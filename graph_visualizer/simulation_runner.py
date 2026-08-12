"""Headless closed-loop simulation runner for overnight, unattended runs.

Wraps the existing (Qt-free) engine -- ``graph_io.load_graph_folder`` ->
``simulation_model.prepare_simulation`` -> ``PreparedSimulation.step_forward`` --
with the robustness a multi-hour unattended run needs: checkpoint/resume, silent-
failure detection, adaptive-dt step-back retry, throttled logging + a live status
file, periodic field snapshots, and an auto-generated report. No Qt; reusable from
a CLI or from the GUI's "run without visualization" path.

Everything for one run lands in a timestamped directory:
    <output_root>/<graph_name>/<YYYYmmdd-HHMMSS>/
        config.json, provenance.json      -- exactly what produced this run
        status.json                        -- live progress (updated as it runs)
        events.log                         -- failures / checks / retries
        timeseries.npz, timeseries.csv     -- the tracked signals
        snapshots/T_<t>.npy                -- periodic full temperature fields
        checkpoints/ckpt_<step>.npz        -- resume points
        report.md, plots/*.png             -- morning-after summary
"""

from __future__ import annotations

import csv
import json
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .graph_io import load_graph_folder
from .simulation_parameters import SimulationParameters
from . import simulation_model


@dataclass
class FailureThresholds:
    """Silent-failure guards. A ``hard`` failure aborts (after finalizing); a soft
    one is logged as an event and the run continues."""

    max_temperature_K: float = 1.0e4  # hard: divergence / non-physical high
    min_temperature_K: float = 0.0  # hard: below absolute zero => broken
    max_temperature_rate_K_per_s: float = 500.0  # hard: runaway (per accepted step)
    energy_drift_rel_tol: float = 0.10  # soft: |net_W - dU/dt| / scale (implicit disc. residual is a few %)
    # Hard divergence guard: if the solve stops conserving energy this badly for
    # this many consecutive steps, the run is producing garbage (the 3443 K / drift
    # ~1.0 case) -- abort with a clear reason instead of burning hours, still
    # finalizing plots. High threshold + sustained count so a real transient never
    # trips it. 0 steps disables it.
    energy_drift_abort_rel: float = 0.9
    energy_drift_abort_steps: int = 20
    forbid_negative_heater_power: bool = True  # soft: controller commanded cooling
    max_dt_retries: int = 6  # step-back halvings before giving up (hard)
    step_wall_timeout_s: float = 0.0  # 0 = disabled watchdog


# A launcher that started the run as a separate, detached process (the headless
# tab) cannot deliver SIGINT/SIGTERM to it -- on Windows terminate() is an
# uncatchable kill that skips _finalize (plots, report, timeseries). Dropping this
# file in the run directory asks the loop to stop at the next step boundary and
# exit through _finalize normally.
# NodeProperties.controller_setpoint_K's default; see the guard in _write_sensor_manifest.
_DEFAULT_SETPOINT_K = 293.15
STOP_REQUEST_FILENAME = "stop.request"


@dataclass
class RunConfig:
    graph_folder: str
    output_root: str = "simulations"
    # Exact output directory. Normally left None so the run gets its own timestamped
    # folder; a launcher that must watch the run's status.json (e.g. the headless
    # tab, which starts the run as a separate process) sets it explicitly.
    run_dir: str | None = None
    controller_path: str | None = None  # modal_controller.npz; None => confirm/open-loop
    allow_no_controller: bool = False  # must be True to run without a controller
    setpoints_K: dict[int, float] = field(default_factory=dict)  # sensor node id -> target
    global_setpoint_K: float | None = None  # applied to every sensor if set
    # Guard against silently running at NodeProperties' 293.15 K default; set True
    # only when room temperature really is the intended target.
    allow_default_setpoint: bool = False
    dt_s: float = 1.0
    t_final_s: float = 3600.0
    gpu_solver_enabled: bool = True
    # Full physics/parameter override. When set (e.g. from the GUI parameter panel),
    # the run uses THESE parameters instead of minimal defaults -- so temperature-
    # dependent properties, radiation, cryocooler config, adaptive substeps, contact
    # conductance etc. match what the user configured. The controller wiring
    # (scheme / path / input_mode) is still enforced by the runner. None => the
    # legacy minimal-default construction (backward compatible for the CLI).
    params: SimulationParameters | None = None
    # Uniform start temperature applied to every node, overriding the graph's saved
    # initial temps (the CLI shortcut for "set the whole system to X K"). For an
    # arbitrary per-node starting state the runner takes an explicit initial_state
    # argument (see SimulationRunner.__init__); that takes precedence over this.
    initial_temperature_uniform_K: float | None = None
    log_interval_steps: int = 1
    snapshot_interval_s: float = 300.0
    checkpoint_interval_s: float = 600.0  # wall-clock seconds between checkpoints
    status_interval_steps: int = 20
    # Prefer the compact nodes.csv + binary-matrix load over parsing graph.json
    # (which costs many GB on a multi-million-cell graph). Falls back automatically
    # when nodes.csv is stale or missing; set False to always use the full loader.
    low_memory_load: bool = True
    # Replay-history depth. Headless runs never scrub back, and each stored step
    # costs 8 bytes/node (256 steps x 3M nodes ~ 6 GB), so keep it minimal.
    history_limit: int = 1
    # Cap on individually-logged sensor series (aggregate RMS error is always kept).
    # Every series is a Python list that grows for the whole run.
    max_logged_sensors: int = 32
    seed: int = 0
    notes: str = ""
    thresholds: FailureThresholds = field(default_factory=FailureThresholds)


# --------------------------------------------------------------------------- #
# Provenance / IO helpers
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _graph_hash(graph_folder: Path) -> str:
    """Cheap content signature of the graph inputs (size+mtime of key files)."""
    parts = []
    for name in ("C.npy", "L_sparse.json", "node_ids.npy", "graph.json", "nodes.csv"):
        p = graph_folder / name
        if p.exists():
            st = p.stat()
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
    return "|".join(parts) or "none"


def _process_rss_gib() -> float:
    """Resident set size of this process in GiB (0.0 if unavailable). Used to make
    memory growth visible in events/status for long unattended runs."""
    try:
        import ctypes
        from ctypes import wintypes

        class _COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # restype/argtypes are required: without them the HANDLE is truncated on
        # 64-bit and the call silently fails (returns 0).
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = _COUNTERS()
        counters.cb = ctypes.sizeof(_COUNTERS)
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return float(counters.WorkingSetSize) / (1024.0**3)
    except Exception:  # noqa: BLE001 - non-Windows or API unavailable
        try:
            import resource  # type: ignore

            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)
        except Exception:  # noqa: BLE001
            return 0.0
    return 0.0


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class SimulationRunner:
    def __init__(
        self,
        config: RunConfig,
        cancel_event: Any | None = None,
        progress_cb: Callable[[dict], None] | None = None,
        initial_state: tuple[Any, Any] | None = None,
        model: Any | None = None,
        matrices: Any | None = None,
    ) -> None:
        self.cfg = config
        self.cancel_event = cancel_event  # anything with .is_set(); optional
        self.progress_cb = progress_cb
        # Optional explicit starting state: (node_ids, temperatures_K) captured from
        # the caller's in-memory model, so the run starts from what the user is
        # looking at rather than the graph's saved-on-disk temperatures. Passed out
        # of band (not through RunConfig) so a million-node vector never bloats
        # config.json. Takes precedence over cfg.initial_temperature_uniform_K.
        self._initial_state = initial_state
        # An already-loaded graph (the GUI has one in memory). Reusing it avoids a
        # SECOND full copy of a multi-million-node model -- the reason a GUI-launched
        # 3M-cell run reached ~50 GB -- and avoids re-running the long Python-level
        # load, which holds the GIL and freezes the window ("Not Responding").
        self._shared_model = model
        self._shared_matrices = matrices
        self._owns_model = model is None
        self.graph_folder = Path(config.graph_folder)
        self.graph_name = self.graph_folder.name
        self.out_dir = (
            Path(config.run_dir)
            if config.run_dir
            else Path(config.output_root) / self.graph_name / _timestamp()
        )
        self.snap_dir = self.out_dir / "snapshots"
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.plots_dir = self.out_dir / "plots"
        self.events_path = self.out_dir / "events.log"
        self.status_path = self.out_dir / "status.json"
        self._series: dict[str, list[float]] = {}
        self._stop = False  # set by SIGTERM/SIGINT or cancel_event
        self._exit_status = "running"
        self._last_step_profile: dict[str, float] = {}
        self._consecutive_high_drift = 0
        self._start_wall = time.time()

    # -- lifecycle ---------------------------------------------------------- #
    def run(self) -> Path:
        for d in (self.out_dir, self.snap_dir, self.ckpt_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)
        # A leftover stop request from a previous run into this same dir (a resume)
        # must not immediately stop this one.
        try:
            (self.out_dir / STOP_REQUEST_FILENAME).unlink()
        except OSError:
            pass
        self._install_signal_handlers()
        self._log_event("run_start", f"graph={self.graph_name} out={self.out_dir}")
        # Graph load can take a while with no visible progress; publish a status so
        # the GUI/`tail` shows the run is alive (and the user doesn't stop it early).
        _atomic_write_json(
            self.status_path,
            {"status": "preparing (loading graph)", "progress": 0.0,
             "updated": datetime.now().isoformat(timespec="seconds")},
        )
        try:
            self._write_config_and_provenance()
            prepared, params, C_diag, sensors, heaters, cryo_idx = self._prepare()
            self._simulate(prepared, params, C_diag, sensors, heaters, cryo_idx)
            if self._exit_status == "running":
                self._exit_status = "completed"
        except _HardFailure as exc:
            self._exit_status = f"failed: {exc}"
            self._log_event("hard_failure", str(exc))
        except KeyboardInterrupt:
            self._exit_status = "interrupted"
            self._log_event("interrupted", "KeyboardInterrupt")
        except Exception as exc:  # noqa: BLE001 -- never leave an overnight run with nothing
            self._exit_status = f"error: {type(exc).__name__}: {exc}"
            self._log_event("error", traceback.format_exc())
        finally:
            self._finalize()
        return self.out_dir

    def _load_graph(self):
        """Load the graph for simulation, preferring the low-memory path.

        The compact path (nodes.csv + binary matrices) skips graph.json entirely --
        1.6 GB at 471k nodes, ~10 GB at 3M, parsed into a Python dict per node, edge
        and cell. It is only safe when nodes.csv is current: it is written when the
        octree is BUILT, so GUI edits saved to graph.json (cryocooler assignments,
        material/capacitance changes) would otherwise be silently lost. Both a
        timestamp check and a capacity cross-check against C.npy guard this, and any
        doubt falls back to the full loader."""
        if self._shared_model is not None and self._shared_matrices is not None:
            self._log_event(
                "graph_load",
                f"reusing the caller's in-memory graph ({len(self._shared_model.nodes)} nodes); "
                "no second copy is loaded",
            )
            return self._shared_model, self._shared_matrices
        if not self.cfg.low_memory_load:
            return load_graph_folder(str(self.graph_folder))
        # A run that drives heaters, runs a controller, or uses the cryocooler needs
        # roles that historically lived only in graph.json. Newer nodes.csv carries
        # them in a role_json column (fast_load_has_roles); when it does NOT, taking
        # the fast path would silently load a model with zero heaters/sensors/
        # cryocoolers and simulate an inert block that only looks like it ran, so we
        # fall back to the full graph.json loader.
        from .fast_graph_io import (
            fast_load_has_roles,
            load_graph_for_simulation,
            validate_against_matrices,
        )

        if self._run_needs_graph_roles() and not fast_load_has_roles(self.graph_folder):
            self._log_event(
                "low_memory_load_skipped",
                "run drives heaters / a controller / the cryocooler, but nodes.csv "
                "has no role_json column; using the full graph.json loader so "
                "heaters, sensors and cryocoolers are not silently dropped "
                "(re-save the graph or run 'Update graph' to add roles to nodes.csv).",
            )
            return load_graph_folder(str(self.graph_folder))
        try:
            model, matrices, report = load_graph_for_simulation(self.graph_folder)
            problem = validate_against_matrices(model, matrices)
            if problem:
                self._log_event("low_memory_load_rejected", problem)
            else:
                for warning in report.warnings:
                    self._log_event("low_memory_load", warning)
                self._log_event(
                    "low_memory_load",
                    f"loaded {report.node_count} nodes and {report.edge_count} conduction "
                    f"edges without graph.json (rss={_process_rss_gib():.1f} GiB)",
                )
                return model, matrices
        except FileNotFoundError as exc:
            self._log_event("low_memory_load_unavailable", str(exc))
        except Exception as exc:  # noqa: BLE001 - never fail a run over an optimisation
            self._log_event("low_memory_load_failed", f"{type(exc).__name__}: {exc}")
        self._log_event("graph_load", "using the full graph.json loader")
        return load_graph_folder(str(self.graph_folder))

    def _run_needs_graph_roles(self) -> bool:
        """Whether the run needs heater/sensor/cryocooler roles that only exist in
        graph.json (so the role-blind nodes.csv fast path would corrupt it)."""
        if self.cfg.controller_path:
            return True
        params = self.cfg.params
        if params is None:
            # Legacy minimal config: a run without an explicit params bundle only
            # forces heaters when a controller is present (handled above).
            return False
        return (
            str(getattr(params, "input_mode", "")) == "heater_inputs"
            or bool(getattr(params, "mimo_controller_enabled", False))
            or bool(getattr(params, "cryocooler_enabled", False))
        )

    # -- setup -------------------------------------------------------------- #
    def _prepare(self):
        model, matrices = self._load_graph()
        # Controller gate: caller must have chosen one, or explicitly allowed none.
        controller_path = self.cfg.controller_path
        if not controller_path:
            default_ctrl = self.graph_folder / "modal_controller.npz"
            if default_ctrl.exists():
                controller_path = str(default_ctrl)
        has_controller = bool(controller_path) and Path(controller_path).exists()
        if not has_controller and not self.cfg.allow_no_controller:
            raise _HardFailure(
                "No controller selected and allow_no_controller=False. Choose a "
                "modal_controller.npz or set allow_no_controller=True to run open-loop."
            )
        # Apply constant setpoints onto the sensor nodes the controller reads.
        sensors = [int(nid) for nid, n in model.nodes.items() if getattr(n, "is_sensor", False)]
        self._apply_setpoints(model, sensors)

        params = self._resolve_params(has_controller, controller_path)
        prepared = simulation_model.prepare_simulation(model, matrices, params)

        # Override the starting temperatures with the caller's in-memory state (or a
        # uniform value) so the run begins from what the user set, not the graph's
        # saved-on-disk temps. Done BEFORE _resume_if_checkpoint so a genuine resume
        # still wins. Both the working state (set_temperatures) and the baseline
        # (initial_temperatures_K, used for prev-step metrics) are updated.
        init_override = self._resolve_initial_temperatures(prepared)
        if init_override is not None:
            prepared.initial_temperatures_K = init_override
            prepared.set_temperatures(init_override)
            self._log_event(
                "initial_state",
                f"start temperatures overridden: min={float(init_override.min()):.2f} "
                f"mean={float(init_override.mean()):.2f} max={float(init_override.max()):.2f} K",
            )
        C_diag = np.asarray(matrices["C"], dtype=float).reshape(-1)
        heaters = [int(x) for x in getattr(prepared, "heater_node_ids", ())]
        cryo_idx = [
            prepared.node_index_by_id[int(x)]
            for x in getattr(prepared, "cryocooler_node_ids", ())
            if int(x) in prepared.node_index_by_id
        ]
        # Release the raw parsed graph.json payload retained on the model
        # (graph_nodes/graph_edges/octree_cells: one Python dict per node, edge and
        # cell -- gigabytes on a multi-million-cell graph). prepare_simulation has
        # already built the matrices, and a headless run never re-derives geometry
        # from it, so it is pure overhead for the rest of the run.
        import gc

        if self._owns_model:
            # Only when WE loaded it: a shared model belongs to the caller (the GUI
            # still needs octree_graph_data to reuse its loaded matrices).
            try:
                model.octree_graph_data = None
            except Exception:  # noqa: BLE001
                pass
        gc.collect()
        history_limit = prepared._effective_history_limit()
        self._log_event(
            "prepared",
            f"nodes={len(C_diag)} sensors={len(sensors)} heaters={len(heaters)} "
            f"cryo={len(cryo_idx)} controller={'yes' if has_controller else 'OPEN-LOOP'} "
            f"history_limit={history_limit} "
            f"(~{history_limit * len(C_diag) * 8 / 1024.0 / 1024.0:.0f} MB) "
            f"rss={_process_rss_gib():.1f} GiB",
        )
        for warning in prepared.warnings:
            self._log_event("prepare_warning", str(warning))
        # The engine appends some warnings LAZILY -- notably "modal controller
        # unavailable (...); nothing is regulating", which is only raised when the
        # controller is first evaluated, i.e. after this point. Remember how many we
        # have logged so the step loop can surface the rest; otherwise a run could
        # silently use a different controller than the one requested.
        self._logged_warning_count = len(prepared.warnings)
        self._warn_if_disconnected()
        self._report_quarantine(prepared)
        self._check_actuator_connectivity(prepared, params)
        self._warn_if_unforced(prepared, params, has_controller, cryo_idx)
        self._resume_if_checkpoint(prepared)
        return prepared, params, C_diag, sensors, heaters, cryo_idx

    def _report_quarantine(self, prepared) -> None:
        """Log which cells were quarantined and which heaters it cost.

        The count alone understates this. A detached solid can be 0.001% of the
        graph and still absorb most of the run's heater power, so the number that
        matters is how much actuator authority now lands on nothing."""
        result = getattr(prepared, "quarantine_result", None)
        if result is None:
            return
        self._quarantine_summary = {
            "quarantined_cells": int(getattr(result, "count", 0)),
            "quarantined_components": int(getattr(result, "quarantined_component_count", 0)),
            "conduction_components": int(getattr(result, "component_count", 0)),
            "orphaned_heaters": [int(v) for v in getattr(prepared, "orphaned_heater_ids", ()) or ()],
            "heaters_with_lost_targets": len(getattr(prepared, "heaters_missing_deposition", {}) or {}),
        }
        if not getattr(result, "any_quarantined", False):
            return
        self._log_event("cell_quarantine", result.summary(getattr(prepared, "node_ids", None)))
        lost = getattr(prepared, "heaters_missing_deposition", {}) or {}
        if lost:
            detail = ", ".join(
                f"{heater_id}->{len(targets)} cell(s)" for heater_id, targets in list(lost.items())[:8]
            )
            self._log_event(
                "cell_quarantine_heaters",
                f"{len(lost)} heater(s) lost deposition targets ({detail}"
                + (", ..." if len(lost) > 8 else "")
                + "). Their commanded power is excluded from power_in_W.",
            )
        orphans = [int(v) for v in getattr(prepared, "orphaned_heater_ids", ()) or ()]
        if orphans:
            self._log_event(
                "cell_quarantine_orphans",
                f"{len(orphans)} heater(s) now deposit into nothing at all (node ids {orphans[:8]}"
                + (", ..." if len(orphans) > 8 else "")
                + "). They remain in the controller by design; rebuild the modal controller so "
                "its DC gain stops assigning them effort.",
            )

    def _check_actuator_connectivity(self, prepared, params) -> None:
        """Verify each heater can actually conduct heat to the sensor it drives.

        A heater deposits into its power_deposition_node_ids and a sensor reads its
        readout_node_ids. If contact detection failed to bond the heater's part to
        the assembly, those two node sets land in DIFFERENT connected components:
        the heater's power heats a small thermally-floating island while the sensor
        reads an inert body that never moves. The tracking error is then constant by
        construction, the integrator winds up to saturation, and the island runs
        away (this is the no_mli_high_res 10,200 K case -- 99.2% of that graph was
        bit-for-bit unchanged after 600 s while 690 W went into ~23.6k stranded
        nodes).

        Cheap to detect here (one connected-components pass), ruinous to discover
        from an overnight run's plots."""
        try:
            from scipy.sparse import issparse, csr_matrix
            from scipy.sparse.csgraph import connected_components

            model = getattr(prepared, "model", None)
            A = getattr(prepared, "A", None)
            if model is None or A is None:
                return
            adjacency = csr_matrix(A) if issparse(A) else csr_matrix(np.asarray(A))
            n_components, labels = connected_components(adjacency, directed=False)
            if n_components <= 1:
                return
            node_ids = np.asarray(prepared.node_ids)
            node_index = getattr(prepared, "node_index_by_id", None) or {
                int(v): i for i, v in enumerate(node_ids)
            }

            def _components_of(ids) -> set[int]:
                return {
                    int(labels[node_index[int(v)]])
                    for v in (ids or [])
                    if int(v) in node_index
                }

            enabled = simulation_model._enabled_node_id_set(
                getattr(params, "enabled_heater_node_ids", None)
            )
            broken: list[int] = []
            checked = 0
            for heater_id in getattr(prepared, "heater_node_ids", []) or []:
                heater_id = int(heater_id)
                if not simulation_model._node_id_enabled(enabled, heater_id):
                    continue
                heater = model.nodes.get(heater_id)
                sensor_id = getattr(heater, "assigned_sensor_id", None) if heater else None
                if sensor_id is None:
                    continue
                sensor = model.nodes.get(int(sensor_id))
                if sensor is None:
                    continue
                deposit = _components_of(getattr(heater, "power_deposition_node_ids", []))
                readout = _components_of(
                    getattr(sensor, "readout_node_ids", [])
                    or getattr(sensor, "sensor_connected_node_ids", [])
                )
                if not deposit or not readout:
                    continue
                checked += 1
                if deposit.isdisjoint(readout):
                    broken.append(heater_id)
            if not checked:
                return
            if broken:
                self._log_event(
                    "actuators_disconnected",
                    f"{len(broken)}/{checked} controlled heater(s) cannot conduct heat to "
                    f"their paired sensor -- deposition and readout nodes are in different "
                    f"connected components (graph has {n_components}). Those heaters warm a "
                    f"thermally-floating island while the sensor never moves, so the "
                    f"tracking error cannot close and the integrator winds up. "
                    f"Heater node ids (first 5): {broken[:5]}",
                )
            if broken and len(broken) == checked:
                raise RuntimeError(
                    f"No enabled heater ({checked} checked) shares a connected component "
                    "with the sensor it drives, so no setpoint is reachable and the "
                    "controller would wind up to saturation heating isolated cells. "
                    "Fix the graph's contact detection (or disable the controller with "
                    "allow_no_controller + input_mode='zero') before running."
                )
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - diagnostic must never break a run
            self._log_event("actuator_connectivity_check_failed", str(exc))

    def _warn_if_unforced(self, prepared, params, has_controller: bool, cryo_idx: list) -> None:
        """Flag a run with no heat source, no heat sink and no radiative gradient.

        Such a run integrates dT/dt = 0 to machine precision for however many
        hours it is given -- the sensor "tracking error" it reports is just the
        gap between the initial condition and the setpoint, never closing. Cheap
        to detect at prepare time; expensive to discover from the plots."""
        reasons = []
        if not has_controller and str(getattr(params, "input_mode", "")) != "heater_inputs":
            reasons.append("no heater input (input_mode='zero' and no controller)")
        if not cryo_idx:
            reasons.append("no cryocooler nodes in the graph")
        env = getattr(prepared, "environment_temperature_K", None)
        initial = np.asarray(prepared.initial_temperatures_K, dtype=float)
        env_vector = (
            np.asarray(env, dtype=float)
            if env is not None
            else np.full(initial.shape, float(getattr(params, "T_env_K", 0.0)))
        )
        if np.allclose(env_vector, initial, atol=1.0e-9):
            reasons.append(
                f"radiative background equals the initial temperature "
                f"(T_env={float(env_vector.flat[0]):.2f} K), so net radiation is zero"
            )
        if len(reasons) >= 3:
            self._log_event(
                "unforced_run",
                "WARNING: nothing drives this simulation -- " + "; ".join(reasons)
                + ". Temperatures will not change.",
            )

    def _apply_setpoints(self, model, sensors: list[int]) -> None:
        if self.cfg.global_setpoint_K is not None:
            for sid in sensors:
                model.nodes[sid].controller_setpoint_K = float(self.cfg.global_setpoint_K)
        for sid, target in (self.cfg.setpoints_K or {}).items():
            if int(sid) in model.nodes:
                model.nodes[int(sid)].controller_setpoint_K = float(target)

    def _resolve_params(self, has_controller: bool, controller_path: str | None) -> SimulationParameters:
        """The parameters the run uses. When the caller supplied a full
        SimulationParameters we keep its physics (tdep properties, radiation,
        cryocooler, substeps, ...) and only enforce the controller wiring; otherwise
        we build the legacy minimal defaults."""
        # --controller carries whichever artifact the caller picked, and the two
        # schemes take DIFFERENT artifacts: modal LQR wants a modal_controller.npz,
        # MIMO PI wants the sys-id run FOLDER holding G. Wiring the folder into
        # modal_controller_path (which is what this used to do unconditionally)
        # produces a modal load failure and an open-loop overnight run.
        scheme = (
            str(getattr(self.cfg.params, "mimo_controller_scheme", "") or "")
            if self.cfg.params is not None else ""
        )
        if not scheme or scheme == "none":
            scheme = "modal_lqr" if has_controller else "none"
        is_mimo_pi = has_controller and scheme == "mimo_pi"
        controller_fields = dict(
            dt_s=float(self.cfg.dt_s),
            t_final_s=float(self.cfg.t_final_s),
            gpu_solver_enabled=bool(self.cfg.gpu_solver_enabled),
            # Replay history exists so the GUI can scrub backwards; a headless run
            # never does, and each stored step costs 8 bytes/node -- the default 256
            # steps is ~6 GB on a 3M-cell graph. Keep the minimum the stepper needs.
            simulation_history_limit=max(1, int(self.cfg.history_limit)),
            input_mode=(
                "heater_inputs" if has_controller
                else ("zero" if self.cfg.params is None else self.cfg.params.input_mode)
            ),
            mimo_controller_enabled=has_controller,
            mimo_controller_scheme=(scheme if has_controller else "none"),
            modal_controller_path="" if is_mimo_pi else (controller_path or ""),
        )
        if is_mimo_pi:
            controller_fields["mimo_pi_gain_matrix_path"] = controller_path or ""
        if self.cfg.params is not None:
            return replace(self.cfg.params, **controller_fields)
        return SimulationParameters(**controller_fields)

    def _resolve_initial_temperatures(self, prepared) -> np.ndarray | None:
        """Starting temperature vector (ordered like ``prepared.node_ids``), or None
        to keep the graph's loaded temps. Explicit per-node ``initial_state`` wins
        over the uniform-scalar config field."""
        node_ids = np.asarray(prepared.node_ids, dtype=int).reshape(-1)
        if self._initial_state is not None:
            ids = np.asarray(self._initial_state[0], dtype=int).reshape(-1)
            temps = np.asarray(self._initial_state[1], dtype=float).reshape(-1)
            if ids.shape != temps.shape:
                self._log_event("initial_state", "ignored: node_ids/temperatures length mismatch")
            elif np.array_equal(ids, node_ids):
                return temps.copy()  # same order as the reloaded graph (the common case)
            else:
                # Different ordering: map by node id, keeping the graph's own temp
                # for any node not supplied.
                mapping = dict(zip(ids.tolist(), temps.tolist()))
                base = np.asarray(prepared.initial_temperatures_K, dtype=float).reshape(-1)
                return np.array(
                    [mapping.get(int(nid), float(base[i])) for i, nid in enumerate(node_ids)],
                    dtype=float,
                )
        if self.cfg.initial_temperature_uniform_K is not None:
            return np.full(node_ids.size, float(self.cfg.initial_temperature_uniform_K), dtype=float)
        return None

    # -- main loop ---------------------------------------------------------- #
    def _simulate(self, prepared, params, C_diag, sensors, heaters, cryo_idx) -> None:
        thr = self.cfg.thresholds
        node_index = prepared.node_index_by_id
        # The engine's own "real body cell" mask (excludes heater/marker nodes), so
        # the runner's step-sanity check polices exactly the cells the engine does
        # and never rejects a step over a marker node's meaningless overshoot.
        try:
            self._real_cell_mask = prepared._physical_step_check_mask()
        except Exception:  # noqa: BLE001 - fall back to policing every cell
            self._real_cell_mask = None
        # Build the row indices and setpoints TOGETHER. Filtering the indices while
        # taking setpoints from the unfiltered sensor list would shift every setpoint
        # onto the wrong sensor as soon as one sensor is missing from the matrices.
        sensor_ids: list[int] = []
        sensor_ix: list[int] = []
        setpoint_values: list[float] = []
        for sensor_id in sensors:
            row = node_index.get(int(sensor_id))
            if row is None:
                self._log_event(
                    "sensor_missing",
                    f"sensor node {sensor_id} is not in the matrices; excluded from tracking.",
                )
                continue
            sensor_ids.append(int(sensor_id))
            sensor_ix.append(int(row))
            setpoint_values.append(
                float(getattr(prepared.model.nodes[int(sensor_id)], "controller_setpoint_K", np.nan))
            )
        setpoints = np.array(setpoint_values, dtype=float)
        self._sensor_ids = sensor_ids
        self._sensor_rows = sensor_ix
        self._sensor_setpoints = setpoints
        # A monitor-only sensor has NO heater assigned, so the controller never acts
        # on it; it only warms by conduction from the regulated regions. Averaging it
        # into the tracking error hides how the loop is actually doing: on
        # no_mli_high_res the 27 controlled sensors reached 0.42 K RMS while the
        # reported figure -- diluted by 64 monitor-only sensors -- still read 4.57 K.
        # Which sensors get an individual series. The cap used to take the first N by
        # INDEX, which on no_mli_high_res meant 32 slots spent almost entirely on
        # monitor-only sensors -- the ones the controller cannot act on -- so the
        # plots showed everything except the loop being tuned. Controlled sensors now
        # claim the budget first; monitor-only fill whatever is left.
        self._sensor_controlled = np.array(
            [
                not bool(getattr(prepared.model.nodes.get(int(sid)), "sensor_monitor_only", False))
                for sid in sensor_ids
            ],
            dtype=bool,
        )
        cap = max(0, int(self.cfg.max_logged_sensors))
        controlled_first = [int(j) for j in np.where(self._sensor_controlled)[0]]
        monitor_rest = [int(j) for j in np.where(~self._sensor_controlled)[0]]
        self._logged_sensor_indices = (controlled_first + monitor_rest)[:cap]
        # Series keep their ORIGINAL sensor index, so sensor_<j> still matches the
        # "series" column in sensors.csv even though the logging order changed.
        self._controlled_series_keys = {
            f"sensor_{j}_K" for j in self._logged_sensor_indices if self._sensor_controlled[j]
        }
        if len(controlled_first) > cap:
            self._log_event(
                "sensor_logging",
                f"{len(controlled_first)} controlled sensors exceed max_logged_sensors={cap}; "
                f"{len(controlled_first) - cap} have no individual series (aggregate RMS still covers all)",
            )
        self._build_sensor_readout_operator(prepared, sensor_ids, sensor_ix)
        self._write_sensor_manifest(prepared, sensor_ids, setpoints)
        prev_temps = np.asarray(prepared.initial_temperatures_K, dtype=float).copy()
        last_snapshot_t = -1.0e18
        last_ckpt_wall = time.time()
        step = 0
        base_dt = float(params.dt_s)
        accepted = False  # bound before the loop so a 0-iteration run finalizes cleanly
        state = None

        while not self._should_stop():
            t_now = self._current_time(prepared, step, base_dt)
            if t_now >= self.cfg.t_final_s - 1e-9:
                break
            # --- adaptive-dt step-back retry ---
            attempt_dt = base_dt
            accepted = False
            for retry in range(thr.max_dt_retries + 1):
                params.dt_s = attempt_dt
                snap = prepared.snapshot_state()
                try:
                    state = prepared.step_forward()
                except Exception as exc:  # noqa: BLE001
                    prepared.restore_state(snap)
                    self._log_event("solver_error", f"step={step} dt={attempt_dt:g}: {exc}")
                    attempt_dt *= 0.5
                    continue
                temps = np.asarray(state.temperatures_K, dtype=float)
                bad, reason = self._step_is_bad(temps, prev_temps, attempt_dt, thr)
                if bad:
                    prepared.restore_state(snap)
                    self._log_event("step_rejected", f"step={step} dt={attempt_dt:g}: {reason}")
                    attempt_dt *= 0.5
                    continue
                accepted = True
                break
            if not accepted:
                raise _HardFailure(
                    f"step {step}: the implicit step could not produce a physical result after "
                    f"{thr.max_dt_retries} dt halvings (down to {attempt_dt:g}s); last reason: "
                    f"{reason or 'unknown'}. This is the graph diverging, not a transient -- most "
                    "often dt is far larger than the graph's fastest thermal time constant (see the "
                    "'dt_s coarse relative to tau' prepare warning) and/or aggressive cryocooler "
                    "cooling overshoots below 0 K. Reduce dt substantially (or raise adaptive "
                    "max substeps) and check for degenerate near-zero-capacity nodes. "
                    "Last state checkpointed."
                )
            params.dt_s = base_dt  # recover after a successful accept

            # --- record ---
            self._collect(prepared, state, temps, prev_temps, attempt_dt,
                          C_diag, sensor_ix, setpoints, heaters, cryo_idx, thr)
            prev_temps = temps.copy()
            self._last_temps = prev_temps  # for the end-of-run verification tables
            self._verification_source = prepared
            # Fail fast on a diverging solve: hours of a run that conserves no energy
            # is worthless. _finalize still runs (via run()'s finally), so the partial
            # plots/report survive and show the divergence.
            if (
                thr.energy_drift_abort_steps > 0
                and self._consecutive_high_drift >= thr.energy_drift_abort_steps
            ):
                self._checkpoint(prepared, state, step)
                raise _HardFailure(
                    f"energy drift exceeded {thr.energy_drift_abort_rel:g} for "
                    f"{self._consecutive_high_drift} consecutive steps at t={state.time_s:.1f}s "
                    f"(max T={float(np.max(temps)):.3g} K): the solve is diverging / not "
                    "conserving energy. Likely causes: dt too large for the graph's stiffness, "
                    "degenerate near-zero node capacitances, or disconnected components. "
                    "Aborting so the run does not waste further time."
                )
            # Per-step timing breakdown, so status.json shows where wall-clock goes
            # (the temperature-dependent operator rebuild vs. the linear solve) --
            # the number needed to decide whether to lag properties or add a GPU.
            self._last_step_profile = dict(getattr(prepared, "last_step_profile_ms", {}) or {})
            step += 1

            # --- snapshots / checkpoints / status ---
            if state.time_s - last_snapshot_t >= self.cfg.snapshot_interval_s:
                np.save(self.snap_dir / f"T_{state.time_s:.1f}.npy", temps)
                last_snapshot_t = state.time_s
            if time.time() - last_ckpt_wall >= self.cfg.checkpoint_interval_s:
                self._checkpoint(prepared, state, step)
                last_ckpt_wall = time.time()
            if step % max(1, self.cfg.status_interval_steps) == 0:
                self._update_status(step, state.time_s)
            self._drain_engine_warnings(prepared)

        if step == 0:
            self._log_event(
                "no_steps",
                f"loop ran 0 steps (stopped/cancelled during setup, or t_final<=0). "
                f"t_final={self.cfg.t_final_s:g}s",
            )
        # final checkpoint + status
        self._checkpoint(prepared, state if accepted else None, step)
        self._update_status(step, self._series.get("time_s", [0.0])[-1] if self._series.get("time_s") else 0.0)

    def _write_sensor_manifest(self, prepared, sensor_ids, setpoints) -> None:
        """sensors.csv: which node each tracked sensor is, what it reads, and the
        setpoint applied to it.

        The time series are named ``sensor_<i>_K``; without this file that index is
        meaningless, so there is no way to confirm a setpoint landed on the intended
        sensor. Written at start (so it exists even if the run dies) and completed
        with final values in _finalize."""
        nodes = getattr(prepared.model, "nodes", {}) or {}
        rows = []
        for index, (node_id, setpoint) in enumerate(zip(sensor_ids, np.asarray(setpoints, dtype=float))):
            node = nodes.get(int(node_id))
            rows.append(
                {
                    "series": f"sensor_{index}",
                    "node_id": int(node_id),
                    "component_name": str(getattr(node, "component_name", "") or ""),
                    "material": str(getattr(node, "material", "") or ""),
                    "setpoint_K": "" if not np.isfinite(setpoint) else f"{float(setpoint):.6g}",
                    "monitor_only": bool(getattr(node, "sensor_monitor_only", False)),
                    "readout_nodes": len(getattr(node, "readout_node_ids", None) or []),
                    "center_mm": ";".join(
                        f"{float(v):.2f}" for v in (getattr(node, "center_mm", None) or ())
                    ),
                }
            )
        self._sensor_manifest_rows = rows
        if not rows:
            return
        with (self.out_dir / "sensors.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        finite = np.asarray(setpoints, dtype=float)
        finite = finite[np.isfinite(finite)]
        # A run that was never given a setpoint silently targets NodeProperties'
        # 293.15 K default -- room temperature. On the low-memory path nodes.csv
        # carries no per-node setpoint, so nothing overrides it, and a cryogenic run
        # then commands every heater flat-out toward 293 K for its whole duration
        # (observed: a full 3600 s run whose tracking error started at -253 K).
        # It is never what a 40 K cryostat run wants, so refuse rather than log it.
        if (
            finite.size
            and self.cfg.global_setpoint_K is None
            and not (self.cfg.setpoints_K or {})
            and bool(np.all(finite == _DEFAULT_SETPOINT_K))
            and not bool(getattr(self.cfg, "allow_default_setpoint", False))
        ):
            raise _HardFailure(
                f"No setpoint was given, so all {finite.size} sensors kept the built-in "
                f"default of {_DEFAULT_SETPOINT_K} K (room temperature). This graph loaded "
                "without per-node setpoints, so nothing overrode it. Pass --setpoint (or "
                "tick 'use setpoint' in the headless tab), or supply --setpoints-json. "
                "Set allow_default_setpoint=True if you really do want 293.15 K."
            )
        if finite.size:
            self._log_event(
                "setpoints",
                f"{finite.size} sensor setpoint(s) applied: min={finite.min():.3f} K "
                f"max={finite.max():.3f} K, {len(np.unique(np.round(finite, 6)))} distinct "
                "(per-sensor detail in sensors.csv)",
            )
        else:
            self._log_event("setpoints", "no finite sensor setpoints were applied")

    def _write_verification_tables(self, prepared) -> None:
        """Final per-sensor results and a per-component temperature summary, so the
        whole assembly can be checked -- not just the tracked sensors."""
        temps = getattr(self, "_last_temps", None)
        if temps is None:
            return
        temps = np.asarray(temps, dtype=float)
        rows = list(getattr(self, "_sensor_manifest_rows", []) or [])
        if rows:
            # Same readout the controller regulates -- NOT the marker node, which is
            # thermally isolated and would report the initial temperature forever.
            readouts = self._sensor_readout_temperatures(temps, self._sensor_rows)
            for index, row in enumerate(rows):
                try:
                    value = float(readouts[index])
                except (IndexError, ValueError):
                    continue
                row["final_K"] = f"{value:.6g}"
                setpoint = self._sensor_setpoints[index]
                row["error_K"] = f"{value - float(setpoint):.6g}" if np.isfinite(setpoint) else ""
            with (self.out_dir / "sensors.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        # Per-component temperature spread. Accumulated in a dict of running stats so
        # a multi-million-node field never materialises grouped copies.
        nodes = getattr(prepared.model, "nodes", {}) or {}
        node_ids = np.asarray(prepared.node_ids, dtype=int).reshape(-1)
        stats: dict[str, list[float]] = {}
        for row, node_id in enumerate(node_ids):
            if row >= temps.size:
                break
            node = nodes.get(int(node_id))
            name = str(getattr(node, "component_name", "") or "(unnamed)")
            value = float(temps[row])
            entry = stats.get(name)
            if entry is None:
                stats[name] = [1.0, value, value, value]  # count, sum, min, max
            else:
                entry[0] += 1.0
                entry[1] += value
                entry[2] = min(entry[2], value)
                entry[3] = max(entry[3], value)
        if not stats:
            return
        with (self.out_dir / "component_temperatures.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["component_name", "nodes", "min_K", "mean_K", "max_K", "spread_K"])
            for name, (count, total, low, high) in sorted(
                stats.items(), key=lambda kv: kv[1][3], reverse=True
            ):
                writer.writerow(
                    [name, int(count), f"{low:.6g}", f"{total / count:.6g}", f"{high:.6g}", f"{high - low:.6g}"]
                )
        self._log_event(
            "verification",
            f"wrote sensors.csv ({len(rows)} sensors) and component_temperatures.csv "
            f"({len(stats)} components)",
        )

    def _warn_if_disconnected(self) -> None:
        """Surface a graph that is not fully connected.

        Islands are thermally isolated: they never exchange heat with the main body,
        so they drift on their own and quietly skew whole-graph metrics (mean/max
        temperature, energy balance). The build already computes this, so read its
        report rather than re-running a connected-components pass on millions of
        nodes."""
        path = self.graph_folder / "connectivity_analysis.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if data.get("connected", True):
            return
        total = int(data.get("node_count", 0) or 0)
        largest = int(data.get("largest_component_size", 0) or 0)
        stranded = max(0, total - largest) if total else len(data.get("disconnected_node_ids", []) or [])
        self._log_event(
            "graph_not_connected",
            f"graph has {int(data.get('component_count', 0))} components; {stranded} node(s) are "
            f"outside the main body ({largest}/{total}). Isolated nodes exchange no heat and "
            "skew whole-graph metrics (as recorded at build time).",
        )

    def _drain_engine_warnings(self, prepared) -> None:
        """Log warnings the engine appended since the last check.

        Some are only raised once a feature is first exercised -- e.g. a modal
        controller artifact that does not match the graph is detected when the
        controller is first evaluated and silently degrades to open-loop
        allocator. Surfacing them keeps the run's log honest about what actually
        ran."""
        warnings = getattr(prepared, "warnings", None)
        if not warnings:
            return
        seen = getattr(self, "_logged_warning_count", 0)
        if len(warnings) <= seen:
            return
        for warning in warnings[seen:]:
            self._log_event("engine_warning", str(warning))
        self._logged_warning_count = len(warnings)

    def _step_is_bad(self, temps, prev, dt, thr) -> tuple[bool, str]:
        # Only genuine SOLVER failures reject-and-halve (smaller dt can help those):
        # non-finite results and hard out-of-bounds (runaway). A high temperature
        # RATE is a divergence INDICATOR, not a solver failure -- halving dt only
        # raises rate = dT/dt, so it must NOT trigger the retry (it's a soft warning
        # in _collect instead). This is what doom-looped on stiff/artifact graphs.
        #
        # Police only REAL body cells -- the SAME mask the engine uses to decide a
        # step is physical (SparseImplicitStepper._implicit_step_is_physical). Heater
        # and other marker nodes carry artificial near-zero capacitance and
        # physically meaningless temperatures (they deposit into real cells); their
        # explicit-source overshoot regularly goes slightly negative and the engine
        # deliberately ignores it. Policing them here made the runner reject a step
        # the engine had already accepted, then halve dt -- which, for TR-BDF2, makes
        # the overshoot WORSE -- so the run doom-looped on graphs with marker nodes.
        if not np.all(np.isfinite(temps)):
            return True, "non-finite temperatures (NaN/Inf)"
        mask = getattr(self, "_real_cell_mask", None)
        policed = temps
        if mask is not None and np.asarray(mask).shape == np.asarray(temps).shape:
            policed = temps[mask]
        if policed.size == 0:
            policed = temps
        tmax = float(np.max(policed))
        tmin = float(np.min(policed))
        if tmax > thr.max_temperature_K:
            return True, f"max real-cell T {tmax:.3g}K > {thr.max_temperature_K:g}K"
        if tmin < thr.min_temperature_K:
            return True, f"min real-cell T {tmin:.3g}K < {thr.min_temperature_K:g}K"
        return False, ""

    # -- readouts / logging ------------------------------------------------- #
    def _build_sensor_readout_operator(self, prepared, sensor_ids, sensor_ix) -> None:
        """Precompute the sensor readout as a sparse (n_sensors x n_nodes) matrix.

        A sensor's temperature is the weighted mean over its readout body cells, the
        same quantity ``sensor_readout_temperature_K`` gives the controller. Building
        it once keeps per-step reporting to a single sparse mat-vec instead of a
        Python loop over ~91 sensors x up to ~500 cells each. Sensors with no readout
        cells fall back to their own row so their series stays populated."""
        self._sensor_readout_matrix = None
        try:
            from scipy.sparse import csr_matrix

            model = getattr(prepared, "model", None)
            if model is None:
                return
            node_index = getattr(prepared, "node_index_by_id", None) or {
                int(v): i for i, v in enumerate(prepared.node_ids)
            }
            n_nodes = int(np.asarray(prepared.node_ids).size)
            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
            fallback = 0
            for j, sensor_id in enumerate(sensor_ids):
                node = model.nodes.get(int(sensor_id))
                ids = [
                    int(v)
                    for v in (
                        getattr(node, "readout_node_ids", None)
                        or getattr(node, "sensor_connected_node_ids", None)
                        or []
                    )
                    if int(v) in node_index
                ]
                raw = [float(w) for w in (getattr(node, "readout_weights", None) or [])]
                if not ids:
                    rows.append(j); cols.append(int(sensor_ix[j])); data.append(1.0)
                    fallback += 1
                    continue
                weights = raw[: len(ids)]
                if len(weights) != len(ids) or not (sum(weights) > 0.0):
                    weights = [1.0 / len(ids)] * len(ids)
                total = float(sum(weights))
                for node_id, weight in zip(ids, weights):
                    rows.append(j); cols.append(node_index[node_id]); data.append(weight / total)
            self._sensor_readout_matrix = csr_matrix(
                (data, (rows, cols)), shape=(len(sensor_ids), n_nodes)
            )
            if fallback:
                self._log_event(
                    "sensor_readout",
                    f"{fallback} sensor(s) have no body readout cells; reporting their own "
                    "node temperature (a marker node exchanges no heat, so it will not move).",
                )
        except Exception as exc:  # noqa: BLE001 - reporting must never break a run
            self._log_event("sensor_readout_unavailable", str(exc))
            self._sensor_readout_matrix = None

    @staticmethod
    def _live_temperatures(prepared, temps) -> np.ndarray:
        """Temperatures of the cells that are actually simulated.

        Quarantined cells have no conduction path to a sink, so they receive no
        power and never change; keeping them in whole-graph statistics reports the
        dead end instead of the body."""
        temps = np.asarray(temps)
        mask = getattr(prepared, "inert_cell_mask", None)
        if mask is None or mask.shape != temps.shape or not np.any(mask):
            return temps
        live = temps[~mask]
        return live if live.size else temps

    def _sensor_readout_temperatures(self, temps, sensor_ix) -> np.ndarray:
        matrix = getattr(self, "_sensor_readout_matrix", None)
        if matrix is None:
            return np.asarray(temps)[sensor_ix]
        return np.asarray(matrix @ np.asarray(temps, dtype=float)).reshape(-1)

    def _collect(self, prepared, state, temps, prev, dt, C_diag, sensor_ix,
                 setpoints, heaters, cryo_idx, thr) -> None:
        s = self._series
        s.setdefault("time_s", []).append(float(state.time_s))
        # Whole-graph metrics exclude quarantined cells. A thermal dead end holds
        # whatever temperature it drifted to and cannot respond to anything, so
        # including it pins max_temp / the rate / the autoscale colour range to a
        # body that is not part of the simulation any more.
        live = self._live_temperatures(prepared, temps)
        s.setdefault("avg_temp_K", []).append(float(np.mean(live)))
        s.setdefault("max_temp_K", []).append(float(np.max(live)))
        s.setdefault("min_temp_K", []).append(float(np.min(live)))
        # Max temperature rate -- soft divergence indicator (logged, not fatal).
        if dt > 0 and prev.shape == temps.shape:
            delta = np.abs(temps - prev)
            mask = getattr(prepared, "inert_cell_mask", None)
            if mask is not None and mask.shape == delta.shape:
                delta = delta[~mask]
            rate = float(np.max(delta)) / dt if delta.size else 0.0
            s.setdefault("max_temp_rate_K_per_s", []).append(rate)
            if rate > thr.max_temperature_rate_K_per_s:
                self._log_event("high_temp_rate", f"t={state.time_s:.1f}s max|dT/dt|={rate:.3g}K/s (soft)")
        # sensor temps + tracking error
        if sensor_ix:
            # Report the READOUT temperature -- the weighted mean over the sensor's
            # body cells -- which is what the controller regulates. Indexing temps at
            # the sensor's own marker row reports the marker instead, and a marker is
            # a thermally isolated single-node component (its role edges carry
            # G = 0 W/K), so it sits at the initial temperature for the whole run.
            # That made every sensor read exactly 40.15 K and pinned the tracking
            # error at its start value while the real readouts had already moved
            # +12.9 K and closed the RMS error from 9.04 to 7.96 K.
            sens = self._sensor_readout_temperatures(temps, sensor_ix)
            # Each per-sensor series is a Python list held for the whole run; with
            # many sensors x an overnight step count that adds up, so log individual
            # sensors up to a cap (the aggregate RMS below always covers them all).
            # Falls back to the plain first-N order when the series order was never
            # built (a partially constructed runner in a unit test).
            logged = getattr(self, "_logged_sensor_indices", None)
            if logged is None:
                logged = range(min(len(sensor_ix), max(0, int(self.cfg.max_logged_sensors))))
            for j in logged:
                s.setdefault(f"sensor_{j}_K", []).append(float(sens[j]))
                if np.isfinite(setpoints[j]):
                    s.setdefault(f"sensor_{j}_err_K", []).append(float(sens[j] - setpoints[j]))
            valid = np.isfinite(setpoints) & np.isfinite(sens)
            if valid.any():
                err = sens[valid] - setpoints[valid]
                s.setdefault("rms_tracking_error_K", []).append(float(np.sqrt(np.mean(err**2))))
                # Split out the sensors the controller actually regulates. Keeping the
                # all-sensor figure as well, since a monitor-only sensor far from
                # setpoint is still worth seeing -- it just is not a control failure.
                controlled = getattr(self, "_sensor_controlled", None)
                if controlled is not None and controlled.shape == valid.shape:
                    for label, mask in (
                        ("rms_tracking_error_controlled_K", valid & controlled),
                        ("rms_tracking_error_monitor_K", valid & ~controlled),
                    ):
                        if mask.any():
                            e = sens[mask] - setpoints[mask]
                            s.setdefault(label, []).append(float(np.sqrt(np.mean(e**2))))
        # cold tip (coldest cryocooler node) / global coldest
        if cryo_idx:
            s.setdefault("cryo_tip_K", []).append(float(np.min(temps[cryo_idx])))
        # Power balance from the engine's own accounting: heater injection in,
        # cryocooler removal out, net radiative load, and net_W (~ dU/dt).
        try:
            pb = prepared.power_balance_W()
            p_in = float(pb.get("heater_W", 0.0))
            p_out = float(pb.get("cryocooler_W", 0.0))
            p_rad = float(pb.get("radiation_W", 0.0))
            net_W = float(pb.get("net_W", p_in + p_rad - p_out))
        except Exception:
            p_in = p_out = p_rad = net_W = float("nan")
        s.setdefault("power_in_W", []).append(p_in)
        # Commanded-but-undelivered heater power: nonzero only when a heater is
        # driving a quarantined cell. A persistent nonzero value means an actuator
        # is burning its full authority on nothing.
        try:
            s.setdefault("heater_undelivered_W", []).append(float(pb.get("heater_undelivered_W", 0.0)))
        except Exception:
            s.setdefault("heater_undelivered_W", []).append(float("nan"))
        s.setdefault("power_out_W", []).append(p_out)
        s.setdefault("radiation_W", []).append(p_rad)
        s.setdefault("net_W", []).append(net_W)
        # Per-heater commanded power, so the report can plot each heater's own
        # trajectory (not just the total power_in_W). One heater_<id>_W series per
        # heater, mirroring the per-sensor temperature series above.
        #
        # Read the ACTUATOR command, not the deposited source vector. A heater
        # deposits its power onto its power_deposition_node_ids (the body cells it
        # touches), not onto its own marker node, so indexing the source vector at
        # the heater row reports 0 W for every heater that has deposition nodes --
        # i.e. every real heater -- even while the controller is driving hundreds
        # of watts into the model.
        try:
            heater_power = prepared.heater_actuator_power_by_node()
        except Exception:
            heater_power = None
        if heater_power is not None:
            heater_set = {int(h) for h in heaters}
            for node_id, value in heater_power.items():
                if int(node_id) in heater_set:
                    s.setdefault(f"heater_{int(node_id)}_W", []).append(float(value))
            if thr.forbid_negative_heater_power:
                neg = [k for k, v in heater_power.items()
                       if int(k) in heater_set and float(v) < -1e-9]
                if neg:
                    self._log_event("controller_check", f"negative heater power on nodes {neg[:5]}")
        # Energy-conservation drift (soft, silent-failure detector): the engine's
        # net power INTO the system should match the observed dU/dt. Skip the first
        # couple of steps (start-up transient of the implicit integrator).
        #
        # Use the LIVE capacitance the solve actually integrates with, not the
        # build-time C. With temperature-dependent properties the two differ a lot
        # (cryogenic cp is ~10x smaller than at build/room temperature), so the
        # static C made dU/dt read ~10x too high -- a spurious "energy drift ~0.9"
        # that is a metrology error, not a physics one.
        inv_C = getattr(prepared, "inv_C", None)
        if inv_C is not None:
            inv_C = np.asarray(inv_C, dtype=float).reshape(-1)
            C_eff = np.where(inv_C > 0.0, 1.0 / np.where(inv_C > 0.0, inv_C, 1.0), C_diag)
        else:
            C_eff = C_diag
        # Compare dU/dt against the power that DROVE this step, i.e. the balance
        # sampled at the previous call. _collect runs after the step, so
        # power_balance_W() above reflects the new state -- the power that will
        # drive the NEXT step. Comparing it to the ΔT just taken is an off-by-one:
        # with a controller whose command moves ~20 W between steps it manufactured
        # a steady "drift" of ~0.1-0.28 on a run that conserved energy fine. Keep
        # the current sample for the next call.
        driving_net_W = getattr(self, "_prev_net_W", net_W)
        self._prev_net_W = net_W
        if dt > 0 and prev.shape == temps.shape and C_eff.shape == temps.shape and len(s["time_s"]) > 2:
            dU_dt = float(np.dot(C_eff, temps - prev) / dt)
            scale = max(abs(driving_net_W), abs(dU_dt), 1.0)
            drift = (
                abs(driving_net_W - dU_dt) / scale if np.isfinite(driving_net_W) else float("nan")
            )
            s.setdefault("energy_drift_rel", []).append(float(drift))
            if np.isfinite(drift) and drift > thr.energy_drift_rel_tol:
                self._log_event(
                    "energy_drift",
                    f"t={state.time_s:.1f}s drift={drift:.3f} (driving net={driving_net_W:.3g}W dU/dt={dU_dt:.3g}W)",
                )
            # Sustained gross drift = the solve is diverging (energy created/lost from
            # nowhere); track a run of it so the loop can abort fast.
            if np.isfinite(drift) and drift > thr.energy_drift_abort_rel:
                self._consecutive_high_drift += 1
            else:
                self._consecutive_high_drift = 0

    # -- checkpoint / resume ------------------------------------------------ #
    def _checkpoint(self, prepared, state, step) -> None:
        try:
            temps = (
                np.asarray(state.temperatures_K, dtype=float)
                if state is not None
                else np.asarray(prepared.z[:-1], dtype=float)
            )
            t = float(state.time_s) if state is not None else 0.0
            # The CONTROLLER's state is part of the simulation state, not incidental
            # to it. Checkpointing only temperatures made a resume silently restart
            # the controller cold: the integral (which holds the accumulated holding
            # power) went back to zero, and MIMO PI re-captured its passive reference
            # from the resumed temperatures with u_prev = 0 -- i.e. it took a plant
            # already warmed by 20 W to BE the unheated equilibrium, and sized the
            # feedforward against that. Resuming is exactly when this matters.
            heater_ids = sorted(prepared.controller_last_power_by_heater)
            np.savez(
                self.ckpt_dir / f"ckpt_{step:08d}.npz",
                temperatures_K=temps,
                time_s=t,
                step=step,
                controller_heater_ids=np.array(heater_ids, dtype=np.int64),
                controller_last_power_W=np.array(
                    [prepared.controller_last_power_by_heater[h] for h in heater_ids],
                    dtype=float,
                ),
                **_optional_state_arrays(prepared),
            )
        except Exception as exc:  # noqa: BLE001
            self._log_event("checkpoint_error", str(exc))

    def _resume_if_checkpoint(self, prepared) -> None:
        # Resume only from a checkpoint under a PRE-EXISTING run dir passed as
        # output; a fresh timestamped dir has none, so this is a no-op for new runs.
        ckpts = sorted(self.ckpt_dir.glob("ckpt_*.npz")) if self.ckpt_dir.exists() else []
        if not ckpts:
            return
        data = np.load(ckpts[-1])
        prepared.set_temperatures(np.asarray(data["temperatures_K"], dtype=float))
        restored = _restore_controller_state(prepared, data)
        self._log_event(
            "resumed",
            f"from {ckpts[-1].name} at t={float(data['time_s']):.1f}s"
            + (f"; controller state restored ({restored})" if restored else
               "; NO controller state in this checkpoint (written before it was saved) -- "
               "the controller starts cold and must re-wind its integral"),
        )

    def _current_time(self, prepared, step, dt) -> float:
        ts = self._series.get("time_s")
        return ts[-1] if ts else 0.0

    # -- status / events / signals ------------------------------------------ #
    def _update_status(self, step, sim_time) -> None:
        wall = time.time() - self._start_wall
        frac = min(1.0, sim_time / self.cfg.t_final_s) if self.cfg.t_final_s > 0 else 0.0
        eta = (wall / frac - wall) if frac > 1e-6 else float("nan")
        payload = {
            "status": self._exit_status,
            "step": step,
            "sim_time_s": sim_time,
            "t_final_s": self.cfg.t_final_s,
            "progress": frac,
            "wall_elapsed_s": wall,
            "eta_s": eta,
            "rss_gib": round(_process_rss_gib(), 2),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        for key in ("rms_tracking_error_K", "max_temp_K", "cryo_tip_K", "power_in_W"):
            if self._series.get(key):
                payload[f"last_{key}"] = self._series[key][-1]
        profile = getattr(self, "_last_step_profile", None)
        if profile:
            for key in ("property_rebuild_ms", "model_solve_ms", "total_ms"):
                if key in profile:
                    payload[key] = round(float(profile[key]), 1)
        _atomic_write_json(self.status_path, payload)
        if self.progress_cb is not None:
            try:
                self.progress_cb(payload)
            except Exception:
                pass

    def _log_event(self, kind: str, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {kind}: {message}\n"
        try:
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ANN001
            self._stop = True
            self._exit_status = f"terminated (signal {signum})"
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass  # not in main thread (e.g. GUI worker) -> caller uses cancel_event

    def _should_stop(self) -> bool:
        if self._stop:
            return True
        if self.cancel_event is not None and getattr(self.cancel_event, "is_set", lambda: False)():
            self._exit_status = "cancelled"
            return True
        # Cross-process graceful stop for a detached/console-less subprocess, where
        # a signal cannot reach us and terminate() would skip _finalize.
        try:
            if (self.out_dir / STOP_REQUEST_FILENAME).exists():
                self._exit_status = "stopped"
                return True
        except OSError:
            pass
        return False

    # -- finalize ----------------------------------------------------------- #
    def _write_config_and_provenance(self) -> None:
        _atomic_write_json(self.out_dir / "config.json", asdict(self.cfg))
        provenance = {
            "graph_name": self.graph_name,
            "graph_folder": str(self.graph_folder),
            "graph_hash": _graph_hash(self.graph_folder),
            "git_commit": _git_commit(),
            "python": sys.version.split()[0],
            "started": datetime.now().isoformat(timespec="seconds"),
            "seed": self.cfg.seed,
            "params_override": self.cfg.params is not None,
        }
        if self._initial_state is not None:
            try:
                temps = np.asarray(self._initial_state[1], dtype=float)
                provenance["initial_state"] = {
                    "source": "in-memory model (out-of-band)",
                    "n": int(temps.size),
                    "min_K": float(temps.min()),
                    "mean_K": float(temps.mean()),
                    "max_K": float(temps.max()),
                }
            except Exception:  # noqa: BLE001
                pass
        elif self.cfg.initial_temperature_uniform_K is not None:
            provenance["initial_state"] = {
                "source": "uniform",
                "temperature_K": float(self.cfg.initial_temperature_uniform_K),
            }
        _atomic_write_json(self.out_dir / "provenance.json", provenance)
        np.random.seed(self.cfg.seed)

    def _finalize(self) -> None:
        source = getattr(self, "_verification_source", None)
        if source is not None:
            try:
                self._write_verification_tables(source)
            except Exception as exc:  # noqa: BLE001 - never lose the run over a report
                self._log_event("verification_failed", f"{type(exc).__name__}: {exc}")
        self._write_timeseries()
        self._write_plots_and_report()
        self._update_status(
            len(self._series.get("time_s", [])),
            self._series.get("time_s", [0.0])[-1] if self._series.get("time_s") else 0.0,
        )
        self._log_event("run_end", f"status={self._exit_status}")

    def _write_timeseries(self) -> None:
        if not self._series.get("time_s"):
            return
        arrs = {k: np.asarray(v, dtype=float) for k, v in self._series.items()}
        np.savez(self.out_dir / "timeseries.npz", **arrs)
        keys = ["time_s"] + [k for k in arrs if k != "time_s"]
        n = len(arrs["time_s"])
        with (self.out_dir / "timeseries.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for i in range(n):
                w.writerow([f"{arrs[k][i]:.6g}" if i < len(arrs[k]) else "" for k in keys])

    def _write_plots_and_report(self) -> None:
        series = self._series
        summary = self._summary_metrics()
        # plots (optional -- guarded so a missing matplotlib never kills a run)
        plotted: list[str] = []
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def _legend(ax) -> None:
                """Adaptive legend that fits: shrink + add columns as the series
                count grows, and for many series (e.g. 90+ sensors) cap the entries,
                prioritise the aggregate curves (rms/avg/max over the per-sensor
                lines), and move it outside the axes so it never covers or overflows
                the plot (bbox_inches='tight' grows the canvas to include it)."""
                handles, labels = ax.get_legend_handles_labels()
                n = len(labels)
                if n == 0:
                    return
                if n <= 8:
                    ax.legend(fontsize=7, ncol=1, loc="best")
                    return
                if n <= 24:
                    ax.legend(fontsize=6, ncol=2, loc="best")
                    return
                # Many series: aggregates first (rms/avg/max/min/power/...), then the
                # bulk per-sensor lines; cap and place outside on the right.
                order = sorted(range(n), key=lambda i: (labels[i].startswith("sensor_"), i))
                handles = [handles[i] for i in order]
                labels = [labels[i] for i in order]
                cap = 16
                ax.legend(
                    handles[:cap], labels[:cap],
                    title=f"{n} series (first {min(cap, n)})",
                    fontsize=5, title_fontsize=6, ncol=1,
                    loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=False, borderaxespad=0.0,
                )

            def _plot(keys, title, ylabel, fname):
                present = [k for k in keys if series.get(k)]
                if not present or not series.get("time_s"):
                    return
                t = series["time_s"]
                fig, ax = plt.subplots(figsize=(9, 4))
                for k in present:
                    ax.plot(t[: len(series[k])], series[k], label=k)
                ax.set_xlabel("time [s]"); ax.set_ylabel(ylabel); ax.set_title(title)
                ax.grid(True, alpha=0.3)
                _legend(ax)
                fig.tight_layout()
                fig.savefig(self.plots_dir / fname, dpi=110, bbox_inches="tight")
                plt.close(fig)
                plotted.append(fname)

            def _heater_id(key: str) -> int:
                try:
                    return int(key.split("_")[1])
                except (IndexError, ValueError):
                    return 0

            # Only the CONTROLLED sensors. A monitor-only sensor has no heater acting
            # on it, so plotting it alongside the regulated ones buries the loop being
            # tuned under traces nothing is steering (64 of 91 here).
            controlled = getattr(self, "_controlled_series_keys", None)
            def _is_controlled(key: str) -> bool:
                return controlled is None or key.replace("_err_K", "_K") in controlled

            sensor_keys = sorted(
                k for k in series
                if k.startswith("sensor_") and k.endswith("_K") and _is_controlled(k)
            )
            err_keys = sorted(
                k for k in series
                if k.endswith("_err_K") and k.startswith("sensor_") and _is_controlled(k)
            )
            heater_keys = sorted(
                (k for k in series if k.startswith("heater_") and k.endswith("_W")), key=_heater_id
            )
            _plot(sensor_keys, "Controlled sensor temperatures", "T [K]", "sensor_temps.png")
            _plot(err_keys + ["rms_tracking_error_controlled_K"], "Tracking error (controlled sensors)", "error [K]", "tracking_error.png")
            _plot(["avg_temp_K", "max_temp_K", "min_temp_K"], "System temperature", "T [K]", "system_temp.png")
            _plot(["cryo_tip_K"], "Cryo tip temperature", "T [K]", "cryo_tip.png")
            _plot(heater_keys, "Heater power (per heater)", "power [W]", "heater_power.png")
            _plot(["power_in_W", "power_out_W", "net_W"], "Power balance", "power [W]", "power_balance.png")
            _plot(["max_temp_rate_K_per_s"], "Max cell temperature rate", "dT/dt [K/s]", "temp_rate.png")
            _plot(["energy_drift_rel"], "Energy-conservation drift (should stay small)", "rel drift", "energy_drift.png")
        except Exception as exc:  # noqa: BLE001
            self._log_event("plot_skipped", str(exc))

        # report
        lines = [
            f"# Simulation report — {self.graph_name}",
            "",
            f"- **Status:** {self._exit_status}",
            f"- **Output:** `{self.out_dir}`",
            f"- **Wall time:** {time.time() - self._start_wall:.1f} s",
            f"- **Sim time reached:** {summary.get('sim_time_s', 0.0):.1f} / {self.cfg.t_final_s:.1f} s",
            f"- **Steps:** {summary.get('steps', 0)}",
            "",
            "## Summary metrics",
        ]
        for k, v in summary.items():
            lines.append(f"- {k}: {v:.6g}" if isinstance(v, float) else f"- {k}: {v}")
        lines += ["", "## Plots"]
        lines += [f"- ![{f}](plots/{f})" for f in plotted] or ["- (none)"]
        lines += ["", "## Events", "See `events.log`."]
        (self.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    def _summary_metrics(self) -> dict[str, Any]:
        s = self._series
        out: dict[str, Any] = {"steps": len(s.get("time_s", []))}
        if s.get("time_s"):
            out["sim_time_s"] = s["time_s"][-1]
        if s.get("rms_tracking_error_K"):
            out["final_rms_tracking_error_K"] = s["rms_tracking_error_K"][-1]
            out["peak_rms_tracking_error_K"] = max(s["rms_tracking_error_K"])
        if s.get("max_temp_K"):
            out["peak_max_temp_K"] = max(s["max_temp_K"])
        if s.get("cryo_tip_K"):
            out["final_cryo_tip_K"] = s["cryo_tip_K"][-1]
        if s.get("power_in_W"):
            out["peak_power_in_W"] = max(v for v in s["power_in_W"] if np.isfinite(v)) if any(np.isfinite(v) for v in s["power_in_W"]) else float("nan")
        if s.get("energy_drift_rel"):
            out["max_energy_drift_rel"] = max(s["energy_drift_rel"])
        out.update(getattr(self, "_quarantine_summary", {}) or {})
        if s.get("heater_undelivered_W"):
            finite = [v for v in s["heater_undelivered_W"] if np.isfinite(v)]
            if finite:
                out["peak_heater_undelivered_W"] = max(finite)
        return out



def _optional_state_arrays(prepared) -> dict:
    """Controller integrator state, when the active scheme has any.

    Stored per scheme rather than as one blob because the two are not
    interchangeable: MIMO PI's integral is indexed by controlled sensor, the modal
    one by heater. Loading the wrong one would be worse than loading neither.
    """
    out = {}
    for name in (
        "controller_mimo_pi_integral",
        "controller_mimo_pi_passive_K",
        "controller_modal_integral",
    ):
        value = getattr(prepared, name, None)
        if value is not None:
            out[name] = np.asarray(value, dtype=float)
    return out


def _restore_controller_state(prepared, data) -> str:
    """Put a checkpoint's controller state back. Returns what was restored.

    Silently tolerates checkpoints written before this existed, and any array whose
    length no longer matches the current controller -- a resume with a different G
    must start clean rather than apply gains to the wrong channels.
    """
    restored = []
    ids = data["controller_heater_ids"] if "controller_heater_ids" in data else None
    if ids is not None and ids.size:
        powers = np.asarray(data["controller_last_power_W"], dtype=float)
        prepared.controller_last_power_by_heater = {
            int(h): float(p) for h, p in zip(ids, powers)
        }
        restored.append(f"{ids.size} heater command(s)")
    expected = {
        "controller_mimo_pi_integral": len(getattr(prepared, "_mimo_pi_sensor_ids", []) or []),
        "controller_mimo_pi_passive_K": len(getattr(prepared, "_mimo_pi_sensor_ids", []) or []),
    }
    for name in (
        "controller_mimo_pi_integral",
        "controller_mimo_pi_passive_K",
        "controller_modal_integral",
    ):
        if name not in data:
            continue
        value = np.asarray(data[name], dtype=float)
        want = expected.get(name, 0)
        if want and value.shape[0] != want:
            continue  # a different controller: start clean
        setattr(prepared, name, value)
        restored.append(name.replace("controller_", ""))
    return ", ".join(restored)


class _HardFailure(Exception):
    """A failure that should abort the run (after finalizing artifacts)."""


def run_simulation(config: RunConfig, cancel_event: Any | None = None,
                   progress_cb: Callable[[dict], None] | None = None,
                   initial_state: tuple[Any, Any] | None = None) -> Path:
    """Convenience entry point: run one closed-loop simulation, return its output dir."""
    return SimulationRunner(
        config, cancel_event=cancel_event, progress_cb=progress_cb, initial_state=initial_state
    ).run()
