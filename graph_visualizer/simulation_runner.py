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
                    f"loaded {report.node_count} nodes without graph.json "
                    f"(rss={_process_rss_gib():.1f} GiB)",
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
        # unavailable (...); using PID+QP allocator", which is only raised when the
        # controller is first evaluated, i.e. after this point. Remember how many we
        # have logged so the step loop can surface the rest; otherwise a run could
        # silently use a different controller than the one requested.
        self._logged_warning_count = len(prepared.warnings)
        self._warn_if_disconnected()
        self._warn_if_unforced(prepared, params, has_controller, cryo_idx)
        self._resume_if_checkpoint(prepared)
        return prepared, params, C_diag, sensors, heaters, cryo_idx

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
            mimo_controller_scheme="modal_lqr" if has_controller else "pid_qp",
            modal_controller_path=controller_path or "",
        )
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
            for index, row in enumerate(rows):
                try:
                    value = float(temps[self._sensor_rows[index]])
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
        controller is first evaluated and silently degrades to the PID+QP
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
        if not np.all(np.isfinite(temps)):
            return True, "non-finite temperatures (NaN/Inf)"
        tmax = float(np.max(temps))
        tmin = float(np.min(temps))
        if tmax > thr.max_temperature_K:
            return True, f"max T {tmax:.3g}K > {thr.max_temperature_K:g}K"
        if tmin < thr.min_temperature_K:
            return True, f"min T {tmin:.3g}K < {thr.min_temperature_K:g}K"
        return False, ""

    # -- readouts / logging ------------------------------------------------- #
    def _collect(self, prepared, state, temps, prev, dt, C_diag, sensor_ix,
                 setpoints, heaters, cryo_idx, thr) -> None:
        s = self._series
        s.setdefault("time_s", []).append(float(state.time_s))
        s.setdefault("avg_temp_K", []).append(float(np.mean(temps)))
        s.setdefault("max_temp_K", []).append(float(np.max(temps)))
        s.setdefault("min_temp_K", []).append(float(np.min(temps)))
        # Max temperature rate -- soft divergence indicator (logged, not fatal).
        if dt > 0 and prev.shape == temps.shape:
            rate = float(np.max(np.abs(temps - prev))) / dt
            s.setdefault("max_temp_rate_K_per_s", []).append(rate)
            if rate > thr.max_temperature_rate_K_per_s:
                self._log_event("high_temp_rate", f"t={state.time_s:.1f}s max|dT/dt|={rate:.3g}K/s (soft)")
        # sensor temps + tracking error
        if sensor_ix:
            sens = temps[sensor_ix]
            # Each per-sensor series is a Python list held for the whole run; with
            # many sensors x an overnight step count that adds up, so log individual
            # sensors up to a cap (the aggregate RMS below always covers them all).
            logged = min(len(sensor_ix), max(0, int(self.cfg.max_logged_sensors)))
            for j in range(logged):
                ix = sensor_ix[j]
                s.setdefault(f"sensor_{j}_K", []).append(float(temps[ix]))
                if np.isfinite(setpoints[j]):
                    s.setdefault(f"sensor_{j}_err_K", []).append(float(temps[ix] - setpoints[j]))
            valid = np.isfinite(setpoints)
            if valid.any():
                err = sens[valid] - setpoints[valid]
                s.setdefault("rms_tracking_error_K", []).append(float(np.sqrt(np.mean(err**2))))
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
        s.setdefault("power_out_W", []).append(p_out)
        s.setdefault("radiation_W", []).append(p_rad)
        s.setdefault("net_W", []).append(net_W)
        # Per-heater commanded power, so the report can plot each heater's own
        # trajectory (not just the total power_in_W). One heater_<id>_W series per
        # heater, mirroring the per-sensor temperature series above.
        try:
            heater_power = prepared.heater_power_by_node()
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
        if dt > 0 and prev.shape == temps.shape and C_diag.shape == temps.shape and len(s["time_s"]) > 2:
            dU_dt = float(np.dot(C_diag, temps - prev) / dt)
            scale = max(abs(net_W), abs(dU_dt), 1.0)
            drift = abs(net_W - dU_dt) / scale if np.isfinite(net_W) else float("nan")
            s.setdefault("energy_drift_rel", []).append(float(drift))
            if np.isfinite(drift) and drift > thr.energy_drift_rel_tol:
                self._log_event(
                    "energy_drift",
                    f"t={state.time_s:.1f}s drift={drift:.3f} (net={net_W:.3g}W dU/dt={dU_dt:.3g}W)",
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
            np.savez(
                self.ckpt_dir / f"ckpt_{step:08d}.npz",
                temperatures_K=temps,
                time_s=t,
                step=step,
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
        self._log_event("resumed", f"from {ckpts[-1].name} at t={float(data['time_s']):.1f}s")

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

            sensor_keys = sorted(k for k in series if k.startswith("sensor_") and k.endswith("_K"))
            err_keys = sorted(k for k in series if k.endswith("_err_K"))
            heater_keys = sorted(
                (k for k in series if k.startswith("heater_") and k.endswith("_W")), key=_heater_id
            )
            _plot(sensor_keys, "Tracked sensor temperatures", "T [K]", "sensor_temps.png")
            _plot(err_keys + ["rms_tracking_error_K"], "Tracking error", "error [K]", "tracking_error.png")
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
        return out


class _HardFailure(Exception):
    """A failure that should abort the run (after finalizing artifacts)."""


def run_simulation(config: RunConfig, cancel_event: Any | None = None,
                   progress_cb: Callable[[dict], None] | None = None,
                   initial_state: tuple[Any, Any] | None = None) -> Path:
    """Convenience entry point: run one closed-loop simulation, return its output dir."""
    return SimulationRunner(
        config, cancel_event=cancel_event, progress_cb=progress_cb, initial_state=initial_state
    ).run()
