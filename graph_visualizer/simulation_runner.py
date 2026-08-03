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
from dataclasses import asdict, dataclass, field
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
    forbid_negative_heater_power: bool = True  # soft: controller commanded cooling
    max_dt_retries: int = 6  # step-back halvings before giving up (hard)
    step_wall_timeout_s: float = 0.0  # 0 = disabled watchdog


@dataclass
class RunConfig:
    graph_folder: str
    output_root: str = "simulations"
    controller_path: str | None = None  # modal_controller.npz; None => confirm/open-loop
    allow_no_controller: bool = False  # must be True to run without a controller
    setpoints_K: dict[int, float] = field(default_factory=dict)  # sensor node id -> target
    global_setpoint_K: float | None = None  # applied to every sensor if set
    dt_s: float = 1.0
    t_final_s: float = 3600.0
    gpu_solver_enabled: bool = True
    log_interval_steps: int = 1
    snapshot_interval_s: float = 300.0
    checkpoint_interval_s: float = 600.0  # wall-clock seconds between checkpoints
    status_interval_steps: int = 20
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
    ) -> None:
        self.cfg = config
        self.cancel_event = cancel_event  # anything with .is_set(); optional
        self.progress_cb = progress_cb
        self.graph_folder = Path(config.graph_folder)
        self.graph_name = self.graph_folder.name
        self.out_dir = Path(config.output_root) / self.graph_name / _timestamp()
        self.snap_dir = self.out_dir / "snapshots"
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.plots_dir = self.out_dir / "plots"
        self.events_path = self.out_dir / "events.log"
        self.status_path = self.out_dir / "status.json"
        self._series: dict[str, list[float]] = {}
        self._stop = False  # set by SIGTERM/SIGINT or cancel_event
        self._exit_status = "running"
        self._start_wall = time.time()

    # -- lifecycle ---------------------------------------------------------- #
    def run(self) -> Path:
        for d in (self.out_dir, self.snap_dir, self.ckpt_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)
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

    # -- setup -------------------------------------------------------------- #
    def _prepare(self):
        model, matrices = load_graph_folder(str(self.graph_folder))
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

        params = SimulationParameters(
            dt_s=float(self.cfg.dt_s),
            t_final_s=float(self.cfg.t_final_s),
            gpu_solver_enabled=bool(self.cfg.gpu_solver_enabled),
            input_mode="heater_inputs" if has_controller else "zero",
            mimo_controller_enabled=has_controller,
            mimo_controller_scheme="modal_lqr" if has_controller else "pid_qp",
            modal_controller_path=controller_path or "",
        )
        prepared = simulation_model.prepare_simulation(model, matrices, params)
        C_diag = np.asarray(matrices["C"], dtype=float).reshape(-1)
        heaters = [int(x) for x in getattr(prepared, "heater_node_ids", ())]
        cryo_idx = [
            prepared.node_index_by_id[int(x)]
            for x in getattr(prepared, "cryocooler_node_ids", ())
            if int(x) in prepared.node_index_by_id
        ]
        self._log_event(
            "prepared",
            f"nodes={len(C_diag)} sensors={len(sensors)} heaters={len(heaters)} "
            f"cryo={len(cryo_idx)} controller={'yes' if has_controller else 'OPEN-LOOP'}",
        )
        self._resume_if_checkpoint(prepared)
        return prepared, params, C_diag, sensors, heaters, cryo_idx

    def _apply_setpoints(self, model, sensors: list[int]) -> None:
        if self.cfg.global_setpoint_K is not None:
            for sid in sensors:
                model.nodes[sid].controller_setpoint_K = float(self.cfg.global_setpoint_K)
        for sid, target in (self.cfg.setpoints_K or {}).items():
            if int(sid) in model.nodes:
                model.nodes[int(sid)].controller_setpoint_K = float(target)

    # -- main loop ---------------------------------------------------------- #
    def _simulate(self, prepared, params, C_diag, sensors, heaters, cryo_idx) -> None:
        thr = self.cfg.thresholds
        node_index = prepared.node_index_by_id
        sensor_ix = [node_index[s] for s in sensors if s in node_index]
        setpoints = np.array(
            [float(getattr(prepared.model.nodes[s], "controller_setpoint_K", np.nan)) for s in sensors],
            dtype=float,
        )
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
                    f"step {step}: solver/step failed after {thr.max_dt_retries} dt halvings "
                    f"(down to {attempt_dt:g}s). Last state checkpointed."
                )
            params.dt_s = base_dt  # recover after a successful accept

            # --- record ---
            self._collect(prepared, state, temps, prev_temps, attempt_dt,
                          C_diag, sensor_ix, setpoints, heaters, cryo_idx, thr)
            prev_temps = temps.copy()
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

        if step == 0:
            self._log_event(
                "no_steps",
                f"loop ran 0 steps (stopped/cancelled during setup, or t_final<=0). "
                f"t_final={self.cfg.t_final_s:g}s",
            )
        # final checkpoint + status
        self._checkpoint(prepared, state if accepted else None, step)
        self._update_status(step, self._series.get("time_s", [0.0])[-1] if self._series.get("time_s") else 0.0)

    def _step_is_bad(self, temps, prev, dt, thr) -> tuple[bool, str]:
        if not np.all(np.isfinite(temps)):
            return True, "non-finite temperatures (NaN/Inf)"
        tmax = float(np.max(temps))
        tmin = float(np.min(temps))
        if tmax > thr.max_temperature_K:
            return True, f"max T {tmax:.3g}K > {thr.max_temperature_K:g}K"
        if tmin < thr.min_temperature_K:
            return True, f"min T {tmin:.3g}K < {thr.min_temperature_K:g}K"
        if dt > 0 and prev.shape == temps.shape:
            rate = float(np.max(np.abs(temps - prev))) / dt
            if rate > thr.max_temperature_rate_K_per_s:
                return True, f"|dT/dt| {rate:.3g}K/s > {thr.max_temperature_rate_K_per_s:g}"
        return False, ""

    # -- readouts / logging ------------------------------------------------- #
    def _collect(self, prepared, state, temps, prev, dt, C_diag, sensor_ix,
                 setpoints, heaters, cryo_idx, thr) -> None:
        s = self._series
        s.setdefault("time_s", []).append(float(state.time_s))
        s.setdefault("avg_temp_K", []).append(float(np.mean(temps)))
        s.setdefault("max_temp_K", []).append(float(np.max(temps)))
        s.setdefault("min_temp_K", []).append(float(np.min(temps)))
        # sensor temps + tracking error
        if sensor_ix:
            sens = temps[sensor_ix]
            for j, ix in enumerate(sensor_ix):
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
        if thr.forbid_negative_heater_power:
            try:
                neg = [k for k, v in prepared.heater_power_by_node().items()
                       if int(k) in set(heaters) and float(v) < -1e-9]
                if neg:
                    self._log_event("controller_check", f"negative heater power on nodes {neg[:5]}")
            except Exception:
                pass
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
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        for key in ("rms_tracking_error_K", "max_temp_K", "cryo_tip_K", "power_in_W"):
            if self._series.get(key):
                payload[f"last_{key}"] = self._series[key][-1]
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
        return False

    # -- finalize ----------------------------------------------------------- #
    def _write_config_and_provenance(self) -> None:
        _atomic_write_json(self.out_dir / "config.json", asdict(self.cfg))
        _atomic_write_json(
            self.out_dir / "provenance.json",
            {
                "graph_name": self.graph_name,
                "graph_folder": str(self.graph_folder),
                "graph_hash": _graph_hash(self.graph_folder),
                "git_commit": _git_commit(),
                "python": sys.version.split()[0],
                "started": datetime.now().isoformat(timespec="seconds"),
                "seed": self.cfg.seed,
            },
        )
        np.random.seed(self.cfg.seed)

    def _finalize(self) -> None:
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

            def _plot(keys, title, ylabel, fname):
                present = [k for k in keys if series.get(k)]
                if not present or not series.get("time_s"):
                    return
                t = series["time_s"]
                fig, ax = plt.subplots(figsize=(9, 4))
                for k in present:
                    ax.plot(t[: len(series[k])], series[k], label=k)
                ax.set_xlabel("time [s]"); ax.set_ylabel(ylabel); ax.set_title(title)
                ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
                fig.tight_layout(); fig.savefig(self.plots_dir / fname, dpi=110); plt.close(fig)
                plotted.append(fname)

            sensor_keys = sorted(k for k in series if k.startswith("sensor_") and k.endswith("_K"))
            err_keys = sorted(k for k in series if k.endswith("_err_K"))
            _plot(sensor_keys, "Tracked sensor temperatures", "T [K]", "sensor_temps.png")
            _plot(err_keys + ["rms_tracking_error_K"], "Tracking error", "error [K]", "tracking_error.png")
            _plot(["avg_temp_K", "max_temp_K", "min_temp_K"], "System temperature", "T [K]", "system_temp.png")
            _plot(["cryo_tip_K"], "Cryo tip temperature", "T [K]", "cryo_tip.png")
            _plot(["power_in_W", "power_out_W"], "Power in / out", "power [W]", "power_balance.png")
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
                   progress_cb: Callable[[dict], None] | None = None) -> Path:
    """Convenience entry point: run one closed-loop simulation, return its output dir."""
    return SimulationRunner(config, cancel_event=cancel_event, progress_cb=progress_cb).run()
