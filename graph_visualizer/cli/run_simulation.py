"""Headless closed-loop simulation runner (overnight, no UI).

Example:
    python run_simulation.py --graph graphs/CRYOSTAT_V2 --setpoint 80 --duration 3600 --dt 1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from graph_visualizer.simulation_parameters import load_simulation_parameters
from graph_visualizer.simulation_runner import RunConfig, FailureThresholds, run_simulation


def _enable_crash_traceback(run_dir: str | None) -> None:
    """Dump a C-level traceback to <run_dir>/crash.log if the process dies natively.

    A segfault / access violation in a compiled extension kills the interpreter
    outright: no Python exception, no finally, no report -- the run directory just
    stops, with status.json still saying "running". That is exactly how a 760-step
    run vanished with nothing to point at. faulthandler installs OS-level signal
    handlers that write a stack for every thread before the process dies, which is
    the only evidence available for this class of failure.

    The file handle is deliberately left open for the life of the process: the
    handler runs during a crash, when opening a file is not safe.
    """
    if not run_dir:
        return
    try:
        import faulthandler

        target = Path(run_dir)
        target.mkdir(parents=True, exist_ok=True)
        handle = open(target / "crash.log", "a", encoding="utf-8")  # noqa: SIM115
        handle.write(f"--- run started {datetime.now().isoformat(timespec='seconds')} ---\n")
        handle.flush()
        faulthandler.enable(file=handle, all_threads=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never block a run
        print(f"Could not enable crash tracebacks: {exc}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, help="path to graphs/<name> folder")
    p.add_argument("--output-root", default="simulations")
    p.add_argument("--controller", default=None, help="modal_controller.npz (default: <graph>/modal_controller.npz)")
    p.add_argument("--allow-no-controller", action="store_true", help="run open-loop if no controller is present")
    p.add_argument("--setpoint", type=float, default=None, help="constant setpoint [K] applied to every sensor")
    p.add_argument("--setpoints-json", default=None,
                   help="JSON {sensor_node_id: setpoint_K} of PER-SENSOR targets. Applied on top "
                        "of --setpoint, so --setpoint sets the baseline and this overrides "
                        "individual sensors.")
    p.add_argument("--heater-overrides-json", default=None,
                   help="JSON {heater_node_id: {heater_max_power_W, heater_slew_rate_W_per_s, "
                        "heater_efficiency}} of PER-HEATER limit overrides. Any field omitted "
                        "keeps the run's controller defaults, and a heater not named here is "
                        "untouched.")
    p.add_argument("--initial-temp", type=float, default=None,
                   help="uniform initial temperature [K] for every node (overrides the graph's saved temps)")
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=3600.0, help="t_final [s]")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--snapshot-interval-s", type=float, default=300.0)
    p.add_argument("--checkpoint-interval-s", type=float, default=600.0)
    p.add_argument("--notes", default="")
    p.add_argument("--run-dir", default=None,
                   help="exact output directory (default: simulations/<graph>/<timestamp>)")
    p.add_argument("--sim-params", default=None,
                   help="simulation_parameters.json giving the FULL physics/solver settings "
                        "(radiation, temperature-dependent properties, cryocooler, solver, "
                        "controller gains). Defaults to the graph's own saved file if present, "
                        "so a headless run matches what the GUI is configured to do.")
    args = p.parse_args()

    _enable_crash_traceback(args.run_dir)

    sim_params = None
    params_path = Path(args.sim_params) if args.sim_params else Path(args.graph) / "simulation_parameters.json"
    if params_path.is_file():
        sim_params, _extras = load_simulation_parameters(params_path)
        print(f"Using simulation parameters from {params_path}")

    # Per-sensor targets are a plain {node_id: K} map. They layer on top of
    # --setpoint, which stays the baseline for every sensor not named here.
    per_sensor_setpoints: dict[int, float] = {}
    if args.setpoints_json:
        setpoints_path = Path(args.setpoints_json)
        if not setpoints_path.is_file():
            raise SystemExit(f"--setpoints-json not found: {setpoints_path}")
        with setpoints_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise SystemExit("--setpoints-json must contain a JSON object {node_id: setpoint_K}.")
        for node_id, target in raw.items():
            try:
                per_sensor_setpoints[int(node_id)] = float(target)
            except (TypeError, ValueError):
                raise SystemExit(f"--setpoints-json has a bad entry: {node_id!r}: {target!r}")
        print(f"Per-sensor setpoints: {len(per_sensor_setpoints)} sensor(s) from {setpoints_path}")

    # Per-heater limit overrides: {node_id: {field: value}}. Only the fields present
    # are applied, so a heater given a slew rate keeps the default max power.
    heater_overrides: dict[int, dict[str, float]] = {}
    if args.heater_overrides_json:
        overrides_path = Path(args.heater_overrides_json)
        if not overrides_path.is_file():
            raise SystemExit(f"--heater-overrides-json not found: {overrides_path}")
        with overrides_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise SystemExit(
                "--heater-overrides-json must contain a JSON object {node_id: {field: value}}."
            )
        for node_id, fields in raw.items():
            if not isinstance(fields, dict):
                raise SystemExit(f"--heater-overrides-json has a bad entry: {node_id!r}: {fields!r}")
            try:
                heater_overrides[int(node_id)] = {
                    str(name): float(value) for name, value in fields.items()
                }
            except (TypeError, ValueError):
                raise SystemExit(f"--heater-overrides-json has a bad entry: {node_id!r}: {fields!r}")
        print(f"Per-heater overrides: {len(heater_overrides)} heater(s) from {overrides_path}")

    # Checked here, AFTER the --setpoints-json / --heater-overrides-json files
    # have been parsed and reported: a malformed argument is the user's mistake to
    # hear about first, whatever graph they aimed at. But still BEFORE
    # run_simulation, because that creates simulations/<name>/<timestamp>/ before
    # it discovers the graph is missing -- so a typo used to leave a litter
    # directory holding a status.json that recorded the failure, and then exit 0.
    graph_folder = Path(args.graph)
    if not graph_folder.is_dir():
        print(f"error: no such graph folder: {graph_folder}", file=sys.stderr)
        return 2
    if not any((graph_folder / name).is_file() for name in ("graph.json", "graph3d.json")):
        print(
            f"error: {graph_folder} is not a graph folder (no graph.json or "
            "graph3d.json). Build one with hts-build-graph.",
            file=sys.stderr,
        )
        return 2

    cfg = RunConfig(
        graph_folder=str(Path(args.graph)),
        output_root=args.output_root,
        run_dir=args.run_dir,
        controller_path=args.controller,
        allow_no_controller=bool(args.allow_no_controller),
        global_setpoint_K=args.setpoint,
        setpoints_K=per_sensor_setpoints,
        heater_overrides=heater_overrides,
        initial_temperature_uniform_K=args.initial_temp,
        dt_s=args.dt,
        t_final_s=args.duration,
        gpu_solver_enabled=not args.no_gpu,
        snapshot_interval_s=args.snapshot_interval_s,
        checkpoint_interval_s=args.checkpoint_interval_s,
        notes=args.notes,
        params=sim_params,
        thresholds=FailureThresholds(),
    )
    out_dir = run_simulation(cfg)
    print(f"\nOutput: {out_dir}")
    status_path = out_dir / "status.json"
    if not status_path.exists():
        print("error: the run wrote no status.json", file=sys.stderr)
        return 1
    text = status_path.read_text(encoding="utf-8")
    print(text)
    # The runner reports failure INSIDE the status file rather than by raising,
    # so the exit code has to be derived from it. Without this, every run exited
    # 0 -- a crashed overnight job was indistinguishable from a completed one,
    # and a batch script would carry on to the next stage regardless.
    try:
        verdict = str(json.loads(text).get("status", ""))
    except ValueError:
        print("error: status.json is not valid JSON", file=sys.stderr)
        return 1
    if verdict != "completed":
        print(f"error: run did not complete ({verdict})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
