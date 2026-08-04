"""Headless closed-loop simulation runner (overnight, no UI).

Example:
    python run_simulation.py --graph graphs/CRYOSTAT_V2 --setpoint 80 --duration 3600 --dt 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from graph_visualizer.simulation_runner import RunConfig, FailureThresholds, run_simulation


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, help="path to graphs/<name> folder")
    p.add_argument("--output-root", default="simulations")
    p.add_argument("--controller", default=None, help="modal_controller.npz (default: <graph>/modal_controller.npz)")
    p.add_argument("--allow-no-controller", action="store_true", help="run open-loop if no controller is present")
    p.add_argument("--setpoint", type=float, default=None, help="constant setpoint [K] applied to every sensor")
    p.add_argument("--initial-temp", type=float, default=None,
                   help="uniform initial temperature [K] for every node (overrides the graph's saved temps)")
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=3600.0, help="t_final [s]")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--snapshot-interval-s", type=float, default=300.0)
    p.add_argument("--checkpoint-interval-s", type=float, default=600.0)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    cfg = RunConfig(
        graph_folder=str(Path(args.graph)),
        output_root=args.output_root,
        controller_path=args.controller,
        allow_no_controller=bool(args.allow_no_controller),
        global_setpoint_K=args.setpoint,
        initial_temperature_uniform_K=args.initial_temp,
        dt_s=args.dt,
        t_final_s=args.duration,
        gpu_solver_enabled=not args.no_gpu,
        snapshot_interval_s=args.snapshot_interval_s,
        checkpoint_interval_s=args.checkpoint_interval_s,
        notes=args.notes,
        thresholds=FailureThresholds(),
    )
    out_dir = run_simulation(cfg)
    print(f"\nOutput: {out_dir}")
    status = (out_dir / "status.json")
    if status.exists():
        print(status.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
