"""Compare implicit sparse CPU simulation against expm_multiply reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_visualizer.simulation_diagnostics import (
    compare_graph_folder_steppers,
    save_stepper_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare implicit sparse CPU trajectory against CPU expm_multiply reference."
    )
    parser.add_argument("graph_folder", type=Path, help="Graph folder containing graph.json and matrix payloads.")
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of simulation steps to compare. Ignored when --to-end is set.",
    )
    parser.add_argument(
        "--to-end",
        action="store_true",
        help="Run through t_final_s from simulation_parameters.json.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional simulation_parameters.json path. Defaults to graph_folder/simulation_parameters.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for .npy trajectory/error matrices and summary.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_graph_folder_steppers(
        args.graph_folder,
        steps=args.steps,
        to_end=bool(args.to_end),
        params_path=args.params,
    )
    summary = {
        "metrics": result.metrics.__dict__,
        "implicit_profile_ms": result.implicit_profile_ms,
        "reference_profile_ms": result.reference_profile_ms,
    }
    if args.output_dir is not None:
        output_dir = save_stepper_comparison(result, args.output_dir)
        summary["output_dir"] = str(output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
