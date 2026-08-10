"""Build a modal-LQR controller artifact WITHOUT loading graph.json.

The Heat Transfer Simulation tab builds controllers from a graph loaded into the
GUI process. For a multi-million-cell graph that means parsing graph.json -- ~10 GB
on disk, ~45 GB once expanded into per-node/per-edge dicts -- and holding all of it
while ``splu`` factorizes the 3M x 3M DC operator. On a 64 GB machine that leaves
only ~15-19 GB for a factorization whose fill-in is hard to predict, and when it
does not fit the symptom is thrashing that looks exactly like a hang (it has
already cost one remote session).

Everything the reduction needs -- C, L, G_rad, node_ids, and the model's
has_cryocooler / heater / sensor roles -- is available from the fast-load
artifacts (``nodes.csv`` + ``edges.npz`` + ``L_sparse.npz`` + ``C.npy``), so this
runs the identical ``design_modal_controller`` at ~20 GB instead of ~45 GB, in a
separate process, with no GUI and no dependence on a remote session staying up.

Example:
    python build_modal_controller.py graphs/no_mli_high_res --t-op 50

Defaults reproduce the descriptors of the existing no_mli_high_res artifact
(r=50, 140 modes, T_op=50 K, effort 0.2, integral gain 0.06).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BUILD_LOG_FILENAME = "build_modal_controller.log"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load(folder: Path, allow_full: bool):
    """(model, matrices). Prefers the low-memory path; that is the whole point."""
    from graph_visualizer.fast_graph_io import can_load_fast, load_graph_for_simulation

    usable, reason = can_load_fast(folder)
    if usable:
        print(f"[{_now()}] loading via the low-memory path (no graph.json) ...", flush=True)
        model, matrices, report = load_graph_for_simulation(folder)
        for warning in report.warnings:
            print(f"[{_now()}]   note: {warning}", flush=True)
        print(
            f"[{_now()}] loaded {report.node_count:,} nodes and {report.edge_count:,} edges",
            flush=True,
        )
        return model, matrices
    if not allow_full:
        raise SystemExit(
            f"Fast load unavailable ({reason}) and --allow-full-load was not passed.\n"
            "Run 'Update graph' (or refresh_fast_load.py) first: the full graph.json loader\n"
            "needs roughly 45 GB on a 3M-cell graph, which is what this script exists to avoid."
        )
    print(f"[{_now()}] fast load unavailable ({reason}); falling back to graph.json "
          f"-- expect very high memory use", flush=True)
    from graph_visualizer.graph_io import load_graph_folder

    model, matrices = load_graph_folder(str(folder))
    return model, matrices


def _describe(path: Path) -> None:
    """Print the checks that say whether the DC grounding fix actually took."""
    try:
        with np.load(path, allow_pickle=False) as data:
            gain = np.asarray(data["dc_gain"], dtype=float)
        columns = np.sort(np.abs(gain).sum(axis=0))
        print(f"[{_now()}] dc_gain: cond={np.linalg.cond(gain):.4g} "
              f"median column sum={np.median(np.abs(gain).sum(axis=0)):.4g} K/W", flush=True)
        print(f"[{_now()}]   four smallest column sums: "
              f"{', '.join(f'{v:.4g}' for v in columns[:4])} K/W", flush=True)
        print(f"[{_now()}]   (a heater grounded at the old fixed 1e3 W/K showed ~0.03 K/W "
              f"against a ~5 K/W median, and carried the whole condition number)", flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail the build
        print(f"[{_now()}] could not summarise dc_gain: {exc}", flush=True)


def build(args: argparse.Namespace) -> Path:
    from graph_visualizer.modal_reduction import design_modal_controller, modal_artifact_filename

    folder = Path(args.graph)
    if not folder.is_dir():
        raise SystemExit(f"Not a graph folder: {folder}")

    out_path = folder / modal_artifact_filename(args.order, args.modes, args.t_op)
    # Rebuilding with identical descriptors intentionally overwrites, so keep the
    # previous artifact: it is the only record of what produced earlier runs.
    if out_path.exists() and not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = out_path.with_name(f"{out_path.stem}.{stamp}.bak.npz")
        shutil.copy2(out_path, backup)
        print(f"[{_now()}] backed up existing artifact -> {backup.name}", flush=True)

    model, matrices = _load(folder, allow_full=bool(args.allow_full_load))
    for key in ("C", "L", "node_ids"):
        if key not in matrices:
            raise SystemExit(f"Graph is missing the '{key}' matrix needed for reduction.")

    capacitance = np.asarray(matrices["C"], dtype=float).reshape(-1)
    radiation = np.asarray(
        matrices.get("G_rad", np.zeros(capacitance.size)), dtype=float
    ).reshape(-1)

    started = time.perf_counter()
    print(f"[{_now()}] designing: r={args.order}, {args.modes} modes, T_op={args.t_op:g} K, "
          f"effort={args.effort:g}, integral={args.integral:g}", flush=True)
    result = design_modal_controller(
        capacitance,
        matrices["L"],
        radiation,
        np.asarray(matrices["node_ids"], dtype=int).reshape(-1),
        model,
        T_op_K=float(args.t_op),
        n_modes=int(args.modes),
        r=int(args.order),
        effort_weight=float(args.effort),
        integral_gain=float(args.integral),
        out_path=str(out_path),
        graph_name=str(getattr(model.metadata, "graph_name", "") or folder.name),
        progress=lambda message: print(f"[{_now()}]   {message}", flush=True),
    )
    elapsed = time.perf_counter() - started

    print(f"\n[{_now()}] wrote {result.path}", flush=True)
    print(f"[{_now()}] {result.main_nodes:,} of {result.total_nodes:,} nodes in the main "
          f"component ({result.components} components)", flush=True)
    print(f"[{_now()}] heaters={result.n_heaters} sensors={result.n_sensors} "
          f"controlled={result.n_controlled}", flush=True)
    print(f"[{_now()}] dc_gain error={result.dc_gain_error:.4g}  "
          f"step-response error={result.step_response_error:.4g}  "
          f"reduced model stable={result.reduced_stable}", flush=True)
    print(f"[{_now()}] design took {elapsed / 60.0:.1f} min", flush=True)
    _describe(Path(result.path))
    if not result.reduced_stable:
        print(f"[{_now()}] WARNING: the reduced model is NOT stable; do not run with this "
              f"artifact without investigating.", flush=True)
    return Path(result.path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph", help="path to graphs/<name>")
    p.add_argument("--t-op", type=float, default=50.0, help="linearization temperature [K]")
    p.add_argument("--modes", type=int, default=140, help="modes retained before truncation")
    p.add_argument("--order", type=int, default=50, help="reduced order r")
    p.add_argument("--effort", type=float, default=0.2, help="LQR effort weight")
    p.add_argument("--integral", type=float, default=0.06, help="integral gain stored in the artifact")
    p.add_argument("--allow-full-load", action="store_true",
                   help="permit the graph.json loader when fast-load artifacts are missing "
                        "(needs ~45 GB on a 3M-cell graph)")
    p.add_argument("--no-backup", action="store_true",
                   help="overwrite an existing artifact without keeping a .bak copy")
    return p


def main() -> None:
    args = build_parser().parse_args()
    build(args)


if __name__ == "__main__":
    main()
