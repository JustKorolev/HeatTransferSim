"""Compute the plant's DC gain G exactly, for the MIMO PI controller.

G (controlled sensors x heaters, K/W) is the only model object a
static-decoupling MIMO PI needs. In simulation there is no reason to identify it
by step tests -- we have L, so ``L T = P`` gives it directly from one sparse
factorization, in minutes rather than months.

Identifying it empirically here would also be much worse. DC contribution scales
as 1/lambda, so the SLOW modes dominate the gain: on no_mli_high_res 74.3% of G
lives in modes slower than 10,000 s against a slowest mode of 86,756 s (24.1 h).
A 300 s step test sees 3.4% of the response and understates G by ~97%, and
extrapolating the asymptote only halves that error -- sometimes diverging
outright. Step-test identification belongs on HARDWARE, where L is unavailable;
there, multiplex the excitation so you pay one settling time instead of 27.

Runs off the fast-load artifacts, so the graph never has to be parsed from
graph.json. Saves in the sys-id run-folder format, so the result appears in the
controller list beside anything produced by the sim tab's sys ID.

Example:
    python build_g_matrix.py graphs/no_mli_high_res --t-op 50
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

G_BUILD_LOG_FILENAME = "build_g_matrix.log"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build(args: argparse.Namespace) -> Path:
    from graph_visualizer.modal_reduction import exact_dc_gain
    from graph_visualizer.sys_id_artifacts import save_sys_id_gain_matrix

    folder = Path(args.graph)
    if not folder.is_dir():
        raise SystemExit(f"Not a graph folder: {folder}")

    from graph_visualizer.fast_graph_io import can_load_fast, load_graph_for_simulation

    usable, reason = can_load_fast(folder)
    if usable:
        print(f"[{_now()}] loading via the low-memory path (no graph.json) ...", flush=True)
        model, matrices, report = load_graph_for_simulation(folder)
        print(f"[{_now()}] loaded {report.node_count:,} nodes and {report.edge_count:,} edges",
              flush=True)
    elif args.allow_full_load:
        print(f"[{_now()}] fast load unavailable ({reason}); falling back to graph.json "
              f"-- expect very high memory use", flush=True)
        from graph_visualizer.graph_io import load_graph_folder

        model, matrices = load_graph_folder(str(folder))
    else:
        raise SystemExit(
            f"Fast load unavailable ({reason}) and --allow-full-load was not passed.\n"
            "Run 'Update graph' first; the graph.json loader needs roughly 45 GB here."
        )

    for key in ("C", "L", "node_ids"):
        if key not in matrices:
            raise SystemExit(f"Graph is missing the '{key}' matrix needed for the DC gain.")
    capacitance = np.asarray(matrices["C"], dtype=float).reshape(-1)
    radiation = np.asarray(matrices.get("G_rad", np.zeros(capacitance.size)), dtype=float).reshape(-1)

    started = time.perf_counter()
    print(f"[{_now()}] solving the exact DC gain at T_op={args.t_op:g} K ...", flush=True)
    result = exact_dc_gain(
        capacitance,
        matrices["L"],
        radiation,
        np.asarray(matrices["node_ids"], dtype=int).reshape(-1),
        model,
        T_op_K=float(args.t_op),
        progress=lambda message: print(f"[{_now()}]   {message}", flush=True),
    )
    elapsed = time.perf_counter() - started

    run_name = args.name or f"G_exact_T{float(args.t_op):g}K".replace(".", "p")
    target = save_sys_id_gain_matrix(
        folder,
        run_name,
        result["controlled_sensor_ids"],
        result["heater_ids"],
        result["G"],
        metadata={
            "method": "exact_dc_solve",
            "T_op_K": float(args.t_op),
            "dc_ground": result["dc_ground"],
            "cond_G": result["cond"],
            "rga_diag_min": result["rga_diag_min"],
            "rga_diag_negative": result["rga_diag_negative"],
            "components": result["components"],
            "main_nodes": result["main_nodes"],
            "total_nodes": result["total_nodes"],
            "elapsed_s": elapsed,
        },
    )

    G = result["G"]
    print(f"\n[{_now()}] wrote {target}", flush=True)
    print(f"[{_now()}] G is {G.shape[0]} controlled sensor(s) x {G.shape[1]} heater(s), "
          f"grounded at the {result['dc_ground']}", flush=True)
    print(f"[{_now()}] cond(G)={result['cond']:.4g}   median column sum="
          f"{np.median(np.abs(G).sum(axis=0)):.4g} K/W", flush=True)
    if result["rga_square"]:
        print(f"[{_now()}] RGA diagonal: {result['rga_diag_negative']} of {G.shape[0]} negative "
              f"(min {result['rga_diag_min']:+.4g})", flush=True)
    else:
        print(f"[{_now()}] RGA diagonal not reported: G is {G.shape[0]}x{G.shape[1]}, so there is "
              f"no one-heater-per-sensor pairing for it to describe.", flush=True)
    if result["rga_diag_negative"]:
        print(f"[{_now()}]   -> a NEGATIVE RGA diagonal means that pairing's gain changes sign "
              f"once the other loops close; per-pair SISO control would drive the wrong way. "
              f"This is why the scheme decouples through G rather than pairing.", flush=True)
    for issue in result["issues"][:5]:
        print(f"[{_now()}]   note: {issue}", flush=True)
    print(f"[{_now()}] took {elapsed / 60.0:.1f} min", flush=True)
    return target


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph", help="path to graphs/<name>")
    p.add_argument("--t-op", type=float, default=50.0,
                   help="operating temperature [K]. G is a LINEARIZATION: conductance depends on "
                        "temperature through k(T) and h(T)=3000*(T/293), so a gain identified at "
                        "the wrong background is systematically wrong (25%% between 40 K and 50 K).")
    p.add_argument("--name", default=None, help="run name for the saved matrix folder")
    p.add_argument("--allow-full-load", action="store_true",
                   help="permit the graph.json loader when fast-load artifacts are missing")
    return p


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
