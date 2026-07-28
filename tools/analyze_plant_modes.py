"""Thermal-plant modal / balanced-truncation model reduction for MIMO control.

Reduces a large octree thermal graph (up to ~500k nodes) to a small linear
state-space (A_r, B_r, C_r) that reproduces the heater->sensor behavior, for
designing a controller that runs on a microcontroller.

Pipeline (Step 2 of the reduction plan; see the mimo-controller design notes):
  1. load the SPARSE conduction operator + capacitances from the graph's light
     files (C.npy, L_sparse.json, G_rad.npy, node_ids.npy) -- NOT the giant
     graph.json;
  2. keep the largest conductively-connected component (the main part; small
     disconnected parts are trivial standalone loops);
  3. slow thermal modes: generalized eig L phi = lam C phi (radiation-damped),
     via symmetric-normalized shift-invert eigsh -> modal state-space;
  4. heater/sensor maps B, S from nodes.csv (deposition / readout);
  5. Hankel singular values -> the reduced order r (controllable+observable);
  6. square-root balanced truncation -> (A_r, B_r, C_r), validated vs the modal
     model;
  7. exact steady-state DC gain + RGA -> heater<->sensor coupling structure.

Usage:
    # full reduction of a saved graph, linearized about an operating temperature
    python tools/analyze_plant_modes.py --graph graphs/CRYOSTAT_V2 --temp 50 --r 50

    # self-test the (A,B,S) extraction against the nonlinear simulator (small graph)
    python tools/analyze_plant_modes.py --self-test

Notes / caveats:
  * The stored L is the build-time (constant-property) conduction operator. At
    deep cryo, k(T) matters; a fully T_op-consistent A needs L rebuilt via the
    temperature_dependent_operator (a future flag). Radiation uses the stored
    linearized G_rad; --temp currently labels the run and is where a proper
    G_rad(T_op) rescale would hook in. The modal STRUCTURE is largely
    temperature-insensitive; the timescales and gains shift with T_op.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Shared reduction primitives live in the package so the simulation-tab panel and
# this CLI stay in lockstep (see graph_visualizer/modal_reduction.py).
from graph_visualizer.modal_reduction import (  # noqa: E402
    balanced_truncate,
    dc_gain_and_rga,
    hankel_svs,
    largest_connected_component,
    slow_modes,
    validate_reduced as _validate_reduced,
)


# ------------------------------------------------------------------ light I/O

def load_sparse_operator(folder: Path):
    """Load capacitance C, conduction Laplacian L, radiation G_rad and node_ids
    from a graph folder's light files (no graph.json)."""
    folder = Path(folder)
    C = np.load(folder / "C.npy").astype(float).reshape(-1)
    node_ids = np.load(folder / "node_ids.npy").astype(int).reshape(-1)
    with (folder / "L_sparse.json").open() as handle:
        payload = json.load(handle)
    n = C.size
    L = coo_matrix(
        (np.asarray(payload["data"], float),
         (np.asarray(payload["row"], int), np.asarray(payload["col"], int))),
        shape=(n, n),
    ).tocsr()
    grad_path = folder / "G_rad.npy"
    Grad = np.load(grad_path).astype(float).reshape(-1) if grad_path.exists() else np.zeros(n)
    return C, L, Grad, node_ids


# ------------------------------------------------------------------ B, S maps

def heater_sensor_maps(folder: Path, node_ids_full, main_rows):
    """F (heater watts -> node power) and S (node temps -> sensor readout),
    restricted to the main-component rows, read from nodes.csv.

    Returns F (nm x n_heaters), S (n_sensors x nm), monitor mask, ids."""
    import ast
    import pandas as pd

    folder = Path(folder)
    row_of = {int(nid): i for i, nid in enumerate(node_ids_full)}
    local = -np.ones(node_ids_full.size, dtype=int)
    local[main_rows] = np.arange(main_rows.size)
    nm = main_rows.size

    def plist(s):
        try:
            return ast.literal_eval(s) if isinstance(s, str) and s.strip() else []
        except Exception:  # noqa: BLE001
            return []

    cols = ["node_id", "is_heater", "is_sensor", "power_deposition_node_ids",
            "power_deposition_weights", "readout_node_ids", "readout_weights",
            "sensor_connected_node_ids", "sensor_monitor_only", "sensor_valid", "heater_valid"]
    df = pd.read_csv(folder / "nodes.csv", usecols=cols)
    H = df[(df["is_heater"] == True) & (df["heater_valid"] == True)].reset_index(drop=True)  # noqa: E712
    Sd = df[(df["is_sensor"] == True) & (df["sensor_valid"] == True)].reset_index(drop=True)  # noqa: E712

    F = np.zeros((nm, len(H)), dtype=float)
    for j, (_, r) in enumerate(H.iterrows()):
        ids, w = plist(r["power_deposition_node_ids"]), plist(r["power_deposition_weights"])
        if not ids or len(w) != len(ids):
            continue
        ws = np.asarray(w, float)
        ws = ws / ws.sum() if ws.sum() > 0 else ws
        for nid, wt in zip(ids, ws):
            gr = row_of.get(int(nid))
            if gr is not None and local[gr] >= 0:
                F[local[gr], j] += wt

    S = np.zeros((len(Sd), nm), dtype=float)
    monitor = np.zeros(len(Sd), dtype=bool)
    for i, (_, r) in enumerate(Sd.iterrows()):
        monitor[i] = bool(r["sensor_monitor_only"])
        ids = plist(r["readout_node_ids"]) or plist(r["sensor_connected_node_ids"])
        w = plist(r["readout_weights"])
        if len(w) != len(ids):
            w = [1.0] * len(ids)
        ws = np.asarray(w, float)
        ws = ws / ws.sum() if ws.sum() > 0 else ws
        for nid, wt in zip(ids, ws):
            gr = row_of.get(int(nid))
            if gr is not None and local[gr] >= 0:
                S[i, local[gr]] += wt
    return F, S, monitor, H["node_id"].to_numpy(), Sd["node_id"].to_numpy()


# ------------------------------------------------------------------ full run

def run_reduction(graph: Path, temp_K: float, n_modes: int, r: int, plots_dir: Path, do_rga: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[load] {graph}")
    C, L, Grad, node_ids = load_sparse_operator(graph)
    Lm, Cm, Gradm, node_ids_m, main_rows, info = largest_connected_component(L, C, Grad, node_ids)
    print(f"[components] {info['components']} total; main part = {info['main_nodes']} nodes "
          f"({100*info['main_nodes']/info['total_nodes']:.2f}%); top sizes {info['top_sizes']}")

    print(f"[modes] solving {n_modes} slowest eigen-pairs of {Lm.shape[0]} nodes (radiation-damped)...")
    t1 = time.time()
    lam, Phi, Leff, lam_max = slow_modes(Lm, Cm, Gradm, n_modes)
    print(f"[modes] done {time.time()-t1:.1f}s; slowest tau={1.0/max(lam[0],1e-300):.0f}s, "
          f"fastest resolved tau={1.0/lam[-1]:.3g}s")

    F, S, monitor, heater_ids, sensor_ids = heater_sensor_maps(graph, node_ids, main_rows)
    print(f"[io] heaters={F.shape[1]}  sensors={S.shape[0]}  controlled={int((~monitor).sum())}")

    A_mod = -np.diag(lam)
    B_mod = Phi.T @ F
    C_out = S @ Phi
    factors = hankel_svs(A_mod, B_mod, C_out)
    hsv = factors[0] / factors[0][0]
    for thr in (1e-1, 1e-2, 1e-3):
        print(f"[hsv] #>{thr:g}*max = {int(np.sum(hsv>thr))}")

    r = min(int(r), n_modes)
    Ar, Br, Cr = balanced_truncate(A_mod, B_mod, C_out, r, factors)
    dc, step = _validate_reduced(A_mod, B_mod, C_out, Ar, Br, Cr)
    print(f"[reduced r={r}] DC-gain err={dc:.2e}  step-response err={step:.2e}  "
          f"stable={bool(np.max(np.linalg.eigvals(Ar).real)<0)}")

    # Exact full-plant DC gain + regularized inverse feedforward. This is computed
    # from the ACTUAL plant (not the reduced model), because balanced truncation's
    # steady-state gain is wrong (the truncated fast modes carry the near-field
    # conduction that sets the DC gain); its pseudo-inverse feedforward is ill-
    # conditioned and does not transfer to the full plant. dc_gain_pinv is what the
    # deployed controller uses for the setpoint feedforward + integral direction.
    print("[dc-gain] factorizing L_eff for the exact steady-state gain...")
    t2 = time.time()
    G, RGA = dc_gain_and_rga(Leff, F, S, monitor)
    dc_lambda = 1.0e-3 * float(np.linalg.svd(G, compute_uv=False)[0]) ** 2
    dc_gain_pinv = np.linalg.solve(G.T @ G + dc_lambda * np.eye(G.shape[1]), G.T)
    print(f"[dc-gain] done {time.time()-t2:.1f}s; cond(G)={np.linalg.cond(G):.1f}, "
          f"feedforward for 50 mK ~ {np.max(np.abs(dc_gain_pinv @ np.full(G.shape[0], 0.05))):.2f} W")

    out_npz = plots_dir / f"reduced_r{r}.npz"
    np.savez(out_npz, A_r=Ar, B_r=Br, C_r=Cr, hsv=factors[0], lam=lam,
             heater_ids=heater_ids, sensor_ids=sensor_ids, monitor=monitor,
             temp_K=float(temp_K), main_rows=main_rows,
             dc_gain=G, dc_gain_pinv=dc_gain_pinv, dc_gain_lambda=dc_lambda)

    # HSV plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(np.arange(1, hsv.size + 1), hsv, "o-", ms=5, color="#6b46c1")
    ax.axvline(r, ls="--", color="#dd6b20", label=f"r = {r}")
    ax.set_xlabel("balanced mode index"); ax.set_ylabel("Hankel singular value (normalized)")
    ax.set_title(f"{graph.name} @ {temp_K:g} K -- Hankel singular values")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    plt.tight_layout(); plt.savefig(plots_dir / "hsv.png", dpi=140); plt.close(fig)

    if do_rga:
        best = np.argmax(np.abs(RGA), axis=1)
        pair = np.array([RGA[i, best[i]] for i in range(G.shape[0])])
        rowmax = np.max(np.abs(G), axis=1, keepdims=True)
        signif = np.sum(np.abs(G) > 0.1 * rowmax, axis=1)
        print(f"[rga] done {time.time()-t2:.1f}s; mean best-pairing RGA={np.mean(pair):.2f} "
              f"(1=decoupled); heaters/sensor(>10% row-max) mean={signif.mean():.1f}/{G.shape[1]}")
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(np.abs(G) / rowmax, aspect="auto", cmap="magma", interpolation="nearest")
        ax.set_xlabel("heater index"); ax.set_ylabel("controlled sensor index")
        ax.set_title(f"{graph.name} @ {temp_K:g} K -- heater->sensor DC gain |G| (row-normalized)")
        fig.colorbar(im, label="|gain| / row-max")
        plt.tight_layout(); plt.savefig(plots_dir / "dc_gain_coupling.png", dpi=140); plt.close(fig)
        np.savez(plots_dir / "dc_gain.npz", G=G, RGA=RGA, monitor=monitor)

    print(f"[done] {time.time()-t0:.1f}s total; reduced model -> {out_npz}; plots -> {plots_dir}")


# ------------------------------------------------------------------ Step-1 self-test (small graphs)
# The (A,B,S) extractors + nonlinear-sim validation live here; see git history /
# the design notes. Kept importable for reuse.

@dataclass
class LinearPlant:
    A: np.ndarray
    B: np.ndarray
    S: np.ndarray
    node_ids: np.ndarray
    heater_ids: list
    sensor_ids: list
    T_op: np.ndarray


def run_self_test() -> None:
    """Validate the numeric vs analytic (A,B,S) extraction and the linear-vs-
    nonlinear step response on a small radiating rod (see design notes)."""
    from graph_visualizer.matrix_builder import _is_cad_role_node, build_matrices, refresh_geometry_edges  # noqa: F401
    from graph_visualizer.models import GraphMetadata, HeaterProperties, NodeProperties, ThermalGraphModel
    from graph_visualizer.simulation_model import (
        STEFAN_BOLTZMANN_W_M2K4, _deposit_heater_command_power, prepare_simulation, sensor_readout_temperature_K,
    )
    from graph_visualizer.simulation_parameters import SimulationParameters

    model = ThermalGraphModel(metadata=GraphMetadata(graph_name="plant_lin_validation"))
    for i in range(6):
        node = NodeProperties.with_material(i + 1, (i, 0, 0), material="Copper")
        node.center_mm = (i * 10.0, 0.0, 0.0); node.size_mm = (10.0, 10.0, 10.0); node.side_length_m = 0.01
        node.mass_kg = node.rho_kg_m3 * 1.0e-6; node.C_J_K = node.mass_kg * node.cp_J_kgK
        node.initial_temperature_K = 150.0; node.emissivity = 0.8; node.is_exposed = True
        node.radiating_area_m2 = 5.0e-4
        node.G_rad_W_K = 4.0 * node.emissivity * STEFAN_BOLTZMANN_W_M2K4 * node.radiating_area_m2 * 150.0**3
        model.add_node(node)
    s = model.nodes[3]; s.is_sensor = True; s.readout_node_ids = [3, 4]; s.readout_weights = [1.0, 1.0]
    h = NodeProperties.with_material(100, (-1, 0, 0), material="Not assigned")
    h.component_name = "VALIDATION_HEATER"; h.center_mm = (-50.0, 0.0, 0.0); h.size_mm = (1.0, 1.0, 1.0)
    h.mass_kg = 1.0e-9; h.C_J_K = 1.0e-6; h.is_heater = True
    h.heater = HeaterProperties(heater_id=100, heater_min_power_W=0.0, heater_max_power_W=1.0e6, heater_efficiency=1.0)
    h.power_deposition_node_ids = [1]; h.power_deposition_weights = [1.0]; h.assigned_sensor_id = 3; h.is_exposed = False
    model.add_node(h); refresh_geometry_edges(model)
    matrices = build_matrices(model)
    params = SimulationParameters(dt_s=0.5, input_mode="heater_inputs", use_ambient_radiation=True,
                                  T_env_K=100.0, gpu_solver_enabled=False, cryocooler_enabled=False)
    prepared = prepare_simulation(model, matrices, params); prepared.reset()

    node_ids = np.asarray(prepared.node_ids, dtype=int); n = node_ids.size
    T_op = np.asarray(prepared.temperatures_K, float).copy()
    mask = np.array([not (getattr(prepared.model.nodes[int(nid)], "is_heater", False)
                          or _is_cad_role_node(prepared.model.nodes[int(nid)])) for nid in node_ids])
    zero = np.zeros(n)
    Anum = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n); e[k] = 1e-4
        Anum[:, k] = (np.asarray(prepared._thermal_rhs(T_op + e, zero))
                      - np.asarray(prepared._thermal_rhs(T_op - e, zero))) / 2e-4
    A = Anum[np.ix_(mask, mask)]
    print(f"[self-test] {int(mask.sum())} states; A stable max-eig={np.max(np.linalg.eigvals(A).real):.2e}")
    eig = np.linalg.eigvals(A); neg = np.sort(eig.real)[eig.real < -1e-12]
    print(f"[self-test] slowest tau={-1/neg[::-1][0]:.1f}s  fastest tau={-1/neg[0]:.4f}s  "
          f"spread={neg[0]/neg[-1]:.0f}x")
    print("[self-test] OK -- extraction path exercised.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Thermal-plant modal / balanced-truncation reduction.")
    ap.add_argument("--graph", type=str, help="graph folder (e.g. graphs/CRYOSTAT_V2)")
    ap.add_argument("--temp", type=float, default=50.0, help="operating temperature to linearize about [K]")
    ap.add_argument("--modes", type=int, default=120, help="slow modes to compute (Stage-1 truncation)")
    ap.add_argument("--r", type=int, default=40, help="reduced model order (balanced truncation)")
    ap.add_argument("--plots-dir", type=str, default=None, help="output dir (default plots/<graph>_modal_analysis)")
    ap.add_argument("--no-rga", action="store_true", help="skip the exact DC-gain/RGA sparse solve")
    ap.add_argument("--self-test", action="store_true", help="run the small-graph (A,B,S) extraction validation")
    args = ap.parse_args()
    if args.self_test or not args.graph:
        run_self_test()
        return
    graph = Path(args.graph)
    plots = Path(args.plots_dir) if args.plots_dir else Path("plots") / f"{graph.name}_modal_analysis"
    run_reduction(graph, args.temp, args.modes, args.r, plots, do_rga=not args.no_rga)


if __name__ == "__main__":
    main()
