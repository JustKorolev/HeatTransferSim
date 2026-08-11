"""Modal / balanced-truncation reduction + reduced-order LQR controller design.

This is the library behind both ``tools/analyze_plant_modes.py`` (CLI, reads a
graph folder's light files) and the "Modal LQR design" panel in the simulation
tab (reduces the in-memory graph). It turns a large thermal graph's sparse
operator into the *exact* artifact consumed by
``PreparedSimulation._load_modal_controller``:

    K            (n_heaters x r)  reduced-order LQR state feedback  (u = -K x)
    E_reg        (r x n_sensors)  regularized static state estimate (x_hat = E_reg (y - T_op))
    Nx, Nu       reduced servo maps (fallback feedforward; ill-conditioned at DC)
    dc_gain_pinv (n_heaters x n_ctrl)  EXACT full-plant DC feedforward (preferred)
    heater_ids, sensor_ids, monitor, T_op_K

Pipeline: largest connected component -> slow radiation-damped modes -> modal
state space (A_mod, B_mod, C_out) -> Hankel singular values -> square-root
balanced truncation to order r -> LQR + estimator + servo design -> exact
steady-state DC gain feedforward.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.linalg import (
    expm,
    solve_continuous_lyapunov,
    solve_discrete_are,
    svd,
)
from scipy.signal import cont2discrete
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, splu


# Grounding conductance (W/K) applied at cryocooler cells when forming the DC-gain
# operator, so the otherwise-singular pure-conduction Laplacian has a sink.
#
# This used to be 1.0e3 W/K -- an "infinitely stiff cold tip" idealization. That is
# ~1000x stiffer than a real PT60, whose lift curve slope dQ/dT is only ~1.07 W/K at
# 50 K, and it has a specific, damaging consequence: a heater mounted ON the
# cryocooler block gets a DC gain of ~27/1e3 = 0.027 K/W instead of the ~5 K/W a
# normal heater sees. The controller then believes heat dumped there is nearly free
# and parks power into it. On no_mli_high_res two such heaters (2988217, 2988222)
# held 73% of the terminal command and absorbed 53% of all energy injected, which
# was almost exactly the run's net heat imbalance. They also accounted for the
# entire dc_gain condition number (779; ~62 without them).
#
# The default is now derived from the cooler's ACTUAL stiffness at the operating
# point (see cryocooler_ground_conductance_W_K). Setting it explicitly overrides
# that; 1.0e3 restores the old fixed-tip behaviour.
CRYOCOOLER_DC_GROUND_W_K = 1.0e3  # legacy fixed-tip value, kept for explicit opt-in


def cryocooler_ground_conductance_W_K(
    operating_temperature_K: float,
    *,
    max_power_w: float = 150.0,
    capacity_scale: float = 1.0,
    delta_K: float = 1.0,
) -> float:
    """Total DC grounding conductance from the cooler's own lift curve.

    The physically meaningful stiffness of the sink is dQ/dT at the operating
    point: warm the cold end by 1 K and the cooler removes this many more watts.
    For a PT60 at 50 K that is ~1.07 W/K, not 1000. Returned as a TOTAL for the
    device; the caller spreads it across the cryocooler cells.
    """
    from .cryocooler import PT60LiftCurve

    curve = PT60LiftCurve(max_power_w=float(max_power_w), capacity_scale=float(capacity_scale))
    T = float(operating_temperature_K)
    half = max(1.0e-6, 0.5 * float(delta_K))
    slope = (curve.cooling_capacity_w(T + half) - curve.cooling_capacity_w(T - half)) / (2.0 * half)
    # Below the curve's floor the cooler has no authority; fall back to a small
    # positive value so the DC operator stays non-singular.
    return float(max(slope, 1.0e-3))


# ------------------------------------------------------------------ reduction primitives
# These are shared with tools/analyze_plant_modes.py (which imports them).

def largest_connected_component(L, C, Grad, node_ids):
    """Restrict to the largest conductively-connected component (the main part)."""
    ncomp, labels = connected_components(L, directed=False)
    sizes = np.bincount(labels)
    main = int(np.argmax(sizes))
    rows = np.where(labels == main)[0]
    info = {
        "components": int(ncomp),
        "main_nodes": int(rows.size),
        "total_nodes": int(C.size),
        "top_sizes": sorted(sizes.tolist(), reverse=True)[:8],
    }
    Lm = L[rows][:, rows].tocsr()
    return Lm, C[rows], Grad[rows], node_ids[rows], rows, info


def slow_modes(Lm, Cm, Gradm, n_modes: int):
    """Slowest n_modes of the radiation-damped generalized problem
    (L + diag(G_rad)) phi = lam C phi, via symmetric shift-invert eigsh.

    Returns (lam, Phi, Leff, lam_max) with Phi C-orthonormal (Phi^T C Phi = I)."""
    Leff = (Lm + diags(Gradm)).tocsr()
    D = diags(1.0 / np.sqrt(Cm))
    A = (D @ Leff @ D).tocsr()
    A = 0.5 * (A + A.T)
    lam_max = float(eigsh(A, k=1, which="LA", return_eigenvectors=False, tol=1e-3)[0])
    vals, vecs = eigsh(A, k=int(n_modes), sigma=-1e-4 * lam_max, which="LM", tol=1e-6)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    Phi = vecs / np.sqrt(Cm)[:, None]  # C^{-1/2} psi
    return vals, Phi, Leff, lam_max


def _sqrtm_psd(M):
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.T


def hankel_svs(A_mod, B_mod, C_out):
    """Gramians of the (stable) modal model -> Hankel singular values + factors."""
    Wc = solve_continuous_lyapunov(A_mod, -(B_mod @ B_mod.T))
    Wo = solve_continuous_lyapunov(A_mod.T, -(C_out.T @ C_out))
    Lc, Lo = _sqrtm_psd(Wc), _sqrtm_psd(Wo)
    U, s, Vt = svd(Lo @ Lc)
    return s, U, Vt, Lc, Lo


def balanced_truncate(A_mod, B_mod, C_out, r, factors):
    """Square-root balanced truncation to order r."""
    s, U, Vt, Lc, Lo = factors
    S12 = np.diag(1.0 / np.sqrt(s[:r]))
    T = Lc @ Vt[:r].T @ S12
    Ti = S12 @ U[:, :r].T @ Lo
    return Ti @ A_mod @ T, Ti @ B_mod, C_out @ T


def validate_reduced(A_mod, B_mod, C_out, Ar, Br, Cr):
    """Relative DC-gain and step-response error of the reduced vs modal model."""
    n, r = A_mod.shape[0], Ar.shape[0]
    Gf = C_out @ np.linalg.solve(-A_mod, B_mod)
    Gr = Cr @ np.linalg.solve(-Ar, Br)
    dc = float(np.max(np.abs(Gf - Gr)) / max(np.max(np.abs(Gf)), 1e-30))
    step_err = 0.0
    for t in (5.0, 20.0, 60.0, 200.0, 600.0):
        yf = C_out @ np.linalg.solve(A_mod, (expm(A_mod * t) - np.eye(n)) @ B_mod)
        yr = Cr @ np.linalg.solve(Ar, (expm(Ar * t) - np.eye(r)) @ Br)
        step_err = max(step_err, float(np.max(np.abs(yf - yr)) / (np.max(np.abs(yf)) + 1e-30)))
    return dc, step_err


def dc_gain_and_rga(Leff, F, S, monitor):
    """Exact steady-state gain G = S_ctrl L_eff^{-1} F and its RGA."""
    lu = splu(Leff.tocsc())
    X = lu.solve(np.asarray(F, dtype=float))
    ctrl = ~monitor
    G = S[ctrl] @ X
    RGA = G * np.linalg.pinv(G).T
    return G, RGA


def drop_inert_cells(
    L: csr_matrix,
    C: np.ndarray,
    Grad: np.ndarray,
    node_ids: np.ndarray,
    *,
    capacitance_floor_frac: float = 1.0e-3,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Remove thermally-inert cells before reduction.

    Voxels assigned a null material ("ZERO MATTER" / "Unassigned") and the
    heater/sensor marker nodes carry near-zero heat capacity (C ~ 1e-9 J/K) and
    near-zero conduction (k ~ 1e-9 W/mK): they store no thermal state and couple
    to nothing. But they are weakly bridged into the main component, so the
    generalized eigenproblem (L + diag(Grad)) phi = lam C phi picks them up as a
    dense cluster of spurious near-zero eigenvalues. That makes the symmetric
    shift-invert slow-mode solve ill-conditioned -- the scaling D = 1/sqrt(C)
    explodes -- and effectively non-convergent (it can hang for many minutes on a
    large graph). Dropping them leaves the real thermal network.

    A cell is kept when ``C >= capacitance_floor_frac * median(C)`` over the
    positive-capacitance cells. This is a NO-OP on operators whose capacitances
    were already floored (e.g. the octree-saved matrices, whose minimum is far
    above the cut), and only trims genuinely degenerate cells otherwise.
    Heater/sensor I/O is unaffected: F/S map to the real deposition/readout
    cells, which are retained. Returns the trimmed (L, C, Grad, node_ids) plus
    the boolean keep mask and an info dict.
    """
    C = np.asarray(C, dtype=float).reshape(-1)
    Grad = np.asarray(Grad, dtype=float).reshape(-1)
    node_ids = np.asarray(node_ids, dtype=int).reshape(-1)
    L = csr_matrix(L)
    positive = C[C > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    floor = max(0.0, float(capacitance_floor_frac)) * scale
    keep = C >= floor
    info = {
        "dropped": int((~keep).sum()),
        "kept": int(keep.sum()),
        "capacitance_floor_J_K": float(floor),
    }
    if info["dropped"] == 0:
        return L, C, Grad, node_ids, keep, info
    Lk = L[keep][:, keep].tocsr()
    return Lk, C[keep], Grad[keep], node_ids[keep], keep, info


# ------------------------------------------------------------------ in-memory plant maps

def heater_sensor_maps_from_model(model: Any, node_ids_full, main_rows, issues: list | None = None):
    """F (heater watts -> node power) and S (node temps -> sensor readout) built
    directly from the in-memory graph model, restricted to the main-component rows.

    Mirrors the nodes.csv path in tools/analyze_plant_modes.py but reads the live
    node deposition/readout maps. Returns F (nm x n_heaters), S (n_sensors x nm),
    monitor mask, heater node ids, sensor node ids."""
    node_ids_full = np.asarray(node_ids_full, dtype=int)
    row_of = {int(nid): i for i, nid in enumerate(node_ids_full)}
    main_rows = np.asarray(main_rows, dtype=int)
    local = -np.ones(node_ids_full.size, dtype=int)
    local[main_rows] = np.arange(main_rows.size)
    nm = int(main_rows.size)

    heater_nodes = []
    sensor_nodes = []
    for nid in sorted(model.nodes):
        node = model.nodes[nid]
        if bool(getattr(node, "is_heater", False)) and bool(getattr(node, "heater_valid", True)):
            heater_nodes.append(node)
        if bool(getattr(node, "is_sensor", False)) and bool(getattr(node, "sensor_valid", True)):
            sensor_nodes.append(node)

    F = np.zeros((nm, len(heater_nodes)), dtype=float)
    heater_ids: list[int] = []
    for j, node in enumerate(heater_nodes):
        heater_ids.append(int(node.node_id))
        ids = list(getattr(node, "power_deposition_node_ids", None) or [])
        weights = list(getattr(node, "power_deposition_weights", None) or [])
        if not ids or len(weights) != len(ids):
            continue
        ws = np.asarray(weights, dtype=float)
        total = ws.sum()
        ws = ws / total if total > 0 else ws
        for nid, wt in zip(ids, ws):
            gr = row_of.get(int(nid))
            if gr is not None and local[gr] >= 0:
                F[local[gr], j] += float(wt)

    S = np.zeros((len(sensor_nodes), nm), dtype=float)
    monitor = np.zeros(len(sensor_nodes), dtype=bool)
    sensor_ids: list[int] = []
    for i, node in enumerate(sensor_nodes):
        sensor_ids.append(int(node.node_id))
        monitor[i] = bool(getattr(node, "sensor_monitor_only", False))
        ids = list(getattr(node, "readout_node_ids", None) or []) or list(
            getattr(node, "sensor_connected_node_ids", None) or []
        )
        weights = list(getattr(node, "readout_weights", None) or [])
        if len(weights) != len(ids):
            weights = [1.0] * len(ids)
        ws = np.asarray(weights, dtype=float)
        total = ws.sum()
        ws = ws / total if total > 0 else ws
        for nid, wt in zip(ids, ws):
            gr = row_of.get(int(nid))
            if gr is not None and local[gr] >= 0:
                S[i, local[gr]] += float(wt)

    # A heater whose deposition map is missing/mismatched, or whose target nodes all
    # fall outside the main component, leaves an ALL-ZERO column in F -- it is still
    # counted as an actuator but can no longer move the plant. The same holds for a
    # sensor with an all-zero row in S: it reads a constant. Both used to pass
    # silently and produce a controller designed around dead channels, so surface
    # them instead of letting the design look healthy.
    if issues is not None:
        dead_heaters = [int(heater_ids[j]) for j in range(F.shape[1]) if not np.any(F[:, j])]
        dead_sensors = [int(sensor_ids[i]) for i in range(S.shape[0]) if not np.any(S[i, :])]
        if dead_heaters:
            issues.append(
                f"{len(dead_heaters)} heater(s) have an empty power-deposition map and "
                f"cannot affect the plant (node ids {dead_heaters[:8]}"
                + (", ..." if len(dead_heaters) > 8 else "")
                + "); the controller would be designed around dead actuators."
            )
        if dead_sensors:
            issues.append(
                f"{len(dead_sensors)} sensor(s) have an empty readout map and report a "
                f"constant (node ids {dead_sensors[:8]}"
                + (", ..." if len(dead_sensors) > 8 else "")
                + ")."
            )
    return F, S, monitor, np.asarray(heater_ids, dtype=int), np.asarray(sensor_ids, dtype=int)


# ------------------------------------------------------------------ LQR + estimator + servo

def lqr_weights(A_r, B_r, C_ctrl, effort_weight: float):
    """The LQR cost weights (Q, R) for tracking the controlled outputs.

    Q = C_ctrl^T C_ctrl (plus a tiny ridge for numerical detectability) and
    R = rho I. Split out from the gain solve so the artifact can store the
    weights and re-derive a gain at any sample rate."""
    r = A_r.shape[0]
    Q = C_ctrl.T @ C_ctrl + 1.0e-9 * np.eye(r)
    R = max(float(effort_weight), 1.0e-12) * np.eye(B_r.shape[1])
    return Q, R


def discrete_lqr_gain(A_r, B_r, Q, R, dt_s: float):
    """Zero-order-hold discretize (A_r, B_r) at dt and solve the discrete LQR.

    Returns K (n_heaters x r) for u = -K x, applied once per sample of length dt.
    The cost weights are scaled by dt so the discrete cost approximates the same
    continuous integral, which keeps rho meaning the same thing across dt.

    There is deliberately NO continuous-time counterpart to this function. A
    continuous-time gain assumes the loop is closed continuously, which only holds
    while dt is short against every mode the gain acts on -- and this plant spans
    tau = 0.0018 s to ~2800 s, so the local heater-to-sensor path presents its full
    DC gain within a single control step. In the ``no_mli_high_res`` run of
    2026-08-10 a continuous-designed gain put 11 eigenvalues of the per-step loop
    gain ``G*K*E_ctrl`` above 2 (peak 5557) -- closed-loop poles below -1 -- and the
    heater commands flipped sign on every single sample (measured
    sign-flip-per-sample of exactly 1.00), bounded only by the u >= 0 clip. On this
    plant a continuous gain leaves the unit circle at every practical dt, including
    0.1 s, so there is no safe default to offer.

    Cost is ~4 ms at r=33..50, so this is cheap enough to redo at runtime whenever
    the sample rate changes."""
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"design timestep must be finite and positive, got {dt_s!r}")
    n = np.asarray(A_r).shape[0]
    m = np.asarray(B_r).shape[1]
    Ad, Bd, *_ = cont2discrete((A_r, B_r, np.zeros((1, n)), np.zeros((1, m))), dt)
    P = solve_discrete_are(Ad, Bd, np.asarray(Q) * dt, np.asarray(R) * dt)
    return np.linalg.solve(np.asarray(R) * dt + Bd.T @ P @ Bd, Bd.T @ P @ Ad)


def regularized_estimator(C_r, reg_frac: float = 1.0e-3):
    """Regularized static state estimate x_hat = E_reg y from all sensor outputs
    (y = C_r x). Tikhonov inverse E_reg = (C_r^T C_r + mu I)^{-1} C_r^T."""
    r = C_r.shape[1]
    sv = np.linalg.svd(C_r, compute_uv=False)
    mu = reg_frac * (float(sv[0]) ** 2 if sv.size else 1.0)
    return np.linalg.solve(C_r.T @ C_r + mu * np.eye(r), C_r.T)


def servo_maps(A_r, B_r, C_ctrl, reg: float = 1.0e-8):
    """Reduced-model steady-state servo maps (Nx, Nu) for a controlled-output
    reference: [A_r B_r; C_ctrl 0][x_ss; u_ss] = [0; r_ctrl]. Regularized least
    squares (the KKT system is generally not square). These are a fallback; the
    deployed controller prefers the exact-plant dc_gain_pinv."""
    r = A_r.shape[0]
    nu = B_r.shape[1]
    nc = C_ctrl.shape[0]
    M = np.block([[A_r, B_r], [C_ctrl, np.zeros((nc, nu))]])
    rhs = np.vstack([np.zeros((r, nc)), np.eye(nc)])
    z = np.linalg.solve(M.T @ M + reg * np.eye(r + nu), M.T @ rhs)
    return z[:r], z[r:]


# ------------------------------------------------------------------ orchestration

@dataclass
class ModalDesignResult:
    path: str
    total_nodes: int
    main_nodes: int
    components: int
    n_modes: int
    reduced_order: int
    n_heaters: int
    n_sensors: int
    n_controlled: int
    dc_gain_error: float
    step_response_error: float
    reduced_stable: bool
    slowest_tau_s: float
    hsv_above_1pct: int
    cond_dc_gain: float
    inert_cells_dropped: int = 0
    T_op_K: float = 0.0
    # Non-fatal problems found while building the heater/sensor maps (e.g. actuators
    # with an empty deposition map). Empty means the design is clean.
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        inert = (
            f" ({self.inert_cells_dropped} inert cells dropped)"
            if self.inert_cells_dropped
            else ""
        )
        text = (
            f"r={self.reduced_order} (from {self.n_modes} modes over {self.main_nodes} nodes{inert}); "
            f"{self.n_heaters} heaters, {self.n_controlled}/{self.n_sensors} controlled sensors; "
            f"reduced DC err {self.dc_gain_error:.1e}, step err {self.step_response_error:.1e}, "
            f"{'stable' if self.reduced_stable else 'UNSTABLE'}; cond(G)={self.cond_dc_gain:.0f}."
        )
        for message in self.warnings:
            text += f"\nWARNING: {message}"
        return text


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def design_modal_controller(
    C: np.ndarray,
    L: csr_matrix,
    Grad: np.ndarray,
    node_ids: np.ndarray,
    model: Any,
    *,
    T_op_K: float,
    n_modes: int,
    r: int,
    effort_weight: float,
    integral_gain: float,
    out_path: str | Path,
    graph_name: str = "",
    design_dt_s: float = 1.0,
    progress: Callable[[str], None] | None = None,
) -> ModalDesignResult:
    """Reduce the plant and design the reduced-order LQR controller, saving the
    artifact consumed by the modal-LQR controller to ``out_path`` (an .npz)."""
    C = np.asarray(C, dtype=float).reshape(-1)
    Grad = np.asarray(Grad, dtype=float).reshape(-1) if Grad is not None else np.zeros_like(C)
    node_ids = np.asarray(node_ids, dtype=int).reshape(-1)
    L = csr_matrix(L)

    total_nodes_full = int(C.size)
    L, C, Grad, node_ids, _inert_keep, inert_info = drop_inert_cells(L, C, Grad, node_ids)
    if inert_info["dropped"]:
        _report(
            progress,
            f"Dropped {inert_info['dropped']} thermally-inert (null-material / marker) "
            f"cells; {inert_info['kept']} thermal cells remain.",
        )

    _report(progress, "Selecting largest connected component…")
    Lm, Cm, Gradm, _node_ids_m, main_rows, info = largest_connected_component(L, C, Grad, node_ids)
    main_nodes = int(info["main_nodes"])
    # eigsh needs 1 <= k <= n-1; keep a small margin.
    n_modes = int(max(2, min(int(n_modes), main_nodes - 2)))

    _report(progress, f"Solving {n_modes} slow modes over {main_nodes} nodes…")
    lam, Phi, Leff, _lam_max = slow_modes(Lm, Cm, Gradm, n_modes)

    _report(progress, "Building heater/sensor maps…")
    design_warnings: list[str] = []
    F, S, monitor, heater_ids, sensor_ids = heater_sensor_maps_from_model(
        model, node_ids, main_rows, issues=design_warnings
    )
    for message in design_warnings:
        _report(progress, f"WARNING: {message}")
    if F.shape[1] == 0:
        raise ValueError("No valid heaters found in the graph (need at least one).")
    if S.shape[0] == 0:
        raise ValueError("No valid sensors found in the graph (need at least one).")
    ctrl_idx = np.where(~monitor)[0]
    if ctrl_idx.size == 0:
        raise ValueError("All sensors are monitor-only; need at least one controlled sensor.")

    A_mod = -np.diag(lam)
    B_mod = Phi.T @ F
    C_out = S @ Phi

    _report(progress, "Hankel singular values + balanced truncation…")
    factors = hankel_svs(A_mod, B_mod, C_out)
    hsv = factors[0]
    r = int(max(1, min(int(r), n_modes)))
    Ar, Br, Cr = balanced_truncate(A_mod, B_mod, C_out, r, factors)
    dc_err, step_err = validate_reduced(A_mod, B_mod, C_out, Ar, Br, Cr)
    reduced_stable = bool(np.max(np.linalg.eigvals(Ar).real) < 0.0)

    _report(progress, "Designing LQR + estimator…")
    C_ctrl = Cr[ctrl_idx]
    design_dt_s = float(design_dt_s)
    Q_lqr, R_lqr = lqr_weights(Ar, Br, C_ctrl, effort_weight)
    K = discrete_lqr_gain(Ar, Br, Q_lqr, R_lqr, design_dt_s)
    _report(progress, f"LQR designed in discrete time at dt={design_dt_s:g} s (|K|={np.linalg.norm(K):.3g}).")
    E_reg = regularized_estimator(Cr)
    Nx, Nu = servo_maps(Ar, Br, C_ctrl)

    _report(progress, "Exact steady-state DC gain + feedforward…")
    # Ground the DC gain at the CRYOCOOLER -- the stable, well-characterized heat
    # sink -- using CONDUCTION ONLY, not radiation. Radiation's linearized sink
    # drifts with ambient temperature (and is poorly modeled here), so baking it
    # into the feedforward gain is fragile. All heaters/sensors sit in the same
    # conductively-connected component as the cryocooler, so a cryocooler-grounded
    # conduction gain is well-posed; the (uncertain) radiation load is left to the
    # controller's integral. Fall back to radiation grounding only if the graph has
    # no cryocooler cells (otherwise pure conduction L is singular).
    main_node_ids = np.asarray(node_ids, dtype=int)[main_rows]
    is_cryo = np.array(
        [bool(getattr(model.nodes.get(int(v)), "has_cryocooler", False)) for v in main_node_ids],
        dtype=bool,
    )
    # Ground with the cooler's OWN stiffness (dQ/dT at T_op), spread across its
    # cells, rather than a 1000 W/K fixed-tip idealization. The idealization gave a
    # heater mounted on the cold block a DC gain ~170x smaller than a normal
    # heater's, so the controller treated dumping power there as free. Spreading the
    # total keeps the whole-device stiffness physical regardless of cell count.
    cryo_cell_count = int(is_cryo.sum())
    if cryo_cell_count:
        total_ground = cryocooler_ground_conductance_W_K(float(T_op_K))
        per_cell = total_ground / float(cryo_cell_count)
        _report(
            progress,
            f"DC grounding: {total_ground:.4g} W/K total over {cryo_cell_count} cryocooler cells "
            f"({per_cell:.4g} W/K each) from the lift-curve slope at T_op={float(T_op_K):g} K.",
        )
    else:
        per_cell = 0.0
    cryo_ground = np.where(is_cryo, per_cell, 0.0)
    if float(cryo_ground.sum()) > 0.0:
        L_dc = (csr_matrix(Lm) + diags(cryo_ground)).tocsr()
        dc_ground = "cryocooler"
    else:
        L_dc = Leff  # no cryocooler present: radiation is the only available sink
        dc_ground = "radiation"
    G, _RGA = dc_gain_and_rga(L_dc, F, S, monitor)
    dc_lambda = 1.0e-3 * float(svd(G, compute_uv=False)[0]) ** 2
    dc_gain_pinv = np.linalg.solve(G.T @ G + dc_lambda * np.eye(G.shape[1]), G.T)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        K=np.asarray(K, dtype=float),
        E_reg=np.asarray(E_reg, dtype=float),
        Nx=np.asarray(Nx, dtype=float),
        Nu=np.asarray(Nu, dtype=float),
        dc_gain_pinv=np.asarray(dc_gain_pinv, dtype=float),
        heater_ids=heater_ids.astype(int),
        sensor_ids=sensor_ids.astype(int),
        monitor=monitor.astype(bool),
        T_op_K=float(T_op_K),
        integral_gain=float(integral_gain),
        r=int(r),
        n_modes=int(n_modes),
        effort_weight=float(effort_weight),
        # The sample rate K was designed at, plus the reduced plant and cost
        # weights it came from. K is only valid at design_dt_s; storing the
        # ingredients lets the simulator re-derive a correct gain at any other dt
        # (~4 ms) instead of silently applying a mis-sampled one. The MCU keeps
        # using the baked K, since it runs at a fixed rate.
        design_dt_s=float(design_dt_s),
        A_r=np.asarray(Ar, dtype=float),
        B_r=np.asarray(Br, dtype=float),
        C_r=np.asarray(Cr, dtype=float),
        Q_lqr=np.asarray(Q_lqr, dtype=float),
        R_lqr=np.asarray(R_lqr, dtype=float),
        hsv=np.asarray(hsv, dtype=float),
        lam=np.asarray(lam, dtype=float),
        dc_gain=np.asarray(G, dtype=float),
        graph=str(graph_name),
    )

    slowest_tau = float(1.0 / max(lam[0], 1e-300))
    hsv_norm = hsv / hsv[0] if hsv.size and hsv[0] > 0 else hsv
    return ModalDesignResult(
        path=str(out_path),
        total_nodes=total_nodes_full,
        main_nodes=main_nodes,
        components=int(info["components"]),
        n_modes=int(n_modes),
        reduced_order=int(r),
        n_heaters=int(F.shape[1]),
        n_sensors=int(S.shape[0]),
        n_controlled=int(ctrl_idx.size),
        dc_gain_error=float(dc_err),
        step_response_error=float(step_err),
        reduced_stable=reduced_stable,
        slowest_tau_s=slowest_tau,
        hsv_above_1pct=int(np.sum(hsv_norm > 1e-2)),
        cond_dc_gain=float(np.linalg.cond(G)),
        inert_cells_dropped=int(inert_info["dropped"]),
        T_op_K=float(T_op_K),
        warnings=list(design_warnings),
    )


def result_as_dict(result: ModalDesignResult) -> dict[str, Any]:
    return asdict(result)


# ------------------------------------------------------------------ artifact discovery

MODAL_ARTIFACT_PREFIX = "modal_controller"
MODAL_ARTIFACT_GLOB = f"{MODAL_ARTIFACT_PREFIX}*.npz"


@dataclass(frozen=True)
class ModalArtifactInfo:
    """A built modal-LQR controller found on disk, with the descriptors that tell
    it apart from other builds of the same graph."""

    path: Path
    reduced_order: int
    n_modes: int
    T_op_K: float
    graph_name: str

    @property
    def label(self) -> str:
        return f"Modal LQR r={self.reduced_order} / {self.n_modes} modes / T_op={_format_T_op(self.T_op_K)}"


def _format_T_op(value: float) -> str:
    return f"{float(value):g} K"


def modal_artifact_filename(reduced_order: int, n_modes: int, T_op_K: float) -> str:
    """Descriptor-derived artifact name, so builds that differ in order, mode count
    or operating point sit side by side instead of overwriting each other.
    Rebuilding with identical descriptors intentionally overwrites."""
    temperature = f"{float(T_op_K):g}".replace(".", "p").replace("-", "m")
    return f"{MODAL_ARTIFACT_PREFIX}_r{int(reduced_order)}_m{int(n_modes)}_T{temperature}K.npz"


def describe_modal_artifact(path: Path | str) -> ModalArtifactInfo | None:
    """Read a built artifact's descriptors, or None if it is unreadable/not one.

    Descriptors come from inside the .npz, not the filename, so artifacts built
    by tools/analyze_plant_modes.py or renamed by hand still describe correctly.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=True) as data:
            if "K" not in data or "dc_gain_pinv" not in data:
                return None
            return ModalArtifactInfo(
                path=path,
                reduced_order=int(data["r"]) if "r" in data else int(np.asarray(data["K"]).shape[1]),
                n_modes=int(data["n_modes"]) if "n_modes" in data else 0,
                T_op_K=float(data["T_op_K"]) if "T_op_K" in data else 0.0,
                graph_name=str(data["graph"]) if "graph" in data else "",
            )
    except Exception:  # noqa: BLE001 - a corrupt/foreign .npz is simply not offered
        return None


def list_modal_artifacts(folder: Path | None) -> list[ModalArtifactInfo]:
    """Every readable modal-LQR artifact in a graph folder, newest build first."""
    if folder is None:
        return []
    folder = Path(folder)
    if not folder.is_dir():
        return []
    infos: list[tuple[float, ModalArtifactInfo]] = []
    for candidate in sorted(folder.glob(MODAL_ARTIFACT_GLOB)):
        info = describe_modal_artifact(candidate)
        if info is None:
            continue
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            mtime = 0.0
        infos.append((mtime, info))
    infos.sort(key=lambda item: (-item[0], str(item[1].path)))
    return [info for _mtime, info in infos]
