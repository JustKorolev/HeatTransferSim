# The Modal-LQR Controller Pipeline

End to end: the physics model, the two-stage reduction, the controller design, and
the runtime law — with the actual matrices and shapes for CRYOSTAT_V2 (~469k
main-component nodes, 33 heaters, 72 sensors, 28 controlled).

---

## 0. The plant: a lumped thermal graph

Every cell is a lumped thermal mass; the network obeys an energy balance. In
deviation variables `x = T − T_op`:

$$\mathbf{C}\,\dot x = -(\mathbf{L} + \operatorname{diag}(\mathbf{G_{rad}}))\,x + \mathbf{F}\,u$$
$$y = \mathbf{S}\,x$$

| matrix | shape | meaning |
|---|---|---|
| `C` | N (diag) | heat capacity `C_i = mass_i·cp_i` (J/K) |
| `L` | N×N | conduction Laplacian: `L_ij=−G_ij`, `L_ii=Σ_j G_ij`. Symmetric PSD, **singular** (a uniform shift is a null mode) |
| `diag(G_rad)` | N | linearized radiation-to-ambient (a sink that grounds `L`) |
| `F` | N×33 | heater map: column *j* = heater *j*'s normalized deposition weights over its cells |
| `u` | 33 | heater powers (W) |
| `S` | 72×N | sensor readout map: row *k* = sensor *k*'s readout weights (rows sum to 1) |

So the state-space is `ẋ = −C⁻¹(L+diag(G_rad))x + C⁻¹F u`, `y = Sx`. This is
N≈469k-dimensional — impossible to run LQR on, or fit on the Nucleo-H563.
Everything below is about shrinking it to r=50 **without losing the input→output
behavior**.

---

## 1. Clean the operator

- `drop_inert_cells` — remove near-zero-capacity cells (markers/void) that would
  inject spurious near-zero eigenvalues.
- `largest_connected_component` — keep the main conductive body (469,315 cells).

---

## 2. Stage-1 reduction — slow thermal modes

The natural modes of `C ẋ = −(L+G_rad)x` solve the **generalized symmetric
eigenproblem**:

$$(\mathbf{L}+\operatorname{diag}(\mathbf{G_{rad}}))\,\phi = \lambda\,\mathbf{C}\,\phi$$

- `λ = 1/τ` — inverse time constant. **Small λ = slow mode** (the slowest here is
  τ≈2813 s ≈ 47 min).
- `φ` — the spatial temperature pattern of that mode.

Solved numerically by symmetric scaling `D = C^{−1/2}`, `A = D(L+G_rad)D`
(symmetric), then shift-invert `eigsh(A, k=120, σ≈0)` for the **120 slowest**.
Modes are **C-orthonormalized**: `ΦᵀCΦ = I`. → `Φ ∈ ℝ^{N×120}`.

Project `x ≈ Φq` (q = modal coordinates):

$$\mathbf{A_{mod}} = -\operatorname{diag}(\lambda)\ (120{\times}120,\ \text{diagonal}),\quad \mathbf{B_{mod}} = \Phi^\top \mathbf{F}\ (120{\times}33),\quad \mathbf{C_{out}} = \mathbf{S}\Phi\ (72{\times}120)$$

Now `q̇ = A_mod q + B_mod u`, `y = C_out q` — 120 states instead of 469k. We kept
the slow heat-diffusion patterns and discarded the fast ones (they settle
instantly on control timescales).

---

## 3. Stage-2 reduction — balanced truncation to r=50

120 slow modes ≠ 120 *useful* modes. Balanced truncation keeps the modes that are
simultaneously **controllable and observable** (i.e. actually connect `u`→`y`),
which "slowest" alone doesn't guarantee.

- Controllability Gramian: `A_mod W_c + W_c A_modᵀ = −B_mod B_modᵀ`
- Observability Gramian: `A_modᵀ W_o + W_o A_mod = −C_outᵀ C_out`
- **Hankel singular values** `σ = svd(L_o L_c)` (from factors of `W_c,W_o`) — each
  measures a balanced mode's input↔output energy.
- Square-root balancing builds `T`, `T_i` keeping the top-r σ:

$$\mathbf{A_r}=T_i A_{mod}T\ (50{\times}50,\ \textbf{full/coupled}),\quad \mathbf{B_r}=T_iB_{mod}\ (50{\times}33),\quad \mathbf{C_r}=C_{out}T\ (72{\times}50)$$

Reduced plant: `ẋ_r = A_r x_r + B_r u`, `y = C_r x_r`. Verified DC error ~3e-5,
step error ~6e-5 vs the 120-mode model — i.e. the 50-state model reproduces the
true input→output map to ~1e-5. **This is the answer to "don't we lose coupling
info?" — no; balanced truncation preserves exactly the input-output map, which
*is* the coupling.**

---

## 4. LQR feedback gain `K` — designed in DISCRETE time

Minimize `∫(y_ctrlᵀy_ctrl + ρ·uᵀu) dt` over the 28 controlled outputs:

- `C_ctrl = C_r[ctrl_idx]` (28×50)
- `Q = C_ctrlᵀC_ctrl (+ tiny ridge)` (50×50), `R = ρI` (33×33)

The controller is **sampled**, not continuous: it computes once per `dt_s` and
holds. So the gain is solved against the zero-order-hold discretization
`(A_d, B_d) = ZOH(A_r, B_r, dt)` and the **discrete** Riccati equation, with the
weights scaled by `dt` so `ρ` keeps its meaning across sample rates:

$$\mathbf{K} = (R\,dt + B_d^\top P B_d)^{-1}B_d^\top P A_d\quad (33{\times}50)$$

**Why not the continuous ARE.** A continuous-time gain assumes the loop is closed
continuously, which is only safe while `dt` is short against every mode the gain
acts on. This plant spans τ = 0.0018 s to ~2800 s, so at `dt = 4 s` the local
heater→sensor path settles ~2000× faster than one control step and presents its
full DC gain within a single sample. Measured on the `no_mli_high_res` run of
2026-08-10: 11 eigenvalues of the per-step loop gain `G·K·E_ctrl` exceeded 2
(peak 5557), i.e. closed-loop poles below −1, and every heater command flipped
sign on every single sample (sign-flip-per-sample of exactly 1.00). It stayed
bounded only because the heaters clipped at 0 W. Designing at the actual `dt`
fixes this by construction — the gain shrinks automatically as `dt` coarsens:

| dt | ‖K‖ | max closed-loop pole | …if you used the continuous `K` |
|---|---|---|---|
| 0.1 s | 2247 | 0.999 | **2.10** |
| 1.0 s | 463 | 0.987 | **82** |
| 4.0 s | 681 | 0.947 | **499** |
| 8.0 s | 932 | 0.897 | **1136** |

`K` is **full MIMO** — every heater's command uses the full 50-state estimate,
coordinating all heaters against the cross-coupling. `ρ` (the effort weight) trades
power vs speed (larger ρ = gentler/slower, smaller ρ = more aggressive/faster).
Feedback: `u = −K x_r`.

The artifact stores `design_dt_s` alongside `A_r, B_r, Q, R`, so running at a
different `dt_s` re-solves the discrete Riccati equation at load (~4 ms) instead
of silently applying a gain that does not belong to that sample rate. The MCU
keeps the baked `K`, since it runs at a fixed rate.

---

## 5. State estimator `E_reg`

At runtime we measure `y` (72 sensors), not `x_r` (50 states). Instead of a dynamic
observer (heavier), we trust the sensors and use a **regularized static inverse**
of `y = C_r x_r`:

$$\mathbf{E_{reg}} = (C_r^\top C_r + \mu I)^{-1}C_r^\top\quad (50{\times}72),\qquad \hat x = E_{reg}\,(y-T_{op})$$

`μ` is a small Tikhonov term (`reg_frac·σ_max²`) for conditioning.

---

## 6. Feedforward — the exact plant DC gain

Two pieces exist; the **exact DC gain is preferred** (the reduced servo maps
`Nx, Nu` are ill-conditioned at DC and kept only as fallback).

$$\mathbf{G} = \mathbf{S}_{ctrl}\,\mathbf{L_{dc}}^{-1}\,\mathbf{F}\quad (28{\times}33)$$

where `L_dc` is the steady-state operator **grounded at the cryocooler,
conduction-only** (radiation left out because its linearized sink drifts with
ambient temperature). `G_ij` = steady-state K rise at sensor *i* per W into heater
*j*. Then the regularized pseudo-inverse:

$$\mathbf{dc\_gain\_pinv} = (G^\top G + \lambda I)^{-1}G^\top\quad (33{\times}28),\qquad u_{ff} = \mathbf{dc\_gain\_pinv}\cdot r_{sp}$$

This is the open-loop power to *reach* the setpoint. (`|G|` is this matrix's norm;
large `|G|` = weak sink = small feedforward.)

---

## 7. The artifact

`modal_controller.npz` stores exactly what the runtime needs:
`K, E_reg, Nx, Nu, dc_gain_pinv, dc_gain(G), heater_ids, sensor_ids, monitor,
T_op_K, integral_gain, r`, plus `design_dt_s, A_r, B_r, Q_lqr, R_lqr` so the gain
can be re-derived at another sample rate.

`integral_gain` here is **metadata only** — the runtime reads
`params.modal_integral_gain` fresh every step and never consults the artifact. So
`ki` is fully decoupled from the LQR build and retunable live, without a rebuild.

---

## 8. Runtime control law (each timestep)

Given measurements `y` and setpoints:

1. `y_dev = y − T_op` (deviations from the 50 K operating point)
2. `x̂ = E_reg · y_dev` — state estimate
3. `r_sp = setpoints − T_op` — controlled-sensor reference (28)
4. `r_full = y_dev; r_full[ctrl] = r_sp;  x_ff = E_reg · r_full` — **target state built
   with the *same* estimator** (this is what stops the `K·x_ff` blow-up), so
   `x̂ − x_ff` is a clean tracking error
5. `u_ff = dc_gain_pinv · r_sp` — feedforward reach
6. `u_int += ki·(dc_gain_pinv · error)·dt` — offset-free integral
   (`error = r_sp − y_dev[ctrl]`)
7. `base = u_ff − K·(x̂ − x_ff)` — feedforward + LQR feedback regulating the estimate
   toward target
8. `u = clip(base + u_int, 0, u_max)`, then **slew-limit** `|Δu| ≤ slew·dt`
9. deposit `u` into the heater cells

$$u = \underbrace{u_{ff}}_{\text{open-loop reach}} \;-\; \underbrace{K(\hat x - x_{ff})}_{\text{MIMO feedback}} \;+\; \underbrace{u_{int}}_{\text{offset-free trim}}\quad\text{(clamped, slew-limited)}$$

---

## Why each piece is there

- **Modal + balanced reduction** (469k→50): fits the Nucleo-H563 and makes LQR
  tractable, while preserving the true I/O map.
- **LQR `K`**: optimal MIMO coordination of the coupled dynamics (vs independent
  SISO PID).
- **Static estimator `E_reg`**: trust the sensors, no dynamic observer — light and
  deployable.
- **Exact-DC feedforward**: correct steady-state reach from the *full-plant* gain,
  not the ill-conditioned reduced one.
- **Integral**: offset-free — absorbs model error and disturbances (like the
  uncertain radiation load) that the feedforward deliberately doesn't model.

The through-line: `K`, `E_reg`, and `dc_gain_pinv` are all **derived from
`C, L, G_rad, F, S`** — so the controller is exactly as good as those matrices,
which is why so much of the modeling work is getting the matrices right
(conductances, sink, materials) rather than tuning the control law.
