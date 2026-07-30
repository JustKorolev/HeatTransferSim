# Handling Mismodeling: Robustness / Adaptive Options for the Controller

The LQR + feedforward do a good job, but they are **model-based** — they depend on
the conductance and other matrices we approximate. So it's worth having an
additional model-agnostic component that adapts to mismodeling error. The integral
already does some of this; the question is whether to implement something more
elaborate. Below is the reasoning and a ranked recommendation.

## First: the integral already *is* the model-agnostic backstop — for steady-state error

The integral is **provably offset-free**: for any constant/slow mismodeling (wrong
conductances, wrong sink, wrong DC gain), it drives the tracking error to **zero**
regardless of how wrong the feedforward/LQR gains are. So for the *dominant* class
of error here (steady-state), correctness is already covered. What the integral
lacks is **speed**, and it does nothing for **dynamic** mismatch. Split the problem:

| mismodeling type | example | who handles it | gap |
|---|---|---|---|
| **steady-state / DC** | wrong conductance, wrong sink | **integral (already, offset-free)** | slow |
| **dynamic / transient** | wrong τ, wrong mode coupling | LQR feedback (model-based) | can be suboptimal, rarely unstable for thermal |

For a **slow, stable, over-damped, minimum-phase** plant like this, dynamic
mismodeling almost never causes instability — it just costs transient optimality.
So the high-value target is the **steady-state** part, and specifically its
**speed**, not robustness to exotic dynamics.

## Recommendation, ranked

**(A) Adaptive / learning feedforward — the best fit here.** The feedforward is the
*only purely open-loop, 100%-model-dependent* piece, and it's exactly what the
integral is *slowly* compensating for. So learn it: as the integral accumulates the
true holding power for a given setpoint, **regress (setpoint-error → steady-state
heater power) online** (recursive least squares) to estimate the *actual* inverse
DC gain, and fold it back into `dc_gain_pinv`. Effect: the controller **learns the
DC gain the matrices got wrong**, so future setpoints get correct feedforward
*immediately* instead of waiting minutes for the integral. It's lightweight (an RLS
update), it *loves* a slow plant (plenty of time, slow disturbances), and it
directly attacks the stated worry. This is essentially online model-correction of
the one piece that can't self-correct.

**(B) Integral-augmented LQR (LQI / servo-LQR).** Instead of a hand-tuned separate
integral, augment the reduced state with the integral-of-error and let the LQR
design the integral action **optimally and MIMO-coordinated** (weights from the
cost, not by hand). It's a principled upgrade of what we already have, still
deployable on the Nucleo, and it removes the ad-hoc `ki` tuning. Modest effort,
solid payoff.

**(C) Periodic offline re-identification.** Not "online adaptive," but the most
*direct* fix for "the matrices are approximate": occasionally step the heaters,
measure, estimate the *actual* conductance/gain matrices, and re-run the modal
pipeline to refresh the controller. Low-risk, leverages the pipeline we already
built, and fixes the root cause instead of patching around it.

## What to avoid (for now)

- **Naive disturbance observer** — we already tried the `d_passive = ṁeas − B_s·u`
  version and it **destabilized** (the rate-gain `B_s` is ill-conditioned on this
  plant). A DOB can work but needs a well-conditioned nominal, which we don't have.
- **MRAC / L1 adaptive** — powerful but hard to certify, and the fast-adaptation
  benefit is wasted on a ~47-minute plant with slow disturbances.
- **Robust H∞ / μ-synthesis** — buys worst-case robustness by *giving up*
  performance, and doesn't *improve* with data. We want adaptation, not conservatism.
- **Full MPC** — handles the one-sided heater constraints elegantly, but the
  per-step QP is heavy for the H563, and it's only worth it with heavier state
  reduction.

## Bottom line

Don't replace the integral — **augment it**. The single highest-leverage "more
elaborate" addition is an **online-learned feedforward (A)**: it turns the
integral's slow, reactive model-correction into a fast, stored model-correction,
targets the exact piece that's purely model-dependent, and is MCU-friendly. Pair it
with **LQI (B)** if you want the feedback/integral to be optimally coordinated
rather than hand-tuned. That gives a controller that's model-based where the model
is good (fast conduction dynamics) and self-correcting where it isn't (the
steady-state gain/sink) — without the certification headaches of full adaptive or
MPC.
