# The big three: validation slides and speaker notes

Three figures, chosen so each isolates a different part of the implementation.
All runs use the converged solver configuration (512 adaptive substeps, target
ΔT 1e-3 K, rtol 1e-11, CPU, direct graph build).

---

## Slide 1 — Heat transfer equation and spatial discretization

**Figure:** `val_2_prism_conduction`

**Setup.** Copper prism, 100 × 10 × 10 mm, 5 mm voxels (20 cells along the
axis). Uniform 300 K initial state, one end face clamped to 200 K, the far end
insulated. Run out to 60 s — copper's L²/α is ~86 s, so this covers most of a
relaxation. Probes at x/L = 0.25, 0.50, 0.75, 1.00.

**Reference.** The closed-form series solution of the 1-D heat equation:

    θ = (4/π) Σ ((−1)ⁿ/(2n+1)) cos((2n+1)πx/2L) exp(−α((2n+1)π/2L)² t)

evaluated to 100 terms. This is derived from the continuum PDE — it knows
nothing about the graph, the voxelisation, or the solver.

**Result.** Max error 0.96 K against a 100 K excursion, ≈1%, inside the 3 K
tolerance by 3×. The error is concentrated in the first ~2 s at the probe
nearest the clamped face and decays monotonically after; by 20 s every probe is
within 0.15 K.

**What this proves.** This is the only case in the suite whose reference is
independent of the discretization, so it is the one that validates the model
rather than the integration. It exercises the whole chain at once: Laplacian
assembly, the conduction area in `G = kA/L`, and the capacitance
`C = ρ·cp·V`. An error in any of those would change the effective diffusivity
and separate the curves visibly — agreement to 1% bounds all of them jointly.

**Say it this way.** "Spatial conduction matches the analytical solution of the
heat equation to about 1% of the temperature excursion."

**Limitation to hold.** One resolution, one geometry. The 1% is discretization
error, and without a mesh-refinement study we can't yet claim it converges at
the expected order — only that it is small here. The error signature (early
transient, nearest the boundary, decaying) is exactly what coarse cells at a
steep gradient produce, which is the expected behaviour, not a defect.

---

## Slide 2 — Solver

**Figure:** `val_6_cryo_regime`

**Setup.** Copper bar starting at 40 K with a 2 W heater, ambient radiation to a
40 K environment, and temperature-dependent cp(T) from the NIST curves — all
three nonlinearities active simultaneously. 60 s, 0.2 s steps. This is the
operating regime the flight hardware actually runs in.

**Reference.** The same nonlinear ODE integrated by scipy's LSODA at rtol/atol
1e-9, from the same initial state and the same operators.

**Result.** Max error 2.9e-4 K, roughly 1700× inside the 0.5 K tolerance. The
error appears as a single transient spike in the first step and then flattens —
it is startup, not drift.

**What this proves.** The implicit stepping and the operator splitting are
converged on the hard problem: a temperature-dependent capacitance matrix that
has to be rebuilt each step, plus a T⁴ radiation term, plus conduction. Two
independent integration schemes on the same operators land on the same
trajectory to sub-millikelvin, so the time discretization contributes
essentially nothing to the error budget.

**Say it this way.** "On the nonlinear coupled problem, our implicit solver and
a high-accuracy reference integrator agree to 0.3 millikelvin."

**Limitation to hold — state this one plainly.** Because the reference is built
from the *same* discrete operators, this validates the integration, not the
operators. It would still pass if a conductance or a radiating area were wrong.
That's why Slide 1 is doing the physics validation and this slide is doing the
solver validation. If asked: the two are deliberately separate claims.

---

## Slide 3 — Coupled-term bookkeeping

**Figure:** `val_3_energy_conservation`

**Setup.** Copper bar 60 × 10 × 10 mm, ~5 cells, 20 W heater, ambient radiation
to 300 K, conduction internal. 200 s. Tracks stored internal energy
`U = Σ Cᵢ(Tᵢ − Tᵢ₀)` against the trapezoidal integral of net power in
(heater minus radiated).

**Result.** Peak imbalance 1.8e-3 J against ~3.6 kJ delivered — about 1 part in
2 × 10⁶, four orders inside the 36 J tolerance. The residual grows smoothly and
monotonically, consistent with trapezoidal quadrature error on the reference
integral rather than a leak in the solver.

**What this proves.** Conduction is internal, so it must net to zero across the
whole body; any sign flip or scale error in the coupled terms shows up here as a
divergence between the two curves. It's the test that catches the class of bug
where each piece looks right in isolation but the assembly double-counts or
drops a term.

**Say it this way.** "With heater, radiation and conduction all active at once,
global energy balance closes to one part in two million."

**Limitation to hold.** Both sides of this comparison use the same radiation
coefficients, so it verifies internal consistency, not the absolute correctness
of the radiating area. It is a strong check on signs and assembly, a weak one on
geometry.

---

## The honest summary slide

If you want one sentence covering all three:

> The time integration is converged to sub-millikelvin, global energy balance
> closes to one part in 2 × 10⁶, and spatial conduction matches the closed-form
> solution of the heat equation to ~1%.

That claim is fully supported by these three figures and survives follow-up
questions. Avoid "9 of 9 cases pass" — it mixes independent-reference tests with
same-operator integration checks, and it invites a question about the distributed
rod case, which is currently degenerate (all nodes flat, comparing zero to zero).

**Known gap, if asked what isn't validated.** Radiating surface area has no
independent check — every radiation test builds its reference from the same
`radiation_coeff_W_K4` the solver uses, so a wrong wetted area cancels on both
sides. Conduction area is covered indirectly by Slide 1. The fix is a
steady-state radiative equilibrium case where `T_final = (P/(εσA) + T_env⁴)^¼`
with A computed by hand from the box dimensions.
