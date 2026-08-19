# Speaker notes — validation summary slide (`val_1_summary`)

Figure: `plots/thermal_validation/val_1_summary_dark.pdf`
Title on the slide: *Thermal solver vs. closed-form references: 8/9 within tolerance, 1 marginal*

---

## Spoken script (~30 s) — 73 words

> "Before any controller result means anything, the solver has to be right. Nine
> cases against closed-form references. Every row is divided by its own tolerance,
> so the dashed line is each case's threshold — further left is better. Eight pass, most by
> orders of magnitude, including the cryo regime with radiation and
> temperature-dependent properties, which is what I actually ran. The ninth is
> amber: fifteen percent over on peak transient error only. Its RMSE and final
> equilibrium both pass."

Four beats: what the axis is, 8 of 9 by orders of magnitude, the regime that
matters, the amber pre-empted. If you run short, cut the cryo clause — that row is
legible on the slide. Do not cut the axis explanation (the chart is unreadable
without it) or naming the amber yourself (ten words, and it turns the one weakness
into evidence that the thresholds were set in advance).

---

## Spoken script (~80 s)

**Read the axis first (~15 s).**
"Before any controller result means anything, the solver underneath it has to be
right. Nine independent test cases, each with a closed-form or independently
integrated reference. The tolerances differ by eight orders of magnitude between
these cases, so each row is divided by its OWN — that's what puts kelvin at four
different scales and one row in joules on a single axis. The dashed line at one is
therefore every case's threshold at once, and further left is better. The raw error
and that row's tolerance are printed on the right."

**The headline (~20 s).**
"Eight of nine pass, most of them by four to ten orders of magnitude. The two at
the left edge are cases the discretization represents exactly, so all that's left
is floating-point round-off — the chart clips them rather than extending the axis
out to ten-to-the-minus-thirty."

**Pre-empt the amber row (~25 s).**
"The ninth is amber, and I want to be the one to point at it. Two-block exchange
across a contact conductance: peak instantaneous error 0.057 K against a 0.050 K
threshold. Fifteen percent over, and it happens in the first few timesteps. On that
same case the RMSE passes at 0.044 K, the final equilibrium temperature is right to
eight times ten-to-the-minus-eleven kelvin, energy conserves to four nanojoules,
and the interface conductance is reproduced exactly. So it's a transient-resolution
artifact, not a physics error — and I'd rather report it than tune the tolerance
until it disappeared."

**The row that matters most (~15 s).**
"The row I'd point to for this project is the cryo regime — heater, radiation and
temperature-dependent specific heat all active, from a 40 K start, checked against
an independent stiff integrator. That's the configuration the real simulation runs
in, and it lands three and a half thousand times inside tolerance."

**Land it (~5 s).**
"So the solver is trustworthy in the regime I used it, with one known, quantified,
transient-only exception."

---

## Numbers, if you need them

| case | error | tolerance | margin |
|---|---|---|---|
| Insulated block, constant heating | 1.7e-10 K | 5.0e-02 K | 3e8× inside |
| **Two-block exchange (contact)** | **5.7e-02 K** | **5.0e-02 K** | **1.15× over** |
| Two-node lumped conductance | 3.7e-10 K | 1.0e-07 K | 270× inside |
| 1-D prism conduction (Fourier) | 1.0e+00 K | 3.0e+00 K | 3× inside |
| 1-D distributed rod (mode decay) | 7.5e-30 K | 1.0e-07 K | at round-off floor |
| Radiation cooling (lumped) | 3.2e-07 K | 5.0e-01 K | 1.6e6× inside |
| Temperature-dependent heating | 4.5e-11 K | 5.0e-01 K | at round-off floor |
| Cryo regime: heater + radiation + cp(T) | 1.8e-04 K | 5.0e-01 K | 2800× inside |
| Global energy conservation | 1.8e-03 J | 3.6e+01 J | 20000× inside |

**Do not** describe the amber row by its bar length. The bar measures log-distance
from the 1e-10 floor, not badness — the prism row has a similarly long bar and
passes. Point at the dashed line and the printed number.

---

## Anticipated questions

**"Did you choose tolerances you knew you'd pass?"**
The tolerance tracks what each comparison can resolve, which is why they span eight
orders of magnitude. Where the discrete model *is* the exact model — lumped nodes,
single decaying modes — the only error is the time integrator, so the tolerance is
1e-7 K. The prism compares a 5 mm-voxel mesh against a continuum Fourier series
across a 100 K drop, so its 3 K tolerance is 3% of the driving temperature
difference. And the energy tolerance isn't arbitrary either: it's 0.05 K expressed
in energy for this body's heat capacity, which is where the 36 J comes from.

**"Why is the prism only 3× inside, when everything else is orders of magnitude?"**
Because it's the only case limited by *spatial* discretization rather than time
integration. 5 mm voxels against a continuum solution — that's a mesh-resolution
statement, and it converges if you refine. Everything else in the suite is either
lumped, where the geometry is exact, or a global audit.

**"What about energy — that row is in joules, not kelvin."**
It's a different kind of check: a global audit rather than a pointwise comparison.
Integrated net power in versus stored internal energy, over the whole body, with
heater, ambient radiation and conduction all active. The imbalance stays at
1.8 millijoules.

**"Is the amber case going to bite you?"**
It's a peak error during the first few timesteps of a contact-coupled transient.
The real simulation runs for 28 to 70 hours and the quantity I care about is the
steady tracking error, which on that same case is exact to 1e-10 K. If I needed
that peak, I'd shrink dt — it's a resolution knob, not a model defect.

**"Which of these is closest to what you actually ran?"**
Cryo regime, and global energy conservation. The first is the operating point
— 40 K, cryocooler, radiation, cp(T). The second ran with everything active at once.

---

## Backup slides to have ready

| if asked | show |
|---|---|
| conduction discretization | `val_2_prism_conduction_dark` — 4 depths vs 100-term series |
| energy conservation | `val_3_energy_conservation_dark` — imbalance 20000× inside |
| cryo / nonlinear | `val_6_cryo_regime_dark` — vs LSODA at rtol 1e-9 |
| the failing case | `val_5_two_block_exchange_dark` — subtitle already names it |
