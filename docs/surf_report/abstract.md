<!-- SFP-formatted submission version: abstract.docx (Verdana 8pt, etc.).
     This Markdown is the plain-text/content copy. -->

# A reduced-order thermal model and multi-input controller for the HISPEC cryostat

Andrey Korolev
*Mentor: Jake Zimmer*

Cryogenic astronomical instruments such as the HISPEC spectrograph must hold
large optical assemblies at a stable operating temperature near 55 K, yet their
thermal behavior is set by hundreds of thousands of coupled conduction and
radiation pathways — far too many states to simulate in real time or to design a
controller against. This project develops an end-to-end pipeline that turns a
CAD assembly into a lumped thermal-network model and reduces it to a compact form
suitable for on-board control. From the instrument geometry an octree mesh builds
a 469,315-node thermal graph; the linearized dynamics are projected onto their
120 slowest physical modes and then reduced by balanced truncation to a 30-state
model that keeps only the jointly controllable and observable dynamics. On this
reduced model I design a multi-input linear-quadratic-regulator controller with a
static state estimator and a regularized steady-state feedforward for its 33
heaters and 28 controlled sensors. The 30-state model reproduces the full plant's
heater-to-sensor step response to within about 0.15 %, small enough to design
against while remaining light enough to run on a microcontroller. Remaining work
adds surface-to-surface radiation and hardware deployment.
