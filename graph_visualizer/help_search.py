"""Ranking for the in-app help search, with no Qt in it.

The app has five tabs and something like two hundred controls, most of them
named for the control theory behind them rather than for what the user is trying
to do. "Where do I cap heater power?" is not answerable by reading labels, and
the answer ("max heater power W (all heaters)", in Controller (limits), on either
simulation tab) is not guessable either.

This module holds the part of the answer that is pure data: a
:class:`HelpTarget` per reachable control, and a ranking that decides which of
them a query means. The Qt side -- walking the widget tree to build the targets,
switching tabs, scrolling, and flashing the result -- lives in
``help_center.py``, so the ranking can be tested without a display.

The targets are BUILT FROM THE LIVE WIDGET TREE rather than listed by hand.
A hand-written index of two hundred controls goes stale the first week: this
repo's own module README claimed the viewer had two tabs when it had five. An
index derived from the widgets cannot drift from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

#: Weights for where a query token matched. A label hit is worth far more than a
#: tooltip hit, because tooltips here are paragraphs -- matching one word of a
#: 200-word tooltip means much less than matching the control's name.
_W_LABEL_EXACT = 1000.0
_W_LABEL_PREFIX = 400.0
_W_LABEL_WORD = 250.0
_W_LABEL_SUBSTRING = 120.0
_W_KEYWORD = 90.0
_W_SECTION = 45.0
_W_TAB = 25.0
_W_TOOLTIP = 12.0
#: Read-only rows stay findable but must never outrank an actionable control.
_READOUT_PENALTY = 0.3


@dataclass(frozen=True)
class HelpTarget:
    """One thing the user can be taken to.

    ``widget`` is whatever the Qt layer needs to reveal it and is never inspected
    here; ``tab_index`` is the tab that must be shown first, or None for something
    that is always visible.
    """

    key: str
    label: str
    tab: str = ""
    section: str = ""
    tooltip: str = ""
    keywords: tuple[str, ...] = ()
    tab_index: int | None = None
    widget: Any = None
    kind: str = "control"

    def describe(self) -> str:
        """'Section > label' with the tab in front, for a result row."""
        parts = [p for p in (self.tab, self.section) if p]
        trail = " > ".join(parts)
        return f"{trail} > {self.label}" if trail else self.label


@dataclass
class ScoredTarget:
    target: HelpTarget
    score: float
    matched: tuple[str, ...] = field(default_factory=tuple)


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]


def _score_token(token: str, target: HelpTarget) -> float:
    """How well one query token matches one target. 0 means it does not."""
    label = target.label.lower()
    if not token:
        return 0.0
    if label == token:
        return _W_LABEL_EXACT
    label_words = _tokens(label)
    if label.startswith(token):
        return _W_LABEL_PREFIX
    if token in label_words:
        return _W_LABEL_WORD
    # A token that is a prefix of some word in the label ("temp" -> "temperature")
    # is a real hit; the user is typing, not reciting.
    if any(word.startswith(token) for word in label_words):
        return _W_LABEL_WORD * 0.8
    if token in label:
        return _W_LABEL_SUBSTRING
    for keyword in target.keywords:
        keyword = keyword.lower()
        if keyword == token or keyword.startswith(token):
            return _W_KEYWORD
        if token in keyword:
            return _W_KEYWORD * 0.6
    if token in target.section.lower():
        return _W_SECTION
    if token in target.tab.lower():
        return _W_TAB
    if token in target.tooltip.lower():
        return _W_TOOLTIP
    return 0.0


def score_target(query: str, target: HelpTarget) -> tuple[float, tuple[str, ...]]:
    """Total score for ``target`` against ``query``, and the tokens that matched.

    EVERY token must match something. A two-word query is a conjunction -- "heater
    power" must not return every control with "heater" in it -- and returning a
    long list of near-misses is the same as returning nothing, because the user
    still has to read all of it.
    """
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0, ()
    total = 0.0
    matched: list[str] = []
    for token in query_tokens:
        value = _score_token(token, target)
        if value <= 0.0:
            return 0.0, ()
        total += value
        matched.append(token)
    # Prefer the shortest label among equal matches: "input mode" should beat
    # "max solver internal step [s]" for the query "mode". Small enough never to
    # outweigh a category difference.
    total -= min(len(target.label), 60) * 0.1
    # A read-only readout is not what "take me to the control" means. The status
    # label beside Export Controller Constants is literally labelled "export", so
    # it beat the button it describes on the query "export controller".
    if target.kind == "readout":
        total *= _READOUT_PENALTY
    return total, tuple(matched)


def search(query: str, targets: list[HelpTarget], limit: int = 12) -> list[ScoredTarget]:
    """The best ``limit`` targets for ``query``, best first.

    Ties break on the target's own order, which is build order, which is the order
    the controls appear on screen -- so equally good answers are offered
    top-to-bottom as the user would find them.
    """
    scored: list[ScoredTarget] = []
    for index, target in enumerate(targets):
        value, matched = score_target(query, target)
        if value > 0.0:
            # index * 1e-6 keeps the sort stable without perturbing real scores.
            scored.append(ScoredTarget(target, value - index * 1.0e-6, matched))
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[: max(0, int(limit))]


# --------------------------------------------------------------------------- #
# Tutorials
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TutorialStep:
    """One step. ``target_key`` makes the step's "Show me" button work."""

    title: str
    body: str
    target_key: str = ""
    image: str = ""


@dataclass(frozen=True)
class Tutorial:
    key: str
    title: str
    summary: str
    steps: tuple[TutorialStep, ...]


#: Written against what the app actually does. Each step that names a control
#: carries its row key, so "Show me" reveals and flashes the real widget instead
#: of describing where it is and hoping.
TUTORIALS: tuple[Tutorial, ...] = (
    Tutorial(
        key="first_run",
        title="Run your first simulation",
        summary="Load a graph, initialize it, and play it with the controller in the loop.",
        steps=(
            TutorialStep(
                title="Pick a graph",
                body=(
                    "On the Heat Transfer Simulation tab, choose a folder from the graph "
                    "dropdown. Graphs live in graphs/ and are built from CAD by "
                    "build_octree_graph.py -- the app does not create them.\n\n"
                    "Node count is shown as it loads. Anything above about 250,000 cells "
                    "will warn you before it tries to draw itself."
                ),
            ),
            TutorialStep(
                title="Set the input mode",
                body=(
                    "'zero' runs the plant with no heater input at all -- useful for "
                    "watching it drift to its passive equilibrium.\n\n"
                    "'heater_inputs' is the one that closes the loop: the controller "
                    "selected below regulates the heaters against each sensor's setpoint."
                ),
                target_key="input_mode",
            ),
            TutorialStep(
                title="Choose a controller",
                body=(
                    "'(no controller selected)' leaves the heaters unregulated -- the run "
                    "is open-loop apart from cryocoolers and any manual heaters.\n\n"
                    "MIMO PI needs a gain matrix G, which you get from a sys ID. Modal LQR "
                    "needs a modal_controller.npz, which you build in Controller Design."
                ),
                target_key="controller",
            ),
            TutorialStep(
                title="Initialize, then play",
                body=(
                    "Initialize builds the operator matrices and prepares the solver. Play "
                    "then advances the simulation and refreshes the 3D view.\n\n"
                    "For anything long, use the Headless Run tab instead: it launches the "
                    "run as a separate process that survives closing the app, and never "
                    "loads the graph into the window."
                ),
                target_key="initialize",
            ),
        ),
    ),
    Tutorial(
        key="controller",
        title="Identify and tune the MIMO PI controller",
        summary="Run a sys ID for G, set Kp and Ki, and understand why it is not per-pair PID.",
        steps=(
            TutorialStep(
                title="Why there is no per-pair PID",
                body=(
                    "On this plant the RGA diagonal is negative on 26 of 27 pairings, and "
                    "only about 0.7% of a heater's steady influence lands on its own "
                    "sensor. A SISO loop per heater/sensor pair drives the WRONG WAY once "
                    "its neighbours close.\n\n"
                    "So the controller inverts the whole DC gain matrix G once. After that "
                    "the loop from the virtual command to the sensors is the identity, and "
                    "the PI runs as independent scalar channels in that decoupled space. "
                    "This is why a gain matrix is mandatory and Kp/Ki alone mean nothing."
                ),
            ),
            TutorialStep(
                title="Run the sys ID",
                body=(
                    "The sys ID steps each heater in turn and measures every sensor's "
                    "steady response, giving G in kelvin per watt. Set the step power high "
                    "enough to move the sensors well clear of numerical noise, and the "
                    "durations long enough for the plant to actually settle -- this plant's "
                    "slowest modes run to many hours.\n\n"
                    "The result is saved as a run folder under the graph's sys_id/."
                ),
                target_key="run_sys_id",
            ),
            TutorialStep(
                title="Set Kp and Ki",
                body=(
                    "After decoupling every channel has unit DC gain, so Ki = 1/lambda for "
                    "a desired closed-loop time constant lambda, and Kp = tau/lambda.\n\n"
                    "Do not ask for lambda faster than the plant's fastest RETAINED mode or "
                    "the command excites dynamics the model truncated. Kp = 0 gives "
                    "feedforward plus integral, which is a reasonable starting point; raise "
                    "it if the approach is too sluggish."
                ),
                target_key="mimo_pi_kp",
            ),
            TutorialStep(
                title="Export it",
                body=(
                    "Export Controller Constants writes a C header and a JSON twin holding "
                    "G, its regularized inverse, per-sensor Kp/Ki and setpoints, per-heater "
                    "power and slew limits, and every loop-shaping constant, with "
                    "provenance naming the gain matrix they came from.\n\n"
                    "The header states the control law in a comment and says exactly how "
                    "far its inverse is from the bounded allocator that runs here."
                ),
                target_key="export_controller",
            ),
        ),
    ),
    Tutorial(
        key="validation",
        title="Check the solver against known answers",
        summary="Run the Thermal Validation tab and read the error it reports.",
        steps=(
            TutorialStep(
                title="What it actually does",
                body=(
                    "The Thermal Validation tab drives the REAL solver -- the same code "
                    "path a production run uses -- against cases with a known answer: "
                    "one-dimensional conduction, lumped radiation cooling, temperature-"
                    "dependent heating, and a published experimental benchmark.\n\n"
                    "It is not a unit test. It reports the error against the reference so "
                    "you can decide whether a modelling choice is good enough."
                ),
            ),
            TutorialStep(
                title="Export the results",
                body=(
                    "Export Results writes the per-case error series to CSV, which is what "
                    "you want when a change moves a number and you need to say by how much."
                ),
            ),
        ),
    ),
    Tutorial(
        key="headless",
        title="Run overnight without the UI",
        summary="Launch a detached run that survives closing the app, and monitor it.",
        steps=(
            TutorialStep(
                title="Why a separate tab",
                body=(
                    "The Headless Run tab never loads a graph into this window. A "
                    "multi-million-cell model would not fit beside the viewer, and a live "
                    "3D redraw during an overnight run is wasted work.\n\n"
                    "It lists graphs by reading the node count from a file header, launches "
                    "run_simulation.py as a detached process, and then monitors that run by "
                    "polling its own status.json and tailing its events.log."
                ),
            ),
            TutorialStep(
                title="Snapshots and checkpoints",
                body=(
                    "Snapshots are the series you plot afterwards. Checkpoints are what a "
                    "resume restarts from, including the controller's integrator state.\n\n"
                    "Checkpoints are large -- multiple megabytes on a big graph -- so the "
                    "interval is a real trade-off between disk and how much you lose if the "
                    "run dies."
                ),
                target_key="checkpoint_interval_s",
            ),
            TutorialStep(
                title="The run folder",
                body=(
                    "Each run writes simulations/<graph>/<timestamp>/ holding the series, "
                    "plots, status.json, events.log, the exact simulation_parameters.json "
                    "it ran with, and the checkpoints.\n\n"
                    "If a run exits with status.json still saying 'running', look for "
                    "crash.log: that is a native crash, and it is the only evidence there "
                    "will be."
                ),
            ),
        ),
    ),
)


def tutorial_by_key(key: str) -> Tutorial | None:
    for tutorial in TUTORIALS:
        if tutorial.key == key:
            return tutorial
    return None
