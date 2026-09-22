"""The help index, built from the REAL control panel.

test_help_search.py checks the ranking against hand-written targets. That proves
the ranking but not that the index finds anything, so these tests build the
actual SimulationControlsPanel against the Qt stub and index THAT. If a row is
renamed, restructured, or added without a key, this is what notices.

VTK cannot get an OpenGL context in this environment, so the real window cannot
be constructed here; the app is stubbed down to the two attributes the index
reads (view_tabs and the tab objects).
"""

from __future__ import annotations

import pytest

from graph_visualizer.help_center import (
    HelpCenter,
    HelpDialog,
    render_tutorial_html,
    _enclosing_scroll_area,
)
from graph_visualizer.help_search import TUTORIALS, tutorial_by_key
from graph_visualizer.simulation_controls_panel import (
    MODE_HEADLESS,
    MODE_LIVE,
    SimulationControlsPanel,
)

import test_simulation_controls_panel as stub


class _Tabs:
    """A tab bar that can be asked WHERE a widget is.

    indexOf matters: the help index used to carry hardcoded tab numbers, and
    adding the Build Graph tab in front of them silently pointed every search
    result one tab to the left. It looks the index up now, so this stub has to be
    able to answer -- which also makes it a fairer model of QTabWidget.
    """

    def __init__(self, entries: list[tuple[str, object]]) -> None:
        self.entries = entries
        self.current = 0

    def count(self) -> int:
        return len(self.entries)

    def tabText(self, index: int) -> str:  # noqa: N802 - Qt name
        return self.entries[index][0]

    def indexOf(self, widget: object) -> int:  # noqa: N802 - Qt name
        for index, (_title, candidate) in enumerate(self.entries):
            if candidate is not None and candidate is widget:
                return index
        return -1

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - Qt name
        self.current = index


class _TabOwner:
    """A tab: its panel, and the widget the tab bar holds for it."""

    def __init__(self, panel) -> None:
        self.panel = panel
        self.widget = object()


class _App:
    """Only what HelpCenter.build_index actually touches."""

    QtCore = stub._QtCore
    QtWidgets = stub._QtWidgets
    window = None

    def __init__(self) -> None:
        live = SimulationControlsPanel(stub._QtStub, mode=MODE_LIVE)
        live.build(stub.QFormLayout())
        live._apply_mode()
        headless = SimulationControlsPanel(stub._QtStub, mode=MODE_HEADLESS)
        headless.build(stub.QFormLayout())
        headless._apply_mode()
        self.simulation_tab = _TabOwner(live)
        self.headless_run_tab = _TabOwner(headless)
        # Deliberately WITHOUT the Build Graph tab, so the live/headless indices
        # stay 2 and 4 and these tests keep testing what they are about.
        self.view_tabs = _Tabs(
            [
                ("3D Octree Graph Editor", None),
                ("2D Network Graph", None),
                ("Heat Transfer Simulation", self.simulation_tab.widget),
                ("Thermal Validation", None),
                ("Headless Run", self.headless_run_tab.widget),
            ]
        )


@pytest.fixture
def center() -> HelpCenter:
    return HelpCenter(_App())


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #
def test_the_index_is_built_from_the_live_panel_not_a_hand_written_list(center) -> None:
    targets = center.build_index()
    keys = {t.key.split(":", 1)[-1] for t in targets}
    panel_keys = set(center.app.simulation_tab.panel._rows)
    # Everything the live panel shows must be findable.
    visible = {k for k in panel_keys if center.app.simulation_tab.panel._rows[k][1].visible}
    assert visible <= keys, sorted(visible - keys)[:10]
    assert len(targets) > 40, "the index is suspiciously small"


def test_every_tab_is_itself_a_destination(center) -> None:
    """'Where is validation?' is answered by a tab, not a control."""
    tabs = [t for t in center.build_index() if t.kind == "tab"]
    assert [t.label for t in tabs] == [title for title, _w in center.app.view_tabs.entries]
    assert all(t.tab_index is not None for t in tabs)


def test_controls_carry_their_section_and_tab(center) -> None:
    # Filter to the live tab: the two simulation panels share row keys, so an
    # unfiltered dict would be whichever tab was indexed last.
    by_key = {
        t.key.split(":", 1)[-1]: t
        for t in center.build_index()
        if t.kind == "control" and t.tab_index == 2
    }
    kp = by_key["mimo_pi_kp"]
    assert kp.section == "Run"
    assert kp.tab == "Heat Transfer Simulation"
    export = by_key["export_controller"]
    assert export.section.startswith("Controller Design")


def test_a_row_hidden_by_the_mode_is_not_offered(center) -> None:
    """Offering a control that is not on screen reads as a bug, not a result."""
    targets = center.build_index()
    live = [t for t in targets if t.tab_index == 2]
    # 'initialize' is live-only; the headless panel hides it.
    live_keys = {t.key.split(":", 1)[-1] for t in live}
    assert "initialize" in live_keys
    headless = [t for t in targets if t.tab_index == 4]
    headless_keys = {t.key.split(":", 1)[-1] for t in headless}
    assert "initialize" not in headless_keys
    # ...and the reverse: the checkpoint cadence is headless-only.
    assert "checkpoint_interval_s" in headless_keys
    assert "checkpoint_interval_s" not in live_keys


def test_searching_the_real_index_finds_the_real_controls(center) -> None:
    """The queries from test_help_search, now against the index the app builds."""
    for query, expected_row in (
        ("kp", "mimo_pi_kp"),
        ("export controller", "export_controller"),
        ("checkpoint", "checkpoint_interval_s"),
        ("undershoot", "mimo_undershoot_weight"),
        ("sys id", "run_sys_id"),
    ):
        results = center.find(query)
        assert results, f"no result for {query!r}"
        assert results[0].target.key.split(":", 1)[-1] == expected_row, (
            query,
            [r.target.describe() for r in results[:3]],
        )


def test_the_removed_role_contact_controls_are_not_findable(center) -> None:
    """They were deleted for doing nothing; search must not resurrect them."""
    assert center.find("role contact") == []


# --------------------------------------------------------------------------- #
# Revealing
# --------------------------------------------------------------------------- #
def test_revealing_a_control_switches_to_its_tab(center) -> None:
    target = next(
        t for t in center.build_index()
        if t.kind == "control" and t.key.split(":", 1)[-1] == "checkpoint_interval_s"
    )
    center.app.view_tabs.current = 0
    assert center.reveal(target)
    assert center.app.view_tabs.current == 4


def test_revealing_flashes_the_widget_and_restores_its_style(center) -> None:
    """The flash must be temporary. A control left permanently outlined looks like
    an error state, and the next search would have nothing left to highlight."""
    target = next(
        t for t in center.build_index()
        if t.kind == "control" and t.key.split(":", 1)[-1] == "mimo_pi_kp"
    )
    widget = target.widget
    original = widget.styleSheet()
    center.reveal(target)
    assert widget.styleSheet() != original, "the widget was never highlighted"
    center._restore_flashed_widget()
    assert widget.styleSheet() == original


def test_flashing_a_second_widget_restores_the_first(center) -> None:
    index = center.build_index()
    first = next(t for t in index if t.kind == "control" and t.widget is not None)
    second = next(
        t for t in index
        if t.kind == "control" and t.widget is not None and t.widget is not first.widget
    )
    before = first.widget.styleSheet()
    center.flash(first.widget)
    center.flash(second.widget)
    assert first.widget.styleSheet() == before, "the previous highlight was left behind"


def test_go_to_row_key_is_what_a_tutorial_uses(center) -> None:
    assert center.go_to_row_key("mimo_pi_kp")
    assert not center.go_to_row_key("no_such_row")


def test_reveal_survives_a_widget_with_no_stylesheet(center) -> None:
    """A stub or an exotic widget must not take the help system down."""
    class Bare:
        visible = True

    from graph_visualizer.help_search import HelpTarget

    center.flash(Bare())  # must not raise
    assert center.reveal(HelpTarget("x", "x", widget=None, kind="tab")) is True


# --------------------------------------------------------------------------- #
# Tutorial rendering
# --------------------------------------------------------------------------- #
def test_a_step_with_a_target_renders_a_show_me_link() -> None:
    html = render_tutorial_html(tutorial_by_key("first_run"))
    assert "show:input_mode" in html
    assert "Show me this control" in html


def test_the_link_scheme_matches_what_the_dialog_parses() -> None:
    """The 'show:' prefix is the contract between the tutorial data and the anchor
    handler. A typo in it fails silently on screen, so it is pinned here."""
    import inspect

    source = inspect.getsource(HelpDialog._handle_anchor)
    assert 'startswith("show:")' in source
    for tutorial in TUTORIALS:
        for step in tutorial.steps:
            if step.target_key:
                assert f"show:{step.target_key}" in render_tutorial_html(tutorial)


def test_tutorial_html_escapes_its_content() -> None:
    """Build && Use Modal Controller is a real label; '&' must survive as text."""
    from graph_visualizer.help_search import Tutorial, TutorialStep

    nasty = Tutorial(
        "k", "A & B", "x <b>y</b>",
        (TutorialStep(title="t", body="1 < 2 & 3 > 2"),),
    )
    html = render_tutorial_html(nasty)
    assert "A &amp; B" in html
    assert "1 &lt; 2 &amp; 3 &gt; 2" in html
    assert "<b>y</b>" not in html


def test_enclosing_scroll_area_returns_none_rather_than_looping(center) -> None:
    class Cycle:
        def parentWidget(self):  # noqa: N802 - Qt name
            return self

    assert _enclosing_scroll_area(Cycle(), stub._QtWidgets) is None


def test_results_do_not_repeat_the_same_control_once_per_tab(center) -> None:
    """The two simulation tabs share one panel, so nearly every control exists
    twice. Uncollapsed, 'heater power' returned twelve matches of which six were
    the same four controls listed again under the other tab."""
    results = center.find("heater power")
    seen = [(r.target.section, r.target.label) for r in results]
    assert len(seen) == len(set(seen)), f"duplicate rows: {seen}"
    # The surviving copy is the live tab's, which is first in build order.
    top = results[0].target
    assert top.label == "max heater power W (all heaters)"
    assert top.tab == "Heat Transfer Simulation"


def test_a_control_only_one_tab_has_is_still_offered(center) -> None:
    """Collapsing must not drop a control that genuinely exists on one tab only."""
    results = center.find("checkpoint")
    assert results
    assert results[0].target.tab == "Headless Run"
