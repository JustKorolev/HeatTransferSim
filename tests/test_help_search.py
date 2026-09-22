"""The help search has to answer the questions people actually ask.

A search that returns the right control somewhere in a list of twenty is not
useful: the user still has to read all twenty, which is the problem they opened
the search to avoid. So these tests assert the TOP hit for realistic queries,
against targets copied from the app's real labels and sections.
"""

from __future__ import annotations

import pytest

from graph_visualizer.help_search import (
    TUTORIALS,
    HelpTarget,
    search,
    score_target,
    tutorial_by_key,
)


# Real labels and sections, lifted from simulation_controls_panel.py.
TARGETS = [
    HelpTarget("input_mode", "input mode", "Heat Transfer Simulation", "Run",
               keywords=("zero", "heater inputs", "open loop")),
    HelpTarget("controller", "controller", "Heat Transfer Simulation", "Run"),
    HelpTarget("mimo_pi_kp", "MIMO PI Kp", "Heat Transfer Simulation", "Run",
               tooltip="proportional gain per controlled sensor"),
    HelpTarget("mimo_pi_ki", "MIMO PI Ki", "Heat Transfer Simulation", "Run",
               tooltip="integral gain, Ki = 1/lambda"),
    HelpTarget("mimo_default_heater_max_power_W", "max heater power W (all heaters)",
               "Heat Transfer Simulation", "Controller (limits)",
               keywords=("ceiling", "watt", "limit")),
    HelpTarget("mimo_heater_slew_rate_W_per_s", "hard slew W/s (all heaters)",
               "Heat Transfer Simulation", "Controller (limits)",
               keywords=("rate limit", "ramp")),
    HelpTarget("mimo_undershoot_weight", "undershoot weight",
               "Heat Transfer Simulation", "Controller (limits)"),
    HelpTarget("mimo_lambda_u", "lambda_u heater effort",
               "Heat Transfer Simulation", "MIMO Thermal-Rate QP"),
    HelpTarget("T_env_K", "T_env K", "Heat Transfer Simulation", "Environment",
               keywords=("ambient", "exterior", "room temperature")),
    HelpTarget("interior_environment_temperature_K", "interior environment K",
               "Heat Transfer Simulation", "Environment"),
    HelpTarget("initialize", "Initialize", "Heat Transfer Simulation", "Run"),
    HelpTarget("export_controller", "Export Controller Constants",
               "Heat Transfer Simulation", "Controller Design",
               keywords=("header", "firmware", "deploy", "C header")),
    HelpTarget("modal_build", "Build && Use Modal Controller",
               "Heat Transfer Simulation", "Controller Design"),
    HelpTarget("run_sys_id", "Run Sys ID", "Heat Transfer Simulation", "Sys ID",
               keywords=("gain matrix", "identify", "G")),
    HelpTarget("checkpoint_interval_s", "checkpoint every s", "Headless Run", "Run"),
    HelpTarget("snapshot_interval_s", "snapshot every s", "Headless Run", "Run"),
    HelpTarget("dt_s", "dt s", "Heat Transfer Simulation", "Run",
               keywords=("timestep", "step size")),
    HelpTarget("color_min_K", "color min K", "Heat Transfer Simulation", "Display"),
    HelpTarget("solver_method", "implicit method", "Headless Run", "Solver"),
]


def top(query: str) -> str:
    results = search(query, TARGETS)
    assert results, f"no result for {query!r}"
    return results[0].target.key


# --------------------------------------------------------------------------- #
# The queries that motivated the feature.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query, expected",
    [
        ("heater power", "mimo_default_heater_max_power_W"),
        ("max power", "mimo_default_heater_max_power_W"),
        ("slew", "mimo_heater_slew_rate_W_per_s"),
        ("kp", "mimo_pi_kp"),
        ("ki", "mimo_pi_ki"),
        ("export", "export_controller"),
        ("firmware", "export_controller"),
        ("sys id", "run_sys_id"),
        ("gain matrix", "run_sys_id"),
        ("checkpoint", "checkpoint_interval_s"),
        ("ambient", "T_env_K"),
        ("timestep", "dt_s"),
        ("undershoot", "mimo_undershoot_weight"),
        ("initialize", "initialize"),
        ("input mode", "input_mode"),
    ],
)
def test_the_obvious_query_returns_the_obvious_control(query, expected) -> None:
    assert top(query) == expected


def test_a_prefix_is_enough_because_the_user_is_typing(tmp_path=None) -> None:
    """Search runs on every keystroke; it must be useful before the word is done."""
    assert top("temp") in {"T_env_K", "interior_environment_temperature_K"}
    assert top("checkp") == "checkpoint_interval_s"
    assert top("undersh") == "mimo_undershoot_weight"


def test_every_token_must_match_so_two_words_narrow_rather_than_widen() -> None:
    """'heater slew' must not return every control with 'heater' in it."""
    results = search("heater slew", TARGETS)
    assert [r.target.key for r in results] == ["mimo_heater_slew_rate_W_per_s"]


def test_a_query_that_matches_nothing_returns_nothing() -> None:
    assert search("kubernetes", TARGETS) == []
    assert search("", TARGETS) == []
    assert search("   ", TARGETS) == []


def test_a_label_hit_outranks_a_tooltip_hit() -> None:
    """Tooltips here are paragraphs. One word of a 200-word tooltip means little."""
    label_hit = HelpTarget("a", "lambda", tooltip="")
    tooltip_hit = HelpTarget("b", "something else", tooltip="the lambda floor is set here")
    label_score, _ = score_target("lambda", label_hit)
    tooltip_score, _ = score_target("lambda", tooltip_hit)
    assert label_score > tooltip_score


def test_an_exact_label_beats_a_longer_label_that_merely_contains_it() -> None:
    exact = HelpTarget("a", "controller")
    longer = HelpTarget("b", "controller scheme selection dropdown")
    assert score_target("controller", exact)[0] > score_target("controller", longer)[0]


def test_the_section_is_searchable_so_you_can_browse_by_area() -> None:
    results = search("environment", TARGETS)
    keys = {r.target.key for r in results}
    assert {"T_env_K", "interior_environment_temperature_K"} <= keys


def test_results_are_capped_and_ordered_best_first() -> None:
    results = search("k", TARGETS, limit=3)
    assert len(results) <= 3
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


def test_describe_names_the_path_a_user_would_follow() -> None:
    target = HelpTarget("x", "hard slew W/s (all heaters)",
                        "Heat Transfer Simulation", "Controller (limits)")
    assert target.describe() == (
        "Heat Transfer Simulation > Controller (limits) > hard slew W/s (all heaters)"
    )
    assert HelpTarget("y", "Bare").describe() == "Bare"


# --------------------------------------------------------------------------- #
# Tutorials
# --------------------------------------------------------------------------- #
def test_every_tutorial_step_that_points_at_a_control_points_at_a_real_one() -> None:
    """A 'Show me' button that reveals nothing is worse than no button: it tells
    the user the control does not exist."""
    from graph_visualizer.simulation_controls_panel import (
        MODE_LIVE,
        SimulationControlsPanel,
    )
    import test_simulation_controls_panel as stub

    from graph_visualizer.build_graph_params import help_row_keys

    panel = SimulationControlsPanel(stub._QtStub, mode=MODE_LIVE)
    panel.build(stub.QFormLayout())
    # Two kinds of tab contribute rows: the panel-backed ones, and the Build
    # Graph tab, which answers for itself via help_targets(). Both are fair game
    # for a tutorial step, so both have to be known here.
    known = set(panel._rows) | help_row_keys()

    missing = [
        (t.key, step.target_key)
        for t in TUTORIALS
        for step in t.steps
        if step.target_key and step.target_key not in known
    ]
    assert not missing, f"tutorial steps naming controls that do not exist: {missing}"


def test_tutorials_are_uniquely_keyed_and_non_empty() -> None:
    keys = [t.key for t in TUTORIALS]
    assert len(keys) == len(set(keys))
    for tutorial in TUTORIALS:
        assert tutorial.title and tutorial.summary and tutorial.steps
        for step in tutorial.steps:
            assert step.title and step.body


def test_tutorial_lookup() -> None:
    assert tutorial_by_key("first_run") is not None
    assert tutorial_by_key("nope") is None
