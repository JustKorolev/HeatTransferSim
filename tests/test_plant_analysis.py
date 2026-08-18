"""The full plant analysis: the statistics, the written report, and the button.

The statistics are asserted against matrices whose answers are known by hand
rather than against a recorded output, because the point of this report is that
someone will act on the numbers. A test that only pins "it produced something"
would let a sign error through unnoticed.
"""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

for _name in ("PySide6", "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

from graph_visualizer.plant_analysis import compute_plant_analysis  # noqa: E402
from graph_visualizer.plant_report import (  # noqa: E402
    report_markdown,
    write_plant_analysis,
)
from graph_visualizer.sys_id_artifacts import save_sys_id_gain_matrix  # noqa: E402


def _save(folder: Path, G: np.ndarray, *, passive_K: float | None = None, name: str = "G_test") -> Path:
    metadata = {"T_op_K": 50.0, "method": "test", "dc_ground": "cryocooler"}
    if passive_K is not None:
        metadata["passive_reference_K"] = float(passive_K)
    sensors = [471300 + i for i in range(G.shape[0])]
    heaters = [471400 + j for j in range(G.shape[1])]
    return save_sys_id_gain_matrix(folder, name, sensors, heaters, G, metadata=metadata)


def _analyse(G: np.ndarray, **kwargs):
    sensors = [471300 + i for i in range(G.shape[0])]
    heaters = [471400 + j for j in range(G.shape[1])]
    return compute_plant_analysis(G, sensors, heaters, **kwargs)


# ---------------------------------------------------------------- statistics
def test_spectrum_is_the_singular_values_and_their_energy():
    stats = _analyse(np.diag([3.0, 2.0, 1.0]))
    spectrum = stats["spectrum"]
    assert spectrum["singular_values"] == pytest.approx([3.0, 2.0, 1.0])
    assert spectrum["condition_number"] == pytest.approx(3.0)
    # Energy is sigma^2 normalised: 9/14, 4/14, 1/14.
    assert spectrum["energy_fraction"] == pytest.approx([9 / 14, 4 / 14, 1 / 14])
    assert spectrum["cumulative_energy_fraction"][-1] == pytest.approx(1.0)
    assert spectrum["directions_for_90pct"] == 2       # 9/14 + 4/14 = 92.9%


def test_effective_rank_counts_directions_above_a_threshold():
    stats = _analyse(np.diag([100.0, 5.0, 0.5]))
    ranks = stats["spectrum"]["effective_rank"]
    assert ranks["tol_0.1"] == 1      # only 100 clears 10 % of sigma_1
    assert ranks["tol_0.01"] == 2     # 5 clears 1 %, 0.5 does not


def test_a_rank_one_plant_puts_everything_in_one_direction():
    """One shared bottleneck: every heater warms every sensor in proportion. The
    spectrum should say so with one number rather than by inspection."""
    G = np.outer(np.arange(1.0, 5.0), np.arange(1.0, 4.0))
    stats = _analyse(G)
    assert stats["spectrum"]["top_energy_fraction"] == pytest.approx(1.0)
    assert stats["modes"]["modes"][0]["sensor_sign_agreement"] == pytest.approx(1.0)
    assert stats["modes"]["modes"][0]["heater_sign_agreement"] == pytest.approx(1.0)


def test_mode_sign_is_normalised_so_agreement_is_stable():
    """Singular vector signs are arbitrary; without pinning them, "does the whole
    structure move together?" would flip answer between runs of the same input."""
    stats = _analyse(-np.outer(np.ones(4), np.ones(3)))
    mode = stats["modes"]["modes"][0]
    assert sum(mode["sensor_pattern"]) > 0
    assert mode["sensor_sign_agreement"] == pytest.approx(1.0)


def test_duplicate_heaters_are_reported_as_one_actuator():
    """Consolidating heaters into one node makes their columns collinear. That is
    a rank statement about the actuator set, and no tuning undoes it."""
    G = np.array([[1.0, 2.0, 1.0], [0.5, 1.0, 3.0]])   # column 1 is 2 x column 0
    actuators = _analyse(G)["actuators"]
    assert actuators["max_cosine_with_another_heater"][0] == pytest.approx(1.0)
    assert actuators["max_cosine_with_another_heater"][1] == pytest.approx(1.0)
    assert actuators["most_similar_heater_id"][0] == 471401
    assert actuators["n_near_duplicate_pairs"] == 2
    assert actuators["max_cosine_with_another_heater"][2] < 0.99


def test_independent_control_cost_is_the_power_to_move_one_sensor_alone():
    """||G+ e_i||. On a diagonal plant it is just 1/gain, which is the sanity
    check that the column-of-the-pseudo-inverse shortcut is the right object."""
    reach = _analyse(np.diag([2.0, 5.0]))["reachability"]
    assert reach["independent_control_cost_W_per_K"] == pytest.approx([0.5, 0.2])
    assert reach["worst_channels"][0]["sensor_id"] == 471300


def test_uniform_lift_is_exact_when_the_plant_can_deliver_it():
    lift = _analyse(np.eye(3))["uniform_lift"]
    assert lift["unconstrained"]["power_W_per_K"] == pytest.approx(3.0)
    assert lift["unconstrained"]["residual_rms_K_per_K"] == pytest.approx(0.0, abs=1e-12)
    assert lift["nonnegative"]["residual_rms_K_per_K"] == pytest.approx(0.0, abs=1e-9)
    assert lift["nonnegativity_penalty_rms_K_per_K"] == pytest.approx(0.0, abs=1e-9)


def test_one_sided_actuation_costs_something_when_the_fit_wants_cooling():
    """The unconstrained fit here asks a heater for negative power. Heaters cannot
    cool, so the bounded answer is strictly worse -- and the gap between them is
    exactly what the bounded allocator gives up."""
    # Exactly solvable, but only with u = [1.25, -0.5]. Held to u >= 0 the best
    # available is u = [0.6, 0], which leaves [+0.4, -0.2] K per K requested.
    G = np.array([[1.0, 0.5], [2.0, 3.0]])
    lift = _analyse(G)["uniform_lift"]
    assert lift["unconstrained"]["power_W_per_K"] == pytest.approx(0.75)
    assert lift["unconstrained"]["negative_power_heaters"] == 1
    assert lift["nonnegative"]["power_per_heater_W_per_K"] == pytest.approx([0.6, 0.0])
    assert lift["nonnegative"]["residual_per_sensor"] == pytest.approx([0.4, -0.2])
    assert (
        lift["nonnegative"]["residual_rms_K_per_K"]
        > lift["unconstrained"]["residual_rms_K_per_K"]
    )
    assert min(lift["nonnegative"]["power_per_heater_W_per_K"]) >= -1e-12
    assert lift["nonnegativity_penalty_rms_K_per_K"] > 0.0


# ------------------------------------------------------------ operating point
def test_operating_point_is_skipped_without_a_reference():
    """The one section that needs more than G says so rather than inventing a
    passive equilibrium to measure a deviation against."""
    operating = _analyse(np.eye(2), setpoints_K={471300: 50.0, 471301: 50.0})["operating_point"]
    assert operating["available"] is False
    assert any("passive reference" in reason for reason in operating["missing"])


def test_operating_point_allocates_against_the_real_caps():
    G = np.eye(2) * 2.0
    operating = _analyse(
        G,
        setpoints_K={471300: 50.0, 471301: 50.0},
        passive_reference_K=45.0,
        heater_max_power_W=30.0,
    )["operating_point"]
    assert operating["available"] is True
    # 5 K wanted at 2 K/W is 2.5 W per channel, comfortably under the cap.
    assert operating["power_per_heater_W"] == pytest.approx([2.5, 2.5])
    assert operating["error_rms_K"] == pytest.approx(0.0, abs=1e-9)
    assert operating["saturated_heaters"] == 0


def test_operating_point_reports_saturation_and_the_error_it_leaves():
    operating = _analyse(
        np.eye(2) * 2.0,
        setpoints_K={471300: 50.0, 471301: 50.0},
        passive_reference_K=45.0,
        heater_max_power_W=1.0,          # 2 K of the 5 K wanted
    )["operating_point"]
    assert operating["saturated_heaters"] == 2
    assert sorted(operating["saturated_heater_ids"]) == [471400, 471401]
    assert operating["error_per_sensor_K"] == pytest.approx([-3.0, -3.0])
    assert operating["error_rms_K"] == pytest.approx(3.0)


def test_per_heater_caps_beat_the_global_one():
    operating = _analyse(
        np.eye(2) * 2.0,
        setpoints_K={471300: 50.0, 471301: 50.0},
        passive_reference_K=45.0,
        heater_max_power_W={471400: 1.0, 471401: 30.0},
    )["operating_point"]
    assert operating["saturated_heater_ids"] == [471400]
    assert operating["error_per_sensor_K"] == pytest.approx([-3.0, 0.0], abs=1e-9)


def test_missing_setpoints_for_some_sensors_is_refused_not_guessed():
    operating = _analyse(
        np.eye(2), setpoints_K={471300: 50.0}, passive_reference_K=45.0
    )["operating_point"]
    assert operating["available"] is False
    assert "setpoints for 1 of 2 sensors" in operating["missing"][0]


# ------------------------------------------------------------------- the files
def test_writes_the_whole_report_under_analysis(tmp_path):
    rng = np.random.default_rng(0)
    G = 0.9 * np.outer(rng.uniform(1, 3, 6), rng.uniform(1, 3, 6)) + 0.1 * rng.uniform(1, 8, (6, 6))
    gain_folder = _save(tmp_path / "graphs" / "demo", G, passive_K=45.0)

    analysis = write_plant_analysis(
        gain_folder,
        setpoints_K={471300 + i: 50.0 for i in range(6)},
        heater_max_power_W=30.0,
    )

    # Under the graph folder, in its own directory, beside the G it describes.
    assert analysis.out_dir == gain_folder / "analysis"
    assert analysis.report_path.is_file() and analysis.json_path.is_file()
    assert analysis.stats["skipped_figures"] == []
    names = {p.name for p in analysis.figures}
    assert names == {
        "gain_matrix.png", "spectrum.png", "mode_shapes.png", "rga.png",
        "pairing.png", "actuators.png", "reachability.png", "operating_point.png",
    }
    assert all(p.stat().st_size > 0 for p in analysis.figures)
    assert {p.name for p in analysis.tables} == {
        "channels.csv", "heaters.csv", "spectrum.csv", "rga.csv"
    }

    saved = json.loads(analysis.json_path.read_text(encoding="utf-8"))
    assert saved["n_sensors"] == 6 and saved["n_heaters"] == 6
    # The full RGA lives in its CSV; keeping it out leaves the JSON openable.
    assert "RGA" not in saved["pairing"]
    assert saved["operating_point"]["available"] is True


def test_channel_table_has_one_row_per_sensor(tmp_path):
    gain_folder = _save(tmp_path / "graphs" / "demo", np.eye(3) + 0.4, passive_K=45.0)
    analysis = write_plant_analysis(gain_folder)
    channels = next(p for p in analysis.tables if p.name == "channels.csv")
    with channels.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["sensor_id"]) for row in rows] == [471300, 471301, 471302]
    # No operating point was possible, so that column is blank rather than absent
    # -- a reader joining on sensor_id should not have to guess the schema.
    assert all(row["operating_point_error_K"] == "" for row in rows)
    assert all(row["independent_control_cost_W_per_K"] for row in rows)


def test_markdown_tables_are_not_broken_by_their_own_labels(tmp_path):
    """A label containing a pipe silently splits a markdown row into extra cells,
    which is invisible in the source and obvious in a renderer."""
    gain_folder = _save(tmp_path / "graphs" / "demo", np.eye(3) + 0.4, passive_K=45.0)
    analysis = write_plant_analysis(gain_folder)
    text = analysis.report_path.read_text(encoding="utf-8")

    widths: list[int] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            widths = []
            continue
        widths.append(line.count("|"))
        assert len(set(widths)) == 1, f"ragged table row: {line}"


def test_report_states_the_headline_numbers(tmp_path):
    G = np.array([[1.0, 2.0], [1.1, 1.0]])
    stats = compute_plant_analysis(G, [10, 11], [20, 21], metadata={"run_name": "demo"})
    text = report_markdown(stats, [], [])
    assert "# Plant analysis — demo" in text
    assert "not** the matrix diagonal" in text
    assert "Niederlinski index" in text
    assert "Skipped" in text          # no operating point was supplied


def test_niederlinski_is_evaluated_at_the_chosen_pairing():
    """det / prod(paired gains) rules out a stable decentralised controller with
    integral action when negative. It is defined FOR A PAIRING, so G's columns are
    permuted into the chosen one first -- on the raw matrix it would answer for
    the sort-order pairing instead, which is a different question.

    Here the sort-order diagonal would give -1.2 ("impossible"); at the pairing
    actually chosen it is +0.545, which excludes nothing.
    """
    G = np.array([[1.0, 2.0], [1.1, 1.0]])
    pairing = compute_plant_analysis(G, [10, 11], [20, 21])["pairing"]
    assert pairing["niederlinski_index"] == pytest.approx(0.5455, abs=1e-3)


def test_paired_influence_uses_the_chosen_partner(tmp_path):
    """The magnitude companion has to use the same pairing as the RGA, or the two
    halves of the pairing figure describe different schemes. Sensor 10's partner
    is heater 21 at 2.0 K/W of a 3.0 row sum, not heater 20 at 1.0."""
    G = np.array([[1.0, 2.0], [1.1, 1.0]])
    result = compute_plant_analysis(G, [10, 11], [20, 21])["pairing"]
    assert result["paired_heater_ids"] == [21, 20]
    assert result["paired_influence_fraction"][0] == pytest.approx(2.0 / 3.0)


def test_enabled_sensor_filter_narrows_the_whole_analysis(tmp_path):
    gain_folder = _save(tmp_path / "graphs" / "demo", np.eye(4) + 0.3, passive_K=45.0)
    analysis = write_plant_analysis(gain_folder, enabled_sensor_ids=[471300, 471301])
    assert analysis.stats["n_sensors"] == 2
    assert analysis.stats["excluded_sensor_ids"] == [471302, 471303]
    # Two sensors against four heaters is still pairable -- each sensor gets its
    # own and the spare heaters go unused -- so the verdict survives the filter.
    summary = analysis.stats["pairing"]["rga_summary"]
    assert summary["n_paired"] == 2
    assert len({p["heater_id"] for p in summary["pairing"]}) == 2


def test_empty_selection_is_refused_rather_than_analysed(tmp_path):
    gain_folder = _save(tmp_path / "graphs" / "demo", np.eye(2))
    with pytest.raises(ValueError, match="no loop left"):
        write_plant_analysis(gain_folder, enabled_sensor_ids=[])


def test_out_dir_overrides_the_default_location(tmp_path):
    gain_folder = _save(tmp_path / "graphs" / "demo", np.eye(2))
    elsewhere = tmp_path / "elsewhere"
    analysis = write_plant_analysis(gain_folder, out_dir=elsewhere)
    assert analysis.out_dir == elsewhere
    assert not (gain_folder / "analysis").exists()


# ----------------------------------------------------------------- the tab hook
def _tab(selected: tuple[str, str], *, default_cap: float = 30.0, heater_rows=None):
    """A HeadlessRunTab with only what analyse_plant touches, so the button can be
    exercised without Qt or a graph."""
    from graph_visualizer.headless_run_tab import HeadlessRunTab

    tab = HeadlessRunTab.__new__(HeadlessRunTab)
    tab.panel = types.SimpleNamespace(selected_controller=lambda: selected)
    tab.inputs = {"mimo_default_heater_max_power_W": types.SimpleNamespace(value=lambda: default_cap)}
    tab._heater_rows_manifest = heater_rows or []
    tab.messages: list[tuple[str, bool]] = []
    tab.on_status = lambda message, is_error: tab.messages.append((message, is_error))
    return tab


def test_tab_button_writes_the_analysis(tmp_path):
    gain_folder = _save(tmp_path / "graphs" / "demo", np.eye(3) + 0.4, passive_K=45.0)
    tab = _tab(("mimo_pi", str(gain_folder)))

    tab.analyse_plant()

    assert (gain_folder / "analysis" / "plant_analysis.md").is_file()
    message, is_error = tab.messages[-1]
    assert is_error is False
    assert "3x3 G" in message and "sigma_1 carries" in message


def test_tab_heater_caps_default_to_the_controller_section(tmp_path):
    tab = _tab(("mimo_pi", ""), default_cap=12.5)
    assert tab._analysis_heater_caps() == pytest.approx(12.5)


def test_tab_heater_caps_apply_per_heater_overrides(tmp_path):
    """Same rule the run uses: a heater with its own limit takes it, everything
    else falls back to the one number on the left. Analysing against a different
    budget than the run would use is how a report ends up disagreeing with the
    loop it describes."""
    tab = _tab(
        ("mimo_pi", ""),
        default_cap=30.0,
        heater_rows=[{"node_id": "471400"}, {"node_id": "471401"}],
    )
    tab.heater_table = None            # collect_heater_overrides reads the table
    assert tab._analysis_heater_caps() == pytest.approx(30.0)

    tab.collect_heater_overrides = lambda: {471401: {"heater_max_power_W": 5.0}}
    assert tab._analysis_heater_caps() == {471400: 30.0, 471401: 5.0}


def test_tab_button_needs_a_mimo_pi_selection(tmp_path):
    tab = _tab(("modal_lqr", str(tmp_path / "some.npz")))
    tab.analyse_plant()
    message, is_error = tab.messages[-1]
    assert is_error is True
    assert "MIMO PI" in message


def test_tab_button_reports_a_missing_matrix(tmp_path):
    tab = _tab(("mimo_pi", str(tmp_path / "gone")))
    tab.analyse_plant()
    message, is_error = tab.messages[-1]
    assert is_error is True
    assert "gone" in message
