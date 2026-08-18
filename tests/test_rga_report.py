"""The RGA section: the numbers, the verdict wording, and when it may be stated.

The point of the RGA is a verdict -- "can this plant be paired one heater per
sensor?" -- and the verdict is only meaningful once you say WHICH pairing. The
matrix diagonal is not it: the two id lists are sorted independently, so G[i, i]
pairs partners by sort order. Most of these tests are about choosing the pairing
rather than assuming it, and about when the verdict may be stated at all.

Where the files land is :mod:`plant_report`'s decision; see test_plant_analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from graph_visualizer.modal_reduction import relative_gain_array
from graph_visualizer.rga_report import (
    diagonal_axis_scale,
    select_pairing,
    has_pairing_verdict,
    render_rga_figure,
    rga_summary,
    verdict_lines,
    write_rga_csv,
)


def test_rga_matches_the_textbook_two_by_two():
    """The classic near-singular pair: every entry of G is strong and positive,
    yet the pairing is hopeless. That gap is the whole reason the RGA is computed
    rather than read off G."""
    G = np.array([[1.0, 1.0], [1.0, 1.1]])
    RGA = relative_gain_array(G)
    assert np.allclose(RGA, [[11.0, -10.0], [-10.0, 11.0]])
    # Defining property: rows and columns each sum to one.
    assert np.allclose(RGA.sum(axis=0), 1.0)
    assert np.allclose(RGA.sum(axis=1), 1.0)


def test_identity_plant_pairs_perfectly():
    G = np.diag([2.0, 5.0, 0.5])
    summary = rga_summary(G, relative_gain_array(G), [1, 2, 3], [4, 5, 6])
    assert summary["rga_paired_negative"] == 0
    assert summary["rga_paired_median"] == pytest.approx(1.0)
    assert summary["pairing_is_index_diagonal"] is True
    assert summary["rga_number"] == pytest.approx(0.0, abs=1e-9)
    assert "viable" in verdict_lines(summary)[-1]


def test_the_pairing_is_chosen_not_assumed_to_be_the_diagonal():
    """The bug this whole mechanism exists for, in miniature.

    sensor and heater ids are sorted independently, so G[i, i] pairs partners by
    sort order. Here that diagonal is NEGATIVE on both channels -- "per-pair
    control is impossible" -- while the anti-diagonal pairing is +1.83 on both,
    i.e. entirely workable. Reading the diagonal gives the opposite conclusion to
    the correct one, which is exactly what happened on the 27x27 cryostat.
    """
    G = np.array([[1.0, 2.0], [1.1, 1.0]])
    RGA = relative_gain_array(G)
    assert np.diag(RGA) == pytest.approx([-0.8333, -0.8333], abs=1e-3)

    summary = rga_summary(G, RGA, [10, 11], [20, 21])
    assert summary["pairing_is_index_diagonal"] is False
    assert [(p["sensor_id"], p["heater_id"]) for p in summary["pairing"]] == [(10, 21), (11, 20)]
    assert summary["rga_paired_negative"] == 0
    assert summary["rga_paired_median"] == pytest.approx(1.8333, abs=1e-3)
    assert "viable" in " ".join(verdict_lines(summary))
    assert "NOT the matrix diagonal" in " ".join(verdict_lines(summary))


def test_more_heaters_than_sensors_can_still_be_paired():
    """Unequal counts are not a reason to withhold a verdict: with heaters to
    spare every sensor still gets its own, and only the leftovers go unused."""
    G = np.array([[1.0, 1.0, 0.5], [1.0, 1.1, 0.4]])
    summary = rga_summary(G, relative_gain_array(G), [10, 11], [20, 21, 22])
    assert has_pairing_verdict(summary) is True
    assert summary["n_paired"] == 2
    assert len({p["heater_id"] for p in summary["pairing"]}) == 2      # no heater reused


def test_fewer_heaters_than_sensors_leaves_some_unpaired():
    G = np.array([[1.0], [0.4], [0.2]])
    summary = rga_summary(G, relative_gain_array(G), [10, 11, 12], [20])
    assert summary["n_paired"] == 1
    assert summary["n_sensors"] == 3


def test_non_finite_rga_still_withholds_the_verdict():
    """A relative gain that came out non-finite holds no verdict, and saying
    "0 negative" would be worse than saying nothing. cond() raises outright on
    this input, so this also pins that the summary survives a degenerate G rather
    than propagating LinAlgError."""
    summary = rga_summary(
        np.full((2, 2), np.nan), np.full((2, 2), np.nan), [10, 11], [20, 21]
    )
    assert has_pairing_verdict(summary) is False
    assert np.isnan(summary["cond_G"])
    assert "No pairing could be scored" in verdict_lines(summary)[1]


def test_diagonal_axis_stays_linear_when_the_whole_diagonal_is_small():
    """symlog spends the axis on empty decades when every value sits inside +/-1,
    collapsing every bar into the linear region so the figure reads "nothing
    here". These are the sort-order diagonal values off the real 27x27, which is
    where the effect was first seen."""
    scale, kwargs = diagonal_axis_scale([-0.6686, -0.0004, 0.0011, -0.117])
    assert scale == "linear" and kwargs == {}


def test_diagonal_axis_goes_symlog_when_it_genuinely_spans_decades():
    scale, kwargs = diagonal_axis_scale([11.0, -10.0, 0.05])
    assert scale == "symlog" and kwargs == {"linthresh": 1.0}


def test_diagonal_axis_ignores_non_finite_entries():
    scale, _ = diagonal_axis_scale([np.nan, 0.5, np.inf])
    assert scale == "linear"


def test_figure_renders_both_layouts(tmp_path):
    """Both shapes are pairable here, so both get the bar panel and must draw."""
    square = np.array([[1.0, 2.0], [1.1, 1.0]])
    summary = rga_summary(square, relative_gain_array(square), [10, 11], [20, 21])  # noqa: E501
    path = render_rga_figure(
        tmp_path / "square.png", relative_gain_array(square), [10, 11], [20, 21], summary
    )
    assert path.is_file() and path.stat().st_size > 0

    wide = np.array([[1.0, 1.0, 0.5], [1.0, 1.1, 0.4]])
    wide_summary = rga_summary(wide, relative_gain_array(wide), [10, 11], [20, 21, 22])
    wide_path = render_rga_figure(
        tmp_path / "wide.png", relative_gain_array(wide), [10, 11], [20, 21, 22], wide_summary
    )
    assert wide_path.is_file() and wide_path.stat().st_size > 0


def test_rga_csv_is_long_form(tmp_path):
    import csv

    G = np.array([[1.0, 1.0], [1.0, 1.1]])
    path = write_rga_csv(tmp_path / "rga.csv", [10, 11], [20, 21], relative_gain_array(G))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {"sensor_id", "heater_id", "rga"} == set(rows[0])
    assert float(rows[0]["rga"]) == pytest.approx(11.0)
