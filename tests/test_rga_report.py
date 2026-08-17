"""The RGA section: the numbers, the verdict wording, and when it may be stated.

The point of the RGA is a verdict -- "can this plant be paired one heater per
sensor?" -- so most of these tests are about when that verdict is allowed to be
stated at all. A number that reads like a pairing answer when no pairing exists
is worse than no number, which is why the non-square cases assert on ABSENCE.

Where the files land is :mod:`plant_report`'s decision; see test_plant_analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from graph_visualizer.modal_reduction import relative_gain_array
from graph_visualizer.rga_report import (
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
    assert summary["rga_diag_negative"] == 0
    assert summary["rga_number"] == pytest.approx(0.0, abs=1e-9)
    assert "not ruled out" in verdict_lines(summary)[-1]


def test_summary_counts_the_negative_diagonal():
    G = np.array([[1.0, 2.0], [1.1, 1.0]])
    summary = rga_summary(G, relative_gain_array(G), [10, 11], [20, 21])
    assert summary["square"] is True
    assert summary["rga_diag_negative"] == 2
    assert summary["rga_diag_min"] < 0.0
    assert "WRONG WAY" in " ".join(verdict_lines(summary))


def test_non_square_withholds_the_diagonal():
    """Unequal counts mean there is no one-heater-per-sensor pairing, so the
    diagonal is not a pairing statement and must not be reported as one."""
    G = np.array([[1.0, 1.0, 0.5], [1.0, 1.1, 0.4]])
    summary = rga_summary(G, relative_gain_array(G), [10, 11], [20, 21, 22])
    assert summary["square"] is False
    assert has_pairing_verdict(summary) is False
    assert summary["rga_diag"] is None
    assert summary["rga_number"] is None
    assert "not reported" in verdict_lines(summary)[1]


def test_square_but_non_finite_diagonal_still_withholds_the_verdict():
    """Square is necessary but not sufficient. A diagonal that came out non-finite
    holds no verdict either, and saying "None of 27 negative" would be worse than
    saying nothing. cond() raises outright on this input, so this also pins that
    the summary survives a degenerate G rather than propagating LinAlgError."""
    summary = rga_summary(
        np.full((2, 2), np.nan), np.full((2, 2), np.nan), [10, 11], [20, 21]
    )
    assert summary["square"] is True
    assert has_pairing_verdict(summary) is False
    assert np.isnan(summary["cond_G"])
    assert "not reported" in verdict_lines(summary)[1]


def test_figure_renders_both_layouts(tmp_path):
    """Square gets the diagonal panel; non-square must not, and must still draw."""
    square = np.array([[1.0, 2.0], [1.1, 1.0]])
    summary = rga_summary(square, relative_gain_array(square), [10, 11], [20, 21])
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
