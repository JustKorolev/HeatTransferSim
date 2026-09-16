"""Spin-box precision, font choice and UI scale.

The dangerous one here is decimals. A QDoubleSpinBox ROUNDS whatever it is given
to its own decimals, and this application reads its parameters straight back out
of the widgets and autosaves them -- so a field that is too narrow does not just
display a number badly, it destroys it on load and writes the damage to disk.
Steps in this UI go down to 1e-14.
"""

from __future__ import annotations

import pytest

from graph_visualizer.ui_theme import (
    MAX_DECIMALS,
    MAX_UI_SCALE,
    MIN_DECIMALS,
    MIN_UI_SCALE,
    UI_SCALE_STEP,
    choose_font_family,
    clamp_scale,
    decimals_for,
    point_size_for,
    step_scale,
    widen_decimals_for,
)


# --------------------------------------------------------------------------- #
# Decimals
# --------------------------------------------------------------------------- #
def test_ordinary_fields_get_three_decimals() -> None:
    """The whole point of the change: 293.15000000 becomes 293.150."""
    assert decimals_for(step=1.0, value=293.15) == 3
    assert decimals_for(step=1.0, value=1.0) == 3
    assert decimals_for(step=0.1, value=30.0) == 3
    assert decimals_for(step=10.0, value=3600.0) == 3


@pytest.mark.parametrize(
    "value, expected",
    [
        (1.0e-4, 4),
        (1.0e-6, 6),
        (1.5e-3, 4),
        (2.5e-5, 6),  # 0.000025 -- six places, not five
        (1.0e-8, 8),
    ],
)
def test_a_small_value_widens_the_field_rather_than_being_rounded_away(value, expected) -> None:
    """Three decimals would turn every one of these into 0.000, and the next
    autosave would write that zero back over the real gain."""
    assert decimals_for(step=1.0, value=value) == expected


def test_a_small_step_widens_the_field_so_the_arrows_do_something() -> None:
    """A 1e-5 step in a 3-decimal box changes nothing on screen, which reads as a
    broken control."""
    assert decimals_for(step=1.0e-5, value=0.0) == 5
    assert decimals_for(step=1.0e-9, value=0.0) == MAX_DECIMALS


def test_nothing_exceeds_the_eight_decimals_the_fields_used_to_have() -> None:
    """No field may end up narrower OR wider than before this change allowed."""
    assert decimals_for(step=1.0e-14, value=1.0e-14) == MAX_DECIMALS
    assert decimals_for(step=1.0e-30, value=1.0e-30) == MAX_DECIMALS


def test_the_floor_holds_for_round_numbers() -> None:
    assert decimals_for(step=1.0, value=0.0) == MIN_DECIMALS
    assert decimals_for(step=100.0, value=1000000.0) == MIN_DECIMALS


def test_junk_does_not_raise() -> None:
    assert decimals_for(step=float("nan"), value=float("inf")) == MIN_DECIMALS
    assert decimals_for(step=None, value=None) == MIN_DECIMALS  # type: ignore[arg-type]


def test_binary_representation_does_not_inflate_the_count() -> None:
    """293.15 is not exactly representable; a naive equality check would decide it
    needs all eight decimals and undo the entire change."""
    assert decimals_for(step=1.0, value=293.15) == 3
    assert decimals_for(step=1.0, value=0.1 + 0.2) == 3


# --------------------------------------------------------------------------- #
# The runtime guard
# --------------------------------------------------------------------------- #
class _Spin:
    def __init__(self, decimals: int = 3) -> None:
        self._decimals = decimals

    def decimals(self) -> int:
        return self._decimals

    def setDecimals(self, value: int) -> None:  # noqa: N802 - Qt name
        self._decimals = int(value)


def test_a_loaded_value_widens_the_box_before_it_can_be_rounded() -> None:
    spin = _Spin(3)
    widen_decimals_for(spin, 1.0e-6)
    assert spin.decimals() == 6


def test_widening_never_shrinks() -> None:
    """A field that grew to hold 1e-6 keeps the room, so typing over the value and
    undoing does not quietly lose it."""
    spin = _Spin(6)
    widen_decimals_for(spin, 1.0)
    assert spin.decimals() == 6


def test_widening_is_capped() -> None:
    spin = _Spin(3)
    widen_decimals_for(spin, 1.0e-20)
    assert spin.decimals() == MAX_DECIMALS


def test_widening_survives_a_widget_that_does_not_cooperate() -> None:
    class Bare:
        pass

    widen_decimals_for(Bare(), 1.0e-6)  # must not raise


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #
def test_the_best_available_font_wins() -> None:
    assert choose_font_family(["Arial", "Segoe UI", "Inter"]) == "Segoe UI"
    assert choose_font_family(["Arial", "Inter"]) == "Inter"


def test_font_matching_is_case_insensitive_but_returns_the_system_spelling() -> None:
    """fontconfig reports lowercase; losing the font over a capital letter would be
    a silly failure."""
    assert choose_font_family(["segoe ui"]) == "segoe ui"
    assert choose_font_family(["  Segoe UI  "]) == "  Segoe UI  "


def test_no_preferred_font_falls_back_to_qts_default() -> None:
    assert choose_font_family(["Comic Sans MS", "Webdings"]) == ""
    assert choose_font_family([]) == ""


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #
def test_the_slider_range_round_trips() -> None:
    for scale in range(MIN_UI_SCALE, MAX_UI_SCALE + 1, UI_SCALE_STEP):
        assert clamp_scale(scale) == scale


def test_the_range_reaches_forty_percent() -> None:
    """The bottom exists for displays whose DPI Qt over-estimates: everything is
    already enlarged before the application gets a say, so the useful correction
    is downward."""
    assert MIN_UI_SCALE == 40
    assert clamp_scale(40) == 40
    assert clamp_scale(10) == 40
    assert point_size_for(40) == pytest.approx(3.5)


def test_values_off_the_step_snap_onto_it() -> None:
    """A slider drag lands anywhere; the stored value should not be arbitrary."""
    assert clamp_scale(63) == 65
    assert clamp_scale(102) == 100
    assert clamp_scale(103) == 105


def test_out_of_range_is_clamped_not_wrapped() -> None:
    assert clamp_scale(1000) == MAX_UI_SCALE
    assert clamp_scale(-50) == MIN_UI_SCALE


def test_junk_scale_is_the_default() -> None:
    assert clamp_scale("wat") == 100
    assert clamp_scale(None) == 100
    assert clamp_scale(float("nan")) == 100


def test_keyboard_steps_move_and_stop_at_the_ends() -> None:
    assert step_scale(100, +1) == 110
    assert step_scale(100, -1) == 90
    assert step_scale(MIN_UI_SCALE, -1) == MIN_UI_SCALE
    assert step_scale(MAX_UI_SCALE, +1) == MAX_UI_SCALE


def test_point_size_tracks_the_scale_and_lands_on_half_points() -> None:
    assert point_size_for(100) == pytest.approx(9.0)
    assert point_size_for(200) == pytest.approx(18.0)
    for scale in range(MIN_UI_SCALE, MAX_UI_SCALE + 1, UI_SCALE_STEP):
        size = point_size_for(scale)
        assert (size * 2) == int(size * 2), f"{scale}% gave {size}, not a half point"
        assert size > 0
