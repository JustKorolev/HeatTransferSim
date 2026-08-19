"""The dark theme's palette has to stay legible on the slide it is drawn for.

A colour that is too close to the background does not fail loudly -- the figure
renders, the curve is simply not there. The light-surface ordinal ramp's dark end
(#0d366b) sits at 1.46:1 against the deck's near-black, so switching themes
without rebuilding the ramp silently loses a series.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "surf_report" / "make_validation_figures.py"


def _module():
    if not SCRIPT.is_file():
        pytest.skip(f"{SCRIPT.name} is not present")
    spec = importlib.util.spec_from_file_location("_val_figs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _luminance(hex_colour: str) -> float:
    raw = hex_colour.lstrip("#")
    channels = [int(raw[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


@pytest.fixture(scope="module")
def dark():
    module = _module()
    module.use_dark()
    return module


# Grid and spines are furniture and are meant to sit under the data; everything
# else carries meaning and has to be visible.
FURNITURE = {"GRID"}
DATA_INK = ["INK", "INK2", "MUTED", "BLUE", "ORANGE", "GOOD", "WARNING", "WARN_TEXT", "AXIS"]


@pytest.mark.parametrize("name", DATA_INK)
def test_every_meaningful_colour_clears_three_to_one(dark, name) -> None:
    colour = getattr(dark, name)
    ratio = _contrast(colour, dark.SURFACE)
    assert ratio >= 3.0, f"{name}={colour} is {ratio:.2f}:1 on {dark.SURFACE}"


def test_the_grid_stays_under_the_data(dark) -> None:
    """The other half of the check: a grid bright enough to pass a contrast floor
    would out-glare the curves. On a light surface this is what going too far the
    other way looks like -- the light theme's #e1e0d9 grid is 11:1 on near-black."""
    ratio = _contrast(dark.GRID, dark.SURFACE)
    assert 1.2 <= ratio <= 2.2, f"grid={dark.GRID} is {ratio:.2f}:1"


def test_the_ordinal_ramp_is_visible_end_to_end(dark) -> None:
    for index, colour in enumerate(dark.SEQ4):
        ratio = _contrast(colour, dark.SURFACE)
        assert ratio >= 3.0, f"SEQ4[{index}]={colour} is {ratio:.2f}:1"


def test_the_ordinal_ramp_steps_are_separable(dark) -> None:
    """Four depths share one axis in the prism figure, so adjacent steps have to be
    tellable apart, and evenly so or the ordering stops reading as an ordering."""
    steps = [_contrast(a, b) for a, b in zip(dark.SEQ4, dark.SEQ4[1:])]
    assert all(s >= 1.35 for s in steps), steps
    assert max(steps) / min(steps) <= 1.5, f"unevenly spaced: {steps}"


def test_the_ramp_is_monotonic(dark) -> None:
    """Light-first. If a theme swap reverses or scrambles it, the figure still draws
    and just encodes depth backwards."""
    lums = [_luminance(c) for c in dark.SEQ4]
    assert lums == sorted(lums, reverse=True), lums


def test_the_light_theme_ramp_would_have_been_invisible() -> None:
    """Why the ramp is rebuilt rather than reused. Guards the reasoning itself: if
    someone drops the override because "the blues look fine", this says what breaks.
    """
    assert _contrast("#0d366b", "#1a1a1a") < 1.5


def test_dark_mode_writes_beside_the_light_figures(dark) -> None:
    """Same run, different suffix -- the light versions are still the report's."""
    assert dark.STEM_SUFFIX == "_dark"
    assert dark.DARK is True
