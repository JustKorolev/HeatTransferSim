"""Typography, UI scale, and how numeric fields are sized and rounded.

Three separate complaints, one cause: every QDoubleSpinBox in the application was
built with ``setDecimals(8)`` and no width limit. "293.15000000" is eleven
characters of which three carry information, and a field wide enough to hold it
crowds the buttons beside it off the panel. Fixing it by simply setting three
decimals everywhere would be worse than the problem: steps in this UI go down to
1e-14, and ``setValue`` ROUNDS TO THE FIELD'S DECIMALS -- so loading a saved
lambda of 1e-6 into a three-decimal box turns it into 0.0, and the next autosave
writes that zero back to disk.

:func:`decimals_for` therefore derives the number from the field itself: three by
default (clean), more when the step or the current value genuinely needs it, and
never more than the eight it used to be. :func:`configure_double_spin` also
widens a box on the fly when a value arrives that would otherwise be rounded
away, so no saved number can be destroyed by being displayed.

The rest is the View menu's business: a font that is not the Qt default, and a
user-settable scale for machines whose DPI Qt guesses badly.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

#: Floor for a spin box's decimals. Three is enough for a temperature in kelvin,
#: a power in watts, or a time in seconds, which is nearly everything here.
MIN_DECIMALS = 3
#: Ceiling. What every field used to use unconditionally.
MAX_DECIMALS = 8

#: Numeric fields are captions, not paragraphs. Without a cap they take whatever
#: the form layout will give them and squeeze out the buttons on the same row --
#: which is exactly how the "Set all" row on the Headless Run tab ended up
#: overlapping the table above it.
SPIN_MAX_WIDTH = 150

#: Width a wrapped help label assumes before it has been laid out, so it can claim
#: a realistic height on the first pass. Slightly under the side panel's 400 px
#: minimum, less the group-box and layout margins -- i.e. the narrowest the label
#: will ever actually be, which is the case that needs the most lines.
ASSUMED_LABEL_WIDTH = 350

#: Point size the UI is designed at, before the View menu's scale is applied.
BASE_POINT_SIZE = 9.0

#: View > UI scale is a slider over this range, in percent. 100% is
#: BASE_POINT_SIZE. The bottom is deliberately far below "small": on a display
#: whose DPI Qt over-estimates, everything is already enlarged before the
#: application gets a say, and the only useful correction is downward.
MIN_UI_SCALE = 40
MAX_UI_SCALE = 200
#: Slider granularity, and how far one Ctrl+plus / Ctrl+minus moves.
UI_SCALE_STEP = 5
UI_SCALE_KEY_STEP = 10
DEFAULT_UI_SCALE = 100

#: Preferred UI fonts, best first. Qt's own default on Windows is a bitmapped
#: fallback in some styles and looks nothing like the rest of the desktop; these
#: are the humanist sans faces that ship with each platform, ending in one that
#: is nearly always present on Linux.
PREFERRED_FONTS: tuple[str, ...] = (
    "Segoe UI Variable Text",
    "Segoe UI",
    "Inter",
    "SF Pro Text",
    "Helvetica Neue",
    "Ubuntu",
    "Cantarell",
    "Noto Sans",
    "DejaVu Sans",
)

#: Monospace, for anything that has to line up in columns.
PREFERRED_MONO_FONTS: tuple[str, ...] = (
    "Cascadia Mono",
    "Consolas",
    "SF Mono",
    "JetBrains Mono",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "monospace",
)


# --------------------------------------------------------------------------- #
# Numeric fields
# --------------------------------------------------------------------------- #
def _decimals_to_represent(value: float, maximum: int = MAX_DECIMALS) -> int:
    """Fewest decimals that display ``value`` without changing it.

    This is the guard against silent data loss: a spin box rounds whatever it is
    given to its own decimals, so a field must never be narrower than the value
    it is about to hold.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number != number or number in (float("inf"), float("-inf")) or number == 0.0:
        return 0
    for digits in range(0, maximum + 1):
        # Relative tolerance: 293.15 must read as exact at 2 decimals despite
        # binary representation, and 1e-14 must not be declared exact at 0.
        if abs(round(number, digits) - number) <= abs(number) * 1.0e-12:
            return digits
    return maximum


def _decimals_for_step(step: float, maximum: int = MAX_DECIMALS) -> int:
    """Enough decimals that pressing the up-arrow visibly changes the number.

    A 1e-5 step in a 3-decimal box does nothing at all on screen, which reads as
    a broken control.
    """
    try:
        size = abs(float(step))
    except (TypeError, ValueError):
        return 0
    if size <= 0.0 or size != size:
        return 0
    return _decimals_to_represent(size, maximum)


def decimals_for(
    step: float = 0.0,
    value: float = 0.0,
    *,
    minimum: int = MIN_DECIMALS,
    maximum: int = MAX_DECIMALS,
) -> int:
    """Decimals for a spin box with this step and current value."""
    needed = max(_decimals_for_step(step, maximum), _decimals_to_represent(value, maximum))
    return max(minimum, min(maximum, needed))


def configure_double_spin(widget: Any, step: float, value: float) -> Any:
    """Apply this module's conventions to an already-constructed spin box.

    Kept separate from construction so both panels can call it on their own
    ``NoWheelDoubleSpinBox`` subclasses without either of them having to know how
    the decimals are chosen.
    """
    widget.setDecimals(decimals_for(step, value))
    widget.setMaximumWidth(SPIN_MAX_WIDTH)
    return widget


def widen_decimals_for(widget: Any, value: float) -> None:
    """Grow ``widget``'s decimals if ``value`` would otherwise be rounded away.

    Call before ``setValue``. Never shrinks: a field that grew to hold 1e-6 keeps
    room for it, so typing over the value and undoing does not lose it.
    """
    try:
        current = int(widget.decimals())
    except Exception:  # noqa: BLE001 - a stub, or an exotic widget
        return
    needed = _decimals_to_represent(value, MAX_DECIMALS)
    if needed > current:
        try:
            widget.setDecimals(min(MAX_DECIMALS, needed))
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Fonts and scale
# --------------------------------------------------------------------------- #
def wrapping_label(QtCore: Any, QtWidgets: Any, text: str = "") -> Any:
    """A word-wrapped QLabel that tells its layout how tall it really is.

    A plain ``QLabel`` with ``setWordWrap(True)`` reports a minimum height for one
    line, because it does not know the width it will be given. A QVBoxLayout
    therefore budgets one line, the label draws four, and everything below it is
    pushed down over whatever comes next -- which is exactly how the "Set all" row
    on the Headless Run tab ended up painted across the sensor table.

    Measured on this group box: the layout's minimum was 275 px where the content
    needed 322.

    Reporting ``heightForWidth`` is not enough on its own, because a QVBoxLayout
    does not propagate it to the parent's minimum. Pinning ``minimumHeight`` on
    every resize is what actually stops the squeeze.
    """

    class _WrappingLabel(QtWidgets.QLabel):
        def __init__(self, content: str = "") -> None:
            super().__init__(content)
            self.setWordWrap(True)
            # Everything below is a refinement on a plain wrapped label, and the
            # panel is also built against a Qt STUB in the tests, which implements
            # only what the panel needs. Degrade to a plain label rather than
            # making the stub carry Qt's whole size-policy machinery.
            try:
                policy = self.sizePolicy()
                policy.setHeightForWidth(True)
                self.setSizePolicy(policy)
            except Exception:  # noqa: BLE001 - not a full QWidget
                pass
            # Claim the height NOW, at the narrowest width the side panel can be.
            # resizeEvent alone is too late: the first layout pass queries
            # minimumSizeHint before the label has ever been given a width, gets
            # one line, and the overlap happens during that first pass.
            try:
                self.setMinimumHeight(self.heightForWidth(ASSUMED_LABEL_WIDTH))
            except Exception:  # noqa: BLE001 - no font metrics on a stub
                pass

        def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt override name
            return True

        def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt override name
            usable = max(1, int(width))
            return (
                self.fontMetrics()
                .boundingRect(
                    QtCore.QRect(0, 0, usable, 0),
                    int(QtCore.Qt.TextWordWrap),
                    self.text(),
                )
                .height()
            )

        def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override name
            super().resizeEvent(event)
            self._claim_height()

        def setText(self, content: str) -> None:  # noqa: N802 - Qt override name
            super().setText(content)
            # graph_info and summary_label have their text replaced at run time;
            # a longer message must claim the extra room it now needs.
            self._claim_height()

        def _claim_height(self) -> None:
            """Pin minimumHeight to what the text actually needs at this width.

            Every call is wrapped: the panel is also built against a Qt stub in the
            tests, which has no geometry at all, and a help label failing to
            measure itself must never be what stops the panel from building.
            """
            try:
                width = int(self.width())
                if width > 0:
                    self.setMinimumHeight(self.heightForWidth(width))
            except Exception:  # noqa: BLE001 - not a full QWidget
                pass

    return _WrappingLabel(text)


def choose_font_family(available: Iterable[str], preferred: Sequence[str] = PREFERRED_FONTS) -> str:
    """First preferred family this system actually has, or "" for Qt's default.

    Matched case-insensitively: Qt reports "Segoe UI" but a fontconfig system may
    report "segoe ui", and falling back to the Qt default over a capital letter
    would be a silly way to lose the font.
    """
    lookup = {str(name).strip().lower(): str(name) for name in available}
    for name in preferred:
        found = lookup.get(name.strip().lower())
        if found:
            return found
    return ""


def clamp_scale(percent: Any) -> int:
    """``percent`` held inside the slider's range and snapped to its step.

    Junk becomes the default rather than raising: this reads a value out of
    QSettings, which can hold anything a previous version or a hand-edited file
    put there, and a bad scale must not stop the window from opening.
    """
    try:
        wanted = float(percent)
    except (TypeError, ValueError):
        return DEFAULT_UI_SCALE
    if wanted != wanted:  # NaN
        return DEFAULT_UI_SCALE
    snapped = int(round(wanted / UI_SCALE_STEP)) * UI_SCALE_STEP
    return max(MIN_UI_SCALE, min(MAX_UI_SCALE, snapped))


def step_scale(percent: Any, direction: int) -> int:
    """One keyboard step up (+1) or down (-1) from ``percent``."""
    return clamp_scale(clamp_scale(percent) + int(direction) * UI_SCALE_KEY_STEP)


def point_size_for(scale_percent: Any, base: float = BASE_POINT_SIZE) -> float:
    """Font point size for a UI scale, rounded to a half point.

    Qt renders fractional point sizes, but quarter-points make text jitter
    between widgets that round differently; halves are stable and still give the
    scale menu eight distinct steps.
    """
    scale = clamp_scale(scale_percent)
    return round(base * scale / 100.0 * 2.0) / 2.0
