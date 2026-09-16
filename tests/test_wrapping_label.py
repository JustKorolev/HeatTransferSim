"""The word-wrapped help labels must claim the height they actually draw.

A plain QLabel with setWordWrap(True) reports a minimum height for ONE line,
because it does not know the width it will be given. A QVBoxLayout budgets that
one line, the label then draws four, and everything below it is pushed down over
whatever comes next -- which is how the "Set all" row on the Headless Run tab came
to be painted across the sensor table.

Run in a SUBPROCESS on purpose. Several test modules install a stub PySide6 into
sys.modules (see test_build_modal_controller), so importing real Qt in-process
would get whatever the test order happened to leave behind. A fresh interpreter
is the only way to be sure this measures Qt and not a stub.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def _run(body: str) -> str:
    script = textwrap.dedent(
        """
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except Exception as exc:
            print("SKIP", exc)
            raise SystemExit(0)
        app = QtWidgets.QApplication([])
        """
    ) + textwrap.dedent(body)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0:
        pytest.fail(f"subprocess failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def _skipped(output: str) -> bool:
    return output.startswith("SKIP")


BOX = """
    from graph_visualizer.ui_theme import wrapping_label

    TEXT = ("Every sensor the graph declares, prefilled at 293.15 K. Edit a row to "
            "give that sensor its own target, or use Randomize / Set all. A row left "
            "blank keeps whatever setpoint the graph itself holds.")

    def build(use_wrapping):
        box = QtWidgets.QGroupBox("Per-sensor setpoints")
        lay = QtWidgets.QVBoxLayout(box)
        if use_wrapping:
            label = wrapping_label(QtCore, QtWidgets, TEXT)
        else:
            label = QtWidgets.QLabel(TEXT)
            label.setWordWrap(True)
        lay.addWidget(label)
        table = QtWidgets.QTableWidget(0, 3)
        table.setMinimumHeight(200)
        lay.addWidget(table, 1)
        row = QtWidgets.QHBoxLayout()
        spin = QtWidgets.QDoubleSpinBox()
        row.addWidget(spin)
        for text in ("Load sensors", "Set all", "Clear"):
            row.addWidget(QtWidgets.QPushButton(text))
        lay.addLayout(row)
        box.setFixedWidth(390)
        box.resize(390, box.minimumSizeHint().height())
        box.show()
        QtWidgets.QApplication.processEvents()
        bottom = table.mapTo(box, QtCore.QPoint(0, 0)).y() + table.height()
        top = spin.mapTo(box, QtCore.QPoint(0, 0)).y()
        return top - bottom
"""


def test_a_wrapping_label_leaves_room_for_the_row_below_it() -> None:
    output = _run(BOX + "\n    print(build(True))\n")
    if _skipped(output):
        pytest.skip(output)
    gap = int(output)
    assert gap >= 0, f"the buttons overlap the table by {-gap} px"


def test_a_plain_wrapped_label_is_the_thing_that_overlaps() -> None:
    """Pins the CAUSE, so this test starts failing if Qt ever fixes it upstream
    and the workaround becomes dead weight."""
    output = _run(BOX + "\n    print(build(False))\n")
    if _skipped(output):
        pytest.skip(output)
    assert int(output) < 0, "a plain wrapped label no longer under-budgets; drop the workaround"


def test_the_label_claims_a_multi_line_height_before_it_is_ever_shown() -> None:
    """The first layout pass queries minimumSizeHint before the label has a width.
    Claiming the height only in resizeEvent is too late -- the overlap has already
    happened by then."""
    output = _run(
        """
    from graph_visualizer.ui_theme import wrapping_label
    label = wrapping_label(QtCore, QtWidgets,
        "Every sensor the graph declares, prefilled at 293.15 K. Edit a row to give "
        "that sensor its own target, or use Randomize / Set all.")
    one_line = label.fontMetrics().height()
    print(label.minimumHeight(), one_line)
    """
    )
    if _skipped(output):
        pytest.skip(output)
    minimum, one_line = (int(value) for value in output.split())
    assert minimum > one_line * 2, f"claimed {minimum}px, barely more than one {one_line}px line"


def test_setting_new_text_reclaims_the_height() -> None:
    """graph_info and summary_label have their text replaced at runtime."""
    output = _run(
        """
    from graph_visualizer.ui_theme import wrapping_label
    label = wrapping_label(QtCore, QtWidgets, "short")
    label.resize(350, 20)
    QtWidgets.QApplication.processEvents()
    before = label.minimumHeight()
    label.setText("A considerably longer message that has to wrap over several lines "
                  "when the side panel is at its narrowest, and must claim the room.")
    print(before, label.minimumHeight())
    """
    )
    if _skipped(output):
        pytest.skip(output)
    before, after = (int(value) for value in output.split())
    assert after > before, f"height did not grow with the text ({before} -> {after})"


# --------------------------------------------------------------------------- #
# Status labels
# --------------------------------------------------------------------------- #
STATUS = """
    class Qt:
        QtCore = QtCore
        QtWidgets = QtWidgets

    from graph_visualizer.simulation_controls_panel import MODE_LIVE, SimulationControlsPanel

    panel = SimulationControlsPanel(Qt, mode=MODE_LIVE)
    host = QtWidgets.QWidget()
    panel.build(QtWidgets.QFormLayout(host))
    label = panel.modal_design_status_label
    label.setFixedWidth(360)

    LONG = ("MIMO PI gain matrix unavailable (the gain matrix references 12 node ids "
            "not in this graph -- it was built for a different graph, so its constants "
            "would not mean anything here); scheme not active. Re-run the sys ID.")
"""


def test_a_long_status_message_expands_the_label_instead_of_being_cut_off() -> None:
    """These are the messages that matter most when something has gone wrong -- a
    gain matrix that does not match the graph, an export that refused. Truncating
    them hid the reason."""
    output = _run(
        STATUS
        + """
    label.setText("Idle.")
    QtWidgets.QApplication.processEvents()
    short = label.sizeHint().height()
    label.setText(LONG)
    QtWidgets.QApplication.processEvents()
    print(short, label.sizeHint().height(), label.maximumHeight())
    """
    )
    if _skipped(output):
        pytest.skip(output)
    short, long_height, maximum = (int(value) for value in output.split())
    assert long_height > short, f"the label did not grow ({short} -> {long_height})"
    assert maximum > 10000, f"a maximum height of {maximum} would clip it again"


def test_a_short_message_keeps_a_two_line_floor() -> None:
    """Without a floor the row collapses and shuffles everything below it on every
    status update."""
    output = _run(
        STATUS
        + """
    label.setText("Idle.")
    QtWidgets.QApplication.processEvents()
    print(label.minimumHeight(), label.fontMetrics().lineSpacing())
    """
    )
    if _skipped(output):
        pytest.skip(output)
    minimum, line = (int(value) for value in output.split())
    assert minimum >= line * 2, f"floor {minimum}px is under two {line}px lines"


def test_the_floor_is_re_measured_after_a_ui_scale_change() -> None:
    """The floor is in pixels, from the font at build time; at a larger scale a
    stale floor is less than two lines."""
    output = _run(
        STATUS
        + """
    before = label.minimumHeight()
    big = QtGui.QFont(QtWidgets.QApplication.font())
    big.setPointSizeF(18.0)
    QtWidgets.QApplication.setFont(big)
    for widget in QtWidgets.QApplication.allWidgets():
        try:
            widget.setFont(big)
        except Exception:
            pass
    panel.repin_two_line_labels()
    print(before, label.minimumHeight())
    """
    )
    if _skipped(output):
        pytest.skip(output)
    before, after = (int(value) for value in output.split())
    assert after > before, f"floor did not grow with the font ({before} -> {after})"
