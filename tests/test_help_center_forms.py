"""Harvesting a hand-built tab's controls out of its real Qt forms.

The Thermal Validation tab and the window's shared left panel keep no table of
their controls: labels go straight into ``addRow`` and buttons into box layouts.
Qt still knows both, and :meth:`HelpCenter._targets_from_forms` asks it -- which
is what turned those two panels from search dead ends into 100-odd findable
controls.

Real Qt in a subprocess, for the same reason test_build_graph_tab.py is: other
modules install a stub PySide6 into sys.modules, and the whole point here is the
behaviour of genuine QFormLayout and QGroupBox. The full window cannot be built
in this environment (VTK gets no OpenGL context), so this exercises the
harvester against a panel of the same shape rather than against the app.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

PREAMBLE = """
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    from PySide6 import QtCore, QtWidgets
except Exception as exc:
    print("SKIP", exc)
    raise SystemExit(0)

app = QtWidgets.QApplication([])
from graph_visualizer.help_center import HelpCenter


class Tab:
    '''A tab shaped like Thermal Validation: widgets in .inputs, labels in Qt.'''

    def __init__(self):
        self.controls_scroll = QtWidgets.QScrollArea()
        content = QtWidgets.QWidget()
        self.controls_scroll.setWidget(content)
        form = QtWidgets.QFormLayout(content)
        self.inputs = {}

        self.inputs["copper_rrr"] = QtWidgets.QSpinBox()
        form.addRow("copper RRR", self.inputs["copper_rrr"])

        box = QtWidgets.QGroupBox("Solver (mirrors the live simulator)")
        box_form = QtWidgets.QFormLayout(box)
        self.inputs["solver_rtol"] = QtWidgets.QDoubleSpinBox()
        box_form.addRow("linear rtol", self.inputs["solver_rtol"])
        self.spanning = QtWidgets.QCheckBox("use adaptive substeps")
        box_form.addRow(self.spanning)
        self.hidden = QtWidgets.QSpinBox()
        box_form.addRow("not for this mode", self.hidden)
        self.hidden.setVisible(False)
        form.addRow(box)

        # A button in a plain box layout: not a form row at all.
        strip = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(strip)
        self.run = QtWidgets.QPushButton("Run Validation")
        row.addWidget(self.run)
        form.addRow(strip)


class Tabs:
    def __init__(self, widget):
        self.widget = widget
        self.current = 0

    def count(self): return 1
    def tabText(self, i): return "Thermal Validation"
    def indexOf(self, w): return 0 if w is self.widget else -1
    def setCurrentIndex(self, i): self.current = i


class App:
    QtCore = QtCore
    QtWidgets = QtWidgets
    window = None


tab = Tab()
tab.widget = QtWidgets.QWidget()
app_obj = App()
app_obj.thermal_validation_tab = tab
app_obj.view_tabs = Tabs(tab.widget)
center = HelpCenter(app_obj)
targets = [t for t in center.build_index() if t.kind != "tab"]
by_label = {t.label: t for t in targets}
"""


def _run(body: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(PREAMBLE) + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=240,
    )
    if result.returncode != 0:
        pytest.fail(f"subprocess failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def _skip(output: str) -> bool:
    return output.startswith("SKIP")


def test_a_form_row_is_found_under_the_label_qt_holds() -> None:
    """The tab passes 'copper RRR' to addRow and keeps no copy. labelForField is
    the only place that text still exists."""
    output = _run("""
        print(sorted(by_label))
        """)
    if _skip(output):
        pytest.skip(output)
    assert "'copper RRR'" in output, output
    assert "'linear rtol'" in output, output


def test_the_key_comes_from_the_inputs_dict_so_a_tutorial_can_name_it() -> None:
    output = _run("""
        print(by_label["copper RRR"].key, by_label["linear rtol"].key)
        """)
    if _skip(output):
        pytest.skip(output)
    assert output.split() == ["0:copper_rrr", "0:solver_rtol"], output


def test_a_control_carries_the_title_of_its_group_box() -> None:
    output = _run("""
        print(repr(by_label["linear rtol"].section), repr(by_label["copper RRR"].section))
        """)
    if _skip(output):
        pytest.skip(output)
    assert output == "'Solver (mirrors the live simulator)' ''", output


def test_a_spanning_row_is_found_under_its_own_text() -> None:
    """addRow(widget) has no label; a checkbox's own text is the label."""
    output = _run("""
        print("use adaptive substeps" in by_label)
        """)
    if _skip(output):
        pytest.skip(output)
    assert output == "True", output


def test_a_button_in_a_box_layout_is_found() -> None:
    """Not a form row, and most of what a user searches by name."""
    output = _run("""
        t = by_label.get("Run Validation")
        print(t.key if t else "MISSING", t.kind if t else "")
        """)
    if _skip(output):
        pytest.skip(output)
    assert output.startswith("0:run_validation"), output


def test_a_hidden_row_is_not_offered() -> None:
    output = _run("""
        print("not for this mode" in by_label)
        """)
    if _skip(output):
        pytest.skip(output)
    assert output == "False", "a row hidden by the mode was offered anyway"


def test_nothing_is_listed_twice() -> None:
    """The form walk and the button sweep overlap; the sweep must skip what the
    walk already took, or every checkbox appears twice."""
    output = _run("""
        keys = [t.key for t in targets]
        print(len(keys), len(set(keys)))
        """)
    if _skip(output):
        pytest.skip(output)
    total, unique = output.split()
    assert total == unique, f"{total} targets but only {unique} distinct keys"


def test_searching_finds_a_harvested_control() -> None:
    output = _run("""
        for q in ("copper rrr", "linear rtol", "run validation", "adaptive substeps"):
            hits = center.find(q)
            print(q, "->", hits[0].target.label if hits else "NO RESULT")
        """)
    if _skip(output):
        pytest.skip(output)
    assert "NO RESULT" not in output, output
