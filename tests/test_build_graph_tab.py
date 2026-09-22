"""The Build Graph tab, driven against real Qt in a subprocess.

Run out of process because several test modules install a stub PySide6 into
sys.modules, so an in-process import would get whatever the test order left
behind. The tab is real Qt here -- offscreen, but the actual widgets.

Nothing in this file starts a build. The voxelizer is memory-hungry enough to
take a machine down, and what needs testing is the form and the command it
produces, not the builder.
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
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:
    print("SKIP", exc)
    raise SystemExit(0)
from pathlib import Path

class Qt:
    QtCore = QtCore
    QtWidgets = QtWidgets

app = QtWidgets.QApplication([])
from graph_visualizer.build_graph_tab import BuildGraphTab

MESSAGES = []

def make(mesh_root="meshes"):
    return BuildGraphTab(
        Qt, None,
        on_status=lambda m, e=False: MESSAGES.append((m, e)),
        mesh_root=lambda: Path(mesh_root),
    )
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


# --------------------------------------------------------------------------- #
# Source discovery
# --------------------------------------------------------------------------- #
def test_only_step_assemblies_are_offered() -> None:
    """GLB is no longer an input format, so a mesh-only folder must not appear --
    offering it and then failing on the button is worse than not offering it."""
    output = _run(
        """
        tab = make()
        names = [tab.folder_combo.itemText(i) for i in range(tab.folder_combo.count())]
        print("|".join(names))
        """
    )
    if _skip(output):
        pytest.skip(output)
    names = [n for n in output.split("|") if n]
    assert "step_test" in names, "the shipped sample should be discoverable"
    assert "cryostat" not in names, "cryostat/ holds only a GLB"
    assert "validation_mesh" not in names, "validation_mesh/ holds only a GLB"


def test_an_empty_mesh_root_says_what_to_do(tmp_path) -> None:
    output = _run(
        f"""
        tab = make({str(tmp_path)!r})
        print(tab.source_info.text())
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert "No STEP assemblies" in output
    assert ".step" in output, "should say what file it is looking for"


def test_the_graph_name_is_prefilled_from_the_folder() -> None:
    output = _run(
        """
        tab = make()
        print(tab.folder_combo.currentText(), tab.graph_name_input.text())
        """
    )
    if _skip(output):
        pytest.skip(output)
    folder, name = output.split()
    assert name == folder.upper()


def test_the_graph_name_follows_the_folder_until_the_user_types_one() -> None:
    """Switching assembly used to leave the previous folder's name in place, so a
    build of cryostat_step landed in a graph called STEP_TEST. An edited name,
    though, is the user's and must survive."""
    output = _run(
        """
        tab = make()
        first = tab.graph_name_input.text()
        tab.folder_combo.setCurrentText("cryostat_step")
        followed = tab.graph_name_input.text()
        tab.graph_name_input.setText("MY_OWN")
        tab.folder_combo.setCurrentText("step_test")
        print(first, followed, tab.graph_name_input.text())
        """
    )
    if _skip(output):
        pytest.skip(output)
    parts = output.split()
    if parts[1] == parts[0]:
        pytest.skip("only one assembly folder here, so there is nothing to switch to")
    assert parts == ["STEP_TEST", "CRYOSTAT_STEP", "MY_OWN"], output


def test_a_missing_materials_lookup_is_called_out(tmp_path) -> None:
    """Without it every part takes the unassigned default, so the conductances do
    not describe the assembly -- that has to be said, not left to be discovered."""
    folder = tmp_path / "bare"
    folder.mkdir()
    (folder / "a.step").write_text("ISO-10303-21;", encoding="utf-8")
    output = _run(
        f"""
        tab = make({str(tmp_path)!r})
        print(tab.source_info.text())
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert "no Materials.xlsx" in output
    assert "unassigned" in output


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #
def test_the_command_the_tab_shows_is_the_command_it_would_run() -> None:
    """The preview exists so a build can be checked before it costs an hour. If it
    and current_argv() could disagree, it would be worse than showing nothing."""
    output = _run(
        """
        tab = make()
        shown = tab.command_view.toPlainText()
        argv = tab.current_argv()
        missing = [a for a in argv if a not in shown]
        print("MISSING" if missing else "MATCH", missing[:3])
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert output.startswith("MATCH"), output


def test_edits_reach_the_command() -> None:
    output = _run(
        """
        tab = make()
        tab.inputs["max_depth"].setValue(7)
        tab.inputs["heater_name_substring"].setText("FOO-HEATER, BAR-HEATER")
        tab.inputs["boundary_refine"].setChecked(True)
        argv = tab.current_argv()
        print("depth7" if "7" in argv else "no-depth",
              "foo" if "FOO-HEATER" in argv else "no-foo",
              "bar" if "BAR-HEATER" in argv else "no-bar",
              "on" if "--boundary-refine" in argv else "off")
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert output.split() == ["depth7", "foo", "bar", "on"], output


def test_the_generated_command_parses(tmp_path) -> None:
    """End to end: the form's argv must satisfy the real parser."""
    output = _run(
        """
        from octree_graph.cli import build_parser
        tab = make()
        args = build_parser().parse_args(tab.current_argv())
        print(args.graph_name, args.max_depth, args.voxel_workers, args.boundary_refine)
        """
    )
    if _skip(output):
        pytest.skip(output)
    name, depth, workers, boundary = output.split()
    assert name == "STEP_TEST"
    assert depth == "10" and workers == "8"
    assert boundary == "False", "the reference build has boundary refine off"


# --------------------------------------------------------------------------- #
# Guards before a long job
# --------------------------------------------------------------------------- #
def test_building_without_a_name_refuses_rather_than_starting(tmp_path) -> None:
    """A build costs real time; a missing name should stop it at the button."""
    output = _run(
        """
        tab = make()
        tab.graph_name_input.setText("   ")
        tab.start_build()
        print("started" if tab.process is not None else "refused", "|", MESSAGES[-1][0])
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert output.startswith("refused"), output
    assert "name" in output.lower()


def test_building_with_no_assembly_selected_refuses(tmp_path) -> None:
    output = _run(
        f"""
        tab = make({str(tmp_path)!r})
        tab.graph_name_input.setText("X")
        tab.start_build()
        print("started" if tab.process is not None else "refused", "|", MESSAGES[-1][0])
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert output.startswith("refused"), output


def test_stop_without_a_build_says_so() -> None:
    output = _run(
        """
        tab = make()
        tab.stop_build()
        print(MESSAGES[-1][0])
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert "No build is running" in output


def test_shutdown_stops_polling_but_leaves_the_build_alone() -> None:
    """The build is detached on purpose: closing the window is not a request to
    discard hours of work."""
    output = _run(
        """
        import inspect
        from graph_visualizer.build_graph_tab import BuildGraphTab
        tab = make()
        tab.shutdown()
        source = inspect.getsource(BuildGraphTab.shutdown)
        print("timer-stopped" if not tab._poll_timer.isActive() else "still-running",
              "kills" if ("terminate" in source or "kill" in source) else "leaves-it")
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert output.split() == ["timer-stopped", "leaves-it"], output


def test_the_log_tail_reads_incrementally(tmp_path) -> None:
    """A long build's log reaches megabytes; re-reading it every second would make
    the UI the slowest part of the build."""
    output = _run(
        f"""
        tab = make()
        folder = Path({str(tmp_path)!r})
        tab._output_folder = folder
        log = folder / "conversion.log"
        log.write_text("line one\\n", encoding="utf-8")
        tab._tail_log()
        first = tab._log_size
        with log.open("a", encoding="utf-8") as h:
            h.write("line two\\n")
        tab._tail_log()
        print(first, tab._log_size, tab.log_view.toPlainText().count("line"))
        """
    )
    if _skip(output):
        pytest.skip(output)
    first, second, lines = (int(v) for v in output.split())
    assert second > first, "the offset did not advance"
    assert lines == 2, f"expected both lines exactly once, got {lines}"


# --------------------------------------------------------------------------- #
# Findability
# --------------------------------------------------------------------------- #
#: Built once: the index walk is the slow part, and every query below reads it.
_HELP_PREAMBLE = """
        from graph_visualizer.help_center import HelpCenter

        class App:
            QtCore = QtCore
            QtWidgets = QtWidgets
            window = None

        app_obj = App()
        app_obj.view_tabs = QtWidgets.QTabWidget()
        tab = make()
        app_obj.view_tabs.addTab(tab.widget, "Build Graph")
        app_obj.view_tabs.addTab(QtWidgets.QWidget(), "3D Octree Graph Editor")
        app_obj.build_graph_tab = tab
        center = HelpCenter(app_obj)
"""


def test_every_build_parameter_is_findable() -> None:
    """The tab's controls are not a SimulationControlsPanel, and the index used to
    walk only that shape -- so this whole tab was a search dead end: 21 parameters
    on screen and the only hit was the tab itself."""
    output = _run(
        _HELP_PREAMBLE
        + """
        from graph_visualizer.build_graph_params import BUILD_FIELDS
        indexed = {t.key.split(":", 1)[-1] for t in center.build_index()}
        missing = [f.dest for _s, fs in BUILD_FIELDS for f in fs if f.dest not in indexed]
        print("MISSING", "|".join(missing))
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert output == "MISSING", output


def test_searching_finds_the_control_a_user_would_describe() -> None:
    output = _run(
        _HELP_PREAMBLE
        + """
        for query, expected in (
            ("min cell size", "min_cell_size_mm"),
            ("voxel workers", "voxel_workers"),
            ("deflection", "step_deflection_mm"),
            ("heater substring", "heater_name_substring"),
            ("max leaf", "max_leaf_cells"),
            ("graph name", "build_graph_name"),
            ("contact detection distance", "contact_detection_distance_mm"),
        ):
            hits = center.find(query)
            top = hits[0].target.key.split(":", 1)[-1] if hits else "NONE"
            print(query, "->", top, "OK" if top == expected else "WRONG(want " + expected + ")")
        """
    )
    if _skip(output):
        pytest.skip(output)
    bad = [line for line in output.splitlines() if not line.endswith("OK")]
    assert not bad, "\n".join([output])


def test_a_build_parameter_carries_its_section_and_tab() -> None:
    """A result reading 'min cell size' with no context does not tell the user
    where to look; the section is what makes the hit legible."""
    output = _run(
        _HELP_PREAMBLE
        + """
        target = center.find("min cell size")[0].target
        print(target.tab, "/", target.section, "/", target.kind)
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert output.startswith("Build Graph / "), output
    assert output.endswith("/ control"), output
    assert "/  /" not in output, "the section came through empty"


def test_revealing_a_build_parameter_switches_to_the_tab_and_flashes_it() -> None:
    output = _run(
        _HELP_PREAMBLE
        + """
        app_obj.view_tabs.setCurrentIndex(1)
        target = center.find("voxel workers")[0].target
        before = target.widget.styleSheet()
        shown = center.reveal(target)
        changed = target.widget.styleSheet() != before
        center._restore_flashed_widget()
        print(shown, app_obj.view_tabs.currentIndex(), changed,
              target.widget.styleSheet() == before)
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert output.split() == ["True", "0", "True", "True"], output


def test_no_build_row_key_shadows_a_simulation_row_key() -> None:
    """'Show me' resolves a bare row key against the first tab that offers it, and
    this tab is first. A key shared with the simulation panel would silently send
    every tutorial step for it to the wrong tab."""
    import test_simulation_controls_panel as stub
    from graph_visualizer.build_graph_params import help_row_keys
    from graph_visualizer.simulation_controls_panel import (
        MODE_LIVE,
        SimulationControlsPanel,
    )

    panel = SimulationControlsPanel(stub._QtStub, mode=MODE_LIVE)
    panel.build(stub.QFormLayout())
    clash = help_row_keys() & set(panel._rows)
    assert not clash, sorted(clash)


def test_a_control_on_a_background_tab_is_still_findable() -> None:
    """The point of the search is to find a control you are NOT looking at.

    Qt reports isVisible() False for a perfectly ordinary widget on a tab page
    that is not the current one, so an index built while another tab was open
    contained nothing but tab names. The Qt stub cannot show this -- its
    isVisible() is a plain flag -- so it has to be pinned against real widgets.
    """
    output = _run(
        _HELP_PREAMBLE
        + """
        app_obj.view_tabs.show()
        QtWidgets.QApplication.processEvents()
        app_obj.view_tabs.setCurrentIndex(1)  # look away from Build Graph
        QtWidgets.QApplication.processEvents()
        field = tab.inputs["min_cell_size_mm"]
        print("isVisible", field.isVisible(), "isHidden", field.isHidden())
        hits = center.find("min cell size")
        print("found", bool(hits))
        """
    )
    if _skip(output):
        pytest.skip(output)
    lines = dict(
        (line.split()[0], line.split()[1:]) for line in output.splitlines()
    )
    assert lines["isVisible"] == ["False", "isHidden", "False"], (
        f"Qt no longer behaves as this test is about: {output}"
    )
    assert lines["found"] == ["True"], "a background tab's control fell out of the index"


def test_the_mouse_wheel_does_not_retune_the_build() -> None:
    """The parameters sit in a scrolling sidebar. A wheel turn that lands on a
    spin box instead of the scroll area would change a build setting with no
    visible cause -- so both kinds of spin box refuse the wheel, as the
    simulation panel's do."""
    output = _run(
        """
        # Imported as QtNS: the preamble's own `Qt` is the injected namespace
        # that make() needs, and `from ... import Qt` would shadow it.
        from PySide6.QtCore import QPoint, QPointF, Qt as QtNS
        from PySide6.QtGui import QWheelEvent
        tab = make()
        checked = []
        for dest in ("max_depth", "min_cell_size_mm"):
            w = tab.inputs[dest]
            before = w.value()
            event = QWheelEvent(
                QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                QtNS.NoButton, QtNS.NoModifier, QtNS.NoScrollPhase, False,
            )
            w.wheelEvent(event)
            checked.append(f"{dest}={'unchanged' if w.value() == before else 'CHANGED'}")
        print(" ".join(checked))
        """
    )
    if _skip(output):
        pytest.skip(output)
    assert output == "max_depth=unchanged min_cell_size_mm=unchanged", output
