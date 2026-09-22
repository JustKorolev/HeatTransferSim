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
