"""Headless modal-controller build.

The simulation tab designs controllers from a graph held in the GUI process. For
no_mli_high_res that means parsing a 10.3 GB graph.json into ~45 GB of per-node
and per-edge dicts, and holding all of it while splu factorizes the 3M x 3M DC
operator. On a 64 GB machine that left ~15-19 GB for a factorization whose fill-in
is unpredictable, and the observed failure mode was thrashing that killed a remote
session rather than a clean error.

Everything the reduction needs is now in the fast-load artifacts, so the same
design runs at ~20 GB in a separate process.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

for _name in ("PySide6", "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

SCRIPT = Path(__file__).resolve().parent.parent / "build_modal_controller.py"


def test_refuses_the_45gb_loader_unless_explicitly_allowed(tmp_path) -> None:
    """Silently falling back to graph.json is the failure this script exists to
    prevent, so it must be opt-in."""
    graph = tmp_path / "g"
    graph.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(graph), "--modes", "4", "--order", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "--allow-full-load" in combined
    assert "45 GB" in combined


def test_missing_graph_folder_is_reported(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "nope")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Not a graph folder" in (result.stdout + result.stderr)


def test_existing_artifact_is_backed_up_not_silently_overwritten(tmp_path) -> None:
    """modal_artifact_filename is deterministic, so a rebuild with the same
    descriptors overwrites. The previous artifact is the only record of what
    produced earlier runs."""
    from graph_visualizer.modal_reduction import modal_artifact_filename

    graph = tmp_path / "g"
    graph.mkdir()
    existing = graph / modal_artifact_filename(50, 140, 50.0)
    existing.write_bytes(b"previous artifact")
    # Fails later (no artifacts to load), but the backup must already have happened.
    subprocess.run(
        [sys.executable, str(SCRIPT), str(graph), "--t-op", "50", "--modes", "140", "--order", "50"],
        capture_output=True, text=True,
    )
    backups = list(graph.glob("*.bak.npz"))
    assert len(backups) == 1, f"expected one backup, found {backups}"
    assert backups[0].read_bytes() == b"previous artifact"


def test_no_backup_flag_is_honoured(tmp_path) -> None:
    from graph_visualizer.modal_reduction import modal_artifact_filename

    graph = tmp_path / "g"
    graph.mkdir()
    (graph / modal_artifact_filename(50, 140, 50.0)).write_bytes(b"previous")
    subprocess.run(
        [sys.executable, str(SCRIPT), str(graph), "--no-backup"],
        capture_output=True, text=True,
    )
    assert list(graph.glob("*.bak.npz")) == []


def test_launcher_passes_the_design_descriptors(monkeypatch, tmp_path) -> None:
    """The tab builds at the panel's operating point, so the artifact matches the
    regime the run will use."""
    import graph_visualizer.fast_graph_io as fio

    captured: dict = {}

    class _Proc:
        returncode = None
        def poll(self): return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    fio.launch_modal_build_subprocess(tmp_path, t_op_K=42.5, n_modes=99, order=7)
    command = captured["command"]
    assert str(SCRIPT) in command[1]
    assert "--t-op" in command and "42.5" in command
    assert "--modes" in command and "99" in command
    assert "--order" in command and "7" in command
    assert (tmp_path / "build_modal_controller.log").exists()


def test_tab_refuses_to_build_without_fast_load_artifacts(tmp_path, monkeypatch) -> None:
    """Otherwise the button quietly triggers the 45 GB path it exists to avoid."""
    import test_simulation_controls_panel as panel_stubs
    from graph_visualizer.headless_run_tab import HeadlessRunTab

    graph = tmp_path / "graphs" / "g"
    graph.mkdir(parents=True)
    (graph / "node_ids.npy").write_bytes(b"")   # listed, but no edges.npz / L_sparse
    messages: list[tuple[str, bool]] = []
    tab = HeadlessRunTab(
        panel_stubs._QtStub, None,
        on_status=lambda m, e: messages.append((m, e)),
        graphs_root=lambda: tmp_path / "graphs",
    )
    tab.build_modal_controller()
    assert tab._modal_build_process is None, "must not have launched anything"
    assert any(error and "Update graph" in message for message, error in messages), messages


def _tab_with_graph(tmp_path, monkeypatch, *, fast_loadable: bool):
    import test_simulation_controls_panel as panel_stubs
    from graph_visualizer.headless_run_tab import HeadlessRunTab

    graph = tmp_path / "graphs" / "g"
    graph.mkdir(parents=True)
    (graph / "node_ids.npy").write_bytes(b"")
    if fast_loadable:
        monkeypatch.setattr(
            "graph_visualizer.fast_graph_io.can_load_fast", lambda folder: (True, "ok")
        )
    return HeadlessRunTab(
        panel_stubs._QtStub, None, graphs_root=lambda: tmp_path / "graphs"
    ), graph


def test_headless_exposes_every_modal_design_control(tmp_path, monkeypatch) -> None:
    """The tab offers a build button, so it must also offer the knobs that decide
    what gets built -- these rows used to be hidden in headless mode."""
    tab, _graph = _tab_with_graph(tmp_path, monkeypatch, fast_loadable=False)
    for name in (
        "modal_temp_spin", "modal_modes_spin", "modal_order_spin",
        "modal_effort_spin", "modal_integral_spin", "modal_design_button",
        "modal_design_status_label",
    ):
        assert getattr(tab, name, None) is not None, name
    from graph_visualizer.simulation_controls_panel import _HEADLESS_HIDDEN_ROWS

    for row in ("modal_operating_temperature", "modal_modes", "modal_order",
                "modal_effort", "modal_build", "modal_status"):
        assert row not in _HEADLESS_HIDDEN_ROWS, row


def test_build_uses_the_panel_values_not_hardcoded_defaults(tmp_path, monkeypatch) -> None:
    tab, _graph = _tab_with_graph(tmp_path, monkeypatch, fast_loadable=True)
    tab.modal_temp_spin.setValue(42.0)
    tab.modal_modes_spin.setValue(64)
    tab.modal_order_spin.setValue(12)
    tab.modal_effort_spin.setValue(0.35)
    tab.modal_integral_spin.setValue(0.02)
    # The LQR gain is only valid at the sample rate it was solved for, so the
    # build has to inherit the run's dt rather than a design-time default.
    tab.panel.inputs["dt_s"].setValue(2.5)

    captured: dict = {}

    def fake_launch(folder, **kwargs):
        captured.update(kwargs)
        class _P:
            def poll(self): return None
        return _P()

    monkeypatch.setattr(
        "graph_visualizer.fast_graph_io.launch_modal_build_subprocess", fake_launch
    )
    tab.build_modal_controller()
    assert captured == {
        "t_op_K": 42.0, "n_modes": 64, "order": 12,
        "effort": 0.35, "integral_gain": 0.02, "design_dt_s": 2.5,
    }, captured


def test_reduced_order_above_the_mode_count_is_rejected(tmp_path, monkeypatch) -> None:
    """r is truncated FROM the modes, so r > modes is not a thing to discover an
    hour into a reduction."""
    tab, _graph = _tab_with_graph(tmp_path, monkeypatch, fast_loadable=True)
    tab.modal_modes_spin.setValue(20)
    tab.modal_order_spin.setValue(50)
    messages: list[tuple[str, bool]] = []
    tab.on_status = lambda m, e: messages.append((m, e))
    tab.build_modal_controller()
    assert tab._modal_build_process is None
    assert any(error and "cannot exceed" in message for message, error in messages), messages
