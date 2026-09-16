"""Ctrl+C in the launching terminal must close the window promptly.

Python runs a signal handler only between bytecodes, and ``QApplication.exec`` is
C++ that does not return until the app quits. So a SIGINT arriving during the
event loop is recorded by the OS and then does nothing at all -- until some other
Python callback happens to fire, or never. The fix is a timer that pulls control
back into the interpreter a few times a second.

Two things are worth pinning: the state machine (first press closes cleanly, a
second gives up and exits), and the mechanism that makes the first press possible
at all -- that Python really does get slices while exec() is blocking.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from graph_visualizer.app import GraphVisualizerApp


class _Timer:
    def __init__(self, *_args) -> None:
        self.interval = None
        self.slots: list = []
        self.timeout = self

    def connect(self, slot) -> None:
        self.slots.append(slot)

    def start(self, interval) -> None:
        self.interval = interval

    def stop(self) -> None:
        self.interval = None


class _QtCore:
    QTimer = _Timer


class _Window:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _App:
    def __init__(self) -> None:
        self.quits = 0

    def quit(self) -> None:
        self.quits += 1


def _app_under_test() -> GraphVisualizerApp:
    """A GraphVisualizerApp with only the attributes the handler touches.

    object.__new__ because the real constructor builds the whole window, which
    needs an OpenGL context this environment does not have.
    """
    app = object.__new__(GraphVisualizerApp)
    app.QtCore = _QtCore
    app.window = _Window()
    app.app = _App()
    return app


# --------------------------------------------------------------------------- #
# The state machine
# --------------------------------------------------------------------------- #
def test_the_first_interrupt_closes_the_window_and_quits() -> None:
    app = _app_under_test()
    app._install_interrupt_handler()
    app._handle_interrupt(2, None)
    assert app.window.closed == 1, "closeEvent never ran, so nothing was shut down"
    assert app.app.quits == 1, "the event loop was not asked to stop"


def test_a_second_interrupt_exits_immediately(monkeypatch) -> None:
    """Shutting a tab down waits on a worker that may be mid-step, and a Ctrl+C
    that hangs is worse than one that is abrupt."""
    app = _app_under_test()
    app._install_interrupt_handler()
    app._handle_interrupt(2, None)

    exits: list[int] = []
    monkeypatch.setattr("os._exit", lambda code: exits.append(code))
    app._handle_interrupt(2, None)
    assert exits == [130], f"expected a hard exit with 130, got {exits}"


def test_the_hard_exit_uses_os_exit_not_sys_exit(monkeypatch) -> None:
    """A SystemExit raised inside a signal handler surfaces at whatever bytecode
    happened to be running, rather than unwinding the Qt loop."""
    import inspect

    source = inspect.getsource(GraphVisualizerApp._handle_interrupt)
    # Only the code lines: the docstring and an inline note both mention sys.exit
    # precisely in order to say it is the wrong thing to use here.
    body = source.split('"""')[0] + "".join(source.split('"""')[2:])
    code_lines = [line.split("#", 1)[0] for line in body.splitlines()]
    assert any("os._exit" in line for line in code_lines)
    assert not any("sys.exit" in line for line in code_lines)


def test_a_failure_to_close_still_quits() -> None:
    """A broken closeEvent must not strand the user in a window they asked to
    close."""
    app = _app_under_test()
    app._install_interrupt_handler()

    def explode() -> None:
        raise RuntimeError("closeEvent blew up")

    app.window.close = explode
    app._handle_interrupt(2, None)
    assert app.app.quits == 1


# --------------------------------------------------------------------------- #
# The mechanism
# --------------------------------------------------------------------------- #
def test_a_poll_timer_is_started_so_python_can_run() -> None:
    app = _app_under_test()
    app._install_interrupt_handler()
    assert app._interrupt_timer.interval == GraphVisualizerApp.INTERRUPT_POLL_MS
    assert app._interrupt_timer.slots, "the timer has no slot, so it wakes nothing"
    assert 0 < GraphVisualizerApp.INTERRUPT_POLL_MS <= 250, "too slow to feel instant"


def test_python_really_gets_slices_while_exec_is_blocking() -> None:
    """The claim the whole feature rests on.

    Runs a real Qt loop in a subprocess with the same timer pattern and counts how
    often Python ran. Without the timer the count would be 0 and Ctrl+C would do
    nothing until the user touched the window.
    """
    script = textwrap.dedent(
        """
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        try:
            from PySide6 import QtCore, QtWidgets
        except Exception as exc:
            print("SKIP", exc)
            raise SystemExit(0)

        app = QtWidgets.QApplication([])
        window = QtWidgets.QMainWindow()
        ticks = []

        poll = QtCore.QTimer(window)
        poll.timeout.connect(lambda: ticks.append(1))
        poll.start(100)

        stop = QtCore.QTimer(window)
        stop.setSingleShot(True)
        stop.timeout.connect(app.quit)
        stop.start(600)

        app.exec()
        print(len(ticks))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    if result.returncode != 0:
        pytest.fail(f"subprocess failed:\n{result.stdout}\n{result.stderr}")
    output = result.stdout.strip()
    if output.startswith("SKIP"):
        pytest.skip(output)
    assert int(output) >= 3, (
        f"Python got only {output} slices in 600 ms of exec(); a signal handler "
        "would not run promptly"
    )


def test_background_builds_are_insulated_from_the_consoles_interrupt() -> None:
    """Ctrl+C means "close the app", not "kill my overnight G-matrix build".

    On Windows the console delivers CTRL_C_EVENT to every process attached to it,
    so a build launched without its own process group dies with the app -- despite
    these launchers documenting that they survive a lost session.
    """
    import inspect

    from graph_visualizer import fast_graph_io

    source = inspect.getsource(fast_graph_io)
    assert source.count("creationflags=_detached_creation_flags()") == 3, (
        "a background build launcher is missing its process-group flag"
    )
    assert "CREATE_NEW_PROCESS_GROUP" in inspect.getsource(
        fast_graph_io._detached_creation_flags
    )
