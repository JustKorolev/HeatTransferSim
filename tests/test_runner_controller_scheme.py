"""The headless runner must wire each controller scheme to ITS OWN artifact.

--controller carries whichever artifact the user picked in the headless tab, but
modal LQR takes a modal_controller.npz while MIMO PI takes the sys-id run folder
holding G. The runner used to assume the modal case unconditionally, so a MIMO PI
run put a folder into modal_controller_path, failed the modal load, and spent the
night open-loop while reporting that a controller was selected.
"""

from __future__ import annotations

from dataclasses import replace

from graph_visualizer.simulation_parameters import SimulationParameters
from graph_visualizer.simulation_runner import SimulationRunner


def _resolve(params, controller_path, *, has_controller=True):
    runner = object.__new__(SimulationRunner)
    runner.cfg = type(
        "Cfg", (),
        {"dt_s": 1.0, "t_final_s": 10.0, "gpu_solver_enabled": False,
         "history_limit": 1, "params": params},
    )()
    return runner._resolve_params(has_controller, controller_path)


def test_mimo_pi_artifact_goes_to_the_gain_path_not_the_modal_path(tmp_path) -> None:
    folder = tmp_path / "sys_id_run"
    folder.mkdir()
    resolved = _resolve(
        replace(SimulationParameters(), mimo_controller_scheme="mimo_pi"), str(folder)
    )
    assert resolved.mimo_controller_scheme == "mimo_pi"
    assert resolved.mimo_pi_gain_matrix_path == str(folder)
    assert resolved.modal_controller_path == "", "a G folder is not a modal artifact"
    assert resolved.mimo_controller_enabled is True
    assert resolved.input_mode == "heater_inputs"


def test_modal_artifact_still_goes_to_the_modal_path(tmp_path) -> None:
    npz = tmp_path / "modal_controller.npz"
    npz.write_bytes(b"")
    resolved = _resolve(
        replace(SimulationParameters(), mimo_controller_scheme="modal_lqr"), str(npz)
    )
    assert resolved.mimo_controller_scheme == "modal_lqr"
    assert resolved.modal_controller_path == str(npz)


def test_a_bare_artifact_with_no_scheme_is_read_as_modal(tmp_path) -> None:
    """Back-compat: `run_simulation.py --controller x.npz` with no --sim-params.

    Every artifact predating MIMO PI is a modal one, so an unset scheme has to keep
    meaning modal_lqr or existing scripts would silently stop controlling anything.
    """
    npz = tmp_path / "modal_controller.npz"
    npz.write_bytes(b"")
    resolved = _resolve(None, str(npz))
    assert resolved.mimo_controller_scheme == "modal_lqr"
    assert resolved.modal_controller_path == str(npz)


def test_no_controller_leaves_nothing_regulating() -> None:
    """The removed PID+QP used to be the no-artifact fallback. Now there isn't one,
    so the scheme must say so rather than naming a controller that no longer exists."""
    resolved = _resolve(SimulationParameters(), None, has_controller=False)
    assert resolved.mimo_controller_scheme == "none"
    assert resolved.mimo_controller_enabled is False
