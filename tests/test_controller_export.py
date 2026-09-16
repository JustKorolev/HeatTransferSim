"""The controller export must describe the controller that actually runs.

An export is only worth anything if a reader can rebuild the loop from it and get
the same commands. So these tests do not check that fields are present -- they
check that the exported constants REPRODUCE the app's own control law, and that
the precedence rules (preset over run parameters, per-sensor over global,
per-heater override over rating over global ceiling) survive the trip.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from graph_visualizer.controller_export import (
    ControllerExportError,
    collect_controller_constants,
    export_controller,
    write_controller_header,
)
from graph_visualizer.models import (
    HeaterProperties,
    NodeProperties,
    SensorProperties,
    ThermalGraphModel,
)
from graph_visualizer.simulation_parameters import SimulationParameters
from graph_visualizer.sys_id_artifacts import save_mimo_pi_preset, save_sys_id_gain_matrix


SENSORS = [10, 11]
HEATERS = [20, 21]
# Strongly coupled on purpose: the off-diagonal is what makes the decoupling (and
# therefore G_pinv) the part of the export that matters.
G = np.array([[5.0, 2.0], [1.5, 4.0]])


def _model(*, setpoints=(80.0, 90.0), max_powers=(30.0, 12.0)) -> ThermalGraphModel:
    model = ThermalGraphModel()
    for index, (node_id, setpoint) in enumerate(zip(SENSORS, setpoints)):
        node = NodeProperties(node_id=node_id, coord=(index, 0, 0))
        node.controller_setpoint_K = float(setpoint)
        node.sensor = SensorProperties()
        model.nodes[node_id] = node
    for index, (node_id, power) in enumerate(zip(HEATERS, max_powers)):
        node = NodeProperties(node_id=node_id, coord=(index, 1, 0))
        node.heater = HeaterProperties()
        node.heater.heater_max_power_W = float(power)
        model.nodes[node_id] = node
    return model


def _gain_folder(tmp_path: Path, *, metadata=None) -> Path:
    return save_sys_id_gain_matrix(
        tmp_path, "run", SENSORS, HEATERS, G, metadata=metadata or {}
    )


def _params(folder: Path, **overrides) -> SimulationParameters:
    base = SimulationParameters(
        mimo_pi_gain_matrix_path=str(folder),
        mimo_pi_kp=0.3,
        mimo_pi_ki=1.0e-3,
        dt_s=30.0,
        # 0 means "no ceiling", which keeps the per-node ratings intact.
        mimo_default_heater_max_power_W=0.0,
    )
    return replace(base, **overrides)


# --------------------------------------------------------------------------- #
# The export has to refuse rather than emit something meaningless.
# --------------------------------------------------------------------------- #
def test_no_gain_matrix_is_refused_with_the_reason(tmp_path) -> None:
    """Kp and Ki alone do not define this controller -- the decoupling is in G."""
    with pytest.raises(ControllerExportError, match="gain matrix"):
        collect_controller_constants(_model(), SimulationParameters(), "")


def test_a_matrix_for_another_graph_is_refused(tmp_path) -> None:
    """Node ids the graph does not have mean the constants describe another plant."""
    folder = save_sys_id_gain_matrix(tmp_path, "run", [999, 998], HEATERS, G)
    with pytest.raises(ControllerExportError, match="different graph"):
        collect_controller_constants(_model(), _params(folder))


def test_a_missing_folder_is_refused(tmp_path) -> None:
    with pytest.raises(ControllerExportError, match="does not exist"):
        collect_controller_constants(_model(), _params(tmp_path / "nope"))


# --------------------------------------------------------------------------- #
# Precedence: an export that silently used the wrong gain would be worse than none.
# --------------------------------------------------------------------------- #
def test_a_preset_beside_the_matrix_beats_the_run_parameters(tmp_path) -> None:
    folder = _gain_folder(tmp_path)
    save_mimo_pi_preset(folder, kp=0.75, ki=2.0e-4)
    constants = collect_controller_constants(_model(), _params(folder))
    assert constants.kp == pytest.approx((0.75, 0.75))
    assert constants.ki_per_s == pytest.approx((2.0e-4, 2.0e-4))


def test_a_per_sensor_override_beats_the_preset_global(tmp_path) -> None:
    folder = _gain_folder(tmp_path)
    save_mimo_pi_preset(folder, kp=0.75, ki=2.0e-4, per_sensor={11: {"kp": 0.1, "ki": 9.0e-5}})
    constants = collect_controller_constants(_model(), _params(folder))
    assert constants.kp == pytest.approx((0.75, 0.1))
    assert constants.ki_per_s == pytest.approx((2.0e-4, 9.0e-5))


def test_without_a_preset_the_run_parameters_are_used_and_said_so(tmp_path) -> None:
    constants = collect_controller_constants(_model(), _params(_gain_folder(tmp_path)))
    assert constants.kp == pytest.approx((0.3, 0.3))
    assert any("run parameters" in note for note in constants.notes)


def test_heater_limits_match_what_the_live_loop_would_command(tmp_path) -> None:
    """Resolved through the app's own helper, so a precedence change cannot drift."""
    from graph_visualizer.simulation_model import _controller_heater_max_power

    model = _model(max_powers=(30.0, 12.0))
    params = _params(_gain_folder(tmp_path), mimo_default_heater_max_power_W=20.0)
    constants = collect_controller_constants(model, params)
    expected = [_controller_heater_max_power(model.nodes[h], params) for h in HEATERS]
    assert constants.heater_max_power_W == pytest.approx(tuple(expected))
    # The global is a CEILING, so the 30 W heater is capped at 20 and the 12 W one
    # keeps its lower rating.
    assert constants.heater_max_power_W == pytest.approx((20.0, 12.0))


def test_an_unticked_heater_is_exported_as_disabled_not_dropped(tmp_path) -> None:
    """G's shape is fixed at build time; a disabled column stays and is bounded."""
    params = _params(_gain_folder(tmp_path), enabled_heater_node_ids=(20,))
    constants = collect_controller_constants(_model(), params)
    assert constants.heater_enabled == (True, False)
    assert len(constants.heater_node_ids) == 2, "the column must survive"


def test_setpoints_come_from_the_graph_nodes(tmp_path) -> None:
    constants = collect_controller_constants(
        _model(setpoints=(45.0, 55.0)), _params(_gain_folder(tmp_path))
    )
    assert constants.setpoint_K == pytest.approx((45.0, 55.0))


# --------------------------------------------------------------------------- #
# The point of the whole thing: the constants must reproduce the control law.
# --------------------------------------------------------------------------- #
def _qp(constants, v_cmd):
    """The app's own allocator, driven from the exported constants alone."""
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    maxima = np.array(constants.heater_max_power_W)
    result = allocate_thermal_rate_qp(
        np.array(constants.G_K_per_W),
        np.zeros(len(constants.sensor_node_ids)),
        v_cmd,
        np.ones(len(constants.sensor_node_ids)),
        maxima,
        np.zeros(len(constants.heater_node_ids)),
        constants.lambda_u,
        constants.rho_du,
        absolute_target=True,
        undershoot_weight=constants.undershoot_weight,
        lambda_u_relative=constants.lambda_u_relative,
    )
    return np.asarray(result.u), maxima


def test_the_pseudo_inverse_is_exact_when_the_objective_is_symmetric(tmp_path) -> None:
    """At undershoot_weight 1 and unbounded, clipping G_pinv @ v IS the QP.

    This is the only case where the substitution is exact, and the export must say
    so rather than implying it always holds -- a firmware port built on a false
    claim drives different power than the simulation that validated it.
    """
    params = _params(_gain_folder(tmp_path), mimo_undershoot_weight=1.0)
    constants = collect_controller_constants(_model(), params)
    v_cmd = np.array([6.0, 5.0])  # well inside the bounds, so nothing saturates
    from_qp, maxima = _qp(constants, v_cmd)
    from_export = np.clip(np.array(constants.G_pinv_W_per_K) @ v_cmd, 0.0, maxima)
    assert from_export == pytest.approx(from_qp, abs=1e-9)
    assert constants.pinv_is_exact
    assert any("exactly" in note for note in constants.notes)


def test_an_asymmetric_objective_is_reported_as_approximate_not_exact(tmp_path) -> None:
    """undershoot_weight != 1 makes the QP reweight over two extra passes, which no
    single matrix reproduces. The export must measure that gap, not hide it."""
    params = _params(_gain_folder(tmp_path), mimo_undershoot_weight=4.0)
    constants = collect_controller_constants(_model(), params)
    assert not constants.pinv_is_exact
    assert constants.pinv_max_error_W > 0.0
    assert any("APPROXIMATION" in note for note in constants.notes)
    assert any("undershoot_weight is 4" in note for note in constants.notes)


def test_the_measured_deviation_is_the_real_one(tmp_path) -> None:
    """pinv_max_error_W must come from running the allocator, not from a constant."""
    params = _params(_gain_folder(tmp_path), mimo_undershoot_weight=4.0)
    constants = collect_controller_constants(_model(), params)
    # Reproduce the probe the export uses: no passive reference here, so it is the
    # command that lands every heater mid-range.
    v_cmd = np.array(constants.G_K_per_W) @ (0.5 * np.array(constants.heater_max_power_W))
    from_qp, maxima = _qp(constants, v_cmd)
    from_export = np.clip(np.array(constants.G_pinv_W_per_K) @ v_cmd, 0.0, maxima)
    assert constants.pinv_max_error_W == pytest.approx(
        float(np.max(np.abs(from_qp - from_export))), rel=1e-6
    )


def test_the_header_states_which_case_it_is_in(tmp_path) -> None:
    exact = collect_controller_constants(
        _model(), _params(_gain_folder(tmp_path), mimo_undershoot_weight=1.0)
    )
    text = write_controller_header(exact, tmp_path / "exact.h").read_text(encoding="utf-8")
    assert "reproduces the app's bounded QP EXACTLY" in text

    approx_constants = collect_controller_constants(
        _model(), _params(_gain_folder(tmp_path), mimo_undershoot_weight=4.0)
    )
    text = write_controller_header(approx_constants, tmp_path / "approx.h").read_text("utf-8")
    assert "APPROXIMATES the app's bounded QP" in text


def test_the_pseudo_inverse_decouples_the_plant(tmp_path) -> None:
    """G @ G_pinv ~ I is the whole reason this controller exists."""
    constants = collect_controller_constants(_model(), _params(_gain_folder(tmp_path)))
    identity = np.array(constants.G_K_per_W) @ np.array(constants.G_pinv_W_per_K)
    assert identity == pytest.approx(np.eye(2), abs=1e-3)


def test_modal_damping_is_all_ones_when_no_regularization_is_asked_for(tmp_path) -> None:
    """The live loop skips the damping entirely at lambda_u = 0; so must the export."""
    params = _params(_gain_folder(tmp_path), mimo_lambda_u=0.0)
    constants = collect_controller_constants(_model(), params)
    assert constants.integral_modal_damping == pytest.approx((1.0, 1.0))
    assert any("lambda_u is 0" in note for note in constants.notes)


def test_modal_damping_matches_the_allocators_own_factor(tmp_path) -> None:
    params = _params(_gain_folder(tmp_path), mimo_lambda_u=0.5, mimo_lambda_u_relative=0.0)
    constants = collect_controller_constants(_model(), params)
    sigma = np.array(constants.singular_values)
    expected = sigma**2 / (sigma**2 + constants.lambda_effective)
    assert constants.integral_modal_damping == pytest.approx(tuple(expected))


# --------------------------------------------------------------------------- #
# The written files.
# --------------------------------------------------------------------------- #
def test_export_writes_both_files_and_the_json_is_strict(tmp_path) -> None:
    """Non-finite floats become strings: json.dump would emit bare NaN/Infinity,
    which a strict parser on the firmware side rejects."""
    folder = _gain_folder(tmp_path)
    out = tmp_path / "out"
    json_path, header_path, constants = export_controller(
        _model(), _params(folder), out, graph_name="TESTGRAPH"
    )
    assert json_path.is_file() and header_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["n_sensors"] == 2 and payload["n_heaters"] == 2
    assert payload["provenance"]["graph_name"] == "TESTGRAPH"
    # Strict mode is what a non-Python reader effectively does.
    json.loads(json_path.read_text(encoding="utf-8"), parse_constant=_reject)


def _reject(value):  # pragma: no cover - only runs if the guard regresses
    raise AssertionError(f"non-finite literal {value!r} leaked into the JSON")


def test_the_header_declares_every_array_at_its_real_size(tmp_path) -> None:
    constants = collect_controller_constants(_model(), _params(_gain_folder(tmp_path)))
    text = write_controller_header(constants, tmp_path / "c.h").read_text(encoding="utf-8")
    assert "#define HTS_CTRL_N_SENSORS 2" in text
    assert "#define HTS_CTRL_N_HEATERS 2" in text
    assert "static const float HTS_CTRL_G_K_PER_W[2][2]" in text
    # G_pinv is heaters x sensors -- transposed relative to G, and easy to get wrong.
    assert "static const float HTS_CTRL_G_PINV_W_PER_K[2][2]" in text
    assert "static const int HTS_CTRL_SENSOR_NODE_IDS[2] = { 10, 11 };" in text
    assert text.count("#ifndef HTS_CTRL_CONSTANTS_H") == 1
    assert text.rstrip().endswith("#endif  /* HTS_CTRL_CONSTANTS_H */")


# A C floating constant: digits with a '.' or an exponent, then the 'f' suffix.
# "30f" does NOT match -- an 'f' suffix on an integer constant is a compile error,
# and every round number produced one before _c_float appended the ".0".
_C_FLOAT = re.compile(
    r"""^-?(
          (\d+\.\d*|\.\d+|\d+)[eE][+-]?\d+   # 1e-3, 1.5e+2, .5E3
        | (\d+\.\d*|\.\d+)                    # 30.0, 30., .5
        )f$""",
    re.VERBOSE,
)


def _float_literals(text: str) -> list[str]:
    """Every token sitting inside a `static const float` initializer.

    Trailing /* ... */ comments are stripped first: they carry prose like
    "1.0 is symmetric" and "0 disables", which is not a literal and would make
    this scan fail on text that never reaches the compiler as code.
    """
    tokens: list[str] = []
    for line in text.splitlines():
        stripped = re.sub(r"/\*.*?\*/", " ", line).strip()
        if not (stripped.startswith("static const float") or stripped.startswith("{")):
            continue
        body = stripped.split("=", 1)[1] if "=" in stripped else stripped
        body = body.replace("{", " ").replace("}", " ").replace(",", " ").replace(";", " ")
        for token in body.split():
            if token and (token[0].isdigit() or token[0] in "-."):
                tokens.append(token)
    return tokens


def test_every_float_literal_in_the_header_is_valid_c(tmp_path) -> None:
    """'%.9g' % 30.0 is "30", and "30f" is an invalid suffix on an integer constant.

    There is no C compiler in this environment, so the grammar is checked directly.
    A header that does not compile is the one failure mode that makes the whole
    export worthless, and round numbers -- 30 W heaters, 50 K setpoints, gains of
    0 and 1 -- are exactly what this controller is full of.
    """
    model = _model(setpoints=(50.0, 50.0), max_powers=(30.0, 30.0))
    # dt 30.0, antiwindup 1.0, filter 0.0: all round, all previously invalid.
    params = _params(_gain_folder(tmp_path), dt_s=30.0)
    constants = collect_controller_constants(model, params)
    text = write_controller_header(constants, tmp_path / "c.h").read_text(encoding="utf-8")

    literals = _float_literals(text)
    assert literals, "the scan found no literals, so it is not testing anything"
    bad = [t for t in literals if not _C_FLOAT.match(t)]
    assert not bad, f"invalid C float literals: {sorted(set(bad))[:10]}"
    # The specific regression: the round values must appear WITH a decimal point.
    assert "30.0f" in text and "50.0f" in text
    assert not re.search(r"(?<![\d.])\d+f\b", text), "a bare integer carries an 'f' suffix"


def test_a_custom_prefix_renames_everything_consistently(tmp_path) -> None:
    constants = collect_controller_constants(_model(), _params(_gain_folder(tmp_path)))
    text = write_controller_header(constants, tmp_path / "c.h", prefix="CRYO").read_text("utf-8")
    assert "#define CRYO_N_SENSORS 2" in text
    assert "HTS_CTRL" not in text


def test_a_missing_passive_reference_tells_the_firmware_to_latch_it(tmp_path) -> None:
    """No T_op_K in the metadata means the baseline is unknowable at export time.
    Emitting a zero there would silently bias every command."""
    constants = collect_controller_constants(_model(), _params(_gain_folder(tmp_path)))
    assert constants.passive_reference_K is None
    text = write_controller_header(constants, tmp_path / "c.h").read_text(encoding="utf-8")
    assert "HTS_CTRL_PASSIVE_REFERENCE_UNKNOWN 1" in text
    assert "HTS_CTRL_PASSIVE_REFERENCE_K[" not in text


def test_a_measured_override_wins_over_the_derived_baseline(tmp_path) -> None:
    folder = _gain_folder(tmp_path, metadata={"dc_ground": "cryocooler", "T_op_K": 50.0})
    params = _params(folder, mimo_pi_passive_reference_K=33.2)
    constants = collect_controller_constants(_model(), params)
    assert constants.passive_reference_K == pytest.approx((33.2, 33.2))
    assert any("measured override" in note for note in constants.notes)


def test_provenance_names_the_matrix_the_constants_came_from(tmp_path) -> None:
    folder = _gain_folder(tmp_path)
    constants = collect_controller_constants(_model(), _params(folder), graph_name="CRYOSTAT_V2")
    assert constants.provenance["gain_matrix_path"] == str(folder)
    assert constants.provenance["graph_name"] == "CRYOSTAT_V2"
    assert constants.provenance["controller_scheme"] == "mimo_pi"
    assert constants.provenance["exported_at"]
