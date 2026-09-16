"""Everything the MIMO PI loop needs, as constants, in a form you can flash.

The controller that runs in this app is spread across three places: the DC gain
matrix ``G`` in a sys-id run folder, the per-sensor Kp/Ki in the preset saved
beside it, and roughly a dozen loop-shaping fields on
:class:`~.simulation_parameters.SimulationParameters`. Reproducing it elsewhere
-- on the STM32H563, in a bench script, in a report -- meant reading all three
and hoping you had not missed one. Every field below was at some point the thing
someone missed.

:func:`collect_controller_constants` resolves all of it ONCE, applying the same
precedence the live loop applies (preset over run parameters, per-sensor over
global, per-heater override over rated over global ceiling), and hands back a
single frozen object. :func:`export_controller` writes it as:

* ``controller_constants.json`` -- the full record, round-trippable, including
  provenance so a header can be traced back to the run that identified it;
* ``controller_constants.h`` -- a dependency-free C header, every array
  ``static const float``, sized by ``#define``, ready to ``#include``.

The header is deliberately data-only: no control law, no solver. The law is
twelve lines and belongs in firmware where it can be reviewed against the
hardware; what firmware cannot do is re-derive these numbers.

WHAT THE FIRMWARE STILL HAS TO DO
---------------------------------
The app allocates heater power with a bounded least-squares QP
(:func:`~.mimo_controller.allocate_thermal_rate_qp`). An MCU generally will not
carry a QP, so the export also ships ``G_pinv`` -- the Tikhonov-regularized
pseudo-inverse of ``G`` at the configured lambda -- and clipping ``G_pinv @ v``
into ``[0, u_max]`` stands in for it.

That substitution is exact in one case and approximate in two others, and the
export refuses to guess which one you are in: it RUNS the real allocator at
export time against a representative command and records the measured deviation
in ``pinv_max_error_W``. The three cases:

* ``undershoot_weight == 1`` and no heater bounded -- exact to machine precision;
* ``undershoot_weight != 1`` (4.0 is the default) -- the QP reweights
  under-served channels over two extra passes, which no single matrix can
  express. The deviation is small but real;
* any heater bounded -- the QP redistributes the demand onto the unsaturated
  heaters and clipping does not. On this plant 10-11 of 27 heaters sit at 0 W in
  normal operation, so this case is the normal one, not the corner.

Firmware that needs the exact law has everything to reproduce it: ``G``,
``lambda_effective`` and ``undershoot_weight`` are all exported, and the
reweighting is two more weighted solves of the same system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .simulation_parameters import SimulationParameters

#: Bumped when the JSON layout changes in a way a reader must notice.
CONTROLLER_EXPORT_VERSION = 1

JSON_FILENAME = "controller_constants.json"
HEADER_FILENAME = "controller_constants.h"


class ControllerExportError(RuntimeError):
    """The controller cannot be exported, with the reason a user can act on."""


@dataclass(frozen=True)
class ControllerConstants:
    """Every number the MIMO PI loop reads, resolved and aligned.

    Vectors indexed ``[i]`` run over ``sensor_node_ids``; vectors indexed ``[j]``
    run over ``heater_node_ids``. ``G[i][j]`` is the steady temperature rise in
    kelvin at sensor ``i`` per watt into heater ``j``.
    """

    # -- structure ---------------------------------------------------------- #
    sensor_node_ids: tuple[int, ...]
    heater_node_ids: tuple[int, ...]
    G_K_per_W: tuple[tuple[float, ...], ...]
    G_pinv_W_per_K: tuple[tuple[float, ...], ...]

    # -- per-sensor gains and references ------------------------------------ #
    kp: tuple[float, ...]
    ki_per_s: tuple[float, ...]
    setpoint_K: tuple[float, ...]
    passive_reference_K: tuple[float, ...] | None
    sensor_enabled: tuple[bool, ...]

    # -- per-heater limits -------------------------------------------------- #
    heater_max_power_W: tuple[float, ...]
    heater_slew_W_per_s: tuple[float, ...]
    heater_enabled: tuple[bool, ...]

    # -- loop shaping ------------------------------------------------------- #
    dt_s: float
    antiwindup_gain: float
    measurement_filter_s: float
    overshoot_integral_scale: float
    integral_hold_error_K: float
    integral_abs_max: float
    passive_latch_rate_K_per_s: float

    # -- allocator ---------------------------------------------------------- #
    lambda_u: float
    lambda_u_relative: float
    lambda_effective: float
    rho_du: float
    undershoot_weight: float

    # -- precomputed, because an MCU will not run an SVD --------------------- #
    singular_values: tuple[float, ...]
    left_singular: tuple[tuple[float, ...], ...]
    integral_modal_damping: tuple[float, ...]
    cond_G: float

    # -- how far the shipped G_pinv is from the QP it stands in for ---------- #
    # Measured at export time by running the real allocator, not asserted. See the
    # module docstring for the three cases.
    pinv_max_error_W: float = 0.0
    pinv_is_exact: bool = False

    # -- provenance --------------------------------------------------------- #
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def n_sensors(self) -> int:
        return len(self.sensor_node_ids)

    @property
    def n_heaters(self) -> int:
        return len(self.heater_node_ids)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #
def collect_controller_constants(
    model: Any,
    params: SimulationParameters,
    gain_matrix_path: str | Path | None = None,
    *,
    graph_name: str = "",
) -> ControllerConstants:
    """Resolve the live controller into a single flat set of constants.

    ``gain_matrix_path`` defaults to ``params.mimo_pi_gain_matrix_path``, i.e. the
    matrix the app is actually configured to run, so an export taken from the UI
    describes the controller on screen rather than a different one.
    """
    path = str(gain_matrix_path if gain_matrix_path is not None
               else getattr(params, "mimo_pi_gain_matrix_path", "") or "").strip()
    if not path:
        raise ControllerExportError(
            "No controller gain matrix is selected. Run a sys ID (or pick an existing "
            "G matrix) before exporting -- Kp and Ki alone do not define this "
            "controller, because the decoupling lives in G."
        )
    folder = Path(path)
    if not folder.is_dir():
        raise ControllerExportError(f"Gain matrix folder does not exist: {folder}")

    from .sys_id_artifacts import load_mimo_pi_preset, load_sys_id_gain_matrix_data

    try:
        data = load_sys_id_gain_matrix_data(folder)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the user
        raise ControllerExportError(f"Could not read the gain matrix: {exc}") from exc

    sensor_ids = [int(v) for v in data.sensor_ids]
    heater_ids = [int(v) for v in data.heater_ids]
    G = np.asarray(data.G, dtype=float)
    if G.shape != (len(sensor_ids), len(heater_ids)):
        raise ControllerExportError(
            f"G shape {G.shape} does not match its own id lists "
            f"({len(sensor_ids)} sensors x {len(heater_ids)} heaters)."
        )
    if not G.size:
        raise ControllerExportError("The gain matrix is empty; there is nothing to export.")
    if not np.all(np.isfinite(G)):
        raise ControllerExportError("G contains non-finite entries; re-run the sys ID.")

    notes: list[str] = []
    nodes = getattr(model, "nodes", {}) or {}
    missing = [n for n in sensor_ids + heater_ids if n not in nodes]
    if missing:
        raise ControllerExportError(
            f"The gain matrix references {len(missing)} node id(s) absent from this graph. "
            "It was built for a different graph, so its constants would not mean anything here."
        )

    # -- gains: preset first, then per-sensor overrides, then run parameters -- #
    preset = load_mimo_pi_preset(folder) or {}
    kp0 = preset.get("kp")
    ki0 = preset.get("ki")
    if kp0 is None:
        kp0 = float(getattr(params, "mimo_pi_kp", 0.0))
        notes.append("Kp came from the run parameters; the matrix has no saved preset.")
    if ki0 is None:
        ki0 = float(getattr(params, "mimo_pi_ki", 0.0))
        notes.append("Ki came from the run parameters; the matrix has no saved preset.")
    per_sensor = preset.get("per_sensor") or {}
    kp = tuple(float(per_sensor.get(int(s), {}).get("kp", kp0)) for s in sensor_ids)
    ki = tuple(float(per_sensor.get(int(s), {}).get("ki", ki0)) for s in sensor_ids)
    if per_sensor:
        notes.append(f"{len(per_sensor)} sensor(s) carry a per-channel Kp/Ki override.")

    # -- references --------------------------------------------------------- #
    setpoints = tuple(
        float(getattr(nodes[int(s)], "controller_setpoint_K", float("nan"))) for s in sensor_ids
    )
    if any(not np.isfinite(v) for v in setpoints):
        notes.append(
            "Some sensors have no setpoint. Those channels are excluded from the loop at "
            "run time (weight 0), not driven to NaN."
        )
    passive = _passive_reference(params, data.metadata or {}, len(sensor_ids), notes)

    # -- limits ------------------------------------------------------------- #
    from .simulation_model import (
        _controller_heater_max_power,
        _controller_slew_limits,
        _enabled_node_id_set,
        _node_id_enabled,
    )

    enabled_heater_ids = _enabled_node_id_set(getattr(params, "enabled_heater_node_ids", None))
    enabled_sensor_ids = _enabled_node_id_set(getattr(params, "enabled_sensor_node_ids", None))
    heater_enabled = tuple(_node_id_enabled(enabled_heater_ids, int(h)) for h in heater_ids)
    sensor_enabled = tuple(_node_id_enabled(enabled_sensor_ids, int(s)) for s in sensor_ids)
    u_max = tuple(
        max(0.0, float(_controller_heater_max_power(nodes[int(h)], params))) for h in heater_ids
    )
    slew = tuple(float(v) for v in _controller_slew_limits(model, heater_ids, params))
    if not all(heater_enabled):
        notes.append(
            f"{heater_enabled.count(False)} heater(s) are unticked in the enabled-I/O table. "
            "Their columns are kept (G's shape is fixed at build time) and bounded to 0 W."
        )
    if not all(sensor_enabled):
        notes.append(
            f"{sensor_enabled.count(False)} sensor(s) are unticked. Their rows are kept and "
            "given weight 0."
        )

    # -- allocator regularization, and the damping it implies ---------------- #
    left, sigma, _ = np.linalg.svd(G, full_matrices=False)
    lambda_u = max(0.0, float(getattr(params, "mimo_lambda_u", 0.0) or 0.0))
    lambda_rel = max(0.0, float(getattr(params, "mimo_lambda_u_relative", 0.0) or 0.0))
    sigma_max = float(sigma[0]) if sigma.size else 0.0
    lambda_eff = max(lambda_u, lambda_rel * sigma_max**2)
    # The same sigma^2/(sigma^2 + lambda) the allocator applies, so the integral
    # respects the reachability judgement the allocator has already made. With
    # lambda_u == 0 the live code skips this entirely, so the factors are all 1.
    if lambda_u > 0.0 and sigma.size:
        damping = tuple(float(s * s / (s * s + lambda_eff)) for s in sigma)
    else:
        damping = tuple(1.0 for _ in sigma)
        if sigma.size:
            notes.append(
                "lambda_u is 0, so the integral's modal damping is disabled (all factors 1), "
                "matching the live loop."
            )
    sigma_min = float(sigma[-1]) if sigma.size else 0.0
    cond = float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")

    # Tikhonov pseudo-inverse at the SAME lambda. How closely clipping it tracks the
    # real QP is MEASURED below rather than assumed.
    G_pinv = np.linalg.solve(G.T @ G + lambda_eff * np.eye(G.shape[1]), G.T)
    undershoot = float(getattr(params, "mimo_undershoot_weight", 1.0))
    pinv_error, pinv_exact = _measure_pinv_against_qp(
        G, G_pinv, np.array(u_max), params, setpoints, passive
    )
    if pinv_exact:
        notes.append(
            "G_pinv reproduces the app's allocator exactly (undershoot weight 1, nothing "
            f"bounded at the probe command): max deviation {pinv_error:.3g} W."
        )
    else:
        reasons = []
        if undershoot != 1.0:
            reasons.append(
                f"undershoot_weight is {undershoot:g}, so the QP reweights under-served "
                "channels over two extra passes -- no single matrix expresses that"
            )
        reasons.append(
            "and any bounded heater makes the QP redistribute onto the unsaturated ones, "
            "which clipping does not"
        )
        notes.append(
            f"G_pinv is an APPROXIMATION of the app's allocator ({'; '.join(reasons)}). "
            f"Measured max deviation at a representative command: {pinv_error:.3g} W. "
            "G, lambda_effective and undershoot_weight are all exported, so firmware that "
            "needs the exact law can reproduce the reweighting."
        )

    provenance = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "export_version": CONTROLLER_EXPORT_VERSION,
        "controller_scheme": str(getattr(params, "mimo_controller_scheme", "mimo_pi")),
        "graph_name": str(graph_name or ""),
        "gain_matrix_path": str(folder),
        "gain_matrix_name": str(data.name),
        "gain_matrix_created_at": str(data.created_at),
        "gain_matrix_metadata": dict(data.metadata or {}),
        "preset_saved_at": str(preset.get("saved_at", "")),
        "git_commit": _git_commit(),
    }

    return ControllerConstants(
        sensor_node_ids=tuple(sensor_ids),
        heater_node_ids=tuple(heater_ids),
        G_K_per_W=_as_rows(G),
        G_pinv_W_per_K=_as_rows(G_pinv),
        kp=kp,
        ki_per_s=ki,
        setpoint_K=setpoints,
        passive_reference_K=passive,
        sensor_enabled=sensor_enabled,
        heater_max_power_W=u_max,
        heater_slew_W_per_s=slew,
        heater_enabled=heater_enabled,
        dt_s=float(getattr(params, "dt_s", 1.0)),
        antiwindup_gain=float(getattr(params, "mimo_pi_antiwindup_gain", 1.0)),
        measurement_filter_s=float(getattr(params, "mimo_pi_measurement_filter_s", 0.0)),
        overshoot_integral_scale=float(getattr(params, "mimo_pi_overshoot_integral_scale", 1.0)),
        integral_hold_error_K=float(getattr(params, "mimo_pi_integral_hold_error_K", 0.0)),
        integral_abs_max=float(getattr(params, "mimo_integral_abs_max", 1.0e6)),
        passive_latch_rate_K_per_s=float(
            getattr(params, "mimo_pi_passive_latch_rate_K_per_s", 1.0e-4)
        ),
        lambda_u=lambda_u,
        lambda_u_relative=lambda_rel,
        lambda_effective=float(lambda_eff),
        rho_du=float(getattr(params, "mimo_rho_du", 0.0)),
        undershoot_weight=float(getattr(params, "mimo_undershoot_weight", 1.0)),
        singular_values=tuple(float(v) for v in sigma),
        left_singular=_as_rows(left),
        integral_modal_damping=damping,
        cond_G=cond,
        pinv_max_error_W=float(pinv_error),
        pinv_is_exact=bool(pinv_exact),
        provenance=provenance,
        notes=tuple(notes),
    )


def _measure_pinv_against_qp(
    G: np.ndarray,
    G_pinv: np.ndarray,
    u_max: np.ndarray,
    params: SimulationParameters,
    setpoints: tuple[float, ...],
    passive: tuple[float, ...] | None,
) -> tuple[float, bool]:
    """Run the real allocator and report how far clipped ``G_pinv @ v`` lands from it.

    The probe command is the plant's own feedforward, ``r - y_passive``, because
    that is what the loop spends nearly all its time asking for -- a synthetic
    command would measure the agreement somewhere the controller never operates.
    When no passive reference exists, fall back to a command that lands mid-range
    on every heater, which is the case most likely to stay unbounded and therefore
    the most favourable honest probe.

    Returns ``(max_abs_error_W, is_exact)``. Never raises: a diagnostic that can
    fail the export it is describing is worse than no diagnostic.
    """
    from .mimo_controller import allocate_thermal_rate_qp

    try:
        n_sensors = G.shape[0]
        if passive is not None and all(np.isfinite(setpoints)):
            v = np.array(setpoints, dtype=float) - np.array(passive, dtype=float)
        else:
            v = G @ (0.5 * np.asarray(u_max, dtype=float))
        v = np.where(np.isfinite(v), v, 0.0).reshape(-1)
        if v.shape != (n_sensors,):
            return 0.0, False
        result = allocate_thermal_rate_qp(
            G,
            np.zeros(n_sensors),
            v,
            np.ones(n_sensors),
            np.asarray(u_max, dtype=float),
            np.zeros(G.shape[1]),
            float(getattr(params, "mimo_lambda_u", 0.0) or 0.0),
            float(getattr(params, "mimo_rho_du", 0.0) or 0.0),
            absolute_target=True,
            undershoot_weight=float(getattr(params, "mimo_undershoot_weight", 1.0)),
            lambda_u_relative=float(getattr(params, "mimo_lambda_u_relative", 0.0) or 0.0),
        )
        from_pinv = np.clip(G_pinv @ v, 0.0, np.asarray(u_max, dtype=float))
        error = float(np.max(np.abs(np.asarray(result.u, dtype=float) - from_pinv)))
        return error, bool(error <= 1.0e-9)
    except Exception:  # noqa: BLE001 - a diagnostic must never break the export
        return float("nan"), False


def _passive_reference(
    params: SimulationParameters,
    metadata: dict,
    n_sensors: int,
    notes: list[str],
) -> tuple[float, ...] | None:
    """(r - y_passive)'s baseline, by the same precedence the live loop uses.

    A measured override wins outright; otherwise it is derived from the matrix's
    operating point. ``None`` means neither is available and the firmware has to
    latch it from a quiet plant, exactly as the app does.
    """
    override = float(getattr(params, "mimo_pi_passive_reference_K", 0.0) or 0.0)
    if override > 0.0:
        notes.append(
            f"passive_reference_K is the measured override ({override:g} K), which wins over "
            "anything derived from the cryocooler lift curve."
        )
        return tuple(override for _ in range(n_sensors))
    from .simulation_model import PreparedSimulation

    # A staticmethod: called on the class so the export never has to build a
    # PreparedSimulation just to read a closed-form property of T_op_K.
    solved = PreparedSimulation._mimo_pi_passive_reference(metadata)
    if solved is not None:
        notes.append(
            f"passive_reference_K ({solved:g} K) is derived from the matrix's operating point "
            "T_op_K and the cryocooler lift curve, not measured."
        )
        return tuple(float(solved) for _ in range(n_sensors))
    notes.append(
        "No passive reference is available (radiation-grounded matrix, or one predating "
        "T_op_K). The firmware must latch y_passive from a quiet plant, as the app does: "
        "hold the feedforward at 0 until max |dy/dt| falls below "
        "passive_latch_rate_K_per_s, then capture y - G u_prev ONCE and never refresh it."
    )
    return None


def _as_rows(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in np.atleast_2d(matrix))


def _git_commit() -> str:
    """The commit the export was taken at, or "" outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def controller_constants_as_dict(constants: ControllerConstants) -> dict[str, Any]:
    """JSON-ready, with tuples flattened to lists and inf/NaN made representable."""
    payload = asdict(constants)
    payload["n_sensors"] = constants.n_sensors
    payload["n_heaters"] = constants.n_heaters
    return _jsonable(payload)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        # json.dump would emit bare Infinity/NaN, which strict parsers reject.
        if not np.isfinite(number):
            return str(number)
        return number
    return value


def write_controller_json(constants: ControllerConstants, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(controller_constants_as_dict(constants), handle, indent=2)
        handle.write("\n")
    return path


def write_controller_header(
    constants: ControllerConstants,
    path: Path,
    *,
    prefix: str = "HTS_CTRL",
) -> Path:
    """A dependency-free C header: sizes as #define, everything else const float."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = f"{prefix}_CONSTANTS_H"
    p = constants.provenance
    lines: list[str] = [
        "/*",
        " * MIMO PI thermal controller constants -- GENERATED, do not edit by hand.",
        " *",
        " * Regenerate with:",
        " *     hts-export-controller --graph graphs/<name>",
        " * or with the Heat Transfer Simulation tab's \"Export Controller\" button.",
        " *",
        f" * exported at   : {p.get('exported_at', '')}",
        f" * graph         : {p.get('graph_name', '') or '(unnamed)'}",
        f" * gain matrix   : {p.get('gain_matrix_name', '')} ({p.get('gain_matrix_created_at', '')})",
        f" * source commit : {p.get('git_commit', '') or '(not a git checkout)'}",
        f" * cond(G)       : {constants.cond_G:.6g}",
        " *",
        " * Control law, per step of DT_S:",
        " *",
        " *     y   = lowpass(sensor_K, MEASUREMENT_FILTER_S)",
        " *     e   = SETPOINT_K - y                            [K]",
        " *     hold channel i if INTEGRAL_HOLD_ERROR_K > 0 and |e_i| > it",
        " *     integrand_i = e_i * (e_i < 0 ? OVERSHOOT_INTEGRAL_SCALE : 1)",
        " *     integrand   = U * (DAMPING .* (U^T integrand))  (modal damping)",
        " *     I  += integrand * DT_S, clamped to +/- INTEGRAL_ABS_MAX",
        " *     v   = (SETPOINT_K - PASSIVE_REFERENCE_K) + KP.*e + KI.*I   [K]",
        " *     u   = clip(G_PINV * v, 0, HEATER_MAX_POWER_W)   [W]",
        " *     I  += ANTIWINDUP_GAIN * (G*u - v) * DT_S        (back-calculation)",
        " *",
        (" * G_PINV reproduces the app's bounded QP EXACTLY."
         if constants.pinv_is_exact else
         " * G_PINV APPROXIMATES the app's bounded QP. Measured max deviation at a"),
        (" * (measured deviation: %.3g W)" % constants.pinv_max_error_W
         if constants.pinv_is_exact else
         " * representative command: %.3g W. See the notes at the end of this file."
         % constants.pinv_max_error_W),
        " *",
        " * Row index i runs over SENSOR_NODE_IDS, column index j over",
        " * HEATER_NODE_IDS. G[i][j] is kelvin at sensor i per watt into heater j.",
        " */",
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f"#define {prefix}_N_SENSORS {constants.n_sensors}",
        f"#define {prefix}_N_HEATERS {constants.n_heaters}",
        f"#define {prefix}_EXPORT_VERSION {CONTROLLER_EXPORT_VERSION}",
        "",
        "/* --- loop shaping ----------------------------------------------------- */",
    ]
    for name, value, unit in (
        ("DT_S", constants.dt_s, "s, control period"),
        ("ANTIWINDUP_GAIN", constants.antiwindup_gain, "back-calculation kt, dimensionless"),
        ("MEASUREMENT_FILTER_S", constants.measurement_filter_s, "s, 0 disables"),
        ("OVERSHOOT_INTEGRAL_SCALE", constants.overshoot_integral_scale, "1.0 is symmetric"),
        ("INTEGRAL_HOLD_ERROR_K", constants.integral_hold_error_K, "K, 0 disables the gate"),
        ("INTEGRAL_ABS_MAX", constants.integral_abs_max, "K*s, integral clamp"),
        ("PASSIVE_LATCH_RATE_K_PER_S", constants.passive_latch_rate_K_per_s, "K/s quiescence"),
        ("LAMBDA_EFFECTIVE", constants.lambda_effective, "regularization used by G_PINV"),
        ("UNDERSHOOT_WEIGHT", constants.undershoot_weight, "allocator asymmetry"),
    ):
        lines.append(f"static const float {prefix}_{name} = {_c_float(value)};  /* {unit} */")

    lines += ["", "/* --- node ids (for telemetry and wiring checks) ------------------------ */"]
    lines.append(
        _c_array("int", f"{prefix}_SENSOR_NODE_IDS", constants.n_sensors, constants.sensor_node_ids)
    )
    lines.append(
        _c_array("int", f"{prefix}_HEATER_NODE_IDS", constants.n_heaters, constants.heater_node_ids)
    )

    lines += ["", "/* --- per-sensor ------------------------------------------------------- */"]
    lines.append(_c_array("float", f"{prefix}_KP", constants.n_sensors, constants.kp))
    lines.append(_c_array("float", f"{prefix}_KI_PER_S", constants.n_sensors, constants.ki_per_s))
    lines.append(
        _c_array("float", f"{prefix}_SETPOINT_K", constants.n_sensors, constants.setpoint_K)
    )
    if constants.passive_reference_K is None:
        lines += [
            f"/* PASSIVE_REFERENCE_K is not exported: no value was available at export time.",
            f" * Latch it on target -- hold the feedforward at 0 until max |dy/dt| drops",
            f" * below PASSIVE_LATCH_RATE_K_PER_S, then capture (y - G*u_prev) ONCE. */",
            f"#define {prefix}_PASSIVE_REFERENCE_UNKNOWN 1",
        ]
    else:
        lines.append(
            _c_array(
                "float",
                f"{prefix}_PASSIVE_REFERENCE_K",
                constants.n_sensors,
                constants.passive_reference_K,
            )
        )
    lines.append(
        _c_array("int", f"{prefix}_SENSOR_ENABLED", constants.n_sensors,
                 tuple(int(v) for v in constants.sensor_enabled))
    )

    lines += ["", "/* --- per-heater ------------------------------------------------------- */"]
    lines.append(
        _c_array("float", f"{prefix}_HEATER_MAX_POWER_W", constants.n_heaters,
                 constants.heater_max_power_W)
    )
    lines.append(
        _c_array("float", f"{prefix}_HEATER_SLEW_W_PER_S", constants.n_heaters,
                 constants.heater_slew_W_per_s)
    )
    lines.append(
        _c_array("int", f"{prefix}_HEATER_ENABLED", constants.n_heaters,
                 tuple(int(v) for v in constants.heater_enabled))
    )

    lines += ["", "/* --- matrices --------------------------------------------------------- */"]
    lines.append(
        _c_matrix(f"{prefix}_G_K_PER_W", constants.n_sensors, constants.n_heaters,
                  constants.G_K_per_W)
    )
    lines.append(
        _c_matrix(f"{prefix}_G_PINV_W_PER_K", constants.n_heaters, constants.n_sensors,
                  constants.G_pinv_W_per_K)
    )
    lines += ["", "/* --- integral modal damping (U and sigma^2/(sigma^2+lambda)) ---------- */"]
    lines.append(
        _c_matrix(f"{prefix}_LEFT_SINGULAR", constants.n_sensors,
                  len(constants.singular_values), constants.left_singular)
    )
    lines.append(
        _c_array("float", f"{prefix}_INTEGRAL_MODAL_DAMPING",
                 len(constants.integral_modal_damping), constants.integral_modal_damping)
    )

    if constants.notes:
        lines += ["", "/* --- notes from the export -------------------------------------------- */", "/*"]
        for note in constants.notes:
            for chunk in _wrap(note, 74):
                lines.append(f" * {chunk}")
            lines.append(" *")
        lines.append(" */")

    lines += ["", f"#endif  /* {guard} */", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _c_float(value: float) -> str:
    """A valid C float literal.

    The 'f' suffix is only legal on a FLOATING constant, so it needs a '.' or an
    exponent somewhere: 30.0f and 3e1f are fine, but "%.9g" % 30.0 gives "30" and
    "30f" is an invalid suffix on an integer constant -- a header full of them does
    not compile. Every round number hit this.
    """
    number = float(value)
    if np.isnan(number):
        return "(0.0f / 0.0f)"
    if np.isinf(number):
        return "(1.0f / 0.0f)" if number > 0 else "(-1.0f / 0.0f)"
    text = f"{number:.9g}"
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return f"{text}f"


def _c_array(ctype: str, name: str, size: int, values: Any) -> str:
    if ctype == "float":
        rendered = ", ".join(_c_float(v) for v in values)
    else:
        rendered = ", ".join(str(int(v)) for v in values)
    return f"static const {ctype} {name}[{size}] = {{ {rendered} }};"


def _c_matrix(name: str, rows: int, cols: int, values: Any) -> str:
    body = ",\n".join(
        "    { " + ", ".join(_c_float(v) for v in row) + " }" for row in values
    )
    return f"static const float {name}[{rows}][{cols}] = {{\n{body}\n}};"


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def export_controller(
    model: Any,
    params: SimulationParameters,
    output_dir: str | Path,
    *,
    gain_matrix_path: str | Path | None = None,
    graph_name: str = "",
    prefix: str = "HTS_CTRL",
) -> tuple[Path, Path, ControllerConstants]:
    """Collect and write both files. Returns ``(json_path, header_path, constants)``."""
    constants = collect_controller_constants(
        model, params, gain_matrix_path, graph_name=graph_name
    )
    target = Path(output_dir)
    json_path = write_controller_json(constants, target / JSON_FILENAME)
    header_path = write_controller_header(constants, target / HEADER_FILENAME, prefix=prefix)
    return json_path, header_path, constants
