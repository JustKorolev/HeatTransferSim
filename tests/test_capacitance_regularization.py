"""Capacitance regularization + positivity floor for stiff cryogenic graphs.

Degenerate near-zero-capacitance cells blow up the implicit stage-matrix
condition number, so the CG solve returns an inaccurate result that overshoots
temperatures negative. Flooring the capacitance shrinks the spread; the
temperature floor is the final safety net. These cover the pure helper and the
parameter defaults (the end-to-end behavior is validated on CRYOSTAT_V2).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from graph_visualizer.simulation_model import _regularize_capacitance
from graph_visualizer.simulation_parameters import SimulationParameters


def test_regularize_floors_degenerate_capacities() -> None:
    params = replace(
        SimulationParameters(),
        implicit_capacitance_floor_J_K=1.0e-3,
        implicit_capacitance_condition_cap=0.0,  # test the fixed absolute floor alone
    )
    C = np.array([1.0e-12, 5.0e-4, 1.0e-3, 0.04, 100.0], dtype=float)
    out = _regularize_capacitance(C, params)
    # Below the floor -> raised to it; at/above -> untouched.
    assert out[0] == 1.0e-3
    assert out[1] == 1.0e-3
    assert out[2] == 1.0e-3
    assert out[3] == 0.04
    assert out[4] == 100.0
    # The extreme spread (1e14) collapses to a tractable one.
    assert out.max() / out.min() == 100.0 / 1.0e-3


def test_regularize_is_noop_when_floor_disabled() -> None:
    params = replace(
        SimulationParameters(),
        implicit_capacitance_floor_J_K=0.0,
        implicit_capacitance_condition_cap=0.0,
    )
    C = np.array([1.0e-12, 0.04, 100.0], dtype=float)
    out = _regularize_capacitance(C, params)
    assert np.array_equal(out, C)


def test_auto_floor_caps_the_capacitance_ratio() -> None:
    # cond_cap scales to the graph: only the pathological spread is floored.
    params = replace(
        SimulationParameters(),
        implicit_capacitance_floor_J_K=0.0,
        implicit_capacitance_condition_cap=100.0,
    )
    C = np.array([1.0e-3, 0.5, 50.0], dtype=float)  # max 50 -> floor 0.5
    out = _regularize_capacitance(C, params)
    assert out[0] == 0.5  # tiny cell raised to max/100
    assert out[1] == 0.5
    assert out[2] == 50.0
    assert out.max() / out.min() == 100.0


def test_auto_floor_leaves_well_conditioned_graph_untouched() -> None:
    params = replace(
        SimulationParameters(),
        implicit_capacitance_floor_J_K=0.0,
        implicit_capacitance_condition_cap=100.0,
    )
    C = np.array([5.0, 8.0, 10.0], dtype=float)  # spread 2x < 100x -> nothing floored
    out = _regularize_capacitance(C, params)
    assert np.array_equal(out, C)


def test_capacitance_floor_is_off_by_default() -> None:
    """The capacitance floor existed to stop tiny-C cells exploding -- a symptom of
    the zero-Laplacian bug (temperature-dependent properties rebuilt L(T) from an
    empty edge set, so every cell was thermally isolated and any deposited power
    ran away). With conduction restored those cells are coupled and the implicit
    solver handles the stiffness, so the floor only adds heat capacity that does
    not exist. Off by default; still available as an explicit knob."""
    p = SimulationParameters()
    assert p.implicit_capacitance_floor_J_K == 0.0
    assert p.implicit_capacitance_condition_cap == 0.0


def test_temperature_floor_stays_on_as_a_positivity_guard() -> None:
    p = SimulationParameters()
    assert p.implicit_temperature_floor_K > 0.0
    # The temperature floor must sit BELOW any legitimate cryogenic temperature
    # (interior can reach ~4 K) so it only ever catches unphysical negatives.
    assert p.implicit_temperature_floor_K < 4.0
    # The ceiling was added for "hot spots" that were artifacts of the same bug;
    # enabling it now would silently discard real energy.
    assert p.implicit_temperature_ceiling_K == 0.0
