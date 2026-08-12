"""A heater is bonded to its mount, not bolted to it.

Classifying interfaces by component name alone made every heater attachment a
bolted joint at h(50 K) ~ 512 W/m2K. Across a 5 mm face that is ~0.01 W/K, so a
heater's entire output crossed a film resistance producing tens of K of local rise
while almost nothing reached the body -- the attachment hot spots.
"""

from __future__ import annotations

import pytest

from graph_visualizer.matrix_builder import (
    _geometry_conductance,
    _is_attachment_interface,
    _series_contact_conductance,
)


class _Cell:
    def __init__(self, component, *, k=400.0, heater=False, sensor=False):
        self.component_name = component
        self.k_W_mK = k
        self.is_heater = heater
        self.is_sensor = sensor


AREA, DIST, H = 2.5e-5, 5.0e-3, 512.0     # a 5 mm face, bolted h at 50 K


def test_a_heater_attachment_is_bonded_not_bolted() -> None:
    heater, plate = _Cell("HTR_01", heater=True), _Cell("COLD_PLATE")
    assert _is_attachment_interface(heater, plate)
    bonded = _geometry_conductance(heater, plate, AREA, DIST, H)
    bolted = _series_contact_conductance(400.0, 400.0, AREA, DIST, interface_conductance_W_m2K=H)
    assert bonded > bolted
    assert bonded > 10 * bolted, (bonded, bolted)


def test_a_sensor_attachment_is_treated_the_same_way() -> None:
    assert _is_attachment_interface(_Cell("TS_04", sensor=True), _Cell("SHIELD"))


def test_it_works_from_either_side() -> None:
    """Face orientation is arbitrary; the classification must not depend on it."""
    heater, plate = _Cell("HTR_01", heater=True), _Cell("COLD_PLATE")
    assert _is_attachment_interface(heater, plate)
    assert _is_attachment_interface(plate, heater)


def test_structural_joints_keep_their_contact_resistance() -> None:
    """Only the attachment changes -- two ordinary parts still meet at a bolted
    joint, which is the whole point of not raising the global default."""
    a, b = _Cell("BRACKET"), _Cell("COLD_PLATE")
    assert not _is_attachment_interface(a, b)
    joined = _geometry_conductance(a, b, AREA, DIST, H)
    pure = _series_contact_conductance(400.0, 400.0, AREA, DIST, interface_conductance_W_m2K=None)
    assert joined < pure, "a bolted joint must still add film resistance"


def test_two_cells_of_the_same_heater_are_unaffected() -> None:
    """Already covered by the same-component test; this must not double-count."""
    assert not _is_attachment_interface(_Cell("HTR_01", heater=True), _Cell("HTR_01", heater=True))


def test_the_attachment_no_longer_dominates_the_path() -> None:
    """The failure mode in numbers: with the bolted film the interface carried the
    whole resistance, so a heater's rise was set by the joint rather than the metal."""
    bolted = _series_contact_conductance(400.0, 400.0, AREA, DIST, interface_conductance_W_m2K=H)
    bonded = _series_contact_conductance(400.0, 400.0, AREA, DIST, interface_conductance_W_m2K=None)
    rise_bolted, rise_bonded = 8.58 / bolted, 8.58 / bonded
    assert rise_bolted > 100.0, rise_bolted        # hundreds of K at 8.6 W
    # Bonded is metal conduction, not zero resistance -- a few K, not nothing.
    assert rise_bonded < 10.0, rise_bonded
    assert rise_bolted / rise_bonded > 100.0, (rise_bolted, rise_bonded)
