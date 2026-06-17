"""Unit tests for the two-cube thermal RC model."""

from __future__ import annotations

import unittest

import numpy as np

from heat_transfer_sim.cube import Cube, zero_heater
from heat_transfer_sim.simulation import TwoCubeSimulation, make_pulsed_heater


def make_simulation(
    temp_1: float = 40.0,
    temp_2: float = 20.0,
    heater_1=zero_heater,
    heater_2=zero_heater,
) -> TwoCubeSimulation:
    cube_1 = Cube(
        name="Cube 1",
        side_length=0.1,
        mass=0.25,
        specific_heat=900.0,
        thermal_conductivity=205.0,
        temperature=temp_1,
        position=(-0.05, 0.0, 0.0),
        heater_power=heater_1,
    )
    cube_2 = Cube(
        name="Cube 2",
        side_length=0.1,
        mass=0.25,
        specific_heat=900.0,
        thermal_conductivity=205.0,
        temperature=temp_2,
        position=(0.05, 0.0, 0.0),
        heater_power=heater_2,
    )
    return TwoCubeSimulation(
        cube_1=cube_1,
        cube_2=cube_2,
        interface_resistance=0.0,
    )


class SimulationTests(unittest.TestCase):
    def test_heat_flow_sign_is_positive_from_hot_cube_to_cold_cube(self) -> None:
        simulation = make_simulation(temp_1=40.0, temp_2=20.0)
        result = simulation.solve((0.0, 1.0), num_points=5, update_cubes=False)
        self.assertGreater(result.heat_flow_1_to_2[0], 0.0)

    def test_contact_area_comes_from_touching_face_overlap(self) -> None:
        simulation = make_simulation(temp_1=40.0, temp_2=20.0)
        self.assertAlmostEqual(simulation.effective_contact_area, 0.01)

    def test_contact_resistance_uses_half_cube_terms_and_interface_resistance(self) -> None:
        simulation = make_simulation(temp_1=40.0, temp_2=20.0)
        simulation.interface_resistance = 0.2
        expected = 0.05 / (205.0 * 0.01) + 0.2 + 0.05 / (205.0 * 0.01)
        self.assertAlmostEqual(simulation.contact_resistance, expected)

    def test_cube_position_is_center_coordinate(self) -> None:
        simulation = make_simulation(temp_1=40.0, temp_2=20.0)
        self.assertEqual(
            simulation.cube_1.flat_bounds,
            (-0.1, 0.0, -0.05, 0.05, -0.05, 0.05),
        )

    def test_separated_cubes_need_contact_area_override(self) -> None:
        simulation = make_simulation(temp_1=40.0, temp_2=20.0)
        simulation.cube_2.position = (0.20, 0.0, 0.0)
        with self.assertRaises(ValueError):
            _ = simulation.contact_resistance

    def test_cube_temperature_derivative_uses_other_cube_and_heater(self) -> None:
        simulation = make_simulation(
            temp_1=40.0,
            temp_2=20.0,
            heater_1=make_pulsed_heater(10.0, 0.0, 10.0),
        )
        derivative = simulation.cube_1.temperature_derivative_with(
            simulation.cube_2, simulation.contact_resistance, time=1.0
        )
        expected = (
            -simulation.cube_1.heat_exchange_with(
                simulation.cube_2, simulation.contact_resistance
            )
            + 10.0
        ) / simulation.cube_1.heat_capacity
        self.assertAlmostEqual(derivative, expected)

    def test_energy_is_conserved_without_heaters(self) -> None:
        simulation = make_simulation(temp_1=60.0, temp_2=20.0)
        initial_energy = (
            simulation.cube_1.heat_capacity * simulation.cube_1.temperature
            + simulation.cube_2.heat_capacity * simulation.cube_2.temperature
        )
        result = simulation.solve((0.0, 300.0), num_points=501, update_cubes=False)
        final_energy = (
            simulation.cube_1.heat_capacity * result.temperature_1[-1]
            + simulation.cube_2.heat_capacity * result.temperature_2[-1]
        )
        self.assertAlmostEqual(initial_energy, final_energy, delta=1e-5)

    def test_temperatures_approach_common_equilibrium_without_heaters(self) -> None:
        simulation = make_simulation(temp_1=60.0, temp_2=20.0)
        result = simulation.solve((0.0, 2000.0), num_points=1001, update_cubes=False)
        self.assertAlmostEqual(result.temperature_1[-1], result.temperature_2[-1], delta=1e-3)

    def test_heater_adds_expected_energy(self) -> None:
        heater_power = 10.0
        heater_stop = 100.0
        simulation = make_simulation(
            temp_1=20.0,
            temp_2=20.0,
            heater_1=make_pulsed_heater(heater_power, 0.0, heater_stop),
        )
        initial_energy = (
            simulation.cube_1.heat_capacity * simulation.cube_1.temperature
            + simulation.cube_2.heat_capacity * simulation.cube_2.temperature
        )
        result = simulation.solve((0.0, 200.0), num_points=1001, update_cubes=False)
        final_energy = (
            simulation.cube_1.heat_capacity * result.temperature_1[-1]
            + simulation.cube_2.heat_capacity * result.temperature_2[-1]
        )
        added_energy = np.trapezoid(
            result.heater_power_1 + result.heater_power_2, result.time
        )
        self.assertAlmostEqual(final_energy - initial_energy, added_energy, delta=1.0)


if __name__ == "__main__":
    unittest.main()
