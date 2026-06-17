"""Two-cube thermal RC simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .cube import Cube


HeaterFunction = Callable[[float], float]


def make_pulsed_heater(power: float, start_time: float, stop_time: float) -> HeaterFunction:
    """Create a constant-power heater active on [start_time, stop_time]."""

    def heater(time: float) -> float:
        return float(power) if start_time <= time <= stop_time else 0.0

    return heater


@dataclass(frozen=True)
class SimulationResult:
    """Solved time history and derived thermal quantities."""

    time: np.ndarray
    temperature_1: np.ndarray
    temperature_2: np.ndarray
    temperature_difference: np.ndarray
    heat_flow_1_to_2: np.ndarray
    heater_power_1: np.ndarray
    heater_power_2: np.ndarray
    contact_resistance: float

    @property
    def final_temperature_1(self) -> float:
        return float(self.temperature_1[-1])

    @property
    def final_temperature_2(self) -> float:
        return float(self.temperature_2[-1])


@dataclass
class TwoCubeSimulation:
    """Coupled ODE simulation for two lumped thermal cubes."""

    cube_1: Cube
    cube_2: Cube
    interface_resistance: float = 0.0
    contact_area: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def effective_contact_area(self) -> float:
        """Contact area [m^2], defaulting to face overlap from cube coordinates."""
        if self.contact_area is not None:
            return self.contact_area
        area = self.cube_1.surface_overlap_area_with(self.cube_2)
        if area <= 0.0:
            raise ValueError(
                "Cubes do not have overlapping touching faces. Set cube coordinates "
                "so their faces touch, or provide a positive contact_area override."
            )
        return area

    @property
    def contact_resistance(self) -> float:
        """Thermal resistance from cube 1 center to cube 2 center [K/W].

        R12 = (s1 / 2) / (k1 A) + R_interface + (s2 / 2) / (k2 A)
        """
        area = self.effective_contact_area
        cube_1_half_resistance = self.cube_1.half_side_length / (
            self.cube_1.thermal_conductivity * area
        )
        cube_2_half_resistance = self.cube_2.half_side_length / (
            self.cube_2.thermal_conductivity * area
        )
        return cube_1_half_resistance + self.interface_resistance + cube_2_half_resistance

    def dynamics(self, time: float, temperatures: np.ndarray) -> list[float]:
        """Return [dT1/dt, dT2/dt] for the current state."""
        temperature_1, temperature_2 = temperatures
        qdot_1_to_2 = (temperature_1 - temperature_2) / self.contact_resistance
        heater_1 = self.cube_1.heater_power_at(time)
        heater_2 = self.cube_2.heater_power_at(time)

        dtemperature_1_dt = (-qdot_1_to_2 + heater_1) / self.cube_1.heat_capacity
        dtemperature_2_dt = (qdot_1_to_2 + heater_2) / self.cube_2.heat_capacity
        return [dtemperature_1_dt, dtemperature_2_dt]

    def solve(
        self,
        time_span: tuple[float, float],
        num_points: int = 1001,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        update_cubes: bool = True,
        max_step: float | None = None,
    ) -> SimulationResult:
        """Solve the two-cube ODE over ``time_span``."""
        if num_points < 2:
            raise ValueError("num_points must be at least 2")
        start_time, stop_time = time_span
        if stop_time <= start_time:
            raise ValueError("time_span stop time must be greater than start time")

        evaluation_times = np.linspace(start_time, stop_time, num_points)
        initial_temperatures = np.array(
            [self.cube_1.temperature, self.cube_2.temperature], dtype=float
        )

        solve_options = {
            "t_eval": evaluation_times,
            "rtol": rtol,
            "atol": atol,
        }
        if max_step is not None:
            if max_step <= 0.0:
                raise ValueError("max_step must be positive when provided")
            solve_options["max_step"] = max_step

        solution = solve_ivp(
            self.dynamics,
            time_span,
            initial_temperatures,
            **solve_options,
        )
        if not solution.success:
            raise RuntimeError(f"ODE solve failed: {solution.message}")

        temperature_1 = solution.y[0]
        temperature_2 = solution.y[1]
        temperature_difference = temperature_1 - temperature_2
        heat_flow_1_to_2 = temperature_difference / self.contact_resistance
        heater_power_1 = np.array([self.cube_1.heater_power_at(t) for t in solution.t])
        heater_power_2 = np.array([self.cube_2.heater_power_at(t) for t in solution.t])

        if update_cubes:
            self.cube_1.set_temperature(temperature_1[-1])
            self.cube_2.set_temperature(temperature_2[-1])

        return SimulationResult(
            time=solution.t,
            temperature_1=temperature_1,
            temperature_2=temperature_2,
            temperature_difference=temperature_difference,
            heat_flow_1_to_2=heat_flow_1_to_2,
            heater_power_1=heater_power_1,
            heater_power_2=heater_power_2,
            contact_resistance=self.contact_resistance,
        )

    def step(
        self, time: float, time_step: float, max_solver_step: float | None = None
    ) -> SimulationResult:
        """Advance the current cube temperatures by one ODE time step."""
        if time_step <= 0.0:
            raise ValueError("time_step must be positive")
        return self.solve(
            (time, time + time_step),
            num_points=2,
            update_cubes=True,
            max_step=max_solver_step,
        )

    def validate(self) -> None:
        self.cube_1.validate()
        self.cube_2.validate()
        if self.interface_resistance < 0.0:
            raise ValueError("interface_resistance must be non-negative")
        if self.contact_area is not None and self.contact_area <= 0.0:
            raise ValueError("contact_area must be positive when provided")
