"""Lumped thermal cube model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


HeaterFunction = Callable[[float], float]


def zero_heater(_: float) -> float:
    """Return zero heater power for all time."""
    return 0.0


@dataclass
class Cube:
    """A single lumped thermal mass represented as a cube.

    SI units are used throughout:
    mass [kg], specific heat [J/(kg K)], thermal conductivity [W/(m K)],
    side length [m], temperature [deg C or K], heater power [W].
    """

    name: str
    side_length: float
    mass: float
    specific_heat: float
    thermal_conductivity: float
    temperature: float
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    heater_power: HeaterFunction = field(default=zero_heater, repr=False)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def heat_capacity(self) -> float:
        """Thermal capacitance C = m cp [J/K]."""
        return self.mass * self.specific_heat

    @property
    def face_area(self) -> float:
        """Area of one cube face [m^2]."""
        return self.side_length**2

    @property
    def volume(self) -> float:
        """Cube volume [m^3]."""
        return self.side_length**3

    @property
    def half_side_length(self) -> float:
        """Half of the cube side length [m]."""
        return 0.5 * self.side_length

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Axis-aligned cube bounds as ((xmin, xmax), (ymin, ymax), (zmin, zmax))."""
        half_side = self.half_side_length
        return tuple(
            (center - half_side, center + half_side) for center in self.position
        )

    @property
    def flat_bounds(self) -> tuple[float, float, float, float, float, float]:
        """Axis-aligned cube bounds as (xmin, xmax, ymin, ymax, zmin, zmax)."""
        return (
            self.bounds[0][0],
            self.bounds[0][1],
            self.bounds[1][0],
            self.bounds[1][1],
            self.bounds[2][0],
            self.bounds[2][1],
        )

    def surface_overlap_area_with(self, other: "Cube", tolerance: float = 1e-9) -> float:
        """Return overlapping face area with another axis-aligned cube [m^2].

        Cubes are considered thermally connected when one pair of opposing faces
        is coincident within ``tolerance`` and their projections overlap on the
        two remaining axes.
        """
        self_bounds = self.bounds
        other_bounds = other.bounds

        for normal_axis in range(3):
            self_min, self_max = self_bounds[normal_axis]
            other_min, other_max = other_bounds[normal_axis]
            faces_touch = (
                abs(self_max - other_min) <= tolerance
                or abs(other_max - self_min) <= tolerance
            )
            if not faces_touch:
                continue

            overlap_area = 1.0
            for tangent_axis in range(3):
                if tangent_axis == normal_axis:
                    continue
                tangent_overlap = _interval_overlap(
                    self_bounds[tangent_axis], other_bounds[tangent_axis]
                )
                overlap_area *= tangent_overlap
            if overlap_area > 0.0:
                return overlap_area

        return 0.0

    def heat_exchange_with(self, other: "Cube", contact_resistance: float) -> float:
        """Return heat flow from this cube to another cube [W].

        Positive heat flow means heat leaves this cube and enters ``other``.
        This method is intentionally non-mutating so the ODE solver owns all
        temperature evolution.
        """
        if contact_resistance <= 0.0:
            raise ValueError("contact_resistance must be positive")
        return (self.temperature - other.temperature) / contact_resistance

    def temperature_derivative_with(
        self, other: "Cube", contact_resistance: float, time: float
    ) -> float:
        """Return dT/dt for this cube exchanging heat with ``other``.

        This is useful for simple stepping methods and for future controller
        experiments. The main simulation still uses ``solve_ivp`` for accuracy.
        """
        heat_leaving = self.heat_exchange_with(other, contact_resistance)
        net_power = -heat_leaving + self.heater_power_at(time)
        return net_power / self.heat_capacity

    def update_temperature_euler(
        self, other: "Cube", contact_resistance: float, time: float, time_step: float
    ) -> None:
        """Advance this cube temperature by one explicit Euler step."""
        if time_step <= 0.0:
            raise ValueError("time_step must be positive")
        self.temperature += (
            self.temperature_derivative_with(other, contact_resistance, time) * time_step
        )

    def heater_power_at(self, time: float) -> float:
        """Return this cube's heater power at ``time`` [W]."""
        return float(self.heater_power(time))

    def set_temperature(self, temperature: float) -> None:
        """Update the displayed/current lumped temperature."""
        self.temperature = float(temperature)

    def validate(self) -> None:
        """Validate physical parameters needed by the lumped model."""
        if self.side_length <= 0.0:
            raise ValueError(f"{self.name}: side_length must be positive")
        if self.mass <= 0.0:
            raise ValueError(f"{self.name}: mass must be positive")
        if self.specific_heat <= 0.0:
            raise ValueError(f"{self.name}: specific_heat must be positive")
        if self.thermal_conductivity <= 0.0:
            raise ValueError(f"{self.name}: thermal_conductivity must be positive")
        if len(self.position) != 3:
            raise ValueError(f"{self.name}: position must be a 3D coordinate")


def _interval_overlap(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    """Return overlap length between two closed intervals."""
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
