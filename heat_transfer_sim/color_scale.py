"""Temperature-to-color mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib import colormaps
from matplotlib.colors import to_hex, to_rgb


@dataclass(frozen=True)
class TemperatureColorScale:
    """Map temperatures onto a blue-to-red color spectrum."""

    min_temperature: float
    max_temperature: float
    cmap_name: str = "coolwarm"

    @classmethod
    def from_temperatures(cls, temperatures: list[float] | tuple[float, ...]):
        return cls(min(temperatures), max(temperatures))

    def normalize(self, temperature: float) -> float:
        """Return a normalized color position in [0, 1]."""
        if self.max_temperature <= self.min_temperature:
            return 0.5
        normalized = (temperature - self.min_temperature) / (
            self.max_temperature - self.min_temperature
        )
        return max(0.0, min(1.0, normalized))

    def hex_color(self, temperature: float) -> str:
        """Return a hex color for ``temperature``."""
        color = colormaps[self.cmap_name](self.normalize(temperature))
        return to_hex(color)

    def rgb_color(self, temperature: float) -> tuple[float, float, float]:
        """Return an RGB color tuple with values in [0, 1]."""
        return to_rgb(self.hex_color(temperature))
