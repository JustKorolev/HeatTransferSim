"""Custom thermal temperature color mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


THERMAL_COLOR_STOPS: tuple[tuple[float, str, str], ...] = (
    (0.00, "#d9d9d9", "Absolute Zero"),
    (0.12, "#4b00b5", "Cold"),
    (0.25, "#0000cc", "Chilled"),
    (0.42, "#00a52a", "Temperate"),
    (0.60, "#d6d100", "Warm"),
    (0.75, "#c46a00", "Hot"),
    (0.90, "#b00020", "Scorching"),
    (1.00, "#5a001f", "Molten"),
)


@dataclass(frozen=True)
class ThermalColormap:
    """Map absolute temperatures in Kelvin to a custom jet-like thermal palette."""

    min_K: float = 0.0
    max_K: float = 400.0

    def normalized(self, temperature_K: float) -> float:
        if self.max_K <= self.min_K:
            return 0.5
        return float(np.clip((float(temperature_K) - self.min_K) / (self.max_K - self.min_K), 0.0, 1.0))

    def hex_color(self, temperature_K: float) -> str:
        t = self.normalized(temperature_K)
        previous = THERMAL_COLOR_STOPS[0]
        for stop in THERMAL_COLOR_STOPS[1:]:
            if t <= stop[0]:
                span = max(stop[0] - previous[0], 1.0e-12)
                local = (t - previous[0]) / span
                rgb = (1.0 - local) * _hex_to_rgb(previous[1]) + local * _hex_to_rgb(stop[1])
                return _rgb_to_hex(rgb)
            previous = stop
        return THERMAL_COLOR_STOPS[-1][1]

    def rgb_color(self, temperature_K: float) -> tuple[float, float, float]:
        rgb = _hex_to_rgb(self.hex_color(temperature_K))
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _hex_to_rgb(color: str) -> np.ndarray:
    cleaned = color.lstrip("#")
    return np.array(
        [int(cleaned[index : index + 2], 16) / 255.0 for index in (0, 2, 4)],
        dtype=float,
    )


def _rgb_to_hex(rgb: np.ndarray) -> str:
    values = np.clip(np.rint(np.asarray(rgb, dtype=float) * 255.0), 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(int(values[0]), int(values[1]), int(values[2]))
