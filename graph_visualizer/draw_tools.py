"""Pure helpers for straight face-normal cell extrusion."""

from __future__ import annotations

from dataclasses import replace
from math import floor
from typing import Iterable

import numpy as np

from .models import HeaterProperties, NodeProperties, SensorProperties


GridCoord = tuple[int, int, int]


def compute_face_normal(center: Iterable[float], picked_point: Iterable[float]) -> GridCoord:
    """Approximate a clicked cube face normal from the largest center offset axis."""
    center_array = np.asarray(tuple(center), dtype=float)
    picked_array = np.asarray(tuple(picked_point), dtype=float)
    offset = picked_array - center_array
    if offset.shape != (3,) or not np.all(np.isfinite(offset)) or np.allclose(offset, 0.0):
        return (1, 0, 0)
    axis = int(np.argmax(np.abs(offset)))
    sign = 1 if offset[axis] >= 0.0 else -1
    normal = [0, 0, 0]
    normal[axis] = sign
    return (normal[0], normal[1], normal[2])


def extrusion_count_from_projected_pixel_drag(
    start_pixel: tuple[int, int] | None,
    current_pixel: tuple[int, int] | None,
    screen_direction: tuple[float, float] | None,
    pixels_per_cell: float = 80.0,
) -> int:
    """Map signed mouse movement along the face-normal screen direction to cells.

    Projection against a picked world ray is the ideal future version. This
    screen-projected fallback avoids raw-distance explosions while keeping cells
    constrained to the clicked face normal.
    """
    if (
        start_pixel is None
        or current_pixel is None
        or screen_direction is None
        or pixels_per_cell <= 0.0
    ):
        return 0
    dx = current_pixel[0] - start_pixel[0]
    dy = current_pixel[1] - start_pixel[1]
    direction = np.asarray(screen_direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9 or not np.isfinite(norm):
        return 0
    direction = direction / norm
    projected = dx * direction[0] + dy * direction[1]
    return max(0, int(floor(projected / pixels_per_cell)))


def preview_coords(
    start_coord: GridCoord,
    normal: GridCoord,
    count: int,
    occupied_coords: set[GridCoord],
) -> list[GridCoord]:
    """Return extruded coordinates until count is reached or collision occurs."""
    coords: list[GridCoord] = []
    count = max(0, int(count))
    for step in range(1, count + 1):
        coord = (
            start_coord[0] + normal[0] * step,
            start_coord[1] + normal[1] * step,
            start_coord[2] + normal[2] * step,
        )
        if coord in occupied_coords:
            break
        coords.append(coord)
    return coords


def next_node_id(existing_node_ids: Iterable[int]) -> int:
    """Return max(existing_node_ids) + 1, or 0 for an empty graph."""
    ids = list(existing_node_ids)
    return max(ids) + 1 if ids else 0


def clone_node_for_extrusion(source: NodeProperties, node_id: int, coord: GridCoord) -> NodeProperties:
    """Copy material/geometry/mass fields and reset heater, sensor, and radiation fields."""
    node = replace(
        source,
        node_id=int(node_id),
        coord=coord,
        Grad_W_K=0.0,
        has_heater=False,
        heater=HeaterProperties(heater_id=int(node_id)),
        has_sensor=False,
        sensor=SensorProperties(sensor_id=int(node_id)),
        has_cryocooler=False,
    )
    if not node.C_manual_override:
        node.recompute_heat_capacity()
    return node
