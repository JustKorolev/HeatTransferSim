"""Geometry backend abstraction for the octree voxelizer.

``build_octree`` subdivides space and assembles the graph identically regardless
of the input geometry; only *cell classification* (what material/component
occupies a cell, and how) is geometry-specific. A ``GeometryProvider`` is that
seam:

- ``TriangleGeometryProvider`` (GLB path) wraps the existing triangle
  spatial-index + ``_classify_cell`` logic and supports multiprocessing workers.
- ``BRepGeometryProvider`` (STEP path, ``brep_geometry.py``) classifies cells by
  exact OpenCASCADE point-in-solid tests, so solid interiors *fill* instead of
  shelling. It runs in-process (OCC handles aren't picklable to workers).

The provider returns an ``octree.CellClassification``; the octree consumes only
that, so both backends drive the same subdivision/refinement/graph assembly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle with octree.py
    from .octree import CellClassification


class GeometryProvider(Protocol):
    """Supplies the octree with a root bounding box and per-cell classification."""

    #: True if per-cell classification is safe to run in ProcessPool workers.
    supports_multiprocessing: bool

    @property
    def bounds_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned root bounds ``(min_xyz, max_xyz)`` in millimetres."""
        ...

    def classify_cell(
        self, center_mm: np.ndarray, size_mm: np.ndarray
    ) -> "CellClassification":
        """Classify one axis-aligned cell (center/size in mm)."""
        ...
