"""The voxelizer's innermost test: triangle vs axis-aligned box.

Rewritten from numpy vector ops to scalar arithmetic. It ran np.cross nine times
per triangle, and each of those goes through moveaxis and normalize_axis_tuple --
so nearly all the time went to numpy dispatch rather than the multiplies a
3-vector cross product actually is. It was also the frame a Windows access
violation landed on during a 10.35M triangle build.

The geometry must be unchanged; only the arithmetic is direct.
"""

from __future__ import annotations

import numpy as np

from octree_graph.octree import _triangle_intersects_aabb as hits

UNIT = (np.zeros(3), np.ones(3))


def test_a_triangle_through_the_box_hits() -> None:
    assert hits(np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), *UNIT)


def test_a_triangle_entirely_outside_misses() -> None:
    assert not hits(np.array([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]]), *UNIT)


def test_a_triangle_inside_the_box_hits() -> None:
    assert hits(np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]), *UNIT)


def test_a_triangle_separated_only_by_an_edge_axis_misses() -> None:
    """The case the nine edge-cross axes exist for: it overlaps every coordinate
    projection yet still misses the box."""
    # The plane x + y = 2 cuts the corner off; the box only reaches x + y = 1.
    tri = np.array([[3.0, -1.0, 0.0], [-1.0, 3.0, 0.0], [3.0, 3.0, 0.0]])
    assert not hits(tri, np.zeros(3), np.array([0.5, 0.5, 0.5]))
    # ...and the same triangle DOES hit once the box is big enough to reach it.
    assert hits(tri, np.zeros(3), np.array([3.0, 3.0, 3.0]))


def test_a_degenerate_triangle_does_not_crash() -> None:
    """Zero-area triangles come out of real tessellations; the plane test must be
    skipped rather than dividing by a zero normal."""
    for tri in (
        np.zeros((3, 3)),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ):
        assert isinstance(hits(tri, *UNIT), bool)


def test_a_malformed_triangle_is_rejected_not_raised() -> None:
    assert hits(np.zeros((2, 3)), *UNIT) is False
    assert hits(np.zeros((3, 2)), *UNIT) is False


def test_the_box_is_respected_in_every_axis_independently() -> None:
    """A box that is thin in one axis must reject a triangle offset along it."""
    tri = np.array([[-1.0, -1.0, 0.9], [1.0, -1.0, 0.9], [0.0, 1.0, 0.9]])
    assert hits(tri, np.zeros(3), np.ones(3))
    assert not hits(tri, np.zeros(3), np.array([1.0, 1.0, 0.5]))


def test_translation_invariance() -> None:
    """Moving triangle and box together cannot change the answer -- the box centre
    is subtracted by hand now, so this guards that arithmetic."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        tri = rng.normal(0.0, 2.0, (3, 3))
        shift = rng.normal(0.0, 50.0, 3)
        assert hits(tri, np.zeros(3), np.ones(3)) == hits(tri + shift, shift, np.ones(3))
