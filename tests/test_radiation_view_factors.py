"""Tests for the ray-traced view-factor / exchange-area module.

Validates the Monte-Carlo view factors against known analytic values and the
gray-diffuse exchange-area extraction against the textbook two-surface result.
"""

from __future__ import annotations

import unittest

import numpy as np

from graph_visualizer.radiation_view_factors import (
    axis_aligned_square_patch,
    compute_view_factors,
    exchange_links_by_group,
    parallel_rectangles_view_factor,
    total_exchange_areas,
)


class ViewFactorTests(unittest.TestCase):
    def test_parallel_squares_match_analytic_view_factor(self) -> None:
        side, gap = 1.0, 1.0
        p1 = axis_aligned_square_patch((0.0, 0.0, 0.0), "z", +1.0, side, group_id=0)
        p2 = axis_aligned_square_patch((0.0, 0.0, gap), "z", -1.0, side, group_id=1)
        F, F_env = compute_view_factors([p1, p2], rays_per_patch=60000, seed=1)
        analytic = parallel_rectangles_view_factor(side, side, gap)  # ~0.1998
        self.assertAlmostEqual(F[0, 1], analytic, delta=0.01)
        self.assertAlmostEqual(F[1, 0], analytic, delta=0.01)

    def test_view_factors_reciprocal_and_closed(self) -> None:
        p1 = axis_aligned_square_patch((0.0, 0.0, 0.0), "z", +1.0, 2.0, group_id=0)
        p2 = axis_aligned_square_patch((0.0, 0.0, 0.5), "z", -1.0, 2.0, group_id=1)
        F, F_env = compute_view_factors([p1, p2], rays_per_patch=40000, seed=2)
        areas = np.array([p1.area, p2.area])
        # Reciprocity A_i F_ij = A_j F_ji (enforced), and row closure sum F + F_env = 1.
        self.assertAlmostEqual(areas[0] * F[0, 1], areas[1] * F[1, 0], places=6)
        self.assertAlmostEqual(F[0].sum() + F_env[0], 1.0, places=6)
        self.assertGreater(F[0, 1], 0.4)  # large closely-spaced plates see most of each other

    def test_occlusion_blocks_view(self) -> None:
        p1 = axis_aligned_square_patch((0.0, 0.0, 0.0), "z", +1.0, 1.0, group_id=0)
        p2 = axis_aligned_square_patch((0.0, 0.0, 2.0), "z", -1.0, 1.0, group_id=1)
        F_open, _ = compute_view_factors([p1, p2], rays_per_patch=40000, seed=3)
        blocker = axis_aligned_square_patch((0.0, 0.0, 1.0), "z", -1.0, 3.0, group_id=2)
        F_blocked, _ = compute_view_factors([p1, p2, blocker], rays_per_patch=40000, seed=3)
        self.assertGreater(F_open[0, 1], 0.05)
        self.assertLess(F_blocked[0, 1], 0.005)  # blocker intercepts essentially everything

    def test_two_surface_gray_exchange_matches_textbook(self) -> None:
        # F=1 between two surfaces (no escape); analytic G12 = A / (1/e1 + 1/e2 - 1).
        F = np.array([[0.0, 1.0], [1.0, 0.0]])
        F_env = np.array([0.0, 0.0])
        for e1, e2, area in [(0.8, 0.8, 1.0), (0.1, 0.9, 2.5), (1.0, 1.0, 3.0)]:
            G, G_env = total_exchange_areas(F, F_env, np.array([e1, e2]), np.array([area, area]))
            expected = area / (1.0 / e1 + 1.0 / e2 - 1.0)
            self.assertAlmostEqual(G[0, 1], expected, places=9)
            self.assertAlmostEqual(G[1, 0], expected, places=9)
            self.assertAlmostEqual(G_env[0], 0.0, places=9)

    def test_nested_cube_enclosure_matches_analytic_concentric_exchange(self) -> None:
        # A convex body inside an enclosure has the textbook gray exchange
        # G12 = A1 / (1/e1 + (A1/A2)(1/e2 - 1)). Build nested cubes (inner faces
        # outward, outer faces inward), ray-trace view factors, solve the gray
        # exchange, and compare the inner->outer total exchange area end to end.
        a, b, e1, e2 = 0.02, 0.06, 0.8, 0.8
        inner = [
            axis_aligned_square_patch(
                tuple((s * a / 2 if ax == i else 0.0) for ax in range(3)), "xyz"[i], s, a, emissivity=e1, group_id=0
            )
            for i in range(3)
            for s in (1.0, -1.0)
        ]
        outer = [
            axis_aligned_square_patch(
                tuple((s * b / 2 if ax == i else 0.0) for ax in range(3)), "xyz"[i], -s, b, emissivity=e2, group_id=1
            )
            for i in range(3)
            for s in (1.0, -1.0)
        ]
        patches = inner + outer
        F, F_env = compute_view_factors(patches, rays_per_patch=40000, seed=1)
        # Inner (convex) faces see only the enclosure: each row sums to ~1.
        self.assertTrue(np.allclose(F[:6].sum(axis=1), 1.0, atol=0.03))
        G, _ = total_exchange_areas(
            F, F_env, np.array([p.emissivity for p in patches]), np.array([p.area for p in patches])
        )
        g12 = float(G[:6, 6:].sum())
        area_inner, area_outer = 6 * a * a, 6 * b * b
        analytic = area_inner / (1.0 / e1 + (area_inner / area_outer) * (1.0 / e2 - 1.0))
        self.assertAlmostEqual(g12 / analytic, 1.0, delta=0.03)

    def test_close_enclosure_removes_the_background_sink(self) -> None:
        # Vacuum enclosure: the escaped fraction is redistributed into surface-to-
        # surface exchange and there is no environment sink.
        F = np.array([[0.0, 0.3], [0.6, 0.0]])
        F_env = np.array([0.7, 0.4])
        areas = np.array([2.0, 1.0])
        eps = np.array([0.9, 0.9])
        G_open, G_env_open = total_exchange_areas(F, F_env, eps, areas, close_enclosure=False)
        G_vac, G_env_vac = total_exchange_areas(F, F_env, eps, areas, close_enclosure=True)
        self.assertGreater(float(np.max(G_env_open)), 0.0)   # sink present when open
        self.assertTrue(np.allclose(G_env_vac, 0.0))         # no sink when enclosed
        self.assertGreater(G_vac[0, 1], G_open[0, 1])        # escape redistributed into coupling

    def test_black_surface_exchange_is_area_times_view_factor(self) -> None:
        F = np.array([[0.0, 0.3], [0.6, 0.0]])
        F_env = np.array([0.7, 0.4])
        areas = np.array([2.0, 1.0])
        # Symmetrize view factors first (reciprocity) so the check is exact.
        s = 0.5 * (areas[:, None] * F + (areas[:, None] * F).T)
        F = s / areas[:, None]
        F_env = 1.0 - F.sum(axis=1)
        G, G_env = total_exchange_areas(F, F_env, np.array([1.0, 1.0]), areas)
        self.assertAlmostEqual(G[0, 1], areas[0] * F[0, 1], places=9)
        self.assertAlmostEqual(G_env[0], areas[0] * F_env[0], places=9)
        self.assertAlmostEqual(G_env[1], areas[1] * F_env[1], places=9)

    def test_exchange_links_aggregate_by_group(self) -> None:
        patches = [
            axis_aligned_square_patch((0.0, 0.0, 0.0), "z", +1.0, 1.0, group_id=5),
            axis_aligned_square_patch((1.0, 0.0, 0.0), "z", +1.0, 1.0, group_id=5),
            axis_aligned_square_patch((0.0, 0.0, 1.0), "z", -1.0, 1.0, group_id=9),
        ]
        exchange = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0.2, 0.3, 0.0]])
        env = np.array([0.1, 0.1, 0.4])
        links, env_by_group = exchange_links_by_group(patches, exchange, env)
        # Two patches in group 5 both couple to group 9: exchange areas sum.
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0][0], 5)
        self.assertEqual(links[0][1], 9)
        self.assertAlmostEqual(links[0][2], 0.5)
        self.assertAlmostEqual(env_by_group[5], 0.2)
        self.assertAlmostEqual(env_by_group[9], 0.4)


if __name__ == "__main__":
    unittest.main()
