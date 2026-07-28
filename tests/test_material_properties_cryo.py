"""Tests for temperature-dependent cryogenic material properties (NIST fits)."""

import unittest

import numpy as np

from graph_visualizer import material_properties_cryo as mp


def _cp(name, T, **kw):
    return mp.specific_heat_J_kgK(name, np.asarray(T, dtype=float), **kw)


def _k(name, T, **kw):
    return mp.thermal_conductivity_W_mK(name, np.asarray(T, dtype=float), **kw)


class RoomTemperatureAnchorTests(unittest.TestCase):
    """At 300 K the NIST fits should match the materials.json room-temp constants."""

    def test_specific_heat_300K(self):
        self.assertAlmostEqual(float(_cp("6061-T6 Aluminum", [300.0])[0]), 896.0, delta=80.0)
        self.assertAlmostEqual(float(_cp("Copper", [300.0])[0]), 385.0, delta=40.0)
        self.assertAlmostEqual(float(_cp("AISI 304", [300.0])[0]), 500.0, delta=60.0)

    def test_thermal_conductivity_300K(self):
        self.assertAlmostEqual(float(_k("6061-T6 Aluminum", [300.0])[0]), 167.0, delta=25.0)
        self.assertAlmostEqual(float(_k("Copper", [300.0], rrr=100)[0]), 401.0, delta=40.0)
        self.assertAlmostEqual(float(_k("AISI 304", [300.0])[0]), 16.2, delta=3.0)


class CryogenicCollapseTests(unittest.TestCase):
    def test_specific_heat_collapses_when_cold(self):
        for name in ("6061-T6 Aluminum", "Copper", "AISI 304"):
            cp_cold = float(_cp(name, [4.0])[0])
            cp_warm = float(_cp(name, [300.0])[0])
            self.assertGreater(cp_warm / max(cp_cold, 1e-12), 100.0, f"{name} cp should collapse cold")

    def test_specific_heat_monotonic_in_temperature(self):
        T = np.array([4.0, 20.0, 77.0, 150.0, 300.0])
        for name in ("6061-T6 Aluminum", "Copper", "AISI 304"):
            cp = _cp(name, T)
            self.assertTrue(np.all(np.diff(cp) > 0.0), f"{name} cp should increase with T")

    def test_copper_conductivity_peaks_cold(self):
        # OFHC copper k peaks in the tens-of-kelvin range, well above its 300 K value.
        k_peak = float(_k("Copper", [20.0], rrr=100)[0])
        k_300 = float(_k("Copper", [300.0], rrr=100)[0])
        self.assertGreater(k_peak, k_300)


class RRRTests(unittest.TestCase):
    def test_higher_rrr_gives_higher_cold_conductivity(self):
        k50 = float(_k("Copper", [4.0], rrr=50)[0])
        k100 = float(_k("Copper", [4.0], rrr=100)[0])
        k300 = float(_k("Copper", [4.0], rrr=300)[0])
        self.assertLess(k50, k100)
        self.assertLess(k100, k300)

    def test_unlisted_rrr_snaps_to_nearest(self):
        k_120 = float(_k("Copper", [10.0], rrr=120)[0])
        k_100 = float(_k("Copper", [10.0], rrr=100)[0])
        self.assertEqual(k_120, k_100)

    def test_default_rrr_is_100(self):
        self.assertEqual(mp.DEFAULT_COPPER_RRR, 100)


class ClampingTests(unittest.TestCase):
    def test_below_and_above_range_are_clamped(self):
        # Aluminum cp fit range is 4-300 K.
        self.assertEqual(float(_cp("6061-T6 Aluminum", [0.5])[0]), float(_cp("6061-T6 Aluminum", [4.0])[0]))
        self.assertEqual(float(_cp("6061-T6 Aluminum", [500.0])[0]), float(_cp("6061-T6 Aluminum", [300.0])[0]))

    def test_invar_cp_clamped_above_27K(self):
        # Invar cp fit is only valid to 27 K; above that it must clamp, not diverge.
        self.assertEqual(float(_cp("Invar36", [100.0])[0]), float(_cp("Invar36", [27.0])[0]))
        self.assertTrue(np.isfinite(float(_cp("Invar36", [300.0])[0])))


class RegistryAndFallbackTests(unittest.TestCase):
    def test_proxy_mappings(self):
        self.assertIs(mp.curve_for_material("17-7PH Stainless Steel"), mp.curve_for_material("AISI 304"))
        self.assertIs(mp.curve_for_material("18-8 Stainless Steel"), mp.curve_for_material("AISI 304"))
        self.assertTrue(mp.has_curve("Phenolic"))
        self.assertTrue(mp.has_curve("Delrin 2700 NC010, Low Viscosity Acetal Copolymer (SS)"))
        self.assertTrue(mp.curve_for_material("Phenolic").approximate)

    def test_unknown_material_uses_fallback(self):
        values = _cp("Unobtainium", [50.0, 100.0], fallback_cp=800.0)
        self.assertTrue(np.allclose(values, 800.0))
        values_k = _k("ZERO MATTER", [50.0], fallback_k=1e-9)
        self.assertTrue(np.allclose(values_k, 1e-9))

    def test_unknown_material_without_fallback_raises(self):
        with self.assertRaises(KeyError):
            _cp("Unobtainium", [50.0])

    def test_array_shape_preserved(self):
        T = np.linspace(4.0, 300.0, 17)
        self.assertEqual(_cp("Copper", T).shape, T.shape)
        self.assertEqual(_k("Copper", T).shape, T.shape)


if __name__ == "__main__":
    unittest.main()
