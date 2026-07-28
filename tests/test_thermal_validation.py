"""Tests for built-in thermal validation experiments."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from graph_visualizer.thermal_validation import (
    CRYO_REGIME,
    CRYOCOOLER_LIFT,
    DISTRIBUTED_ROD,
    ENERGY_CONSERVATION,
    GEOMETRY_CONTACT_PAIR,
    INSULATED_BLOCK,
    ONE_D_PRISM,
    RADIATION_COOLING,
    RADIATIVE_COUPLING,
    SANDIA_THERMAL_CHALLENGE,
    TDEP_CONDUCTION,
    TDEP_HEATING,
    TWO_NODE_LUMPED,
    TWO_BLOCK_EXCHANGE,
    experiments_by_name,
    prism_dirichlet_insulated_solution,
    sandia_challenge_flux_solution,
)


def _converged_solver(params):
    """Tighten the exposed solver knobs so the sim converges to the analytical
    reference. The validation tab defaults to the real-sim (looser) settings so
    it surfaces the live solver's accuracy; these correctness tests instead
    confirm the sim CAN match the reference when well resolved."""
    params.solver_adaptive_max_substeps = 512
    params.solver_adaptive_target_delta_K = 1.0e-3
    params.solver_rtol = 1.0e-11
    params.gpu_solver_enabled = False
    return params


class ThermalValidationAnalyticalTests(unittest.TestCase):
    def test_insulated_block_capacitance_and_linear_slope(self) -> None:
        experiment = experiments_by_name()[INSULATED_BLOCK]
        params = experiment.default_parameters()
        params.use_octree_pipeline = False
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))
            body_ids = [
                node_id
                for node_id, node in build.model.nodes.items()
                if node.component_name.startswith("VALIDATION_BLOCK")
            ]
            capacitance = sum(build.model.nodes[node_id].C_J_K for node_id in body_ids)
            material = build.model.material_library[params.material]
            expected_volume = params.length_mm * params.width_mm * params.height_mm * 1.0e-9
            expected_capacitance = (
                material["rho_kg_m3"] * material["cp_J_kgK"] * expected_volume
            )
            self.assertAlmostEqual(build.imported_volume_m3, expected_volume)
            self.assertAlmostEqual(capacitance, expected_capacitance)
            values = experiment.analytical_solution(np.array([0.0, 10.0]), params, build)[
                "average_temperature_K"
            ]
            self.assertAlmostEqual(values[0], params.initial_temperature_K)
            self.assertAlmostEqual(
                values[1] - values[0],
                params.heater_power_W * 10.0 / expected_capacitance,
            )

    def test_exact_validation_graph_does_not_warn_about_glb_generation(self) -> None:
        experiment = experiments_by_name()[INSULATED_BLOCK]
        params = experiment.default_parameters()
        params.use_octree_pipeline = False
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))

        self.assertFalse(any("Generated GLB unavailable" in warning for warning in build.warnings))

    def test_two_block_solution_conserves_energy_and_uses_total_interface_g(self) -> None:
        experiment = experiments_by_name()[TWO_BLOCK_EXCHANGE]
        params = experiment.default_parameters()
        params.use_octree_pipeline = False
        params.interface_conductance_W_K = 0.25
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))
            actual_g = sum(
                edge.Gij_W_K
                for edge in build.model.edges.values()
                if {
                    build.model.nodes[edge.source].component_name,
                    build.model.nodes[edge.target].component_name,
                }
                == {"VALIDATION_HOT_BLOCK", "VALIDATION_COLD_BLOCK"}
            )
            self.assertAlmostEqual(actual_g, params.interface_conductance_W_K)
            solution = experiment.analytical_solution(np.array([0.0, 1000.0]), params, build)
            hot = solution["hot_average_temperature_K"]
            cold = solution["cold_average_temperature_K"]
            equilibrium = solution["equilibrium_temperature_K"][0]
            self.assertAlmostEqual(hot[0], params.hot_initial_temperature_K)
            self.assertAlmostEqual(cold[0], params.cold_initial_temperature_K)
            self.assertLess(abs(hot[-1] - equilibrium), abs(hot[0] - equilibrium))
            self.assertLess(abs(cold[-1] - equilibrium), abs(cold[0] - equilibrium))

    def test_geometry_derived_interface_conductance_is_informational(self) -> None:
        experiment = experiments_by_name()[TWO_BLOCK_EXCHANGE]
        params = experiment.default_parameters()
        params.use_octree_pipeline = False
        params.interface_model = "geometry_derived_conductance"
        params.duration_s = params.dt_s
        params.output_sample_interval_s = params.dt_s
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))
            result = experiment.run(build, params)

        metric = next(
            metric for metric in result.metrics if metric.name == "Geometry-derived interface conductance"
        )
        self.assertEqual(metric.status, "PASS")
        self.assertIsNone(metric.tolerance)
        self.assertGreater(metric.value, 0.0)

    def test_two_node_lumped_conductance_experiment_passes_strict_reference(self) -> None:
        experiment = experiments_by_name()[TWO_NODE_LUMPED]
        params = _converged_solver(experiment.default_parameters())
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))
            result = experiment.run(build, params)

        self.assertEqual(result.status, "PASS")
        self.assertEqual(len(build.model.nodes), 2)
        self.assertTrue(all(metric.status == "PASS" for metric in result.metrics))

    def test_geometry_contact_pair_reports_expected_geometry_conductance(self) -> None:
        experiment = experiments_by_name()[GEOMETRY_CONTACT_PAIR]
        params = _converged_solver(experiment.default_parameters())
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))
            result = experiment.run(build, params)

        metric = next(metric for metric in result.metrics if metric.name == "Expected geometry conductance error")
        self.assertEqual(metric.status, "PASS")
        self.assertAlmostEqual(metric.value, 0.0, places=10)
        self.assertEqual(result.status, "PASS")

    def test_distributed_rod_experiment_passes_discrete_mode_reference(self) -> None:
        experiment = experiments_by_name()[DISTRIBUTED_ROD]
        params = _converged_solver(experiment.default_parameters())
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))
            result = experiment.run(build, params)

        self.assertEqual(result.status, "PASS")
        self.assertIn("mode_amplitude_K", result.simulated)
        self.assertTrue(any(key.startswith("node_") for key in result.simulated))

    def test_prism_series_boundary_initial_and_insulated_end(self) -> None:
        length_m = 0.1
        alpha = 1.0e-4
        times = np.array([0.0, 0.05, 0.2])
        fixed = prism_dirichlet_insulated_solution(0.0, times, length_m, alpha, 300.0, 200.0, 200)
        insulated = prism_dirichlet_insulated_solution(length_m, times, length_m, alpha, 300.0, 200.0, 200)
        self.assertAlmostEqual(fixed[0], 300.0)
        self.assertAlmostEqual(fixed[1], 200.0, places=6)
        self.assertTrue(np.all(insulated >= 200.0))
        self.assertTrue(np.all(insulated <= 300.0))
        near_end = prism_dirichlet_insulated_solution(length_m - 1.0e-5, np.array([0.2]), length_m, alpha, 300.0, 200.0, 300)
        at_end = prism_dirichlet_insulated_solution(length_m, np.array([0.2]), length_m, alpha, 300.0, 200.0, 300)
        self.assertLess(abs(float(at_end[0] - near_end[0])), 0.1)

    def test_radiation_cooling_matches_solve_ivp_reference(self) -> None:
        experiment = experiments_by_name()[RADIATION_COOLING]
        params = experiment.default_parameters()
        with TemporaryDirectory() as tmp:
            result = experiment.run(experiment.build(params, Path(tmp)), params)
        self.assertEqual(result.status, "PASS")
        avg = result.simulated["average_temperature_K"]
        self.assertLess(avg[-1], avg[0])  # radiates toward the colder ambient
        self.assertGreater(avg[0] - avg[-1], 1.0)  # meaningful cooling signal

    def test_temperature_dependent_heating_matches_solve_ivp_reference(self) -> None:
        experiment = experiments_by_name()[TDEP_HEATING]
        params = experiment.default_parameters()
        with TemporaryDirectory() as tmp:
            result = experiment.run(experiment.build(params, Path(tmp)), params)
        self.assertEqual(result.status, "PASS")
        avg = result.simulated["average_temperature_K"]
        self.assertGreater(avg[-1], avg[0])  # constant heater warms the body

    def test_cryo_regime_matches_solve_ivp_reference(self) -> None:
        experiment = experiments_by_name()[CRYO_REGIME]
        params = experiment.default_parameters()
        with TemporaryDirectory() as tmp:
            result = experiment.run(experiment.build(params, Path(tmp)), params)
        self.assertEqual(result.status, "PASS")
        self.assertIn("hot_end_temperature_K", result.simulated)
        # localized heater injection => hot end at/above the body average
        self.assertGreaterEqual(
            result.simulated["hot_end_temperature_K"][-1],
            result.simulated["average_temperature_K"][-1],
        )

    def test_temperature_dependent_conduction_matches_kt_steady_profile(self) -> None:
        # Steady conduction with k(T) must match an independent discrete steady-state
        # solve built directly from the k(T) curve (operator-independent), and must
        # be clearly distinct from the wrong constant-k (linear) profile.
        experiment = experiments_by_name()[TDEP_CONDUCTION]
        params = experiment.default_parameters()
        with TemporaryDirectory() as tmp:
            result = experiment.run(experiment.build(params, Path(tmp)), params)
        self.assertEqual(result.status, "PASS")
        sim = np.asarray(result.simulated["steady_state_temperature_K"], dtype=float)
        ref = np.asarray(result.analytical["steady_state_temperature_K"], dtype=float)
        self.assertLess(float(np.max(np.abs(sim - ref))), params.absolute_tolerance_K)
        # The k(T) reference must curve away from the constant-k straight line
        # (otherwise the test could not tell correct k(T) from a constant one).
        linear = np.linspace(ref[0], ref[-1], ref.size)
        self.assertGreater(float(np.max(np.abs(ref - linear))), 15.0)

    def test_cryocooler_lift_curve_reproduces_pt60_capacity(self) -> None:
        # A cold head under each constant load must settle at the PT60 lift-curve
        # temperature (capacity(T*) = load). Monotonic increase with load, tight
        # agreement with the manufacturer curve.
        experiment = experiments_by_name()[CRYOCOOLER_LIFT]
        params = experiment.default_parameters()
        with TemporaryDirectory() as tmp:
            result = experiment.run(experiment.build(params, Path(tmp)), params)
        self.assertEqual(result.status, "PASS")
        sim = np.asarray(result.simulated["cold_head_temperature_K"], dtype=float)
        ref = np.asarray(result.analytical["cold_head_temperature_K"], dtype=float)
        self.assertLess(float(np.max(np.abs(sim - ref))), params.absolute_tolerance_K)
        self.assertTrue(np.all(np.diff(sim) > 0.0))  # more load => warmer cold head

    def test_global_energy_conservation_balances_stored_and_supplied(self) -> None:
        # With conduction + ambient radiation + a heater all active, stored internal
        # energy must track the time integral of net power in to a tiny fraction.
        experiment = experiments_by_name()[ENERGY_CONSERVATION]
        params = experiment.default_parameters()
        with TemporaryDirectory() as tmp:
            result = experiment.run(experiment.build(params, Path(tmp)), params)
        self.assertEqual(result.status, "PASS")
        relative = next(m for m in result.metrics if m.name == "relative energy imbalance")
        self.assertLess(relative.value, 1.0e-3)
        supplied = next(m for m in result.metrics if m.name == "net energy supplied")
        self.assertGreater(supplied.value, 0.0)  # the heater does net positive work

    def test_sandia_challenge_matches_model_and_experimental_data(self) -> None:
        experiment = experiments_by_name()[SANDIA_THERMAL_CHALLENGE]
        params = experiment.default_parameters()
        params.gpu_solver_enabled = False
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))
            result = experiment.run(build, params)
        self.assertEqual(result.status, "PASS")
        # The simulator must reproduce the challenge closed-form model (flux BC).
        model_errors = [
            abs(v) for label, series in result.errors.items() for v in series
        ]
        self.assertLess(max(model_errors), 8.0)
        # And it must land within the experimental scatter at t=1000 s.
        experiment_metrics = [m for m in result.metrics if "vs experiment" in m.name]
        self.assertEqual(len(experiment_metrics), 3)
        self.assertTrue(all(m.status == "PASS" for m in experiment_metrics))

    def test_radiative_coupling_conserves_energy_and_matches_reference(self) -> None:
        experiment = experiments_by_name()[RADIATIVE_COUPLING]
        params = experiment.default_parameters()
        params.gpu_solver_enabled = False
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))
            result = experiment.run(build, params)
        self.assertEqual(result.status, "PASS")
        hot = np.asarray(result.simulated["hot_plate_temperature_K"], dtype=float)
        cold = np.asarray(result.simulated["cold_plate_temperature_K"], dtype=float)
        # Pure surface-to-surface exchange conserves energy between the plates
        # (equal C => T_hot + T_cold constant) and drives them together.
        self.assertLess(abs((hot[-1] + cold[-1]) - (hot[0] + cold[0])), 0.05)
        self.assertLess(hot[-1], hot[0])
        self.assertGreater(cold[-1], cold[0])
        self.assertLess(hot[-1] - cold[-1], hot[0] - cold[0])
        # Must track the independent solve_ivp reference for the coupled T^4 term.
        model_errors = [abs(v) for series in result.errors.values() for v in series]
        self.assertLess(max(model_errors), 0.5)

    def test_sandia_challenge_forces_its_material_ignoring_caller_override(self) -> None:
        # The tab's material selector defaults to Copper; the challenge material
        # is intrinsic to the problem and must win, or the slab goes isothermal
        # and cold (large negative errors vs the experiment).
        experiment = experiments_by_name()[SANDIA_THERMAL_CHALLENGE]
        params = experiment.default_parameters()
        params.gpu_solver_enabled = False
        params.material = "Copper"  # simulate the tab overriding the material
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))
            result = experiment.run(build, params)
        self.assertEqual(params.material, "SandiaChallengeSlab")
        self.assertEqual(result.status, "PASS")
        self.assertTrue(all(m.status == "PASS" for m in result.metrics if "vs experiment" in m.name))

    def test_sandia_reference_never_undershoots_below_initial_even_when_truncated(self) -> None:
        # The truncated Fourier series Gibbs-oscillates near t=0; with few terms
        # and a cryogenic initial temperature it used to dip below 0 K. The exact
        # solution (flux in, insulated back, no losses) is >= T0 everywhere, so
        # the reference must be clamped to that physical floor.
        times = np.linspace(0.0, 1000.0, 400)
        for T0 in (4.0, 20.0, 298.15):
            for terms in (1, 5, 100):
                values = sandia_challenge_flux_solution(
                    0.0095, times, 0.06, 4.2e5, 3000.0, 0.019, T0, terms
                )
                self.assertGreaterEqual(float(np.min(values)), T0 - 1e-9)
        # Converged late-time value must be independent of the term count.
        late = np.array([1000.0])
        a = sandia_challenge_flux_solution(0.0, late, 0.06, 4.2e5, 3000.0, 0.019, 298.15, 20)
        b = sandia_challenge_flux_solution(0.0, late, 0.06, 4.2e5, 3000.0, 0.019, 298.15, 400)
        self.assertLess(abs(float(a[0] - b[0])), 1e-6)

    def test_default_settings_flag_real_error_that_tightening_removes(self) -> None:
        # The tab defaults to the real-sim solver settings (so a run exposes the
        # live solver's accuracy); tightening the exposed knobs must converge.
        experiment = experiments_by_name()[TWO_NODE_LUMPED]
        with TemporaryDirectory() as tmp:
            default_params = experiment.default_parameters()
            default_result = experiment.run(experiment.build(default_params, Path(tmp)), default_params)
            tight_params = _converged_solver(experiment.default_parameters())
            tight_result = experiment.run(experiment.build(tight_params, Path(tmp)), tight_params)
        default_error = max(abs(v) for series in default_result.errors.values() for v in series)
        tight_error = max(abs(v) for series in tight_result.errors.values() for v in series)
        self.assertGreater(default_error, 10.0 * tight_error)

    def test_short_default_runs_produce_finite_temperatures(self) -> None:
        with TemporaryDirectory() as tmp:
            for name, experiment in experiments_by_name().items():
                params = experiment.default_parameters()
                params.use_octree_pipeline = False
                params.duration_s = min(params.duration_s, params.dt_s * 3.0)
                params.output_sample_interval_s = params.dt_s
                build = experiment.build(params, Path(tmp))
                result = experiment.run(build, params)
                self.assertIn(result.status, {"PASS", "WARNING", "FAIL"}, name)
                self.assertGreaterEqual(len(result.times_s), 2, name)
                for series in result.simulated.values():
                    if isinstance(series, list) and series:
                        values = np.asarray(series, dtype=float)
                        self.assertTrue(np.all(np.isfinite(values)), name)


if __name__ == "__main__":
    unittest.main()
