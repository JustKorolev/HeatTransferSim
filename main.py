"""Run the two-cube thermal RC simulation example or interactive UI."""

from __future__ import annotations

import argparse

from heat_transfer_sim.cube import Cube, zero_heater
from heat_transfer_sim.plotting import plot_simulation_result
from heat_transfer_sim.simulation import TwoCubeSimulation, make_pulsed_heater


def build_example_simulation() -> TwoCubeSimulation:
    """Create a simple two-cube example with a pulsed heater on cube 1."""
    cube_1 = Cube(
        name="Cube 1",
        side_length=0.1,
        mass=0.25,
        specific_heat=900.0,
        thermal_conductivity=205.0,
        temperature=20.0,
        position=(-0.05, 0.0, 0.0),
        heater_power=make_pulsed_heater(power=30.0, start_time=0.0, stop_time=120.0),
    )
    cube_2 = Cube(
        name="Cube 2",
        side_length=0.1,
        mass=0.25,
        specific_heat=900.0,
        thermal_conductivity=205.0,
        temperature=20.0,
        position=(0.05, 0.0, 0.0),
        heater_power=zero_heater,
    )
    return TwoCubeSimulation(
        cube_1=cube_1,
        cube_2=cube_2,
        interface_resistance=0.0,
    )


def run_example(show_plot: bool = True):
    simulation = build_example_simulation()
    result = simulation.solve((0.0, 600.0), num_points=1001)
    plot_simulation_result(result, show=show_plot)
    return result


def run_ui() -> None:
    from heat_transfer_sim.ui import ThermalUI

    ThermalUI().show()


def run_graph_ui() -> None:
    from graph_visualizer.app import GraphVisualizerApp

    GraphVisualizerApp().show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the PyVista/Qt interactive cube viewer.",
    )
    parser.add_argument(
        "--graph-ui",
        action="store_true",
        help="Launch the sparse 3D lumped thermal graph visualizer.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run the example without opening Matplotlib plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.graph_ui:
        run_graph_ui()
    elif args.ui:
        run_ui()
    else:
        run_example(show_plot=not args.no_plot)
