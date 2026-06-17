"""Matplotlib plotting helpers for thermal simulation results."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .simulation import SimulationResult


def plot_simulation_result(result: SimulationResult, show: bool = True):
    """Plot temperatures, temperature difference, and heat flow."""
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(result.time, result.temperature_1, label="T1")
    axes[0].plot(result.time, result.temperature_2, label="T2")
    axes[0].set_ylabel("Temperature")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(result.time, result.temperature_difference, color="tab:purple")
    axes[1].set_ylabel("T1 - T2")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result.time, result.heat_flow_1_to_2, color="tab:red")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Qdot 1->2 [W]")
    axes[2].grid(True, alpha=0.3)

    figure.suptitle("Two-Cube Thermal RC Simulation")
    figure.tight_layout()

    if show:
        plt.show()
    return figure, axes
