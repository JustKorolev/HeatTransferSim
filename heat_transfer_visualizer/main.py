"""Entry point for the existing two-cube heat-transfer visualizer."""

from __future__ import annotations

from heat_transfer_sim.ui import ThermalUI


def main() -> None:
    ThermalUI().show()


if __name__ == "__main__":
    main()
