"""Compatibility package for the existing heat-transfer simulation UI."""

__all__ = ["ThermalUI"]


def __getattr__(name: str):
    """Lazily expose the old UI without importing plotting dependencies early."""
    if name == "ThermalUI":
        from heat_transfer_sim.ui import ThermalUI

        return ThermalUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
