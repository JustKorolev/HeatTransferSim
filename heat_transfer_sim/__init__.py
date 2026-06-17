"""Two-cube thermal RC simulation package."""

__all__ = ["Cube", "SimulationResult", "TwoCubeSimulation", "make_pulsed_heater"]


def __getattr__(name: str):
    """Lazily expose public classes without importing heavy dependencies early."""
    if name == "Cube":
        from .cube import Cube

        return Cube
    if name in {"SimulationResult", "TwoCubeSimulation", "make_pulsed_heater"}:
        from .simulation import SimulationResult, TwoCubeSimulation, make_pulsed_heater

        return {
            "SimulationResult": SimulationResult,
            "TwoCubeSimulation": TwoCubeSimulation,
            "make_pulsed_heater": make_pulsed_heater,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
