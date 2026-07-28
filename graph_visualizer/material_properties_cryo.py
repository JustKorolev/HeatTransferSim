"""Temperature-dependent cryogenic material properties from NIST fits.

Source: NIST Cryogenic Materials Properties Database,
https://trc.nist.gov/cryogenics/materials/materialproperties.htm

Most materials use the NIST log-polynomial form

    log10(y) = a + b*(log10 T) + c*(log10 T)^2 + ... + i*(log10 T)^8
    =>  y = 10 ** poly(log10 T)

with y = k in W/(m.K) or cp in J/(kg.K). Copper thermal conductivity is the
exception: an RRR-dependent rational form in T (not log10 T),

    k = 10 ** [ (a + c*T^.5 + e*T + g*T^1.5 + i*T^2)
                / (1 + b*T^.5 + d*T + f*T^1.5 + h*T^2) ]

Outside a fit's validity range the temperature is CLAMPED to the range endpoint
before evaluation (NIST polynomials diverge wildly if extrapolated). Materials
without a NIST curve fall back to a caller-supplied constant.

Known approximations (see nist-cryo-property-curves memory):
- 18-8 / AISI 304 / 17-7PH stainless all use the 304 fit.
- Phenolic and Delrin/acetal use the G-10 epoxy fit (polymer proxy, flagged).
- Invar specific heat is only fitted 4-27 K; it is clamped above 27 K and
  flagged approximate there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

DEFAULT_COPPER_RRR = 100


def _log_poly(coeffs: list[float], t_min: float, t_max: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return f(T) = 10 ** sum(coeffs[n] * (log10 T_clamped)^n)."""
    poly = np.asarray(coeffs, dtype=float)

    def evaluate(temperatures_K: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(temperatures_K, dtype=float), t_min, t_max)
        x = np.log10(t)
        exponent = np.zeros_like(x)
        for power, coefficient in enumerate(poly):
            if coefficient != 0.0:
                exponent = exponent + coefficient * (x ** power)
        return np.power(10.0, exponent)

    return evaluate


def _copper_k(coeffs: dict[str, float], t_min: float = 4.0, t_max: float = 300.0) -> Callable[[np.ndarray], np.ndarray]:
    """Return the NIST OFHC-copper rational thermal-conductivity evaluator."""
    a, b, c, d, e, f, g, h, i = (coeffs[k] for k in "abcdefghi")

    def evaluate(temperatures_K: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(temperatures_K, dtype=float), t_min, t_max)
        r = np.sqrt(t)
        numerator = a + c * r + e * t + g * t * r + i * t * t
        denominator = 1.0 + b * r + d * t + f * t * r + h * t * t
        return np.power(10.0, numerator / denominator)

    return evaluate


@dataclass(frozen=True)
class CryoCurveSet:
    """cp(T) and k(T) evaluators for one material (k may be RRR-dependent)."""

    name: str
    cp: Callable[[np.ndarray], np.ndarray] | None
    k: Callable[[np.ndarray], np.ndarray] | None = None
    k_by_rrr: dict[int, Callable[[np.ndarray], np.ndarray]] | None = None
    approximate: bool = False

    def thermal_conductivity(self, temperatures_K: np.ndarray, rrr: int = DEFAULT_COPPER_RRR) -> np.ndarray | None:
        if self.k_by_rrr:
            if rrr not in self.k_by_rrr:
                rrr = min(self.k_by_rrr, key=lambda available: abs(available - int(rrr)))
            return self.k_by_rrr[rrr](temperatures_K)
        return self.k(temperatures_K) if self.k is not None else None

    def specific_heat(self, temperatures_K: np.ndarray) -> np.ndarray | None:
        return self.cp(temperatures_K) if self.cp is not None else None


# --------------------------------------------------------------------------- #
# NIST coefficient tables (verbatim from the per-material pages)
# --------------------------------------------------------------------------- #
_AL_6061 = CryoCurveSet(
    name="6061-T6 Aluminum",
    k=_log_poly([0.07918, 1.0957, -0.07277, 0.08084, 0.02803, -0.09464, 0.04179, -0.00571, 0.0], 1.0, 300.0),
    cp=_log_poly([46.6467, -314.292, 866.662, -1298.3, 1162.27, -637.795, 210.351, -38.3094, 2.96344], 4.0, 300.0),
)

_SS_304 = CryoCurveSet(
    name="304 Stainless Steel",
    k=_log_poly([-1.4087, 1.3982, 0.2543, -0.6260, 0.2334, 0.4256, -0.4658, 0.1650, -0.0199], 1.0, 300.0),
    cp=_log_poly([22.0061, -127.5528, 303.647, -381.0098, 274.0328, -112.9212, 24.7593, -2.239153, 0.0], 4.0, 300.0),
)

_COPPER_K_COEFFS = {
    50: dict(a=1.8743, b=-0.41538, c=-0.6018, d=0.13294, e=0.26426, f=-0.0219, g=-0.051276, h=0.0014871, i=0.003723),
    100: dict(a=2.2154, b=-0.47461, c=-0.88068, d=0.13871, e=0.29505, f=-0.02043, g=-0.04831, h=0.001281, i=0.003207),
    150: dict(a=2.3797, b=-0.4918, c=-0.98615, d=0.13942, e=0.30475, f=-0.019713, g=-0.046897, h=0.0011969, i=0.0029988),
    300: dict(a=1.357, b=0.3981, c=2.669, d=-0.1346, e=-0.6683, f=0.01342, g=0.05773, h=0.0002147, i=0.0),
    500: dict(a=2.8075, b=-0.54074, c=-1.2777, d=0.15362, e=0.36444, f=-0.02105, g=-0.051727, h=0.0012226, i=0.0030964),
}
_COPPER = CryoCurveSet(
    name="OFHC Copper",
    cp=_log_poly([-1.91844, -0.15973, 8.61013, -18.996, 21.9661, -12.7328, 3.54322, -0.3797, 0.0], 4.0, 300.0),
    k_by_rrr={rrr: _copper_k(coeffs) for rrr, coeffs in _COPPER_K_COEFFS.items()},
)

_INVAR = CryoCurveSet(
    name="Invar (Fe-36Ni)",
    k=_log_poly([-2.7064, 8.5191, -15.923, 18.276, -11.9116, 4.40318, -0.86018, 0.068508, 0.0], 4.0, 300.0),
    # Specific heat fit is only valid 4-27 K; clamped above (flagged approximate).
    cp=_log_poly([28.08, -228.23, 777.587, -1448.423, 1596.567, -1040.294, 371.2125, -56.004, 0.0], 4.0, 27.0),
    approximate=True,
)

_G10 = CryoCurveSet(
    name="G-10 CR Fiberglass Epoxy (normal)",
    k=_log_poly([-4.1236, 13.788, -26.068, 26.272, -14.663, 4.4954, -0.6905, 0.0397, 0.0], 10.0, 300.0),
    cp=_log_poly([-2.4083, 7.6006, -8.2982, 7.3301, -4.2386, 1.4294, -0.24396, 0.015236, 0.0], 4.0, 300.0),
    approximate=True,
)


def _normalize(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


# materials.json name (normalized) -> NIST curve set
_REGISTRY: dict[str, CryoCurveSet] = {}
for _curve, _names in [
    (_AL_6061, ["6061-t6 aluminum", "6061-t6 (ss)", "aluminum 6061", "aluminum 6061-t6", "5052-h32 aluminum", "5052-h32"]),
    (_SS_304, ["aisi 304", "aisi 304 stainless steel", "18-8 stainless steel", "17-7ph stainless steel", "304 stainless steel", "aisi 303 stainless steel", "aisi 316 stainless steel", "aisi 316 stainless steel sheet (ss)"]),
    (_COPPER, ["copper", "ofhc copper"]),
    (_INVAR, ["invar36", "invar, al 36", "invar", "invar (fe-36ni)", "carpenter invar 36"]),
    (_G10, ["phenolic", "g-10", "delrin 2700 nc010, low viscosity acetal copolymer (ss)", "delrin", "cryogenic g10-cr", "g-10 glass epoxy laminate", "fiberglass pcb"]),
]:
    for _name in _names:
        _REGISTRY[_name] = _curve


def curve_for_material(material_name: str) -> CryoCurveSet | None:
    """Return the NIST curve set for a materials.json material name, or None."""
    return _REGISTRY.get(_normalize(material_name))


def has_curve(material_name: str) -> bool:
    return _normalize(material_name) in _REGISTRY


def specific_heat_J_kgK(
    material_name: str,
    temperatures_K: np.ndarray,
    *,
    fallback_cp: float | None = None,
) -> np.ndarray:
    """cp(T) for a material; falls back to constant fallback_cp where no curve exists."""
    temperatures = np.asarray(temperatures_K, dtype=float)
    curve = curve_for_material(material_name)
    values = curve.specific_heat(temperatures) if curve is not None else None
    if values is None:
        if fallback_cp is None:
            raise KeyError(f"No cryogenic cp curve for {material_name!r} and no fallback provided.")
        return np.full(temperatures.shape, float(fallback_cp))
    return values


def thermal_conductivity_W_mK(
    material_name: str,
    temperatures_K: np.ndarray,
    *,
    rrr: int = DEFAULT_COPPER_RRR,
    fallback_k: float | None = None,
) -> np.ndarray:
    """k(T) for a material; falls back to constant fallback_k where no curve exists."""
    temperatures = np.asarray(temperatures_K, dtype=float)
    curve = curve_for_material(material_name)
    values = curve.thermal_conductivity(temperatures, rrr=rrr) if curve is not None else None
    if values is None:
        if fallback_k is None:
            raise KeyError(f"No cryogenic k curve for {material_name!r} and no fallback provided.")
        return np.full(temperatures.shape, float(fallback_k))
    return values
