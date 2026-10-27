"""Pipe-body ratings for casing: burst, collapse and tension.

Source for every formula: API TR 5C3 (2008) / ISO/TR 10400:2018, "Petroleum
and natural gas industries - Equations and calculations for the properties of
casing, tubing, drill pipe and line pipe used as casing or tubing", as
summarised in spec Section 3.3. These are the API minimum-performance
(serviceability) equations, not the ultimate-limit-state equations.

TODO(citation): confirm the section and equation numbers against the user's
copy of ISO/TR 10400:2018 and add them to each docstring (CLAUDE.md rule 3).

Units: field units throughout (psi, in, lbf). Functions take floats or numpy
arrays and broadcast, so the Monte Carlo layer can pass whole sample vectors.
Ratings are returned unrounded; the API tables round to the nearest 10 psi.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

# API minimum wall tolerance: wall may be as thin as 87.5% of nominal.
API_WALL_TOLERANCE = 0.875

# Elastic-collapse constant, psi (46.95e6 = 2E / (1 - nu^2) with API's E and nu).
ELASTIC_COLLAPSE_CONST_PSI = 46.95e6

# Numerical guard for the biaxial collapse rating; see collapse_api_combined_psi.
YPA_FLOOR_PSI = 20_000.0

YIELD = "yield"
PLASTIC = "plastic"
TRANSITION = "transition"
ELASTIC = "elastic"


@dataclass(frozen=True)
class CollapseCoefficients:
    """Empirical collapse coefficients A, B, C, F, G for one yield strength.

    A, B, F and G are dimensionless; C is in psi.
    """

    a: np.ndarray | float
    b: np.ndarray | float
    c_psi: np.ndarray | float
    f: np.ndarray | float
    g: np.ndarray | float


@dataclass(frozen=True)
class CollapseRegimeLimits:
    """D/t boundaries between the four API collapse regimes."""

    dt_yield_plastic: np.ndarray | float
    dt_plastic_transition: np.ndarray | float
    dt_transition_elastic: np.ndarray | float


def burst_api_psi(od_in: ArrayLike, wall_in: ArrayLike, yield_psi: ArrayLike,
                  k_wall: float = API_WALL_TOLERANCE) -> np.ndarray | float:
    """Internal yield (burst) pressure, Barlow's formula with minimum wall.

    P_iy = k_wall * 2 * Y_p * t / D, with k_wall = 0.875 (API 12.5% wall
    tolerance). API TR 5C3 / ISO/TR 10400, internal yield pressure.
    """
    od_in, wall_in, yield_psi = np.broadcast_arrays(*map(np.asarray, (od_in, wall_in, yield_psi)))
    return _scalar_or_array(k_wall * 2.0 * yield_psi * wall_in / od_in)


def pipe_body_yield_lbf(od_in: ArrayLike, wall_in: ArrayLike,
                        yield_psi: ArrayLike) -> np.ndarray | float:
    """Pipe-body tensile yield strength, F_y = Y_p * (pi/4) * (D^2 - d^2).

    Uses nominal wall (d = D - 2t). Connection strength is checked separately
    from the manufacturer's data sheet. API TR 5C3 / ISO/TR 10400, pipe-body
    yield strength.
    """
    od_in = np.asarray(od_in, dtype=float)
    id_in = od_in - 2.0 * np.asarray(wall_in, dtype=float)
    return _scalar_or_array(np.asarray(yield_psi) * np.pi / 4.0 * (od_in**2 - id_in**2))


def collapse_coefficients(yield_psi: ArrayLike) -> CollapseCoefficients:
    """Collapse coefficients A, B, C, F, G as functions of yield strength (psi).

    API TR 5C3 / ISO/TR 10400, plastic and transition collapse coefficients.
    """
    yp = np.asarray(yield_psi, dtype=float)
    a = 2.8762 + 0.10679e-5 * yp + 0.21301e-10 * yp**2 - 0.53132e-16 * yp**3
    b = 0.026233 + 0.50609e-6 * yp
    c_psi = -465.93 + 0.030867 * yp - 0.10483e-7 * yp**2 + 0.36989e-13 * yp**3
    ba = b / a
    r = 3.0 * ba / (2.0 + ba)
    f = ELASTIC_COLLAPSE_CONST_PSI * r**3 / (yp * (r - ba) * (1.0 - r) ** 2)
    g = f * b / a
    return CollapseCoefficients(*(_scalar_or_array(x) for x in (a, b, c_psi, f, g)))


def collapse_regime_limits(yield_psi: ArrayLike) -> CollapseRegimeLimits:
    """D/t boundaries between yield, plastic, transition and elastic collapse.

    API TR 5C3 / ISO/TR 10400, applicable D/t ranges.
    """
    yp = np.asarray(yield_psi, dtype=float)
    k = collapse_coefficients(yp)
    bc = k.b + k.c_psi / yp
    dt_yp = (np.sqrt((k.a - 2.0) ** 2 + 8.0 * bc) + (k.a - 2.0)) / (2.0 * bc)
    dt_pt = yp * (k.a - k.f) / (k.c_psi + yp * (k.b - k.g))
    dt_te = (2.0 + k.b / k.a) / (3.0 * k.b / k.a)
    return CollapseRegimeLimits(*(_scalar_or_array(x) for x in (dt_yp, dt_pt, dt_te)))


def collapse_regime(od_in: ArrayLike, wall_in: ArrayLike,
                    yield_psi: ArrayLike) -> np.ndarray | str:
    """Name of the governing API collapse regime ("yield", "plastic", ...)."""
    dt, yp = _dt_and_yield(od_in, wall_in, yield_psi)
    lim = collapse_regime_limits(yp)
    regime = np.select(
        [dt <= lim.dt_yield_plastic, dt <= lim.dt_plastic_transition,
         dt <= lim.dt_transition_elastic],
        [YIELD, PLASTIC, TRANSITION], default=ELASTIC)
    return regime.item() if regime.ndim == 0 else regime


def collapse_api_psi(od_in: ArrayLike, wall_in: ArrayLike,
                     yield_psi: ArrayLike) -> np.ndarray | float:
    """API collapse resistance with no axial load, psi.

    The regime is chosen by D/t (see collapse_regime_limits):
      yield:      P_y = 2 Y_p ((D/t) - 1) / (D/t)^2
      plastic:    P_p = Y_p (A/(D/t) - B) - C
      transition: P_t = Y_p (F/(D/t) - G)
      elastic:    P_e = 46.95e6 / ((D/t) ((D/t) - 1)^2)
    API TR 5C3 / ISO/TR 10400, collapse pressure equations.
    """
    dt, yp = _dt_and_yield(od_in, wall_in, yield_psi)
    k = collapse_coefficients(yp)
    lim = collapse_regime_limits(yp)
    p_yield = 2.0 * yp * (dt - 1.0) / dt**2
    p_plastic = yp * (k.a / dt - k.b) - k.c_psi
    p_transition = yp * (k.f / dt - k.g)
    p_elastic = ELASTIC_COLLAPSE_CONST_PSI / (dt * (dt - 1.0) ** 2)
    p = np.select(
        [dt <= lim.dt_yield_plastic, dt <= lim.dt_plastic_transition,
         dt <= lim.dt_transition_elastic],
        [p_yield, p_plastic, p_transition], default=p_elastic)
    return _scalar_or_array(p)


def reduced_yield_biaxial_psi(yield_psi: ArrayLike,
                              axial_stress_psi: ArrayLike) -> np.ndarray | float:
    """Yield strength of axial-stress equivalent grade, Y_pa.

    Y_pa = [sqrt(1 - 0.75 (s_a/Y_p)^2) - 0.5 s_a/Y_p] Y_p,
    with s_a positive in tension. Tension lowers Y_pa; compression raises it.
    API TR 5C3 / ISO/TR 10400, collapse under combined axial load (spec 3.4).

    TODO(citation): check whether the standard's axial term is s_a alone (as
    in the spec) or s_a + p_i (internal pressure). Needs the standard's text.
    """
    yp = np.asarray(yield_psi, dtype=float)
    # Outside -2/sqrt(3) <= s_a/Y_p <= 1 the formula has no meaning (axial yield in
    # tension gives Y_pa = 0); clipping keeps sampled extremes finite.
    ratio = np.clip(np.asarray(axial_stress_psi, dtype=float) / yp, -2.0 / np.sqrt(3.0), 1.0)
    return _scalar_or_array((np.sqrt(np.maximum(1.0 - 0.75 * ratio**2, 0.0)) - 0.5 * ratio) * yp)


def collapse_api_combined_psi(od_in: ArrayLike, wall_in: ArrayLike,
                              yield_psi: ArrayLike,
                              axial_stress_psi: ArrayLike) -> np.ndarray | float:
    """API collapse resistance under axial stress, psi.

    Replaces Y_p with Y_pa (reduced_yield_biaxial_psi), then recomputes the
    coefficients, the regime and the collapse rating. Compare the result with
    equivalent_external_pressure_psi, not with the raw external pressure.
    """
    ypa = np.asarray(reduced_yield_biaxial_psi(yield_psi, axial_stress_psi), dtype=float)
    # Below YPA_FLOOR_PSI (under the API formulas' range) the rating is scaled
    # linearly to zero at Y_pa = 0, so heavy tension never yields a NaN or a jump.
    low = ypa < YPA_FLOOR_PSI
    rating = collapse_api_psi(od_in, wall_in, np.where(low, YPA_FLOOR_PSI, ypa))
    return _scalar_or_array(np.where(low, rating * ypa / YPA_FLOOR_PSI, rating))


def equivalent_external_pressure_psi(p_external_psi: ArrayLike, p_internal_psi: ArrayLike,
                                     od_in: ArrayLike, wall_in: ArrayLike) -> np.ndarray | float:
    """Equivalent external pressure for collapse, P_e - (1 - 2/(D/t)) P_i, psi.

    API TR 5C3 / ISO/TR 10400, collapse with internal pressure (spec 3.4).
    """
    dt = np.asarray(od_in, dtype=float) / np.asarray(wall_in, dtype=float)
    return _scalar_or_array(np.asarray(p_external_psi) - (1.0 - 2.0 / dt) * np.asarray(p_internal_psi))


def _dt_and_yield(od_in: ArrayLike, wall_in: ArrayLike,
                  yield_psi: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    od_in, wall_in, yp = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (od_in, wall_in, yield_psi)))
    return od_in / wall_in, yp


def _scalar_or_array(x: np.ndarray) -> np.ndarray | float:
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


# --- Pipe specification ---------------------------------------------------------

# Minimum yield strength by API 5CT grade, psi (API Spec 5CT / ISO 11960).
GRADE_YIELD_PSI = {
    "H40": 40_000.0, "J55": 55_000.0, "K55": 55_000.0, "M65": 65_000.0, "N80": 80_000.0,
    "L80": 80_000.0, "C90": 90_000.0, "R95": 95_000.0, "T95": 95_000.0, "C110": 110_000.0,
    "P110": 110_000.0, "Q125": 125_000.0,
}


@dataclass(frozen=True)
class Pipe:
    """One pipe: body dimensions, grade and optional connection ratings.

    Connection ratings come from the manufacturer's data sheet. When one is
    missing, only the pipe body is checked and the design check says so.
    """

    od_in: float
    wall_in: float
    weight_lb_per_ft: float
    grade: str
    yield_psi: float | None = None
    drift_in: float | None = None
    connection: str | None = None
    conn_tension_lbf: float | None = None
    conn_compression_lbf: float | None = None
    conn_internal_psi: float | None = None
    cost_per_ft: float | None = None

    def __post_init__(self) -> None:
        if self.yield_psi is None:
            if self.grade not in GRADE_YIELD_PSI:
                raise ValueError(f"unknown grade {self.grade!r}; give yield_psi")
            object.__setattr__(self, "yield_psi", GRADE_YIELD_PSI[self.grade])
        if not 0 < self.wall_in < self.od_in / 2:
            raise ValueError("wall_in must be positive and less than half the OD")

    @property
    def id_in(self) -> float:
        return self.od_in - 2.0 * self.wall_in

    @property
    def dt(self) -> float:
        return self.od_in / self.wall_in

    @property
    def area_steel_in2(self) -> float:
        return np.pi / 4.0 * (self.od_in**2 - self.id_in**2)

    @property
    def label(self) -> str:
        return f"{self.od_in:g}-in {self.weight_lb_per_ft:g} lb/ft {self.grade}"
