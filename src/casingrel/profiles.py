"""Pore pressure, fracture gradient, overburden and temperature vs TVD.

Tables are interpolated linearly in TVD. Outside a table the end value is
held constant, and every such extrapolation inside the well's TVD range is
reported in `Profiles.warnings` (spec Section 2.1).

Gradients are in ppg equivalent mud weight referenced to KB, so
P_psi = 0.052 * ppg * TVD_ft (spec Section 3). The overburden is stored as an
average gradient Sv/TVD in psi/ft.

Uncertainty: the curve functions take shift and scale modifiers that
broadcast with numpy. The Monte Carlo layer passes sampled modifiers shaped
(n_samples, 1) together with a node grid shaped (n_nodes,) and gets back
(n_samples, n_nodes). The fracture gradient is always computed from the
modified pore pressure, never sampled on its own (spec Section 2.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .schema import (DEFAULT_OVERBURDEN_PSI_PER_FT, CurveUncertainty, Distribution, FracEaton,
                     PsiPerFtCurve, TemperatureGradient, WellInput)
from .trajectory import Survey, build_survey
from .units import PSI_PER_FT_PER_PPG

# Eaton needs 0 <= nu < 0.5; sampled values are clipped into this range.
_POISSON_MAX = 0.4999
_CHECK_STEP_FT = 50.0


@dataclass(frozen=True)
class DepthCurve:
    """Piecewise-linear curve of some value vs TVD (ft)."""

    name: str
    unit: str
    tvd_ft: np.ndarray
    value: np.ndarray

    @classmethod
    def from_table(cls, name: str, unit: str, table: dict[float, float]) -> DepthCurve:
        tvd = np.array(sorted(table), dtype=float)
        return cls(name, unit, tvd, np.array([table[z] for z in tvd], dtype=float))

    @classmethod
    def constant(cls, name: str, unit: str, value: float) -> DepthCurve:
        return cls(name, unit, np.array([0.0]), np.array([float(value)]))

    def __call__(self, tvd_ft: ArrayLike) -> np.ndarray:
        return np.interp(tvd_ft, self.tvd_ft, self.value)

    def coverage_warnings(self, max_tvd_ft: float) -> list[str]:
        """Depth ranges in [0, max_tvd_ft] that the table does not cover."""
        if len(self.tvd_ft) == 1:
            return []  # a single value is a deliberate constant, not an extrapolation
        out = []
        if self.tvd_ft[0] > 0.5:
            out.append(f"{self.name}: table starts at {self.tvd_ft[0]:.0f} ft TVD; "
                       f"held at {self.value[0]:g} {self.unit} from 0 ft (extrapolated)")
        if self.tvd_ft[-1] < max_tvd_ft - 0.5:
            out.append(f"{self.name}: table ends at {self.tvd_ft[-1]:.0f} ft TVD; held at "
                       f"{self.value[-1]:g} {self.unit} down to {max_tvd_ft:.0f} ft (extrapolated)")
        return out


@dataclass(frozen=True)
class Profiles:
    """All depth profiles for one well, at the input's base values."""

    survey: Survey
    pore: DepthCurve
    overburden: DepthCurve
    poisson: DepthCurve | None
    frac_table: DepthCurve | None
    temperature_curve: DepthCurve | None
    surface_F: float | None
    gradient_F_per_100ft: float | None
    pore_uncertainty: CurveUncertainty | None = None
    poisson_uncertainty: CurveUncertainty | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def max_tvd_ft(self) -> float:
        return self.survey.max_tvd_ft

    # --- Pore pressure and overburden ---------------------------------------

    def pore_ppg(self, tvd_ft: ArrayLike, shift_ppg: ArrayLike = 0.0, scale: ArrayLike = 1.0) -> np.ndarray:
        """Pore-pressure gradient, ppg: base curve * scale + shift."""
        return self.pore(tvd_ft) * np.asarray(scale) + np.asarray(shift_ppg)

    def pore_psi(self, tvd_ft: ArrayLike, shift_ppg: ArrayLike = 0.0, scale: ArrayLike = 1.0) -> np.ndarray:
        return ppg_to_psi(self.pore_ppg(tvd_ft, shift_ppg, scale), tvd_ft)

    def overburden_ppg(self, tvd_ft: ArrayLike, scale: ArrayLike = 1.0) -> np.ndarray:
        """Overburden as an equivalent gradient, ppg."""
        return self.overburden(tvd_ft) * np.asarray(scale) / PSI_PER_FT_PER_PPG

    def overburden_psi(self, tvd_ft: ArrayLike, scale: ArrayLike = 1.0) -> np.ndarray:
        return self.overburden(tvd_ft) * np.asarray(scale) * np.asarray(tvd_ft, dtype=float)

    # --- Fracture gradient ---------------------------------------------------

    def poisson_ratio(self, tvd_ft: ArrayLike, shift: ArrayLike = 0.0, scale: ArrayLike = 1.0) -> np.ndarray:
        if self.poisson is None:
            raise ValueError("fracture gradient was given as a table; there is no Poisson's ratio")
        nu = self.poisson(tvd_ft) * np.asarray(scale) + np.asarray(shift)
        return np.clip(nu, 0.0, _POISSON_MAX)

    def frac_ppg(self, tvd_ft: ArrayLike, pore_shift_ppg: ArrayLike = 0.0, pore_scale: ArrayLike = 1.0,
                 poisson_shift: ArrayLike = 0.0, poisson_scale: ArrayLike = 1.0,
                 overburden_scale: ArrayLike = 1.0) -> np.ndarray:
        """Fracture gradient, ppg.

        Eaton (1969): FG = nu/(1 - nu) * (Sv - Pp) + Pp, all as gradients.
        Computed from the (possibly shifted or scaled) pore pressure, so the
        two stay linked. A frac-gradient table ignores the modifiers.
        TODO(citation): add the equation number from Eaton (1969), JPT.
        """
        if self.frac_table is not None:
            return self.frac_table(tvd_ft) + 0.0 * np.asarray(pore_shift_ppg)
        pp = self.pore_ppg(tvd_ft, pore_shift_ppg, pore_scale)
        nu = self.poisson_ratio(tvd_ft, poisson_shift, poisson_scale)
        return eaton_frac_ppg(pp, self.overburden_ppg(tvd_ft, overburden_scale), nu)

    def frac_psi(self, tvd_ft: ArrayLike, **modifiers: ArrayLike) -> np.ndarray:
        return ppg_to_psi(self.frac_ppg(tvd_ft, **modifiers), tvd_ft)

    # --- Temperature ----------------------------------------------------------

    def temperature_F(self, tvd_ft: ArrayLike, surface_F: ArrayLike | None = None,
                      gradient_F_per_100ft: ArrayLike | None = None) -> np.ndarray:
        """Static formation temperature, degF. Overrides apply to the gradient form only."""
        if self.temperature_curve is not None:
            return self.temperature_curve(tvd_ft)
        t0 = self.surface_F if surface_F is None else np.asarray(surface_F)
        g = self.gradient_F_per_100ft if gradient_F_per_100ft is None else np.asarray(gradient_F_per_100ft)
        return t0 + g * np.asarray(tvd_ft, dtype=float) / 100.0


def ppg_to_psi(ppg: ArrayLike, tvd_ft: ArrayLike) -> np.ndarray:
    """Hydrostatic pressure, P_psi = 0.052 * rho_ppg * TVD_ft (spec Section 3)."""
    return PSI_PER_FT_PER_PPG * np.asarray(ppg) * np.asarray(tvd_ft, dtype=float)


def eaton_frac_ppg(pore_ppg: ArrayLike, overburden_ppg: ArrayLike, poisson: ArrayLike) -> np.ndarray:
    """Eaton (1969) fracture gradient: nu/(1 - nu) * (Sv - Pp) + Pp, gradients in ppg."""
    nu = np.asarray(poisson)
    pp = np.asarray(pore_ppg)
    return nu / (1.0 - nu) * (np.asarray(overburden_ppg) - pp) + pp


def build_profiles(well: WellInput, survey: Survey | None = None) -> Profiles:
    """Build every depth profile for a well at its base values and check them."""
    survey = build_survey(well.trajectory) if survey is None else survey
    pore = DepthCurve.from_table("pore pressure", "ppg", well.pore_pressure_ppg.table)

    ob_in = well.overburden_psi_per_ft
    if ob_in is None:
        overburden = DepthCurve.constant("overburden", "psi/ft", DEFAULT_OVERBURDEN_PSI_PER_FT)
    elif isinstance(ob_in, PsiPerFtCurve):
        overburden = DepthCurve.from_table("overburden", "psi/ft", ob_in.table)
    else:
        overburden = DepthCurve.constant("overburden", "psi/ft", _base(ob_in))

    poisson = frac_table = None
    poisson_uncertainty = None
    if isinstance(well.frac_gradient, FracEaton):
        poisson = DepthCurve.from_table("Poisson's ratio", "", well.frac_gradient.poisson.table)
        poisson_uncertainty = well.frac_gradient.poisson.uncertainty
    else:
        frac_table = DepthCurve.from_table("fracture gradient", "ppg", well.frac_gradient.table)

    temperature_curve = surface_F = gradient = None
    if isinstance(well.temperature, TemperatureGradient):
        surface_F = _base(well.temperature.surface_F)
        gradient = _base(well.temperature.gradient_F_per_100ft)
    else:
        temperature_curve = DepthCurve.from_table("temperature", "degF", well.temperature.table)

    prof = Profiles(survey=survey, pore=pore, overburden=overburden, poisson=poisson,
                    frac_table=frac_table, temperature_curve=temperature_curve, surface_F=surface_F,
                    gradient_F_per_100ft=gradient, pore_uncertainty=well.pore_pressure_ppg.uncertainty,
                    poisson_uncertainty=poisson_uncertainty)
    prof.warnings.extend(_check(prof))
    return prof


def _base(x: float | Distribution) -> float:
    return x.base if isinstance(x, Distribution) else float(x)


def _check(p: Profiles) -> list[str]:
    """Coverage and physical-sanity warnings (printed, never fatal)."""
    out = []
    for curve in (p.pore, p.overburden, p.poisson, p.frac_table, p.temperature_curve):
        if curve is not None:
            out += curve.coverage_warnings(p.max_tvd_ft)
    if p.frac_table is not None and (p.pore_uncertainty is not None):
        out.append("fracture gradient is a table, so it will not follow sampled pore pressure "
                   "(spec 2.2 expects Eaton)")
    z = np.arange(_CHECK_STEP_FT, p.max_tvd_ft + _CHECK_STEP_FT, _CHECK_STEP_FT)
    z[-1] = min(z[-1], p.max_tvd_ft)
    if p.poisson is not None:
        nu = p.poisson(z)
        if np.any((nu <= 0) | (nu >= 0.5)):
            out.append(f"Poisson's ratio outside (0, 0.5) at {_where(z, (nu <= 0) | (nu >= 0.5))}")
    pp, ob, fg = p.pore_ppg(z), p.overburden_ppg(z), p.frac_ppg(z)
    _flag(out, z, pp >= ob, "pore pressure at or above overburden")
    _flag(out, z, fg <= pp, "fracture gradient at or below pore pressure")
    _flag(out, z, fg >= ob, "fracture gradient at or above overburden")
    return out


def _flag(out: list[str], z: np.ndarray, bad: np.ndarray, what: str) -> None:
    if np.any(bad):
        out.append(f"{what} at {_where(z, bad)}")


def _where(z: np.ndarray, bad: np.ndarray) -> str:
    return f"{z[bad].min():.0f}-{z[bad].max():.0f} ft TVD"
