"""Sensitivity outputs (spec Section 4.5).

For the governing limit state of each string (the one with the highest Pf):
  * Tornado: FORM reliability index beta with one input held at its P10, then
    its P90, while the other inputs stay random. A wide bar means Pf depends
    strongly on knowing that input well.
  * Sobol indices of the safety margin g (first-order and total), by the
    Saltelli (2010) first-order and Jansen (1999) total-effect estimators.
    They say how much of g's variance each input explains, alone and with
    interactions.
  * Pf against maximum surface treating pressure, for every string with an
    injection-down-casing (frac) load: FORM on the frac case's limit states
    over a range of surface pressures, and the highest pressure that still
    meets the reliability target.

All of it reuses reliability.StringLimitStates, so it uses the same loads,
ratings and random variables as the Pf tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats

from .design import StringDesign
from .loads import StringGeometry
from .profiles import Profiles
from .ratings import Pipe
from .reliability import (StringLimitStates, StringReliability, VariableSet, collect_variables, form,
                          form_many)
from .schema import Distribution, WellInput

Z90 = float(stats.norm.ppf(0.90))
SOBOL_SAMPLES = 2048
TREATING_FACTORS = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)


@dataclass(frozen=True)
class TornadoBar:
    variable: str
    p10: float
    p90: float
    beta_p10: float
    beta_p90: float

    @property
    def swing(self) -> float:
        a, b = self.beta_p10, self.beta_p90
        if not (math.isfinite(a) and math.isfinite(b)):
            return math.inf if a != b else 0.0
        return abs(b - a)


@dataclass(frozen=True)
class SobolIndex:
    variable: str
    first: float
    total: float


@dataclass(frozen=True)
class TreatingPoint:
    surface_psi: float
    pf: float
    beta: float


@dataclass
class StringSensitivity:
    string: str
    case: str
    check: str
    beta_base: float
    tornado: list[TornadoBar]
    sobol: list[SobolIndex]
    sobol_samples: int
    treating: list[TreatingPoint] | None = None
    treating_limit_psi: float | None = None  # highest surface pressure with frac Pf <= target
    notes: list[str] = field(default_factory=list)


# --- Generic tools -----------------------------------------------------------------------------

def sobol_indices(g: Callable[[np.ndarray], np.ndarray], variables: VariableSet, n: int = SOBOL_SAMPLES,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """First-order (Saltelli 2010) and total (Jansen 1999) Sobol indices of g.

    Sampling is in independent standard normal space, so with correlated
    inputs the indices belong to the independent components. Uses n (k + 2)
    evaluations of g, in one batch.
    """
    k = len(variables.variables)
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, k))
    b = rng.standard_normal((n, k))
    blocks = [a, b]
    for i in range(k):
        ab = a.copy()
        ab[:, i] = b[:, i]
        blocks.append(ab)
    f = np.asarray(g(variables.to_physical(np.vstack(blocks))), dtype=float).reshape(-1)
    fa, fb, fab = f[:n], f[n:2 * n], f[2 * n:].reshape(k, n)
    var = np.var(np.concatenate([fa, fb]))
    if var <= 0:
        return np.full(k, np.nan), np.full(k, np.nan)
    first = np.mean(fb * (fab - fa), axis=1) / var
    total = 0.5 * np.mean((fa - fab) ** 2, axis=1) / var
    return first, total


def tornado(g: Callable[[np.ndarray], np.ndarray], variables: VariableSet) -> list[TornadoBar]:
    """beta with each variable fixed at its P10 and P90 (the others random), largest swing first."""
    vs = variables.variables
    bars = []
    for i, v in enumerate(vs):
        lo, hi = (float(v.from_standard_normal(np.array([z]))[0]) for z in (-Z90, Z90))
        betas = [_beta_fixed(g, variables, i, val) for val in (lo, hi)]
        bars.append(TornadoBar(v.name, lo, hi, *betas))
    return sorted(bars, key=lambda b: -b.swing)


def _beta_fixed(g, variables: VariableSet, idx: int, value: float) -> float:
    keep = [j for j in range(len(variables.variables)) if j != idx]

    def g_reduced(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return g(np.insert(x, idx, value, axis=1))

    if not keep:
        val = float(np.asarray(g(np.array([[value]]))).reshape(-1)[0])
        return math.inf if val > 0 else -math.inf
    reduced = VariableSet([variables.variables[j] for j in keep], variables.correlation[np.ix_(keep, keep)])
    return form(g_reduced, reduced).beta


# --- Per string --------------------------------------------------------------------------------

def string_sensitivity(well: WellInput, profiles: Profiles, geometries: list[StringGeometry], name: str,
                       design: Pipe | StringDesign, reliability: StringReliability | None = None,
                       sobol_n: int = SOBOL_SAMPLES, seed: int = 0,
                       treating_factors: tuple[float, ...] = TREATING_FACTORS) -> StringSensitivity:
    """Tornado, Sobol and (for frac strings) Pf vs surface treating pressure for one string."""
    variables = collect_variables(well, geometries, name)
    ls = StringLimitStates(well, profiles, geometries, name, design, variables)
    if reliability is not None:
        case, check = reliability.governing.case, reliability.governing.check
    else:
        case, check = ls.labels[int(np.argmin(ls.base_g))]
    j = ls.labels.index((case, check))

    def g(x: np.ndarray) -> np.ndarray:
        return ls(x)[:, j]

    notes = []
    base = form(g, variables)
    s1, st = sobol_indices(g, variables, sobol_n, seed)
    if np.isnan(s1).any():
        notes.append(f"{name}: the margin for {check} under {case} does not vary with the inputs; no Sobol indices")
    if np.any(variables.correlation != np.eye(len(variables.variables))):
        notes.append(f"{name}: inputs are correlated; Sobol indices refer to the independent components")
    sobol = sorted((SobolIndex(v.name, float(a), float(b)) for v, a, b in zip(variables.variables, s1, st)),
                   key=lambda s: -(s.total if math.isfinite(s.total) else 0.0))
    out = StringSensitivity(name, case, check, base.beta, tornado(g, variables), sobol, sobol_n, notes=notes)
    if "frac_down_casing" in {c for c, _ in ls.labels}:
        out.treating, out.treating_limit_psi = pf_vs_treating_pressure(
            well, profiles, geometries, name, design, well.reliability_target_pf, treating_factors)
    return out


def pf_vs_treating_pressure(well: WellInput, profiles: Profiles, geometries: list[StringGeometry], name: str,
                            design: Pipe | StringDesign, target_pf: float,
                            factors: tuple[float, ...] = TREATING_FACTORS
                            ) -> tuple[list[TreatingPoint], float | None]:
    """Pf of the frac case (worst of its checks, by FORM) over a range of maximum surface treating pressures.

    The pressure is scaled about the input's value; if the input gives a
    distribution, its mean moves and its spread is kept. Returns the points and
    the highest pressure with Pf <= target (log-linear interpolation), or None
    if the target is met everywhere or nowhere in the range.
    """
    cfg = well.load_cases[name].frac_down_casing
    base_ps = cfg.max_surface_psi.base if isinstance(cfg.max_surface_psi, Distribution) else float(cfg.max_surface_psi)
    points = []
    for f in factors:
        ps = base_ps * f
        w = _with_surface_pressure(well, name, ps)
        variables = collect_variables(w, geometries, name)
        ls = StringLimitStates(w, profiles, geometries, name, design, variables)
        cols = [i for i, (c, _) in enumerate(ls.labels) if c == "frac_down_casing"]
        res = form_many(ls, variables, cols)
        worst = max(res.values(), key=lambda r: r.pf)
        points.append(TreatingPoint(ps, worst.pf, worst.beta))
    return points, _crossing(points, target_pf)


def _with_surface_pressure(well: WellInput, name: str, ps: float) -> WellInput:
    d = well.model_dump()
    current = d["load_cases"][name]["frac_down_casing"]["max_surface_psi"]
    if isinstance(current, dict):
        current = {k: v for k, v in current.items() if v is not None}
        shift = ps - current["mean"]
        current["mean"] = ps
        for k in ("lower", "upper"):
            if k in current:
                current[k] += shift
        d["load_cases"][name]["frac_down_casing"]["max_surface_psi"] = current
    else:
        d["load_cases"][name]["frac_down_casing"]["max_surface_psi"] = ps
    return WellInput.model_validate(d)


def _crossing(points: list[TreatingPoint], target: float) -> float | None:
    ok = [p.pf <= target for p in points]
    if all(ok) or not any(ok):
        return None
    for a, b in zip(points, points[1:]):
        if a.pf <= target < b.pf:
            la, lb = math.log10(max(a.pf, 1e-300)), math.log10(b.pf)
            t = (math.log10(target) - la) / (lb - la)
            return a.surface_psi + t * (b.surface_psi - a.surface_psi)
    return None
