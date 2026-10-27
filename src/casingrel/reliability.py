"""Probability of failure for casing strings (spec Section 4.1-4.3).

Limit states (4.1): for each string, load case and check, g = R - L, taken as
the minimum over all depth nodes; failure is g <= 0. R and L come from
design.demands, the same code the deterministic check uses. The resistance
model is the API TR 5C3 serviceability model. The ISO/TR 10400 ultimate-limit-
state model (Klever-Tamano collapse, ductile-rupture burst) is not built yet:
it needs the standard's equations from the user.

Methods (4.2):
  * Monte Carlo with a fixed seed: 1e5 samples to start, more added until the
    coefficient of variation of each non-zero Pf is below 10% (up to max_n).
    95% confidence intervals use the Wilson score interval.
  * FORM: Hasofer-Lind / Rackwitz-Fiessler iteration in standard normal space,
    with finite-difference gradients. Gives beta and importance factors alpha.
  * Importance sampling around the FORM design point, to check FORM.

Inputs: every Distribution in the input file, curve shift/scale uncertainty,
and pipe manufacturing ratios (yield, OD, wall). They are independent unless
the input gives a correlation (Gaussian copula). Frac gradient is never
sampled on its own; it follows the sampled pore pressure (spec 2.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats

from .design import demands, setting_depths
from .loads import (PipeBody, Realization, StringGeometry, required_mud_ppg, resolve_strings,
                    string_loads)
from .profiles import Profiles, ProfileView, build_profiles
from .ratings import Pipe
from .schema import CurveUncertainty, Distribution, PsiPerFtCurve, TemperatureGradient, WellInput

Z95 = 1.959963984540054
DEFAULT_SAMPLES = 100_000
DEFAULT_MAX_SAMPLES = 1_000_000
BATCH = 20_000
TARGET_COV = 0.10
SWEEP_TARGETS = (1e-2, 1e-3, 1e-4, 1e-5)  # spec 4.3

# Placeholders for manufacturing variability, used only when the input's
# pipe_statistics block is missing an entry. They are NOT from a source:
# replace them with ISO/TR 10400:2018 or Gouveia et al. (2020) Table 1 values.
PLACEHOLDER_PIPE_STATS = {
    "yield_ratio": Distribution(dist="normal", mean=1.10, sd=0.05),
    "od_ratio": Distribution(dist="normal", mean=1.00, sd=0.003),
    "wall_ratio": Distribution(dist="normal", mean=1.00, sd=0.03, lower=0.875),
}
PIPE_KEYS = ("pipe.yield_ratio", "pipe.od_ratio", "pipe.wall_ratio")


# --- Random variables -------------------------------------------------------------------

@dataclass(frozen=True)
class RandomVariable:
    """One uncertain input. `name` is its Realization key (or a pipe.* ratio)."""

    name: str
    dist: Distribution

    @property
    def base(self) -> float:
        return self.dist.base

    def _frozen(self):
        d = self.dist
        if d.dist == "normal":
            return stats.norm(d.mean, d.sd)
        if d.dist == "lognormal":
            s = math.sqrt(math.log1p((d.sd / d.mean) ** 2))
            return stats.lognorm(s=s, scale=math.exp(math.log(d.mean) - s * s / 2))
        if d.dist == "uniform":
            return stats.uniform(d.low, d.high - d.low)
        return stats.triang((d.mode - d.low) / (d.high - d.low), loc=d.low, scale=d.high - d.low)

    def from_standard_normal(self, z: np.ndarray) -> np.ndarray:
        """Map standard normal values to this variable (inverse-CDF / Rosenblatt)."""
        d = self.dist
        rv = self._frozen()
        if d.lower is None and d.upper is None:
            if d.dist == "normal":
                return d.mean + d.sd * z  # exact in the far tails
            if d.dist == "lognormal":
                s = rv.kwds["s"]
                return rv.kwds["scale"] * np.exp(s * z)
        fa = rv.cdf(d.lower) if d.lower is not None else 0.0
        fb = rv.cdf(d.upper) if d.upper is not None else 1.0
        u = np.clip(stats.norm.cdf(z), 1e-300, 1.0 - 1e-16)
        return rv.ppf(fa + u * (fb - fa))


@dataclass(frozen=True)
class VariableSet:
    variables: list[RandomVariable]
    correlation: np.ndarray  # in standard normal space (Gaussian copula)
    notes: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def to_physical(self, u: np.ndarray) -> np.ndarray:
        """Independent standard normals u (n, k) to physical values x (n, k)."""
        u = np.atleast_2d(u)
        z = u @ np.linalg.cholesky(self.correlation).T
        return np.column_stack([v.from_standard_normal(z[:, i]) for i, v in enumerate(self.variables)]) \
            if self.variables else np.zeros((u.shape[0], 0))


def collect_variables(well: WellInput, geometries: list[StringGeometry], name: str) -> VariableSet:
    """Random variables that affect one string's loads and resistance."""
    rvs: list[RandomVariable] = []
    notes: list[str] = []

    def curve(prefix: str, unc: CurveUncertainty | None) -> None:
        if unc is None:
            return
        if unc.shift_sd is not None:
            rvs.append(RandomVariable(f"{prefix}.shift" + ("_ppg" if prefix == "pore" else ""),
                                      Distribution(dist="normal", mean=0.0, sd=unc.shift_sd)))
        else:
            rvs.append(RandomVariable(f"{prefix}.scale", Distribution(dist="normal", mean=1.0, sd=unc.scale_sd)))

    curve("pore", well.pore_pressure_ppg.uncertainty)
    if well.frac_gradient.method == "eaton":
        curve("poisson", well.frac_gradient.poisson.uncertainty)
    ob = well.overburden_psi_per_ft
    if isinstance(ob, Distribution):
        rvs.append(RandomVariable("overburden.scale", _scaled(ob, 1.0 / ob.base)))
    if isinstance(ob, PsiPerFtCurve) and ob.uncertainty is not None:
        if ob.uncertainty.scale_sd is None:
            notes.append("overburden table uncertainty must be a scale_sd; shift ignored")
        else:
            rvs.append(RandomVariable("overburden.scale",
                                      Distribution(dist="normal", mean=1.0, sd=ob.uncertainty.scale_sd)))
    t = well.temperature
    if isinstance(t, TemperatureGradient):
        for key in ("surface_F", "gradient_F_per_100ft"):
            if isinstance(getattr(t, key), Distribution):
                rvs.append(RandomVariable(f"temperature.{key}", getattr(t, key)))

    g = next(x for x in geometries if x.name == name)
    for s in well.strings:
        if s.name in (name, g.next_name) and isinstance(s.mud_ppg, Distribution):
            rvs.append(RandomVariable(f"strings.{s.name}.mud_ppg", s.mud_ppg))
    for key, val in well.fluids:
        if isinstance(val, Distribution):
            rvs.append(RandomVariable(f"fluids.{key}", val))
    cases = well.load_cases.get(name)
    if cases is not None:
        for case, cfg in cases:
            if cfg in (None, False):
                continue
            for key, val in cfg:
                if isinstance(val, Distribution):
                    rvs.append(RandomVariable(f"load_cases.{name}.{case}.{key}", val))

    stats_in = well.pipe_statistics
    for key in ("yield_ratio", "od_ratio", "wall_ratio"):
        d = getattr(stats_in, key)
        if d is None:
            d = PLACEHOLDER_PIPE_STATS[key]
            notes.append(f"pipe_statistics.{key} not given; used placeholder {_describe(d)} "
                         "(not from a source: supply ISO/TR 10400 or Gouveia et al. 2020 Table 1 values)")
        rvs.append(RandomVariable(f"pipe.{key}", d))

    names = [v.name for v in rvs]
    corr = np.eye(len(rvs))
    for c in well.correlations:
        if c.a in names and c.b in names:
            i, j = names.index(c.a), names.index(c.b)
            corr[i, j] = corr[j, i] = c.rho
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the input correlations do not form a valid correlation matrix") from exc
    return VariableSet(rvs, corr, notes)


def _scaled(d: Distribution, f: float) -> Distribution:
    def sc(x):
        return None if x is None else x * f
    return Distribution(dist=d.dist, mean=sc(d.mean), sd=sc(d.sd), low=sc(d.low), mode=sc(d.mode),
                        high=sc(d.high), lower=sc(d.lower), upper=sc(d.upper))


def _describe(d: Distribution) -> str:
    if d.dist in ("normal", "lognormal"):
        out = f"{d.dist}(mean {d.mean:g}, sd {d.sd:g})"
        if d.lower is not None:
            out += f" truncated below {d.lower:g}"
        return out
    return f"{d.dist}({d.low:g}-{d.high:g})"


# --- Limit states for one string -------------------------------------------------------------

class StringLimitStates:
    """g(x) for every (case, check) of one string: x (n, k) -> g (n, n_states)."""

    def __init__(self, well: WellInput, profiles: Profiles, geometries: list[StringGeometry], name: str,
                 pipe: Pipe, variables: VariableSet, yield_derate: Callable | None = None):
        self.well, self.profiles, self.geometries, self.name = well, profiles, geometries, name
        self.pipe, self.variables, self.yield_derate = pipe, variables, yield_derate
        self.body = PipeBody(pipe.od_in, pipe.id_in, pipe.weight_lb_per_ft)
        base = self(np.array([[v.base for v in variables.variables]]))  # also sets labels
        self.base_g = base[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        cols = {v.name: x[:, [i]] for i, v in enumerate(self.variables.variables)}
        real = Realization({k: v for k, v in cols.items() if not k.startswith("pipe.")})
        sl = string_loads(self.well, self.profiles, self.geometries, self.name, self.body, real)
        factor = 1.0 if self.yield_derate is None else np.asarray(self.yield_derate(sl.temperature_F))
        yp = self.pipe.yield_psi * cols.get("pipe.yield_ratio", 1.0) * factor
        od = self.pipe.od_in * cols.get("pipe.od_ratio", 1.0)
        wall = self.pipe.wall_in * cols.get("pipe.wall_ratio", 1.0)
        dem = demands(sl, self.pipe, self.well.design_factors, yp, od, wall)
        self.labels = [(d.case, d.check) for d in dem]
        n, m = x.shape[0], len(sl.md_ft)
        return np.column_stack([np.min(np.broadcast_to(d.rating - d.load, (n, m)), axis=1) for d in dem])


# --- Monte Carlo ------------------------------------------------------------------------------

@dataclass(frozen=True)
class MCResult:
    n: int
    failures: np.ndarray  # per limit state
    system_failures: int  # samples where any limit state fails

    @property
    def pf(self) -> np.ndarray:
        return self.failures / self.n

    @property
    def system_pf(self) -> float:
        return self.system_failures / self.n

    def ci(self, failures=None) -> tuple[np.ndarray, np.ndarray]:
        return wilson_interval(self.failures if failures is None else failures, self.n)

    @property
    def cov(self) -> np.ndarray:
        pf = self.pf
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(pf > 0, np.sqrt((1 - pf) / (self.n * pf)), np.inf)


def wilson_interval(failures, n: int, z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for a binomial proportion (sound even with 0 failures)."""
    f = np.asarray(failures, dtype=float)
    p = f / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return np.maximum(center - half, 0.0), np.minimum(center + half, 1.0)


def monte_carlo(g: Callable[[np.ndarray], np.ndarray], variables: VariableSet, n: int = DEFAULT_SAMPLES,
                seed: int = 0, max_n: int = DEFAULT_MAX_SAMPLES, target_cov: float = TARGET_COV,
                batch: int = BATCH) -> MCResult:
    """Crude Monte Carlo with a fixed seed. Starts at n samples and adds batches
    while any limit state that has failures has a CoV above target_cov."""
    rng = np.random.default_rng(seed)
    k = len(variables.variables)
    failures = None
    system = 0
    done = 0
    while True:
        size = min(batch, max(n, done + batch) - done) if done < n else min(batch, max_n - done)
        if size <= 0:
            break
        x = variables.to_physical(rng.standard_normal((size, k)))
        fail = np.atleast_2d(g(x)) <= 0
        failures = fail.sum(axis=0) if failures is None else failures + fail.sum(axis=0)
        system += int(fail.any(axis=1).sum())
        done += size
        if done < n:
            continue
        res = MCResult(done, failures, system)
        need_more = (res.failures > 0) & (res.cov > target_cov)
        if not need_more.any() or done >= max_n:
            return res
    return MCResult(done, failures, system)


# --- FORM and importance sampling --------------------------------------------------------------

@dataclass(frozen=True)
class FORMResult:
    beta: float
    u_star: np.ndarray
    x_star: np.ndarray
    alpha: np.ndarray  # unit vector from origin towards failure; alpha_i^2 sum to 1
    converged: bool
    iterations: int

    @property
    def pf(self) -> float:
        return float(stats.norm.sf(self.beta)) if math.isfinite(self.beta) else (0.0 if self.beta > 0 else 1.0)


LINE_SEARCH = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])


def form_many(G: Callable[[np.ndarray], np.ndarray], variables: VariableSet, columns: list[int],
              tol: float = 1e-4, max_iter: int = 100, h: float = 1e-4) -> dict[int, FORMResult]:
    """FORM for several limit states of one function G(x (n,k)) -> (n, L) at once.

    Improved HL-RF (Zhang & Der Kiureghian 1997): the Rackwitz-Fiessler search
    direction d = [(grad . u - g)/|grad|^2] grad - u, with a backtracking line
    search on the merit m(u) = |u|^2/2 + c|g(u)|, c = 2(|u| + 1)/|grad|.
    Gradients by forward differences in independent standard normal space.
    All active limit states share each call to G, so a string's loads are
    built once per iteration. beta = sign(g(0)) |u*|.
    """
    k = len(variables.variables)
    eye = np.eye(k)

    def g_at(u: np.ndarray) -> np.ndarray:
        return np.atleast_2d(G(variables.to_physical(u)))

    g0 = g_at(np.zeros((1, k)))[0]
    u = {c: np.zeros(k) for c in columns}
    grad = {c: np.zeros(k) for c in columns}
    done: dict[int, FORMResult] = {}
    active = list(columns)
    for it in range(1, max_iter + 1):
        if not active:
            break
        blocks = np.vstack([np.vstack([u[c], u[c] + h * eye]) for c in active])
        vals = g_at(blocks)
        steps, gu, cmerit = {}, {}, {}
        for i, c in enumerate(active):
            v = vals[i * (k + 1):(i + 1) * (k + 1), c]
            gu[c], grad[c] = v[0], (v[1:] - v[0]) / h
            n2 = float(grad[c] @ grad[c])
            if n2 < 1e-30:
                beta = math.inf if gu[c] > 0 else -math.inf
                done[c] = FORMResult(beta, u[c], variables.to_physical(u[c][None, :])[0], np.zeros(k), False, it)
                continue
            target = (grad[c] @ u[c] - gu[c]) / n2 * grad[c]
            steps[c] = target - u[c]
            cmerit[c] = 2.0 * (np.linalg.norm(u[c]) + 1.0) / math.sqrt(n2)
            scale = abs(g0[c]) if g0[c] != 0 else 1.0
            if np.linalg.norm(steps[c]) < tol * (1 + np.linalg.norm(u[c])) and abs(gu[c]) < tol * scale:
                beta = math.copysign(float(np.linalg.norm(u[c])), g0[c])
                done[c] = FORMResult(beta, u[c], variables.to_physical(u[c][None, :])[0],
                                     -grad[c] / math.sqrt(n2), True, it)
        active = [c for c in active if c not in done]
        if not active:
            break
        trials = np.vstack([u[c] + lam * steps[c] for c in active for lam in LINE_SEARCH])
        tvals = g_at(trials)
        for i, c in enumerate(active):
            tg = tvals[i * len(LINE_SEARCH):(i + 1) * len(LINE_SEARCH), c]
            merit0 = 0.5 * float(u[c] @ u[c]) + cmerit[c] * abs(gu[c])
            cand = [u[c] + lam * steps[c] for lam in LINE_SEARCH]
            merits = [0.5 * float(x @ x) + cmerit[c] * abs(t) for x, t in zip(cand, tg)]
            pick = next((j for j, m in enumerate(merits) if m < merit0), int(np.argmin(merits)))
            u[c] = cand[pick]
    for c in active:
        n2 = float(grad[c] @ grad[c])
        alpha = -grad[c] / math.sqrt(n2) if n2 > 0 else np.zeros(k)
        beta = math.copysign(float(np.linalg.norm(u[c])), g0[c])
        done[c] = FORMResult(beta, u[c], variables.to_physical(u[c][None, :])[0], alpha, False, max_iter)
    return done


def form(g: Callable[[np.ndarray], np.ndarray], variables: VariableSet, tol: float = 1e-4,
         max_iter: int = 100, h: float = 1e-4) -> FORMResult:
    """FORM for one scalar limit state g(x (n,k)) -> (n,); see form_many."""
    return form_many(lambda x: np.asarray(g(x)).reshape(-1, 1), variables, [0], tol, max_iter, h)[0]


def importance_sampling(g: Callable[[np.ndarray], np.ndarray], variables: VariableSet, u_star: np.ndarray,
                        n: int = 20_000, seed: int = 0) -> tuple[float, float]:
    """Pf by importance sampling with a unit normal centred on the FORM design point.
    Returns (pf, coefficient of variation)."""
    rng = np.random.default_rng(seed)
    u = u_star + rng.standard_normal((n, len(u_star)))
    w = np.exp(-u @ u_star + 0.5 * float(u_star @ u_star))
    ind = np.asarray(g(variables.to_physical(u))).reshape(-1) <= 0
    terms = ind * w
    pf = float(terms.mean())
    cov = float(terms.std(ddof=1) / (math.sqrt(n) * pf)) if pf > 0 else math.inf
    return pf, cov


# --- Results per string and per well --------------------------------------------------------------

@dataclass(frozen=True)
class LimitStateResult:
    case: str
    check: str
    g_base: float
    mc_pf: float
    mc_ci: tuple[float, float]
    mc_cov: float
    form: FORMResult | None

    @property
    def pf(self) -> float:
        """Recommended Pf: Monte Carlo when its CoV meets the target, else FORM."""
        if self.mc_cov <= TARGET_COV or self.form is None or not self.form.converged:
            return self.mc_pf
        return self.form.pf

    @property
    def method(self) -> str:
        if self.mc_cov <= TARGET_COV or self.form is None or not self.form.converged:
            return "MC" if self.mc_pf > 0 else "MC (no failures)"
        return "FORM"

    @property
    def beta(self) -> float:
        if self.form is not None and self.form.converged:
            return self.form.beta
        p = self.pf
        return float(stats.norm.isf(p)) if 0 < p < 1 else (math.inf if p == 0 else -math.inf)


@dataclass
class StringReliability:
    string: str
    pipe: Pipe
    variables: VariableSet
    rows: list[LimitStateResult]
    mc: MCResult
    notes: list[str] = field(default_factory=list)

    @property
    def system_bounds(self) -> tuple[float, float]:
        """max(Pf_i) <= Pf_sys <= sum(Pf_i) (spec 4.3)."""
        pfs = [r.pf for r in self.rows]
        return (max(pfs, default=0.0), min(sum(pfs), 1.0))

    def meets(self, target_pf: float) -> bool:
        return all(r.pf <= target_pf for r in self.rows)

    def importance(self, case: str, check: str) -> dict[str, float]:
        """FORM importance factors alpha_i^2 (signed by alpha_i) for one limit state."""
        r = next(x for x in self.rows if x.case == case and x.check == check)
        if r.form is None:
            return {}
        return {v.name: float(np.sign(a) * a * a) for v, a in zip(self.variables.variables, r.form.alpha)}


def string_reliability(well: WellInput, profiles: Profiles, geometries: list[StringGeometry], name: str,
                       pipe: Pipe, n: int = DEFAULT_SAMPLES, seed: int = 0, max_n: int = DEFAULT_MAX_SAMPLES,
                       use_form: bool = True, yield_derate: Callable | None = None) -> StringReliability:
    """Pf per load case and check for one string (Monte Carlo, plus FORM where useful)."""
    variables = collect_variables(well, geometries, name)
    ls = StringLimitStates(well, profiles, geometries, name, pipe, variables, yield_derate)
    mc = monte_carlo(ls, variables, n=n, seed=seed, max_n=max_n)
    lo, hi = mc.ci()
    need_form = [j for j in range(len(ls.labels)) if use_form and mc.cov[j] > TARGET_COV]
    forms = form_many(ls, variables, need_form) if need_form else {}
    rows = [LimitStateResult(case, check, float(ls.base_g[j]), float(mc.pf[j]), (float(lo[j]), float(hi[j])),
                             float(mc.cov[j]), forms.get(j))
            for j, (case, check) in enumerate(ls.labels)]
    notes = list(variables.notes)
    if yield_derate is None:
        notes.append(f"{name}: no temperature derating of yield (manufacturer table not supplied)")
    notes.append(f"{name}: API TR 5C3 (serviceability) resistance model; the ISO/TR 10400 ultimate-limit-state "
                 "model is not built yet, so Pf is conservative and uneven across D/t (Ju et al. 1998)")
    return StringReliability(name, pipe, variables, rows, mc, notes)


@dataclass
class ShoeMove:
    string: str
    shoe_tvd_ft: float
    next_td_tvd_ft: float
    probability: float  # fraction of samples where this shoe would have to move shallower
    n: int


def shoe_move_probability(well: WellInput, profiles: Profiles, n: int = 10_000, seed: int = 0,
                          step_ft: float = 50.0) -> list[ShoeMove]:
    """How often each shoe would move (spec 3.1): with sampled pore, Poisson's
    ratio and overburden, the share of samples where the hole below the shoe
    can no longer be drilled to the next shoe (mud window or kick tolerance)."""
    from .design import _required_kick, kick_tolerance_ppg  # noqa: PLC0415

    depths = setting_depths(well, profiles)
    geoms, _ = resolve_strings(well, profiles.survey, profiles, depths.chosen_tvd_ft)
    g0 = geoms[0]
    variables = collect_variables(well, geoms, g0.name)
    keep = [v for v in variables.variables if v.name.split(".")[0] in ("pore", "poisson", "overburden")]
    vs = VariableSet(keep, np.eye(len(keep)))
    rng = np.random.default_rng(seed)
    x = vs.to_physical(rng.standard_normal((n, len(keep))))
    view = Realization({v.name: x[:, [i]] for i, v in enumerate(keep)}).profile_view(profiles)
    trip, kick = well.margins.trip_margin_ppg, well.margins.kick_margin_ppg
    out = []
    for g in geoms:
        if g.next_td_tvd_ft is None or g.kind in ("conductor", "tieback"):
            continue
        z = np.append(np.arange(g.shoe_tvd_ft, g.next_td_tvd_ft, step_ft), g.next_td_tvd_ft)
        need = required_mud_ppg(np.max(view.pore_ppg(z), axis=-1), trip)
        allow = np.min(view.frac_ppg(z), axis=-1) - kick
        fg_shoe = view.frac_ppg(g.shoe_tvd_ft).reshape(-1)
        kt = kick_tolerance_ppg(g.shoe_tvd_ft, fg_shoe, g.next_td_tvd_ft, np.reshape(need, -1))
        moved = (np.reshape(need, -1) > np.reshape(allow, -1) + 1e-9) | (kt < _required_kick(well, g.name, []) - 1e-9)
        out.append(ShoeMove(g.name, g.shoe_tvd_ft, g.next_td_tvd_ft, float(moved.mean()), n))
    return out


def well_reliability(well: WellInput, design: str | None = None, n: int = DEFAULT_SAMPLES, seed: int = 0,
                     max_n: int = DEFAULT_MAX_SAMPLES, use_form: bool = True) -> dict[str, StringReliability]:
    """Pf for every string that has a pipe in the chosen reference design."""
    from .design import design_pipes  # noqa: PLC0415

    profiles = build_profiles(well)
    depths = setting_depths(well, profiles)
    geoms, _ = resolve_strings(well, profiles.survey, profiles, depths.chosen_tvd_ft)
    _, pipes, _ = design_pipes(well, design)
    return {g.name: string_reliability(well, profiles, geoms, g.name, pipes[g.name], n=n, seed=seed,
                                       max_n=max_n, use_form=use_form)
            for g in geoms if g.name in pipes}


def target_sweep(results: dict[str, StringReliability],
                 targets: tuple[float, ...] = SWEEP_TARGETS) -> dict[float, dict[str, bool]]:
    """Whether each string meets each reliability target (spec 4.3 sweep)."""
    return {t: {name: r.meets(t) for name, r in results.items()} for t in targets}
