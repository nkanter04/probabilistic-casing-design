"""Load cases for casing design (spec Sections 3.2 and 3.5).

SIGN CONVENTIONS (CLAUDE.md rule 6):
    tension positive, compression negative
    burst load    = Pi - Pe   (internal minus external)
    collapse load = Pe - Pi   (external minus internal)

DEPTH: pressures use TVD; weights, bending and lengths use MD (rule 5).

Each load case is an internal profile Pi(z) and an external profile Pe(z),
evaluated at every node along the string, not only at the top and shoe.

The primitive functions (hydrostatic_psi, burst_backup_psi, ...) take numpy
arrays and broadcast, so the Monte Carlo layer can call them with sampled
inputs shaped (n_samples, 1) against nodes shaped (n_nodes,).
`string_loads` assembles the cases for one string at base values.

Every value this module has to assume is listed in StringLoads.notes, so the
report can print it (spec Section 5, output 6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .profiles import Profiles, ProfileView, ppg_to_psi
from .schema import (Cementing, CasingString, Depletion, Distribution, Evacuation, FracDownCasing,
                     GasKick, LostReturns, PressureTest, Running, StringLoadCases, TubingLeak,
                     WellInput)
from .trajectory import Survey
from .units import PSI_PER_FT_PER_PPG

# --- Constants (field units) ---------------------------------------------------

STEEL_DENSITY_PPG = 65.5  # buoyancy factor 1 - rho/65.5 (spec 3.5)
BENDING_PSI_PER_IN_PER_DEG_100FT = 218.0  # sigma_b = 218 D DLS (spec 3.5; E D kappa / 2 with E = 30e6 psi)
THERMAL_LBF_PER_IN2_PER_F = 207.0  # E * alpha = 30e6 psi * 6.9e-6 /F (spec 3.5)
BALLOONING_2NU = 0.6  # 2 * Poisson's ratio of steel, nu = 0.3 (Lubinski et al. 1962)

# Placeholders, used only when the input gives no value. Each use is noted.
PLACEHOLDER_GAS_GRADIENT_PSI_PER_FT = 0.10  # spec 2.2: 0.08-0.12 psi/ft placeholder
PLACEHOLDER_WATER_PPG = 8.33  # fresh water
PLACEHOLDER_MIX_WATER_PPG = 8.33  # spec 3.2 gives 8.3-8.6 ppg; the low end is the conservative burst backup
PLACEHOLDER_SLURRY_PPG = 15.8
PLACEHOLDER_OVERPULL_LBF = 100_000.0
# Shock load while running, lbf per lb/ft of pipe weight.
# TODO(citation): confirm 3,200 * w against Bourgoyne et al. (1986) Ch. 7 / Rabia (1987).
SHOCK_LBF_PER_LB_PER_FT = 3200.0
NODE_STEP_FT = 100.0

BURST: Literal["burst"] = "burst"
COLLAPSE: Literal["collapse"] = "collapse"

# Cases switched on by default for each kind when the input does not mention
# them (spec 3.2 table). Cases that need a user value (pressure test, frac,
# tubing leak, depletion) are only run when the input gives them.
DEFAULT_CASES = {
    "conductor": (),
    "surface": ("gas_kick", "lost_returns", "evacuation", "cementing", "running"),
    "intermediate": ("gas_kick", "lost_returns", "evacuation", "cementing", "running"),
    "production": ("evacuation", "cementing", "running"),
    "liner": ("evacuation", "cementing", "running"),
    "tieback": ("evacuation", "running"),
}
_CASE_MODELS = {"gas_kick": GasKick, "lost_returns": LostReturns, "pressure_test": PressureTest,
                "frac_down_casing": FracDownCasing, "tubing_leak": TubingLeak, "evacuation": Evacuation,
                "cementing": Cementing, "depletion": Depletion, "running": Running}


# --- Pipe and geometry ------------------------------------------------------------

@dataclass(frozen=True)
class PipeBody:
    """Pipe-body dimensions for axial calculations (one section)."""

    od_in: float
    id_in: float
    weight_lb_per_ft: float

    @classmethod
    def from_wall(cls, od_in: float, wall_in: float, weight_lb_per_ft: float) -> PipeBody:
        return cls(od_in, od_in - 2.0 * wall_in, weight_lb_per_ft)

    @property
    def area_outer_in2(self) -> float:
        return math.pi / 4.0 * self.od_in**2

    @property
    def area_inner_in2(self) -> float:
        return math.pi / 4.0 * self.id_in**2

    @property
    def area_steel_in2(self) -> float:
        return self.area_outer_in2 - self.area_inner_in2


@dataclass(frozen=True)
class StringGeometry:
    """One string with every depth resolved to numbers (ft)."""

    name: str
    kind: str
    od_in: float
    hole_in: float
    top_md_ft: float
    top_tvd_ft: float
    shoe_md_ft: float
    shoe_tvd_ft: float
    toc_md_ft: float
    toc_tvd_ft: float
    mud_ppg: float  # mud the string was run and cemented in
    outer_shoe_tvd_ft: float | None  # shoe of the next string outward
    next_td_tvd_ft: float | None  # TD of the next (deeper) hole section
    next_mud_ppg: float | None
    next_name: str | None = None


def resolve_strings(well: WellInput, survey: Survey, profiles: Profiles,
                    shoe_tvd_ft: dict[str, float] | None = None) -> tuple[list[StringGeometry], list[str]]:
    """Resolve every string's depths. Shoes marked "choose" must be given in shoe_tvd_ft.

    A missing mud weight is replaced by the highest pore pressure in the hole
    section plus the trip margin, rounded up to 0.1 ppg, and noted.
    """
    shoe_tvd_ft = shoe_tvd_ft or {}
    notes: list[str] = []
    resolved: list[dict] = []
    for s in well.strings:
        shoe_md, shoe_tvd = _resolve_shoe(s, survey, shoe_tvd_ft)
        top_tvd = float(survey.tvd_at_md(s.top_md_ft)[0])
        if s.toc_ft == "surface":
            toc_md = s.top_md_ft
        else:
            toc_md = min(max(float(s.toc_ft), s.top_md_ft), shoe_md)
            if toc_md != s.toc_ft:
                notes.append(f"{s.name}: toc_ft {s.toc_ft:g} outside the string; moved to {toc_md:.0f} ft MD")
        resolved.append(dict(name=s.name, kind=s.kind, od_in=s.od_in, hole_in=s.hole_in,
                             top_md_ft=s.top_md_ft, top_tvd_ft=top_tvd, shoe_md_ft=shoe_md,
                             shoe_tvd_ft=shoe_tvd, toc_md_ft=toc_md,
                             toc_tvd_ft=float(survey.tvd_at_md(toc_md)[0])))

    nested = [r for r in resolved if r["kind"] != "tieback"]
    for r in resolved:
        outer = [o for o in nested if o["od_in"] > r["od_in"] and o["shoe_tvd_ft"] <= r["shoe_tvd_ft"]]
        r["outer_shoe_tvd_ft"] = outer[-1]["shoe_tvd_ft"] if outer else None
    for r, s in zip(resolved, well.strings):
        if s.mud_ppg is not None:
            r["mud_ppg"] = _base(s.mud_ppg)
        else:
            z = np.linspace(r["outer_shoe_tvd_ft"] or 0.0, r["shoe_tvd_ft"], 200)
            mud = float(required_mud_ppg(profiles.pore_ppg(z).max(), well.margins.trip_margin_ppg))
            r["mud_ppg"] = mud
            notes.append(f"{s.name}: mud_ppg not given; used {mud:.1f} ppg "
                         f"(max pore pressure in the section + trip margin, placeholder)")
    for r in resolved:
        deeper = [d for d in nested if d["od_in"] < r["od_in"] and d["shoe_tvd_ft"] > r["shoe_tvd_ft"]]
        r["next_td_tvd_ft"] = deeper[0]["shoe_tvd_ft"] if deeper else None
        r["next_mud_ppg"] = deeper[0]["mud_ppg"] if deeper else None
        r["next_name"] = deeper[0]["name"] if deeper else None
    return [StringGeometry(**r) for r in resolved], notes


def required_mud_ppg(max_pore_ppg: ArrayLike, trip_margin_ppg: float) -> np.ndarray:
    """Mud weight for a hole section: highest pore pressure + trip margin, rounded up to 0.1 ppg.

    Rounds to 0.01 first so float noise (10.2003) does not round up a whole 0.1 ppg.
    """
    need = np.round(np.asarray(max_pore_ppg, dtype=float) + trip_margin_ppg, 2)
    return np.ceil(need * 10 - 1e-9) / 10


def _resolve_shoe(s: CasingString, survey: Survey, chosen: dict[str, float]) -> tuple[float, float]:
    value = s.shoe_value
    if value == "choose":
        if s.name not in chosen:
            raise ValueError(f"string {s.name!r} has shoe 'choose'; give its TVD (setting-depth design)")
        tvd = float(chosen[s.name])
        return survey.md_at_tvd(tvd), tvd
    if value == "TD":
        return survey.td_md_ft, survey.td_tvd_ft
    if s.shoe_md_ft is not None:
        return float(value), float(survey.tvd_at_md(value)[0])
    return survey.md_at_tvd(float(value)), float(value)


def node_md(geom: StringGeometry, survey: Survey, step_ft: float = NODE_STEP_FT) -> np.ndarray:
    """MD nodes along a string: a regular grid plus the top, shoe, TOC, the
    outer shoe and every survey station inside the string."""
    md = [np.arange(geom.top_md_ft, geom.shoe_md_ft, step_ft), [geom.shoe_md_ft, geom.toc_md_ft]]
    md.append(survey.md_ft[(survey.md_ft > geom.top_md_ft) & (survey.md_ft < geom.shoe_md_ft)])
    if geom.outer_shoe_tvd_ft is not None and geom.top_tvd_ft < geom.outer_shoe_tvd_ft < geom.shoe_tvd_ft:
        md.append([survey.md_at_tvd(geom.outer_shoe_tvd_ft)])
    return np.unique(np.concatenate([np.asarray(m, dtype=float) for m in md]))


# --- Pressure primitives (psi) ------------------------------------------------------

def hydrostatic_psi(ppg: ArrayLike, tvd_ft: ArrayLike) -> np.ndarray:
    return ppg_to_psi(ppg, tvd_ft)


def burst_backup_psi(tvd_ft: ArrayLike, mud_ppg: ArrayLike, toc_tvd_ft: float,
                     outer_shoe_tvd_ft: float | None, mix_water_ppg: ArrayLike,
                     pore_psi: ArrayLike) -> np.ndarray:
    """External (backup) pressure for burst (spec 3.2, Devon/EPA guidance):
    mud gradient to TOC, mix-water gradient from TOC to the outer shoe, and
    pore pressure below that (or below TOC if TOC is deeper)."""
    z = np.asarray(tvd_ft, dtype=float)
    p_toc = hydrostatic_psi(mud_ppg, toc_tvd_ft)
    p_mix = p_toc + PSI_PER_FT_PER_PPG * np.asarray(mix_water_ppg) * (z - toc_tvd_ft)
    mix_bottom = max(toc_tvd_ft, outer_shoe_tvd_ft) if outer_shoe_tvd_ft is not None else toc_tvd_ft
    return np.where(z <= toc_tvd_ft, hydrostatic_psi(mud_ppg, z),
                    np.where(z <= mix_bottom, p_mix, np.asarray(pore_psi)))


def collapse_backup_psi(tvd_ft: ArrayLike, mud_ppg: ArrayLike, pore_psi: ArrayLike) -> np.ndarray:
    """External pressure for collapse: the mud the string was run in (Bourgoyne
    et al. 1986, Ch. 7), but never less than pore pressure."""
    return np.maximum(hydrostatic_psi(mud_ppg, tvd_ft), np.asarray(pore_psi))


def gas_to_surface_psi(tvd_ft: ArrayLike, p_reservoir_psi: ArrayLike, reservoir_tvd_ft: float,
                       gas_gradient_psi_per_ft: ArrayLike) -> np.ndarray:
    """Gas column from a kick at reservoir_tvd_ft up to surface: P_res - g_gas (TVD_res - z)."""
    z = np.asarray(tvd_ft, dtype=float)
    p = np.asarray(p_reservoir_psi) - np.asarray(gas_gradient_psi_per_ft) * (reservoir_tvd_ft - z)
    return np.maximum(p, 0.0)


def column_below_psi(tvd_ft: ArrayLike, p_ref_psi: ArrayLike, ref_tvd_ft: float, ppg: ArrayLike) -> np.ndarray:
    """Fluid column hanging from a reference pressure at ref_tvd_ft, floored at 0."""
    z = np.asarray(tvd_ft, dtype=float)
    p = np.asarray(p_ref_psi) - PSI_PER_FT_PER_PPG * np.asarray(ppg) * (ref_tvd_ft - z)
    return np.maximum(p, 0.0)


def surface_pressure_column_psi(tvd_ft: ArrayLike, p_surface_psi: ArrayLike, ppg: ArrayLike) -> np.ndarray:
    """Surface pressure on a static fluid column: P_s + 0.052 rho TVD.

    Used for injection down casing (spec 3.2: the frac load grows with depth;
    friction is not subtracted), the pressure test and the tubing leak.
    """
    return np.asarray(p_surface_psi) + hydrostatic_psi(ppg, tvd_ft)


def evacuated_psi(tvd_ft: ArrayLike, level_tvd_ft: ArrayLike | None, ppg: ArrayLike) -> np.ndarray:
    """Fluid below a level (TVD); None means full evacuation (Pi = 0)."""
    z = np.asarray(tvd_ft, dtype=float)
    if level_tvd_ft is None:
        return np.zeros_like(z)
    return PSI_PER_FT_PER_PPG * np.asarray(ppg) * np.maximum(z - np.asarray(level_tvd_ft), 0.0)


def cementing_external_psi(tvd_ft: ArrayLike, mud_ppg: ArrayLike, toc_tvd_ft: float,
                           slurry_ppg: ArrayLike) -> np.ndarray:
    """Wet cement outside: mud above TOC, slurry column from TOC down."""
    z = np.asarray(tvd_ft, dtype=float)
    p_toc = hydrostatic_psi(mud_ppg, toc_tvd_ft)
    return np.where(z <= toc_tvd_ft, hydrostatic_psi(mud_ppg, z),
                    p_toc + PSI_PER_FT_PER_PPG * np.asarray(slurry_ppg) * (z - toc_tvd_ft))


# --- Axial primitives -----------------------------------------------------------------

def buoyancy_factor(fluid_ppg: ArrayLike) -> np.ndarray:
    return 1.0 - np.asarray(fluid_ppg) / STEEL_DENSITY_PPG


def effective_tension_lbf(tvd_ft: ArrayLike, shoe_tvd_ft: float, weight_lb_per_ft: float,
                          fluid_ppg: ArrayLike = 0.0) -> np.ndarray:
    """Buoyed weight of the pipe below each node (spec 3.5).

    Without friction, the axial component of weight over a length of hole is
    w * delta-TVD, so w * BF * (TVD_shoe - TVD) holds in any trajectory.
    With fluid_ppg = 0 this is the dry hanging weight.
    """
    return weight_lb_per_ft * buoyancy_factor(fluid_ppg) * (shoe_tvd_ft - np.asarray(tvd_ft, dtype=float))


def real_axial_lbf(effective_lbf: ArrayLike, p_internal_psi: ArrayLike, p_external_psi: ArrayLike,
                   pipe: PipeBody) -> np.ndarray:
    """True axial force from effective tension: F = F_eff + Pi Ai - Pe Ao (Lubinski et al. 1962)."""
    return (np.asarray(effective_lbf) + np.asarray(p_internal_psi) * pipe.area_inner_in2
            - np.asarray(p_external_psi) * pipe.area_outer_in2)


def bending_stress_psi(od_in: float, dls_deg_per_100ft: ArrayLike) -> np.ndarray:
    """Bending stress at the outer fibre, sigma_b = 218 D DLS (spec 3.5)."""
    return BENDING_PSI_PER_IN_PER_DEG_100FT * od_in * np.asarray(dls_deg_per_100ft)


def thermal_force_lbf(area_steel_in2: float, delta_t_F: ArrayLike) -> np.ndarray:
    """Axial force change of pipe fixed at both ends: -207 A_s dT (cooling adds tension)."""
    return -THERMAL_LBF_PER_IN2_PER_F * area_steel_in2 * np.asarray(delta_t_F)


def ballooning_force_lbf(delta_pi_psi: ArrayLike, delta_po_psi: ArrayLike, pipe: PipeBody) -> np.ndarray:
    """Axial force change of pipe fixed at both ends: 0.6 (dPi Ai - dPo Ao) (Lubinski et al. 1962).
    More pressure inside makes the pipe balloon, shorten and so pull in tension."""
    return BALLOONING_2NU * (np.asarray(delta_pi_psi) * pipe.area_inner_in2
                             - np.asarray(delta_po_psi) * pipe.area_outer_in2)


def shock_load_lbf(weight_lb_per_ft: float) -> float:
    return SHOCK_LBF_PER_LB_PER_FT * weight_lb_per_ft


# --- Assembled loads for one string -------------------------------------------------------

@dataclass(frozen=True)
class LoadCase:
    """One pressure load case on one string, at every node."""

    name: str
    mode: Literal["burst", "collapse"]
    md_ft: np.ndarray
    tvd_ft: np.ndarray
    p_internal_psi: np.ndarray
    p_external_psi: np.ndarray
    axial_real_lbf: np.ndarray | None = None  # with service changes; None without a pipe

    @property
    def load_psi(self) -> np.ndarray:
        """Net load: Pi - Pe for burst, Pe - Pi for collapse."""
        if self.mode == BURST:
            return self.p_internal_psi - self.p_external_psi
        return self.p_external_psi - self.p_internal_psi


@dataclass(frozen=True)
class AxialCase:
    """An axial-only case (installation, running)."""

    name: str
    md_ft: np.ndarray
    tvd_ft: np.ndarray
    real_lbf: np.ndarray
    effective_lbf: np.ndarray


@dataclass(frozen=True)
class StringLoads:
    geometry: StringGeometry
    md_ft: np.ndarray
    tvd_ft: np.ndarray
    temperature_F: np.ndarray
    dls_deg_per_100ft: np.ndarray
    cases: list[LoadCase]
    axial: list[AxialCase] = field(default_factory=list)
    bending_stress_psi: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    def case(self, name: str) -> LoadCase:
        return next(c for c in self.cases if c.name == name)


def active_cases(kind: str, settings: StringLoadCases | None) -> dict[str, object]:
    """Load cases to run for a string: kind defaults, then the input's on/off/settings."""
    out: dict[str, object] = {name: _CASE_MODELS[name]() for name in DEFAULT_CASES[kind]}
    if settings is not None:
        for name, value in settings:
            if value is False:
                out.pop(name, None)
            elif value is not None:
                out[name] = value
    return out


@dataclass(frozen=True)
class Realization:
    """Sampled values for uncertain inputs, each shaped (n, 1) (Monte Carlo batch
    or FORM points). Keys are input paths such as "strings.surface.mud_ppg",
    "fluids.mix_water_ppg" or "load_cases.production.frac_down_casing.max_surface_psi",
    plus the profile modifiers in PROFILE_KEYS. Inputs that are not sampled use
    their base values, so loads with an empty Realization equal the deterministic loads."""

    values: dict[str, np.ndarray] = field(default_factory=dict)

    def profile_view(self, profiles: Profiles) -> ProfileView:
        v = self.values
        return ProfileView(profiles, v.get("pore.shift_ppg", 0.0), v.get("pore.scale", 1.0),
                           v.get("poisson.shift", 0.0), v.get("poisson.scale", 1.0),
                           v.get("overburden.scale", 1.0), v.get("temperature.surface_F"),
                           v.get("temperature.gradient_F_per_100ft"))


PROFILE_KEYS = ("pore.shift_ppg", "pore.scale", "poisson.shift", "poisson.scale", "overburden.scale",
                "temperature.surface_F", "temperature.gradient_F_per_100ft")


def string_loads(well: WellInput, profiles: Profiles, geometries: list[StringGeometry], name: str,
                 pipe: PipeBody | None = None, realization: Realization | None = None) -> StringLoads:
    """All load cases for one string. With a pipe, also the axial loads.

    Without a realization every input is at its base value and each profile is
    shaped (n_nodes,). With one, sampled inputs are (n, 1) arrays and the
    profiles broadcast to (n, n_nodes).
    """
    g = next(x for x in geometries if x.name == name)
    real = realization or Realization()
    prof = real.profile_view(profiles) if realization is not None else profiles
    survey = profiles.survey
    md = node_md(g, survey)
    pts = survey.at_md(md)
    tvd = pts.tvd_ft
    notes: list[str] = []
    fl = well.fluids

    def value(path: str, x, default=None, label: str | None = None):
        """Sampled value at `path` if x is uncertain and sampled, else x's base value."""
        if x is None:
            if label is not None:
                notes.append(f"{label} not given; used {default:g} (placeholder)")
            return default
        if isinstance(x, Distribution) and path in real.values:
            return real.values[path]
        return _base(x)

    spec = next(s for s in well.strings if s.name == name)
    mud = value(f"strings.{name}.mud_ppg", spec.mud_ppg, g.mud_ppg)
    next_mud = None
    if g.next_name is not None:
        nxt = next(s for s in well.strings if s.name == g.next_name)
        next_mud = value(f"strings.{g.next_name}.mud_ppg", nxt.mud_ppg, g.next_mud_ppg)
    pore = prof.pore_psi(tvd)
    mix_water = value("fluids.mix_water_ppg", fl.mix_water_ppg, PLACEHOLDER_MIX_WATER_PPG, "fluids.mix_water_ppg")
    burst_pe = burst_backup_psi(tvd, mud, g.toc_tvd_ft, g.outer_shoe_tvd_ft, mix_water, pore)
    collapse_pe = collapse_backup_psi(tvd, mud, pore)
    completion = value("fluids.completion_ppg", fl.completion_ppg)
    cases: list[LoadCase] = []
    settings = active_cases(g.kind, well.load_cases.get(name))

    def add(case: str, mode: str, pi, pe) -> None:
        shape = np.broadcast_shapes(np.shape(pi), np.shape(pe), tvd.shape)
        cases.append(LoadCase(case, mode, md, tvd, np.broadcast_to(pi, shape).copy(),
                              np.broadcast_to(pe, shape).copy()))

    for case, cfg in settings.items():
        path = f"load_cases.{name}.{case}"
        if case == "gas_kick":
            if g.next_td_tvd_ft is None:
                notes.append(f"{name}: gas_kick skipped (no deeper hole section)")
                continue
            grad = value(f"{path}.gas_gradient_psi_per_ft", cfg.gas_gradient_psi_per_ft,
                         PLACEHOLDER_GAS_GRADIENT_PSI_PER_FT, f"{name}.gas_kick.gas_gradient_psi_per_ft")
            ki = value(f"{path}.kick_intensity_ppg", cfg.kick_intensity_ppg, 0.0)
            if cfg.kick_volume_bbl is not None:
                notes.append(f"{name}: kick volume model not built yet; used full gas to surface (conservative)")
            p_res = hydrostatic_psi(prof.pore_ppg(g.next_td_tvd_ft) + ki, g.next_td_tvd_ft)
            pi = gas_to_surface_psi(tvd, p_res, g.next_td_tvd_ft, grad)
            if cfg.limit_to_shoe_frac:
                p_shoe = np.minimum(gas_to_surface_psi(g.shoe_tvd_ft, p_res, g.next_td_tvd_ft, grad),
                                    prof.frac_psi(g.shoe_tvd_ft))
                pi = gas_to_surface_psi(tvd, p_shoe, g.shoe_tvd_ft, grad)
            add(case, BURST, pi, burst_pe)
        elif case == "lost_returns":
            if g.next_td_tvd_ft is None:
                notes.append(f"{name}: lost_returns skipped (no deeper hole section)")
                continue
            water = value(f"{path}.fluid_ppg", cfg.fluid_ppg, PLACEHOLDER_WATER_PPG, f"{name}.lost_returns.fluid_ppg")
            add(case, BURST, column_below_psi(tvd, prof.frac_psi(g.shoe_tvd_ft), g.shoe_tvd_ft, water), burst_pe)
        elif case == "pressure_test":
            fluid = value(f"{path}.fluid_ppg", cfg.fluid_ppg, _default_fluid(g, mud, completion))
            add(case, BURST, surface_pressure_column_psi(tvd, value(f"{path}.surface_psi", cfg.surface_psi), fluid),
                burst_pe)
        elif case == "frac_down_casing":
            add(case, BURST, surface_pressure_column_psi(tvd, value(f"{path}.max_surface_psi", cfg.max_surface_psi),
                                                         value(f"{path}.fluid_ppg", cfg.fluid_ppg)), burst_pe)
        elif case == "tubing_leak":
            fluid = value(f"{path}.packer_fluid_ppg", cfg.packer_fluid_ppg,
                          completion if completion is not None else mud)
            if cfg.packer_fluid_ppg is None:
                notes.append(f"{name}: tubing_leak packer fluid not given; used "
                             f"{'completion fluid' if completion is not None else 'mud'} (placeholder)")
            add(case, BURST, surface_pressure_column_psi(tvd, value(f"{path}.surface_psi", cfg.surface_psi), fluid),
                burst_pe)
        elif case == "evacuation":
            level = value(f"{path}.level_ft", cfg.level_ft)
            fluid = next(f for f in (next_mud, completion, mud) if f is not None)
            add(case, COLLAPSE, evacuated_psi(tvd, level, fluid), collapse_pe)
        elif case == "cementing":
            slurries = [value(f"fluids.{k}", x) for k, x in
                        (("cement_lead_ppg", fl.cement_lead_ppg), ("cement_tail_ppg", fl.cement_tail_ppg))
                        if x is not None]
            if not slurries:
                notes.append(f"{name}: no cement density given; used {PLACEHOLDER_SLURRY_PPG} ppg (placeholder)")
            slurry = PLACEHOLDER_SLURRY_PPG if not slurries else (
                slurries[0] if len(slurries) == 1 else np.maximum(*slurries))
            disp = value(f"{path}.displacement_ppg", cfg.displacement_ppg, mud)
            add(case, COLLAPSE, hydrostatic_psi(disp, tvd), cementing_external_psi(tvd, mud, g.toc_tvd_ft, slurry))
        elif case == "depletion":
            fluid = value("fluids.completion_ppg", fl.completion_ppg, PLACEHOLDER_WATER_PPG,
                          f"{name}.depletion fluid (completion_ppg)")
            p_dep = hydrostatic_psi(value(f"{path}.pore_ppg", cfg.pore_ppg), g.shoe_tvd_ft)
            pi = column_below_psi(tvd, p_dep, g.shoe_tvd_ft, fluid)
            if cfg.annulus_level_ft is not None:
                pi = np.minimum(pi, evacuated_psi(tvd, value(f"{path}.annulus_level_ft", cfg.annulus_level_ft), fluid))
            add(case, COLLAPSE, pi, collapse_pe)

    loads = StringLoads(geometry=g, md_ft=md, tvd_ft=tvd, temperature_F=prof.temperature_F(tvd),
                        dls_deg_per_100ft=pts.dls_deg_per_100ft, cases=cases, notes=notes)
    if pipe is None:
        return loads
    return _with_axial(loads, settings, pipe, mud, prof.temperature_F(0.0),
                       float(profiles.temperature_F(0.0)), value, notes)


def _with_axial(loads: StringLoads, settings: dict, pipe: PipeBody, mud, surface_F, surface_F_base: float,
                value, notes: list[str]) -> StringLoads:
    """Add installation, running and service axial loads for one pipe."""
    g, md, tvd = loads.geometry, loads.md_ft, loads.tvd_ft
    p_ref = hydrostatic_psi(mud, tvd)  # inside and outside when the cement sets
    eff0 = effective_tension_lbf(tvd, g.shoe_tvd_ft, pipe.weight_lb_per_ft, mud)
    real0 = real_axial_lbf(eff0, p_ref, p_ref, pipe)
    axial = [AxialCase("installation", md, tvd, real0, eff0)]

    run = settings.get("running")
    if run is not None:
        overpull = run.overpull_lbf
        if overpull is None:
            overpull = PLACEHOLDER_OVERPULL_LBF
            notes.append(f"{g.name}: running.overpull_lbf not given; used {overpull:,.0f} lbf (placeholder)")
        extra = max(overpull, shock_load_lbf(pipe.weight_lb_per_ft) if run.shock else 0.0)
        axial.append(AxialCase("running", md, tvd, real0 + extra, eff0 + extra))

    # Service changes act on the free pipe between the top and TOC, held at both ends.
    free = md <= g.toc_md_ft
    cases = []
    for c in loads.cases:
        d_force = 0.0
        if free.sum() >= 2:
            d_force = ballooning_force_lbf(_avg(md, c.p_internal_psi - p_ref, free),
                                           _avg(md, c.p_external_psi - p_ref, free), pipe)
            if c.name == "frac_down_casing":
                cfg = settings["frac_down_casing"]
                t_inj = value(f"load_cases.{g.name}.frac_down_casing.fluid_temp_F", cfg.fluid_temp_F, surface_F)
                if cfg.fluid_temp_F is None:
                    notes.append(f"{g.name}: frac fluid temperature not given; used surface temperature "
                                 f"{surface_F_base:.0f} F (placeholder, maximum cooling)")
                d_force = d_force + thermal_force_lbf(pipe.area_steel_in2,
                                                      _avg(md, t_inj - loads.temperature_F, free))
        cases.append(LoadCase(c.name, c.mode, md, tvd, c.p_internal_psi, c.p_external_psi,
                              real0 + np.where(free, d_force, 0.0)))
    return StringLoads(geometry=g, md_ft=md, tvd_ft=tvd, temperature_F=loads.temperature_F,
                       dls_deg_per_100ft=loads.dls_deg_per_100ft, cases=cases, axial=axial,
                       bending_stress_psi=bending_stress_psi(pipe.od_in, loads.dls_deg_per_100ft),
                       notes=notes)


def _avg(md: np.ndarray, y: ArrayLike, mask: np.ndarray) -> np.ndarray:
    """MD-weighted average of y over the masked nodes (last axis), kept as (..., 1)."""
    x = md[mask]
    v = np.broadcast_to(y, np.broadcast_shapes(np.shape(y), md.shape))[..., mask]
    return np.trapezoid(v, x, axis=-1)[..., None] / (x[-1] - x[0])


def _default_fluid(g: StringGeometry, mud, completion):
    if g.kind in ("production", "liner", "tieback") and completion is not None:
        return completion
    return mud


def _base(x: float | Distribution) -> float:
    return x.base if isinstance(x, Distribution) else float(x)
