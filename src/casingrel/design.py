"""Setting depths (spec 3.1) and the deterministic design-factor check (spec 3.6).

Setting depths: the operational mud window is pore pressure + trip margin
(lowest usable mud) to fracture gradient - kick margin (highest usable mud).
The top-down method sets each "choose" shoe as deep as the open hole below
the previous shoe can be drilled: the mud needed for the deepest pore
pressure must stay below the weakest fracture gradient in that open hole.
Each shoe is then checked for kick tolerance.

Design check: for each string, node and load case, design factor =
rating / load, compared with the input's design factors. Signs follow
loads.py: tension positive, burst load Pi - Pe, collapse load Pe - Pi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import ratings as rt
from .loads import (PipeBody, StringGeometry, StringLoads, TaperedBody, required_mud_ppg, resolve_strings,
                    section_index, string_loads)
from .profiles import Profiles, build_profiles
from .ratings import Pipe
from .schema import Distribution, ReferenceDesign, WellInput
from .units import PSI_PER_FT_PER_PPG

SETTING_DEPTH_STEP_FT = 10.0
# Kick intensity each shoe must tolerate when the input gives none (placeholder, noted).
PLACEHOLDER_KICK_INTENSITY_PPG = 0.5


# --- Setting depths ------------------------------------------------------------------

@dataclass(frozen=True)
class MudWindow:
    tvd_ft: np.ndarray
    min_mud_ppg: np.ndarray  # pore + trip margin
    max_mud_ppg: np.ndarray  # frac - kick margin


def mud_window(profiles: Profiles, trip_margin_ppg: float, kick_margin_ppg: float,
               tvd_ft: np.ndarray) -> MudWindow:
    z = np.asarray(tvd_ft, dtype=float)
    return MudWindow(z, profiles.pore_ppg(z) + trip_margin_ppg, profiles.frac_ppg(z) - kick_margin_ppg)


def deepest_reachable_tvd(profiles: Profiles, trip_margin_ppg: float, kick_margin_ppg: float,
                          shoe_tvd_ft: float, limit_tvd_ft: float, kick_intensity_ppg: float | None = None,
                          influx_height_ft: float = 0.0, gas_gradient_psi_per_ft: float = 0.1,
                          step_ft: float = SETTING_DEPTH_STEP_FT) -> float:
    """Deepest TVD (on a step_ft grid, rounded down) that can be drilled below a shoe.

    Open hole [shoe, z] is drillable when the mud needed for its highest pore
    pressure is no heavier than its weakest fracture gradient allows:
    max(pore + trip) <= min(frac - kick) over the open hole, shoe included.
    With kick_intensity_ppg, the shoe must also tolerate that kick taken at z.
    """
    z = np.append(np.arange(shoe_tvd_ft, limit_tvd_ft, step_ft), limit_tvd_ft)
    w = mud_window(profiles, trip_margin_ppg, kick_margin_ppg, z)
    # Mud for drilling to z, in the 0.1 ppg steps that resolve_strings will use.
    need = required_mud_ppg(np.maximum.accumulate(profiles.pore_ppg(z)), trip_margin_ppg)
    allow = np.minimum.accumulate(w.max_mud_ppg)
    ok = need <= allow + 1e-9
    if kick_intensity_ppg is not None and shoe_tvd_ft > 0:
        kt = kick_tolerance_ppg(shoe_tvd_ft, float(profiles.frac_ppg(shoe_tvd_ft)), np.maximum(z, 1e-9),
                                need, influx_height_ft, gas_gradient_psi_per_ft)
        ok &= (kt >= kick_intensity_ppg - 1e-9) | (z <= shoe_tvd_ft)
    ok = np.logical_and.accumulate(ok)
    if ok.all():
        return float(limit_tvd_ft)
    first_bad = int(np.argmin(ok))
    return float(z[first_bad - 1]) if first_bad > 0 else float(shoe_tvd_ft)


@dataclass(frozen=True)
class KickTolerance:
    string: str
    shoe_tvd_ft: float
    next_td_tvd_ft: float
    next_mud_ppg: float
    tolerance_ppg: float  # largest kick intensity the shoe can take
    required_ppg: float

    @property
    def ok(self) -> bool:
        return self.tolerance_ppg >= self.required_ppg - 1e-9


def kick_tolerance_ppg(shoe_tvd_ft: float, frac_shoe_ppg: float, next_td_tvd_ft: float, mud_ppg: float,
                       influx_height_ft: float = 0.0, gas_gradient_psi_per_ft: float = 0.1) -> float:
    """Largest kick intensity (ppg over the mud) that can be shut in without breaking down the shoe.

    KT = (D_shoe / D_td)(FG_shoe - MW) - (h / D_td)(MW - rho_gas), with an
    influx of height h at TD (h = 0 gives the intensity-only form).
    TODO(citation): confirm the form and equation number against Bourgoyne et al. (1986).
    """
    rho_gas_ppg = gas_gradient_psi_per_ft / PSI_PER_FT_PER_PPG
    return ((shoe_tvd_ft / next_td_tvd_ft) * (frac_shoe_ppg - mud_ppg)
            - (influx_height_ft / next_td_tvd_ft) * (mud_ppg - rho_gas_ppg))


@dataclass
class SettingDepthResult:
    chosen_tvd_ft: dict[str, float]  # shoe TVD for each "choose" string
    shoe_tvd_ft: dict[str, float]  # every nested string's shoe
    kick_tolerance: list[KickTolerance]
    warnings: list[str] = field(default_factory=list)


def setting_depths(well: WellInput, profiles: Profiles, influx_height_ft: float = 0.0) -> SettingDepthResult:
    """Top-down setting depths for "choose" shoes, with kick tolerance.

    Each chosen shoe goes as deep as the mud window allows below the shoe
    above it, and no deeper than that shoe's kick tolerance allows (bottom-up
    kick-tolerance check, spec 3.1). Fixed shoes are kept and only checked.
    """
    survey = profiles.survey
    trip, kick = well.margins.trip_margin_ppg, well.margins.kick_margin_ppg
    nested = [s for s in well.strings if s.kind != "tieback"]

    def fixed_tvd(s) -> float | None:
        v = s.shoe_value
        if v == "choose":
            return None
        if v == "TD":
            return survey.td_tvd_ft
        return float(survey.tvd_at_md(v)[0]) if s.shoe_md_ft is not None else float(v)

    fixed = [fixed_tvd(s) for s in nested]
    chosen: dict[str, float] = {}
    shoes: dict[str, float] = {}
    warnings: list[str] = []
    prev_name, prev_tvd, prev_kick = "surface", 0.0, (None, 0.1)
    for i, s in enumerate(nested):
        reach = deepest_reachable_tvd(profiles, trip, kick, prev_tvd, survey.max_tvd_ft, prev_kick[0],
                                      influx_height_ft, prev_kick[1])
        if fixed[i] is not None:
            shoe = fixed[i]
            if shoe > reach + 1e-6:
                warnings.append(f"{s.name}: shoe at {shoe:.0f} ft TVD is below the deepest depth drillable "
                                f"from {prev_name} ({reach:.0f} ft TVD); an extra string or a shallower shoe is needed")
        else:
            limit = next((f for f in fixed[i + 1:] if f is not None), survey.max_tvd_ft)
            shoe = min(reach, limit)
            if shoe >= limit - 1e-6:
                warnings.append(f"{s.name}: the hole below {prev_name} reaches {limit:.0f} ft TVD without it; "
                                f"this string is not needed by the mud window")
            if shoe <= prev_tvd + 1e-6:
                warnings.append(f"{s.name}: no drillable depth below {prev_name} at {prev_tvd:.0f} ft TVD")
            chosen[s.name] = shoe
        shoes[s.name] = shoe
        prev_name, prev_tvd = s.name, shoe
        has_deeper = i + 1 < len(nested)
        prev_kick = (_required_kick(well, s.name, warnings) if has_deeper and s.kind != "conductor" else None,
                     _gas_gradient(well, s.name))

    geoms, notes = resolve_strings(well, survey, profiles, chosen)
    by_name = {g.name: g for g in geoms}
    for s in nested:
        g = by_name[s.name]
        z = np.linspace(g.outer_shoe_tvd_ft or 0.0, g.shoe_tvd_ft, 200)
        need = float(required_mud_ppg(profiles.pore_ppg(z).max(), trip))
        if s.mud_ppg is not None and g.mud_ppg < need - 1e-9:
            warnings.append(f"{s.name}: mud {g.mud_ppg:g} ppg is below pore pressure + trip margin "
                            f"({need:.1f} ppg) in its hole section")
    kts = []
    for s in nested:
        g = by_name[s.name]
        if g.next_td_tvd_ft is None or s.kind == "conductor":
            continue
        required = _required_kick(well, s.name, [])
        grad = _gas_gradient(well, s.name)
        kt = kick_tolerance_ppg(g.shoe_tvd_ft, float(profiles.frac_ppg(g.shoe_tvd_ft)), g.next_td_tvd_ft,
                                g.next_mud_ppg, influx_height_ft, grad)
        kts.append(KickTolerance(s.name, g.shoe_tvd_ft, g.next_td_tvd_ft, g.next_mud_ppg, kt, required))
        if kt < required - 1e-9:
            warnings.append(f"{s.name}: kick tolerance {kt:.2f} ppg is below the required {required:.2f} ppg")
    return SettingDepthResult(chosen, shoes, kts, warnings + notes)


# --- Stresses for the triaxial check ------------------------------------------------------

def lame_stresses_psi(p_internal_psi, p_external_psi, r_inner_in: float, r_outer_in: float, r_in):
    """Lame radial and hoop stress at radius r in a thick cylinder (compression negative).

    sigma_r = A - B/r^2, sigma_theta = A + B/r^2, with
    A = (Pi ri^2 - Po ro^2) / (ro^2 - ri^2), B = (Pi - Po) ri^2 ro^2 / (ro^2 - ri^2).
    """
    pi, po = np.asarray(p_internal_psi, dtype=float), np.asarray(p_external_psi, dtype=float)
    ri2, ro2 = r_inner_in**2, r_outer_in**2
    a = (pi * ri2 - po * ro2) / (ro2 - ri2)
    b = (pi - po) * ri2 * ro2 / (ro2 - ri2)
    r2 = np.asarray(r_in, dtype=float) ** 2
    return a - b / r2, a + b / r2


def von_mises_psi(sigma_r, sigma_theta, sigma_z):
    """Von Mises equivalent stress (spec 3.4)."""
    return np.sqrt(0.5 * ((sigma_z - sigma_theta) ** 2 + (sigma_theta - sigma_r) ** 2 + (sigma_r - sigma_z) ** 2))


def triaxial_vme_inner_psi(pipe: Pipe, p_internal_psi, p_external_psi, axial_lbf, bending_psi):
    """Largest VME stress at the inner wall, with bending added and subtracted."""
    return _vme_inner(pipe.od_in, pipe.id_in, pipe.area_steel_in2, p_internal_psi, p_external_psi,
                      axial_lbf, bending_psi)


def _vme_inner(od_in, id_in, area_steel_in2, p_internal_psi, p_external_psi, axial_lbf, bending_psi):
    ri, ro = np.asarray(id_in) / 2.0, np.asarray(od_in) / 2.0
    s_r, s_t = lame_stresses_psi(p_internal_psi, p_external_psi, ri, ro, ri)
    s_z = np.asarray(axial_lbf) / area_steel_in2
    return np.maximum(von_mises_psi(s_r, s_t, s_z + bending_psi), von_mises_psi(s_r, s_t, s_z - bending_psi))


# --- String designs (one pipe, or a tapered string) ----------------------------------------

@dataclass(frozen=True)
class StringDesign:
    """Pipes for one string, top section first. crossovers_md_ft[i] is the
    bottom of section i; a single pipe has no crossovers."""

    pipes: tuple[Pipe, ...]
    crossovers_md_ft: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.crossovers_md_ft) != len(self.pipes) - 1:
            raise ValueError("a string design needs one crossover depth between each pair of pipes")

    @classmethod
    def of(cls, x: Pipe | StringDesign) -> StringDesign:
        return x if isinstance(x, StringDesign) else cls((x,))

    @property
    def tapered(self) -> bool:
        return len(self.pipes) > 1

    @property
    def label(self) -> str:
        if not self.tapered:
            return self.pipes[0].label
        parts = [f"{p.label} to {x:,.0f} ft" for p, x in zip(self.pipes, self.crossovers_md_ft)]
        return " / ".join(parts + [f"{self.pipes[-1].label} below"])

    def body(self) -> PipeBody | TaperedBody:
        bodies = tuple(PipeBody(p.od_in, p.id_in, p.weight_lb_per_ft) for p in self.pipes)
        return bodies[0] if not self.tapered else TaperedBody(bodies, self.crossovers_md_ft)

    def section_lengths_ft(self, top_md_ft: float, shoe_md_ft: float) -> list[float]:
        edges = [top_md_ft, *self.crossovers_md_ft, shoe_md_ft]
        return [b - a for a, b in zip(edges, edges[1:])]

    def at(self, md_ft: np.ndarray) -> NodeRatings:
        i = section_index(md_ft, self.crossovers_md_ft)

        def pick(attr: str) -> np.ndarray:
            vals = [getattr(p, attr) for p in self.pipes]
            return np.array([np.nan if v is None else v for v in vals], dtype=float)[i]

        regime_yield = np.array([rt.collapse_regime(p.od_in, p.wall_in, p.yield_psi) == rt.YIELD
                                 for p in self.pipes])[i]
        return NodeRatings(pick("od_in"), pick("wall_in"), pick("yield_psi"), pick("conn_internal_psi"),
                           pick("conn_tension_lbf"), pick("conn_compression_lbf"), ~regime_yield)


@dataclass(frozen=True)
class NodeRatings:
    """Nominal pipe properties at every node; missing connection ratings are NaN."""

    od_in: np.ndarray
    wall_in: np.ndarray
    yield_psi: np.ndarray
    conn_internal_psi: np.ndarray
    conn_tension_lbf: np.ndarray
    conn_compression_lbf: np.ndarray
    thin_wall: np.ndarray  # True where D/t is outside the yield-collapse range

    @property
    def id_in(self) -> np.ndarray:
        return self.od_in - 2.0 * self.wall_in

    @property
    def area_steel_in2(self) -> np.ndarray:
        return np.pi / 4.0 * (self.od_in**2 - self.id_in**2)


# --- Design-factor check ----------------------------------------------------------------------

@dataclass(frozen=True)
class CheckRow:
    string: str
    case: str
    check: str  # burst, collapse, tension, compression, triaxial
    design_factor: float  # minimum over nodes; inf when the case puts no load of this kind
    required: float
    md_ft: float | None
    tvd_ft: float | None

    @property
    def ok(self) -> bool:
        return self.design_factor >= self.required - 1e-9


@dataclass
class StringCheck:
    string: str
    design: StringDesign
    rows: list[CheckRow]
    loads: StringLoads
    max_surface_treating_psi: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.rows)

    def governing(self, check: str) -> CheckRow | None:
        rows = [r for r in self.rows if r.check == check]
        return min(rows, key=lambda r: r.design_factor / r.required) if rows else None


YieldDerate = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Demand:
    """Rating (capacity) and load for one case and check, at every node.

    Shapes are (n_nodes,) in the deterministic run and (n, n_nodes) with
    samples. The design factor is rating / load; the limit state is
    g = rating - load (spec 4.1). Both come from this one function, so the
    deterministic and probabilistic answers cannot disagree.
    """

    case: str
    check: str
    rating: np.ndarray
    load: np.ndarray
    required: float


def demands(sl: StringLoads, design: Pipe | StringDesign, design_factors, yield_factor=1.0, yield_ratio=1.0,
            od_ratio=1.0, wall_ratio=1.0) -> list[Demand]:
    """Every (case, check) rating and load for one string and its pipe(s).

    yield_factor is temperature derating; the *_ratio arguments are sampled
    manufacturing ratios (1 for the deterministic run). Areas for axial stress
    and the Lame stresses use the nominal pipe.
    """
    df = design_factors
    n = StringDesign.of(design).at(sl.md_ft)
    yp = n.yield_psi * yield_ratio * yield_factor
    od = n.od_in * od_ratio
    wall = n.wall_in * wall_ratio
    burst_rating = np.fmin(rt.burst_api_psi(od, wall, yp), n.conn_internal_psi)
    body_yield = rt.pipe_body_yield_lbf(od, wall, yp)
    tension_rating = np.fmin(body_yield, n.conn_tension_lbf)
    comp_rating = np.fmin(body_yield, n.conn_compression_lbf)
    a_s = n.area_steel_in2
    bend_psi = sl.bending_stress_psi
    bend_lbf = bend_psi * a_s

    out: list[Demand] = []

    def axial(case: str, force) -> None:
        out.append(Demand(case, "tension", tension_rating, force + bend_lbf, df.tension))
        out.append(Demand(case, "compression", comp_rating, -(force - bend_lbf), df.compression))

    for c in sl.cases:
        if c.mode == "burst":
            out.append(Demand(c.name, "burst", burst_rating, c.load_psi, df.burst))
        else:
            rating = rt.collapse_api_combined_psi(od, wall, yp, c.axial_real_lbf / a_s)
            pe_eq = rt.equivalent_external_pressure_psi(c.p_external_psi, c.p_internal_psi, od, wall)
            out.append(Demand(c.name, "collapse", rating, pe_eq, df.collapse))
        axial(c.name, c.axial_real_lbf)
        if c.mode == "collapse" and n.thin_wall.all():
            continue  # spec 3.4: triaxial is not a collapse criterion for thin-wall pipe
        vme = _vme_inner(n.od_in, n.id_in, a_s, c.p_internal_psi, c.p_external_psi, c.axial_real_lbf, bend_psi)
        if c.mode == "collapse":
            vme = np.where(n.thin_wall, 0.0, vme)
        out.append(Demand(c.name, "triaxial", yp, vme, df.triaxial))
    for a in sl.axial:
        axial(a.name, a.real_lbf)
    return out


def check_string(well: WellInput, profiles: Profiles, geometries: list[StringGeometry], name: str,
                 pipe: Pipe | StringDesign, yield_derate: YieldDerate | None = None) -> StringCheck:
    """Design factors for one string (one pipe or a tapered design), for every case and check.

    yield_derate(temperature_F) -> factor on Y_p from the manufacturer's
    table. Without one, no temperature derating is applied and a note says so.
    """
    design = StringDesign.of(pipe)
    sl = string_loads(well, profiles, geometries, name, design.body())
    df = well.design_factors
    notes = list(sl.notes)
    factor = _derate_factor(sl, yield_derate, name, notes)
    if any(p.conn_tension_lbf is None for p in design.pipes):
        notes.append(f"{name}: no connection ratings given; pipe body only")
    dem = demands(sl, design, df, factor)
    rows = [_min_df(name, d.case, d.check, np.asarray(d.rating, dtype=float), np.asarray(d.load, dtype=float),
                    d.required, sl) for d in dem]
    thin = [p for p in design.pipes if rt.collapse_regime(p.od_in, p.wall_in, p.yield_psi) != rt.YIELD]
    if thin and any(c.mode == "collapse" for c in sl.cases):
        notes.append(f"{name}: triaxial not applied to collapse cases where D/t is outside the yield-collapse "
                     f"range (D/t {', '.join(f'{p.dt:.1f}' for p in thin)}; spec 3.4)")

    ps_max = None
    frac = next((c for c in sl.cases if c.name == "frac_down_casing"), None)
    if frac is not None:
        cfg = well.load_cases[name].frac_down_casing
        ps = _base(cfg.max_surface_psi)
        burst = next(d for d in dem if d.case == "frac_down_casing" and d.check == "burst")
        # Largest P_s with burst load <= rating / DF at every node (burst only; axial
        # and triaxial also change with P_s through ballooning).
        ps_max = float(np.min(burst.rating / df.burst - (frac.p_internal_psi - ps) + frac.p_external_psi))
    return StringCheck(name, design, rows, sl, ps_max, notes)


def _derate_factor(sl: StringLoads, yield_derate: YieldDerate | None, name: str, notes: list[str]) -> np.ndarray:
    if yield_derate is None:
        notes.append(f"{name}: no temperature derating of yield (manufacturer table not supplied)")
        return np.ones_like(np.asarray(sl.temperature_F, dtype=float))
    return np.asarray(yield_derate(sl.temperature_F), dtype=float)


# --- Whole-well deterministic run ------------------------------------------------------

@dataclass
class WellCheck:
    """Deterministic run of one well: depths, every string's loads and, where a
    pipe is assigned, its design-factor check."""

    well: WellInput
    profiles: Profiles
    depths: SettingDepthResult
    geometries: list[StringGeometry]
    design_name: str | None
    checks: dict[str, StringCheck]  # strings with a pipe
    loads: dict[str, StringLoads]  # every string
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool | None:
        """True or False when every string has a pipe; None when some are unchecked."""
        if len(self.checks) < len(self.geometries):
            return None
        return all(c.ok for c in self.checks.values())


def design_pipes(well: WellInput, design: str | None = None
                 ) -> tuple[str | None, dict[str, StringDesign], list[str]]:
    """String designs from a reference design in the input (the first one by default).

    A string with several sections is a tapered design: list the sections top
    first and give bottom_md_ft for every section except the last.
    """
    if not well.reference_designs:
        return None, {}, ["no reference design in the input; loads are reported without pipe checks"]
    ref = _pick_design(well.reference_designs, design)
    notes: list[str] = []
    out: dict[str, StringDesign] = {}
    od_by_name = {s.name: s.od_in for s in well.strings}
    for name, sections in ref.strings.items():
        pipes = []
        for sec in sections:
            od = sec.od_in if sec.od_in is not None else od_by_name[name]
            if abs(od - od_by_name[name]) > 1e-6:
                notes.append(f"{name}: design OD {od:g} in differs from the string's {od_by_name[name]:g} in")
            if sec.wall_in is None:
                notes.append(f"{name}: reference design gives no wall_in; string not checked")
                break
            try:
                pipes.append(Pipe(od, sec.wall_in, sec.weight_lb_per_ft, sec.grade, yield_psi=sec.yield_psi,
                                  drift_in=sec.drift_in, connection=sec.connection,
                                  conn_tension_lbf=sec.conn_tension_lbf,
                                  conn_compression_lbf=sec.conn_compression_lbf,
                                  conn_internal_psi=sec.conn_internal_psi))
            except ValueError as exc:
                notes.append(f"{name}: {exc}; string not checked")
                break
        else:
            crossovers = [sec.bottom_md_ft for sec in sections[:-1]]
            if any(x is None for x in crossovers):
                notes.append(f"{name}: tapered design needs bottom_md_ft on every section but the last; "
                             "string not checked")
                continue
            try:
                out[name] = StringDesign(tuple(pipes), tuple(crossovers))
            except ValueError as exc:
                notes.append(f"{name}: {exc}; string not checked")
    return ref.name, out, notes


def _pick_design(designs: list[ReferenceDesign], name: str | None) -> ReferenceDesign:
    if name is None:
        return designs[0]
    for d in designs:
        if d.name == name:
            return d
    raise ValueError(f"no reference design named {name!r}; have {[d.name for d in designs]}")


def next_bit_in(well: WellInput, g: StringGeometry) -> float | None:
    """Bit size of the hole section drilled through this string, if any."""
    if g.kind in ("liner", "tieback"):
        return None
    deeper = [s for s in well.strings if s.kind != "tieback" and s.od_in < g.od_in and s.hole_in < g.od_in]
    return deeper[0].hole_in if deeper else None


def drift_notes(well: WellInput, geometries: list[StringGeometry], designs: dict[str, StringDesign]) -> list[str]:
    """Flag pipes whose drift does not pass the next hole section's bit."""
    out = []
    for g in geometries:
        design = designs.get(g.name)
        bit_in = next_bit_in(well, g)
        if design is None or bit_in is None:
            continue
        for pipe in design.pipes:
            if pipe.drift_in is None:
                out.append(f"{g.name}: no drift_in given for {pipe.label}; cannot confirm the {bit_in:g}-in bit "
                           f"passes (ID {pipe.id_in:.3f} in)")
            elif pipe.drift_in < bit_in - 1e-9:
                out.append(f"{g.name}: drift {pipe.drift_in:.3f} in is smaller than the next bit "
                           f"({bit_in:g} in); a special-drift pipe or a smaller bit is needed")
    return out


def check_well(well: WellInput, design: str | None = None, profiles: Profiles | None = None,
               yield_derate: YieldDerate | None = None, designs: dict[str, StringDesign] | None = None,
               design_name: str | None = None) -> WellCheck:
    """Run the deterministic pipeline: profiles, setting depths, loads and design factors.

    The pipes come from `designs` when given (e.g. the optimizer's choice),
    otherwise from the input's reference design named `design` (or the first).
    """
    profiles = build_profiles(well) if profiles is None else profiles
    depths = setting_depths(well, profiles)
    geoms, _ = resolve_strings(well, profiles.survey, profiles, depths.chosen_tvd_ft)
    if designs is None:
        design_name, pipes, notes = design_pipes(well, design)
    else:
        pipes, notes = dict(designs), []
    notes += drift_notes(well, geoms, pipes)
    checks: dict[str, StringCheck] = {}
    loads: dict[str, StringLoads] = {}
    for g in geoms:
        if g.name in pipes:
            checks[g.name] = check_string(well, profiles, geoms, g.name, pipes[g.name], yield_derate)
            loads[g.name] = checks[g.name].loads
        else:
            loads[g.name] = string_loads(well, profiles, geoms, g.name)
    return WellCheck(well, profiles, depths, geoms, design_name, checks, loads, notes)


def _min_df(string: str, case: str, check: str, rating: np.ndarray, load: np.ndarray, required: float,
            sl: StringLoads) -> CheckRow:
    rating = np.broadcast_to(rating, load.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        dfs = np.where(load > 0, rating / np.where(load > 0, load, 1.0), np.inf)
    i = int(np.argmin(dfs))
    if not np.isfinite(dfs[i]):
        return CheckRow(string, case, check, float("inf"), required, None, None)
    return CheckRow(string, case, check, float(dfs[i]), required, float(sl.md_ft[i]), float(sl.tvd_ft[i]))


def _gas_kick_settings(well: WellInput, name: str):
    cfg = well.load_cases.get(name)
    return cfg.gas_kick if cfg is not None and cfg.gas_kick not in (None, False) else None


def _required_kick(well: WellInput, name: str, notes: list[str]) -> float:
    gk = _gas_kick_settings(well, name)
    if gk is not None and gk.kick_intensity_ppg is not None:
        return _base(gk.kick_intensity_ppg)
    notes.append(f"{name}: kick_intensity_ppg not given; shoe must tolerate "
                 f"{PLACEHOLDER_KICK_INTENSITY_PPG} ppg (placeholder)")
    return PLACEHOLDER_KICK_INTENSITY_PPG


def _gas_gradient(well: WellInput, name: str) -> float:
    gk = _gas_kick_settings(well, name)
    return _base(gk.gas_gradient_psi_per_ft) if gk is not None and gk.gas_gradient_psi_per_ft is not None else 0.1


def _base(x: float | Distribution) -> float:
    return x.base if isinstance(x, Distribution) else float(x)

