"""Pipe catalog and least-cost string selection (spec Section 4.4).

For each string:
  1. Candidates: every catalog pipe with the string's OD that drifts the next
     bit (or, when the catalog has no drift, whose ID passes it; flagged).
  2. Designs: every single pipe, plus every two-section tapered string (any
     ordered pair of different candidates) with its crossover on a 100-ft MD grid.
  3. Cost = sum of section length x $/ft. When any candidate has no price, all
     candidates use a relative cost (weight x grade multiplier), labelled
     illustrative and never printed as dollars.
  4. The chosen design is the cheapest one that passes the deterministic check
     at the input's design factors (spec 6.3, unless switched off) and has
     Pf <= target for every load case and check. Search is exhaustive in cost
     order: designs are screened with FORM, and the winner is confirmed by
     Monte Carlo. If Monte Carlo disagrees, the search moves on.
  5. The same search gives the choice for every target in the spec 4.3 sweep,
     and each reference design in the input is scored the same way.

Strings are optimized one at a time: a string's loads depend only on its own
pipe, so the cheapest well is the cheapest choice for each string.
"""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ratings as rt
from .design import StringDesign, check_string, design_pipes, next_bit_in, setting_depths
from .loads import StringGeometry, resolve_strings
from .profiles import Profiles, build_profiles
from .ratings import Pipe
from .reliability import DEFAULT_MAX_SAMPLES, DEFAULT_SAMPLES, SWEEP_TARGETS, StringReliability, string_reliability
from .schema import WellInput

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "catalogs" / "api_starter.csv"
CROSSOVER_STEP_FT = 100.0
# Relative cost per lb of pipe by grade, used only when the catalog has no
# prices. ILLUSTRATIVE placeholders, not market data; results using them are
# labelled and never shown as dollars.
ILLUSTRATIVE_GRADE_MULTIPLIER = {"H40": 0.9, "J55": 1.0, "K55": 1.0, "M65": 1.05, "N80": 1.1, "L80": 1.15,
                                 "C90": 1.3, "R95": 1.3, "T95": 1.35, "C110": 1.5, "P110": 1.3, "Q125": 1.45}
ILLUSTRATIVE_UNIT = "relative units (illustrative: weight x grade multiplier; not dollars)"


# --- Catalog -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CatalogPipe:
    pipe: Pipe
    api_collapse_psi: float | None  # published value, for the spec 6.1 check
    api_burst_psi: float | None
    source: str


def load_catalog(path: str | Path = DEFAULT_CATALOG) -> list[CatalogPipe]:
    """Read a pipe catalog CSV. Blank cells mean "not known"."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            def num(key: str) -> float | None:
                v = (row.get(key) or "").strip()
                return float(v) if v else None
            try:
                pipe = Pipe(od_in=num("od_in"), wall_in=num("wall_in"), weight_lb_per_ft=num("weight_lb_per_ft"),
                            grade=row["grade"].strip(), yield_psi=num("yield_psi"), drift_in=num("drift_in"),
                            connection=(row.get("connection") or "").strip() or None,
                            conn_tension_lbf=num("conn_tension_lbf"), conn_compression_lbf=num("conn_compression_lbf"),
                            conn_internal_psi=num("conn_internal_psi"), cost_per_ft=num("cost_per_ft"))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{path}, line {i}: {exc}") from exc
            out.append(CatalogPipe(pipe, num("api_collapse_psi"), num("api_burst_psi"),
                                   (row.get("source") or "").strip()))
    return out


def resolve_catalog_path(well: WellInput, input_path: str | Path | None = None) -> Path:
    """The input's catalog (relative to the input file, then the working folder), else the starter catalog."""
    if well.catalog is None:
        return DEFAULT_CATALOG
    p = Path(well.catalog)
    if not p.is_absolute() and input_path is not None and (Path(input_path).parent / p).exists():
        return Path(input_path).parent / p
    if p.exists():
        return p
    raise FileNotFoundError(f"catalog {well.catalog!r} not found")


def candidates_for(well: WellInput, g: StringGeometry, catalog: list[CatalogPipe]) -> tuple[list[Pipe], list[str]]:
    """Catalog pipes that fit this string: same OD, and drift (or ID) passes the next bit."""
    notes = []
    same_od = [c.pipe for c in catalog if abs(c.pipe.od_in - g.od_in) < 1e-3]
    if not same_od:
        return [], [f"{g.name}: the catalog has no {g.od_in:g}-in pipe; add data sheets for this size"]
    bit = next_bit_in(well, g)
    out = []
    for p in same_od:
        if bit is None:
            out.append(p)
        elif p.drift_in is not None:
            if p.drift_in >= bit - 1e-9:
                out.append(p)
        elif p.id_in >= bit:
            out.append(p)
            notes.append(f"{g.name}: {p.label} has no drift in the catalog; accepted on ID {p.id_in:.3f} in "
                         f"vs the {bit:g}-in bit (API drift is smaller than ID; confirm)")
    if not out:
        notes.append(f"{g.name}: no {g.od_in:g}-in catalog pipe passes the {bit:g}-in bit")
    return out, notes


# --- Designs and cost ------------------------------------------------------------------------

def enumerate_designs(g: StringGeometry, pipes: list[Pipe], step_ft: float = CROSSOVER_STEP_FT,
                      tapered: bool = True) -> list[StringDesign]:
    """Every single-pipe design, plus two-section designs with one crossover on a step_ft MD grid."""
    designs = [StringDesign((p,)) for p in pipes]
    if tapered and len(pipes) > 1:
        first = math.floor(g.top_md_ft / step_ft) * step_ft + step_ft
        grid = np.arange(first, g.shoe_md_ft - step_ft / 2, step_ft)
        for upper, lower in itertools.permutations(pipes, 2):
            designs += [StringDesign((upper, lower), (float(x),)) for x in grid]
    return designs


def cost_mode(pipes: list[Pipe]) -> str:
    return "dollars" if pipes and all(p.cost_per_ft is not None for p in pipes) else "illustrative"


def design_cost(design: StringDesign, g: StringGeometry, mode: str) -> float:
    total = 0.0
    for pipe, length in zip(design.pipes, design.section_lengths_ft(g.top_md_ft, g.shoe_md_ft)):
        if mode == "dollars":
            total += length * pipe.cost_per_ft
        else:
            total += length * pipe.weight_lb_per_ft * ILLUSTRATIVE_GRADE_MULTIPLIER.get(pipe.grade, 1.0)
    return total


def cost_unit(mode: str) -> str:
    return "$" if mode == "dollars" else ILLUSTRATIVE_UNIT


# --- Search ----------------------------------------------------------------------------------------

@dataclass
class Candidate:
    design: StringDesign
    cost: float
    deterministic_ok: bool
    worst_df_ratio: float  # min over checks of DF / required
    reliability: StringReliability | None = None  # FORM screen, or MC-confirmed for a winner
    confirmed: bool = False  # Monte Carlo run on this design

    @property
    def max_pf(self) -> float | None:
        return None if self.reliability is None else max(r.pf for r in self.reliability.rows)

    @property
    def governing(self) -> tuple[str, str] | None:
        if self.reliability is None:
            return None
        r = self.reliability.governing
        return (r.case, r.check)


@dataclass
class StringOptimum:
    string: str
    mode: str  # "dollars" or "illustrative"
    designs_considered: int
    chosen: Candidate | None
    by_target: dict[float, Candidate | None]
    best_effort: Candidate | None  # highest DF ratio when nothing is feasible
    notes: list[str] = field(default_factory=list)

    @property
    def unit(self) -> str:
        return cost_unit(self.mode)


@dataclass
class ReferenceScore:
    design_name: str
    string: str
    candidate: Candidate | None
    note: str | None = None


@dataclass
class Optimization:
    well: WellInput
    profiles: Profiles
    geometries: list[StringGeometry]
    target_pf: float
    strings: dict[str, StringOptimum]
    references: list[ReferenceScore]
    catalog_path: Path
    notes: list[str] = field(default_factory=list)

    @property
    def designs(self) -> dict[str, StringDesign]:
        return {k: v.chosen.design for k, v in self.strings.items() if v.chosen is not None}


def _evaluate(well, profiles, geoms, g, design, mode, require_det) -> Candidate:
    chk = check_string(well, profiles, geoms, g.name, design)
    finite = [r.design_factor / r.required for r in chk.rows if math.isfinite(r.design_factor)]
    return Candidate(design, design_cost(design, g, mode), chk.ok, min(finite, default=math.inf))


def optimize_string(well: WellInput, profiles: Profiles, geoms: list[StringGeometry], g: StringGeometry,
                    catalog: list[CatalogPipe], target_pf: float, require_deterministic: bool = True,
                    samples: int = DEFAULT_SAMPLES, seed: int = 0, tapered: bool = True,
                    sweep: tuple[float, ...] = SWEEP_TARGETS, max_samples: int = DEFAULT_MAX_SAMPLES) -> StringOptimum:
    pipes, notes = candidates_for(well, g, catalog)
    mode = cost_mode(pipes)
    if mode == "illustrative" and pipes:
        notes.append(f"{g.name}: catalog has no prices; costs are {ILLUSTRATIVE_UNIT}")
    designs = enumerate_designs(g, pipes, tapered=tapered)
    cands = sorted((_evaluate(well, profiles, geoms, g, d, mode, require_deterministic) for d in designs),
                   key=lambda c: (c.cost, -c.worst_df_ratio))
    eligible = [c for c in cands if c.deterministic_ok or not require_deterministic]
    targets = sorted(set(sweep) | {target_pf}, reverse=True)
    by_target: dict[float, Candidate | None] = {t: None for t in targets}
    chosen = None
    for c in eligible:
        if all(by_target[t] is not None for t in targets) and chosen is not None:
            break
        c.reliability = string_reliability(well, profiles, geoms, g.name, c.design, use_mc=False)
        for t in targets:
            if by_target[t] is None and c.max_pf <= t:
                by_target[t] = c
        if chosen is None and c.max_pf <= target_pf:
            confirmed = string_reliability(well, profiles, geoms, g.name, c.design, n=samples, seed=seed,
                                           max_n=max_samples)
            c.confirmed = True
            if confirmed.meets(target_pf):
                c.reliability = confirmed
                chosen = c
            else:
                c.reliability = confirmed
                notes.append(f"{g.name}: {c.design.label} passed the FORM screen but Monte Carlo gives "
                             f"Pf {max(r.pf for r in confirmed.rows):.1e} > target; skipped")
                if by_target.get(target_pf) is c:
                    by_target[target_pf] = None
    best = max(cands, key=lambda c: c.worst_df_ratio) if cands and chosen is None else None
    if pipes and chosen is None:
        if not eligible:
            notes.append(f"{g.name}: no catalog design passes the deterministic check at the input's design "
                         "factors; a heavier wall or stronger grade is needed in the catalog")
        else:
            notes.append(f"{g.name}: no catalog design meets Pf <= {target_pf:g} in every load case")
    return StringOptimum(g.name, mode, len(designs), chosen, by_target, best, notes)


def optimize_well(well: WellInput, input_path: str | Path | None = None, catalog_path: str | Path | None = None,
                  target_pf: float | None = None, require_deterministic: bool = True,
                  samples: int = DEFAULT_SAMPLES, seed: int = 0, tapered: bool = True,
                  max_samples: int = DEFAULT_MAX_SAMPLES) -> Optimization:
    """Cheapest design per string meeting the deterministic check and the reliability target."""
    profiles = build_profiles(well)
    depths = setting_depths(well, profiles)
    geoms, _ = resolve_strings(well, profiles.survey, profiles, depths.chosen_tvd_ft)
    path = Path(catalog_path) if catalog_path is not None else resolve_catalog_path(well, input_path)
    catalog = load_catalog(path)
    target = well.reliability_target_pf if target_pf is None else target_pf
    notes = [] if require_deterministic else ["deterministic design-factor check switched off for the optimizer"]
    if path == DEFAULT_CATALOG:
        notes.append(f"starter catalog used ({len(catalog)} pipes from the spec 6.1 table only); "
                     "supply manufacturer data sheets for a real design")
    strings = {g.name: optimize_string(well, profiles, geoms, g, catalog, target, require_deterministic,
                                       samples, seed, tapered, max_samples=max_samples)
               for g in geoms if g.kind != "conductor"}
    refs = []
    for ref in well.reference_designs:
        _, designs, ref_notes = design_pipes(well, ref.name)
        for g in geoms:
            if g.name not in ref.strings:
                continue
            d = designs.get(g.name)
            if d is None:
                refs.append(ReferenceScore(ref.name, g.name, None, "; ".join(n for n in ref_notes if g.name in n)))
                continue
            mode = strings[g.name].mode if g.name in strings else "illustrative"
            c = _evaluate(well, profiles, geoms, g, d, mode, require_deterministic)
            c.reliability = string_reliability(well, profiles, geoms, g.name, d, n=samples, seed=seed,
                                               max_n=max_samples)
            c.confirmed = True
            refs.append(ReferenceScore(ref.name, g.name, c, _compare(strings.get(g.name), c)))
    return Optimization(well, profiles, geoms, target, strings, refs, path, notes)


def _compare(opt: StringOptimum | None, ref: Candidate) -> str | None:
    """Spec 6.3: flag a large gap between the optimized string and a reference string."""
    if opt is None or opt.chosen is None:
        return None
    gap = (ref.cost - opt.chosen.cost) / opt.chosen.cost
    if abs(gap) > 0.25:
        return (f"reference costs {gap:+.0%} vs the optimized design; a large gap can mean an input is off "
                "or the reference used a load case this model lacks")
    return None


def published_rating_errors(catalog: list[CatalogPipe]) -> list[tuple[str, str, float, float]]:
    """Spec 6.1 check for every catalog pipe with published ratings:
    (label, rating, formula psi, published psi) where they differ by more than max(0.5%, 10 psi)."""
    bad = []
    for c in catalog:
        p = c.pipe
        for kind, published, value in (("collapse", c.api_collapse_psi, rt.collapse_api_psi(p.od_in, p.wall_in, p.yield_psi)),
                                       ("burst", c.api_burst_psi, rt.burst_api_psi(p.od_in, p.wall_in, p.yield_psi))):
            if published is not None and abs(value - published) > max(0.005 * published, 10.0):
                bad.append((p.label, kind, float(value), published))
    return bad
