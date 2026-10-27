import copy

import numpy as np
import pytest

from casingrel import ratings as rt
from casingrel.cli import main
from casingrel.design import StringDesign, check_string, check_well, design_pipes, setting_depths
from casingrel.loads import (CROSSOVER_GAP_FT, PipeBody, TaperedBody, effective_tension_lbf,
                             hanging_effective_tension_lbf, resolve_strings, service_force_lbf)
from casingrel.optimize import (DEFAULT_CATALOG, ILLUSTRATIVE_UNIT, Candidate, StringOptimum, _compare,
                                candidates_for, cost_mode, design_cost, enumerate_designs, load_catalog,
                                optimize_string, optimize_well, published_rating_errors, resolve_catalog_path)
from casingrel.profiles import build_profiles
from casingrel.ratings import Pipe
from casingrel.report import write_report
from casingrel.schema import WellInput, load_well
from casingrel.synthetic import DEFAULT_EXAMPLES_DIR

WELL_C = DEFAULT_EXAMPLES_DIR / "synthetic_c_horizontal_frac.yaml"
J55 = Pipe(9.625, 0.395, 40.0, "J55")
L80 = Pipe(9.625, 0.395, 40.0, "L80")


@pytest.fixture(scope="module")
def well_c():
    w = load_well(WELL_C)
    p = build_profiles(w)
    geoms, _ = resolve_strings(w, p.survey, p, setting_depths(w, p).chosen_tvd_ft)
    return w, p, geoms


def geom(geoms, name):
    return next(g for g in geoms if g.name == name)


# --- Catalog -------------------------------------------------------------------------------------

def test_starter_catalog_matches_published_ratings():
    # Spec 6.1: the rating check extends to every pipe in the starter catalog.
    cat = load_catalog()
    assert len(cat) == 6 and all("spec 6.1" in c.source for c in cat)
    assert all(c.api_burst_psi and c.api_collapse_psi for c in cat)
    assert published_rating_errors(cat) == []


def test_catalog_errors_and_paths(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("od_in,weight_lb_per_ft,grade,wall_in\n7,26,P110,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_catalog(bad)
    w = load_well(WELL_C)
    assert resolve_catalog_path(w) == DEFAULT_CATALOG
    d = w.model_dump()
    d["catalog"] = "../catalogs/api_starter.csv"
    assert resolve_catalog_path(WellInput.model_validate(d), WELL_C).resolve() == DEFAULT_CATALOG.resolve()
    d["catalog"] = "nowhere.csv"
    with pytest.raises(FileNotFoundError):
        resolve_catalog_path(WellInput.model_validate(d), WELL_C)


def test_candidates_filter_by_od_and_drift(well_c):
    w, _, geoms = well_c
    cat = load_catalog()
    surface = geom(geoms, "surface")
    pipes, notes = candidates_for(w, surface, cat)
    assert {p.grade for p in pipes} == {"J55", "L80"}
    assert any("accepted on ID" in n for n in notes)
    tight = [c for c in cat]
    tight[1] = type(tight[1])(Pipe(9.625, 0.395, 40.0, "L80", drift_in=8.679), None, None, "test")
    pipes, _ = candidates_for(w, surface, tight)
    assert [p.grade for p in pipes] == ["J55"]  # API drift 8.679 in does not pass the 8.75-in bit
    d = w.model_dump()
    d["strings"][0]["od_in"] = 10.75
    d["strings"][0]["hole_in"] = 13.5
    w2 = WellInput.model_validate(d)
    p2 = build_profiles(w2)
    g2, _ = resolve_strings(w2, p2.survey, p2)
    assert candidates_for(w2, g2[0], cat) == ([], ["surface: the catalog has no 10.75-in pipe; add data sheets "
                                                   "for this size"])


# --- Designs, tapering and cost ------------------------------------------------------------------

def test_enumerate_designs_and_cost(well_c):
    _, _, geoms = well_c
    g = geom(geoms, "surface")
    designs = enumerate_designs(g, [J55, L80])
    singles = [d for d in designs if not d.tapered]
    tapered = [d for d in designs if d.tapered]
    assert len(singles) == 2 and len(tapered) == 2 * 29  # 100-2,900 ft
    assert all(d.crossovers_md_ft[0] % 100 == 0 and 0 < d.crossovers_md_ft[0] < 3000 for d in tapered)
    assert cost_mode([J55, L80]) == "illustrative"
    d = StringDesign((L80, J55), (1000.0,))
    assert design_cost(d, g, "illustrative") == pytest.approx(1000 * 40 * 1.15 + 2000 * 40 * 1.0)
    priced = [Pipe(9.625, 0.395, 40.0, "J55", cost_per_ft=30.0), Pipe(9.625, 0.395, 40.0, "L80", cost_per_ft=36.0)]
    assert cost_mode(priced) == "dollars"
    assert design_cost(StringDesign(tuple(priced), (500.0,)), g, "dollars") == pytest.approx(500 * 30 + 2500 * 36)
    assert d.label == "9.625-in 40 lb/ft L80 to 1,000 ft / 9.625-in 40 lb/ft J55 below"


def test_tapered_hanging_weight_hand_calc():
    tvd = np.array([0.0, 1000.0, 1000.0 + CROSSOVER_GAP_FT, 3000.0])
    w = np.array([47.0, 47.0, 40.0, 40.0])
    eff = hanging_effective_tension_lbf(tvd, w, 0.0)
    assert eff[-1] == 0
    assert eff[0] == pytest.approx(47 * (1000 + CROSSOVER_GAP_FT) + 40 * (2000 - CROSSOVER_GAP_FT))
    one = hanging_effective_tension_lbf(tvd, np.full(4, 40.0), 9.0)
    assert one == pytest.approx(effective_tension_lbf(tvd, 3000.0, 40.0, 9.0))


def test_service_force_two_sections_hand_calc():
    md = np.array([0.0, 500.0, 1000.0, 1000.01, 2000.0])
    bodies = (PipeBody.from_wall(7.0, 0.408, 29.0), PipeBody.from_wall(7.0, 0.362, 26.0))
    pipe = TaperedBody(bodies, (1000.0,)).at(md)
    free = np.ones(5, dtype=bool)
    got = service_force_lbf(md, free, np.full(5, 1000.0), 0.0, 0.0, pipe)
    ratio = pipe.area_inner_in2 / pipe.area_steel_in2
    expect = 0.6 * 1000.0 * np.trapezoid(ratio, md) / np.trapezoid(1 / pipe.area_steel_in2, md)
    assert float(got[0]) == pytest.approx(expect)


def test_tapered_check_uses_each_section(well_c):
    w, p, geoms = well_c
    d = StringDesign((L80, J55), (1500.0,))
    sc = check_string(w, p, geoms, "surface", d)
    md = sc.loads.md_ft
    assert 1500.0 in md and 1500.0 + CROSSOVER_GAP_FT in md
    nr = d.at(md)
    assert nr.yield_psi[md == 1500.0][0] == 80_000 and nr.yield_psi[md == 1500.0 + CROSSOVER_GAP_FT][0] == 55_000
    single_l80 = check_string(w, p, geoms, "surface", L80)
    row = lambda sc, case, check: next(r for r in sc.rows if r.case == case and r.check == check)  # noqa: E731
    # Burst load falls with depth, but J55 is weaker: the tapered string's gas-kick burst
    # governs at the top of the J55 section, below the L80.
    gk = row(sc, "gas_kick", "burst")
    assert gk.md_ft == pytest.approx(1500.0 + CROSSOVER_GAP_FT)
    load = sc.loads.case("gas_kick").load_psi[md == 1500.0 + CROSSOVER_GAP_FT][0]
    assert gk.design_factor == pytest.approx(rt.burst_api_psi(9.625, 0.395, 55_000) / load)
    assert gk.design_factor < row(single_l80, "gas_kick", "burst").design_factor
    # Same 40 lb/ft weight in both grades, so the hook load and the top tension check are unchanged.
    assert row(sc, "installation", "tension").design_factor == pytest.approx(
        row(single_l80, "installation", "tension").design_factor)


def test_tapered_reference_design(well_c):
    w, _, _ = well_c
    d = w.model_dump()
    d["reference_designs"][0]["strings"]["surface"] = [
        {"grade": "L80", "weight_lb_per_ft": 40.0, "wall_in": 0.395, "bottom_md_ft": 1500},
        {"grade": "J55", "weight_lb_per_ft": 40.0, "wall_in": 0.395}]
    _, designs, notes = design_pipes(WellInput.model_validate(d))
    assert designs["surface"].crossovers_md_ft == (1500.0,) and not notes
    del d["reference_designs"][0]["strings"]["surface"][0]["bottom_md_ft"]
    _, designs, notes = design_pipes(WellInput.model_validate(d))
    assert "surface" not in designs and any("bottom_md_ft" in n for n in notes)


# --- Optimizer -------------------------------------------------------------------------------------

def test_optimizer_picks_cheapest_reliable_surface_string(well_c):
    w, p, geoms = well_c
    s = optimize_string(w, p, geoms, geom(geoms, "surface"), load_catalog(), 1e-3, samples=20_000,
                        max_samples=20_000)
    assert s.chosen is not None and s.chosen.design.label == "9.625-in 40 lb/ft J55"
    assert s.chosen.deterministic_ok and s.chosen.confirmed and s.chosen.max_pf <= 1e-3
    # Stricter targets buy L80 at the top (where gas-kick burst is highest): a tapered string.
    strict = s.by_target[1e-5]
    assert strict is not None and strict.design.tapered and strict.design.pipes[0].grade == "L80"
    assert strict.cost > s.chosen.cost
    assert s.unit == ILLUSTRATIVE_UNIT


def test_optimizer_reports_infeasible_string(well_c):
    w, p, geoms = well_c
    s = optimize_string(w, p, geoms, geom(geoms, "intermediate"), load_catalog(), 1e-3, tapered=False)
    assert s.chosen is None and s.designs_considered == 3
    assert s.best_effort.design.label == "7-in 26 lb/ft P110"
    assert any("no catalog design passes the deterministic check" in n for n in s.notes)


def test_reference_gap_flag():
    opt = StringOptimum("x", "illustrative", 1, Candidate(StringDesign((J55,)), 100.0, True, 1.2), {}, None)
    assert _compare(opt, Candidate(StringDesign((L80,)), 110.0, True, 1.5)) is None
    assert "+50%" in _compare(opt, Candidate(StringDesign((L80,)), 150.0, True, 1.5))


@pytest.fixture(scope="module")
def optimized(well_c, tmp_path_factory):
    w, _, _ = well_c
    o = optimize_well(w, WELL_C, samples=20_000, tapered=False, max_samples=20_000)
    result = check_well(w, profiles=o.profiles, designs=o.designs, design_name="optimized")
    rel = {k: s.chosen.reliability for k, s in o.strings.items() if s.chosen is not None}
    out = tmp_path_factory.mktemp("opt")
    return o, result, write_report(result, out, rel, None, o).read_text(encoding="utf-8")


def test_optimize_well_and_report(optimized):
    o, result, text = optimized
    assert set(o.designs) == {"surface", "liner"}
    assert result.checks["surface"].design.label == "9.625-in 40 lb/ft J55"
    assert len(o.references) == 3 and o.references[1].candidate.deterministic_ok is False
    for heading in ("## Optimized design", "### Choice by target (FORM screen)", "### Reference designs",
                    "## Probability of failure", "### Target sweep"):
        assert heading in text
    assert "**none feasible**" in text and "not dollars" in text
    assert "starter catalog used" in text and "pipe_statistics.yield_ratio not given" in text


def test_cli_optimize_and_probabilistic(tmp_path, capsys):
    small = ["--samples", "5000", "--max-samples", "5000"]
    assert main(["optimize", str(WELL_C), "-o", str(tmp_path / "o"), "--no-taper", *small]) == 0
    assert "## Optimized design" in (tmp_path / "o" / "report.md").read_text(encoding="utf-8")
    assert main(["run", str(WELL_C), "-o", str(tmp_path / "r"), "--probabilistic", *small]) == 0
    text = (tmp_path / "r" / "report.md").read_text(encoding="utf-8")
    assert "## Probability of failure" in text and "Share of samples where the shoe must move" in text
    assert "## Sensitivity" in text and "Sobol total" in text
    assert (tmp_path / "r" / "plots" / "tornado_intermediate.png").exists()
    assert (tmp_path / "r" / "plots" / "pf_vs_ps_intermediate.png").exists()
    assert "max Pf" in capsys.readouterr().out
