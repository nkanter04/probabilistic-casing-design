import copy
import math

import numpy as np
import pytest
from scipy import stats

from casingrel import ratings as rt
from casingrel.design import demands, design_pipes, setting_depths
from casingrel.loads import PipeBody, Realization, resolve_strings, string_loads
from casingrel.profiles import build_profiles
from casingrel.reliability import (RandomVariable, StringLimitStates, VariableSet, collect_variables, form,
                                   form_many, importance_sampling, monte_carlo, shoe_move_probability,
                                   string_reliability, target_sweep, wilson_interval)
from casingrel.schema import Distribution, WellInput, load_well
from casingrel.synthetic import DEFAULT_EXAMPLES_DIR

WELL_C = DEFAULT_EXAMPLES_DIR / "synthetic_c_horizontal_frac.yaml"


def normal(name, mean, sd, **kw):
    return RandomVariable(name, Distribution(dist="normal", mean=mean, sd=sd, **kw))


def vset(*rvs, corr=None):
    return VariableSet(list(rvs), np.eye(len(rvs)) if corr is None else np.asarray(corr))


def g_linear(x):
    return x[:, 0] - x[:, 1]


# --- Spec 6.2: closed form, MC vs FORM, seeds ---------------------------------------------------

def test_form_matches_closed_form_for_normal_r_and_l():
    # beta = (muR - muL) / sqrt(sR^2 + sL^2) exactly for a linear limit state.
    vs = vset(normal("R", 10.0, 1.5), normal("L", 6.0, 1.0))
    r = form(g_linear, vs)
    beta = 4.0 / math.sqrt(1.5**2 + 1.0**2)
    assert r.converged
    assert r.beta == pytest.approx(beta, rel=1e-6)
    assert r.pf == pytest.approx(stats.norm.sf(beta), rel=1e-5)
    assert np.sum(r.alpha**2) == pytest.approx(1.0)
    assert r.alpha[0] < 0 < r.alpha[1]  # lower R and higher L push towards failure


def test_monte_carlo_matches_form_near_1e3():
    # Spec 6.2: MC and FORM within a factor of 2 for a case with Pf near 1e-3.
    vs = vset(RandomVariable("R", Distribution(dist="lognormal", mean=10.0, sd=1.0)), normal("L", 6.2, 0.8))
    fr = form(g_linear, vs)
    mc = monte_carlo(lambda x: g_linear(x)[:, None], vs, n=400_000, seed=1, max_n=400_000)
    assert 3e-4 < mc.pf[0] < 3e-3
    assert 0.5 < fr.pf / mc.pf[0] < 2.0
    lo, hi = mc.ci()
    assert lo[0] <= mc.pf[0] <= hi[0]


def test_same_seed_same_pf():
    vs = vset(normal("R", 10.0, 1.5), normal("L", 6.0, 1.0))
    g = lambda x: g_linear(x)[:, None]  # noqa: E731
    a = monte_carlo(g, vs, n=50_000, seed=7, max_n=50_000)
    b = monte_carlo(g, vs, n=50_000, seed=7, max_n=50_000)
    c = monte_carlo(g, vs, n=50_000, seed=8, max_n=50_000)
    assert a.pf[0] == b.pf[0] and a.n == b.n
    assert a.pf[0] != c.pf[0]


def test_monte_carlo_adds_samples_until_cov_target():
    vs = vset(normal("R", 10.0, 1.5), normal("L", 6.0, 1.0))  # Pf ~ 1.3e-2
    mc = monte_carlo(lambda x: g_linear(x)[:, None], vs, n=1_000, seed=0, max_n=200_000, batch=1_000)
    assert mc.cov[0] <= 0.10 and mc.n > 1_000


def test_importance_sampling_checks_form():
    vs = vset(RandomVariable("R", Distribution(dist="lognormal", mean=10.0, sd=1.0)), normal("L", 5.0, 0.8))
    fr = form(g_linear, vs)
    pf_is, cov = importance_sampling(g_linear, vs, fr.u_star, n=50_000, seed=3)
    assert cov < 0.05
    assert 0.7 < pf_is / fr.pf < 1.3


def test_form_matches_openturns():
    ot = pytest.importorskip("openturns")
    vs = vset(RandomVariable("R", Distribution(dist="lognormal", mean=10.0, sd=1.0)), normal("L", 5.0, 0.8))
    ours = form(g_linear, vs)
    s = math.sqrt(math.log1p(0.01))
    dist = ot.JointDistribution([ot.LogNormal(math.log(10.0) - s * s / 2, s), ot.Normal(5.0, 0.8)])
    model = ot.SymbolicFunction(["R", "L"], ["R - L"])
    event = ot.ThresholdEvent(ot.CompositeRandomVector(model, ot.RandomVector(dist)), ot.LessOrEqual(), 0.0)
    algo = ot.FORM(ot.AbdoRackwitz(), event, dist.getMean())
    algo.run()
    assert ours.beta == pytest.approx(algo.getResult().getHasoferReliabilityIndex(), rel=1e-3)


# --- Random variables ---------------------------------------------------------------------------

def test_variable_transforms():
    z = np.random.default_rng(0).standard_normal(400_000)
    ln = RandomVariable("x", Distribution(dist="lognormal", mean=2.0, sd=0.3)).from_standard_normal(z)
    assert ln.mean() == pytest.approx(2.0, rel=5e-3) and ln.std() == pytest.approx(0.3, rel=1e-2)
    wall = RandomVariable("w", Distribution(dist="normal", mean=1.0, sd=0.08, lower=0.875))
    w = wall.from_standard_normal(z)
    assert w.min() >= 0.875 and (w < 0.9).any()
    tri = RandomVariable("t", Distribution(dist="triangular", low=0, mode=1, high=4)).from_standard_normal(z)
    assert tri.mean() == pytest.approx(5 / 3, rel=1e-2)


def test_correlation_and_bad_matrix():
    vs = vset(normal("a", 0, 1), normal("b", 0, 1), corr=[[1, 0.7], [0.7, 1]])
    x = vs.to_physical(np.random.default_rng(0).standard_normal((200_000, 2)))
    assert np.corrcoef(x.T)[0, 1] == pytest.approx(0.7, abs=0.01)
    d = load_well(WELL_C).model_dump()
    d["correlations"] = [{"a": "pore.shift_ppg", "b": "pipe.yield_ratio", "rho": 1.0},
                         {"a": "pore.shift_ppg", "b": "pipe.wall_ratio", "rho": 1.0},
                         {"a": "pipe.yield_ratio", "b": "pipe.wall_ratio", "rho": -1.0}]
    w = WellInput.model_validate(d)
    p = build_profiles(w)
    geoms, _ = resolve_strings(w, p.survey, p)
    with pytest.raises(ValueError, match="correlation"):
        collect_variables(w, geoms, "intermediate")


def test_wilson_interval_with_zero_failures():
    lo, hi = wilson_interval(0, 100_000)
    assert lo == 0 and hi == pytest.approx(3.84e-5, rel=0.01)


# --- Well-level wiring ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def well_c():
    w = load_well(WELL_C)
    p = build_profiles(w)
    geoms, _ = resolve_strings(w, p.survey, p, setting_depths(w, p).chosen_tvd_ft)
    _, pipes, _ = design_pipes(w)
    return w, p, geoms, pipes


def test_collect_variables_and_placeholders(well_c):
    w, p, geoms, _ = well_c
    vs = collect_variables(w, geoms, "intermediate")
    assert vs.names == ["pore.shift_ppg", "pipe.yield_ratio", "pipe.od_ratio", "pipe.wall_ratio"]
    assert sum("pipe_statistics" in n and "placeholder" in n for n in vs.notes) == 3
    d = w.model_dump()
    d["load_cases"]["intermediate"]["frac_down_casing"]["max_surface_psi"] = {"dist": "normal", "mean": 9500,
                                                                            "sd": 300}
    d["overburden_psi_per_ft"] = {"dist": "normal", "mean": 1.0, "sd": 0.02}
    d["pipe_statistics"] = {"yield_ratio": {"dist": "normal", "mean": 1.08, "sd": 0.04}}
    w2 = WellInput.model_validate(d)
    vs2 = collect_variables(w2, geoms, "intermediate")
    assert "load_cases.intermediate.frac_down_casing.max_surface_psi" in vs2.names
    ob = next(v for v in vs2.variables if v.name == "overburden.scale")
    assert (ob.dist.mean, ob.dist.sd) == (pytest.approx(1.0), pytest.approx(0.02))
    assert sum("placeholder" in n for n in vs2.notes) == 2


def test_limit_states_at_base_equal_deterministic_margins(well_c):
    # The probabilistic g at base values must equal min(rating - load) from the deterministic code.
    w, p, geoms, pipes = well_c
    pipe = pipes["intermediate"].pipes[0]
    vs = collect_variables(w, geoms, "intermediate")
    ls = StringLimitStates(w, p, geoms, "intermediate", pipe, vs)
    sl = string_loads(w, p, geoms, "intermediate", PipeBody(pipe.od_in, pipe.id_in, pipe.weight_lb_per_ft))
    dem = demands(sl, pipe, w.design_factors, 1.0, vs.variables[1].base, vs.variables[2].base,
                  vs.variables[3].base)
    assert [(d.case, d.check) for d in dem] == ls.labels
    expect = [float(np.min(np.broadcast_to(d.rating - d.load, sl.md_ft.shape))) for d in dem]
    assert ls.base_g == pytest.approx(expect, rel=1e-12)


def test_loads_with_base_realization_equal_deterministic(well_c):
    w, p, geoms, pipes = well_c
    body = PipeBody(7.0, 7.0 - 2 * 0.362, 26.0)
    det = string_loads(w, p, geoms, "intermediate", body)
    one = string_loads(w, p, geoms, "intermediate", body, Realization({"pore.shift_ppg": np.zeros((1, 1))}))
    for a, b in zip(det.cases, one.cases):
        assert np.atleast_2d(b.p_internal_psi)[0] == pytest.approx(a.p_internal_psi)
        assert np.atleast_2d(b.axial_real_lbf)[0] == pytest.approx(a.axial_real_lbf)


def test_string_reliability_intermediate_frac(well_c):
    w, p, geoms, pipes = well_c
    r = string_reliability(w, p, geoms, "intermediate", pipes["intermediate"], n=40_000, max_n=40_000)
    frac = next(x for x in r.rows if x.case == "frac_down_casing" and x.check == "burst")
    assert 1e-3 < frac.mc_pf < 3e-2  # fails the 1e-3 target, as the deterministic check said
    fr = form_many(StringLimitStates(w, p, geoms, "intermediate", pipes["intermediate"], r.variables),
                   r.variables, [r.rows.index(frac)])[r.rows.index(frac)]
    assert fr.converged and 0.5 < fr.pf / frac.mc_pf < 2.0  # MC vs FORM on the real limit state
    lo, hi = r.system_bounds
    assert lo <= r.mc.system_pf * 1.2 and lo <= hi
    assert not r.meets(1e-3) and r.meets(0.5)
    sweep = target_sweep({"intermediate": r})
    assert sweep[1e-2]["intermediate"] is False  # the pressure test case fails even 1e-2
    assert any("API TR 5C3" in n for n in r.notes)


def test_form_vs_importance_sampling_on_liner_collapse(well_c):
    w, p, geoms, pipes = well_c
    vs = collect_variables(w, geoms, "liner")
    ls = StringLimitStates(w, p, geoms, "liner", pipes["liner"], vs)
    j = ls.labels.index(("evacuation", "collapse"))
    fr = form_many(ls, vs, [j])[j]
    assert fr.converged and 3.5 < fr.beta < 5.5
    pf_is, cov = importance_sampling(lambda x: ls(x)[:, j], vs, fr.u_star, n=20_000, seed=0)
    assert cov < 0.1 and 0.5 < pf_is / fr.pf < 2.0


def test_pf_rises_when_load_rises(well_c):
    # Spec 6.3 sanity: a heavier load (lower design factor) must give a higher Pf.
    w, p, geoms, pipes = well_c
    pfs = []
    for mean in (8500, 9500, 10500):
        d = w.model_dump()
        d["load_cases"]["intermediate"]["frac_down_casing"]["max_surface_psi"] = mean
        w2 = WellInput.model_validate(d)
        vs = collect_variables(w2, geoms, "intermediate")
        ls = StringLimitStates(w2, p, geoms, "intermediate", pipes["intermediate"], vs)
        j = ls.labels.index(("frac_down_casing", "burst"))
        pfs.append(form_many(ls, vs, [j])[j].pf)
    assert pfs[0] < pfs[1] < pfs[2]


def test_shoe_move_probability(well_c):
    w, p, _, _ = well_c
    moves = shoe_move_probability(w, p, n=5_000)
    assert [m.string for m in moves] == ["surface", "intermediate"]
    assert all(0.0 <= m.probability <= 1.0 for m in moves)
    d = w.model_dump()
    d["pore_pressure_ppg"]["uncertainty"] = None
    w2 = WellInput.model_validate(d)
    assert all(m.probability == 0.0 for m in shoe_move_probability(w2, build_profiles(w2), n=1_000))


def test_biaxial_guard_beyond_axial_yield():
    assert rt.reduced_yield_biaxial_psi(80_000, 80_000) == pytest.approx(0.0)
    assert rt.collapse_api_combined_psi(7.0, 0.362, 80_000, 200_000) == pytest.approx(0.0)
    vals = rt.collapse_api_combined_psi(7.0, 0.362, 80_000, np.linspace(-120_000, 120_000, 241))
    assert np.all(np.isfinite(vals)) and np.all(vals >= 0)
