import math

import numpy as np
import pytest
from scipy import stats

from casingrel.design import design_pipes, setting_depths
from casingrel.loads import resolve_strings
from casingrel.profiles import build_profiles
from casingrel.reliability import RandomVariable, VariableSet
from casingrel.schema import Distribution, load_well
from casingrel.sensitivity import (TreatingPoint, _crossing, _with_surface_pressure, pf_vs_treating_pressure,
                                   sobol_indices, string_sensitivity, tornado)
from casingrel.synthetic import DEFAULT_EXAMPLES_DIR

WELL_C = DEFAULT_EXAMPLES_DIR / "synthetic_c_horizontal_frac.yaml"


def rv(name, **kw):
    return RandomVariable(name, Distribution(**kw))


def test_sobol_matches_ishigami():
    # Ishigami (a=7, b=0.1) on U(-pi, pi)^3: exact S1 = (0.3139, 0.4424, 0), ST = (0.5576, 0.4424, 0.2437).
    vs = VariableSet([rv(f"x{i}", dist="uniform", low=-math.pi, high=math.pi) for i in range(3)], np.eye(3))

    def ishigami(x):
        return np.sin(x[:, 0]) + 7 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])

    s1, st = sobol_indices(ishigami, vs, n=2**15, seed=1)
    assert s1 == pytest.approx([0.3139, 0.4424, 0.0], abs=0.03)
    assert st == pytest.approx([0.5576, 0.4424, 0.2437], abs=0.03)


def test_sobol_additive_model():
    # g = x1 + 2 x2 with unit normals: variance shares 1/5 and 4/5, no interactions.
    vs = VariableSet([rv("a", dist="normal", mean=0, sd=1), rv("b", dist="normal", mean=0, sd=1)], np.eye(2))
    s1, st = sobol_indices(lambda x: x[:, 0] + 2 * x[:, 1], vs, n=2**14, seed=0)
    assert s1 == pytest.approx([0.2, 0.8], abs=0.02)
    assert st == pytest.approx([0.2, 0.8], abs=0.02)


def test_tornado_closed_form():
    # g = R - L, R ~ N(10, 1.5), L ~ N(6, 1): holding one input leaves a linear normal problem.
    vs = VariableSet([rv("R", dist="normal", mean=10, sd=1.5), rv("L", dist="normal", mean=6, sd=1)], np.eye(2))
    bars = tornado(lambda x: x[:, 0] - x[:, 1], vs)
    z = stats.norm.ppf(0.9)
    r_bar, l_bar = bars
    assert [b.variable for b in bars] == ["R", "L"]  # R has the larger swing
    assert (r_bar.beta_p10, r_bar.beta_p90) == (pytest.approx(10 - 1.5 * z - 6), pytest.approx(10 + 1.5 * z - 6))
    assert (l_bar.beta_p10, l_bar.beta_p90) == (pytest.approx((10 - (6 - z)) / 1.5), pytest.approx((10 - (6 + z)) / 1.5))
    one = tornado(lambda x: x[:, 0] - 5.0, VariableSet([rv("R", dist="normal", mean=10, sd=1)], np.eye(1)))
    assert one[0].beta_p10 == math.inf and one[0].beta_p90 == math.inf


def test_crossing_interpolates_in_log_pf():
    pts = [TreatingPoint(8000, 1e-5, 4.3), TreatingPoint(9000, 1e-1, 1.3)]
    assert _crossing(pts, 1e-3) == pytest.approx(8500)
    assert _crossing(pts, 1.0) is None and _crossing(pts, 1e-9) is None


@pytest.fixture(scope="module")
def well_c():
    w = load_well(WELL_C)
    p = build_profiles(w)
    geoms, _ = resolve_strings(w, p.survey, p, setting_depths(w, p).chosen_tvd_ft)
    _, designs, _ = design_pipes(w)
    return w, p, geoms, designs


def test_pf_rises_with_treating_pressure(well_c):
    w, p, geoms, designs = well_c
    pts, limit = pf_vs_treating_pressure(w, p, geoms, "intermediate", designs["intermediate"], 1e-3,
                                         (0.8, 0.9, 1.0, 1.1))
    pfs = [t.pf for t in pts]
    assert all(a < b for a, b in zip(pfs, pfs[1:]))
    assert [t.surface_psi for t in pts] == pytest.approx([7600, 8550, 9500, 10450])
    assert 8550 < limit < 9500
    assert 2e-3 < pts[2].pf < 2e-2  # agrees with the Monte Carlo frac Pf (~7e-3) at the input's 9,500 psi


def test_surface_pressure_distribution_keeps_spread(well_c):
    w, _, _, _ = well_c
    d = w.model_dump()
    d["load_cases"]["intermediate"]["frac_down_casing"]["max_surface_psi"] = {"dist": "normal", "mean": 9500,
                                                                            "sd": 300, "lower": 9000}
    from casingrel.schema import WellInput
    moved = _with_surface_pressure(WellInput.model_validate(d), "intermediate", 8000.0)
    ps = moved.load_cases["intermediate"].frac_down_casing.max_surface_psi
    assert (ps.mean, ps.sd, ps.lower) == (8000.0, 300.0, 7500.0)


def test_string_sensitivity_liner(well_c):
    w, p, geoms, designs = well_c
    s = string_sensitivity(w, p, geoms, "liner", designs["liner"], sobol_n=1024,
                           treating_factors=(1.0, 1.15))
    assert (s.case, s.check) == ("evacuation", "collapse")
    assert 3.5 < s.beta_base < 5.0
    # Tornado (beta swing at P10/P90) and Sobol (variance of the margin) rank differently: holding
    # the wall at its P10 still removes the thin-wall tail that drives failure, so yield swings beta
    # slightly more, while the wall explains most of the margin's variance.
    assert {b.variable for b in s.tornado[:2]} == {"pipe.yield_ratio", "pipe.wall_ratio"}
    assert s.sobol[0].variable == "pipe.wall_ratio" and s.sobol[0].total > 0.6
    assert s.treating is not None and len(s.treating) == 2
