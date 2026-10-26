import copy

import numpy as np
import pytest

from casingrel import ratings as rt
from casingrel.design import (check_string, deepest_reachable_tvd, kick_tolerance_ppg, lame_stresses_psi,
                              setting_depths, triaxial_vme_inner_psi, von_mises_psi)
from casingrel.loads import resolve_strings
from casingrel.profiles import build_profiles
from casingrel.ratings import Pipe
from casingrel.schema import WellInput, load_well
from casingrel.synthetic import DEFAULT_EXAMPLES_DIR

L80_40 = Pipe(9.625, 0.395, 40.0, "L80")
P110_26 = Pipe(7.0, 0.362, 26.0, "P110")

BASE = {
    "well": {"name": "T", "kb_ft": 25},
    "trajectory": {"td_ft": 9_600},
    "pore_pressure_ppg": {0: 8.33, 2600: 8.33, 8963: "4521 psi", 9600: 9.7},
    "frac_gradient": {"method": "eaton", "poisson": {0: 0.30, 9600: 0.30}},
    "overburden_psi_per_ft": 1.0,
    "temperature": {"surface_F": 75, "gradient_F_per_100ft": 1.4},
    "fluids": {"mix_water_ppg": 8.4, "cement_tail_ppg": 15.6, "completion_ppg": 8.6},
    "strings": [
        {"name": "surface", "hole_in": 12.25, "od_in": 9.625, "shoe_tvd_ft": 2600, "toc": "surface",
         "mud_ppg": 9.0},
        {"name": "intermediate", "hole_in": 8.75, "od_in": 7.0, "shoe_tvd_ft": 8963, "toc_ft": 2000,
         "mud_ppg": 10.3},
        {"name": "production", "hole_in": 6.125, "od_in": 4.5, "shoe_md_ft": "TD", "toc_ft": 8000,
         "mud_ppg": 11.3},
    ],
    "load_cases": {"surface": {"gas_kick": {"gas_gradient_psi_per_ft": 0.10, "kick_intensity_ppg": 0.5},
                               "lost_returns": {"fluid_ppg": 8.33}, "running": {"overpull_lbf": 50_000}}},
}


def setup(**changes):
    d = copy.deepcopy(BASE)
    d.update(changes)
    w = WellInput.model_validate(d)
    p = build_profiles(w)
    return w, p


def check(name, pipe, **changes):
    w, p = setup(**changes)
    geoms, _ = resolve_strings(w, p.survey, p)
    return check_string(w, p, geoms, name, pipe)


def row(sc, case, kind):
    return next(r for r in sc.rows if r.case == case and r.check == kind)


# --- Stresses (spec 6.2) ---------------------------------------------------------------------

def test_lame_equal_pressures_give_hoop_minus_p_everywhere():
    r = np.linspace(3.0, 3.5, 11)
    s_r, s_t = lame_stresses_psi(5000.0, 5000.0, 3.0, 3.5, r)
    assert s_t == pytest.approx(-5000.0) and s_r == pytest.approx(-5000.0)


def test_lame_internal_pressure_hand_calc():
    ri, ro, pi = 3.0, 3.5, 10_000.0
    s_r, s_t = lame_stresses_psi(pi, 0.0, ri, ro, np.array([ri, ro]))
    assert s_r == pytest.approx([-pi, 0.0], abs=1e-9)
    assert s_t[0] == pytest.approx(pi * (ri**2 + ro**2) / (ro**2 - ri**2))
    assert s_t[1] == pytest.approx(2 * pi * ri**2 / (ro**2 - ri**2))


def test_von_mises_uniaxial_and_hydrostatic():
    assert von_mises_psi(0.0, 0.0, 50_000.0) == pytest.approx(50_000.0)
    assert von_mises_psi(-3000.0, -3000.0, -3000.0) == pytest.approx(0.0)


def test_triaxial_takes_worse_bending_side():
    with_bend = triaxial_vme_inner_psi(P110_26, 0.0, 0.0, 100_000.0, 10_000.0)
    s_z = 100_000.0 / P110_26.area_steel_in2
    assert with_bend == pytest.approx(s_z + 10_000.0)


# --- Kick tolerance and setting depths ------------------------------------------------------

def test_kick_tolerance_hand_calc():
    assert kick_tolerance_ppg(3000, 12.5, 10_000, 10.0) == pytest.approx(0.75)
    rho_gas = 0.1 / 0.052
    assert kick_tolerance_ppg(3000, 12.5, 10_000, 10.0, 500, 0.1) == pytest.approx(0.75 - 0.05 * (10 - rho_gas))


def test_deepest_reachable_window_and_kick_limits():
    # Pore ramps 0.0004 ppg/ft; frac constant 13.0 ppg; 0.5 ppg margins.
    w, p = setup(pore_pressure_ppg={0: 8.6, 10_000: 12.6},
                 frac_gradient={"method": "table", "table": {0: 13.0, 10_000: 13.0}},
                 trajectory={"td_ft": 10_000})
    # Window: mud (pore + 0.5, in 0.1 ppg steps) <= 12.5 until pore exceeds 12.005 ppg at 8,512.5 ft.
    assert deepest_reachable_tvd(p, 0.5, 0.5, 3000.0, 10_000.0) == 8510.0
    # Kick tolerance from a 3,000 ft shoe: (3000/z)(13.0 - mud) >= 0.5 is stricter.
    z_kt = deepest_reachable_tvd(p, 0.5, 0.5, 3000.0, 10_000.0, kick_intensity_ppg=0.5)
    assert z_kt < 8510.0
    mud = np.ceil(np.round(p.pore_ppg(z_kt) + 0.5, 2) * 10 - 1e-9) / 10
    assert kick_tolerance_ppg(3000.0, 13.0, z_kt, mud) >= 0.5
    mud_next = np.ceil(np.round(p.pore_ppg(z_kt + 10) + 0.5, 2) * 10 - 1e-9) / 10
    assert kick_tolerance_ppg(3000.0, 13.0, z_kt + 10, mud_next) < 0.5
    assert deepest_reachable_tvd(p, 0.5, 0.5, 3000.0, 5000.0) == 5000.0  # capped at the limit


def test_well_b_needs_two_intermediates():
    w = load_well(DEFAULT_EXAMPLES_DIR / "synthetic_b_deep_overpressured.yaml")
    r = setting_depths(w, build_profiles(w))
    assert set(r.chosen_tvd_ft) == {"intermediate_1", "intermediate_2"}
    assert 3000 < r.chosen_tvd_ft["intermediate_1"] < r.chosen_tvd_ft["intermediate_2"] < 16_000
    assert all(k.ok for k in r.kick_tolerance)
    assert not [x for x in r.warnings if "placeholder" not in x]
    # With one intermediate fixed at the second one's depth, the first is still needed:
    # the hole from the surface shoe cannot reach it.
    d = load_well(DEFAULT_EXAMPLES_DIR / "synthetic_b_deep_overpressured.yaml").model_dump()
    d["strings"] = [s for s in d["strings"] if s["name"] != "intermediate_1"]
    d["strings"][2]["shoe_tvd_ft"] = r.chosen_tvd_ft["intermediate_2"]
    d["load_cases"].pop("intermediate_1")
    w1 = WellInput.model_validate(d)
    r1 = setting_depths(w1, build_profiles(w1))
    assert any("below the deepest depth drillable from surface" in x for x in r1.warnings)


def test_unneeded_string_is_reported():
    strings = copy.deepcopy(BASE["strings"])
    strings.insert(1, {"name": "intermediate_extra", "hole_in": 8.75, "od_in": 7.625, "shoe_tvd_ft": "choose",
                       "toc_ft": 2000, "mud_ppg": 9.5})
    strings[2]["hole_in"] = 6.75
    strings[2]["od_in"] = 5.5
    strings[3]["hole_in"] = 4.75
    strings[3]["od_in"] = 3.5
    w, p = setup(strings=strings, pore_pressure_ppg={0: 8.6})
    r = setting_depths(w, p)
    assert any("intermediate_extra" in x and "not needed" in x for x in r.warnings)


def test_underweight_mud_is_flagged():
    strings = copy.deepcopy(BASE["strings"])
    strings[1]["mud_ppg"] = 9.0  # pore at the shoe is 9.70 ppg
    w, p = setup(strings=strings)
    assert any("intermediate: mud 9 ppg is below pore pressure + trip margin" in x
               for x in setting_depths(w, p).warnings)


# --- Design-factor check -------------------------------------------------------------------------

def test_burst_df_hand_calc_gas_to_surface():
    sc = check("surface", L80_40)
    r = row(sc, "gas_kick", "burst")
    # Reservoir pressure includes the 0.5 ppg kick intensity at the 8,963 ft next-section TD.
    p_surface = 4521 + 0.052 * 0.5 * 8963 - 0.1 * 8963
    assert r.design_factor == pytest.approx(rt.burst_api_psi(9.625, 0.395, 80_000) / p_surface, rel=1e-6)
    assert r.md_ft == 0.0 and r.ok


def test_tension_df_at_top_is_body_yield_over_hook_load():
    sc = check("surface", L80_40)
    r = row(sc, "installation", "tension")
    hook = 40.0 * (1 - 9.0 / 65.5) * 2600
    assert r.design_factor == pytest.approx(rt.pipe_body_yield_lbf(9.625, 0.395, 80_000) / hook)
    run = row(sc, "running", "tension")
    assert run.design_factor == pytest.approx(rt.pipe_body_yield_lbf(9.625, 0.395, 80_000) / (hook + 3200 * 40))


def test_collapse_df_uses_biaxial_rating_and_equivalent_pressure():
    sc = check("surface", L80_40)
    c = sc.loads.case("evacuation")
    sigma_a = c.axial_real_lbf / L80_40.area_steel_in2
    rating = rt.collapse_api_combined_psi(9.625, 0.395, 80_000, sigma_a)
    pe_eq = rt.equivalent_external_pressure_psi(c.p_external_psi, c.p_internal_psi, 9.625, 0.395)
    expect = np.min(np.where(pe_eq > 0, rating / np.where(pe_eq > 0, pe_eq, 1), np.inf))
    assert row(sc, "evacuation", "collapse").design_factor == pytest.approx(expect)


def test_triaxial_skipped_for_thin_wall_collapse_cases():
    sc = check("surface", L80_40)
    assert not [r for r in sc.rows if r.case == "evacuation" and r.check == "triaxial"]
    assert any("triaxial not applied to collapse cases" in n for n in sc.notes)
    assert row(sc, "gas_kick", "triaxial").design_factor > 0


def test_connection_rating_limits_tension():
    weak = Pipe(9.625, 0.395, 40.0, "L80", conn_tension_lbf=300_000)
    sc = check("surface", weak)
    hook = 40.0 * (1 - 9.0 / 65.5) * 2600
    assert row(sc, "installation", "tension").design_factor == pytest.approx(300_000 / hook)
    assert not any("no connection ratings" in n for n in sc.notes)


def test_temperature_derating_lowers_ratings():
    w, p = setup()
    geoms, _ = resolve_strings(w, p.survey, p)
    base = check_string(w, p, geoms, "surface", L80_40)
    derated = check_string(w, p, geoms, "surface", L80_40, yield_derate=lambda t: np.full_like(t, 0.9))
    assert row(derated, "gas_kick", "burst").design_factor == pytest.approx(
        0.9 * row(base, "gas_kick", "burst").design_factor)
    assert any("no temperature derating" in n for n in base.notes)


def test_max_surface_treating_pressure_is_where_burst_df_hits_required():
    lc = {"intermediate": {"frac_down_casing": {"max_surface_psi": 8000, "fluid_ppg": 8.4}}}
    sc = check("intermediate", P110_26, load_cases=lc)
    ps_max = sc.max_surface_treating_psi
    lc["intermediate"]["frac_down_casing"]["max_surface_psi"] = ps_max
    at_max = check("intermediate", P110_26, load_cases=lc)
    assert row(at_max, "frac_down_casing", "burst").design_factor == pytest.approx(1.10, abs=1e-9)


def test_well_c_intermediate_fails_frac_at_9500_psi():
    # 7-in 26 lb/ft P110 under 9,500 psi injection down casing on an 8.4 ppg column:
    # burst DF < 1.10, so the deterministic check rejects it (spec 3.2 frac lesson).
    w = load_well(DEFAULT_EXAMPLES_DIR / "synthetic_c_horizontal_frac.yaml")
    p = build_profiles(w)
    geoms, _ = resolve_strings(w, p.survey, p)
    sc = check_string(w, p, geoms, "intermediate", P110_26)
    assert not row(sc, "frac_down_casing", "burst").ok
    assert sc.max_surface_treating_psi < 9500
    assert not sc.ok


def test_pipe_defaults_and_validation():
    assert Pipe(5.5, 0.304, 17.0, "P110").yield_psi == 110_000
    assert Pipe(5.5, 0.304, 17.0, "X95", yield_psi=95_000).yield_psi == 95_000
    with pytest.raises(ValueError):
        Pipe(5.5, 0.304, 17.0, "X95")
    with pytest.raises(ValueError):
        Pipe(5.5, 3.0, 17.0, "P110")
