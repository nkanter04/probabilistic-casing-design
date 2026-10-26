import copy
import math

import numpy as np
import pytest

from casingrel import loads as L
from casingrel.loads import PipeBody, resolve_strings, string_loads
from casingrel.profiles import build_profiles
from casingrel.schema import WellInput, load_well
from casingrel.synthetic import DEFAULT_EXAMPLES_DIR

# Vertical well with the spec 6.2 numbers: 4,521 psi pore pressure at 8,963 ft.
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
         "mud_ppg": 9.8},
        {"name": "production", "hole_in": 6.125, "od_in": 4.5, "shoe_md_ft": "TD", "toc_ft": 8000,
         "mud_ppg": 11.3},
    ],
    "load_cases": {"surface": {"gas_kick": {"gas_gradient_psi_per_ft": 0.10},
                               "lost_returns": {"fluid_ppg": 8.33}, "running": {"overpull_lbf": 50_000}}},
}
SURFACE_PIPE = PipeBody.from_wall(9.625, 0.395, 40.0)
INTER_PIPE = PipeBody.from_wall(7.0, 0.362, 26.0)


def setup(**changes):
    d = copy.deepcopy(BASE)
    d.update(changes)
    w = WellInput.model_validate(d)
    p = build_profiles(w)
    geoms, notes = resolve_strings(w, p.survey, p)
    return w, p, geoms, notes


def loads_for(name, pipe=None, **changes):
    w, p, geoms, _ = setup(**changes)
    return string_loads(w, p, geoms, name, pipe)


# --- Spec 6.2 required values ---------------------------------------------------------

def test_spec_gas_to_surface_3625_psi():
    sl = loads_for("surface")
    gas = sl.case("gas_kick")
    assert gas.p_internal_psi[0] == pytest.approx(4521 - 0.10 * 8963, abs=0.5)
    assert gas.p_internal_psi[0] == pytest.approx(3625, abs=0.5)
    # Linear in TVD with the gas gradient.
    assert np.diff(gas.p_internal_psi) / np.diff(gas.tvd_ft) == pytest.approx(0.10)


def test_spec_dry_hanging_loads():
    assert L.effective_tension_lbf(0.0, 2600, 40.0) == pytest.approx(104_000)
    assert L.effective_tension_lbf(0.0, 8963, 26.0) == pytest.approx(233_038)


def test_spec_bending_stress():
    # 9 deg/100 ft on 4-1/2 in pipe gives about 8,800 psi.
    assert L.bending_stress_psi(4.5, 9.0) == pytest.approx(8829)


# --- Sign conventions -------------------------------------------------------------------

def test_load_signs():
    c = L.LoadCase("x", L.BURST, np.zeros(1), np.zeros(1), np.array([5000.0]), np.array([2000.0]))
    assert c.load_psi[0] == 3000
    c = L.LoadCase("x", L.COLLAPSE, np.zeros(1), np.zeros(1), np.array([5000.0]), np.array([2000.0]))
    assert c.load_psi[0] == -3000


# --- Pressure profiles ------------------------------------------------------------------

def test_frac_down_casing_grows_with_depth():
    # Spec 3.2 common mistake: injection pressure sits on a full column, it is not uniform.
    z = np.array([0.0, 5000.0, 9600.0])
    pi = L.surface_pressure_column_psi(z, 9500.0, 8.4)
    assert pi == pytest.approx(9500 + 0.052 * 8.4 * z)
    assert pi[-1] - pi[0] == pytest.approx(0.052 * 8.4 * 9600)


def test_burst_backup_is_mud_then_mix_water_then_pore():
    z = np.array([0.0, 1000.0, 2000.0, 2600.0, 5000.0])
    pore = 0.052 * 8.6 * z
    pe = L.burst_backup_psi(z, mud_ppg=10.0, toc_tvd_ft=1000.0, outer_shoe_tvd_ft=2600.0,
                            mix_water_ppg=8.4, pore_psi=pore)
    assert pe[:2] == pytest.approx(0.052 * 10.0 * z[:2])
    assert pe[2] == pytest.approx(520 + 0.052 * 8.4 * 1000)
    assert pe[3] == pytest.approx(520 + 0.052 * 8.4 * 1600)
    assert pe[4] == pytest.approx(pore[4])
    # TOC below the outer shoe: mud to TOC, then pore pressure.
    pe = L.burst_backup_psi(z, 10.0, 3000.0, 2600.0, 8.4, pore)
    assert pe[3] == pytest.approx(0.052 * 10.0 * 2600) and pe[4] == pytest.approx(pore[4])


def test_collapse_backup_never_below_pore():
    z = np.array([1000.0, 9000.0])
    pe = L.collapse_backup_psi(z, 9.0, 0.052 * np.array([8.6, 12.0]) * z)
    assert pe == pytest.approx([0.052 * 9.0 * 1000, 0.052 * 12.0 * 9000])


def test_cementing_external_continuous_at_toc():
    z = np.array([0.0, 2000.0, 2000.0 + 1e-6, 5000.0])
    pe = L.cementing_external_psi(z, 9.0, 2000.0, 15.6)
    assert pe[1] == pytest.approx(pe[2], abs=1e-3)
    assert pe[3] == pytest.approx(0.052 * 9.0 * 2000 + 0.052 * 15.6 * 3000)


def test_evacuation_full_and_partial():
    z = np.array([0.0, 3000.0, 5000.0])
    assert np.all(L.evacuated_psi(z, None, 9.8) == 0)
    assert L.evacuated_psi(z, 3000.0, 10.0) == pytest.approx([0, 0, 0.052 * 10 * 2000])


def test_lost_returns_starts_at_shoe_frac_pressure():
    w, p, geoms, _ = setup()
    sl = string_loads(w, p, geoms, "surface")
    lr = sl.case("lost_returns")
    shoe = np.argmax(sl.md_ft)
    assert lr.p_internal_psi[shoe] == pytest.approx(float(p.frac_psi(2600.0)))
    assert lr.p_internal_psi[0] == pytest.approx(float(p.frac_psi(2600.0)) - 0.052 * 8.33 * 2600)


def test_gas_kick_limited_to_shoe_frac():
    lc = {"surface": {"gas_kick": {"gas_gradient_psi_per_ft": 0.10, "limit_to_shoe_frac": True}}}
    w, p, geoms, _ = setup(load_cases=lc)
    gas = string_loads(w, p, geoms, "surface").case("gas_kick")
    p_frac = float(p.frac_psi(2600.0))
    assert p_frac < 4521 - 0.10 * (8963 - 2600)  # the cap is active in this well
    assert gas.p_internal_psi.max() == pytest.approx(p_frac)
    assert gas.p_internal_psi[0] == pytest.approx(p_frac - 0.10 * 2600)


def test_every_node_evaluated_including_key_depths():
    sl = loads_for("intermediate")
    for md in (0.0, 2000.0, 2600.0, 8963.0):  # top, TOC, outer shoe, shoe
        assert np.any(np.isclose(sl.md_ft, md))
    assert np.max(np.diff(sl.md_ft)) <= L.NODE_STEP_FT + 1e-9
    for c in sl.cases:
        assert c.load_psi.shape == sl.md_ft.shape


def test_primitives_broadcast_samples_by_nodes():
    z = np.linspace(0, 8963, 10)
    grad = np.array([[0.08], [0.10], [0.12]])
    pi = L.gas_to_surface_psi(z, 4521.0, 8963.0, grad)
    assert pi.shape == (3, 10) and np.all(pi[0] >= pi[2])  # lighter gas, higher pressure up the hole
    assert L.surface_pressure_column_psi(z, np.array([[9000.0], [9500.0]]), 8.4).shape == (2, 10)


# --- Axial ---------------------------------------------------------------------------------

def test_real_axial_at_bottom_is_pressure_area_compression():
    sl = loads_for("surface", SURFACE_PIPE)
    inst = next(a for a in sl.axial if a.name == "installation")
    p_shoe = 0.052 * 9.0 * 2600
    assert inst.effective_lbf[-1] == pytest.approx(0.0, abs=1e-3)
    assert inst.real_lbf[-1] == pytest.approx(-p_shoe * SURFACE_PIPE.area_steel_in2)
    assert inst.real_lbf[0] == pytest.approx(40.0 * (1 - 9.0 / 65.5) * 2600)


def test_running_adds_larger_of_overpull_and_shock():
    sl = loads_for("surface", SURFACE_PIPE)
    inst, run = sl.axial
    assert run.real_lbf - inst.real_lbf == pytest.approx(max(50_000, 3200 * 40.0))


def test_ballooning_and_thermal_signs():
    pipe = INTER_PIPE
    assert L.ballooning_force_lbf(1000.0, 0.0, pipe) == pytest.approx(0.6 * 1000 * pipe.area_inner_in2)
    assert L.ballooning_force_lbf(0.0, 1000.0, pipe) < 0
    assert L.thermal_force_lbf(pipe.area_steel_in2, -50.0) == pytest.approx(207 * pipe.area_steel_in2 * 50)


def test_frac_service_load_only_above_toc():
    lc = {"intermediate": {"frac_down_casing": {"max_surface_psi": 9500, "fluid_ppg": 8.4, "fluid_temp_F": 75}}}
    w, p, geoms, _ = setup(load_cases=lc)
    sl = string_loads(w, p, geoms, "intermediate", INTER_PIPE)
    inst = sl.axial[0]
    frac = sl.case("frac_down_casing")
    free = sl.md_ft <= 2000.0
    md, pi = sl.md_ft[free], frac.p_internal_psi[free] - 0.052 * 9.8 * sl.tvd_ft[free]
    d_balloon = 0.6 * np.trapezoid(pi, md) / 2000.0 * INTER_PIPE.area_inner_in2
    t_static = 75 + 1.4 * md / 100
    d_thermal = 207 * INTER_PIPE.area_steel_in2 * np.trapezoid(t_static - 75, md) / 2000.0
    assert frac.axial_real_lbf[free] - inst.real_lbf[free] == pytest.approx(d_balloon + d_thermal)
    assert np.all(frac.axial_real_lbf[~free] == inst.real_lbf[~free])
    assert d_balloon > 0 and d_thermal > 0  # both add tension


def test_no_service_change_when_cemented_to_surface():
    sl = loads_for("surface", SURFACE_PIPE)
    for c in sl.cases:
        assert np.all(c.axial_real_lbf == sl.axial[0].real_lbf)


def test_liner_hangs_from_hanger():
    plan = {"kop_ft": 8963, "build_deg_per_100ft": 9, "lateral_ft": 5300}
    strings = copy.deepcopy(BASE["strings"])
    strings[2] = {"name": "liner", "hole_in": 6.125, "od_in": 4.5, "top_md_ft": 8760, "shoe_md_ft": "TD",
                  "toc_ft": 8760, "mud_ppg": 11.3}
    pipe = PipeBody.from_wall(4.5, 0.25, 11.6)
    w, p, geoms, _ = setup(trajectory=plan, strings=strings)
    sl = string_loads(w, p, geoms, "liner", pipe)
    assert sl.md_ft[0] == 8760
    inst = sl.axial[0]
    assert inst.effective_lbf[0] == pytest.approx(11.6 * (1 - 11.3 / 65.5) * (9599.6 - 8760), abs=1)
    assert sl.bending_stress_psi.max() == pytest.approx(218 * 4.5 * 9)


# --- Case selection and geometry --------------------------------------------------------------

def test_default_cases_by_kind_and_switches():
    assert set(L.active_cases("surface", None)) == {"gas_kick", "lost_returns", "evacuation", "cementing", "running"}
    assert set(L.active_cases("conductor", None)) == set()
    w, p, geoms, _ = setup(load_cases={"production": {"cementing": False,
                                                      "depletion": {"pore_ppg": 6.0}}})
    names = {c.name for c in string_loads(w, p, geoms, "production").cases}
    assert names == {"evacuation", "depletion"}


def test_cases_needing_values_are_off_unless_given():
    names = {c.name for c in loads_for("production").cases}
    assert not names & {"frac_down_casing", "pressure_test", "tubing_leak", "depletion"}


def test_placeholders_are_noted():
    sl = loads_for("intermediate", INTER_PIPE)
    text = " ".join(sl.notes)
    assert "gas_gradient_psi_per_ft not given; used 0.1 (placeholder)" in text
    assert "overpull_lbf not given" in text
    strings = copy.deepcopy(BASE["strings"])
    del strings[1]["mud_ppg"]
    _, _, geoms, notes = setup(strings=strings)
    # Max pore in the section is 9.70 ppg at the shoe; + 0.5 trip margin = 10.2 ppg.
    assert "intermediate: mud_ppg not given; used 10.2 ppg" in " ".join(notes)


def test_geometry_resolution():
    _, p, geoms, _ = setup()
    surface, inter, prod = geoms
    assert (surface.outer_shoe_tvd_ft, surface.next_td_tvd_ft, surface.next_mud_ppg) == (None, 8963, 9.8)
    assert (inter.outer_shoe_tvd_ft, inter.next_td_tvd_ft) == (2600, 9600)
    assert (prod.outer_shoe_tvd_ft, prod.next_td_tvd_ft) == (8963, None)
    assert surface.toc_md_ft == 0 and inter.toc_tvd_ft == 2000


def test_choose_needs_a_depth():
    strings = copy.deepcopy(BASE["strings"])
    strings[1]["shoe_tvd_ft"] = "choose"
    w = WellInput.model_validate({**copy.deepcopy(BASE), "strings": strings})
    p = build_profiles(w)
    with pytest.raises(ValueError, match="choose"):
        resolve_strings(w, p.survey, p)
    geoms, _ = resolve_strings(w, p.survey, p, {"intermediate": 8000.0})
    assert geoms[1].shoe_md_ft == pytest.approx(8000.0)


@pytest.mark.parametrize("fname,chosen", [
    ("synthetic_a_shallow_vertical.yaml", {}),
    ("synthetic_b_deep_overpressured.yaml", {"intermediate_1": 11_370.0, "intermediate_2": 14_490.0}),
    ("synthetic_c_horizontal_frac.yaml", {}),
])
def test_bundled_wells_run(fname, chosen):
    w = load_well(DEFAULT_EXAMPLES_DIR / fname)
    p = build_profiles(w)
    geoms, notes = resolve_strings(w, p.survey, p, chosen)
    for g in geoms:
        pipe = PipeBody.from_wall(g.od_in, 0.4, 40.0)
        sl = string_loads(w, p, geoms, g.name, pipe)
        for note in notes + sl.notes:
            assert "placeholder" in note or "skipped" in note
        for c in sl.cases:
            assert np.all(np.isfinite(c.load_psi)) and np.all(np.isfinite(c.axial_real_lbf))
