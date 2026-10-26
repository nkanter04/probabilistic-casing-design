import copy

import pytest
import yaml
from pydantic import ValidationError

from casingrel.schema import (DEFAULT_OVERBURDEN_PSI_PER_FT, Distribution, Evacuation, GasKick,
                              PlanTrajectory, WellInput)

# Spec Section 2.1 minimal example, with kop_ft removed (see test below).
SPEC_EXAMPLE = yaml.safe_load("""
well: {name: Example-1, kb_ft: 25}
trajectory: {build_deg_per_100ft: 8, landing_tvd_ft: 8700, lateral_ft: 7500}
pore_pressure_ppg:
  table: {0: 8.6, 6000: 8.6, 8700: 11.5}
  uncertainty: {dist: normal, shift_sd: 0.3}
frac_gradient: {method: eaton, poisson: {0: 0.30, 8700: 0.27}}
overburden_psi_per_ft: 1.0
temperature: {surface_F: 75, gradient_F_per_100ft: 1.4}
strings:
  - {name: surface, hole_in: 12.25, od_in: 9.625, shoe_tvd_ft: choose, toc: surface}
  - {name: production, hole_in: 8.75, od_in: 5.5, shoe_md_ft: TD, toc_ft: 5000}
load_cases:
  production: {frac_down_casing: {max_surface_psi: 9500, fluid_ppg: 8.4}, evacuation: full}
design_factors: {burst: 1.10, collapse: 1.125, tension: 1.60, triaxial: 1.25}
reliability_target_pf: 1.0e-3
catalog: catalogs/api_starter.csv
""")


def example(**changes):
    d = copy.deepcopy(SPEC_EXAMPLE)
    d.update(changes)
    return d


def errors(data):
    with pytest.raises(ValidationError) as exc:
        WellInput.model_validate(data)
    return str(exc.value)


def test_spec_example_parses():
    w = WellInput.model_validate(SPEC_EXAMPLE)
    assert isinstance(w.trajectory, PlanTrajectory)
    assert w.trajectory.kop_tvd_ft == pytest.approx(7983.8, abs=0.1)
    assert w.trajectory.td_md_ft == pytest.approx(7983.8 + 1125 + 7500, abs=0.1)
    assert w.strings[0].kind == "surface" and w.strings[0].toc_ft == "surface"
    assert w.strings[1].shoe_md_ft == "TD"
    assert w.load_cases["production"].evacuation == Evacuation(level_ft=None)
    assert w.pore_pressure_ppg.table == {0.0: 8.6, 6000.0: 8.6, 8700.0: 11.5}


def test_spec_example_as_written_is_inconsistent():
    # KOP 8000 ft + 8 deg/100 ft build to 90 deg lands at 8716.2 ft TVD, not 8700.
    traj = {"kop_ft": 8000, "build_deg_per_100ft": 8, "landing_tvd_ft": 8700, "lateral_ft": 7500}
    assert "8716.2" in errors(example(trajectory=traj))


def test_si_input_matches_field_units():
    si = example(
        well={"name": "Example-1", "kb_ft": "7.62 m"},
        trajectory={"build_deg_per_100ft": "8 deg / (100 ft)", "landing_tvd_ft": "2651.76 m",
                    "lateral_ft": "2286 m"},
        pore_pressure_ppg={"table": {"0 m": "1.0305 g/cm**3", "1828.8 m": "1.0305 g/cm**3",
                                     "2651.76 m": "1.378 g/cm**3"},
                           "uncertainty": {"dist": "normal", "shift_sd": 0.3}},
        overburden_psi_per_ft="22.6206 kPa/m",
        temperature={"surface_F": "23.8889 degC", "gradient_F_per_100ft": "2.5520 delta_degC / (100 m)"},
    )
    field = WellInput.model_validate(SPEC_EXAMPLE).model_dump()
    si_dump = WellInput.model_validate(si).model_dump()
    assert si_dump["well"]["kb_ft"] == pytest.approx(25.0)
    assert si_dump["trajectory"] == pytest.approx(field["trajectory"], rel=1e-6)
    assert list(si_dump["pore_pressure_ppg"]["table"]) == pytest.approx(list(field["pore_pressure_ppg"]["table"]))
    assert list(si_dump["pore_pressure_ppg"]["table"].values()) == pytest.approx(
        list(field["pore_pressure_ppg"]["table"].values()), abs=0.005)
    assert si_dump["overburden_psi_per_ft"] == pytest.approx(1.0, abs=1e-4)
    assert si_dump["temperature"]["surface_F"] == pytest.approx(75.0, abs=1e-3)
    assert si_dump["temperature"]["gradient_F_per_100ft"] == pytest.approx(1.4, abs=1e-3)


def test_pore_pressure_given_as_pressure_becomes_gradient():
    w = WellInput.model_validate(example(pore_pressure_ppg={0: 8.6, 8963: "4521 psi"}))
    assert w.pore_pressure_ppg.table[8963.0] == pytest.approx(4521 / (0.052 * 8963))


def test_distribution_converts_mean_and_sd():
    d = Distribution(dist="normal", mean=24.0, sd=2.0, unit="degC").in_field_unit("degF")
    assert (d.mean, d.sd, d.unit) == (pytest.approx(75.2), pytest.approx(3.6), None)
    temp = {"surface_F": {"dist": "normal", "mean": 24, "sd": 2, "unit": "degC"}, "gradient_F_per_100ft": 1.4}
    w = WellInput.model_validate(example(temperature=temp))
    assert w.temperature.surface_F.mean == pytest.approx(75.2)


@pytest.mark.parametrize("bad", [
    {"dist": "normal", "mean": 9.7},
    {"dist": "normal", "mean": 9.7, "sd": -0.1},
    {"dist": "uniform", "low": 10, "high": 9},
    {"dist": "triangular", "low": 1, "mode": 5, "high": 3},
    {"dist": "normal", "mean": 9.7, "sd": 0.3, "low": 9},
])
def test_bad_distributions_rejected(bad):
    with pytest.raises(ValidationError):
        Distribution.model_validate(bad)


def test_unknown_keys_rejected():
    assert "extra" in errors(example(well={"name": "x", "kb_ft": 25, "kb_fT": 30}))
    assert "extra" in errors(example(pore_presure_ppg={0: 8.6}))


def test_wrong_dimension_rejected():
    assert "cannot be converted" in errors(example(well={"name": "x", "kb_ft": "25 psi"}))


def test_string_rules():
    s = SPEC_EXAMPLE["strings"]
    assert "outermost first" in errors(example(strings=[s[1], s[0]]))
    assert "exactly one of shoe" in errors(example(strings=[{**s[0], "shoe_md_ft": 2000}], load_cases={}))
    assert "below TD" in errors(example(strings=[{**s[0], "shoe_tvd_ft": None, "shoe_md_ft": 20_000}],
                                        load_cases={}))
    liner = {"name": "liner", "hole_in": 6.125, "od_in": 4.5, "shoe_md_ft": "TD", "toc_ft": 7000}
    assert "hanger" in errors(example(strings=[s[0], liner]))
    assert "kind" in errors(example(strings=[{**s[0], "name": "shallow string"}], load_cases={}))


def test_kind_inferred_from_name():
    s = {**SPEC_EXAMPLE["strings"][1], "name": "intermediate_1"}
    w = WellInput.model_validate(example(strings=[SPEC_EXAMPLE["strings"][0], s], load_cases={}))
    assert w.strings[1].kind == "intermediate"


def test_load_case_switches():
    lc = {"production": {"gas_kick": True, "cementing": False, "evacuation": {"level_ft": "1000 m"}}}
    w = WellInput.model_validate(example(load_cases=lc))
    cases = w.load_cases["production"]
    assert cases.gas_kick == GasKick()
    assert cases.cementing is False
    assert cases.evacuation.level_ft == pytest.approx(3280.84, abs=0.01)
    assert cases.pressure_test is None
    assert "unknown string" in errors(example(load_cases={"intermediate": {}}))


def test_placeholders_list_defaults_only():
    w = WellInput.model_validate(SPEC_EXAMPLE)
    assert w.placeholders() == [
        "design_factors.compression = 1.2 (spec 3.6 default)",
        "margins.trip_margin_ppg = 0.5 ppg (spec 3.1 default)",
        "margins.kick_margin_ppg = 0.5 ppg (spec 3.1 default)",
    ]
    bare = {k: v for k, v in SPEC_EXAMPLE.items()
            if k not in ("design_factors", "reliability_target_pf", "overburden_psi_per_ft")}
    notes = WellInput.model_validate(bare).placeholders()
    assert len([n for n in notes if n.startswith("design_factors.")]) == 5
    assert any("reliability_target_pf" in n for n in notes)
    assert any(str(DEFAULT_OVERBURDEN_PSI_PER_FT) in n and "overburden" in n for n in notes)


def test_overburden_forms():
    assert WellInput.model_validate(example(overburden_psi_per_ft={0: 0.9, 9000: 1.05})) \
        .overburden_psi_per_ft.table[9000.0] == 1.05
    dist = WellInput.model_validate(
        example(overburden_psi_per_ft={"dist": "normal", "mean": 1.0, "sd": 0.02})).overburden_psi_per_ft
    assert isinstance(dist, Distribution)


def test_frac_table_cannot_carry_its_own_uncertainty():
    # Spec 2.2: frac gradient follows sampled pore pressure, never sampled alone.
    frac = {"method": "table", "table": {0: 13.0}, "uncertainty": {"shift_sd": 0.3}}
    assert "extra" in errors(example(frac_gradient=frac))
