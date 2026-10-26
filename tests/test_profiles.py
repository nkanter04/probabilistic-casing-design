import copy

import numpy as np
import pytest

from casingrel.profiles import DepthCurve, build_profiles, eaton_frac_ppg, ppg_to_psi
from casingrel.schema import WellInput, load_well
from casingrel.synthetic import DEFAULT_EXAMPLES_DIR, bundled_wells

BASE = {
    "well": {"name": "T", "kb_ft": 25},
    "trajectory": {"td_ft": 10_000},
    "pore_pressure_ppg": {0: 8.6, 6000: 8.6, 10_000: 12.6},
    "frac_gradient": {"method": "eaton", "poisson": {0: 0.25, 10_000: 0.25}},
    "overburden_psi_per_ft": 1.0,
    "temperature": {"surface_F": 75, "gradient_F_per_100ft": 1.5},
    "strings": [{"name": "surface", "hole_in": 12.25, "od_in": 9.625, "shoe_tvd_ft": 2000, "toc": "surface"}],
}
OB_PPG = 1.0 / 0.052


def profiles(**changes):
    d = copy.deepcopy(BASE)
    d.update(changes)
    return build_profiles(WellInput.model_validate(d))


def test_hydrostatic_and_gas_gradient_hand_calc():
    assert ppg_to_psi(10.0, 10_000) == pytest.approx(5_200.0)
    # Spec 6.2 inputs: 4,521 psi at 8,963 ft is 9.70 ppg.
    assert 4521 / (0.052 * 8963) == pytest.approx(9.70, abs=0.005)


def test_pore_interpolation_and_pressure():
    p = profiles()
    assert p.pore_ppg([0, 3000, 8000, 10_000]) == pytest.approx([8.6, 8.6, 10.6, 12.6])
    assert p.pore_psi(8000) == pytest.approx(0.052 * 10.6 * 8000)


def test_eaton_hand_calc():
    # nu = 0.25 -> nu/(1 - nu) = 1/3; FG = (19.2308 - 8.6)/3 + 8.6 = 12.1436 ppg.
    p = profiles()
    assert p.frac_ppg(3000) == pytest.approx((OB_PPG - 8.6) / 3 + 8.6, abs=1e-9)
    assert eaton_frac_ppg(8.6, OB_PPG, 0.25) == pytest.approx(12.1436, abs=1e-4)
    # Eaton limits: nu -> 0 gives FG = Pp; nu = 0.5 gives FG = Sv.
    assert eaton_frac_ppg(9.0, OB_PPG, 0.0) == pytest.approx(9.0)
    assert eaton_frac_ppg(9.0, OB_PPG, 0.5) == pytest.approx(OB_PPG)


def test_frac_follows_sampled_pore_pressure():
    # Spec 2.2: frac is computed from the sampled pore pressure. With nu = 0.25,
    # a +1 ppg pore shift moves FG by 1 - 1/3 = 2/3 ppg.
    p = profiles()
    base = p.frac_ppg(8000)
    assert p.frac_ppg(8000, pore_shift_ppg=1.0) - base == pytest.approx(2 / 3)
    assert p.frac_ppg(8000, overburden_scale=1.05) > base


def test_modifiers_broadcast_samples_by_nodes():
    p = profiles()
    z = np.linspace(0, 10_000, 11)
    shift = np.array([[-0.3], [0.0], [0.3]])
    fg = p.frac_ppg(z, pore_shift_ppg=shift, poisson_shift=np.zeros((3, 1)))
    assert fg.shape == (3, 11)
    assert fg[1] == pytest.approx(p.frac_ppg(z))
    assert np.all(fg[2] > fg[0])
    assert p.pore_psi(z, scale=np.array([[1.0], [1.1]])).shape == (2, 11)


def test_poisson_clipped_below_half():
    p = profiles()
    assert p.poisson_ratio(5000, shift=0.4) == pytest.approx(0.4999)
    assert p.poisson_ratio(5000, shift=-0.4) == 0.0


def test_overburden_forms():
    assert profiles().overburden_psi(10_000) == pytest.approx(10_000)
    p = profiles(overburden_psi_per_ft={0: 0.9, 10_000: 1.0})
    assert p.overburden_psi(5000) == pytest.approx(0.95 * 5000)
    p = profiles(overburden_psi_per_ft={"dist": "normal", "mean": 0.98, "sd": 0.02})
    assert p.overburden_psi(1000) == pytest.approx(980)
    d = copy.deepcopy(BASE)
    del d["overburden_psi_per_ft"]
    assert build_profiles(WellInput.model_validate(d)).overburden_psi(1000) == pytest.approx(1000)


def test_temperature_forms():
    p = profiles()
    assert p.temperature_F([0, 10_000]) == pytest.approx([75, 225])
    assert p.temperature_F(10_000, gradient_F_per_100ft=np.array([[1.0], [2.0]])) == pytest.approx(np.array([[175], [275]]))
    p = profiles(temperature={0: 70, 10_000: 250})
    assert p.temperature_F(5000) == pytest.approx(160)
    p = profiles(temperature={"surface_F": {"dist": "uniform", "low": 70, "high": 80}, "gradient_F_per_100ft": 1.5})
    assert p.temperature_F(0) == pytest.approx(75)


def test_extrapolation_is_flagged():
    # Appendix A pattern: pore table stops above TD.
    p = profiles(pore_pressure_ppg={2600: 8.3, 8726: 9.7})
    text = " ".join(p.warnings)
    assert "pore pressure: table starts at 2600" in text
    assert "pore pressure: table ends at 8726" in text
    assert p.pore_ppg(9500) == pytest.approx(9.7)
    assert DepthCurve.constant("x", "ppg", 9.0).coverage_warnings(20_000) == []


def test_sanity_warnings():
    p = profiles(pore_pressure_ppg={0: 8.6, 10_000: 20.0})
    assert any("pore pressure at or above overburden" in w for w in p.warnings)
    p = profiles(frac_gradient={"method": "table", "table": {0: 12.0, 10_000: 12.0}})
    assert any("fracture gradient at or below pore pressure" in w for w in p.warnings)
    p = profiles(frac_gradient={"method": "eaton", "poisson": {0: 0.25, 10_000: 0.6}})
    assert any("Poisson's ratio outside" in w for w in p.warnings)


def test_frac_table_ignores_pore_shift_and_is_flagged_when_pore_is_uncertain():
    pore = {"table": BASE["pore_pressure_ppg"], "uncertainty": {"shift_sd": 0.3}}
    p = profiles(pore_pressure_ppg=pore, frac_gradient={"method": "table", "table": {0: 13.0, 10_000: 16.0}})
    assert p.frac_ppg(5000, pore_shift_ppg=np.array([[0.0], [1.0]])) == pytest.approx(np.array([[14.5], [14.5]]))
    assert any("will not follow sampled pore pressure" in w for w in p.warnings)


@pytest.mark.parametrize("fname", sorted(bundled_wells()))
def test_bundled_wells_have_no_profile_warnings(fname):
    # Spec 6.2: bundled synthetic wells run with no warnings beyond labelled placeholders.
    p = build_profiles(load_well(DEFAULT_EXAMPLES_DIR / fname))
    assert p.warnings == []
