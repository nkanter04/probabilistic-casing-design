"""Tests for casingrel.ratings against spec Section 6.1 and 6.2.

Published values are API ratings as listed in Tenaris data sheets (spec 6.1).
API tables round ratings to the nearest 10 psi, so the tolerance against
published values is +/-0.5% or +/-10 psi, whichever is larger.
"""

import numpy as np
import pytest

from casingrel import ratings as r

J55_PSI, L80_PSI, P110_PSI = 55_000.0, 80_000.0, 110_000.0

# (label, OD in, wall in, yield psi, D/t, regime,
#  collapse formula psi, collapse table psi, burst formula psi, burst table psi)
SPEC_6_1 = [
    ("9-5/8 40 J55", 9.625, 0.395, J55_PSI, 24.37, r.PLASTIC, 2570, 2570, 3950, 3950),
    ("9-5/8 40 L80", 9.625, 0.395, L80_PSI, 24.37, r.TRANSITION, 3086, 3090, 5745, 5750),
    ("7 26 L80", 7.000, 0.362, L80_PSI, 19.34, r.PLASTIC, 5411, 5410, 7240, 7240),
    ("7 29 L80", 7.000, 0.408, L80_PSI, 17.16, r.PLASTIC, 7026, 7020, 8160, 8160),
    ("7 26 P110", 7.000, 0.362, P110_PSI, 19.34, r.PLASTIC, 6232, 6230, 9955, 9950),
    ("4-1/2 11.6 P110", 4.500, 0.250, P110_PSI, 18.00, r.PLASTIC, 7576, 7580, 10694, 10690),
]
IDS = [row[0] for row in SPEC_6_1]


def published_tol_psi(value_psi):
    return max(0.005 * value_psi, 10.0)


@pytest.mark.parametrize("row", SPEC_6_1, ids=IDS)
def test_dt_and_regime(row):
    _, od, wall, yp, dt, regime, *_ = row
    assert od / wall == pytest.approx(dt, abs=0.005)
    assert r.collapse_regime(od, wall, yp) == regime


@pytest.mark.parametrize("row", SPEC_6_1, ids=IDS)
def test_collapse_matches_formula_and_table(row):
    _, od, wall, yp, _, _, formula, table, _, _ = row
    p = r.collapse_api_psi(od, wall, yp)
    assert p == pytest.approx(formula, abs=1.0)
    assert p == pytest.approx(table, abs=published_tol_psi(table))


@pytest.mark.parametrize("row", SPEC_6_1, ids=IDS)
def test_burst_matches_formula_and_table(row):
    _, od, wall, yp, *_, formula, table = row
    p = r.burst_api_psi(od, wall, yp)
    assert p == pytest.approx(formula, abs=1.0)
    assert p == pytest.approx(table, abs=published_tol_psi(table))


def test_p110_collapse_coefficients():
    k = r.collapse_coefficients(P110_PSI)
    assert k.a == pytest.approx(3.181, abs=0.0005)
    assert k.b == pytest.approx(0.0819, abs=0.00005)
    assert k.c_psi == pytest.approx(2852, abs=0.5)
    assert k.f == pytest.approx(2.066, abs=0.0005)
    assert k.g == pytest.approx(0.0532, abs=0.00005)


@pytest.mark.parametrize("yp", [40e3, 55e3, 65e3, 80e3, 90e3, 95e3, 110e3, 125e3, 140e3])
def test_collapse_curve_is_continuous_at_regime_boundaries(yp):
    # Each boundary is defined as the D/t where adjacent regime formulas meet,
    # so the rating must not jump there. Checks the coefficient algebra.
    lim = r.collapse_regime_limits(yp)
    assert lim.dt_yield_plastic < lim.dt_plastic_transition < lim.dt_transition_elastic
    for dt in (lim.dt_yield_plastic, lim.dt_plastic_transition, lim.dt_transition_elastic):
        below = r.collapse_api_psi(dt * (1 - 1e-9), 1.0, yp)
        above = r.collapse_api_psi(dt * (1 + 1e-9), 1.0, yp)
        assert below == pytest.approx(above, rel=1e-6)


def test_collapse_falls_as_dt_rises():
    dt = np.linspace(10, 50, 401)
    p = r.collapse_api_psi(dt, 1.0, L80_PSI)
    assert np.all(np.diff(p) < 0)


def test_vectorised_matches_scalar():
    od = np.array([row[1] for row in SPEC_6_1])
    wall = np.array([row[2] for row in SPEC_6_1])
    yp = np.array([row[3] for row in SPEC_6_1])
    collapse = r.collapse_api_psi(od, wall, yp)
    burst = r.burst_api_psi(od, wall, yp)
    for i, row in enumerate(SPEC_6_1):
        assert collapse[i] == r.collapse_api_psi(row[1], row[2], row[3])
        assert burst[i] == r.burst_api_psi(row[1], row[2], row[3])
    assert list(r.collapse_regime(od, wall, yp)) == [row[5] for row in SPEC_6_1]


def test_pipe_body_yield_hand_calc():
    # 1.0 in OD, 0.25 in wall -> area = pi/4 * (1.0 - 0.25) in^2.
    expected = 80_000.0 * np.pi / 4.0 * (1.0**2 - 0.5**2)
    assert r.pipe_body_yield_lbf(1.0, 0.25, 80_000.0) == pytest.approx(expected, rel=1e-12)


# Spec 6.2: biaxial collapse.
def test_biaxial_zero_axial_stress_returns_full_yield():
    assert r.reduced_yield_biaxial_psi(P110_PSI, 0.0) == P110_PSI


def test_biaxial_collapse_falls_with_tension():
    tension_psi = np.linspace(0.0, 60_000.0, 31)
    p = r.collapse_api_combined_psi(7.0, 0.362, P110_PSI, tension_psi)
    assert p[0] == pytest.approx(r.collapse_api_psi(7.0, 0.362, P110_PSI), rel=1e-12)
    assert np.all(np.diff(p) < 0)


def test_equivalent_external_pressure():
    # D/t = 20 -> factor (1 - 2/20) = 0.9.
    assert r.equivalent_external_pressure_psi(5000.0, 1000.0, 20.0, 1.0) == pytest.approx(4100.0)
