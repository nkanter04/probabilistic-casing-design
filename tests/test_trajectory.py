import numpy as np
import pytest

from casingrel.schema import PlanTrajectory, SurveyTrajectory, VerticalTrajectory
from casingrel.trajectory import build_survey, minimum_curvature


def test_spec_build_0_to_90_at_9_deg_per_100ft():
    # Spec 6.2: 1,000 ft MD and 636.6 ft TVD gain.
    s = minimum_curvature([0, 1000], [0, 90], [0, 0])
    assert s.tvd_ft[-1] == pytest.approx(636.6, abs=0.05)
    assert s.north_ft[-1] == pytest.approx(18000 / np.pi / 9, rel=1e-12)
    assert s.dls_deg_per_100ft[-1] == pytest.approx(9.0)


def test_vertical_and_straight_tangent():
    s = minimum_curvature([0, 5000], [0, 0], [0, 0])
    assert (s.tvd_ft[-1], s.north_ft[-1], s.dls_deg_per_100ft[-1]) == (5000, 0, 0)
    s = minimum_curvature([0, 1000], [30, 30], [45, 45])
    assert s.tvd_ft[-1] == pytest.approx(1000 * np.cos(np.radians(30)))
    assert s.north_ft[-1] == pytest.approx(500 * np.cos(np.radians(45)))
    assert s.east_ft[-1] == pytest.approx(500 * np.sin(np.radians(45)))


def test_interpolation_follows_the_arc():
    # Any point on a constant-rate build lies on a circle of radius R.
    r_ft = 18000 / np.pi / 9
    s = minimum_curvature([0, 1000], [0, 90], [90, 90])
    md = np.linspace(0, 1000, 41)
    pts = s.at_md(md)
    theta = np.radians(md * 9 / 100)
    assert pts.tvd_ft == pytest.approx(r_ft * np.sin(theta), abs=1e-9)
    assert pts.east_ft == pytest.approx(r_ft * (1 - np.cos(theta)), abs=1e-9)
    assert pts.north_ft == pytest.approx(0, abs=1e-9)
    assert pts.inc_deg == pytest.approx(md * 9 / 100, abs=1e-9)
    assert np.all(pts.dls_deg_per_100ft == pytest.approx(9.0))


def test_interpolation_hits_stations_in_3d():
    md, inc, azi = [0, 1500, 3000, 4500, 6000], [0, 10, 35, 60, 88], [0, 20, 75, 140, 200]
    s = minimum_curvature(md, inc, azi)
    pts = s.at_md(md)
    assert pts.tvd_ft == pytest.approx(s.tvd_ft, abs=1e-9)
    assert pts.north_ft == pytest.approx(s.north_ft, abs=1e-9)
    assert pts.east_ft == pytest.approx(s.east_ft, abs=1e-9)
    assert pts.inc_deg[1:] == pytest.approx(inc[1:], abs=1e-9)
    assert pts.azi_deg[1:] == pytest.approx(azi[1:], abs=1e-9)


def test_splitting_an_arc_does_not_change_it():
    # Adding an interpolated station must leave the survey unchanged (exactness of the arc).
    s = minimum_curvature([0, 2000, 4000], [0, 40, 70], [10, 60, 120])
    mid = s.at_md([3100.0])
    s2 = minimum_curvature([0, 2000, 3100, 4000], [0, 40, mid.inc_deg[0], 70], [10, 60, mid.azi_deg[0], 120])
    assert s2.tvd_ft[-1] == pytest.approx(s.tvd_ft[-1], abs=1e-8)
    assert s2.north_ft[-1] == pytest.approx(s.north_ft[-1], abs=1e-8)
    assert s2.east_ft[-1] == pytest.approx(s.east_ft[-1], abs=1e-8)


def test_small_dogleg_is_stable():
    s = minimum_curvature([0, 100], [10.0, 10.0 + 1e-10], [0, 0])
    assert s.tvd_ft[-1] == pytest.approx(100 * np.cos(np.radians(10)), rel=1e-12)


def test_plan_survey_matches_plan_geometry():
    # Appendix A geometry: KOP 8,963 ft, 9 deg/100 ft, lands at 9,600 ft TVD.
    plan = PlanTrajectory(kop_ft=8963, build_deg_per_100ft=9, lateral_ft=5300)
    s = build_survey(plan)
    assert s.td_tvd_ft == pytest.approx(plan.td_tvd_ft) == pytest.approx(9599.6, abs=0.05)
    assert s.td_md_ft == pytest.approx(plan.td_md_ft) == pytest.approx(8963 + 1000 + 5300)
    assert s.at_md([8963 + 500]).inc_deg[0] == pytest.approx(45)


def test_vertical_and_survey_inputs():
    assert build_survey(VerticalTrajectory(td_ft=6000)).td_tvd_ft == 6000
    st = SurveyTrajectory(stations=[{"md_ft": 0, "inc_deg": 0, "azi_deg": 0},
                                    {"md_ft": 1000, "inc_deg": 90, "azi_deg": 0}])
    assert build_survey(st).td_tvd_ft == pytest.approx(636.6, abs=0.05)


def test_md_at_tvd_inverts_tvd_at_md():
    s = build_survey(PlanTrajectory(kop_ft=8000, build_deg_per_100ft=8, lateral_ft=5000))
    for tvd in (0.0, 5000.0, 8000.0, 8200.0, 8700.0):
        md = s.md_at_tvd(tvd)
        assert s.tvd_at_md(md)[0] == pytest.approx(tvd, abs=1e-4)
    assert s.md_at_tvd(5000.0) == pytest.approx(5000.0, abs=1e-4)
    with pytest.raises(ValueError):
        s.md_at_tvd(9000.0)


def test_resample_and_range_checks():
    s = build_survey(PlanTrajectory(kop_ft=8000, build_deg_per_100ft=8, lateral_ft=500))
    pts = s.resample(100.0, extra_md_ft=[8050.5])
    assert pts.md_ft[0] == 0 and pts.md_ft[-1] == s.td_md_ft
    assert 8050.5 in pts.md_ft and set(s.md_ft) <= set(pts.md_ft)
    assert np.all(np.diff(pts.md_ft) > 0)
    with pytest.raises(ValueError):
        s.at_md([s.td_md_ft + 10])
    with pytest.raises(ValueError):
        minimum_curvature([0, 100, 100], [0, 0, 0], [0, 0, 0])
