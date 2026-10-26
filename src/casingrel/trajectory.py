"""Well trajectory by the minimum-curvature method.

Each pair of survey stations is joined by a circular arc. The arc's length
is the measured-depth step; its dogleg is the angle between the two
stations' unit tangent vectors. Positions between stations are interpolated
along the same arc, so the result is exact for a constant-rate build or turn.

Sources: Sawaryn & Thorogood (2005), "A compendium of directional
calculations based on the minimum curvature method", SPE Drilling &
Completion 20(1), SPE-84246-PA; Bourgoyne et al. (1986), Ch. 8.
TODO(citation): add equation numbers from the user's copies (CLAUDE.md rule 3).

Depth: MD for lengths and weights, TVD for pressures (CLAUDE.md rule 5).
Coordinates: north, east and TVD (positive down), all in ft from KB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .schema import PlanTrajectory, SurveyTrajectory, VerticalTrajectory

_SMALL_RAD = 1e-9  # below this dogleg the arc is treated as a straight line


@dataclass(frozen=True)
class SurveyPoints:
    """Trajectory values at a set of measured depths."""

    md_ft: np.ndarray
    tvd_ft: np.ndarray
    inc_deg: np.ndarray
    azi_deg: np.ndarray
    north_ft: np.ndarray
    east_ft: np.ndarray
    dls_deg_per_100ft: np.ndarray


@dataclass(frozen=True)
class Survey:
    """A minimum-curvature survey. dls_deg_per_100ft[i] is the dogleg severity
    of the arc that ends at station i (0 for the first station)."""

    md_ft: np.ndarray
    inc_deg: np.ndarray
    azi_deg: np.ndarray
    tvd_ft: np.ndarray
    north_ft: np.ndarray
    east_ft: np.ndarray
    dls_deg_per_100ft: np.ndarray

    @property
    def td_md_ft(self) -> float:
        return float(self.md_ft[-1])

    @property
    def td_tvd_ft(self) -> float:
        return float(self.tvd_ft[-1])

    @property
    def max_tvd_ft(self) -> float:
        return float(self.tvd_ft.max())

    def at_md(self, md_ft: ArrayLike) -> SurveyPoints:
        """Interpolate along the minimum-curvature arcs at any MD in [0, TD]."""
        md = np.atleast_1d(np.asarray(md_ft, dtype=float))
        if np.any(md < self.md_ft[0] - 1e-9) or np.any(md > self.md_ft[-1] + 1e-9):
            raise ValueError(f"MD outside survey range 0 to {self.td_md_ft:.1f} ft")
        i = np.clip(np.searchsorted(self.md_ft, md, side="right") - 1, 0, len(self.md_ft) - 2)
        md1, md2 = self.md_ft[i], self.md_ft[i + 1]
        t1 = _tangent(self.inc_deg[i], self.azi_deg[i])
        t2 = _tangent(self.inc_deg[i + 1], self.azi_deg[i + 1])
        beta = _dogleg_rad(self.inc_deg[i], self.azi_deg[i], self.inc_deg[i + 1], self.azi_deg[i + 1])
        frac = (md - md1) / (md2 - md1)
        t = _slerp(t1, t2, beta, frac)
        step = _min_curvature_step(md - md1, t1, t, beta * frac)
        inc = np.degrees(np.arccos(np.clip(t[2], -1.0, 1.0)))
        azi = np.where(np.hypot(t[0], t[1]) > 1e-12,
                       np.degrees(np.arctan2(t[1], t[0])) % 360.0, self.azi_deg[i])
        return SurveyPoints(
            md_ft=md, tvd_ft=self.tvd_ft[i] + step[2], inc_deg=inc, azi_deg=azi,
            north_ft=self.north_ft[i] + step[0], east_ft=self.east_ft[i] + step[1],
            dls_deg_per_100ft=self.dls_deg_per_100ft[i + 1])

    def tvd_at_md(self, md_ft: ArrayLike) -> np.ndarray:
        return self.at_md(md_ft).tvd_ft

    def md_at_tvd(self, tvd_ft: float, tol_ft: float = 1e-6) -> float:
        """Shallowest MD at which the well reaches `tvd_ft` (bisection on the arcs)."""
        if not 0.0 <= tvd_ft <= self.max_tvd_ft + 1e-9:
            raise ValueError(f"TVD {tvd_ft} ft outside 0 to {self.max_tvd_ft:.1f} ft")
        k = int(np.argmax(self.tvd_ft >= tvd_ft - 1e-9))
        if k == 0:
            return float(self.md_ft[0])
        lo, hi = float(self.md_ft[k - 1]), float(self.md_ft[k])
        while hi - lo > tol_ft:
            mid = 0.5 * (lo + hi)
            if self.tvd_at_md(mid)[0] < tvd_ft:
                lo = mid
            else:
                hi = mid
        return hi

    def resample(self, step_ft: float = 100.0, extra_md_ft: ArrayLike = ()) -> SurveyPoints:
        """Points every `step_ft` of MD, plus every station and any extra MDs."""
        md = np.concatenate([np.arange(0.0, self.td_md_ft, step_ft), self.md_ft,
                             np.atleast_1d(np.asarray(extra_md_ft, dtype=float))])
        return self.at_md(np.unique(md))


def minimum_curvature(md_ft: ArrayLike, inc_deg: ArrayLike, azi_deg: ArrayLike) -> Survey:
    """Compute TVD, north, east and DLS at each station by minimum curvature.

    For each arc: dogleg beta from the two tangents, ratio factor
    RF = (2/beta) tan(beta/2), and displacement = (dMD/2) RF (t1 + t2).
    """
    md = np.asarray(md_ft, dtype=float)
    inc = np.asarray(inc_deg, dtype=float)
    azi = np.asarray(azi_deg, dtype=float)
    if md.ndim != 1 or len(md) < 2 or not (md.shape == inc.shape == azi.shape):
        raise ValueError("need at least two stations with matching md, inc and azi")
    if np.any(np.diff(md) <= 0):
        raise ValueError("station MDs must increase")
    beta = _dogleg_rad(inc[:-1], azi[:-1], inc[1:], azi[1:])
    step = _min_curvature_step(np.diff(md), _tangent(inc[:-1], azi[:-1]), _tangent(inc[1:], azi[1:]), beta)
    pos = np.concatenate([np.zeros((3, 1)), np.cumsum(step, axis=1)], axis=1)
    dls = np.concatenate([[0.0], np.degrees(beta) / np.diff(md) * 100.0])
    return Survey(md_ft=md, inc_deg=inc, azi_deg=azi % 360.0, tvd_ft=pos[2], north_ft=pos[0],
                  east_ft=pos[1], dls_deg_per_100ft=dls)


def build_survey(trajectory: SurveyTrajectory | VerticalTrajectory | PlanTrajectory) -> Survey:
    """Turn any schema trajectory into a minimum-curvature survey."""
    if isinstance(trajectory, SurveyTrajectory):
        st = trajectory.stations
        return minimum_curvature([s.md_ft for s in st], [s.inc_deg for s in st], [s.azi_deg for s in st])
    if isinstance(trajectory, VerticalTrajectory):
        return minimum_curvature([0.0, trajectory.td_ft], [0.0, 0.0], [0.0, 0.0])
    # Plan: vertical to KOP, build at constant rate to final inclination, then the lateral.
    kop = trajectory.kop_tvd_ft
    eob = kop + trajectory.final_inc_deg / trajectory.build_deg_per_100ft * 100.0
    md, inc = [0.0], [0.0]
    if kop > 0:
        md, inc = md + [kop], inc + [0.0]
    md, inc = md + [eob], inc + [trajectory.final_inc_deg]
    if trajectory.lateral_ft > 0:
        md, inc = md + [eob + trajectory.lateral_ft], inc + [trajectory.final_inc_deg]
    return minimum_curvature(md, inc, [trajectory.azimuth_deg] * len(md))


def _tangent(inc_deg: ArrayLike, azi_deg: ArrayLike) -> np.ndarray:
    """Unit tangent (north, east, down), shape (3, ...)."""
    i, a = np.radians(inc_deg), np.radians(azi_deg)
    return np.array([np.sin(i) * np.cos(a), np.sin(i) * np.sin(a), np.cos(i)])


def _dogleg_rad(inc1, azi1, inc2, azi2) -> np.ndarray:
    """Dogleg angle between two stations, in the haversine form that stays
    accurate for small angles (the arccos form loses precision there)."""
    i1, i2 = np.radians(inc1), np.radians(inc2)
    da = np.radians(np.asarray(azi2) - np.asarray(azi1))
    h = np.sin((i2 - i1) / 2.0) ** 2 + np.sin(i1) * np.sin(i2) * np.sin(da / 2.0) ** 2
    return 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _ratio_factor(beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    safe = np.where(beta > _SMALL_RAD, beta, 1.0)
    return np.where(beta > _SMALL_RAD, 2.0 / safe * np.tan(safe / 2.0), 1.0 + beta**2 / 12.0)


def _min_curvature_step(dmd, t1, t2, beta) -> np.ndarray:
    return 0.5 * np.asarray(dmd) * _ratio_factor(beta) * (t1 + t2)


def _slerp(t1: np.ndarray, t2: np.ndarray, beta: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Tangent a fraction `frac` of the way along the arc from t1 to t2."""
    big = beta > _SMALL_RAD
    sb = np.where(big, np.sin(beta), 1.0)
    w1 = np.where(big, np.sin((1.0 - frac) * beta) / sb, 1.0 - frac)
    w2 = np.where(big, np.sin(frac * beta) / sb, frac)
    t = w1 * t1 + w2 * t2
    return t / np.linalg.norm(t, axis=0)
