from pathlib import Path

import numpy as np
import pytest

from casingrel.schema import FracEaton, PlanTrajectory, VerticalTrajectory, load_well
from casingrel.synthetic import (DEFAULT_EXAMPLES_DIR, SYNTHETIC_NOTE, PoreProfile, bundled_wells,
                                 make_input, pore_table, random_params, write_bundled, write_input)

BUNDLED = sorted(bundled_wells())


@pytest.mark.parametrize("fname", BUNDLED)
def test_bundled_files_are_up_to_date(fname, tmp_path):
    # The committed example files must equal what the generator writes now.
    write_bundled(tmp_path)
    committed = (DEFAULT_EXAMPLES_DIR / fname).read_text(encoding="utf-8")
    assert committed == (tmp_path / fname).read_text(encoding="utf-8")


@pytest.mark.parametrize("fname", BUNDLED)
def test_bundled_files_validate_and_are_labelled_synthetic(fname):
    path = DEFAULT_EXAMPLES_DIR / fname
    w = load_well(path)
    assert w.meta.synthetic and w.sources["all"] == SYNTHETIC_NOTE
    assert w.well.name.startswith("SYNTH-")
    assert path.read_text(encoding="utf-8").startswith(f"# {SYNTHETIC_NOTE}")
    assert isinstance(w.frac_gradient, FracEaton)
    # Only spec defaults may be placeholders in a bundled well.
    assert all("default" in note for note in w.placeholders())


def test_well_a_is_shallow_normal_pressure_two_strings():
    w = load_well(DEFAULT_EXAMPLES_DIR / "synthetic_a_shallow_vertical.yaml")
    assert isinstance(w.trajectory, VerticalTrajectory)
    assert [s.kind for s in w.strings] == ["surface", "production"]
    assert set(w.pore_pressure_ppg.table.values()) == {8.6}


def test_well_b_has_overpressure_and_two_intermediates_to_choose():
    w = load_well(DEFAULT_EXAMPLES_DIR / "synthetic_b_deep_overpressured.yaml")
    inters = [s for s in w.strings if s.kind == "intermediate"]
    assert len(inters) == 2 and all(s.shoe_tvd_ft == "choose" for s in inters)
    pp = w.pore_pressure_ppg.table
    assert pp[0.0] == pp[10_000.0] == 8.6 and pp[16_000.0] == 16.0


def test_well_c_is_horizontal_with_liner_and_frac_loads():
    w = load_well(DEFAULT_EXAMPLES_DIR / "synthetic_c_horizontal_frac.yaml")
    assert isinstance(w.trajectory, PlanTrajectory)
    assert w.trajectory.final_inc_deg == 90 and w.trajectory.lateral_ft == 7_500
    liner = next(s for s in w.strings if s.kind == "liner")
    assert liner.top_md_ft > 0 and liner.shoe_md_ft == "TD"
    assert w.load_cases["liner"].frac_down_casing.max_surface_psi == 9_500


def test_same_seed_same_file(tmp_path):
    p = bundled_wells()["synthetic_b_deep_overpressured.yaml"]
    a = write_input(make_input(p), tmp_path / "a.yaml").read_bytes()
    b = write_input(make_input(p), tmp_path / "b.yaml").read_bytes()
    assert a == b
    assert random_params(7) == random_params(7)
    assert random_params(7) != random_params(8)


def test_pore_table_shape():
    rng = np.random.default_rng(0)
    t = pore_table(10_000, PoreProfile(transition_tvd_ft=6_000, max_ppg=13.0), rng)
    assert t[0.0] == t[6_000.0] == 8.6 and t[10_000.0] == 13.0
    values = list(t.values())
    assert all(b >= a for a, b in zip(values, values[1:]))
    with pytest.raises(ValueError):
        pore_table(10_000, PoreProfile(transition_tvd_ft=12_000, max_ppg=13.0), rng)


@pytest.mark.parametrize("well_type", ["vertical", "horizontal"])
@pytest.mark.parametrize("seed", range(25))
def test_random_wells_validate(seed, well_type):
    make_input(random_params(seed, well_type))  # raises if invalid
