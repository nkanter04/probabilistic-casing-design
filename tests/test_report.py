import csv

import pytest

from casingrel.cli import main
from casingrel.design import check_well, design_pipes
from casingrel.report import DISCLAIMER, write_report
from casingrel.schema import WellInput, load_well
from casingrel.synthetic import DEFAULT_EXAMPLES_DIR, bundled_wells

WELL_C = DEFAULT_EXAMPLES_DIR / "synthetic_c_horizontal_frac.yaml"


@pytest.fixture(scope="module")
def report_c(tmp_path_factory):
    out = tmp_path_factory.mktemp("c")
    result = check_well(load_well(WELL_C))
    return result, write_report(result, out), out


def test_report_files_written(report_c):
    result, path, out = report_c
    assert path == out / "report.md"
    assert (out / "plots" / "pressure_window.png").stat().st_size > 10_000
    for g in result.geometries:
        assert (out / "plots" / f"loads_{g.name}.png").exists()


def test_report_content(report_c):
    result, path, _ = report_c
    text = path.read_text(encoding="utf-8")
    assert DISCLAIMER in text and "Synthetic well" in text
    for heading in ("## Summary", "## Setting depths and kick tolerance", "## surface", "## intermediate",
                    "## liner", "## Assumptions and warnings", "### Placeholders and defaults"):
        assert heading in text
    # The intermediate fails the frac and pressure-test burst checks; the failing cells are marked.
    assert "| intermediate | 7-in 26 lb/ft P110 |" in text and "**FAIL**" in text
    assert "**1.05 ✗**" in text
    assert "can take a surface treating pressure of up to 9,050 psi" in text
    # Every placeholder from the input and the loads reaches the assumptions list.
    for note in result.well.placeholders():
        assert note in text
    assert "running.overpull_lbf not given" in text
    assert "no drift_in given" in text


def test_every_table_has_a_plain_sentence(report_c):
    _, path, _ = report_c
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and (i + 1 == len(lines) or not lines[i + 1].startswith("|")):
            following = next(x for x in lines[i + 1:] if x.strip())
            assert not following.startswith(("|", "#", "!")), f"table ending at line {i} has no sentence"


def test_csv_matches_check_rows(report_c):
    result, _, out = report_c
    rows = list(csv.DictReader(open(out / "design_factors.csv", encoding="utf-8")))
    assert len(rows) == sum(len(c.rows) for c in result.checks.values())
    frac = next(r for r in rows if r["string"] == "intermediate" and r["case"] == "frac_down_casing"
                and r["check"] == "burst")
    assert float(frac["design_factor"]) == pytest.approx(1.05, abs=0.005) and frac["ok"] == "False"


@pytest.mark.parametrize("fname", sorted(bundled_wells()))
def test_bundled_wells_report(fname, tmp_path):
    result = check_well(load_well(DEFAULT_EXAMPLES_DIR / fname))
    text = write_report(result, tmp_path).read_text(encoding="utf-8")
    assert "## Assumptions and warnings" in text
    for g in result.geometries:
        assert f"## {g.name}" in text


def test_loads_only_report_without_design(tmp_path):
    result = check_well(load_well(DEFAULT_EXAMPLES_DIR / "synthetic_a_shallow_vertical.yaml"))
    assert result.checks == {} and result.ok is None
    text = write_report(result, tmp_path).read_text(encoding="utf-8")
    assert "Design checked: **none (loads only)**" in text
    assert "these are the loads any pipe for this string must carry" in text


def test_design_pipes_notes():
    d = load_well(WELL_C).model_dump()
    d["reference_designs"][0]["strings"]["surface"][0]["wall_in"] = None
    d["reference_designs"][0]["strings"]["liner"].append({"grade": "P110", "weight_lb_per_ft": 13.5, "wall_in": 0.29})
    name, pipes, notes = design_pipes(WellInput.model_validate(d))
    assert set(pipes) == {"intermediate"}
    assert any("no wall_in" in n for n in notes) and any("tapered" in n for n in notes)
    with pytest.raises(ValueError, match="no reference design named"):
        design_pipes(WellInput.model_validate(d), "missing")


def test_drift_check_flags_small_drift():
    d = load_well(WELL_C).model_dump()
    d["reference_designs"][0]["strings"]["surface"][0]["drift_in"] = 8.679  # API drift; 8.75-in bit
    result = check_well(WellInput.model_validate(d))
    assert any("drift 8.679 in is smaller than the next bit (8.75 in)" in n for n in result.notes)


def test_cli_run_and_errors(tmp_path, capsys):
    assert main(["run", str(WELL_C), "-o", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "report.md").exists()
    assert "intermediate" in capsys.readouterr().out
    bad = tmp_path / "bad.yaml"
    bad.write_text("well: {name: x}\n", encoding="utf-8")
    assert main(["run", str(bad)]) == 2
    assert "is not valid" in capsys.readouterr().err
    assert main(["run", str(WELL_C), "--design", "nope", "-o", str(tmp_path / "o2")]) == 2
