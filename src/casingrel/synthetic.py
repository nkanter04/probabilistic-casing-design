"""Synthetic-well generator (spec Section 2.3).

Writes valid input files from a few parameters: TD, well type, a pore-pressure
profile (hydrostatic to a transition depth, then a ramp to a chosen maximum),
a Poisson's ratio profile, an overburden gradient and a geothermal gradient.
Random draws use a seeded generator, so the same parameters always give the
same file. Every file is marked synthetic in `meta`, `sources` and its header.

Run `python -m casingrel.synthetic [out_dir]` to rewrite the bundled wells.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from .schema import WellInput

GENERATOR = "casingrel.synthetic"
SYNTHETIC_NOTE = "SYNTHETIC: generated test data, not from any real well."
DEFAULT_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@dataclass(frozen=True)
class PoreProfile:
    """Hydrostatic to transition_tvd_ft, then a linear ramp to max_ppg at TD.

    With noise_sd_ppg > 0, the interior ramp points get seeded normal noise.
    """

    transition_tvd_ft: float
    max_ppg: float
    normal_ppg: float = 8.6
    ramp_points: int = 6
    noise_sd_ppg: float = 0.0


@dataclass(frozen=True)
class SyntheticWellParams:
    name: str
    td_tvd_ft: float
    pore: PoreProfile
    strings: list[dict[str, Any]]
    well_type: Literal["vertical", "horizontal"] = "vertical"
    build_deg_per_100ft: float | None = None
    lateral_ft: float = 0.0
    kb_ft: float = 25.0
    poisson_surface: float = 0.25
    poisson_td: float = 0.30
    overburden_psi_per_ft: float = 1.0
    surface_F: float = 75.0
    gradient_F_per_100ft: float = 1.4
    pore_shift_sd_ppg: float = 0.3
    fluids: dict[str, Any] = field(default_factory=dict)
    load_cases: dict[str, Any] = field(default_factory=dict)
    seed: int = 0


def pore_table(td_tvd_ft: float, pore: PoreProfile, rng: np.random.Generator) -> dict[float, float]:
    """Pore-pressure table {tvd_ft: ppg} for a hydrostatic-then-ramp profile."""
    if not 0 < pore.transition_tvd_ft <= td_tvd_ft:
        raise ValueError("transition_tvd_ft must lie in (0, TD]")
    table = {0.0: pore.normal_ppg, float(pore.transition_tvd_ft): pore.normal_ppg}
    if pore.transition_tvd_ft < td_tvd_ft:
        tvd = np.linspace(pore.transition_tvd_ft, td_tvd_ft, pore.ramp_points + 1)[1:]
        ppg = np.interp(tvd, [pore.transition_tvd_ft, td_tvd_ft], [pore.normal_ppg, pore.max_ppg])
        if pore.noise_sd_ppg > 0:
            ppg[:-1] += rng.normal(0.0, pore.noise_sd_ppg, size=len(ppg) - 1)
        table.update({round(float(z), 1): round(float(p), 3) for z, p in zip(tvd, ppg)})
    return table


def make_input(p: SyntheticWellParams) -> dict[str, Any]:
    """Build the input-file mapping for one synthetic well."""
    rng = np.random.default_rng(p.seed)
    if p.well_type == "vertical":
        trajectory: dict[str, Any] = {"td_ft": p.td_tvd_ft}
    else:
        if p.build_deg_per_100ft is None:
            raise ValueError("a horizontal well needs build_deg_per_100ft")
        trajectory = {"landing_tvd_ft": p.td_tvd_ft, "build_deg_per_100ft": p.build_deg_per_100ft,
                      "lateral_ft": p.lateral_ft}
    data: dict[str, Any] = {
        "well": {"name": p.name, "kb_ft": p.kb_ft},
        "trajectory": trajectory,
        "pore_pressure_ppg": {"table": pore_table(p.td_tvd_ft, p.pore, rng),
                              "uncertainty": {"dist": "normal", "shift_sd": p.pore_shift_sd_ppg}},
        "frac_gradient": {"method": "eaton",
                          "poisson": {0.0: p.poisson_surface, float(p.td_tvd_ft): p.poisson_td}},
        "overburden_psi_per_ft": p.overburden_psi_per_ft,
        "temperature": {"surface_F": p.surface_F, "gradient_F_per_100ft": p.gradient_F_per_100ft},
        "strings": [dict(s) for s in p.strings],
    }
    if p.fluids:
        data["fluids"] = dict(p.fluids)
    if p.load_cases:
        data["load_cases"] = {k: dict(v) for k, v in p.load_cases.items()}
    data["sources"] = {"all": SYNTHETIC_NOTE}
    # Record the generator parameters that are not already visible in the file.
    params = {k: v for k, v in dataclasses.asdict(p).items()
              if k not in ("name", "strings", "fluids", "load_cases", "seed")}
    data["meta"] = {"synthetic": True, "generator": GENERATOR, "seed": p.seed,
                    "params": _plain(params), "note": SYNTHETIC_NOTE}
    WellInput.model_validate(data)  # fail here, not in a later stage
    return data


def write_input(data: dict[str, Any], path: str | Path) -> Path:
    """Write an input mapping as YAML with a synthetic-data header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# {SYNTHETIC_NOTE}\n# Written by {GENERATOR}; regenerate with "
              f"`python -m casingrel.synthetic`. Do not edit by hand.\n")
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=100)
    path.write_text(header + body, encoding="utf-8", newline="\n")
    return path


def random_params(seed: int, well_type: Literal["vertical", "horizontal"] = "vertical") -> SyntheticWellParams:
    """Draw a plausible synthetic well from `seed` (for fuzz and smoke tests).

    The ranges are generic placeholders chosen to give a valid, drillable
    geometry. They do not describe any basin.
    """
    rng = np.random.default_rng(seed)
    td_tvd_ft = float(round(rng.uniform(6_000, 14_000), -2))
    transition_tvd_ft = float(round(rng.uniform(0.5, 0.9) * td_tvd_ft, -2))
    max_ppg = float(round(rng.uniform(9.0, 14.0), 1))
    surface_shoe_ft = float(round(rng.uniform(0.10, 0.20) * td_tvd_ft, -2))
    strings = [
        {"name": "surface", "hole_in": 12.25, "od_in": 9.625, "shoe_tvd_ft": surface_shoe_ft, "toc": "surface"},
        {"name": "production", "hole_in": 8.75, "od_in": 5.5, "shoe_md_ft": "TD",
         "toc_ft": float(round(0.5 * surface_shoe_ft + 0.5 * transition_tvd_ft, -2))},
    ]
    return SyntheticWellParams(
        name=f"SYNTH-RANDOM-{seed}", td_tvd_ft=td_tvd_ft,
        pore=PoreProfile(transition_tvd_ft=transition_tvd_ft, max_ppg=max_ppg, noise_sd_ppg=0.1),
        strings=strings, well_type=well_type,
        build_deg_per_100ft=8.0 if well_type == "horizontal" else None,
        lateral_ft=float(round(rng.uniform(3_000, 10_000), -2)) if well_type == "horizontal" else 0.0,
        poisson_surface=float(round(rng.uniform(0.22, 0.28), 3)),
        poisson_td=float(round(rng.uniform(0.28, 0.35), 3)),
        gradient_F_per_100ft=float(round(rng.uniform(1.0, 2.0), 2)), seed=seed)


# --- Bundled test wells (spec Section 2.3, item 2) ---------------------------
# All values are synthetic placeholders picked to exercise the program.

_FLUIDS = {"completion_ppg": 8.6, "frac_fluid_ppg": 8.4, "cement_lead_ppg": 12.5,
           "cement_tail_ppg": 15.8, "mix_water_ppg": 8.4}


def bundled_wells() -> dict[str, SyntheticWellParams]:
    """The three bundled synthetic wells, keyed by file name."""
    shallow = SyntheticWellParams(
        name="SYNTH-A-shallow-vertical", td_tvd_ft=6_000,
        pore=PoreProfile(transition_tvd_ft=6_000, max_ppg=8.6),
        poisson_surface=0.25, poisson_td=0.30,
        strings=[
            {"name": "surface", "hole_in": 12.25, "od_in": 8.625, "shoe_tvd_ft": 1_200, "toc": "surface",
             "mud_ppg": 9.1},
            {"name": "production", "hole_in": 7.875, "od_in": 5.5, "shoe_md_ft": "TD", "toc_ft": 900,
             "mud_ppg": 9.2},
        ],
        fluids=_FLUIDS,
        load_cases={
            "surface": {"gas_kick": {"gas_gradient_psi_per_ft": 0.1}, "pressure_test": {"surface_psi": 1_500},
                        "evacuation": "full", "cementing": True},
            "production": {"pressure_test": {"surface_psi": 3_000},
                           "tubing_leak": {"surface_psi": 2_000, "packer_fluid_ppg": 8.6},
                           "evacuation": "full", "cementing": True},
        },
        seed=1)

    deep = SyntheticWellParams(
        name="SYNTH-B-deep-overpressured", td_tvd_ft=16_000,
        pore=PoreProfile(transition_tvd_ft=10_000, max_ppg=16.0, noise_sd_ppg=0.1),
        poisson_surface=0.25, poisson_td=0.40, gradient_F_per_100ft=1.6,
        strings=[
            {"name": "conductor", "hole_in": 26.0, "od_in": 20.0, "shoe_tvd_ft": 150, "toc": "surface"},
            {"name": "surface", "hole_in": 17.5, "od_in": 13.375, "shoe_tvd_ft": 3_000, "toc": "surface",
             "mud_ppg": 9.1},
            {"name": "intermediate_1", "hole_in": 12.25, "od_in": 9.625, "shoe_tvd_ft": "choose",
             "toc_ft": 2_500},
            {"name": "intermediate_2", "hole_in": 8.5, "od_in": 7.0, "shoe_tvd_ft": "choose",
             "toc_ft": 8_000},
            {"name": "production", "hole_in": 6.125, "od_in": 4.5, "shoe_md_ft": "TD", "toc_ft": 11_000},
        ],
        fluids=_FLUIDS,
        load_cases={
            "surface": {"gas_kick": {"gas_gradient_psi_per_ft": 0.1}, "pressure_test": {"surface_psi": 2_500},
                        "evacuation": {"level_ft": 2_000}, "cementing": True},
            "intermediate_1": {"gas_kick": {"gas_gradient_psi_per_ft": 0.1},
                               "pressure_test": {"surface_psi": 5_000}, "evacuation": {"level_ft": 5_000},
                               "cementing": True},
            "intermediate_2": {"gas_kick": {"gas_gradient_psi_per_ft": 0.1},
                               "pressure_test": {"surface_psi": 8_000}, "evacuation": {"level_ft": 7_000},
                               "cementing": True},
            "production": {"pressure_test": {"surface_psi": 10_000},
                           "tubing_leak": {"surface_psi": 9_000, "packer_fluid_ppg": 10.0},
                           "evacuation": "full", "cementing": True},
        },
        seed=2)

    horizontal = SyntheticWellParams(
        name="SYNTH-C-horizontal-frac", td_tvd_ft=8_700, well_type="horizontal",
        build_deg_per_100ft=8.0, lateral_ft=7_500,
        pore=PoreProfile(transition_tvd_ft=6_000, max_ppg=11.5, noise_sd_ppg=0.05),
        poisson_surface=0.25, poisson_td=0.30,
        strings=[
            {"name": "surface", "hole_in": 12.25, "od_in": 9.625, "shoe_tvd_ft": 3_000, "toc": "surface",
             "mud_ppg": 9.1},
            {"name": "intermediate", "hole_in": 8.75, "od_in": 7.0, "shoe_md_ft": 7_800, "toc_ft": 2_500,
             "mud_ppg": 11.0},
            {"name": "liner", "hole_in": 6.125, "od_in": 4.5, "top_md_ft": 7_600, "shoe_md_ft": "TD",
             "toc_ft": 7_600, "mud_ppg": 12.0},
        ],
        fluids=_FLUIDS,
        load_cases={
            "surface": {"gas_kick": {"gas_gradient_psi_per_ft": 0.1}, "pressure_test": {"surface_psi": 1_500},
                        "evacuation": "full", "cementing": True},
            "intermediate": {"frac_down_casing": {"max_surface_psi": 9_500, "fluid_ppg": 8.4},
                             "pressure_test": {"surface_psi": 10_000}, "evacuation": "full",
                             "cementing": True},
            "liner": {"frac_down_casing": {"max_surface_psi": 9_500, "fluid_ppg": 8.4},
                      "evacuation": "full", "depletion": {"pore_ppg": 6.0}, "cementing": True},
        },
        seed=3)

    return {"synthetic_a_shallow_vertical.yaml": shallow,
            "synthetic_b_deep_overpressured.yaml": deep,
            "synthetic_c_horizontal_frac.yaml": horizontal}


def write_bundled(out_dir: str | Path = DEFAULT_EXAMPLES_DIR) -> list[Path]:
    return [write_input(make_input(p), Path(out_dir) / fname) for fname, p in bundled_wells().items()]


def _plain(x: Any) -> Any:
    """Make a structure YAML-safe (tuples to lists, numpy scalars to Python)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write the bundled synthetic test wells.")
    ap.add_argument("out_dir", nargs="?", default=str(DEFAULT_EXAMPLES_DIR))
    for path in write_bundled(ap.parse_args(argv).out_dir):
        print(path)


if __name__ == "__main__":
    main()
