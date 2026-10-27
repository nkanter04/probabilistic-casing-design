# Probabilistic Casing Design

`casingrel` is a Python tool that designs the casing program for an oil or gas well from its input data: pore pressure, fracture gradient, overburden, temperature, trajectory, hole sections, fluids and design rules.

- **Deterministic core:** setting depths, burst, collapse, axial and triaxial load cases, and pipe ratings per API TR 5C3 / ISO/TR 10400.
- **Probabilistic layer:** any input can be a distribution. The tool estimates the probability of failure of each candidate string (Monte Carlo and FORM) and picks the cheapest string that meets a reliability target.

See [docs/spec.md](docs/spec.md) for the full specification.

> **Status:** early development. Done: pipe-body ratings, input schema with unit handling, the synthetic-well generator with three bundled test wells in [examples/](examples/), the minimum-curvature trajectory, the depth profiles (pore pressure, Eaton fracture gradient, overburden, temperature), the burst, collapse and axial load cases, setting depths with kick tolerance, the deterministic design-factor check, the deterministic report, probability of failure by Monte Carlo and FORM, and least-cost string selection from a pipe catalog (single and tapered strings).

## Usage

```bash
casingrel run examples/synthetic_c_horizontal_frac.yaml -o reports/well_c
```

This writes `report.md` (summary, setting depths and kick tolerance, design-factor tables, assumptions list), `plots/` and `design_factors.csv`. Add `--probabilistic` for probability of failure.

```bash
casingrel optimize examples/synthetic_c_horizontal_frac.yaml -o reports/well_c_opt
```

This chooses the cheapest design per string from the pipe catalog (`catalogs/api_starter.csv` unless the input names one) that passes the design factors and the reliability target. It also compares the choice with any reference designs. `casingrel synthetic` regenerates the bundled synthetic wells.

The starter catalog holds only the six pipes from the spec's Section 6.1 table. Real designs need manufacturer data sheets (dimensions, drift, connection ratings, and prices if available).

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # on macOS/Linux: .venv/bin/python
.venv/Scripts/python -m pytest
```

> **Disclaimer:** this is a student design tool. It is not for field use without review by a licensed engineer and the operator's own standards.
