# Probabilistic Casing Design

`casingrel` is a Python tool that designs the casing program for an oil or gas well from its input data: pore pressure, fracture gradient, overburden, temperature, trajectory, hole sections, fluids and design rules.

- **Deterministic core:** setting depths, burst, collapse, axial and triaxial load cases, and pipe ratings per API TR 5C3 / ISO/TR 10400.
- **Probabilistic layer:** any input can be a distribution. The tool estimates the probability of failure of each candidate string (Monte Carlo and FORM) and picks the cheapest string that meets a reliability target.

See [docs/spec.md](docs/spec.md) for the full specification.

> **Status:** early development. Done: pipe-body ratings, input schema with unit handling, the synthetic-well generator with three bundled test wells in [examples/](examples/), the minimum-curvature trajectory, the depth profiles (pore pressure, Eaton fracture gradient, overburden, temperature), the burst, collapse and axial load cases, setting depths with kick tolerance, the deterministic design-factor check, and the deterministic report.

## Usage

```bash
casingrel run examples/synthetic_c_horizontal_frac.yaml -o reports/well_c
```

This writes `report.md` (summary, setting depths and kick tolerance, design-factor tables, assumptions list), `plots/` and `design_factors.csv`. `casingrel synthetic` regenerates the bundled synthetic wells.

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # on macOS/Linux: .venv/bin/python
.venv/Scripts/python -m pytest
```

> **Disclaimer:** this is a student design tool. It is not for field use without review by a licensed engineer and the operator's own standards.
