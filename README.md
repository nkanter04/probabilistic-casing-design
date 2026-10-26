# Probabilistic Casing Design

`casingrel` is a Python tool that designs the casing program for an oil or gas well from its input data: pore pressure, fracture gradient, overburden, temperature, trajectory, hole sections, fluids and design rules.

- **Deterministic core:** setting depths, burst, collapse, axial and triaxial load cases, and pipe ratings per API TR 5C3 / ISO/TR 10400.
- **Probabilistic layer:** any input can be a distribution. The tool estimates the probability of failure of each candidate string (Monte Carlo and FORM) and picks the cheapest string that meets a reliability target.

See [docs/spec.md](docs/spec.md) for the full specification.

> **Status:** early development. Nothing has been built yet.

> **Disclaimer:** this is a student design tool. It is not for field use without review by a licensed engineer and the operator's own standards.
