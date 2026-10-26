# CLAUDE.md

Project instructions for Claude. This file lives in the repo, so the same instructions apply on any machine that clones it.

## Project

`casingrel`: a Python package that designs a well's casing program deterministically (API TR 5C3 / ISO/TR 10400) and probabilistically (Monte Carlo + FORM). It then picks the cheapest string that meets a reliability target. It is a student design tool, built for any well.

The full specification is [docs/spec.md](docs/spec.md). Read it before starting work.

## How to treat the spec

- An LLM generated the spec. Treat it as a strong starting point, not as gospel. A deviation is fine when it makes the result more accurate, more correct or easier to maintain, but record each one in [docs/decisions.md](docs/decisions.md) with the reason.
- Section 5's package-layout diagram was lost when the spec was exported. The user agreed this layout instead (2026-10-26):

  ```
  src/casingrel/
    ratings.py      API burst/collapse/tension
    schema.py       pydantic input model, YAML/JSON load + validation
    units.py        pint conversions at the input/report boundary
    trajectory.py   minimum-curvature survey, simple-plan builder
    profiles.py     pore, frac (Eaton), overburden, temperature vs TVD
    loads.py        load cases -> Pi(z), Pe(z); axial loads
    design.py       setting depths, deterministic design factors
    reliability.py  distributions, Monte Carlo, FORM
    optimize.py     candidate strings, tapered search, cost
    report.py       tables, plots, assumptions list
    synthetic.py    synthetic-well generator
    cli.py          casingrel run well.yaml -> report
  examples/         bundled synthetic wells (generated; do not hand-edit)
  catalogs/         pipe catalog CSV(s)
  ```

- Input files put the unit in the key (`kb_ft: 25`). Any value may instead be a string with its own unit ("7.6 m"), which pint converts on load.
- Never change the user's design basis (setting depths, design factors, load cases) without asking first (spec §7 rule 7).

## Rules that matter most (from spec §7)

1. Build in order: ratings → loads → deterministic report → probabilistic layer → optimizer. A stage starts only after the previous stage's tests pass.
2. No well-specific data in package code. Data that is missing or uncertain gets a clearly labelled placeholder or a question to the user, never an invented value.
3. Every equation's docstring cites its source and equation number. Ask the user for excerpts of paid standards instead of reconstructing them from memory.
4. Field units internally (psi, ft, in, lbf, ppg, °F), with pint at the boundaries. Put the unit in the variable name (`p_psi`, `tvd_ft`).
5. Pressures use TVD. Weights, bending and lengths use MD.
6. Signs: tension positive, compression negative. Burst load = Pi − Pe; collapse load = Pe − Pi.
7. Every report carries the line: not for field use without review by a licensed engineer and the operator's own standards.

## Environment

- Python 3.11+ (developed on 3.13). Tests: `pytest`.
- Primary machine: a Windows PC. A second machine (laptop) may also be used, so keep all project context in this repo, not in local files.
