# Design decisions

This log records each place where the build departs from [spec.md](spec.md), with the reason.

| Date | Spec section | Decision | Reason |
| --- | --- | --- | --- |
| 2026-10-26 | §5 (pint) | Rating functions take plain floats or numpy arrays in field units, with the unit in the parameter name. pint converts only at the input/report boundary. | The Monte Carlo layer passes vectors of 10⁵+ samples; pint wrapping at every call is slow and adds nothing inside the package. CLAUDE.md rule 4 already says pint at the boundary. |
| 2026-10-26 | §3.3, §6.1 | Ratings are returned unrounded; tests compare with published tables at ±max(0.5%, 10 psi). | The API's rounding convention is not documented in the spec, and some published values (7-in 29 lb/ft L80 collapse 7,020 psi vs. 7,025.6 computed; 7-in 26 lb/ft P110 burst 9,950 vs. 9,955.0) do not follow simple round-half-up. Rounding is a display concern only. |
| 2026-10-26 | §5 | Module layout agreed with the user (listed in CLAUDE.md). It replaces the spec's lost 8-module diagram. | The diagram did not survive export. |
| 2026-10-26 | §2.1 | Units go in the key (`kb_ft`); a value may be a unit string instead. Source notes go in a top-level `sources:` map, not on each value. | The files stay as short as the spec's example, every value still has a unit, and SI input works (needed for the §6.2 round-trip test). |
| 2026-10-26 | §2.1 example | The spec's minimal example is rejected as written: KOP 8,000 ft + 8°/100 ft build lands at 8,716 ft TVD, not 8,700. A plan takes `kop_ft` or `landing_tvd_ft`; both are allowed only if they agree within 1 ft. | An over-specified, inconsistent trajectory would silently put pressures at the wrong TVD. The tests use the example without `kop_ft`. |
| 2026-10-26 | §2.2 | A frac-gradient table cannot carry its own uncertainty block. | The spec fixes frac gradient to follow sampled pore pressure. Eaton inputs (Poisson's ratio, overburden) can be uncertain instead. |
| 2026-10-26 | §2.3 | Synthetic well (b)'s parameters were tuned so a top-down check (Eaton, 0.5 ppg margins) needs exactly two intermediates (~10,290 and ~14,070 ft TVD). Their shoes are `choose`. | The spec requires (b) to need two intermediates. design.py tests should confirm this once built. |
| 2026-10-26 | §3.4 | Biaxial collapse uses the spec's formula with σ_a alone. | Open question: the standard may use (σ_a + p_i) in the axial term. Waiting on the ISO/TR 10400 excerpt before changing it. |
