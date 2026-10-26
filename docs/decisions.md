# Design decisions

This log records each place where the build departs from [spec.md](spec.md), with the reason.

| Date | Spec section | Decision | Reason |
| --- | --- | --- | --- |
| 2026-10-26 | §5 (pint) | Rating functions take plain floats or numpy arrays in field units, with the unit in the parameter name. pint converts only at the input/report boundary. | The Monte Carlo layer passes vectors of 10⁵+ samples; pint wrapping at every call is slow and adds nothing inside the package. CLAUDE.md rule 4 already says pint at the boundary. |
| 2026-10-26 | §3.3, §6.1 | Ratings are returned unrounded; tests compare with published tables at ±max(0.5%, 10 psi). | The API's rounding convention is not documented in the spec, and some published values (7-in 29 lb/ft L80 collapse 7,020 psi vs. 7,025.6 computed; 7-in 26 lb/ft P110 burst 9,950 vs. 9,955.0) do not follow simple round-half-up. Rounding is a display concern only. |
| 2026-10-26 | §3.4 | Biaxial collapse uses the spec's formula with σ_a alone. | Open question: the standard may use (σ_a + p_i) in the axial term. Waiting on the ISO/TR 10400 excerpt before changing it. |
