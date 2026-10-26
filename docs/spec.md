# Probabilistic Casing Design — LLM Build Instructions

Oct 25, 2026 · @Kanter

## 1. Role and objective

You are a drilling engineer and Python developer. Build a general-purpose program that designs the casing program for **any well** from its input data: pore pressure, fracture gradient, overburden, temperature, trajectory, hole sections, fluids and design rules. For each string it selects grade, weight and connection, and it can also choose setting depths. It reports design factors and probability of failure, and picks the cheapest string that meets the user's reliability target.

The program has two layers:

1. **Deterministic core.** Setting depths, load cases (burst, collapse, axial, triaxial) and pipe ratings per API TR 5C3 / ISO/TR 10400. It must reproduce hand calculations and published worked examples to within 0.5%.
2. **Probabilistic layer.** Any input can be given as a distribution instead of a single value. The program estimates probability of failure for each candidate string and chooses the least-cost string under the target.

Nothing about a particular well, basin or operator is hard-coded. Every well-specific number comes from the input file. The program handles vertical, directional and horizontal wells with any number of strings: conductor, surface, one or more intermediates, production casing or liner, and an optional tieback.

Version 1 scope: onshore wells. Salt creep, subsea annular pressure buildup and casing wear are left out; each can be added later as an optional load module.

Deliverables: a Python package, a command-line tool that takes an input file and writes a report, a synthetic-well generator for testing, and unit tests against published values.

## 2. Inputs and data

Each well is one YAML or JSON input file, checked against a schema (pydantic). Every value carries a unit and an optional source note. The engineering modules contain no well data.

### 2.1 Required inputs

| Group | Fields | Notes |
| --- | --- | --- |
| Well | Name, KB elevation, ground level | Depths are measured from KB |
| Trajectory | Survey stations (MD, inclination, azimuth), or a simple plan: KOP, build rate, landing TVD, lateral length | Converted to a minimum-curvature survey |
| Pore pressure | Table of TVD vs. gradient (ppg) or pressure (psi) | Linear interpolation; any extrapolation is flagged |
| Fracture gradient | Table of TVD vs. ppg or psi, or Poisson's ratio vs. depth to compute it with Eaton (1969) | Eaton needs overburden and pore pressure |
| Overburden | Table of TVD vs. Sv, or a density log (LAS) to integrate | Default: constant gradient, flagged |
| Temperature | Surface temperature and geothermal gradient, or a TVD vs. °F table | For yield derating and thermal loads |
| Hole sections | Per string: hole size, casing OD, setting depth or "choose", top depth (surface or hanger), top of cement, mud weight drilled with | Strings listed outermost first |
| Fluids | Completion or packer fluid, frac fluid, cement slurry densities, mix-water gradient |  |
| Load-case settings | Kick intensity or volume, gas gradient, max surface treating pressure, test pressure, evacuation level, wellhead rating | Each case can be switched on or off per string |
| Design rules | Design factors (burst, collapse, tension, compression, triaxial), trip and kick margins, reliability target | Defaults in Section 3.6; user can override |
| Pipe catalog | CSV: OD, weight, grade, wall, ID, drift, Y\_p, connection and its ratings, $/ft | Ship a starter catalog of API sizes; connection ratings and prices come from the user |
| Regulatory rules | Optional plug-ins, such as a state's minimum surface-casing depth | Off by default |

A minimal input file looks like this:

```yaml
well: {name: Example-1, kb_ft: 25}
trajectory: {kop_ft: 8000, build_deg_per_100ft: 8, landing_tvd_ft: 8700, lateral_ft: 7500}
pore_pressure_ppg:  # TVD ft: gradient
  table: {0: 8.6, 6000: 8.6, 8700: 11.5}
  uncertainty: {dist: normal, shift_sd: 0.3}
frac_gradient: {method: eaton, poisson: {0: 0.30, 8700: 0.27}}
overburden_psi_per_ft: 1.0
temperature: {surface_F: 75, gradient_F_per_100ft: 1.4}
strings:
  - {name: surface, hole_in: 12.25, od_in: 9.625, shoe_tvd_ft: choose, toc: surface}
  - {name: production, hole_in: 8.75, od_in: 5.5, shoe_md_ft: TD, toc_ft: 5000}
load_cases:
  production: {frac_down_casing: {max_surface_psi: 9500, fluid_ppg: 8.4}, evacuation: full}
design_factors: {burst: 1.10, collapse: 1.125, tension: 1.60, triaxial: 1.25}
reliability_target_pf: 1.0e-3
catalog: catalogs/api_starter.csv
```

### 2.2 Uncertain variables

Any input can be a fixed value or a distribution, such as `{dist: normal, mean: 9.7, sd: 0.3, unit: ppg}`. A depth curve is varied by a sampled shift or scale factor applied to the whole curve, so it stays smooth.

| Variable | Suggested distribution | Where parameters usually come from |
| --- | --- | --- |
| Pore pressure gradient | Shift or scale of the input curve | Spread of offset-well mud weights and kicks; Eaton (1975) from sonic or resistivity logs |
| Fracture gradient | Derived: sample Poisson's ratio and overburden, then Eaton | Lab tests or dipole sonic for Poisson's ratio; density logs for overburden |
| Max surface treating pressure | Empirical from offset jobs, or lognormal | Offset frac reports (Enverus, state well records). FracFocus does not report pressures |
| Gas gradient | Uniform | Gas gravity and temperature; 0.08 to 0.12 psi/ft is a labelled placeholder |
| Mud weight actually used | Normal around plan | Offset mud recaps |
| Yield strength (ratio to minimum) | Normal or lognormal, mean above 1.0 | ISO/TR 10400:2018 statistics; Gouveia et al. (2020), Table 1 |
| OD and wall (ratio to nominal) | Normal, wall bounded below by the API 87.5% tolerance | Same as above |
| Ovality, eccentricity, residual stress | Normal or lognormal | ISO/TR 10400:2018 (Klever–Tamano inputs) |
| Model error (test / predicted) | Normal | ISO/TR 10400:2018; Ju, Power and Tallin (1998) |

Treat inputs as independent unless the input file gives a correlation. One rule is fixed: fracture gradient is always computed from the sampled pore pressure, never sampled on its own, because Eaton links them.

### 2.3 Test wells

Real well data is not needed to build or test the program.

1. **Synthetic-well generator** (`synthetic.py`). It writes valid input files from a few parameters: TD, well type, a pore-pressure profile (hydrostatic to a transition depth, then a ramp to a chosen maximum), Poisson's ratio profile, overburden gradient and geothermal gradient. Randomness is seeded so files are reproducible, and every value is labelled synthetic.
2. **Three bundled synthetic wells:** (a) shallow vertical well at normal pressure, surface + production; (b) deep vertical well with an overpressure ramp that needs two intermediates; (c) horizontal unconventional well with a liner and frac-down-casing loads.
3. **Textbook worked example.** Reproduce at least one casing-design example from Bourgoyne et al. (1986, Ch. 7) or Mitchell and Miska (2011) end to end. The user supplies the pages.
4. **Optional real-well example.** Appendix A gives the user's PGE 373L design as one more example input file. It is a test case, never a default.

## 3. Engineering method (deterministic core)

Use field units throughout: psi, ft, in, lbf, ppg, °F. Hydrostatic pressure is P = 0.052 × ρ × TVD. Pressures use TVD; lengths, weights and bending use MD from a minimum-curvature survey.

### 3.1 Setting depths

Setting depths are either given in the input or chosen by the program. To choose them, build the operational mud window (pore + trip margin, fracture − kick margin; 0.5 ppg each by default), apply the top-down method, then check each shoe with the bottom-up kick-tolerance method (Bourgoyne et al. 1986). In the probabilistic run, report how often each shoe would move. Regulatory minimums, such as Texas RRC Statewide Rule 13 (16 TAC §3.13), are optional plug-ins that act as hard constraints.

### 3.2 Load cases

Each load case is an internal pressure profile Pi(z) and an external (backup) profile Pe(z) along the string. Net load = Pi − Pe for burst and Pe − Pi for collapse, evaluated at every node, not only at the top and shoe.

| String | Burst cases | Collapse cases |
| --- | --- | --- |
| Surface | Gas kick from next-section TD with gas to surface; lost returns with water; pressure test | Full or partial evacuation; cementing (wet cement outside, displacement fluid inside) |
| Production casing, or any intermediate the well is fractured through | Injection down casing: max surface treating pressure on a static fluid column, friction not subtracted (frac screenout); tubing leak near the hanger; pressure test | Full evacuation; cementing; depleted reservoir with low annulus level |
| Liner | Injection down casing below the liner top; tubing leak | Full evacuation; depletion at end of life |

Backup for burst follows Devon's guidance in the EPA workshop deck: mud gradient to top of cement, mix-water gradient (8.3 to 8.6 ppg) to the outer shoe, pore pressure below that.

**Common mistake to avoid.** Using the formation breakdown pressure as a uniform internal pressure understates the frac load. In the injection-down-casing case, surface pressure sits on a full fluid column, so internal pressure grows with depth: Pi(z) = P\_surface + 0.052 × ρ\_fluid × TVD(z). The program must compute it that way and report the maximum surface treating pressure each candidate string can take at its design factor.

### 3.3 Pipe ratings (API TR 5C3 / ISO/TR 10400:2018)

Burst (internal yield, Barlow with 87.5% wall):

```latex
P_{iy} = 0.875\,\frac{2\,Y_p\,t}{D}
```

Collapse uses four regimes chosen by D/t. Y\_p is in psi.

```latex
\begin{aligned}
P_{y} &= 2Y_p\,\frac{(D/t)-1}{(D/t)^2} && \text{yield}\\
P_{p} &= Y_p\left[\frac{A}{D/t}-B\right]-C && \text{plastic}\\
P_{t} &= Y_p\left[\frac{F}{D/t}-G\right] && \text{transition}\\
P_{e} &= \frac{46.95\times10^{6}}{(D/t)\,[(D/t)-1]^2} && \text{elastic}
\end{aligned}
```

```latex
\begin{aligned}
A &= 2.8762 + 0.10679\times10^{-5}Y_p + 0.21301\times10^{-10}Y_p^2 - 0.53132\times10^{-16}Y_p^3\\
B &= 0.026233 + 0.50609\times10^{-6}Y_p\\
C &= -465.93 + 0.030867\,Y_p - 0.10483\times10^{-7}Y_p^2 + 0.36989\times10^{-13}Y_p^3\\
F &= \frac{46.95\times10^{6}\left(\frac{3B/A}{2+B/A}\right)^3}{Y_p\left(\frac{3B/A}{2+B/A}-\frac{B}{A}\right)\left(1-\frac{3B/A}{2+B/A}\right)^2},\qquad G = \frac{FB}{A}
\end{aligned}
```

Regime boundaries:

```latex
\left(\tfrac{D}{t}\right)_{yp} = \frac{\sqrt{(A-2)^2+8\left(B+\frac{C}{Y_p}\right)}+(A-2)}{2\left(B+\frac{C}{Y_p}\right)},\quad
\left(\tfrac{D}{t}\right)_{pt} = \frac{Y_p(A-F)}{C+Y_p(B-G)},\quad
\left(\tfrac{D}{t}\right)_{te} = \frac{2+B/A}{3B/A}
```

Tensile pipe-body yield: F\_y = Y\_p × (π/4)(D² − d²). Connection strength comes from the manufacturer data sheet and is checked separately; the string is limited by the weaker of body and connection.

### 3.4 Combined loads

Collapse under tension and internal pressure (API TR 5C3 biaxial): replace Y\_p with the reduced value below, then recompute the regime and collapse rating. Use the equivalent external pressure Pe − (1 − 2/(D/t))Pi.

```latex
Y_{pa} = \left[\sqrt{1-0.75\left(\frac{\sigma_a}{Y_p}\right)^2}-0.5\,\frac{\sigma_a}{Y_p}\right]Y_p
```

Triaxial check: von Mises equivalent stress at the inner wall, with Lamé hoop and radial stresses and axial stress including bending. The triaxial design factor is Y\_p / σ\_VME. Do not use triaxial as a collapse criterion for thin-wall pipe (PetroWiki, Strength of casing and tubing).

```latex
\sigma_{VME} = \sqrt{\tfrac{1}{2}\left[(\sigma_z-\sigma_\theta)^2+(\sigma_\theta-\sigma_r)^2+(\sigma_r-\sigma_z)^2\right]}
```

### 3.5 Axial load

Sum these along MD, from the shoe up:

- Buoyed weight: air weight × (1 − ρ\_mud / 65.5), or pressure-area at each cross-section change.
- Bending in the build: σ\_b = 218 × D × DLS (psi; D in inches, DLS in °/100 ft). For example, 9°/100 ft on 4½-in pipe gives about 8,800 psi. Apply it to every string section that crosses a dogleg, using the survey's local DLS.
- Overpull and shock while running (Bourgoyne et al. 1986).
- Service changes after cementing: temperature ΔF = −207 × A\_s × ΔT (lbf; ΔT in °F, cooling during frac adds tension) and ballooning ΔF = 0.6(ΔPi·Ai − ΔPo·Ao) (Lubinski; Mitchell & Miska 2011). Devon's EPA guidance calls out both for frac loads.
- Temperature derating of yield from the manufacturer table at the static and flowing temperature profiles.

### 3.6 Deterministic check

For each string, node and load case, report the design factor = rating / load and compare it with the design factors in the input file. Defaults when none are given: burst 1.10, collapse 1.125, tension 1.60, compression 1.20, triaxial 1.25, within the ranges in Devon's EPA guidance and common textbooks. The report states which defaults were used.

## 4. Probabilistic and optimization layer

### 4.1 Limit states

Write each failure mode as g = R − L, where R is the sampled pipe resistance and L the sampled load. Failure is g ≤ 0. Compute one g per string, load case and mode (burst, collapse, axial, triaxial), taking the minimum over all depth nodes.

Two resistance models are allowed:

1. **Serviceability (API TR 5C3 equations, Section 3).** These are minimum-performance formulas, so reliability from them is conservative and uneven across D/t. Ju, Power and Tallin (1998) showed this.
2. **Ultimate limit state (ISO/TR 10400:2018).** Klever–Tamano collapse (Klever & Tamano 2006) and the ductile-rupture burst equation, with sampled ovality, eccentricity, residual stress and model error. Use this for the real reliability numbers, and the API model as the baseline to compare against.

### 4.2 Estimating probability of failure

- **Monte Carlo** with a fixed random seed. Start at 10⁵ samples. Report Pf with a 95% confidence interval; increase samples until the coefficient of variation of Pf is below 10%.
- **FORM** (Hasofer–Lind / Rackwitz–Fiessler) through OpenTURNS for small Pf (below 10⁻⁴), where plain Monte Carlo needs too many samples. Check FORM against importance sampling for at least one case.
- Report the reliability index β and the FORM importance factors, so the user sees which input drives each failure mode.

### 4.3 Reliability target

The industry has no single accepted target. Gouveia et al. (2020) treat Pf above 10⁻³ as a concern. Make the target a user setting, default 10⁻³ per load case. Always also produce a sweep from 10⁻² to 10⁻⁵ so the result does not depend on one choice. Report system Pf for each string as well, using the bound max(Pf\_i) ≤ Pf\_sys ≤ Σ Pf\_i.

### 4.4 Cost optimization

1. Build a candidate list for each hole section: every API 5CT grade and weight that drifts the next bit and fits the hole. Include J55, K55, L80, N80, P110 and Q125. Use manufacturer data sheets for dimensions and connection ratings.
2. Allow two-section (tapered) strings with one crossover depth, searched on a 100-ft grid.
3. Cost = Σ length × $/ft. The user supplies $/ft per grade and weight. If none is given, use relative cost = weight × grade multiplier, label it **illustrative**, and never print it as a dollar figure.
4. Choose the cheapest string with Pf ≤ target for every load case. Use exhaustive search when the candidate count is small (it is here), not a heuristic.
5. If the input lists reference designs (a deterministic design, or an offset well's actual string), compare the optimized result with each: grades, weights, cost difference and Pf.

### 4.5 Sensitivity outputs

- Tornado chart of Pf (or β) against each input at P10 and P90.
- Sobol indices (SALib or OpenTURNS) for the governing load case of each string.
- Pf against maximum surface treating pressure for every string exposed to frac loads.

## 5. Software architecture

Build a Python 3.11+ package named `casingrel` with the modules below. The optimizer calls the same load, rating and limit-state code for every candidate, so the deterministic and probabilistic answers can never disagree.

&#91;embedded content: package layout · 8 modules, run in order\]

The deterministic run is the same pipeline with every variable at its base value and Pf replaced by design factors.

| Library | Use | Notes |
| --- | --- | --- |
| numpy, scipy, pandas | Arrays, distributions, tables | Vectorize over samples; no Python loops over Monte Carlo draws |
| [OpenTURNS](https://openturns.github.io/openturns/latest/index.html) | FORM, SORM, importance sampling, Sobol | Cross-check its FORM against your own HL-RF code on one case |
| [SALib](https://salib.readthedocs.io/) | Sobol sensitivity (alternative) | Optional |
| pint | Units | Every public function takes and returns quantities with units |
| pydantic | Input schema | Reject missing units or sources |
| [welleng](https://pypi.org/project/welleng/) | Minimum-curvature survey | Or write minimum curvature yourself and test against it |
| lasio | Read LAS density and sonic logs | For overburden and Eaton pore pressure |
| matplotlib | Plots | Pressure-depth load lines, tornado, Pf vs. treating pressure |
| pytest | Tests | Section 6 cases are required tests |

Required outputs:

1. Load-line plot for each string: burst and collapse loads with depth, overlaid on rating lines (both derated and nominal).
2. Design-factor table per string, case and mode, with the governing depth.
3. Pf and β table per string and case, with confidence intervals.
4. Optimized string versus any reference designs in the input: grades, weights, crossover depths, cost, Pf.
5. Tornado and Sobol charts, and Pf against surface treating pressure for strings with frac loads.
6. A one-page assumptions list: every placeholder value, every extrapolated depth, every warning.

## 6. Verification and accuracy checks

The program is not done until every test below passes. API tables round ratings to the nearest 10 psi, so the tolerance is ±0.5% or ±10 psi, whichever is larger.

### 6.1 Pipe ratings against published values

The table values are published API ratings, as listed in Tenaris data sheets. Extend the test to every pipe in the starter catalog. The formula column is what the Section 3.3 equations give; they were checked before this spec was written.

| Pipe | D/t | Collapse regime | Collapse, formula (psi) | Collapse, table (psi) | Burst, formula (psi) | Burst, table (psi) |
| --- | --- | --- | --- | --- | --- | --- |
| 9⅝-in 40 lb/ft J55 (t = 0.395 in) | 24.37 | Plastic | 2,570 | 2,570 | 3,950 | 3,950 |
| 9⅝-in 40 lb/ft L80 (t = 0.395 in) | 24.37 | Transition | 3,086 | 3,090 | 5,745 | 5,750 |
| 7-in 26 lb/ft L80 (t = 0.362 in) | 19.34 | Plastic | 5,411 | 5,410 | 7,240 | 7,240 |
| 7-in 29 lb/ft L80 (t = 0.408 in) | 17.16 | Plastic | 7,026 | 7,020 | 8,160 | 8,160 |
| 7-in 26 lb/ft P110 (t = 0.362 in) | 19.34 | Plastic | 6,232 | 6,230 | 9,955 | 9,950 |
| 4½-in 11.6 lb/ft P110 (t = 0.250 in) | 18.00 | Plastic | 7,576 | 7,580 | 10,694 | 10,690 |

Also check the collapse coefficients for P110 (Y\_p = 110,000 psi): A = 3.181, B = 0.0819, C = 2,852, F = 2.066, G = 0.0532. These match the published API TR 5C3 coefficient table.

### 6.2 Other required tests

- [ ] Gas-to-surface burst: pore pressure 4,521 psi at 8,963 ft with a 0.10 psi/ft gas gradient gives 3,625 psi at surface.
- [ ] Dry hanging load: 2,600 ft of 40 lb/ft gives 104,000 lbf; 8,963 ft of 26 lb/ft gives 233,038 lbf.
- [ ] Biaxial collapse: at σ\_a = 0, Y\_pa = Y\_p exactly; collapse falls as tension rises.
- [ ] Lamé stresses: with Pi = Po, hoop stress equals −P at every radius.
- [ ] Minimum curvature: build from 0° to 90° at 9°/100 ft gives 1,000 ft MD and 636.6 ft TVD gain.
- [ ] Monte Carlo against FORM: Pf within a factor of 2 for one case with Pf near 10⁻³.
- [ ] Reliability against a closed form: one normal load and one normal resistance gives β = (μR − μL)/√(σR² + σL²) exactly.
- [ ] Same random seed gives the same Pf to the last digit.
- [ ] Unit round-trip: inputs in SI give the same answer as field units. Each bundled synthetic well also runs end to end with no warnings other than labelled placeholders.

### 6.3 Engineering sanity checks (print warnings, do not stop)

- Pf rises when design factor falls. If not, something is wrong with the sign of g.
- The optimizer never picks a string that fails the deterministic check at the user's design factors, unless the user switches that check off.
- When an offset well's actual string is given, the optimized result should be close to it. A large gap means an input is off or the offset operator used a load case this model lacks.

## 7. Rules for the LLM

1. **Build in order.** Ratings module and its tests first (Section 6.1), then loads, then the deterministic report, then the probabilistic layer, then the optimizer. Do not start a stage until the tests of the one before it pass.
2. **Stay well-agnostic and never invent data.** No well name, depth, basin or pressure appears in package code; tests use synthetic wells or example files. Pipe dimensions, connection ratings, derating tables, manufacturing statistics, treating pressures and prices come from the user or a cited source. If missing, stop and ask, or use a clearly labelled placeholder that the report lists.
3. **Cite in code.** Each equation's docstring names its source and equation number (for example, API TR 5C3 §8.3 or Bourgoyne et al. 1986, Eq. 7.x). Look up the number; do not guess it.
4. **Units.** Field units internally, pint at the boundary. State the unit in every variable name or docstring (`p_psi`, `tvd_ft`).
5. **Depth.** Pressures use TVD; weights, bending and lengths use MD. Mixing them is the most common bug in this kind of code.
6. **Signs.** Tension positive, compression negative. Burst load = Pi − Pe; collapse load = Pe − Pi. Write this at the top of `loads.py`.
7. **Ask before changing the user's design basis** (setting depths, design factors, load cases). Add new load cases as options, never as silent changes.
8. **Explain results in plain words.** Each report table gets one sentence that says what it means for the well.
9. **Scope.** This is a student design tool. Add a line to every report: not for field use without review by a licensed engineer and the operator's own standards.

### Starter prompt to paste

```markdown
You are a drilling engineer and Python developer. Read this whole spec first.
Then build the `casingrel` package in the order given in Section 7, Rule 1.
Start with ratings.py and its pytest file using the Section 6.1 table.
Show me the test results, then build synthetic.py and the three test wells.
Ask me for any input marked "user supplies" before you use a placeholder.
```

## 8. References

Paid standards and papers are available through UT Libraries (OnePetro, IHS/Accuris). The LLM should ask the user for excerpts rather than reconstruct equations from memory.

### Standards and regulation

- [ISO/TR 10400:2018](https://www.iso.org/standard/75259.html), formulae for casing and tubing properties (2nd ed., Aug 2018; technically the same as API TR 5C3, 2018). Source for ratings, the ultimate-limit-state equations and manufacturing statistics. It is not a design code and gives no design factors.
- API Spec 5CT / ISO 11960 (latest edition), casing and tubing specifications: grades, dimensions, wall tolerance.
- [16 Tex. Admin. Code §3.13](https://law.cornell.edu/regulations/texas/16-Tex-Admin-Code-SS-3-13), Statewide Rule 13: casing, cementing and completion rules; plus the [RRC Rule 13 page](https://rrc.texas.gov/oil-and-gas/compliance-enforcement/rule-13-geologic-formation-info).

### Reliability-based design

- [Gouveia et al. (2020)](https://link.springer.com/article/10.1007/s13202-020-00858-9), Probabilistic assessment of API casing strength in serviceability limit state, *J. Petrol. Explor. Prod. Technol.* 10: 2089–2104 (open access). Worked Monte Carlo and FORM on API 5C3 equations, with input statistics.
- [Ju, Power and Tallin (1998)](https://www.stress.com/publication/a-reliability-approach-to-the-design-of-octg-tubulars-against-collapse/), A reliability approach to the design of OCTG tubulars against collapse, SPE Workshop on Risk Based Design of Well Casing and Tubing.
- Adams, Parfitt, Reeves and Thorogood (1993), Casing system risk analysis using structural reliability, SPE/IADC Drilling Conference.
- Klever and Tamano (2006), A new OCTG strength equation for collapse under combined loads, *SPE Drilling & Completion*.
- Hasofer and Lind (1974) and Rackwitz and Fiessler (1978), the FORM algorithms.

### Load cases and casing practice

- [PetroWiki: Strength of casing and tubing](https://petrowiki.spe.org/Strength_of_casing_and_tubing): Barlow, collapse regimes, biaxial, triaxial factor = burst factor × 8/7.
- [PetroWiki: Internal pressure loads on casing and tubing strings](https://petrowiki.spe.org/Internal_pressure_loads_on_casing_and_tubing_strings): gas kick, lost returns, tubing leak, injection down casing.
- [Hansen (Devon Energy), Production casing design considerations](https://19january2021snapshot.epa.gov/sites/static/files/documents/productioncasingdesignconsiderations.pdf), EPA hydraulic fracturing workshop. Frac-specific load cases and Devon's minimum design factors (burst 1.25, collapse 1.1, tension 1.4, triaxial 1.25).
- Bourgoyne, Millheim, Chenevert and Young (1986), *Applied Drilling Engineering*, SPE Textbook Series Vol. 2, Ch. 7 (casing design).
- Mitchell and Miska (2011), *Fundamentals of Drilling Engineering*, SPE Textbook Series Vol. 12: tubular mechanics, ballooning, temperature loads.
- Byrom (2015), *Casing and Liners for Drilling and Completion*, 2nd ed., Gulf Professional Publishing.
- Lubinski, Althouse and Logan (1962), Helical buckling of tubing sealed in packers, *JPT*: ballooning and piston forces.

### Pore pressure, fracture gradient and stress

- Eaton (1969), Fracture gradient prediction and its application in oilfield operations, *JPT*.
- Eaton (1975), The equation for geopressure prediction from well logs, SPE-5544-MS.
- [Lund Snee and Zoback (2018)](https://stress.stanford.edu/sites/g/files/sbiybj23561/files/media/file/lund_snee_zoback_2018_state_of_stress_in_the_permian_basin_texas_and_new_mexico_implications_for_induced_seismicity_0.pdf), State of stress in the Permian Basin, *The Leading Edge*: SHmax direction and faulting regime for the Delaware Basin.
- Zoback (2007), *Reservoir Geomechanics*, Cambridge University Press.

### Data sources

- Tenaris and Vallourec pipe-body and connection data sheets: dimensions, ratings, temperature derating. These are the user's WR5 source.
- [Engineers Edge API casing tables](https://ns2.engineersedge.com/fluid_flow/api_casing_engineering_data_table_3.htm): free cross-check of dimensions.
- Enverus and Texas RRC well records: offset casing, mud weights and frac treating pressures (HAY LANDON 9 analogs).
- [FracFocus data download](https://www.fracfocus.org/data-download): fluid volumes and dates by well (CSV). Treating pressures are not included.
- Optional example data from the user's course files: PP&FP Permian Basin workbook (López 2026), PGE 334 HW 2 (HCLonghorn overburden), PGE 334 Labs 1–3 (elastic properties), PGE 430 course notes (Van Oort 2025).

### Software

- [OpenTURNS](https://openturns.github.io/openturns/latest/index.html) · [SALib](https://salib.readthedocs.io/) · [welleng](https://pypi.org/project/welleng/) · numpy, scipy, pandas, pint, pydantic, lasio, matplotlib, pytest.

## Appendix A. Example input: PGE 373L Wolfcamp A well

An optional real-well test case from the user's senior design (Weekly Report 5 and Midterm Report). Save it as `examples/pge373l_wolfcamp.yaml`; it is never a default.

| Item | Value | Source |
| --- | --- | --- |
| Surface casing | 9⅝-in 40 lb/ft L80, 0 to 2,600 ft TVD; 8¾-in bit below it needs 8.750-in special drift | WR5 Table 5 |
| Intermediate casing | 7-in 26 lb/ft P110, 0 to 8,963 ft TVD | WR5 Table 5 |
| Production liner | 4½-in 11.6 lb/ft P110, 8,760 to 15,210 ft MD | WR5 Table 5 |
| Trajectory | Vertical to KOP 8,963 ft, 9°/100 ft build, 636.6 ft radius, land at 9,600 ft TVD, 90°, 5,300 ft lateral | Midterm 5.4 |
| Pore pressure | 8.30 ppg at 2,600 ft; 9.70 ppg at 8,963 and 9,600 ft (extrapolated below 8,726 ft) | PP&FP Permian workbook (López 2026) |
| Fracture gradient | 13.69 ppg at 2,600 ft; 14.73 ppg at 8,963 and 9,600 ft | Same workbook, Eaton (1969) |
| Drilling mud | 9.8 ppg intermediate hole, 11.3 ppg production hole | WR5 3.3 |
| Design factors | Burst 1.10, collapse 1.125, tension 1.60 | WR5 3.1 |
| Reference design | Offset HAY LANDON 9 0023WA: 9⅝-in 40 L80IC, 7-in 29 TN110S, 4½-in 11.6 P110 | RRC 2022 |

Two things this case should reveal: WR5 used the 7,355 psi breakdown pressure as a uniform frac load (Section 3.2), and it left out bending in the build (Section 3.5).
