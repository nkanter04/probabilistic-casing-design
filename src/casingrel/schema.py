"""Input schema for one well (spec Section 2).

Each well is one YAML or JSON file. Units follow the key: `kb_ft: 25` is in
feet. Any value may instead be a string with its own unit ("7.6 m"), which is
converted to the field unit on load. After validation every number in the
model is in field units, so the engineering modules never see a unit string.

Any scalar marked "uncertain" below may be a distribution instead of a number:
`{dist: normal, mean: 9.7, sd: 0.3}` (in the key's unit) or with its own
`unit: g/cm**3`. Depth curves take an `uncertainty:` block that shifts or
scales the whole curve, so it stays smooth (spec Section 2.2).

Unknown keys are rejected, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import (BaseModel, BeforeValidator, ConfigDict, Discriminator, Field,
                      Tag, field_validator, model_validator)

from .units import PSI_PER_FT_PER_PPG, convert_spread, to_field

# Spec Section 3.6 defaults, used when the input file gives none.
DEFAULT_DESIGN_FACTORS = {"burst": 1.10, "collapse": 1.125, "tension": 1.60,
                          "compression": 1.20, "triaxial": 1.25}
DEFAULT_MARGINS_PPG = {"trip_margin_ppg": 0.5, "kick_margin_ppg": 0.5}
DEFAULT_RELIABILITY_TARGET_PF = 1.0e-3
# Spec Section 2.1: overburden defaults to a constant gradient, flagged as a placeholder.
DEFAULT_OVERBURDEN_PSI_PER_FT = 1.0

# Radius of curvature, ft, for a build rate in deg/100 ft: R = 18000 / (pi * BUR).
_DEG_PER_100FT_TO_RADIUS_FT = 18000.0 / math.pi


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Quantities -------------------------------------------------------------

def _q(unit: str, scale: float = 1.0) -> BeforeValidator:
    return BeforeValidator(lambda v: to_field(v, unit, scale))


Ft = Annotated[float, _q("ft")]
In = Annotated[float, _q("inch")]
Deg = Annotated[float, _q("deg")]
DegPer100Ft = Annotated[float, _q("deg/ft", 100.0)]
Ratio = Annotated[float, _q("dimensionless")]


class Distribution(_Model):
    """A probability distribution for one uncertain input.

    Parameters are in the field unit of the key they sit under, unless `unit`
    is given. On load they are converted to the field unit and `unit` is
    cleared. lognormal takes the mean and sd of the variable itself, not of
    its logarithm.
    """

    dist: Literal["normal", "lognormal", "uniform", "triangular"]
    mean: float | None = None
    sd: float | None = None
    low: float | None = None
    mode: float | None = None
    high: float | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def _check_parameters(self) -> Distribution:
        needed = {"normal": ("mean", "sd"), "lognormal": ("mean", "sd"),
                  "uniform": ("low", "high"), "triangular": ("low", "mode", "high")}[self.dist]
        given = {k for k in ("mean", "sd", "low", "mode", "high") if getattr(self, k) is not None}
        if given != set(needed):
            raise ValueError(f"{self.dist} distribution takes exactly {', '.join(needed)}")
        if self.sd is not None and self.sd <= 0:
            raise ValueError("sd must be positive")
        if self.dist == "lognormal" and self.mean <= 0:
            raise ValueError("lognormal mean must be positive")
        if self.low is not None and not self.low < self.high:
            raise ValueError("low must be less than high")
        if self.mode is not None and not self.low <= self.mode <= self.high:
            raise ValueError("mode must lie between low and high")
        return self

    @property
    def base(self) -> float:
        """Value used in the deterministic run: mean (normal, lognormal),
        midpoint (uniform) or mode (triangular)."""
        if self.dist in ("normal", "lognormal"):
            return self.mean
        if self.dist == "uniform":
            return 0.5 * (self.low + self.high)
        return self.mode

    def in_field_unit(self, unit: str, scale: float = 1.0) -> Distribution:
        if self.unit is None:
            return self

        def conv(x: float | None) -> float | None:
            return None if x is None else to_field(f"{x} {self.unit}", unit, scale)

        sd = None if self.sd is None else convert_spread(self.mean, self.sd, self.unit, unit, scale)
        return Distribution(dist=self.dist, mean=conv(self.mean), sd=sd, low=conv(self.low),
                            mode=conv(self.mode), high=conv(self.high))


def _uq(unit: str, scale: float = 1.0) -> BeforeValidator:
    def conv(v: Any) -> Any:
        if isinstance(v, Distribution):
            return v.in_field_unit(unit, scale)
        if isinstance(v, dict) and "dist" in v:
            return Distribution.model_validate(v).in_field_unit(unit, scale)
        return to_field(v, unit, scale)
    return BeforeValidator(conv)


# Uncertain scalars: a number, a quantity string, or a Distribution.
UFt = Annotated[Union[float, Distribution], _uq("ft")]
UPsi = Annotated[Union[float, Distribution], _uq("psi")]
UPpg = Annotated[Union[float, Distribution], _uq("ppg")]
UPsiPerFt = Annotated[Union[float, Distribution], _uq("psi/ft")]
UDegF = Annotated[Union[float, Distribution], _uq("degF")]
UDegFPer100Ft = Annotated[Union[float, Distribution], _uq("delta_degF/ft", 100.0)]
UBbl = Annotated[Union[float, Distribution], _uq("oil_barrel")]


# --- Depth curves -----------------------------------------------------------

def _table(unit: str, scale: float = 1.0, allow_pressure: bool = False) -> BeforeValidator:
    """Convert {TVD: value} to {tvd_ft: value in field unit}, sorted by depth.

    With allow_pressure (ppg curves only), a value given as a pressure such as
    "4521 psi" is converted to an equivalent gradient at that TVD.
    """
    def conv(raw: Any) -> dict[float, float]:
        if not isinstance(raw, dict) or not raw:
            raise ValueError("expected a non-empty mapping of TVD -> value")
        out: dict[float, float] = {}
        for k, v in raw.items():
            tvd_ft = to_field(k, "ft")
            if tvd_ft < 0:
                raise ValueError(f"negative TVD {k!r}")
            if tvd_ft in out:
                raise ValueError(f"TVD {k!r} appears twice")
            out[tvd_ft] = _table_value(v, tvd_ft, unit, scale, allow_pressure)
        return dict(sorted(out.items()))
    return BeforeValidator(conv)


def _table_value(v: Any, tvd_ft: float, unit: str, scale: float, allow_pressure: bool) -> float:
    if allow_pressure and isinstance(v, str):
        try:
            p_psi = to_field(v, "psi")
        except ValueError:
            pass
        else:
            if tvd_ft <= 0:
                raise ValueError(f"pressure {v!r} at TVD 0 has no equivalent gradient")
            return p_psi / (PSI_PER_FT_PER_PPG * tvd_ft)
    return to_field(v, unit, scale)


class CurveUncertainty(_Model):
    """Shift (in the curve's unit) or scale (fraction) applied to the whole curve."""

    dist: Literal["normal"] = "normal"
    shift_sd: float | None = None
    scale_sd: float | None = None

    @model_validator(mode="after")
    def _one_of(self) -> CurveUncertainty:
        if (self.shift_sd is None) == (self.scale_sd is None):
            raise ValueError("give exactly one of shift_sd or scale_sd")
        if (self.shift_sd or self.scale_sd) <= 0:
            raise ValueError("shift_sd / scale_sd must be positive")
        return self


class _Curve(_Model):
    uncertainty: CurveUncertainty | None = None

    @model_validator(mode="before")
    @classmethod
    def _wrap_plain_table(cls, v: Any) -> Any:
        # Accept a bare {TVD: value} mapping as shorthand for {table: {...}}.
        if isinstance(v, dict) and "table" not in v:
            return {"table": v}
        return v


class PpgCurve(_Curve):
    """Gradient vs TVD in ppg; entries may also be pressures ("4521 psi")."""

    table: Annotated[dict[float, float], _table("ppg", allow_pressure=True)]


class RatioCurve(_Curve):
    table: Annotated[dict[float, float], _table("dimensionless")]


class PsiPerFtCurve(_Curve):
    """Overburden as average gradient Sv/TVD (psi/ft) vs TVD, so Sv = gradient * TVD."""

    table: Annotated[dict[float, float], _table("psi/ft")]


class DegFCurve(_Curve):
    table: Annotated[dict[float, float], _table("degF")]


class FracEaton(_Model):
    """Fracture gradient from Eaton (1969); needs overburden and pore pressure."""

    method: Literal["eaton"]
    poisson: RatioCurve


class FracTable(_Model):
    """Fracture gradient given directly. It takes no uncertainty block: spec
    Section 2.2 requires frac gradient to follow sampled pore pressure."""

    method: Literal["table"]
    table: Annotated[dict[float, float], _table("ppg", allow_pressure=True)]


class TemperatureGradient(_Model):
    surface_F: UDegF
    gradient_F_per_100ft: UDegFPer100Ft


def _has_key(key: str, yes: str, no: str):
    def pick(v: Any) -> str:
        if isinstance(v, dict):
            return yes if key in v else no
        return yes if hasattr(v, key) else no
    return pick


Temperature = Annotated[
    Union[Annotated[DegFCurve, Tag("table")], Annotated[TemperatureGradient, Tag("gradient")]],
    Discriminator(_has_key("surface_F", "gradient", "table"))]

Overburden = Annotated[
    Union[Annotated[PsiPerFtCurve, Tag("table")], Annotated[UPsiPerFt, Tag("constant")]],
    Discriminator(lambda v: "table" if isinstance(v, (dict, PsiPerFtCurve)) and not
                  (isinstance(v, dict) and "dist" in v) else "constant")]


# --- Trajectory -------------------------------------------------------------

class Station(_Model):
    md_ft: Ft
    inc_deg: Deg
    azi_deg: Deg


class SurveyTrajectory(_Model):
    """Survey stations; converted to a minimum-curvature survey downstream."""

    stations: list[Station] = Field(min_length=2)

    @field_validator("stations")
    @classmethod
    def _check(cls, stations: list[Station]) -> list[Station]:
        if stations[0].md_ft != 0:
            raise ValueError("first station must be at MD 0 (KB)")
        for a, b in zip(stations, stations[1:]):
            if b.md_ft <= a.md_ft:
                raise ValueError(f"station MDs must increase ({a.md_ft} then {b.md_ft})")
        for s in stations:
            if not 0 <= s.inc_deg <= 180:
                raise ValueError(f"inclination {s.inc_deg} outside 0-180 deg")
        return stations

    @property
    def td_md_ft(self) -> float:
        return self.stations[-1].md_ft


class VerticalTrajectory(_Model):
    td_ft: Ft

    @property
    def td_md_ft(self) -> float:
        return self.td_ft

    @property
    def td_tvd_ft(self) -> float:
        return self.td_ft


class PlanTrajectory(_Model):
    """Vertical to KOP, constant build to final inclination, then a straight lateral.

    Give kop_ft or landing_tvd_ft (or both, if they agree within 1 ft): with a
    build to final_inc_deg, landing TVD = KOP + R sin(final_inc), R = 18000/(pi BUR).
    """

    kop_ft: Ft | None = None
    landing_tvd_ft: Ft | None = None
    build_deg_per_100ft: DegPer100Ft
    final_inc_deg: Deg = 90.0
    lateral_ft: Ft = 0.0
    azimuth_deg: Deg = 0.0

    @model_validator(mode="after")
    def _check(self) -> PlanTrajectory:
        if self.build_deg_per_100ft <= 0:
            raise ValueError("build_deg_per_100ft must be positive")
        if not 0 < self.final_inc_deg <= 90:
            raise ValueError("final_inc_deg must be in (0, 90]")
        if self.kop_ft is None and self.landing_tvd_ft is None:
            raise ValueError("give kop_ft or landing_tvd_ft")
        gain_ft = self.radius_ft * math.sin(math.radians(self.final_inc_deg))
        if self.kop_ft is not None and self.landing_tvd_ft is not None:
            if abs(self.kop_ft + gain_ft - self.landing_tvd_ft) > 1.0:
                raise ValueError(
                    f"kop_ft {self.kop_ft:g} + build TVD gain {gain_ft:.1f} = "
                    f"{self.kop_ft + gain_ft:.1f} ft, not landing_tvd_ft {self.landing_tvd_ft:g}; "
                    "give only one of them")
        if self.kop_tvd_ft < 0:
            raise ValueError("landing_tvd_ft is shallower than the build section needs")
        return self

    @property
    def radius_ft(self) -> float:
        return _DEG_PER_100FT_TO_RADIUS_FT / self.build_deg_per_100ft

    @property
    def kop_tvd_ft(self) -> float:
        if self.kop_ft is not None:
            return self.kop_ft
        return self.landing_tvd_ft - self.radius_ft * math.sin(math.radians(self.final_inc_deg))

    @property
    def td_tvd_ft(self) -> float:
        inc = math.radians(self.final_inc_deg)
        return self.kop_tvd_ft + self.radius_ft * math.sin(inc) + self.lateral_ft * math.cos(inc)

    @property
    def td_md_ft(self) -> float:
        return self.kop_tvd_ft + self.final_inc_deg / self.build_deg_per_100ft * 100.0 + self.lateral_ft


def _trajectory_kind(v: Any) -> str:
    keys = v.keys() if isinstance(v, dict) else type(v).model_fields.keys()
    if "stations" in keys:
        return "survey"
    if "td_ft" in keys:
        return "vertical"
    return "plan"


Trajectory = Annotated[
    Union[Annotated[SurveyTrajectory, Tag("survey")], Annotated[VerticalTrajectory, Tag("vertical")],
          Annotated[PlanTrajectory, Tag("plan")]],
    Discriminator(_trajectory_kind)]


# --- Strings ----------------------------------------------------------------

StringKind = Literal["conductor", "surface", "intermediate", "production", "liner", "tieback"]
_KINDS = ("conductor", "surface", "intermediate", "production", "liner", "tieback")


def _depth_or(*keywords: str) -> BeforeValidator:
    def conv(v: Any) -> Any:
        if isinstance(v, str) and v.strip() in keywords:
            return v.strip()
        return to_field(v, "ft")
    return BeforeValidator(conv)


ShoeDepth = Annotated[Union[float, Literal["choose", "TD"]], _depth_or("choose", "TD")]
TocDepth = Annotated[Union[float, Literal["surface"]], _depth_or("surface")]


class CasingString(_Model):
    """One casing string or liner. Strings are listed outermost first."""

    name: str
    kind: StringKind
    hole_in: In
    od_in: In
    shoe_md_ft: ShoeDepth | None = None
    shoe_tvd_ft: ShoeDepth | None = None
    top_md_ft: Ft = 0.0
    toc_ft: TocDepth  # measured depth, as read from a cement log
    mud_ppg: UPpg | None = None

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        v = dict(v)
        if "toc" in v and "toc_ft" not in v:
            v["toc_ft"] = v.pop("toc")
        if "kind" not in v and isinstance(v.get("name"), str):
            # Infer kind from a name such as "surface" or "intermediate_1".
            name = v["name"].strip().lower()
            for kind in _KINDS:
                if name.startswith(kind):
                    v["kind"] = kind
                    break
        return v

    @model_validator(mode="after")
    def _check(self) -> CasingString:
        if (self.shoe_md_ft is None) == (self.shoe_tvd_ft is None):
            raise ValueError(f"string {self.name!r}: give exactly one of shoe_md_ft or shoe_tvd_ft")
        if self.od_in >= self.hole_in:
            raise ValueError(f"string {self.name!r}: od_in {self.od_in} must be smaller than hole_in {self.hole_in}")
        if self.kind == "liner" and self.top_md_ft <= 0:
            raise ValueError(f"liner {self.name!r} needs top_md_ft (hanger depth) below surface")
        if self.kind != "liner" and self.top_md_ft != 0:
            raise ValueError(f"string {self.name!r}: only a liner hangs below surface")
        if isinstance(self.shoe_value, float) and self.shoe_value <= self.top_md_ft:
            raise ValueError(f"string {self.name!r}: shoe must be below its top")
        return self

    @property
    def shoe_value(self) -> float | str:
        return self.shoe_md_ft if self.shoe_md_ft is not None else self.shoe_tvd_ft


class Fluids(_Model):
    completion_ppg: UPpg | None = None
    frac_fluid_ppg: UPpg | None = None
    cement_lead_ppg: UPpg | None = None
    cement_tail_ppg: UPpg | None = None
    mix_water_ppg: UPpg | None = None


# --- Load cases (spec Section 3.2) -------------------------------------------
# Each case is a settings block, `true` (use defaults), or `false` (off).
# A case left out gets the default for the string's kind in loads.py.

class GasKick(_Model):
    """Gas to surface from the next section's TD. kick_intensity_ppg is added
    to the pore pressure there. With limit_to_shoe_frac, pressure at this
    string's shoe is capped at the shoe's fracture pressure."""

    gas_gradient_psi_per_ft: UPsiPerFt | None = None
    kick_intensity_ppg: UPpg | None = None
    kick_volume_bbl: UBbl | None = None
    limit_to_shoe_frac: bool = False


class LostReturns(_Model):
    fluid_ppg: UPpg | None = None


class PressureTest(_Model):
    surface_psi: UPsi
    fluid_ppg: UPpg | None = None


class FracDownCasing(_Model):
    """Injection down casing; fluid_temp_F is the injected fluid temperature
    (cooling adds tension). If omitted, the surface temperature is used."""

    max_surface_psi: UPsi
    fluid_ppg: UPpg
    fluid_temp_F: UDegF | None = None


class TubingLeak(_Model):
    surface_psi: UPsi
    packer_fluid_ppg: UPpg | None = None


class Evacuation(_Model):
    """Fluid level below surface, ft; omit (or write `evacuation: full`) for full evacuation."""

    level_ft: UFt | None = None


class Cementing(_Model):
    displacement_ppg: UPpg | None = None


class Depletion(_Model):
    pore_ppg: UPpg
    annulus_level_ft: UFt | None = None


class Running(_Model):
    """Axial load while running: buoyed weight plus the larger of overpull and shock."""

    overpull_lbf: float | None = None
    shock: bool = True


def _switch(v: Any) -> Any:
    if v is True or v == "full":
        return {}
    return v


def _case(model: type) -> Any:
    return Annotated[Union[model, Literal[False], None], BeforeValidator(_switch)]


class StringLoadCases(_Model):
    gas_kick: _case(GasKick) = None
    lost_returns: _case(LostReturns) = None
    pressure_test: _case(PressureTest) = None
    frac_down_casing: _case(FracDownCasing) = None
    tubing_leak: _case(TubingLeak) = None
    evacuation: _case(Evacuation) = None
    cementing: _case(Cementing) = None
    depletion: _case(Depletion) = None
    running: _case(Running) = None


# --- Design rules and the rest -----------------------------------------------

class DesignFactors(_Model):
    burst: float = DEFAULT_DESIGN_FACTORS["burst"]
    collapse: float = DEFAULT_DESIGN_FACTORS["collapse"]
    tension: float = DEFAULT_DESIGN_FACTORS["tension"]
    compression: float = DEFAULT_DESIGN_FACTORS["compression"]
    triaxial: float = DEFAULT_DESIGN_FACTORS["triaxial"]


class Margins(_Model):
    trip_margin_ppg: float = DEFAULT_MARGINS_PPG["trip_margin_ppg"]
    kick_margin_ppg: float = DEFAULT_MARGINS_PPG["kick_margin_ppg"]


class Well(_Model):
    name: str
    kb_ft: Ft
    ground_level_ft: Ft | None = None
    wellhead_rating_psi: UPsi | None = None


class ReferenceSection(_Model):
    grade: str
    weight_lb_per_ft: float
    od_in: In | None = None
    connection: str | None = None
    top_md_ft: Ft | None = None
    bottom_md_ft: Ft | None = None


class ReferenceDesign(_Model):
    """A design to compare the optimized result with (spec Section 4.4, step 5)."""

    name: str
    source: str | None = None
    strings: dict[str, list[ReferenceSection]]


class Correlation(_Model):
    a: str
    b: str
    rho: float = Field(ge=-1.0, le=1.0)


class Meta(_Model):
    synthetic: bool = False
    generator: str | None = None
    seed: int | None = None
    params: dict[str, Any] | None = None
    note: str | None = None


class WellInput(_Model):
    """One well's complete input (spec Section 2.1)."""

    well: Well
    trajectory: Trajectory
    pore_pressure_ppg: PpgCurve
    frac_gradient: Annotated[Union[FracEaton, FracTable], Field(discriminator="method")]
    overburden_psi_per_ft: Overburden | None = None
    temperature: Temperature
    strings: list[CasingString] = Field(min_length=1)
    fluids: Fluids = Fluids()
    load_cases: dict[str, StringLoadCases] = {}
    design_factors: DesignFactors = DesignFactors()
    margins: Margins = Margins()
    reliability_target_pf: float = Field(DEFAULT_RELIABILITY_TARGET_PF, gt=0.0, lt=1.0)
    catalog: str | None = None
    reference_designs: list[ReferenceDesign] = []
    regulatory: list[str] = []
    correlations: list[Correlation] = []
    sources: dict[str, str] = {}
    meta: Meta = Meta()

    @model_validator(mode="after")
    def _check(self) -> WellInput:
        names = [s.name for s in self.strings]
        if len(set(names)) != len(names):
            raise ValueError("string names must be unique")
        for key in self.load_cases:
            if key not in names:
                raise ValueError(f"load_cases names unknown string {key!r}")
        for ref in self.reference_designs:
            for key in ref.strings:
                if key not in names:
                    raise ValueError(f"reference design {ref.name!r} names unknown string {key!r}")
        nested = [s for s in self.strings if s.kind != "tieback"]
        for outer, inner in zip(nested, nested[1:]):
            if inner.hole_in >= outer.od_in:
                raise ValueError(f"string {inner.name!r}: hole_in {inner.hole_in} must pass through "
                                 f"{outer.name!r} (od_in {outer.od_in}); list strings outermost first")
        td_md_ft = self.trajectory.td_md_ft
        for s in self.strings:
            if isinstance(s.shoe_md_ft, float) and s.shoe_md_ft > td_md_ft + 1.0:
                raise ValueError(f"string {s.name!r}: shoe_md_ft {s.shoe_md_ft:g} is below TD ({td_md_ft:.0f} ft MD)")
        return self

    def placeholders(self) -> list[str]:
        """Inputs that fell back to a default; the report must list each one."""
        out = []
        df_set = self.design_factors.model_fields_set if "design_factors" in self.model_fields_set else set()
        out += [f"design_factors.{k} = {getattr(self.design_factors, k)} (spec 3.6 default)"
                for k in DesignFactors.model_fields if k not in df_set]
        m_set = self.margins.model_fields_set if "margins" in self.model_fields_set else set()
        unset_m = [k for k in Margins.model_fields if k not in m_set]
        out += [f"margins.{k} = {getattr(self.margins, k)} ppg (spec 3.1 default)" for k in unset_m]
        if "reliability_target_pf" not in self.model_fields_set:
            out.append(f"reliability_target_pf = {DEFAULT_RELIABILITY_TARGET_PF:g} (spec 4.3 default)")
        if self.overburden_psi_per_ft is None:
            out.append(f"overburden_psi_per_ft = {DEFAULT_OVERBURDEN_PSI_PER_FT} (constant-gradient placeholder)")
        return out


def load_well(path: str | Path) -> WellInput:
    """Read and validate a well input file (.yaml, .yml or .json)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return WellInput.model_validate(data)
