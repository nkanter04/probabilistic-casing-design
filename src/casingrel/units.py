"""Unit handling at the input/report boundary.

Inside the package every number is in field units (psi, ft, in, lbf, ppg, degF)
with the unit in the variable name. This module converts user input to those
units with pint. A plain number is taken to be in the field unit already; a
string such as "2600 m" or "1.38 g/cm**3" is parsed and converted.
"""

from __future__ import annotations

from functools import lru_cache

import pint

# Hydrostatic gradient per unit mud weight, psi/ft per ppg: P = 0.052 * rho * TVD
# (spec Section 3; Bourgoyne et al. 1986, Ch. 4). The exact value is 0.051948.
PSI_PER_FT_PER_PPG = 0.052

ureg = pint.UnitRegistry()
ureg.define("ppg = pound / gallon")

_PRESSURE_GRADIENT = ureg.psi / ureg.ft


@lru_cache(maxsize=None)
def _unit(unit: str) -> pint.Unit:
    return ureg.Unit(unit)


def to_field(value: float | int | str, unit: str, scale: float = 1.0) -> float:
    """Convert a number or a unit string to field unit `unit`, times `scale`.

    Plain numbers are returned unchanged (already in the field unit). For
    unit "ppg", a pressure gradient ("0.465 psi/ft", "11.3 kPa/m") is also
    accepted and divided by PSI_PER_FT_PER_PPG. `scale` handles per-100-ft
    units: gradient_F_per_100ft uses unit "delta_degF/ft" with scale 100.
    """
    if isinstance(value, bool):
        raise ValueError(f"expected a number or a quantity string, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"expected a number or a quantity string, got {value!r}")
    try:
        q = ureg.Quantity(value.strip())
    except (pint.errors.PintError, ValueError, AttributeError) as exc:
        raise ValueError(f"cannot parse quantity {value!r}: {exc}") from exc
    if q.dimensionless and _unit(unit).dimensionless:
        return float(q.magnitude) * scale
    if q.dimensionless:
        raise ValueError(f"{value!r} has no unit; give a plain number in {unit} or add a unit")
    target = _unit(unit)
    if unit == "ppg" and q.dimensionality == _PRESSURE_GRADIENT.dimensionality:
        return float(q.to(_PRESSURE_GRADIENT).magnitude) / PSI_PER_FT_PER_PPG
    if q.dimensionality != target.dimensionality:
        raise ValueError(f"{value!r} cannot be converted to {unit}")
    return float(q.to(target).magnitude) * scale


def convert_spread(center: float, spread: float, src_unit: str,
                   unit: str, scale: float = 1.0) -> float:
    """Convert a spread (standard deviation, half-width) from `src_unit` to field `unit`.

    A spread is a difference, so offset units (degC to degF) must not shift
    it. Converting center + spread and subtracting the converted center gives
    the right factor for both linear and offset units.
    """
    return (to_field(f"{center + spread} {src_unit}", unit, scale)
            - to_field(f"{center} {src_unit}", unit, scale))
