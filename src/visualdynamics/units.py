"""Unit handling for visualdynamics.

Values whose units are known are stored in SI (m, kg, s, N, Pa, K, ...).
Values read from a source that does not declare units are stored exactly
as they appear in the file and tagged `unknown` until the user defines
units. Defining units converts them to SI once and records the unit
chosen, so a wrong choice can be corrected without losing anything.

Conversions are affine, `si = raw * scale + offset`, so offset scales such
as degC and degF are handled correctly along with purely multiplicative
units.

pint provides the unit registry and is used only inside this module. The
core objects carry plain numpy arrays and a dimension tag: a string such
as 'length' or 'acceleration', or a compound expression such as
'acceleration/force'.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from numpy.typing import ArrayLike

UNKNOWN = 'unknown'

_ureg = None


def unit_registry() -> Any:
    """The pint unit registry, built on first use.

    There is one registry because pint units from two registries cannot
    be compared. It is built on first use because building it takes about
    0.1 s, which a script that converts no units should not have to wait
    for.

    Returns
    -------
    pint.UnitRegistry
        The shared registry, with `gn` defined as standard gravity.
    """
    global _ureg
    if _ureg is None:
        import pint

        _ureg = pint.UnitRegistry()  # pint ships slinch (lbf*s^2/in) already
        _ureg.define('@alias standard_gravity = gn')
    return _ureg


# In visualdynamics, 'g' is standard gravity, never grams. Accelerometers are read in
# g all day long, and pint's own reading — 0.001 kg — is both wrong and
# plausible enough to pass unnoticed. Grams are still reachable as 'gram';
# it is the one-letter shorthand that is reserved. Word boundaries keep
# 'kg', 'mg' and 'gn' to themselves.
_BARE_G = re.compile(r'(?<![A-Za-z0-9_])g(?![A-Za-z0-9_])')

# The same trap, one unit over: a mil of vibration displacement is a
# thousandth of an inch, but pint's 'mil' is the angular one — 1/6400 of a
# turn, and dimensionless, so it neither converts nor complains. 'thou' is
# pint's name for the length.
_BARE_MIL = re.compile(r'(?<![A-Za-z0-9_])mils?(?![A-Za-z0-9_])')
# 'lbm' is the engineering spelling of pound-mass; pint has only 'lb', and
# refuses 'lbm' outright — which made it a dead entry in the mass shortlist
_BARE_LBM = re.compile(r'\blbm\b')


def _spell(unit: str) -> str:
    """The three substitutions above, applied to a string as written."""
    unit = _BARE_LBM.sub('lb', unit)
    return _BARE_MIL.sub('thou', _BARE_G.sub('standard_gravity', unit))


# The spellings the toolset speaks, for reading a unit whose case is not
# pint's. A controller's channel table holds whatever was typed into it,
# and `G`, `LBF` and `Volts` are typed all day: pint reads `G` as gauss
# and refuses the other two, so a run's accelerometers arrived unit-less
# while its `V` channels came through (Brandon, 2026-09-17, on a real
# import). The rule is that the string as written wins whenever it names
# a quantity this toolset tracks — `mV` and `MV` are different voltages
# and stay so — and only a string that means nothing here is re-read
# token by token against this list, ignoring case. No prefix arithmetic:
# a token folds to one of these spellings or is left alone.
_UNIT_SPELLINGS = (
    'g', 'gn', 'standard_gravity',
    'in', 'inch', 'inches', 'ft', 'foot', 'feet', 'm', 'meter', 'meters',
    'mm', 'cm', 'km', 'um', 'mil', 'mils', 'thou',
    's', 'sec', 'second', 'seconds', 'ms', 'us', 'min', 'minute', 'hr', 'hour',
    'kg', 'gram', 'slinch', 'slug', 'lbm', 'lb', 'tonne',
    'lbf', 'N', 'kN', 'newton', 'newtons',
    'psi', 'Pa', 'kPa', 'MPa',
    'V', 'mV', 'volt', 'volts',
    'strain', 'microstrain', 'dimensionless',
    'degC', 'degF', 'K', 'degR',
    'rad', 'radian', 'radians', 'deg', 'degree', 'degrees',
    'Hz', 'kHz',
)
_FOLD = {spelling.lower(): spelling for spelling in _UNIT_SPELLINGS}
_UNIT_TOKEN = re.compile(r'[A-Za-z_]+')


@lru_cache(maxsize=4096)
def fold_unit_case(unit: str) -> str:
    """`unit` with its letter case corrected, when only the corrected
    spelling is a unit this toolset can read.

    `'G'` becomes `'g'`, `'LBF/IN'` becomes `'lbf/in'` and `'Volts'`
    becomes `'volts'`. A string that already names a tracked quantity is
    returned unchanged, so `'mV'` stays millivolts and `'MV'` stays
    megavolts. A string that no change of case makes readable is also
    returned unchanged, so a caller can still tell an unknown unit from a
    known one.

    Parameters
    ----------
    unit : str
        The unit as written, for example in a channel table.

    Returns
    -------
    str
        The respelled unit, or `unit` itself.
    """
    if not isinstance(unit, str) or _reading(unit) == 2:
        return unit
    folded = _UNIT_TOKEN.sub(
        lambda m: _FOLD.get(m.group(0).lower(), m.group(0)), unit)
    if folded != unit and _reading(folded) > _reading(unit):
        return folded
    return unit


def _reading(unit: str) -> int:
    """How much a spelling means here: 2 names a dimension this toolset
    tracks, 1 is a unit pint can at least parse (a stiffness in lbf/in,
    say — a compound a reference unit may carry), 0 is nothing. The fold
    above trades up and never sideways: `G`, gauss as written, is a 1
    that `g` makes a 2, while `MV` is a 2 as it stands."""
    spelled = _spell(unit)
    if _dimension_of_spelled(spelled):
        return 2
    try:
        unit_registry().Quantity(1.0, spelled)
    except Exception:  # noqa: BLE001 - unparseable is simply nothing
        return 0
    return 1


def normalize_unit(unit: str) -> str:
    """A unit string as visualdynamics reads it, before it is given to pint.

    The case is corrected as in `fold_unit_case`, then `g` is read as
    standard gravity (never grams, which are written `gram`), `mil` as a
    thousandth of an inch and `lbm` as pound-mass.

    Parameters
    ----------
    unit : str
        The unit as written.

    Returns
    -------
    str
        The spelling pint is given.
    """
    if not isinstance(unit, str):
        return unit
    return _spell(fold_unit_case(unit))


# Dimensionality strings pint should report for each visualdynamics dimension tag
_DIMENSIONALITY = {
    'length': '[length]',
    'mass': '[mass]',
    'time': '[time]',
    'frequency': '1 / [time]',
    'force': '[length] * [mass] / [time] ** 2',
    'acceleration': '[length] / [time] ** 2',
    'velocity': '[length] / [time]',
    'pressure': '[mass] / [length] / [time] ** 2',
    'temperature': '[temperature]',
    'voltage': '[length] ** 2 * [mass] / [current] / [time] ** 3',
    # pint spells "no dimension" as the empty expression; asking it for the
    # dimensionality of the word 'dimensionless' raises KeyError('')
    'strain': '',
    'dimensionless': '',
    # Rotations about a point, for the virtual point transformation
    # (PLAN.md "The virtual point arc"): a unit rigid-body rotation
    # shape is length per radian, so a response carried through it
    # comes out per radian. pint holds the radian dimensionless, which
    # is why an angle looks like a strain and an angular velocity like
    # a frequency here — `dimension_of` tells them apart by the unit's
    # own spelling.
    'angle': '',
    'angular_velocity': '1 / [time]',
    'angular_acceleration': '1 / [time] ** 2',
    'moment': '[length] ** 2 * [mass] / [time] ** 2',
    # Modal responses through mass-normalized shapes, which are in
    # 1/√mass: `[q] = [u]/[Φ]` puts a half power of mass on the
    # response. The dimension algebra below is integer-powered by
    # design — these are base tags precisely because the half power
    # cannot be composed — and pint takes the fractional exponent.
    'modal_acceleration': '[length] * [mass] ** 0.5 / [time] ** 2',
    'modal_velocity': '[length] * [mass] ** 0.5 / [time]',
    'modal_length': '[length] * [mass] ** 0.5',
    'modal_force': '[length] * [mass] ** 1.5 / [time] ** 2',
}

#: the tags whose display unit follows from a system's base units,
#: filled in for any system that does not name them itself
ANGULAR_UNITS = {'angle': 'rad', 'angular_velocity': 'rad/s',
                 'angular_acceleration': 'rad/s**2'}
MODAL_QUANTITIES = ('acceleration', 'velocity', 'length', 'force')

_ANGLE_TOKEN = re.compile(r'\b(rad|radian|radians|deg|degree|degrees)\b')


def _derived_units(units: dict) -> dict:
    """The angular, moment and modal units a system's base units imply."""
    out = {tag: unit for tag, unit in ANGULAR_UNITS.items()
           if tag not in units}
    if 'moment' not in units and 'force' in units and 'length' in units:
        out['moment'] = f"{units['force']}*{units['length']}"
    if 'mass' in units:
        for quantity in MODAL_QUANTITIES:
            tag = f'modal_{quantity}'
            if tag not in units and quantity in units:
                out[tag] = f"{units[quantity]}*{units['mass']}**0.5"
    return out


class UnitError(ValueError):
    """A unit could not be read, or is not of the dimension required.

    It is a ValueError, since a bad unit is a bad value, and it has its
    own name so the app can tell a unit problem, which it explains to the
    user, from a programming error, which it does not catch.
    """


class UnitsRequired(UnitError):
    """Raised when an operation needs units that have not been defined.

    `required` names the unit settings that are missing (by default
    `('length_unit',)`), so the caller can ask for exactly those.
    """

    def __init__(self, message: str,
                 required: Sequence[str] = ('length_unit',)) -> None:
        super().__init__(message)
        self.required: tuple[str, ...] = tuple(required)


_SUPERSCRIPT_DIGITS = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
_POWER = re.compile(r'\*\*(-?\d+(?:\.\d+)?)')
_FROM_SUPERSCRIPT = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻', '0123456789-')
_PRETTY_POWER = re.compile(r'([A-Za-z\)])([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+|½)')


def _superscript(power: str) -> str:
    # the one fractional power in use, the modal units' √mass, as the
    # glyph a reader knows rather than a superscript decimal point
    if power == '0.5':
        return '½'
    return power.translate(_SUPERSCRIPT_DIGITS)


def _is_compound(unit: str) -> bool:
    """Does the unit already contain an operator that needs bracketing?"""
    return '/' in unit or '·' in unit or '*' in unit


def _exponents_to_unicode(unit: str) -> str:
    return _POWER.sub(lambda m: _superscript(m.group(1)),
                      unit).replace('*', '·')


def pretty_unit(unit: str) -> str:
    """A unit string with superscript exponents, for display:
    'm/s**2' -> 'm/s²'.

    Parameters
    ----------
    unit : str
        The unit in pint's spelling.

    Returns
    -------
    str
        The same unit with superscripts and `·` for products.
    """
    return _exponents_to_unicode(unit)


def plain_unit(text: str) -> str:
    """The inverse of `pretty_unit`, so a displayed unit can be read back.

    Parameters
    ----------
    text : str
        A unit as `pretty_unit` writes it, such as 'm/s²'.

    Returns
    -------
    str
        The unit in pint's spelling, such as 'm/s**2'.
    """
    text = text.replace('·', '*')
    return _PRETTY_POWER.sub(
        lambda m: f'{m.group(1)}**'
        + ('0.5' if m.group(2) == '½'
           else m.group(2).translate(_FROM_SUPERSCRIPT)),
        text)


def is_compound_unit(unit: str) -> bool:
    """Whether a unit needs brackets before it is raised to a power or
    divided by another.

    Parameters
    ----------
    unit : str
        The unit string.

    Returns
    -------
    bool
        True if it contains `/`, `*` or `·`.
    """
    return _is_compound(unit)


def _exponents_to_html(unit: str) -> str:
    return _POWER.sub(
        lambda m: f'<sup>{"½" if m.group(1) == "0.5" else m.group(1)}</sup>',
        unit).replace('*', '·')


def _exponents_to_ascii(unit: str) -> str:
    # '*' stays: it is the compound marker `_is_compound` keys on, and
    # ASCII has no middle dot to trade it for
    return _POWER.sub(lambda m: f'^{m.group(1)}', unit)


def _render_parts(parts, render_unit, apply_power):
    """Render dimension parts into (numerator, denominator) pieces.

    Each piece is (text, atomic) where atomic means the text needs no
    brackets if it is combined further — which is what keeps a squared
    compound unit from ending up double-bracketed.
    """
    numerator, denominator = [], []
    for unit, power in parts:
        text = render_unit(unit)
        atomic = not _is_compound(text)
        if abs(power) != 1:
            text = apply_power(text if atomic else f'({text})', abs(power))
            atomic = True
        (numerator if power > 0 else denominator).append((text, atomic))

    def combine(pieces: list[tuple[str, bool]]) -> tuple[str, bool]:
        if not pieces:
            return '', True
        if len(pieces) == 1:
            return pieces[0]
        return '·'.join(text for text, _ in pieces), False

    return combine(numerator), combine(denominator)


def _split_products(chunk: str) -> list[str]:
    """Split 'a*b**2*c' on product '*' while keeping '**' powers intact."""
    out, token, i = [], '', 0
    while i < len(chunk):
        if chunk.startswith('**', i):
            token += '**'
            i += 2
        elif chunk[i] == '*':
            out.append(token)
            token = ''
            i += 1
        else:
            token += chunk[i]
            i += 1
    out.append(token)
    return out


@lru_cache(maxsize=4096)
def _parse_dimension(expression: str) -> tuple[tuple[str, int], ...]:
    return tuple(_parse_dimension_uncached(expression))


def parse_dimension(expression: str) -> list[tuple[str, int]]:
    """Parse a dimension expression into (base_dimension, power) parts.

    'acceleration/force' gives `[('acceleration', 1), ('force', -1)]`.
    The results are cached.

    Parameters
    ----------
    expression : str
        A dimension tag or an expression of tags joined by `*`, `/` and
        integer `**` powers.

    Returns
    -------
    list of (str, int)
        Each base dimension and its power.

    Raises
    ------
    UnitError
        If the expression names a dimension that is not tracked.
    """
    # Cached because this is a pure function of a short string and the
    # plot asks for the same few expressions once per record: drawing
    # one curve out of a 1356-record FRF used to parse and convert 1356
    # times.
    return list(_parse_dimension(expression))


def _parse_dimension_uncached(expression: str) -> list[tuple[str, int]]:
    """The parsing itself; see parse_dimension."""
    if expression == UNKNOWN:
        return [(UNKNOWN, 1)]
    parts = []
    for i, chunk in enumerate(expression.split('/')):
        for factor in _split_products(chunk):
            name, _, exponent = factor.partition('**')
            name = name.strip()
            power = int(exponent) if exponent else 1
            if name not in _DIMENSIONALITY:
                raise UnitError(f"Unknown dimension {name!r} in {expression!r}")
            parts.append((name, -power if i > 0 else power))
    return parts


@lru_cache(maxsize=4096)
def si_transform(unit: str, dimension: str | None = None) -> tuple[float, float]:
    """(scale, offset) such that `si_value = value * scale + offset`.

    The offset is non-zero only for affine units such as degC and degF.

    Parameters
    ----------
    unit : str
        The unit, for example 'mm' or 'degF'.
    dimension : str, optional
        If given, the unit is checked to be of this dimension.

    Returns
    -------
    tuple of (float, float)
        The scale and the offset.

    Raises
    ------
    UnitError
        If the unit cannot be read, or is not of `dimension`.
    """
    ureg = unit_registry()
    unit = normalize_unit(unit)
    try:
        zero = ureg.Quantity(0.0, unit).to_base_units().magnitude
        one = ureg.Quantity(1.0, unit).to_base_units().magnitude
    except Exception as e:  # noqa: BLE001 - surface any pint failure uniformly
        raise UnitError(f"Cannot interpret unit {unit!r}: {e}")
    scale = one - zero
    if dimension is not None and dimension != UNKNOWN:
        expected = _DIMENSIONALITY.get(dimension)
        if expected is None:
            raise UnitError(f"Unknown dimension {dimension!r}")
        actual = ureg.Quantity(1.0, unit).dimensionality
        if actual != ureg.get_dimensionality(expected):
            article = 'an' if dimension[:1] in 'aeiou' else 'a'
            raise UnitError(f"Unit {unit!r} is not {article} {dimension} unit "
                            f"(got {actual})")
    return scale, zero


def si_factor(unit: str, dimension: str | None = None) -> float:
    """Multiplier converting values in `unit` to SI.

    Affine units (degC, degF) have no single factor; convert those with
    `to_si` and `from_si`.

    Parameters
    ----------
    unit : str
        The unit, for example 'mm' or 'lbf'.
    dimension : str, optional
        If given, the unit is checked to be of this dimension.

    Returns
    -------
    float
        The factor: a value in `unit` times this is the value in SI.

    Raises
    ------
    UnitError
        If the unit has an offset, cannot be read, or is not of
        `dimension`.
    """
    scale, offset = si_transform(unit, dimension)
    if offset:
        raise UnitError(
            f"Unit {unit!r} has an offset; use to_si()/from_si() instead of a "
            "bare scale factor")
    return scale


def to_si(values: ArrayLike, unit: str,
          dimension: str | None = None) -> Any:
    """`values`, read as `unit`, converted to SI.

    Offsets are applied, so degrees Celsius are converted to kelvin
    correctly.

    Parameters
    ----------
    values : array-like
        The values, in `unit`.
    unit : str
        The unit they are in.
    dimension : str, optional
        If given, the unit is checked to be of this dimension.

    Returns
    -------
    numpy.ndarray or float
        The values in SI.
    """
    scale, offset = si_transform(unit, dimension)
    return values * scale + offset


def from_si(values: ArrayLike, unit: str,
            dimension: str | None = None) -> Any:
    """SI `values` converted to `unit`; the inverse of `to_si`.

    Parameters
    ----------
    values : array-like
        The values, in SI.
    unit : str
        The unit wanted.
    dimension : str, optional
        If given, the unit is checked to be of this dimension.

    Returns
    -------
    numpy.ndarray or float
        The values in `unit`.
    """
    scale, offset = si_transform(unit, dimension)
    return (values - offset) / scale


def convert(values: ArrayLike, from_unit: str, to_unit: str) -> Any:
    """`values` converted from one unit to another, through SI.

    Parameters
    ----------
    values : array-like
        The values, in `from_unit`.
    from_unit : str
        The unit they are in, for example 'in/s**2'.
    to_unit : str
        The unit wanted, of the same dimension.

    Returns
    -------
    numpy.ndarray or float
        The values in `to_unit`, the shape of `values`.
    """
    return from_si(to_si(values, from_unit), to_unit)


def dimension_of(unit: str) -> str | None:
    """The visualdynamics dimension tag of a unit string, or None.

    Only single tags are returned ('m/s**2' -> 'acceleration'). Compound
    quantities such as FRFs carry their dimension expression explicitly.
    A dimensionless unit is 'angle' if it is spelled with an angle (rad,
    deg) and 'dimensionless' otherwise, and rad/s is 'angular_velocity'
    where Hz is 'frequency'.

    Parameters
    ----------
    unit : str
        The unit string.

    Returns
    -------
    str or None
        The dimension tag, or None if the unit cannot be read or matches
        no tracked dimension.
    """
    return _dimension_of_spelled(normalize_unit(unit))


def _dimension_of_spelled(unit: str) -> str | None:
    """`dimension_of` for a string that has already been through
    `normalize_unit` — the half that asks pint, kept apart so the case
    fold above can ask it without normalizing again."""
    ureg = unit_registry()
    try:
        dimensionality = ureg.Quantity(1.0, unit).dimensionality
    except Exception:  # noqa: BLE001 - any unparseable unit is simply unknown
        return None
    # pint holds the radian dimensionless, so rad/s and Hz share a
    # dimensionality; a unit spelled with an angle is the angular one
    angular = bool(_ANGLE_TOKEN.search(unit))
    for tag, expr in _DIMENSIONALITY.items():
        if tag in ('dimensionless', 'strain', 'angle'):
            continue
        if (tag in ANGULAR_UNITS) != angular:
            continue
        if dimensionality == ureg.get_dimensionality(expr):
            return tag
    if not dimensionality:
        return 'angle' if angular else 'dimensionless'
    return None


@dataclass(frozen=True)
class UnitSystem:
    """A named mapping of dimension tag to display unit.

    `base` is the coherent system a derived system was made from, such as
    in-slinch-lbf-s for in-slinch-lbf-s (g), which shows accelerations in
    g. It is None for a coherent system. Most exporters write in the
    coherent system (see `coherent`), because most file formats cannot
    label a value as g.

    The angular, moment and modal units are filled in from the base
    units when the system does not name them.
    """

    name: str
    units: dict = field(default_factory=dict)
    base: UnitSystem | None = None

    def __post_init__(self) -> None:
        # the angular, moment and modal units follow from the base
        # units, so a system that names its length, mass and force
        # has them without listing them (the dict is filled in place:
        # the dataclass is frozen, its dictionary is not)
        self.units.update(_derived_units(self.units))

    @property
    def coherent(self) -> UnitSystem:
        """The coherent system this one is based on, or this system if it
        is coherent."""
        return self.base or self

    def unit(self, dimension: str) -> str:
        """Display unit for a dimension tag or dimension expression.

        Expressions are built from the base units: in in-slinch-lbf-s,
        'acceleration/force' gives '(in/s**2)/lbf'.

        Parameters
        ----------
        dimension : str
            A dimension tag, such as 'acceleration', or an expression,
            such as 'acceleration/force'.

        Returns
        -------
        str
            The unit string in pint's spelling; empty for 'unknown'.

        Raises
        ------
        UnitError
            If the system has no unit for a dimension in the expression.
        """
        if dimension == UNKNOWN:
            return ''
        if dimension in self.units:
            return self.units[dimension]
        numerator, denominator = [], []
        for name, power in parse_dimension(dimension):
            unit = self.units.get(name)
            if unit is None:
                raise UnitError(
                    f"Unit system {self.name!r} has no unit for dimension {name!r}")
            label = f'({unit})' if ('/' in unit or '*' in unit) else unit
            if abs(power) != 1:
                label += f'**{abs(power)}'
            (numerator if power > 0 else denominator).append(label)
        out = '*'.join(numerator) if numerator else '1'
        for label in denominator:
            out += f'/{label}'
        return out

    def transform(self, dimension: str) -> tuple[float, float]:
        """(scale, offset) converting a value in this system's unit to SI,
        `si = value * scale + offset`.

        Parameters
        ----------
        dimension : str
            A dimension tag or expression, such as 'acceleration' or
            'acceleration/force'.

        Returns
        -------
        tuple of (float, float)
            The scale and offset. The offset is non-zero only for
            temperature; (1.0, 0.0) for 'unknown'.

        Raises
        ------
        UnitError
            If an expression includes an offset unit such as degF.
        """
        if dimension == UNKNOWN:
            return 1.0, 0.0
        if dimension in self.units:
            return si_transform(self.units[dimension], dimension)
        scale = 1.0
        for name, power in parse_dimension(dimension):
            part_scale, part_offset = si_transform(self.units[name], name)
            if part_offset:
                raise UnitError(
                    f"Dimension {dimension!r} combines the offset unit "
                    f"{self.units[name]!r}; use an absolute unit instead")
            scale *= part_scale ** power
        return scale, 0.0

    def factor(self, dimension: str) -> float:
        """SI value per display unit, for dimensions with no offset.

        Parameters
        ----------
        dimension : str
            A dimension tag or expression, such as 'acceleration' or
            'acceleration/force'.

        Returns
        -------
        float
            What a value in this system's unit is multiplied by to give
            SI; divide an SI value by it to display it.

        Raises
        ------
        UnitError
            If the display unit has an offset (degC, degF).
        """
        scale, offset = self.transform(dimension)
        if offset:
            raise UnitError(
                f"Display unit for {dimension!r} has an offset; use "
                "from_si()/to_si()")
        return scale

    def from_si(self, values: ArrayLike, dimension: str) -> Any:
        """Convert SI values to this system's display unit for `dimension`.

        Parameters
        ----------
        values : array_like
            Values in SI.
        dimension : str
            A dimension tag or expression, such as 'acceleration' or
            'force'.

        Returns
        -------
        array_like
            The same values in this system's units.
        """
        scale, offset = self.transform(dimension)
        return (values - offset) / scale

    def to_si(self, values: ArrayLike, dimension: str) -> Any:
        """Convert values from this system's units into SI.

        Parameters
        ----------
        values : array_like
            Values in this system's units.
        dimension : str
            A dimension tag or expression, such as 'acceleration'.

        Returns
        -------
        array_like
            The same values in SI.
        """
        scale, offset = self.transform(dimension)
        return values * scale + offset

    def label(self, dimension: str) -> str:
        """Display unit text, the same as `unit`; empty for 'unknown'.

        Parameters
        ----------
        dimension : str
            A dimension tag or expression, such as 'acceleration' or
            'force'.

        Returns
        -------
        str
            The unit string in this system.
        """
        return self.unit(dimension)

    def _unit_parts(self, dimension: str) -> list[tuple[str, int]]:
        """[(unit string, power)] making up a dimension expression."""
        if dimension == UNKNOWN:
            return []
        if dimension in self.units:
            return [(self.units[dimension], 1)]
        parts = []
        for name, power in parse_dimension(dimension):
            unit = self.units.get(name)
            if unit is None:
                raise UnitError(
                    f"Unit system {self.name!r} has no unit for dimension {name!r}")
            parts.append((unit, power))
        return parts

    def label_text(self, dimension: str) -> str:
        """Display unit for plain-text output, with superscript
        exponents: in/s².

        Parameters
        ----------
        dimension : str
            A dimension tag or expression, such as 'acceleration' or
            'force'.

        Returns
        -------
        str
            The label as plain text; empty for 'unknown'.
        """
        return self._slash_label(
            dimension, _exponents_to_unicode,
            lambda text, power: text + _superscript(str(power)))

    def label_ascii(self, dimension: str) -> str:
        """Display unit in plain ASCII, with exponents as carets: in/s^2.

        This is for renderers that drop characters they cannot draw. VTK's
        3-D axis titles drop unicode superscripts in every text mode, so
        `label_text`'s (in/s²)²/Hz is drawn as (in/s)/Hz there.

        Parameters
        ----------
        dimension : str
            A dimension tag or expression, such as 'acceleration' or
            'force'.

        Returns
        -------
        str
            The label with no unicode characters; empty for 'unknown'.
        """
        return self._slash_label(dimension, _exponents_to_ascii,
                                 lambda text, power: f'{text}^{power}')

    def _slash_label(self, dimension, render_unit, apply_power):
        """numerator/denominator on one line, the numerator bracketed
        when it is more than one unit — the one shape `label_text` and
        `label_ascii` share, differing only in how an exponent is
        drawn."""
        parts = self._unit_parts(dimension)
        if not parts:
            return ''
        numerator, denominator = _render_parts(parts, render_unit,
                                               apply_power)
        if not denominator[0]:
            return numerator[0] or '1'
        text, atomic = numerator
        if not atomic:
            text = f'({text})'
        return f'{text or "1"}/{denominator[0]}'

    def label_html(self, dimension: str) -> str:
        """Display unit as HTML, drawn as a stacked fraction when it has a
        denominator.

        Parameters
        ----------
        dimension : str
            A dimension tag or expression, such as 'acceleration' or
            'force'.

        Returns
        -------
        str
            The label with HTML superscripts; empty for 'unknown'.
        """
        parts = self._unit_parts(dimension)
        if not parts:
            return ''
        numerator, denominator = _render_parts(
            parts, _exponents_to_html,
            lambda text, power: f'{text}<sup>{power}</sup>')
        numerator, denominator = numerator[0] or '1', denominator[0]
        if not denominator:
            return numerator
        return (
            '<table cellpadding="0" cellspacing="0" style="display:inline">'
            f'<tr><td align="center" style="border-bottom:1px solid">'
            f'{numerator}</td></tr>'
            f'<tr><td align="center">{denominator}</td></tr></table>')

    def with_units(self, name: str | None = None, **overrides) -> UnitSystem:
        """A copy of this system with some dimensions' units replaced.

        `IN_LBF_S.with_units(acceleration='g')` shows acceleration in g
        and every other dimension in inch-pound-second units. The copy's
        `base` is this system's coherent system.

        Parameters
        ----------
        name : str, optional
            What to call the derived system. The default is this
            system's name followed by ' (custom)'.
        **overrides
            Dimension/unit pairs to change.

        Returns
        -------
        UnitSystem
            A copy with those units replaced.

        Raises
        ------
        UnitError
            If a unit cannot be read or is not of its dimension.
        """
        for dimension, unit in overrides.items():
            si_transform(unit, dimension)  # validates dimension and unit
        return UnitSystem(name or f'{self.name} (custom)',
                          {**self.units, **overrides}, base=self.coherent)


SI = UnitSystem('m-kg-N-s', {
    'length': 'm', 'mass': 'kg', 'time': 's', 'frequency': 'Hz',
    'force': 'N', 'acceleration': 'm/s**2', 'velocity': 'm/s',
    'pressure': 'Pa', 'temperature': 'K', 'voltage': 'V',
    'strain': '', 'dimensionless': '',
})

MMKS = UnitSystem('mm-kg-N-s', {
    'length': 'mm', 'mass': 'kg', 'time': 's', 'frequency': 'Hz',
    'force': 'N', 'acceleration': 'mm/s**2', 'velocity': 'mm/s',
    'pressure': 'MPa', 'temperature': 'degC', 'voltage': 'V',
    'strain': '', 'dimensionless': '',
})

IN_LBF_S = UnitSystem('in-slinch-lbf-s', {
    'length': 'in', 'mass': 'slinch', 'time': 's', 'frequency': 'Hz',
    'force': 'lbf', 'acceleration': 'in/s**2', 'velocity': 'in/s',
    'pressure': 'psi', 'temperature': 'degF', 'voltage': 'V',
    'strain': '', 'dimensionless': '',
})

FT_LBF_S = UnitSystem('ft-slug-lbf-s', {
    'length': 'ft', 'mass': 'slug', 'time': 's', 'frequency': 'Hz',
    'force': 'lbf', 'acceleration': 'ft/s**2', 'velocity': 'ft/s',
    'pressure': 'lbf/ft**2', 'temperature': 'degF', 'voltage': 'V',
    'strain': '', 'dimensionless': '',
})

_COHERENT = (SI, MMKS, IN_LBF_S, FT_LBF_S)

# Accelerometers are read in g, so every system gets a variant that shows
# accelerations that way. Only the display changes: most exports are
# written in the coherent system, since most file formats cannot record
# "g" as a unit (the .mat writer, which labels each value, is the
# exception; io/exporters.py).
_IN_G = tuple(s.with_units(f'{s.name} (g)', acceleration='g')
              for s in _COHERENT)

SYSTEMS = {s.name: s for s in _COHERENT + _IN_G}

# in-slinch-lbf-s with accelerations displayed in g (Brandon,
# 2026-08-28): the units his tests are specified and read in. Display
# only: most exports still write the coherent system's in/s**2, since
# most file formats cannot record "g" as a unit.
DEFAULT_SYSTEM = SYSTEMS['in-slinch-lbf-s (g)']
