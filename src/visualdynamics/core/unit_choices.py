"""Which units the interface offers, grouped by the quantity they measure.

Objects are imported without units when their source does not declare
them. The user sets the units in the imported-units pane (see
`MainWindow.show_units_panel`), and these lists are what it offers.

The lists are grouped by dimension, so a channel the file calls an
acceleration is offered the four acceleration units instead of every
unit in the list. They are shortlists, not limits: every unit cell
accepts any unit visualdynamics can read, typed or pasted.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..units import is_compound_unit, plain_unit, pretty_unit

# In core rather than beside the widgets that offer these lists: which
# units an acceleration could be in is a fact about accelerations, and
# `ChannelTable` needs it to narrow a channel's unit column. It lived in
# `gui/` and core reached up into it, which is the one direction imports
# must not go.

LENGTH_UNITS = ['in', 'ft', 'm', 'mm', 'cm']

# the mass units of visualdynamics's unit systems first, then other common ones.
# No grams: in visualdynamics 'g' is standard gravity, and offering it here would
# invite exactly the confusion that rule exists to prevent.
MASS_UNITS = ['kg', 'slinch', 'slug', 'lbm', 'tonne']

# Grouped by dimension so a record that knows what it is can be offered only
# the units it could possibly be in. Order within a group is the order the
# choices are offered in, so the likeliest comes first.
ORDINATE_UNITS = {
    'acceleration': ['g', 'in/s**2', 'm/s**2', 'mm/s**2'],  # 'g' is 9.80665
    'velocity': ['in/s', 'm/s', 'mm/s'],
    'length': ['mil', 'in', 'm', 'mm'],        # displacement
    'force': ['lbf', 'N', 'kN'],
    'pressure': ['psi', 'Pa', 'kPa', 'MPa'],
    'voltage': ['V', 'mV'],
    # strain has its own entry although its dimensionality is empty:
    # it is a quantity a channel measures, where 'dimensionless' is
    # the claim that a number has no units at all. Offering a strain
    # gauge every unit in the list was what the fallback did.
    # No 'ue': pint reads it as a *micro-elementary-charge*
    # (1.6e-25), not a microstrain, which is the same trap as bare 'g'
    # and 'mil' one unit over — it parses, it is wrong, and nothing
    # downstream would have said so.
    'strain': ['strain', 'microstrain', 'm/m'],
    'temperature': ['degC', 'degF', 'K', 'degR'],
    # rotations at a virtual point, and the moments conjugate to them
    'angle': ['rad', 'deg'],
    'angular_velocity': ['rad/s', 'deg/s'],
    'angular_acceleration': ['rad/s**2', 'deg/s**2'],
    'moment': ['lbf*in', 'N*m', 'lbf*ft'],
    # modal responses through mass-normalized shapes: the response's
    # unit times the square root of the mass unit the shapes carry
    'modal_acceleration': ['in/s**2*slinch**0.5', 'm/s**2*kg**0.5',
                           'g*slinch**0.5'],
    'modal_velocity': ['in/s*slinch**0.5', 'm/s*kg**0.5'],
    'modal_length': ['in*slinch**0.5', 'm*kg**0.5'],
    'modal_force': ['lbf*slinch**0.5', 'N*kg**0.5'],
    # spelled out, because '' means 'not declared' in every unit cell
    'dimensionless': ['dimensionless'],         # strain, ratio
}

ALL_ORDINATE_UNITS = [unit for units in ORDINATE_UNITS.values()
                      for unit in units]

#: What a dimension is called where a person reads it.
#:
#: The stored dimension is the physical one, and it has to be: it is
#: derived from the unit, and meters cannot say whether they are a
#: coordinate or a motion — a geometry's nodes, a plate's thickness and
#: a proximity probe's output are all `length`. But a *channel* measured
#: in meters is a displacement, and 'length' offered in a Type cell
#: reads as a mistake to anyone who has ever run a survey. So the
#: dimension keeps its name and the interface uses this one.
DISPLAY_DIMENSIONS = {'length': 'displacement',
                      'angular_velocity': 'angular velocity',
                      'angular_acceleration': 'angular acceleration',
                      'modal_acceleration': 'modal acceleration',
                      'modal_velocity': 'modal velocity',
                      'modal_length': 'modal displacement',
                      'modal_force': 'modal force'}
_STORED = {shown: stored for stored, shown in DISPLAY_DIMENSIONS.items()}


def shown_dimension(dimension: str) -> str:
    """The word for a dimension in the interface: 'length' is shown as
    'displacement', 'modal_velocity' as 'modal velocity'.

    Parameters
    ----------
    dimension : str
        The stored dimension tag.

    Returns
    -------
    str
        The word shown, or `dimension` itself if it has no other name.
    """
    return DISPLAY_DIMENSIONS.get(dimension, dimension)


def stored_dimension(shown: str) -> str:
    """The stored dimension tag for a word the interface shows; the
    inverse of `shown_dimension`, ignoring case and surrounding spaces.

    Parameters
    ----------
    shown : str
        The word, such as 'displacement'.

    Returns
    -------
    str
        The dimension tag, such as 'length'.
    """
    return _STORED.get(shown.strip().lower(), shown.strip().lower())

# What an FRF is usually driven by, when the file did not say
REFERENCE_UNITS = ['lbf', 'N', 'kN', 'g', 'm/s**2', 'in/s**2', 'V']


def base_dimension(dimension: str) -> str:
    """The quantity whose unit the user picks, given a record's dimension.

    This is the leading term of the expression: an FRF's ordinate is
    declared by its numerator, and a PSD by the engineering unit whose
    square it stores.

    Parameters
    ----------
    dimension : str
        A dimension tag or expression, such as 'acceleration/force'.

    Returns
    -------
    str
        The leading tag, such as 'acceleration'.
    """
    return dimension.split('/')[0].split('*')[0]


def units_for(dimension: str | None,
              fallback: Sequence[str] | None = None) -> Sequence[str]:
    """The units to offer for a dimension, or the fallback if it has no
    shortlist.

    A file often names a quantity without giving its unit; a UNV with no
    dataset 164 says 'acceleration' and nothing more. The list is then
    limited to the units of that quantity.

    Parameters
    ----------
    dimension : str or None
        A dimension tag or expression; only its leading term is used
        (`base_dimension`).
    fallback : sequence of str, optional
        What to offer when the dimension has no shortlist. The default
        is every unit in every shortlist.

    Returns
    -------
    sequence of str
        The units, the most likely first.
    """
    if fallback is None:
        fallback = ALL_ORDINATE_UNITS
    return ORDINATE_UNITS.get(base_dimension(dimension or ''), fallback)




# A PSD stores the square-per-Hz of an engineering unit, so a cell reading
# 'g' is telling only a third of the truth. Offer the whole label instead.
_SQUARED_PER_HZ = re.compile(r'\s*(?:\*\*\s*2|\^\s*2|²)\s*/\s*Hz\s*$', re.IGNORECASE)


def squared_per_hz(unit: str | None) -> str:
    """The full unit of a PSD declared in `unit`: 'g' -> 'g²/Hz'.

    A compound unit is bracketed, '(m/s²)²/Hz', so the square cannot be
    read as applying to the seconds only.

    Parameters
    ----------
    unit : str or None
        The engineering unit the PSD was declared in.

    Returns
    -------
    str
        The label, or an empty string if `unit` is empty or None.
    """
    if not unit:
        return ''
    shown = pretty_unit(unit)
    return f'({shown})²/Hz' if is_compound_unit(unit) else f'{shown}²/Hz'


def engineering_unit(text: str) -> str:
    """The unit inside a PSD label; the inverse of `squared_per_hz`.

    'g' and 'g²/Hz' both give 'g'. The interface adds the suffix for
    clarity, and the user does not have to type it.

    Parameters
    ----------
    text : str
        The label as shown or typed.

    Returns
    -------
    str
        The engineering unit in pint's spelling.
    """
    text = _SQUARED_PER_HZ.sub('', text.strip())
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]
    return plain_unit(text).strip()
