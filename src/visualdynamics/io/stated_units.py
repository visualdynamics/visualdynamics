"""Declaring the units a source states, one record at a time.

A reader whose source writes a unit beside each channel (a SEP 005
dict's ``unit_str``, a column header's "(g)") declares it on the record,
so the values are converted to SI the way `define_units` converts them.
Two things can stop that, and each reader decides what follows: the
unit is not one this package reads, or it reads as something no channel
measures. "Gs" is the case that brought the second up: pint reads it as
gigaseconds, a time, and a record declared in it would hold an
acceleration channel as a duration a billion seconds long. A unit that
reads as a time, a frequency or a mass is refused here, because a
channel of a time history measures none of them.

A claimed quantity can be checked against the unit at the same time:
a column stated to be an acceleration and headed "(lbf)" is a
contradiction, and the reader is told so rather than either claim
being preferred.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.unit_choices import ORDINATE_UNITS, shown_dimension
from ..units import dimension_of

# One helper for every reader that declares units a source stated: the
# SEP 005 converter had its own loop, and the delimited-text reader
# would have been a second copy of it (review, 2026-10-23).

#: the quantities a channel of a time history can measure, as stored
#: dimension tags: the unit shortlists' own groups
CHANNEL_QUANTITIES = frozenset(ORDINATE_UNITS)


def unit_problem(unit: str, quantity: str | None = None) -> str | None:
    """Why `unit` cannot be declared on a channel, or None when it can.

    Parameters
    ----------
    unit : str
        The unit as the source wrote it.
    quantity : str, optional
        The stored dimension tag the channel is claimed to measure
        ('acceleration'); the unit must then be one of its units.

    Returns
    -------
    str or None
        The reason in words, written to follow the quoted unit
        ("'counts' is ..."), or None.
    """
    dimension = dimension_of(unit)
    if dimension is None:
        return 'is not a unit Visual Dynamics reads'
    if dimension not in CHANNEL_QUANTITIES:
        return (f'reads as {_a(shown_dimension(dimension))}, which no '
                'channel of a time history measures')
    if quantity is not None and dimension != quantity:
        return (f'is {_a(shown_dimension(dimension))} unit, and the channel is '
                f'stated to be {_a(shown_dimension(quantity))}')
    return None


def _a(word: str) -> str:
    """`word` with its indefinite article: 'a force', 'an acceleration'."""
    return f'{"an" if word[:1].lower() in "aeiou" else "a"} {word}'


def declare_stated_units(data: Any, units: Sequence[str | None],
                         quantities: Sequence[str | None] | None = None
                         ) -> list[tuple[int, str, str]]:
    """Declare each record's stated unit where it can be declared.

    `units` holds one entry per record (None or '' for a record whose
    source stated nothing); `quantities`, when given, the dimension tag
    each record is claimed to measure, which its unit must agree with.
    A unit that passes `unit_problem` is declared with `define_units`,
    converting that record to SI. One that does not leaves the record's
    values as they are, and is returned with the reason.

    Parameters
    ----------
    data : DataArray
        The records, declared in place.
    units : sequence of str or None
        The unit each record's source stated.
    quantities : sequence of str or None, optional
        The quantity each record is claimed to measure.

    Returns
    -------
    list of (int, str, str)
        (record, unit, reason) for each unit that was not declared.
    """
    refused = []
    for i, unit in enumerate(list(units)[:data.num_records]):
        if not unit:
            continue
        unit = str(unit)
        quantity = quantities[i] if quantities is not None else None
        reason = unit_problem(unit, quantity)
        if reason is None:
            try:
                data.define_units({i: unit})
            except Exception as error:  # noqa: BLE001 — pint's parse
                # failures span UndefinedUnitError (an AttributeError),
                # DefinitionSyntaxError and ValueError; the answer is the
                # same whichever it is: the unit was not declared
                reason = f'could not be declared ({error})'
        if reason is not None:
            refused.append((i, unit, reason))
    return refused
