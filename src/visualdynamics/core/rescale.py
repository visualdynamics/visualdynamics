"""Rescaling a time history: some or all of its channels times a factor.

The usual reason is a channel recorded with the wrong sensitivity. The
acquisition divides volts by the sensitivity it was given, so a record
taken with `recorded` mV per unit where the sensor is `correct` mV per
unit is off by `recorded / correct` (`sensitivity_factor`). A gain
entered on the wrong channel, or a sign, is the same operation with the
factor typed directly.

The result keeps the record's clock, its channels and what is set on it
(the averaging and shock windows, the filter and truncation settings,
the channel roles), so everything read or computed from the record
reads the same way from the rescaled one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def sensitivity_factor(recorded: float, correct: float) -> float:
    """The factor that corrects data recorded with sensitivity `recorded`
    to what sensitivity `correct` gives, both in the same units (mV per
    engineering unit, say).

    Engineering units are volts over the sensitivity, so the factor is
    `recorded / correct`: data taken with 100 mV/g on a 10 mV/g sensor
    reads a tenth of the true level and is multiplied by 10.

    Raises
    ------
    ValueError
        When either sensitivity is not a finite number above zero.
    """
    for name, value in (('recorded', recorded), ('correct', correct)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f'the {name} sensitivity has to be above zero, '
                             f'not {value:g}')
    return float(recorded) / float(correct)


def chosen_records(data: Any, records: Sequence[int] | None) -> list[int]:
    """The records a rescale applies to, sorted: all of them for None.

    Raises
    ------
    ValueError
        For an empty list or a record the history does not have.
    """
    if records is None:
        return list(range(data.num_records))
    chosen = sorted({int(i) for i in records})
    if not chosen:
        raise ValueError('no records to rescale')
    outside = [i for i in chosen if not 0 <= i < data.num_records]
    if outside:
        raise ValueError(f'record {outside[0]} is not in a history of '
                         f'{data.num_records} records')
    return chosen


def rescaled(data: Any, factor: float,
             records: Sequence[int] | None = None) -> Any:
    """The time history with `records` (all of them for None) multiplied
    by `factor`, the rest unchanged.

    Raises
    ------
    ValueError
        When the factor is zero or not finite, or a record is not in the
        history.
    """
    from .data import TimeHistory
    from .filters import carrying_marks

    factor = float(factor)
    if not (math.isfinite(factor) and factor != 0.0):
        raise ValueError(f'the factor has to be a finite number other than '
                         f'zero, not {factor:g}')
    chosen = chosen_records(data, records)
    rows = np.array(data.ordinate, dtype=float, copy=True)
    rows[chosen] *= factor
    result = TimeHistory(data.abscissa, rows, **data.record_fields())
    carrying_marks(data, result)
    # the same channels on the same clock: what is set on the record
    # still means what it meant
    result.roles = data.roles
    result.filtering = data.filtering
    result.truncation = data.truncation
    result.sine_extraction = data.sine_extraction
    return result
