"""Truncating a time history to a start and stop time.

A recording rarely starts and stops where the test does: there is a
lead-in before the excitation reaches level, a tail after it stops,
and sometimes a false start at the front of the file. A `Truncation`
gives the part to keep as a start and a stop in seconds, and Truncate
Data makes a record that holds only that part, with every channel cut
at the same two times so the channels stay one record.

The time values are kept. A truncated record's samples keep their
measured times rather than being shifted to start at zero: the sample
recorded at t = 1.25 s is still at 1.25 s, so events line up with the
original recording, with the setup log, and with any untruncated
channels beside them. Shifting to zero would make a second time base
to reconcile.

Averaging frames and shock windows are not carried over.
`core.filters` carries them onto what it makes because its abscissa is
unchanged, so they still refer to the same samples. Truncation changes
the abscissa, and a frame or a window could refer to samples that no
longer exist, so the truncated record has neither; find the shocks and
set the averaging again on the truncated data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory


@dataclass(frozen=True)
class Truncation:
    """The part of a record to keep: start and stop, in seconds.

    Both times are on the record's own time axis, which may not begin
    at zero (a truncated record keeps its measured times), so a second
    truncation of an already truncated record uses the times shown for
    it. Both must be finite and `start` must be less than `stop`, or it
    raises ValueError.

    Frozen like `Averaging` and `Filtering`, for the same reason: the
    settings are stored with the history, and the check for out-of-date
    results compares their fields, which would be unreliable if the
    settings could change in place.
    """

    start: float
    stop: float

    def __post_init__(self) -> None:
        start, stop = float(self.start), float(self.stop)
        if not np.isfinite(start) or not np.isfinite(stop):
            raise ValueError('a truncation needs finite start and stop '
                             'times')
        if not start < stop:
            raise ValueError(
                f'a truncation runs forward: start ({start:g} s) must '
                f'sit before stop ({stop:g} s)')
        object.__setattr__(self, 'start', start)
        object.__setattr__(self, 'stop', stop)

    def describe(self) -> str:
        """The truncation in words, for example 'keeping 0.5 to 2 s'.

        The status line and the out-of-date message both use this
        function, so they describe it the same way."""
        return f'keeping {self.start:g} to {self.stop:g} s'


def truncate(data: TimeHistory, truncation: Truncation) -> TimeHistory:
    """A new record holding the samples between the truncation's start
    and stop, inclusive at both ends, with every channel cut at the
    same two times.

    A sample is kept when its time on the abscissa is inside the span,
    so an unevenly sampled record is cut correctly too. The records'
    DOFs, quantities, units and comments are copied unchanged. Raises
    ValueError for a span holding fewer than two samples, because one
    sample has no sample rate and nothing later can be computed from
    it.
    """
    from .data import TimeHistory

    abscissa = np.asarray(data.abscissa, dtype=float)
    keep = (abscissa >= truncation.start) & (abscissa <= truncation.stop)
    if int(keep.sum()) < 2:
        raise ValueError(
            f'the span {truncation.start:g} to {truncation.stop:g} s '
            'holds fewer than two samples — nothing downstream can '
            'read a record that short')
    return TimeHistory(abscissa[keep], np.asarray(data.ordinate)[..., keep],
                       **data.record_fields())
