"""Truncating a record to the stretch that matters.

A recording rarely starts and stops where the test does: there is a
lead-in before the excitation reaches level, a tail after it stops,
sometimes a false start at the front of the file. The truncation names
the stretch to keep — a start and a stop, in seconds — and Truncate
Data makes the record that holds only that stretch, every channel cut
at the same two instants so the set stays one record.

**The clock is kept.** A truncated record's samples keep their
measured instants rather than being re-zeroed: the sample recorded at
t = 1.25 s still says 1.25 s, so events line up with the recording
they came from, with the setup log, and with any untruncated channel
set beside them. A re-zeroed clock would be a second time base to
reconcile, silently.

**No marks are carried.** `core.filters` carries the averaging frames
and shock windows onto what it makes because the abscissa is the same
and the marks still say what they said. Here the abscissa changed —
that is the whole point — and a frame or a window can refer to samples
that no longer exist, so the honest answer is a clean record: re-find
the shocks, re-frame the averaging, on the data that is actually
there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory


@dataclass(frozen=True)
class Truncation:
    """The stretch a record is cut to: start and stop, in seconds.

    On the record's own clock — which may not begin at zero, since a
    truncated record keeps its measured instants — so a second cut of
    an already-cut record means what it says.

    Frozen like `Averaging` and `Filtering`, and for the same reason:
    the settings ride the history, the staleness fingerprint is their
    fields, and a mutable setting would be a fingerprint that lies.
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
        """The cut in words — 'keeping 0.5 to 2 s'. One implementation:
        the status line and the staleness story both say it this way."""
        return f'keeping {self.start:g} to {self.stop:g} s'


def truncate(data: TimeHistory, truncation: Truncation) -> TimeHistory:
    """The record between the truncation's start and stop, inclusive
    at both ends, every channel cut at the same two instants.

    A sample is kept when its own instant lies inside the span — read
    off the abscissa, so an unevenly sampled record cuts correctly
    too. Refuses a span holding fewer than two samples: one sample has
    no sample rate, and no downstream reading can be made from it.
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
