"""What a record says about the channel that recorded it: missing
samples and a channel that never moved.

Two things a recorded channel can carry that no spectrum or statistic
will report on its own. A missing sample (stored as NaN) is a gap: a
recorder that dropped a value, a column shorter than its neighbors, a
cell left empty in an export. A channel whose every sample is the same
number is dead: unplugged, unpowered, or never wired to anything. Both
import without complaint, because both are what the file holds, and
both are worth saying before anything is computed from the record.

`channel_health` reads every record of a time history (or any data
array) and returns one `ChannelHealth` per record that has something to
say. The delimited-text reader turns each into an import note, and its
preview marks the same records, so the note and the picture cannot
disagree.

A dead channel here is one whose present samples are all exactly equal.
A live channel that has failed usually still shows a few counts of
noise, and a threshold that caught that would also catch a quiet
channel that is working; the exact test says only what is certain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Written for the delimited-text import (2026-10-23), where a reviewer
# asked for missing samples and dead channels to be handled with notes
# rather than silently imported or silently dropped.


@dataclass(frozen=True)
class ChannelHealth:
    """What one record says about its channel.

    Attributes
    ----------
    record : int
        The record's index in the data array.
    samples : int
        How many samples the record holds.
    missing : int
        How many of them are missing (NaN or infinite).
    first_missing : int or None
        The index of the first missing sample, or None when none is.
    constant : float or None
        The one value every present sample holds, for a dead channel;
        None for a channel that varies, or one with no present samples.
    """

    record: int
    samples: int
    missing: int
    first_missing: int | None
    constant: float | None

    @property
    def empty(self) -> bool:
        """Whether every sample is missing."""
        return self.missing == self.samples

    @property
    def dead(self) -> bool:
        """Whether the present samples are all one value."""
        return self.constant is not None

    def describe(self, label: str, abscissa: Any = None,
                 unit: str | None = None) -> str:
        """The finding in words, naming the channel `label`.

        With `abscissa` (the record's time in seconds) the first missing
        sample is placed in time as well as by index; with `unit` a dead
        channel's value carries it.

        Parameters
        ----------
        label : str
            What to call the channel.
        abscissa : array-like, optional
            The time of each sample, in seconds.
        unit : str, optional
            The unit the constant value is written in.

        Returns
        -------
        str
            One or two sentences.
        """
        if self.empty:
            return f'{label} holds no values: all {self.samples} samples are missing'
        said = []
        if self.missing:
            where = f'sample {self.first_missing + 1}'
            if abscissa is not None:
                where += f', {float(np.asarray(abscissa)[self.first_missing]):g} s'
            said.append(f'{label} has {self.missing} missing sample'
                        f'{"s" * (self.missing != 1)} of {self.samples} '
                        f'(the first at {where}), kept as NaN')
        if self.dead:
            value = f'{self.constant:g}' + (f' {unit}' if unit else '')
            said.append(f'{label} is constant at {value} over every present '
                        'sample: a dead or unconnected channel')
        return '; '.join(said)


def channel_health(data: Any) -> list[ChannelHealth]:
    """The records of `data` that have missing samples or never vary.

        for finding in channel_health(history):
            print(finding.describe(history.record_label(finding.record)))

    Only records with something to say are returned, in record order, so
    an empty list is a clean bill. A record is listed when any sample is
    NaN or infinite, or when every present sample holds the same value;
    a record of one sample is not called dead, since one sample cannot
    vary.

    Parameters
    ----------
    data : DataArray or array-like
        A time history (or any data array), or its ordinate as a
        (records, samples) array.

    Returns
    -------
    list of ChannelHealth
        One per record with a finding.
    """
    values = np.real(np.atleast_2d(np.asarray(getattr(data, 'ordinate', data))))
    out = []
    for i, row in enumerate(values):
        bad = ~np.isfinite(row)
        missing = int(bad.sum())
        present = row[~bad]
        constant = None
        if present.size > 1 and float(present.max()) == float(present.min()):
            constant = float(present[0])
        if not missing and constant is None:
            continue
        first = int(np.flatnonzero(bad)[0]) if missing else None
        out.append(ChannelHealth(i, int(row.size), missing, first, constant))
    return out
