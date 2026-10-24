"""A random specification written by slope: the level a slope reaches,
each segment's area, and the rms accumulated from the low end.

A random specification arrives as a breakpoint table, and many tables
do not print a level at every breakpoint. They give a level, a slope in
decibels per octave and the next frequency, and leave the level there
to the reader. Between two breakpoints the specification is a straight
line on log-log axes, a power law W = C f**n, and a decibel on a
density is ten times the logarithm of a ratio, so a slope of N dB per
octave is an exponent

    n = N / (10 log10 2) = N / 3.0103

and the level it reaches at f2 from (f1, W1) is W1 (f2 / f1)**n. Plus
3 dB per octave is an exponent of 0.9966, minus 6 is -1.993; the
documents write 3 where they mean 3.01, and the difference in a level
is a third of a percent per octave. `OCTAVE_DB` is the exact 3.0103,
for a slope meant as an exponent of exactly one.

The rms of the specification is the square root of the area under it,
and a breakpoint table makes the area a sum of segments, each with a
closed form (`core.compliance.log_log_areas`, which every rms here is
computed by). For a segment from (f1, W1) to (f2, W2) with exponent n,

    W1 f1 ((f2/f1)**(n+1) - 1) / (n + 1)      for every n but -1
    W1 f1 ln(f2/f1)                           when n is -1

The second line is the special case the first divides by zero at: an
exponent of minus one is minus 3.0103 dB per octave (minus 10 dB per
decade), common at the top of a profile. A segment typed at minus 3 dB
per octave is an exponent of -0.9966 and takes the first line, which
is continuous with the second.

The areas say where the rms comes from, which the plot does not: on
log axes a slope at the top of the band looks short, and on linear
ones it is often the largest area of all, because a hertz at 1,500 Hz
counts as much as a hertz at 100 Hz. The running rms, the root of the
area from the lowest breakpoint up to each frequency, shows the same
thing as a curve: it climbs where the area is. Averaging a sloped
segment's two end levels and multiplying by its width (the trapezoid)
is right only at an exponent of one, and is the error a hand check
that disagrees with these numbers usually holds.

`level_at_slope`, `segment_slopes` and `levels_from_slopes` convert
between the two ways a table is written. `levels_with_slope` and
`levels_holding_slopes` are the two edits a table written by slope
takes, each holding every slope it was not asked to change: they are
what `core.author.SpecificationDraft.with_slope` and `with_level` do
to each channel. `segment_areas`, `running_rms` and `read_breakpoints`
read a table, whichever way it was written. Every function takes
levels in any unit of density and returns areas in that unit times
hertz and rms in its square root; nothing here converts units.

References
----------
The slope in decibels per octave as a power-law exponent, the level it
reaches, and the closed-form area under each segment, with its
logarithmic case at an exponent of minus one, are standard results of
random vibration practice; these are where they are set out, the
first with a worked example, which `tests/test_breakpoint_slopes.py`
repeats, and the second with a written spectrum whose printed overall
level they reproduce.

1. Irvine, T. (2000). ["Integration of the Power Spectral Density
   Function"](https://www.vibrationdata.com/tutorials2/psdinteg.pdf),
   Revision B. The area under a segment that is a straight line
   on log-log axes, in its two cases (its equation 11), the overall
   rms as the root of the summed areas, and, in its Appendix A, the
   level a slope in decibels per octave reaches over any frequency
   ratio (equation A-12).
2. GSFC-STD-7000B (2021), *General Environmental Verification
   Standard (GEVS)*. Table 2.4-4: a workmanship random vibration
   spectrum written as levels joined by slopes of plus and minus 3 dB
   per octave, with its overall level printed as 6.8 grms, which these
   closed forms give as 6.78.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from .compliance import log_log_areas

#: the slope, in decibels per octave, of a density that goes as the
#: frequency to the first power: ten log ten of two, 3.0103. A slope
#: typed as plus or minus this is an exponent of exactly plus or minus
#: one, and minus it takes the logarithmic area exactly.
OCTAVE_DB: float = 10.0 * float(np.log10(2.0))


def slope_exponent(db_per_octave: ArrayLike) -> np.ndarray:
    """The power-law exponent of a slope in decibels per octave,
    `db_per_octave / OCTAVE_DB`: 3 dB per octave is 0.9966."""
    return np.asarray(db_per_octave, dtype=float) / OCTAVE_DB


def slope_db(exponent: ArrayLike) -> np.ndarray:
    """The slope in decibels per octave of a power-law exponent,
    `exponent * OCTAVE_DB`: the inverse of `slope_exponent`."""
    return np.asarray(exponent, dtype=float) * OCTAVE_DB


def level_at_slope(frequency: float, level: float, next_frequency: float,
                   db_per_octave: float) -> float:
    """The level a slope reaches: from `level` at `frequency`, at
    `db_per_octave`, the level at `next_frequency`.

    This is the unprinted level of a breakpoint table written as a
    level, a slope and the next frequency. The next frequency may lie
    below the first, in which case the slope is followed downward.
    Raises `ValueError` for a frequency that is not positive or a level
    that is not positive and finite.
    """
    frequency, next_frequency = float(frequency), float(next_frequency)
    level = float(level)
    if not (frequency > 0 and next_frequency > 0):
        raise ValueError('a breakpoint frequency is positive')
    if not (np.isfinite(level) and level > 0):
        raise ValueError('a level is positive and finite')
    n = float(slope_exponent(db_per_octave))
    return level * (next_frequency / frequency) ** n


def _table(frequencies: ArrayLike,
           levels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The table as float arrays, refused when no specification could be
    written from it: fewer than two breakpoints, frequencies that are
    not positive and increasing, levels that are not positive and
    finite, or the two of different lengths."""
    f = np.asarray(frequencies, dtype=float)
    w = np.asarray(levels, dtype=float)
    if f.ndim != 1 or f.shape != w.shape:
        raise ValueError(f'{w.size} levels for {f.size} breakpoints')
    if f.size < 2:
        raise ValueError('a specification needs at least two breakpoints')
    if not np.all(np.isfinite(f) & (f > 0)):
        raise ValueError('breakpoint frequencies must be positive')
    if not np.all(np.diff(f) > 0):
        raise ValueError('breakpoint frequencies must increase')
    if not np.all(np.isfinite(w) & (w > 0)):
        raise ValueError('a level is positive and finite at every breakpoint')
    return f, w


def segment_slopes(frequencies: ArrayLike, levels: ArrayLike) -> np.ndarray:
    """Each segment's slope in decibels per octave, one fewer than the
    breakpoints: ten log ten of the level ratio over log two of the
    frequency ratio. A flat segment is 0. Raises `ValueError` for a
    table no specification could be written from."""
    f, w = _table(frequencies, levels)
    return 10.0 * np.log10(w[1:] / w[:-1]) / np.log2(f[1:] / f[:-1])


def levels_from_slopes(frequencies: ArrayLike, first_level: float,
                       slopes: ArrayLike) -> np.ndarray:
    """The level at every breakpoint of a table written by slope: the
    first breakpoint's level, and a slope in decibels per octave into
    each breakpoint after it.

    The levels are accumulated in logarithms, so a long table does not
    gather rounding from a running product. Raises `ValueError` when
    there is not one slope per segment, or for frequencies or a level
    no specification could hold.
    """
    f = np.asarray(frequencies, dtype=float)
    s = np.asarray(slopes, dtype=float)
    if s.ndim != 1 or s.size != f.size - 1:
        raise ValueError(f'{s.size} slopes for {max(f.size - 1, 0)} segments')
    if not np.all(np.isfinite(s)):
        raise ValueError('a slope is a finite number of decibels per octave')
    first = float(first_level)
    _table(f, np.full(f.shape, first))           # the frequencies and the level
    steps = slope_exponent(s) * np.log(f[1:] / f[:-1])
    return first * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))


def levels_with_slope(frequencies: ArrayLike, levels: ArrayLike,
                      segment: int, db_per_octave: float) -> np.ndarray:
    """The levels after one segment's slope is typed: the segment's end
    level is the one the slope reaches from its start, and every segment
    after it keeps its slope, so the levels past it move by the same
    factor. The levels before the segment are unchanged.

    Raises `ValueError` for a segment that is not one of the table's.
    """
    f, w = _table(frequencies, levels)
    segment = int(segment)
    if not 0 <= segment < f.size - 1:
        raise ValueError(f'the table has {f.size - 1} segments; no segment '
                         f'{segment + 1}')
    db_per_octave = float(db_per_octave)
    if not np.isfinite(db_per_octave):
        raise ValueError('a slope is a finite number of decibels per octave')
    end = level_at_slope(f[segment], w[segment], f[segment + 1], db_per_octave)
    # the tail scaled rather than rebuilt from the first level, so the
    # levels it does not move keep every digit they had
    out = w.copy()
    out[segment + 1:] = w[segment + 1:] * (end / w[segment + 1])
    out[segment + 1] = end
    return out


def levels_holding_slopes(frequencies: ArrayLike, levels: ArrayLike,
                          index: int, level: float) -> np.ndarray:
    """The levels after one breakpoint's level is typed in a table
    written by slope: the slope into it changes, and every slope after
    it is held, so the levels past it move by the same factor. Typed at
    the first breakpoint, the whole table moves with it.

    Raises `ValueError` for a breakpoint that is not one of the table's
    or a level that is not positive and finite.
    """
    f, w = _table(frequencies, levels)
    index = int(index)
    if not 0 <= index < f.size:
        raise ValueError(f'the table has {f.size} breakpoints; no '
                         f'breakpoint {index + 1}')
    level = float(level)
    if not (np.isfinite(level) and level > 0):
        raise ValueError('a level is positive and finite')
    out = w.copy()
    out[index:] = w[index:] * (level / w[index])
    out[index] = level                           # exactly as typed
    return out


def segment_areas(frequencies: ArrayLike, levels: ArrayLike) -> np.ndarray:
    """The area under each segment of the table, one fewer than the
    breakpoints, by the closed form of `core.compliance.log_log_areas`
    (the logarithmic case at an exponent of minus one included). The
    unit is the levels' times hertz: g^2 for levels in g^2/Hz. Raises
    `ValueError` for a table no specification could be written from."""
    f, w = _table(frequencies, levels)
    return log_log_areas(f, w, f[:-1], f[1:])


def running_rms(frequencies: ArrayLike, levels: ArrayLike,
                at: ArrayLike | None = None) -> np.ndarray:
    """The rms accumulated from the lowest breakpoint up to each of `at`:
    the square root of the area under the table between its first
    breakpoint and that frequency.

    With `at` omitted, the running rms at every breakpoint, the first
    of which is zero and the last the table's rms. A frequency below
    the first breakpoint reads zero, and one above the last reads the
    whole table's rms, since outside the table nothing is specified.
    """
    f, w = _table(frequencies, levels)
    at = f if at is None else np.atleast_1d(np.asarray(at, dtype=float))
    area = log_log_areas(f, w, np.full(at.shape, f[0]), at)
    return np.sqrt(np.maximum(area, 0.0))


class BreakpointReading(NamedTuple):
    """A breakpoint table read: its slopes, its areas and its rms.

    Attributes:
        frequencies: The breakpoints, in Hz.
        levels: The level at each breakpoint, in the table's unit of
            density.
        slopes: Each segment's slope in decibels per octave.
        areas: Each segment's area, in the levels' unit times hertz.
        running: The rms accumulated from the first breakpoint to the
            end of each segment, one per segment; the last is `rms`.
    """

    frequencies: np.ndarray
    levels: np.ndarray
    slopes: np.ndarray
    areas: np.ndarray
    running: np.ndarray

    @property
    def total_area(self) -> float:
        """The whole table's area, the sum of the segments'."""
        return float(np.sum(self.areas))

    @property
    def rms(self) -> float:
        """The table's rms, the square root of its whole area."""
        return float(np.sqrt(self.total_area))

    @property
    def shares(self) -> np.ndarray:
        """Each segment's share of the whole area, from 0 to 1: where
        the rms comes from."""
        total = self.total_area
        return self.areas / total if total > 0 else np.zeros_like(self.areas)


def read_breakpoints(frequencies: ArrayLike,
                     levels: ArrayLike) -> BreakpointReading:
    """Read a breakpoint table: each segment's slope in decibels per
    octave, its area by the closed form, and the rms accumulated from
    the low end at the end of each segment.

    Parameters
    ----------
    frequencies : array_like
        The breakpoints in Hz, positive and increasing, at least two.
    levels : array_like
        The level at each breakpoint, positive and finite, in any unit
        of density. A table written by slope has its levels computed
        first (`levels_from_slopes`).

    Returns
    -------
    BreakpointReading
        The slopes, the areas in the levels' unit times hertz, and the
        running rms in its square root.

    Raises
    ------
    ValueError
        For a table no specification could be written from.
    """
    f, w = _table(frequencies, levels)
    areas = segment_areas(f, w)
    return BreakpointReading(f, w, segment_slopes(f, w), areas,
                             np.sqrt(np.cumsum(areas)))
