"""Q read from the half-power width of one peak.

A resonance on a magnitude curve (an FRF magnitude, a transmissibility,
a tracked sine level) has two numbers a test engineer reads off it: the
frequency of its peak and how sharp it is. The sharpness is Q, the peak
frequency over the half-power width, the width between the two
frequencies where the curve has fallen to 1/sqrt(2) of its peak (0.707,
3 dB down; half power because power goes with amplitude squared). The
damping ratio is 1/(2Q). For one mode well clear of its neighbors this
agrees with the mode's own damping to within a few times the damping
squared: read finely off a one-mass accelerance, Q comes out 1 % low at
a Q of 10, 0.25 % at 20 and 0.04 % at 50 (a receptance or a
transmissibility half that).

`read_half_power` takes the curve and a frequency near the peak and
returns a `HalfPower`: the peak's frequency and height, the two
crossings, the width, Q, the damping ratio and how many lines span the
width. The reading is the one the eye makes off a plot, put in numbers,
and it is not a modal fit: it reads where the curve is, not the
resonator that made it. Two modes closer than their widths read as one
wide peak, and a swept level read too fast reads wide and low; both are
properties of the curve, and the reading reports the curve faithfully.

Three things are said rather than hidden:

- A crossing that is not inside the curve. The curve stops before it
  falls to 0.707 of the peak on that side (the peak is near an end of
  the measured band, or rides a shoulder). The crossing is None, so are
  the width, Q and damping, and a `HalfPowerWarning` says which side.
  Borrowing the other side's width, as the modal fit's seed does, is a
  guess about a curve that was not measured; a seed may guess, a reading
  may not.
- Too few lines across the peak. A peak sampled at a line spacing near
  its own width is read low (the tallest line misses the top) and wide
  (the crossings are interpolated across straight segments of a curved
  flank), and so its Q comes out wrong, nearly always low. The reading
  is still returned, since it is what the curve says, but with a
  `HalfPowerWarning` when fewer than `MIN_LINES` lines span the width.
- A peak so much narrower than its lines that the top, placed between
  them, stands more than 3 dB over every line. No crossing can be
  placed at all, so neither is, and the warning says why.

References
----------
The half-power reading is a standard method; this is where it is
described. The parabola through the logarithm of the tallest line and
its two neighbors, which places the top between lines, is worked
through where the top is read.

1. Piersol, A. G., & Paez, T. L. (Eds.) (2010). *Harris' Shock and
   Vibration Handbook*, 6th ed. McGraw-Hill. The half-power bandwidth
   of a resonance: its Q as the peak frequency over the width between
   the points where the response has fallen to 1/sqrt(2) of its peak,
   and the damping ratio as 1/(2Q).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

#: the fewest lines across the half-power width that a reading is
#: trusted at without a warning. Measured on a one-mass accelerance and
#: transmissibility at Q 20 and Q 50, over twenty placements of the
#: lines against the peak: at 4 lines across the width the Q read is at
#: most 3 % low and the height under 1 %; at 2 lines Q is up to 14 % low,
#: and at 1 line up to 44 % low with the height 22 % low. From two
#: lines up it was low at every placement; at one and a half lines one
#: placement in forty read 2 % high (`tests/test_half_power.py` pins
#: the two-line and four-line figures). Without the
#: parabola through the top (`_top`) the same 4 lines read up to 7 %
#: low, which is why the top is placed between lines at all.
MIN_LINES = 4

#: 1/sqrt(2): the half-power level as a share of the peak
HALF_POWER = 1.0 / np.sqrt(2.0)


def _figures(value: float, figures: int) -> str:
    """`value` to `figures` significant figures, never in exponent form:
    '180.1', '10240', '0.0125'. A reading is copied into reports and
    spreadsheets, where 1.024e+04 Hz is a number to re-type."""
    if not value or not math.isfinite(value):
        return f'{value:g}'
    decimals = max(0, figures - 1 - math.floor(math.log10(abs(value))))
    text = f'{value:.{decimals}f}'
    return text.rstrip('0').rstrip('.') if '.' in text else text


def _lines(lines: float) -> str:
    """A count of lines: two figures below ten, whole lines past it."""
    return f'{lines:.2g}' if lines < 10.0 else f'{lines:.0f}'


class HalfPowerWarning(UserWarning):
    """Warned when a half-power reading is missing a crossing or rests
    on too few lines. The reading is returned either way; the warning
    says what it cannot be trusted for, and `HalfPower.notes` carries
    the same words for a caller that shows them rather than warns."""


@dataclass(frozen=True)
class HalfPower:
    """One peak read by its half-power width: what `read_half_power`
    returns.

    Frequencies are in Hz and `height` is in the curve's own units.
    `low` and `high` are None where the curve ends before falling to the
    half-power level on that side, and then `width`, `q` and `damping`
    are None too. `lines` is the width over the line spacing at the
    peak: how many lines the reading rests on. `notes` holds the
    warnings the reading raised, as text, empty when it raised none.
    `min_lines` is the count `resolved` is judged against: the
    `min_lines` the reading was taken with.
    """

    frequency: float
    height: float
    low: float | None
    high: float | None
    spacing: float
    notes: tuple[str, ...] = ()
    min_lines: float = MIN_LINES

    @property
    def level(self) -> float:
        """The half-power level: 0.707 of the height."""
        return self.height * HALF_POWER

    @property
    def width(self) -> float | None:
        """The half-power width in Hz, `high` minus `low`."""
        if self.low is None or self.high is None:
            return None
        return self.high - self.low

    @property
    def q(self) -> float | None:
        """Q: the peak frequency over the half-power width."""
        width = self.width
        return self.frequency / width if width else None

    @property
    def damping(self) -> float | None:
        """The damping ratio, a fraction of critical: 1/(2Q)."""
        q = self.q
        return 0.5 / q if q else None

    @property
    def lines(self) -> float | None:
        """How many lines span the half-power width."""
        width = self.width
        return width / self.spacing if width is not None and self.spacing \
            else None

    @property
    def resolved(self) -> bool:
        """True when both crossings were found and at least `min_lines`
        lines span the width."""
        # 2026-10-16: read the module's MIN_LINES until review found a
        # reading taken with min_lines=2 calling itself unresolved at
        # three lines, after warning nothing
        lines = self.lines
        return lines is not None and lines >= self.min_lines

    def describe(self) -> str:
        """The reading in one line, as the application shows it:
        'Q 19.9 at 179.9 Hz, half-power 175.3 to 184.3 Hz (9.03 Hz,
        14 lines)'. Numbers are given to four figures (the width and Q
        to three) and never in exponent form."""
        where = f'{_figures(self.frequency, 4)} Hz'
        if self.low is None and self.high is None:
            return f'peak at {where}: no half-power width can be read'
        if self.q is None:
            side = ('below' if self.low is None else 'above')
            return (f'peak at {where}: no half-power crossing {side} it '
                    'inside the curve')
        assert self.low is not None and self.high is not None
        assert self.width is not None and self.lines is not None
        return (f'Q {_figures(self.q, 3)} at {where}, half-power '
                f'{_figures(self.low, 4)} to {_figures(self.high, 4)} Hz '
                f'({_figures(self.width, 3)} Hz, {_lines(self.lines)} '
                'lines)')


def crossings(frequencies: Any, magnitude: Any, index: int,
              level: float) -> tuple[float | None, float | None]:
    """Where the curve first falls below `level` on each side of line
    `index`, as (low, high) in Hz, interpolated linearly between the
    last line at or above the level and the first below it.

    A side that never falls below the level inside the curve gives
    None. The level must not be above line `index` itself, which is the
    top the crossings are measured down from; a ValueError says so. This
    is the one place the package finds half-power crossings: the reading
    here and the modal fit's damping seed both call it.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    m = np.asarray(magnitude, dtype=np.float64)
    if level > m[index]:
        raise ValueError(f'a level of {level:g} is above the line it is '
                         f'measured down from ({m[index]:g})')

    # 2026-10-16: these were Python loops a line at a time, 6 ms on
    # 40 000 lines across a broad peak, and the fitting screen reads on
    # every tick of a drag; one comparison over the array is the same
    # rule at a small fraction of that
    def between(i: int, j: int) -> float:
        # m[i] >= level > m[j], so the step is never flat
        fraction = (m[i] - level) / (m[i] - m[j])
        return float(f[i] + fraction * (f[j] - f[i]))

    below = np.flatnonzero(m[:index] < level)
    above = np.flatnonzero(m[index + 1:] < level)
    low = between(int(below[-1]) + 1, int(below[-1])) if len(below) \
        else None
    high = (between(index + int(above[0]), index + int(above[0]) + 1)
            if len(above) else None)
    return low, high


def _climb(m: np.ndarray, index: int) -> int:
    """From line `index`, uphill to the top of the peak it sits on.

    The way is chosen at the first line, toward the taller neighbor
    (down in frequency on a tie), and the climb goes on that way while
    each next line is strictly taller. It never turns back: having
    stepped one way, the line behind is lower than the one stood on.
    """
    # 2026-10-16: a step at a time until review measured it among the
    # loops costing a drag tick; the run of rises is found in one pass
    last = len(m) - 1
    below = m[index - 1] if index > 0 else -np.inf
    above = m[index + 1] if index < last else -np.inf
    if below > m[index] and below >= above:
        falls = np.flatnonzero(~(m[:index][::-1] > m[1:index + 1][::-1]))
        return index - (int(falls[0]) if len(falls) else index)
    if above > m[index]:
        falls = np.flatnonzero(~(m[index + 1:] > m[index:-1]))
        return index + (int(falls[0]) if len(falls) else last - index)
    return index


def _top(f: np.ndarray, m: np.ndarray, i: int) -> tuple[float, float]:
    """The peak's frequency and height between lines: the vertex of the
    parabola through the logarithms of line `i` and its two neighbors.

    In the logarithm because a resonance's top is closer to a parabola
    there than in the magnitude, which the three points over-sharpen.
    The vertex is held between the neighbors. The fit is in local
    frequency, about line `i`, so uneven spacing (a tracked level is
    read evenly in time, not in frequency) is honored rather than
    assumed away.
    """
    if i == 0 or i == len(m) - 1 or np.any(m[i - 1:i + 2] <= 0.0):
        return float(f[i]), float(m[i])
    x = f[i - 1:i + 2] - f[i]
    y = np.log(m[i - 1:i + 2])
    a, b, c = np.polyfit(x, y, 2)
    if not a < 0.0:
        return float(f[i]), float(m[i])
    vertex = float(np.clip(-b / (2.0 * a), x[0], x[2]))
    return float(f[i] + vertex), float(np.exp(np.polyval((a, b, c), vertex)))


def read_half_power(frequencies: Any, magnitude: Any, near: float, *,
                    within: tuple[float, float] | None = None,
                    min_lines: float = MIN_LINES,
                    warn: bool = True, stacklevel: int = 1) -> HalfPower:
    """Read Q from the half-power width of the peak nearest `near`.

    The peak is the one the line nearest `near` sits on, found by
    climbing uphill from that line, so a frequency anywhere on a peak's
    flank reads that peak and not a taller one elsewhere. With `within`,
    a (low, high) band in Hz, the peak is instead the tallest line
    inside the band, which is the reading for a band that holds one
    mode. The top is placed between lines by a parabola through the
    logarithms of the tallest line and its two neighbors, and the two
    crossings of 0.707 of that height are interpolated linearly between
    lines. Q is the peak frequency over the width between the
    crossings, and the damping ratio is 1/(2Q).

    A `HalfPowerWarning` is warned, and its text kept in `notes`, when a
    crossing is not inside the curve and when fewer than `min_lines`
    lines span the width (see the module's notes on resolution). Pass
    `warn=False` to keep the notes without warning, as a display that
    reads on every mouse move does.

    Parameters
    ----------
    frequencies : array-like
        The curve's frequencies in Hz, ascending. They need not be
        evenly spaced.
    magnitude : array-like
        The curve, one value per frequency. Complex values are read by
        their magnitude.
    near : float
        A frequency on the peak to read, in Hz.
    within : tuple of float, optional
        A band (low, high) in Hz; the tallest line inside it is the
        peak, and `near` is not used to find it.
    min_lines : float, default `MIN_LINES`
        The fewest lines across the half-power width read without a
        warning.
    warn : bool, default True
        Warn the notes as `HalfPowerWarning` as well as keeping them.
    stacklevel : int, default 1
        Which line the warning names: 1 is the line that called this
        function, 2 its caller, as for `warnings.warn`. A function that
        reads on its caller's behalf passes 2, so the warning points at
        the caller's code and not its own.

    Returns
    -------
    HalfPower
        The peak's frequency and height, the two crossings, the line
        spacing at the peak, and any notes.

    Raises
    ------
    ValueError
        When the frequencies and the magnitude differ in length, hold
        fewer than three lines, are not ascending, or `within` holds no
        line of the curve.
    """
    f = np.asarray(frequencies, dtype=np.float64).ravel()
    m = np.abs(np.asarray(magnitude)).astype(np.float64).ravel()
    if len(f) != len(m):
        raise ValueError(f'{len(f)} frequencies for {len(m)} values')
    if len(f) < 3:
        raise ValueError('a peak needs at least three lines to be read')
    if not np.all(np.diff(f) > 0.0):
        raise ValueError('the frequencies must be ascending')
    # a line the source could not read (NaN) is no part of any peak
    m = np.where(np.isfinite(m), m, -np.inf)
    if within is not None:
        low, high = sorted((float(within[0]), float(within[1])))
        band = np.flatnonzero((f >= low) & (f <= high))
        if not len(band):
            raise ValueError(f'no line of the curve lies between {low:g} '
                             f'and {high:g} Hz')
        i = int(band[np.argmax(m[band])])
    else:
        i = _climb(m, int(np.argmin(np.abs(f - float(near)))))
    peak, height = _top(f, np.where(np.isfinite(m), m, 0.0), i)
    # the spacing the peak was sampled at: the mean of the two steps
    # about it, which is the one step on an even axis
    spacing = float((f[min(i + 1, len(f) - 1)] - f[max(i - 1, 0)])
                    / (min(i + 1, len(f) - 1) - max(i - 1, 0)))
    notes = []
    narrow = height * HALF_POWER > m[i]
    if narrow:
        # the top, placed between lines, stands more than 3 dB over the
        # tallest line: the half-power level is above every line, so no
        # crossing can be placed. Both crossings would land on the
        # tallest line, a width of zero and a Q of infinity.
        below = above = None
        notes.append(f'the peak at {_figures(peak, 4)} Hz is narrower '
                     f'than its lines, {_figures(spacing, 3)} Hz apart: '
                     'its half-power level is above every line, and no '
                     'width or Q is read')
    else:
        below, above = crossings(f, m, i, height * HALF_POWER)
    for crossing, word, end in ((below, 'below', f[0]),
                                (above, 'above', f[-1])):
        if crossing is None and not narrow:
            notes.append(f'the curve ends at {_figures(end, 4)} Hz before '
                         f'falling to half power {word} the peak at '
                         f'{_figures(peak, 4)} Hz; no width or Q is read')
    reading = HalfPower(peak, height, below, above, spacing,
                        min_lines=min_lines)
    lines = reading.lines
    if lines is not None and lines < min_lines:
        assert reading.width is not None
        notes.append(f'the half-power width of the peak at '
                     f'{_figures(peak, 4)} Hz is '
                     f'{_figures(reading.width, 3)} Hz, {_lines(lines)} lines '
                     f'of {_figures(spacing, 3)} Hz; with fewer than '
                     f'{min_lines:g} lines across it the top is missed and '
                     'the flanks are cut straight, and Q is biased, '
                     'usually low')
    if warn:
        for note in notes:
            warnings.warn(note, HalfPowerWarning, stacklevel=stacklevel + 1)
    return HalfPower(peak, height, below, above, spacing, tuple(notes),
                     min_lines)
