"""Proportional-band spectra: a PSD resampled onto octave bands.

A narrowband PSD gives the power per hertz at each frequency line. A
proportional-band PSD gives the power in each band, where the bands get
wider as the frequency increases. Responses are usually specified this
way, and it is also how the ear hears.

The bands are the base-ten system of ANSI S1.11 / IEC 61260:

- One octave is a factor of ten to the three tenths, which is close to
  two but not equal to it. For 1/n octave the band ratio is
  `10 ** (3 / (10 n))`. At a sixth of an octave that is 1.122018, where
  a base-two ratio would be 1.122462. The two differ in the fourth
  digit, which is enough to move every band edge.
- The grid is absolute. It does not depend on the frequency range
  requested or on any specification; the range only selects which bands
  of the one fixed grid are returned. 1000 Hz is on the grid.
- Whether the edges or the centers land on `10 ** (3 k / (10 n))`
  depends on whether `n` is odd or even. For odd fractions (whole
  octaves, thirds) the band centers are on the grid. For even fractions
  (sixths, twelfths) the edges are, and the centers fall half a step
  between. Getting this backwards moves every band by half a step,
  which is a real difference in the result and much larger than
  rounding.
- The bands tile: each band's upper edge is the next band's lower edge,
  and a band's center is the geometric mean of its edges.

Converting a spectrum is an integration. The value in a band is the
mean-square content of that band divided by its width, so the area under
the spectrum is unchanged and the RMS is the same before and after.
Reading the narrowband curve at each band center would ignore
everything between the centers and would not conserve the RMS.

References
----------
The bands are the ones these two standards define; they agree on every
band this module builds.

1. ANSI/ASA S1.11-2014, *Electroacoustics - Octave-band and
   Fractional-octave-band Filters - Part 1: Specifications*.
2. IEC 61260-1:2014, *Electroacoustics - Octave-band and
   fractional-octave-band filters - Part 1: Specifications*.
3. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. On the relation between a
   density and the power in a band, which is why a band takes the
   integral over its width rather than a reading at its center.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# The base-ten system is also the one sdynpy uses, and so the one a
# result here has to agree with when the two are compared.

#: one octave, as a power of ten. The base-ten system's defining
#: constant: an octave is 10**0.3, near enough to 2 for the names to
#: have stuck and far enough for the edges to differ.
DECADE_FRACTION = 3.0 / 10.0

#: bands to the octave when nobody says. A sixth is fine enough to
#: follow the shape of a response and coarse enough to be worth doing.
PER_OCTAVE = 6

#: the fractions offered in the app. Whole octaves through
#: twenty-fourths, which is the range anyone asks for in practice.
CHOICES = (1, 2, 3, 6, 12, 24)


def ratio(per_octave: int = PER_OCTAVE) -> float:
    """The ratio of each band's center to the center of the band below it.

    Raises `ValueError` when `per_octave` is less than one.
    """
    per_octave = int(per_octave)
    if per_octave < 1:
        raise ValueError('there is at least one band to an octave')
    return 10.0 ** (DECADE_FRACTION / per_octave)


def edges(low: float, high: float,
          per_octave: int = PER_OCTAVE) -> np.ndarray:
    """The edges of every band that overlaps `low` to `high`, in Hz.

    There is one more edge than there are bands, since the bands tile.
    The edges are taken from the fixed grid instead of starting at
    `low`, so two spectra covering different ranges land on the same
    bands and can be compared. Raises `ValueError` unless
    `0 < low < high`.
    """
    low, high = float(low), float(high)
    if not (low > 0.0 and high > low):
        raise ValueError(f'band range {low} to {high} is not a range')
    per_octave = int(per_octave)
    step = DECADE_FRACTION / per_octave
    # odd fractions put the centers on the grid, so their edges fall
    # half a step off it; even fractions put the edges on it
    offset = 0.5 if per_octave % 2 else 0.0
    # snapped before rounding: an edge that is already a grid point
    # comes back from ten-to-the-power and a logarithm a float's
    # breadth adrift, and flooring that adds a band on every re-banding
    below = np.log10(low) / step - offset
    above = np.log10(high) / step - offset
    first = int(np.floor(below + SNAP))
    last = int(np.ceil(above - SNAP))
    return 10.0 ** (step * (np.arange(first, last + 1) + offset))


def bands(low: float, high: float, per_octave: int = PER_OCTAVE
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centers, widths, edges) for every band that overlaps the range.

    The center is the geometric mean of the band's edges. The
    arithmetic mean would be higher, and on a log axis the band would
    then be drawn off center from the frequency that names it.
    """
    bounds = edges(low, high, per_octave)
    lower, upper = bounds[:-1], bounds[1:]
    return np.sqrt(lower * upper), upper - lower, bounds


#: how close to a whole step counts as being on it. Band edges come
#: back as powers of ten and go into a logarithm again; without a
#: tolerance, an edge that lands a float's breadth under its own grid
#: point is floored to the step below and the range grows a band every
#: time a spectrum is re-banded.
SNAP = 1e-9


def per_octave_of(centers: ArrayLike) -> int:
    """The number of bands per octave that a set of band centers is on.

    It is read from the ratio of the first two positive centers, so two
    bands are enough. Raises `ValueError` with fewer than two.
    """
    centers = np.asarray(centers, dtype=float)
    centers = centers[np.isfinite(centers) & (centers > 0.0)]
    if centers.size < 2:
        raise ValueError('a fraction needs at least two bands to be read from')
    return int(np.rint(DECADE_FRACTION / np.log10(centers[1] / centers[0])))


def bin_bounds(centers: ArrayLike, widths: ArrayLike | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) edges of the bin that each line represents.

    With no widths, the bin edges are the midpoints between neighboring
    lines, which is exact for evenly spaced FFT lines. With widths, the
    edges follow from `c = sqrt(l u)` and `w = u - l`:

        u = (w + sqrt(w^2 + 4 c^2)) / 2,   l = u - w

    using the positive root. Taking `c +/- w/2` would assume the center
    is the arithmetic mean of the edges, which is not true for a
    proportional band.
    """
    centers = np.asarray(centers, dtype=float)
    if widths is not None:
        widths = np.asarray(widths, dtype=float)
        upper = (widths + np.sqrt(widths * widths
                                  + 4.0 * centers * centers)) / 2.0
        return upper - widths, upper
    spacing = np.gradient(centers)
    return centers - spacing / 2.0, centers + spacing / 2.0


def resample(frequencies: ArrayLike, values: ArrayLike, bounds: ArrayLike,
             widths: ArrayLike | None = None,
             source: ArrayLike | None = None) -> np.ndarray:
    """`values` integrated onto the bands whose edges are `bounds`.

    Each line represents its own bin and is constant across it, which
    is what a discrete density is. A band's content is the part of each
    bin that falls inside the band, and the band's value is that
    content divided by the band's width, so the area under the spectrum
    is unchanged. `source` gives the (left, right) bin edges of the
    input lines; without it they are taken from `bin_bounds`, which is
    only exact for evenly spaced lines. `widths` overrides the band
    widths, which are otherwise the differences of `bounds`.

    Complex values stay complex: the cross terms of a CPSD are averaged
    over a band the same way the diagonal is.

    A band that the data only partly covers holds that content over its
    whole width, so it reads lower than its neighbors. The bands are
    fixed frequency bands and are never cut down to the part that has
    data. A band with no data under it at all is NaN. Raises
    `ValueError` with fewer than two lines.
    """
    # Bands are whole, never cut or read over part of themselves
    # (Brandon, 2026-09-19, twice).
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.atleast_2d(np.asarray(values))
    if frequencies.size < 2:
        raise ValueError('a spectrum needs at least two lines to be banded')

    # the bins the source actually has. Inferring them from the centers
    # is exact for evenly spaced lines and wrong for a spectrum that is
    # already banded — the guess runs a fraction of a percent wide
    # through the middle and six percent wide at the first band, which
    # smears content into its neighbors every time one is re-banded.
    left, right = source if source is not None else bin_bounds(frequencies)
    lower, upper = bounds[:-1], bounds[1:]

    # (bands, lines): how much of each line's bin lies in each band
    overlap = np.clip(
        np.minimum(right[None, :], upper[:, None])
        - np.maximum(left[None, :], lower[:, None]), 0.0, None)
    held = np.nan_to_num(values, nan=0.0)
    content = held @ overlap.T                      # (records, bands)
    widths = (upper - lower) if widths is None else np.asarray(widths)
    with np.errstate(divide='ignore', invalid='ignore'):
        banded = np.where(widths > 0.0, content / widths, 0.0)
    # a band with no data under it says nothing rather than nothing-much
    covered = overlap.sum(axis=1)
    return np.where(covered > 0.0, banded, np.nan)
