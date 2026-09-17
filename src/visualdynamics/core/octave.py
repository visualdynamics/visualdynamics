"""Proportional-band spectra: a PSD resampled onto octave bands.

A narrowband PSD answers "how much power per hertz, at this hertz". A
proportional-band one answers "how much power in this band", where the
bands get wider as the frequency rises — which is how a structure's
response is usually specified, and how an ear hears.

The bands are the base-ten system of ANSI S1.11 / IEC 61260, which is
what sdynpy uses and therefore what a result has to agree with:

- one octave is a factor of ten to the three tenths, not a factor of
  two. For nth-octave the band ratio is `10 ** (3 / (10 n))` — at a
  sixth of an octave that is 1.122018, where a base-two reading would
  give 1.122462. They differ in the fourth digit, which is enough to
  put every band edge in a slightly different place.
- the grid is **absolute**. It does not depend on the frequency range
  asked for, or on any specification: the range only chooses which
  bands of the one fixed grid are returned. 1000 Hz anchors it.
- and which of edges or centers lands on `10 ** (3 k / (10 n))`
  depends on whether `n` is odd or even. For **odd** fractions — whole
  octaves, thirds — the band *centers* sit on the grid; for **even**
  ones — sixths, twelfths — the *edges* do, and the centers fall half
  a step between. Getting this backwards puts every band half a step
  out, which is a real disagreement and not a rounding one.
- bands tile — each one's upper edge is the next one's lower — and a
  band's center is the geometric mean of its edges.

Converting a spectrum is an integration, not a resampling. The value in
a band is the mean-square content of that band divided by its width, so
the area under the spectrum is unchanged and the RMS it carries is the
RMS it carried before. Reading the narrowband curve at each band center
instead would throw away everything between the centers, and would not
conserve anything.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

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
    """How much wider each band is than the one below it."""
    per_octave = int(per_octave)
    if per_octave < 1:
        raise ValueError('there is at least one band to an octave')
    return 10.0 ** (DECADE_FRACTION / per_octave)


def edges(low: float, high: float,
          per_octave: int = PER_OCTAVE) -> np.ndarray:
    """The edges of every band overlapping `low` to `high`.

    One more edge than there are bands, since they tile. Snapped to the
    fixed grid rather than started at `low`: two spectra covering
    different ranges land on the same bands, which is the whole point
    of a standard grid and the reason two runs can be compared at all.
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
    """(centers, widths, edges) for every band overlapping the range.

    The center is the geometric mean of the band's own edges — the
    arithmetic mean would sit above it, and on a log axis a band would
    then be drawn off-center from the number naming it.
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


def bin_bounds(centers: ArrayLike, widths: ArrayLike | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) of the bin each line stands for.

    With no widths the bins are the midpoints between neighbors, which
    is exact for evenly spaced FFT lines. With them the edges follow
    from `c = sqrt(l u)` and `w = u - l`:

        u = (w + sqrt(w^2 + 4 c^2)) / 2,   l = u - w

    the positive root, which is where a proportional band's edges
    actually are. Taking `c +/- w/2` instead would be assuming the
    center is the arithmetic mean of the edges, and it is not.
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
    """`values` integrated onto the bands `bounds` describes.

    Each line stands for its own bin, flat across it, which is what a
    discrete density is. A band's content is the part of every bin that
    falls inside it, and the band's value is that content over the
    band's width — so the area under the spectrum survives and a band
    straddling the end of the data is credited only for the part of it
    that has data underneath.

    Complex values pass through as complex: the cross terms of a CPSD
    average over a band the same way the diagonal does.
    """
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
