"""Peak-keeping decimation: fewer points, with every peak kept.

A record of a million samples is too many to draw point for point, and
keeping every n-th sample can skip the one large peak that was the
reason to look at the record. Peak-keeping decimation cuts the samples
into equal slices and keeps each slice's smallest and largest value at
their own abscissa positions, so no resonance, dropout or shock is lost.
It is the same method as pyqtgraph's `peak` downsampling, which the 2-D
plot uses.

The report and the 3-D waterfall both use this module. It works on all
of an object's records at once, because the waterfall passes it hundreds
of them. It needs only numpy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Vectorized over whole records because of the waterfall: as a per-bin
# Python loop the same work took 10.6 s on a 320-record million-sample
# history, and batched it is 1.3 s (both measured).


def peak_decimate(x: ArrayLike, y: ArrayLike,
                  budget: int) -> tuple[np.ndarray, np.ndarray]:
    """`x` and `y` thinned to at most `budget` points that keep every
    extreme.

    The samples are cut into `budget // 2` slices, and each slice keeps
    its smallest and largest value. `y` must be real: a caller with
    complex data takes the component it wants first, because the
    extremes of a complex value are not defined. Input at or under the
    budget is returned as given.

    NaN marks a gap (a specification is NaN outside its band) and is
    never kept as an extreme. A slice that is all NaN keeps its first
    point, so the gap is still drawn as a gap.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) <= budget:
        return x, y
    keep = _extreme_indices(y[np.newaxis], budget // 2)[0]
    return x[keep], y[keep]


def peak_decimate_rows(x: ArrayLike, rows: ArrayLike,
                       budget: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """`peak_decimate` for every record of one object at once.

    `rows` is a (records, samples) array on the shared abscissa `x`. The
    result is one (x, y) pair per record, and the records no longer share
    an abscissa: each keeps its own extremes at their own positions. The
    whole block is decimated in one vectorized pass.
    """
    x = np.asarray(x)
    rows = np.atleast_2d(np.asarray(rows))
    if rows.shape[1] <= budget:
        return [(x, rows[k]) for k in range(rows.shape[0])]
    kept = _extreme_indices(rows, budget // 2)
    return [(x[keep], rows[k][keep]) for k, keep in enumerate(kept)]


def shared_peak_indices(rows: ArrayLike, budget: int) -> np.ndarray:
    """The samples of one shared abscissa to keep, at most `budget` of
    them, so that every row keeps its extremes.

    `peak_decimate_rows` gives each row its own positions, which suits
    curves drawn on their own. Rows that share an abscissa on the page,
    a measurement and the limits shaded about it, must keep one set of
    positions between them, or a band is shaded between points that are
    not above each other. This keeps, for every slice of the samples,
    the smallest and largest value of every row, with the slices as fine
    as the budget allows, and the first and last sample so the extent is
    unchanged. Rows with fewer than `budget` samples keep them all.

    Parameters
    ----------
    rows : array_like
        (rows, samples) values on one abscissa; NaN marks a gap.
    budget : int
        The most samples to keep, at least two for each row and two for
        the ends.

    Returns
    -------
    numpy.ndarray
        Sorted sample indices.

    Raises
    ------
    ValueError
        If the budget cannot hold one slice's extremes for every row.
    """
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    count, samples = rows.shape
    budget = int(budget)
    if samples <= budget:
        return np.arange(samples)
    if budget < 2 * count + 2:
        raise ValueError(f'{budget} points cannot keep the extremes of '
                         f'{count} curves; at least {2 * count + 2} can')
    ends = np.array([0, samples - 1])

    def kept(bins):
        return np.unique(np.concatenate(
            [ends, *_extreme_indices(rows, bins)]))

    # the most slices whose extremes fit: rows that rise and fall
    # together share most of theirs, so the answer usually lies well
    # above budget / (2 * rows), and a bisection finds it in a dozen
    # vectorized passes
    low, high = 1, budget // 2
    best = kept(low)
    while low < high:
        middle = (low + high + 1) // 2
        trial = kept(middle)
        if len(trial) <= budget:
            low, best = middle, trial
        else:
            high = middle - 1
    return best


def _extreme_indices(rows: np.ndarray, bins: int) -> list[np.ndarray]:
    """Per record, the sorted sample indices keeping each slice's extremes.

    Slices are a uniform chunk with the tail padded, rather than
    `linspace` edges, because a uniform chunk is what one `reshape`
    and one `argmin` over the whole block can answer — the per-bin
    Python loop this replaces was the entire cost of a large
    waterfall. Infinities stand in for NaN during the argmin/argmax
    so an all-gap slice cannot raise; such a slice keeps its first
    sample instead, and the gap survives.
    """
    n, m = rows.shape
    chunk = -(-m // bins)
    padded = np.full((n, bins * chunk), np.nan)
    padded[:, :m] = rows
    shaped = padded.reshape(n, bins, chunk)
    finite = np.isfinite(shaped)
    any_finite = finite.any(axis=-1)
    lo = np.argmin(np.where(finite, shaped, np.inf), axis=-1)
    hi = np.argmax(np.where(finite, shaped, -np.inf), axis=-1)
    starts = np.arange(bins) * chunk
    # an all-gap slice keeps its start; slices entirely in the padding
    # (starts past the data) are dropped below
    lo = starts + np.where(any_finite, lo, 0)
    hi = starts + np.where(any_finite, hi, 0)
    real = starts < m
    return [np.unique(np.concatenate([row_lo[real], row_hi[real]]))
            for row_lo, row_hi in zip(lo, hi)]


def envelope_rows(x: ArrayLike, rows: ArrayLike, bins: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Every record of one object thinned onto the same bins, at most
    `bins` of them: two points a bin, the bin's least and greatest value
    in the order they occurred, at the bin's first and last sample time.

    `peak_decimate_rows` keeps each record's extremes at their own
    positions, so a page carrying two dozen records carries two dozen
    abscissas. An envelope on common bins lets a report's time-history
    page share one abscissa among its curves, and draws the same
    picture, since a bin is far narrower than a pixel at any zoom the
    page offers. Input with at most two points a bin is returned as
    given.

    Returns ``(axis, envelope)``: the shared abscissa and a (records,
    points) array.
    """
    # The shared axis halved the size of a report's time-history page
    # (2026-09-20).
    x = np.asarray(x, dtype=float)
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    n, m = rows.shape
    if m <= 2 * bins:
        return x, rows
    chunk = -(-m // bins)
    real = int(-(-m // chunk))
    padded = np.full((n, real * chunk), np.nan)
    padded[:, :m] = rows
    shaped = padded.reshape(n, real, chunk)
    finite = np.isfinite(shaped)
    lo = np.argmin(np.where(finite, shaped, np.inf), axis=-1)
    hi = np.argmax(np.where(finite, shaped, -np.inf), axis=-1)
    least = np.take_along_axis(shaped, lo[..., None], axis=-1)[..., 0]
    most = np.take_along_axis(shaped, hi[..., None], axis=-1)[..., 0]
    any_finite = finite.any(axis=-1)
    least = np.where(any_finite, least, np.nan)
    most = np.where(any_finite, most, np.nan)
    # the extreme that came first goes at the bin's start
    first_is_low = lo <= hi
    start = np.where(first_is_low, least, most)
    end = np.where(first_is_low, most, least)
    starts = np.arange(real) * chunk
    ends = np.minimum(starts + chunk - 1, m - 1)
    axis = np.column_stack([x[starts], x[ends]]).ravel()
    out = np.empty((n, 2 * real))
    out[:, 0::2] = start
    out[:, 1::2] = end
    return axis, out
