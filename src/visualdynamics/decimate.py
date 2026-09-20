"""Peak-keeping decimation: fewer points, every peak kept.

A million-sample record cannot go to a renderer point for point, and
plain striding is the wrong thinning: it walks straight past the one
spike that was the reason to look at the record. The peak-keeping
reading cuts the samples into equal slices and keeps each slice's
smallest and largest value at their own abscissa positions, so no
resonance, dropout or shock thins away — the same reading pyqtgraph's
`peak` downsampling gives the 2-D plot.

One implementation, used by the report (a document, not a scope) and
by the 3-D waterfall (a scene with a point budget). The core is
vectorized over whole records at once because the waterfall hands it
hundreds of them: as a per-bin Python loop the same work took 10.6 s
on a 320-record million-sample history, batched it is 1.3 s — both
measured. numpy only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def peak_decimate(x: ArrayLike, y: ArrayLike,
                  budget: int) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) thinned to about `budget` points that keep every extreme.

    Real values; a caller reading complex data extracts its component
    first, because "the extremes" of a complex value is not a question
    with one answer. Input at or under the budget is returned as given.

    NaN marks a gap (a specification is NaN outside its band) and never
    wins an extreme. A slice that is *all* gap keeps its first point, so
    the gap survives to be drawn as a gap.
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

    The records share `x` going in but not coming out: each keeps its
    own extremes at their own positions. One vectorized pass over the
    whole (records, samples) block, which is what makes decimating a
    320-record million-sample history cost tenths of a second rather
    than tens.
    """
    x = np.asarray(x)
    rows = np.atleast_2d(np.asarray(rows))
    if rows.shape[1] <= budget:
        return [(x, rows[k]) for k in range(rows.shape[0])]
    kept = _extreme_indices(rows, budget // 2)
    return [(x[keep], rows[k][keep]) for k, keep in enumerate(kept)]


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
    """Every record of one object thinned onto the same `bins`: two
    points a bin, the bin's least and greatest in the order they
    occurred, at the bin's first and last sample time.

    `peak_decimate_rows` keeps each record's extremes at their own
    positions, so a page carrying two dozen records carries two dozen
    time axes. A report's time-history page shares one axis among its
    curves (2026-09-20: it is what halved the page), and an envelope
    on common bins draws the same picture — the bin is far narrower
    than a pixel at any zoom the page offers. Input at or under two
    points a bin is returned as given, one axis for all.
    """
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
