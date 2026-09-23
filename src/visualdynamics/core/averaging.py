"""How a time history is cut into frames before it is averaged.

A spectral average is not one number but a recipe: where to start, how
long each frame is, how far they overlap, what window shapes them, and
how many of them there are. Those five are independent; everything else
— the hop between frames, where the analysis ends, how long it runs —
follows from them, and is derived here rather than stored, so nothing
can disagree.

The frame count is what is kept, not the stop time. Dragging the end of
the analysis in the GUI adds or removes whole frames, because half a
frame is not an average.

References
----------
1. Welch, P. D. (1967). "The Use of Fast Fourier Transform for the
   Estimation of Power Spectra: A Method Based on Time Averaging Over
   Short, Modified Periodograms." *IEEE Transactions on Audio and
   Electroacoustics*, 15(2), 70-73.
   [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
   The averaged-periodogram method these frames feed, and why
   overlapping them recovers what the window threw away.
2. Harris, F. J. (1978). "On the Use of Windows for Harmonic Analysis
   with the Discrete Fourier Transform." *Proceedings of the IEEE*,
   66(1), 51-83.
   [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)
   The window shapes, their equivalent noise bandwidths and their
   overlap correlation - the table behind a half overlap for a Hann
   window.
3. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The random error of an
   averaged estimate against the number of frames, which is what the
   frame count buys.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

#: the windows offered, under scipy's own names (Brandon, 2026-08-28):
#: the shapes come from `scipy.signal.get_window` now, and a name that
#: is also scipy's is one a reader can look up in scipy's documentation
#: and get the same numbers. 'boxcar' is the one spelling that changed
#: — engineering prose says rectangle, and the aliases keep every such
#: spelling working, along with what Rattlesnake writes ('Hann',
#: 'rectangle').
WINDOWS = ('boxcar', 'hann', 'hamming', 'blackman',
           'blackmanharris', 'flattop', 'tukey', 'kaiser')

#: the windows that take a number, with what it is called, its
#: default and the range a box may offer. Tukey's alpha is the
#: tapered fraction — 0 is a boxcar, 1 is a hann — and 0.5 is
#: scipy's own default; Kaiser's beta trades main-lobe width for
#: sidelobes, and 14 sits in the blackman-harris class.
WINDOW_PARAMETERS = {'tukey': ('alpha', 0.5, (0.0, 1.0)),
                     'kaiser': ('beta', 14.0, (0.0, 40.0))}

_ALIASES = {'uniform': 'boxcar', 'rectangle': 'boxcar', 'none': 'boxcar',
            'rect': 'boxcar', 'hanning': 'hann', 'flat_top': 'flattop',
            'flat top': 'flattop', 'blackman_harris': 'blackmanharris',
            'blackman harris': 'blackmanharris'}


#: how a frame is leveled before it is windowed. 'none' is the
#: convention here and in sdynpy — the oracle pins it — while scipy's
#: welch removes each segment's mean by default; offering the choice
#: is scipy parity (Brandon, 2026-08-29). 'mean' subtracts the
#: frame's own mean; 'linear' takes a least-squares line out, for a
#: record with drift the frames should not carry into the low bins.
DETRENDS = ('none', 'mean', 'linear')

_DETREND_ALIASES = {'constant': 'mean', 'off': 'none'}


def normalize_detrend(name: str | None) -> str:
    """The key for a detrend named any of the ways it might be."""
    if name is None:
        return 'none'
    key = str(name).strip().lower()
    key = _DETREND_ALIASES.get(key, key)
    if key not in DETRENDS:
        raise ValueError(
            f'unknown detrend {name!r}; visualdynamics has '
            + ', '.join(DETRENDS))
    return key


def normalize_window(name: str | None) -> str:
    """The key for a window named any of the ways a file might name it."""
    if name is None:
        return 'boxcar'
    key = str(name).strip().lower().replace('-', '_')
    key = _ALIASES.get(key, key)
    if key not in WINDOWS:
        raise ValueError(
            f'unknown window {name!r}; visualdynamics has ' + ', '.join(WINDOWS))
    return key


def from_span(averaging: Averaging, low: float, high: float,
              sample_rate: float, samples: int,
              origin: float = 0.0) -> Averaging:
    """The averaging a dragged span means, in whole frames.

    The frame length and the window are the table's; a drag says only
    where the analysis starts and how many frames it holds. One
    implementation for both editors — the 2-D region and the stage's
    handles commit through this, so they cannot disagree about what a
    span means.
    """
    rate = sample_rate
    n = averaging.frame_length
    # the drag is on the record's clock; the start counts from the
    # record's beginning (`Averaging.stop` says why they differ)
    low, high = float(low) - origin, float(high) - origin
    last_start = max(samples - n, 0) / rate
    start = min(max(low, 0.0), last_start)
    moved = replace(averaging, start=start)
    width = round((float(high) - start) * rate)
    wanted = 1 if width < n else 1 + (width - n) // moved.hop
    frames = max(1, min(int(wanted), moved.most_frames(samples, rate)))
    return replace(moved, frames=frames)


def window_shape(name: str | None, length: int,
                 parameter: float | None = None) -> np.ndarray:
    """The window itself, as the values each frame is multiplied by.

    scipy's `get_window`, periodic (`fftbins=True`) — the convention
    for spectral work, where a frame is one period of an
    assumed-repeating signal.

    These were hand-written cosine sums until 2026-08-28, because scipy
    was a test-only dependency and fifty lines was the better trade
    (principle 7). scipy became a runtime dependency that week, and the
    trade reversed: the tables were verified identical to scipy's to
    machine epsilon at every length that can hold a frame before being
    deleted, and the sdynpy oracle now checks the windows through the
    whole spectral pipeline on every run. The one behavioral change is
    a one-sample window, which reads 1.0 where the old taper read its
    own first point — and a one-sample frame cannot exist
    (`Averaging` requires two).
    """
    from scipy.signal import get_window

    key = normalize_window(name)
    if length < 1:
        raise ValueError('a window needs at least one sample')
    if key in WINDOW_PARAMETERS:
        what, default, _bounds = WINDOW_PARAMETERS[key]
        del what
        return get_window((key, default if parameter is None
                           else float(parameter)), length, fftbins=True)
    return get_window(key, length, fftbins=True)


#: the flat rail's proportions, in fractions of the visible height.
#: RAIL_TOP is where the tallest window peaks, RAIL_SLOT is one
#: frame's share, and RAIL_BAND caps what the whole rail may take
#: however many levels it needs — at a fixed step, four levels would
#: walk down into the data. GLYPH_SHARE is how much of a slot the
#: window itself fills, RAIL_GAP the clear air between the trace and
#: the rail's foot, and RAIL_CAP the tick dropped under each end of a
#: window — a tapering window touches its baseline at both ends and
#: would otherwise say nothing about where it stopped.
#:
#: The stage restates these for itself (`viz.marks`): its ceiling is
#: fixed at the stage's own height rather than wherever the view is,
#: so the proportions differ on purpose.
RAIL_TOP = 0.97
RAIL_SLOT = 0.09
RAIL_BAND = 0.34
GLYPH_SHARE = 0.78
RAIL_GAP = 0.04
RAIL_CAP = 0.016

#: how solid a band gets where its window peaks, and how solid the
#: fill under a window in the rail is — as fractions, so a canvas and
#: a QColor can each say them their own way
BAND_PEAK_ALPHA = 64 / 255
GLYPH_ALPHA = 110 / 255


@dataclass(frozen=True)
class Averaging:
    """The six independent numbers, and everything else derived.

    `start` is in seconds from the beginning of the record; `overlap` is
    a fraction of a frame, so 0.5 is the usual half-overlap; `frames` is
    how many are averaged *per record*. A history that already holds one
    frame per record — a burst-random run saved as 20 captures — is
    described by a frame the length of a record with `frames=1`, which
    is what `for_records` builds and what averaging with no parameters
    has always done.

    There is no zero-padding field, and there was for two days
    (2026-08-27/28, `pad`). It was removed once the mode fitter was
    shown not to need it — the apparent accuracy it bought was the
    fitter's own search railing on a leash, fixed in the search — and
    what remained was presentation: interpolated lines that resolve
    nothing new. The demonstration projects were regenerated rather
    than kept compatible with a two-day experiment.
    """

    frame_length: int
    overlap: float = 0.0
    window: str = 'hann'
    frames: int = 1
    start: float = 0.0
    detrend: str = 'none'
    #: the parameterized windows' own number — tukey's alpha, kaiser's
    #: beta — None for every window that takes none. A parameterized
    #: window given None adopts its documented default, so a stored
    #: Averaging always carries the number it was computed with.
    window_parameter: float | None = None

    def __post_init__(self) -> None:
        if int(self.frame_length) < 2:
            raise ValueError('a frame needs at least two samples')
        if not 0.0 <= float(self.overlap) < 1.0:
            raise ValueError('overlap is a fraction of a frame, from 0 up to '
                             'but not including 1')
        if int(self.frames) < 1:
            raise ValueError('averaging needs at least one frame')
        if float(self.start) < 0.0:
            raise ValueError('the analysis cannot start before the record')
        object.__setattr__(self, 'frame_length', int(self.frame_length))
        object.__setattr__(self, 'overlap', float(self.overlap))
        object.__setattr__(self, 'frames', int(self.frames))
        object.__setattr__(self, 'start', float(self.start))
        object.__setattr__(self, 'window', normalize_window(self.window))
        object.__setattr__(self, 'detrend',
                           normalize_detrend(self.detrend))
        settings = WINDOW_PARAMETERS.get(self.window)
        if settings is None:
            if self.window_parameter is not None:
                raise ValueError(
                    f'{self.window} takes no parameter; refuse it here '
                    'rather than carrying a number that means nothing')
        else:
            what, default, (lowest, highest) = settings
            value = (default if self.window_parameter is None
                     else float(self.window_parameter))
            if not lowest <= value <= highest:
                raise ValueError(
                    f"{self.window}'s {what} runs {lowest:g} to "
                    f'{highest:g}; {value:g} is outside it')
            object.__setattr__(self, 'window_parameter', value)

    @classmethod
    def for_records(cls, samples: int,
                    window: str = 'boxcar') -> Averaging:
        """Each record its own frame, which is what a stored capture is.

        Everything but the window follows from the record: it starts
        where the record starts, runs its whole length, and there is no
        second frame inside it to overlap with. The window is still the
        user's, so a saved capture can be re-averaged under Hann without
        the frames themselves moving.
        """
        return cls(frame_length=samples, overlap=0.0, window=window,
                   frames=1, start=0.0)

    def shape(self) -> np.ndarray:
        """This averaging's window, as the values each frame is
        multiplied by — `window_shape` with this frame length and
        this parameter, so no caller can pair them wrongly.

        Returns
        -------
        numpy.ndarray
            One value per sample of a frame.
        """
        return window_shape(self.window, self.frame_length,
                            self.window_parameter)

    @property
    def hop(self) -> int:
        """Samples from the start of one frame to the start of the next."""
        return max(round(self.frame_length * (1.0 - self.overlap)), 1)

    @property
    def span(self) -> int:
        """Samples the whole analysis covers, first frame to last."""
        return self.frame_length + (self.frames - 1) * self.hop

    def start_sample(self, sample_rate: float) -> int:
        return round(self.start * sample_rate)

    def stop(self, sample_rate: float, origin: float = 0.0) -> float:
        """When the analysis ends, in seconds — derived, never stored.

        `origin` is the record's first instant. `start` counts from
        the record's beginning, a sample count in disguise, and stays
        so whatever the clock says; but a record whose clock does not
        begin at zero — a truncation keeps its measured instants, an
        import window reads the run's — is drawn and dragged on that
        clock, and the frames of one that began at 1060 s were being
        drawn at 0 s, off the plot (Brandon, 2026-09-18). Every
        reading in seconds takes the origin here, once.
        """
        return origin + (self.start_sample(sample_rate) + self.span) / sample_rate

    def frame_bounds(self, sample_rate: float,
                     origin: float = 0.0) -> list[tuple[float, float]]:
        """[(start, stop)] in seconds on the record's clock, one per
        frame, overlaps included — what the time history plot shades.
        `origin` as in `stop`."""
        first = self.start_sample(sample_rate)
        return [(origin + (first + i * self.hop) / sample_rate,
                 origin + (first + i * self.hop + self.frame_length) / sample_rate)
                for i in range(self.frames)]

    def fits(self, samples: int, sample_rate: float) -> bool:
        """Is there record enough for every frame asked for?"""
        return self.start_sample(sample_rate) + self.span <= samples

    def most_frames(self, samples: int, sample_rate: float) -> int:
        """How many whole frames the record can carry from `start`.

        What the drag handle snaps to: a frame is added only when there
        is room for all of it.
        """
        room = samples - self.start_sample(sample_rate) - self.frame_length
        return 0 if room < 0 else 1 + room // self.hop

    def levels(self, sample_rate: float) -> tuple[list[int], int]:
        """(level per frame, how many levels).

        Frames that overlap cannot share a row without their windows
        running through each other, so each takes the lowest row whose
        last frame has already ended. With no overlap they all sit on
        one row; at half overlap they alternate between two.

        Here rather than in the overlay because the report draws the
        same marks and the two must not drift — a figure that stacked
        them differently from the app would be describing a different
        analysis.
        """
        ends, levels = [], []
        for low, high in self.frame_bounds(sample_rate):
            for level, end in enumerate(ends):
                if low >= end:
                    ends[level] = high
                    levels.append(level)
                    break
            else:
                ends.append(high)
                levels.append(len(ends) - 1)
        return levels, max(len(ends), 1)

    def rail(self, sample_rate: float) -> dict[str, Any]:
        """Where the flat rail's slots sit, in fractions of the
        visible height — the 2-D overlay's own geometry, and the
        report's, from one set of numbers.

        Here for the same reason `levels` is: the app draws this rail
        over a live plot and the report draws it in a canvas, and a
        figure whose rail sat differently from the app's would be
        describing a different analysis. The page used to restate
        these constants in its JavaScript and was held to them by
        matching strings, which is a weaker seam than being told.

        Returns the level of each frame, how many levels there are,
        one slot's share, the glyph's height inside its slot, the tick
        under each end, the baseline of every frame, and `foot` — the
        share of the height the trace keeps once the rail has its
        room above.
        """
        levels, count = self.levels(sample_rate)
        slot = min(RAIL_SLOT, RAIL_BAND / count)
        glyph = slot * GLYPH_SHARE
        return {
            'levels': list(levels), 'rows': count,
            'slot': slot, 'glyph': glyph, 'cap': RAIL_CAP,
            'baselines': [RAIL_TOP - glyph - level * slot
                          for level in levels],
            'band_alpha': BAND_PEAK_ALPHA, 'glyph_alpha': GLYPH_ALPHA,
            # the trace keeps this share of the height and the rail
            # takes the rest; never less than a quarter, or four
            # levels of rail would walk down into the data
            'foot': max(RAIL_TOP - glyph - (count - 1) * slot - RAIL_GAP,
                        0.25)}

    def filled(self, samples: int, sample_rate: float) -> Averaging:
        """This averaging with as many frames as actually fit."""
        return replace(self, frames=max(self.most_frames(samples,
                                                         sample_rate), 1))
