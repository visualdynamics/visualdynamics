"""How a time history is cut into frames before it is averaged.

A spectral average is set by a few independent choices: where to
start, how long each frame is, how far the frames overlap, which window
shapes them (with its parameter, for the windows that take one),
whether each frame is detrended, and how many frames there are.
Everything else, such as the hop between frames, where the analysis
ends and how long it runs, follows from those and is derived here
instead of stored, so nothing can disagree.

The frame count is stored, not the stop time. Dragging the end of the
analysis in the GUI adds or removes whole frames, because half a frame
is not an average.

What an average is worth follows from the same settings. An
autospectrum averaged over n independent frames is a chi-square
variable with 2n statistical degrees of freedom (`statistical_dof`):
two per frame, the real and the imaginary part of each line. Frames
that overlap are not independent, so the equivalent count is lower, by
how much the window lets two overlapped frames share
(`frame_correlation`). From the equivalent count follows the
confidence band on the estimate (`chi_square_factors`,
`confidence_bands`): the range the true spectrum lies in, at a stated
confidence, as a factor below and a factor above the averaged value.
These degrees of freedom are statistical. They have nothing to do with
the measurement DOFs (node and direction) a record is taken at.

References
----------
These frames feed Welch's averaged periodogram, and the window choices
are Harris's.

1. Welch, P. D. (1967). "The Use of Fast Fourier Transform for the
   Estimation of Power Spectra: A Method Based on Time Averaging Over
   Short, Modified Periodograms." *IEEE Transactions on Audio and
   Electroacoustics*, 15(2), 70-73.
   [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
2. Harris, F. J. (1978). "On the Use of Windows for Harmonic Analysis
   with the Discrete Fourier Transform." *Proceedings of the IEEE*,
   66(1), 51-83.
   [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)
   Window shapes, equivalent noise bandwidths and overlap correlation.
3. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The random error of an
   averaged estimate against the number of frames, which is what the
   frame count sets, and the chi-square confidence interval on an
   autospectral density estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

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
    """The key for a detrend named any of the ways it might be.

    'none', 'mean' or 'linear', case-insensitive, or the aliases
    'constant' and 'off'; None means 'none'. An unknown name raises
    ValueError.
    """
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
    """The key for a window named any of the ways a file might name it.

    One of scipy's names in `WINDOWS`, case-insensitive, or an alias
    such as 'rectangle', 'uniform' or 'hanning'; None means 'boxcar'.
    An unknown name raises ValueError.
    """
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

    `low` and `high` are the span's edges in seconds on the record's
    clock, which starts at `origin`, and `samples` is the record's
    length. The frame length and the window stay as they are; a drag
    sets only where the analysis starts and how many frames it holds.
    The start is kept inside the record, and the frame count is at
    least one and no more than fit. The 2-D region and the 3-D stage's
    handles both commit through this, so they agree on what a span
    means.
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
    """The window `name` for a frame of `length` samples, as the values
    each frame is multiplied by.

    This is scipy's `get_window`, periodic (`fftbins=True`), the
    convention for spectral work, where a frame is one period of a
    signal assumed to repeat. `parameter` is the Tukey window's alpha or
    the Kaiser window's beta, the default in `WINDOW_PARAMETERS` when
    None; other windows ignore it. A length under one is refused.
    """
    # These were hand-written cosine sums until 2026-08-28, because
    # scipy was a test-only dependency and fifty lines was the better
    # trade (principle 7). scipy became a runtime dependency that week,
    # and the trade reversed: the tables were verified identical to
    # scipy's to machine epsilon at every length that can hold a frame
    # before being deleted, and the sdynpy oracle now checks the windows
    # through the whole spectral pipeline on every run. The one change
    # in behavior is a one-sample window, which reads 1.0 where the old
    # taper read its own first point, and a one-sample frame cannot
    # exist (`Averaging` requires two).
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
    """The independent settings of an average; everything else is
    derived from them.

    `frame_length` is in samples, at least two. `overlap` is a fraction
    of a frame, from 0 up to but not including 1, so 0.5 is the usual
    half overlap. `window` is any name `normalize_window` accepts. Each
    frame is leveled by `detrend` ('none', 'mean' or 'linear') before
    it is windowed. `frames` is how many frames are averaged per record,
    at least one. `start` is in seconds from the beginning of the
    record. `window_parameter` is the Tukey or Kaiser window's number,
    filled in with its default when None and refused for a window that
    takes none. `start_set` says the start was chosen (by the detector,
    a drag or a typed number) instead of left at the record's
    beginning; it is not part of equality.

    A history that already holds one frame per record, such as a
    burst-random run saved as 20 captures, is described by a frame the
    length of a record with `frames=1`, which is what `for_records`
    builds and what averaging with no parameters does.

    There is no zero-padding setting: interpolated lines resolve nothing
    new.
    """

    # There was a zero-padding field for two days (2026-08-27/28,
    # `pad`). It was removed once the mode fitter was shown not to need
    # it: the apparent accuracy it gave was the fitter's own search
    # stopping at its limit, fixed in the search. What remained was
    # presentation. The demonstration projects were regenerated instead
    # of kept compatible with a two-day experiment.

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
    #: whether the start was chosen — by the detector, a drag, a typed
    #: number — rather than left at the record's beginning. A file's
    #: recipe says frame length, window and overlap and never when
    #: (Brandon, 2026-09-30: the view should open on the detected
    #: start when none was assigned). Not part of equality: two
    #: averagings with the same numbers are the same averaging.
    start_set: bool = field(default=False, compare=False)

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
        """An averaging with each record as its own frame, which is what
        a stored capture is.

        Everything but the window follows from the record: the frame
        starts where the record starts, runs its whole length
        (`samples`), and has no second frame to overlap with. The window
        is still the user's choice (boxcar by default), so a saved
        capture can be averaged again under Hann without the frames
        moving.
        """
        return cls(frame_length=samples, overlap=0.0, window=window,
                   frames=1, start=0.0)

    def shape(self) -> np.ndarray:
        """This averaging's window, as the values each frame is
        multiplied by: `window_shape` with this frame length and this
        parameter, so a caller cannot pair them wrongly.

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
        """The index of the sample the analysis starts on, at
        `sample_rate`."""
        return round(self.start * sample_rate)

    def stop(self, sample_rate: float, origin: float = 0.0) -> float:
        """When the analysis ends, in seconds on the record's clock;
        derived, never stored.

        `origin` is the record's first instant. `start` counts from the
        record's beginning, in effect a sample count, whatever the clock
        says. A record whose clock does not begin at zero (a truncation
        keeps its measured times, an import window reads the run's) is
        drawn and dragged on that clock, so every reading in seconds
        adds the origin here.
        """
        # The frames of a record that began at 1060 s were being drawn
        # at 0 s, off the plot (Brandon, 2026-09-18).
        return origin + (self.start_sample(sample_rate) + self.span) / sample_rate

    def frame_bounds(self, sample_rate: float,
                     origin: float = 0.0) -> list[tuple[float, float]]:
        """[(start, stop)] in seconds on the record's clock, one per
        frame, overlaps included; the time history plot shades these.
        `origin` is as in `stop`."""
        first = self.start_sample(sample_rate)
        return [(origin + (first + i * self.hop) / sample_rate,
                 origin + (first + i * self.hop + self.frame_length) / sample_rate)
                for i in range(self.frames)]

    def fits(self, samples: int, sample_rate: float) -> bool:
        """Whether a record of `samples` samples has room for every
        frame asked for."""
        return self.start_sample(sample_rate) + self.span <= samples

    def most_frames(self, samples: int, sample_rate: float) -> int:
        """How many whole frames a record of `samples` samples can hold
        from `start`, 0 when not even one fits.

        The drag handle snaps to this: a frame is added only when there
        is room for all of it.
        """
        room = samples - self.start_sample(sample_rate) - self.frame_length
        return 0 if room < 0 else 1 + room // self.hop

    def levels(self, sample_rate: float) -> tuple[list[int], int]:
        """(level per frame, how many levels), the rows the frames are
        drawn on.

        Frames that overlap cannot share a row without their windows
        running through each other, so each takes the lowest row whose
        last frame has already ended. With no overlap they all sit on
        one row; at half overlap they alternate between two. This is
        here, not in the overlay, because the report draws the same
        marks, and a figure that stacked them differently from the app
        would be describing a different analysis.
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
        """Where the flat rail's slots sit, in fractions of the visible
        height: the 2-D overlay's geometry and the report's, from one set
        of numbers.

        It is here for the same reason as `levels`: the app draws this
        rail over a live plot and the report draws it in a canvas, and a
        figure whose rail sat differently from the app's would be
        describing a different analysis.

        Returns a dict: `levels` (each frame's level), `rows` (how many
        levels), `slot` (one slot's share), `glyph` (the window's height
        inside its slot), `cap` (the tick under each end of a window),
        `baselines` (each frame's baseline), `band_alpha` and
        `glyph_alpha` (the fill opacities), and `foot` (the share of the
        height the trace keeps below the rail, at least a quarter).
        """
        # The page used to restate these constants in its JavaScript and
        # was held to them by matching strings, which is weaker than
        # being given them.
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

    def statistical_dof(self, records: int = 1) -> StatisticalDof:
        """The statistical degrees of freedom of a spectrum averaged
        over these frames: `statistical_dof` with this averaging's
        frame count, window and hop, so a caller cannot pair them
        wrongly.

        Parameters
        ----------
        records : int, optional
            How many records each channel holds, all cut the same way
            and pooled, as a capture saved one record per average is.
            Separate records share no samples, so each adds its frames'
            degrees of freedom in full.

        Returns
        -------
        StatisticalDof
            The frames averaged, the nominal 2n and the equivalent
            count discounted for overlap.
        """
        return statistical_dof(self.frames, self.shape(), self.hop,
                               records=records)

    def overlap_text(self) -> str:
        """The overlap as every summary writes it, a percentage of a
        frame to three significant figures: '50%', '66.7%', '0%'."""
        return f'{self.overlap * 100.0:.3g}%'

    def window_text(self) -> str:
        """The window as every summary writes it, with its parameter
        where it takes one: 'hann', 'tukey (alpha 0.5)'."""
        settings = WINDOW_PARAMETERS.get(self.window)
        if settings is None or self.window_parameter is None:
            return self.window
        return f'{self.window} ({settings[0]} {self.window_parameter:g})'

    def describe(self) -> str:
        """The settings in one phrase, the one statement of them a
        caption, a label or a summary uses, so no two say the same
        averaging two ways: 'hann, 50% overlap, frames of 1024
        samples', with the window's parameter and the detrend when
        there are ('tukey (alpha 0.5), 66.7% overlap, frames of 2048
        samples, mean removed from each frame').

        Returns
        -------
        str
        """
        said = (f'{self.window_text()}, {self.overlap_text()} overlap, '
                f'frames of {self.frame_length} samples')
        if self.detrend != 'none':
            removed = 'mean' if self.detrend == 'mean' else 'linear trend'
            said += f', {removed} removed from each frame'
        return said

    def filled(self, samples: int, sample_rate: float) -> Averaging:
        """This averaging with as many frames as fit in a record of
        `samples` samples, at least one."""
        return replace(self, frames=max(self.most_frames(samples,
                                                         sample_rate), 1))


# ---- what an average is worth -------------------------------------------

#: the confidence levels a band is given at when none is asked for:
#: 90 percent, the usual reading of a test's spectrum, and 95, the
#: stricter one
CONFIDENCE_LEVELS = (0.90, 0.95)


@dataclass(frozen=True)
class StatisticalDof:
    """How many statistical degrees of freedom an averaged autospectrum
    has.

    `frames` is how many frames were averaged, every record's together.
    `nominal` is two per frame, 2n, which is the count when no two
    frames share a sample. `equivalent` is the count discounted for
    overlap, Welch's: overlapped frames carry part of the same record,
    so they vary together and an average of them steadies less than
    the same number of independent frames would. With no overlap the
    two are equal.
    """

    frames: int
    nominal: int
    equivalent: float


def frame_correlation(shape: ArrayLike, hop: int, apart: int = 1) -> float:
    """How much two frames `apart` hops from each other share, as their
    window sees it: 0 for frames that do not overlap, 1 for a frame and
    itself.

    `shape` is the window, one value per sample of a frame, and `hop`
    is the samples from one frame's start to the next
    (`Averaging.hop`). The value is the window's overlap correlation,
    the sum over the shared samples of one frame's weight times the
    other's, over the sum of the weights squared. It is the number
    Harris tabulates for each window: a Hann window at half overlap
    gives 1/6, about 0.167, a rectangle at half overlap 0.5.
    """
    weights = np.asarray(shape, dtype=float)
    length = len(weights)
    shift = int(hop) * int(apart)
    if shift < 0:
        raise ValueError('frames are apart by a whole number of hops, '
                         'zero or more')
    if shift >= length:
        return 0.0
    power = float(np.sum(weights ** 2))
    if power == 0.0:
        raise ValueError('a window of zeros weighs nothing')
    return float(np.sum(weights[:length - shift] * weights[shift:]) / power)


def statistical_dof(frames: int, shape: ArrayLike, hop: int,
                    records: int = 1) -> StatisticalDof:
    """The statistical degrees of freedom of an autospectrum averaged over
    `frames` frames of the window `shape`, each `hop` samples after the
    last, in each of `records` records.

    The nominal count is two per frame. The equivalent count is Welch's
    for overlapped frames: the variance of the average is the variance
    of independent frames times

        1 + 2 * sum over j of (1 - j/n) * c(j)**2

    where n is the frame count and c(j) is the window's correlation
    between frames j hops apart (`frame_correlation`), and the
    equivalent count is 2n over that factor. Only the lags at which two
    frames still share samples contribute. Records are separate pieces
    of signal and share nothing, so each adds its frames' count in full.

    The count is that of a line between zero and the Nyquist frequency.
    The line at zero (and at Nyquist, for an even frame) is real, so it
    carries one degree of freedom per frame rather than two. Taking out
    each frame's mean or trend costs the lowest lines a little more.
    Neither is folded in here: the count is the one the spectrum's
    interior lines have, which is where the band is read.
    """
    # Welch (1967) gives the factor for any window and overlap; the
    # Harris (1978) correlations are its c(j) for the common windows.
    # The textbook ratio for Hann at half overlap (about 0.95 of 2n for
    # many frames) is this formula's limit, not a constant to store.
    frames, hop, records = int(frames), int(hop), int(records)
    if frames < 1:
        raise ValueError('an average needs at least one frame')
    if records < 1:
        raise ValueError('an average needs at least one record')
    if hop < 1:
        raise ValueError('frames start at least one sample apart')
    weights = np.asarray(shape, dtype=float)
    spread = 1.0
    lag = 1
    while lag < frames and lag * hop < len(weights):
        shared = frame_correlation(weights, hop, lag)
        spread += 2.0 * (1.0 - lag / frames) * shared ** 2
        lag += 1
    return StatisticalDof(frames=frames * records,
                          nominal=2 * frames * records,
                          equivalent=records * 2.0 * frames / spread)


def chi_square_factors(dof: float, level: float = 0.90
                       ) -> tuple[float, float]:
    """(lower, upper): the factors the true spectrum lies between, as
    multiples of an estimate with `dof` statistical degrees of freedom,
    at confidence `level`.

    An averaged autospectrum is the true one times a chi-square variable
    over its degrees of freedom, so the true value lies between
    dof / chi2(1 - a/2) and dof / chi2(a/2) times the estimate, where a
    is one minus the level and chi2(p) is the chi-square quantile at p.
    The band is not symmetric: the factor above is the larger, in
    decibels as well as in ratio, because an average of few frames
    reads low more often than it reads high. `dof` need not be a whole
    number, since an equivalent count seldom is.

    Examples: at two degrees of freedom (one frame) the 90 percent band
    runs from 0.334 to 19.5 times the estimate; at 120 it runs from
    0.819 to 1.25.
    """
    # The quantile is scipy's regularized incomplete gamma inverse,
    # chi2(p) = 2 * gammaincinv(dof / 2, p), which is exact for any dof;
    # the Wilson-Hilferty cube-root approximation is good to a few
    # percent above ten degrees of freedom and poor below, which is
    # exactly where a band matters most.
    from scipy.special import gammaincinv

    dof, level = float(dof), float(level)
    if not dof > 0.0:
        raise ValueError('degrees of freedom are more than zero')
    if not 0.0 < level < 1.0:
        raise ValueError('a confidence level is a fraction between 0 and 1')
    tail = (1.0 - level) / 2.0
    high_quantile = 2.0 * float(gammaincinv(dof / 2.0, 1.0 - tail))
    low_quantile = 2.0 * float(gammaincinv(dof / 2.0, tail))
    return dof / high_quantile, dof / low_quantile


@dataclass(frozen=True)
class ConfidenceBand:
    """Where the true spectrum lies about an averaged estimate, at one
    confidence level.

    `level` is the confidence, such as 0.9. `dof` is the statistical
    degrees of freedom the band was worked out at. `lower_factor` and
    `upper_factor` are the multiples of the estimate the band runs
    between (`chi_square_factors`), and `lower` and `upper` are the
    band itself: the estimate times each factor, the same shape and in
    the same units as the estimate it was given.
    """

    level: float
    dof: float
    lower_factor: float
    upper_factor: float
    lower: np.ndarray
    upper: np.ndarray

    @property
    def lower_db(self) -> float:
        """The band's lower edge in decibels from the estimate, a
        negative number (ten times the log of the factor, as for any
        power quantity)."""
        return 10.0 * float(np.log10(self.lower_factor))

    @property
    def upper_db(self) -> float:
        """The band's upper edge in decibels from the estimate, a
        positive number."""
        return 10.0 * float(np.log10(self.upper_factor))


def confidence_bands(estimate: ArrayLike, dof: float,
                     levels: Sequence[float] = CONFIDENCE_LEVELS
                     ) -> list[ConfidenceBand]:
    """The chi-square confidence band on an averaged autospectrum, one
    per level in `levels`, 90 and 95 percent by default.

    `estimate` is the averaged spectrum's values, any shape, and is not
    converted: the band comes back in whatever units it was given,
    because each edge is the estimate times a pure number. `dof` is the
    statistical degrees of freedom the estimate was averaged to,
    normally the equivalent count from `statistical_dof`. The band
    belongs to an autospectrum (a PSD, or the squared magnitude of a
    spectrum), not to a cross-spectrum or a transfer function, whose
    errors follow other laws.
    """
    values = np.asarray(estimate)
    if np.iscomplexobj(values):
        raise ValueError('a chi-square band is on a power, which is '
                         'real; this estimate is complex')
    out = []
    for level in levels:
        low, high = chi_square_factors(dof, level)
        out.append(ConfidenceBand(level=float(level), dof=float(dof),
                                  lower_factor=low, upper_factor=high,
                                  lower=values * low, upper=values * high))
    return out
