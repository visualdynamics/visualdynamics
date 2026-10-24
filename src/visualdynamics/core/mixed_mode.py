"""A mixed-mode test as one specification, and its data sheet.

A sine-on-random or random-on-random test is controlled to one drive
made of parts: a broadband random floor written as breakpoints, tones
written in amplitude (peak), fixed or swept, and narrow bands of random
laid over the floor, fixed or swept. Each part is judged by its own
rule, and the parts are only told apart if the specification says which
is which. `MixedSpecification` says it: the floor (a `Specification`),
the tones (a `SineSweepSpecification`), the narrowbands (`Narrowband`)
and the width left out of the floor either side of each tone.

Nothing new is written to a file for it. The floor is saved as the
random specification it is and the tones as the sine specification they
are, the two objects a random and sine project already holds. A fixed
tone is the dwell a controller's file writes, a linear segment across a
hair of a hertz (`SineTone.dwell`), so no segment type is added; the
tone table (`Tone`) is the form it is entered and read in. The
narrowbands and the width left out are attributes of the floor
(`Specification.narrowbands`, `Specification.tone_exclusion`), saved
with it the way its band constraints are.

`judge_mixed` reads a record against it, one channel at a time:

1. Where the tones begin in the record is found by matched filter
   (`core.sine.find_environment`), unless an onset is given. Each tone
   is read through a tracking band locked to its frequency
   (`core.sine_tracking.track_sine`), the way a controller reads it.
   The band settles in about one over its width, so the readings before
   that are not used; the tone's level is the median of the rest, in
   amplitude, compared with the tone table, by default at plus or minus
   10 percent (an example of what test documents ask for). The median,
   not the rms of the readings' deviation in dB that a sweep comparison
   scores (`core.compliance.sine_errors`): a small tone near the floor
   scatters through its band from reading to reading, and that scatter
   is the floor's, not an error in the tone's level.
2. What each band passed (`track_waveform`) is taken out of the record.
   What is left, the residual, is the floor and the narrowbands, and its
   averaged autospectrum is read for them.
3. The floor is judged on the residual's lines with the lines that held
   a tone, and the lines of every narrowband, left out. A line held a
   tone when the tone spent at least `LEFT_OUT_SHARE` of its run within
   the left-out width of it: every line beside a fixed tone, and few or
   none of the lines a long sweep passes, which it holds only briefly.
   The floor is judged as a comparison is (`core.compliance.judge`,
   line by line against the reference and the reference moved by the
   line tolerance, by default plus or minus 3 dB): its rms against the
   reference's over the same lines (by default plus or minus 10
   percent), and its lines outside against the number chance puts
   outside at their degrees of freedom. An averaged line scatters, so a
   correctly controlled floor of a few hundred lines has some outside a
   3 dB band on a short record; the lines fail only when more are
   outside than chance would put there one time in a thousand
   (`FLOOR_CHANCE`). A floor with no line left to judge is not judged,
   and the reading says so.
4. Each narrowband is judged at the degrees of freedom a narrow band
   has, which are fewer than the floor's. A fixed band's level is the
   mean of the residual's lines between its edges, worth the per-line
   degrees of freedom times the lines over the window's noise bandwidth.
   A swept band is read in slices of the run, each half as long as the
   band takes to move its own width, over the lines within a quarter of
   its width of its center at the slice's middle, which the band covered
   for the whole slice. Each slice is worth only the frames it holds.
   A swept band's level is its slices pooled, each weighted by its
   degrees of freedom, and is worth their sum; the share of slices that
   lie inside the tolerance widened to their own scatter is reported
   beside it, as how far the band wandered moment to moment. A band's
   tolerance is the one the plan states (`tolerance_db`), or the floor's
   line tolerance, widened in either case to the chi-square scatter of
   an estimate at those degrees of freedom (90 percent by default), so
   a band is never failed for scatter its statistics cannot avoid.
5. The total is the rms of the whole record over the span read, every
   line of the composite's autospectrum. It has no tolerance: it is the
   figure the amplifier, the coil and the fixture see.

The parts are independent, so their mean squares add: the reference
total is the root of the floor's mean square, each tone's A²/2 and each
band's level times its width.

References
----------
The tracking band is `core.sine_tracking`'s. The averaged periodogram,
its degrees of freedom and the chi-square scatter of an estimate are
`core.averaging`'s; that a band of width B averaged over T seconds is
worth about 2BT degrees of freedom is the band-limited estimate's
normalized random error.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The autospectrum's random
   error, 1/sqrt(B T) for a band B wide averaged over T, and the
   chi-square distribution of an averaged autospectrum.
2. Welch, P. D. (1967). "The Use of Fast Fourier Transform for the
   Estimation of Power Spectra." *IEEE Transactions on Audio and
   Electroacoustics*, 15(2), 70-73.
   [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
   The variance of an average of overlapped, windowed frames.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .sine import DWELL_SPAN, LINEAR, LOG, SineSweepSpecification, SineTone

#: the tolerance on a tone's level, percent of its amplitude either way:
#: an example of what test documents ask for
TONE_TOLERANCE = 10.0

#: the tolerance on each line of the floor, dB either way
FLOOR_DB = 3.0

#: the tolerance on the floor's rms, percent either way
FLOOR_RMS = 10.0

#: the width left out of the floor either side of a tone, Hz, when the
#: specification does not say: enough to hold a Hann window's spread
#: of a tone at 1 Hz lines, which leaks into about four lines
EXCLUSION = 2.0

#: the tracking band's width, Hz, when none is given
TRACKING_BAND = 0.5

#: the confidence the chi-square scatter of a narrowband's estimate is
#: read at
CONFIDENCE = 0.90

#: the share of a tone's run a line has to hold it, within the width
#: left out, to be left out of the floor. A fixed tone holds its lines
#: for the whole run; a sweep many times the width passes each line for
#: a sliver of it, and the tracking band takes out with the tone at
#: most this share of such a line's power, under half a decibel
LEFT_OUT_SHARE = 0.1

#: how seldom a correctly controlled floor fails on its lines by
#: chance: the lines fail only when more of them are outside the line
#: tolerance than chance puts there at this quantile of the binomial
FLOOR_CHANCE = 0.999

def _db(ratio: float) -> float:
    """Ten times the log of a power ratio; NaN for one that is not
    above zero."""
    ratio = float(ratio)
    return 10.0 * float(np.log10(ratio)) if ratio > 0.0 else float('nan')


# ---- the parts --------------------------------------------------------------

@dataclass(frozen=True)
class Narrowband:
    """A narrow band of random laid over the floor: a rectangle of
    higher density, fixed or swept.

    `center` is the band's center in Hz, where a fixed band stays and a
    swept one starts. `level` is the density the band adds over the
    floor between its edges, in SI ((m/s²)²/Hz for an acceleration), so
    its mean square is `level` times its width. The width is given one
    of two ways: `width`, a full width in Hz, or `share`, a full width as
    a share of the center, which widens with the center the way a
    harmonic's band does (0.10 for a band written plus or minus 5
    percent). A swept band sweeps its center from `center` to
    `sweep_to` and back, over and over, at `rate`: Hz/s for a LINEAR
    sweep, octaves a minute for a LOG one (`mode`). `tolerance_db` is
    the tolerance the plan states for the band, dB either way, or None
    for the floor's. `name` is what the sheet calls it.

    Raises ValueError for a center, level or width that is not above
    zero, for both widths or neither, for a band reaching below zero
    frequency, and for a sweep without a rate, a rate without a sweep,
    or a mode that is neither.
    """

    center: float
    level: float
    width: float | None = None
    share: float | None = None
    sweep_to: float | None = None
    rate: float | None = None
    mode: int = LINEAR
    tolerance_db: float | None = None
    name: str = ''

    def __post_init__(self) -> None:
        for name in ('center', 'level'):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f'a narrowband\'s {name} must be above zero')
        if (self.width is None) == (self.share is None):
            raise ValueError('give a narrowband one width: width in Hz or '
                             'share of its center')
        given = self.width if self.width is not None else self.share
        if not float(given) > 0.0:
            raise ValueError('a narrowband\'s width must be above zero')
        if self.share is not None and not float(self.share) < 2.0:
            raise ValueError('a band as wide as twice its center reaches '
                             'below zero frequency')
        if self.mode not in (LINEAR, LOG):
            raise ValueError('a sweep is LINEAR or LOG')
        if (self.sweep_to is None) != (self.rate is None):
            raise ValueError('a swept band needs both sweep_to and rate')
        if self.sweep_to is not None:
            if not float(self.sweep_to) > 0.0 \
                    or float(self.sweep_to) == float(self.center):
                raise ValueError('a band sweeps to another frequency above '
                                 'zero')
            if not float(self.rate) > 0.0:
                raise ValueError('a sweep rate is above zero; the direction '
                                 'is the one from center to sweep_to')
        if self.tolerance_db is not None and not float(self.tolerance_db) > 0:
            raise ValueError('a tolerance is above zero dB')
        for low in (self.edges(c)[0] for c in (self.center, self.sweep_to)
                    if c is not None):
            if not low > 0.0:
                raise ValueError('a narrowband reaches below zero frequency')

    @classmethod
    def plus_minus(cls, center: float, percent: float, level: float,
                   **options: Any) -> Narrowband:
        """A band written as a standard writes it, plus or minus
        `percent` of its center: `Narrowband.plus_minus(400, 5, w)` runs
        from 380 to 420 Hz. Its width is kept as a share of the center,
        so a swept band widens as it climbs.

        Parameters
        ----------
        center : float
            Hz.
        percent : float
            The half-width, percent of the center.
        level : float
            The density added over the floor, SI.
        **options
            `sweep_to`, `rate`, `mode`, `tolerance_db` and `name`.

        Returns
        -------
        Narrowband
            The band, its full width twice `percent`.
        """
        return cls(center, level, share=2.0 * float(percent) / 100.0,
                   **options)

    @classmethod
    def between(cls, low: float, high: float, level: float,
                **options: Any) -> Narrowband:
        """A band written as its two edges in Hz, the form a plan checks
        the control plot against.

        Parameters
        ----------
        low, high : float
            The edges, Hz, `low` under `high`.
        level : float
            The density added over the floor, SI.
        **options
            `tolerance_db` and `name`; a band given by its edges is
            fixed.

        Returns
        -------
        Narrowband
            The band, centered between its edges.
        """
        low, high = float(low), float(high)
        if not high > low:
            raise ValueError('a band\'s upper edge is above its lower one')
        return cls(0.5 * (low + high), level, width=high - low, **options)

    @property
    def swept(self) -> bool:
        """Whether the band moves."""
        return self.sweep_to is not None

    @property
    def label(self) -> str:
        """The band's name, or its center and width in words."""
        if self.name:
            return self.name
        low, high = self.edges()
        return f'{low:g}-{high:g} Hz band'

    def width_at(self, center: float | None = None) -> float:
        """The band's full width in Hz with its center at `center` (its
        own by default)."""
        center = self.center if center is None else float(center)
        return float(self.width) if self.width is not None \
            else float(self.share) * center

    def edges(self, center: float | None = None) -> tuple[float, float]:
        """(low, high), the band's edges in Hz with its center at
        `center` (its own by default)."""
        center = self.center if center is None else float(center)
        half = 0.5 * self.width_at(center)
        return center - half, center + half

    def mean_square(self, center: float | None = None) -> float:
        """What the band adds to the record's mean square: its level
        times its width, in SI."""
        return float(self.level) * self.width_at(center)

    @property
    def rms(self) -> float:
        """The band's rms at its own center, the root of its mean
        square."""
        return float(np.sqrt(self.mean_square()))

    def span(self) -> tuple[float, float]:
        """(low, high), every frequency the band covers over its sweep;
        its edges for a fixed band."""
        ends = [self.edges(self.center)]
        if self.sweep_to is not None:
            ends.append(self.edges(self.sweep_to))
        return (min(low for low, _high in ends),
                max(high for _low, high in ends))

    def leg_seconds(self) -> float:
        """How long the center takes from `center` to `sweep_to`; zero
        for a fixed band."""
        if self.sweep_to is None:
            return 0.0
        if self.mode == LINEAR:
            return abs(float(self.sweep_to) - self.center) / float(self.rate)
        return abs(np.log2(float(self.sweep_to) / self.center)) \
            / float(self.rate) * 60.0

    def center_at(self, seconds: Any) -> np.ndarray:
        """The band's center in Hz at `seconds` after the band started:
        from `center` to `sweep_to` and back, over and over; its own
        center throughout for a fixed band.

        Parameters
        ----------
        seconds : array-like
            Seconds from the band's start.

        Returns
        -------
        ndarray
            Hz, the shape of `seconds`.
        """
        t = np.asarray(seconds, dtype=np.float64)
        if self.sweep_to is None:
            return np.full(t.shape, float(self.center))
        leg = self.leg_seconds()
        phase = np.mod(np.maximum(t, 0.0), 2.0 * leg) / leg
        u = np.where(phase <= 1.0, phase, 2.0 - phase)
        c0, c1 = float(self.center), float(self.sweep_to)
        return c0 + (c1 - c0) * u if self.mode == LINEAR \
            else c0 * (c1 / c0) ** u

    def crossing_seconds(self, center: float) -> float:
        """How long the swept band takes to move its own width with its
        center at `center`: how long any one frequency near it is
        covered. Infinite for a fixed band."""
        if self.sweep_to is None:
            return float('inf')
        low, high = self.edges(center)
        if self.mode == LINEAR:
            return (high - low) / float(self.rate)
        return float(np.log2(high / low)) / float(self.rate) * 60.0

    def to_dict(self) -> dict[str, Any]:
        """The band's fields as plain numbers and text, the form it is
        saved in."""
        return asdict(self)

    @classmethod
    def from_dict(cls, fields: dict[str, Any]) -> Narrowband:
        """The band `to_dict` wrote.

        Parameters
        ----------
        fields : dict
            As `to_dict` returns them.

        Returns
        -------
        Narrowband
            The band.

        A field this version does not know, written by a later one, is
        passed over, so a file stays readable as the band gains fields.
        """
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{name: value for name, value in fields.items()
                      if name in known})


def narrowbands_json(bands: Sequence[Narrowband]) -> str:
    """Narrowbands as the text a file holds them in: a JSON list of
    `Narrowband.to_dict`."""
    return json.dumps([band.to_dict() for band in bands])


def narrowbands_from_json(text: str) -> tuple[Narrowband, ...]:
    """The narrowbands `narrowbands_json` wrote."""
    return tuple(Narrowband.from_dict(fields) for fields in json.loads(text))


@dataclass(frozen=True)
class Tone:
    """One row of a mixed test's tone table: a tone at a fixed frequency
    or swept between two, at one amplitude (peak).

    `frequency` is where a fixed tone stands and a swept one starts, in
    Hz; `sweep_to` is where a swept tone ends, None for a fixed tone.
    `rate` is the sweep rate, Hz/s for a LINEAR sweep and octaves a
    minute for a LOG one (`mode`), always above zero, the direction being
    the one from `frequency` to `sweep_to`; None on a row read from a
    tone of several segments, whose rates are its segments'. `amplitude`
    is the level in SI, peak (m/s² for an acceleration); a tone read
    from a specification whose level changes along its sweep gives its
    largest. `name` is the controller's name for it.

    `sine_tone` writes the row as the `SineTone` a sine specification
    holds; `Tone.of` reads one back.
    """

    frequency: float
    amplitude: float
    sweep_to: float | None = None
    rate: float | None = None
    mode: int = LINEAR
    name: str = ''

    def __post_init__(self) -> None:
        if not float(self.frequency) > 0.0:
            raise ValueError('a tone\'s frequency is above zero')
        if not float(self.amplitude) > 0.0:
            raise ValueError('a tone\'s amplitude is above zero')
        if self.mode not in (LINEAR, LOG):
            raise ValueError('a sweep is LINEAR or LOG')
        if self.sweep_to is not None and not float(self.sweep_to) > 0.0:
            raise ValueError('a tone sweeps to a frequency above zero')
        if self.rate is not None and not float(self.rate) > 0.0:
            raise ValueError('a sweep rate is above zero; the direction is '
                             'the one from frequency to sweep_to')

    @property
    def fixed(self) -> bool:
        """Whether the tone stands at one frequency."""
        return self.sweep_to is None

    @property
    def label(self) -> str:
        """The tone's name, or its frequency in words."""
        if self.name:
            return self.name
        if self.fixed:
            return f'{self.frequency:g} Hz'
        return f'{self.frequency:g}-{self.sweep_to:g} Hz'

    def sine_tone(self, seconds: float | None = None, channels: int = 1, *,
                  start_time: float = 0.0) -> SineTone:
        """This row as a sine specification holds it.

        A fixed tone is a dwell held `seconds` (`SineTone.dwell`), the
        form a controller's file gives one; a swept tone is one segment
        from `frequency` to `sweep_to` at `rate`. The amplitude is the
        same on every one of `channels` control channels.

        Parameters
        ----------
        seconds : float, optional
            How long a fixed tone holds; needed for a fixed tone only.
        channels : int, default 1
            How many control channels the specification has.
        start_time : float, default 0.0
            Seconds after the environment started that the tone turns
            on.

        Returns
        -------
        SineTone
            The tone, named by `label`.
        """
        amplitude = np.full(int(channels), float(self.amplitude))
        if self.fixed:
            if seconds is None:
                raise ValueError(f'{self.label}: a fixed tone needs the '
                                 'seconds it holds')
            return SineTone.dwell(self.label, self.frequency, amplitude,
                                  seconds, start_time=start_time)
        if self.rate is None:
            raise ValueError(f'{self.label}: a swept tone needs its rate')
        rising = float(self.sweep_to) > float(self.frequency)
        return SineTone(self.label, start_time,
                        [self.frequency, self.sweep_to],
                        np.tile(amplitude, (2, 1)), [self.mode],
                        [self.rate if rising else -self.rate])

    @classmethod
    def of(cls, tone: SineTone, column: int = 0) -> Tone:
        """The tone-table row of a sine specification's tone, on control
        channel `column`.

        A tone every segment of which is a dwell (`SineTone.is_dwell`)
        at one frequency is fixed; any other is swept from its first
        breakpoint to its last, with its one segment's rate and mode, or
        no rate when it has several segments.

        Parameters
        ----------
        tone : SineTone
            The tone.
        column : int, default 0
            Which control channel's amplitude.

        Returns
        -------
        Tone
            The row.
        """
        amplitude = float(np.max(tone.amplitude[:, column]))
        segments = len(tone.segment_rate)
        held = (len(tone.dwell_segments()) == segments
                and np.ptp(tone.frequency) <= DWELL_SPAN
                * float(np.max(tone.frequency)) * max(segments, 1))
        if held:
            return cls(float(tone.frequency[0]), amplitude, name=tone.name)
        rate = mode = None
        if segments == 1:
            rate = abs(float(tone.segment_rate[0]))
            mode = int(tone.segment_type[0])
        return cls(float(tone.frequency[0]), amplitude,
                   sweep_to=float(tone.frequency[-1]), rate=rate,
                   mode=LINEAR if mode is None else mode, name=tone.name)


def tone_span(tone: SineTone) -> tuple[float, float]:
    """(low, high), the frequencies a tone passes, in Hz."""
    return float(np.min(tone.frequency)), float(np.max(tone.frequency))


def held_share(tone: SineTone, frequencies: Any, width: float) -> np.ndarray:
    """The share of a tone's run it spends within `width` Hz of each of
    `frequencies`: 1 beside a fixed tone, and about twice the width over
    the range for a line a linear sweep passes.

    Parameters
    ----------
    tone : SineTone
        The tone.
    frequencies : array-like
        Hz.
    width : float
        Hz either side.

    Returns
    -------
    ndarray
        A share from 0 to 1 for each frequency.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    seconds = tone.duration()
    if seconds <= 0.0 or len(tone.dwell_segments()) == len(tone.segment_rate):
        low, high = tone_span(tone)
        return ((f >= low - width) & (f <= high + width)).astype(float)
    t, path = tone.trajectory(seconds / 4000.0)
    # each point of the path weighed by the time it stands for, since a
    # segment's steps are whole and may differ from the next segment's
    weight = np.gradient(np.asarray(t, dtype=np.float64))
    order = np.argsort(path, kind='stable')
    path, weight = np.asarray(path)[order], weight[order]
    held = np.concatenate([[0.0], np.cumsum(weight)])
    low = np.searchsorted(path, f - width, side='left')
    high = np.searchsorted(path, f + width, side='right')
    return (held[high] - held[low]) / held[-1]


def tone_mean_square(tone: SineTone, column: int = 0) -> float:
    """A tone's mean square as written: A²/2 for a fixed tone, the time
    average of its amplitude squared over two along a sweep."""
    seconds = tone.duration()
    if seconds <= 0.0 or len(tone.dwell_segments()) == len(tone.segment_rate):
        amplitude = float(np.max(tone.amplitude[:, column]))
        return 0.5 * amplitude ** 2
    _t, f = tone.trajectory(seconds / 400.0)
    level = tone.target(f)[:, column]
    return 0.5 * float(np.nanmean(level ** 2))


# ---- the specification -----------------------------------------------------

class MixedSpecification:
    """A mixed-mode test's requirement as one object: the floor, the
    tones, the narrowbands and the width left out around each tone.

    `floor` is the random specification the broadband part was written
    as, one record per control channel. `tones` is the sine
    specification of the tones, or the tone table as `Tone` rows, which
    is written into one (a fixed tone as a dwell held `seconds`), or
    None for a random-on-random test. `narrowbands` and `exclusion` (the
    width left out of the floor either side of each tone, Hz) are kept
    on the floor (`Specification.narrowbands`,
    `Specification.tone_exclusion`), which is how they are saved; given
    here, they are written onto a copy of the floor, which `floor` then
    holds (add it to a project to keep them), and left out, the floor's
    own are used (none, and `EXCLUSION`). Nothing else is saved for the
    object:
    the floor and the tones are the two specifications a random and
    sine project holds, and `MixedSpecification(floor, tones)` puts them
    back together.

    Raises ValueError when the tones' control channels are not the
    floor's, or when a fixed tone is given as a row without `seconds`.

    Attributes:
        floor: The random specification, carrying the narrowbands and
            the width left out.
        tones: The sine specification, or None.
    """

    def __init__(self, floor: Any,
                 tones: SineSweepSpecification | Sequence[Tone] | None = None,
                 narrowbands: Sequence[Narrowband] | None = None,
                 exclusion: float | None = None, *,
                 seconds: float | None = None) -> None:
        if narrowbands is not None or exclusion is not None:
            # written onto a copy, so a project's floor is never changed
            # by asking a question of it; the copy shares its arrays
            floor = copy.copy(floor)
        if narrowbands is not None:
            floor.narrowbands = tuple(narrowbands)
        if exclusion is not None:
            if not float(exclusion) >= 0.0:
                raise ValueError('the width left out is zero or more Hz')
            floor.tone_exclusion = float(exclusion)
        self.floor: Any = floor
        channels = [str(dof) for dof in floor.response_dof]
        if tones is not None and not isinstance(tones,
                                                SineSweepSpecification):
            rows = list(tones)
            tones = (SineSweepSpecification(
                [row.sine_tone(seconds, len(channels)) for row in rows],
                channels, ordinate_unit='m/s**2') if rows else None)
        if tones is not None:
            foreign = [dof for dof in tones.response_dof
                       if str(dof) not in channels]
            if foreign:
                raise ValueError(f'the tones are controlled at {foreign}, '
                                 f'which the floor ({channels}) is not '
                                 'written at')
        self.tones: SineSweepSpecification | None = tones

    # ---- what it holds ------------------------------------------------

    @property
    def narrowbands(self) -> tuple[Narrowband, ...]:
        """The narrowbands, as the floor carries them."""
        return tuple(getattr(self.floor, 'narrowbands', ()) or ())

    @property
    def exclusion(self) -> float:
        """The width left out of the floor either side of each tone, Hz:
        the floor's, or `EXCLUSION` when it does not say."""
        width = getattr(self.floor, 'tone_exclusion', None)
        return EXCLUSION if width is None else float(width)

    @property
    def channels(self) -> list[str]:
        """The control channels, the floor's records."""
        return [str(dof) for dof in self.floor.response_dof]

    def record(self, channel: str | None = None) -> int:
        """The floor's record for `channel` (its first by default);
        ValueError for a channel it is not written at."""
        if channel is None:
            return 0
        channels = self.channels
        if str(channel) not in channels:
            raise ValueError(f'the floor is not written at {channel}; it is '
                             f'written at {channels}')
        return channels.index(str(channel))

    def column(self, channel: str | None = None) -> int | None:
        """The tones' amplitude column for `channel` (the floor's first
        by default), or None when there are no tones or they are not
        controlled there."""
        if self.tones is None:
            return None
        dof = self.channels[self.record(channel)]
        names = [str(d) for d in self.tones.response_dof]
        return names.index(dof) if dof in names else None

    def tone_table(self, channel: str | None = None) -> list[Tone]:
        """The tones as tone-table rows (`Tone.of`) at `channel`; empty
        without tones."""
        column = self.column(channel)
        if column is None:
            return []
        return [Tone.of(tone, column) for tone in self.tones.tones]

    # ---- reading it onto lines ------------------------------------------

    def floor_at(self, frequencies: Any, channel: str | None = None
                 ) -> np.ndarray:
        """The floor's density at `frequencies`, read between its
        breakpoints as the floor is drawn (log-log), NaN where it is not
        written."""
        from .compliance import log_interpolate

        row = np.real(np.atleast_2d(self.floor.ordinate)[self.record(channel)])
        return log_interpolate(frequencies, self.floor.abscissa, row)

    def reference(self, frequencies: Any, channel: str | None = None
                  ) -> np.ndarray:
        """The composite reference density at `frequencies`: the floor,
        with each fixed narrowband's level added between its edges. A
        swept band and the tones have no fixed place on a density and are
        not added."""
        f = np.asarray(frequencies, dtype=np.float64)
        out = self.floor_at(f, channel)
        for band in self.narrowbands:
            if not band.swept:
                low, high = band.edges()
                out = np.where((f >= low) & (f <= high), out + band.level,
                               out)
        return out

    def left_out(self, frequencies: Any, channel: str | None = None
                 ) -> np.ndarray:
        """Which of `frequencies` are left out of the floor for a tone:
        those a tone at `channel` spent at least `LEFT_OUT_SHARE` of its
        run within `exclusion` of (`held_share`). Every line beside a
        fixed tone; of a sweep, only the lines it lingers near."""
        f = np.asarray(frequencies, dtype=np.float64)
        out = np.zeros(f.shape, dtype=bool)
        if self.column(channel) is None:
            return out
        for tone in self.tones.tones:
            out |= held_share(tone, f, self.exclusion) >= LEFT_OUT_SHARE
        return out

    def band_lines(self, frequencies: Any) -> np.ndarray:
        """Which of `frequencies` any narrowband covers, over its whole
        sweep: the lines the floor is not judged on, since they hold a
        band."""
        f = np.asarray(frequencies, dtype=np.float64)
        out = np.zeros(f.shape, dtype=bool)
        for band in self.narrowbands:
            low, high = band.span()
            out |= (f >= low) & (f <= high)
        return out

    def reference_rms(self, channel: str | None = None) -> dict[str, float]:
        """The written levels of each part as rms, from the
        specification alone: 'floor' (the floor's own area,
        `compliance.specification_rms`), 'tones' (the root of the tones'
        mean squares), 'narrowbands' (the root of the bands' level times
        width) and 'total' (the root of all of them), in SI."""
        from .compliance import specification_rms

        floor = specification_rms(self.floor, self.record(channel))
        column = self.column(channel)
        tones = 0.0 if column is None else sum(
            tone_mean_square(tone, column) for tone in self.tones.tones)
        bands = sum(band.mean_square() for band in self.narrowbands)
        return {'floor': floor, 'tones': float(np.sqrt(tones)),
                'narrowbands': float(np.sqrt(bands)),
                'total': float(np.sqrt(floor ** 2 + tones + bands))}

    def __repr__(self) -> str:
        tones = 0 if self.tones is None else len(self.tones.tones)
        return (f'<MixedSpecification: floor at {self.channels}, {tones} '
                f'tones, {len(self.narrowbands)} narrowbands, '
                f'{self.exclusion:g} Hz left out>')


def mixed(floor: Any, tones: Any = None, **options: Any) -> MixedSpecification:
    """`floor` as a `MixedSpecification`, whether it is one already or
    the random specification of one; `tones` and the options as for the
    class, applied only when `floor` is not one already."""
    if isinstance(floor, MixedSpecification):
        return floor
    return MixedSpecification(floor, tones, **options)


# ---- the data sheet ---------------------------------------------------------

@dataclass(frozen=True)
class FloorRow:
    """The floor's row of the data sheet, in SI.

    `reference` and `measured` are the rms over the lines judged (the
    floor's band, less the lines left out for tones and narrowbands), of
    the reference and of the residual. `residual` is the residual's rms
    over every line, tones taken out and nothing left out. `rms_percent`
    and `db` are the two tolerances; `judged` and `outside` count the
    lines judged and those outside `db`, `expected` the lines chance
    puts outside `db` at `dof` and `allowed` the most that may be
    outside before the lines fail (`FLOOR_CHANCE`). `left_out` is the
    share of the floor's lines left out, `exclusion` the width left out
    either side of a tone, `spacing` the line spacing and `dof` each
    line's statistical degrees of freedom.
    """

    reference: float
    measured: float
    residual: float
    rms_percent: float
    db: float
    judged: int
    outside: int
    left_out: float
    exclusion: float
    spacing: float
    dof: float
    expected: float = 0.0
    allowed: int = 0

    @property
    def error(self) -> float:
        """The measured rms over the reference, less one, in percent."""
        return 100.0 * (self.measured / self.reference - 1.0) \
            if self.reference > 0.0 else float('nan')

    @property
    def assessed(self) -> bool:
        """Whether any line was left to judge the floor on."""
        return self.judged > 0

    @property
    def passed(self) -> bool | None:
        """Whether the rms is within its tolerance and no more lines are
        outside theirs than chance allows; None when nothing was left to
        judge."""
        if not self.assessed:
            return None
        return bool(abs(self.error) <= self.rms_percent
                    and self.outside <= self.allowed)


@dataclass(frozen=True)
class ToneRow:
    """One tone's row of the data sheet, in SI.

    `reference` and `measured` are the tone's amplitude (peak), written
    and read: for a swept tone the median, over the readings, of the
    level written at each and of the level read. `readings` counts the
    readings used (those after the band settled) and `inside` the share
    of them within the tolerance of the level written where each was
    read. `tolerance` is percent either way; `low` and `high` the
    frequencies the tone passes; `band` the tracking band in words.
    """

    name: str
    low: float
    high: float
    reference: float
    measured: float
    tolerance: float
    readings: int
    inside: float
    band: str

    @property
    def fixed(self) -> bool:
        """Whether the tone stands at one frequency."""
        return self.high - self.low <= DWELL_SPAN * self.high

    @property
    def error(self) -> float:
        """The measured amplitude over the written, less one, percent."""
        return 100.0 * (self.measured / self.reference - 1.0) \
            if self.reference > 0.0 else float('nan')

    @property
    def passed(self) -> bool:
        """Whether the tone's level is within its tolerance."""
        return bool(self.readings > 0 and abs(self.error) <= self.tolerance)


@dataclass(frozen=True)
class BandRow:
    """One narrowband's row of the data sheet, in SI.

    `low` and `high` are its edges at its own center (a swept band's
    `span` is where it travels). `reference` and `measured` are its rms
    above the floor: the root of its level times its width, and of the
    residual's lines between its edges less the floor, times the
    spacing (a fixed band; a swept band's slices give the median of
    each slice's measured level times its width). `error_db` is the
    band's level read over the floor's plus the band's, in dB (over all
    its slices together for a swept band); `low_db` and `high_db` the
    tolerance it was judged against, the stated or the floor's, widened
    to the scatter at `dof`, the degrees of freedom of its estimate (all
    its slices' together for a swept band). `slices` counts the slices
    of a swept band (one for a fixed band), `slice_dof` what each was
    worth (the median), and `inside` the share of slices within the
    tolerance widened to their own scatter, which says how far a swept
    band wandered from moment to moment.
    """

    name: str
    low: float
    high: float
    swept: bool
    span: tuple[float, float]
    reference: float
    measured: float
    error_db: float
    low_db: float
    high_db: float
    dof: float
    slices: int
    slice_dof: float
    inside: float

    @property
    def passed(self) -> bool:
        """Whether the band's level is within its tolerance."""
        return bool(self.slices
                    and self.low_db <= self.error_db <= self.high_db)


@dataclass(frozen=True)
class TotalRow:
    """The total's row: the rms of the whole drive, which has no
    tolerance. `reference` is the root of every part's written mean
    square, `measured` the composite's every line over the span read,
    and `record` the record's own rms over the same samples."""

    reference: float
    measured: float
    record: float


@dataclass(frozen=True, eq=False)
class MixedReading:
    """A record read against a mixed specification: what `judge_mixed`
    returns.

    The four parts of the data sheet (`floor`, `tones`, `narrowbands`,
    `total`) and the tones together (`tones_reference`,
    `tones_measured`, rms), with what they were read from:
    `frequencies` and the averaged densities of the record over the span
    read (`composite`), of what is left with the tones taken out
    (`residual`) and of the reference (`reference`, floor and fixed
    bands); `left_out` and `band_lines` mark the lines left out of the
    floor for the tones and held by narrowbands. `residual_history` is
    the residual as a record, `averaging` how its spectrum was averaged,
    `start` the second the span read begins (the band settled), `onset`
    where the tones began in the record. `dimension` is the record's
    quantity, `channel` its DOF, and `notes` what the reading says
    rather than hides. Every value is in SI.
    """

    channel: str
    dimension: str
    floor: FloorRow
    tones: tuple[ToneRow, ...]
    narrowbands: tuple[BandRow, ...]
    total: TotalRow
    tones_reference: float
    tones_measured: float
    frequencies: np.ndarray
    composite: np.ndarray
    residual: np.ndarray
    reference: np.ndarray
    left_out: np.ndarray
    band_lines: np.ndarray
    residual_history: Any
    averaging: Any
    start: float
    onset: float
    notes: tuple[str, ...] = field(default=())

    @property
    def passed(self) -> bool:
        """Whether every judged part passes: the floor when any line was
        left to judge it on, each tone and each narrowband. The total is
        not judged."""
        return bool(self.floor.passed is not False
                    and all(t.passed for t in self.tones)
                    and all(b.passed for b in self.narrowbands))

    def describe(self, unit: str = '', scale: float = 1.0) -> str:
        """The reading in one line: the floor, the tones and the total,
        each labeled, and what failed. `scale` converts SI into `unit`
        ('g' with 1/9.80665)."""
        rms = f'{unit} rms' if unit else 'rms'
        parts = [(f'floor {self.floor.measured * scale:.3g} {rms} '
                 f'(written {self.floor.reference * scale:.3g})')
                 if self.floor.assessed else 'floor not judged']
        if self.tones:
            parts.append(f'{len(self.tones)} tone'
                         f'{"" if len(self.tones) == 1 else "s"} '
                         f'{self.tones_measured * scale:.3g} {rms}')
        if self.narrowbands:
            parts.append(f'{len(self.narrowbands)} narrowband'
                         f'{"" if len(self.narrowbands) == 1 else "s"}')
        parts.append(f'total {self.total.measured * scale:.3g} {rms}')
        failed = ([] if self.floor.passed is not False else ['the floor']) \
            + [t.name for t in self.tones if not t.passed] \
            + [b.name for b in self.narrowbands if not b.passed]
        verdict = ('every part passes' if not failed
                   else 'outside: ' + ', '.join(failed))
        return f'{self.channel}: ' + ', '.join(parts) + f'; {verdict}'


# ---- reading a record against it ----------------------------------------------

def settling_seconds(band: Any, tones: Sequence[SineTone] = ()) -> float:
    """How long a tracking band takes to settle: about one over its
    width, at the lowest frequency of any tone for a proportional band
    (2 s for a 0.5 Hz band); zero for no band."""
    kind = getattr(band, 'kind', 'unfiltered')
    if kind == 'fixed':
        return 1.0 / float(band.fixed)
    if kind == 'proportional':
        lowest = min((tone_span(t)[0] for t in tones), default=1.0)
        return 1.0 / (float(band.proportional) * lowest)
    return 0.0


def _channel_history(history: Any, channel: str, values: np.ndarray | None
                     = None) -> Any:
    """One channel of `history` as a record of its own, with other
    `values` in its place when given."""
    from .data import TimeHistory

    dofs = [str(d) for d in history.response_dof]
    row = dofs.index(channel)
    y = np.real(np.asarray(history.ordinate[row], dtype=np.float64)) \
        if values is None else values
    return TimeHistory(np.asarray(history.abscissa, dtype=np.float64),
                       y[None], response_dof=[channel],
                       ordinate_dim=[history.ordinate_dim[row]],
                       ordinate_unit=[history.ordinate_unit[row]])


def _averaging_from(history: Any, averaging: Any, start: float,
                    rate: float, samples: int) -> Any:
    """The averaging the residual is read with: the one given, the
    record's own, or half-overlapped Hann frames at 1 Hz lines or finer;
    starting where the span read starts and filling the record."""
    from dataclasses import replace

    from .averaging import Averaging

    if averaging is None:
        averaging = getattr(history, 'averaging', None)
    if averaging is None:
        length = int(2 ** np.ceil(np.log2(max(rate, 2.0))))
        averaging = Averaging(frame_length=length, overlap=0.5,
                              window='hann')
    begin = max(float(averaging.start), float(start))
    used = replace(averaging, start=begin, start_set=True)
    most = used.most_frames(samples, rate)
    if most < 1:
        raise ValueError(f'a frame of {used.frame_length} samples does not '
                         f'fit after {begin:g} s of the record')
    return replace(used, frames=most)


def _outside_by_chance(dof: float, db: float) -> float:
    """The chance that a line averaged to `dof` degrees of freedom, its
    truth on the reference, reads more than `db` dB from it either way:
    the chi-square's two tails, computed directly so a small share keeps
    its digits."""
    from scipy.special import gammainc, gammaincc

    half = 0.5 * max(float(dof), 1e-6)
    return float(gammainc(half, half * 10.0 ** (-float(db) / 10.0))
                 + gammaincc(half, half * 10.0 ** (float(db) / 10.0)))


def _judge_floor(spec, channel, residual_psd, lines_kept, floor_db):
    """The floor judged as a comparison is (`compliance.judge`): the
    reference, and the reference moved `floor_db` either way, against the
    residual, cell by cell. Returns (measured power, reference power,
    judged cells, cells out) over the cells whose line is in
    `lines_kept`."""
    from .compliance import judge

    record = spec.record(channel)
    bounded = copy.copy(spec.floor)
    level = np.real(np.atleast_2d(np.asarray(spec.floor.ordinate)))
    factor = 10.0 ** (float(floor_db) / 10.0)
    bounded.limits = {'warning_lower': level / factor,
                      'warning_upper': level * factor}
    target = judge(bounded, residual_psd, record, 0, None, True, 0)
    upper = judge(bounded, residual_psd, record, 0, 'warning_upper', True, 0)
    lower = judge(bounded, residual_psd, record, 0, 'warning_lower', False, 0)
    left, _right = residual_psd.bin_bounds()
    middles = np.array([0.5 * (cell['low'] + cell['high'])
                        for cell in target['cells']])
    if not len(middles):
        empty = np.zeros(0, dtype=bool)
        return 0.0, 0.0, empty, empty
    line = np.clip(np.searchsorted(left, middles, side='right') - 1, 0,
                   len(lines_kept) - 1)
    judged = target['judged'] & lines_kept[line]
    out = judged & (upper['out'] | lower['out'])
    return (float(np.sum(target['held'][judged])),
            float(np.sum(target['asked'][judged])), judged, out)


def _scatter_band(tolerance: float, dof: float, confidence: float
                  ) -> tuple[float, float]:
    """(low dB, high dB): the band an estimate at `dof` degrees of
    freedom is judged in, `tolerance` dB either way, widened to the
    chi-square scatter of the estimate about the truth at `confidence`.
    An estimate is the truth times chi-square over its degrees of
    freedom, so it lies, at that confidence, between the truth times the
    inverse of `chi_square_factors`' upper and lower factors."""
    from .averaging import chi_square_factors

    lower, upper = chi_square_factors(max(float(dof), 1e-6), confidence)
    return (min(-float(tolerance), -_db(upper)),
            max(float(tolerance), -_db(lower)))


def _judge_tone(history, spec, tone, column, band, onset, start, tolerance,
                channel):
    """One tone's row, and what its band passed as (first sample,
    waveform)."""
    from .sine_tracking import track_sine, track_waveform

    tone_onset = float(onset) + float(tone.start_time)
    level = track_sine(history, spec, band, tone=tone.name,
                       onset=tone_onset, channels=[channel])[0]
    read = np.abs(np.asarray(level.ordinate[0])).astype(np.float64)
    written = tone.target(level.abscissa)[:, column]
    settled = np.asarray(level.seconds) >= start
    if not settled.any():
        settled = np.ones(read.shape, dtype=bool)
    read, written = read[settled], written[settled]
    good = np.isfinite(written) & (written > 0.0)
    measured = float(np.median(read[good])) if good.any() else float('nan')
    reference = float(np.median(written[good])) if good.any() else float('nan')
    inside = (float(np.mean(np.abs(read[good] / written[good] - 1.0)
                            <= tolerance / 100.0)) if good.any() else 0.0)
    low, high = tone_span(tone)
    row = ToneRow(name=tone.name, low=low, high=high, reference=reference,
                  measured=measured, tolerance=float(tolerance),
                  readings=int(good.sum()), inside=inside,
                  band=band.describe())
    wave = track_waveform(history, spec, band, tone=tone.name,
                          onset=tone_onset, channel=channel)
    return row, wave


def _fixed_band(band, f, residual, floor, dof_line, bandwidth, db,
                confidence, notes):
    """A fixed narrowband's row, read off the residual's lines between
    its edges."""
    low, high = band.edges()
    lines = (f >= low) & (f <= high) & np.isfinite(floor)
    spacing = float(f[1] - f[0])
    k = int(lines.sum())
    tolerance = band.tolerance_db if band.tolerance_db is not None else db
    if not k:
        notes.append(f'{band.label}: no line of the spectrum lies between '
                     'its edges over the floor; it is not judged')
        return BandRow(band.label, low, high, False, band.span(), band.rms,
                       float('nan'), float('nan'), -tolerance, tolerance,
                       0.0, 0, 0.0, 0.0)
    got = float(np.mean(residual[lines]))
    wanted = float(np.mean(floor[lines])) + band.level
    above = float(np.sum(residual[lines] - floor[lines])) * spacing
    dof = dof_line * max(1.0, k / bandwidth)
    low_db, high_db = _scatter_band(tolerance, dof, confidence)
    error = _db(got / wanted)
    return BandRow(band.label, low, high, False, band.span(), band.rms,
                   float(np.sqrt(max(above, 0.0))), error, low_db, high_db,
                   dof, 1, dof, float(low_db <= error <= high_db))


def _swept_band(band, residual_history, spec, channel, rate, onset, start,
                db, confidence, notes):
    """A swept narrowband's row, read in slices of the run half as long
    as the band takes to move its own width."""
    from .averaging import Averaging
    from .tone_readout import noise_bandwidth

    c_low, c_high = sorted((band.center, float(band.sweep_to)))
    centers = np.geomspace(c_low, c_high, 32)
    slice_seconds = 0.5 * min(band.crossing_seconds(c) for c in centers)
    widths = [band.width_at(c) for c in centers]
    narrowest = min(widths)
    t = np.asarray(residual_history.abscissa, dtype=np.float64)
    samples = len(t)
    slice_samples = int(slice_seconds * rate)
    wanted_frame = int(2 ** np.ceil(np.log2(8.0 * rate / narrowest)))
    frame = wanted_frame
    if frame > slice_samples:
        frame = int(2 ** np.floor(np.log2(max(slice_samples, 2))))
        notes.append(f'{band.label}: the band moves half its width in '
                     f'{slice_seconds:.3g} s, shorter than a frame fine '
                     'enough to resolve it; each slice is read at '
                     f'{rate / frame:.3g} Hz lines and may read low')
    hop = max(frame // 2, 1)
    tolerance = band.tolerance_db if band.tolerance_db is not None else db
    begin = int(np.ceil(start * rate))
    errors, dofs, inside, levels = [], [], [], []
    got_all, wanted_all = [], []
    bandwidth = noise_bandwidth('hann', frame)
    step = max(slice_samples, frame)
    for first in range(begin, samples - step + 1, step):
        frames = 1 + (step - frame) // hop
        averaging = Averaging(frame_length=frame, overlap=0.5,
                              window='hann', frames=frames,
                              start=float(t[first] - t[0]), start_set=True)
        asd = residual_history.compute_psds(averaging)
        f = np.asarray(asd.abscissa, dtype=np.float64)
        g = np.real(np.asarray(asd.ordinate[0], dtype=np.float64))
        middle = t[first] - t[0] + 0.5 * step / rate - onset
        center = float(band.center_at(middle))
        quarter = 0.25 * band.width_at(center)
        floor = spec.floor_at(f, channel)
        lines = (np.abs(f - center) <= quarter) & np.isfinite(floor)
        k = int(lines.sum())
        if not k:
            continue
        got = float(np.mean(g[lines]))
        wanted = float(np.mean(floor[lines])) + band.level
        dof = averaging.statistical_dof().equivalent * max(1.0, k / bandwidth)
        low_db, high_db = _scatter_band(tolerance, dof, confidence)
        error = _db(got / wanted)
        errors.append(error)
        got_all.append(got * dof)
        wanted_all.append(wanted * dof)
        dofs.append(dof)
        inside.append(low_db <= error <= high_db)
        levels.append(max(got - float(np.mean(floor[lines])), 0.0)
                      * band.width_at(center))
    low, high = band.edges()
    if not errors:
        notes.append(f'{band.label}: the record holds no slice of the sweep '
                     'with a line under the band; it is not judged')
        return BandRow(band.label, low, high, True, band.span(), band.rms,
                       float('nan'), float('nan'), -tolerance, tolerance,
                       0.0, 0, 0.0, 0.0)
    # the band's level over the whole sweep: the slices pooled, each by
    # the degrees of freedom it carries, which is the estimate the band
    # is judged by; a swept band gathers few of them a slice
    dof = float(np.sum(dofs))
    low_db, high_db = _scatter_band(tolerance, dof, confidence)
    return BandRow(band.label, low, high, True, band.span(), band.rms,
                   float(np.sqrt(np.median(levels))),
                   _db(float(np.sum(got_all)) / float(np.sum(wanted_all))),
                   low_db, high_db, dof, len(errors),
                   float(np.median(dofs)), float(np.mean(inside)))


def judge_mixed(history: Any, specification: Any, tones: Any = None, *,
                channel: str | None = None, band: Any = None,
                onset: float | None = None, settle: float | None = None,
                averaging: Any = None,
                tone_tolerance: float = TONE_TOLERANCE,
                floor_db: float = FLOOR_DB, floor_rms: float = FLOOR_RMS,
                confidence: float = CONFIDENCE) -> MixedReading:
    """Read one channel of a record against a mixed specification, part
    by part: the data sheet's floor, tones, narrowbands and total.

    The tones are tracked and taken out, the floor is judged on what is
    left with the tone and band lines left out, each narrowband at its
    own degrees of freedom, and the total is reported unjudged; the
    module's notes give the method.

    Parameters
    ----------
    history : TimeHistory
        The record, evenly sampled.
    specification : MixedSpecification or Specification
        The requirement, or its floor, with `tones` beside it.
    tones : SineSweepSpecification, optional
        The tones, when `specification` is the floor alone.
    channel : str, optional
        The control channel read; the floor's first by default.
    band : SineTracking, optional
        The tracking band the tones are read through; a fixed band
        `TRACKING_BAND` Hz wide by default.
    onset : float, optional
        Seconds into the record where the tones' environment begins;
        each tone starts its own `start_time` later. Found by matched
        filter when omitted (`core.sine.find_environment`), as the
        extraction finds it; zero without tones.
    settle : float, optional
        Seconds from `onset` before which nothing is read, so the band
        has settled; `settling_seconds` of the band by default, and
        none without tones.
    averaging : Averaging, optional
        How the residual's spectrum is averaged; the record's own, or
        half-overlapped Hann frames at 1 Hz lines or finer, by default.
        Its frames start no earlier than the span read and fill the
        record.
    tone_tolerance : float, default `TONE_TOLERANCE`
        Percent either way on each tone's amplitude.
    floor_db : float, default `FLOOR_DB`
        dB either way on each line of the floor, and the narrowbands'
        tolerance where the plan states none.
    floor_rms : float, default `FLOOR_RMS`
        Percent either way on the floor's rms.
    confidence : float, default `CONFIDENCE`
        The confidence a narrowband's scatter is read at.

    Returns
    -------
    MixedReading
        The data sheet and the spectra it was read from, in SI.

    Raises
    ------
    ValueError
        For a channel the record or the floor does not hold, a record
        of another quantity than the floor or the tones, a record not
        evenly sampled, or a span too short for one frame.
    """
    from scipy.stats import binom

    from ..units import UNKNOWN
    from .compliance import rms, units_check
    from .sine import find_environment
    from .sine_tracking import SineTracking
    from .tone_readout import noise_bandwidth

    spec = mixed(specification, tones)
    channel = spec.channels[0] if channel is None else str(channel)
    spec.record(channel)
    dofs = [str(d) for d in history.response_dof]
    if channel not in dofs:
        raise ValueError(f'the record holds no channel {channel}; it holds '
                         f'{dofs}')
    if band is None:
        band = SineTracking(fixed=TRACKING_BAND)
    t = np.asarray(history.abscissa, dtype=np.float64)
    rate = float(history.sample_rate)
    row = dofs.index(channel)
    x = np.real(np.asarray(history.ordinate[row], dtype=np.float64))
    notes: list[str] = []
    column = spec.column(channel)
    tone_list = [] if column is None else list(spec.tones.tones)
    # the record against what it had to meet, by the one rule every
    # comparison uses: a velocity record says nothing about an
    # acceleration floor
    held = str(history.known_dim(row))
    density = UNKNOWN if held == UNKNOWN else f'{held}**2/frequency'
    refused, note = units_check(
        str(spec.floor.known_dim(spec.record(channel))), density,
        wanted_name='the floor', held_name='the record')
    if refused is None and tone_list:
        asked = (UNKNOWN if getattr(spec.tones, 'ordinate_unit', None) is None
                 else str(spec.tones.ordinate_dim))
        refused, _said = units_check(asked, held, wanted_name='the tones',
                                     held_name='the record')
    if refused is not None:
        raise ValueError(refused)
    if note is not None:
        notes.append(note)
    if onset is None:
        onset = (find_environment([x], 1.0 / rate, tone_list) if tone_list
                 else 0.0)
    if settle is None:
        # nothing to settle when there is no tone to track
        settle = settling_seconds(band, tone_list) if tone_list else 0.0
    start = float(onset) + float(settle)

    # 1 and 2: each tone tracked, its band's waveform taken out
    residual = x.copy()
    tone_rows = []
    for tone in tone_list:
        tone_row, wave = _judge_tone(history, spec.tones, tone, column, band,
                                     onset, start, tone_tolerance, channel)
        tone_rows.append(tone_row)
        first = round((wave.time[0] - t[0]) * rate)
        residual[first:first + len(wave.passed)] -= wave.passed
    one = _channel_history(history, channel)
    left = _channel_history(history, channel, residual)
    used = _averaging_from(history, averaging, start, rate, len(t))
    composite_psd = one.compute_psds(used)
    residual_psd = left.compute_psds(used)
    f = np.asarray(residual_psd.abscissa, dtype=np.float64)
    g_res = np.real(np.asarray(residual_psd.ordinate[0], dtype=np.float64))
    g_all = np.real(np.asarray(composite_psd.ordinate[0], dtype=np.float64))
    dof_line = float(used.statistical_dof().equivalent)

    # 3: the floor, tone and band lines left out
    floor = spec.floor_at(f, channel)
    written = np.isfinite(floor)
    tone_lines = spec.left_out(f, channel)
    band_held = spec.band_lines(f)
    if not written.any():
        raise ValueError('the floor is not written over any line of the '
                         'spectrum')
    power, asked_power, judged, outside = _judge_floor(
        spec, channel, residual_psd, ~tone_lines & ~band_held, floor_db)
    count = int(judged.sum())
    # what chance puts outside the line tolerance at each line's degrees
    # of freedom, and the most that may be outside before the lines fail
    chance = _outside_by_chance(dof_line, floor_db)
    allowed = int(binom.ppf(FLOOR_CHANCE, count, chance)) if count else 0
    share = float(np.mean((tone_lines | band_held)[written]))
    floor_row = FloorRow(
        reference=float(np.sqrt(asked_power)), measured=float(np.sqrt(power)),
        residual=rms(f, g_res), rms_percent=float(floor_rms),
        db=float(floor_db), judged=count,
        outside=int(outside.sum()), left_out=share,
        exclusion=spec.exclusion, spacing=float(f[1] - f[0]), dof=dof_line,
        expected=count * chance, allowed=allowed)
    if not count:
        notes.append('no line of the floor is left to judge once the tones '
                     'and narrowbands are left out; the floor is not judged')
    elif share > 0.5:
        notes.append(f'{100 * share:.0f} % of the floor\'s lines are left '
                     'out for tones and narrowbands; what is left is judged '
                     'and the rest is not')
    for tone in tone_list:
        low, high = tone_span(tone)
        if not np.isfinite(spec.floor_at([low, high], channel)).all():
            notes.append(f'{tone.name}: the floor is not written under all '
                         'of it')

    # 4: each narrowband at its own degrees of freedom
    bandwidth = noise_bandwidth(used, used.frame_length) \
        if hasattr(used, 'window') else 1.5
    band_rows = []
    for narrow in spec.narrowbands:
        if narrow.swept:
            band_rows.append(_swept_band(narrow, left, spec, channel, rate,
                                         float(onset), start, floor_db,
                                         confidence, notes))
        else:
            band_rows.append(_fixed_band(narrow, f, g_res, floor, dof_line,
                                         bandwidth, floor_db, confidence,
                                         notes))

    # 5: the total, unjudged; and the tones together
    written_rms = spec.reference_rms(channel)
    tones_measured = float(np.sqrt(sum(0.5 * r.measured ** 2
                                       for r in tone_rows)))
    tones_reference = float(np.sqrt(sum(0.5 * r.reference ** 2
                                        for r in tone_rows)))
    first = used.start_sample(rate)
    last = first + used.hop * (used.frames - 1) + used.frame_length
    total = TotalRow(reference=written_rms['total'],
                     measured=rms(f, g_all),
                     record=float(np.sqrt(np.mean(x[first:last] ** 2))))
    return MixedReading(
        channel=channel, dimension=str(history.ordinate_dim[row]),
        floor=floor_row, tones=tuple(tone_rows),
        narrowbands=tuple(band_rows), total=total,
        tones_reference=tones_reference, tones_measured=tones_measured,
        frequencies=f, composite=g_all, residual=g_res,
        reference=spec.reference(f, channel), left_out=tone_lines,
        band_lines=band_held, residual_history=left, averaging=used,
        start=start, onset=float(onset), notes=tuple(notes))


__all__ = [
    'CONFIDENCE',
    'EXCLUSION',
    'FLOOR_CHANCE',
    'FLOOR_DB',
    'FLOOR_RMS',
    'LEFT_OUT_SHARE',
    'TONE_TOLERANCE',
    'TRACKING_BAND',
    'BandRow',
    'FloorRow',
    'MixedReading',
    'MixedSpecification',
    'Narrowband',
    'Tone',
    'ToneRow',
    'TotalRow',
    'held_share',
    'judge_mixed',
    'mixed',
    'narrowbands_from_json',
    'narrowbands_json',
    'settling_seconds',
    'tone_mean_square',
    'tone_span',
]
