"""Judge a measured classical shock pulse against the pulse it was meant
to be.

A classical pulse is written as three things: a shape, a peak and a
duration. Three shapes are in common use. The half-sine is half a cycle
of a sine, the shape a drop onto a resilient pad makes. The
terminal-peak sawtooth rises in a straight line over the duration and
falls back to zero quickly; the sharp fall sets every resonator
ringing, which is why it is chosen where a flat spectrum is wanted. The
trapezoid rises, holds a flat top and falls. A sawtooth's fall, and a
trapezoid's rise and fall, are its edges, and they are part of what is
asked for.

A test document states the pulse with its tolerance, and the
measurement is judged against that, point by point and as a whole. What
test documents commonly ask for, and the defaults here (each is an
argument, and the document the procedure cites decides its values):

- **The band on the pulse.** The measured pulse stays inside a band
  drawn around the ideal shape, a fifth of the peak either side of it
  (`BAND`), so a 40 g pulse peaks between 32 and 48 g.
- **The duration**, within a tenth of nominal either way
  (`DURATION_TOLERANCE`): 9.9 to 12.1 ms for an 11 ms pulse.
- **The edges.** A sawtooth's fall, and a trapezoid's rise and fall,
  take no longer than a tenth of the duration (`EDGE_LIMIT`).
- **The quiet either side.** For two and a half durations before the
  pulse and after it (`QUIET`), the acceleration stays inside the same
  fifth of the peak. Before the pulse is where a shaker puts its
  compensation; after it is where a fixture's ringing shows.
- **The velocity change**, the area under the pulse, integrated from
  0.4 durations before the pulse to 0.1 after it (`VELOCITY_WINDOW`),
  within a fifth of the nominal pulse's (`VELOCITY_TOLERANCE`). The
  window is part of the rule: a shaker's compensated pulse changes
  velocity by nothing over the whole record, by design, so the area is
  taken around the main pulse only. The fifth is this package's
  default, not a document's figure; documents and their revisions
  differ on this tolerance, so it is set from the one the procedure
  cites.

The nominal velocity change is the area formula of each shape, with
`A` the peak and `T` the duration: ``2AT/π`` for the half-sine,
``AT/2`` for the sawtooth (with its fall taken as instant, as the
formula is usually stated) and ``A(T - e)`` for a trapezoid whose rise
and fall each take ``e``. A sawtooth whose fall takes its whole
allowance therefore changes velocity by about a tenth more than its
nominal, half of a fifth's tolerance spent before anything else varies,
and the check reports it as it is rather than moving the nominal.

How the envelope is drawn
-------------------------
Documents draw the envelope as a figure, and the shapes of their lines
at the start and end of the pulse differ. Here it is built one way for
all three shapes, so that it agrees with the separate duration and
edge tolerances rather than contradicting them: every pulse of the
shape whose duration is inside the duration tolerance and whose edges
are inside the edge limit is admissible, and the envelope is the
highest and lowest of them at each instant, widened by the band. A
pulse a tenth long is then inside the envelope and inside the duration
tolerance, and a sawtooth that falls in a tenth of its duration is
inside both as well. Away from the pulse every admissible shape is
zero, and the envelope is the band alone. The admissible family is
sampled (`FAMILY` durations by `FAMILY_EDGES` edges); the largest error
that leaves, at a half-sine's crest, is a few parts in a hundred
thousand of the peak.

How the measured pulse is read
------------------------------
The pulse is found by its largest excursion in the nominal's direction,
and its start, end and edges by where it crosses a tenth and nine
tenths of that peak on either side, walking out from the peak so that
noise before and after the pulse is not mistaken for it. The edges of a
sawtooth and a trapezoid are straight, so each is extended through its
two crossings to zero: the start, the end, and how long each edge took.
A half-sine has no straight edge, and its duration is the span between
its two tenth-of-peak crossings over the share of a half-sine's
duration that span covers, ``1 - 2 asin(0.1)/π``. The measurement
cannot resolve a time finer than a sample, so the duration and edge
tolerances are read with half a sample's grace.

The velocity change is integrated raw, by the trapezoid rule with
nothing removed (`filters.integrate` with ``drift_corner=None``), over
the window placed on the measured pulse: from 0.4 nominal durations
before its start to 0.1 after its end. A mean removed or a drift
high-pass applied first would take out exactly the net area being
measured. The integration needs evenly spaced samples, so a record
whose samples are not evenly spaced is refused, as is one holding a
sample that is not a number.

The velocity change here is the area over the window: the velocity
where the window closes, from rest where it opens. The shock check
(`core.shock_check`) reads a different number under the same name,
the span from the lowest velocity to the highest across its whole
window. For a lone one-signed pulse the two agree. They part where the
record swings the other way: an overshoot of the other sign after the
pulse, inside the window, takes from the area here and leaves the span
alone, and compensation that runs into the window is counted here and
not by the span.

References
----------
The check is a standard one, and its tolerances are the test
document's; these set out the pulses, their envelopes and their
velocity change.

1. IEC 60068-2-27:2008, *Environmental testing - Part 2-27: Tests -
   Test Ea and guidance: Shock*. The classical pulses, their tolerance
   envelopes, and the velocity change read over a window around the
   pulse.
2. Lalanne, C. (2009). *Mechanical Vibration and Shock Analysis,
   Volume 2: Mechanical Shock* (2nd ed.). Wiley. The simple shock
   shapes, their velocity changes and their shock response spectra.
3. Harris, C. M. and Piersol, A. G., eds. (2002). *Harris' Shock and
   Vibration Handbook* (5th ed.). McGraw-Hill. Shock testing machines
   and the pulses they are asked to make.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

# Asked for on 2026-10-16, for a worked example in which a 40 g, 11 ms
# sawtooth whose fall took its full allowance changed velocity by
# 93.4 in/s against the formula's 84.9: a tenth over, in tolerance, and
# half the way to the edge of it before anything else varied.
#
# 2026-10-16, on review: the velocity tolerance's default was described
# as what IEC 60068-2-27 commonly asks for, with older revisions at a
# tenth. Neither figure was checked against the standard, so the fifth
# is stated as this package's default and the attribution is gone; the
# value stands until a source says otherwise.

#: the shapes, as `ClassicalPulse` names them
SHAPES = ('half-sine', 'sawtooth', 'trapezoid')

#: the band either side of the ideal shape on the pulse, and either side
#: of zero before and after it, as a share of the peak
BAND = 0.2

#: how far the duration may sit from nominal, as a share of it
DURATION_TOLERANCE = 0.1

#: the longest a sawtooth's fall, or a trapezoid's rise or fall, may
#: take, as a share of the nominal duration
EDGE_LIMIT = 0.1

#: how far the measured velocity change may sit from the nominal pulse's,
#: as a share of it
VELOCITY_TOLERANCE = 0.2

#: the quiet held to the band before the pulse and after it, in durations
QUIET = 2.5

#: the velocity window, in durations before the pulse starts and after
#: it ends
VELOCITY_WINDOW = (0.4, 0.1)

#: durations, across the duration tolerance, the admissible family is
#: sampled at; odd, so the nominal is one of them
FAMILY = 41

#: edges, from instant to the edge limit, the family is sampled at
FAMILY_EDGES = 11

#: the levels the measured pulse's crossings are read at, as shares of
#: its peak: a tenth, clear of the noise either side of the pulse, and
#: nine tenths, clear of a flat top's ripple
LOW, HIGH = 0.1, 0.9

#: the share of a half-sine's duration between its two tenth-of-peak
#: crossings
_HALF_SINE_SPAN = 1.0 - 2.0 * float(np.arcsin(LOW)) / np.pi


@dataclass(frozen=True)
class ClassicalPulse:
    """A nominal classical pulse: its shape, peak, duration and edge.

    `peak` is signed, in SI (m/s² for an acceleration pulse), so a pulse
    in the negative direction is a negative peak. `duration` is in
    seconds. `edge` is the sawtooth's fall, after its rise over the
    duration, or the trapezoid's rise and its fall, each, inside the
    duration; a half-sine has none. A sawtooth's fall is instant unless
    given, and a trapezoid's edge has to be given.
    """

    #: one of `SHAPES`
    shape: str
    #: the peak, signed, in SI
    peak: float
    #: the duration, in seconds: the half-sine's half cycle, the
    #: sawtooth's rise, the trapezoid's rise, top and fall together
    duration: float
    #: the sawtooth's fall or the trapezoid's rise and fall, in seconds
    edge: float | None = None

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f'a classical pulse is one of '
                             f'{", ".join(SHAPES)}; {self.shape!r} is none '
                             'of them')
        if not np.isfinite(self.peak) or self.peak == 0.0:
            raise ValueError(f'the peak ({self.peak:g}) has to be a number '
                             'other than zero')
        if not self.duration > 0.0:
            raise ValueError(f'the duration ({self.duration:g} s) has to '
                             'be above zero')
        edge = self.edge
        if self.shape == 'half-sine':
            if edge:
                raise ValueError('a half-sine has no edge to set')
            edge = None
        elif self.shape == 'sawtooth':
            edge = 0.0 if edge is None else float(edge)
        elif edge is None:
            raise ValueError('a trapezoid needs its edge: how long its '
                             'rise and its fall each take')
        if edge is not None:
            if edge < 0.0:
                raise ValueError(f'the edge ({edge:g} s) cannot be '
                                 'negative')
            if self.shape == 'trapezoid' and not 2.0 * edge < self.duration:
                raise ValueError(
                    f'a trapezoid of {self.duration:g} s cannot rise and '
                    f'fall in {edge:g} s each and still have a top')
        object.__setattr__(self, 'peak', float(self.peak))
        object.__setattr__(self, 'duration', float(self.duration))
        object.__setattr__(self, 'edge', edge)

    @property
    def extent(self) -> float:
        """From the pulse's start to its end, in seconds: the duration,
        and the fall after it for a sawtooth."""
        if self.shape == 'sawtooth':
            return self.duration + float(self.edge or 0.0)
        return self.duration

    @property
    def velocity_change(self) -> float:
        """The nominal velocity change, in SI (m/s), signed.

        ``2AT/π`` for a half-sine, ``AT/2`` for a sawtooth with its fall
        taken as instant (the formula as it is usually stated, so a fall
        that takes time adds to the measured change and not to this),
        and ``A(T - e)`` for a trapezoid with edges of ``e``.
        """
        a, t = self.peak, self.duration
        if self.shape == 'half-sine':
            return 2.0 * a * t / np.pi
        if self.shape == 'sawtooth':
            return 0.5 * a * t
        return a * (t - float(self.edge))

    def at(self, time: ArrayLike) -> np.ndarray:
        """The nominal pulse at `time`, in seconds from its start.

        Zero before the start and after the end.

        Parameters
        ----------
        time : array_like
            Times, in seconds, measured from the pulse's start.

        Returns
        -------
        numpy.ndarray
            The pulse, in the units of `peak`.
        """
        return self.peak * _shape(self.shape, np.asarray(time, dtype=float),
                                  self.duration, self.edge)

    def describe(self, unit: str = 'm/s²', factor: float = 1.0) -> str:
        """The pulse in words, as '50 g, 11 ms half-sine'.

        `factor` scales the SI peak to `unit` (1/9.80665 for g).
        """
        text = f'{self.peak * factor:.4g} {unit}, {self.duration * 1e3:.4g} ms'
        if self.shape == 'sawtooth' and self.edge:
            text += f', {self.edge * 1e3:.3g} ms fall'
        elif self.shape == 'trapezoid':
            text += f', {self.edge * 1e3:.3g} ms edges'
        return f'{text} {self.shape}'


def _shape(shape: str, t: np.ndarray, duration: float,
           edge: float | None) -> np.ndarray:
    """A unit-peak pulse of `shape` at times `t` from its start."""
    inside = (t >= 0.0) & (t <= duration)
    if shape == 'half-sine':
        return np.where(inside, np.sin(np.pi * np.clip(t, 0.0, duration)
                                       / duration), 0.0)
    if shape == 'sawtooth':
        fall = float(edge or 0.0)
        rise = np.where(inside, t / duration, 0.0)
        if fall > 0.0:
            falling = (t > duration) & (t <= duration + fall)
            rise = np.where(falling, 1.0 - (t - duration) / fall, rise)
        return rise
    e = float(edge or 0.0)
    if e == 0.0:
        return inside.astype(float)
    ramp = np.minimum(t / e, (duration - t) / e)
    return np.where(inside, np.clip(ramp, 0.0, 1.0), 0.0)


def admissible_reach(pulse: ClassicalPulse,
                     duration_tolerance: float = DURATION_TOLERANCE,
                     edge_limit: float = EDGE_LIMIT) -> float:
    """The latest instant, in seconds from the start, any admissible
    pulse is still away from zero: the longest duration, and the longest
    fall after it for a sawtooth."""
    reach = pulse.duration * (1.0 + duration_tolerance)
    if pulse.shape == 'sawtooth':
        reach += edge_limit * pulse.duration
    return reach


def pulse_envelope(pulse: ClassicalPulse, time: ArrayLike, *,
                   band: float = BAND,
                   duration_tolerance: float = DURATION_TOLERANCE,
                   edge_limit: float = EDGE_LIMIT
                   ) -> tuple[np.ndarray, np.ndarray]:
    """The tolerance envelope of `pulse` at `time`.

    The highest and the lowest of every admissible pulse at each
    instant, widened by `band` of the peak either way: admissible is the
    shape at any duration inside `duration_tolerance` and, for the
    sawtooth and the trapezoid, any edge from instant to `edge_limit` of
    the nominal duration (the module docstring says why it is built
    this way). Away from the pulse it is the band either side of zero.

    Parameters
    ----------
    pulse : ClassicalPulse
        The nominal pulse.
    time : array_like
        Times, in seconds, measured from the pulse's start.
    band : float, default `BAND`
        The band's half-width, as a share of the peak.
    duration_tolerance : float, default `DURATION_TOLERANCE`
        How far the duration may sit from nominal, as a share of it.
    edge_limit : float, default `EDGE_LIMIT`
        The longest an edge may take, as a share of the duration.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The lower and the upper edge of the envelope, in the units of
        the peak.
    """
    t = np.asarray(time, dtype=float)
    reach = admissible_reach(pulse, duration_tolerance, edge_limit)
    on = (t >= 0.0) & (t <= reach)
    highest = np.zeros_like(t)
    lowest = np.zeros_like(t)
    if on.any():
        inner = t[on]
        top = np.full(inner.shape, -np.inf)
        bottom = np.full(inner.shape, np.inf)
        durations = pulse.duration * np.linspace(
            1.0 - duration_tolerance, 1.0 + duration_tolerance, FAMILY)
        edges = ([None] if pulse.shape == 'half-sine'
                 else list(np.linspace(0.0, edge_limit * pulse.duration,
                                       FAMILY_EDGES)))
        for duration in durations:
            for edge in edges:
                if pulse.shape == 'trapezoid' and not 2.0 * edge < duration:
                    continue
                values = pulse.peak * _shape(pulse.shape, inner, duration,
                                             edge)
                np.maximum(top, values, out=top)
                np.minimum(bottom, values, out=bottom)
        highest[on] = top
        lowest[on] = bottom
    width = band * abs(pulse.peak)
    return lowest - width, highest + width


@dataclass(frozen=True)
class PulseMeasurement:
    """Where a measured pulse is and what shape it took, on the record's
    clock (seconds) and in the record's SI units.

    `rise` is the trapezoid's rise and `fall` the sawtooth's or the
    trapezoid's fall; each is None where the shape has no such edge.
    `peak` is signed, in the direction of the nominal pulse.
    """

    #: the pulse's start, in seconds on the record's clock
    start: float
    #: its end, after the sawtooth's fall, in seconds
    end: float
    #: its duration, in seconds, measured as the nominal is stated
    duration: float
    #: its largest value in the nominal's direction, signed, SI
    peak: float
    #: when that was, in seconds
    peak_time: float
    #: the trapezoid's rise, in seconds, or None
    rise: float | None = None
    #: the sawtooth's or the trapezoid's fall, in seconds, or None
    fall: float | None = None


def _crossing(t: np.ndarray, y: np.ndarray, level: float, at: int,
              ahead: bool) -> float:
    """Where `y` last rises through `level` before sample `at`, or first
    falls through it after, by straight-line interpolation."""
    if ahead:
        below = np.flatnonzero(y[at:] < level)
        if not below.size:
            raise ValueError('the pulse has not come back down by the end '
                             'of the record')
        b = at + int(below[0])
        a = b - 1
    else:
        below = np.flatnonzero(y[:at] < level)
        if not below.size:
            raise ValueError('the pulse is already up at the start of the '
                             'record')
        a = int(below[-1])
        b = a + 1
    return float(t[a] + (level - y[a]) * (t[b] - t[a]) / (y[b] - y[a]))


def measure_pulse(time: ArrayLike, acceleration: ArrayLike,
                  shape: str, direction: float = 1.0) -> PulseMeasurement:
    """Find a classical pulse in a record and read its start, end,
    duration, peak and edges.

    The pulse is the record's largest excursion in `direction`, and its
    timing comes from where it crosses a tenth and nine tenths of that
    peak, walking out from the peak; the module docstring says how each
    shape's duration and edges follow from them.

    Parameters
    ----------
    time : array_like
        Each sample's time, in seconds.
    acceleration : array_like
        The record, in SI.
    shape : str
        One of `SHAPES`: which reading of the crossings applies.
    direction : float, default 1.0
        The sign of the nominal pulse; the pulse looked for is the
        largest excursion this way.

    Returns
    -------
    PulseMeasurement

    Raises
    ------
    ValueError
        If `shape` is not a classical shape, a time or a value is not
        a finite number, the record never moves in `direction`, or the
        pulse runs off either end of the record.
    """
    if shape not in SHAPES:
        raise ValueError(f'a classical pulse is one of {", ".join(SHAPES)}; '
                         f'{shape!r} is none of them')
    t = np.asarray(time, dtype=float)
    sign = 1.0 if direction >= 0.0 else -1.0
    y = sign * np.real(np.asarray(acceleration, dtype=complex))
    if t.shape != y.shape or t.size < 3:
        raise ValueError('a pulse is read from matching time and value '
                         'arrays of at least three samples')
    # 2026-10-16, on review: one NaN anywhere made argmax return it, and
    # the refusal said the record never moved in the pulse's direction
    broken = np.flatnonzero(~(np.isfinite(t) & np.isfinite(y)))
    if broken.size:
        raise ValueError(
            f'{broken.size} of the record\'s samples are not finite '
            f'numbers (NaN or infinite), the first at sample '
            f'{int(broken[0])}; a pulse is not read across them')
    at = int(np.argmax(y))
    peak = float(y[at])
    if not peak > 0.0:
        raise ValueError('the record never moves in the pulse\'s '
                         'direction')
    low_up = _crossing(t, y, LOW * peak, at, False)
    low_down = _crossing(t, y, LOW * peak, at, True)
    if shape == 'half-sine':
        duration = (low_down - low_up) / _HALF_SINE_SPAN
        start = low_up - duration * float(np.arcsin(LOW)) / np.pi
        return PulseMeasurement(start=start, end=start + duration,
                                duration=duration, peak=sign * peak,
                                peak_time=float(t[at]))
    high_up = _crossing(t, y, HIGH * peak, at, False)
    high_down = _crossing(t, y, HIGH * peak, at, True)
    # each straight edge extended through its two crossings to zero and
    # to the peak: the crossings are LOW and HIGH of the way up, so the
    # whole edge is their spacing over HIGH - LOW
    rising = (high_up - low_up) / (HIGH - LOW)
    falling = (low_down - high_down) / (HIGH - LOW)
    start = low_up - LOW * rising
    end = low_down + LOW * falling
    if shape == 'sawtooth':
        # the duration is the rise, and the fall comes after it
        return PulseMeasurement(start=start, end=end, duration=rising,
                                peak=sign * peak, peak_time=float(t[at]),
                                fall=falling)
    return PulseMeasurement(start=start, end=end, duration=end - start,
                            peak=sign * peak, peak_time=float(t[at]),
                            rise=rising, fall=falling)


def suggest_pulse(time: ArrayLike, acceleration: ArrayLike,
                  shape: str = 'half-sine') -> ClassicalPulse:
    """The classical pulse of `shape` a record most resembles: its
    largest excursion either way, with the duration (and the edge) read
    from it by `measure_pulse`.

    A starting point for a reading that has no nominal yet, as the app's
    Pulse Check opens on a record; the nominal a test is judged against
    is the one its document states, never one read off the record being
    judged.

    Parameters
    ----------
    time : array_like
        Each sample's time, in seconds.
    acceleration : array_like
        The record, in SI.
    shape : str, default 'half-sine'
        One of `SHAPES`.

    Returns
    -------
    ClassicalPulse

    Raises
    ------
    ValueError
        As `measure_pulse` raises.
    """
    values = np.real(np.asarray(acceleration, dtype=complex))
    if not values.size or not np.any(values):
        raise ValueError('the record holds no pulse: it never leaves zero')
    direction = 1.0 if values.max() >= -values.min() else -1.0
    measured = measure_pulse(time, values, shape, direction)
    edge = None
    if shape == 'sawtooth':
        edge = measured.fall
    elif shape == 'trapezoid':
        edge = min(max(measured.rise or 0.0, measured.fall or 0.0),
                   0.49 * measured.duration)
    return ClassicalPulse(shape, measured.peak, measured.duration, edge)


@dataclass(frozen=True, eq=False)
class PulseCheck:
    """A measured pulse judged against its nominal.

    The arrays are on the record's clock and in SI. `time`,
    `acceleration`, `lower`, `upper` and `outside` cover the judged
    span, from `QUIET` durations before the measured start to as long
    after the latest admissible end, as far as the record reaches; the
    envelope is placed on the measured start. `velocity_time` and
    `velocity` cover the velocity window, the velocity from rest at its
    first sample. The verdicts are read from them.
    """

    #: the nominal pulse
    pulse: ClassicalPulse
    #: where the measured pulse is, and what shape it took
    measured: PulseMeasurement
    #: each judged sample's time, in seconds on the record's clock
    time: np.ndarray
    #: the record over the judged span, SI
    acceleration: np.ndarray
    #: the envelope's lower edge at each judged sample, SI
    lower: np.ndarray
    #: and its upper edge
    upper: np.ndarray
    #: which judged samples are outside the envelope
    outside: np.ndarray
    #: each sample's time across the velocity window, in seconds
    velocity_time: np.ndarray
    #: the velocity across the window, from rest at its start, SI
    velocity: np.ndarray
    #: the judged span and the velocity window as the rule places them,
    #: (start, end) in seconds, whether or not the record reaches
    span: tuple[float, float]
    #: the velocity window, (start, end) in seconds
    window: tuple[float, float]
    #: the band, as a share of the peak
    band: float = BAND
    #: the duration's tolerance, as a share of nominal
    duration_tolerance: float = DURATION_TOLERANCE
    #: the edges' limit, as a share of the nominal duration
    edge_limit: float = EDGE_LIMIT
    #: the velocity change's tolerance, as a share of nominal
    velocity_tolerance: float = VELOCITY_TOLERANCE
    #: the time a sample stands for, in seconds: the grace the timing
    #: tolerances are read with is half of it
    sample_time: float = 0.0
    #: the record's first and last instant, in seconds
    record: tuple[float, float] = (0.0, 0.0)

    @property
    def points(self) -> int:
        """How many samples were judged against the envelope."""
        return int(self.time.size)

    @property
    def outside_count(self) -> int:
        """How many judged samples are outside the envelope."""
        return int(np.count_nonzero(self.outside))

    @property
    def covered(self) -> bool:
        """Whether the record reaches across the whole judged span and
        the whole velocity window."""
        # a sample's grace: the span's edges fall between samples
        first, last = self.record
        return bool(first <= min(self.span[0], self.window[0])
                    + self.sample_time
                    and last >= max(self.span[1], self.window[1])
                    - self.sample_time)

    @property
    def velocity_change(self) -> float:
        """The measured velocity change over the window, SI, signed.

        The area under the record across the window, the velocity where
        the window closes. Not the span from the lowest velocity to the
        highest that `core.shock_check.ShockCheck.velocity_change`
        reads; the module docstring says where the two differ.
        """
        return float(self.velocity[-1]) if self.velocity.size else 0.0

    @property
    def nominal_velocity_change(self) -> float:
        """The nominal pulse's velocity change, SI, signed."""
        return self.pulse.velocity_change

    @property
    def velocity_ratio(self) -> float:
        """The measured velocity change over the nominal."""
        return self.velocity_change / self.nominal_velocity_change

    @property
    def peak_ratio(self) -> float:
        """The measured peak over the nominal."""
        return self.measured.peak / self.pulse.peak

    @property
    def duration_ratio(self) -> float:
        """The measured duration over the nominal."""
        return self.measured.duration / self.pulse.duration

    @property
    def edge_allowance(self) -> float:
        """The longest an edge may take, in seconds."""
        return self.edge_limit * self.pulse.duration

    @property
    def envelope_ok(self) -> bool:
        """Every judged sample is inside the envelope."""
        return self.outside_count == 0

    @property
    def peak_ok(self) -> bool:
        """The measured peak is inside the band about the nominal's."""
        return bool(abs(self.peak_ratio - 1.0) <= self.band)

    @property
    def duration_ok(self) -> bool:
        """The measured duration is inside its tolerance, with half a
        sample's grace."""
        grace = 0.5 * self.sample_time
        return bool(abs(self.measured.duration - self.pulse.duration)
                    <= self.duration_tolerance * self.pulse.duration + grace)

    def _edge_ok(self, edge: float | None) -> bool:
        return (edge is None
                or edge <= self.edge_allowance + 0.5 * self.sample_time)

    @property
    def rise_ok(self) -> bool:
        """The trapezoid's rise takes no longer than the limit, with half
        a sample's grace; true where the shape has no rise to hold."""
        return self._edge_ok(self.measured.rise)

    @property
    def fall_ok(self) -> bool:
        """The sawtooth's or the trapezoid's fall takes no longer than
        the limit, with half a sample's grace; true for a half-sine."""
        return self._edge_ok(self.measured.fall)

    @property
    def edges_ok(self) -> bool:
        """Every edge the shape has is inside the limit."""
        return self.rise_ok and self.fall_ok

    @property
    def velocity_ok(self) -> bool:
        """The velocity change is inside its tolerance of nominal."""
        return bool(abs(self.velocity_ratio - 1.0)
                    <= self.velocity_tolerance)

    @property
    def passed(self) -> bool:
        """Every part of the check is met, over a record that covers it."""
        return (self.covered and self.envelope_ok and self.peak_ok
                and self.duration_ok and self.edges_ok and self.velocity_ok)

    def failures(self) -> list[str]:
        """What was not met, a phrase apiece; empty for a pass."""
        found = []
        if not self.covered:
            found.append('the record does not reach across the quiet '
                         'either side of the pulse and its velocity window')
        if not self.envelope_ok:
            found.append(f'{self.outside_count} of {self.points} points '
                         'outside the envelope')
        if not self.peak_ok:
            found.append(f'peak {_signed_percent(self.peak_ratio)} of '
                         'nominal')
        if not self.duration_ok:
            found.append(f'duration {_signed_percent(self.duration_ratio)} '
                         'of nominal')
        if not self.edges_ok:
            names = {'sawtooth': 'fall', 'trapezoid': 'rise or fall'}
            edge = max(e for e in (self.measured.rise, self.measured.fall)
                       if e is not None)
            found.append(f'{names[self.pulse.shape]} of {edge * 1e3:.3g} ms '
                         f'against {self.edge_allowance * 1e3:.3g} ms')
        if not self.velocity_ok:
            found.append(
                f'velocity change {_signed_percent(self.velocity_ratio)} '
                'of nominal')
        return found

    def describe(self) -> str:
        """The verdict in a sentence or two, as the view and the panel
        state it."""
        dv = _signed_percent(self.velocity_ratio)
        if self.passed:
            return (f'Passes: every point inside the envelope, velocity '
                    f'change {dv} of nominal.')
        return f'Flagged: {"; ".join(self.failures())}.'


def _signed_percent(ratio: float) -> str:
    """A ratio's distance from one as a signed percentage, '+10.0 %'."""
    return f'{100.0 * (ratio - 1.0):+.1f} %'.replace('-', '−')


def judge_pulse(time: ArrayLike, acceleration: ArrayLike,
                pulse: ClassicalPulse, *, band: float = BAND,
                duration_tolerance: float = DURATION_TOLERANCE,
                edge_limit: float = EDGE_LIMIT,
                velocity_tolerance: float = VELOCITY_TOLERANCE,
                quiet: float = QUIET,
                window: tuple[float, float] = VELOCITY_WINDOW
                ) -> PulseCheck:
    """Judge a measured pulse against `pulse`, point by point and as a
    whole.

    The pulse is found and measured (`measure_pulse`), the envelope is
    placed on its measured start (`pulse_envelope`) and every sample
    from `quiet` durations before the start to as long after the latest
    admissible end is judged against it, and the velocity change is
    integrated raw over `window` (`filters.integrate`, no drift
    corner) and compared with the nominal's.

    Parameters
    ----------
    time : array_like
        Each sample's time, in seconds, evenly spaced and in order.
    acceleration : array_like
        The record, in SI.
    pulse : ClassicalPulse
        The pulse the record was meant to be.
    band : float, default `BAND`
        The band either side of the shape and of zero, a share of the
        peak; it holds the peak as well.
    duration_tolerance : float, default `DURATION_TOLERANCE`
        How far the duration may sit from nominal, as a share.
    edge_limit : float, default `EDGE_LIMIT`
        The longest an edge may take, a share of the nominal duration.
    velocity_tolerance : float, default `VELOCITY_TOLERANCE`
        How far the velocity change may sit from nominal, as a share.
    quiet : float, default `QUIET`
        The quiet judged before and after the pulse, in durations.
    window : (float, float), default `VELOCITY_WINDOW`
        The velocity window, in durations before the start and after
        the end.

    Returns
    -------
    PulseCheck

    Raises
    ------
    ValueError
        If a tolerance is not a positive share, the samples are not
        evenly spaced, or the pulse cannot be found in the record
        (`measure_pulse`).
    """
    from .data import TimeHistory, _even_steps
    from .filters import integrate

    for name, value in (('band', band),
                        ('duration tolerance', duration_tolerance),
                        ('velocity tolerance', velocity_tolerance)):
        if not 0.0 < value < 1.0:
            raise ValueError(f'the {name} ({value:g}) is a share of '
                             'nominal and has to be between 0 and 1')
    if not 0.0 <= edge_limit < 0.5:
        raise ValueError(f'the edge limit ({edge_limit:g}) is a share of '
                         'the duration from 0 to below a half')
    if quiet < 0.0 or min(window) < 0.0:
        raise ValueError('the quiet and the velocity window are lengths, '
                         'in durations, and cannot be negative')
    t = np.asarray(time, dtype=float)
    a = np.real(np.asarray(acceleration, dtype=complex))
    measured = measure_pulse(t, a, pulse.shape, pulse.peak)
    step = _even_steps(t, 'A pulse check')
    duration = pulse.duration
    span = (measured.start - quiet * duration,
            measured.start + admissible_reach(pulse, duration_tolerance,
                                              edge_limit)
            + quiet * duration)
    judged = (t >= span[0]) & (t <= span[1])
    lower, upper = pulse_envelope(
        pulse, t[judged] - measured.start, band=band,
        duration_tolerance=duration_tolerance, edge_limit=edge_limit)
    values = a[judged]
    outside = (values < lower) | (values > upper)

    opens = (measured.start - window[0] * duration,
             measured.end + window[1] * duration)
    inside = (t >= opens[0]) & (t <= opens[1])
    across = t[inside]
    # 2026-10-16, on review: integrated through filters.integrate, raw,
    # as the shock check is, so the package has one trapezoid rule; it
    # was scipy's cumulative_trapezoid called here, which also took
    # unevenly spaced samples that nothing else in the package does
    if across.size > 1:
        windowed = TimeHistory(across, a[inside], response_dof='1X+',
                               ordinate_dim='acceleration')
        velocity = np.real(np.asarray(integrate(windowed, None).ordinate[0]))
    else:
        velocity = np.zeros(across.size)
    return PulseCheck(
        pulse=pulse, measured=measured, time=t[judged], acceleration=values,
        lower=lower, upper=upper, outside=outside, velocity_time=across,
        velocity=velocity, span=(float(span[0]), float(span[1])),
        window=(float(opens[0]), float(opens[1])), band=float(band),
        duration_tolerance=float(duration_tolerance),
        edge_limit=float(edge_limit),
        velocity_tolerance=float(velocity_tolerance), sample_time=step,
        record=(float(t[0]), float(t[-1])))


def check_pulse(history: TimeHistory, pulse: ClassicalPulse,
                channel: int | str | None = None, *, band: float = BAND,
                duration_tolerance: float = DURATION_TOLERANCE,
                edge_limit: float = EDGE_LIMIT,
                velocity_tolerance: float = VELOCITY_TOLERANCE,
                quiet: float = QUIET,
                window: tuple[float, float] = VELOCITY_WINDOW
                ) -> PulseCheck:
    """Judge one acceleration channel of a record against `pulse`
    (`judge_pulse`).

    Parameters
    ----------
    history : TimeHistory
        The shock record.
    pulse : ClassicalPulse
        The pulse it was meant to be, its peak in SI (m/s²).
    channel : int or str, optional
        The record's row, or its DOF; the first acceleration channel by
        default.
    band, duration_tolerance, edge_limit, velocity_tolerance, quiet, window
        As for `judge_pulse`.

    Returns
    -------
    PulseCheck

    Raises
    ------
    ValueError
        If the channel is not in the record or is not an acceleration,
        or as `judge_pulse` raises.
    """
    row = pulse_channel(history, channel)
    return judge_pulse(np.asarray(history.abscissa, dtype=float),
                       np.asarray(history.ordinate)[row], pulse, band=band,
                       duration_tolerance=duration_tolerance,
                       edge_limit=edge_limit,
                       velocity_tolerance=velocity_tolerance, quiet=quiet,
                       window=window)


def pulse_channel(history: TimeHistory,
                  channel: int | str | None = None) -> int:
    """The row of `history` a pulse is read from: `channel` by index or
    DOF, or the first acceleration channel; refused when it is not an
    acceleration, since a pulse's velocity change is the area under an
    acceleration and nothing else."""
    rows = range(history.num_records)
    if channel is None:
        row = next((i for i in rows
                    if history.known_dim(i) == 'acceleration'), None)
        if row is None:
            raise ValueError('a pulse is read from an acceleration, and '
                             'this record holds none')
        return row
    if isinstance(channel, str):
        dofs = [str(dof) for dof in history.response_dof]
        if channel not in dofs:
            raise ValueError(f'the record has no channel {channel}')
        row = dofs.index(channel)
    else:
        row = int(channel)
        if not 0 <= row < history.num_records:
            raise ValueError(f'the record has {history.num_records} '
                             f'channels; there is no channel {row}')
    if history.known_dim(row) != 'acceleration':
        raise ValueError(f'{history.response_dof[row]} is not an '
                         'acceleration, and a pulse is read from one')
    return row
