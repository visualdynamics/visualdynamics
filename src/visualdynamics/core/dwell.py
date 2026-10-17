"""A dwell read against time: how the level builds, holds and rings down.

A dwell holds one tone at one frequency, and what a test engineer wants
from it is a picture against time rather than frequency. A sweep's level
is read against frequency because the frequency is what changes; a
dwell's frequency does not change, so against frequency its level is one
point, and the story it tells is in time. At a resonance the response
does not reach its steady height when the tone arrives: it builds cycle
by cycle, and when the tone stops it rings down at the same pace. For a
mode of quality factor Q driven at its own frequency from rest, the
share of the steady height reached after n cycles is very nearly
1 - exp(-pi n / Q), so after Q cycles it is 1 - exp(-pi), 95.7 %, and
after 2Q cycles 99.8 %, whatever the frequency or the Q; and n cycles
after the tone stops, exp(-pi n / Q) of it remains. That count, about Q
cycles to build and about Q to die away, is why a sweep must spend Q
cycles near a resonance to read it, why a few cycles of a transient
never bring a sharp mode to its full response, and how long a dwell's
first seconds are spent arriving rather than holding.

`read_dwell` reads one dwell of a sine specification on one channel of
a recording with the tracking detectors of `core.sine_tracking` (the
peak over each cycle, unfiltered, by default; any `SineTracking`
setting may be given) at the end of every cycle of the dwell, and on at
the dwell's frequency after the tone stops, to the end of the record.
It returns a `DwellReading`: the level against time, the steady level
the dwell reached, the cycles it took to reach a stated share of it (95 %
by default), and the cycles after the stop for the level to fall to the
complement of that share (5 %), so the build-up and the ring-down are
read by one rule. `DwellReading.share_at` and `remaining_at` read the
level at any count of cycles into the dwell or after the stop.

The steady level is the median of the readings over the dwell's last
quarter, which is a reading of the record and not a model of it: a
dwell too short to settle, or a resonance that moved during it, reaches
a steady level that is not the mode's, and the reading says so
(`DwellReading.settled`, and a line in `notes`) rather than borrowing a
number from a formula. Two other things are said rather than hidden: a
dwell that a sweep runs into does not start from rest, so its build-up
is the rest of an arrival, and a dwell that a sweep leaves does not
stop, so nothing rings down from a stop and no ring-down is read.

The readings are taken at the dwell's own whole cycles, counted from its
start in the record and, after the stop, from the stop, so the peak in
cycle n covers exactly that cycle. A peak is the peak of the samples, as
a sampled controller reads it, and under-reads a sine by at most
cos(pi x frequency / sample rate), 0.01 % at 200 samples a cycle.

A reading is reported at the end of the span it reads, but it stands for
the response somewhere inside that span, and where depends on the
detector and on which way the level is going. The peak over a cycle of
a rising level is that cycle's last crest, at its end; over a cycle of a
falling level it is the first crest, at its start, a whole cycle before
the reading is reported. RMS and mean average the span and stand for
its middle either way; the band's own output stands at its instant, its
lag the filter's and the same both ways. Counted on the reported clock,
the peak detector's ring-down would read a cycle longer than the
build-up for no reason in the response. So each reading is also placed
at the instant it stands for
(`DwellReading.stands`, the reported time less `LAG_RISING` or
`LAG_FALLING` spans), and every number the reading gives is read on that
clock: the two counts, `share_at` and `remaining_at`, and the plot. The
build-up and the ring-down of one Q then read alike, and each number is
the response's rather than the detector's late report of it: Q cycles
after the stop, 4.3 % (exp(-pi)) remains, where the peak over the
thirtieth cycle after the stop, reported at its end, says 4.8 %.

References
----------
The build-up and the free decay are the transient and the free response
of a damped one-degree-of-freedom system, a standard result; these are
where it and the dwell test it is read for are set out.

1. Thomson, W. T., & Dahleh, M. D. (1998). *Theory of Vibration with
   Applications*, 5th ed. Prentice Hall. The response of a damped
   oscillator to a harmonic force switched on at rest, its transient
   decaying as exp(-zeta omega t) into the steady state, and the free
   decay of the same system, read as the logarithmic decrement.
2. IEC 60068-2-6:2007, *Environmental testing - Part 2-6: Tests - Test
   Fc: Vibration (sinusoidal)*. The endurance test by dwelling at fixed
   frequencies and at resonances found in a search, whose level a
   tracking detector reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import _even_steps
from .sine import SineSweepSpecification, _control_rows
from .sine_tracking import SineTracking, _Demodulated, _detect, _rows, _tone_at

# On the reported clock the peak detector's ring-down to 5 % read 29.6
# cycles at Q 30 against 28.6 for the build-up to 95 %, and 4.8 % Q
# cycles after the stop where the response is at 4.3 %: the measurement
# behind the `stands` clock (Kevin Cross, 2026-10-16; the narration moved
# here from the docstring on review, 2026-10-16).

#: the share of the steady level the build-up is read to, and whose
#: complement the ring-down is read to: 95 %, the share a resonance
#: reaches after about 0.95 Q cycles and keeps after the stop for the
#: same count (exp(-pi n / Q) = 0.05 at n = Q ln 20 / pi)
DWELL_SHARE = 0.95

#: the share of the dwell, at its end, whose readings give the steady
#: level: the last quarter, long enough that a median holds out a stray
#: cycle and short enough that a dwell of a few Q is past its build-up
STEADY_SPAN = 0.25

#: how far before its report a reading stands on a rising level, and on
#: a falling one, in the detector's spans: the peak of a rising cycle is
#: at its end and of a falling one at its start, RMS and mean stand for
#: the middle of what they average, and the band's output for its
#: instant (see the module docstring)
LAG_RISING = {'peak': 0.0, 'rms': 0.5, 'mean': 0.5, 'filtered': 0.0}
LAG_FALLING = {'peak': 1.0, 'rms': 0.5, 'mean': 0.5, 'filtered': 0.0}


@dataclass(frozen=True, kw_only=True, eq=False)
class DwellReading:
    """One dwell's level against time on one channel: what
    `read_dwell` returns.

    Times are seconds into the recording, from its first sample, the
    clock `onset` is given on and `SineLevel.seconds` is stamped on.
    `seconds` and `level` hold every reading, in the dwell and after
    the stop, reported at the end of each whole cycle; `level` is in the
    record's own units (SI), as the tracking detector read it, and
    `stands` is the instant each reading stands for (the detector's lag
    taken out, `LAG_RISING` in the dwell and `LAG_FALLING` after it).
    `steady` is the level the dwell reached (the median over its last
    `STEADY_SPAN`), and `settled` whether it held there (the median of
    that span's second half within half of 1 - `share` of its first
    half's). `built` is the cycles from the dwell's start to `share` of
    it and `rung` the cycles from the stop down to 1 - `share` of it,
    each interpolated on the `stands` clock, or None where the record
    does not say (a record that starts after the dwell has built, a
    dwell that a sweep leaves, a record that ends first).
    `unit` is the unit the record declared, kept as metadata; the
    numbers are SI whatever it says.
    `notes` holds what the reading cannot be trusted for, as text, empty
    when it is clean. Two of these compare equal only if they are the
    same object, as `TrackedWaveform` does, since the arrays have no
    single truth value.
    """

    tone: str
    segment: int
    dof: str
    dimension: str
    unit: str | None
    setting: SineTracking
    frequency: float
    onset: float
    start: float
    stop: float
    seconds: np.ndarray
    level: np.ndarray
    stands: np.ndarray
    share: float
    steady: float
    settled: bool
    built: float | None
    rung: float | None
    stops: bool
    notes: tuple[str, ...] = field(default=())

    @property
    def cycles(self) -> np.ndarray:
        """Each reading's count of the dwell's cycles since its start:
        whole numbers through the dwell, and on past the stop."""
        return (self.seconds - self.start) * self.frequency

    @property
    def held(self) -> np.ndarray:
        """True where a reading is inside the dwell (at or before the
        stop), False where it is in the ring-down."""
        return self.seconds <= self.stop + 1e-6 / self.frequency

    @property
    def after(self) -> np.ndarray:
        """Each reading's count of cycles since the stop: negative
        inside the dwell, whole numbers in the ring-down."""
        return (self.seconds - self.stop) * self.frequency

    @property
    def built_seconds(self) -> float | None:
        """`built` in seconds from the dwell's start."""
        return None if self.built is None else self.built / self.frequency

    @property
    def rung_seconds(self) -> float | None:
        """`rung` in seconds from the stop."""
        return None if self.rung is None else self.rung / self.frequency

    def share_at(self, cycles: float) -> float:
        """The level `cycles` cycles into the dwell, as a share of the
        steady level: the response itself, each reading placed where it
        stands (`stands`) and interpolated between them. Raises
        ValueError outside the dwell's readings.

        Parameters
        ----------
        cycles : float
            Cycles of the dwell's frequency since its start.

        Returns
        -------
        float
            The share, 1.0 at the steady level.
        """
        x = (self.stands[self.held] - self.start) * self.frequency
        if not x.size or not x[0] - 1e-9 <= cycles <= x[-1] + 1e-9:
            raise ValueError(f'{cycles:g} cycles is outside the dwell, which '
                             f'is read from {x[0]:g} to {x[-1]:g} cycles'
                             if x.size else 'the dwell holds no reading')
        return float(np.interp(cycles, x, self.level[self.held]) / self.steady)

    def remaining_at(self, cycles: float) -> float:
        """The level `cycles` cycles after the stop, as a share of the
        steady level: what remains of the response once the tone has
        gone, each reading placed where it stands (`stands`), so Q cycles
        after the stop it is exp(-pi), 4.3 %, whatever the detector.
        Raises ValueError where the ring-down was not read.

        Parameters
        ----------
        cycles : float
            Cycles of the dwell's frequency since the stop.

        Returns
        -------
        float
            The share remaining.
        """
        ringing = ~self.held
        placed = (self.stands - self.stop) * self.frequency
        x = np.concatenate((placed[self.held][-1:], placed[ringing]))
        y = np.concatenate(([self.level[self.held][-1]],
                            self.level[ringing]))
        if not ringing.any() or not 0.0 <= cycles <= x[-1] + 1e-9:
            raise ValueError(
                'no ring-down was read' if not ringing.any()
                else f'{cycles:g} cycles after the stop is past the '
                     f'record, which ends {x[-1]:g} cycles after it')
        return float(np.interp(cycles, x, y) / self.steady)

    def describe(self, unit_system: Any = None) -> str:
        """The reading in one line, such as 'steady 294.3 m/s², 95 % in
        28.6 cycles (1.43 s), down to 5 % 28.6 cycles after the stop'.

        The steady level is in SI by default, or in a unit system's
        display units when one is given.
        """
        from ..units import SI

        us = SI if unit_system is None else unit_system
        steady = float(us.from_si(self.steady, self.dimension))
        unit = us.label_text(self.dimension)
        percent = f'{100.0 * self.share:g} %'
        rest = f'{100.0 * (1.0 - self.share):g} %'
        words = [f'steady {steady:.4g} {unit}'.rstrip()]
        if self.built is not None:
            words.append(f'{percent} in {self.built:.1f} cycles '
                         f'({self.built_seconds:.3g} s)')
        if self.rung is not None:
            words.append(f'down to {rest} {self.rung:.1f} cycles after the '
                         'stop')
        return ', '.join(words)


def _crossing(x: np.ndarray, y: np.ndarray, level: float,
              rising: bool) -> float | None:
    """Where `y` first reaches `level` along `x` (at or over it when
    `rising`, at or under it otherwise), interpolated linearly from the
    reading before; None when it never does."""
    reached = y >= level if rising else y <= level
    if not reached.any():
        return None
    k = int(np.argmax(reached))
    if k == 0 or y[k] == y[k - 1]:
        return float(x[k])
    share = (level - y[k - 1]) / (y[k] - y[k - 1])
    return float(x[k - 1] + share * (x[k] - x[k - 1]))


def default_channel(history: Any,
                    specification: SineSweepSpecification) -> str:
    """The channel `read_dwell` reads when none is named: the first the
    specification does not control, since a dwell's build-up is a
    response's (the control is held at its level by the controller, and
    barely builds), else the first control channel. Raises ValueError
    when the recording carries neither.

    Parameters
    ----------
    history : TimeHistory
        The recording.
    specification : SineSweepSpecification
        Whose control channels are passed over.

    Returns
    -------
    str
        The channel's DOF.
    """
    return str(history.response_dof[_default_row(history, specification)])


def _default_row(history: Any, specification: SineSweepSpecification) -> int:
    """`default_channel`'s row."""
    dofs = [str(dof) for dof in history.response_dof]
    controlled = {str(dof) for dof in specification.response_dof}
    for row, dof in enumerate(dofs):
        if dof not in controlled:
            return row
    return _control_rows(history, specification)[0]


def _chosen(specification: SineSweepSpecification, tone: str | None,
            segment: int | None) -> tuple[Any, int]:
    """The tone and the dwell segment read: the named tone or the first
    that holds a dwell, and the named segment or its first dwell."""
    if tone is not None:
        chosen = specification.tone(tone)
        if not chosen.dwell_segments():
            raise ValueError(f'{chosen.name} holds no dwell: every segment '
                             'sweeps')
    else:
        chosen = next((t for t in specification.tones if t.dwell_segments()),
                      None)
        if chosen is None:
            raise ValueError('no tone of this specification holds a dwell')
    dwells = chosen.dwell_segments()
    if segment is None:
        return chosen, dwells[0]
    if segment not in dwells:
        raise ValueError(f'{chosen.name}: segment {segment} sweeps; its '
                         f'dwells are segments {dwells}')
    return chosen, int(segment)


def read_dwell(history: Any, specification: SineSweepSpecification, *,
               tone: str | None = None, segment: int | None = None,
               channel: str | None = None,
               setting: SineTracking | None = None,
               share: float = DWELL_SHARE, onset: float | None = None,
               after: float | None = None) -> DwellReading:
    """Read a dwell's level against time: the build-up, the steady
    level and the ring-down after the tone stops.

    The tone is placed on the recording where it begins (found by
    matched filter on the control channels, as `track_sine` finds it,
    or at `onset` seconds into the recording), and the channel is read
    with the tracking `setting` at the end of every whole cycle of the
    dwell's frequency from the dwell's start to its stop, then from the
    stop on at the same frequency until the record ends (or `after`
    seconds past the stop). The steady level is the median of the
    readings over the dwell's last `STEADY_SPAN`; `share` of it marks
    the build-up and 1 - `share` the ring-down, each counted where the
    readings stand (`LAG_RISING`, `LAG_FALLING`). A dwell is a segment
    that sweeps no more than `DWELL_SPAN` of its frequency
    (`SineTone.is_dwell`), the form `SineTone.dwell` and a controller's
    file both give it.

    Raises ValueError when no tone holds a dwell, when the named tone or
    segment is not one, when `share` is not between zero and one, when
    the recording is not evenly sampled or lacks the channel, when it
    holds no whole cycle of the dwell, or when the channel reads no level
    over the dwell's last quarter (zero, as a dead channel reads, or not
    a number).

    Parameters
    ----------
    history : TimeHistory
        The recording, evenly sampled.
    specification : SineSweepSpecification
        What the drive followed; one of its tones holds the dwell.
    tone : str, optional
        Which tone; the first that holds a dwell by default.
    segment : int, optional
        Which of the tone's segments; its first dwell by default.
    channel : str, optional
        The DOF read. By default the first channel the specification does
        not control, a response, since a controlled channel is held at
        its level and does not build; else the first control channel.
    setting : SineTracking, optional
        How each cycle is read: the peak over the cycle, unfiltered, by
        default; any band and detector `track_sine` takes.
    share : float, default `DWELL_SHARE`
        The share of the steady level the build-up is read to; the
        ring-down is read down to one minus it.
    onset : float, optional
        Seconds into the recording where the tone begins (its own time
        zero, not the dwell's when a sweep comes first). Found by matched
        filter when omitted.
    after : float, optional
        Seconds of ring-down to read past the stop; to the record's end
        by default.

    Returns
    -------
    DwellReading
        The readings against time, the steady level and the two counts,
        with notes on anything they cannot be trusted for.
    """
    setting = SineTracking(detector='peak') if setting is None else setting
    share = float(share)
    if not 0.0 < share < 1.0:
        raise ValueError(f'a share of {share:g} of the steady level must '
                         'sit between zero and one')
    dt = _even_steps(np.asarray(history.abscissa, dtype=float),
                     'Reading a dwell')
    rate = 1.0 / dt
    chosen, segment = _chosen(specification, tone, segment)
    dofs = [str(dof) for dof in history.response_dof]
    row = (_default_row(history, specification) if channel is None
           else _rows(history, specification, [str(channel)])[0])
    rows = [row]
    if onset is None:
        try:
            rows = _control_rows(history, specification)
        except ValueError:
            pass                  # no control channel: place it on this one
    chosen, onset = _tone_at(history, specification, rows, chosen.name,
                             onset, dt)

    starts = np.concatenate(([0.0], np.cumsum(chosen.segment_seconds())))
    frequency = float(chosen.frequency[segment])
    begins = onset + float(starts[segment])
    stop = onset + float(starts[segment + 1])
    stops = segment == len(chosen.segment_rate) - 1
    length = history.ordinate.shape[1]
    end = (length - 1) * dt
    if stops and after is not None:
        end = min(end, stop + float(after))
    elif not stops:
        end = min(end, stop)

    # the read instants: whole cycles from the dwell's start to its
    # stop, then whole cycles from the stop, each a sample index
    period = 1.0 / frequency
    # (a dwell's span is a rate times a hair of a hertz, so its count of
    # cycles comes back a rounding short of whole: hence the 1e-6)
    held = begins + period * np.arange(1, int(np.floor(
        (stop - begins) * frequency + 1e-6)) + 1)
    ringing = (stop + period * np.arange(1, int(np.floor(
        (end - stop) * frequency + 1e-6)) + 1)) if stops else np.empty(0)
    instants = np.concatenate((held, ringing))
    first = max(round(begins / dt), 0)
    last = min(round(end / dt) + 1, length)
    if last - first < 2:
        raise ValueError(f'{chosen.name}: the recording holds no whole cycle '
                         'of the dwell to read')
    reads = np.round(instants / dt).astype(np.int64) - first
    keep = (reads >= 0) & (reads < last - first)
    reads, instants = reads[keep], instants[keep]

    # the drive's phase over the samples read: the tone's own law to its
    # end, then held on at its last frequency, so the ring-down is read
    # cycle by cycle at the frequency the response rings at
    local = (np.arange(first, last) * dt) - onset
    duration = chosen.duration()
    phase = chosen.phase_at(np.minimum(local, duration))
    drive = chosen.frequency_at(local)
    past = local > duration
    phase[past] += 2.0 * np.pi * float(chosen.frequency[-1]) * (
        local[past] - duration)
    # a read needs its whole span of cycles inside the samples held,
    # which only a record that starts after the dwell does not give
    spans = setting.cycles if setting.detector != 'filtered' else 0.0
    whole = phase[reads] - 2.0 * np.pi * spans >= phase[0] - 1e-9
    reads, instants = reads[whole], instants[whole]
    inside = instants <= stop + 1e-6 * period
    if not inside.any():
        raise ValueError(f'{chosen.name}: the recording holds no whole cycle '
                         'of the dwell to read')

    x = np.real(np.asarray(history.ordinate[row, first:last], dtype=float))
    if setting.kind == 'unfiltered':
        level = _detect(x, phase, reads, setting.detector, setting.cycles)
    else:
        demodulated = _Demodulated(x, phase)
        amplitude = demodulated.through(setting, drive, rate)
        level = (np.abs(amplitude[reads]) if setting.detector == 'filtered'
                 else _detect(demodulated.waveform(amplitude), phase, reads,
                              setting.detector, setting.cycles))

    cycles = (instants - begins) * frequency
    in_dwell = cycles[inside]
    total = (stop - begins) * frequency
    tail = inside & (cycles >= total * (1.0 - STEADY_SPAN) - 1e-6)
    if not tail.any():
        tail = inside & (cycles == in_dwell[-1])
    steady = float(np.median(level[tail]))
    if not np.isfinite(steady) or steady <= 0.0:
        raise ValueError(
            f'{dofs[row]} reads no level over the last quarter of the dwell '
            f'(steady {steady:g}): a dead channel, or readings that are not '
            'numbers, give no build-up to count')
    # settled: the tail's second half holds where its first half was, to
    # within half the complement of the share. Halves' medians rather
    # than the extremes, so one stray cycle neither moves the steady
    # level nor calls a held level unsettled
    held_level = level[tail]
    half = len(held_level) // 2
    drift = (float(np.median(held_level[half:]))
             - float(np.median(held_level[:half])) if half else 0.0)
    settled = bool(abs(drift) <= 0.5 * (1.0 - share) * steady)

    notes = []
    if not settled:
        notes.append(f'the level moved {100.0 * drift / steady:+.1f} % over '
                     'the last quarter of the dwell: it had not settled, so '
                     'the steady level and the build-up are not the mode\'s')
    if segment > 0:
        notes.append('the tone runs before the dwell, so its build-up '
                     'starts from the response it left, not from rest')
    # where each reading stands: its report less the detector's lag,
    # rising in the dwell and falling after it (LAG_RISING, LAG_FALLING)
    rise = LAG_RISING[setting.detector] * spans
    fall = LAG_FALLING[setting.detector] * spans
    stands = instants - np.where(inside, rise, fall) * period
    # a record that starts after the dwell has built holds no crossing:
    # its first reading is already over the share, and the count would
    # be wherever the record happens to start
    if round(begins / dt) < 0 and level[inside][0] >= share * steady:
        built = None
        notes.append(f'the record starts {-begins * frequency:.1f} cycles '
                     f'into the dwell, after the level had reached '
                     f'{100.0 * share:g} % of its steady value, so no '
                     'build-up is read')
    else:
        built = _crossing(in_dwell - rise, level[inside], share * steady,
                          rising=True)
    rung = None
    if not stops:
        notes.append('a sweep leaves the dwell, so the tone does not stop '
                     'and no ring-down is read')
    else:
        x_after = np.concatenate(([-rise], (stands[~inside] - stop)
                                  * frequency))
        y_after = np.concatenate(([level[inside][-1]], level[~inside]))
        rung = _crossing(x_after, y_after, (1.0 - share) * steady,
                         rising=False)
        if rung is None:
            notes.append(f'the record ends before the level falls to '
                         f'{100.0 * (1.0 - share):g} % of its steady value')
    return DwellReading(
        tone=chosen.name, segment=segment, dof=dofs[row],
        dimension=history.ordinate_dim[row],
        unit=history.ordinate_unit[row], setting=setting,
        frequency=frequency, onset=onset, start=begins, stop=stop,
        seconds=instants, level=np.asarray(level, dtype=float),
        stands=stands, share=share,
        steady=steady, settled=settled, built=built, rung=rung, stops=stops,
        notes=tuple(notes))
