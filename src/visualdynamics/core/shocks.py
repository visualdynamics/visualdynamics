"""Finding the shocks in a recording, and the window to analyze each in.

A shock test is a series of events in one continuous stream: the
article sits quiet, something hits it, it rings down, and it sits quiet
again until the next one. Nothing in the file says where they were.
`Averaging` answers "which stretch of this record is the test?" for a
stationary run; this answers "which stretches are the events?" for a
transient one, and there is more than one answer per record.

The shape of the problem is different enough from averaging to want its
own algorithm, but the first move is the same: compress the record to
one level per hop, across all channels at once, and decide everything
afterwards on that short series. A shock record can be minutes long at
50 kHz, and no amount of care about thresholds is worth a second pass
over the samples.

Three things make the detection robust rather than merely simple:

- **A Schmitt trigger, not a threshold.** One threshold chatters: a
  ringdown crosses it on every cycle and one event comes back as forty.
  An event starts when the level rises `ARM` dB over the floor and does
  not end until it falls back under `RELEASE`, so the decay is followed
  down instead of being chopped where it happens to dip.
- **The floor is a low quantile, not the mean.** Shocks are sparse and
  enormous; averaging them into the floor they are measured against
  raises it by tens of dB. A quantile low enough to sit in the quiet is
  unmoved by how hard, or how many, the shocks were.
- **Everything is measured against the floor, not against the peak.**
  A shock series is usually walked up in level, so the last event can
  be ten times the first. Thresholds relative to each event's own peak
  would find them all and give them different windows for no physical
  reason; relative to the floor, they get the window their ringdown
  actually needs.

The window is not the event. It reaches back before the rise, because
an SRS oscillator has to start from rest and a window that opens mid-
pulse invents a step that was never there, and it reaches past the
release, because the ringdown that is still above the floor is the part
the low-frequency oscillators are still answering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from .detect import hop_levels

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

#: how finely the record is chopped before anything is decided, in
#: seconds. A shock pulse is milliseconds, so the hop has to be a
#: fraction of one — but every hop is a point in the series every later
#: judgment walks, so it cannot be a single sample either.
HOP_SECONDS = 0.001

#: where the quiet is, as a percentile of the levels. Low enough to sit
#: in the gaps between events even when the ringdowns are long.
FLOOR_QUANTILE = 10.0

#: dB over the floor at which an event starts, and the dB over the floor
#: it has to fall back under before it is over. The gap between them is
#: the hysteresis, and it is what stops a ringdown reading as many
#: events instead of one.
ARM = 12.0
RELEASE = 6.0

#: events closer than this, in seconds, are one event. A double hit and
#: a ringdown that dips through the release level for a moment look the
#: same from here, and treating them as two windows would analyze the
#: second half of one shock as though it were a shock.
MERGE = 0.05

#: how much of its own length a window reaches back before the rise and
#: on past the release. The front matters most: an oscillator has to
#: start at rest, so the window opens in the quiet ahead of the event.
#: The tail fraction is a *floor* — the ringdown follow below usually
#: reaches further, but an impulse-like event whose ringing dies inside
#: the run still gets a margin past its release.
LEAD = 0.15
TAIL = 0.25

#: the hop the ringdown is followed at, in seconds — deliberately much
#: coarser than `HOP_SECONDS`. A hop's level is the variance *within*
#: the hop, which makes the hop a high-pass filter: content whose
#: period dwarfs the hop is mostly mean, and the mean is removed. At
#: one millisecond that blinds the series below a couple of hundred
#: hertz — precisely the slow ringing that needs the longest windows,
#: which is how a 63 Hz mode ringing on one channel was invisible to
#: the series that decided where its window closed. Twenty-five
#: milliseconds holds a whole cycle of it, and averages power over the
#: cycle rather than sampling the flicker within one.
RING_HOP = 0.025

#: how far under its own peak an event's ringing has settled, in dB,
#: before the window may close.
#:
#: Peak-relative, where the detection thresholds are floor-relative,
#: because they answer different questions. Whether something *is* an
#: event is judged against the quiet — a walked-up series must not
#: need different thresholds per level. How long its ringing still
#: matters is judged against how hard it rang: a tail this far under
#: the peak can no longer raise any oscillator's own peak, and
#: identical events at different levels settle in identical time,
#: which is the level-independence the floor gave the thresholds.
#: Never past the quiet either way — the follow stops at
#: `RELEASE` over the loudest channel's own floor when that comes
#: first, so a fast event is not chased 30 dB into its noise.
#:
#: Measured, not chosen: at 25 dB the drone stress set's worst channel
#: reads 0.14 dB low at the bottom of its band; at 30 dB every record
#: in the stress and test sets reads exact.
SETTLED = 30.0

#: whether every event in one record is analyzed in a window of the
#: same length.
#:
#: The default, and the reason is comparability rather than accuracy.
#: An SRS is the peak of a lightly damped oscillator, and past its
#: knee lengthening a window changes the spectrum by exactly nothing —
#: so what one length buys is not a better answer but the same
#: question asked of every event: with identical support, what differs
#: between the spectra is the events.
#:
#: It matters because window length is derived from each event's own
#: ringing, and detection noise gives near-identical events windows
#: that differ for no physical reason. Not forced: a record can hold
#: genuinely unlike events, a drop test and a pyroshock in one stream,
#: and then per-event is the honest answer. Hence a default rather
#: than a rule.
COMMON_LENGTH = True

#: an event shorter than this in seconds is a glitch, not a shock — a
#: dropped sample, a switch transient, a spike on one channel
MIN_DURATION = 0.002

#: how far under the loudest hop the floor is allowed to be, in dB.
#:
#: A guard against digital silence, not a tuning knob. A record with
#: exact zeros between its events — a synthesized one, or a stream that
#: was gated — has a floor of negative infinity, and every threshold
#: measured from it arms on the first bit of dither. No real noise floor
#: is anywhere near a hundred dB under the peak of the same recording,
#: so this never binds on a measurement and always binds on silence.
DYNAMIC = 100.0

#: the most events worth reporting from one record. A record that
#: appears to hold more than this is one whose floor was misjudged, and
#: a thousand windows is not an answer anybody wanted.
MAX_EVENTS = 200


@dataclass(frozen=True)
class Shock:
    """One event's analysis window: when it opens and how long it runs.

    Two numbers in seconds, both from the start of the record, and
    everything else derived. The window rather than the event: it
    already includes the lead-in and the tail, because what an SRS is
    computed from is the window and there is nothing to be gained by
    storing the event and re-deriving the window every time it is
    wanted.
    """

    start: float
    duration: float

    def __post_init__(self) -> None:
        if float(self.duration) <= 0.0:
            raise ValueError('a shock window has to have some length')
        if float(self.start) < 0.0:
            raise ValueError('a shock cannot start before the record')
        object.__setattr__(self, 'start', float(self.start))
        object.__setattr__(self, 'duration', float(self.duration))

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def start_sample(self, sample_rate: float) -> int:
        return round(self.start * sample_rate)

    def samples(self, sample_rate: float) -> int:
        """How many samples the window holds — at least two, since one
        sample is not a transient."""
        return max(round(self.duration * sample_rate), 2)

    def bounds(self, sample_rate: float) -> tuple[int, int]:
        """(first, last) sample indices, last exclusive."""
        first = self.start_sample(sample_rate)
        return first, first + self.samples(sample_rate)

    def fits(self, samples: int, sample_rate: float) -> bool:
        return self.bounds(sample_rate)[1] <= samples

    def clipped(self, samples: int, sample_rate: float) -> Shock:
        """This window pulled inside the record it belongs to."""
        first, last = self.bounds(sample_rate)
        first = max(min(first, samples - 2), 0)
        last = min(max(last, first + 2), samples)
        return replace(self, start=first / sample_rate,
                       duration=(last - first) / sample_rate)

    def cut(self, history: TimeHistory) -> np.ndarray:
        """The samples this window covers, every channel, as a view."""
        first, last = self.bounds(history.sample_rate)
        return np.asarray(history.ordinate)[:, first:last]


def hop_for(sample_rate: float, seconds: float = HOP_SECONDS) -> int:
    """Samples per hop — at least one, whatever the rate."""
    return max(round(float(seconds) * float(sample_rate)), 1)


def envelope(history: TimeHistory, seconds: float = HOP_SECONDS
             ) -> tuple[np.ndarray, int]:
    """(levels in dB, hop in samples): the record as a short series.

    The same compression the averaging detector uses — median across
    channels so one bad channel is a vote and not the answer, mean
    removed within each hop so a DC offset is not read as level — but
    at a hop measured in milliseconds rather than in seconds, because
    what is being looked for lasts milliseconds.
    """
    hop = hop_for(history.sample_rate, seconds)
    # sparse: a shock record's channels are quiet most of the time, and
    # the stationary rule for a live channel would call them all dead
    return hop_levels(history.ordinate, hop, sparse=True), hop


def loudest(history: TimeHistory, seconds: float = RING_HOP
            ) -> np.ndarray:
    """The loudest channel's level per hop, in dB, at the follow hop.

    The series the ringdown is followed down. Detection stays on the
    median — one bad channel is a vote, not the answer — but an SRS is
    one spectrum *per channel*, so one channel still ringing under
    twenty quiet ones is a channel whose spectrum a median-sized
    window cuts short. The maximum is the series that can see it, and
    the coarse hop (`RING_HOP`) is what lets it see slow ringing at
    all.
    """
    hop = hop_for(history.sample_rate, seconds)
    return hop_levels(history.ordinate, hop, sparse=True, loudest=True)


def floor_of(level: np.ndarray, quantile: float = FLOOR_QUANTILE,
             dynamic: float = DYNAMIC) -> float | None:
    """The quiet the events are measured against.

    A quantile rather than a mean or a median: shocks are sparse and
    enormous, and any average of a record containing them sits well
    above the quiet between them.

    Never more than `dynamic` dB under the loudest hop. That is the
    silence guard described at `DYNAMIC`, and it is why a synthesized
    record whose gaps are exact zeros still gets thresholds that mean
    something.
    """
    if level.size == 0:
        return None
    return float(max(np.percentile(level, quantile),
                     level.max() - float(dynamic)))


def _runs(level: np.ndarray, arm: float,
          release: float) -> list[tuple[int, int]]:
    """[(first, last)] hops, last exclusive: the Schmitt trigger.

    A run opens on the first hop at or over `arm` and closes on the
    first hop under `release` after it. Written out rather than done
    with masks because that is exactly what hysteresis is — the state
    depends on the state — and the loop is over hops, of which there
    are thousands, not samples, of which there are millions.
    """
    out = []
    live = False
    first = 0
    for i, value in enumerate(level):
        if not live and value >= arm:
            live, first = True, i
        elif live and value < release:
            out.append((first, i))
            live = False
    if live:
        out.append((first, len(level)))
    return out


def _merged(runs: list[tuple[int, int]],
            apart: int) -> list[tuple[int, int]]:
    """Runs closer together than `apart` hops, joined."""
    out = []
    for first, last in runs:
        if out and first - out[-1][1] <= apart:
            out[-1] = (out[-1][0], last)
        else:
            out.append((first, last))
    return out


def _windowed(runs: list[tuple[int, int]], hops: int, loud: np.ndarray,
              scale: float, lead: float = LEAD, tail: float = TAIL,
              settled: float = SETTLED) -> list[tuple[int, int]]:
    """Each run opened out into the window it should be analyzed in,
    in detection hops.

    The front reaches back into the quiet ahead of the rise, a
    fraction of the run, because an oscillator has to start from rest.

    The back follows the ringdown down the loudest-channel series
    (`loud`, at `scale` detection hops per follow hop) to where it has
    settled `SETTLED` dB under its own peak — or into the quiet, when
    that comes first, so a fast event is not chased into its noise.
    The follow reads the *first* settled hop, not the last loud one:
    the loudest channel jumps at the next event before the median
    arms, so a last-exceedance read walks into the neighbor and calls
    the whole gap ringdown. The tail fraction stands underneath as a
    floor, so an event whose ringing dies inside its own run keeps the
    margin it always had.

    No two windows share a sample: each is capped at the next run's
    rise, and each opens no earlier than the previous window closed —
    a follow that ran long must not have the next event's lead-in
    reaching back through it. A run that never settles before the next
    one keeps the whole gap, which is the honest answer for a record
    still ringing when it was hit again.
    """
    quiet = floor_of(loud)
    quiet_line = (quiet + RELEASE) if quiet is not None else np.inf
    out: list[tuple[int, int]] = []
    for k, (first, last) in enumerate(runs):
        span = max(last - first, 1)
        earliest = out[-1][1] if out else 0
        start = max(first - round(span * lead), earliest, 0)
        limit = hops if k == len(runs) - 1 else runs[k + 1][0]
        j0 = int(np.ceil(last / scale))
        j1 = min(int(limit / scale), len(loud))
        peak = loud[int(first / scale):max(j0, int(first / scale) + 1)]
        line = max(quiet_line, float(peak.max()) - settled)             if peak.size else quiet_line
        under = np.nonzero(loud[j0:j1] < line)[0]
        settle = round((j0 + (int(under[0]) if under.size else j1 - j0))
                       * scale)
        stop = max(settle, last + round(span * tail))
        out.append((start, min(stop, limit, hops)))
    return out


def uniform(shocks: Sequence[Shock], places: int = 6) -> bool:
    """Whether these windows are all one length.

    The mode is read off the windows rather than stored beside them. A
    second copy of it in the project file is a copy that can disagree
    with the lengths it describes, and then the checkbox and the data
    say different things about the same series — so there is one place
    it lives, and it is the series.

    Rounded, because the lengths are built by rounding to samples and
    two windows meant to match can differ in the last bit.
    """
    lengths = {round(s.duration, places) for s in shocks}
    return len(lengths) == 1


#: below this a duration change is jitter, not a resize
DRAG_EPSILON = 1e-6


def with_added(shocks: Sequence[Shock],
               limit: float) -> tuple[Shock, ...]:
    """The series with one more window, placed where there is room.

    The Add button's rule (Brandon, 2026-08-24): the detector misses
    events the user knows about — below the arm threshold, or close
    under a louder neighbor — and Remove's rationale cuts both ways.
    The new window takes the series' own length (they are usually
    uniform; the median otherwise, a tenth of the record when there
    is nothing to copy), and stands in the middle of the largest
    unwindowed stretch, shrunk when even the largest gap is tighter
    than that. The caller drags or types it into place from there.
    """
    ordered = tuple(sorted(shocks, key=lambda shock: shock.start))
    if ordered:
        durations = sorted(shock.duration for shock in ordered)
        length = durations[len(durations) // 2]
    else:
        length = limit / 10.0
    edges = [0.0]
    for shock in ordered:
        edges.extend([shock.start, shock.stop])
    edges.append(float(limit))
    gaps = [(edges[k + 1] - edges[k], edges[k])
            for k in range(0, len(edges), 2)]
    width, low = max(gaps)
    if width <= 0:
        return ordered            # wall to wall; nothing fits anywhere
    length = min(length, width * 0.9)
    start = low + (width - length) / 2.0
    return tuple(sorted(ordered + (Shock(start, length),),
                        key=lambda shock: shock.start))


def drag_settled(shocks: Sequence[Shock], index: int, low: float,
                 high: float, common: bool,
                 limit: float | None) -> tuple[Shock, ...] | None:
    """What a drag of one window's edges means for the whole series.

    One implementation for both editors — the 2-D regions and the
    stage's handles commit through this. *Which* window moved is
    always just the dragged one; what can be shared is the *length*
    (an analysis choice, not a measurement), so a resize under a
    shared length regrows every window, while a move never resizes a
    neighbor. Either way the result is stopped at its neighbors.

    Returns the settled series, or None when the drag changed nothing
    or asked for something the series has no room for — the caller
    puts its handles back.
    """
    if index >= len(shocks) or high - low <= 0:
        return None
    was = shocks[index]
    moved = Shock(max(low, 0.0), high - max(low, 0.0))
    if moved == was:
        return None
    edited = tuple(moved if k == index else shock
                   for k, shock in enumerate(shocks))
    resized = abs(moved.duration - was.duration) > DRAG_EPSILON
    settled = (same_length(edited, moved.duration, limit)
               if common and resized
               else held_apart(edited, moved=index))
    return None if settled == tuple(shocks) else settled


def held_apart(shocks: Sequence[Shock],
               moved: int | None = None) -> tuple[Shock, ...]:
    """The windows with no two of them sharing a sample.

    `_windowed` guarantees this at detection, but a window can also be
    dragged on the plot or typed into the table, and neither of those
    went through it — so one shock's window could be pulled over its
    neighbor and its spectrum computed from the neighbor's ringdown.
    This is where that invariant is restated, once, for every path that
    can break it.

    `moved` is which window the edit was aimed at, and it decides who
    yields. Given it, that window is held inside the gap its untouched
    neighbors leave: drag an edge into the next event and it stops at
    the edge, which is what a person pulling it expects to happen and
    would not expect of the event they did not touch. Without it — a
    list arriving from somewhere with no single author — the earlier
    window of an overlapping pair is cut back to where the later one
    opens. Neither rule moves a window that was not in the way, so
    neither can cascade down the record from one bad edit.
    """
    out = list(shocks)
    if len(out) < 2:
        return tuple(out)
    if moved is not None and 0 <= moved < len(out):
        one = out[moved]
        low = out[moved - 1].stop if moved > 0 else 0.0
        high = out[moved + 1].start if moved + 1 < len(out) else None
        start = max(one.start, low)
        stop = one.stop if high is None else min(one.stop, high)
        if stop > start:
            out[moved] = Shock(start, stop - start)
        return tuple(out)
    for k in range(len(out) - 1):
        if out[k + 1].start < out[k].stop:
            room = out[k + 1].start - out[k].start
            if room > 0:
                out[k] = Shock(out[k].start, room)
    return tuple(out)


def same_length(shocks: Sequence[Shock], length: float,
                limit: float | None = None) -> tuple[Shock, ...]:
    """Every window given `length`, each keeping its own start.

    What a *resize* does to a series held at one length: pulling any
    edge sets the length for all of them, and they grow together until
    the closest pair runs out of room. Clamping only the window that
    collided and leaving the rest is how a uniform series quietly stops
    being uniform, which is the failure the mode exists to prevent, so
    the cap is shared too.

    The starts are not touched, and that is the whole division of
    labor: a start belongs to the event it was measured from, so
    moving one window is moving one window, and only the length — the
    part that is an analysis choice rather than a measurement — is held
    in common.
    """
    out = list(shocks)
    if not out:
        return ()
    starts = [s.start for s in out]
    length = float(length)
    if len(starts) > 1:
        length = min(length, min(b - a for a, b in pairwise(starts)))
    if limit is not None:
        length = min(length, float(limit) - starts[-1])
    if length <= 0.0:
        return tuple(out)
    return tuple(Shock(start, length) for start in starts)


def find(history: TimeHistory, hop_seconds: float = HOP_SECONDS,
         arm: float = ARM, release: float = RELEASE,
         quantile: float = FLOOR_QUANTILE, merge: float = MERGE,
         lead: float = LEAD, tail: float = TAIL,
         minimum: float = MIN_DURATION,
         most: int = MAX_EVENTS,
         common: bool = COMMON_LENGTH) -> tuple[Shock, ...]:
    """The shocks in a record, as the windows to analyze them in.

    Empty when there are none — which is the right answer for a random
    vibration run, where the level never rises `arm` dB over its own
    quiet because there is no quiet.

    `common` gives every event a window of one length, which is the
    default and the reasoning is at `COMMON_LENGTH`. Pass False for a
    record whose events are not the same kind of event.
    """
    level, hop = envelope(history, hop_seconds)
    floor = floor_of(level, quantile)
    if floor is None or level.size < 2:
        return ()
    rate = float(history.sample_rate)
    runs = _runs(level, floor + float(arm), floor + float(release))
    if not runs:
        return ()
    runs = _merged(runs, max(round(float(merge) * rate / hop), 1))
    loud = loudest(history)
    scale = hop_for(rate, RING_HOP) / hop
    windows = _windowed(runs, len(level), loud, scale, lead, tail)

    samples = np.asarray(history.ordinate).shape[-1]
    found = []
    for first, last in windows:
        start = first * hop / rate
        duration = (last - first) * hop / rate
        if duration < float(minimum):
            continue
        found.append(Shock(start, duration).clipped(samples, rate))
    found = found[:int(most)]
    if common and len(found) > 1:
        # one length for the series: the longest, so no event is
        # truncated, capped inside the record — the reasoning is at
        # COMMON_LENGTH, and same_length is the same rule a drag obeys
        found = list(same_length(found,
                                 max(s.duration for s in found),
                                 limit=samples / rate))
    return tuple(found)


def suggest(history: TimeHistory, **kwargs: Any) -> tuple[Shock, ...]:
    """`find`, or the whole record when it finds nothing.

    What a caller wanting *something* to analyze asks for. A record with
    no quiet in it has no shock the detector can point at, but it is
    still a transient somebody wants a spectrum of, and the honest
    fallback is the record itself rather than nothing.
    """
    found = find(history, **kwargs)
    if found:
        return found
    samples = np.asarray(history.ordinate).shape[-1]
    rate = float(history.sample_rate)
    if samples < 2:
        return ()
    return (Shock(0.0, samples / rate),)
