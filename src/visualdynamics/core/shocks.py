"""Find the shocks in a recording, and the window to analyze each one in.

A shock test is a series of events in one continuous record: the
article is quiet, something hits it, it rings down, and it is quiet
again until the next one. Nothing in the file says where the events
are. `detect` finds the one stretch of a stationary run to average;
this finds every event in a transient run, and a record usually holds
more than one.

As in `detect`, the record is first compressed to one level per hop
(1 ms by default) across all channels at once, and everything after
that is decided on that short series. A shock record can be minutes
long at 50 kHz, so every later step works on thousands of points
instead of millions of samples.

Three choices set how the events are found:

- The trigger has hysteresis (a Schmitt trigger). An event starts when
  the level rises `ARM` dB (12) over the floor and ends only when it
  falls back under `RELEASE` dB (6), so a ringdown is followed down
  instead of being cut where it dips. With a single threshold, a
  ringdown crosses it on every cycle and one event comes back as forty.
- The floor is a low quantile of the levels (the 10th percentile).
  Shocks are short and large, and averaging them into the floor would
  raise it by tens of dB; a low quantile stays in the quiet however
  large or numerous the shocks are.
- The thresholds are measured from the floor, not from each event's
  peak. A shock series is usually run up in level, so the last event
  can be ten times the first, and thresholds relative to each peak
  would give the events different windows for no physical reason.

The window is longer than the event at both ends. It opens before the
rise (by `LEAD`, 0.15 of the event's length), because an SRS oscillator has
to start from rest, and a window that opens partway into the pulse adds
a step that was never there. It closes past the release, once the
loudest channel's ringing has settled `SETTLED` dB (30) under its peak
or reached the quiet, because the low-frequency oscillators are still
responding to the ringdown above the floor. By default every event in
one record gets the same window length (`COMMON_LENGTH`).
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

    `start` and `duration` are in seconds, `start` from the beginning
    of the record; everything else is derived from the two. The window
    already includes the lead-in and the tail, since an SRS is computed
    from the window, and storing the window saves working it out again
    each time it is used. A window with no length, or one that starts
    before the record, is refused.
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
        """When the window closes, in seconds from the start of the
        record."""
        return self.start + self.duration

    def start_sample(self, sample_rate: float) -> int:
        """The index of the sample the window opens on, at
        `sample_rate`."""
        return round(self.start * sample_rate)

    def samples(self, sample_rate: float) -> int:
        """How many samples the window holds at `sample_rate`, at least
        two, since one sample is not a transient."""
        return max(round(self.duration * sample_rate), 2)

    def bounds(self, sample_rate: float) -> tuple[int, int]:
        """(first, last) sample indices at `sample_rate`, last
        exclusive."""
        first = self.start_sample(sample_rate)
        return first, first + self.samples(sample_rate)

    def fits(self, samples: int, sample_rate: float) -> bool:
        """Whether the window ends inside a record of `samples`
        samples."""
        return self.bounds(sample_rate)[1] <= samples

    def clipped(self, samples: int, sample_rate: float) -> Shock:
        """This window moved and cut to lie inside a record of `samples`
        samples, at least two samples long."""
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
    """Samples per hop of `seconds` at `sample_rate`, at least one."""
    return max(round(float(seconds) * float(sample_rate)), 1)


def envelope(history: TimeHistory, seconds: float = HOP_SECONDS
             ) -> tuple[np.ndarray, int]:
    """(levels in dB, hop in samples): the record compressed to one
    level per hop of `seconds`.

    This is the compression the averaging detector uses
    (`detect.hop_levels`): the median across channels, so one bad
    channel does not decide the answer, with the mean removed within
    each hop, so a DC offset is not read as level. The hop is
    milliseconds instead of a large fraction of a second, because the
    events last milliseconds. A channel counts as live if it carries
    power anywhere, since a shock record's channels are quiet most of
    the time.
    """
    hop = hop_for(history.sample_rate, seconds)
    # sparse: a shock record's channels are quiet most of the time, and
    # the stationary rule for a live channel would call them all dead
    return hop_levels(history.ordinate, hop, sparse=True), hop


def loudest(history: TimeHistory, seconds: float = RING_HOP
            ) -> np.ndarray:
    """The loudest channel's level per hop of `seconds`, in dB (25 ms by
    default, `RING_HOP`).

    The end of each window is found on this series. Detection uses the
    median across channels, but an SRS is one spectrum per channel,
    and a window sized on the median would cut short the spectrum of
    one channel still ringing under twenty quiet ones. The maximum
    across channels shows that channel. The coarse hop is what lets the
    series show slow ringing at all: a hop's level is the variance
    within the hop, so content much slower than the hop is removed with
    the mean.
    """
    hop = hop_for(history.sample_rate, seconds)
    return hop_levels(history.ordinate, hop, sparse=True, loudest=True)


def floor_of(level: np.ndarray, quantile: float = FLOOR_QUANTILE,
             dynamic: float = DYNAMIC) -> float | None:
    """The quiet level the events are measured against, in dB, or None
    for an empty series.

    The floor is the `quantile` percentile of the levels (10 by
    default), not a mean or a median: shocks are short and large, and
    any average of a record holding them sits well above the quiet
    between them. It is never more than `dynamic` dB (100) under the
    loudest hop, so a synthesized record whose gaps are exact zeros
    still gets thresholds that mean something (see `DYNAMIC`).
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
        line = (max(quiet_line, float(peak.max()) - settled)
                if peak.size else quiet_line)
        under = np.nonzero(loud[j0:j1] < line)[0]
        settle = round((j0 + (int(under[0]) if under.size else j1 - j0))
                       * scale)
        stop = max(settle, last + round(span * tail))
        out.append((start, min(stop, limit, hops)))
    return out


def uniform(shocks: Sequence[Shock], places: int = 6) -> bool:
    """Whether these windows are all one length, compared to `places`
    decimal places of a second.

    The common-length mode is read off the windows instead of being
    stored beside them. A stored copy could disagree with the lengths
    it describes, and then the checkbox and the data would say
    different things about the same series. The lengths are rounded
    because they are built by rounding to samples, and two windows
    meant to match can differ in the last bit.
    """
    lengths = {round(s.duration, places) for s in shocks}
    return len(lengths) == 1


#: below this a duration change is jitter, not a resize
DRAG_EPSILON = 1e-6


def with_added(shocks: Sequence[Shock],
               limit: float) -> tuple[Shock, ...]:
    """The series with one more window, placed where there is room.

    This is what the Add button does: the detector can miss events the
    user knows about, below the arm threshold or close under a louder
    neighbor. `limit` is the record's length in seconds. The new window
    takes the median length of the series (the common length, when they
    share one), or a tenth of `limit` when there are no windows yet. It
    is centered in the largest stretch with no window in it, and is
    never longer than 90 % of that stretch. With no room anywhere the series comes back unchanged.
    The caller drags or types the new window into place from there.
    """
    # The Add button's rule (Brandon, 2026-08-24): the detector misses
    # events the user knows about, and the reasoning that gave the
    # user Remove applies the other way as well.
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

    `index` is the dragged window, and `low` and `high` are its new
    edges in seconds. `common` says the series is held at one length,
    and `limit` is the record's length in seconds, or None. The 2-D
    regions and the 3-D stage's handles both commit through this. Only
    the dragged window moves. A resize under a common length gives
    every window the new length (`same_length`), since the length is an
    analysis choice and not a measurement; a move, or any change
    without a common length, holds the dragged window inside the gap
    its neighbors leave (`held_apart`).

    Returns the settled series, or None when the drag changed nothing
    or asked for something the series has no room for, in which case
    the caller puts its handles back.
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

    Detection already keeps the windows apart, but a window can also be
    dragged on the plot or typed into the table, and without this one
    shock's window could be pulled over its neighbor and its spectrum
    computed from the neighbor's ringdown. This is the one place that
    rule is applied after detection.

    `moved` is the index of the window the edit was aimed at, and it
    decides which window gives way. Given it, that window is held inside
    the gap its neighbors leave: an edge dragged into the next event
    stops at that event's edge, and the event that was not touched does
    not move. Without it, for a list that did not come from one edit,
    the earlier window of each overlapping pair is cut back to where the
    later one opens. Neither rule moves a window that was not in the
    way, so one bad edit cannot cascade down the record.
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
    """Every window given `length`, in seconds, each keeping its own
    start.

    This is what a resize does to a series held at one length: pulling
    any edge sets the length for all of them. The length is capped by
    the closest pair of starts, and by `limit` (the record's length in
    seconds, when given) less the last start. Capping only the window
    that collided would leave the series no longer uniform, which is
    what the mode exists to prevent, so the cap is shared. The starts
    are not touched, because a start belongs to the event it was
    measured from; only the length, an analysis choice, is held in
    common. When the cap leaves no length, the windows come back
    unchanged.
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

    The defaults are the module's constants: `hop_seconds` is the
    detection hop in seconds (`HOP_SECONDS`), `arm` and `release` are dB
    over the floor, `quantile` is the floor's percentile, events closer
    than `merge` seconds are joined, `lead` and `tail` are fractions of
    each event's length, windows shorter than `minimum` seconds are
    dropped, and at most `most` windows are returned.

    Empty when there are none, which is the right answer for a random
    vibration run: its level never rises `arm` dB over its own floor
    because there is no quiet in it.

    `common` (the default, `COMMON_LENGTH`) gives every event the length
    of the longest window, capped so that no two windows overlap and
    none runs past the record. Pass False for a record whose events are
    not the same kind of event.
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
    """`find`, or one window over the whole record when it finds nothing.

    This is for a caller that needs something to analyze. A record with
    no quiet in it has no shock the detector can point at, but it is
    still a transient someone wants a spectrum of. Keyword arguments go
    to `find`. Empty only for a record of fewer than two samples.
    """
    found = find(history, **kwargs)
    if found:
        return found
    samples = np.asarray(history.ordinate).shape[-1]
    rate = float(history.sample_rate)
    if samples < 2:
        return ()
    return (Shock(0.0, samples / rate),)
