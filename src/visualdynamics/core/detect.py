"""Finding the stretch of a record worth averaging.

A run holds more than the test: the shaker coming up, a reduced-level
check, the full level, and whatever was still recording after the
environment stopped. Averaging the lot mixes them into one PSD that
describes none of them. This works out which stretch to use.

The record is compressed to one number per hop before anything else is
decided — a level, in dB — and every judgment after that is made on
that short series. That is what keeps it quick on the sets where it
matters: three hundred channels of a ten-minute run cost one pass, and
the rest is arithmetic on a few thousand points.

Two things make the compression trustworthy rather than merely small:

- the consensus is the median across channels, not the mean, so a
  channel that drops out or spikes or sits forty dB above its
  neighbors is one vote and not the answer;
- the mean is removed within each hop, so a DC offset or a slow drift
  is not read as level.

Nothing normalizes the levels against each other or centers the series.
Both were tried: the median across channels already does that work, and
the threshold sweep walks every bin edge from the top, so scaling a
whole record by a million moves the bins together and lands on the same
hops. Neither changed an answer, so neither is here.

The level is in dB rather than in power because a level change is
multiplicative. In power, scatter grows with the level, so the flattest
stretch of a record is reliably its quietest — which is the opposite of
what is wanted. In dB a quiet stretch and a loud one scatter alike, and
flat means flat.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .averaging import Averaging

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

#: how much of a detected stretch to keep, as fractions of its length.
#: Closed-loop control does not arrive at a new level instantly — under
#: match-trace especially it walks in over several frames — so the front
#: is trimmed hardest. The tail is trimmed a little too, against the
#: level starting to come off before the environment formally stops.
KEEP = (0.20, 0.90)

#: frames the detector tries to find room for before it will settle for
#: a shorter stretch at a higher level
WANT_FRAMES = 30

#: how far below a plateau's own level a hop may sit and still belong to
#: it, in multiples of the scatter the record actually shows.
#:
#: Two, and it has to be about two. A random run's levels step by 3 dB,
#: and the full-level stretch of a closed-loop run wanders by about a dB
#: hop to hop, so the tolerance must be wide enough to hold a plateau
#: together and narrower than the step that separates two of them.
#: Both fit because a run is grown from its own median rather than read
#: off the threshold that found it: the threshold only has to land
#: somewhere inside a level, and the growing does the rest.
TOLERANCE = 2.0

#: where the top level is taken to be, as a percentile of the hops. Low
#: enough to sit inside the highest level rather than on its noisiest
#: hop, high enough that a top level occupying a twentieth of a record
#: still sets it.
TOP_QUANTILE = 95.0

#: how many times a run is re-centered on itself and grown again. Three
#: is enough for anything seen: the first pass from a fragment's own
#: median, the second from the median of what that reached, and the
#: third to confirm it has stopped moving.
GROW_PASSES = 3

#: channels are read in chunks this many samples wide, so the working
#: array stays a few tens of megabytes whatever the record's size
CHUNK_SAMPLES = 4_000_000


def frame_length_for(sample_rate: float) -> int:
    """The nearest power of two to one second of record.

    A power of two for the FFT, and about a second because that is the
    resolution a random vibration test is usually specified at. On a
    rate that is itself a power of two — which most are — the two come
    out the same number.
    """
    if sample_rate <= 0:
        raise ValueError('a sample rate must be positive')
    return int(2 ** max(round(np.log2(float(sample_rate))), 1))


def scatter_floor(samples: int) -> float:
    """The least a stretch can scatter, in dB, for estimator noise alone.

    A power estimated from `samples` samples has at most that many
    degrees of freedom, and a chi-squared with dof has a relative
    standard deviation of sqrt(2/dof); in dB that is 4.343 times it, for
    the small values this deals in.

    A floor and not an estimate. Real records scatter more, and by a
    lot: the full-level stretch of a closed-loop run scatters about a dB
    against a floor here of a quarter of one, because the loop is
    adjusting the whole time and the response really is wandering.
    Band-limited data has fewer independent lines than samples, too. So
    this bounds the answer from below and `observed_scatter` supplies
    the rest.
    """
    return 4.343 * np.sqrt(2.0 / max(float(samples), 2.0))


def observed_scatter(level: np.ndarray) -> float:
    """How much this record actually scatters hop to hop, in dB.

    From the first differences, which a level *change* barely touches —
    one step in a hundred hops moves one difference — so what is left is
    the wander within a level. Robust rather than a standard deviation,
    for the same reason: the steps must not set the tolerance for what
    counts as the same level.
    """
    if len(level) < 3:
        return 0.0
    steps = np.diff(level)
    spread = float(np.median(np.abs(steps - np.median(steps))))
    # MAD to sigma for a normal, then the difference of two hops is
    # sqrt(2) times as noisy as one
    return spread / 0.6745 / np.sqrt(2.0)


def hop_levels(ordinate: ArrayLike, hop: int,
               sparse: bool = False, loudest: bool = False) -> np.ndarray:
    """The record as one level per hop, in dB.

    Returns an empty array when there is not a whole hop of record, or
    when every channel is flat — a file of zeros has no level to speak
    of, and saying so is better than reporting a floor.

    `loudest` reduces across channels by the maximum instead of the
    median. The median is the right voter for *detection* — one bad
    channel is a vote, not the answer — but it is blind to one channel
    still ringing under twenty quiet ones, and a shock window sized on
    it closes while that channel's spectrum is still being written.

    `sparse` changes what counts as a flat channel, and it exists
    because the two detectors ask different questions of the same
    compression. A stationary run's channel is live when its *median*
    hop carries power: anything quieter than that half the time is a
    channel that dropped out. A shock record's channel is quiet most of
    the time by definition — three events in a minute leaves the median
    hop empty — so there, a channel is live if it carries power
    anywhere. Judged by the stationary rule, a record of sparse
    transients reports no level at all.
    """
    ordinate = np.asarray(ordinate)
    if ordinate.ndim == 1:
        ordinate = ordinate[None]
    hop = int(hop)
    count = ordinate.shape[1] // hop
    if count < 1:
        return np.zeros(0)

    channels = ordinate.shape[0]
    power = np.empty((channels, count))
    # by hops rather than by channels: the temporary is the chunk, and
    # squaring a whole 300-channel record at once is gigabytes
    per_chunk = max(CHUNK_SAMPLES // max(channels * hop, 1), 1)
    for first in range(0, count, per_chunk):
        last = min(first + per_chunk, count)
        block = ordinate[:, first * hop:last * hop].reshape(
            channels, last - first, hop)
        mean = block.mean(axis=2)
        # einsum contracts without materializing the squares
        power[:, first:last] = (
            np.einsum('cks,cks->ck', block, block) / hop - mean * mean)

    power = np.maximum(power, 0.0)          # rounding, on a flat channel
    live = (power.max(axis=1) if sparse else np.median(power, axis=1)) > 0
    if not live.any():
        return np.zeros(0)
    across = power[live].max(axis=0) if loudest else np.median(
        power[live], axis=0)
    return 10.0 * np.log10(np.maximum(across, 1e-30))


def _longest_run(mask):
    """(first, last) of the longest run of True, last exclusive."""
    best = (0, 0)
    start = None
    for i, ok in enumerate([*mask, False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def _grow(level, first, last, sigma, tolerance=TOLERANCE):
    """Extend a run outward to the whole of the level it sits in.

    A threshold swept in steps lands where the steps fall, not where the
    level changes, so the run it finds is a fragment of a plateau rather
    than the plateau. Grown from that fragment's own median it reaches
    both ends of the level and stops at the step to the next one —
    which is three dB away, and the tolerance is one.

    Bounded above as well as below, so a plateau does not grow into an
    overload sitting next to it.

    Repeated until it settles, because the first median is the median of
    a fragment and a fragment of a wandering plateau sits wherever the
    threshold happened to cut it. Grown once from a median near the top
    of the wander it reaches only the upper half; re-centered on what it
    then holds, it reaches the rest.
    """
    for _ in range(GROW_PASSES):
        if last <= first:
            return first, last
        mid = np.median(level[first:last])
        inside = np.abs(level - mid) <= tolerance * sigma
        low, high = first, last
        while low > 0 and inside[low - 1]:
            low -= 1
        while high < len(level) and inside[high]:
            high += 1
        if (low, high) == (first, last):
            break
        first, last = low, high
    return first, last


def plateau(level: np.ndarray, sigma: float, want_hops: int,
            tolerance: float = TOLERANCE) -> tuple[int, int]:
    """The highest level the record holds for `want_hops` hops.

    Level alone is the wrong thing to maximize. A ten-second overload in
    the middle of a two-minute test is the highest level in the record
    and the last thing anyone wants to average; sought that way it wins,
    and returns a fifth of the frames the test itself would have. So
    duration comes first and level second.

    Sweeping a threshold down from the top does both at once: the
    longest run above a threshold only grows as the threshold falls, so
    the first threshold whose run is long enough is the highest level
    that is held long enough. When nothing is held that long — a short
    record, or a test that never settled — the longest stretch near the
    top is taken instead, and it is the caller's business that it is
    shorter than asked for.
    """
    if len(level) == 0:
        return (0, 0)
    step = max(tolerance * sigma, 1e-9)
    for edge in np.sort(np.unique(np.round(level / step)))[::-1]:
        first, last = _longest_run(level >= (edge - 0.5) * step)
        if last - first >= want_hops:
            return _grow(level, first, last, sigma, tolerance)

    # Nothing is held that long, and the threshold must not simply keep
    # falling until something is. On a record that is eight seconds of
    # test inside eighty of quiet, what is held long enough is the
    # quiet, and an analysis started there is an analysis of the noise
    # floor — the exact mistake the whole exercise is meant to avoid.
    #
    # So take the highest level the record sustains at all, however
    # briefly, and all of it. Fewer frames than were asked for, of the
    # right thing.
    return _grow(level, *_top_level_run(level, sigma, tolerance),
                 sigma, tolerance)


def _top_level_run(level: np.ndarray, sigma: float,
                   tolerance: float = TOLERANCE) -> tuple[int, int]:
    """The longest stretch at the highest level the record sustains.

    The anchor is a high quantile and not the maximum. A maximum is
    biased upward by however much the record scatters — on a run
    wandering by a dB it lands two above the level it is meant to
    describe, the tolerance below it then excludes the level's own
    typical hops, and what comes back is a second and a half of noise
    excursion. A quantile sits inside the top level rather than above
    it, and the growing settles the rest.
    """
    if len(level) == 0:
        return (0, 0)
    top = float(np.percentile(level, TOP_QUANTILE))
    return _longest_run(level >= top - max(tolerance * sigma, 1e-9))


def trim(first: int, last: int,
         keep: tuple[float, float] = KEEP) -> tuple[int, int]:
    """Take the settled middle of a stretch.

    Closed-loop control arrives at a level over several frames rather
    than at once, so the front of a stretch is not yet the level the
    rest of it is. The tail is trimmed a little as well, because the
    level often starts to come off before the environment formally
    stops. What is left is the part that is all one thing.
    """
    low, high = keep
    if not 0.0 <= low < high <= 1.0:
        raise ValueError('keep is a pair of fractions, low < high, in [0, 1]')
    span = last - first
    if span <= 0:
        return first, last
    inner = (first + int(np.floor(span * low)),
             first + int(np.ceil(span * high)))
    # a stretch too short to trim is better used whole than lost
    return inner if inner[1] > inner[0] else (first, last)


def compress(history: TimeHistory, overlap: float = 0.5,
             frame_length: int | None = None
             ) -> tuple[np.ndarray, int, int, float]:
    """(levels in dB, hops per frame, hop in samples, scatter floor).

    The one pass over the samples, and everything a caller needs to
    reason about the record afterwards. Separated out because more than
    one question gets asked of the same compression, and asking twice
    means reading three gigabytes twice.
    """
    ordinate = np.asarray(history.ordinate)
    if ordinate.ndim == 1:
        ordinate = ordinate[None]
    samples = ordinate.shape[1]
    length = int(frame_length or frame_length_for(history.sample_rate))
    length = max(min(length, samples), 2)
    hop = max(round(length * (1.0 - overlap)), 1)
    level = hop_levels(ordinate, hop)
    # whichever of the two scatters is larger is the one to allow for
    sigma = max(scatter_floor(hop), observed_scatter(level))
    return level, max(length // hop, 1), hop, sigma


def analysis_window(history: TimeHistory, want_frames: int = WANT_FRAMES,
                    overlap: float = 0.5, frame_length: int | None = None,
                    keep: tuple[float, float] = KEEP) -> tuple[int, int]:
    """(first sample, last sample) worth averaging in this history.

    The whole record when there is nothing to choose between one part of
    it and another — a noise floor capture is a level like any other,
    and the point of finding it is to be able to put a PSD on it.
    """
    samples = np.asarray(history.ordinate).reshape(
        -1, len(history.abscissa)).shape[1]
    level, per_frame, hop, sigma = compress(history, overlap, frame_length)
    if len(level) == 0:
        return 0, samples
    # frames to hops: a frame spans length/hop of them, and F frames
    # spanning from one start take (F - 1) hops more
    first, last = plateau(level, sigma, want_frames - 1 + per_frame)
    first, last = trim(first, last, keep)
    return first * hop, min(last * hop, samples)


def suggest(history: TimeHistory, want_frames: int = WANT_FRAMES,
            window: str = 'hann', overlap: float = 0.5,
            frame_length: int | None = None,
            keep: tuple[float, float] = KEEP) -> Averaging:
    """The averaging to use for a record nobody has set parameters on.

    Hann at half overlap, a frame about a second long, and as many
    frames as the settled stretch will carry — which is the answer to
    'how many averages', rather than a number chosen in advance and
    hoped for.
    """
    rate = history.sample_rate
    samples = len(history.abscissa)
    length = int(frame_length or frame_length_for(rate))
    length = max(min(length, samples), 2)
    first, last = analysis_window(history, want_frames=want_frames,
                                  overlap=overlap, frame_length=length,
                                  keep=keep)
    averaging = Averaging(frame_length=length, overlap=overlap,
                          window=window, frames=1, start=first / rate)
    room = averaging.most_frames(last, rate)
    return Averaging(frame_length=length, overlap=overlap, window=window,
                     frames=max(room, 1), start=first / rate)
