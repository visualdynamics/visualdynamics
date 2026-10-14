"""Find the stretch of a record worth averaging.

A run holds more than the test: the shaker coming up, a reduced-level
check, the full level, and whatever was still recorded after the
environment stopped. Averaging all of it mixes them into one PSD that
describes none of them. This finds which stretch to use.

The record is first compressed to one level per hop, in dB, and every
decision after that is made on that short series. That keeps it quick
on large sets: three hundred channels of a ten-minute run take one pass
over the samples, and the rest is arithmetic on a few thousand points.

Two choices make the compression reliable:

- the levels are combined across channels by the median, not the mean,
  so a channel that drops out, spikes, or sits forty dB above its
  neighbors does not decide the answer;
- the mean is removed within each hop, so a DC offset or a slow drift
  is not read as level.

The levels are not normalized against each other and the series is not
centered. The median across channels already does that work, and the
threshold sweep walks every bin edge from the top, so scaling a whole
record by a million moves the bins together and lands on the same hops.

The level is in dB, not power, because a level change is
multiplicative. In power the scatter grows with the level, so the
flattest stretch of a record is reliably its quietest, the opposite of
what is wanted. In dB a quiet stretch and a loud one scatter alike.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .averaging import Averaging

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

# Normalizing the levels against each other and centering the series
# were both tried; neither changed an answer, so neither is here.

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

#: how far below the record's top level a stretch may sit and still be
#: the test, in dB. The threshold sweep in `plateau` stops here: a run
#: eight seconds at level inside eighty of quiet holds no thirty frames
#: at level, and sweeping on found thirty frames of noise floor — the
#: importer's own guard against exactly that, folded into the detector
#: once the import became the Detect answer (2026-09-19). Twelve sits
#: between the two things it must tell apart: an overload nearly ten
#: decibels above the test (duration still wins, as it must) and an
#: ambient twenty and more below it (duration must not).
FULL_LEVEL_MARGIN = 12.0

#: how many times a run is re-centered on itself and grown again. Three
#: is enough for anything seen: the first pass from a fragment's own
#: median, the second from the median of what that reached, and the
#: third to confirm it has stopped moving.
GROW_PASSES = 3

#: channels are read in chunks this many samples wide, so the working
#: array stays a few tens of megabytes whatever the record's size
CHUNK_SAMPLES = 4_000_000


def frame_length_for(sample_rate: float) -> int:
    """The power of two nearest one second of record at
    `sample_rate`, in samples (nearest on a log scale, at least 2).

    A power of two for the FFT, and about a second because that is the
    resolution (1 Hz) a random vibration test is usually specified at.
    When the rate is itself a power of two, as most are, the two are
    the same number. A rate that is not positive is refused.
    """
    if sample_rate <= 0:
        raise ValueError('a sample rate must be positive')
    return int(2 ** max(round(np.log2(float(sample_rate))), 1))


def scatter_floor(samples: int) -> float:
    """The least a stretch can scatter, in dB, from estimator noise
    alone, for a power estimated from `samples` samples.

    That power has at most `samples` degrees of freedom, and a
    chi-squared variable with dof degrees of freedom has a relative
    standard deviation of sqrt(2/dof); in dB that is 4.343 times it, for
    the small values used here.

    This is a lower bound, not an estimate. Real records scatter more:
    the full-level stretch of a closed-loop run scatters about a dB
    against a floor here of about a quarter of one, because the loop is
    adjusting the whole time and the response does wander. Band-limited
    data also has fewer independent values than samples. The larger of
    this and `observed_scatter` is used.
    """
    return 4.343 * np.sqrt(2.0 / max(float(samples), 2.0))


def observed_scatter(level: np.ndarray) -> float:
    """How much this record actually scatters hop to hop, in dB.

    It is taken from the first differences of the levels, which a level
    change barely affects (one step in a hundred hops moves one
    difference), so what is left is the wander within a level. It uses
    the median absolute deviation, scaled to a standard deviation,
    instead of the standard deviation itself, for the same reason: the
    steps must not set the tolerance for what counts as the same level.
    0 for fewer than three hops.
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
    """The record as one level per hop of `hop` samples, in dB.

    Each channel's power in a hop is its variance within the hop (the
    mean removed), and the live channels are combined by the median.
    Returns an empty array when there is not a whole hop of record, or
    when every channel is flat: a file of zeros has no level, and
    reporting a floor for it would be wrong.

    `loudest` combines the channels by the maximum instead of the
    median. The median is the right choice for detection, where one bad
    channel should not decide the answer, but it cannot see one channel
    still ringing under twenty quiet ones, and a shock window sized on
    it closes while that channel is still responding.

    `sparse` changes what counts as a flat channel, because the two
    detectors ask different questions of the same compression. A
    stationary run's channel is live when its median hop carries power:
    a channel quieter than that half the time has dropped out. A shock
    record's channel is quiet most of the time (three events in a
    minute leave the median hop empty), so with `sparse` a channel is
    live if it carries power anywhere. Judged by the stationary rule, a
    record of sparse transients reports no level at all.
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
    """The highest level the record holds for `want_hops` hops, as
    (first, last) hops, last exclusive.

    `sigma` is the scatter in dB, and `tolerance` is how many of it a
    hop may sit from the plateau's level. Level alone is the wrong
    thing to maximize: a ten-second overload in the middle of a
    two-minute test is the highest level in the record and the last
    thing anyone wants to average, and chosen by level it would return
    a fifth of the frames the test holds. So duration comes first and
    level second.

    A threshold is swept down from the top in steps of `tolerance`
    times `sigma`. The longest run above a threshold only grows as the
    threshold falls, so the first threshold whose run is long enough
    gives the highest level that is held long enough. The sweep stops
    `FULL_LEVEL_MARGIN` (12 dB) under the record's top level (its
    `TOP_QUANTILE` percentile), since below that a run is not the test
    at full level however long it lasts. The run found is grown to the
    whole level it sits in. When nothing is held that long, as in a
    short record or a test that never settled, the longest stretch at
    the highest level the record holds is returned instead, shorter
    than asked for.
    """
    if len(level) == 0:
        return (0, 0)
    step = max(tolerance * sigma, 1e-9)
    # the sweep stops a margin under the top: below that a run is not
    # the test at level, however long it is held
    floor = float(np.percentile(level, TOP_QUANTILE)) - FULL_LEVEL_MARGIN
    for edge in np.sort(np.unique(np.round(level / step)))[::-1]:
        if (edge - 0.5) * step < floor:
            break
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
    """The settled middle of the stretch (`first`, `last`), keeping
    the fractions `keep` of its length (`KEEP`, 0.20 to 0.90).

    Closed-loop control reaches a level over several frames instead of
    at once, so the front of a stretch is not yet at the level the rest
    of it is. The tail is trimmed a little as well, because the level
    often starts to come off before the environment formally stops. A
    stretch too short to trim is returned whole. `keep` must be two
    fractions in [0, 1], the first below the second.
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
    """(levels in dB, hops per frame, hop in samples, scatter in dB).

    This is the one pass over the samples, and it returns everything a
    caller needs to reason about the record afterward. The frame is
    `frame_length` samples, or about a second (`frame_length_for`) when
    None, and never longer than the record; the hop is the frame length
    times (1 - `overlap`). The scatter is the larger of `scatter_floor`
    and `observed_scatter`. It is separate from `analysis_window`
    because more than one question is asked of the same compression,
    and asking twice would mean reading the samples twice.
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
    """(first sample, last sample) worth averaging in this history,
    last exclusive.

    The stretch is the highest level held for `want_frames` frames of
    `frame_length` samples at `overlap` (see `plateau`), trimmed to its
    settled middle by `keep` (see `trim`). A noise floor capture is a
    level like any other and is found the same way, so a PSD can be
    computed from it. The whole record comes back when the record has
    no level to compare (shorter than a hop, or every channel flat).
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
    """The averaging to use for a record that has no parameters set.

    By default a Hann `window` at half `overlap` and a frame about a
    second long (`frame_length_for`), starting where `analysis_window`
    puts the stretch, with as many frames as fit before the stretch
    ends, and at least one. The frame count is the answer to how many
    averages, not a number chosen in advance. The start is marked as
    chosen (`start_set`).
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
                     frames=max(room, 1), start=first / rate, start_set=True)
