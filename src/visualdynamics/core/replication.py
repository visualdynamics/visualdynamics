"""How closely a transient test reproduced the waveform it was
controlled to.

A random test is judged against a specification that is already a
statistic: the PSD is an average, and it is compared with another
average. A transient specification is one waveform, which the
controller played over and over, so the record is first cut back into
its repeats and each repeat is compared with the same target.

The frames follow from the specification. Each repeat plays exactly the
control signal, so a frame is the specification's length, the repeats
are back to back, and the window is rectangular, because any taper
would change the waveform being compared. The one number that is
measured rather than derived is how many repeats the record holds, the
record's length divided by the frame's.

None of this can be read from the file. Rattlesnake's transient
environment saves the control signal, the channel indices, the ramp
time and the control-law wiring, and nothing about repeats: `repeat` is
a runtime instruction that never reaches the netCDF file. The file does
carry a set of `sysid_*` attributes giving a Hann window, half overlap
and a 2048-sample frame. Those describe the random excitation run
beforehand to measure the FRF the controller inverts, and as averaging
parameters for the transient all three are wrong.

Three metrics are reported, and they answer different questions:

- `waveform` is ||m - s|| / ||s||, in percent. Amplitude, phase and
  shape all change it, it is what the controller minimizes, and it
  cannot look good by accident.
- `srs` is the largest deviation of the measured SRS from the target's,
  in dB, signed. It asks whether the shock is equivalent: two
  transients with the same SRS damage the same hardware however little
  they resemble each other.
- `level` is the scale alone, in dB, with shape and phase left out. It
  is the ratio of the mean squares, which by Parseval is the ratio of
  the two PSDs' totals, so it reads as a PSD level. Paired with
  `waveform` it tells "too small" from "wrong", which one number
  cannot do.

TRAC and a complex-spectrum FRAC are left out on purpose. They are the
same number: both are normalized inner products and the DFT is unitary,
so by Parseval a complex FRAC equals TRAC to within the half-bin the
real FFT keeps at DC and Nyquist. Reporting both would present one
measurement as two.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .averaging import Averaging

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import DataArray, TimeHistory

#: How far the alignment search looks either side, as a fraction of a
#: frame. The delay being measured is a handful of samples of hardware
#: latency; a wide search would let a badly replicated channel slide
#: until it found something to sit on.
LAG_FRACTION = 0.02

#: The waveform error a bar is shaded past, in percent. Unlike the dB
#: thresholds either side of a level — which are the convention a
#: specification is written to — this one is a display default and
#: nothing more. There is no accepted tolerance on how closely a
#: replicated transient must follow its target, so this marks a line
#: for the eye rather than a requirement, and it is meant to be moved.
WAVEFORM_PERCENT = 20.0

#: How far below the loudest control channel a target may sit and still
#: be a target. A MIMO transient names every control channel, including
#: ones the input barely reaches — a lateral DOF under an axial pulse
#: asks for nothing, and its target is the solver's residual rather
#: than a requirement. Every metric here is a ratio against the target,
#: and against the residual they do not become large so much as stop
#: meaning anything: the airplane's lateral channel is 230 dB down and
#: reported a waveform error of 1.3e12 percent before this existed.
ZERO_TARGET_DB = 120.0

#: How far below its own peak the target's SRS may fall and still be
#: compared. The bottom of an SRS band answers what a 0.25 Hz
#: oscillator did during a 4-second record, which for a shock is the
#: residual drift and not the shock; the target there is 80 dB below
#: its own peak, and a ratio taken against it reports 20 dB of nothing.
#: This is the same principle a written specification gets for free —
#: outside its band it says nothing — derived for a target that is a
#: waveform and carries no band of its own.
SRS_FLOOR_DB = 40.0


def averaging_for(measured: TimeHistory,
                  specification: DataArray) -> Averaging | None:
    """The averaging a transient run implies, or None if there is none.

    Everything but the count follows from the specification: the frame
    is its length, the repeats are back to back with no overlap, and
    the window is rectangular. The count is how many whole repeats the
    record holds. A record that stops partway through a repeat (a
    profile whose stop lands near a frame boundary does this) leaves
    that fraction out, since a fraction of a frame is not an average.

    When the record already carries an averaging with the
    specification's frame length, such as one set in the averaging
    panel, that averaging is used instead, so narrowing the analysis
    there narrows the comparison too. None when the specification has
    fewer than two samples or the record is shorter than one repeat.
    """
    frame = int(specification.ordinate.shape[1])
    samples = int(measured.ordinate.shape[1])
    if frame < 2 or samples < frame:
        return None
    # the record's own averaging wins when it is about these playings.
    # It is what the averaging panel edits, so narrowing the analysis to
    # the first three events there has to narrow the comparison too —
    # otherwise the shading on the time history says three and every
    # number beside it is still worked out from six.
    #
    # Recognized by the frame length. Anything else is describing some
    # other cutting-up of the record: a frame that is not the
    # specification's length does not line up with a playing of it, and
    # honoring it would compare each event against a slice of the
    # target chosen by accident.
    held = getattr(measured, 'averaging', None)
    if held is not None and held.frame_length == frame:
        return held
    return Averaging(frame_length=frame, overlap=0.0, window='rectangle',
                     frames=samples // frame, start=0.0)


def playings(measured: TimeHistory, specification: DataArray,
             averaging: Averaging | None = None,
             lag: int | None = None) -> list[dict[str, int]]:
    """Where each whole playing of the waveform sits in the record.

    Returns one dict per playing: `frame` (its index), `start` (its
    first sample, with the alignment `lag` applied) and `samples` (its
    length). The plot, the grid, the numbers and the shading on the
    time history all read this list, so they agree on how many playings
    there are and where they start. `averaging` defaults to
    `averaging_for`, and `lag` to `lag_of`.

    Only whole playings are listed. A record rarely stops on a
    boundary, since Rattlesnake repeats until it is told to stop and
    streaming often ends before the environment does, so there is
    usually a fragment at the end. The fragment is not analyzed: a
    waveform error over part of a frame is taken against a different
    stretch of the target, and does not compare with the ones beside
    it. `leftover` reports what was recorded past the last whole
    playing.
    """
    # One list for every view because they disagreed once: an earlier
    # version counted a part-played repeat as a seventh event while the
    # shading, which comes from `Averaging.frame_bounds`, knew only
    # whole frames. Two views of one record gave two answers.
    averaging = averaging or averaging_for(measured, specification)
    if averaging is None:
        return []
    if lag is None:
        lag = lag_of(measured, specification, averaging)
    n = averaging.frame_length
    total = measured.ordinate.shape[1]
    first = averaging.start_sample(float(measured.sample_rate))
    out = []
    for k in range(averaging.frames):
        # hop, not the frame length: an averaging that arrived from the
        # panel may carry an overlap, and where the frames sit is the
        # averaging's business rather than something to assume here
        start = first + k * averaging.hop + lag
        if start < 0 or start + n > total:
            continue
        out.append({'frame': k, 'start': start, 'samples': n})
    return out


def leftover(measured: TimeHistory, specification: DataArray,
             averaging: Averaging | None = None,
             lag: int | None = None) -> tuple[int, float] | None:
    """What the record holds past the last whole playing, as
    `(samples, fraction of a playing)`, or None when it ends on a
    boundary.

    Nothing is scored from it (see `playings`), but a run that recorded
    two and a half seconds of a further event should say so. The
    fragment is measured against how many whole playings the record
    could hold, not how many are being analyzed. `averaging` and `lag`
    default as in `playings`.
    """
    averaging = averaging or averaging_for(measured, specification)
    if averaging is None:
        return None
    if lag is None:
        lag = lag_of(measured, specification, averaging)
    n = averaging.frame_length
    total = measured.ordinate.shape[1]
    # measured against how many whole playings the record *could* hold,
    # not against how many are being analyzed. Choosing to look at three
    # of six is a decision and needs no announcing; a recording that
    # stopped part way through one is a fact about the file.
    left = (total - lag) - ((total - lag) // n) * n
    if left <= 0:
        return None
    return left, left / float(n)


def _rows_by_dof(data: DataArray, dofs: Sequence[str],
                 dim: str) -> list[int | None]:
    """First record at each DOF carrying `dim`, or None where there is
    none. First, not best: a channel is picked by what it is, never by
    which one happens to compare well."""
    out = []
    for dof in dofs:
        match = [i for i in range(data.num_records)
                 if data.channel_key(i)[0] == dof
                 and data.channel_key(i)[1] == dim]
        out.append(match[0] if match else None)
    return out


def lag_of(measured: TimeHistory, specification: DataArray,
           averaging: Averaging | None = None) -> int:
    """One integer sample shift aligning the record to the target,
    used for every channel and every repeat.

    The delay belongs to the acquisition, not to a channel, and letting
    each channel choose its own shift would let a poorly replicated one
    find its most favorable alignment and report the error there. The
    shift is the one that minimizes the total squared difference over
    the first frame of the analysis, across every control channel at
    once. It is searched up to `LAG_FRACTION` of a frame (2 %) either
    side of where the analysis starts, and never before the record's
    first sample, so an analysis that starts at 0 is searched forward
    only. 0 when there is nothing to align.
    """
    # It used to search 0..reach from sample 0 whatever the averaging
    # said, so an analysis held at a later start was aligned against a
    # stretch of the record it does not score, and could not move back.
    averaging = averaging or averaging_for(measured, specification)
    if averaging is None:
        return 0
    n = averaging.frame_length
    dofs = list(specification.response_dof)
    dim = specification.ordinate_dim[0]
    rows = _rows_by_dof(measured, dofs, dim)
    pairs = [(specification.ordinate[c], measured.ordinate[r])
             for c, r in enumerate(rows) if r is not None]
    pairs = [(s, m) for s, m in pairs if np.abs(s).max() > 0.0]
    if not pairs:
        return 0
    reach = max(1, round(n * LAG_FRACTION))
    total = measured.ordinate.shape[1]
    first = averaging.start_sample(float(measured.sample_rate))
    best, best_cost = 0, np.inf
    for lag in range(-reach, reach + 1):
        at = first + lag
        if at < 0 or at + n > total:
            continue
        cost = sum(float(np.sum((m[at:at + n] - s) ** 2))
                   for s, m in pairs)
        if cost < best_cost:
            best, best_cost = lag, cost
    return best


def _level_db(measured_frame, spec_frame):
    """The level the frame came out at against the target, in dB.

    The ratio of mean squares, which by Parseval is the ratio of the two
    PSDs' totals — so it is read as a PSD level — without a PSD to get
    wrong. It used to go through one, a second PSD implementation beside
    `TimeHistory.compute_psds` that doubled the DC and Nyquist lines with
    the rest, and so counted a steady offset twice (2026-09-23).
    """
    got = float(np.mean(np.square(measured_frame)))
    want = float(np.mean(np.square(spec_frame)))
    if not (np.isfinite(got) and np.isfinite(want)) or want <= 0.0:
        return float('nan')
    return 10.0 * np.log10(got / want)


def _srs_db(measured_frame, want, sample_rate, band, q):
    """The worst deviation of the frame's SRS from the target's, in dB.

    Worst rather than mean, and signed. A channel that runs 5 dB high
    across the low bands and 5 dB low across the high ones averages to
    nothing, and nothing is not what happened to it. Signed, because a
    shock that overtests and one that undertests are different
    failures and a bar either side of zero says which.

    The target's spectrum comes in already worked out. It is the same
    waveform at every playing, so computing it per repeat doubles the
    cost of the single most expensive thing here for no answer that was
    not already known.
    """
    from .srs import maximax

    got = maximax(measured_frame, band, sample_rate, q=q)
    good = np.isfinite(got) & np.isfinite(want) & (want > 0.0) & (got > 0.0)
    if not good.any():
        return float('nan')
    # only where the target actually asks for something
    good &= want >= want[good].max() * 10.0 ** (-SRS_FLOOR_DB / 20.0)
    if not good.any():
        return float('nan')
    deviation = 20.0 * np.log10(got[good] / want[good])
    return float(deviation[np.argmax(np.abs(deviation))])


def _band_for(frame_length, sample_rate, per_octave=None):
    """The SRS band a frame of this length can support.

    From a frequency low enough that the frame holds a cycle of it, up
    to a fifth of the sample rate — above which the ramp-invariant
    filter is being asked about frequencies the record cannot resolve.
    """
    from .srs import PER_OCTAVE, octave_frequencies

    low = sample_rate / float(frame_length)
    high = sample_rate / 5.0
    if not high > low:
        return None
    return octave_frequencies(low, high, per_octave or PER_OCTAVE)


#: every reading `compare` can take, and what it costs. The SRS is
#: three orders of magnitude dearer than the other two — a
#: ramp-invariant filter run at every band, against a record rather
#: than a spectrum — so a caller that does not want it must be able to
#: say so. Measured on the airplane: 70 ms a cell against 0.2 ms for a
#: level and under 0.05 ms for a waveform error.
METRICS = ('waveform', 'srs', 'level')


def compare(measured: TimeHistory, specification: DataArray,
            averaging: Averaging | None = None, lag: int | None = None,
            q: float | None = None, frame: int | None = None,
            metrics: Sequence[str] = METRICS) -> list[dict[str, Any]]:
    """One row per control channel per repeat: how well each was
    replicated, and which playing of the waveform it was.

    Each row is a dict with `label` (the channel's DOF), `frame` (the
    playing's index), `zero_target`, and one entry per metric asked
    for: `waveform` in percent, `srs` and `level` in dB. Every repeat is
    reported and none is singled out, because which event counts as the
    bad one depends on which reading matters and on what the article is
    for. `frame` narrows the answer to one repeat.

    `metrics` picks which readings are taken (any of `METRICS`), and a
    row carries only those; an unknown name raises ValueError. The SRS
    costs a ramp-invariant filter per band per cell, about 70 ms against
    under 1 ms for the other two, so a caller that does not need it
    should leave it out. `q` is the SRS quality factor (`srs.DEFAULT_Q`
    when None), and `averaging` and `lag` default as in `playings`.

    Only whole playings are scored (see `playings`); a fragment at the
    end of the record is reported by `leftover`. A channel whose target
    is more than `ZERO_TARGET_DB` (120 dB) below the loudest control
    channel's gets NaN for every metric, with `zero_target` True,
    because every metric is a ratio against the target. A control
    channel with no matching record is left out.
    """
    # Every repeat is reported: an earlier version reduced these to the
    # worst repeat per channel and labeled it so, which is a judgment
    # and not a measurement. `metrics` exists because a screen showing a
    # grid of waveform errors was spending two seconds an update on
    # sixty shock spectra nobody had asked to see.
    from .srs import DEFAULT_Q, maximax

    metrics = tuple(metrics)
    unknown = [name for name in metrics if name not in METRICS]
    if unknown:
        raise ValueError(f'not a reading of a replication: {unknown}; '
                         'visualdynamics has ' + ', '.join(METRICS))
    averaging = averaging or averaging_for(measured, specification)
    if averaging is None:
        return []
    if lag is None:
        lag = lag_of(measured, specification, averaging)
    q = DEFAULT_Q if q is None else float(q)

    rate = float(measured.sample_rate)
    n = averaging.frame_length
    band = _band_for(n, rate) if 'srs' in metrics else None
    dofs = list(specification.response_dof)
    dim = specification.ordinate_dim[0]
    rows = _rows_by_dof(measured, dofs, dim)

    scales = [float(np.linalg.norm(specification.ordinate[c]))
              for c in range(len(dofs))]
    loudest = max(scales) if scales else 0.0
    floor = loudest * 10.0 ** (-ZERO_TARGET_DB / 20.0)

    found_playings = playings(measured, specification, averaging, lag)
    if frame is not None:
        found_playings = [p for p in found_playings
                          if p['frame'] == int(frame)]
    out = []
    for c, dof in enumerate(dofs):
        row = rows[c]
        if row is None:
            continue
        whole = specification.ordinate[c]
        silent = scales[c] <= floor
        for playing in found_playings:
            k, a, m = playing['frame'], playing['start'], playing['samples']
            found = {'label': dof, 'frame': k, 'zero_target': bool(silent)}
            if silent:
                found.update(dict.fromkeys(metrics, float('nan')))
                out.append(found)
                continue
            target, scale = whole, scales[c]
            block = measured.ordinate[row, a:a + m]
            if 'waveform' in metrics:
                found['waveform'] = (
                    100.0 * float(np.linalg.norm(block - target)) / scale
                    if scale > 0.0 else float('nan'))
            if 'srs' in metrics:
                aimed = maximax(target, band, rate, q=q) \
                    if band is not None else None
                found['srs'] = (_srs_db(block, aimed, rate, band, q)
                                if band is not None else float('nan'))
            if 'level' in metrics:
                found['level'] = _level_db(block, target)
            out.append(found)
    return out



def event_slice(measured: TimeHistory, specification: DataArray,
                frame: int, averaging: Averaging | None = None,
                lag: int | None = None) -> TimeHistory | None:
    """One repeat of the control channels, on the target's own clock,
    as a TimeHistory.

    The repeat is moved onto the specification's abscissa, which starts
    at zero, instead of keeping its place in the record, and the
    alignment (`lag`) is applied, so the two are drawn where they are
    compared. Only the control channels are kept, in the
    specification's order, so the two line up record for record. None
    when that repeat is not in the record, or none of the control
    channels is.
    """
    averaging = averaging or averaging_for(measured, specification)
    if averaging is None:
        return None
    if lag is None:
        lag = lag_of(measured, specification, averaging)
    wanted = [p for p in playings(measured, specification, averaging, lag)
              if p['frame'] == int(frame)]
    if not wanted:
        return None
    a, n = wanted[0]['start'], wanted[0]['samples']
    dofs = list(specification.response_dof)
    dim = specification.ordinate_dim[0]
    rows = _rows_by_dof(measured, dofs, dim)
    keep = [(dof, row) for dof, row in zip(dofs, rows) if row is not None]
    if not keep:
        return None
    from .data import TimeHistory

    out = TimeHistory(
        np.asarray(specification.abscissa[:n], dtype=float),
        np.asarray([measured.ordinate[row, a:a + n] for _dof, row in keep]),
        response_dof=[dof for dof, _row in keep],
        ordinate_dim=[measured.ordinate_dim[row] for _dof, row in keep],
        ordinate_unit=[measured.ordinate_unit[row] for _dof, row in keep],
        comment=f'event {int(frame) + 1}')
    return out
