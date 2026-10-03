"""Signal to noise: how far a driven measurement stands above its floor.

A system identification records each channel twice — once with the
drives still (the ambient: the room, the amplifiers, the cabling and the
sensors) and once with them running. The driven recording holds the
signal *and* that same noise, so its power is the sum of the two, and
the signal's own power is what is left when the noise is taken away:

    SNR = (P_driven - P_ambient) / P_ambient,    in dB 10 log10(SNR)

That is the textbook definition — signal power over noise power, as
IEEE Std 1241 states it for digitizers and as Bendat and Piersol
(*Random Data*, 4th ed., Wiley, 2010) use it for random data — with the
signal's power estimated by subtraction, which is exact when the noise
is uncorrelated with the signal, the one assumption a separate ambient
recording makes anyway. The plain quotient P_driven / P_ambient is
SNR + 1: the two agree when the margin is wide and part near the floor,
where a quotient of 3 dB is a signal-to-noise of 0 dB. It flatters
exactly the channels the reading exists to find, so it is not what is
drawn here (Brandon, 2026-10-03: the per-line curve moved to this
definition with the per-channel bars).

It is also the quantity coherence speaks for: with the noise on the
response, ordinary coherence is SNR / (1 + SNR) (Bendat and Piersol),
so the `THRESHOLD_DB` the bars are judged against is the margin at
which that coherence reaches 0.9.

Two readings, one rule:

- `signal_to_noise` — line by line, every shared channel: the curve,
  the stage and the report's figure all divide here.
- `rms_signal_to_noise` — one number per channel, from the RMS of each
  recording over the band the two densities share: P is the area under
  the density, read the way the density is drawn (`Psd.area`). A
  summary, dominated by wherever the energy is, so it ranks channels
  and the curve stays the diagnostic.

Where the driven power does not exceed the ambient — estimation scatter
right at the floor, or a channel the drives never reached — there is no
signal to state, and the answer is NaN rather than a number or minus
infinity: the channel is at its noise floor, and the readings say so.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from .data import paired_channels

__all__ = ['AT_FLOOR', 'THRESHOLD_DB', 'at_floor', 'rms_signal_to_noise',
           'signal_to_noise']

#: the margin a channel's RMS signal-to-noise is judged against. 10 dB
#: is where the coherence the same noise allows, SNR / (1 + SNR),
#: reaches 0.91 — the level a coherence is usually asked to clear — so
#: the bars and the coherence map judge a channel by one standard. A
#: starting point, as every threshold here is: it is dragged in the app.
THRESHOLD_DB = 10.0

#: what a channel with no signal above its noise is called beside its
#: name, where its bar would be: a bar of nothing reads as a channel
#: nobody measured, and this one was measured and found wanting
AT_FLOOR = 'at noise floor'


def at_floor(rows: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    """The rows with every NaN channel's label saying why it has no
    bar — one place, so the app, the headless plot and the report name
    it alike."""
    return [(f'{label} ({AT_FLOOR})' if not np.isfinite(value) else label,
             value) for label, value in rows]


def _subtracted(driven: np.ndarray, ambient: np.ndarray) -> np.ndarray:
    """(driven - ambient) / ambient where the driven stands above a
    nonzero ambient, NaN everywhere else."""
    driven = np.real(np.asarray(driven, dtype=complex))
    ambient = np.real(np.asarray(ambient, dtype=complex))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((ambient > 0.0) & (driven > ambient),
                        (driven - ambient) / ambient, np.nan)


def signal_to_noise(signal: Any, floor: Any):
    """The signal-to-noise of a driven density over its ambient one,
    channel by channel, line by line, as a power ratio (take
    10 log10 for decibels).

    Channels pair by `paired_channels`: same DOF, same quantity, on
    the same frequency lines, or it refuses. A line whose ambient is
    zero answers NaN, never infinity — on hardware zero means below
    resolution, in a simulation it means the quiet was exactly silent,
    and neither is an infinite signal-to-noise. A line where the
    driven density does not exceed the ambient answers NaN too: it is
    at the noise floor.

    Parameters
    ----------
    signal : Psd
        The driven densities.
    floor : Psd
        The ambient densities, on the same lines.

    Returns
    -------
    tuple
        ``(abscissa, rows, dofs, dims)`` — the frequency lines, the
        ratio as one row per shared channel, each row's DOF and its
        quantity, so a mixed object's rows can be told apart.
    """
    rows, dofs, dims = [], [], []
    for i, j, dof, dim in paired_channels(signal, floor):
        rows.append(_subtracted(signal.ordinate[i], floor.ordinate[j]))
        dofs.append(dof)
        dims.append(dim)
    return np.asarray(signal.abscissa), np.asarray(rows), dofs, dims


def rms_signal_to_noise(signal: Any, floor: Any,
                        records: Sequence[int] | None = None
                        ) -> tuple[list[tuple[str, float]], list[str]]:
    """([(channel label, RMS signal-to-noise in dB)], [silent labels])
    for the channels the two densities share.

    P_driven and P_ambient are each channel's mean square over the
    whole band the densities stand on — the area under the density —
    and the reading is 10 log10((P_driven - P_ambient) / P_ambient),
    which is 20 log10 of the signal's RMS over the noise's. NaN where
    the driven power does not exceed the ambient: that channel is at
    its noise floor.

    A channel whose ambient recorded nothing at all is the opposite
    case and is kept apart: on hardware zero is noise below the
    recording's resolution, in a simulation a quiet that was exactly
    silent, and either way there is no noise to divide by. Reading it
    as at the floor would flag the cleanest channel as the worst (the
    plate's drive-point load cells did exactly that, 2026-10-03), so
    it is left out of the rows and named in the second list.

    Parameters
    ----------
    signal : Psd
        The driven densities.
    floor : Psd
        The ambient densities, on the same lines.
    records : sequence of int, optional
        The driven rows to read, as the selection names them; every
        shared channel when None.

    Returns
    -------
    rows : list of (str, float)
        One row per channel, labeled as the driven object labels it —
        with the quantity added where one DOF carries two (a drive
        point's accelerometer and load cell), so the bars can tell
        them apart.
    silent : list of str
        The labels of the channels whose ambient recorded nothing.
    """
    pairs = paired_channels(signal, floor)
    if records is not None:
        wanted = {int(k) for k in records}
        pairs = [pair for pair in pairs if pair[0] in wanted]
    repeated = Counter(dof for _i, _j, dof, _dim in pairs)
    rows, silent = [], []
    for i, j, dof, dim in pairs:
        driven, ambient = signal.area(i), floor.area(j)
        label = signal.record_label(i)
        if repeated[dof] > 1:
            label = f'{label} ({_quantity_word(dim)})'
        if not ambient > 0.0:
            silent.append(label)
            continue
        ratio = float(_subtracted(np.array([driven]),
                                  np.array([ambient]))[0])
        rows.append((label, float(10.0 * np.log10(ratio))
                     if np.isfinite(ratio) else float('nan')))
    return rows, silent


def _quantity_word(dim: str) -> str:
    """'acceleration' for acceleration**2/frequency: the word the
    interface names a quantity by, not the dimension's spelling."""
    from .data import channel_quantities
    from .unit_choices import shown_dimension

    response, _reference = channel_quantities(dim)
    return shown_dimension(response)
