"""Signal-to-noise ratio of a driven measurement over its noise floor.

A system identification records each channel twice: once with the
drives off (the ambient recording, which holds the noise of the room,
the amplifiers, the cabling and the sensors) and once with them
running. The driven recording holds the signal and that same noise, so
its power is the sum of the two, and the signal power is the driven
power minus the ambient power:

    SNR = (P_driven - P_ambient) / P_ambient,    in dB 10 log10(SNR)

This is the standard definition, signal power over noise power, as
IEEE Std 1241 states it for digitizers and as Bendat and Piersol
(*Random Data*, 4th ed., Wiley, 2010) use it for random data. The
signal power is estimated by subtraction, which is exact when the
noise is uncorrelated with the signal; a separate ambient recording
assumes that already. The plain quotient P_driven / P_ambient is
SNR + 1. The two agree when the margin is large and differ near the
floor, where a quotient of 3 dB is a signal-to-noise ratio of 0 dB.
The quotient overstates the margin of the channels closest to their
floor, which are the channels this check is meant to find, so it is
not used here.

The ratio also sets the coherence: with the noise on the response,
ordinary coherence is SNR / (1 + SNR) (Bendat and Piersol), so
`THRESHOLD_DB`, the level the per-channel bars are compared against,
is the margin at which that coherence is about 0.91.

Two functions compute it by the same rule:

- `signal_to_noise`: line by line, for every shared channel. The
  curve, the stage and the report's figure all use it.
- `rms_signal_to_noise`: one value per channel, from the mean square
  of each recording over the band the two densities share, taken as
  the area under the density the way the density is drawn
  (`Psd.area`). The band's largest peaks dominate it, so it is for
  ranking channels, and the curve is the one to diagnose with.

Where the driven power does not exceed the ambient (estimation scatter
at the floor, or a channel the drives did not excite), there is no
signal to report, and the result is NaN rather than a number or minus
infinity: the channel is at its noise floor.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from .data import paired_channels

# The plain quotient P_driven / P_ambient is not drawn: the per-line
# curve moved to the subtracted definition together with the
# per-channel bars (Brandon, 2026-10-03).

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
    """The rows, with each NaN channel's label marked as at the noise
    floor, where its bar would be.

    The app, the headless plot and the report all label these channels
    through this function, so they label them the same way."""
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
    """The signal-to-noise ratio of a driven density over its ambient
    density, channel by channel and line by line, as a power ratio
    (take 10 log10 for decibels).

    Channels are paired by `paired_channels`: same DOF, same quantity
    and the same frequency lines, or it raises. A line whose ambient is
    zero gives NaN, not infinity. On hardware a zero is noise below the
    recording's resolution, and in a simulation it is an ambient run
    with no noise in it; neither is an infinite signal-to-noise ratio.
    A line where the driven density does not exceed the ambient also
    gives NaN, because it is at the noise floor.

    Parameters
    ----------
    signal : Psd
        The driven densities.
    floor : Psd
        The ambient densities, on the same lines.

    Returns
    -------
    tuple
        ``(abscissa, rows, dofs, dims)``: the frequency lines, the
        ratio as one row per shared channel, and each row's DOF and
        quantity, so the rows of an object holding several quantities
        can be told apart.
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
    """The RMS signal-to-noise ratio in dB of each channel the two
    densities share, and the labels of the channels with no ambient
    noise.

    P_driven and P_ambient are each channel's mean square over the
    whole band of the densities (the area under the density), and the
    value is 10 log10((P_driven - P_ambient) / P_ambient), which is
    20 log10 of the signal RMS over the noise RMS. It is NaN where the
    driven power does not exceed the ambient, because that channel is
    at its noise floor.

    A channel whose ambient recording is exactly zero is the opposite
    case and is kept separate. On hardware a zero is noise below the
    recording's resolution, and in a simulation it is an ambient run
    with no noise in it; either way there is no noise to divide by.
    Treating it as at the floor would mark the cleanest channel as the
    worst, so it is left out of the rows and listed in the second
    list instead.

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
        One row per channel, labeled as the driven object labels it,
        with the quantity added where one DOF has two channels (a drive
        point's accelerometer and load cell), so the bars can be told
        apart.
    silent : list of str
        The labels of the channels whose ambient recording is zero.
    """
    # A zero ambient read as at the floor marked the plate's
    # drive-point load cells, the cleanest channels, as the worst
    # (2026-10-03); they are listed as silent instead.
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
