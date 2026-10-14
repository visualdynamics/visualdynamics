"""Read Rattlesnake sine specification files (.npz), one tone per file.

The controller's sine environment saves each specification the way its
own `save_specification` writes it: `frequency` (the breakpoints),
`amplitude` (channels by breakpoints, the specification table's column
transposed), and `sweep_type` and `sweep_rate` for each segment; and,
from the controller versions that have them, `phase` in degrees,
`warning` and `abort` as (lower/upper, left/right, channels,
breakpoints), `start_time` and `name`. Files from older versions lack
some of the optional arrays, and read as well.

The file names no channels and has no units. The channels are named in
the controller's own order, '1', '2' and so on, which is also how a run
through a response transformation records them, unless `response_dof`
names them; the amplitudes are read as written unless `ordinate_unit`
gives their unit. Nine files are nine specifications of one tone each;
`Project.merge` joins them into one.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.sine import SineSweepSpecification, SineTone
from .sniffing import npz_has

# Brandon, 2026-10-02: nine older files, one per sweep, were refused as "an
# npz archive holding 5 arrays", which is why the later arrays are optional.

_RANDOM_KEYS = {'f', 'cpsd'}


def sniff(path: str | os.PathLike) -> bool:
    """True for an `.npz` with the four arrays every sine specification
    has, and without the random specification's `f` and `cpsd`, since
    the two formats come from the same controller and are otherwise
    unrelated."""
    return (npz_has(path, {'frequency', 'amplitude', 'sweep_type', 'sweep_rate'},
                    allow_pickle=False)
            and not npz_has(path, _RANDOM_KEYS, allow_pickle=False))


def load(path: str | os.PathLike,
         response_dof: Sequence[str] | None = None,
         ordinate_dim: str = 'acceleration',
         ordinate_unit: str | None = None) -> SineSweepSpecification:
    """Read one tone as the controller wrote it.

    The phase is converted from degrees to radians. The warning and
    abort bands must be the same on the left and right sides of each
    breakpoint, since a tone here has one value per breakpoint; a file
    where they differ is refused. The tone is named by the file's `name`,
    or by the file name without it.

    Parameters
    ----------
    path : str or os.PathLike
        The `.npz` file.
    response_dof : sequence of str, optional
        The control channels the amplitude columns belong to, in the
        file's order. Defaults to '1', '2', ..., the controller's
        channel order, which is how a run through a response
        transformation names them.
    ordinate_dim : str, default 'acceleration'
        What the amplitudes measure.
    ordinate_unit : str, optional
        The amplitudes' unit. None keeps the file's values as they are,
        to be read beside data in the same unit.

    Returns
    -------
    SineSweepSpecification
        A specification of one tone.

    Raises
    ------
    ValueError
        For a file with no breakpoints, a `response_dof` whose length is
        not the number of amplitude columns, a band of the wrong shape,
        or a band that differs between the sides of a breakpoint.
    """
    with np.load(path, allow_pickle=False) as d:
        frequency = np.asarray(d['frequency'], dtype=float).ravel()
        amplitude = np.asarray(d['amplitude'], dtype=float)
        sweep_type = np.asarray(d['sweep_type']).ravel()
        sweep_rate = np.asarray(d['sweep_rate'], dtype=float).ravel()
        phase = np.asarray(d['phase'], dtype=float) if 'phase' in d else None
        warning = np.asarray(d['warning'], dtype=float) if 'warning' in d else None
        abort = np.asarray(d['abort'], dtype=float) if 'abort' in d else None
        start_time = float(np.asarray(d['start_time']).ravel()[0]) \
            if 'start_time' in d else 0.0
        name = str(np.asarray(d['name']).ravel()[0]) if 'name' in d else ''
    n = len(frequency)
    if n < 1:
        raise ValueError(f'{os.path.basename(str(path))}: no breakpoints')
    # the table's amplitude column is (breakpoints, channels) and the
    # writer transposes it; a file that did not is read by its shape
    amplitude = _by_breakpoint(amplitude, n, 'amplitude')
    phase = None if phase is None else np.deg2rad(_by_breakpoint(phase, n, 'phase'))
    m = amplitude.shape[1]
    if response_dof is None:
        response_dof = [str(k + 1) for k in range(m)]
    response_dof = [str(dof) for dof in response_dof]
    if len(response_dof) != m:
        raise ValueError(
            f'{os.path.basename(str(path))}: {m} amplitude columns but '
            f'{len(response_dof)} channel names')
    bands: dict[str, Any] = {}
    for kind, matrix in (('warning', warning), ('abort', abort)):
        if matrix is None or not np.isfinite(matrix).any():
            continue
        # (lower/upper, left/right, channels, breakpoints), the table's
        # (breakpoints, lower/upper, left/right, channels) transposed
        if matrix.shape != (2, 2, m, n):
            raise ValueError(
                f'{os.path.basename(str(path))}: the {kind} band has shape '
                f'{matrix.shape}, expected (2, 2, {m}, {n})')
        left, right = matrix[:, 0, :, :], matrix[:, 1, :, :]
        if not np.allclose(left, right, equal_nan=True):
            raise ValueError(
                f'{os.path.basename(str(path))}: the {kind} band differs '
                'between the left and right sides of a breakpoint')
        for side, curve in enumerate(('lower', 'upper')):
            values = left[side].T                      # (breakpoints, channels)
            if np.isfinite(values).any():
                bands[f'{kind}_{curve}'] = values
    tone = SineTone(name or os.path.splitext(os.path.basename(str(path)))[0],
                    start_time, frequency, amplitude,
                    sweep_type[:n - 1], sweep_rate[:n - 1], phase=phase, **bands)
    return SineSweepSpecification([tone], response_dof,
                                  ordinate_dim=ordinate_dim,
                                  ordinate_unit=ordinate_unit,
                                  comment=f'from {os.path.basename(str(path))}')


def _by_breakpoint(matrix: np.ndarray, n: int, what: str) -> np.ndarray:
    matrix = np.atleast_2d(matrix)
    if matrix.shape[0] == n and matrix.shape[1] != n:
        return matrix
    if matrix.shape[1] == n:
        return matrix.T
    if matrix.shape[0] == n:
        return matrix
    raise ValueError(f'{what} has shape {matrix.shape}; {n} breakpoints expected '
                     'along one axis')
