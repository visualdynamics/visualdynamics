"""Rattlesnake's random-vibration specification file, both ways.

The controller's own target for a random test is not its `.nc4` (that
is the recording of a run, and stays read-only here) but a small file
its Random environment loads before the test: `f`, the frequency lines
in Hz; `cpsd`, the full control-channel cross-spectral matrix at each
line, `(lines, n, n)` complex; optionally `warning_upper`,
`warning_lower`, `abort_upper`, `abort_lower`, `(lines, n)` real bands
on the autospectra; and optionally `coordinate`, `(n, n, 2)` node and
direction pairs the controller uses to put the matrix in its own
channel order. Numpy `.npz` or MATLAB `.mat`; this module writes and
reads the `.npz`. (Brandon, 2026-09-05: a specification edited in
Visual Dynamics should come out as something Rattlesnake can read.)

The controller reads a whole matrix, and a specification here more
often than not holds autospectra alone — that is how most random
tests are run (Brandon, 2026-09-05). So a pair the specification
does not hold is written as the controller's own targets carry it,
a zero, and the file says which those were: `held`, an `(n, n)`
mask of the cross terms the specification actually held, which the
controller ignores and this reader honors, so a zero that was
never stated does not come back as a statement of independence. A
Hermitian half is completed, as the transform allows. The file has
nowhere to record units, so values go out in the unit system asked
for, the coherent one, and the status line names it; reading takes
the unit as a parameter, as sdynpy's format does.

The file also carries `dof`, the channel names as Visual Dynamics
writes them, which the controller ignores and this reader prefers:
`coordinate` is written only when every channel is a node and a
direction, since a modal coordinate (`M3`) has neither, and the
strings are what bring the specification back exactly.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.data import Specification, direction_code, dof_string, parse_dof
from .sniffing import npz_has

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem

#: the band arrays the controller reads, by the name Visual Dynamics
#: keeps them under
BANDS = ('warning_lower', 'warning_upper', 'abort_lower', 'abort_upper')

_COORDINATE = np.dtype([('node', '<u8'), ('direction', 'i1')])


def sniff(path: str | os.PathLike) -> bool:
    return npz_has(path, {'f', 'cpsd'}, allow_pickle=False)


def handles(obj: Any) -> bool:
    return isinstance(obj, Specification)


def _channels(spec: Specification) -> list[str]:
    """The channels, in the order of their autospectra."""
    references = spec.reference_dof or list(spec.response_dof)
    channels = [spec.response_dof[i] for i in range(spec.num_records)
                if references[i] == spec.response_dof[i]]
    if not channels:
        raise ValueError('the specification holds no autospectrum; the '
                         'controller reads a matrix with the autospectra '
                         'on its diagonal')
    return channels


def save(spec: Specification, path: str | os.PathLike,
         unit_system: UnitSystem | None = None) -> None:
    """Write a specification as the controller's target file.

    Values go out in `unit_system`'s coherent system, or as stored
    without one; the file records no units. A pair the specification
    does not hold is written as a zero, the controller's own
    placeholder, and `held` records that it was one.
    """
    from .exporters import data_values

    channels = _channels(spec)
    references = spec.reference_dof or list(spec.response_dof)
    index = {name: k for k, name in enumerate(channels)}
    held = {}
    for i in range(spec.num_records):
        a, b = spec.response_dof[i], references[i]
        if a not in index or b not in index:
            raise ValueError(f'{a}/{b} is a cross term of a channel with no '
                             'autospectrum in the specification')
        held[(a, b)] = i
    frequencies, ordinate = data_values(spec, unit_system)
    n, lines = len(channels), len(frequencies)
    cpsd = np.zeros((lines, n, n), dtype=np.complex128)
    stated = np.eye(n, dtype=bool)
    for (a, b), i in held.items():
        cpsd[:, index[a], index[b]] = ordinate[i]
        stated[index[a], index[b]] = True
    for a in channels:
        for b in channels:
            if (a, b) not in held and (b, a) in held:
                # the Hermitian half the file needs and the object did
                # not hold: the conjugate of the one it did
                cpsd[:, index[a], index[b]] = np.conj(cpsd[:, index[b], index[a]])
                stated[index[a], index[b]] = True
    out: dict[str, Any] = {'f': np.asarray(frequencies, dtype=np.float64),
                           'cpsd': cpsd,
                           'dof': np.asarray(channels, dtype=str),
                           'held': stated}
    for name in BANDS:
        if name not in spec.limits:
            continue
        values = (spec.display_limit(name, unit_system.coherent)
                  if unit_system is not None and spec.units_defined
                  else spec.limits[name])
        out[name] = np.stack([np.real(values[held[(c, c)]])
                              for c in channels], axis=1)
    parsed = [parse_dof(c) for c in channels]
    if all(node is not None and direction for node, direction in parsed):
        coordinate = np.zeros((n, n, 2), dtype=_COORDINATE)
        for i, (node_i, dir_i) in enumerate(parsed):
            for j, (node_j, dir_j) in enumerate(parsed):
                coordinate[i, j, 0] = (node_i, direction_code(dir_i))
                coordinate[i, j, 1] = (node_j, direction_code(dir_j))
        out['coordinate'] = coordinate
    np.savez(str(path), **out)


def load(path: str | os.PathLike,
         ordinate_unit: str | None = None) -> Specification:
    """Read the controller's target file as a Specification.

    Channels come from `dof` when the file has it, else from the
    `coordinate` diagonal; a file with neither names nothing, and is
    refused rather than given invented names. A file with `held` was
    written here, and that mask says which cross terms were stated:
    those come back, zeros included, and the rest were placeholders.
    Any other file is the controller's or sdynpy's, and its cross
    terms are kept only when one is a meaningful number, as the
    `.nc4` reader does — an off-diagonal all NaN or zero is their
    placeholder for autospectra alone. `ordinate_unit` declares what
    the values are in — the file cannot say.
    """
    with np.load(path, allow_pickle=False) as d:
        frequencies = np.asarray(d['f'], dtype=np.float64).reshape(-1)
        cpsd = np.asarray(d['cpsd'])
        if cpsd.ndim != 3 or cpsd.shape[1] != cpsd.shape[2] \
                or cpsd.shape[0] != len(frequencies):
            raise ValueError(f'{os.path.basename(str(path))}: cpsd is '
                             f'{cpsd.shape}, not (lines, n, n) over '
                             f'{len(frequencies)} lines')
        n = cpsd.shape[1]
        held = (np.asarray(d['held'], dtype=bool) if 'held' in d.files
                else None)
        if 'dof' in d.files:
            channels = [str(c) for c in d['dof']]
        elif 'coordinate' in d.files:
            diagonal = np.asarray(d['coordinate'])
            channels = [dof_string(int(diagonal[i, i, 0]['node']),
                                   int(diagonal[i, i, 0]['direction']))
                        for i in range(n)]
        else:
            raise ValueError(f'{os.path.basename(str(path))} names no '
                             'channels — neither a dof list nor a '
                             'coordinate array — and the controller matches '
                             'it to its channel table by order alone')
        if len(channels) != n:
            raise ValueError(f'{len(channels)} channel names for a '
                             f'{n}-channel matrix')
        bands = {name: np.asarray(d[name], dtype=np.float64)
                 for name in BANDS if name in d.files}
    off = ~np.eye(n, dtype=bool)
    if held is None:
        meaningful = bool(np.any(np.isfinite(cpsd[:, off])
                                 & (cpsd[:, off] != 0)))
        held = np.eye(n, dtype=bool) | (
            meaningful & np.isfinite(cpsd).any(axis=0))
    rows, resp, refs, limits = [], [], [], {name: [] for name in bands}
    blank = np.full(len(frequencies), np.nan)
    for i in range(n):
        for j in range(n):
            if not held[i, j]:
                continue
            rows.append(cpsd[:, i, j])
            resp.append(channels[i])
            refs.append(channels[j])
            for name, values in bands.items():
                limits[name].append(values[:, i] if i == j else blank)
    spec = Specification(
        frequencies, np.asarray(rows, dtype=np.complex128),
        response_dof=resp, reference_dof=refs,
        comment=f'Rattlesnake specification {os.path.basename(str(path))}',
        **{name: np.asarray(values) for name, values in limits.items()})
    if ordinate_unit is not None:
        spec.define_units(ordinate_unit)
    spec.interpolation = Specification.reading_of(frequencies)
    return spec
