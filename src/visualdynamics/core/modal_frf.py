"""Each mode's own frequency response, synthesized from a shape set.

A real mode's modal FRF is the response of its modal coordinate to a
unit modal force,

    H_r(f) = (iw)^p / (m_r (w_r^2 - w^2 + 2i z_r w_r w)),

with w = 2*pi*f and p = 0, 1 or 2 for displacement, velocity or
acceleration (`ShapeSet.modal_frf`). It needs only the mode's
frequency, damping and modal mass, not its shape. A physical FRF is the
shapes on both sides of it, H_jk = sum_r phi_jr H_r phi_kr, which is
what `ShapeSet.synthesize_frf` sums.

The result is an `Frf` with one record per mode, its response and its
reference both at the mode's modal coordinate (`M1` ... `Mn`). The
lines are a test's: `frame_length` samples at `sample_rate` give
lines every `sample_rate / frame_length` Hz from 0 Hz to the Nyquist
frequency, the lines Compute FRFs gives a record framed that way.

The units follow the modal coordinates (`transform`): through
mass-normalized shapes a modal displacement is m·kg½ and a modal force
N/kg½, so the modal FRF is in s² for displacement, s for velocity and
is dimensionless for acceleration. A set with no mass unit gives modal
FRFs with no unit, the modal dimension kept as the hint.

`frfs_at_channels` goes back: from modal FRFs and the shapes at a
channel table's channels to the physical FRFs between them, each
response channel against each reference channel, each row in what its
channel measures. Modal FRFs that hold each mode against itself are a
diagonal matrix; a transform of a measured FRF holds every mode against
every mode, and both expand the same way.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..units import UNKNOWN
from .data import Frf, channel_quantities
from .modal_response import POWERS, QUANTITIES
from .shapes import ShapeSet
from .transform import TransformReport, _columns, _modal_dim, _physical_quantity, modal_dofs
from .validate import modal_coordinate

#: (iw)^p for each physical motion, as a channel table names it
MOTION_POWERS = {'length': 0, 'velocity': 1, 'acceleration': 2}


def frequency_lines(sample_rate: float, frame_length: int) -> np.ndarray:
    """The lines a frame of `frame_length` samples at `sample_rate`
    gives: every `sample_rate / frame_length` Hz from 0 Hz to the
    Nyquist frequency.

    Parameters
    ----------
    sample_rate : float
        Samples per second, in Hz.
    frame_length : int
        Samples in a frame, two or more.

    Returns
    -------
    numpy.ndarray
        The lines, in Hz.

    Raises
    ------
    ValueError
        If the sample rate is not positive or the frame has fewer than
        two samples.
    """
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f'the sample rate must be positive, not {sample_rate}')
    if int(frame_length) != frame_length or frame_length < 2:
        raise ValueError('a frame needs two samples or more, '
                         f'not {frame_length}')
    return np.fft.rfftfreq(int(frame_length), 1.0 / float(sample_rate))


def picked_modes(shapes: ShapeSet, modes: Sequence[int] | None) -> list[int]:
    """The modes asked for, by index from zero, sorted and once each;
    every mode when `modes` is None.

    Raises
    ------
    ValueError
        If the set has no modes, a mode is not in it, or none is asked
        for.
    """
    if shapes.num_shapes == 0:
        raise ValueError('the set has no modes')
    if modes is None:
        return list(range(shapes.num_shapes))
    chosen = sorted({int(k) for k in modes})
    outside = [k + 1 for k in chosen if not 0 <= k < shapes.num_shapes]
    if outside:
        raise ValueError(f'the set has {shapes.num_shapes} modes; mode '
                         f'{outside[0]} is not one of them')
    if not chosen:
        raise ValueError('no modes are picked')
    return chosen


def synthesize_modal_frfs(shapes: ShapeSet, sample_rate: float,
                          frame_length: int,
                          quantity: str = 'acceleration',
                          modes: Sequence[int] | None = None,
                          label: str = 'shapes') -> Frf:
    """Each picked mode's modal FRF, on the lines of a test's frame.

    Parameters
    ----------
    shapes : ShapeSet
        A real shape set. Only its frequencies, damping, modal masses and
        mass unit are read.
    sample_rate : float
        The sample rate the lines follow from, in Hz.
    frame_length : int
        Samples in a frame; the line spacing is `sample_rate /
        frame_length`.
    quantity : str, default 'acceleration'
        The response per modal force: 'displacement', 'velocity' or
        'acceleration'.
    modes : sequence of int, optional
        Which modes, by index from zero. All of them when omitted.
    label : str, default 'shapes'
        The set's name, written in each record's comment.

    Returns
    -------
    Frf
        One record per mode, at `M1` ... `Mn` against itself. A rigid
        mode's displacement and velocity at 0 Hz are unbounded and are
        nan; its acceleration there is 1/m_r.

    Raises
    ------
    ValueError
        If the set is complex, a mode is not in it, the quantity is not
        one of the three, or the lines cannot be made (as
        `frequency_lines`).
    """
    if quantity not in QUANTITIES:
        raise ValueError(f'the response is one of {", ".join(QUANTITIES)}, '
                         f'not {quantity!r}')
    lines = frequency_lines(sample_rate, frame_length)
    chosen = picked_modes(shapes, modes)
    values = shapes.modal_frf(lines, chosen, POWERS[quantity])
    names = modal_dofs(shapes)
    dims, hints = [], []
    for k in chosen:
        response, response_hint = _modal_dim(shapes, QUANTITIES[quantity], k)
        force, force_hint = _modal_dim(shapes, 'force', k)
        if UNKNOWN in (response, force):
            dims.append(UNKNOWN)
            hints.append(f'{response_hint or response}/{force_hint or force}')
        else:
            dims.append(f'{response}/{force}')
            hints.append(None)
    comments = [f'{label}: modal {quantity} per modal force, mode {k + 1} at '
                f'{float(shapes.frequency[k]):.4g} Hz' for k in chosen]
    return Frf(lines, values, response_dof=[names[k] for k in chosen],
               reference_dof=[names[k] for k in chosen], ordinate_dim=dims,
               dimension_hint=hints, comment=comments)


def above_nyquist(shapes: ShapeSet, modes: Sequence[int] | None,
                  sample_rate: float) -> int:
    """How many of the picked modes lie above the Nyquist frequency:
    they are kept, since each still adds its stiffness below it, and the
    pane says how many there are."""
    chosen = picked_modes(shapes, modes)
    return int(np.sum(shapes.frequency[chosen] > float(sample_rate) / 2.0))


def modal_matrix(modal: Frf, records: Sequence[int] | None = None
                 ) -> tuple[list[int], np.ndarray, str, bool]:
    """The modal FRF matrix a modal FRF's records hold.

    Each mode against itself, as Synthesize Modal FRFs writes them, is
    the diagonal and the rest is zero: a mode's coordinate answers only
    its own modal force. Every mode against every mode, as a transform
    of a measured FRF gives, is the whole matrix.

    Parameters
    ----------
    modal : Frf
        Records at modal coordinates (`M1` ...), response and reference.
    records : sequence of int, optional
        Which records; all of them when omitted. Picking some modes'
        records sums those modes alone.

    Returns
    -------
    tuple of (list of int, numpy.ndarray, str, bool)
        The modes by index from zero, the matrix (lines × modes ×
        modes), the physical quantity the records answer in
        ('acceleration', 'velocity' or 'length'), and whether their units
        are defined.

    Raises
    ------
    ValueError
        If a record is not at a modal coordinate, or not a motion per
        modal force, the records answer in two quantities, or some modes
        hold cross terms and others do not.
    """
    picked = range(modal.num_records) if records is None else records
    entries: dict[tuple[int, int], int] = {}
    quantities, known = set(), True
    for i in picked:
        row = modal_coordinate(modal.response_dof[i])
        column = modal_coordinate(modal.reference_dof[i])
        if row is None or column is None:
            raise ValueError(f'{modal.response_dof[i]}/'
                             f'{modal.reference_dof[i]} is not a modal FRF: '
                             'its response and reference are modal '
                             'coordinates, M1 and on')
        response, force = channel_quantities(modal.known_dim(i))
        quantity = _physical_quantity(response)
        if quantity not in MOTION_POWERS or _physical_quantity(force) != 'force':
            raise ValueError(f'{modal.response_dof[i]} is not a motion per '
                             'modal force')
        quantities.add(quantity)
        known = known and modal.ordinate_dim[i] != UNKNOWN
        entries[(row - 1, column - 1)] = i
    if not entries:
        raise ValueError('no modal FRF is picked')
    if len(quantities) != 1:
        raise ValueError('the records answer in '
                         f'{" and ".join(sorted(quantities))}; expand one '
                         'response at a time')
    modes = sorted({k for pair in entries for k in pair})
    where = {k: n for n, k in enumerate(modes)}
    diagonal = all(row == column for row, column in entries)
    if not diagonal:
        missing = [(row, column) for row in modes for column in modes
                   if (row, column) not in entries]
        if missing:
            row, column = missing[0]
            raise ValueError(
                f'no FRF between M{row + 1} and M{column + 1}: give each '
                'mode against itself, or every mode against every mode')
    matrix = np.zeros((modal.abscissa.size, len(modes), len(modes)),
                      dtype=complex)
    for (row, column), i in entries.items():
        matrix[:, where[row], where[column]] = modal.ordinate[i]
    return modes, matrix, quantities.pop(), known


def frfs_at_channels(modal: Frf, shapes: ShapeSet, references: Sequence[str],
                     responses: Sequence[tuple[str, str]],
                     records: Sequence[int] | None = None
                     ) -> tuple[Frf, TransformReport]:
    """Physical FRFs between channels, from modal FRFs and the shapes at
    those channels:

        H_jk(f) = sum_r sum_s phi_jr H_rs(f) phi_ks,

    which for modal FRFs each against itself is the modal sum
    `ShapeSet.synthesize_frf` makes.

    Parameters
    ----------
    modal : Frf
        Modal FRFs (`modal_matrix`).
    shapes : ShapeSet
        The set the modal FRFs are of, or that set at the channels'
        DOFs: its mode `r` is the modal coordinate `M{r+1}`. A DOF it
        does not cover is dropped and reported.
    references : sequence of str
        The force channels' DOFs, the columns.
    responses : sequence of tuple of (str, str)
        Each response channel's DOF and what it measures:
        'acceleration', 'velocity' or 'length'. Each row is turned from
        the modal FRFs' quantity into its own by (iw)^p; where that
        divides by zero frequency, the line is nan.
    records : sequence of int, optional
        Which modal records; all of them when omitted.

    Returns
    -------
    tuple of (Frf, TransformReport)
        The FRFs, one record per response and reference pair in
        response order, and what was dropped.

    Raises
    ------
    ValueError
        As `modal_matrix`; if a modal coordinate is past the set's last
        mode; if a response measures something other than a motion; or
        if no response or no reference is on a DOF the set covers.
    """
    modes, matrix, quantity, known = modal_matrix(modal, records)
    if modes[-1] >= shapes.num_shapes:
        raise ValueError(f'the set has {shapes.num_shapes} modes, and the '
                         f'modal FRFs reach M{modes[-1] + 1}')
    outside = [measured for _dof, measured in responses
               if measured not in MOTION_POWERS]
    if outside:
        raise ValueError(f'a response measures {outside[0]}, not a motion')
    report = TransformReport()
    columns, phi_f = _columns(shapes, list(references), report)
    kinds = dict(responses)
    rows, phi_u = _columns(shapes, [dof for dof, _measured in responses],
                           report)
    if not columns:
        raise ValueError('no reference is on a DOF the set covers')
    if not rows:
        raise ValueError('no response is on a DOF the set covers')
    phi_u = np.real(np.asarray(phi_u))[:, modes]
    phi_f = np.real(np.asarray(phi_f))[:, modes]
    physical = phi_u @ matrix @ phi_f.T                    # lines × nu × nf
    omega = 2.0 * np.pi * modal.abscissa
    values, response_dof, reference_dof, dims, hints = [], [], [], [], []
    for a, dof in enumerate(rows):
        measured = kinds[dof]
        step = MOTION_POWERS[measured] - MOTION_POWERS[quantity]
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = (1j * omega) ** step
        if step < 0:
            factor[omega == 0.0] = np.nan
        dim = f'{measured}/force'
        for b, reference in enumerate(columns):
            values.append(physical[:, a, b] * factor)
            response_dof.append(dof)
            reference_dof.append(reference)
            dims.append(dim if known else UNKNOWN)
            hints.append(None if known else dim)
    report.shared[quantity] = [f'M{k + 1}' for k in modes]
    return (Frf(modal.abscissa, np.asarray(values), response_dof=response_dof,
                reference_dof=reference_dof, ordinate_dim=dims,
                dimension_hint=hints), report)
