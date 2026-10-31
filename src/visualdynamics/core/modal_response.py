"""Modal responses: how each mode of a model answers the forces a record
holds, computed in modal coordinates.

A model's modes are a compact description of how it moves: a large
finite element model has many more physical DOFs than modes in a
bandwidth, and the response anywhere follows from the modal responses
by u = Φq. So the response to a test's forces is computed and kept as
modal responses, one record per mode, in a time history of their own on
the forces' time axis; the physical response at any DOF is a later
expansion.

The forces are made modal first, pᵣ = Σₖ φₖᵣ fₖ, the projection
`transform.to_modal` makes of a force record (a record already holding
modal forces is used as it is). Each mode r then answers its own
equation,

    q̈ᵣ + 2ζᵣωᵣ q̇ᵣ + ωᵣ² qᵣ = pᵣ(t) / mᵣ,

with ωᵣ, ζᵣ and mᵣ the set's frequency, damping and modal mass (one for
a mass-normalized set). It is solved exactly for a force that varies
linearly between samples: the mode's state equation discretized with a
first-order hold, which is a second-order recursive filter run over the
record. Nothing else is approximated, so the start of a record, a
transient, and a rigid-body mode (ωᵣ = 0, whose displacement under a
net force grows without bound, as it does) are all answered as they are.
The record starts at rest.

A force CPSD is answered the same way, line by line in frequency:
each mode's frequency response H_r carries the modal force density to
the modal response density, every mode against every other
(`modal_response_density`).

The response is a displacement, a velocity or an acceleration of each
modal coordinate, `M1` … `Mn` named as everywhere in the package (the
mode's 1-based index in its set). Through a mass-normalized set the
units carry a half power of mass, the `modal_*` dimensions, as
`transform` gives them; through any other set they are left undeclared
and hinted.

References
----------
.. [1] G. F. Franklin, J. D. Powell and M. L. Workman, *Digital Control
   of Dynamic Systems*, 3rd ed., Addison-Wesley, 1998, sec. 6.3.2
   (the triangle-hold equivalent).
.. [2] R. R. Craig and A. J. Kurdila, *Fundamentals of Structural
   Dynamics*, 2nd ed., Wiley, 2006, ch. 10.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

from ..units import UNKNOWN
from .data import Psd, TimeHistory, channel_quantities
from .shapes import ShapeSet
from .transform import MODAL, _hermitian_block, modal_dofs, reads_as, to_modal

#: what each response is, by the name a user picks
QUANTITIES: dict[str, str] = {'displacement': 'length',
                              'velocity': 'velocity',
                              'acceleration': 'acceleration'}

#: the modal force dimension `transform` gives a mass-normalized set
MODAL_FORCE = MODAL['force']


def _mode_response(force: np.ndarray, frequency: float, damping: float,
                   mass: float, dt: float, quantity: str) -> np.ndarray:
    """A mode's response to its modal force, sampled every `dt` and
    linear between samples, from rest.

    The state x = (q, q̇) over one step with the force linear from u[k]
    to u[k+1] is exactly x[k+1] = Φ x[k] + (Γa − Γb) u[k] + Γb u[k+1],
    with Φ, Γa and Γb blocks of the exponential of the step's augmented
    matrix. Written in w[k] = x[k] − Γb u[k], it is a causal state
    equation, and so a second-order filter; at rest x[0] = 0 means
    w[0] = −Γb u[0], whose decay C Φᵏ w[0] is added to the filter's
    output, which starts from w[0] = 0.
    """
    # scipy's cont2discrete(method='foh') is the triangle-hold
    # equivalent, which is centered: a rigid mode under a constant force
    # came out half a sample early (3·dt·T/2 too far after T seconds),
    # and a two-DOF model 0.37 % off a direct simulation (2026-10-25)
    from scipy.linalg import expm
    from scipy.signal import lfilter, ss2tf

    w = 2.0 * np.pi * float(frequency)
    z = float(damping)
    a = np.array([[0.0, 1.0], [-w * w, -2.0 * z * w]])
    b = np.array([[0.0], [1.0 / mass]])
    if quantity == 'displacement':
        c, d = np.array([[1.0, 0.0]]), np.array([[0.0]])
    elif quantity == 'velocity':
        c, d = np.array([[0.0, 1.0]]), np.array([[0.0]])
    else:
        c, d = np.array([[-w * w, -2.0 * z * w]]), np.array([[1.0 / mass]])
    augmented = np.zeros((4, 4))
    augmented[:2, :2] = a * dt
    augmented[:2, 2:3] = b * dt
    augmented[2, 3] = 1.0
    blocks = expm(augmented)
    phi, gamma_a, gamma_b = blocks[:2, :2], blocks[:2, 2:3], blocks[:2, 3:4]
    num, den = ss2tf(phi, phi @ gamma_b + gamma_a - gamma_b, c,
                     c @ gamma_b + d)
    out = lfilter(np.real(num[0]), np.real(den), force)
    if force.size and force[0] != 0.0:
        # C Φᵏ Γb for k = 0, 1, …: the impulse response of (Φ, Γb, C),
        # one sample on
        num_0, den_0 = ss2tf(phi, gamma_b, c, np.zeros((1, 1)))
        impulse = np.zeros(force.size + 1)
        impulse[0] = 1.0
        decay = lfilter(np.real(num_0[0]), np.real(den_0), impulse)[1:]
        out -= force[0] * decay
    return out


def modal_forces(history: TimeHistory, shapes: ShapeSet,
                 records: Sequence[int] | None = None) -> TimeHistory:
    """The modal forces the record's force channels put on every mode of
    the set, one record per mode, `M1` … `Mn`.

    Force channels on DOFs the set covers are projected, pᵣ = Σ φₖᵣ fₖ;
    a record whose channels are already the set's modal forces is used
    as it is. Other channels (the responses) are passed over.

    Raises
    ------
    ValueError
        If no picked record is a force on a DOF the set covers, the set
        is complex or has no modes.
    """
    if shapes.num_shapes == 0:
        raise ValueError('the set has no modes')
    if shapes.is_complex:
        raise ValueError('the set is complex: a real force cannot drive a '
                         'complex modal coordinate here')
    picked = (list(range(history.num_records)) if records is None
              else sorted({int(i) for i in records}))
    names = set(modal_dofs(shapes))
    modal = [i for i in picked if history.response_dof[i] in names
             and history.ordinate_dim[i] in (MODAL_FORCE, 'force', UNKNOWN)
             and (history.ordinate_dim[i] == MODAL_FORCE
                  or history.dimension_hint[i] == MODAL_FORCE)]
    if modal and reads_as(history, shapes) == 'modal':
        rows = np.zeros((shapes.num_shapes, len(history.abscissa)))
        order = modal_dofs(shapes)
        for i in modal:
            rows[order.index(history.response_dof[i])] += history.ordinate[i]
        dim = history.ordinate_dim[modal[0]]
        return TimeHistory(history.abscissa, rows, response_dof=order,
                           ordinate_dim=dim,
                           dimension_hint=None if dim != UNKNOWN else MODAL_FORCE)
    forces = [i for i in picked if history.ordinate_dim[i] == 'force']
    if not forces:
        raise ValueError('no force channel picked: the modal response is '
                         'the answer to forces, and the record holds none '
                         'on the DOFs picked')
    projected, _report = to_modal(history, shapes, records=forces)
    return projected


def modal_responses(history: TimeHistory, shapes: ShapeSet, *,
                    modes: Sequence[int] | None = None,
                    quantity: str = 'acceleration',
                    records: Sequence[int] | None = None,
                    label: str = 'shapes') -> TimeHistory:
    """Each picked mode's response to the record's forces, one record per
    mode, at the modal coordinates `M1` … `Mn`.

    Parameters
    ----------
    history : TimeHistory
        Evenly sampled, holding force channels on DOFs the set covers,
        or the set's modal forces.
    shapes : ShapeSet
        Real modes with frequency, damping and modal mass.
    modes : sequence of int, optional
        Which modes, by index in the set from zero; every mode when
        omitted.
    quantity : str, default 'acceleration'
        'displacement', 'velocity' or 'acceleration' of each modal
        coordinate.
    records : sequence of int, optional
        Which of the history's records are the forces; every force
        channel when omitted.
    label : str, default 'shapes'
        The set's name, written into each record's comment, which is how
        the records find their shapes again when they are played on a
        geometry or merged in with the forces.

    Returns
    -------
    TimeHistory
        One record per mode picked, on the history's own time axis.

    Raises
    ------
    ValueError
        If the quantity is not one of `QUANTITIES`, a mode is not in the
        set, the record is not evenly sampled, or as `modal_forces`.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f'{quantity!r} is not a response: one of '
                         f'{", ".join(QUANTITIES)}')
    chosen = (list(range(shapes.num_shapes)) if modes is None
              else sorted({int(k) for k in modes}))
    bad = [k for k in chosen if not 0 <= k < shapes.num_shapes]
    if bad:
        raise ValueError(f'no mode {bad[0] + 1}: the set has '
                         f'{shapes.num_shapes}')
    if not chosen:
        raise ValueError('no mode picked')
    t = np.asarray(history.abscissa, dtype=float)
    steps = np.diff(t)
    if t.size < 2 or not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
        raise ValueError('the record is not evenly sampled')
    dt = float(steps[0])
    forces = modal_forces(history, shapes, records)
    names = modal_dofs(shapes)
    rows = np.empty((len(chosen), t.size))
    for row, k in enumerate(chosen):
        rows[row] = _mode_response(np.asarray(forces.ordinate[k], dtype=float),
                                   shapes.frequency[k], shapes.damping[k],
                                   float(shapes.modal_mass[k]), dt, quantity)
    force_dim = forces.ordinate_dim[0]
    physical = QUANTITIES[quantity]
    known = force_dim == MODAL_FORCE
    dim = MODAL[physical] if known else UNKNOWN
    hint = None if known else MODAL[physical]
    comments = [f'{label}: modal {quantity}, mode {k + 1} at '
                f'{float(shapes.frequency[k]):.4g} Hz' for k in chosen]
    return TimeHistory(t, rows, response_dof=[names[k] for k in chosen],
                       ordinate_dim=dim, dimension_hint=hint,
                       comment=comments)


# ---- a force CPSD: the same responses, line by line ----------------------

#: (iw)^p for each quantity, the power `ShapeSet.modal_frf` takes
POWERS: dict[str, int] = {'displacement': 0, 'velocity': 1,
                          'acceleration': 2}


def modal_force_density(density: Psd, shapes: ShapeSet,
                        records: Sequence[int] | None = None) -> Psd:
    """The modal force CPSD the density's forces put on the set,
    S_pp = Φᵀ S_ff Φ, every mode against every mode.

    It is `transform.to_modal` on the force records alone (the matrix
    needs every cross term between the drives, and a density of
    autospectra alone is refused there); a density whose records are
    already the set's modal forces is used as it is, as a time history's
    are in `modal_forces`.

    Raises
    ------
    ValueError
        If the set is complex or has no modes, no picked record is a
        force density, or as `transform.to_modal`.
    """
    if shapes.num_shapes == 0:
        raise ValueError('the set has no modes')
    if shapes.is_complex:
        raise ValueError('the set is complex: a real force cannot drive a '
                         'complex modal coordinate here')
    picked = (list(range(density.num_records)) if records is None
              else sorted({int(i) for i in records}))
    wanted = f'{MODAL_FORCE}**2/frequency'
    if reads_as(density, shapes, picked) == 'modal':
        modal = [i for i in picked if density.known_dim(i) == wanted]
        if not modal:
            raise ValueError('no modal force density picked: the modal '
                             'response is the answer to forces')
        return _subset(density, modal)
    forces = [i for i in picked
              if channel_quantities(density.known_dim(i)) == ('force', 'force')]
    if not forces:
        raise ValueError('no force density picked: the modal response is '
                         'the answer to forces, and the density holds none '
                         'between the DOFs picked')
    projected, _report = to_modal(density, shapes, records=forces)
    return projected


def _subset(density, records):
    out = copy.copy(density)
    for attribute in ('response_dof', 'reference_dof', 'ordinate_dim',
                      'dimension_hint'):
        values = getattr(density, attribute)
        if values is not None:
            setattr(out, attribute, [values[i] for i in records])
    out.ordinate = np.asarray(density.ordinate)[records]
    return out


def modal_response_density(density: Psd, shapes: ShapeSet, *,
                           modes: Sequence[int] | None = None,
                           quantity: str = 'acceleration',
                           records: Sequence[int] | None = None,
                           cross: bool = True) -> Psd:
    """Each picked mode's response to a force CPSD, as the modal
    response CPSD: every picked mode against every other, `M1` … `Mn`.

    The forces are made modal first, S_pp = Φᵀ S_ff Φ
    (`modal_force_density`). Each mode answers its modal force through
    its own frequency response H_r (`ShapeSet.modal_frf`), so with each
    cross term stored as conj(Q_r)·Q_s, as `DataArray.compute_cpsds`
    stores it,

        S_qq,rs = conj(H_r) S_pp,rs H_s.

    Expanding it through the shapes, Φ S_qq Φᵀ, gives conj(H) S_ff Hᵀ
    with H the FRF matrix `ShapeSet.synthesize_frf` gives, so the
    response density at the DOFs is the same as driving the synthesized
    FRFs with the forces. A rigid mode's acceleration at 0 Hz is finite
    (its (iω)² cancels its −ω² exactly); its displacement and velocity
    there are nan.

    Parameters
    ----------
    density : Psd
        A force CPSD with every cross term between the drives (zero
        terms, uncorrelated drives, are terms), or the set's modal force
        CPSD.
    shapes : ShapeSet
        Real modes with frequency, damping and modal mass.
    modes : sequence of int, optional
        Which modes, by index in the set from zero; every mode when
        omitted.
    quantity : str, default 'acceleration'
        'displacement', 'velocity' or 'acceleration' of each modal
        coordinate.
    records : sequence of int, optional
        Which of the density's records are the forces; every force
        record when omitted.
    cross : bool, default True
        Every pair of picked modes; with False, each mode's
        autospectrum alone, as a preview draws them.

    Returns
    -------
    Psd
        One record per pair of picked modes (per mode without `cross`),
        on the density's lines.

    Raises
    ------
    ValueError
        If the quantity is not one of `QUANTITIES`, a mode is not in the
        set, the density is on proportional bands or written at
        breakpoints (its `interpolation` not 'bin'), or as
        `modal_force_density`.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f'{quantity!r} is not a response: one of '
                         f'{", ".join(QUANTITIES)}')
    chosen = (list(range(shapes.num_shapes)) if modes is None
              else sorted({int(k) for k in modes}))
    bad = [k for k in chosen if not 0 <= k < shapes.num_shapes]
    if bad:
        raise ValueError(f'no mode {bad[0] + 1}: the set has '
                         f'{shapes.num_shapes}')
    if not chosen:
        raise ValueError('no mode picked')
    if getattr(density, 'bandwidth', None) is not None:
        # a mode's response varies across a band, so one H at its center
        # would not be the band's response
        raise ValueError('the density is on proportional bands: a mode '
                         'answers line by line, and a band holds many')
    if getattr(density, 'interpolation', 'bin') != 'bin':
        # H at the breakpoints alone: four breakpoints gave four lines
        # and every mode between them was missed, without a word
        # (Brandon, 2026-10-30, on the BARC model)
        raise ValueError('the density is written at breakpoints: a mode '
                         'answers line by line, and the modes between '
                         'breakpoints would be missed. Read it onto lines '
                         'first, at a frequency spacing: Convert to CPSD '
                         'on the specification it came from')
    forces = modal_force_density(density, shapes, records)
    names = modal_dofs(shapes)
    entries = {(forces.response_dof[i], forces.reference_dof[i]): i
               for i in range(forces.num_records)}
    block = _hermitian_block(entries, np.asarray(forces.ordinate),
                             [names[k] for k in chosen], 'modal force density')
    terms = shapes.modal_frf(density.abscissa, chosen, POWERS[quantity]).T
    picked = [names[k] for k in chosen]
    if cross:
        response = (np.conj(terms)[:, :, None] * block * terms[:, None, :]
                    ).reshape(len(density.abscissa), -1).T
        rows = [k for k in picked for _l in picked]
        columns = [l for _k in picked for l in picked]
    else:
        diagonal = np.diagonal(block, axis1=1, axis2=2)
        response = (np.conj(terms) * diagonal * terms).T
        rows = columns = picked
    known = all(forces.ordinate_dim[i] != UNKNOWN for i in entries.values())
    dim = f'{MODAL[QUANTITIES[quantity]]}**2/frequency'
    return Psd(density.abscissa, response, response_dof=rows,
               reference_dof=columns, ordinate_dim=dim if known else UNKNOWN,
               dimension_hint=None if known else dim)


#: the most samples a preview computes, so that it answers while the
#: pane is open; Compute runs the whole record
PREVIEW_SAMPLES = 2 ** 16


def preview_responses(history: TimeHistory, shapes: ShapeSet, *,
                      modes: Sequence[int] | None = None,
                      quantity: str = 'acceleration',
                      records: Sequence[int] | None = None,
                      label: str = 'shapes') -> TimeHistory:
    """`modal_responses` over the first `PREVIEW_SAMPLES` of the record:
    the same responses there, since each starts from rest at the
    record's start."""
    short = copy.copy(history)
    short.abscissa = np.asarray(history.abscissa)[:PREVIEW_SAMPLES]
    short.ordinate = np.asarray(history.ordinate)[:, :PREVIEW_SAMPLES]
    return modal_responses(short, shapes, modes=modes, quantity=quantity,
                           records=records, label=label)
