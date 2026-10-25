"""Modal responses: how each mode of a model answers the forces a record
holds, computed in modal coordinates and kept beside the forces.

A model's modes are a compact description of how it moves: a large
finite element model has many more physical DOFs than modes in a
bandwidth, and the response anywhere follows from the modal responses
by u = Φq. So the response to a test's forces is computed and kept as
modal responses, one record per mode, appended to the time history the
forces are in; the physical response at any DOF is a later expansion.

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
from .data import TimeHistory
from .shapes import ShapeSet
from .transform import MODAL, modal_dofs, reads_as, to_modal

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
        a later run through the same set finds the records to replace.

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


def made_through(history: TimeHistory, label: str) -> list[int]:
    """The records of `history` that an earlier run through the set
    named `label` appended (`modal_responses` writes the name into each
    record's comment)."""
    prefix = f'{label}: modal '
    return [i for i, comment in enumerate(history.comment)
            if comment.startswith(prefix)]


def appended(history: TimeHistory, responses: TimeHistory,
             label: str) -> TimeHistory:
    """`history` with `responses` appended as records after its own, the
    records an earlier run through the same set made taken out first.
    Its averaging, shock windows, roles and other settings are kept;
    `history` itself is not changed."""
    keep = [i for i in range(history.num_records)
            if i not in set(made_through(history, label))]
    out = copy.copy(history)
    n = len(responses.response_dof)

    def joined(field: str, new: list) -> list | None:
        old = getattr(history, field)
        if old is None and all(v is None for v in new):
            return None
        old = [None] * history.num_records if old is None else list(old)
        return [old[i] for i in keep] + list(new)

    out.ordinate = np.vstack([np.asarray(history.ordinate)[keep],
                              responses.ordinate])
    out.response_dof = [history.response_dof[i] for i in keep] + list(
        responses.response_dof)
    out.ordinate_dim = [history.ordinate_dim[i] for i in keep] + list(
        responses.ordinate_dim)
    out.ordinate_unit = [history.ordinate_unit[i] for i in keep] + list(
        responses.ordinate_unit)
    out.reference_unit = [history.reference_unit[i] for i in keep] + list(
        responses.reference_unit)
    out.comment = [history.comment[i] for i in keep] + list(responses.comment)
    out.dimension_hint = [history.dimension_hint[i] for i in keep] + list(
        responses.dimension_hint)
    out.reference_dof = joined('reference_dof', [None] * n)
    # a history with no capture labels keeps none: blank labels would
    # read as one capture named '' (2026-10-25)
    out.block = (None if history.block is None
                 else [b or '' for b in joined('block', [''] * n)])
    return out


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
