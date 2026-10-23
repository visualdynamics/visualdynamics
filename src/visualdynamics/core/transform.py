"""Transform physical responses through a shape set to modal responses,
and back.

The relationship is `u = Φq`. Physical to modal is `q = Φ⁺u`, the
least-squares fit over the DOFs the data and the set share; modal to
physical is `u = Φq`, exact by definition. Any shape set can be used: a
fitted set, an eigensolution, or the six rigid-body shapes of a
geometry, which gives the virtual point transformation of the
substructuring literature with no special case in the code.

Each kind of data goes through the same two rules, in the form its kind
needs. A time history or a spectrum is rows, one per channel, and is
transformed as `q = Φ⁺u` line by line, complex where the rows are. A
cross-spectral matrix is a quadratic form, `S_qq = Φ⁺ S_uu Φ⁺ᴴ`, so it
needs every cross term between the shared channels: the phase between
two channels is what tells a translation from a rotation. A set of
autospectra alone is refused, not completed with assumed cross terms;
the user computes or defines the cross terms when they are needed. An
FRF is a matrix from references to responses. Its response rows
transform through `Φ⁺` like any response, and its reference columns
through the force rule, `f = (Φᵀ)⁺ f_m`, when the set covers the
drives, which gives the virtual point's FRF in both halves. References
the set does not cover stay physical, and the report says so. A shock
response spectrum (a maximum), a coherence (a ratio) and a sine
level (read tone by tone beside a noise floor) do not transform at
all, and the refusal says to transform the time history and compute
them again.

Records are matched to shape columns by DOF, within each quantity, with
the sign honored: a `101Z-` channel against a `101Z+` column is
negated. Every shape column is a displacement today.

Each record's quantity sets how it transforms. Motions (acceleration,
velocity, displacement) transform as responses, `q = Φ⁺u`, each
quantity separately with its own unit. Forces transform the other way:
a force is work-conjugate to a motion, so the modal force is `Φᵀf`,
with no pseudo-inverse and no rank condition. Every other quantity
(temperature, voltage, pressure, strain) is left out and reported,
since a displacement shape set says nothing about it. A record whose
units are not defined is left out and reported too, since its values
are raw numbers.

The unit rule is `[q] = [u]/[Φ]`. Unit rigid-body shapes have
dimensionless translations and length-per-radian rotations, so the
translational modal responses keep the data's quantity and the
rotational ones come out per radian (rad/s², or a moment for a force).
A mass-normalized set is in 1/√kg, so its modal responses carry a half
power of mass, the `modal_*` dimensions. Any other set with no declared
mass unit gives modal responses with no unit, hinted with the `modal_*`
dimension for Define Units to declare. The expansion applies the same
rule the other way, so a round trip returns the original quantity and
unit.

A specification's bands carry through when every channel has the same
band. The transform is linear, so a band that is the same number of
decibels on every channel at a line is the same on every modal channel
there. Bands that differ between channels have no single answer, and
the transform refuses them.

A modal coordinate is named `M` and the mode's 1-based index (`M1` …
`M6` for a rigid set, `M1` … `M22` for a fitted one), one rule for
every set (`validate.modal_coordinate`). The names are visibly not
nodes and do not change when a set is renamed; which set a coordinate
belongs to is in the object's provenance and in each record's comment,
which names the mode as the set describes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..units import UNKNOWN
from .filters import carrying_marks
from .rigid import is_rigid_set
from .shapes import _split_sign
from .validate import MODAL_PREFIX, modal_coordinate

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import DataArray
    from .shapes import ShapeSet

# The design is PLAN.md, "The virtual point arc", phase 2. No cross
# terms are ever assumed (Brandon, 2026-09-04: no assumptions about
# cross terms; the user defines or computes them when they are needed).

#: the response quantities that transform as motions, `q = Φ⁺u`
MOTIONS = ('acceleration', 'velocity', 'length')
#: the quantities that transform as their work-conjugates, `Φᵀf`
FORCES = ('force',)
#: what a physical quantity becomes through a unit rigid rotation
#: shape — per radian — and through a mass-normalized shape
ROTATIONAL = {'acceleration': 'angular_acceleration',
              'velocity': 'angular_velocity', 'length': 'angle',
              'force': 'moment'}
MODAL = {quantity: f'modal_{quantity}' for quantity in (*MOTIONS, *FORCES)}
#: the modal quantity back to the physical one it came from
PHYSICAL = {**{modal: physical for physical, modal in MODAL.items()},
            **{angular: physical for physical, angular in ROTATIONAL.items()}}

#: how far two channels' bands may differ, as a ratio, and still be
#: called the same band — the round-off of a controller writing them
BAND_TOLERANCE = 1e-6


def modal_dofs(shapes: ShapeSet) -> list[str]:
    """The DOF names a set's modal coordinates take: `M1` … `Mn`.

    Parameters
    ----------
    shapes : ShapeSet
        The set.

    Returns
    -------
    list of str
        One DOF string per mode, in mode order.
    """
    return [f'{MODAL_PREFIX}{k + 1}' for k in range(shapes.num_shapes)]


def _cannot(data: Any) -> str | None:
    """Why a kind of data does not transform, or None when it does."""
    from .data import DataArray, Srs, _CoherenceBase
    from .sine import SineLevel, SineLevelSet

    # A SineLevel is a Spectrum, so it used to pass as rows: a tracked
    # level is a magnitude (track_sine's are real), and an extracted
    # one carries a noise floor per channel the rows cannot carry.
    if isinstance(data, (SineLevel, SineLevelSet)):
        return ('a sine level is read out of the record tone by tone, '
                'each reading beside its own noise floor — transform the '
                'time history and extract the levels again')
    if not isinstance(data, DataArray):
        return f'{type(data).__name__} is not a data object'
    if isinstance(data, Srs):
        return ('a shock response spectrum is a maximum of an oscillator\'s '
                'response, not a linear function of the record — transform '
                'the time history and compute the SRS again')
    if isinstance(data, _CoherenceBase):
        return ('a coherence is a ratio — transform the time history and '
                'compute it again')
    return None


def _is_modal(dof: str, shapes: ShapeSet) -> bool:
    index = modal_coordinate(dof)
    return index is not None and index <= shapes.num_shapes


def reads_as(data: DataArray, shapes: ShapeSet) -> str | None:
    """Which way a transform of `data` through `shapes` would go.

    `'physical'` when any of the data's DOFs is on the set's
    coordinates, `'modal'` when any is one of the set's modal names, and
    None when neither, or when the kind of data does not transform at
    all.

    Parameters
    ----------
    data : DataArray
        The object to read.
    shapes : ShapeSet
        The set to read it against.

    Returns
    -------
    str or None
        Which way a transform would go.
    """
    if _cannot(data) is not None:
        return None
    lookup = shapes._dof_lookup()
    dofs = list(data.response_dof) + list(data.reference_dof or [])
    if any(_split_sign(dof)[0] in lookup for dof in dofs):
        return 'physical'
    if any(_is_modal(dof, shapes) for dof in dofs):
        return 'modal'
    return None


@dataclass
class TransformReport:
    """What a transform used, left out, and left unexplained.

    Attributes
    ----------
    shared : dict of str to list of str
        The DOFs matched to shape columns, per quantity.
    dropped : list of str
        The DOFs of transformable quantities the set has no column for,
        such as a shaker's synthetic drive DOF or a node the set never
        covered.
    skipped : dict of str to int
        How many records of each quantity that does not transform were
        left out (`{'temperature': 2}`), including records whose units
        are not defined.
    rank : dict of str to (int, int)
        Per motion quantity, the rank of the shared shape matrix and the
        number of modes.
    residual : dict of str to float
        Per shared motion DOF, the fraction of its RMS (or of its power,
        for a density) the modes do not explain; 0 when the fit is
        exact.
    notes : list of str
        Anything else worth a sentence, such as references kept physical
        or cross-quantity terms left out.
    """

    shared: dict[str, list[str]] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    rank: dict[str, tuple[int, int]] = field(default_factory=dict)
    residual: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def worst_residual(self) -> float:
        """The largest per-DOF residual fraction, or 0 with none."""
        return max(self.residual.values(), default=0.0)

    def describe(self) -> str:
        """One line saying what was shared, dropped, skipped and left
        unexplained, worded as the status line shows it."""
        shared = len({dof for dofs in self.shared.values() for dof in dofs})
        parts = [f'{shared} DOF{"s" * (shared != 1)} shared']
        if self.dropped:
            parts.append(f'{len(self.dropped)} not in the shapes')
        if self.skipped:
            parts.append('not transformed: ' + ', '.join(
                f'{count} {quantity}' for quantity, count in
                self.skipped.items()))
        if self.residual:
            parts.append(f'residual {self.worst_residual:.1%} at worst')
        parts.extend(self.notes)
        return ', '.join(parts)


# ---- what every kind shares -----------------------------------------------------


def _check_set(shapes):
    if shapes.is_complex:
        raise ValueError('a complex shape set cannot carry real data into '
                         'modal coordinates')
    if not shapes.num_shapes:
        raise ValueError('the shape set has no modes')


def _picked(data, records):
    """The record indices a pick names, or every record."""
    if records is None:
        return list(range(data.num_records))
    chosen = sorted({int(i) for i in records})
    bad = [i for i in chosen if not 0 <= i < data.num_records]
    if bad:
        raise ValueError(f'no record {bad[0]}: the object has '
                         f'{data.num_records}')
    if not chosen:
        raise ValueError('no records picked')
    return chosen


def _modal_dim(shapes, quantity, mode):
    """The dimension a response of `quantity` takes through mode `mode`
    of the set, and the hint when it has none — the unit rule."""
    if shapes.mass_unit is not None:
        return MODAL[quantity], None
    if is_rigid_set(shapes) and shapes.unscaled:
        return (quantity if mode < 3 else ROTATIONAL[quantity]), None
    return UNKNOWN, MODAL[quantity]


def _columns(shapes, dofs, report, quantity=None):
    """(kept dofs, shape rows (len × modes) in the data's signs), with
    the DOFs the set has no column for reported as dropped."""
    lookup = shapes._dof_lookup()
    kept, rows = [], []
    for dof in dofs:
        base, sign = _split_sign(dof)
        found = lookup.get(base)
        if found is None:
            if dof not in report.dropped:
                report.dropped.append(dof)
            continue
        column, column_sign = found
        kept.append(dof)
        rows.append(np.real(shapes.shape_matrix[:, column]) * sign * column_sign)
    return kept, (np.asarray(rows) if rows else np.empty((0, shapes.num_shapes)))


def _fit(phi, quantity, count, shapes, report):
    """`Φ⁺` for a motion block, after the rank check that refuses a
    measurement too thin to resolve the modes."""
    rank = int(np.linalg.matrix_rank(phi))
    report.rank[quantity] = (rank, shapes.num_shapes)
    if rank < shapes.num_shapes:
        raise ValueError(
            f'the {count} shared {quantity} DOFs resolve only {rank} of '
            f'{shapes.num_shapes} modes — measure more directions, or '
            'transform through fewer modes')
    return np.linalg.pinv(phi)


def _quantity_class(quantity):
    """'motion', 'force' or None — how a physical quantity transforms."""
    if quantity in MOTIONS:
        return 'motion'
    if quantity in FORCES:
        return 'force'
    return None


def _physical_quantity(dim):
    """The physical quantity a modal record's dimension came from, or
    None when it did not come through a transform."""
    return PHYSICAL.get(dim, dim if dim in MOTIONS or dim in FORCES else None)


def _rms(values):
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


# ---- rows: a time history, a spectrum ------------------------------------------


def _rows_to_modal(data, shapes, records, report):
    from .data import Spectrum, TimeHistory

    names = modal_dofs(shapes)
    rows, dofs, dims, hints, comments = [], [], [], [], []
    groups: dict[str, list[int]] = {}
    for i in _picked(data, records):
        groups.setdefault(data.known_dim(i), []).append(i)
    for quantity, chosen in groups.items():
        kind = _quantity_class(quantity)
        if kind is None:
            report.skipped[quantity] = len(chosen)
            continue
        if any(data.ordinate_dim[i] == UNKNOWN for i in chosen):
            # a hinted record's values are raw: the scale is what is
            # missing, and a transform of raw numbers would carry the
            # hint's authority without its meaning
            report.skipped[f'{quantity} (units undefined)'] = len(chosen)
            continue
        by_dof = {data.response_dof[i]: i for i in chosen}
        kept, phi = _columns(shapes, list(by_dof), report)
        if not kept:
            continue
        report.shared[quantity] = kept
        values = np.asarray(data.ordinate[[by_dof[dof] for dof in kept]])
        if kind == 'motion':
            modal = _fit(phi, quantity, len(kept), shapes, report) @ values
            unexplained = values - phi @ modal
            for dof, before, after in zip(kept, values, unexplained):
                scale = _rms(before)
                report.residual[dof] = (_rms(after) / scale if scale > 0.0
                                        else 0.0)
        else:
            modal = phi.T @ values
        for mode in range(shapes.num_shapes):
            dim, hint = _modal_dim(shapes, quantity, mode)
            rows.append(modal[mode])
            dofs.append(names[mode])
            dims.append(dim)
            hints.append(hint)
            comments.append(f'{quantity} of mode {mode + 1}'
                            + (f': {shapes.description[mode]}'
                               if shapes.description[mode] else ''))
    if not rows:
        raise ValueError(
            'nothing to transform: no motion or force channel of this '
            'record is on a DOF the shapes cover'
            + (f' (not transformed: {report.describe()})'
               if report.skipped or report.dropped else ''))
    cls = Spectrum if isinstance(data, Spectrum) else TimeHistory
    return cls(data.abscissa, np.asarray(rows), response_dof=dofs,
               ordinate_dim=dims, dimension_hint=hints, comment=comments)


def _rows_to_physical(data, shapes, records, report):
    from .data import Spectrum, TimeHistory

    names = modal_dofs(shapes)
    grouped: dict[str, dict[int, int]] = {}
    for i in _picked(data, records):
        dim = data.known_dim(i)
        quantity = _physical_quantity(dim)
        # modal forces do not expand: many force distributions give
        # one modal force, so they are left out and said, like any
        # other quantity the expansion cannot carry
        if quantity is None or quantity in FORCES:
            report.skipped[dim] = report.skipped.get(dim, 0) + 1
            continue
        index = modal_coordinate(data.response_dof[i])
        if index is None or index > shapes.num_shapes:
            report.dropped.append(data.response_dof[i])
            continue
        grouped.setdefault(quantity, {})[index - 1] = i
    if not grouped:
        forces = any(_physical_quantity(dim) in FORCES
                     for dim in report.skipped)
        raise ValueError(
            'nothing to expand: no motion record of this object is at '
            'one of the shapes\' modal DOFs'
            + (' (modal forces do not expand: many distributions give '
               'the same modal force)' if forces else ''))
    rows, dofs, dims, hints, comments = [], [], [], [], []
    for quantity, by_mode in grouped.items():
        modes = sorted(by_mode)
        chosen = [by_mode[k] for k in modes]
        report.shared[quantity] = [data.response_dof[i] for i in chosen]
        undefined = any(data.ordinate_dim[i] == UNKNOWN for i in chosen)
        modal = np.asarray(data.ordinate[chosen])
        physical = np.real(shapes.shape_matrix[modes]).T @ modal
        carried = _carried(modes, shapes, names)
        for column, dof in enumerate(shapes.coordinate):
            rows.append(physical[column])
            dofs.append(dof)
            dims.append(UNKNOWN if undefined else quantity)
            hints.append(quantity if undefined else None)
            comments.append(f'{quantity} from {carried}')
    cls = Spectrum if isinstance(data, Spectrum) else TimeHistory
    return cls(data.abscissa, np.asarray(rows), response_dof=dofs,
               ordinate_dim=dims, dimension_hint=hints, comment=comments)


def _carried(modes, shapes, names):
    return ('all modes' if len(modes) == shapes.num_shapes
            else 'modes ' + ', '.join(names[k] for k in modes))


# ---- matrices: a cross-spectral density, a specification ---------------------


def _hermitian_block(entries, ordinate, dofs, what):
    """The (lines, n, n) matrix over `dofs` from the records in
    `entries` ({(row dof, column dof): record}), the missing half of a
    pair filled from its conjugate — a CPSD is Hermitian by definition,
    so that is a fact and not a guess — and a pair present neither way
    refused by name."""
    lines = ordinate.shape[1]
    block = np.zeros((lines, len(dofs), len(dofs)), dtype=np.complex128)
    missing = []
    for a, da in enumerate(dofs):
        for b, db in enumerate(dofs):
            i, j = entries.get((da, db)), entries.get((db, da))
            if i is not None:
                block[:, a, b] = ordinate[i]
            elif j is not None:
                block[:, a, b] = np.conj(ordinate[j])
            else:
                missing.append((da, db))
    if missing:
        da, db = missing[0]
        raise ValueError(
            f'no cross terms between {da} and {db}'
            + (f' and {len(missing) - 1} more pair'
               f'{"s" * (len(missing) != 2)}' if len(missing) > 1 else '')
            + f': a {what} transforms as a matrix and needs every pair '
            '— compute the CPSDs from the time history, or read the '
            'specification with its cross terms')
    return block


def _band_ratios(data, autos, lines, quantity):
    """{limit name: ratio per line} when every auto record wears the
    same band at every line, else a refusal naming the two that differ."""
    ratios = {}
    for name, limit in getattr(data, 'limits', {}).items():
        per_record = []
        for dof, i in autos:
            target = np.real(data.ordinate[i])
            with np.errstate(divide='ignore', invalid='ignore'):
                per_record.append((dof, np.where(target > 0,
                                                 limit[i] / target, np.nan)))
        first_dof, first = per_record[0]
        for dof, ratio in per_record[1:]:
            both = np.isfinite(first) & np.isfinite(ratio)
            if not np.allclose(first[both], ratio[both], rtol=BAND_TOLERANCE):
                where = int(np.flatnonzero(both & ~np.isclose(
                    first, ratio, rtol=BAND_TOLERANCE))[0])
                raise ValueError(
                    f'the {name.replace("_", " ")} band differs between '
                    f'{first_dof} ({10 * np.log10(first[where]):+.2f} dB) '
                    f'and {dof} ({10 * np.log10(ratio[where]):+.2f} dB) at '
                    f'{lines[where]:.4g} Hz — a band carries through the '
                    'transform only when every channel wears the same one')
        ratios[name] = first
    return ratios


def _density_dims(dims_per_mode, hints_per_mode, k, l):
    """The dimension of a density between modal channels k and l, and
    its hint when either has none."""
    dk, dl = dims_per_mode[k], dims_per_mode[l]
    if dk == UNKNOWN or dl == UNKNOWN:
        hk, hl = hints_per_mode[k], hints_per_mode[l]
        return UNKNOWN, (f'{hk}**2/frequency' if hk == hl
                         else f'{hk}*{hl}/frequency')
    return (f'{dk}**2/frequency' if dk == dl
            else f'{dk}*{dl}/frequency'), None


def _matrix_to_modal(data, shapes, records, report):
    from .data import channel_quantities

    if data.reference_dof is None:
        raise ValueError(
            'no cross terms: this density holds autospectra only, and the '
            'transform needs the matrix — compute the CPSDs from the time '
            'history, or read the specification with its cross terms')
    names = modal_dofs(shapes)
    blocks: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    for i in _picked(data, records):
        pair = channel_quantities(data.known_dim(i))
        blocks.setdefault(pair, {})[
            (data.response_dof[i], data.reference_dof[i])] = i
    rows, resp, refs, dims, hints, limits = [], [], [], [], [], {}
    left_out = 0
    for (qi, qj), entries in blocks.items():
        if qi != qj:
            left_out += len(entries)
            continue
        kind = _quantity_class(qi)
        if kind is None:
            report.skipped[qi] = len(entries)
            continue
        if any(data.ordinate_dim[i] == UNKNOWN for i in entries.values()):
            report.skipped[f'{qi} (units undefined)'] = len(entries)
            continue
        ordered = list(dict.fromkeys(
            dof for pair in entries for dof in pair))
        kept, phi = _columns(shapes, ordered, report)
        if not kept:
            continue
        block = _hermitian_block(entries, data.ordinate, kept,
                                 f'{qi} density')
        report.shared[qi] = kept
        if kind == 'motion':
            inverse = _fit(phi, qi, len(kept), shapes, report)
            modal = inverse @ block @ inverse.conj().T
            explained = phi @ modal @ phi.T
            for a, dof in enumerate(kept):
                power = float(np.sum(np.real(block[:, a, a])))
                short = float(np.sum(np.real(block[:, a, a]
                                             - explained[:, a, a])))
                report.residual[dof] = (max(short, 0.0) / power
                                        if power > 0.0 else 0.0)
        else:
            modal = phi.T @ block @ phi
        # the bands, when every channel wears the same one: the autos
        # of this block, by the record that holds each
        autos = [(dof, entries.get((dof, dof))) for dof in kept]
        autos = [(dof, i) for dof, i in autos if i is not None]
        ratios = _band_ratios(data, autos, data.abscissa, qi) if autos else {}
        per_mode = [_modal_dim(shapes, qi, k) for k in range(shapes.num_shapes)]
        mode_dims = [dim for dim, _hint in per_mode]
        mode_hints = [hint for _dim, hint in per_mode]
        for k in range(shapes.num_shapes):
            for l in range(shapes.num_shapes):
                dim, hint = _density_dims(mode_dims, mode_hints, k, l)
                rows.append(modal[:, k, l])
                resp.append(names[k])
                refs.append(names[l])
                dims.append(dim)
                hints.append(hint)
                for name, ratio in ratios.items():
                    limits.setdefault(name, []).append(
                        ratio * np.real(modal[:, k, k]) if k == l
                        else np.full(len(data.abscissa), np.nan))
    if left_out:
        report.notes.append(f'{left_out} cross-quantity terms left out')
    if not rows:
        raise ValueError(
            'nothing to transform: no motion or force density of this '
            'object is on DOFs the shapes cover'
            + (f' (not transformed: {report.describe()})'
               if report.skipped or report.dropped else ''))
    return _density_like(data, np.asarray(rows), resp, refs, dims, hints,
                         limits)


def _matrix_to_physical(data, shapes, records, report):
    from .data import channel_quantities

    if data.reference_dof is None:
        raise ValueError(
            'no cross terms: these modal densities are autospectra only, '
            'and the expansion needs the matrix between the modes')
    names = modal_dofs(shapes)
    blocks: dict[str, dict[tuple[int, int], int]] = {}
    for i in _picked(data, records):
        qi, qj = (_physical_quantity(q)
                  for q in channel_quantities(data.known_dim(i)))
        if qi is None or qj is None or qi in FORCES or qj in FORCES:
            dim = data.known_dim(i)
            report.skipped[dim] = report.skipped.get(dim, 0) + 1
            continue
        if qi != qj:
            report.notes.append('a cross-quantity term left out')
            continue
        k = modal_coordinate(data.response_dof[i])
        l = modal_coordinate(data.reference_dof[i])
        if (k is None or l is None or k > shapes.num_shapes
                or l > shapes.num_shapes):
            report.dropped.append(data.response_dof[i])
            continue
        blocks.setdefault(qi, {})[(k - 1, l - 1)] = i
    if not blocks:
        raise ValueError('nothing to expand: no motion density of this '
                         'object is between the shapes\' modal DOFs')
    rows, resp, refs, dims, hints, limits = [], [], [], [], [], {}
    for quantity, entries in blocks.items():
        modes = sorted({k for pair in entries for k in pair})
        block = _hermitian_block(
            {(names[k], names[l]): i for (k, l), i in entries.items()},
            data.ordinate, [names[k] for k in modes], f'{quantity} density')
        report.shared[quantity] = [names[k] for k in modes]
        undefined = any(data.ordinate_dim[i] == UNKNOWN
                        for i in entries.values())
        shape = np.real(shapes.shape_matrix[modes]).T          # dofs × modes
        physical = shape @ block @ shape.T
        autos = [(names[k], entries.get((k, k))) for k in modes]
        autos = [(dof, i) for dof, i in autos if i is not None]
        ratios = _band_ratios(data, autos, data.abscissa, quantity) \
            if autos else {}
        dim = f'{quantity}**2/frequency'
        for a, da in enumerate(shapes.coordinate):
            for b, db in enumerate(shapes.coordinate):
                rows.append(physical[:, a, b])
                resp.append(da)
                refs.append(db)
                dims.append(UNKNOWN if undefined else dim)
                hints.append(dim if undefined else None)
                for name, ratio in ratios.items():
                    limits.setdefault(name, []).append(
                        ratio * np.real(physical[:, a, a]) if a == b
                        else np.full(len(data.abscissa), np.nan))
    return _density_like(data, np.asarray(rows), resp, refs, dims, hints,
                         limits)


def _density_like(data, ordinate, resp, refs, dims, hints, limits):
    """A density of the source's own class — a specification stays a
    specification, with the bands that carried — on a new matrix."""
    from .data import Specification

    cls = type(data)
    kwargs: dict[str, Any] = {
        'response_dof': resp, 'reference_dof': refs, 'ordinate_dim': dims,
        'dimension_hint': hints}
    bandwidth = getattr(data, 'bandwidth', None)
    if bandwidth is not None:
        kwargs['bandwidth'] = bandwidth
    if isinstance(data, Specification):
        kwargs.update({name: np.asarray(values)
                       for name, values in limits.items()})
    result = cls(data.abscissa, ordinate, **kwargs)
    result.interpolation = data.interpolation
    result.scale_db = getattr(data, 'scale_db', None)
    return result


# ---- an FRF: response rows, reference columns ---------------------------------


def _frf_to_modal(data, shapes, records, report):
    from .data import Frf, channel_quantities

    names = modal_dofs(shapes)
    blocks: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    for i in _picked(data, records):
        pair = channel_quantities(data.known_dim(i))
        blocks.setdefault(pair, {})[
            (data.response_dof[i], data.reference_dof[i])] = i
    rows, resp, refs, dims, hints = [], [], [], [], []
    for (qu, qf), entries in blocks.items():
        if _quantity_class(qu) != 'motion':
            report.skipped[f'{qu}/{qf}'] = len(entries)
            continue
        if any(data.ordinate_dim[i] == UNKNOWN for i in entries.values()):
            report.skipped[f'{qu}/{qf} (units undefined)'] = len(entries)
            continue
        responses = list(dict.fromkeys(u for u, _f in entries))
        references = list(dict.fromkeys(f for _u, f in entries))
        kept, phi_u = _columns(shapes, responses, report)
        if not kept:
            continue
        missing = [(u, f) for u in kept for f in references
                   if (u, f) not in entries]
        if missing:
            raise ValueError(
                f'no FRF between {missing[0][0]} and {missing[0][1]}'
                + (f' and {len(missing) - 1} more' if len(missing) > 1
                   else '')
                + ': the responses transform together, so every one '
                'needs the same references')
        matrix = np.stack([[data.ordinate[entries[(u, f)]] for f in references]
                           for u in kept], axis=0)          # nu × nf × lines
        matrix = np.moveaxis(matrix, 2, 0)                  # lines × nu × nf
        report.shared[qu] = kept
        inverse = _fit(phi_u, qu, len(kept), shapes, report)
        modal = inverse @ matrix                            # lines × m × nf
        explained = phi_u @ modal
        for a, dof in enumerate(kept):
            scale = _rms(matrix[:, a, :])
            report.residual[dof] = (
                _rms(matrix[:, a, :] - explained[:, a, :]) / scale
                if scale > 0.0 else 0.0)
        # the references: through the force rule when the set covers
        # every drive and they resolve the modes; kept physical, and
        # said, otherwise
        ref_names, ref_dims, ref_hints = references, [], []
        physical_refs = True
        if _quantity_class(qf) == 'force':
            covered, phi_f = _columns(shapes, references, TransformReport())
            if len(covered) == len(references):
                rank = int(np.linalg.matrix_rank(phi_f))
                if rank == shapes.num_shapes:
                    modal = modal @ np.linalg.pinv(phi_f.T)  # lines × m × m
                    physical_refs = False
                    report.shared[qf] = list(references)
                else:
                    report.notes.append(
                        f'references kept physical: the {len(references)} '
                        f'drives resolve only {rank} of {shapes.num_shapes} '
                        'modes')
            else:
                outside = [f for f in references if f not in covered]
                report.notes.append(
                    'references kept physical: '
                    + ', '.join(outside[:3])
                    + (f' and {len(outside) - 3} more' if len(outside) > 3
                       else '') + ' not in the shapes')
        else:
            report.notes.append(f'references kept physical: {qf} is not '
                                'a force')
        if physical_refs:
            ref_dims = [qf] * len(references)
            ref_hints = [None] * len(references)
        else:
            ref_names = names
            per_mode = [_modal_dim(shapes, qf, l)
                        for l in range(shapes.num_shapes)]
            ref_dims = [dim for dim, _hint in per_mode]
            ref_hints = [hint for _dim, hint in per_mode]
        per_mode = [_modal_dim(shapes, qu, k) for k in range(shapes.num_shapes)]
        for k in range(shapes.num_shapes):
            dk, hk = per_mode[k]
            for j, ref in enumerate(ref_names):
                df, hf = ref_dims[j], ref_hints[j]
                rows.append(modal[:, k, j])
                resp.append(names[k])
                refs.append(ref)
                if dk == UNKNOWN or df == UNKNOWN:
                    dims.append(UNKNOWN)
                    hints.append(f'{hk or dk}/{hf or df}')
                else:
                    dims.append(f'{dk}/{df}')
                    hints.append(None)
    if not rows:
        raise ValueError(
            'nothing to transform: no motion response of this FRF is on '
            'a DOF the shapes cover'
            + (f' (not transformed: {report.describe()})'
               if report.skipped or report.dropped else ''))
    return Frf(data.abscissa, np.asarray(rows), response_dof=resp,
               reference_dof=refs, ordinate_dim=dims, dimension_hint=hints)


def _frf_to_physical(data, shapes, records, report):
    from .data import Frf, channel_quantities

    names = modal_dofs(shapes)
    blocks: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
    for i in _picked(data, records):
        qu, qf = channel_quantities(data.known_dim(i))
        pu = _physical_quantity(qu)
        if pu is None or pu in FORCES:
            dim = data.known_dim(i)
            report.skipped[dim] = report.skipped.get(dim, 0) + 1
            continue
        pf = _physical_quantity(qf) or qf
        blocks.setdefault((pu, pf), {})[
            (data.response_dof[i], data.reference_dof[i])] = i
    if not blocks:
        raise ValueError('nothing to expand: no motion response of this '
                         'FRF is at one of the shapes\' modal DOFs')
    rows, resp, refs, dims, hints = [], [], [], [], []
    for (pu, pf), entries in blocks.items():
        modes = sorted({modal_coordinate(u) - 1 for u, _f in entries
                        if _is_modal(u, shapes)})
        if not modes:
            report.dropped.extend(dict.fromkeys(u for u, _f in entries))
            continue
        references = list(dict.fromkeys(f for _u, f in entries))
        modal_refs = all(_is_modal(f, shapes) for f in references)
        missing = [(names[k], f) for k in modes for f in references
                   if (names[k], f) not in entries]
        if missing:
            raise ValueError(
                f'no FRF between {missing[0][0]} and {missing[0][1]}: the '
                'modes expand together, so every one needs the same '
                'references')
        matrix = np.stack([[data.ordinate[entries[(names[k], f)]]
                            for f in references] for k in modes], axis=0)
        matrix = np.moveaxis(matrix, 2, 0)                  # lines × m × nf
        report.shared[pu] = [names[k] for k in modes]
        undefined = any(data.ordinate_dim[i] == UNKNOWN
                        for i in entries.values())
        shape = np.real(shapes.shape_matrix[modes]).T          # dofs × modes
        physical = shape @ matrix                              # lines × nu × nf
        if modal_refs:
            ref_modes = [modal_coordinate(f) - 1 for f in references]
            physical = physical @ np.real(shapes.shape_matrix[ref_modes])
            ref_names = list(shapes.coordinate)
        else:
            ref_names = references
        dim = f'{pu}/{pf}'
        for a, da in enumerate(shapes.coordinate):
            for j, ref in enumerate(ref_names):
                rows.append(physical[:, a, j])
                resp.append(da)
                refs.append(ref)
                dims.append(UNKNOWN if undefined else dim)
                hints.append(dim if undefined else None)
    return Frf(data.abscissa, np.asarray(rows), response_dof=resp,
               reference_dof=refs, ordinate_dim=dims, dimension_hint=hints)


# ---- the two verbs' cores -------------------------------------------------------


def to_modal(data: DataArray, shapes: ShapeSet,
             records: Sequence[int] | None = None
             ) -> tuple[DataArray, TransformReport]:
    """Transform physical responses through the set into modal
    responses at the set's modal DOFs `M1` … `Mn`.

    The result takes the form the data's kind needs: rows for a time
    history or spectrum, the matrix for a density or specification, and
    both halves for an FRF. A time history's averaging frames and shock
    windows are carried onto the result.

    Parameters
    ----------
    data : DataArray
        The physical record. Motions are fitted, forces projected,
        anything else left out and reported. A density needs every
        cross term between the shared channels; an SRS, a coherence or
        a sine level does not transform at all.
    shapes : ShapeSet
        The set to transform through. Complex sets are refused: real
        data cannot carry a complex modal coordinate.
    records : sequence of int, optional
        Which records to carry through, the channels picked in the
        tree. All of them when omitted.

    Returns
    -------
    tuple of (DataArray, TransformReport)
        The modal responses, and what was shared, dropped and left
        unexplained. A density or FRF comes back as the source's own
        class; a time history or spectrum as a `TimeHistory` or
        `Spectrum`.

    Raises
    ------
    ValueError
        When the kind of data does not transform, the set is complex or
        has no modes, the shared DOFs of a motion quantity resolve fewer
        modes than the set has, cross terms or FRF pairs are missing,
        the bands differ between channels, or no motion or force record
        is on a DOF the shapes cover.
    """
    from .data import Frf, Psd, TimeHistory

    refusal = _cannot(data)
    if refusal is not None:
        raise ValueError(refusal)
    _check_set(shapes)
    report = TransformReport()
    if isinstance(data, Psd):
        result = _matrix_to_modal(data, shapes, records, report)
    elif isinstance(data, Frf):
        result = _frf_to_modal(data, shapes, records, report)
    else:
        result = _rows_to_modal(data, shapes, records, report)
    if isinstance(data, TimeHistory):
        carrying_marks(data, result)
    return result, report


def to_physical(data: DataArray, shapes: ShapeSet,
                records: Sequence[int] | None = None
                ) -> tuple[DataArray, TransformReport]:
    """Expand modal responses back through the set, `u = Φq`, at every
    DOF the set covers.

    The result is summed over the modes present, in the form the
    data's kind takes. With every mode it is the motion of the
    structure; with a few (the records picked in the tree) it is those
    modes' contribution to it, as in a modal contribution plot. The
    result's comments say which modes it carries. Modal forces are left
    out, because many force distributions give the same modal force.

    Parameters
    ----------
    data : DataArray
        The modal record, its DOFs the set's modal names.
    shapes : ShapeSet
        The set the record was transformed through.
    records : sequence of int, optional
        Which modal records to expand, the modes picked in the tree.
        All of them when omitted.

    Returns
    -------
    tuple of (DataArray, TransformReport)
        The physical responses, and which modal DOFs were used.

    Raises
    ------
    ValueError
        When the kind of data does not transform, the set is complex or
        has no modes, the record's DOFs are not this set's modal
        coordinates, cross terms or FRF pairs are missing, the bands
        differ between modal channels, or no motion record is at one of
        the set's modal DOFs.
    """
    from .data import Frf, Psd, TimeHistory

    refusal = _cannot(data)
    if refusal is not None:
        raise ValueError(refusal)
    _check_set(shapes)
    names = modal_dofs(shapes)
    if reads_as(data, shapes) != 'modal':
        raise ValueError('the record\'s DOFs are not modal coordinates '
                         f'of this set ({names[0]} … {names[-1]})')
    report = TransformReport()
    if isinstance(data, Psd):
        result = _matrix_to_physical(data, shapes, records, report)
    elif isinstance(data, Frf):
        result = _frf_to_physical(data, shapes, records, report)
    else:
        result = _rows_to_physical(data, shapes, records, report)
    if isinstance(data, TimeHistory):
        carrying_marks(data, result)
    return result, report


#: what the Shape Values table shows of a set at a choice of DOFs: the
#: shapes, the response transformation or the force transformation
FORMS = ('shapes', 'response', 'force')


@dataclass
class MatrixAt:
    """A shape set's matrix at a choice of DOFs, in one of `FORMS`.

    Attributes
    ----------
    form : str
        'shapes', 'response' or 'force'.
    rows, columns : list of str
        The labels: DOFs down and modes across for the shapes, modes
        down and DOFs across for the two transformations.
    values : numpy.ndarray
        `(rows, columns)`, NaN where a DOF is not in the set.
    missing : list of str
        The DOFs asked for that the set has no column for.
    """

    form: str
    rows: list[str]
    columns: list[str]
    values: np.ndarray
    missing: list[str] = field(default_factory=list)


def matrix_at(shapes: ShapeSet, dofs: Sequence[str] | None = None,
              form: str = 'shapes') -> MatrixAt:
    """A shape set's matrix at `dofs`, as the transform uses it.

    'shapes' is `Φ` at the DOFs, a row per DOF and a column per mode:
    how each modal coordinate moves each DOF, `u = Φq`, each value as
    stored, complex where the set is. 'response' is `Φ⁺`, a row per mode
    and a column per DOF, what `to_modal` applies to motions,
    `q = Φ⁺u`. 'force' is `Φᵀ`, what it applies to forces, `f_m = Φᵀf`.
    The two transformations are over the real shapes, as the transform
    takes them, and the pseudo-inverse is refused when the DOFs resolve
    fewer modes than the set has.

    A DOF is matched to the set's column as the transform matches a
    channel, its sign honored: `2X-` is the negative of the set's `2X+`.
    A DOF the set has no column for is kept, its row or column NaN, and
    listed in `missing`. With `dofs` left out, every DOF of the set.

    Parameters
    ----------
    shapes : ShapeSet
        The set.
    dofs : sequence of str, optional
        The DOFs, in the order they are wanted.
    form : str, default 'shapes'
        One of `FORMS`.

    Returns
    -------
    MatrixAt
        The labels, the values and the DOFs not in the set.

    Raises
    ------
    ValueError
        For a form not in `FORMS`, a transformation at DOFs none of
        which is in the set, or a response transformation the DOFs are
        too few to resolve.
    """
    # Brandon, 2026-10-23: the kinematic transformation at a channel
    # table's references, responses or control channels, beside the
    # shapes themselves, read the way the transform reads them.
    if form not in FORMS:
        raise ValueError(f'{form!r} is not a form; use one of {FORMS}')
    names = modal_dofs(shapes)
    if dofs is None:
        if form == 'shapes':
            return MatrixAt(form, list(shapes.coordinate), names,
                            np.asarray(shapes.shape_matrix).T)
        dofs = list(shapes.coordinate)
    dofs = [str(dof) for dof in dofs]
    report = TransformReport()
    if form == 'shapes':
        lookup = shapes._dof_lookup()
        matrix = np.asarray(shapes.shape_matrix)
        values = np.full((len(dofs), shapes.num_shapes), np.nan,
                         dtype=np.result_type(matrix.dtype, np.float64))
        missing = []
        for row, dof in enumerate(dofs):
            base, sign = _split_sign(dof)
            found = lookup.get(base)
            if found is None:
                missing.append(dof)
                continue
            column, column_sign = found
            values[row] = matrix[:, column] * sign * column_sign
        return MatrixAt(form, dofs, names, values, missing)
    kept, phi = _columns(shapes, dofs, report)
    if not kept:
        raise ValueError('none of these DOFs is in the set')
    block = (_fit(phi, 'motion', len(kept), shapes, report)
             if form == 'response' else phi.T)
    values = np.full((shapes.num_shapes, len(dofs)), np.nan)
    at = {dof: k for k, dof in enumerate(kept)}
    for column, dof in enumerate(dofs):
        if dof in at:
            values[:, column] = block[:, at[dof]]
    return MatrixAt(form, names, dofs, values, list(report.dropped))


def carried_modes(report: TransformReport, shapes: ShapeSet) -> list[str]:
    """The modal DOFs an expansion carried, in mode order, or an empty
    list when it carried every mode, so a name lists the modes only
    when that matters."""
    used = sorted({dof for dofs in report.shared.values() for dof in dofs
                   if modal_coordinate(dof) is not None},
                  key=lambda dof: modal_coordinate(dof) or 0)
    return [] if len(used) == shapes.num_shapes else used
