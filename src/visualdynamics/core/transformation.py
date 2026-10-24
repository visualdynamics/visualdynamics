"""A transformation between control channels and degrees of freedom, as
an object of its own: a matrix with named inputs and outputs and the
units its entries carry.

A six-DOF test controls the motion of a point, and measures it with
accelerometers spread over the article. The matrix that turns the
measured channels into the point's three translations and three
rotations is the most consequential setting in such a test, and it is
usually typed into a controller once and never read again. A
`Transformation` holds it so it can be read: which channel goes in
where, what comes out, in what unit, how well the inputs resolve the
outputs, and how much of the measured motion the outputs leave
unexplained.

The matrix `T` maps inputs to outputs, `y = T u`. Built from geometry it
is the pseudo-inverse of the six rigid-body shapes at the input DOFs,
`T = Φ⁺`, the same matrix `transform.to_modal` applies through a rigid
set (`transform.matrix_at`'s response form); typed, it is whatever the
user states, with its inputs and outputs named. Either way it is
applied the same way, to a time history or spectrum row by row,
`y = T u`, and to a cross-spectral density or a specification as the
quadratic form `G_yy = T G_uu Tᵀ`, which needs every cross term between
the inputs: the phase between two channels is what tells a translation
from a rotation.

Every input and output is a translation or a rotation. An entry's unit
follows from the two it joins: `[T_ij] = length^(r_j − r_i)`, with `r`
one for a rotation and zero for a translation, so a translation from
translations is a plain number, a rotation from translations is per
meter, and a translation from a rotation is in meters. Stored values
are SI like every other value in the package. An output rotation comes
out in the input's angular quantity: acceleration in, angular
acceleration out, in rad/s².

The residual is what the outputs cannot carry back. `Q = T⁺T` is the
orthogonal projector onto the motions the outputs describe, and
`P = I − Q` onto the rest. For a rigid-body transformation, `P u` is the
part of the measured motion that is not rigid: a channel's local
flexing, a mode of the article, a loose mount or a channel wired to the
wrong DOF. For a density the measured power splits exactly in two,
`trace G = trace(Q G Q) + trace(P G P)`, because `P Q = 0`, so the
residual is a fraction of the measured power at every line, and
`read_residual` says up to what frequency that fraction stays under a
stated threshold: where the rigid-body assumption holds.

The conditioning is the ratio of the largest to the smallest singular
value. Entries in mixed units make that ratio depend on the length
unit, so it is computed with every rotation scaled by a characteristic
length, `length_scale`: the RMS lever arm of the inputs about the point
for a transformation built from geometry, one of the stated length unit
for a typed one. A pseudo-inverse whose conditioning is past a stated
limit is refused, since an error of one part in a hundred at the inputs
can then reach the outputs a hundred times larger.

A force does not transform by this matrix. A force is work-conjugate
to a motion, so a controller's drives transform by the transpose of the
shapes (`transform.matrix_at(form='reference')`), and the transform
verbs apply that rule by quantity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..units import UNKNOWN
from .shapes import _split_sign
from .transform import (
    MOTIONS,
    ROTATIONAL,
    MatrixAt,
    TransformReport,
    _band_ratios,
    _columns,
    _density_dims,
    _density_like,
    _hermitian_block,
    _picked,
    _rms,
    modal_dofs,
)

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import DataArray
    from .geometry import Geometry
    from .shapes import ShapeSet

#: what an input or output of a transformation is
KINDS = ('translation', 'rotation')

#: the conditioning past which a pseudo-inverse is refused unless a
#: larger limit is asked for: an error of one percent at the inputs may
#: then reach the outputs as one hundred percent
MAX_CONDITION = 100.0

#: the share of the measured power the residual may reach at a line and
#: the rigid-body assumption still be said to hold there: a tenth, ten
#: decibels down
RESIDUAL_THRESHOLD = 0.1

#: an entry of a row at or under this share of the row's largest is
#: written in a table as zero: it is the pseudo-inverse's round-off
ROUND_OFF = 1e-12

#: the angular quantity back to the motion it is the rotation of
_ANGULAR = {ROTATIONAL[quantity]: quantity for quantity in MOTIONS}

# The transformation as an object nobody has to take on trust, built
# from geometry or typed, with its conditioning, its weights and its
# residual against frequency.


def _kind_of(dof: str) -> str | None:
    """A node DOF's kind from its direction ('101RX+' is a rotation),
    or None for a modal coordinate, which says nothing of its own."""
    from .data import parse_dof

    node, direction = parse_dof(dof)
    if node is None:
        return None
    return 'rotation' if direction.startswith('R') else 'translation'


def _kinds(names: Sequence[str], kinds: Sequence[str] | None,
           what: str) -> tuple[str, ...]:
    """The kinds of a set of inputs or outputs: as given, or read from
    each node DOF's direction. A modal coordinate given no kind is
    refused, since `M4` may be a translation or a rotation."""
    if kinds is not None:
        kinds = tuple(str(kind) for kind in kinds)
        if len(kinds) != len(names):
            raise ValueError(f'{len(kinds)} {what} kinds for {len(names)} '
                             f'{what}s')
        bad = [kind for kind in kinds if kind not in KINDS]
        if bad:
            raise ValueError(f'{bad[0]!r} is not a kind; use one of {KINDS}')
        return kinds
    read = [_kind_of(name) for name in names]
    unknown = [name for name, kind in zip(names, read) if kind is None]
    if unknown:
        raise ValueError(
            f'{unknown[0]} says nothing of whether it is a translation or '
            f'a rotation: state the {what} kinds')
    return tuple(read)                                # type: ignore[arg-type]


def _scaled(matrix: np.ndarray, input_kinds: Sequence[str],
            output_kinds: Sequence[str], length: float) -> np.ndarray:
    """The matrix with every entry made a plain number: rotation outputs
    times the length, rotation inputs over it."""
    rows = np.where(np.asarray(output_kinds) == 'rotation', length, 1.0)
    columns = np.where(np.asarray(input_kinds) == 'rotation', 1.0 / length,
                       1.0)
    return matrix * rows[:, np.newaxis] * columns[np.newaxis, :]


def _singular(scaled: np.ndarray) -> np.ndarray:
    return np.linalg.svd(scaled, compute_uv=False)


def _condition(values: np.ndarray) -> float:
    """Largest over smallest of a matrix's singular values, infinite
    when the smallest is nothing next to the largest."""
    if not len(values) or values[0] == 0.0:
        return float('inf')
    smallest = values[-1]
    if smallest <= values[0] * len(values) * np.finfo(np.float64).eps:
        return float('inf')
    return float(values[0] / smallest)


def _checked_pinv(matrix: np.ndarray, input_kinds: Sequence[str],
                  output_kinds: Sequence[str], length: float,
                  max_condition: float, needed: int,
                  what: str) -> np.ndarray:
    """The pseudo-inverse of `matrix`, refused when its rank is short of
    what the inverse must resolve or its conditioning is past
    `max_condition`. `needed` is the rank the inverse must have: the
    six motions of a point for a set of shapes, every output for the
    inverse of a transformation. One check for a transformation built from shapes
    and for the inverse of any transformation, so the two cannot differ
    in what they let through."""
    scaled = _scaled(matrix, input_kinds, output_kinds, length)
    values = _singular(scaled)
    rank = int(np.sum(values > values[0] * max(matrix.shape)
                      * np.finfo(np.float64).eps)) if values.size and \
        values[0] > 0.0 else 0
    if rank < needed:
        raise ValueError(
            f'{what}: the matrix has rank {rank} of {needed}, so '
            f'{needed - rank} direction{"s" * (needed - rank != 1)} cannot '
            'be resolved — measure more directions, or farther from the '
            'point')
    condition = _condition(values)
    if not condition <= max_condition:
        raise ValueError(
            f'{what}: the conditioning is {condition:.4g}, past the limit '
            f'of {max_condition:.4g} (the matrix inverted has singular '
            f'values {values[0]:.4g} to '
            f'{values[-1]:.4g}) — an error at the inputs reaches the '
            'outputs that many times larger. Spread the channels, or '
            'raise max_condition knowingly')
    return np.linalg.pinv(matrix)


@dataclass(frozen=True, eq=False)
class Transformation:
    """A matrix from named inputs to named outputs, `y = T u`, with the
    kind of each so its entries' units are known.

    Made with `from_geometry` (the rigid-body transformation to a point),
    `from_shapes` (through any rigid set at chosen DOFs) or `typed` (a
    matrix stated by hand). Checked when it is created: the matrix is
    real and finite, its shape is the outputs by the inputs, the names
    are DOF strings with none repeated, and the length scale is a
    positive number of meters.

    Attributes
    ----------
    matrix : numpy.ndarray
        `(outputs, inputs)`, SI: a rotation row from translations per
        meter, a translation row from a rotation in meters.
    inputs, outputs : tuple of str
        The DOF strings, in the matrix's order. An input's sign is
        honored when a record is matched to it: a `101Z-` record at a
        `101Z+` input is negated.
    input_kinds, output_kinds : tuple of str
        'translation' or 'rotation' for each (`KINDS`).
    description : tuple of str
        A line per output saying what it is ('Rigid rotation about Y'),
        or empty.
    length_scale : float
        The length, in meters, the rotations are scaled by for the
        conditioning (`condition`).
    origin : str
        How the transformation was made, in a line.
    """

    matrix: np.ndarray
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    input_kinds: tuple[str, ...]
    output_kinds: tuple[str, ...]
    description: tuple[str, ...] = ()
    length_scale: float = 1.0
    origin: str = ''

    def __post_init__(self) -> None:
        from .validate import dofs

        matrix = np.asarray(self.matrix)
        if np.iscomplexobj(matrix):
            raise ValueError('a transformation is real: a complex entry '
                             'would turn a motion into a phase shift')
        matrix = np.array(matrix, dtype=np.float64)
        if matrix.ndim != 2 or not matrix.size:
            raise ValueError('a transformation is a matrix of outputs by '
                             'inputs, with at least one of each')
        if not np.all(np.isfinite(matrix)):
            raise ValueError('a transformation entry is not a finite number')
        inputs = tuple(dofs(list(self.inputs), 'input'))
        outputs = tuple(dofs(list(self.outputs), 'output'))
        if matrix.shape != (len(outputs), len(inputs)):
            raise ValueError(
                f'the matrix is {matrix.shape[0]} by {matrix.shape[1]}, but '
                f'there are {len(outputs)} outputs and {len(inputs)} inputs')
        for names, what in ((inputs, 'input'), (outputs, 'output')):
            bases = [_split_sign(name)[0] for name in names]
            repeated = sorted({base for base in bases if bases.count(base) > 1})
            if repeated:
                raise ValueError(f'the {what} {repeated[0]} is named twice')
        for kinds, names, what in ((self.input_kinds, inputs, 'input'),
                                   (self.output_kinds, outputs, 'output')):
            if len(kinds) != len(names) or any(k not in KINDS for k in kinds):
                raise ValueError(f'each {what} is one of {KINDS}')
        length = float(self.length_scale)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError('the length scale is a positive number of '
                             'meters')
        description = tuple(str(line) for line in self.description)
        if description and len(description) != len(outputs):
            raise ValueError(f'{len(description)} descriptions for '
                             f'{len(outputs)} outputs')
        matrix.setflags(write=False)
        object.__setattr__(self, 'matrix', matrix)
        object.__setattr__(self, 'inputs', inputs)
        object.__setattr__(self, 'outputs', outputs)
        object.__setattr__(self, 'input_kinds', tuple(self.input_kinds))
        object.__setattr__(self, 'output_kinds', tuple(self.output_kinds))
        object.__setattr__(self, 'description',
                           description or ('',) * len(outputs))
        object.__setattr__(self, 'length_scale', length)

    # ---- made three ways ------------------------------------------------

    @classmethod
    def from_shapes(cls, shapes: ShapeSet, inputs: Sequence[str], *,
                    max_condition: float = MAX_CONDITION) -> Transformation:
        """The transformation from motions at `inputs` to the coordinates
        of a rigid set, `T = Φ⁺`, the matrix `transform.to_modal` applies
        through the same set.

        Parameters
        ----------
        shapes : ShapeSet
            Six unscaled rigid-body shapes (`rigid.rigid_body_shapes`
            about a point, with no mass properties). A mass-normalized
            set is refused: its coordinates carry a half power of mass
            and are not the motions of the point.
        inputs : sequence of str
            The input DOFs, in the order wanted. A DOF the set has no
            column for is refused by name.
        max_condition : float, default `MAX_CONDITION`
            The conditioning past which the pseudo-inverse is refused.

        Returns
        -------
        Transformation
            Outputs `M1` … `M6`, three translations then three
            rotations, described as the set describes them, with the
            RMS lever arm of the inputs as the length scale.

        Raises
        ------
        ValueError
            When the set is not unscaled rigid shapes, an input is not in
            the set, the inputs resolve fewer than six motions, or the
            conditioning is past the limit.
        """
        from .rigid import is_rigid_set

        if not is_rigid_set(shapes) or not shapes.unscaled:
            raise ValueError(
                'a transformation to the motions of a point is built from '
                'six unscaled rigid-body shapes; this set is '
                + ('mass-normalized, so its coordinates are not motions'
                   if is_rigid_set(shapes) else 'not a rigid set'))
        names = [str(dof) for dof in inputs]
        if not names:
            raise ValueError('no inputs: a transformation needs channels '
                             'to transform')
        report = TransformReport()
        kept, phi = _columns(shapes, names, report)
        if report.dropped:
            raise ValueError(
                f'{report.dropped[0]} is not a DOF of the set'
                + (f' (nor {len(report.dropped) - 1} more)'
                   if len(report.dropped) > 1 else '')
                + ': the shapes say nothing of how it moves')
        output_kinds = ('translation',) * 3 + ('rotation',) * 3
        input_kinds = ('translation',) * len(kept)
        # the rotation columns of Φ are each input's lever arm about an
        # axis through the point; their RMS over the inputs is the
        # length that makes a rotation and a translation comparable
        arms = phi[:, 3:]
        length = float(np.sqrt(np.mean(np.sum(arms ** 2, axis=1))))
        if not length > 0.0:
            raise ValueError('every input is on the point itself, so no '
                             'rotation can be resolved')
        # Φ is (inputs, modes): its columns are the outputs' kinds and
        # its rows the inputs'
        inverse = _checked_pinv(phi, output_kinds, input_kinds, length,
                                max_condition, phi.shape[1],
                                f'the {len(kept)} inputs')
        return cls(inverse, tuple(kept), tuple(modal_dofs(shapes)),
                   input_kinds, output_kinds,
                   description=tuple(shapes.description),
                   length_scale=length,
                   origin=f'rigid body through {len(kept)} channels')

    @classmethod
    def from_geometry(cls, geometry: Geometry,
                      point: Sequence[float], inputs: Sequence[str], *,
                      max_condition: float = MAX_CONDITION
                      ) -> Transformation:
        """The rigid-body transformation from motions at `inputs` to the
        three translations and three rotations of `point`.

        The six unit rigid-body shapes about the point
        (`rigid.rigid_body_shapes`) at the inputs, pseudo-inverted
        (`from_shapes`). The geometry's length unit must be declared:
        the rotation rows are per meter, and coordinates with no unit
        would make them per nothing anyone could name.

        Parameters
        ----------
        geometry : Geometry
            The nodes the inputs are on, in declared units.
        point : sequence of float
            The point, three coordinates in meters.
        inputs : sequence of str
            The input DOFs, in the order wanted.
        max_condition : float, default `MAX_CONDITION`
            The conditioning past which the pseudo-inverse is refused.

        Returns
        -------
        Transformation
            As `from_shapes` makes it.

        Raises
        ------
        ValueError
            When the geometry's length unit is not declared, or as
            `from_shapes` refuses.
        """
        from .rigid import MassProperties, rigid_body_shapes

        if not geometry.units_defined:
            raise ValueError(
                'the geometry\'s length unit is not declared, so a rotation '
                'row would be per an unknown length — declare it with '
                'define_units first')
        properties = MassProperties(point=point)
        shapes = rigid_body_shapes(geometry, properties)
        made = cls.from_shapes(shapes, inputs, max_condition=max_condition)
        where = ', '.join(f'{v:.4g}' for v in properties.point)
        return replace(made, origin=f'rigid body about ({where}) m through '
                                    f'{len(made.inputs)} channels')

    @classmethod
    def typed(cls, matrix: Any, inputs: Sequence[str],
              outputs: Sequence[str], *,
              input_kinds: Sequence[str] | None = None,
              output_kinds: Sequence[str] | None = None,
              length_unit: str = 'm',
              description: Sequence[str] = ()) -> Transformation:
        """A transformation stated by hand: a matrix and the names and
        kinds of what goes in and what comes out.

        The entries are read in `length_unit`, as a controller's matrix
        is usually typed: a rotation row from translations per that
        unit, a translation row from a rotation in it. They are stored
        in SI. The length scale for the conditioning is one of that
        unit.

        Parameters
        ----------
        matrix : array-like
            `(outputs, inputs)`, real.
        inputs, outputs : sequence of str
            The DOF strings. A virtual point is a node of its own
            (`'1000X+'` … `'1000RZ+'`), or the modal coordinates
            `M1` … `Mn`.
        input_kinds, output_kinds : sequence of str, optional
            'translation' or 'rotation' for each. Read from each DOF's
            direction when left out (`'1000RX+'` is a rotation); a modal
            coordinate needs its kind stated.
        length_unit : str, default 'm'
            The length the entries are read in.
        description : sequence of str, optional
            A line per output.

        Returns
        -------
        Transformation

        Raises
        ------
        ValueError
            When the matrix, the names or the kinds do not agree, an
            entry is not a finite real number, or the unit is not a
            length.
        """
        from ..units import si_transform

        matrix = np.asarray(matrix)
        if np.iscomplexobj(matrix):
            raise ValueError('a transformation is real: a complex entry '
                             'would turn a motion into a phase shift')
        matrix = np.array(matrix, dtype=np.float64)
        inputs = [str(dof) for dof in inputs]
        outputs = [str(dof) for dof in outputs]
        ins = _kinds(inputs, input_kinds, 'input')
        outs = _kinds(outputs, output_kinds, 'output')
        try:
            meters, _offset = si_transform(length_unit, 'length')
        except Exception as error:
            raise ValueError(f'{length_unit!r} is not a length unit') \
                from error
        if matrix.ndim == 2 and matrix.shape == (len(outs), len(ins)):
            # a rotation row from a translation is per length unit, so
            # to per meter it divides by the unit's meters; a
            # translation row from a rotation multiplies
            power = (np.asarray([kind == 'rotation' for kind in ins], float)
                     [np.newaxis, :]
                     - np.asarray([kind == 'rotation' for kind in outs],
                                  float)[:, np.newaxis])
            matrix = matrix * float(meters) ** power
        return cls(matrix, tuple(inputs), tuple(outputs), ins, outs,
                   description=tuple(description),
                   length_scale=float(meters),
                   origin=f'typed, entries in {length_unit}')

    # ---- what it is -------------------------------------------------------

    @property
    def scaled(self) -> np.ndarray:
        """The matrix with every entry a plain number: rotation rows
        times `length_scale`, rotation columns over it. What the
        singular values and the conditioning are read from."""
        return _scaled(self.matrix, self.input_kinds, self.output_kinds,
                       self.length_scale)

    @property
    def singular_values(self) -> np.ndarray:
        """The singular values of `scaled`, largest first."""
        return _singular(self.scaled)

    @property
    def condition(self) -> float:
        """The largest singular value of `scaled` over the smallest:
        how many times larger a relative error at the inputs can reach
        the outputs. Infinite when the matrix is short of full rank."""
        return _condition(self.singular_values)

    def entry_unit(self, row: int, column: int,
                   length_unit: str = 'm') -> str:
        """The unit of one entry, in `length_unit`: '' for a plain
        number, '1/m' or 'm'."""
        power = ((self.input_kinds[column] == 'rotation')
                 - (self.output_kinds[row] == 'rotation'))
        return {0: '', 1: length_unit, -1: f'1/{length_unit}'}[int(power)]

    def output_dimension(self, quantity: str, row: int) -> str:
        """The dimension output `row` takes for an input motion
        `quantity`: the quantity for a translation, its angular quantity
        for a rotation."""
        return (ROTATIONAL[quantity] if self.output_kinds[row] == 'rotation'
                else quantity)

    def matrix_at(self) -> MatrixAt:
        """The matrix as `transform.matrix_at` gives a shape set's, rows
        the outputs and columns the inputs, so its weights draw on a
        geometry as arrows (`viz.geometry.plot_transformation`)."""
        return MatrixAt('response', list(self.outputs), list(self.inputs),
                        np.array(self.matrix))

    def table(self, length_unit: str = 'm'
              ) -> tuple[list[str], list[list[str]]]:
        """The matrix as text for a controller: a row per output, a
        column per input, each entry in `length_unit` with the row's
        unit named in the last column.

        Parameters
        ----------
        length_unit : str, default 'm'
            The length a rotation row's entries are written per.

        Returns
        -------
        tuple of (list of str, list of list of str)
            The column headings and the rows.
        """
        from ..units import si_transform

        meters, _offset = si_transform(length_unit, 'length')
        headings = ['Output', *self.inputs, 'Unit']
        rows = []
        for i, name in enumerate(self.outputs):
            values = []
            units = set()
            # a pseudo-inverse leaves round-off where an entry is zero
            # (1e-20 for a sensor that cannot see a rotation); written
            # out it reads as a weight, so it is written as the zero it is
            largest = float(np.max(np.abs(self.matrix[i])))
            for j in range(len(self.inputs)):
                power = ((self.input_kinds[j] == 'rotation')
                         - (self.output_kinds[i] == 'rotation'))
                value = self.matrix[i, j]
                if abs(value) <= ROUND_OFF * largest:
                    value = 0.0
                values.append(f'{value / float(meters) ** power:.6g}')
                units.add(self.entry_unit(i, j, length_unit))
            unit = ' or '.join(sorted(u or '1' for u in units))
            rows.append([name, *values, unit])
        return headings, rows

    def describe(self) -> str:
        """One line: the shape, the conditioning and how it was made."""
        condition = self.condition
        text = (f'{len(self.inputs)} input{"s" * (len(self.inputs) != 1)} '
                f'to {len(self.outputs)} output'
                f'{"s" * (len(self.outputs) != 1)}, conditioning '
                + (f'{condition:.3g}' if np.isfinite(condition)
                   else 'infinite (short of full rank)'))
        return text + (f', {self.origin}' if self.origin else '')

    # ---- inverted ---------------------------------------------------------

    def inverse(self, *, max_condition: float = MAX_CONDITION
                ) -> Transformation:
        """The pseudo-inverse, outputs back to inputs, `u = T⁺ y`.

        For a rigid-body transformation this is the expansion: the six
        motions of the point carried out to the channels, the rigid
        part of what they measure. It is refused when the outputs are
        more than the matrix's rank (they could not all be told apart)
        or the conditioning is past `max_condition`.

        Parameters
        ----------
        max_condition : float, default `MAX_CONDITION`
            The conditioning past which it is refused.

        Returns
        -------
        Transformation
            Inputs and outputs exchanged, with the same length scale.

        Raises
        ------
        ValueError
            When the rank is short or the conditioning is past the limit.
        """
        inverse = _checked_pinv(self.matrix, self.input_kinds,
                                self.output_kinds, self.length_scale,
                                max_condition, len(self.outputs),
                                'the inverse')
        return Transformation(
            inverse, self.outputs, self.inputs, self.output_kinds,
            self.input_kinds, length_scale=self.length_scale,
            origin=f'inverse of: {self.origin}' if self.origin
            else 'inverse')

    @property
    def residual_projector(self) -> np.ndarray:
        """`P = I − T⁺T`, `(inputs, inputs)`: the part of an input
        motion the outputs cannot carry back. Zero for a square,
        invertible transformation, which explains everything it is
        given and so cannot say anything about the residual."""
        return np.eye(len(self.inputs)) - np.linalg.pinv(self.matrix) @ \
            self.matrix

    # ---- applied ----------------------------------------------------------

    def apply(self, data: DataArray, records: Sequence[int] | None = None,
              *, quantity: str | None = None
              ) -> tuple[DataArray, TransformReport]:
        """Apply the transformation to a record: a time history or a
        spectrum row by row, a cross-spectral density or specification
        as the quadratic form.

        Every input must be present as a record of one motion quantity
        in declared units: a translation input as that quantity, a
        rotation input as its angular one. Records at other DOFs are
        left out and listed (the report's `dropped`), and records of
        another quantity at the inputs left out and counted by quantity
        (`skipped`). A specification's bands carry through when
        every input wears the same band, as `transform.to_modal`
        carries them.

        Parameters
        ----------
        data : DataArray
            A `TimeHistory`, `Spectrum`, or a `Psd` with its cross terms
            (a `Specification` stays one).
        records : sequence of int, optional
            Which records to take the inputs from; all when omitted.
        quantity : str, optional
            The motion to transform ('acceleration', 'velocity' or
            'length') when the record holds more than one at the inputs.

        Returns
        -------
        tuple of (DataArray, TransformReport)
            The outputs, of the source's kind, and the report: the
            inputs used, the records left out, and each input's residual
            fraction (of its RMS, or of its power for a density).

        Raises
        ------
        ValueError
            When an input is missing, in undefined units, of the wrong
            quantity or kind, or holds a value that is not a finite
            number, or a density's cross terms are missing.
        TypeError
            When the data is an FRF or another kind that does not
            transform by one matrix.
        """
        from .data import Frf, Psd, Spectrum, TimeHistory

        if isinstance(data, Psd):
            return self._apply_density(data, records, quantity)
        if isinstance(data, Frf):
            raise TypeError(
                'an FRF transforms in both halves, its responses by this '
                'matrix and its references by the force rule — use '
                'transform.to_modal through the shape set')
        if not isinstance(data, (TimeHistory, Spectrum)):
            raise TypeError(f'{type(data).__name__} does not transform '
                             'row by row; transform the time history')
        return self._apply_rows(data, records, quantity)

    def _base(self, data, chosen, quantity):
        """The one motion quantity the inputs are read as, from the
        records at the inputs' DOFs: as asked, or the only one there."""
        bases = {_split_sign(name)[0] for name in self.inputs}
        found = set()
        for i in chosen:
            if _split_sign(data.response_dof[i])[0] not in bases:
                continue
            dim = data.known_dim(i).partition('/')[0].removesuffix('**2')
            dim = dim.partition('*')[0]
            base = _ANGULAR.get(dim, dim)
            if base in MOTIONS:
                found.add(base)
        if quantity is not None:
            if quantity not in MOTIONS:
                raise ValueError(f'{quantity!r} is not a motion; use one of '
                                 f'{MOTIONS}')
            return quantity
        if not found:
            raise ValueError(
                'no motion record is at the transformation\'s inputs ('
                + ', '.join(self.inputs[:3])
                + (' …' if len(self.inputs) > 3 else '') + ')')
        if len(found) > 1:
            raise ValueError(
                'the inputs hold ' + ' and '.join(sorted(found))
                + ': name the quantity to transform')
        return found.pop()

    def _expected(self, base, column):
        return (ROTATIONAL[base] if self.input_kinds[column] == 'rotation'
                else base)

    def _apply_rows(self, data, records, quantity):
        from .data import Spectrum, TimeHistory

        chosen = _picked(data, records)
        base = self._base(data, chosen, quantity)
        report = TransformReport()
        by_base: dict[str, int] = {}
        for i in chosen:
            dim = data.known_dim(i)
            dof = str(data.response_dof[i])
            column = self._column(dof)
            if column is None:
                # a DOF the transformation has no input for
                report.dropped.append(dof)
            elif dim != self._expected(base, column):
                # an input's DOF holding another quantity: left out, and
                # counted by its quantity as the transform counts it
                report.skipped[dim] = report.skipped.get(dim, 0) + 1
            else:
                by_base[_split_sign(dof)[0]] = i
        rows = []
        for column, name in enumerate(self.inputs):
            i = self._record_at(by_base, name, data, chosen, base, column)
            values = np.asarray(data.ordinate[i])
            if not np.all(np.isfinite(values)):
                raise ValueError(f'{data.response_dof[i]} holds a value that '
                                 'is not a finite number; a transformation '
                                 'of it would spread that to every output')
            rows.append(values * self._sign(name, data.response_dof[i]))
        values = np.asarray(rows)
        report.shared[base] = list(self.inputs)
        report.rank[base] = (int(np.linalg.matrix_rank(self.matrix)),
                             len(self.outputs))
        outputs = self.matrix @ values
        unexplained = self.residual_projector @ values
        for name, before, after in zip(self.inputs, values, unexplained):
            scale = _rms(before)
            report.residual[name] = _rms(after) / scale if scale > 0 else 0.0
        dims = [self.output_dimension(base, k)
                for k in range(len(self.outputs))]
        comments = [f'{dim} of {line}' if line else dim
                    for dim, line in zip(dims, self.description)]
        cls = Spectrum if isinstance(data, Spectrum) else TimeHistory
        result = cls(data.abscissa, outputs, response_dof=list(self.outputs),
                     ordinate_dim=dims, comment=comments)
        if isinstance(data, TimeHistory):
            from .filters import carrying_marks

            carrying_marks(data, result)
        return result, report

    def _translational(self):
        return all(kind == 'translation' for kind in self.input_kinds)

    def _column(self, dof):
        base = _split_sign(dof)[0]
        for column, name in enumerate(self.inputs):
            if _split_sign(name)[0] == base:
                return column
        return None

    @staticmethod
    def _sign(name, dof):
        return _split_sign(name)[1] * _split_sign(str(dof))[1]

    def _record_at(self, by_base, name, data, chosen, base, column):
        """The record at input `name`, or a refusal saying why there is
        none: absent, of another quantity, or in undefined units."""
        found = by_base.get(_split_sign(name)[0])
        expected = self._expected(base, column)
        if found is not None:
            if data.ordinate_dim[found] == UNKNOWN:
                raise ValueError(
                    f'{data.response_dof[found]} has no declared units, so '
                    'its values are raw numbers — define its units first')
            return found
        there = [i for i in chosen
                 if _split_sign(str(data.response_dof[i]))[0]
                 == _split_sign(name)[0]]
        if there:
            dims = sorted({data.known_dim(i) for i in there})
            raise ValueError(
                f'{name} holds {", ".join(dims)}, but the transformation '
                f'takes {expected} there ({self.input_kinds[column]})')
        raise ValueError(
            f'no record at {name}: the transformation needs every one of '
            f'its {len(self.inputs)} inputs')

    def _apply_density(self, data, records, quantity):
        from .data import channel_quantities

        if data.reference_dof is None:
            raise ValueError(
                'no cross terms: this density holds autospectra only, and '
                'the transformation needs the matrix — compute the CPSDs '
                'from the time history, or read the specification with its '
                'cross terms')
        chosen = _picked(data, records)
        base = self._base(data, chosen, quantity)
        allowed = {base, ROTATIONAL[base]}
        report = TransformReport()
        entries: dict[tuple[str, str], int] = {}
        stored: dict[str, str] = {}
        for i in chosen:
            row, col = str(data.response_dof[i]), str(data.reference_dof[i])
            qi, qj = channel_quantities(data.known_dim(i))
            ci, cj = self._column(row), self._column(col)
            if ci is None or cj is None:
                for dof in (row, col):
                    if self._column(dof) is None and dof not in report.dropped:
                        report.dropped.append(dof)
                continue
            if (qi not in allowed or qj not in allowed
                    or qi != self._expected(base, ci)
                    or qj != self._expected(base, cj)):
                dim = data.known_dim(i)
                report.skipped[dim] = report.skipped.get(dim, 0) + 1
                continue
            if data.ordinate_dim[i] == UNKNOWN:
                raise ValueError(
                    f'the density between {row} and {col} has no declared '
                    'units — define its units first')
            stored.setdefault(_split_sign(row)[0], row)
            stored.setdefault(_split_sign(col)[0], col)
            entries[(row, col)] = i
        missing = [name for name in self.inputs
                   if _split_sign(name)[0] not in stored]
        if missing:
            raise ValueError(
                f'no {self._expected(base, self.inputs.index(missing[0]))} '
                f'density at {missing[0]}: the transformation needs every '
                f'one of its {len(self.inputs)} inputs')
        dofs = [stored[_split_sign(name)[0]] for name in self.inputs]
        block = _hermitian_block(entries, data.ordinate, dofs,
                                 f'{base} density')
        if not np.all(np.isfinite(block)):
            raise ValueError('a density at the inputs holds a value that is '
                             'not a finite number')
        signs = np.asarray([self._sign(name, dof)
                            for name, dof in zip(self.inputs, dofs)])
        block = block * signs[:, np.newaxis] * signs[np.newaxis, :]
        report.shared[base] = list(self.inputs)
        report.rank[base] = (int(np.linalg.matrix_rank(self.matrix)),
                             len(self.outputs))
        outputs = self.matrix @ block @ self.matrix.T
        projector = self.residual_projector
        unexplained = projector @ block @ projector.T
        for a, name in enumerate(self.inputs):
            power = float(np.sum(np.real(block[:, a, a])))
            short = float(np.sum(np.real(unexplained[:, a, a])))
            report.residual[name] = (max(short, 0.0) / power
                                     if power > 0.0 else 0.0)
        autos = [(name, entries.get((dof, dof)))
                 for name, dof in zip(self.inputs, dofs)]
        autos = [(name, i) for name, i in autos if i is not None]
        ratios = _band_ratios(data, autos, data.abscissa, base) if autos else {}
        dims = [self.output_dimension(base, k)
                for k in range(len(self.outputs))]
        hints = [None] * len(dims)
        rows, resp, refs, out_dims, out_hints, limits = [], [], [], [], [], {}
        for k, name_k in enumerate(self.outputs):
            for l, name_l in enumerate(self.outputs):
                dim, hint = _density_dims(dims, hints, k, l)
                rows.append(outputs[:, k, l])
                resp.append(name_k)
                refs.append(name_l)
                out_dims.append(dim)
                out_hints.append(hint)
                for limit, ratio in ratios.items():
                    limits.setdefault(limit, []).append(
                        ratio * np.real(outputs[:, k, k]) if k == l
                        else np.full(len(data.abscissa), np.nan))
        result = _density_like(data, np.asarray(rows), resp, refs, out_dims,
                               out_hints, limits)
        return result, report

    def read_residual(self, density: DataArray,
                      records: Sequence[int] | None = None, *,
                      threshold: float = RESIDUAL_THRESHOLD,
                      quantity: str | None = None) -> TransformationResidual:
        """What the outputs leave unexplained of a cross-spectral
        density at the inputs, line by line (`read_residual`)."""
        return read_residual(self, density, records, threshold=threshold,
                             quantity=quantity)


@dataclass
class TransformationResidual:
    """The measured power at a transformation's inputs split in two at
    every line: the part the outputs carry back, and the residual.

    Attributes
    ----------
    transformation : Transformation
        What was read.
    frequency : numpy.ndarray
        The lines, Hz.
    channels : list of str
        The inputs, in the transformation's order.
    quantity : str
        The motion the density is of.
    measured, explained, residual : numpy.ndarray
        `(channels, lines)`: each input's autospectrum, the part of it
        the outputs explain, `Q G Q`, and the rest, `P G P`, in SI.
        `measured = explained + residual` channel by channel.
    threshold : float
        The share of the measured power the residual may reach and the
        transformation still be said to hold (`holds_to`).
    notes : list of str
        Anything the numbers should be read with: dead channels, lines
        with no measured power.
    """

    transformation: Transformation
    frequency: np.ndarray
    channels: list[str]
    quantity: str
    measured: np.ndarray
    explained: np.ndarray
    residual: np.ndarray
    threshold: float = RESIDUAL_THRESHOLD
    notes: list[str] = field(default_factory=list)

    @property
    def dimension(self) -> str:
        """The dimension of the spectra: the quantity's density."""
        return f'{self.quantity}**2/frequency'

    @property
    def total(self) -> np.ndarray:
        """The measured power summed over the inputs, at every line."""
        return np.sum(self.measured, axis=0)

    @property
    def fraction(self) -> np.ndarray:
        """The residual's share of the summed measured power at every
        line, 0 to 1, not a number where nothing was measured."""
        total = self.total
        with np.errstate(divide='ignore', invalid='ignore'):
            share = np.sum(self.residual, axis=0) / total
        return np.where(np.isfinite(total) & (total > 0.0),
                        np.clip(share, 0.0, 1.0), np.nan)

    @property
    def overall(self) -> float:
        """The residual's share of the measured power over the whole
        band, or not a number when nothing was measured."""
        total = float(np.sum(self.total))
        return (float(np.sum(self.residual)) / total if total > 0.0
                else float('nan'))

    @property
    def channel_fractions(self) -> np.ndarray:
        """Each input's residual share of its own power over the band,
        not a number for a channel that measured nothing."""
        power = np.sum(self.measured, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            share = np.sum(self.residual, axis=1) / power
        return np.where(power > 0.0, share, np.nan)

    @property
    def worst_channel(self) -> str | None:
        """The input whose own residual share is largest, or None when
        no channel measured anything."""
        shares = self.channel_fractions
        if not np.any(np.isfinite(shares)):
            return None
        return self.channels[int(np.nanargmax(shares))]

    @property
    def holds_to(self) -> float | None:
        """The highest frequency up to which the residual stays at or
        under `threshold` of the measured power at every line, from the
        first line with power. The last line when it never passes the
        threshold; None when it is past it at the first line, or nothing
        was measured."""
        fraction = self.fraction
        defined = np.flatnonzero(np.isfinite(fraction))
        if not len(defined):
            return None
        past = defined[fraction[defined] > self.threshold]
        if not len(past):
            return float(self.frequency[defined[-1]])
        before = defined[defined < past[0]]
        return float(self.frequency[before[-1]]) if len(before) else None

    @property
    def holds_throughout(self) -> bool:
        """Whether the residual never passes the threshold."""
        fraction = self.fraction
        defined = np.isfinite(fraction)
        return bool(defined.any()) and bool(
            np.all(fraction[defined] <= self.threshold))

    def describe(self) -> str:
        """One line, worded as the status line shows it."""
        if not np.isfinite(self.overall):
            return 'nothing measured at the inputs'
        share = f'residual {self.overall:.1%} of the measured power'
        limit = f'{self.threshold:.0%}'
        if self.holds_throughout:
            return f'{share}; under {limit} at every line'
        holds = self.holds_to
        if holds is None:
            return f'{share}; past {limit} from the first line'
        return f'{share}; under {limit} up to {holds:.4g} Hz'


def read_residual(transformation: Transformation, density: DataArray,
                  records: Sequence[int] | None = None, *,
                  threshold: float = RESIDUAL_THRESHOLD,
                  quantity: str | None = None) -> TransformationResidual:
    """Split the measured power at a transformation's inputs into the
    part its outputs explain and the residual, at every line.

    With `Q = T⁺T` and `P = I − Q`, the explained part of the
    cross-spectral matrix at the inputs is `Q G Q` and the residual
    `P G P`; on the diagonal they sum to the measured autospectra
    because `P Q = 0`. For a rigid-body transformation the residual is
    the motion that is not rigid, and `holds_to` says up to what
    frequency it stays under `threshold` of the measured power.

    Parameters
    ----------
    transformation : Transformation
        Its inputs all translations, so their powers can be summed.
    density : DataArray
        A `Psd` with its cross terms at every input.
    records : sequence of int, optional
        Which records to read; all when omitted.
    threshold : float, default `RESIDUAL_THRESHOLD`
        The share of the measured power, above 0 and under 1.
    quantity : str, optional
        As for `Transformation.apply`.

    Returns
    -------
    TransformationResidual

    Raises
    ------
    ValueError
        When an input is a rotation, the threshold is not between 0 and
        1, the density has no cross terms, or as `Transformation.apply`
        refuses.
    TypeError
        When the density is not a `Psd`.
    """
    from .data import Psd

    if not 0.0 < float(threshold) < 1.0:
        raise ValueError('the threshold is a share of the measured power, '
                         'above 0 and under 1')
    if not transformation._translational():
        raise ValueError('the residual is summed over the inputs, so they '
                         'must all be translations of one quantity')
    if not isinstance(density, Psd):
        raise TypeError(f'a residual against frequency is read from a '
                         f'cross-spectral density, not a '
                         f'{type(density).__name__}')
    if np.linalg.matrix_rank(transformation.matrix) >= len(
            transformation.inputs):
        notes = [('the transformation is square and invertible, so it '
                  'explains everything at its inputs and the residual is '
                  'zero by construction')]
    else:
        notes = []
    expansion = transformation.inverse(max_condition=np.inf)
    # through the matrix both ways: the outputs, then back out to the
    # inputs, `Q G Q`; the measured matrix itself from the same records
    outputs, _report = transformation.apply(density, records,
                                            quantity=quantity)
    explained, _back = expansion.apply(outputs)
    identity = Transformation(np.eye(len(transformation.inputs)),
                              transformation.inputs, transformation.inputs,
                              transformation.input_kinds,
                              transformation.input_kinds)
    measured, _same = identity.apply(density, records, quantity=quantity)
    base = next(iter(_report.shared))

    def diagonal(psd):
        at = {(str(r), str(c)): i for i, (r, c) in enumerate(
            zip(psd.response_dof, psd.reference_dof))}
        return np.real(np.asarray([psd.ordinate[at[(name, name)]]
                                   for name in transformation.inputs]))

    g = diagonal(measured)
    q = diagonal(explained)
    # the residual by difference, which is `P G P` on the diagonal; a
    # few parts in 1e15 under zero is the subtraction
    r = np.maximum(g - q, 0.0)
    dead = [name for name, row in zip(transformation.inputs, g)
            if not np.any(row > 0.0)]
    if dead:
        notes.append(f'{", ".join(dead)} measured nothing, so '
                     f'{"they say" if len(dead) > 1 else "it says"} nothing '
                     'about the residual')
    total = np.sum(g, axis=0)
    empty = int(np.sum(~(total > 0.0)))
    if empty and empty < len(total):
        notes.append(f'{empty} line{"s" * (empty != 1)} with no measured '
                     'power, where the residual is not read')
    return TransformationResidual(
        transformation, np.asarray(density.abscissa, dtype=np.float64),
        list(transformation.inputs), base, g, np.minimum(q, g), r,
        float(threshold), notes)
