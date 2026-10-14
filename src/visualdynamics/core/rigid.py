"""Rigid-body mode shapes of a geometry.

Every free structure has six rigid-body shapes: three translations and
three rotations about a reference point. They are the first step of the
virtual point workflow, where the responses measured at the control
DOFs are transformed through them to the six motions of one point, and
a specification is transformed the same way so the two can be compared
at that point.

The shapes depend only on the geometry and the reference point. A rigid
translation moves every node one unit along an axis, and a rotation of
one radian about an axis `e` through the point `c` moves a node at `r`
by `e × (r − c)`. Mass does not enter. The point is usually the center
of gravity (CG), and `Geometry.suggest_mass_properties` uses the
centroid of the nodes.

The mass and inertia only scale the shapes. `ShapeSet` holds
mass-normalized shapes, `φᵀMφ = I`. A rigid body's 6×6 mass matrix
about its CG is diagonal in its principal axes (the mass three times,
then the three principal moments), so normalizing divides each
translation by `√m` and each rotation by `√I` about its own axis. That
is all the mass properties are used for, and it is why they are given
together or not at all: mass without inertia would scale three shapes
and not the other three, and the resulting modal masses would have no
meaning. Without them the set is `unscaled`, the same as a fit without a
drive point: unit shapes, with the modal mass only a convention.

The rotation axes are the principal axes of the inertia tensor.
Rotations about the global axes are mass-orthogonal only when the
products of inertia are zero. With a coupled tensor the six orthonormal
shapes must rotate about its eigenvectors, so the principal
decomposition is always used. A diagonal tensor's principal axes are
the global axes, so three moments typed in give X, Y and Z back; a
coupled tensor gives its own axes, which are named in the mode
descriptions. Using only the diagonal of a coupled tensor would give
shapes that look mass-normalized and are not.

Mass normalization only has meaning about the CG. About any other point
the rigid mass matrix couples translation and rotation and the six
shapes are not orthogonal, so the mass properties, when given, are
taken to be about the reference point. Unscaled shapes about any point
span the same six-dimensional space, which is what the virtual point
transformation needs: it builds unscaled shapes about the virtual
point, an interface location and not the CG, with this same function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem
    from .geometry import Geometry
    from .shapes import ShapeSet

# The virtual point workflow these shapes start is described in PLAN.md,
# "The virtual point arc".

#: the order the six inertia terms are stated in, and the tensor
#: entries each one fills. `Ixy` is the product of inertia as a tensor
#: entry — the negative of the integral ∫xy dm some handbooks tabulate
#: — so a tensor typed straight from a CAD mass-properties report,
#: which states the tensor itself, needs no sign flipped.
INERTIA_TERMS = ('Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz')
_TENSOR_SLOTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

AXIS_NAMES = ('X', 'Y', 'Z')

def is_rigid_set(shapes: ShapeSet) -> bool:
    """Whether a set is the six rigid-body shapes `rigid_body_shapes` writes.

    True for six modes at exactly 0 Hz whose descriptions start with
    'Rigid translation' (the first three) and 'Rigid rotation' (the last
    three). It is decided from the set's contents, not from a flag, so a
    rigid set that was saved to a file, copied or built in a script is
    still recognized.
    """
    return (shapes.num_shapes == 6
            and bool(np.all(shapes.frequency == 0.0))
            and all(d.startswith('Rigid translation')
                    for d in shapes.description[:3])
            and all(d.startswith('Rigid rotation')
                    for d in shapes.description[3:]))


@dataclass(frozen=True)
class MassProperties:
    """The reference point, and optionally the mass and inertia about it.

    `point` is in meters, `mass` in kilograms and `inertia` in kg·m²,
    since every stored value is SI. The six inertia terms are the
    tensor's `Ixx, Iyy, Izz, Ixy, Ixz, Iyz` (`INERTIA_TERMS`).

    The object is frozen, like `Averaging` and `Truncation`, because it
    is stored on the geometry and its fields are compared to tell when
    shapes made from it are out of date; a value that could change in
    place would make that comparison wrong. It is checked when it is
    created: the point must be three finite numbers, mass and inertia
    must be given together or not at all, the mass must be positive, and
    the tensor must be one a real body could have (symmetric positive
    definite, which a typo in a product of inertia usually breaks; the
    triangle inequality `Ixx + Iyy ≥ Izz` follows from it). Any failure
    raises ValueError.
    """

    point: tuple[float, float, float]
    mass: float | None = None
    inertia: tuple[float, float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        point = tuple(float(v) for v in np.asarray(self.point).ravel())
        if len(point) != 3 or not all(np.isfinite(point)):
            raise ValueError('the reference point is three finite '
                             'coordinates')
        object.__setattr__(self, 'point', point)
        if (self.mass is None) != (self.inertia is None):
            raise ValueError(
                'mass and inertia scale the set together: give both, '
                'for mass-normalized shapes, or neither, for unit shapes')
        if self.mass is None:
            return
        mass = float(self.mass)
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f'mass must be positive, not {self.mass!r}')
        object.__setattr__(self, 'mass', mass)
        inertia = tuple(float(v) for v in np.asarray(self.inertia).ravel())
        if len(inertia) != 6 or not all(np.isfinite(inertia)):
            raise ValueError('inertia is six finite terms: '
                             + ', '.join(INERTIA_TERMS))
        object.__setattr__(self, 'inertia', inertia)
        moments = np.linalg.eigvalsh(self.tensor())
        if moments[0] <= 0.0:
            raise ValueError(
                'the inertia tensor is not positive definite — no body '
                'has it; check the products of inertia')

    @property
    def scaled(self) -> bool:
        """True when mass and inertia are given and the shapes are
        mass-normalized, False for unit shapes."""
        return self.mass is not None

    def tensor(self) -> np.ndarray:
        """The 3×3 inertia tensor, or the identity when no inertia is
        given, which makes the unscaled rotation axes X, Y and Z."""
        tensor = np.eye(3)
        if self.inertia is None:
            return tensor
        for value, (i, j) in zip(self.inertia, _TENSOR_SLOTS):
            tensor[i, j] = tensor[j, i] = value
        return tensor

    def principal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """``(moments, axes)``: the principal moments, and the unit axis
        of each as the rows of `axes`, one per rotation mode.

        They are ordered and signed so that mode 4 rotates about the
        principal axis nearest X, mode 5 nearest Y and mode 6 nearest Z,
        each with a positive component along its global axis. A diagonal
        tensor gives exactly X, Y and Z, and a lightly coupled one gives
        axes that are still close to X, Y and Z. Each rotation shape is
        built from its own axis, so the three axes are not forced to form
        a right-handed set.
        """
        moments, vectors = np.linalg.eigh(self.tensor())
        taken: list[int] = []
        rows, picked = [], []
        for axis in range(3):
            candidates = [k for k in range(3) if k not in taken]
            best = max(candidates, key=lambda k: abs(vectors[axis, k]))
            taken.append(best)
            vector = vectors[:, best]
            if vector[axis] < 0.0:
                vector = -vector
            rows.append(vector)
            picked.append(moments[best])
        return np.asarray(picked), np.asarray(rows)

    def describe(self, unit_system: UnitSystem | None = None, *,
                 defined: bool = True) -> str:
        """The settings as text, for example
        `'about (0.1, 0, 0.05) m, unit shapes'`.

        Used by the status line and by the message that explains why
        derived shapes are out of date. The values are in SI by default,
        or in a unit system's display units when one is given.
        `defined=False` means the geometry's coordinates have no units,
        so the point is labeled 'units undefined' instead of meters.
        """
        import numpy as np

        point = np.asarray(self.point)
        length, mass_unit = 'm', 'kg'
        mass = self.mass
        if not defined:
            length = 'units undefined'
        elif unit_system is not None:
            point = unit_system.from_si(point, 'length')
            length = unit_system.label_text('length')
            mass_unit = unit_system.label_text('mass')
            if mass is not None:
                mass = float(unit_system.from_si(mass, 'mass'))
        coordinates = ', '.join(f'{v:.4g}' for v in point)
        scaling = (f'mass-normalized ({mass:.4g} {mass_unit})' if self.scaled
                   else 'unit shapes')
        return f'about ({coordinates}) {length}, {scaling}'


def axis_name(axis: np.ndarray) -> str:
    """The name of an axis in a rotation mode's description.

    `'X'`, `'Y'` or `'Z'` for a global axis, otherwise the axis's three
    components to two decimals, as in `'(0.71, 0.71, 0.00)'`.
    """
    for k, name in enumerate(AXIS_NAMES):
        unit = np.zeros(3)
        unit[k] = 1.0
        if np.allclose(axis, unit, atol=1e-9):
            return name
    return '(' + ', '.join(f'{v:.2f}' for v in axis) + ')'


def rigid_body_shapes(geometry: Geometry,
                      properties: MassProperties) -> ShapeSet:
    """The six rigid-body mode shapes of a geometry about a point.

    Three translations, then three rotations, over the three
    translational DOFs of every node, with each coefficient in the
    node's own displacement coordinate system. The DOF directions come
    from `deform.dof_directions`, the same function the animation uses,
    so the shapes and the animation of them agree. Rotational DOFs are
    not included, because nothing measures them.

    The frequencies are exactly 0.0, which is how the FRF synthesis
    recognizes a rigid mode, and the damping is 0.0.

    Parameters
    ----------
    geometry : Geometry
        The nodes the shapes are written over.
    properties : MassProperties
        The reference point and, optionally, the mass and inertia
        that mass-normalize the set. Mass properties are refused on a
        geometry whose length unit has not been defined, because an
        inertia in kg·m² cannot be combined with coordinates that have
        no unit.

    Returns
    -------
    ShapeSet
        Six modes over `3 × nodes` DOFs, `unscaled` unless mass
        properties were given.

    Raises
    ------
    ValueError
        If the geometry has no nodes, or if mass properties are given
        and the geometry's units are not defined.
    """
    from ..deform import dof_directions
    from .shapes import ShapeSet

    if geometry.num_nodes == 0:
        raise ValueError('the geometry has no nodes to move')
    if properties.scaled and not geometry.units_defined:
        raise ValueError(
            'mass properties need the geometry\'s length unit — declare '
            'it with define_units first, or leave the shapes unscaled')

    dofs = [f'{int(node)}{axis}+' for node in geometry.node_id
            for axis in AXIS_NAMES]
    rows, directions, used = dof_directions(geometry, dofs)
    # every DOF names a node the geometry has and a translational
    # axis, so nothing is dropped; said here so a future change to
    # dof_directions cannot silently thin the set
    assert used.all(), 'a translational DOF of the geometry was dropped'

    center = np.asarray(properties.point, dtype=np.float64)
    offsets = geometry.node_xyz[rows] - center
    moments, axes = properties.principal_axes()
    shapes = np.empty((6, len(dofs)), dtype=np.float64)
    for k in range(3):
        unit = np.zeros(3)
        unit[k] = 1.0
        shapes[k] = directions @ unit
        # a small rotation about the axis moves a point by the cross
        # product of the axis with its offset from the center
        motion = np.cross(axes[k], offsets)
        shapes[3 + k] = np.einsum('ij,ij->i', directions, motion)
    if properties.scaled:
        shapes[:3] /= np.sqrt(properties.mass)
        shapes[3:] /= np.sqrt(moments)[:, np.newaxis]

    description = ([f'Rigid translation {name}' for name in AXIS_NAMES]
                   + [f'Rigid rotation about {axis_name(axis)}'
                      for axis in axes])
    return ShapeSet(
        frequency=np.zeros(6), damping=np.zeros(6), coordinate=dofs,
        shape_matrix=shapes, description=description,
        comment=f'rigid body {properties.describe()}',
        mass_unit='kg' if properties.scaled else None,
        unscaled=not properties.scaled)
