"""Rigid-body mode shapes of a geometry.

Six shapes every free structure has: three translations and three
rotations about a reference point. They are the start of the virtual
point workflow (PLAN.md, "The virtual point arc"): responses measured
at the control DOFs are transformed through them to the six motions
of a point, and a specification is transformed the same way, so the
two can be compared there.

**The shapes are kinematic; only the reference point shapes them.** A
rigid translation moves every node one unit along an axis; a rotation
of one radian about an axis `e` through the point `c` moves a node at
`r` by `e × (r − c)`. No mass enters. The point is what a test
engineer calls the CG, and the geometry's centroid is the default.

**Mass and inertia only scale.** `ShapeSet` holds mass-normalized
shapes, `φᵀMφ = I`. A rigid body's 6×6 mass matrix about its CG is
diagonal in its principal axes — the mass three times, then the three
principal moments — so normalizing divides each translation by `√m`
and each rotation by `√I` about its own axis. That is all the mass
properties do, and it is why they come as one optional group: mass
without inertia could scale three rows and not the other three, and a
set normalized by halves carries modal masses nobody can read. Without
them the set is exactly what a fit without a drive point is,
`unscaled` — unit shapes, modal mass a convention.

**The rotation axes are the inertia's principal axes.** Rotations
about the global axes are mass-orthogonal only when the products of
inertia vanish. With a coupled tensor the six orthonormal shapes
rotate about its eigenvectors, and there is no choice about it, so the
principal decomposition is always taken: a diagonal tensor's principal
axes are the global axes and the user who typed three numbers gets X,
Y and Z back; a coupled one gets its own axes, named in the mode
descriptions. Taking the diagonal of a coupled tensor and calling the
result mass-normalized is a wrong answer that looks right.

Mass normalization is only meaningful about the CG — about any other
point the rigid mass matrix couples translation to rotation and the
six are not orthogonal — so the mass properties, when given, are taken
to be about the point. Unscaled shapes about *any* point span the same
six-dimensional space, which is what the later transformation wants:
it builds unscaled shapes about the virtual point, an interface
location and not the CG at all, through this same function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem
    from .geometry import Geometry
    from .shapes import ShapeSet

#: the order the six inertia terms are stated in, and the tensor
#: entries each one fills. `Ixy` is the product of inertia as a tensor
#: entry — the negative of the integral ∫xy dm some handbooks tabulate
#: — so a tensor typed straight from a CAD mass-properties report,
#: which states the tensor itself, needs no sign flipped.
INERTIA_TERMS = ('Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz')
_TENSOR_SLOTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

AXIS_NAMES = ('X', 'Y', 'Z')

def is_rigid_set(shapes: ShapeSet) -> bool:
    """Whether a set is the six rigid-body shapes `rigid_body_shapes`
    writes: six modes at exactly 0 Hz, described as such.

    Read off the set rather than flagged on it, so a rigid set that
    traveled through a file, a copy or a script is still one.
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

    `point` is in meters, `mass` in kilograms, `inertia` in kg·m² —
    SI, like every stored value here — and the six inertia terms are
    the tensor's `Ixx, Iyy, Izz, Ixy, Ixz, Iyz` (`INERTIA_TERMS`).

    Frozen like `Averaging` and `Truncation`, and for the same reason:
    the settings ride the geometry, the staleness fingerprint is their
    fields, and a mutable setting would be a fingerprint that lies.
    Validated at entry rather than at use: mass and inertia both or
    neither, a positive mass, and a tensor a real body could have —
    symmetric positive definite, which a typo in a product of inertia
    breaks and which the triangle inequality (`Ixx + Iyy ≥ Izz` for a
    real body) is a consequence of.
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
        """Whether the shapes are mass-normalized (mass and inertia
        given) or unit shapes."""
        return self.mass is not None

    def tensor(self) -> np.ndarray:
        """The 3×3 inertia tensor, or the identity when unscaled —
        which is what makes the unscaled rotation axes X, Y and Z."""
        tensor = np.eye(3)
        if self.inertia is None:
            return tensor
        for value, (i, j) in zip(self.inertia, _TENSOR_SLOTS):
            tensor[i, j] = tensor[j, i] = value
        return tensor

    def principal_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """(moments, axes): the principal moments and the unit axis of
        each, as the rows of `axes`, one per rotation mode.

        Ordered and signed so that mode 4 rotates about the principal
        axis nearest X, 5 nearest Y, 6 nearest Z, each pointed to have
        a positive component along its global axis — a diagonal tensor
        gives exactly X, Y, Z, and a lightly coupled one gives axes a
        reader can still call X, Y and Z. Each rotation shape is built
        from its own axis, so the three need not form a right-handed
        triad and are not forced into one.
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
        """The settings in words, for the status line and the
        staleness story alike: `'about (0.1, 0, 0.05) m, unit shapes'`.

        In SI by default — the journal's and the staleness story's
        units — or in a unit system's display units when one is given.
        `defined=False` says the geometry's coordinates are raw, so the
        point is labeled as such rather than as meters.
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
    """`'X'` for a global axis, else the axis's components — what a
    rotation mode's description says it turns about."""
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
    translational DOFs of every node, each coefficient expressed in
    the node's own displacement frame — `deform.dof_directions` is the
    one owner of a DOF's global direction, so the shapes and the
    animation that draws them agree by construction. Rotational DOFs
    are not carried: nothing here measures them.

    Frequencies are exactly 0.0, the convention the FRF synthesis
    relies on to recognize a rigid mode, and damping is 0.0.

    Parameters
    ----------
    geometry : Geometry
        The nodes the shapes are written over.
    properties : MassProperties
        The reference point and, optionally, the mass and inertia
        that mass-normalize the set. Mass properties on a geometry
        whose length unit is undeclared are refused: an inertia in
        kg·m² against coordinates in nothing is not a number.

    Returns
    -------
    ShapeSet
        Six modes over `3 × nodes` DOFs, `unscaled` unless mass
        properties were given.
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
