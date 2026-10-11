"""Rules in `core.fem` pinned by value where a mutation pass found them
unguarded (2026-10-11).

`tools/mutate.py` over fem.py against its twelve test files caught 1443
of 1747. Most survivors are equivalent by construction — a triangle's
local frame puts its first node at the origin and its second on the x
axis, a brick's corners are at plus or minus one, so many sign and
operator flips cannot change a number — but two were real: the rotary
inertia `distribute_mass` gives each node, and the mass distribution
of a brick that is not a parallelepiped, which the tests of its total
mass and its stiffness cannot see.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_fem import MASSLESS, MEMBER, cube

from visualdynamics.core import fem


def test_each_node_spins_as_its_patch_of_structure():
    """`spin * m * L^2`, L the mean length of the members meeting at
    the node: a cube's corner carries three equal edges, so its share is
    an eighth and its inertia a twelfth of that share times the edge
    squared — on all three axes."""
    model = cube()
    model.wire_faces(MASSLESS, MEMBER)
    edges = {round(model._length(beam), 12) for beam in model.beams}
    assert len(edges) == 1, 'a cube: every edge the same'
    (size,) = edges
    model.distribute_mass(0.8)
    shared = [item for item in model.masses if item.name == 'shared']
    assert len(shared) == 8
    for item in shared:
        assert item.mass == pytest.approx(0.1)
        assert item.inertia == pytest.approx((0.1 * size ** 2 / 12.0,) * 3)
    model.distribute_mass(0.8, spin=0.5)
    item = next(item for item in model.masses if item.name == 'shared')
    assert item.inertia[0] == pytest.approx(0.5 * 0.1 * size ** 2)


def _trilinear(xyz, point):
    """The brick's map, written out here from its definition rather
    than taken from the element: corner k at natural (sx, sy, sz)."""
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    weights = np.prod(1.0 + corners * np.asarray(point), axis=1) / 8.0
    return weights @ xyz


def test_a_distorted_bricks_mass_sits_where_its_material_does():
    """Every row of the consistent mass, summed against the node
    coordinates, is the brick's first moment, rho times the integral of
    x over its volume — exact at 2x2x2 for a trilinear map. Checked
    against the same integral done here independently: the map written
    out by hand, its Jacobian by central differences, a 4x4x4 rule. A
    parallelepiped cannot tell a mirrored shape function from the true
    one, and the tests of total mass and of stiffness never looked at
    where the mass sits."""
    steel = fem.material('1018 steel')
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.3, 1.1, 0.0],
                    [0.0, 1.0, 0.1], [0.0, 0.1, 1.0], [1.2, 0.0, 0.9],
                    [1.0, 1.2, 1.4], [-0.1, 1.0, 1.1]])
    _stiffness, mass = fem._hex_matrices(steel, xyz)
    points, weights = np.polynomial.legendre.leggauss(4)
    h = 1e-6
    moment, volume = np.zeros(3), 0.0
    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            for zeta, wz in zip(points, weights):
                here = np.array([xi, eta, zeta])
                jacobian = np.column_stack([
                    (_trilinear(xyz, here + h * e) - _trilinear(xyz, here - h * e))
                    / (2.0 * h) for e in np.eye(3)])
                dv = abs(np.linalg.det(jacobian)) * wx * wy * wz
                volume += dv
                moment += _trilinear(xyz, here) * dv
    rho = steel.density
    for axis in range(3):
        rows = mass[axis::3, axis::3]
        assert rows.sum() == pytest.approx(rho * volume, rel=1e-8)
        assert (rows @ xyz[:, axis]).sum() == pytest.approx(rho * moment[axis],
                                                            rel=1e-8)
