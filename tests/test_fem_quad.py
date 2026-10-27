"""The four-node shell on any convex quad (2026-10-26).

Rectangles only until a student's free-meshed circuit board needed more:
1,111 of its 11,489 quads off square, the worst by 49 degrees. The
general element is MITC4 in its covariant form. It has to be the
rectangle's element on a rectangle (the old formulation is kept below,
verbatim, as the reference), strain nothing in rigid motion, pass the
membrane and bending patch tests on a distorted patch, and keep a thin
free plate's frequencies on a distorted mesh where a locking element
would not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from visualdynamics.core import fem
from visualdynamics.core.fem import (
    DRILLING_FRACTION,
    SHEAR_CORRECTION,
    Material,
    _quad_matrices,
)

ALUMINUM = Material('aluminum', 69e9, 2700.0, 0.33)


# ---- the rectangle's element, as it was until 2026-10-26 -----------------------

def _rectangle_matrices(material: Material, thickness: float, a: float,
                    b: float) -> tuple[np.ndarray, np.ndarray]:
    """Local stiffness and consistent mass for the rectangular shell.

    24x24, six DOFs per node in DIRECTIONS order, nodes at
    (0,0) (a,0) (a,b) (0,b) in the element's own plane. Membrane and
    bending integrate 2x2 Gauss (exact for the rectangle); the
    transverse shear is the MITC4 assumed field — gamma_xz tied at the
    midpoints of the two eta edges and interpolated linearly in eta,
    gamma_yz the mirror of that — which is what keeps the thin limit
    honest instead of shear-locked.

    Sign conventions, written down because they were derived once and
    should not be re-derived: the tilt of the normal toward +x is
    +theta_y and toward +y is -theta_x, so the curvatures are
    (theta_y,x | -theta_x,y | theta_y,y - theta_x,x) and the shears
    (w,x + theta_y | w,y - theta_x). Both vanish identically on every
    rigid motion, which the rigid-mode test holds to machine zero.
    """
    E, nu = material.youngs_modulus, material.poissons_ratio
    G, rho, t = material.shear_modulus, material.density, thickness
    plane = E / (1.0 - nu * nu) * np.array([[1.0, nu, 0.0],
                                            [nu, 1.0, 0.0],
                                            [0.0, 0.0, (1.0 - nu) / 2.0]])
    d_membrane = t * plane
    d_bending = t ** 3 / 12.0 * plane
    d_shear = SHEAR_CORRECTION * G * t * np.eye(2)

    def basis(xi: float, eta: float):
        signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
        n = np.array([(1 + xi * sx) * (1 + eta * sy) / 4.0
                      for sx, sy in signs])
        dx = np.array([sx * (1 + eta * sy) / 4.0 * 2.0 / a
                       for sx, sy in signs])
        dy = np.array([sy * (1 + xi * sx) / 4.0 * 2.0 / b
                       for sx, sy in signs])
        return n, dx, dy

    def shear_rows(xi: float, eta: float) -> np.ndarray:
        n, dx, dy = basis(xi, eta)
        rows = np.zeros((2, 24))
        for i in range(4):
            rows[0, 6 * i + 2] = dx[i]          # gamma_xz = w,x + theta_y
            rows[0, 6 * i + 4] = n[i]
            rows[1, 6 * i + 2] = dy[i]          # gamma_yz = w,y - theta_x
            rows[1, 6 * i + 3] = -n[i]
        return rows

    stiffness = np.zeros((24, 24))
    consistent = np.zeros((24, 24))
    gauss = 1.0 / math.sqrt(3.0)
    area_weight = a * b / 4.0                    # dA per unit natural area
    for xi in (-gauss, gauss):
        for eta in (-gauss, gauss):
            n, dx, dy = basis(xi, eta)
            b_membrane = np.zeros((3, 24))
            b_bending = np.zeros((3, 24))
            fields = np.zeros((6, 24))
            for i in range(4):
                col = 6 * i
                b_membrane[0, col] = dx[i]
                b_membrane[1, col + 1] = dy[i]
                b_membrane[2, col] = dy[i]
                b_membrane[2, col + 1] = dx[i]
                b_bending[0, col + 4] = dx[i]
                b_bending[1, col + 3] = -dy[i]
                b_bending[2, col + 3] = -dx[i]
                b_bending[2, col + 4] = dy[i]
                for field in range(6):
                    fields[field, col + field] = n[i]
            # the assumed shear: the raw rows evaluated at the tying
            # points, blended linearly across the element
            b_shear = np.zeros((2, 24))
            b_shear[0] = ((1 - eta) * shear_rows(0.0, -1.0)[0]
                          + (1 + eta) * shear_rows(0.0, 1.0)[0]) / 2.0
            b_shear[1] = ((1 - xi) * shear_rows(-1.0, 0.0)[1]
                          + (1 + xi) * shear_rows(1.0, 0.0)[1]) / 2.0

            # the drilling tie: theta_z minus the rotation the membrane
            # field describes, penalized softly (see DRILLING_FRACTION)
            b_drilling = np.zeros((1, 24))
            for i in range(4):
                col = 6 * i
                b_drilling[0, col] = dy[i] / 2.0
                b_drilling[0, col + 1] = -dx[i] / 2.0
                b_drilling[0, col + 5] = n[i]

            stiffness += area_weight * (
                b_membrane.T @ d_membrane @ b_membrane
                + b_bending.T @ d_bending @ b_bending
                + b_shear.T @ d_shear @ b_shear
                + DRILLING_FRACTION * G * t
                * (b_drilling.T @ b_drilling))
            # translations carry rho t, the bending rotations the
            # section's rotary inertia rho t^3/12. The drilling
            # rotation carries a thousandth of that — not zero, because
            # a massless DOF breaks the Cholesky factorization the
            # eigensolution stands on, and not the full value, because
            # inertia against penalty stiffness sets where the drilling
            # *artifact* modes land: at the full inertia a half-inch
            # plate grew a wall of a hundred and sixty fake modes at
            # 4.2 kHz, mid-band; at a thousandth they sit near 130 kHz,
            # far above anything the element is valid for. Rigid
            # rotation about the normal loses nothing measurable — its
            # inertia is the translations' r^2 terms, not this.
            spin = rho * t ** 3 / 12.0
            weights = np.diag([rho * t] * 3 + [spin, spin, spin * 1e-3])
            consistent += area_weight * (fields.T @ weights @ fields)

    return stiffness, consistent




# ---- the general element reduces to it --------------------------------------

@pytest.mark.parametrize(('a', 'b', 't'), [(1.0, 1.0, 0.01), (0.3, 0.07, 0.002),
                                           (2.0, 0.5, 0.2)])
def test_a_rectangle_is_the_rectangles_element(a, b, t):
    xy = np.array([(0.0, 0.0), (a, 0.0), (a, b), (0.0, b)])
    k, m = _quad_matrices(ALUMINUM, t, xy)
    k0, m0 = _rectangle_matrices(ALUMINUM, t, a, b)
    assert np.allclose(k, k0, rtol=0, atol=1e-11 * abs(k0).max())
    assert np.allclose(m, m0, rtol=0, atol=1e-11 * abs(m0).max())


def _rigid_motions(xy):
    """The six rigid motions of a plate in its own plane, as 24-vectors."""
    out = []
    for axis in range(3):
        u = np.zeros(24)
        u[axis::6] = 1.0
        out.append(u)
    x, y = xy[:, 0], xy[:, 1]
    # about z: u = -y, v = x, theta_z = 1
    about_z = np.zeros(24)
    about_z[0::6], about_z[1::6], about_z[5::6] = -y, x, 1.0
    # about x: w = y, theta_x = 1; about y: w = -x, theta_y = 1
    about_x = np.zeros(24)
    about_x[2::6], about_x[3::6] = y, 1.0
    about_y = np.zeros(24)
    about_y[2::6], about_y[4::6] = -x, 1.0
    return out + [about_x, about_y, about_z]


@pytest.mark.parametrize('xy', [
    [(0.0, 0.0), (1.0, 0.0), (1.3, 1.0), (0.0, 1.0)],
    [(0.0, 0.0), (2.0, 0.3), (1.6, 1.4), (-0.2, 0.9)],
    [(0.0, 0.0), (1.0, 0.0), (0.52, 0.52), (0.0, 1.0)],    # nearly a triangle
])
def test_rigid_motion_strains_nothing(xy):
    xy = np.array(xy, dtype=float)
    k, _m = _quad_matrices(ALUMINUM, 0.01, xy)
    scale = abs(k).max()
    for u in _rigid_motions(xy):
        assert abs(k @ u).max() < 1e-9 * scale


def test_a_distorted_quads_mass_is_its_area_times_rho_t():
    xy = np.array([(0.0, 0.0), (2.0, 0.3), (1.6, 1.4), (-0.2, 0.9)])
    _k, m = _quad_matrices(ALUMINUM, 0.01, xy)
    area = fem._quad_area(xy)
    for axis in range(3):
        u = np.zeros(24)
        u[axis::6] = 1.0
        assert u @ m @ u == pytest.approx(2700.0 * 0.01 * area, rel=1e-12)


# ---- patch tests --------------------------------------------------------------

#: MacNeal and Harder's patch: a 0.24 x 0.12 rectangle cut into five
#: distorted quads around four irregular interior nodes
PATCH_NODES = {1: (0.0, 0.0), 2: (0.24, 0.0), 3: (0.24, 0.12), 4: (0.0, 0.12),
               5: (0.04, 0.02), 6: (0.18, 0.03), 7: (0.16, 0.08),
               8: (0.08, 0.08)}
PATCH_QUADS = [(1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8),
               (5, 6, 7, 8)]
INTERIOR = (5, 6, 7, 8)


def _patch(thickness):
    model = fem.Model('patch')
    for node, (x, y) in PATCH_NODES.items():
        model.add_node(node, x, y, 0.0)
    for quad in PATCH_QUADS:
        model.add_plate(quad, ALUMINUM, thickness)
    _mass, stiffness = model.sparse_matrices()
    return model, stiffness.toarray()


def _residual_at_the_interior(model, stiffness, field):
    """K u for the nodal values `field(x, y)` gives, at the interior
    nodes, against the largest force at the boundary."""
    u = np.concatenate([field(*PATCH_NODES[n]) for n in model.node_ids])
    force = (stiffness @ u).reshape(-1, 6)
    interior = [model.node_ids.index(n) for n in INTERIOR]
    boundary = [i for i in range(len(model.node_ids)) if i not in interior]
    return abs(force[interior]).max() / abs(force[boundary]).max()


def test_the_membrane_patch_test():
    """A constant strain state, u = 1e-3 (x + y/2), v = 1e-3 (x/2 + y),
    with the drilling rotation it implies: the interior nodes carry no
    force."""
    model, stiffness = _patch(0.001)

    def field(x, y):
        u, v = 1e-3 * (x + y / 2), 1e-3 * (x / 2 + y)
        return np.array([u, v, 0.0, 0.0, 0.0, (0.5e-3 - 0.5e-3) / 2.0])

    assert _residual_at_the_interior(model, stiffness, field) < 1e-10


def test_the_bending_patch_test():
    """A constant curvature state, w = 1e-3 (x^2 + xy + y^2) / 2, with the
    rotations that make the transverse shear zero (theta_y = -w,x,
    theta_x = w,y): the interior nodes carry no force."""
    model, stiffness = _patch(0.001)

    def field(x, y):
        w = 1e-3 * (x * x + x * y + y * y) / 2.0
        wx, wy = 1e-3 * (x + y / 2), 1e-3 * (x / 2 + y)
        return np.array([0.0, 0.0, w, wy, -wx, 0.0])

    assert _residual_at_the_interior(model, stiffness, field) < 1e-8


# ---- a thin plate on a distorted mesh -----------------------------------------

FREE_SQUARE_PLATE = [13.468, 19.596, 24.271, 34.801, 34.801, 61.093]


def _distorted_plate(mesh, thickness, side=1.0, shake=0.3, seed=1):
    """The free square plate meshed `mesh` x `mesh` with every interior
    node moved up to `shake` of a cell at random: quads of every shape."""
    rng = np.random.default_rng(seed)
    h = side / mesh
    model = fem.Model('distorted')
    for j in range(mesh + 1):
        for i in range(mesh + 1):
            x, y = i * h, j * h
            if 0 < i < mesh and 0 < j < mesh:
                x += shake * h * rng.uniform(-1, 1)
                y += shake * h * rng.uniform(-1, 1)
            model.add_node(1 + j * (mesh + 1) + i, x, y, 0.0)
    for j in range(mesh):
        for i in range(mesh):
            n = 1 + j * (mesh + 1) + i
            model.add_plate((n, n + 1, n + mesh + 2, n + mesh + 1), ALUMINUM,
                            thickness)
    return model


def _lambdas(model, side, thickness):
    shapes = model.eigensolution(num_modes=12)
    d = 69e9 * thickness ** 3 / (12.0 * (1.0 - 0.33 ** 2))
    elastic = shapes.frequency[shapes.frequency > 0.0][:6]
    return shapes, (elastic * 2 * math.pi * side ** 2
                    * math.sqrt(2700.0 * thickness / d))


@pytest.mark.parametrize('thickness', [0.01, 0.001])
def test_a_distorted_thin_plate_keeps_its_frequencies(thickness):
    """Six rigid modes and the free plate's six lambda^2 within a few
    percent on a 16 x 16 mesh shaken by 30 % of a cell, at a hundred and
    at a thousand to one: a shear-locking element would be several times
    high at the thinner one. (At ten thousand to one the rigid modes come
    apart on the regular rectangles too, which are the element as it
    was: the eigensolution's conditioning, not this element.)"""
    shapes, lambdas = _lambdas(_distorted_plate(16, thickness), 1.0, thickness)
    assert int(np.sum(shapes.frequency == 0.0)) == 6
    for computed, exact in zip(lambdas, FREE_SQUARE_PLATE):
        assert computed == pytest.approx(exact, rel=0.04), (computed, exact)
