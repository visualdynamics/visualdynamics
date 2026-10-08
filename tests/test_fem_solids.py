"""Solid elements: the hexahedron with incompatible modes, the wedge and
the tetrahedron (2026-09-30, for the four-unit frame example — a
half-inch ladder frame that beams describe poorly and plates not at
all).

What is pinned: a brick's volume and mass; rigid motions strain nothing,
distorted or not; a constant strain is exact on a distorted brick (the
patch test the incompatible modes were given Taylor's form to pass); a
bar of bricks one element deep bends like the beam element, which a
plain trilinear brick cannot do; a plate of bricks one element thick
vibrates like the plate element; the rotations only solids touch are
grounded; dense and sparse agree; a solid block builds from a geometry
and comes back out of one.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_fem import ALUMINUM

from visualdynamics.core import fem
from visualdynamics.core.fem import (
    GroupProperties,
    Model,
    Section,
    _hex_matrices,
    _solid_volume,
    _tet_matrices,
    _wedge_matrices,
)

CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * 0.01


def _rigid(xyz):
    """Six rigid motions over these nodes, three translations per node."""
    out = np.zeros((3 * len(xyz), 6))
    for i, x in enumerate(xyz):
        out[3 * i:3 * i + 3, :3] = np.eye(3)
        out[3 * i:3 * i + 3, 3:] = [[0, x[2], -x[1]], [-x[2], 0, x[0]],
                                    [x[1], -x[0], 0]]
    return out


@pytest.mark.parametrize('distort', [0.0, 0.002])
def test_a_brick_weighs_its_volume_and_strains_nothing_when_rigid(distort):
    rng = np.random.default_rng(1)
    xyz = CUBE + rng.uniform(-distort, distort, CUBE.shape)
    stiffness, mass = _hex_matrices(ALUMINUM, xyz)
    volume = _solid_volume(xyz)
    assert mass.sum() / 3 == pytest.approx(ALUMINUM.density * volume, rel=1e-12)
    assert np.abs(stiffness @ _rigid(xyz)).max() < 1e-9 * np.abs(stiffness).max()
    assert np.allclose(stiffness, stiffness.T)


def test_a_distorted_brick_carries_a_constant_strain_exactly():
    """The patch test: u = eps x on a brick that is not a parallelepiped
    stores exactly V eps^2 (lambda + 2 mu) — Taylor's form of the
    incompatible modes, with the bubble strains taken from the centroid."""
    rng = np.random.default_rng(1)
    xyz = CUBE + rng.uniform(-0.002, 0.002, CUBE.shape)
    stiffness, _mass = _hex_matrices(ALUMINUM, xyz)
    u = np.zeros(24)
    u[0::3] = 1e-3 * xyz[:, 0]
    E, nu = ALUMINUM.youngs_modulus, ALUMINUM.poissons_ratio
    lame = E * nu / ((1 + nu) * (1 - 2 * nu))
    energy = _solid_volume(xyz) * 1e-6 * (lame + 2 * ALUMINUM.shear_modulus)
    assert u @ stiffness @ u == pytest.approx(energy, rel=1e-10)


def test_the_wedge_and_the_tetrahedron_agree_with_themselves():
    wedge = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                      [0, 0, 1], [1, 0, 1], [0, 1, 1]], float) * 0.01
    tet = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) * 0.01
    for xyz, matrices in ((wedge, _wedge_matrices), (tet, _tet_matrices)):
        stiffness, mass = matrices(ALUMINUM, xyz)
        assert mass.sum() / 3 == pytest.approx(
            ALUMINUM.density * _solid_volume(xyz), rel=1e-12)
        assert np.abs(stiffness @ _rigid(xyz)).max() < 1e-9 * np.abs(stiffness).max()
    assert _solid_volume(wedge) == pytest.approx(0.5e-6)
    assert _solid_volume(tet) == pytest.approx(1e-6 / 6)


def _bar_of_bricks(length=0.5, depth=0.02, along=25, deep=1):
    """A bar meshed with bricks, `deep` through its depth and its width."""
    model = Model()
    xs = np.linspace(0, length, along + 1)
    ys = np.linspace(0, depth, deep + 1)
    ids = {}
    for k, z in enumerate(ys):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                ids[i, j, k] = model.add_node(len(ids) + 1, x, y, z)
    for k in range(deep):
        for j in range(deep):
            for i in range(along):
                model.add_solid([ids[i, j, k], ids[i + 1, j, k],
                                 ids[i + 1, j + 1, k], ids[i, j + 1, k],
                                 ids[i, j, k + 1], ids[i + 1, j, k + 1],
                                 ids[i + 1, j + 1, k + 1], ids[i, j + 1, k + 1]],
                                ALUMINUM)
    return model


def test_one_layer_of_bricks_bends_like_the_beam():
    """A free bar 25 to 1, one brick deep, against the beam element: the
    first bending pair within two percent (the beam is Euler-Bernoulli;
    the brick carries the little shear a bar this slender has). A plain
    trilinear brick puts the same mode about a third high."""
    bricks = _bar_of_bricks()
    shapes = bricks.eigensolution(num_modes=10)
    assert int(np.sum(shapes.frequency == 0.0)) == 6
    beam = Model()
    for i, x in enumerate(np.linspace(0, 0.5, 26)):
        beam.add_node(i + 1, x, 0, 0)
    beam.add_chain(range(1, 27), ALUMINUM, Section.rectangle('bar', 0.02, 0.02))
    reference = beam.eigensolution(num_modes=10).frequency
    bending = shapes.frequency[6:8]
    assert bending == pytest.approx(reference[6:8], rel=0.02), (
        bending, reference[6:8])
    # the rotations of every node are grounded: only solids touch them
    assert bricks.dangling_rotations() == bricks.node_ids


def test_a_plate_of_bricks_vibrates_like_the_plate():
    """A thin free plate, a hundred to one, meshed 16 x 16 one brick
    thick, against the plate element on the same grid: the first seven
    elastic modes within 1.5 % (measured: 0.3 % on the bending pairs,
    1.3 % on the twisting pair). A second layer of bricks changes
    nothing — the thickness is one brick's business — while the in-plane
    mesh is what converges, an 8 x 8 grid of bricks sitting 18 % high on
    the twisting pair where the plate element is already there."""
    from test_fem import square_plate

    side, thickness, n = 1.0, 0.01, 16
    plate = square_plate(n, side, thickness)   # the plate element's mesh
    bricks = Model()
    grid = np.linspace(0, side, n + 1)
    ids = {}
    for k, z in enumerate((0.0, thickness)):
        for j, y in enumerate(grid):
            for i, x in enumerate(grid):
                ids[i, j, k] = bricks.add_node(len(ids) + 1, x, y, z)
    for j in range(n):
        for i in range(n):
            bricks.add_solid([ids[i, j, 0], ids[i + 1, j, 0], ids[i + 1, j + 1, 0],
                              ids[i, j + 1, 0], ids[i, j, 1], ids[i + 1, j, 1],
                              ids[i + 1, j + 1, 1], ids[i, j + 1, 1]], ALUMINUM)
    reference = plate.eigensolution(num_modes=13).frequency[6:13]
    found = bricks.eigensolution(num_modes=13).frequency[6:13]
    assert found == pytest.approx(reference, rel=0.015), (found, reference)


def test_dense_and_sparse_agree_on_solids():
    model = _bar_of_bricks(along=10)
    dense = model.eigensolution(num_modes=12, solver='dense').frequency
    sparse = model.eigensolution(num_modes=12, solver='sparse').frequency
    assert sparse == pytest.approx(dense, rel=1e-7, abs=1e-6)


def test_a_solid_is_refused_flat_inverted_or_rigid():
    model = Model()
    for node, xyz in enumerate(CUBE, 1):
        model.add_node(node, *xyz)
    for node, xyz in enumerate(CUBE[:4] * [1, 1, 0] + [0.02, 0, 0], 9):
        model.add_node(node, *xyz)                  # four more, all flat
    with pytest.raises(ValueError, match='no volume'):
        model.add_solid([1, 2, 3, 4, 9, 10, 11, 12], ALUMINUM)
    with pytest.raises(ValueError, match='not rigid'):
        model.add_solid(range(1, 9), fem.RIGID)
    with pytest.raises(ValueError, match='eight, six or four'):
        model.add_solid(range(1, 6), ALUMINUM)


def test_inertia_on_a_node_only_solids_touch_is_refused():
    model = _bar_of_bricks(along=4)
    model.add_mass(1, 0.01, inertia=(1e-6, 0, 0))
    with pytest.raises(ValueError, match='rotary inertia'):
        model.eigensolution(num_modes=8)


def test_a_solid_block_round_trips_through_a_geometry():
    model = _bar_of_bricks(along=6)
    model.solids[0].group = 'bar'
    for solid in model.solids:
        solid.group = 'bar'
    geometry = model.geometry()
    assert set(geometry.elem_type.tolist()) == {115}
    assert list(geometry.group_name) == ['bar']
    geometry.group_properties = {int(geometry.group_id[0]): GroupProperties(ALUMINUM)}
    assert geometry.group_properties[int(geometry.group_id[0])].kind == 'solid'
    rebuilt = Model.from_geometry(geometry)
    assert len(rebuilt.solids) == 6
    assert rebuilt.structural_mass == pytest.approx(model.structural_mass)
    assert rebuilt.eigensolution(num_modes=8).frequency == pytest.approx(
        model.eigensolution(num_modes=8).frequency)
    # and the wrong properties are refused by what the elements are
    geometry.group_properties = {int(geometry.group_id[0]):
                                 GroupProperties(ALUMINUM, thickness=0.01)}
    with pytest.raises(ValueError, match='hex8 elements, which take a material alone'):
        Model.from_geometry(geometry)


def test_a_solid_draws_as_its_skin():
    """Two bricks side by side draw as ten quads — the faces no other
    brick shares — not as a polyline through every corner."""
    import pyvista as pv

    from visualdynamics.core import mesh
    from visualdynamics.viz.geometry import add_geometry

    two = mesh.block((0, 0, 0), (0.2, 0, 0), (0, 0.1, 0), (0, 0, 0.1), 0.1, 'two')
    assert len(two.elem_conn) == 2
    plotter = pv.Plotter(off_screen=True)
    meshes: list = []
    add_geometry(plotter, two, meshes=meshes)
    faces = [m for m in meshes if m.n_faces_strict]
    assert len(faces) == 1 and faces[0].n_faces_strict == 10
    assert not any(m.n_lines for m in meshes), 'no polylines through the cells'
    plotter.close()


def test_a_node_nothing_touches_is_grounded_not_refused():
    """A deck's reference point — a grid no element names — used to
    refuse the whole model as 'connected to nothing'. Built from blocks
    it is grounded at solve time instead, and the modes are the
    structure's."""
    model = _bar_of_bricks(along=6)
    for solid in model.solids:
        solid.group = 'bar'
    reference = model.eigensolution(num_modes=8).frequency
    geometry = model.geometry()
    geometry.add_nodes([[0.25, 0.5, 0.5]])          # a point in space
    geometry.group_properties = {int(geometry.group_id[0]): GroupProperties(ALUMINUM)}
    rebuilt = Model.from_geometry(geometry)
    assert rebuilt.loose_nodes() == [int(geometry.node_id[-1])]
    shapes = rebuilt.eigensolution(num_modes=8)
    assert shapes.frequency == pytest.approx(reference)
    assert int(np.sum(shapes.frequency == 0.0)) == 6, 'no extra free body'
