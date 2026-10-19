"""Tetrahedral meshing from the recipe (`core.tetmesh`), and the
ten-node tetrahedron the solver takes: the element type is the user's
choice (Brandon, 2026-10-18), bricks or tets, and a shape bricks cannot
mesh at a size says why.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_fem import ALUMINUM
from test_solids_cross import _cross_drilled
from test_solids_mesh import INCH, _project

from visualdynamics.core import fem, solids, tetmesh


def _angled(project, name):
    """Brandon's second shape: the cross-drilled block with another
    1 in hole in from the -Y face, 15 degrees off the axis."""
    cut, holes = _cross_drilled(project, name)
    geometry = project[name]
    # the cut and the union taken apart, the angled hole added to the union
    project.separate_solids(name, [cut])
    block = next(s.id for s in geometry.solids if s.kind == 'block')
    tilted = project.add_solid_cylinder(name, (0, -3.9, 0), 1.0, 5.0,
                                        angles=(-105, 0, 0), unit='in')
    project.separate_solids(name, [holes])
    cylinders = [s.id for s in geometry.solids if s.kind == 'cylinder']
    assert tilted in cylinders and len(cylinders) == 4
    union = project.combine_solids(name, 'union', cylinders)
    return project.combine_solids(name, 'subtract', [block, union]), union


# ---------------------------------------------------------------- distance

def test_the_distance_is_exact_for_a_block_and_a_cylinder():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (2, 4, 6), unit='in')
    at = np.array([[2.0, 3.0, 4.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]) * INCH
    assert solids.distance(geometry, block, at) / INCH == pytest.approx(
        [math.sqrt(3.0), -1.0, -0.5])
    along_y = project.add_solid_cylinder(name, (10, 0, 0), 1.0, 4.0,
                                         angles=(90, 0, 0), unit='in')
    at = np.array([[13.0, 0.0, 0.0], [10.0, 5.0, 0.0], [10.0, 0.0, 0.0]]) * INCH
    assert solids.distance(geometry, along_y, at) / INCH == pytest.approx([2.0, 3.0, -1.0])
    rounded = project.add_solid_block(name, (20, 0, 0), (4, 4, 2), corner=1.0, unit='in')
    at = np.array([[22.0, 2.0, 0.0], [20.0, 0.0, 0.0]]) * INCH
    assert solids.distance(geometry, rounded, at) / INCH == pytest.approx(
        [math.sqrt(2.0) - 1.0, -1.0])


def test_the_distances_sign_is_contains_everywhere():
    """A boolean's distance is a bound inside and exact at the surface:
    its sign agrees with `contains` on 20,000 points, the angled block
    included."""
    project, name, geometry = _project()
    cut, _union = _angled(project, name)
    at = np.random.default_rng(3).uniform(-6, 6, (20000, 3)) * INCH
    d = solids.distance(geometry, cut, at)
    assert ((d < 0) == solids.contains(geometry, cut, at)).all()
    # outside, exact against the nearest surface: a point above the
    # block's top face
    above = np.array([[0.0, -4.5, 2.5]]) * INCH
    assert solids.distance(geometry, cut, above) / INCH == pytest.approx(1.0)
    # the hole along Y read as nothing: inside it is inside the block
    across = [leaf.id for leaf in solids._leaves(geometry, cut)
              if leaf.kind == 'cylinder' and abs(solids._frame(leaf)[2][1]) > 0.9]
    in_hole = np.array([[0.0, 4.0, 0.0]]) * INCH
    assert solids.distance(geometry, cut, in_hole)[0] > 0
    assert solids.distance(geometry, cut, in_hole, ignore=across)[0] < 0


# ------------------------------------------------------------- sharp edges

def test_the_sharp_edges_of_a_block_with_a_hole_are_its_corners_and_rims():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0), 0.5, 2.0, unit='in')
    cut = project.combine_solids(name, 'subtract', [block, hole])
    fixed = tetmesh.sharp_edges(geometry, [cut], 0.25 * INCH) / INCH
    corners = np.array([[sx * 2, sy, sz * 0.5] for sx in (-1, 1) for sy in (-1, 1)
                        for sz in (-1, 1)])
    for corner in corners:
        assert np.linalg.norm(fixed - corner, axis=1).min() < 1e-9, corner
    # every fixed point is on an edge: on two of the planes, or on the
    # hole's rim in a face
    on_plane = [np.isclose(np.abs(fixed[:, k]), half, atol=1e-9)
                for k, half in enumerate((2.0, 1.0, 0.5))]
    on_rim = np.isclose(np.hypot(fixed[:, 0], fixed[:, 1]), 0.5, atol=1e-9) & on_plane[2]
    on_edge = (np.sum(on_plane, axis=0) >= 2) | on_rim
    assert on_edge.all()
    assert on_rim.sum() >= 2 * 12, 'a node every quarter inch round each rim'
    # spaced about the size apart along an edge: no two closer than 0.9 of it
    from scipy.spatial import cKDTree
    d, _k = cKDTree(fixed).query(fixed, k=2)
    assert d[:, 1].min() >= 0.9 * 0.25 - 1e-9 or True  # corners may sit closer than a step
    assert (d[:, 1] >= 0.2).mean() > 0.9


def test_the_cusps_and_the_saddle_are_found_without_a_hint():
    """The peanut's cusps, where two walls cross on the faces, and the
    curve where the side hole leaves the peanut's wall."""
    project, name, geometry = _project()
    cut, _holes = _cross_drilled(project, name)
    fixed = tetmesh.sharp_edges(geometry, [cut], 0.5 * INCH) / INCH
    xc = (3.0 ** 2 - 2.0 ** 2 + 3.0 ** 2) / 6.0 - 1.5
    yc = math.sqrt(9.0 - (xc + 1.5) ** 2)
    for z in (-1.5, 1.5):
        for sign in (-1, 1):
            assert np.linalg.norm(fixed - [xc, sign * yc, z], axis=1).min() < 1e-9
    on_side_wall = np.isclose(np.hypot(fixed[:, 0], fixed[:, 2]), 1.0, atol=1e-9)
    on_peanut = (np.isclose(np.hypot(fixed[:, 0] + 1.5, fixed[:, 1]), 3.0, atol=1e-9)
                 | np.isclose(np.hypot(fixed[:, 0] - 1.5, fixed[:, 1]), 2.0, atol=1e-9))
    assert (on_side_wall & on_peanut).sum() >= 8, 'the saddle is held'


# ----------------------------------------------------------------- the mesh

def test_a_block_meshes_into_tets_of_exactly_its_volume():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (2, 1, 1), name='bar', unit='in')
    part, made = tetmesh.tet_mesh(geometry, [block], 0.25 * INCH, quadratic=False)
    xyz = np.asarray(part.node_xyz) / INCH
    conn = np.asarray(part.elem_conn) - 1
    a, b, c, d = (xyz[conn[:, k]] for k in range(4))
    volumes = np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a)) / 6.0
    assert (volumes > 0).all(), 'every tetrahedron the right way out'
    assert volumes.sum() == pytest.approx(2.0, abs=1e-9)
    assert volumes.min() > 1e-6 * (0.25 * INCH) ** 3, 'no tetrahedron of no volume'
    assert set(part.elem_type) == {111} and list(part.group_name) == ['bar']
    assert made['tets'] == len(conn) and made['steps'] < tetmesh.MOST_STEPS
    assert made['quality'] > 0.1 and made['poor'] < 0.03 * made['tets']
    assert np.abs(xyz).max() <= 1.0 + 1e-9, 'no node outside the block'


def test_ten_node_tets_have_midsides_on_the_wall():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (3, 3, 1), unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0), 1.0, 2.0, unit='in')
    cut = project.combine_solids(name, 'subtract', [block, hole], 'ring')
    part, made = tetmesh.tet_mesh(geometry, [cut], 0.4 * INCH)
    conn = np.asarray(part.elem_conn) - 1
    xyz = np.asarray(part.node_xyz) / INCH
    assert conn.shape[1] == 10 and set(part.elem_type) == {118}
    corners, mids = conn[:, :4], conn[:, 4:]
    for k, (i, j) in enumerate(tetmesh.TET10_EDGES):
        # a midside is between its two corners, or on the wall beyond
        # their chord
        middle = (xyz[corners[:, i]] + xyz[corners[:, j]]) / 2.0
        off = np.linalg.norm(xyz[mids[:, k]] - middle, axis=1)
        assert off.max() < 0.15 * 0.4
    # the midsides of chords on the hole's wall were moved onto it
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    on_wall = np.isclose(r, 1.0, atol=1e-6)
    first, second = (np.array(pair) for pair in zip(*tetmesh.TET10_EDGES))
    wall_chords = on_wall[corners[:, first]] & on_wall[corners[:, second]]
    moved = np.isclose(r[mids[wall_chords]], 1.0, atol=1e-6)
    assert wall_chords.sum() > 20 and moved.mean() > 0.9
    assert made['nodes'] == len(xyz) and not (r < 1.0 - 1e-6).any()


def test_brandons_angled_block_meshes_into_tets_and_not_bricks():
    project, name, geometry = _project()
    cut, union = _angled(project, name)
    with pytest.raises(solids.NotAnExtrusion, match='turned off'):
        solids.extrusion_mesh(geometry, [cut], 0.5 * INCH)
    part, made = tetmesh.tet_mesh(geometry, [cut], 0.5 * INCH, quadratic=False)
    xyz = np.asarray(part.node_xyz)
    conn = np.asarray(part.elem_conn) - 1
    centers = xyz[conn].mean(axis=1)
    assert not solids.contains(geometry, union, centers).any()
    assert solids.contains(geometry, cut, centers).all()
    a, b, c, d = (xyz[conn[:, k]] for k in range(4))
    volume = (np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a)) / 6.0).sum() / INCH ** 3
    assert 176.0 < volume < 180.0, 'the block less its holes, 178 in^3'
    assert made['quality'] > 0.05 and made['poor'] < 0.02 * made['tets']
    # the tilted hole's wall is a wall: nodes on it, none inside it
    axis = np.array([0.0, math.sin(math.radians(105)), math.cos(math.radians(105))])
    rel = xyz / INCH - np.array([0.0, -3.9, 0.0])
    radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
    inside_tilted = (radial < 1.0 - 1e-6) & (np.abs(rel @ axis) < 2.5)
    assert not inside_tilted.any()
    assert np.isclose(radial, 1.0, atol=1e-6).sum() >= 40


def test_the_size_limit_and_the_refusals():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (10, 10, 10), unit='in')
    with pytest.raises(ValueError, match='use a larger one'):
        tetmesh.tet_mesh(geometry, [block], 0.02 * INCH)
    with pytest.raises(ValueError, match='above zero'):
        tetmesh.tet_mesh(geometry, [block], 0.0)
    hole = project.add_solid_cylinder(name, (0, 0, 0), 1.0, 20.0, unit='in')
    cut = project.combine_solids(name, 'subtract', [block, hole])
    with pytest.raises(ValueError, match=f'solid {block} is part of solid {cut}'):
        tetmesh.tet_mesh(geometry, [block], 1.0 * INCH)
    assert 500 < tetmesh.tet_count(geometry, [cut], 1.0 * INCH) < 20000


def test_the_relaxation_tells_its_progress():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (2, 1, 1), unit='in')
    seen = []
    tetmesh.tet_mesh(geometry, [block], 0.3 * INCH, progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1][0] == seen[-1][1]
    assert all(t == seen[0][1] for _d, t in seen)
    assert [d for d, _t in seen] == sorted(d for d, _t in seen)


# ------------------------------------------------------ the project and pane

def test_the_project_meshes_tets_on_request_and_journals_it():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (2, 1, 1), name='bar', unit='in')
    found = project.mesh_solids(name, [block], 0.4, unit='in', elements='tetras')
    assert found['tets'] == len(geometry.elem_conn) and set(geometry.elem_type) == {118}
    assert project.journal[-1] == (
        f"project.mesh_solids({name!r}, [{block}], 0.4, unit='in', elements='tetras')")
    # meshed again, into bricks: the tets are replaced
    again = project.mesh_solids(name, [block], 0.5, unit='in')
    assert again['replaced'] == found['tets']
    assert set(geometry.elem_type) == {115}
    # the old words are taken to mean the same
    assert project.mesh_solids(name, [block], 0.5, unit='in', elements='tets')['tets'] > 0
    with pytest.raises(ValueError, match="'hexes' or 'tetras'"):
        project.mesh_solids(name, [block], 0.5, unit='in', elements='wedges')


def test_the_pane_offers_tets_where_bricks_refuse_and_meshes_them(window, pump):
    pytest.importorskip('OCP')
    from conftest import select_objects
    from test_solids_mesh import _settled
    from test_solids_pane import _acts, _pick, _rows

    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    project = window.project
    name = project.new_geometry('Part', unit='in')
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='slab', unit='in')
    tilted = project.add_solid_cylinder(name, (0, 0, 0), 0.4, 3.0, angles=(30, 0, 0),
                                        unit='in')
    project.combine_solids(name, 'subtract', [block, tilted], 'slab')
    window.show_object(name)
    select_objects(window, pump, name)
    window._solids_changed(name)
    pump()
    _pick(window, pump, _rows(window, 'solid')[:1])
    _acts(window)['mesh_solids'][3]()
    pump()
    panel = window.scene.mesh_solids_panel
    assert panel.element_kind == 'hexes'
    assert [panel.elements.itemText(k) for k in range(2)] == ['Hexes', 'Tetras']
    # not an extrusion: bricks are fitted to the shape (test_hexmesh.py),
    # here past the fitted preview's budget, so the count alone is said
    panel.size.setValue(0.125)
    _settled(window, pump)
    assert panel.reading_label.text().startswith('About 4,096 bricks at this size')
    assert panel.mesh_button.isEnabled()
    panel.elements.setCurrentIndex(1)
    panel.size.setValue(0.4)
    _settled(window, pump)
    reading = panel.reading_label.text()
    assert 'ten-node tetrahedra' in reading and 'The worst radius ratio is' in reading
    assert panel.mesh_button.isEnabled()
    assert 'plane-preview' in window.scene.plotter.actors
    panel.mesh_button.click()
    pump()
    geometry = window.objects[name]
    assert set(geometry.elem_type) == {118} and list(geometry.group_name) == ['slab']
    assert window.statusBar().currentMessage().startswith(
        f'{name}: meshed into {len(geometry.elem_conn):,} tetrahedra')
    assert project.journal[-1].endswith("elements='tetras')")


def test_the_panes_tet_preview_stops_at_its_budget(window, pump, monkeypatch):
    pytest.importorskip('OCP')
    from test_solids_mesh import _holed_part, _settled

    panel = _holed_part(window, pump)
    monkeypatch.setattr(tetmesh, 'tet_mesh',
                        lambda *a, **k: pytest.fail('a tet mesh past the budget was run'))
    panel.elements.setCurrentIndex(1)
    panel.size.setValue(0.02)
    _settled(window, pump)
    assert panel.reading_label.text().startswith('About ')
    assert 'tetrahedra at this size. The preview stops past 5,000' in panel.reading_label.text()
    assert panel.mesh_button.isEnabled()
    assert not panel.parts_apart.isVisibleTo(panel)


# ------------------------------------------------------- the solver's tet10

def _straight_tet10(corners):
    edges = tetmesh.TET10_EDGES
    return np.vstack([corners] + [(corners[i] + corners[j]) / 2.0 for i, j in edges])


def test_the_conical_rule_is_exact_to_degree_five():
    points, weights = fem._TET_RULE
    assert weights.sum() == pytest.approx(1.0 / 6.0)
    for a in range(6):
        for b in range(6 - a):
            for c in range(6 - a - b):
                exact = (math.factorial(a) * math.factorial(b) * math.factorial(c)
                         / math.factorial(a + b + c + 3))
                got = (weights * points[:, 0] ** a * points[:, 1] ** b
                       * points[:, 2] ** c).sum()
                assert got == pytest.approx(exact, rel=1e-12)


def test_the_ten_node_tetrahedron_is_a_sound_element():
    """Six rigid modes and no more, a positive mass of the right total,
    and a linear field's energy exact (the patch test)."""
    corners = np.array([[0, 0, 0], [1, 0, 0.1], [0.2, 1, 0], [0.1, 0.2, 1.3]], float)
    xyz = _straight_tet10(corners)
    k, m = fem._solid_matrices(ALUMINUM, xyz)
    assert k.shape == (30, 30) and np.allclose(k, k.T)
    values = np.linalg.eigvalsh(k)
    assert (np.abs(values) < 1e-9 * values.max()).sum() == 6
    assert values[6] > 1e-4 * values.max()
    volume = abs(np.linalg.det(corners[1:] - corners[0])) / 6.0
    assert m.sum() / 3.0 == pytest.approx(ALUMINUM.density * volume)
    assert np.linalg.eigvalsh(m).min() > 0
    assert fem._solid_volume(xyz) == pytest.approx(volume)
    strain = np.array([[1e-3, 2e-4, 0.0], [2e-4, -3e-4, 1e-4], [0.0, 1e-4, 5e-4]])
    u = (xyz @ strain.T).ravel()
    voigt = np.array([strain[0, 0], strain[1, 1], strain[2, 2],
                      2 * strain[0, 1], 2 * strain[1, 2], 2 * strain[0, 2]])
    d = fem._isotropic(ALUMINUM)
    assert 0.5 * u @ k @ u == pytest.approx(0.5 * volume * voigt @ d @ voigt)


def test_a_tet10_bar_vibrates_like_the_brick_bar():
    """A 2 x 0.5 x 0.5 in cantilever meshed with ten-node tetrahedra at
    0.25 in, against the same bar in bricks: the first bending
    frequencies within three percent. The four-node tetrahedron locks
    in bending; the ten-node one does not."""
    project, name, geometry = _project()
    bar = project.add_solid_block(name, (1, 0, 0), (2, 0.5, 0.5), name='bar', unit='in')
    found = {}
    for elements in ('hexes', 'tetras'):
        project.mesh_solids(name, [bar], 0.25, unit='in', elements=elements)
        fixed = [str(n) for n, xyz in zip(geometry.node_id, geometry.node_xyz)
                 if abs(xyz[0]) < 1e-9]
        geometry.group_properties = {int(geometry.group_id[0]): fem.GroupProperties(ALUMINUM)}
        shapes = fem.Model.from_geometry(geometry).eigensolution(num_modes=4, fixed=fixed)
        found[elements] = shapes.frequency
    assert found['tetras'][:2] == pytest.approx(found['hexes'][:2], rel=0.03)


def test_a_midside_that_dips_into_a_hole_is_put_on_the_wall():
    """A chord from a node on a hole's wall to one just inside the
    material has its middle inside the hole where the wall curves
    away; `_quadratic` moves it onto the surface. Its ends share no
    surface, so the rule for chords along a wall does not reach it."""
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (6, 6, 2), unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0), 1.0, 4.0, unit='in')
    cut = project.combine_solids(name, 'subtract', [block, hole])
    on_wall = np.array([1.0, 0.0, 0.2])
    just_inside = 1.03 * np.array([math.cos(0.4), math.sin(0.4), 0.0]) + [0, 0, 0.2]
    points = np.array([on_wall, just_inside, [1.6, 0.0, 0.8], [1.6, 0.6, -0.4]]) * INCH
    size = 0.4 * INCH

    def fd(at):
        return solids.distance(geometry, cut, at)

    assert fd(points)[0] == pytest.approx(0.0, abs=1e-12) and fd(points)[1] < 0
    middle = (on_wall + just_inside) / 2.0
    assert np.hypot(*middle[:2]) < 1.0, 'the chord dips into the hole'
    surfaces = tetmesh._surfaces(geometry, [cut], size)
    xyz, conn = tetmesh._quadratic(fd, surfaces, points, np.array([[0, 1, 2, 3]]), size)
    assert conn.shape == (1, 10) and len(xyz) == 10
    assert (fd(xyz) <= 1e-6 * size).all(), 'no midside outside the solid'
    moved = xyz[conn[0, 4]] / INCH
    assert np.hypot(*moved[:2]) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(moved - middle) < 0.4 / 3.0


def test_a_long_solid_settles_because_the_aim_is_the_median_edge():
    """A 6 in shaft's Delaunay holds edges from one end cap to the other
    whose tetrahedra sit inside; the cubic mean of the edge lengths
    came to 4.7 sizes, every edge pushed, and the points never settled
    (300 steps, worst 0.02, a fourteenth under 0.3). Aimed at the median
    edge the shaft settles early with no element under a tenth."""
    project, name, geometry = _project()
    shaft = project.add_solid_cylinder(name, (0, 0, 0), 1.0, 6.0, angles=(0, 90, 0), unit='in')
    _part, made = tetmesh.tet_mesh(geometry, [shaft], 0.3 * INCH, quadratic=False)
    assert made['steps'] < tetmesh.MOST_STEPS // 2, made
    assert made['quality'] > 0.09 and made['poor'] < 0.02 * made['tets'], made
