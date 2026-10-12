"""Mesh Solids: parts built from solids meshed into bricks (Brandon,
2026-10-11; `core.solids.extrusion_mesh`).

The grid is cut at every flat face, so flat faces mesh exactly; about
a circle with room the grid is swapped for a ring mesh, the circle's
nodes evenly spaced on it (2026-10-11, after a two-inch hole came out
jagged), and a circle without room, or a rounded corner's arc, has its
rim pulled onto it a node at a time, a pull that would fold a brick
undone. Pinned: the block and the windows to the arithmetic, the hole's
volume converging with no brick inside out at any size tried (the first
cut folded bricks into darts at a hole six across), the large hole's
ring, a washer about a hole as one ring, the fallbacks, the boss
rounded as the hole is, two parts meshed together sharing their nodes,
the refusals, the project's verb, and the pane.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core import solids
from visualdynamics.core.geometry import _CORNER_EDGES

INCH = 0.0254
IN3 = INCH ** 3


def _project():
    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    return project, name, project[name]


def _points(geometry):
    """(bricks, 8, 3) corners in inches, the nodes looked up once."""
    conn = np.array(geometry.elem_conn)
    return (geometry.node_xyz[geometry.node_index(conn.ravel())]
            .reshape(len(conn), 8, 3) / INCH)


def _corner_volumes(geometry):
    edges = np.asarray(_CORNER_EDGES[8])
    points = _points(geometry)
    legs = points[:, edges] - points[:, :, None, :]
    return np.einsum('mci,mci->mc', legs[:, :, 0],
                     np.cross(legs[:, :, 1], legs[:, :, 2]))


def _volume(geometry):
    """The bricks' volume, in³: each split into six tetrahedra about its
    first diagonal, exact for bricks with flat faces."""
    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6),
            (0, 4, 5, 6), (0, 5, 1, 6)]
    points = _points(geometry)
    total = 0.0
    for a, b, c, d in tets:
        total += np.abs(np.einsum('mi,mi->m', points[:, b] - points[:, a], np.cross(
            points[:, c] - points[:, a], points[:, d] - points[:, a]))).sum() / 6
    return total


def _holed_plate(radius=0.3):
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, -0.25), (4, 2, 0.5), name='plate',
                                    unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, -0.25), radius, 1.0, unit='in')
    return project, name, geometry, project.combine_solids(
        name, 'subtract', [plate, hole], 'plate')


def test_a_block_meshes_exactly_along_its_thinnest_axis():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 0.5), unit='in')
    part, made = solids.extrusion_mesh(geometry, [block], 0.25 * INCH)
    assert made['axis'] == 2, 'the thinnest'
    assert made['bricks'] == 16 * 8 * 2 and made['nodes'] == 17 * 9 * 3
    assert made['rounded'] == 0
    assert set(part.elem_type) == {115}, 'eight-node bricks'
    assert _volume(part) == pytest.approx(4.0)
    assert (_corner_volumes(part) > 0).all(), 'every brick the right way out'
    assert list(part.group_name) == ['Block 1']


def test_flat_faces_fall_on_grid_lines():
    """A window cut from a plate off the size's grid: the grid is cut at
    its faces, so the window is exact and no brick straddles it."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0), (4, 3, 0.5), unit='in')
    window = project.add_solid_block(name, (0.13, 0.07, 0), (1.37, 1.11, 1.0),
                                     unit='in')
    framed = project.combine_solids(name, 'subtract', [plate, window], 'framed')
    part, _made = solids.extrusion_mesh(geometry, [framed], 0.25 * INCH)
    assert _volume(part) == pytest.approx((12 - 1.37 * 1.11) * 0.5)


@pytest.mark.parametrize('size', [0.15, 0.1, 0.05, 0.03])
def test_a_hole_is_rounded_without_folding_a_brick(size):
    _unused, _name, geometry, plate = _holed_plate()
    part, made = solids.extrusion_mesh(geometry, [plate], size * INCH)
    assert made['rounded'] > 0
    assert (_corner_volumes(part) > 0).all(), 'no brick inside out'
    exact = (4 * 2 - np.pi * 0.3 ** 2) * 0.5
    assert _volume(part) == pytest.approx(exact, rel=0.004)
    # a pull undone so as not to fold a brick leaves its node where it
    # was: within a brick of the circle, never further in
    radius = np.hypot(*(part.node_xyz[:, :2] / INCH).T)
    assert radius.min() >= 0.3 - size and np.isclose(radius, 0.3).sum() > 8


def test_a_boss_is_rounded_as_a_hole_is():
    """A cylinder unioned onto a plate: the part inside the circle, the
    empty space beyond it — a boundary the circle draws, rounded."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0), (2, 2, 0.25), unit='in')
    boss = project.add_solid_cylinder(name, (0, 0, 0.375), 0.4, 0.5, unit='in')
    part = project.combine_solids(name, 'union', [plate, boss], 'bossed')
    mesh, made = solids.extrusion_mesh(geometry, [part], 0.1 * INCH)
    assert made['rounded'] > 0 and (_corner_volumes(mesh) > 0).all()
    exact = 2 * 2 * 0.25 + np.pi * 0.4 ** 2 * 0.5
    assert _volume(mesh) == pytest.approx(exact, rel=0.01)


def test_parts_meshed_together_share_their_nodes_where_they_touch():
    """A plate with a hole drilled through and an insert in its top
    third: one grid, the insert's bricks in a group of their own sharing
    the hole's wall with the plate's, nothing below the insert."""
    project, name, geometry, plate = _holed_plate(0.25)
    insert = project.add_solid_cylinder(name, (0, 0, -1 / 12), 0.25, 1 / 6,
                                        name='insert', unit='in')
    part, _made = solids.extrusion_mesh(geometry, [plate, insert], 0.05 * INCH)
    ids = dict(zip(part.group_name, part.group_id.tolist()))
    assert list(ids) == ['plate', 'insert']
    shared = set(part.nodes_in([ids['plate']])) & set(part.nodes_in([ids['insert']]))
    assert len(shared) > 20
    insert_rows = part.node_index(part.nodes_in([ids['insert']]))
    assert (part.node_xyz[insert_rows, 2] / INCH).min() == pytest.approx(-1 / 6)
    assert (_corner_volumes(part) > 0).all()
    # the hole's wall is the ring's circle, and the insert has a core
    wall = part.node_xyz[part.node_index(sorted(shared)), :2] / INCH
    assert np.allclose(np.hypot(*wall.T), 0.25)
    inside = part.node_xyz[insert_rows, :2] / INCH
    assert np.isclose(np.hypot(*inside.T), 0.0).any(), 'a node at the middle'


def test_what_is_not_an_extrusion_is_refused_by_name():
    project, name, geometry = _project()
    turned = project.add_solid_block(name, (0, 0, 0), (1, 1, 1), angles=(0, 0, 30),
                                     unit='in')
    with pytest.raises(ValueError, match=f"solid {turned} is turned off the geometry's axes"):
        solids.extrusion_mesh(geometry, [turned], 0.1 * INCH)
    a = project.add_solid_cylinder(name, (5, 0, 0), 0.2, 1.0, unit='in')
    b = project.add_solid_cylinder(name, (8, 0, 0), 0.2, 1.0, angles=(0, 90, 0),
                                   unit='in')
    with pytest.raises(ValueError, match='the cylinders and rounded corners run along different axes'):
        solids.extrusion_mesh(geometry, [a, b], 0.1 * INCH)
    both = project.combine_solids(name, 'union', [a, b])
    with pytest.raises(ValueError, match=f'solid {a} is part of solid {both}'):
        solids.extrusion_mesh(geometry, [a], 0.1 * INCH)
    with pytest.raises(ValueError, match='an element size above zero'):
        solids.extrusion_mesh(geometry, [turned], 0.0)
    plain = project.add_solid_block(name, (0, 0, 9), (4, 4, 4), unit='in')
    with pytest.raises(ValueError, match='bricks at that size: use a larger one'):
        solids.extrusion_mesh(geometry, [plain], 0.01 * INCH)


def test_the_project_meshes_journals_and_keeps_apart():
    """The verb at the default size (half the thinnest feature) joins
    the bricks to the geometry; a second part kept apart lies on the
    first's nodes without sharing them."""
    project, name, geometry, plate = _holed_plate()
    assert solids.mesh_size(geometry, [plate]) / INCH == pytest.approx(0.25)
    found = project.mesh_solids(name, [plate], 0.1, unit='in')
    assert found['bricks'] == len(geometry.elem_conn) and found['shared'] == 0
    assert project.journal[-1] == "project.mesh_solids('Part', [3], 0.1, unit='in')"
    assert list(geometry.group_name) == ['plate']
    cap = project.add_solid_block(name, (0, 1.25, 0.25), (4, 0.5, 0.5), name='cap',
                                  unit='in')
    found = project.mesh_solids(name, [cap], 0.1, unit='in', apart=True)
    assert found['shared'] == 0
    ids = dict(zip(geometry.group_name, geometry.group_id.tolist()))
    assert len(geometry.coincident_spots(1e-9, [ids['plate'], ids['cap']])) > 0


# ---- the pane ---------------------------------------------------------------

def test_the_pane_previews_the_bricks_and_meshes(window, pump):
    pytest.importorskip('OCP')
    from conftest import select_objects
    from test_solids_pane import _acts, _pick, _rows

    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    project = window.project
    project.new_geometry('Part', unit='in')
    plate = project.add_solid_block('Part', (0, 0, -0.25), (4, 2, 0.5),
                                    name='plate', unit='in')
    hole = project.add_solid_cylinder('Part', (0, 0, -0.25), 0.3, 1.0, unit='in')
    project.combine_solids('Part', 'subtract', [plate, hole], 'plate')
    window.show_object('Part')
    select_objects(window, pump, 'Part')
    window._solids_changed('Part')
    pump()
    _pick(window, pump, _rows(window, 'solid')[:1])
    _acts(window)['mesh_solids'][3]()
    pump()
    panel = window.scene.mesh_solids_panel
    assert panel.isVisibleTo(window)
    assert panel.size_label.text() == 'Element size [in]'
    assert panel.size.value() == pytest.approx(0.25)
    assert panel.parts_label.text() == 'Solids: plate'
    assert 'plane-preview' in window.scene.plotter.actors
    panel.size.setValue(0.1)
    pump()
    assert panel.reading_label.text().startswith('3,840 bricks, ')
    # the ring's 32 on the circle (its rectangle eight bricks a side) in
    # each of the six layers of nodes
    assert 'extruded along Z. 192 nodes on circles.' in panel.reading_label.text()
    panel.mesh_button.click()
    pump()
    geometry = window.objects['Part']
    assert len(geometry.elem_conn) == 3840 and list(geometry.group_name) == ['plate']
    assert not panel.isVisibleTo(window), 'put away once meshed'
    assert 'meshed into 3,840 bricks' in window.statusBar().currentMessage()
    assert window.project.journal[-1] == (
        "project.mesh_solids('Part', [3], 0.1, unit='in')")


def test_a_circle_inside_one_material_draws_no_boundary():
    """A cylinder unioned wholly inside a plate leaves the plate's
    material on both sides of its circle: no boundary there, so no node
    is pulled onto it — the mesh stays the plain grid."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0), (2, 2, 0.5), unit='in')
    inner = project.add_solid_cylinder(name, (0, 0, 0), 0.4, 0.5, unit='in')
    part = project.combine_solids(name, 'union', [plate, inner], 'plate')
    _mesh, made = solids.extrusion_mesh(geometry, [part], 0.1 * INCH)
    assert made['rounded'] == 0 and made['rings'] == 0
    assert made['bricks'] == 20 * 20 * 5


def test_a_window_with_rounded_corners_leaves_round_fillets():
    """A window cut with a corner radius: the frame left around it has a
    fillet in each corner, meshed round as a hole is — its arcs drawn
    by the corner circles, no brick inside out."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0), (5, 6, 0.5), unit='in')
    window = project.add_solid_block(name, (0, 0, 0), (3.375, 5, 1.0), corner=0.138,
                                     unit='in')
    frame = project.combine_solids(name, 'subtract', [plate, window], 'frame')
    part, made = solids.extrusion_mesh(geometry, [frame], INCH / 12)
    assert made['rounded'] > 0 and (_corner_volumes(part) > 0).all()
    exact = (5 * 6 - 3.375 * 5 + (4 - np.pi) * 0.138 ** 2) * 0.5
    assert _volume(part) == pytest.approx(exact, rel=0.002)


def test_once_meshed_the_solids_draw_only_when_picked(window, pump):
    """Over a mesh whose modes animate, see-through solids are in the
    way: once a geometry has elements they are drawn only with the Solids
    row or a solid picked (2026-10-11)."""
    pytest.importorskip('OCP')
    from conftest import select_objects
    from test_solids_pane import _pick, _rows

    project = window.project
    project.new_geometry('Part', unit='in')
    block = project.add_solid_block('Part', (0, 0, 0), (2, 2, 0.5), unit='in')
    window.show_object('Part')
    select_objects(window, pump, 'Part')
    assert 'solid-1' in window.scene.plotter.actors, 'nothing meshed yet'
    project.mesh_solids('Part', [block], 0.25, unit='in')
    window._solids_changed('Part')
    select_objects(window, pump, 'Part')
    assert 'solid-1' not in window.scene.plotter.actors
    _pick(window, pump, _rows(window, 'solids'))
    assert 'solid-1' in window.scene.plotter.actors


def test_where_parts_overlap_the_later_one_takes_the_bricks():
    """A steel screw added over an insert: meshed after it, it takes the
    bricks it overlaps — how the Frame-Wing's screws are threaded into
    their inserts (2026-10-11) — and meshed before it, the insert keeps
    them."""
    project, name, geometry = _project()
    insert = project.add_solid_cylinder(name, (0, 0, 0), 0.3, 0.5, name='insert',
                                        unit='in')
    screw = project.add_solid_cylinder(name, (0, 0, 0.125), 0.3, 0.25, name='screw',
                                       unit='in')
    later, made = solids.extrusion_mesh(geometry, [insert, screw], 0.1 * INCH)
    assert made['rings'] == 1, 'a lone cylinder: its ring stops at the grid'
    earlier, _made = solids.extrusion_mesh(geometry, [screw, insert], 0.1 * INCH)
    assert len(later.elements_in('screw')) == len(earlier.elements_in('insert')) / 2
    assert len(earlier.elements_in('screw')) == 0


def _on_circle(part, center, radius, z):
    """In the layer of nodes at `z` (inches): how many lie on the circle,
    the widest angle between neighbors on it over the narrowest, and
    how many lie inside it."""
    xyz = part.node_xyz / INCH
    layer = np.isclose(xyz[:, 2], z)
    far = np.hypot(xyz[:, 0] - center[0], xyz[:, 1] - center[1])
    on = layer & np.isclose(far, radius, atol=1e-9)
    angles = np.sort(np.arctan2(xyz[on, 1] - center[1], xyz[on, 0] - center[0]))
    gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
    return int(on.sum()), gaps.max() / gaps.min(), int((layer & (far < radius - 1e-9)).sum())


def _edge_lengths(part):
    """Every brick's twelve edges, inches."""
    points = _points(part)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.stack([np.linalg.norm(points[:, a] - points[:, b], axis=1)
                     for a, b in pairs], axis=1)


def _scaled_jacobian(part):
    """The worst brick's worst corner: the triple product of its three
    unit edges, 1 for a cube's corner, 0 for a flat one."""
    edges = np.asarray(_CORNER_EDGES[8])
    points = _points(part)
    legs = points[:, edges] - points[:, :, None, :]
    legs /= np.linalg.norm(legs, axis=3, keepdims=True)
    return np.einsum('mci,mci->mc', legs[:, :, 0],
                     np.cross(legs[:, :, 1], legs[:, :, 2])).min()


@pytest.mark.parametrize('size', [0.1, 0.25])
def test_a_large_hole_is_a_ring_of_evenly_spaced_nodes(size):
    """Brandon's two-inch hole (2026-10-11): pulled onto its circle from
    the grid, a pull that would fold a brick was not made and its node
    stayed inside the hole — 23 of the rim's 140 a layer at a tenth of
    an inch, the hole's edge toothed — and the staircase put twice the
    nodes on the circle at 45 degrees as along the grid. As a ring mesh:
    no node inside, the nodes on the circle evenly spaced (the hole off
    the grid's center, so the rectangle about it is too), the worst
    brick's corner well short of flat (it was 0.43), the volume to the
    arithmetic."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, -0.25), (12, 8, 0.5), name='plate',
                                    unit='in')
    hole = project.add_solid_cylinder(name, (0.37, 0.21, -0.25), 2.0, 1.0, unit='in')
    part = project.combine_solids(name, 'subtract', [plate, hole], 'plate')
    mesh, made = solids.extrusion_mesh(geometry, [part], size * INCH)
    on, uneven, inside = _on_circle(mesh, (0.37, 0.21), 2.0, 0.0)
    assert made['rings'] == 1
    assert inside == 0 and on > 4 * np.pi / size
    assert uneven < 1.1
    assert made['rounded'] == on * len(np.unique(np.round(mesh.node_xyz[:, 2], 12)))
    assert _scaled_jacobian(mesh) > 0.65
    # layers of about the size between the circle and the grid, so no
    # brick is long; and the shortest edge read off all twelve of every
    # brick's (a ring's short edge is on the circle)
    lengths = _edge_lengths(mesh)
    assert lengths.max() < 2.0 * size
    assert made['shortest'] / INCH == pytest.approx(lengths.min())
    assert _volume(mesh) == pytest.approx((96 - np.pi * 4) * 0.5, rel=2e-4)


@pytest.mark.parametrize('size, between', [(0.1, []), (0.05, [0.15])])
def test_a_washer_about_a_hole_is_one_ring(size, between):
    """A plate with a hole and a washer under it about the same center,
    meshed together: one ring with both circles in it, so the washer's
    bricks lie between the circles exactly and share the plate's nodes
    on both, in as many layers as the size asks — one at a tenth of an
    inch, two at a twentieth, their nodes on the middle circle."""
    x, y = 0.13, 0.07
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0.125), (2, 2, 0.25), unit='in')
    hole = project.add_solid_cylinder(name, (x, y, 0.125), 0.1, 1.0, unit='in')
    holed = project.combine_solids(name, 'subtract', [plate, hole], 'plate')
    disk = project.add_solid_cylinder(name, (x, y, -0.015), 0.2, 0.03, unit='in')
    bore = project.add_solid_cylinder(name, (x, y, -0.015), 0.1, 0.1, unit='in')
    washer = project.combine_solids(name, 'subtract', [disk, bore], 'washer')
    mesh, made = solids.extrusion_mesh(geometry, [holed, washer], size * INCH)
    assert made['rings'] == 1
    rows = mesh.node_index(mesh.nodes_in([dict(zip(mesh.group_name,
                                                   mesh.group_id.tolist()))['washer']]))
    far = np.hypot(*(mesh.node_xyz[rows, :2] / INCH - (x, y)).T)
    assert np.unique(np.round(far, 9)).tolist() == pytest.approx([0.1, *between, 0.2])
    # off the grid's lines, the rectangle is off the circles' center and
    # each side takes its own share of them: even along a side, not
    # quite from one side to the next
    for radius in (0.1, 0.2):
        on, uneven, _inside = _on_circle(mesh, (x, y), radius, -0.03)
        assert on > 8 and uneven < 1.2
    assert (_corner_volumes(mesh) > 0).all()
    assert _volume(mesh) == pytest.approx(
        (4 - np.pi * 0.01) * 0.25 + np.pi * (0.04 - 0.01) * 0.03, rel=0.005)


@pytest.mark.parametrize('x, ring', [(0.8, True), (0.84, False)])
def test_a_ring_stops_at_the_parts_edge_with_room_for_a_layer(x, ring):
    """A hole near a plate's edge: the rectangle about it stops at the
    edge, a ring if a seventh of the radius is left for a layer of
    bricks (the Frame-Wing's screws, a quarter inch from its wing's
    edge), else the rim pulled round from the grid — its spacing as
    uneven as the staircase it came from, no brick inside out."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0.125), (2, 2, 0.25), unit='in')
    hole = project.add_solid_cylinder(name, (x, 0, 0.125), 0.15, 1.0, unit='in')
    part = project.combine_solids(name, 'subtract', [plate, hole], 'plate')
    mesh, made = solids.extrusion_mesh(geometry, [part], 0.1 * INCH)
    assert made['rings'] == int(ring)
    assert (_corner_volumes(mesh) > 0).all()


def test_holes_too_close_for_rings_are_pulled_round():
    """Two holes whose rectangles would take in each other's circle:
    neither is ringed, both rounded from the grid, nothing folded."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0.125), (3, 2, 0.25), unit='in')
    holes = [project.add_solid_cylinder(name, (x, 0, 0.125), 0.1, 1.0, unit='in')
             for x in (-0.13, 0.13)]
    part = project.combine_solids(name, 'subtract', [plate, *holes], 'plate')
    mesh, made = solids.extrusion_mesh(geometry, [part], 0.1 * INCH)
    assert made['rings'] == 0
    assert made['rounded'] > 0 and (_corner_volumes(mesh) > 0).all()


def test_a_face_clipping_a_rings_rectangle_keeps_the_grid():
    """A window whose corner reaches into the rectangle a ring about a
    hole would take, outside the band the ring samples: the grid's own
    bricks there are seen to be empty, so the hole is pulled round from
    the grid and the window's faces stay on its lines, exact."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0.125), (2, 2, 0.25), unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0.125), 0.3, 1.0, unit='in')
    window = project.add_solid_block(name, (0.66, 0.66, 0.125), (0.68, 0.68, 1.0),
                                     unit='in')
    part = project.combine_solids(name, 'subtract', [plate, hole, window], 'plate')
    mesh, made = solids.extrusion_mesh(geometry, [part], 0.1 * INCH)
    assert made['rings'] == 0 and made['rounded'] > 0
    # no node inside the window, so its faces are the grid's lines
    xyz = mesh.node_xyz / INCH
    assert not ((xyz[:, 0] > 0.32 + 1e-9) & (xyz[:, 1] > 0.32 + 1e-9)).any()
    assert (_corner_volumes(mesh) > 0).all()


def test_an_insert_off_the_grids_center_has_its_core_about_its_center():
    """A hole off the grid's lines with an insert in its top half: the
    insert's core is a grid about the circle's own center, so its
    nodes there average to that center, and layers of about the size
    run from it to the circle, so no brick is long."""
    x, y = 0.13, 0.07
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, -0.25), (2, 2, 0.5), name='plate',
                                    unit='in')
    hole = project.add_solid_cylinder(name, (x, y, -0.25), 0.3, 1.0, unit='in')
    holed = project.combine_solids(name, 'subtract', [plate, hole], 'plate')
    insert = project.add_solid_cylinder(name, (x, y, -0.125), 0.3, 0.25, name='insert',
                                        unit='in')
    mesh, made = solids.extrusion_mesh(geometry, [holed, insert], 0.05 * INCH)
    assert made['rings'] == 1 and (_corner_volumes(mesh) > 0).all()
    assert _edge_lengths(mesh).max() < 2 * 0.05
    rows = mesh.node_index(mesh.nodes_in([dict(zip(mesh.group_name,
                                                   mesh.group_id.tolist()))['insert']]))
    xy = mesh.node_xyz[rows, :2] / INCH - (x, y)
    # the core is a square half the circle across: its inside, clear of
    # the rings about it
    core = xy[np.abs(xy).max(axis=1) < 0.1]
    assert len(core) and np.allclose(core.mean(axis=0), 0.0, atol=1e-9)


def test_a_pattern_meshes_every_copy():
    """Six pegs, three along +X and two along -Y, a pattern meshed alone:
    the grid spans the pattern's extent, every copy's both ways, and each
    copy is a ring of its own — six cylinders' volume, to the polygons
    the rings make."""
    project, name, geometry = _project()
    peg = project.add_solid_cylinder(name, (0, 0, 0), 0.3, 0.5, name='pegs',
                                     unit='in')
    grid = project.pattern_solids(name, [peg], (3, 2, 1), (1, -1, 0), unit='in')
    mesh, made = solids.extrusion_mesh(geometry, [grid], 0.1 * INCH)
    assert made['rings'] == 6 and list(mesh.group_name) == ['pegs']
    low, high = mesh.node_xyz.min(axis=0) / INCH, mesh.node_xyz.max(axis=0) / INCH
    # to the rings' polygons: a node need not sit at a circle's extreme
    assert np.allclose(low[:2], (-0.3, -1.3), atol=0.002)
    assert np.allclose(high[:2], (2.3, 0.3), atol=0.002)
    assert _volume(mesh) == pytest.approx(6 * np.pi * 0.09 * 0.5, rel=0.02)
    assert (_corner_volumes(mesh) > 0).all()


def test_parts_kept_apart_share_a_grid_and_no_nodes():
    """Two blocks side by side, meshed together: they share the face where
    they touch. Kept apart from each other (the BARC's parts, 2026-10-11),
    the same bricks on the same grid, each part with nodes of its own —
    the shared face's 3 x 3 nodes twice — every one of them on one of the
    other part's, for bolts to be merged at."""
    project, name, geometry = _project()
    left = project.add_solid_block(name, (0.5, 0, 0), (1, 1, 1), name='left',
                                   unit='in')
    right = project.add_solid_block(name, (1.5, 0, 0), (1, 1, 1), name='right',
                                    unit='in')
    _together, made = solids.extrusion_mesh(geometry, [left, right], 0.5 * INCH)
    apart, made_apart = solids.extrusion_mesh(geometry, [left, right], 0.5 * INCH,
                                              parts_apart=True)
    assert made_apart['bricks'] == made['bricks'] == 16
    assert made_apart['nodes'] == made['nodes'] + 9 == 45 + 9
    ids = dict(zip(apart.group_name, apart.group_id.tolist()))
    one = set(apart.nodes_in([ids['left']]))
    other = set(apart.nodes_in([ids['right']]))
    assert not one & other, 'no node shared'
    face = apart.node_xyz[apart.node_index(sorted(one))]
    face = face[np.isclose(face[:, 0], INCH)]
    there = apart.node_xyz[apart.node_index(sorted(other))]
    assert len(face) == 9 and all(np.isclose(there, p).all(axis=1).any() for p in face)


def test_the_pane_keeps_two_parts_apart_when_asked(window, pump):
    """Keep the parts apart shows with two parts picked, not one, and
    Mesh passes it on."""
    pytest.importorskip('OCP')
    from conftest import select_objects
    from test_solids_pane import _acts, _pick, _rows

    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    project = window.project
    project.new_geometry('Part', unit='in')
    project.add_solid_block('Part', (0.5, 0, 0), (1, 1, 1), name='left', unit='in')
    project.add_solid_block('Part', (1.5, 0, 0), (1, 1, 1), name='right', unit='in')
    window.show_object('Part')
    select_objects(window, pump, 'Part')
    window._solids_changed('Part')
    pump()
    panel = window.scene.mesh_solids_panel
    _pick(window, pump, _rows(window, 'solid')[:1])
    _acts(window)['mesh_solids'][3]()
    pump()
    assert panel.parts_apart.isHidden(), 'one part has nothing to keep apart from'
    window.close_mesh_solids_pane()
    _pick(window, pump, _rows(window, 'solid'))
    _acts(window)['mesh_solids'][3]()
    pump()
    assert not panel.parts_apart.isHidden()
    panel.size.setValue(0.5)
    pump()
    assert panel.reading_label.text().startswith('16 bricks, 45 nodes')
    panel.parts_apart.setChecked(True)
    pump()
    assert panel.reading_label.text().startswith('16 bricks, 54 nodes')
    panel.mesh_button.click()
    pump()
    assert window.project.journal[-1] == (
        "project.mesh_solids('Part', [1, 2], 0.5, unit='in', parts_apart=True)")
