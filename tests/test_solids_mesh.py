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
from visualdynamics.core.geometry import _CORNER_EDGES, Geometry

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
    # cylinders along two axes are an extrusion along one of them, the
    # other pulled round from the grid (test_solids_cross.py)
    _part, made = solids.extrusion_mesh(geometry, [a, b], 0.1 * INCH)
    assert made['axis'] == 0 and made['rings'] == 1
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

def _settled(window, pump):
    """Let the preview's settle timer fire, as a person's pause does."""
    for _ in range(200):
        if not window._mesh_settle.isActive():
            break
        pump()
        import time
        time.sleep(0.01)
    pump()


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
    _settled(window, pump)
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
    # meshed again: the pane says what it replaces, and the status what it did
    window.tree.clearSelection()
    _pick(window, pump, _rows(window, 'solid')[:1])
    _acts(window)['mesh_solids'][3]()
    pump()
    panel.size.setValue(0.1)
    _settled(window, pump)
    assert panel.reading_label.text().endswith(
        'It replaces the 3,840 elements meshed before.')
    panel.mesh_button.click()
    pump()
    assert 'replacing the 3,840 meshed before' in window.statusBar().currentMessage()
    assert len(geometry.elem_conn) == 3840


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


@pytest.mark.parametrize('x, ring', [(0.8, True), (0.84, True), (0.9, False)])
def test_a_ring_stops_at_the_parts_edge_with_room_for_a_layer(x, ring):
    """A hole near a plate's edge: the rectangle about it stops at the
    edge, and the hole is a ring while it fits inside the plate — 0.05
    in from the edge, or 0.01 (Brandon, 2026-10-12: a leg nearly flush
    with a table's corner), the circle held a tenth of a brick in where
    it comes closer. One breaking out through the edge is pulled round
    from the grid, no brick inside out."""
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
    bricks there are seen to be empty, so that rectangle is refused and
    the ring takes one clear of the window (2026-10-12; it was pulled
    round from the grid before) — and the window's faces stay on the
    grid's lines, exact."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0.125), (2, 2, 0.25), unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0.125), 0.3, 1.0, unit='in')
    window = project.add_solid_block(name, (0.66, 0.66, 0.125), (0.68, 0.68, 1.0),
                                     unit='in')
    part = project.combine_solids(name, 'subtract', [plate, hole, window], 'plate')
    mesh, made = solids.extrusion_mesh(geometry, [part], 0.1 * INCH)
    assert made['rings'] == 1
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
    _settled(window, pump)
    assert panel.reading_label.text().startswith('16 bricks, 54 nodes')
    panel.mesh_button.click()
    pump()
    assert window.project.journal[-1] == (
        "project.mesh_solids('Part', [1, 2], 0.5, unit='in', parts_apart=True)")


@pytest.mark.parametrize('radius', [4.0, 4.5])
def test_a_large_hole_in_a_table_is_a_ring_short_of_its_legs(radius):
    """Brandon's table (2026-10-12): a 12 in top on four 1 in legs at
    its corners, a large hole through the top. The ring's rectangle a
    quarter radius out reached the legs' columns — the grid's bricks
    there are leg in the leg layers — and the hole was pulled round from
    the grid, toothed. The largest rectangle short of the legs is taken
    instead, half an inch of top left between hole and leg at 4.5 in."""
    project, name, geometry = _project()
    top = project.add_solid_block(name, (0, 0, 0), (12, 12, 0.5), name='table',
                                  unit='in')
    legs = [project.add_solid_block(name, (sx * 5.5, sy * 5.5, -3.25), (1, 1, 6),
                                    unit='in')
            for sx in (-1, 1) for sy in (-1, 1)]
    hole = project.add_solid_cylinder(name, (0, 0, 0), radius, 1.0, unit='in')
    table = project.combine_solids(name, 'subtract', [
        project.combine_solids(name, 'union', [top, *legs]), hole])
    mesh, made = solids.extrusion_mesh(geometry, [table], 0.25 * INCH)
    assert made['rings'] == 1
    _on, uneven, inside = _on_circle(mesh, (0, 0), radius, 0.25)
    assert inside == 0 and uneven < 1.01
    assert (_corner_volumes(mesh) > 0).all()


def _table(project, name, legs=True):
    top = project.add_solid_block(name, (0, 0, 0), (12, 12, 0.5), name='table', unit='in')
    feet = [project.add_solid_block(name, (sx * 5.5, sy * 5.5, -3.25), (1, 1, 6), unit='in')
            for sx in (-1, 1) for sy in (-1, 1)] if legs else []
    hole = project.add_solid_cylinder(name, (0, 0, 0), 3.0, 1.0, unit='in')
    whole = project.combine_solids(name, 'union', [top, *feet]) if feet else top
    return project.combine_solids(name, 'subtract', [whole, hole]), top, feet, hole


def test_meshing_a_part_again_replaces_its_earlier_mesh():
    """Brandon's table (2026-10-12): meshed with its legs, the legs taken
    off, meshed again — the new bricks that landed on old ones were left
    out as already there, and the legs' bricks stayed. Now the part's
    earlier bricks go first, with the nodes only they used: the table is
    its top alone, its element group and material kept."""
    from visualdynamics.core.fem import GroupProperties, material

    project, name, geometry = _project()
    table, top, feet, hole = _table(project, name)
    first = project.mesh_solids(name, [table], 0.5, unit='in')
    assert first['replaced'] == 0
    group = int(geometry.group_id[0])
    geometry.group_properties[group] = GroupProperties(material('6061-T6'))
    project.separate_solids(name, [table])
    project.separate_solids(name, [solids.used_by(geometry)[top]])
    project.delete_solids(name, feet)
    again = project.combine_solids(name, 'subtract', [top, hole])
    found = project.mesh_solids(name, [again], 0.5, unit='in')
    assert found['replaced'] == first['bricks']
    assert len(geometry.elem_id) == found['bricks'] < first['bricks']
    used = {int(n) for conn in geometry.elem_conn for n in conn}
    assert used == {int(n) for n in geometry.node_id}, 'no node left from the legs'
    assert (geometry.node_xyz[:, 2] / INCH).min() >= -0.25 - 1e-9, 'nothing below the top'
    assert list(geometry.group_id) == [group] and \
        geometry.group_properties[group].material.name == '6061-T6'


def test_meshing_again_keeps_the_nodes_data_is_measured_at():
    """The earlier mesh's nodes go unless data uses them — the rule
    deleting elements follows (2026-10-12)."""
    from visualdynamics.core.shapes import ShapeSet

    project, name, geometry = _project()
    table, top, feet, hole = _table(project, name)
    project.mesh_solids(name, [table], 0.5, unit='in')
    row = int(np.argmin(geometry.node_xyz[:, 2]))
    lowest, where = int(geometry.node_id[row]), geometry.node_xyz[row].copy()
    project.add('Modes', ShapeSet([10.0], [0.01], [f'{lowest}Z+'], np.ones((1, 1))))
    project.separate_solids(name, [table])
    project.separate_solids(name, [solids.used_by(geometry)[top]])
    project.delete_solids(name, feet)
    project.mesh_solids(name, [project.combine_solids(name, 'subtract', [top, hole])],
                        0.5, unit='in')
    # the same node, where it was: an id alone could be a new node's,
    # numbered from one again once every old node had gone
    rows = np.flatnonzero(geometry.node_id == lowest)
    assert len(rows) == 1 and np.allclose(geometry.node_xyz[rows[0]], where), \
        'measured at, kept'
    assert (geometry.node_xyz[:, 2] < -0.25 * INCH - 1e-9).sum() == 1, 'and only it'


@pytest.mark.parametrize('size', [0.25, 0.5])
def test_a_hole_nearly_as_wide_as_the_table_steps_round_its_legs(size):
    """An 11 in hole in the 12 in top (Brandon, 2026-10-12): every
    rectangle round it took in the legs at its corners, and the hole was
    pulled round from the grid, toothed. The ring's outline steps round
    the legs instead — the grid's cells inside a disk clear of them — so
    the hole is round, no brick inside out, and the legs keep their
    square faces: their bricks are four 1 by 1 by 6 in posts exactly."""
    project, name, geometry = _project()
    top = project.add_solid_block(name, (0, 0, 0), (12, 12, 0.5), name='table', unit='in')
    legs = [project.add_solid_block(name, (sx * 5.5, sy * 5.5, -3.25), (1, 1, 6),
                                    unit='in')
            for sx in (-1, 1) for sy in (-1, 1)]
    hole = project.add_solid_cylinder(name, (0, 0, 0), 5.5, 1.0, unit='in')
    table = project.combine_solids(name, 'subtract', [
        project.combine_solids(name, 'union', [top, *legs]), hole])
    mesh, made = solids.extrusion_mesh(geometry, [table], size * INCH)
    assert made['rings'] == 1
    _on, _uneven, inside = _on_circle(mesh, (0, 0), 5.5, 0.25)
    assert inside == 0
    assert (_corner_volumes(mesh) > 0).all()
    points = _points(mesh)
    below = points[:, :, 2].max(axis=1) <= -0.25 + 1e-9
    posts = mesh.__class__(node_id=mesh.node_id, node_xyz=mesh.node_xyz,
                           elem_conn=[mesh.elem_conn[i] for i in np.flatnonzero(below)],
                           elem_type=[115] * int(below.sum()))
    assert _volume(posts) == pytest.approx(4 * 6.0)


def test_a_stepped_outline_leaves_a_tenth_of_a_brick():
    """The stepped outline is not held in from what it comes near as a
    rectangle is, so it needs that room: a hole 0.01 in short of the
    table's edges would put bricks of a few thousandths there, and it
    is pulled round from the grid instead, no brick edge under a tenth
    of a brick (2026-10-12)."""
    project, name, geometry = _project()
    table, _top, _feet, hole = _table(project, name)
    project.edit_solid(name, hole, radius=5.99, unit='in')
    mesh, made = solids.extrusion_mesh(geometry, [table], 0.25 * INCH)
    assert made['rings'] == 0
    assert _edge_lengths(mesh)[:, :8].min() > 0.025


@pytest.mark.parametrize('inset', [0.25, 0.05, 0.0])
def test_round_legs_are_round_up_to_flush_with_the_tables_corners(inset):
    """A 9 in hole in the 12 in top on four 1 in round legs (Brandon,
    2026-10-12). A leg 0.05 in from the top's edges was refused its ring
    for want of room, and a flush one, touching two edges, was pulled
    round from the grid, faceted. Now each leg is a ring: exactly round
    where it has a tenth of a brick of room, and held that far in from
    an edge it touches, flattened by no more there — no node outside
    its circle, the hole round past the legs, no brick inside out."""
    project, name, geometry = _project()
    at = 5.5 - inset
    top = project.add_solid_block(name, (0, 0, 0), (12, 12, 0.5), name='table', unit='in')
    legs = [project.add_solid_cylinder(name, (sx * at, sy * at, -3.25), 0.5, 6.0,
                                       unit='in')
            for sx in (-1, 1) for sy in (-1, 1)]
    hole = project.add_solid_cylinder(name, (0, 0, 0), 4.5, 1.0, unit='in')
    table = project.combine_solids(name, 'subtract', [
        project.combine_solids(name, 'union', [top, *legs]), hole])
    mesh, made = solids.extrusion_mesh(geometry, [table], 0.25 * INCH)
    assert made['rings'] == 5
    _on, _uneven, inside = _on_circle(mesh, (0, 0), 4.5, 0.25)
    assert inside == 0
    assert (_corner_volumes(mesh) > 0).all()
    xyz = mesh.node_xyz / INCH
    for z in (-1.0, -0.25, 0.25):
        layer = xyz[np.isclose(xyz[:, 2], z)]
        far = np.hypot(layer[:, 0] - at, layer[:, 1] - at)
        if z < -0.25:
            # below the top, the leg alone
            assert far[far < 1.0].max() < 0.5 + 1e-9, 'no node outside the leg'
        circle = far[(far > 0.45) & (far < 0.5 + 1e-9)]
        if inset >= 0.05:
            assert np.allclose(circle, 0.5, atol=1e-9), 'exactly round'
        else:
            assert circle.min() > 0.5 - 0.025 - 1e-9 and (np.abs(circle - 0.5) < 1e-9).sum() >= 12



def test_a_peanut_cut_is_meshed_to_the_unions_outline():
    """Two overlapping cylinders unioned and subtracted from a block
    (Brandon, 2026-10-18). No ring can be laid about either circle,
    since each has the other in its rectangle, so the rim is pulled
    onto the circles; at the two cusps the nearer circle's arc runs
    inside the other hole, and twelve bricks of material stood inside
    the cut with a sliver edge of 0.008 in beside them. A rim node is
    pulled to the union's outline: where its arc crosses the other
    circle, the crossing itself."""
    from visualdynamics.core import solids

    IN = 0.0254
    geometry = Geometry(node_id=[], node_xyz=np.zeros((0, 3)), elem_id=[],
                        elem_type=[], elem_conn=[], length_unit='in')
    block = solids.add_block(geometry, (0, 0, 0), (10 * IN, 10 * IN, 3 * IN))
    left = solids.add_cylinder(geometry, (-1.5 * IN, 0, 0), 3 * IN, 3 * IN)
    right = solids.add_cylinder(geometry, (1.5 * IN, 0, 0), 3 * IN, 3 * IN)
    peanut = solids.combine(geometry, 'union', [left, right])
    cut = solids.combine(geometry, 'subtract', [block, peanut])
    project = visualdynamics.Project()
    project.add('Geometry', geometry)
    made = project.mesh_solids('Geometry', [cut], 0.5, unit='in')
    xyz = np.asarray(geometry.node_xyz)
    index = {int(n): i for i, n in enumerate(geometry.node_id)}
    corners = np.array([[index[int(n)] for n in c] for c in geometry.elem_conn])
    centers = xyz[corners].mean(axis=1)
    assert not solids.contains(geometry, peanut, centers).any(), \
        'no brick of material inside the cut'
    cusp = np.sqrt(3.0 ** 2 - 1.5 ** 2) * IN
    at_cusps = (np.isclose(xyz[:, 0], 0.0, atol=1e-9)
                & np.isclose(np.abs(xyz[:, 1]), cusp, atol=1e-9))
    assert int(at_cusps.sum()) == 14, 'a node at each cusp in every layer'
    assert made['shortest'] > 0.15 * IN, 'no sliver bricks at the waist'
    # one ring about the union's outline, not two stair-stepped pulls:
    # every node on the rim lies on an arc of the outline, and the
    # bricks are all the right way out
    assert made['rings'] == 1
    rim = _rim_nodes(geometry, [(-1.5 * IN, 0.0), (1.5 * IN, 0.0)], 3 * IN)
    assert len(rim) >= 14 * 7, 'the rim has a node every brick or so, in every layer'
    assert (_corner_volumes(geometry) > 0).all()


def _rim_nodes(geometry, centers, radius, tol=1e-9):
    """The nodes on the outline of a union of equal circles: on one
    circle and not inside another."""
    xyz = np.asarray(geometry.node_xyz)
    far = np.stack([np.hypot(xyz[:, 0] - cx, xyz[:, 1] - cy) for cx, cy in centers])
    on_one = np.any(np.abs(far - radius) <= tol, axis=0)
    inside_another = np.any(far < radius - tol, axis=0)
    return np.flatnonzero(on_one & ~inside_another)


def test_the_peanut_rings_at_a_finer_size_too():
    from visualdynamics.core import solids

    IN = 0.0254
    geometry = Geometry(node_id=[], node_xyz=np.zeros((0, 3)), elem_id=[],
                        elem_type=[], elem_conn=[], length_unit='in')
    block = solids.add_block(geometry, (0, 0, 0), (10 * IN, 10 * IN, 3 * IN))
    left = solids.add_cylinder(geometry, (-1.5 * IN, 0, 0), 3 * IN, 3 * IN)
    right = solids.add_cylinder(geometry, (1.5 * IN, 0, 0), 3 * IN, 3 * IN)
    cut = solids.combine(geometry, 'subtract',
                         [block, solids.combine(geometry, 'union', [left, right])])
    project = visualdynamics.Project()
    project.add('Geometry', geometry)
    made = project.mesh_solids('Geometry', [cut], 0.25, unit='in')
    assert made['rings'] == 1
    assert (_corner_volumes(geometry) > 0).all()
    assert len(_rim_nodes(geometry, [(-1.5 * IN, 0.0), (1.5 * IN, 0.0)], 3 * IN)) > 14 * 13


def test_a_slot_of_three_circles_is_one_ring():
    """Three overlapping circles in a row, cut from a plate: one lobed
    outline, one ring, with a node at each of its four cusps."""
    from visualdynamics.core import solids

    IN = 0.0254
    geometry = Geometry(node_id=[], node_xyz=np.zeros((0, 3)), elem_id=[],
                        elem_type=[], elem_conn=[], length_unit='in')
    plate = solids.add_block(geometry, (0, 0, 0), (12 * IN, 6 * IN, 1 * IN))
    holes = [solids.add_cylinder(geometry, (x * IN, 0, 0), 1 * IN, 1 * IN)
             for x in (-1.5, 0.0, 1.5)]
    slot = solids.combine(geometry, 'union', holes)
    cut = solids.combine(geometry, 'subtract', [plate, slot])
    project = visualdynamics.Project()
    project.add('Geometry', geometry)
    made = project.mesh_solids('Geometry', [cut], 0.25, unit='in')
    assert made['rings'] == 1
    xyz = np.asarray(geometry.node_xyz)
    y = np.sqrt(1.0 ** 2 - 0.75 ** 2) * IN
    cusps = sum(int((np.isclose(xyz[:, 0], x * IN, atol=1e-9)
                     & np.isclose(np.abs(xyz[:, 1]), y, atol=1e-9)).sum())
                for x in (-0.75, 0.75))
    assert cusps == 4 * 5, 'four cusps in each of the five layers of nodes'
    assert not solids.contains(geometry, slot, _centers(geometry)).any()
    assert (_corner_volumes(geometry) > 0).all()


def _centers(geometry):
    xyz = np.asarray(geometry.node_xyz)
    index = {int(n): i for i, n in enumerate(geometry.node_id)}
    return np.array([xyz[[index[int(n)] for n in c]].mean(axis=0)
                     for c in geometry.elem_conn])


def test_a_circle_inside_another_is_not_a_lobe():
    """A small hole wholly inside a large one's disk (the two cut
    together) is no outline of its own: the large circle is ringed as a
    single hole and the small one adds nothing."""
    from visualdynamics.core import solids

    IN = 0.0254
    geometry = Geometry(node_id=[], node_xyz=np.zeros((0, 3)), elem_id=[],
                        elem_type=[], elem_conn=[], length_unit='in')
    plate = solids.add_block(geometry, (0, 0, 0), (8 * IN, 8 * IN, 1 * IN))
    big = solids.add_cylinder(geometry, (0, 0, 0), 2 * IN, 1 * IN)
    small = solids.add_cylinder(geometry, (0.5 * IN, 0, 0), 0.5 * IN, 1 * IN)
    cut = solids.combine(geometry, 'subtract',
                         [plate, solids.combine(geometry, 'union', [big, small])])
    project = visualdynamics.Project()
    project.add('Geometry', geometry)
    made = project.mesh_solids('Geometry', [cut], 0.25, unit='in')
    assert made['rings'] == 1
    assert (_corner_volumes(geometry) > 0).all()
    far = np.hypot(*np.asarray(geometry.node_xyz)[:, :2].T)
    assert int(np.isclose(far, 2 * IN, atol=1e-9).sum()) >= 5 * 32


def _holed_part(window, pump):
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
    return window.scene.mesh_solids_panel


def test_the_preview_waits_for_the_size_box_to_settle(window, pump, monkeypatch):
    """Each change of the size meshed and drew at once, so a spin of the
    box at a fine size was one mesh per step on this thread. Three quick
    changes are one mesh, once the box rests."""
    pytest.importorskip('OCP')
    panel = _holed_part(window, pump)
    meshed = []
    real = solids.extrusion_mesh

    def counted(*args, **kwargs):
        meshed.append(args[2])
        return real(*args, **kwargs)

    monkeypatch.setattr(solids, 'extrusion_mesh', counted)
    for value in (0.2, 0.15, 0.1):
        panel.size.setValue(value)
        pump()
    assert meshed == [], 'nothing meshed while the box is changing'
    assert window._mesh_settle.isActive()
    _settled(window, pump)
    assert len(meshed) == 1 and meshed[0] == pytest.approx(0.1 * 0.0254)
    assert panel.reading_label.text().startswith('3,840 bricks, ')


def test_a_preview_past_the_budget_is_counted_not_drawn(window, pump):
    """A fine size on a large part: the bricks are counted from the
    grid and said, the preview is taken down, and Mesh is what makes
    them."""
    pytest.importorskip('OCP')
    from visualdynamics.gui.main_window import MESH_PREVIEW_BRICKS

    panel = _holed_part(window, pump)
    assert 'plane-preview' in window.scene.plotter.actors
    panel.size.setValue(0.02)        # 200 x 100 x 25 cells
    _settled(window, pump)
    assert panel.reading_label.text() == (
        f'About 500,000 bricks at this size. The preview stops past '
        f'{MESH_PREVIEW_BRICKS:,}; press Mesh to make them.')
    assert 'plane-preview' not in window.scene.plotter.actors
    assert 500_000 == solids.grid_cells(window.objects['Part'], [3], 0.02 * 0.0254)


def test_mesh_runs_with_the_bar_up(window, pump, monkeypatch):
    """Mesh goes through the window's long-verb path: the bar is fed
    (done, total) from the mesher's stages and ends full."""
    pytest.importorskip('OCP')
    panel = _holed_part(window, pump)
    ticks = []
    monkeypatch.setattr(window, '_strip_tick',
                        lambda done, total: ticks.append((done, total)))
    panel.size.setValue(0.1)
    _settled(window, pump)
    panel.mesh_button.click()
    pump()
    assert len(window.objects['Part'].elem_conn) == 3840
    assert ticks and ticks[-1][0] == ticks[-1][1] > 0
    assert [d for d, _t in ticks] == sorted(d for d, _t in ticks)


def test_the_mesher_tells_its_progress():
    """`extrusion_mesh` ticks its stages: done never falls, never passes
    the total, and the last word is done == total."""
    from visualdynamics.core import solids

    project, name, geometry, cut = _holed_plate()
    ids = [cut]
    heard = []
    solids.extrusion_mesh(geometry, ids, 0.1 * 0.0254,
                          progress=lambda d, t: heard.append((d, t)))
    assert heard[-1][0] == heard[-1][1] == 4
    done = [d for d, _t in heard]
    assert done == sorted(done) and all(d <= t for d, t in heard)
    project.mesh_solids(name, ids, 0.1, unit='in',
                        progress=lambda d, t: heard.append((d, t)))
    assert heard[-1][0] == heard[-1][1]


def test_mesh_takes_the_size_as_typed_before_the_edit_is_finished(window, pump):
    """0.1 typed into the size box and Mesh pressed meshed at 0.25
    (Brandon, 2026-10-19): the box reports only finished edits, and a
    button takes no focus on macOS, so the click finished nothing.
    The apply reads the text as typed."""
    pytest.importorskip('OCP')
    panel = _holed_part(window, pump)
    assert panel.size.value() == pytest.approx(0.25)
    panel.size.lineEdit().setText('0.4')        # typed, not finished
    assert panel.size.value() == pytest.approx(0.25), 'the box still reports the old size'
    panel.mesh_button.click()
    pump()
    assert window.project.journal[-1].startswith("project.mesh_solids('Part', [")
    assert ', 0.4, unit=' in window.project.journal[-1]
