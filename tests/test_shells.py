"""Plates, disks and shells (Brandon, 2026-10-26): a plate and a disk on
the Solids row, flat, and Mesh Solids' Shells (quads) and Shells
(triangles), on a flat part's plane or a solid's outer surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import shells, solids
from visualdynamics.core.geometry import Geometry
from visualdynamics.rotate import frame_from_angles

pytest.importorskip('OCP')

INCH = 0.0254


def _empty():
    return Geometry(node_id=[], node_xyz=np.empty((0, 3)))


def _edges(conn):
    """Each edge of these faces, by its two node ids, with how many faces
    hold it."""
    count: dict[tuple[int, int], int] = {}
    for face in conn:
        face = [int(n) for n in face]
        for a, b in zip(face, face[1:] + face[:1]):
            key = (min(a, b), max(a, b))
            count[key] = count.get(key, 0) + 1
    return count


# ---- the solids ---------------------------------------------------------------


def test_a_plate_and_a_disk_are_flat_faces_with_an_area():
    g = _empty()
    plate = solids.add_plate(g, (0, 0, 0), (4, 2), name='skin')
    disk = solids.add_disk(g, (0, 0, 5), 1.0, angles=(90, 0, 0))
    hole = solids.add_cylinder(g, (1, 0, 0), 0.25, 1.0)
    cut = solids.combine(g, 'subtract', [plate, hole])
    assert solids.solid(g, plate).widths == (4.0, 2.0, 0.0)
    assert solids.area(g, disk) == pytest.approx(np.pi)
    assert solids.area(g, cut) == pytest.approx(8 - np.pi / 16)
    assert solids.volume(g, cut) == pytest.approx(0.0, abs=1e-12)
    assert solids.is_flat(g, cut) and solids.is_flat(g, disk)
    assert not solids.is_flat(g, hole)
    solids.edit(g, plate, widths=(5, 3))
    assert solids.area(g, cut) == pytest.approx(15 - np.pi / 16), 'the cut follows'
    solids.edit(g, disk, radius=2.0)
    assert solids.area(g, disk) == pytest.approx(4 * np.pi)
    with pytest.raises(ValueError, match='a plate needs a width along its X and its Y'):
        solids.add_plate(g, (0, 0, 0), (4, 0))
    with pytest.raises(ValueError, match='a disk needs a radius'):
        solids.add_disk(g, (0, 0, 0), 0.0)


def test_a_union_of_flat_and_solid_is_refused_and_nothing_flat_is_cut_away():
    g = _empty()
    plate = solids.add_plate(g, (0, 0, 0), (4, 2))
    block = solids.add_block(g, (0, 0, 2), (1, 1, 1))
    disk = solids.add_disk(g, (0, 0, 0), 0.5)
    with pytest.raises(ValueError, match='a union is of plates and disks, or of solids'):
        solids.combine(g, 'union', [plate, block])
    with pytest.raises(ValueError, match='is flat and has no volume to take away'):
        solids.combine(g, 'subtract', [block, disk])
    joined = solids.combine(g, 'union', [plate, disk])
    assert solids.is_flat(g, joined)


def test_a_plate_saves_and_reads_back():
    g = _empty()
    plate = solids.solid(g, solids.add_plate(g, (1, 2, 3), (4, 2), (10, 20, 30), 'skin'))
    assert solids.Solid.from_record(plate.as_record()) == plate


# ---- shells of a flat part ------------------------------------------------------


def test_a_plate_meshes_into_quads_on_its_plane():
    g = _empty()
    solids.add_plate(g, (0, 0, 0), (4, 2), name='skin')
    part, made = shells.shell_mesh(g, [1], 0.5)
    assert made['elements'] == 8 * 4 and made['nodes'] == 9 * 5
    assert set(part.elem_type.tolist()) == {shells.QUAD}
    assert np.abs(part.node_xyz[:, 2]).max() == 0.0
    assert list(part.group_name) == ['skin']
    # every face the right way up: the plate's own Z
    xyz = part.node_xyz
    index = {int(n): k for k, n in enumerate(part.node_id)}
    for face in part.elem_conn:
        p = xyz[[index[int(n)] for n in face]]
        assert np.cross(p[2] - p[0], p[3] - p[1])[2] > 0.0


def test_a_turned_plate_with_a_hole_meshes_on_its_plane_round_the_hole():
    g = _empty()
    turn, at = (30, 20, 10), np.array([1.0, 2.0, 3.0])
    plate = solids.add_plate(g, at, (4, 2), angles=turn, name='skin')
    hole = solids.add_cylinder(g, at + (0.0, 0.0, 0.0), 0.4, 0.05, angles=turn)
    cut = solids.combine(g, 'subtract', [plate, hole])
    part, _made = shells.shell_mesh(g, [cut], 0.2)
    local = (part.node_xyz - at) @ frame_from_angles(turn)[:3].T
    assert np.abs(local[:, 2]).max() < 1e-12, 'on the plate\'s plane'
    radius = np.hypot(local[:, 0], local[:, 1])
    assert radius.min() == pytest.approx(0.4), 'the hole round, not stepped'
    assert int(np.sum(np.isclose(radius, 0.4))) >= 16
    assert np.abs(local[:, 0]).max() == pytest.approx(2.0)
    assert np.abs(local[:, 1]).max() == pytest.approx(1.0)


def test_a_disk_meshes_round_and_vibrates_as_a_free_circular_plate():
    from visualdynamics.core.fem import GroupProperties, Material, Model

    g = _empty()
    solids.add_disk(g, (0, 0, 0), 1.0, name='disk')
    part, _made = shells.shell_mesh(g, [1], 0.1)
    radius = np.hypot(part.node_xyz[:, 0], part.node_xyz[:, 1])
    assert radius.max() == pytest.approx(1.0)
    alu = Material('aluminum', youngs_modulus=70e9, density=2700, poissons_ratio=0.33)
    thickness = 0.01
    part.group_properties = {int(part.group_id[0]): GroupProperties(alu, thickness=thickness)}
    frequency = Model.from_geometry(part).eigensolution(num_modes=9).frequency
    assert int(np.sum(frequency == 0.0)) == 6
    rigidity = 70e9 * thickness ** 3 / (12 * (1 - 0.33 ** 2))
    lam = 2 * np.pi * frequency[6:9] * np.sqrt(2700 * thickness / rigidity)
    # Leissa, a free circular plate at a Poisson's ratio of 0.33: two
    # nodal diameters 5.253, one nodal circle 9.084
    assert lam == pytest.approx([5.253, 5.253, 9.084], rel=0.005)


def test_triangles_split_each_quad_on_its_shorter_diagonal():
    g = _empty()
    plate = solids.add_plate(g, (0, 0, 0), (4, 2))
    hole = solids.add_cylinder(g, (1, 0, 0), 0.4, 1.0)
    cut = solids.combine(g, 'subtract', [plate, hole])
    quads, _q = shells.shell_mesh(g, [cut], 0.2)
    triangles, made = shells.shell_mesh(g, [cut], 0.2, triangles=True)
    assert made['elements'] == 2 * len(quads.elem_conn)
    assert set(triangles.elem_type.tolist()) == {shells.TRIANGLE}
    index = {int(n): k for k, n in enumerate(triangles.node_id)}
    xyz = triangles.node_xyz
    for quad in quads.elem_conn:
        a, b, c, d = (xyz[index[int(n)]] for n in quad)
        shorter = min(np.linalg.norm(a - c), np.linalg.norm(b - d))
        corners = {int(n) for n in quad}
        halves = [t for t in triangles.elem_conn if {int(n) for n in t} <= corners]
        assert len(halves) == 2
        shared = set(map(int, halves[0])) & set(map(int, halves[1]))
        p, q = (xyz[index[n]] for n in shared)
        assert np.linalg.norm(p - q) == pytest.approx(shorter)


def test_what_cannot_cut_a_plate_square_is_refused_by_name():
    g = _empty()
    plate = solids.add_plate(g, (0, 0, 0), (4, 2), name='skin')
    slanted = solids.add_cylinder(g, (0, 0, 0), 0.3, 1.0, angles=(30, 0, 0), name='bolt')
    cut = solids.combine(g, 'subtract', [plate, slanted])
    with pytest.raises(ValueError, match='bolt crosses the plane of skin at an angle'):
        shells.shell_mesh(g, [cut], 0.2)
    g = _empty()
    plate = solids.add_plate(g, (0, 0, 0), (4, 2), name='skin')
    cone = solids.add_cone(g, (0, 0, 0), 0.3, 0.1, 1.0, name='punch')
    cut = solids.combine(g, 'subtract', [plate, cone])
    with pytest.raises(ValueError, match='punch is a cone'):
        shells.shell_mesh(g, [cut], 0.2)
    g = _empty()
    plate = solids.add_plate(g, (0, 0, 0), (4, 2), name='skin')
    wall = solids.add_plate(g, (0, 0, 1), (4, 2), angles=(90, 0, 0), name='wall')
    joined = solids.combine(g, 'union', [plate, wall])
    with pytest.raises(ValueError, match='wall is not on the plane of skin'):
        shells.shell_mesh(g, [joined], 0.2)


def test_plates_meshed_together_share_their_edge():
    g = _empty()
    floor = solids.add_plate(g, (0, 0, 0), (2, 2), name='floor')
    wall = solids.add_plate(g, (0, 1, 1), (2, 2), angles=(90, 0, 0), name='wall')
    part, made = shells.shell_mesh(g, [floor, wall], 0.5)
    assert made['flat'] == 2
    assert part.num_nodes == 2 * 25 - 5, 'the five nodes along the edge are shared'
    assert sorted(part.group_name) == ['floor', 'wall']


# ---- shells of a solid's outer surface -----------------------------------------


@pytest.mark.parametrize('triangles', [False, True])
def test_a_block_meshes_on_its_outer_surface_closed(triangles):
    g = _empty()
    solids.add_block(g, (0, 0, 0), (1, 2, 3), name='box')
    part, made = shells.shell_mesh(g, [1], 0.25, triangles=triangles)
    assert made['surfaces'] == 1 and made['flat'] == 0
    if not triangles:
        assert made['elements'] == 2 * (1 * 2 + 1 * 3 + 2 * 3) / 0.25 ** 2
    half = np.array([0.5, 1.0, 1.5])
    on_a_face = np.isclose(np.abs(part.node_xyz), half).any(axis=1)
    assert on_a_face.all(), 'nothing from inside the block'
    assert set(part.elem_type.tolist()) == {shells.TRIANGLE if triangles else shells.QUAD}
    assert set(_edges(part.elem_conn).values()) == {2}, 'closed: every edge on two faces'


# ---- through the project --------------------------------------------------------


def _project():
    from visualdynamics import Project

    project = Project()
    project.new_geometry('Part', unit='in')
    return project


def test_the_project_meshes_a_plate_into_shells_and_again_into_triangles():
    project = _project()
    plate = project.add_solid_plate('Part', (0, 0, 0), (4, 2), name='skin', unit='in')
    hole = project.add_solid_cylinder('Part', (1, 0, 0), 0.25, 0.1, unit='in')
    cut = project.combine_solids('Part', 'subtract', [plate, hole])
    with pytest.raises(ValueError, match='skin is flat: a plate or a disk is meshed into '
                                         'shells'):
        project.mesh_solids('Part', [cut], 0.25, unit='in')
    found = project.mesh_solids('Part', [cut], 0.25, unit='in', elements='quads')
    geometry = project['Part']
    assert set(geometry.elem_type.tolist()) == {shells.QUAD}
    assert found['elements'] == len(geometry.elem_conn)
    assert np.ptp(geometry.node_xyz[:, 0]) == pytest.approx(4 * INCH)
    again = project.mesh_solids('Part', [cut], 0.25, unit='in', elements='triangles')
    assert again['replaced'] == found['elements']
    assert set(geometry.elem_type.tolist()) == {shells.TRIANGLE}
    assert len(geometry.elem_conn) == 2 * found['elements']
    assert project.journal[-1] == ("project.mesh_solids('Part', [3], 0.25, unit='in', "
                                   "elements='triangles')")
    assert project.journal[-5] == ("project.add_solid_plate('Part', (0, 0, 0), (4, 2), "
                                   "name='skin', unit='in')")


def test_a_disk_journals_and_its_default_size_is_across_its_plane():
    project = _project()
    disk = project.add_solid_disk('Part', (0, 0, 0), 1.0, name='lid', unit='in')
    assert project.journal[-1] == ("project.add_solid_disk('Part', (0, 0, 0), 1.0, "
                                   "name='lid', unit='in')")
    # half the thinnest feature, the diameter: no thickness to read as 0
    assert shells.mesh_size(project['Part'], [disk]) == pytest.approx(1.0 * INCH)
    found = project.mesh_solids('Part', [disk], elements='quads')
    assert found['elements'] > 0


# ---- in the app -----------------------------------------------------------------


def test_the_solids_row_adds_a_plate_and_a_disk_in_the_pane(window, pump):
    from test_solids_pane import _add, _part, _rows

    geometry = _part(window, pump)
    panel = _add(window, pump, 'add_solid_plate', widths=(4, 2, 0), group='skin')
    assert panel.title.text() == 'Add Plate'
    assert panel.widths[2].isHidden() and not panel.widths[1].isHidden()
    assert panel.reading_label.text().startswith('A plate 4 by 2 in, 8 in² — flat')
    panel.add_button.click()
    pump()
    assert window.project.journal[-1] == (
        "project.add_solid_plate('Part', (0.0, 0.0, 0.0), (4.0, 2.0), (0.0, 0.0, 0.0), "
        "'skin', unit='in')")
    assert solids.solid(geometry, 1).widths == pytest.approx((4 * INCH, 2 * INCH, 0.0))
    panel = _add(window, pump, 'add_solid_disk', radius=1.5)
    assert panel.title.text() == 'Add Disk'
    assert not panel.radius.isHidden() and panel.widths[0].isHidden()
    assert panel.reading_label.text().startswith('A disk 3 in across')
    panel.add_button.click()
    pump()
    assert solids.solid(geometry, 2).kind == 'disk'
    assert solids.solid(geometry, 2).radius == pytest.approx(1.5 * INCH)
    assert len(_rows(window, 'solid')) == 2


def test_mesh_solids_offers_shells_only_for_a_plate_and_meshes_it(window, pump):
    from conftest import select_objects
    from test_solids_mesh import _settled
    from test_solids_pane import _acts, _pick, _rows

    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    project = window.project
    name = project.new_geometry('Part', unit='in')
    project.add_solid_plate(name, (0, 0, 0), (4, 2), name='skin', unit='in')
    window.show_object(name)
    select_objects(window, pump, name)
    window._solids_changed(name)
    pump()
    _pick(window, pump, _rows(window, 'solid')[:1])
    _acts(window)['mesh_solids'][3]()
    pump()
    panel = window.scene.mesh_solids_panel
    assert [panel.elements.itemText(k) for k in range(panel.elements.count())] == \
        ['Shells (quads)', 'Shells (triangles)']
    assert panel.element_kind == 'quads'
    panel.size.setValue(0.5)
    _settled(window, pump)
    assert panel.reading_label.text().startswith(
        '32 four-node plates on their plane, 45 nodes. Their thickness is set on the '
        'element group.')
    panel.mesh_button.click()
    pump()
    geometry = window.objects[name]
    assert set(geometry.elem_type.tolist()) == {shells.QUAD}
    assert list(geometry.group_name) == ['skin']
    assert window.statusBar().currentMessage().startswith(f'{name}: meshed into 32 plates')
