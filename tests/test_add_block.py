"""Add Block: a meshed box of bricks typed into a geometry, the project
verb and the app's dialog (2026-09-30, for the four-unit frame example).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

import visualdynamics
from visualdynamics.core import mesh
from visualdynamics.core.fem import GroupProperties, Model, material

INCH = 0.0254


def test_a_block_is_a_box_of_bricks():
    bar = mesh.block((0, 0, 0), (0.4, 0, 0), (0, 0.1, 0), (0, 0, 0.02), 0.02,
                     'bar')
    assert bar.num_nodes == 21 * 6 * 2 and len(bar.elem_conn) == 20 * 5
    assert set(bar.elem_type.tolist()) == {115}
    assert list(bar.group_name) == ['bar'] and bar.length_unit == 'm'
    with pytest.raises(ValueError, match='not perpendicular'):
        mesh.block((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), 0.5, 'skew')
    with pytest.raises(ValueError, match='no length'):
        mesh.block((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 1), 0.5, 'flat')


def test_a_block_weighs_and_bends_as_its_solids():
    bar = mesh.block((0, 0, 0), (0.4, 0, 0), (0, 0.02, 0), (0, 0, 0.02),
                     0.02, 'bar')
    bar.group_properties = {1: GroupProperties(material('6061-T6'))}
    model = Model.from_geometry(bar)
    assert model.structural_mass == pytest.approx(
        0.4 * 0.02 * 0.02 * material('6061-T6').density)
    shapes = model.eigensolution(num_modes=8)
    assert int(np.sum(shapes.frequency == 0.0)) == 6
    assert shapes.frequency[6] == pytest.approx(shapes.frequency[7], rel=1e-6), \
        'a square bar bends the same either way'


def test_holes_leave_the_block_or_go_to_an_insert_block():
    box = mesh.block((0, 0, 0), (0.4, 0, 0), (0, 0.1, 0), (0, 0, 0.02), 0.01,
                     'bar', holes=[((0.1, 0.05, 0), 0.02, 2)])
    whole = 40 * 10 * 2
    assert len(box.elem_conn) < whole and list(box.group_name) == ['bar']
    assert box.num_nodes < 41 * 11 * 3, 'the hole\'s own nodes went too'
    centers = np.array([box.node_xyz[box.node_index(c)].mean(axis=0)
                        for c in box.elem_conn])
    assert np.linalg.norm(centers[:, :2] - [0.1, 0.05], axis=1).min() > 0.02
    inserts = mesh.block((0, 0, 0), (0.4, 0, 0), (0, 0.1, 0), (0, 0, 0.02),
                         0.01, 'bar', holes=[((0.1, 0.05, 0), 0.02, 2),
                                             ((0.3, 0.05, 0.02), 0.02, 2, 0.012)],
                         hole_name='inserts')
    assert list(inserts.group_name) == ['bar', 'inserts']
    assert len(inserts.elem_conn) == whole, 'nothing removed, only moved'
    insert_rows = np.flatnonzero(inserts.elem_group == 2)
    depths = centers_z = np.array([inserts.node_xyz[inserts.node_index(
        inserts.elem_conn[r])].mean(axis=0) for r in insert_rows])
    through = depths[np.abs(centers_z[:, 0] - 0.1) < 0.021]
    blind = depths[np.abs(centers_z[:, 0] - 0.3) < 0.021]
    assert through[:, 2].min() < 0.01 < through[:, 2].max()
    assert blind[:, 2].min() > 0.02 - 0.012, 'blind from the far face'
    with pytest.raises(ValueError, match='leave nothing'):
        mesh.block((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 0.5, 'x',
                   holes=[((0.5, 0.5, 0), 5, 2)])


def test_the_project_verb_is_journaled_and_ties_blocks_that_meet():
    project = visualdynamics.Project('p')
    name = project.new_geometry('Frame', unit='in')
    first = project.add_block(name, (0, 0, 0), (4, 0, 0), (0, 0.5, 0),
                              (0, 0, 0.5), 0.5, 'frame', unit='in')
    second = project.add_block(name, (0, 0.5, 0), (0.5, 0, 0), (0, 2, 0),
                               (0, 0, 0.5), 0.5, 'frame', unit='in')
    assert first['elements'] == 8 and second['elements'] == 4
    assert second['shared'] == 4, 'the upright stands on the rail\'s face'
    frame = project[name]
    assert list(frame.group_name) == ['frame']
    assert np.allclose(frame.node_xyz.max(axis=0), [4 * INCH, 2.5 * INCH,
                                                    0.5 * INCH])
    assert project.journal[-1] == (
        "project.add_block('Frame', (0, 0.5, 0), (0.5, 0, 0), (0, 2, 0), "
        "(0, 0, 0.5), 0.5, 'frame', unit='in')")
    frame.group_properties = {1: GroupProperties(material('6061-T6'))}
    modes = project.solve_modes(name, num_modes=8)
    assert int(np.sum(project[modes].frequency == 0.0)) == 6, 'one piece'


def test_add_block_types_blocks_in_display_units(window, pump):
    """The same pane: a center and three widths in the display unit,
    the element group previewed by its skin, what it shares said, and each Add
    the project's verb (2026-10-02, no window)."""
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Frame', unit='in')
    window.show_object('Frame')
    select_objects(window, pump, 'Frame')
    assert 'add_block' in [act[0] for act in window.acts_for(['Frame'])]
    window.add_block_act()
    pump()
    panel = window.scene.mesh_panel
    assert panel.isVisibleTo(window) and panel.title.text() == 'Add Block'
    from PySide6.QtWidgets import QLabel

    labels = [label.text() for label in panel.findChildren(QLabel)]
    assert {'Center [in]', 'Width [in]', 'Element size [in]'} <= set(labels)
    panel.set_values(group='rail', center=(2, 0.5, 0.25), widths=(4, 1, 0.5),
                     size=0.5)
    pump()
    assert panel.reading_label.text() == (
        "16 bricks of 0.5 by 0.5 by 0.5 in, into a new element group 'rail': "
        '54 nodes to add, 0 on nodes already there.')
    assert 'plane-preview' in window.scene.plotter.actors
    panel.add_button.click()
    pump()
    frame = window.objects['Frame']
    assert np.allclose(frame.node_xyz.max(axis=0), [4 * INCH, INCH, 0.5 * INCH])
    assert 'added 16 bricks — 54 nodes, 0 shared' in \
        window.statusBar().currentMessage()
    panel.set_values(widths=(4, 1, 0))
    pump()
    assert 'all three axes' in panel.reading_label.text()
    assert not panel.add_button.isEnabled()
    panel.close_button.click()
    pump()
    assert 'plane-preview' not in window.scene.plotter.actors


def test_a_plate_never_joins_a_block_of_bricks_by_name():
    """A plate added under the name of an element group of bricks made one element group
    of both, deleted whole from either family's row (Brandon,
    2026-10-02). The join refuses it, and a plate of its own name goes
    in an element group of its own."""
    import pytest

    project = visualdynamics.Project('p')
    g = project.new_geometry(unit='in')
    project.add_block(g, (0, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 0.5), 0.5,
                      'group', unit='in')
    with pytest.raises(ValueError, match="element group 'group' holds hexes"):
        project.add_plane(g, (0, 0, 2), (2, 0, 0), (0, 1, 0), 0.5, 'group',
                          unit='in')
    geometry = project[g]
    assert list(geometry.group_name) == ['group'], 'nothing half-added'
    project.add_plane(g, (0, 0, 2), (2, 0, 0), (0, 1, 0), 0.5, 'plate',
                      unit='in')
    assert list(geometry.group_name) == ['group', 'plate']
    assert not geometry.mixed_groups()


def test_the_pane_opens_on_a_block_of_its_own_family(window, pump):
    """After an element group of bricks, Add Plane opens on a fresh name, not the
    bricks' element group; after a plate, it opens on the plate's element group, so
    plates keep joining plates. A name typed onto the bricks' element group is
    refused before Add."""
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Frame', unit='in')
    window.show_object('Frame')
    select_objects(window, pump, 'Frame')
    window.add_block_act()
    pump()
    panel = window.scene.mesh_panel
    assert panel.values()['group'] == 'block', 'named for the box'
    panel.add_button.click()
    pump()
    window.add_plane_act()
    pump()
    assert panel.values()['group'] == 'plate', 'not the bricks\' element group'
    panel.set_values(group='block')
    pump()
    assert 'holds hexes' in panel.reading_label.text()
    assert not panel.add_button.isEnabled()
    panel.set_values(group='plate')
    pump()
    panel.add_button.click()
    pump()
    window.add_plane_act()
    pump()
    assert panel.values()['group'] == 'plate', 'the plate\'s element group again'
    window.add_block_act()
    pump()
    assert panel.values()['group'] == 'block'
    assert not window.objects['Frame'].mixed_groups()


def test_adding_many_elements_keeps_one_family_per_block():
    """`add_elements` keeps the rule `add_element` keeps, before it adds
    anything; it was the path the join took around it (2026-10-02)."""
    import pytest

    from visualdynamics.core.geometry import Geometry

    geometry = Geometry(node_id=[1, 2, 3, 4], node_xyz=np.eye(4, 3),
                        length_unit='m')
    geometry.add_elements([[1, 2, 3, 4]], [44], [1])
    with pytest.raises(ValueError, match='one element family'):
        geometry.add_elements([[1, 2]], [21], [1])
    assert len(geometry.elem_conn) == 1, 'nothing added by the refusal'
    with pytest.raises(ValueError, match='one element family'):
        geometry.add_elements([[1, 2, 3], [1, 2]], [41, 21], [2, 2])
    geometry.add_elements([[1, 2]], [21], [2])
    assert not geometry.mixed_groups()


def _cross(project):
    """Two bars crossing at the origin: an X cross-section, its overlap a
    cell of eight bricks both bars mesh."""
    g = project.new_geometry(unit='in')
    project.add_block(g, (-2, -0.5, 0), (4, 0, 0), (0, 1, 0), (0, 0, 1), 0.5,
                      'a', unit='in')
    return g


def test_crossing_blocks_fill_their_overlap_once():
    """The overlap of two crossing bars was meshed by both and counted
    twice, and the pair of bricks in each cell drew no faces at all, so
    the middle of the X vanished (Brandon, 2026-10-02). The second bar
    leaves out the bricks already there; the skin is the X's."""
    from collections import Counter

    from visualdynamics.viz.geometry import solid_faces

    project = visualdynamics.Project('p')
    g = _cross(project)
    found = project.add_block(g, (-0.5, -2, 0), (1, 0, 0), (0, 4, 0),
                              (0, 0, 1), 0.5, 'b', unit='in')
    geometry = project[g]
    assert (found['elements'], found['duplicates']) == (24, 8)
    assert len(geometry.elem_conn) == 56 and not geometry.duplicate_elements()
    faces = Counter()
    for conn in geometry.elem_conn:
        for face in solid_faces(len(conn)):
            faces[tuple(sorted(int(conn[i]) for i in face))] += 1
    # 7 + 7 sq in top and bottom, 3 + 3 + 3 + 3 along the sides, four
    # 1 sq in ends: 30 sq in of quarter-inch-square faces
    assert sum(1 for count in faces.values() if count == 1) == 120


def test_merging_nodes_makes_elements_on_the_same_nodes_one():
    """Bars meshed apart and tied after: the merge leaves two bricks in
    every overlap cell, and makes each pair one, the earlier."""
    from visualdynamics.core import mesh
    from visualdynamics.core.geometry import Geometry

    a = mesh.block((-2, -0.5, 0), (4, 0, 0), (0, 1, 0), (0, 0, 1), 0.5, 'a', unit='in')
    b = mesh.block((-0.5, -2, 0), (1, 0, 0), (0, 4, 0), (0, 0, 1), 0.5, 'b', unit='in')
    geometry = Geometry(
        node_id=np.concatenate([a.node_id, b.node_id + 1000]),
        node_xyz=np.vstack([a.node_xyz, b.node_xyz]),
        elem_conn=[*a.elem_conn, *[c + 1000 for c in b.elem_conn]],
        elem_type=[*a.elem_type, *b.elem_type],
        elem_group=[1] * len(a.elem_conn) + [2] * len(b.elem_conn),
        group_id=[1, 2], group_name=['a', 'b'], length_unit='m')
    assert len(geometry.elem_conn) == 64
    found = geometry.merge_coincident_nodes(1e-9)
    assert found['duplicates'] == 8
    assert len(geometry.elem_conn) == 56 and not geometry.duplicate_elements()
    assert int(np.sum(geometry.elem_group == 1)) == 32, 'the earlier bar keeps the cell'


def test_the_pane_says_the_overlap_before_the_add(window, pump):
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    _cross(window.project)
    window.show_object('Geometry')
    select_objects(window, pump, 'Geometry')
    window.add_block_act()
    pump()
    panel = window.scene.mesh_panel
    panel.set_values(group='b', center=(0, 0, 0.5), widths=(1, 4, 1), size=0.5)
    pump()
    assert panel.reading_label.text().endswith(
        '8 of the elements are already there where it overlaps, and are left out.')
    panel.add_button.click()
    pump()
    assert 'added 24 bricks, 8 more already there where it overlaps' in \
        window.statusBar().currentMessage()
