"""New Geometry, the join that adds a part to it, and the add pane's
gizmo: a model built in the app.

Until 2026-09-26 the planes a plate model is made of could be meshed only
from a script (`mesh.plane`, `mesh.assemble`), and a geometry could only
be imported; the BARC's workflow page had to start in Python. The
project's **+** makes an empty geometry. Add Plane typed meshed planes
into it until 2026-10-26; a plate is now a solid on the Solids row,
meshed by Mesh Solids, and each part is tied to what is there where they
meet (`mesh.join`).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import bar_acts, select_objects

import visualdynamics
from visualdynamics import mesh
from visualdynamics.core.geometry import Geometry

INCH = 0.0254


def _empty(unit='m'):
    return Geometry(node_id=[], node_xyz=np.empty((0, 3)), length_unit=unit)


def test_joining_keeps_the_ids_already_there():
    """Data linked to a geometry names its nodes, so a plane joined to it
    leaves them alone: its own nodes that fall on them become them, the
    rest are numbered after the highest."""
    floor = mesh.plane((0, 0, 0), (4, 0, 0), (0, 2, 0), 1, 'floor')
    floor.node_id = floor.node_id + 100
    floor.elem_conn = [c + 100 for c in floor.elem_conn]
    before = floor.node_id.copy()
    wall = mesh.plane((0, 0, 0), (4, 0, 0), (0, 0, 2), 1, 'wall')
    found = mesh.join(floor, wall)
    assert found == {'added': 10, 'shared': 5, 'elements': 8, 'duplicates': 0,
                     'groups': [2]}
    assert np.array_equal(floor.node_id[:15], before)
    assert floor.node_id[15:].tolist() == list(range(116, 126))
    shared_line = [n for n in floor.elem_conn[8] if n < 116]  # its first
    assert shared_line == [101, 102], 'the wall is tied along the floor edge'


def test_a_block_of_the_same_name_is_joined_and_an_unnamed_one_is_not():
    geometry = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, 'box')
    assert mesh.join(geometry, mesh.plane((1, 0, 0), (0, 1, 0), (0, 0, 1),
                                          0.5, 'box'))['groups'] == [1]
    assert mesh.join(geometry, mesh.plane((0, 0, 1), (1, 0, 0), (0, 1, 0),
                                          0.5, ''))['groups'] == [2]
    assert mesh.join(geometry, mesh.plane((0, 1, 0), (1, 0, 0), (0, 0, 1),
                                          0.5, ''))['groups'] == [3]
    assert list(geometry.group_name) == ['box', '', '']


def test_landing_says_what_joining_would_share_without_joining():
    floor = mesh.plane((0, 0, 0), (4, 0, 0), (0, 2, 0), 1, 'floor')
    wall = mesh.plane((0, 0, 0), (4, 0, 0), (0, 0, 2), 1, 'wall')
    on, rows = mesh.landing(floor, wall)
    assert on.sum() == 5 and floor.num_nodes == 15
    assert np.allclose(floor.node_xyz[rows[on]], wall.node_xyz[on])
    assert not mesh.landing(_empty(), wall)[0].any()


def test_a_part_kept_apart_joins_only_its_own_group():
    """Kept apart (2026-10-11), a plane lands only on the nodes of the
    element group it joins by name: a second floor plane still shares
    the first's edge, a foot laid on the floor keeps nodes of its own
    over it, coincident and unjoined, for a merge to join at the bolts."""
    floor = mesh.plane((0, 0, 0), (4, 0, 0), (0, 2, 0), 1, 'floor')
    foot = mesh.plane((1, 0, 0), (2, 0, 0), (0, 1, 0), 1, 'foot')
    on, _rows = mesh.landing(floor, foot, apart=True)
    assert not on.any() and mesh.landing(floor, foot)[0].sum() == 6
    found = mesh.join(floor, foot, apart=True)
    assert found['shared'] == 0 and found['added'] == 6
    assert len(floor.coincident_spots(0.1)) == 6, 'each over a floor node'
    more = mesh.plane((4, 0, 0), (1, 0, 0), (0, 2, 0), 1, 'floor')
    assert mesh.join(floor, more, apart=True)['shared'] == 3, \
        'the floor still joins the floor'
    assert mesh.landing(floor, mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 1),
                        apart=True)[0].sum() == 0, 'an unnamed group joins none'


def test_assembling_apart_keeps_each_group_whole_and_the_groups_unjoined():
    walls = [mesh.plane((0, 0, 0), (2, 0, 0), (0, 1, 0), 1, 'box'),
             mesh.plane((2, 0, 0), (2, 0, 0), (0, 1, 0), 1, 'box'),
             mesh.plane((1, 0, 0), (2, 0, 0), (0, 1, 0), 1, 'lid')]
    joined = mesh.assemble(*walls)
    apart = mesh.assemble(*walls, apart=True)
    assert joined.num_nodes == 10 and apart.num_nodes == 16
    lid = set(apart.elem_conn[4]) | set(apart.elem_conn[5])
    spots = apart.coincident_spots(0.1)
    assert len(spots) == 6 and all(b > 10 and b in lid and a <= 10
                                   for a, b in spots), spots


def test_coordinates_that_mean_different_things_do_not_join():
    raw = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, unit=None)
    with pytest.raises(ValueError, match='do not mean the same thing'):
        mesh.join(raw, mesh.plane((0, 0, 0), (1, 0, 0), (0, 0, 1), 0.5))


def test_assembling_numbers_the_nodes_without_gaps():
    """Built on `join`, the assembly allocates no id to a node that lands
    on one already there, so its nodes run 1..n."""
    whole = mesh.assemble(
        mesh.plane((0, 0, 0), (4, 0, 0), (0, 2, 0), 1, 'floor'),
        mesh.plane((0, 0, 0), (4, 0, 0), (0, 0, 2), 1, 'wall'))
    assert whole.node_id.tolist() == list(range(1, 26))


def test_the_project_builds_a_model_from_nothing():
    """Plates added as solids and meshed, journaled as a script would
    write them; lengths in the unit named, held in SI. Two plates of one
    name are one element group, tied where they meet."""
    project = visualdynamics.Project('p')
    name = project.new_geometry('Box', unit='in')
    assert project[name].num_nodes == 0 and project[name].length_unit == 'in'
    assert [verb for verb, _ in project.verbs(name)] == [
        'generate_rigid_body_modes',
        'add_solid_block', 'add_solid_cylinder', 'add_solid_cone', 'add_solid_sphere',
        'add_solid_section', 'add_solid_plate', 'add_solid_disk', 'set_view'], \
        'an empty geometry has no elements to merge or element groups to solve'
    floor = project.add_solid_plate(name, (2, 1, 0), (4, 2), name='box', unit='in')
    wall = project.add_solid_plate(name, (2, 0, 1), (4, 2), (90, 0, 0), 'box', unit='in')
    found = project.mesh_solids(name, [floor, wall], 1, unit='in', elements='quads')
    box = project[name]
    assert (found['added'], box.num_nodes, len(box.elem_conn)) == (25, 25, 16), \
        'fifteen and fifteen, five of them shared along the edge'
    assert np.allclose(box.node_xyz.max(axis=0), [4 * INCH, 2 * INCH,
                                                  2 * INCH])
    assert list(box.group_name) == ['box']
    assert project.journal[-3:] == [
        "project.add_solid_plate('Box', (2, 1, 0), (4, 2), name='box', unit='in')",
        "project.add_solid_plate('Box', (2, 0, 1), (4, 2), (90, 0, 0), 'box', unit='in')",
        "project.mesh_solids('Box', [1, 2], 1, unit='in', elements='quads')"]


def test_a_geometry_without_units_takes_the_numbers_as_given():
    project = visualdynamics.Project('p')
    raw = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, 'floor',
                     unit=None)
    project.add('Raw', raw)
    wall = project.add_solid_plate('Raw', (0.5, 0, 0.5), (1, 1), (90, 0, 0), 'wall',
                                   unit='in')
    project.mesh_solids('Raw', [wall], 0.5, unit='in', elements='quads')
    assert not raw.units_defined
    assert np.isclose(raw.node_xyz[:, 2].max(), 1.0)
    assert raw.num_nodes == 15, 'tied to the floor along their edge'


def test_the_project_offers_a_new_geometry(window, pump):
    """The **+** on the project's bar: an empty geometry in the display
    length unit, shown and selected. Its Solids row offers the shapes a
    part is built from, a plate among them, and its family rows offer
    nothing to add: Add Plane and Add Block, which typed meshed planes
    and blocks onto the Quads and Hexes rows, retired for the plate and
    the block meshed by Mesh Solids (Brandon, 2026-10-26)."""
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.tree.clearSelection()
    window.test_item.setSelected(True)
    pump()
    acts = {verb: icon for verb, _label, icon, _handler, _tip
            in window.acts_for()}
    assert acts.get('new_geometry') == 'new_geometry'
    window.new_geometry_act()
    pump()
    geometry = window.objects['Geometry']
    assert geometry.num_nodes == 0 and geometry.length_unit == 'in'
    assert [act[0] for act in window.acts_for()] == ['set_view'], (
        'not on the geometry itself')
    item = window._item_for_object('Geometry')
    item.setExpanded(True)

    def pick(label):
        row = next(item.child(i) for i in range(item.childCount())
                   if item.child(i).text(0).startswith(label))
        window.tree.clearSelection()
        window.tree.setCurrentItem(row)
        pump()
        return row

    pick('Solids')
    assert [act[0] for act in window.acts_for()] == list(window.SOLID_ACTS)
    assert 'Add Plate' in bar_acts(window.scene), 'on the 3-D bar'
    for family in ('Quads', 'Hexes', 'Beams'):
        pick(family)
        assert window.acts_for() == [], family
    # opened from the Solids row, the pane stays open through an Add,
    # which rebuilds the geometry's rows under it
    pick('Solids')
    window.add_solid_plate_act()
    pump()
    panel = window.scene.mesh_panel
    panel.set_values(group='floor', center=(2, 1, 0), widths=(4, 2, 0))
    pump()
    panel.add_button.click()
    pump()
    assert len(window.objects['Geometry'].solids) == 1
    assert panel.isVisibleTo(window), 'still open for the next plate'


def test_the_gizmo_turns_the_box_about_its_center_and_slides_it_onto_the_grid(
        window, pump):
    """The coordinate system's rings and arrows, on the solid: a quarter
    turn about Z turns the plate and keeps its center; a slide along X
    lands the center on a tenth of an inch and the field shows it
    (2026-10-02)."""
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Box', unit='in')
    window.show_object('Box')
    select_objects(window, pump, 'Box')
    window.add_solid_plate_act()
    pump()
    panel = window.scene.mesh_panel
    panel.set_values(center=(0, 0, 0), widths=(4, 2, 0))
    pump()
    window._rotating = {'row': 'mesh', 'axis': 2,
                        'start': np.array(window._mesh['frame']), 'from': 0.0}
    window._apply_rotation(np.pi / 2)
    window._commit_rotation()
    pump()
    _verb, call = window._mesh_call()
    assert call['angles'] == pytest.approx((0, 0, 90)), 'the plate turned'
    assert np.allclose(window._mesh['frame'][0], (0, 1, 0))
    assert np.allclose(window._mesh['frame'][3], (0, 0, 0)), 'about its center'
    window._sliding = {'row': 'mesh', 'axis': 1,
                       'start': np.array(window._mesh['frame']), 'from': 0.0}
    window._apply_slide(1.234)
    window._commit_slide()
    pump()
    center = panel.values()['center']
    assert np.allclose(center, (-1.2, 0, 0)), center
    assert 'Moved the box' in window.statusBar().currentMessage()
    panel.square_button.click()
    pump()
    assert np.allclose(window._mesh['frame'][:3], np.eye(3))
    assert np.allclose(panel.values()['center'], (-1.2, 0, 0)), 'kept where it is'
    select_objects(window, pump, 'Box')


def test_selecting_another_object_puts_the_pane_away(window, pump):
    window.project.new_geometry('Box', unit='m')
    window.project.new_geometry('Other', unit='m')
    window.show_object('Box')
    window.show_object('Other')
    select_objects(window, pump, 'Box')
    window.add_solid_block_act()
    pump()
    assert window.scene.mesh_panel.isVisibleTo(window)
    select_objects(window, pump, 'Other')
    assert not window.scene.mesh_panel.isVisibleTo(window)
    assert window._mesh is None
    assert 'plane-preview' not in window.scene.plotter.actors


def test_dragging_the_box_draws_its_outline_snaps_the_angle_and_says_it(
        window, pump, monkeypatch):
    """While the box is dragged only its outline is drawn — the solid is
    not built on every move — the turn snaps to whole degrees and the
    angle is written over the view; release brings the full preview and
    its reading back, and takes the number away (2026-10-02)."""
    from visualdynamics.core import solids

    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Box', unit='in')
    window.show_object('Box')
    select_objects(window, pump, 'Box')
    window.add_solid_block_act()
    pump()
    panel = window.scene.mesh_panel
    panel.set_values(center=(0, 0, 0), widths=(4, 2, 1))
    pump()
    built = []
    real = solids.shape_triangles
    monkeypatch.setattr(solids, 'shape_triangles',
                        lambda *a, **k: built.append(1) or real(*a, **k))
    start = np.array(window._mesh['frame'])
    window._rotating = {'row': 'mesh', 'axis': 2, 'start': start, 'from': 0.0}
    applied = window._apply_rotation(np.radians(37.4), snap=True)
    assert np.degrees(applied) == pytest.approx(37.0)
    assert np.allclose(window._mesh['frame'][0],
                       [np.cos(np.radians(37)), np.sin(np.radians(37)), 0])
    actors = window.scene.plotter.actors
    assert 'mesh-outline' in actors and 'plane-preview' not in actors
    assert 'gizmo-reading' in actors and window._gizmo_reading == 'Turn about Z: +37°'
    assert built == [], 'nothing built while dragging'
    window._commit_rotation()
    pump()
    actors = window.scene.plotter.actors
    assert 'plane-preview' in actors and 'mesh-outline' not in actors
    assert 'gizmo-reading' not in actors and window._gizmo_reading is None
    assert built == [1], 'built once, on release'
    turned = np.array(window._mesh['frame'])
    window._sliding = {'row': 'mesh', 'axis': 0, 'start': turned, 'from': 0.0}
    window._apply_slide(1.234)
    # the center lands on the grid in the geometry's axes, so along the
    # turned axis the slide reads what it really moved, not a round step
    moved = float((window._mesh['frame'][3] - turned[3]) @ turned[0])
    assert window._gizmo_reading == f'Slide along X: {moved:+.4g} in'
    assert np.allclose(np.round(window._mesh['frame'][3] / 0.1) * 0.1,
                       window._mesh['frame'][3]), 'the center on the grid'
    assert 'mesh-outline' in window.scene.plotter.actors
    window._commit_slide()
    pump()
    assert 'plane-preview' in window.scene.plotter.actors


def test_the_pane_turns_the_box_by_typed_angles_and_the_rings_write_them_back(
        window, pump):
    """Three angle fields, degrees about the geometry's X, Y and Z, for an
    odd angle; a ring's turn writes them back (Brandon, 2026-10-02)."""
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Box', unit='in')
    window.show_object('Box')
    select_objects(window, pump, 'Box')
    window.add_solid_plate_act()
    pump()
    panel = window.scene.mesh_panel
    from PySide6.QtWidgets import QLabel

    assert 'Rotation [°]' in [label.text() for label in panel.findChildren(QLabel)]
    panel.set_values(center=(0, 0, 0), widths=(4, 2, 0), angles=(0, 0, 37.5))
    pump()
    c, s = np.cos(np.radians(37.5)), np.sin(np.radians(37.5))
    assert np.allclose(window._mesh['frame'][0], (c, s, 0))
    window._rotating = {'row': 'mesh', 'axis': 2,
                        'start': np.array(window._mesh['frame']), 'from': 0.0}
    window._apply_rotation(np.radians(10.0), snap=True)
    window._commit_rotation()
    pump()
    assert panel.values()['angles'] == pytest.approx((0.0, 0.0, 47.5))
    panel.square_button.click()
    pump()
    assert panel.values()['angles'] == (0.0, 0.0, 0.0)


def test_the_pane_keeps_an_inch_fraction_to_the_64th(window):
    """The BARC guide places its channels at 1.96875 and 1.53125 in, and
    four decimals turned them into 1.9688 and 1.5312: gaps where the
    planes should meet, the model in four pieces (2026-10-17)."""
    panel = window.scene.mesh_panel
    for box in (*panel.center, *panel.widths):
        box.setValue(1.96875)
        assert box.value() == pytest.approx(1.96875, abs=1e-9)
    for box in panel.center:            # a width is never below zero
        box.setValue(-0.15625)
        assert box.value() == pytest.approx(-0.15625, abs=1e-9)


def test_every_length_box_keeps_an_inch_fraction_to_the_64th(window):
    """The same places for every length typed into the scene's panes:
    the rigid-body reference point kept four and read 1.96875 in as
    1.9688."""
    scene = window.scene
    boxes = (*scene.modes_panel.point_boxes, *scene.pattern_panel.spacings,
             scene.mesh_solids_panel.size, scene.merge_panel.tolerance)
    for box in boxes:
        box.setValue(1.96875)
        assert box.value() == pytest.approx(1.96875, abs=1e-9)
