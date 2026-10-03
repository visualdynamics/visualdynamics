"""New Geometry and Add Plane: a plate model built in the app.

Until 2026-09-26 the planes a plate model is made of could be meshed only
from a script (`mesh.plane`, `mesh.assemble`), and a geometry could only
be imported; the BARC's workflow page had to start in Python. The
project's **+** makes an empty geometry, and Add Plane types planes into
it, each tied to what is there where they meet (`mesh.join`).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

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
                     'blocks': [2]}
    assert np.array_equal(floor.node_id[:15], before)
    assert floor.node_id[15:].tolist() == list(range(116, 126))
    shared_line = [n for n in floor.elem_conn[8] if n < 116]  # its first
    assert shared_line == [101, 102], 'the wall is tied along the floor edge'


def test_a_block_of_the_same_name_is_joined_and_an_unnamed_one_is_not():
    geometry = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, 'box')
    assert mesh.join(geometry, mesh.plane((1, 0, 0), (0, 1, 0), (0, 0, 1),
                                          0.5, 'box'))['blocks'] == [1]
    assert mesh.join(geometry, mesh.plane((0, 0, 1), (1, 0, 0), (0, 1, 0),
                                          0.5, ''))['blocks'] == [2]
    assert mesh.join(geometry, mesh.plane((0, 1, 0), (1, 0, 0), (0, 0, 1),
                                          0.5, ''))['blocks'] == [3]
    assert list(geometry.block_name) == ['box', '', '']


def test_landing_says_what_joining_would_share_without_joining():
    floor = mesh.plane((0, 0, 0), (4, 0, 0), (0, 2, 0), 1, 'floor')
    wall = mesh.plane((0, 0, 0), (4, 0, 0), (0, 0, 2), 1, 'wall')
    on, rows = mesh.landing(floor, wall)
    assert on.sum() == 5 and floor.num_nodes == 15
    assert np.allclose(floor.node_xyz[rows[on]], wall.node_xyz[on])
    assert not mesh.landing(_empty(), wall)[0].any()


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
    """The verbs the two acts call, journaled as a script would write
    them; lengths in the unit named, held in SI."""
    project = visualdynamics.Project('p')
    name = project.new_geometry('Box', unit='in')
    assert project[name].num_nodes == 0 and project[name].length_unit == 'in'
    assert [verb for verb, _ in project.verbs(name)] == [
        'generate_rigid_body_modes', 'add_plane', 'add_block', 'set_view'], \
        'an empty geometry has no elements to merge or blocks to solve'
    first = project.add_plane(name, (0, 0, 0), (4, 0, 0), (0, 2, 0), 1,
                              'box', unit='in')
    second = project.add_plane(name, (0, 0, 0), (4, 0, 0), (0, 0, 2), 1,
                               'box', unit='in')
    assert (first['added'], second['added'], second['shared']) == (15, 10, 5)
    box = project[name]
    assert np.allclose(box.node_xyz.max(axis=0), [4 * INCH, 2 * INCH,
                                                  2 * INCH])
    assert list(box.block_name) == ['box']
    assert project.journal[-2:] == [
        ("project.add_plane('Box', (0, 0, 0), (4, 0, 0), (0, 2, 0), 1, "
         "'box', unit='in')"),
        ("project.add_plane('Box', (0, 0, 0), (4, 0, 0), (0, 0, 2), 1, "
         "'box', unit='in')")]


def test_a_geometry_without_units_takes_the_numbers_as_given():
    project = visualdynamics.Project('p')
    raw = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, 'floor',
                     unit=None)
    project.add('Raw', raw)
    project.add_plane('Raw', (0, 0, 0), (1, 0, 0), (0, 0, 1), 0.5, 'wall',
                      unit='in')
    assert not raw.units_defined
    assert np.isclose(raw.node_xyz[:, 2].max(), 1.0)


def test_the_project_offers_a_new_geometry(window, pump):
    """The **+** on the project's bar: an empty geometry in the display
    length unit, shown and selected so its bar offers Add Plane."""
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.tree.clearSelection()
    window.test_item.setSelected(True)
    pump()
    acts = {verb: icon for verb, _label, icon, _handler, _tip
            in window.acts_for()}
    assert acts.get('new_geometry') == 'add'
    window.new_geometry_act()
    pump()
    geometry = window.objects['Geometry']
    assert geometry.num_nodes == 0 and geometry.length_unit == 'in'
    assert [act[0] for act in window.acts_for()] == ['add_plane', 'add_block',
                                                    'set_view']


def test_add_plane_types_planes_in_display_units(window, pump):
    """The pane beside the view (no window, 2026-10-02) reads a center
    and widths in the display unit, the zero width naming the plane;
    previews the plate, says what it shares before it is added, refuses
    widths that make no plate, and each Add is the project's verb."""
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Box', unit='in')
    window.show_object('Box')
    select_objects(window, pump, 'Box')
    window.add_plane_act()
    pump()
    panel = window.scene.mesh_panel
    assert panel.isVisibleTo(window) and panel.title.text() == 'Add Plane'
    from PySide6.QtWidgets import QDialog, QLabel

    assert not [w for w in window.findChildren(QDialog) if w.isVisible()], \
        'no window of its own'
    labels = [label.text() for label in panel.findChildren(QLabel)]
    assert {'Center [in]', 'Width [in]', 'Element size [in]'} <= set(labels)
    panel.set_values(block='floor', center=(2, 1, 0), widths=(4, 2, 0), size=1)
    pump()
    assert panel.reading_label.text() == (
        "8 plates of 1 by 1 in, into a new block 'floor': 15 nodes to add, "
        '0 on nodes already there.')
    assert 'plane-preview' in window.scene.plotter.actors
    assert any('rotate-arrow' in name for name in window.scene.plotter.actors), \
        'the gizmo is around the plate'
    panel.add_button.click()
    pump()
    box = window.objects['Box']
    assert np.allclose(box.node_xyz.max(axis=0), [4 * INCH, 2 * INCH, 0])
    assert window.project.journal[-1].startswith("project.add_plane('Box', "
                                                 '(0.0, 0.0, 0.0), (4.0, ')
    # a wall: the zero width now along Y, the plate in X-Z
    panel.set_values(block='wall', center=(2, 0, 1), widths=(4, 0, 2))
    pump()
    assert panel.reading_label.text().endswith(
        '10 nodes to add, 5 on nodes already there.')
    panel.add_button.click()
    pump()
    assert box.num_nodes == 25 and list(box.block_name) == ['floor', 'wall']
    assert 'added 8 plates — 10 nodes, 5 shared' in \
        window.statusBar().currentMessage()
    panel.set_values(widths=(4, 2, 1))
    pump()
    assert 'exactly one width at zero' in panel.reading_label.text()
    assert not panel.add_button.isEnabled()
    assert 'plane-preview' not in window.scene.plotter.actors, \
        'a refused plate is not drawn'
    panel.close_button.click()
    pump()
    assert not panel.isVisibleTo(window)
    assert 'plane-preview' not in window.scene.plotter.actors, \
        'closing takes the preview away'
    assert not any('rotate-' in name for name in window.scene.plotter.actors)


def test_the_gizmo_turns_the_box_about_its_center_and_slides_it_onto_the_grid(
        window, pump):
    """The coordinate system's rings and arrows, on the box: a quarter
    turn about Z swaps the plate's edges and keeps its center; a slide
    along X lands the center on a tenth of an inch and the field shows
    it (2026-10-02)."""
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Box', unit='in')
    window.show_object('Box')
    select_objects(window, pump, 'Box')
    window.add_plane_act()
    pump()
    panel = window.scene.mesh_panel
    panel.set_values(center=(0, 0, 0), widths=(4, 2, 0), size=1)
    pump()
    window._rotating = {'row': 'mesh', 'axis': 2,
                        'start': np.array(window._mesh['frame']), 'from': 0.0}
    window._apply_rotation(np.pi / 2)
    window._commit_rotation()
    pump()
    _verb, call = window._mesh_call()
    assert np.allclose(call['edges'][0], (0, 4, 0)) and \
        np.allclose(call['edges'][1], (-2, 0, 0)), 'the plate turned'
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
    window.add_block_act()
    pump()
    assert window.scene.mesh_panel.isVisibleTo(window)
    select_objects(window, pump, 'Other')
    assert not window.scene.mesh_panel.isVisibleTo(window)
    assert window._mesh is None
    assert 'plane-preview' not in window.scene.plotter.actors


def test_dragging_the_box_draws_its_outline_snaps_the_angle_and_says_it(
        window, pump, monkeypatch):
    """While the box is dragged only its outline is drawn — the bricks
    are not meshed on every move — the turn snaps to whole degrees and
    the angle is written over the view; release brings the full preview
    and its reading back, and takes the number away (2026-10-02)."""
    from visualdynamics.core import mesh

    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Box', unit='in')
    window.show_object('Box')
    select_objects(window, pump, 'Box')
    window.add_block_act()
    pump()
    panel = window.scene.mesh_panel
    panel.set_values(center=(0, 0, 0), widths=(4, 2, 1), size=0.5)
    pump()
    built = []
    real = mesh.block
    monkeypatch.setattr(mesh, 'block', lambda *a, **k: built.append(1) or real(*a, **k))
    start = np.array(window._mesh['frame'])
    window._rotating = {'row': 'mesh', 'axis': 2, 'start': start, 'from': 0.0}
    applied = window._apply_rotation(np.radians(37.4), snap=True)
    assert np.degrees(applied) == pytest.approx(37.0)
    assert np.allclose(window._mesh['frame'][0],
                       [np.cos(np.radians(37)), np.sin(np.radians(37)), 0])
    actors = window.scene.plotter.actors
    assert 'mesh-outline' in actors and 'plane-preview' not in actors
    assert 'gizmo-reading' in actors and window._gizmo_reading == 'Turn about Z: +37°'
    assert built == [], 'nothing meshed while dragging'
    window._commit_rotation()
    pump()
    actors = window.scene.plotter.actors
    assert 'plane-preview' in actors and 'mesh-outline' not in actors
    assert 'gizmo-reading' not in actors and window._gizmo_reading is None
    assert built == [1], 'meshed once, on release'
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
    window.add_plane_act()
    pump()
    panel = window.scene.mesh_panel
    from PySide6.QtWidgets import QLabel

    assert 'Rotation [°]' in [label.text() for label in panel.findChildren(QLabel)]
    panel.set_values(center=(0, 0, 0), widths=(4, 2, 0), size=1, angles=(0, 0, 37.5))
    pump()
    _verb, call = window._mesh_call()
    c, s = np.cos(np.radians(37.5)), np.sin(np.radians(37.5))
    assert np.allclose(call['edges'][0], (4 * c, 4 * s, 0))
    window._rotating = {'row': 'mesh', 'axis': 2,
                        'start': np.array(window._mesh['frame']), 'from': 0.0}
    window._apply_rotation(np.radians(10.0), snap=True)
    window._commit_rotation()
    pump()
    assert panel.values()['angles'] == pytest.approx((0.0, 0.0, 47.5))
    panel.square_button.click()
    pump()
    assert panel.values()['angles'] == (0.0, 0.0, 0.0)
