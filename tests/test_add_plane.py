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
    assert found == {'added': 10, 'shared': 5, 'elements': 8, 'blocks': [2]}
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
    """The dialog reads in the display unit, previews the plane, says what
    it shares before it is added, refuses skewed edges, and each Add is
    the project's verb."""
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Box', unit='in')
    window.show_object('Box')
    select_objects(window, pump, 'Box')
    window.add_plane_act()
    pump()
    dialog = window.plane_dialog
    from PySide6.QtWidgets import QLabel

    labels = [label.text() for label in dialog.findChildren(QLabel)]
    assert {'Corner [in]', 'Edge A [in]', 'Edge B [in]',
            'Element size [in]'} <= set(labels)
    dialog.set_values(block='floor', corner=(0, 0, 0), edge_a=(4, 0, 0),
                      edge_b=(0, 2, 0), size=1)
    pump()
    assert dialog.reading_label.text() == (
        "8 plates of 1 by 1 in, into a new block 'floor': 15 nodes to add, "
        '0 on nodes already there.')
    assert 'plane-preview' in window.scene.plotter.actors
    dialog.add_button.click()
    pump()
    box = window.objects['Box']
    assert np.allclose(box.node_xyz.max(axis=0), [4 * INCH, 2 * INCH, 0])
    dialog.set_values(block='wall', edge_b=(0, 0, 2))
    pump()
    assert dialog.reading_label.text().endswith(
        '10 nodes to add, 5 on nodes already there.')
    dialog.add_button.click()
    pump()
    assert box.num_nodes == 25 and list(box.block_name) == ['floor', 'wall']
    assert 'added 8 plates — 10 nodes, 5 shared' in \
        window.statusBar().currentMessage()
    dialog.set_values(edge_a=(4, 0, 1))
    pump()
    assert 'not perpendicular' in dialog.reading_label.text()
    assert not dialog.add_button.isEnabled()
    assert 'plane-preview' not in window.scene.plotter.actors, \
        'a refused plane is not drawn'
    dialog.set_values(edge_a=(4, 0, 0))
    pump()
    assert 'plane-preview' in window.scene.plotter.actors
    dialog.close()
    pump()
    assert 'plane-preview' not in window.scene.plotter.actors, \
        'closing takes the preview away'
    assert window.project.journal[-1].startswith("project.add_plane('Box', "
                                                 '(0.0, 0.0, 0.0), (4.0, ')
