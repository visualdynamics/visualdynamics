"""Turning a coordinate system in the window: the rings and the angle.

`test_rotate.py` has the math — what a frame becomes when it is turned
about an axis, with no window in sight. This is the gesture on top of it:
when the rings are offered at all, what appears with them, that a typed
angle turns the frame and only the frame, that Reset puts it back square
without moving it, and that switching off takes the rings out of the
scene.

It is a good example of why a control's *visibility* is worth asserting:
turning is a single-object act, so with two rows picked there is no
answer to which frame the rings belong to, and the button has to be gone
rather than present and confusing.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path


@pytest.fixture
def systems(window, pump):
    """A geometry open on its coordinate systems, one row picked."""
    window.import_paths([fixture_path('plate', 'geometry.exo')])
    pump()
    geometry = window.objects['Geometry']
    # a second one, because 'several picked' cannot be expressed in a
    # table with a single row and that is half of what is checked here
    geometry.add_coordinate_system(origin=(0.1, 0.0, 0.0))
    item = window._item_for_object('Geometry')
    item.setExpanded(True)
    child = next(item.child(i) for i in range(item.childCount())
                 if item.child(i).text(0).startswith('Coordinate'))
    window.tree.clearSelection()
    window.tree.setCurrentItem(child)
    child.setSelected(True)
    window.edit_entities()
    pump()
    return geometry


def rings(window):
    return [name for name in window.scene.plotter.renderer.actors
            if 'rotate-ring' in name]


def test_the_rings_are_offered_only_for_one_picked_system(systems, window,
                                                          pump):
    assert not window.rotate_action.isVisible(), 'nothing picked'
    window.table.selectRow(0)
    pump()
    assert window.rotate_action.isVisible()
    assert not window.reset_rotation_action.isVisible(), 'only once turning'
    assert not window._angle_actions[0].isVisible()
    window.table.selectAll()
    pump()
    assert not window.rotate_action.isVisible(), (
        'with several picked there is no single frame to turn')


def test_turning_on_puts_three_rings_in_the_scene(systems, window, pump):
    window.table.selectRow(0)
    pump()
    window.rotate_action.setChecked(True)
    pump()
    assert len(rings(window)) == 3, 'one per axis'
    assert window.reset_rotation_action.isVisible()
    assert window._angle_actions[0].isVisible(), 'the angle box comes with it'
    window.rotate_action.setChecked(False)
    pump()
    assert not rings(window), 'and they go with it'
    assert not window.reset_rotation_action.isVisible()


def test_a_typed_angle_turns_the_frame_and_leaves_the_origin(systems, window,
                                                             pump):
    window.table.selectRow(0)
    pump()
    window.rotate_action.setChecked(True)
    pump()
    before = np.array(systems.cs_matrix[0])
    window.angle_box.setValue(90.0)
    pump()
    turned = np.array(systems.cs_matrix[0])
    assert not np.allclose(turned[:3], before[:3]), 'the frame turned'
    assert np.allclose(turned[3], before[3]), 'the origin stayed put'
    assert np.allclose(turned[:3] @ turned[:3].T, np.eye(3), atol=1e-9), (
        'and it is still a rotation, not a shear')
    assert sorted({i.row() for i
                   in window.table.selectionModel().selectedIndexes()}) == [0], (
        'the row it belongs to stays picked, so the rings keep their target')
    assert window.rotate_action.isVisible()


def test_reset_squares_it_up_where_it_sits(systems, window, pump):
    window.table.selectRow(0)
    pump()
    window.rotate_action.setChecked(True)
    pump()
    origin = np.array(systems.cs_matrix[0][3])
    window.angle_box.setValue(35.0)
    pump()
    window.reset_rotation_action.trigger()
    pump()
    reset = np.array(systems.cs_matrix[0])
    assert np.allclose(reset[:3], np.eye(3)), 'back to the global directions'
    assert np.allclose(reset[3], origin), 'without moving it'
    assert window.angle_box.value() == 0.0
