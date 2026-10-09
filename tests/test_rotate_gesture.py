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


def arrows(window):
    return [name for name in window.scene.plotter.renderer.actors
            if 'rotate-arrow' in name]


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
    assert len(arrows(window)) == 3, 'and an arrow along each, for sliding'
    assert window.reset_rotation_action.isVisible()
    assert window._angle_actions[0].isVisible(), 'the angle box comes with it'
    window.rotate_action.setChecked(False)
    pump()
    assert not rings(window) and not arrows(window), 'and they go with it'
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


def test_a_slide_moves_the_origin_along_the_axis_onto_the_grid(systems, window,
                                                                pump):
    """Dragging an arrow slides the origin along that axis and lands it
    on the grid in the geometry's own axes — a centimetre here, the
    display unit being metres — leaving the basis alone (2026-10-02)."""
    from visualdynamics.rotate import grid_step

    # the fixture's exodus file says no unit; the grid needs one
    systems.define_units('m')
    window.table.selectRow(0)
    pump()
    window.rotate_action.setChecked(True)
    pump()
    before = np.array(systems.cs_matrix[0])
    step = grid_step(window.unit_system.unit('length'))
    assert systems.units_defined and step > 0.0
    window._sliding = {'row': 0, 'axis': 0, 'start': before.copy(), 'from': 0.0}
    window._apply_slide(12.3 * step)
    moved = np.array(systems.cs_matrix[0])
    assert np.allclose(moved[:3], before[:3]), 'the basis stayed put'
    shown = window.unit_system.from_si(moved[3], 'length')
    assert np.allclose(shown / step, np.round(shown / step)), 'on the grid'
    assert np.allclose(moved[3] - before[3],
                       window.unit_system.to_si(12.0 * step, 'length') * before[0],
                       atol=1e-9), 'twelve steps along x, the third of a step dropped'
    window._commit_slide()
    pump()
    assert window._sliding is None
    assert 'Moved coordinate system' in window.statusBar().currentMessage()
    assert sorted({i.row() for i
                   in window.table.selectionModel().selectedIndexes()}) == [0]


def test_a_dragged_turn_snaps_to_a_degree_and_a_typed_one_does_not(systems,
                                                                    window, pump):
    window.table.selectRow(0)
    pump()
    window.rotate_action.setChecked(True)
    pump()
    before = np.array(systems.cs_matrix[0])
    window._rotating = {'row': 0, 'axis': 2, 'start': before.copy(), 'from': 0.0}
    applied = window._apply_rotation(np.radians(37.4), snap=True)
    assert np.degrees(applied) == pytest.approx(37.0)
    assert window._gizmo_reading == 'Turn about Z: +37°'
    window._commit_rotation()
    pump()
    assert window._gizmo_reading is None
    systems.cs_matrix[0] = before
    window._rotating = {'row': 0, 'axis': 2, 'start': before.copy(), 'from': 0.0}
    applied = window._apply_rotation(np.radians(37.4))
    assert np.degrees(applied) == pytest.approx(37.4), 'typed exactly'
    window._commit_rotation()
    pump()


def test_turns_and_slides_journal_as_calls_that_replay(systems, window,
                                                      pump):
    """The rings and arrows wrote the frame and journaled nothing, so a
    replayed session lost every turn (found 2026-10-09). Each gesture
    now ends as one `place_coordinate_system` call, a second nudge of
    the same kind settling the line rather than adding one."""
    window.table.selectRow(0)
    pump()
    window.rotate_action.setChecked(True)
    pump()
    start = np.array(systems.cs_matrix[0])
    window._rotating = {'row': 0, 'axis': 2, 'start': start.copy(),
                        'from': 0.0}
    window._apply_rotation(np.radians(37.4), snap=True)
    window._commit_rotation()
    pump()
    assert window.project.journal[-1].endswith(
        'place_coordinate_system(1, angles=(0.0, 0.0, 37.0))'), \
        'the drag ends as its own line'
    window.angle_box.setValue(12.5)
    pump()
    window._sliding = {'row': 0, 'axis': 0,
                       'start': np.array(systems.cs_matrix[0]), 'from': 0.0}
    window._apply_slide(0.25)
    window._commit_slide()
    pump()
    lines = [line for line in window.project.journal
             if '.place_coordinate_system(' in line]
    assert len(lines) == 2, lines
    assert 'angles=' in lines[0] and 'origin=' in lines[1]
    assert not np.allclose(systems.cs_matrix[0], start), 'it did move'
    room: dict = {}
    exec(window.project.session_script(), room)          # noqa: S102
    replayed = room['project']['Geometry']
    assert np.allclose(replayed.cs_matrix[0], systems.cs_matrix[0],
                       atol=1e-9)


def test_place_coordinate_system_turns_and_moves_one_frame():
    import visualdynamics
    from visualdynamics.rotate import frame_from_angles

    geometry = visualdynamics.import_file(fixture_path('plate',
                                                       'geometry.npz'))
    cs = geometry.add_coordinate_system(origin=(1.0, 2.0, 3.0))
    geometry.place_coordinate_system(cs, angles=(0.0, 0.0, 90.0))
    row = list(geometry.cs_id).index(cs)
    assert np.allclose(geometry.cs_matrix[row, :3],
                       frame_from_angles((0, 0, 90))[:3])
    assert np.allclose(geometry.cs_matrix[row, 3], (1.0, 2.0, 3.0)), \
        'a turn keeps the origin'
    geometry.place_coordinate_system(cs, origin=(4.0, 5.0, 6.0))
    assert np.allclose(geometry.cs_matrix[row, :3],
                       frame_from_angles((0, 0, 90))[:3]), 'a move keeps the turn'
    geometry.place_coordinate_system(cs, rotation=np.eye(3))
    assert np.allclose(geometry.cs_matrix[row, :3], np.eye(3))
    with pytest.raises(ValueError, match='not both'):
        geometry.place_coordinate_system(cs, angles=(0, 0, 0),
                                         rotation=np.eye(3))
    with pytest.raises(KeyError, match='no coordinate system 999'):
        geometry.place_coordinate_system(999, angles=(0, 0, 0))
