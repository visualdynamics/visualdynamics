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


def test_the_rings_stay_after_a_turn_and_a_slide(systems, window, pump):
    """A committed drag refreshes the table row, and an edited row
    redraws the scene from scratch — which took the rings with it,
    leaving them answering clicks where they could no longer be seen
    (2026-10-10). They are back after every kind of commit."""
    systems.define_units('m')
    window.table.selectRow(0)
    pump()
    window.rotate_action.setChecked(True)
    pump()
    before = np.array(systems.cs_matrix[0])
    window._rotating = {'row': 0, 'axis': 2, 'start': before.copy(), 'from': 0.0}
    window._apply_rotation(np.radians(20.0), snap=True)
    window._commit_rotation()
    pump()
    assert (len(rings(window)), len(arrows(window))) == (3, 3), 'after a turn'
    start = np.array(systems.cs_matrix[0])
    window._sliding = {'row': 0, 'axis': 0, 'start': start, 'from': 0.0}
    window._apply_slide(0.05)
    window._commit_slide()
    pump()
    assert (len(rings(window)), len(arrows(window))) == (3, 3), 'after a slide'
    window.table.selectRow(1)
    pump()
    assert window.rotate_action.isChecked()
    assert (len(rings(window)), len(arrows(window))) == (3, 3), (
        'and on the next system picked')


def clear_of_the_rest(window, systems, kind, axis, offset):
    """A pixel `offset` off one ring or arrow, where nothing else of the
    gizmo is near. Returns (pixel, its clearance)."""
    from visualdynamics.viz.pick import segment_distances

    parts = ([('ring', a, p) for a, p in window._ring_screen_points()]
             + [('arrow', a, p) for a, p in window._arrow_screen_points()])
    line = next(p for k, a, p in parts if (k, a) == (kind, axis))
    others = [p for k, a, p in parts if (k, a) != (kind, axis)]

    def clearance(point):
        return min(segment_distances(point, p[:-1], p[1:]).min()
                   for p in others)

    i = max(range(len(line) - 1),
            key=lambda i: clearance((line[i] + line[i + 1]) / 2))
    a, b = line[i], line[i + 1]
    normal = np.array([-(b - a)[1], (b - a)[0]])
    normal /= np.linalg.norm(normal)
    pixel = max(((a + b) / 2 + offset * normal, (a + b) / 2 - offset * normal),
                key=clearance)
    return pixel, clearance(pixel)


def turning(systems, window, pump):
    """The rings on for the first system, and the camera closed in on
    them, so they are big enough to have stretches clear of each other
    by more than the reach."""
    systems.define_units('m')
    window.table.selectRow(0)
    pump()
    window.rotate_action.setChecked(True)
    pump()
    plotter = window.scene.plotter
    plotter.camera.focal_point = systems.cs_matrix[0][3]
    plotter.camera.zoom(4.0)
    plotter.render()


def test_a_press_reaches_fourteen_points_on_any_screen(systems, window, pump,
                                                       monkeypatch):
    """The cursor and the projected rings are both in device pixels, so
    the reach is too: 28 of them on a Retina screen, where 14 was a
    7-point target (2026-10-10)."""
    turning(systems, window, pump)
    monkeypatch.setattr(window, 'devicePixelRatioF', lambda: 2.0)
    assert window._gizmo_reach() == 28.0
    press, clear = clear_of_the_rest(window, systems, 'ring', 0, 20.0)
    assert clear > 28.0, 'nothing else within reach'
    monkeypatch.setattr(window, '_cursor_position', lambda: press)
    window._on_rotate_press(None, None)
    assert window._rotating is not None and window._rotating['axis'] == 0


def colors_of(window):
    """Every gizmo actor's color, by name."""
    from pyvista import Color

    return {name: Color(actor.prop.color).hex_rgb
            for name, actor in window.scene.plotter.renderer.actors.items()
            if name.startswith(('rotate-ring', 'rotate-arrow', 'rotate-head'))}


def test_what_a_press_would_take_lights_up_under_the_cursor(systems, window,
                                                            pump, monkeypatch):
    """Hovering a ring paints it in the scene's highlight — the color a
    hovered node takes — and only it; an arrow lights shaft and head
    together; moving off puts the axis colors back; and the light
    survives the redraw a drag makes (2026-10-10)."""
    from visualdynamics.theme import theme
    from visualdynamics.viz.geometry import AXIS_COLORS

    turning(systems, window, pump)
    highlight = theme(window.theme_name)['scene_highlight']
    resting = colors_of(window)
    assert len(resting) == 9
    assert all(color != highlight for color in resting.values())

    cursor = [None]
    monkeypatch.setattr(window, '_cursor_position', lambda: cursor[0])
    cursor[0], clear = clear_of_the_rest(window, systems, 'ring', 1, 4.0)
    assert clear > window._gizmo_reach()
    window._on_rotate_move(None, None)
    lit = {n for n, c in colors_of(window).items() if c == highlight}
    assert lit == {'rotate-ring-1'}
    assert window._gizmo_under_cursor(cursor[0]) == ('ring', 1), (
        'the press would take what is lit')
    window._draw_rings()
    assert {n for n, c in colors_of(window).items()
            if c == highlight} == {'rotate-ring-1'}, 'redrawn still lit'

    cursor[0], _ = clear_of_the_rest(window, systems, 'arrow', 2, 4.0)
    window._on_rotate_move(None, None)
    assert {n for n, c in colors_of(window).items() if c == highlight} == {
        'rotate-arrow-2', 'rotate-head-2'}

    cursor[0] = np.array([-1000.0, -1000.0])
    window._on_rotate_move(None, None)
    assert colors_of(window)['rotate-arrow-2'] == AXIS_COLORS[2]
    assert not any(c == highlight for c in colors_of(window).values())


def test_switching_off_forgets_what_was_hovered(systems, window, pump,
                                                monkeypatch):
    turning(systems, window, pump)
    point, _ = clear_of_the_rest(window, systems, 'ring', 0, 4.0)
    monkeypatch.setattr(window, '_cursor_position', lambda: point)
    window._on_rotate_move(None, None)
    assert window._gizmo_hovered == ('ring', 0)
    window.rotate_action.setChecked(False)
    pump()
    assert window._gizmo_hovered is None
    window.rotate_action.setChecked(True)
    pump()
    assert not any(c == theme_highlight(window)
                   for c in colors_of(window).values()), 'drawn unlit'


def theme_highlight(window):
    from visualdynamics.theme import theme

    return theme(window.theme_name)['scene_highlight']


def test_nothing_lights_while_the_camera_is_being_dragged(systems, window,
                                                          pump, monkeypatch):
    """An orbit sweeps the rings past a cursor whose button is down;
    lighting them as they pass would offer what a press cannot take."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    turning(systems, window, pump)
    point, _ = clear_of_the_rest(window, systems, 'ring', 0, 4.0)
    monkeypatch.setattr(window, '_cursor_position', lambda: point)
    monkeypatch.setattr(QApplication, 'mouseButtons',
                        staticmethod(lambda: Qt.MouseButton.LeftButton))
    window._on_rotate_move(None, None)
    assert window._gizmo_hovered is None
    assert not any(c == theme_highlight(window)
                   for c in colors_of(window).values())
