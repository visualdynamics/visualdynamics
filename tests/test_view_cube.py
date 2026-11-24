"""The orientation marker's cube: a click on a face turns the view to
look straight at that face, and again to the opposite one; while the
view is square, its arrows turn it a quarter turn."""

import numpy as np
import pytest

from visualdynamics.demo import plate
from visualdynamics.viz.geometry import annotate_scene
from visualdynamics.viz.view_cube import (
    ViewCube,
    look_at_face,
    show_view_cube,
    turn_view,
    view_cube,
)

FACES = [(axis, sign) for axis in range(3) for sign in (1, -1)]


@pytest.fixture
def plotter():
    import pyvista as pv

    made = pv.Plotter(off_screen=True, window_size=(600, 400))
    made.add_mesh(pv.Cone())
    made.camera_position = [(5.0, -7.0, 4.0), (0.5, 0.25, 0.0), (0, 0, 1)]
    made.show(auto_close=False)
    yield made
    made.close()


@pytest.mark.parametrize('axis, sign', FACES)
def test_a_face_is_looked_at_from_its_own_side(plotter, axis, sign):
    camera = plotter.camera
    focus, distance = np.array(camera.focal_point), camera.distance
    assert look_at_face(plotter, axis, sign) == (axis, sign)
    toward = np.zeros(3)
    toward[axis] = sign
    assert np.allclose(camera.direction, -toward)
    # what is looked at and how far away stay: the scene keeps its size
    assert np.allclose(camera.focal_point, focus)
    assert np.isclose(camera.distance, distance)
    # Z up from the sides; Y up from above and down from below
    up = (0.0, 0.0, 1.0) if axis != 2 else (0.0, float(sign), 0.0)
    assert np.allclose(camera.up, up)
    # the same face again is the opposite one
    assert look_at_face(plotter, axis, sign) == (axis, -sign)
    assert np.allclose(camera.direction, toward)
    assert np.allclose(camera.focal_point, focus)
    assert look_at_face(plotter, axis, sign) == (axis, sign)


def test_the_marker_turns_with_the_camera_and_its_faces_are_picked(plotter):
    show_view_cube(plotter, True, 'black')
    cube = view_cube(plotter)
    assert sorted(cube.faces) == sorted(FACES)
    assert cube.renderer.GetLayer() > plotter.renderer.GetLayer(), \
        'over the scene, not painting its corner'
    for face in FACES:
        look_at_face(plotter, *face)
        plotter.render()
        assert np.allclose(
            cube.renderer.GetActiveCamera().GetDirectionOfProjection(),
            plotter.camera.direction)
        assert np.allclose(cube.renderer.GetActiveCamera().GetViewUp(),
                           plotter.camera.up)
        # the face looked at is the one in front, and its opposite's
        # center lies behind it
        assert cube.face_at(*cube.face_pixel(*face)) == face
        assert cube.face_at(*cube.face_pixel(face[0], -face[1])) == face
    # from a corner three faces show, each under its own center
    plotter.camera_position = [(5.0, 5.0, 5.0), (0, 0, 0), (0, 0, 1)]
    plotter.render()
    for face in ((0, 1), (1, 1), (2, 1)):
        assert cube.face_at(*cube.face_pixel(*face)) == face
    # an arrow is not a face: past the cube, along X, nothing is picked
    cube.renderer.SetWorldPoint(0.8, 0.0, 0.0, 1.0)
    cube.renderer.WorldToDisplay()
    assert cube.face_at(*cube.renderer.GetDisplayPoint()[:2]) is None
    # off the cube, in the marker's corner and out of it
    assert cube.face_at(3, 3) is None
    assert cube.face_at(300, 200) is None


def test_the_marker_is_kept_recolored_and_taken_away(plotter):
    window = plotter.render_window
    before = window.GetRenderers().GetNumberOfItems()
    watched = plotter.renderer.HasObserver('StartEvent')
    show_view_cube(plotter, True, 'black')
    first = view_cube(plotter)
    assert isinstance(first, ViewCube)
    assert window.GetRenderers().GetNumberOfItems() == before + 1
    show_view_cube(plotter, True, 'black')
    assert view_cube(plotter) is first, 'drawn again, the same one'
    show_view_cube(plotter, True, 'white')
    assert view_cube(plotter) is not first
    assert window.GetRenderers().GetNumberOfItems() == before + 1
    show_view_cube(plotter, False)
    assert view_cube(plotter) is None
    assert window.GetRenderers().GetNumberOfItems() == before
    assert plotter.renderer.HasObserver('StartEvent') == watched, \
        'nothing left watching the scene'
    show_view_cube(plotter, False)      # nothing there: nothing to do


def test_a_figure_keeps_the_plain_triad(plotter):
    colors = {'scene_text': 'black'}
    annotate_scene(plotter, 'm', colors, bounds=False, orientation=True)
    assert view_cube(plotter) is None
    assert plotter.renderer.axes_enabled
    plotter.hide_axes()
    annotate_scene(plotter, 'm', colors, bounds=False, orientation=True,
                   cube=True)
    assert view_cube(plotter) is not None
    assert not plotter.renderer.axes_enabled
    annotate_scene(plotter, 'm', colors, bounds=False, orientation=False,
                   cube=True)
    assert view_cube(plotter) is None


def test_a_click_on_the_cube_turns_the_view(window, pump):
    from conftest import select_objects

    window.add_object('Plate', plate.build(4).geometry())
    pump()
    select_objects(window, pump, 'Plate')
    scene = window.scene
    plotter = scene.plotter
    plotter.render()
    cube = view_cube(plotter)
    assert cube is not None
    held = plotter.camera.position
    assert not scene.click_view_cube(3, 3)
    assert plotter.camera.position == held, 'off the cube, the view stays'
    assert scene.click_view_cube(*cube.face_pixel(2, 1))
    assert np.allclose(plotter.camera.direction, (0, 0, -1))
    # a redraw keeps the marker and the view it turned to
    window.render_current()
    pump()
    assert view_cube(plotter) is cube
    assert np.allclose(plotter.camera.direction, (0, 0, -1))
    # the triad's own toggle takes the cube with it
    scene.orientation_action.trigger()
    assert view_cube(plotter) is None
    assert not scene.click_view_cube(*cube.face_pixel(2, 1))
    scene.orientation_action.trigger()
    assert view_cube(plotter) is not None


def test_the_turn_arrows_show_only_while_square_and_turn_a_quarter(plotter):
    show_view_cube(plotter, True, 'black')
    cube = view_cube(plotter)
    look_at_face(plotter, 2, 1)
    plotter.render()
    held = plotter.camera.position
    assert cube.arrow_at(*cube.arrow_pixel(1)) == 1
    assert cube.arrow_at(*cube.arrow_pixel(-1)) == -1
    # an arrow is not a face, and a face is not an arrow
    assert cube.face_at(*cube.arrow_pixel(1)) is None
    assert cube.arrow_at(*cube.face_pixel(2, 1)) is None
    # from above, Y up and X right: clockwise puts Y on the right
    turn_view(plotter, 1)
    assert np.allclose(plotter.camera.up, (-1, 0, 0))
    assert plotter.camera.position == held, 'only which way is up changes'
    # still square, so the arrows stay, drawn where they were
    assert cube.arrow_at(*cube.arrow_pixel(1)) == 1
    turn_view(plotter, -1)
    turn_view(plotter, -1)
    assert np.allclose(plotter.camera.up, (1, 0, 0))
    # from a side, about the line of sight too
    look_at_face(plotter, 0, 1)
    turn_view(plotter, 1)
    assert np.allclose(plotter.camera.direction, (-1, 0, 0))
    assert np.allclose(plotter.camera.up, (0, -1, 0))
    # turned off square, they are gone
    plotter.camera_position = [(5.0, 5.0, 5.0), (0, 0, 0), (0, 0, 1)]
    plotter.render()
    assert cube.arrow_at(*cube.arrow_pixel(1)) is None
    assert cube.arrow_at(*cube.arrow_pixel(-1)) is None


def test_a_click_on_a_turn_arrow_turns_the_view(window, pump):
    from conftest import select_objects

    window.add_object('Plate', plate.build(4).geometry())
    pump()
    select_objects(window, pump, 'Plate')
    scene = window.scene
    plotter = scene.plotter
    plotter.render()
    cube = view_cube(plotter)
    assert scene.click_view_cube(*cube.face_pixel(2, 1))
    plotter.render()
    assert np.allclose(plotter.camera.up, (0, 1, 0))
    assert scene.click_view_cube(*cube.arrow_pixel(-1))
    assert np.allclose(plotter.camera.up, (1, 0, 0))
    assert np.allclose(plotter.camera.direction, (0, 0, -1))
