"""The Perspective toggle on the 3-D views: perspective or parallel
projection, kept through every scene, with the scene's size on screen
held across the switch and across a redraw."""

import numpy as np
import pytest

import visualdynamics
from visualdynamics.demo import plate
from visualdynamics.viz.geometry import (
    held_camera,
    keep_parallel_width,
    restore_camera,
    set_perspective,
)


@pytest.fixture
def plotter():
    import pyvista as pv

    made = pv.Plotter(off_screen=True, window_size=(300, 200))
    made.add_mesh(pv.Cube())
    made.camera_position = [(0.0, -8.0, 3.0), (0.0, 0.0, 0.0), (0, 0, 1)]
    yield made
    made.close()


def _width(camera):
    """Half the height of what the camera shows at its focal point."""
    if camera.parallel_projection:
        return camera.parallel_scale
    return camera.distance * np.tan(np.radians(camera.view_angle) / 2.0)


def test_the_switch_keeps_the_scene_its_size(plotter):
    camera = plotter.camera
    assert not camera.parallel_projection
    before = _width(camera)
    set_perspective(plotter, False)
    assert camera.parallel_projection
    assert np.isclose(_width(camera), before)
    # zooming a parallel view moves its scale and not the camera: back in
    # perspective the camera is where it shows the same
    camera.parallel_scale = before / 4.0
    direction = np.asarray(camera.position) / np.linalg.norm(camera.position)
    set_perspective(plotter, True)
    assert not camera.parallel_projection
    assert np.isclose(_width(camera), before / 4.0)
    assert np.allclose(
        np.asarray(camera.position) / np.linalg.norm(camera.position),
        direction), 'from the same side'
    # asked for what it already is, nothing moves
    camera.parallel_scale = 5.0     # unused in perspective, and left so
    held = (camera.position, camera.parallel_scale)
    set_perspective(plotter, True)
    assert (camera.position, camera.parallel_scale) == held


def test_a_saved_camera_carries_the_parallel_scale(plotter):
    set_perspective(plotter, False)
    plotter.camera.parallel_scale = 0.37
    held = held_camera(plotter)
    plotter.reset_camera()
    assert not np.isclose(plotter.camera.parallel_scale, 0.37)
    restore_camera(plotter, held)
    assert np.isclose(plotter.camera.parallel_scale, 0.37)
    assert np.allclose(plotter.camera.position, (0.0, -8.0, 3.0))


def test_a_placed_camera_is_as_wide_in_parallel(plotter):
    wide = _width(plotter.camera)
    set_perspective(plotter, False)
    plotter.camera.parallel_scale = 99.0
    keep_parallel_width(plotter)
    assert np.isclose(plotter.camera.parallel_scale, wide)
    set_perspective(plotter, True)
    plotter.camera.parallel_scale = scale = 99.0
    keep_parallel_width(plotter)
    assert plotter.camera.parallel_scale == scale, 'perspective: left alone'


def test_the_scene_bar_toggles_it_and_it_stays(window, pump):
    from conftest import select_objects

    built = plate.build(4)
    window.add_object('Plate', built.geometry())
    window.add_object('Other', visualdynamics.Geometry(
        [1, 2, 3], [[0, 0, 0], [1, 0, 0], [0, 2, 0]]))
    pump()
    select_objects(window, pump, 'Plate')
    scene = window.scene
    action = scene.perspective_action
    camera = scene.plotter.camera
    assert action.isVisible() and action.isChecked()
    assert not camera.parallel_projection
    action.trigger()
    pump()
    assert not scene.perspective and scene.plotter.camera.parallel_projection
    # another object, Reset View and a redraw: still parallel
    select_objects(window, pump, 'Other')
    assert scene.plotter.camera.parallel_projection
    scene.reset_view_action.trigger()
    assert scene.plotter.camera.parallel_projection
    # the view belongs to the user: a redraw keeps a parallel view's zoom
    scene.plotter.camera.parallel_scale *= 0.5
    zoomed = scene.plotter.camera.parallel_scale
    window.render_current()
    pump()
    assert np.isclose(scene.plotter.camera.parallel_scale, zoomed)
    # a camera a drawing brought with it takes the pane's projection
    scene.plotter.camera.parallel_projection = False
    scene.apply_annotations()
    assert scene.plotter.camera.parallel_projection
    scene.set_perspective(True)
    assert action.isChecked() and not scene.plotter.camera.parallel_projection


def test_the_data_pane_offers_it_while_its_3d_form_is_up(window, pump):
    from conftest import select_objects

    lines = np.linspace(0.0, 100.0, 64)
    window.add_object('Spectra', visualdynamics.Psd(
        lines, np.abs(np.random.default_rng(0).normal(size=(4, 64))) + 1.0,
        response_dof=[f'{n}Z+' for n in range(1, 5)],
        ordinate_dim=['acceleration**2/frequency'] * 4))
    pump()
    select_objects(window, pump, 'Spectra')
    pane = window.data_pane
    assert pane.showing_waterfall and pane.waterfall_plotter is not None
    action = pane.perspective_action
    assert action.isVisible() and action.isChecked()
    assert not pane.waterfall_plotter.camera.parallel_projection
    action.trigger()
    pump()
    assert pane.waterfall_plotter.camera.parallel_projection
    window.render_current()
    pump()
    assert pane.waterfall_plotter.camera.parallel_projection
    pane.waterfall_action.trigger()      # the flat plot: nothing to project
    pump()
    assert not action.isVisible()
    pane.waterfall_action.trigger()
    pump()
    assert action.isVisible() and not action.isChecked()
    assert pane.waterfall_plotter.camera.parallel_projection
