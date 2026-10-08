"""Screen-space picking: is the right thing under the cursor?"""


import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics


def scene(model):
    """A rendered off-screen view plus a projector over its nodes.
    `model` is a fixture path, or a Geometry brought along ready-made."""
    import pyvista as pv

    from visualdynamics.viz.geometry import geometry_scene
    from visualdynamics.viz.pick import ScreenProjector

    geometry = (model if isinstance(model, visualdynamics.Geometry)
                else visualdynamics.import_file(fixture_path(model),
                                                length_unit='m'))
    plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    geometry_scene(geometry, plotter=plotter, unit_system=visualdynamics.SI)
    plotter.camera_position = 'iso'
    plotter.render()
    return geometry, plotter, ScreenProjector(plotter.renderer,
                                              geometry.node_xyz)


def test_projection_matches_vtk():
    """Our matrix must agree with VTK's own world-to-display conversion."""
    import vtk

    geometry, plotter, projector = scene('plate/geometry.exo')
    screen, _ = projector.screen()
    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    for row in (0, 7, 22, 44):
        coordinate.SetValue(*geometry.node_xyz[row])
        expected = coordinate.GetComputedDoubleDisplayValue(plotter.renderer)
        assert np.allclose(screen[row], expected, atol=1e-6)
    plotter.close()


def test_every_node_picks_itself():
    from visualdynamics.viz.pick import EntityPicker

    geometry, plotter, projector = scene('plate/geometry.exo')
    screen, _ = projector.screen()
    picker = EntityPicker(geometry, 'nodes', projector)
    for row in range(geometry.num_nodes):
        assert picker.pick(*screen[row]) == int(geometry.node_id[row])
    plotter.close()


def test_clicking_inside_a_face_picks_that_element():
    from visualdynamics.viz.pick import EntityPicker

    geometry, plotter, projector = scene('plate/geometry.exo')
    screen, _ = projector.screen()
    picker = EntityPicker(geometry, 'elements', projector)
    lookup = {int(n): i for i, n in enumerate(geometry.node_id)}
    for element in range(len(geometry.elem_conn)):
        rows = [lookup[int(n)] for n in geometry.elem_conn[element]]
        assert picker.pick(*screen[rows].mean(axis=0)) == element
    plotter.close()


def test_empty_space_picks_nothing():
    from visualdynamics.viz.pick import EntityPicker

    geometry, plotter, projector = scene('plate/geometry.exo')
    picker = EntityPicker(geometry, 'nodes', projector)
    assert picker.pick(2, 2) is None
    plotter.close()


def test_screen_distance_beats_depth():
    """A node under the cursor wins over a nearer-to-camera node beside it.

    This is the failure the sdynpy editor had: picking by 3-D proximity to a
    ray hit selects things that are not under the pointer.
    """
    from visualdynamics.viz.pick import _nearest_candidate

    distances = np.array([0.0, 4.0])
    depth = np.array([0.9, 0.1])         # the far-from-cursor one is in front
    assert _nearest_candidate(distances, depth, tolerance=12.0) == 0


def test_depth_breaks_a_genuine_tie():
    from visualdynamics.viz.pick import _nearest_candidate

    distances = np.array([0.5, 0.4])     # both effectively under the cursor
    depth = np.array([0.9, 0.1])
    assert _nearest_candidate(distances, depth, tolerance=12.0) == 1


def test_tolerance_is_respected():
    from visualdynamics.viz.pick import _nearest_candidate

    assert _nearest_candidate(np.array([20.0]), np.array([0.5]),
                              tolerance=12.0) is None


def test_beam_picking_is_accurate_where_unambiguous():
    """Every drawn-line segment whose midpoint is not shared with
    another (the survey's lines are element groups of beams, 2026-09-30)."""
    from visualdynamics.viz.pick import EntityPicker, _segment_distances

    # the survey's drawn lines on their own: its faces' edges lie on
    # the same pixels and would make every segment ambiguous
    survey = visualdynamics.import_file(fixture_path('plate/test_geometry.npz'),
                                        length_unit='m')
    drawn = {line['group'] for line in survey.drawn_lines()}
    survey.delete_groups([int(b) for b in survey.group_id
                          if int(b) not in drawn])
    geometry, plotter, projector = scene(survey)
    screen, _ = projector.screen()
    picker = EntityPicker(geometry, 'elements', projector)
    starts, ends = picker.segments
    lookup = {int(n): i for i, n in enumerate(geometry.node_id)}

    checked = 0
    for index in range(len(geometry.elem_conn)):
        rows = [lookup[int(n)] for n in geometry.elem_conn[index]]
        midpoint = screen[rows].mean(axis=0)
        distances = _segment_distances(midpoint, screen[starts], screen[ends])
        others = distances[picker.owner != index]
        if others.size and others.min() <= 2.0:
            continue          # another segment lies on the same pixels
        checked += 1
        assert picker.pick(*midpoint) == index
    # the survey display threads six lines through the grid; the ones
    # whose midpoints land on a crossing line are skipped as ambiguous,
    # and at least a couple always stand clear
    assert checked >= 2, 'some segments should be unambiguous'
    plotter.close()


def test_projection_caches_until_the_camera_moves():
    _geometry, plotter, projector = scene('plate/geometry.exo')
    first, _ = projector.screen()
    assert projector.screen()[0] is first, 'cached while the camera is still'
    plotter.camera.azimuth = 45
    plotter.render()
    assert projector.screen()[0] is not first, 'recomputed after a camera move'
    plotter.close()


def test_coordinate_systems_pick_by_origin():
    """Coordinate systems are picked by their origins, which must be apart —
    the demo models stack all of theirs at (0, 0, 0)."""
    import pyvista as pv

    from visualdynamics.viz.geometry import geometry_scene
    from visualdynamics.viz.pick import EntityPicker, ScreenProjector

    identity = np.vstack([np.eye(3), np.zeros(3)])
    offset = np.vstack([np.eye(3), [1.0, 0.0, 0.0]])
    geometry = visualdynamics.Geometry(
        node_id=[1, 2], node_xyz=[[0, 0, 0], [1, 0, 0]],
        cs_id=[1, 2], cs_name=['', ''], cs_type=[0, 0],
        cs_matrix=[identity, offset], length_unit='m')
    plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
    geometry_scene(geometry, plotter=plotter, unit_system=visualdynamics.SI)
    plotter.render()

    projector = ScreenProjector(plotter.renderer, geometry.node_xyz)
    picker = EntityPicker(geometry, 'coordinate_systems', projector)
    origins = ScreenProjector(plotter.renderer,
                              geometry.cs_matrix[:, 3, :]).screen()[0]
    for index, origin in enumerate(origins):
        assert picker.pick(*origin) == int(geometry.cs_id[index])
    plotter.close()


def test_coincident_origins_pick_one_of_them():
    """The survey stacks three coordinate systems at the same point; a
    pick must return one of them rather than nothing."""
    from visualdynamics.viz.pick import EntityPicker

    geometry, plotter, projector = scene('plate/geometry.npz')
    picker = EntityPicker(geometry, 'coordinate_systems', projector)
    from visualdynamics.viz.pick import ScreenProjector
    origin = ScreenProjector(plotter.renderer,
                             geometry.cs_matrix[:, 3, :]).screen()[0][0]
    assert picker.pick(*origin) in set(geometry.cs_id.tolist())
    plotter.close()


def test_picking_is_fast_enough_to_hover():
    """A mouse move must not cost more than a frame."""
    import time

    from visualdynamics.viz.pick import EntityPicker

    geometry, plotter, projector = scene('plate/geometry.npz')
    picker = EntityPicker(geometry, 'nodes', projector)
    picker.pick(400, 300)                       # warm the projection cache
    start = time.perf_counter()
    for _ in range(200):
        picker.pick(400, 300)
    per_pick = (time.perf_counter() - start) / 200
    assert per_pick < 0.005, f'{per_pick * 1000:.2f} ms per pick'
    plotter.close()


@pytest.mark.parametrize('component,model', [
    ('nodes', 'plate/geometry.exo'),
    ('elements', 'plate/test_geometry.npz'),  # the survey's drawn lines
    ('elements', 'plate/geometry.exo'),
])
def test_hover_cells_reference_scene_points(component, model):
    from visualdynamics.gui.main_window import _hover_cells

    geometry = visualdynamics.import_file(fixture_path(model),
                                length_unit='m')
    entity = int(geometry.node_id[3]) if component == 'nodes' else 0
    cells = _hover_cells(geometry, component, entity)
    assert len(cells['verts']) or len(cells['lines'])
    indices = np.concatenate([cells['verts'], cells['lines']])
    assert indices.max() <= geometry.num_nodes


def test_hover_cells_are_empty_for_a_missing_entity():
    from visualdynamics.gui.main_window import _hover_cells

    geometry = visualdynamics.import_file(
        fixture_path('plate', 'geometry.exo'), length_unit='m')
    cells = _hover_cells(geometry, 'nodes', 999999)
    assert not len(cells['verts']) and not len(cells['lines'])


def test_unprojection_round_trips():
    """A node's own pixel must unproject back to the node."""
    geometry, plotter, projector = scene('plate/geometry.exo')
    screen, depth = projector.screen()
    for row in range(geometry.num_nodes):
        back = projector.unproject(*screen[row], depth[row])
        assert np.allclose(back, geometry.node_xyz[row], atol=1e-9)
    plotter.close()


def test_new_points_land_on_a_flat_model():
    """Clicking between nodes of a flat plate places the point on its plane,
    not floating above it."""
    _geometry, plotter, projector = scene('plate/geometry.exo')
    screen, _ = projector.screen()
    assert projector._plane_normal() is not None, 'the plate is planar'
    point = projector.place_point(*(screen[0] + screen[8]) / 2)
    assert abs(point[2]) < 1e-9
    plotter.close()


def test_place_point_reproduces_a_clicked_node():
    geometry, plotter, projector = scene('plate/geometry.npz')
    screen, _ = projector.screen()
    for row in (0, 40, 112):
        assert np.allclose(projector.place_point(*screen[row]),
                           geometry.node_xyz[row], atol=1e-9)
    plotter.close()


def solid_box():
    """A genuinely three-dimensional wireframe — the plate is planar,
    which is exactly what this test needs the model not to be."""
    corners = [(x, y, z) for z in (0.0, 0.4)
               for y in (0.0, 0.3) for x in (0.0, 0.5)]
    lines = [[1, 2, 4, 3, 1], [5, 6, 8, 7, 5],
             [1, 5], [2, 6], [3, 7], [4, 8]]
    geometry = visualdynamics.Geometry(
        node_id=list(range(1, 9)), node_xyz=corners, length_unit='m')
    for line in lines:
        geometry.add_beams(line)
    return geometry


def test_a_three_dimensional_model_has_no_plane():
    _geometry, plotter, projector = scene(solid_box())
    assert projector._plane_normal() is None
    plotter.close()
