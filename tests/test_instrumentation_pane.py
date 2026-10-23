"""Plan Instrumentation in the window: the act on a shape set, the pane
beside the 3-D view, the sensors drawn as the counts change, and Keep.

The pane follows select, see, set, apply (principle 13). The plan
itself is `core.instrumentation`'s and is tested there; these check
that the window offers it where it applies, shows what it chose, reads
it, refuses what cannot be planned, and keeps what `Project.
plan_instrumentation` would make.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from conftest import select_objects

from visualdynamics.core.fem import MATERIALS, GroupProperties
from visualdynamics.core.instrumentation import FewerDirectionsWarning


@pytest.fixture
def bar(window, pump):
    """A free 8 x 2 x 1 in steel bar and its first fourteen modes."""
    project = window.project
    name = project.new_geometry('Bar', unit='in')
    solid = project.add_solid_block(name, (0, 0, 0), (8, 2, 1), name='bar', unit='in')
    project.mesh_solids(name, [solid], 0.5, unit='in')
    geometry = project[name]
    for group in geometry.group_id:
        geometry.group_properties[int(group)] = GroupProperties(MATERIALS['1018 steel'])
    modes = project.solve_modes(name, num_modes=14)
    window.show_object(name)
    window.show_object(modes)
    pump()
    return modes


def _arrows(window):
    return [name for name in window.scene.plotter.renderer.actors if name.startswith('plan-arrows')]


def _open(window, pump):
    acts = {act[0]: act for act in window.acts_for()}
    acts['plan_instrumentation'][3]()
    pump()
    return window.scene.instrumentation_panel


def _to_the_edges(window, panel):
    """Let sensors sit up to the edges: a quarter of an inch in, the
    half-inch bar's 1 in faces keep only their center lines, and these
    tests are about other things."""
    panel.edge_distance.setValue(0.0)
    window._plan_now()


def test_shapes_on_a_model_offer_the_plan_and_the_model_alone_does_not(bar, window, pump):
    select_objects(window, pump, bar)
    assert 'plan_instrumentation' in {act[0] for act in window.acts_for()}
    select_objects(window, pump, 'Bar')
    assert 'plan_instrumentation' not in {act[0] for act in window.acts_for()}


def test_the_pane_opens_on_the_elastic_modes_and_draws_the_sensors(bar, window, pump):
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    _to_the_edges(window, panel)
    assert panel.isVisible() and window._planning is not None
    shapes = window.objects[bar]
    assert window._planning_modes() == [k for k, f in enumerate(shapes.frequency) if f > 0]
    assert panel.modes_label.text().startswith('Measuring 8 modes (7–14), ')
    assert 'every mode above 0 Hz' in panel.modes_label.text()
    assert panel.picked_groups == [int(g) for g in window.objects['Bar'].group_id]
    assert panel.surfaces_label.text().startswith('Sensors on 6 of 6 surfaces.')
    assert panel.edge_angle.value() == 20, 'surfaces split at 20 degrees by default'
    # eight modes: three tri-axial to measure them and one more
    assert (panel.triaxial.value(), panel.uniaxial.value()) == (4, 0)
    assert panel.name.text() == f'{bar} Plan'
    plan = window._planning['plan']
    assert plan is not None and len(plan.placement.triaxial) == 4
    assert panel.keep_button.isEnabled()
    assert 'tri-axial' in panel.reading_label.text() and 'Worst off-diagonal MAC' in \
        panel.reading_label.text()
    assert sorted(_arrows(window)) == ['plan-arrows0', 'plan-arrows1', 'plan-arrows2']
    assert 'plan-surfaces' in window.scene.plotter.renderer.actors


def test_a_count_changed_chooses_again_without_finding_the_places_again(bar, window, pump):
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    places, blocks = window._planning['places'], window._planning['blocks']
    panel.triaxial.setValue(2)
    panel.uniaxial.setValue(3)
    window._plan_now()
    plan = window._planning['plan']
    assert len(plan.placement.triaxial) == 2 and len(plan.placement.uniaxial) == 3
    assert plan.channels.num_channels == 2 * 3 + 3
    assert window._planning['places'] is places, 'the places were kept'
    assert window._planning['blocks'] is blocks, 'and the modes read at them'
    assert '2 tri-axial and 3 uni-axial' in panel.reading_label.text()


def test_a_plan_that_cannot_be_made_says_why_and_offers_no_keep(bar, window, pump):
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    panel.triaxial.setValue(0)
    panel.uniaxial.setValue(0)
    window._plan_now()
    assert window._planning['plan'] is None
    assert not panel.keep_button.isEnabled()
    assert panel.reading_label.text() == 'No plan: no sensors to place.'
    assert not _arrows(window), 'no sensors drawn'
    panel.groups.item(0).setCheckState(panel.groups.item(0).checkState().Unchecked)
    window._plan_now()
    assert 'check an element group' in panel.reading_label.text()
    assert not panel.keep_button.isEnabled()


def _pick_modes(window, pump, bar, modes):
    """Pick modes in the shape set's grid, as a click there does."""
    window.record_grids[bar].select_records(modes)
    window.record_grids[bar].selection_changed.emit()
    pump()


def test_the_modes_picked_are_the_modes_planned_for_and_may_change(bar, window, pump):
    item = window._item_for_object(bar)
    window.tree.clearSelection()
    item.setExpanded(True)
    pump()
    # the row first, then the modes in its grid: selecting the owner
    # row releases its grid picks, as a person's click on it does
    window.tree.setCurrentItem(item)
    pump()
    _pick_modes(window, pump, bar, [6, 7, 8])
    acts = window.acts_for()
    assert [act[0] for act in acts] == ['plan_instrumentation'], 'only the plan on mode picks'
    panel = _open(window, pump)
    _to_the_edges(window, panel)
    assert window._planning_modes() == [6, 7, 8]
    assert panel.modes_label.text().startswith('Measuring 3 modes (7–9), ')
    assert window._planning['plan'].shapes.num_shapes == 3
    places = window._planning['places']
    # more modes picked with the pane open: it stays and plans for them
    _pick_modes(window, pump, bar, [6, 7, 8, 10])
    assert panel.isVisible()
    assert panel.modes_label.text().startswith('Measuring 4 modes (7–9, 11), ')
    assert window._planning['plan'].shapes.num_shapes == 4
    assert window._planning['places'] is places, 'the places were kept'
    panel.keep_button.click()
    pump()
    assert window.project.journal[-1] == (
        f'project.plan_instrumentation({bar!r}, 2, 0, modes=[6, 7, 8, 10], '
        'edge_distance=0.0)'), 'the picks kept'


def _click(window, monkeypatch, start, end=None):
    """A press at `start` and a release at `end` (the same place when
    omitted), as the view reports them."""
    where = iter([start, end or start])
    monkeypatch.setattr(window, '_cursor_position', lambda: next(where))
    window._on_plan_press(None, None)
    window._on_plan_release(None, None)


def _on_screen(window, surface):
    """A pixel inside one of `surface`'s faces, as the view draws it."""
    picker = window._planning['picker']
    screen, _depth = picker.projector.screen()
    owned = np.flatnonzero(picker.owner == surface)
    corners = picker.triangles[owned[len(owned) // 2]]
    return tuple(screen[corners].mean(axis=0))


def test_a_click_on_a_surface_leaves_it_out_and_another_puts_it_back(
        bar, window, pump, monkeypatch):
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    found = window._planning_skin()[1]
    # the face nearest the camera at the middle of a surface's drawing
    pixel = _on_screen(window, found[0].id)
    surface = window.plan_surface_at(*pixel)
    assert surface is not None
    _click(window, monkeypatch, pixel, (pixel[0] + 20, pixel[1]))
    assert window._planning['off'] == set(), 'a drag turns the view and toggles nothing'
    _click(window, monkeypatch, pixel)
    assert window._planning['off'] == {surface}
    assert surface not in window._planning['places'][1].surface
    assert panel.surfaces_label.text().startswith('Sensors on 5 of 6 surfaces.')
    colors = window._planning['mesh']['mesh'].cell_data['colors']
    owner = window._planning['mesh']['owner']
    assert len({tuple(c) for c in colors[owner == surface]}) == 1
    assert not np.array_equal(colors[owner == surface][0], colors[owner != surface][0]), (
        'drawn apart from the surfaces kept')
    _click(window, monkeypatch, pixel)
    assert window._planning['off'] == set()
    assert panel.surfaces_label.text().startswith('Sensors on 6 of 6 surfaces.')
    # a surface of an element group left unchecked is not to be clicked
    panel.groups.item(0).setCheckState(panel.groups.item(0).checkState().Unchecked)
    assert window.plan_surface_at(*pixel) is None


def test_a_click_reaches_the_pane_through_a_real_interactor(bar, window, pump):
    """Clicks and hovers as VTK delivers them, through a real interactor
    with the trackball camera's style: the style takes the mouse to
    itself from the press, so the interactor's observers never hear the
    release. Listening there, no click toggled a surface, and the press
    left behind put out the hover's light for good (Brandon, 2026-10-22:
    clicking surfaces did nothing). A click toggles, a drag does not, and
    after either, the surface under the cursor still lights."""
    import vtk

    select_objects(window, pump, bar)
    _open(window, pump)
    found = window._planning_skin()[1]
    pixel = _on_screen(window, found[0].id)
    surface = window.plan_surface_at(*pixel)
    # the view's own interactor, on the view's own render window, which
    # is the one the pane's observers are on. A second window built here
    # with an interactor of its own crashed CI workers in the press, on
    # one Python with the platform's interactor and on both with the
    # generic one (2026-10-23): a second GL context on a runner with no
    # display. The app never makes one.
    # Its style is the view's own trackball camera, whose observers the
    # pane put on when it opened; replacing it with a bare VTK style left
    # pyvista's observers driving one style and the pane listening to
    # another.
    raw = window.scene.plotter.iren.interactor
    assert isinstance(raw.GetInteractorStyle(), vtk.vtkInteractorStyleTrackballCamera)
    raw.Enable()

    def at(x, y, event):
        raw.SetEventInformation(round(x), round(y), 0, 0, '0', 0)
        getattr(raw, event)()

    at(*pixel, 'LeftButtonPressEvent')
    at(*pixel, 'LeftButtonReleaseEvent')
    assert window._planning['off'] == {surface}, 'the click toggled it'
    at(*pixel, 'MouseMoveEvent')
    assert window._planning['hovered'] == surface, 'and the light still follows the cursor'
    # the view's camera really turns now: put it back after each gesture,
    # so the pixel is still on the surface
    camera = window.scene.plotter.camera_position
    at(*pixel, 'LeftButtonPressEvent')
    at(pixel[0] + 30, pixel[1], 'MouseMoveEvent')
    at(pixel[0] + 30, pixel[1], 'LeftButtonReleaseEvent')
    assert window._planning['off'] == {surface}, 'a drag turns the view and toggles nothing'
    window.scene.plotter.camera_position = camera
    at(*pixel, 'MouseMoveEvent')
    assert window._planning['hovered'] == surface
    # a wheel turn ends an interaction too, with no press: nothing toggles
    raw.SetEventInformation(round(pixel[0]), round(pixel[1]), 0, 0, '0', 0)
    raw.MouseWheelForwardEvent()
    assert window._planning['off'] == {surface}
    window.scene.plotter.camera_position = camera
    at(*pixel, 'LeftButtonPressEvent')
    at(*pixel, 'LeftButtonReleaseEvent')
    assert window._planning['off'] == set(), 'a second click puts it back'


def test_none_and_all_and_keep_with_surfaces_left_out(bar, window, pump):
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    _to_the_edges(window, panel)
    panel.none_button.click()
    assert panel.surfaces_label.text().startswith('Sensors on 0 of 6 surfaces.')
    assert window._planning['plan'] is None and not panel.keep_button.isEnabled()
    panel.all_button.click()
    assert panel.surfaces_label.text().startswith('Sensors on 6 of 6 surfaces.')
    found = window._planning_skin()[1]
    top = next(s.id for s in found if np.allclose(s.normals[0], [0, 0, 1]))
    window.toggle_plan_surface(top)
    places = window._planning['places'][1]
    assert top not in places.surface and window._planning['plan'] is not None
    panel.keep_button.click()
    pump()
    kept = [s.id for s in found if s.id != top]
    assert window.project.journal[-1] == (
        f'project.plan_instrumentation({bar!r}, 4, 0, surfaces={kept}, edge_distance=0.0)')
    placed = window.objects[f'{bar} Plan Geometry']
    assert not np.isclose(placed.node_xyz[:, 2], placed.node_xyz[:, 2].max()).all()


def test_a_wider_edge_angle_joins_surfaces_and_starts_the_picks_again(bar, window, pump):
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    window.toggle_plan_surface(1)
    panel.edge_angle.setValue(120)
    window._plan_now()
    assert panel.surfaces_label.text().startswith('Sensors on 1 of 1 surfaces.'), (
        "the block's edges are 90 degrees: one surface")
    assert window._planning['off'] == set(), 'a number means another surface now'
    panel.keep_button.click()
    pump()
    assert window.project.journal[-1] == (
        f'project.plan_instrumentation({bar!r}, 4, 0, edge_angle=120.0)')


def test_keep_adds_the_plan_as_the_project_call_does(bar, window, pump):
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    panel.triaxial.setValue(3)
    panel.uniaxial.setValue(2)
    window._plan_now()
    previewed = window._planning['plan']
    panel.keep_button.click()
    pump()
    assert not panel.isVisible() and window.__dict__.get('_planning') is None
    names = [f'{bar} Plan {part}' for part in ('Geometry', 'Channels', 'Shapes')]
    for name in names:
        assert window._item_for_object(name) is not None, name
    assert window.project.journal[-1] == f"project.plan_instrumentation({bar!r}, 3, 2)"
    kept = window.objects[names[2]]
    assert kept.coordinate == previewed.shapes.coordinate
    assert np.allclose(kept.shape_matrix, previewed.shapes.shape_matrix)
    assert window.tree.currentItem() is window._item_for_object(names[0])


def test_close_adds_nothing_and_selecting_elsewhere_puts_it_away(bar, window, pump):
    before = set(window.objects)
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    panel.close_button.click()
    pump()
    assert not panel.isVisible() and set(window.objects) == before
    assert not window._plan_observers, 'the clicks are the camera\'s again'
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    select_objects(window, pump, 'Bar')
    assert not panel.isVisible() and window.__dict__.get('_planning') is None


def test_the_plan_is_on_the_bar_that_is_up(bar, window, pump):
    """A shape set selected alone shows its mode table and auto-MAC,
    with neither the plot nor the 3-D view up: the act is on the
    table's bar, where it can be pressed."""
    select_objects(window, pump, bar)
    bars = (window.data_pane.toolbar, window.scene.toolbar, window.table_bar)
    up = [b for b in bars if b.isVisibleTo(window)]
    offered = [a for b in up for a in b.actions()
               if a.isVisible() and a.text() == 'Plan Instrumentation']
    assert len(offered) == 1
    offered[0].trigger()
    pump()
    assert window.scene.instrumentation_panel.isVisible()


def test_fewer_directions_than_modes_warn_and_still_keep(bar, window, pump):
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    panel.triaxial.setValue(0)
    panel.uniaxial.setValue(3)
    window._plan_now()
    reading = panel.reading_label.text()
    assert reading.startswith('0 tri-axial and 3 uni-axial sensors')
    assert reading.endswith('Warning: 3 directions for 8 modes, planned for the lowest 3; '
                            'modes 10–14 cannot be told apart by these sensors.')
    assert panel.keep_button.isEnabled()
    assert window._planning['plan'].shapes.num_shapes == 8
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter('always')
        panel.keep_button.click()
        pump()
    assert not [w for w in said if issubclass(w.category, FewerDirectionsWarning)], (
        'the reading said it; Keep does not say it again')
    assert window.objects[f'{bar} Plan Shapes'].num_shapes == 8


def test_the_distance_from_the_edges_is_shown_and_kept_in_the_display_unit(bar, window, pump):
    """A quarter of an inch by default, in the display unit system and
    following it; a distance changed is kept, in meters, the
    geometry's own unit."""
    window._units_changed('in-slinch-lbf-s')
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    assert np.isclose(panel.edge_distance.value(), 0.25)
    assert panel.edge_distance.suffix() == ' in'
    near = len(window._planning['places'][1].source)
    window._units_changed('mm-kg-N-s')
    assert np.isclose(panel.edge_distance.value(), 6.35) and panel.edge_distance.suffix() == ' mm'
    window._plan_now()
    assert len(window._planning['places'][1].source) == near, 'the same distance'
    panel.edge_distance.setValue(15.24)          # 0.6 in: only the 8 x 2 faces' middles
    window._plan_now()
    assert len(window._planning['places'][1].source) == 26 < near
    panel.keep_button.click()
    pump()
    line = window.project.journal[-1]
    assert line.startswith(f'project.plan_instrumentation({bar!r}, 4, 0, edge_distance=0.01524')
    placed = window.objects[f'{bar} Plan Geometry'].node_xyz / 0.0254
    assert (np.abs(placed[:, 0]) <= 3.4 + 1e-9).all() and (np.abs(placed[:, 1]) <= 0.4 + 1e-9).all()


def test_the_default_distance_is_not_journaled(bar, window, pump):
    window._units_changed('mm-kg-N-s')
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    panel.triaxial.setValue(2)
    panel.uniaxial.setValue(3)
    window._plan_now()
    panel.keep_button.click()
    pump()
    assert window.project.journal[-1] == f"project.plan_instrumentation({bar!r}, 2, 3)"


@pytest.fixture
def bar_and_cube(window, pump):
    """The bar with a half-inch cube on it, meshed together on one grid
    so they share the seam: the cube's five open faces are each under
    1 % of the outside's area."""
    project = window.project
    name = project.new_geometry('Bar', unit='in')
    bar = project.add_solid_block(name, (0, 0, 0), (8, 2, 1), name='bar', unit='in')
    cube = project.add_solid_block(name, (0.25, 0.25, 0.75), (0.5, 0.5, 0.5), name='cube',
                                   unit='in')
    project.mesh_solids(name, [bar, cube], 0.5, unit='in')
    geometry = project[name]
    for group in geometry.group_id:
        geometry.group_properties[int(group)] = GroupProperties(MATERIALS['1018 steel'])
    modes = project.solve_modes(name, num_modes=20)
    window.show_object(name)
    window.show_object(modes)
    pump()
    return modes


def test_small_surfaces_start_out_left_out_and_keep_says_so_only_when_changed(
        bar_and_cube, window, pump):
    bar = bar_and_cube
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    assert panel.surfaces_label.text().startswith('Sensors on 6 of 11 surfaces.')
    found = window._planning_skin()[1]
    geometry = window.objects['Bar']
    cube = dict(zip(geometry.group_name, (int(g) for g in geometry.group_id),
                    strict=True))['cube']
    small = {s.id for s in found if s.group == cube}
    assert window._planning['off'] == small
    colors = window._planning['mesh']['mesh'].cell_data['colors']
    owner = window._planning['mesh']['owner']
    assert not np.array_equal(colors[np.isin(owner, list(small))][0],
                              colors[~np.isin(owner, list(small))][0]), 'drawn as left out'
    panel.keep_button.click()
    pump()
    assert 'surfaces=' not in window.project.journal[-1], 'the default is not journaled'
    select_objects(window, pump, bar)
    panel = _open(window, pump)
    back = min(small)
    window.toggle_plan_surface(back)
    assert panel.surfaces_label.text().startswith('Sensors on 7 of 11 surfaces.')
    panel.name.setText('Again')
    panel.keep_button.click()
    pump()
    kept = sorted({s.id for s in found} - small | {back})
    assert f'surfaces={kept}' in window.project.journal[-1]
