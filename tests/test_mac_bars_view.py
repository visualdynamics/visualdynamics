"""The MAC's 3D reading: the default, the toggle, and picking parity.

The auto-MAC beside a mode table, the fit's MAC, and the cross-MAC all
draw through the same pair of surfaces, and the bars are the default —
made so once they could be picked like the grid. Picking runs through
`_pick_mac_pair`, one implementation for both readings, so the bars
cannot select differently from the grid; the marks (outline, boldest
outline, stripes) are drawn by the scene and asserted by actor.
"""

import numpy as np

from visualdynamics.core.shapes import ShapeSet


def shapes(n=3, offset=0.0):
    rng = np.random.default_rng(4)
    return ShapeSet(np.linspace(10.0, 90.0, n) + offset,
                    np.full(n, 0.01), ['1Z+', '2Z+', '3Z+', '4Z+'],
                    rng.standard_normal((n, 4)))


def select(window, pump, *names):
    window.tree.clearSelection()
    for name in names:
        window._item_for_object(name).setSelected(True)
    if len(names) == 1:
        # setCurrentItem clears a multi-selection, so it is only for
        # the single case — the animation tests select the same way
        window.tree.setCurrentItem(window._item_for_object(names[0]))
    pump()


def bars_up(window):
    page = window._mac_bars_page
    return (page is not None and page.isVisibleTo(window)
            and 'mac-bars' in window._mac_bars_plotter_obj.renderer.actors
            and not window.mac_frame.isVisibleTo(window))


def comparing(window, pump):
    window.add_object('Test Modes', shapes())
    window.add_object('FEM Modes', shapes(n=4, offset=1.0))
    select(window, pump, 'Test Modes', 'FEM Modes')
    return window._mac_bars_plotter_obj.renderer.actors


def test_the_bars_are_the_default_reading(window, pump):
    window.add_object('Modes', shapes())
    select(window, pump, 'Modes')
    assert window.mac_bars_action.isVisible()
    assert window.mac_bars_action.isChecked(), 'checked from the start'
    assert bars_up(window)
    window.mac_bars_action.trigger()   # uncheck: the flat grid
    pump()
    assert window.mac_frame.isVisibleTo(window)
    assert not window._mac_bars_page.isVisibleTo(window)


def test_the_cross_mac_defaults_to_bars_with_the_active_mark(window, pump):
    actors = comparing(window, pump)
    assert bars_up(window)
    assert 'mac-bars-active' in actors, \
        'the animated pair wears the boldest outline from the start'


def test_picking_runs_the_selection_idiom(window, pump):
    """One implementation for both readings: a plain pick replaces, a
    modifier extends, a modifier on a picked pair removes it."""
    comparing(window, pump)
    window._pick_mac_pair(0, 1, extend=False)
    pump()
    assert window._compare['pairs'] == [(0, 1)]
    assert window._compare['cell'] == (0, 1)
    window._pick_mac_pair(1, 2, extend=True)
    pump()
    assert window._compare['pairs'] == [(0, 1), (1, 2)]
    actors = window._mac_bars_plotter_obj.renderer.actors
    selected = actors['mac-bars-selected'].mapper.dataset
    assert selected.n_lines == 2 * 12, 'both picked bars wear outlines'
    window._pick_mac_pair(0, 1, extend=True)   # and taken back out
    pump()
    assert window._compare['pairs'] == [(1, 2)]


def test_out_of_range_picks_are_refused(window, pump):
    comparing(window, pump)
    before = dict(window._compare)
    window._pick_mac_pair(99, 0, extend=False)
    assert window._compare == before, 'a miss changes nothing'


def test_committed_matches_wear_stripes_in_both_readings(window, pump):
    comparing(window, pump)
    window._pick_mac_pair(0, 0, extend=False)
    pump()
    window.add_matches()
    pump()
    actors = window._mac_bars_plotter_obj.renderer.actors
    assert 'mac-bars-matched' in actors, \
        'the committed pair wears the stripes the flat grid gives it'


def test_the_fit_mac_defaults_to_bars(window, pump):
    from conftest import fixture_path

    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    pump()
    window.tree.clearSelection()
    item = window._item_for_object('FRF')
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    window.start_modal_fit()
    pump()
    assert window.fit is not None
    assert not window.mac_bars_action.isVisible(), \
        'a fresh session has no modes and rightly no MAC'
    window.confirm_mode_button.click()
    pump()
    assert window.mac_bars_action.isVisible()
    assert not window.mode_table_action.isVisible(), \
        'comparison controls mean nothing over one set against itself'
    assert bars_up(window)
    window.stop_fitting()


def test_the_camera_holds_while_a_comparison_redraws(window, pump):
    window.add_object('Modes', shapes())
    select(window, pump, 'Modes')
    plotter = window._mac_bars_plotter_obj
    plotter.camera_position = [(9.0, 9.0, 9.0), (0.0, 0.0, 0.0),
                               (0.0, 0.0, 1.0)]
    window.render_current()
    pump()
    assert np.allclose(np.asarray(plotter.camera_position[0]),
                       (9.0, 9.0, 9.0)), 'same comparison, same view'
    window.add_object('Other', shapes(offset=2.0))
    select(window, pump, 'Other')
    assert not np.allclose(np.asarray(plotter.camera_position[0]),
                           (9.0, 9.0, 9.0))


def test_the_flat_grid_returns_wherever_the_mac_goes_away(window, pump):
    """Anything that hides the MAC hides both surfaces — the bars must
    not outlive the grid on a selection with no MAC at all."""
    window.add_object('Modes', shapes())
    select(window, pump, 'Modes')
    assert bars_up(window)
    window.add_object('Line', __import__('visualdynamics').Geometry(
        node_id=[1], node_xyz=[[0, 0, 0]], length_unit='m'))
    select(window, pump, 'Line')
    assert not window._mac_bars_page.isVisibleTo(window)


def test_the_flat_mac_comes_back_with_real_width(window, pump):
    """A splitter child keeps the width it had when hidden, and the
    two MAC surfaces trade places while hidden — so the flat grid
    could return at the zero it was stored at: `isVisible()`, 0 px,
    and the 2-D toggle appearing to show only the table (Brandon,
    2026-08-30)."""
    window.add_object('Shapes', shapes())
    select(window, pump, 'Shapes')
    window.render_current()
    pump()
    # what a session's worth of resizes can leave behind: the flat
    # frame's stored share squeezed to nothing while the bars were up
    window.tables_row.setSizes([1000, 0, 100])
    pump()
    window.mac_bars_action.trigger()
    pump()
    assert window.mac_frame.isVisible()
    assert window.mac_frame.width() > 66, \
        'wide enough to read — the left axis alone wants 66 px'
    assert window.table.isVisible(), 'the table stays beside it'


def test_the_3d_toggle_leads_the_table_bar(window):
    """The same control sits in the same place whichever view is up:
    the data pane's toolbar leads with its 2D/3D toggle, and this bar
    leads with its own (Brandon, 2026-08-30)."""
    actions = window.table_bar.actions()
    assert actions[0] is window.mac_bars_action
    assert actions[1] is window.mode_table_action


def test_confirming_a_mode_reframes_the_fit_mac(window, pump):
    """Keyed per session alone, the camera was placed for the opening
    1x1 matrix and never again — a grown MAC came up out of frame with
    the view centered on its first corner (Brandon, 2026-08-30). A
    growth reframes; a same-size redraw still keeps the user's view."""
    from conftest import fixture_path

    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    pump()
    window.tree.clearSelection()
    item = window._item_for_object('FRF')
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    window.start_modal_fit()
    pump()
    window.confirm_mode_button.click()
    pump()
    plotter = window._mac_bars_plotter_obj
    first = np.asarray(plotter.camera_position[0])

    # the user's adjustment survives a same-size redraw
    plotter.camera_position = [(9.0, 9.0, 9.0), (0.0, 0.0, 0.0),
                               (0.0, 0.0, 1.0)]
    window._render_fit_mac()
    pump()
    assert np.allclose(np.asarray(plotter.camera_position[0]),
                       (9.0, 9.0, 9.0)), 'same size, same view'

    window.find_mode_button.click()
    pump()
    window.confirm_mode_button.click()
    pump()
    grown = np.asarray(plotter.camera_position[0])
    assert not np.allclose(grown, (9.0, 9.0, 9.0)), \
        'the grid grew, so the camera reframed for the new extent'
    assert not np.allclose(grown, first), 'and not to the 1x1 framing'
    window.stop_fitting()
