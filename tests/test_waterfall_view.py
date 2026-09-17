"""The 2D/3D toggle: the default, what it swaps, what the camera keeps.

The scene's own claims — one mesh, one color scale, log mapping — are
`test_waterfall.py`'s. This file holds the window's half: the 3-D
reading is the default wherever plain curves draw (a single record
included), unchecking asks for the flat plot, the surfaces swap without
losing each other's state, and the camera obeys the standing rule
(placed when the displayed object changes, never on a reread).
"""

import numpy as np

from visualdynamics.core.data import MultipleCoherence, Psd, TimeHistory

PSD_DIM = 'acceleration**2/frequency'
PSD_UNIT = '(m/s**2)**2/Hz'


def psd(rows=6, dims=None, units=None):
    rng = np.random.default_rng(5)
    f = np.linspace(0.0, 2000.0, 257)
    values = 1e-4 * (1 + rng.random((rows, 257)))
    return Psd(f, values, response_dof=[f'{100 + i}Z+' for i in range(rows)],
               ordinate_dim=dims or [PSD_DIM] * rows,
               ordinate_unit=units or [PSD_UNIT] * rows)


def history(rows=3):
    t = np.linspace(0.0, 1.0, 4096)
    return TimeHistory(t, np.tile(np.sin(60.0 * t), (rows, 1)),
                       response_dof=[f'{i}Z+' for i in range(1, rows + 1)],
                       ordinate_dim=['acceleration'] * rows,
                       ordinate_unit=['m/s**2'] * rows)


def select(window, pump, name):
    window.tree.clearSelection()
    window.tree.setCurrentItem(window._item_for_object(name))
    pump()


def camera_of(plotter):
    return np.asarray([list(point) for point in plotter.camera_position])


def in_3d(pane):
    return (pane.waterfall_plotter is not None
            and not pane.graphics.isVisibleTo(pane)
            and 'waterfall' in pane.waterfall_plotter.renderer.actors)


def test_the_waterfall_is_the_default_reading(window, pump):
    """Selecting data lands in 3-D with the toggle untouched."""
    window.add_object('Many', psd())
    select(window, pump, 'Many')
    pane = window.data_pane
    assert pane.waterfall_action.isVisible()
    assert pane.waterfall_action.isChecked(), 'checked from the start'
    assert in_3d(pane)
    assert '6 channels as a waterfall' in window.statusBar().currentMessage()


def test_a_single_record_is_a_waterfall_too(window, pump):
    """One line on the stage, color by level — offered and default,
    like any other record count."""
    window.add_object('Solo', psd(rows=1))
    select(window, pump, 'Solo')
    pane = window.data_pane
    assert pane.waterfall_action.isVisible()
    assert in_3d(pane)
    assert '1 channel as a waterfall' in window.statusBar().currentMessage()


def test_unchecking_asks_for_the_flat_plot(window, pump):
    window.add_object('Many', psd())
    select(window, pump, 'Many')
    pane = window.data_pane
    assert in_3d(pane)
    pane.waterfall_action.trigger()   # uncheck: the flat plot
    pump()
    assert pane.graphics.isVisibleTo(pane)
    assert not pane.waterfall_action.isChecked()
    pane.waterfall_action.trigger()   # and back
    pump()
    assert in_3d(pane)


def test_the_camera_survives_a_reread_and_resets_on_a_new_object(window,
                                                                 pump):
    window.add_object('First', psd())
    window.add_object('Second', psd())
    select(window, pump, 'First')
    pane = window.data_pane
    assert in_3d(pane)
    plotter = pane.waterfall_plotter
    home = camera_of(plotter)
    # the user orbits somewhere, then redraws the same object
    plotter.camera_position = [(5.0, 5.0, 5.0), (0.0, 0.0, 0.0),
                               (0.0, 0.0, 1.0)]
    window.render_current()
    pump()
    assert np.allclose(camera_of(plotter)[0], (5.0, 5.0, 5.0)), \
        'a reread must redraw under the view the user set'
    # a different object is a different scene: the camera goes home
    select(window, pump, 'Second')
    assert np.allclose(camera_of(plotter), home)


def test_the_choice_stays_armed_across_selections(window, pump):
    """Unchecked on one object stays unchecked on the next — the
    reading is a view choice, not a per-object setting."""
    window.add_object('Many', psd())
    window.add_object('Other', psd(rows=2))
    select(window, pump, 'Many')
    pane = window.data_pane
    pane.waterfall_action.trigger()   # flat
    pump()
    select(window, pump, 'Other')
    assert pane.graphics.isVisibleTo(pane)
    assert not pane.waterfall_action.isChecked()


def test_a_whole_coherence_lands_in_the_waterfall(window, pump):
    """The map was the whole-object default because flat lines overlay
    into a band; the waterfall shows every channel at once too, and it
    is the resting default now (Brandon's call). An explicit click on
    Map still outranks it — every explicit choice does — and clicking
    Curves hands back to the checked 3-D reading."""
    f = np.linspace(0.0, 100.0, 65)
    coherence = MultipleCoherence(
        f, np.tile(np.linspace(0.2, 0.9, 65), (3, 1)),
        response_dof=['1Z+', '2Z+', '3Z+'])
    window.add_object('Coherence', coherence)
    select(window, pump, 'Coherence')
    pane = window.data_pane
    assert in_3d(pane), 'the default reading, for a coherence like anything'
    pane.map_action.trigger()
    pump()
    assert pane.graphics.isVisibleTo(pane), 'Map means the map'
    assert not pane.waterfall_action.isVisible()
    pane.curves_action.trigger()
    pump()
    assert in_3d(pane)


def test_the_averaging_view_marks_the_stage(window, pump):
    """The averaging reading works on the stage now: opening the view
    in 3-D keeps 3-D, the frames draw as stage geometry, and the side
    panel — the editor in both views — comes up beside it (Brandon,
    2026-08-23; the view used to force the flat plot)."""
    window.add_object('History', history())
    select(window, pump, 'History')
    pane = window.data_pane
    assert in_3d(pane)
    pane.averaging_action.trigger()
    pump()
    assert in_3d(pane), 'the stage holds; the marks join it'
    assert pane.showing_averaging
    assert pane.averaging_panel.isVisible(), 'the panel is the editor'
    names = list(pane.waterfall_plotter.actors)
    assert any('marks-averaging-windows' in name for name in names)
    assert any('marks-averaging-edge-start' in name for name in names)
    # standing down to 2-D swaps the same reading onto the flat plot
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    assert not pane._waterfall_page.isVisible()
    assert pane.showing_averaging and pane.averaging_panel.isVisible()
    # and back again, marks intact
    pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    assert in_3d(pane)
    assert any('marks-averaging-windows' in name
               for name in pane.waterfall_plotter.actors)


def mixed_psd():
    return psd(rows=5,
               dims=[PSD_DIM] * 3 + ['voltage**2/frequency'] * 2,
               units=[PSD_UNIT] * 3 + ['V**2/Hz'] * 2)


def test_the_quantity_box_reaches_every_floor(window, pump):
    """The flat plot stacks an axis per quantity; the stage holds one,
    and the box is how the others are reached. Largest group first and
    drawn by default; choosing another redraws with its records."""
    window.add_object('Mixed', mixed_psd())
    select(window, pump, 'Mixed')
    pane = window.data_pane
    assert pane.quantity_action.isVisible()
    labels = [pane.quantity_box.itemText(k)
              for k in range(pane.quantity_box.count())]
    assert labels == ['acceleration', 'voltage'], 'largest floor first'
    message = window.statusBar().currentMessage()
    assert '3 channels as a waterfall' in message
    assert '2 of other quantities' in message
    pane.quantity_box.setCurrentIndex(1)
    pump()
    assert '2 channels as a waterfall' in window.statusBar().currentMessage()


def test_the_quantity_choice_is_sticky_by_key(window, pump):
    window.add_object('Mixed', mixed_psd())
    select(window, pump, 'Mixed')
    pane = window.data_pane
    pane.quantity_box.setCurrentIndex(1)   # voltage
    pump()
    window.render_current()
    pump()
    assert pane.quantity_box.currentText() == 'voltage', \
        'a reread must not walk the choice back to the default'


def test_one_quantity_offers_no_box(window, pump):
    window.add_object('Plain', psd())
    select(window, pump, 'Plain')
    assert not window.data_pane.quantity_action.isVisible(), \
        'one quantity is not a choice'


def test_other_drawings_never_inherit_the_3d_surface(window, pump):
    """Anything that is not the plain-curve path — here a geometry,
    whose render never reaches `_render_series` at all — must leave the
    2-D surface standing, however the last selection left the pane.
    That is `render_current`'s own reset, not the curve path's."""
    import visualdynamics

    window.add_object('Many', psd())
    window.add_object('Plate', visualdynamics.Geometry(
        node_id=[1, 2, 3], node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        length_unit='m'))
    select(window, pump, 'Many')
    pane = window.data_pane
    assert in_3d(pane)
    select(window, pump, 'Plate')
    assert pane.graphics.isVisibleTo(pane)


# ---- the page stepper ---------------------------------------------------


def big_psd(rows=40, samples=4001):
    rng = np.random.default_rng(6)
    f = np.linspace(0.0, 2000.0, samples)
    return Psd(f, 1e-4 * (1 + rng.random((rows, samples))),
               response_dof=[f'{100 + i}Z+' for i in range(rows)],
               ordinate_dim=[PSD_DIM] * rows, ordinate_unit=[PSD_UNIT] * rows)


def test_one_page_offers_no_stepper(window, pump):
    window.add_object('Small', psd())
    select(window, pump, 'Small')
    assert not window.data_pane.page_action.isVisible(), \
        'one page is not a choice'


def test_many_records_page_and_the_arrows_walk_them(window, pump,
                                                    monkeypatch):
    """Brandon's `<- 1/3 ->`: what a scene can draw exactly it draws,
    and the rest is a page away rather than thinned."""
    from visualdynamics.viz import waterfall

    # a budget small enough that the fixture pages, without needing a
    # 900-record CPSD in a unit test
    monkeypatch.setattr(waterfall, 'SCENE_BUDGET', 20 * 2 * 4001)
    window.add_object('Many', big_psd())
    select(window, pump, 'Many')
    pane = window.data_pane
    assert pane.page_action.isVisible(), 'the stepper is offered'
    assert pane.page_box.value() == 1
    assert pane.page_box.suffix() == ' / 2'
    assert '20 channels as a waterfall' in window.statusBar().currentMessage()
    assert 'page 1 of 2' in window.statusBar().currentMessage()
    assert '40 in all' in window.statusBar().currentMessage()

    first = list(window._waterfall_drawn)
    pane.page_box.stepUp()
    pump()
    assert pane.page_box.value() == 2
    assert 'page 2 of 2' in window.statusBar().currentMessage()
    assert not set(window._waterfall_drawn) & set(first), \
        'the next page shows different channels'

    pane.page_box.stepDown()
    pump()
    assert pane.page_box.value() == 1
    assert list(window._waterfall_drawn) == first


def test_the_page_resets_when_the_object_changes(window, pump, monkeypatch):
    """Page 2 of something else is not a place."""
    from visualdynamics.viz import waterfall

    monkeypatch.setattr(waterfall, 'SCENE_BUDGET', 20 * 2 * 4001)
    window.add_object('A', big_psd())
    window.add_object('B', big_psd())
    select(window, pump, 'A')
    window.data_pane.page_box.stepUp()
    pump()
    assert window.data_pane.page_box.value() == 2
    select(window, pump, 'B')
    assert window.data_pane.page_box.value() == 1, 'a new object starts at 1'


# ---- the flat plot pages by the same control ----------------------------


def test_the_flat_plot_pages_too(window, pump):
    """One pager, both readings. The 2-D limit is MAX_RECORDS rather
    than a vertex budget — what stops a flat plot is legibility, not
    memory — but the control and its vocabulary are the same."""
    from visualdynamics.plot import MAX_RECORDS

    rows = MAX_RECORDS * 2 + 10
    window.add_object('Many', psd(rows=rows))
    select(window, pump, 'Many')
    pane = window.data_pane
    pane.waterfall_action.trigger()          # the flat reading
    pump()
    assert pane.graphics.isVisibleTo(pane)
    assert pane.page_action.isVisible(), 'the flat plot pages as well'
    assert pane.page_box.suffix() == ' / 3'
    message = window.statusBar().currentMessage()
    assert 'page 1 of 3' in message and f'{rows} in all' in message
    assert 'first' not in message, 'a page, not a truncation'

    pane.page_box.stepUp()
    pump()
    assert 'page 2 of 3' in window.statusBar().currentMessage()


def test_a_small_object_pages_in_neither_reading(window, pump):
    window.add_object('Small', psd(rows=6))
    select(window, pump, 'Small')
    pane = window.data_pane
    assert not pane.page_action.isVisible()
    pane.waterfall_action.trigger()
    pump()
    assert not pane.page_action.isVisible()
