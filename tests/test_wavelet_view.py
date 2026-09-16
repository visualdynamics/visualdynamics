"""The scalogram as a view: a button, a panel, and a picture drawn.

The maths has its own tests (`test_wavelet`). This is the half those
cannot see — that the button reaches it, that the panel says what the
picture is of, and that a reader can get back out again.

That last one is not incidental. Kurtosis used to hide the other
reading buttons while it was up, on the argument that a record showing
bars has no trace to mark; when the readings became a visible group,
that left the group collapsed to whichever button had been pressed and
no way to another except unpicking it first (Brandon, 2026-08-27). The
readings stay offered; the *2-D/3-D toggle* is what goes away, because
neither the bars nor the scalogram has a depth axis to use.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

from visualdynamics.core.data import TimeHistory

MODAL = ('plate', 'modal.nc4')


def looking_at(window, pump, name='Time History'):
    window.import_paths([fixture_path(*MODAL)])
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    return window.objects[name]


@pytest.fixture
def showing(window, pump):
    history = looking_at(window, pump)
    window.data_pane.wavelet_action.trigger()
    pump()
    return window, history


def test_a_time_history_offers_the_wavelet(window, pump):
    looking_at(window, pump)
    assert window.data_pane.wavelet_action.isVisible()


def test_it_opens_in_three_dimensions(showing):
    """A scalogram is already a function of two variables, so a surface
    is its natural form and the flat picture is the fallback — the
    opposite of the toggle's default elsewhere (Brandon, 2026-08-27)."""
    window, _history = showing
    assert window.data_pane.showing_waterfall, 'the 3-D reading is default'
    assert window.data_pane.waterfall_plotter is not None, 'a surface drew'


def test_the_flat_picture_is_a_toggle_away(window, pump):
    window_ = window
    looking_at(window, pump)
    window_.data_pane.wavelet_action.trigger()
    pump()
    window_.data_pane.waterfall_action.setChecked(False)
    window_.render_current()
    pump()
    assert not window_.data_pane.showing_waterfall
    assert window_.data_pane.graphics.ci.items, 'the flat scalogram drew'


def test_both_forms_are_offered_for_this_reading(showing):
    """Brandon's general rule: the toggle is there whenever both a 2-D
    and a 3-D form exist for the current view mode."""
    window, _history = showing
    assert window.data_pane.waterfall_action.isVisible()


def test_the_panel_comes_up_with_it(showing):
    window, _history = showing
    assert window.data_pane.wavelet_panel.isVisible()
    assert window.data_pane.wavelet_panel.derived['transforms'].text() != \
        '—', 'the derived rows are computed, not placeholders'


def test_the_status_line_says_what_was_computed(showing):
    window, _history = showing
    said = window._status_text
    assert 'Hz' in said and 'cone' in said, said


def test_the_range_it_opens_on_is_one_the_record_can_carry(showing):
    """A default whose bottom is all cone would be a default that draws
    an artefact."""
    window, history = showing
    settings = window.data_pane.wavelet_settings
    assert settings['high'] < history.sample_rate / 2.0
    duration = len(history.abscissa) / history.sample_rate
    from visualdynamics.core.wavelet import cone_of_influence

    cone = cone_of_influence([settings['low']], history.sample_rate,
                             settings['omega0'])[0]
    assert 2 * cone < duration, 'the record outlasts its own cone'


def test_a_tuned_range_survives_looking_away_and_back(window, pump):
    """Sticky like every other view choice on this bar."""
    looking_at(window, pump)
    window.data_pane.wavelet_action.trigger()
    pump()
    window.data_pane.wavelet_panel.low_box.setValue(45.0)
    pump()
    kept = dict(window.data_pane.wavelet_settings)
    assert kept['low'] == pytest.approx(45.0)

    looking_at(window, pump)
    pump()
    assert window.data_pane.wavelet_settings['low'] == pytest.approx(45.0)


# ---- getting back out again ---------------------------------------------


@pytest.mark.parametrize('other', ['averaging_action', 'filter_action',
                                   'kurtosis_action', 'shocks_action'])
def test_the_other_readings_stay_reachable(showing, other):
    """The bug this view was written alongside: a reading that hides its
    alternatives is a reading with no way out."""
    window, _history = showing
    action = getattr(window.data_pane, other)
    assert action.isVisible(), f'{other} went away'


def test_choosing_another_reading_puts_the_scalogram_away(showing, pump):
    window, _history = showing
    window.data_pane.averaging_action.trigger()
    pump()
    assert not window.data_pane.wavelet_wanted
    assert not window.data_pane.wavelet_panel.isVisible()


def test_the_3d_toggle_goes_away_only_where_there_is_no_3d_form(window, pump):
    """The setting half of principle 3, and the only half of the old
    hiding rule that survives: the kurtosis bars are flat, so a depth
    axis would have nothing to put along it. The scalogram has both
    forms, so it keeps the toggle."""
    looking_at(window, pump)
    window.data_pane.offer_waterfall(True)
    assert window.data_pane.waterfall_action.isVisible()

    window.data_pane.kurtosis_action.trigger()
    pump()
    window.data_pane.offer_waterfall(True)
    assert not window.data_pane.waterfall_action.isVisible()


def test_the_toggle_comes_back_when_the_flat_reading_goes(window, pump):
    looking_at(window, pump)
    window.data_pane.kurtosis_action.trigger()
    pump()
    window.data_pane.kurtosis_action.trigger()      # off again
    pump()
    window.data_pane.offer_waterfall(True)
    assert window.data_pane.waterfall_action.isVisible()


# ---- what it refuses to draw --------------------------------------------


def test_a_range_above_this_records_nyquist_is_brought_back_into_it(
        window, pump):
    """A range tuned on a fast run and carried to a slower one asks for
    frequencies that are not there.

    Clamped into the record rather than refused: the range is sticky on
    purpose, and a person who moves between runs should not have to
    retype it — but it must never be *transformed* as asked, which
    would be inventing frequencies the record cannot carry. The first
    version clamped each end on its own and settled both onto Nyquist,
    leaving a range of no width that the panel's derived rows then
    raised on.
    """
    history = looking_at(window, pump)
    window.data_pane.wavelet_action.trigger()
    pump()
    rate = history.sample_rate
    window.data_pane.wavelet_settings = {
        'low': rate, 'high': rate * 2, 'per_octave': 12, 'omega0': 6.0}

    said = window._render_wavelet(history)
    settled = window.data_pane.wavelet_settings
    assert settled['high'] < rate / 2.0, 'inside Nyquist'
    assert settled['low'] < settled['high'], 'and still a range'
    assert 'Hz' in said, said


def test_the_picture_is_of_the_selected_record(showing, pump):
    """The tree is the one selection (Brandon, 2026-08-29): picking a
    record in the grid is what changes what the scalogram is of. The
    Channel combo this replaced was the same choice in two places."""
    window, history = showing
    if len(history.response_dof) < 2:
        pytest.skip('this fixture has one channel')
    window._select_records('Time History', [1])
    window.render_current()
    pump()
    assert history.record_label(1) in window._status_text
    assert not hasattr(window.data_pane.wavelet_panel, 'channel_box'), \
        'the panel no longer offers the choice the tree owns'


def test_a_whole_object_selection_settles_on_its_first_record(showing):
    """Selecting the object whole shows *a* record, and the tree is
    settled onto it — expanded, first record selected — so the tree
    always says what the picture is of."""
    window, history = showing
    if len(history.response_dof) < 2:
        pytest.skip('this fixture has one channel')
    grid = window.record_grids['Time History']
    assert grid.selected_records() == [0]
    assert window._item_for_object('Time History').isExpanded(), \
        'expanded so the settled pick is visible'
    assert history.record_label(0) in window._status_text


def test_a_multi_record_selection_collapses_to_the_first(showing, pump):
    """The scalogram shows one record; leaving three selected while
    drawing one would be the tree saying one thing and the picture
    another — the disagreement this rework exists to remove."""
    window, history = showing
    if len(history.response_dof) < 3:
        pytest.skip('this fixture has too few channels')
    window._select_records('Time History', [1, 2])
    window.render_current()
    pump()
    assert window.record_grids['Time History'].selected_records() == [1]
    assert history.record_label(1) in window._status_text


def test_the_magnitudes_drawn_are_the_records_own_amplitudes(showing):
    """The normalisation reaching the picture: the colour bar carries
    the record's units, so its top has to be an amplitude the record
    actually contains rather than an arbitrary scale."""
    window, history = showing
    settings = window.data_pane.wavelet_settings
    from visualdynamics.core.wavelet import log_frequencies, scalogram

    frequencies = log_frequencies(settings['low'], settings['high'],
                                  settings['per_octave'])
    frequencies = frequencies[frequencies < history.sample_rate / 2]
    magnitude = np.abs(scalogram(
        np.real(np.asarray(history.ordinate)[0]),
        history.sample_rate, frequencies, settings['omega0']))
    peak = np.abs(np.real(np.asarray(history.ordinate)[0])).max()
    assert magnitude.max() <= peak * 1.5, 'no invented amplitude'


# ---- the transient report carries the reading ---------------------------


def test_the_transient_report_carries_the_scalogram():
    """Brandon's ask (2026-08-28): the 3-D wavelet plot in the
    transient report. The template names the figure and the prose
    references it, so `test_report`'s reference checks hold them to
    each other like every other figure."""
    from visualdynamics.core.report import transient_template

    report = transient_template({})
    blocks = report.blocks
    scalograms = [b for b in blocks if b.get('mode') == 'scalogram']
    assert len(scalograms) == 1
    assert scalograms[0]['select'] == 'dim:acceleration', \
        'the control channels, not the drives'
    prose = ' '.join(b.get('text', '') for b in blocks)
    assert '{{figure:Control scalogram}}' in prose, \
        'the text points at the figure it discusses'


def test_the_scalogram_figure_builds_from_a_record(qt_app):
    """End to end through the report builder: a burst at a known time
    and frequency comes out as stage geometry whose loudest run sits
    at the burst's own frequency."""
    from visualdynamics.report import _build_block
    from visualdynamics.units import DEFAULT_SYSTEM

    fs = 1024.0
    t = np.arange(8192) / fs
    burst = np.exp(-((t - 3.0) / 0.2) ** 2) * np.sin(2 * np.pi * 180.0 * t)
    history = TimeHistory(
        t, np.vstack([burst, np.zeros_like(t)]),
        response_dof=['101Z+', '901X+'],
        ordinate_dim=['acceleration', 'force'])
    built = _build_block(
        {'kind': 'plot', 'source': 'Time History', 'mode': 'scalogram',
         'select': 'dim:acceleration', 'caption': 'Control scalogram'},
        {'Time History': history}, DEFAULT_SYSTEM)
    figure = built if isinstance(built, dict) else built[0]
    assert figure['kind'] == 'stage', 'rides the payload the page draws'
    assert '101Z+' in figure['caption'], 'and says whose record it is'

    # a surface, not a fan of lines (Brandon, 2026-08-29): the sheet
    # carries a level grid, and the loudest row sits at the burst's
    # own frequency — the geometry carries the measurement
    sheet = figure['sheet']
    assert 'runs' not in figure
    peaks = [max(row) for row in sheet['levels']]
    loudest = int(np.argmax(peaks))
    from visualdynamics.viz.waterfall import STAGE

    share = sheet['stations'][loudest] / STAGE[1]
    import re

    low, high = map(float, re.search(
        r'([\d.]+) to ([\d.]+) Hz', figure['caption']).groups())
    frequency = low * (high / low) ** share
    assert 150.0 < frequency < 215.0, frequency
    # the cone rides the sheet, widest at the lowest row, and the
    # low rows are marked as leaning on the record's ends
    cone = sheet['cone']
    assert cone[0] > cone[-1] > 0.0, 'widest at the lowest frequency'
    assert 'walls mark where' in figure['caption'], \
        'the cone reads as walls, the same as the app\'s stage'


def test_a_record_with_no_matching_channel_builds_nothing(qt_app):
    from visualdynamics.report import _build_block
    from visualdynamics.units import DEFAULT_SYSTEM

    fs = 512.0
    t = np.arange(1024) / fs
    history = TimeHistory(t, np.atleast_2d(np.sin(2 * np.pi * 50 * t)),
                          response_dof=['901X+'], ordinate_dim=['force'])
    built = _build_block(
        {'kind': 'plot', 'source': 'Time History', 'mode': 'scalogram',
         'select': 'dim:acceleration', 'caption': 'Control scalogram'},
        {'Time History': history}, DEFAULT_SYSTEM)
    assert built is None, 'a figure of nothing is absent, not empty'


def test_the_editor_offers_the_scalogram_a_channel_choice(qt_app):
    """The figure shows one channel, so the reader chooses which
    (Brandon, 2026-08-29): the drop-down offers exactly what the
    block's select admits, and the field op stores the pick."""
    from visualdynamics.report import scalogram_channel_options

    fs = 1024.0
    t = np.arange(4096) / fs
    history = TimeHistory(
        t, np.zeros((3, len(t))),
        response_dof=['101Z+', '104Z+', '901X+'],
        ordinate_dim=['acceleration', 'acceleration', 'force'])
    block = {'kind': 'plot', 'source': 'Time History',
             'mode': 'scalogram', 'select': 'dim:acceleration'}
    assert scalogram_channel_options(
        block, {'Time History': history}) == ['101Z+', '104Z+'], \
        'the force channel is outside the select, so it is not offered'

    # the pick lands on the block through the editor's own op, and
    # the figure honours it
    from visualdynamics.core.report import Report
    from visualdynamics.gui.report_editor import ReportEditor
    from visualdynamics.units import DEFAULT_SYSTEM

    editor = ReportEditor()
    report = Report('R', [dict(block)])
    editor.show_report(report, lambda: {'Time History': history},
                       DEFAULT_SYSTEM)
    editor._operate({'op': 'field', 'at': 0, 'field': 'channel',
                     'value': '104Z+'})
    assert report.blocks[0]['channel'] == '104Z+'


# ---- a long record is drawn at a picture's width, never at its own ----


def test_the_surface_is_held_to_the_column_budget(showing):
    """A five-minute record at 16 kHz is 121 rows of 4.9M columns, and
    building that as a surface is what crashed the app (Brandon,
    2026-09-15: the flat picture drew, the 3-D one died). The fixture is
    129 024 samples — already thirty times the budget — so the mesh's
    vertex count is the pin: rows times `COLUMNS`, not rows times
    samples."""
    from visualdynamics.core.wavelet import COLUMNS

    window, history = showing
    samples = len(history.abscissa)
    assert samples > COLUMNS, 'the fixture is over budget'
    actor = window.data_pane.waterfall_plotter.renderer.actors['scalogram']
    n_points = actor.mapper.dataset.n_points
    columns = held_columns(samples)
    rows = n_points / columns
    assert rows == int(rows) and 10 < rows < 200, (
        f'{n_points} vertices is not rows x {columns}')


def held_columns(samples):
    """What `scalogram_peaks` holds `samples` to: equal slices of the
    smallest whole-sample width that fits the budget — at most
    `COLUMNS`, and a little under it when the slices do not divide."""
    from visualdynamics.core.wavelet import COLUMNS

    step = -(-samples // COLUMNS)
    return -(-samples // step)


def test_the_flat_picture_is_held_to_the_column_budget(window, pump):
    from visualdynamics.core.wavelet import COLUMNS

    looking_at(window, pump)
    window.data_pane.wavelet_action.trigger()
    window.data_pane.waterfall_action.trigger()
    pump()
    plot = window.data_pane.graphics.getItem(0, 0)
    images = [item for item in plot.items
              if type(item).__name__ == 'ImageItem']
    assert len(images) == 1
    samples = len(window.objects['Time History'].abscissa)
    assert images[0].image.shape[0] == held_columns(samples) <= COLUMNS, (
        'time across, held')
