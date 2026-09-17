"""The averaging view: a button on the bar, a table beside the plot.

Clicking a time history shows the trace as it always did. The button
adds the frames — shaded on the plot, listed in the table — and takes
them away again. Nothing here computes: the calculator in the tree is
still what turns a history into a PSD, with whatever is set at the
moment it is clicked.
"""

from __future__ import annotations

import pytest
from conftest import fixture_path

from visualdynamics.core.averaging import Averaging

MODAL = ('plate', 'modal.nc4')
SPLIT = ('plate', 'modal_spectra.nc4')


pytestmark = pytest.mark.usefixtures('flat_reading')

def looking_at(window, pump, fixture, name='Time History'):
    """Import a file and select its time history."""
    window.import_paths([fixture_path(*fixture)])
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    return window.objects[name]


def show_averaging(window, pump, wanted=True):
    """Click the bar's averaging button."""
    action = window.data_pane.averaging_action
    if action.isChecked() != wanted:
        action.trigger()          # a checkable action toggles and emits
    pump()


def test_a_time_history_offers_the_averaging_button(window, pump):
    looking_at(window, pump, MODAL)
    assert window.data_pane.averaging_action.isVisible()
    assert window.data_pane.toolbar.isVisible()


def test_nothing_else_offers_it(window, pump):
    """An FRF has no record to cut; a button that did nothing would
    still have to be explained."""
    looking_at(window, pump, SPLIT)
    frf = next(name for name, obj in window.objects.items()
               if type(obj).__name__ == 'Frf')
    item = window._item_for_object(frf)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert not window.data_pane.averaging_action.isVisible()


def test_the_frames_are_off_until_they_are_asked_for(window, pump):
    """The plot is what it always was until the button goes down."""
    looking_at(window, pump, MODAL)
    assert window.averaging_overlays == []
    assert not window.data_pane.averaging_panel.isVisible()


def test_the_button_puts_the_frames_up(window, pump):
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    assert window.averaging_overlays, 'nothing was drawn'
    assert window.data_pane.averaging_panel.isVisible()
    for overlay in window.averaging_overlays:
        assert len(overlay.bands) == history.averaging.frames


def test_the_button_takes_them_away_again(window, pump):
    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    show_averaging(window, pump, wanted=False)
    assert window.averaging_overlays == []
    assert not window.data_pane.averaging_panel.isVisible()


def test_every_plot_of_the_record_is_marked(window, pump):
    """A modal capture holds force and acceleration, drawn a row each.
    The frames are cut from the same record, so they belong on both —
    marking one and not the other would read as two analyses."""
    import pyqtgraph as pg

    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    plots = [item for item in window.data_pane.graphics.ci.items
             if isinstance(item, pg.PlotItem)]
    assert len(plots) > 1, 'the fixture really does draw more than one row'
    assert len(window.averaging_overlays) == len(plots)


def test_moving_to_another_object_takes_the_frames_off(window, pump):
    """They describe the record that was up. Left behind, they would be
    marks on somebody else's data."""
    looking_at(window, pump, SPLIT)
    show_averaging(window, pump)
    frf = next(name for name, obj in window.objects.items()
               if type(obj).__name__ == 'Frf')
    item = window._item_for_object(frf)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert window.averaging_overlays == []
    assert not window.data_pane.averaging_panel.isVisible()


# ---- the table and the plot say the same thing --------------------------


def test_the_table_opens_on_what_the_file_said(window, pump):
    """The whole point of reading the parameters at import: they are
    already right, and the table shows what will be used."""
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    assert panel.averaging() == history.averaging
    assert panel.length_box.value() == 2048
    assert panel.frames_box.value() == 20


def test_an_edit_in_the_table_moves_the_shading(window, pump):
    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    panel.frames_box.setValue(4)
    pump()
    for overlay in window.averaging_overlays:
        assert len(overlay.bands) == 4


def test_an_edit_in_the_table_is_what_the_next_psd_uses(window, pump):
    """No Apply button and no compute button: the calculator computes
    with whatever is set, so an edit has to reach the object itself."""
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    panel.window_box.setCurrentIndex(panel.window_box.findData('blackman'))
    panel.frames_box.setValue(3)
    pump()
    assert history.averaging.window == 'blackman'
    assert history.averaging.frames == 3
    psd = history.compute_psds()
    assert len(psd.abscissa) == 2048 // 2 + 1


def test_a_drag_on_the_plot_moves_the_table(window, pump):
    """The two are one control shown twice; either may be the one used."""
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    overlay = window.averaging_overlays[0]
    overlay.region.setRegion((1.0, 3.0))
    pump()
    panel = window.data_pane.averaging_panel
    assert panel.start_box.value() == pytest.approx(1.0)
    assert panel.averaging() == history.averaging
    assert history.averaging.start == pytest.approx(1.0)


def test_a_drag_on_one_plot_moves_the_other(window, pump):
    """Both are marks on the same record; two answers would be a lie
    about which one the calculator is going to use."""
    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    first, second = window.averaging_overlays[0], window.averaging_overlays[1]
    first.region.setRegion((2.0, 4.0))
    pump()
    assert second.averaging == first.averaging
    assert second.region.getRegion() == pytest.approx(
        first.region.getRegion())


# ---- a capture the controller already cut -------------------------------


def test_a_split_capture_is_still_the_users_to_read(window, pump):
    """The frames the controller wrote are the right default and not the
    only reading of the record.

    They were locked once, on the argument that the file had settled
    them. But a burst-random capture is half excitation and half
    ringdown, and analyzing the burst alone — or putting an exponential
    window on the decay — is an ordinary thing to want and was
    impossible. Every parameter is editable; the machinery underneath
    always supported it.
    """
    looking_at(window, pump, SPLIT)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    assert panel.isVisible()
    for box in (panel.frames_box, panel.length_box, panel.overlap_box,
                panel.start_box, panel.window_box):
        assert box.isEnabled()
    assert not panel.detect_button.isEnabled(), (
        'detection searches a stream for the stretch that is the test, '
        'and inside one already-cut frame there is no such search')


def test_half_a_split_record_is_half_its_samples(window, pump):
    """Shortening the frame on a pre-cut capture analyses part of each
    record — the burst without the ringdown, which is the reason to want
    this at all."""
    history = looking_at(window, pump, SPLIT)
    show_averaging(window, pump)
    whole = history.compute_psds()
    samples = len(history.abscissa)
    panel = window.data_pane.averaging_panel
    panel.length_box.setValue(samples // 2)
    pump()
    assert history.averaging.frame_length == samples // 2
    half = history.compute_psds()
    assert len(half.abscissa) == samples // 4 + 1, 'half the lines'
    assert half.abscissa[1] == pytest.approx(2 * whole.abscissa[1]), (
        'and twice the spacing between them')


def test_a_split_capture_counts_its_records_as_the_averages(window, pump):
    """One frame per record and twenty records is twenty averages, and
    the count in the table has to say twenty rather than one."""
    looking_at(window, pump, SPLIT)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    assert panel.frames_box.value() == 1, 'one frame inside each record'
    assert panel.derived['averages'].text() == '20 × 1 = 20', \
        'the arithmetic said, not just the product: how many playings, '\
        'cut how many ways (Brandon, 2026-08-28)'


def test_a_split_capture_can_be_dragged_too(window, pump):
    """Dragging the shaded span is how you say 'just the burst', and a
    pre-cut capture is exactly where you want to say it."""
    looking_at(window, pump, SPLIT)
    show_averaging(window, pump)
    assert window.averaging_overlays
    for overlay in window.averaging_overlays:
        assert overlay.region.movable


def test_a_continuous_capture_can_be(window, pump):
    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    assert all(overlay.region.movable
               for overlay in window.averaging_overlays)
    assert window.data_pane.averaging_panel.frames_box.isEnabled()


# ---- what the table works out for you -----------------------------------


def test_the_table_shows_what_follows_from_the_five(window, pump):
    """Frame length is set in samples, and what it means is a frequency
    resolution — the trade is worth seeing while it is being made."""
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    panel.length_box.setValue(8192)
    pump()
    rate = history.sample_rate
    assert panel.derived['resolution'].text() == f'{rate / 8192:.4g} Hz'
    assert panel.derived['duration'].text() == f'{8192 / rate:.4g} s'
    # and they really do move with the frame length, in opposite ways
    assert float(panel.derived['duration'].text().split()[0]) > 1.0
    assert float(panel.derived['resolution'].text().split()[0]) < 1.0


def test_the_table_will_not_ask_for_more_record_than_there_is(window, pump):
    """Lengthening a frame with the count already at what the record
    held would ask for record that is not there. The count comes down
    rather than the edit being refused."""
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    panel.frames_box.setValue(panel.frames_box.maximum())
    panel.length_box.setValue(8192)
    pump()
    settled = panel.averaging()
    assert settled.fits(len(history.abscissa), history.sample_rate)
    assert history.averaging.fits(len(history.abscissa), history.sample_rate)


def test_a_start_past_the_end_is_pulled_back(window, pump):
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    panel.start_box.setValue(1e6)
    pump()
    assert panel.averaging().fits(len(history.abscissa), history.sample_rate)


def test_the_averaging_is_offered_even_when_the_file_said_nothing(window,
                                                                  pump):
    """A history assembled in a script has no parameters on it; the view
    still opens, on the whole record as one frame, which is what
    computing a PSD without any averaging has always meant."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory

    t = np.arange(4096) / 1024.0
    history = TimeHistory(abscissa=t, ordinate=np.sin(2 * np.pi * 30 * t)[None],
                          response_dof=['101X+'], ordinate_dim=['acceleration'])
    assert history.averaging is None
    window.add_object('Bare', history)
    item = window._item_for_object('Bare')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    assert panel.averaging() == Averaging.for_records(4096)
    assert window.averaging_overlays


# ---- worked out from the record -----------------------------------------


def test_detect_fills_the_table_in_from_the_record(window, pump):
    """The button answers 'where should I average and over how many
    frames' from the record itself, rather than leaving it to be typed."""
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    before = panel.averaging()
    panel.detect_button.click()
    pump()
    found = panel.averaging()
    assert found != before
    assert found == history.suggest_averaging(window=found.window,
                                              overlap=found.overlap)
    assert found.fits(len(history.abscissa), history.sample_rate)


def test_detect_reaches_the_object_and_the_shading(window, pump):
    """It is an edit like any other: the next PSD uses it, and the
    frames on the plot move to where it says."""
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    panel.detect_button.click()
    pump()
    assert history.averaging == panel.averaging()
    for overlay in window.averaging_overlays:
        assert len(overlay.bands) == history.averaging.frames


def test_detect_keeps_the_window_the_user_chose(window, pump):
    """Which window to apply is a judgment about the measurement, not
    something to read off the record; the button settles where and how
    many, and leaves that alone."""
    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    panel.window_box.setCurrentIndex(panel.window_box.findData('flattop'))
    pump()
    panel.detect_button.click()
    pump()
    assert panel.averaging().window == 'flattop'


def test_a_settled_capture_has_nothing_to_detect(window, pump):
    """Every record is one frame and they are all of the test; there is
    no stretch to find and a live button would promise otherwise."""
    looking_at(window, pump, SPLIT)
    show_averaging(window, pump)
    assert not window.data_pane.averaging_panel.detect_button.isEnabled()


def test_the_panel_does_not_take_qts_own_window(window, pump):
    """`widget.window()` is Qt's own verb for the top-level window a
    widget sits in, and every Qt codebase uses it that way — to parent a
    dialog, to raise the window an edit came from.

    The panel used to define `window()` as *which window function is
    selected*, which is a different question with an unrelated answer.
    Nothing outside the panel had called it yet, so the collision cost
    nothing; the day someone inside this class wrote
    `QMessageBox.warning(self.window(), ...)` it would have cost an
    afternoon, because a string where a QWidget belongs fails somewhere
    else entirely.
    """
    from PySide6.QtWidgets import QWidget

    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    assert isinstance(panel.window(), QWidget), (
        f'panel.window() answered {panel.window()!r}; Qt promises the '
        'top-level widget')
    assert panel.chosen_window() == panel.averaging().window


def test_the_stage_toggle_survives_the_marks_views(window, pump):
    """The 2D/3D button stays live while the averaging or shocks view
    is up — the marks draw in whichever reading is chosen (Brandon,
    2026-08-23; the button used to vanish, then to force 2-D)."""
    pane = window.data_pane
    looking_at(window, pump, MODAL)
    pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    assert pane.waterfall_action.isVisible()
    assert pane.showing_waterfall
    show_averaging(window, pump)
    assert pane.showing_averaging
    assert pane.showing_waterfall, 'the stage holds; the marks join it'
    assert pane.waterfall_action.isVisible()
    assert pane.averaging_panel.isVisible()
    # both readings stay reachable with the view up
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    assert pane.showing_averaging and pane.averaging_panel.isVisible()
    pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    assert pane.showing_averaging
    # the shocks view rides the stage the same way
    show_averaging(window, pump, wanted=False)
    pane.shocks_action.trigger()
    pump()
    assert pane.showing_shocks and pane.showing_waterfall
    assert pane.shock_panel.isVisible()


# ---- the acts these frames parameterize ----------------------------------


def test_the_panel_computes_psds_over_exactly_these_frames(window, pump):
    """The act where its settings are set (Brandon, 2026-08-28): the
    button is a second entrance to the calculator's own verb, so the
    result carries the same provenance and the same badge."""
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    window.data_pane.averaging_panel.psds_button.click()
    pump()
    name = next(n for n, obj in window.objects.items()
                if type(obj).__name__ == 'Psd')
    assert window.project.provenance[name]['verb'] == 'compute_psds'
    # the fingerprint reads the framing this panel edits: move it and
    # the record just made goes stale. Selected again first — making
    # the PSDs showed *them*, and the panel edits whatever is up
    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    window.data_pane.averaging_panel.frames_box.setValue(
        max(history.averaging.frames - 1, 1))
    pump()
    assert name in window.project.stale()


def test_the_panel_computes_cpsds_the_same_way(window, pump):
    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    window.data_pane.averaging_panel.cpsds_button.click()
    pump()
    assert any(window.project.provenance[n]['verb'] == 'compute_cpsds'
               for n in window.project.provenance)


def test_the_panel_computes_frfs_and_coherence_over_these_frames(
        window, pump, monkeypatch):
    """The whole spectral family lives with the framing now: FRFs ask
    their estimator exactly as the menu entry did, and the coherence
    beside them reads the same frames."""
    from PySide6.QtWidgets import QInputDialog

    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    assert panel.frfs_button.isVisible(), 'a hammer test holds a drive'
    assert panel.coherence_button.isVisible()

    monkeypatch.setattr(
        QInputDialog, 'getItem',
        staticmethod(lambda *_a, **_k: ('H1 — noise on the response',
                                        True)))
    panel.frfs_button.click()
    pump()
    assert any(window.project.provenance[n]['verb'] == 'compute_frfs'
               for n in window.project.provenance)

    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel.coherence_button.click()
    pump()
    assert any(window.project.provenance[n]['verb']
               == 'compute_multiple_coherence'
               for n in window.project.provenance)


def test_the_reference_buttons_hide_where_nothing_drives(window, pump):
    """Principle 3: a history of bare responses cannot say what drove
    it, so the estimates that need references are not offered."""
    import numpy as np

    import visualdynamics

    rng = np.random.default_rng(4)
    t = np.arange(4096) / 1024.0
    window.add_object('Responses', visualdynamics.TimeHistory(
        t, rng.standard_normal((2, len(t))),
        response_dof=['101Z+', '104Z+'],
        ordinate_dim=['acceleration', 'acceleration']))
    item = window._item_for_object('Responses')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    assert not panel.frfs_button.isVisible()
    assert not panel.coherence_button.isVisible()
    assert panel.psds_button.isVisible(), 'the power estimates stay'


def test_the_panel_computes_spectra_too(window, pump):
    looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    window.data_pane.averaging_panel.spectra_button.click()
    pump()
    assert any(window.project.provenance[n]['verb'] == 'compute_spectra'
               for n in window.project.provenance)


def test_a_parameterized_window_brings_its_one_box(window, pump):
    """Tukey and kaiser each take a number; the row for it appears
    only while such a window is chosen, wearing the parameter's own
    name — the UI stays minimal for the windows that take none
    (Brandon, 2026-08-29)."""
    history = looking_at(window, pump, MODAL)
    show_averaging(window, pump)
    panel = window.data_pane.averaging_panel
    assert not panel.parameter_box.isVisibleTo(panel), \
        'hann takes no parameter, so no row asks for one'

    panel.window_box.setCurrentIndex(panel.window_box.findData('kaiser'))
    pump()
    assert panel.parameter_box.isVisibleTo(panel)
    assert panel._parameter_label.text() == 'Beta'
    assert panel.parameter_box.value() == 14.0, 'the documented default'
    assert history.averaging.window_parameter == 14.0, \
        'and the record heard the switch, default included'

    panel.parameter_box.setValue(8.0)
    pump()
    assert history.averaging.window_parameter == 8.0

    panel.window_box.setCurrentIndex(panel.window_box.findData('tukey'))
    pump()
    assert panel._parameter_label.text() == 'Alpha'
    assert panel.parameter_box.value() == 0.5, \
        'each window seeds its own default at the moment it is chosen'

    panel.window_box.setCurrentIndex(panel.window_box.findData('hann'))
    pump()
    assert not panel.parameter_box.isVisibleTo(panel)
    assert history.averaging.window_parameter is None
