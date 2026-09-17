"""The truncate view: the cut drawn where it is decided.

The flat plot grays the ends being discarded and hands the span's
edges to the mouse; the stage draws the same cut as gray slabs with
the averaging span's own handles; the panel holds the numbers. All
three commit through one place, so they cannot disagree about what a
drag means — and a drag settles before it is said, the filter
corner's rule (2026-08-28).
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.truncate import Truncation


def _record(window, pump):
    rng = np.random.default_rng(7)
    t = np.arange(8192) / 2048.0
    history = visualdynamics.TimeHistory(
        t, rng.standard_normal((3, len(t))),
        response_dof=['101Z+', '104Z+', '9001X+'],
        ordinate_dim=['acceleration'] * 2 + ['force'])
    window.add_object('Record', history)
    item = window._item_for_object('Record')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    return history


def _toggle_truncate(window, pump):
    pane = window.data_pane
    pane.truncate_action.setChecked(True)
    pane._choose_truncate(True)
    pump()
    return pane


def _flat(window, pump):
    """Stand the 3-D reading down: the stage is the default for a
    record, so a test of the *flat* marks has to say so."""
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
        pump()
    window.render_current()
    pump()
    return pane


# ---- the flat view -------------------------------------------------------


def test_the_toggle_grays_the_ends_and_shows_the_panel(window, pump):
    _record(window, pump)
    _toggle_truncate(window, pump)
    pane = _flat(window, pump)
    assert window.truncation_overlays, 'marks are on the plot'
    overlay = window.truncation_overlays[0]
    assert len(overlay.shades) == 2, 'a shade over each discarded end'
    panel = pane.truncate_panel
    assert panel.isVisible()
    assert panel.start_box.value() == pytest.approx(0.0)
    assert panel.stop_box.value() == pytest.approx(8191 / 2048.0,
                                                   abs=1e-3), \
        'the record\'s own end, at the boxes\' four-decimal precision'
    assert '4' in panel.derived['record'].text(), 'the record names '\
        'its own length'


def test_an_edit_lands_on_the_history_and_raises_the_badge(window, pump):
    """The panel stores what it says, and the stored value is what the
    provenance fingerprints — so the edit is the moment a truncated
    record's refresh badge appears."""
    history = _record(window, pump)
    history.truncation = Truncation(0.5, 2.0)
    derived = window.project.truncate_data('Record')
    assert not window.project.stale()
    _toggle_truncate(window, pump)
    _flat(window, pump)
    panel = window.data_pane.truncate_panel
    panel.start_box.setValue(0.25)
    pump()
    assert history.truncation == Truncation(0.25, 2.0)
    assert derived in window.project.stale()
    assert window._stale.get(derived), 'the badge state re-read itself'


def test_a_region_drag_settles_once(window, pump):
    """Mid-drag the shading follows and nothing else happens; the
    release commits exactly one edit — the same contract as the
    filter corner, and for the same reason: whoever listens stores
    the span, re-reads staleness and re-settles the report."""
    history = _record(window, pump)
    _toggle_truncate(window, pump)
    _flat(window, pump)
    overlay = window.truncation_overlays[0]

    heard: list = []
    overlay.changed.connect(heard.append)
    for stop in (3.5, 3.0, 2.5, 2.0):
        overlay.region.lines[1].setValue(stop)   # the drag, line by line
        pump()
    assert not heard, 'mid-drag, nothing is said'
    assert overlay.shades[1].getRegion()[0] == pytest.approx(2.0), \
        'but the shading follows the hand'
    overlay.region.sigRegionChangeFinished.emit(overlay.region)
    pump()
    assert len(heard) == 1, 'the release says it, once'
    assert history.truncation == Truncation(0.0, 2.0)
    assert window.data_pane.truncate_panel.stop_box.value() == \
        pytest.approx(2.0), 'and the panel followed'


def test_the_boxes_cannot_state_a_backwards_span(window, pump):
    """The ranges are the enforcement, the filter panel's rule: a
    value typed past the other edge clamps one sample short of it."""
    history = _record(window, pump)
    _toggle_truncate(window, pump)
    _flat(window, pump)
    panel = window.data_pane.truncate_panel
    panel.stop_box.setValue(2.0)
    pump()
    panel.start_box.setValue(3.0)                # past the stop
    pump()
    assert panel.start_box.value() < 2.0
    assert history.truncation.start < history.truncation.stop


# ---- the calculator ------------------------------------------------------


def test_the_bar_does_not_offer_what_cannot_act(window, pump):
    """Every verb a view parameterizes lives on that view's panel
    (Brandon, 2026-08-28): Filter and Truncate left first — a blind
    suggestion and an outright refusal — and Compute SRS followed to
    the shock view, where a whole-record spectrum is one window set
    over the whole record."""
    _record(window, pump)
    labels = [label for _v, label, *_rest in window.acts_for(['Record'])]
    assert 'Truncate Data' not in labels
    assert 'Filter Data' not in labels
    assert 'Compute SRS' not in labels, \
        'the shock view owns it: a whole-record SRS is a window set '\
        'over the whole record (Brandon, 2026-08-28)'


def test_truncate_data_refuses_politely_without_a_span(window, pump):
    _record(window, pump)
    window.truncate_data()
    assert 'no span set' in window._status_text, \
        'the refusal says where to set one'


def test_truncate_data_makes_the_record(window, pump):
    history = _record(window, pump)
    history.truncation = Truncation(0.5, 2.0)
    window.truncate_data()
    cut = window.objects['Record Truncated']
    assert cut.abscissa[0] == pytest.approx(0.5)
    assert cut.abscissa[-1] == pytest.approx(2.0)
    assert 'keeping 0.5 to 2 s' in window._status_text


# ---- the stage -----------------------------------------------------------


def _stage(window, pump):
    history = _record(window, pump)
    pane = window.data_pane
    pane.truncate_action.trigger()
    pump()
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible(), 'the 3-D reading is the default'
    return history, pane


def test_the_truncation_marks_the_stage(window, pump):
    history, pane = _stage(window, pump)
    history.truncation = Truncation(0.5, 2.0)
    window.render_current()
    pump()
    names = set(pane.waterfall_plotter.actors)
    assert {'marks-truncation-head', 'marks-truncation-tail',
            'marks-truncation-edge-start', 'marks-truncation-edge-stop',
            'marks-truncation-handle-start', 'marks-truncation-handle-move',
            'marks-truncation-handle-stop'} <= names
    assert pane.truncate_panel.isVisible(), 'the panel rides the stage'


def test_a_whole_record_span_sheds_its_slabs(window, pump):
    """Nothing is being discarded, so nothing is grayed — a zero-width
    slab would still draw a seam at the wall."""
    _history, pane = _stage(window, pump)
    window.render_current()
    pump()
    names = set(pane.waterfall_plotter.actors)
    assert 'marks-truncation-head' not in names
    assert 'marks-truncation-tail' not in names
    assert 'marks-truncation-handle-move' in names, \
        'the handles still offer the drag'


def test_dragging_the_stop_handle_moves_the_stop(window, pump):
    history, pane = _stage(window, pump)
    dragger = window._stage_dragger
    assert dragger is not None and dragger._context is not None
    assert dragger.begin('marks-truncation-handle-stop')
    dragger.drag_to(2.0)
    dragger.finish(2.0)
    pump()
    assert history.truncation == Truncation(0.0, 2.0)
    assert pane.truncate_panel.stop_box.value() == pytest.approx(2.0), \
        'the panel followed — one control shown twice'


def test_a_move_slides_the_span_with_its_width_kept(window, pump):
    history, _pane = _stage(window, pump)
    history.truncation = Truncation(0.5, 2.5)
    window.render_current()
    pump()
    dragger = window._stage_dragger
    assert dragger.begin('marks-truncation-handle-move')
    dragger.finish(2.5)                        # anchor was 1.5: +1 s
    pump()
    assert history.truncation == Truncation(1.5, 3.5)


def test_a_slide_stops_at_the_records_wall(window, pump):
    history, _pane = _stage(window, pump)
    history.truncation = Truncation(0.5, 2.5)
    window.render_current()
    pump()
    dragger = window._stage_dragger
    assert dragger.begin('marks-truncation-handle-move')
    dragger.finish(100.0)                      # far past the end
    pump()
    last = 8191 / 2048.0
    assert history.truncation.stop == pytest.approx(last)
    assert history.truncation.start == pytest.approx(last - 2.0), \
        'the width survived the clamp'


def test_the_apply_button_sleeps_while_there_is_nothing_to_cut(window,
                                                               pump):
    """A whole-record cut is a copy, and a button that makes one is a
    button that does nothing — so it wakes exactly when the span
    leaves the walls (Brandon, 2026-08-28)."""
    _record(window, pump)
    _toggle_truncate(window, pump)
    _flat(window, pump)
    panel = window.data_pane.truncate_panel
    assert not panel.apply_button.isEnabled(), \
        'the opening span is the whole record: nothing to cut'
    panel.start_box.setValue(0.5)
    pump()
    assert panel.apply_button.isEnabled()
    # opened back out by hand — the boxes reach every span, so there
    # is no reset button (Brandon, 2026-08-28)
    panel.start_box.setValue(panel.start_box.minimum())
    panel.stop_box.setValue(panel.stop_box.maximum())
    pump()
    assert not panel.apply_button.isEnabled(), 'opened back out, it sleeps'


def test_the_panel_applies_the_truncation_it_is_showing(window, pump):
    _record(window, pump)
    _toggle_truncate(window, pump)
    _flat(window, pump)
    panel = window.data_pane.truncate_panel
    panel.start_box.setValue(0.5)
    panel.stop_box.setValue(2.0)
    pump()
    panel.apply_button.click()
    pump()
    cut = window.objects['Record Truncated']
    assert cut.abscissa[0] == pytest.approx(0.5)
    assert cut.abscissa[-1] == pytest.approx(2.0)
    assert window.project.provenance['Record Truncated']['verb'] == \
        'truncate_data'
