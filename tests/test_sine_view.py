"""The sine view: the toggle, the panel, the sampled preview, the
verb (Brandon, 2026-09-30). The extraction's settings ride the record
the way the filter's do — a taskbar toggle, a panel beside the plot,
the Extract button on it — with the smoothing chosen from the data
unless set, and the readings sampled at it before anything is made.
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

from visualdynamics.core.sine import SineExtraction, SineLevelSet


def _sweep(window, pump, noise=6.0, with_spec=True):
    from test_extract_sine import _flat_sweep

    history, spec = _flat_sweep(noise=noise)
    window.add_object('Record', history)
    if with_spec:
        window.add_object('Sweep', spec)
    select_objects(window, pump, 'Record')
    window.render_current()
    pump()
    return history, spec


def _toggle_sine(window, pump):
    pane = window.data_pane
    pane.sine_action.setChecked(True)
    pane._choose_sine(True)
    pump()
    return pane


def _flat(window, pump):
    pane = window.data_pane
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    return pane


# ---- the toggle ----------------------------------------------------------


def test_the_button_is_offered_only_beside_a_sweep_specification(window,
                                                                 pump):
    """A record with no sweep to be read against has no sine levels
    to speak of; the button is not shown (principle 3)."""
    from test_extract_sine import _flat_sweep

    _sweep(window, pump, with_spec=False)
    assert not window.data_pane.sine_action.isVisible()
    window.add_object('Sweep', _flat_sweep(noise=6.0)[1])
    select_objects(window, pump, 'Record')
    window.render_current()
    pump()
    assert window.data_pane.sine_action.isVisible()


def test_toggling_it_shows_the_panel_on_the_automatic_setting(window, pump):
    history, _spec = _sweep(window, pump)
    pane = _toggle_sine(window, pump)
    _flat(window, pump)
    panel = pane.sine_panel
    assert panel.isVisible()
    setting = panel.setting()
    assert setting.automatic and setting.chosen >= 60.0, \
        'the panel opens on the automatic choice for this loud record'
    assert panel.derived['chosen'].text() == f'{setting.chosen:g}'
    assert 'dB' in panel.derived['scatter'].text()
    assert panel.preview.items, 'the sampled readings are drawn'
    assert history.sine_extraction is None, 'opening the view sets nothing'


def test_the_panel_shows_only_what_applies(window, pump):
    _sweep(window, pump)
    pane = _toggle_sine(window, pump)
    _flat(window, pump)
    panel = pane.sine_panel
    assert panel.target_box.isVisible() and not panel.cycles_box.isVisible()
    panel.mode_box.setCurrentIndex(1)
    pump()
    assert panel.cycles_box.isVisible() and not panel.target_box.isVisible()


# ---- edits reach the record ----------------------------------------------


def test_an_edit_lands_on_the_record_and_in_the_journal(window, pump):
    history, _spec = _sweep(window, pump)
    pane = _toggle_sine(window, pump)
    _flat(window, pump)
    panel = pane.sine_panel
    panel.mode_box.setCurrentIndex(1)
    panel.cycles_box.setValue(40.0)
    pump()
    assert history.sine_extraction == SineExtraction(cycles=40.0)
    assert window.project.journal[-1] == \
        f"project['Record'].sine_extraction = {SineExtraction(cycles=40.0)!r}"


def test_the_button_makes_the_levels_with_the_setting_shown(window, pump):
    _sweep(window, pump)
    pane = _toggle_sine(window, pump)
    _flat(window, pump)
    panel = pane.sine_panel
    panel.mode_box.setCurrentIndex(1)
    panel.cycles_box.setValue(40.0)
    pump()
    panel.extract_button.click()
    pump()
    levels = next(obj for obj in window.objects.values()
                  if isinstance(obj, SineLevelSet))
    assert levels.cycles == 40.0
    assert '40 cycles of smoothing' in window.statusBar().currentMessage()
    assert window.project.journal[-1] == "project.extract_sine('Record')"
    # and a later edit badges the levels stale, as a filter edit does
    # (the press selected the levels; the edit is made on the record)
    select_objects(window, pump, 'Record')
    window.render_current()
    pump()
    assert panel.isVisible()
    panel.cycles_box.setValue(80.0)
    pump()
    name = next(n for n, o in window.objects.items() if o is levels)
    assert name in window.project.stale()


def test_the_automatic_press_records_what_it_chose(window, pump):
    history, _spec = _sweep(window, pump)
    pane = _toggle_sine(window, pump)
    _flat(window, pump)
    pane.sine_panel.extract_button.click()
    pump()
    levels = next(obj for obj in window.objects.values()
                  if isinstance(obj, SineLevelSet))
    assert history.sine_extraction.automatic
    assert history.sine_extraction.chosen == levels.cycles >= 60.0
    assert 'automatic smoothing' in window.statusBar().currentMessage()


def test_the_readings_are_exclusive(window, pump):
    """One reading of the record at a time: the sine view up puts the
    filter view away, and the other way round."""
    _sweep(window, pump)
    pane = _toggle_sine(window, pump)
    _flat(window, pump)
    assert pane.sine_panel.isVisible()
    pane.filter_action.setChecked(True)
    pane._choose_filter(True)
    window.render_current()
    pump()
    assert pane.filter_panel.isVisible() and not pane.sine_panel.isVisible()
    pane.sine_action.setChecked(True)
    pane._choose_sine(True)
    window.render_current()
    pump()
    assert pane.sine_panel.isVisible() and not pane.filter_panel.isVisible()


# ---- the flagged readings on the comparison ------------------------------


def test_readings_under_the_floor_are_ringed_on_the_comparison(window, pump):
    """A reading the tone was under the noise at is drawn as a hollow
    ring at the floor, beside the solid curve of the resolved ones."""
    import pyqtgraph as pg

    from visualdynamics.plot import build_plots

    history, spec = _sweep(window, pump, noise=12.0)
    name = window.project.extract_sine('Record')[0]
    history.sine_extraction = SineExtraction(cycles=10.0)
    window.project.refresh(name)
    levels = window.project[name]
    level = next(iter(levels))
    assert level.below_floor.any()
    pane = window.data_pane
    build_plots(pane.graphics, [('Up level', level, [0]),
                                ('Up requirement', spec.tone_curve('Up'), [0])],
                unit_system=window.unit_system, theme=window.theme_name)
    pump()
    rings = [item for plot in pane.graphics.ci.items
             if isinstance(plot, pg.PlotItem)
             for item in plot.listDataItems()
             if item.opts.get('symbol') == 'o' and item.opts.get('pen') is None]
    assert len(rings) == 1
    x, _y = rings[0].getData()
    assert len(x) == int(level.below_floor[0].sum())


def test_the_glyph_is_its_own(window):
    from visualdynamics.gui.icons import control_icon

    ours = control_icon('sine_levels').pixmap(32).toImage()
    assert ours != control_icon('no-such-glyph').pixmap(32).toImage()
    assert ours != control_icon('sine').pixmap(32).toImage()


@pytest.mark.parametrize('noise', [0.0])
def test_a_clean_record_opens_on_the_base_smoothing(window, pump, noise):
    _sweep(window, pump, noise=noise)
    pane = _toggle_sine(window, pump)
    _flat(window, pump)
    assert pane.sine_panel.setting().chosen == 10.0
    assert np.isfinite(float(pane.sine_panel.derived['scatter'].text().split()[0]))


def test_extracting_moves_the_strip_bar_and_puts_it_away(window, pump, monkeypatch):
    """The window's extraction ticks the strip's bar — the import's
    bar, shared (2026-10-01) — to full, then hides it."""
    _sweep(window, pump)
    seen = []
    real = window._strip_tick

    def spy(done, total):
        seen.append((done, total, not window._import_progress.isHidden()))
        real(done, total)

    monkeypatch.setattr(window, '_strip_tick', spy)
    window.extract_sine_levels('Record')
    pump()
    assert seen and seen[-1][0] == seen[-1][1] > 0
    assert all(shown for _d, _t, shown in seen[1:]), 'the bar was shown while it moved'
    assert window._import_progress.isHidden()
    assert 'extracted from Record' in window.statusBar().currentMessage()


def test_a_long_verb_runs_off_the_loop_and_the_loop_turns(window, pump):
    """`_run_long`: the verb runs on a worker thread while this thread's
    event loop keeps turning — the only way a status line set as the
    work starts, or a bar, reaches a macOS screen (Brandon, 2026-10-01:
    neither did when they were pumped from inside the blocked loop)."""
    import threading
    import time

    from PySide6.QtCore import QTimer

    turned = []
    seen = {}

    def verb(name, progress=None):
        seen['thread'] = threading.current_thread() is threading.main_thread()
        progress(1, 2)
        time.sleep(0.2)
        progress(2, 2)
        return name

    QTimer.singleShot(50, lambda: turned.append(True))
    assert window._run_long('Working…', verb, 'x') == 'x'
    assert seen['thread'] is False, 'the verb ran on the loop thread'
    assert turned, 'the loop did not turn while the verb ran'
    assert window._import_progress.isHidden()
    assert window._status_text == 'Working…', 'said on the strip while it ran'


def test_what_a_long_verb_raises_is_raised_on_the_loop_thread(window):
    def verb(name, progress=None):
        raise ValueError('no')

    with pytest.raises(ValueError, match='no'):
        window._run_long('Working…', verb, 'x')
    assert window._import_progress.isHidden()


def test_a_long_verb_shows_the_strip_and_its_bar_and_puts_them_away(window, pump):
    """The strip that holds the bar is hidden except while something
    runs — and the first runner never showed it, so a27's bar sat in
    a hidden strip (2026-10-02). Visible-to-the-window while the verb
    runs, both of them and the Cancel button; all gone after."""
    import time

    seen = []

    def verb(name, progress=None):
        progress(1, 2)
        time.sleep(0.05)
        progress(2, 2)
        return name

    real = window._strip_tick

    def spy(done, total):
        real(done, total)
        seen.append((window._import_strip.isVisibleTo(window),
                     window._import_progress.isVisibleTo(window),
                     window._cancel_button.isVisibleTo(window),
                     window._import_label.text()))

    window._strip_tick = spy
    try:
        window._run_long('Working…', verb, 'x')
    finally:
        window._strip_tick = real
    assert seen and all(strip and bar and cancel for strip, bar, cancel, _t in seen)
    assert seen[-1][3] == 'Working…', 'the status rides the strip, not the message'
    assert not window._import_strip.isVisibleTo(window)
    assert not window._cancel_button.isVisibleTo(window)
    assert not window._importing


def _cancellable(stopped):
    import time

    def verb(name, progress=None):
        for k in range(200):
            progress(k, 200)
            time.sleep(0.02)
        stopped.append('ran to the end')
        return name
    return verb


def test_the_cancel_button_stops_a_long_verb(window, pump):
    from PySide6.QtCore import QTimer

    from visualdynamics.core.progress import Cancelled

    stopped = []
    QTimer.singleShot(150, window._cancel_button.click)
    with pytest.raises(Cancelled):
        window._run_long('Working…', _cancellable(stopped), 'x')
    assert stopped == [], 'the verb was stopped at a tick, not run out'
    assert not window._cancel_button.isVisibleTo(window)
    # and the next run's button is live again
    assert window._run_long('Again…', lambda name, progress=None: name, 'y') == 'y'
    assert window._cancel_button.isEnabled()


def test_escape_cancels_a_long_verb_and_the_action_says_so(window, pump):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest

    from visualdynamics.core.data import TimeHistory

    _sweep(window, pump)
    stopped = []
    QTimer.singleShot(150, lambda: QTest.keyClick(window, Qt.Key.Key_Escape))
    acted = window._act_on(
        TimeHistory, 'nothing selected',
        lambda name, **kw: window._run_long('Working…', _cancellable(stopped),
                                            name, **kw))
    assert acted is None and stopped == []
    assert 'cancelled' in window.statusBar().currentMessage()
