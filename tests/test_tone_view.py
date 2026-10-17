"""The tone readout in the window: a cursor on a plain autospectrum,
the tone under it read and drawn, the reading in the pane beside it.

A reading of a spectrum that exists, so a toggle on the bar with no act
(as the octave preview is before its button), offered for a plain
autospectrum on even lines and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import Psd, TimeHistory
from visualdynamics.core.tone_readout import noise_bandwidth

G = 9.80665
FS = 2048.0


def _record(seconds=16.0):
    """A 1 grms floor from 5 to 500 Hz with two tones: 1.944 g at
    19.44 Hz and 1.5 g at 58.32 Hz, stored in m/s^2."""
    rng = np.random.default_rng(3304)
    n = int(seconds * FS)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    spectrum[(f < 5.0) | (f > 500.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x = x / np.sqrt(np.mean(x ** 2))
    t = np.arange(n) / FS
    for frequency, amplitude in ((19.44, 1.944), (58.32, 1.5)):
        x = x + amplitude * np.sin(2.0 * np.pi * frequency * t)
    history = TimeHistory(t, x[None] * G, response_dof=['1Z+'],
                          ordinate_dim=['acceleration'],
                          ordinate_unit=['g'])
    history.averaging = Averaging(frame_length=2048, overlap=0.5,
                                  window='hann').filled(n, FS)
    return history


def _asd(window, pump):
    window.add_object('Record', _record())
    name = window.project.compute_psds('Record')
    window.show_object(name)
    pump()
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
    window.render_current()
    pump()
    return name, window.objects[name], pane


def _tones_on(window, pump):
    name, asd, pane = _asd(window, pump)
    pane.tone_action.trigger()
    pump()
    return name, asd, pane


def test_the_reading_is_offered_on_a_plain_autospectrum_alone(window, pump):
    _name, asd, pane = _asd(window, pump)
    assert pane.tone_action.isVisible()
    item = window._item_for_object('Record')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert not pane.tone_action.isVisible(), 'a record has no lines'
    window.add_object('Banded', asd.to_octave(3))
    window.show_object('Banded')
    pump()
    window.render_current()
    pump()
    assert not pane.tone_action.isVisible(), \
        'octave bands have no line spacing to undo'


def test_the_toggle_puts_a_cursor_on_the_tallest_spike(window, pump):
    name, _asd, pane = _tones_on(window, pump)
    assert pane.tone_panel.isVisible()
    state = window._tone
    assert state is not None and state['name'] == name
    reading = state['reading']
    assert reading.frequency == pytest.approx(19.44, abs=0.05), \
        'the cursor starts on the tallest tone'
    assert reading.amplitude / G == pytest.approx(1.944, rel=0.02)
    derived = pane.tone_panel.derived
    assert derived['amplitude'].text() == f'{reading.amplitude / G:.4g} g peak'
    assert derived['spike'].text().endswith('g²/Hz')
    assert derived['spacing'].text() == '1 Hz'
    assert derived['bandwidth'].text() == '1.5 lines'
    assert derived['source'].text() == 'the averaging'
    # 19.44 Hz sits 0.44 of a line off a 1 Hz line, and the pane says so
    assert reading.between_lines
    assert 'sits 0.44 of a line' in pane.tone_panel.notes.text()


def test_the_overlay_stands_on_the_spike_in_display_units(window, pump):
    _name, _asd, _pane = _tones_on(window, pump)
    state = window._tone
    reading, overlay = state['reading'], state['overlay']
    assert overlay.top.isVisible()
    # drawn in g^2/Hz on the plot, read in SI
    shown = overlay.top.yData[0]
    expected = (reading.floor + reading.height) / G ** 2
    assert shown == pytest.approx(expected, rel=1e-9)
    assert overlay.label.toPlainText().startswith('1.94 g peak')


def test_moving_the_cursor_reads_the_tone_under_it(window, pump):
    _name, _asd, pane = _tones_on(window, pump)
    state = window._tone
    cursor = state['cursor']
    cursor.setValue(np.log10(58.0) if state['log_x'] else 58.0)
    pump()
    reading = window._tone['reading']
    assert reading.frequency == pytest.approx(58.32, abs=0.05)
    assert reading.amplitude / G == pytest.approx(1.5, rel=0.02)
    assert pane.tone_panel.derived['frequency'].text() == \
        f'{reading.frequency:.4g} Hz'
    assert window.statusBar().currentMessage() == \
        '1Z+: ' + reading.describe('g', 1.0 / G)


def test_a_window_chosen_in_the_pane_reads_again_under_it(window, pump):
    name, _asd, pane = _tones_on(window, pump)
    hann = window._tone['reading']
    box = pane.tone_panel.window_box
    box.setCurrentIndex(box.findData('boxcar'))
    pump()
    boxcar = window._tone['reading']
    assert pane.tone_panel.derived['bandwidth'].text() == '1 line'
    assert pane.tone_panel.derived['source'].text() == 'chosen'
    assert boxcar.window == 'boxcar' and hann.window == 'hann'
    assert boxcar.on_line > hann.on_line, \
        'a narrower bandwidth, a taller spike for the same tone'
    # and the choice is kept for this spectrum across a redraw
    window.render_current()
    pump()
    assert window._tone['reading'].window == 'boxcar'
    assert window._tone_windows[name] == ('boxcar', None)


def test_the_stage_stands_down_while_the_tones_are_read(window, pump):
    _name, _asd, pane = _tones_on(window, pump)
    pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    assert window._tone is not None, 'the cursor is on the flat plot'
    assert not (pane._waterfall_page is not None
                and pane._waterfall_page.isVisible())


def test_putting_the_reading_away_takes_the_cursor_off(window, pump):
    _name, _asd, pane = _tones_on(window, pump)
    pane.tone_action.trigger()
    pump()
    assert window._tone is None
    assert not pane.tone_panel.isVisible()


def test_the_reading_is_journaled_as_its_plot_call(window, pump):
    name, _asd, _pane = _tones_on(window, pump)
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_tone_readout(')]
    assert len(lines) == 1
    assert f'project[{name!r}]' in lines[0]
    assert f'window=project.averaging_of({name!r})' in lines[0]
    # and moving the cursor settles the same line rather than adding one
    state = window._tone
    state['cursor'].setValue(np.log10(58.0) if state['log_x'] else 58.0)
    pump()
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_tone_readout(')]
    assert len(lines) == 1 and '58.3' in lines[0]


def test_the_pane_s_notes_are_in_the_rows_units(window, pump):
    """At 300 Hz there is no tone; the note gives the floor in g²/Hz,
    the number the floor row shows, not the SI value."""
    _name, _asd, pane = _tones_on(window, pump)
    state = window._tone
    state['cursor'].setValue(np.log10(300.0) if state['log_x'] else 300.0)
    pump()
    reading = window._tone['reading']
    notes = pane.tone_panel.notes.text()
    assert f'({reading.floor / G ** 2:.3g} g²/Hz)' in notes
    assert '(m/s²)²' not in notes


def test_a_kaiser_s_beta_is_set_in_the_pane(window, pump):
    name, _asd, pane = _tones_on(window, pump)
    panel = pane.tone_panel
    assert panel.parameter_box.isHidden(), 'Hann takes no number'
    panel.window_box.setCurrentIndex(panel.window_box.findData('kaiser'))
    pump()
    assert not panel.parameter_box.isHidden()
    assert panel._parameter_label.text() == 'Beta'
    assert panel.parameter_box.value() == 14.0
    panel.parameter_box.setValue(6.0)
    pump()
    reading = window._tone['reading']
    assert reading.noise_bandwidth == pytest.approx(
        noise_bandwidth('kaiser', parameter=6.0))
    assert window._tone_windows[name] == ('kaiser', 6.0)
    line = next(line for line in window.project.journal
                if line.startswith('visualdynamics.plot.plot_tone_readout('))
    assert "window='kaiser', parameter=6.0" in line
    # kept across a redraw, the number with the window
    window.render_current()
    pump()
    assert panel.parameter_box.value() == 6.0
    assert window._tone['reading'].noise_bandwidth == pytest.approx(
        reading.noise_bandwidth)


def test_a_cross_spectrum_is_not_offered_the_reading(window, pump):
    """A real-valued off-diagonal record has no phase to show, but it
    is still not a density of one signal."""
    f = np.arange(0.0, 64.0)
    cross = Psd(f, np.ones((1, 64)), response_dof=['1Z+'],
                reference_dof=['2Z+'],
                ordinate_dim='acceleration*acceleration/frequency')
    window.add_object('Cross', cross)
    window.show_object('Cross')
    pump()
    window.render_current()
    pump()
    assert not window.data_pane.tone_action.isVisible()


def test_a_record_with_no_number_is_said_not_read(window, pump):
    f = np.arange(0.0, 64.0)
    empty = Psd(f, np.full((1, 64), np.nan), response_dof=['1Z+'],
                ordinate_dim='acceleration**2/frequency')
    window.add_object('Empty', empty)
    window.show_object('Empty')
    pump()
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
    window.render_current()
    pump()
    pane.tone_action.trigger()
    pump()
    assert window._tone is not None and window._tone['reading'] is None
    assert window.statusBar().currentMessage() == \
        'No tone read: no line of this spectrum holds a number'
