"""The rms check in the window: a toggle on a plain autospectrum's bar,
the judgment and the gap in the plot's corner, the whole check in the
pane beside it.

A reading of a spectrum that exists, so a toggle on the bar with no act
(as the tones are), offered for a plain autospectrum with a time history
in the project to check it against, and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import Psd, TimeHistory

G = 9.80665
FS = 2048.0


def _record(seconds=8.0, seed=1502, grms=2.0):
    """2 grms of noise from 20 to 800 Hz, Hann frames at half overlap
    set on it, stored in m/s^2."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    spectrum[(f < 20.0) | (f > 800.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x = x * grms / np.sqrt(np.mean(x ** 2))
    history = TimeHistory(np.arange(n) / FS, x[None] * G,
                          response_dof=['1Z+'],
                          ordinate_dim=['acceleration'],
                          ordinate_unit=['g'])
    history.averaging = Averaging(frame_length=1024, overlap=0.5,
                                  window='hann').filled(n, FS)
    return history


def _show(window, pump, name):
    window.show_object(name)
    pump()
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
    window.render_current()
    pump()
    return pane


def _asd(window, pump):
    window.add_object('Record', _record())
    name = window.project.compute_psds('Record')
    return name, window.objects[name], _show(window, pump, name)


def _checked(window, pump):
    name, asd, pane = _asd(window, pump)
    pane.rms_check_action.trigger()
    pump()
    return name, asd, pane


def test_the_check_is_offered_on_a_plain_autospectrum_alone(window, pump):
    _name, asd, pane = _asd(window, pump)
    assert pane.rms_check_action.isVisible()
    item = window._item_for_object('Record')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert not pane.rms_check_action.isVisible(), \
        'a record is what the spectrum is checked against'
    window.add_object('Banded', asd.to_octave(3))
    _show(window, pump, 'Banded')
    assert not pane.rms_check_action.isVisible(), \
        'octave bands are not a density on lines'


def test_an_imported_spectrum_is_offered_only_with_its_channel(window, pump):
    psd = _record().compute_psds()
    window.add_object('Controller', Psd(
        psd.abscissa, psd.ordinate, response_dof=['5X+'],
        ordinate_dim=['acceleration**2/frequency']))
    window.add_object('Record', _record())
    pane = _show(window, pump, 'Controller')
    assert not pane.rms_check_action.isVisible(), \
        'no record in the project holds a 5X+ acceleration'
    window.add_object('Other', Psd(
        psd.abscissa, psd.ordinate, response_dof=['1Z+'],
        ordinate_dim=['acceleration**2/frequency']))
    pane = _show(window, pump, 'Other')
    assert pane.rms_check_action.isVisible()
    pane.rms_check_action.trigger()
    pump()
    state = window._rms_check
    assert state['history'] == 'Record' and state['source'] is None
    assert pane.rms_check_panel.history_box.currentText() == 'Record'


def test_the_toggle_writes_the_check_in_the_pane_and_the_corner(window,
                                                                pump):
    name, _spectrum, pane = _checked(window, pump)
    panel = pane.rms_check_panel
    assert panel.isVisible()
    state = window._rms_check
    assert state is not None and state['name'] == name
    check = state['check']
    assert check == window.project.rms_check(name)
    assert check.passed and check.window == 'hann'
    derived = panel.derived
    assert derived['spectrum'].text() == f'{check.spectrum_rms / G:.4g} g rms'
    assert derived['time'].text() == f'{check.time_rms / G:.4g} g rms'
    assert derived['gap'].text() == f'{check.gap_db:+.3f} dB'
    assert derived['framed'].text() == f'{check.framed_rms / G:.4g} g rms'
    assert derived['lines'].text() == '513, 2 Hz apart'
    assert derived['window'].text() == 'hann, 1.5 lines'
    assert derived['span'].text() == '0 to 8 s'
    assert panel.verdict.text() == ('Agree within 0.05 dB: the spectrum '
                                    "carries the record's mean square.")
    assert panel.history_box.currentText() == 'Record (computed from)'
    text = state['label'].text()
    assert 'rms check: agree' in text and f'{check.gap_db:+.3f} dB' in text
    assert window.statusBar().currentMessage() == \
        '1Z+: ' + check.describe('g', 1.0 / G)


def test_a_threshold_set_in_the_pane_judges_again_and_is_kept(window,
                                                               pump):
    name, _spectrum, pane = _checked(window, pump)
    panel = pane.rms_check_panel
    panel.threshold_box.setValue(0.0)
    pump()
    check = window._rms_check['check']
    assert check.threshold_db == 0.0 and not check.passed
    assert check.explained, 'Hann at half overlap: the window, not a scale'
    assert "the window's gap" in window._rms_check['label'].text()
    assert panel.verdict.text().startswith('The spectrum reads ')
    assert window._rms_check_settings[name]['threshold_db'] == 0.0
    window.render_current()
    pump()
    assert window._rms_check['check'].threshold_db == 0.0
    assert panel.threshold_box.value() == 0.0


def test_another_record_can_be_chosen_to_check_against(window, pump):
    name, _spectrum, pane = _asd(window, pump)
    window.add_object('Louder', _record(seed=7, grms=4.0))
    pane = _show(window, pump, name)
    pane.rms_check_action.trigger()
    pump()
    box = pane.rms_check_panel.history_box
    assert [box.itemData(i) for i in range(box.count())] == \
        ['Record', 'Louder'], 'the record it came from first'
    box.setCurrentIndex(box.findData('Louder'))
    pump()
    check = window._rms_check['check']
    assert check.gap_db == pytest.approx(-6.02, abs=0.05), \
        'checked against a record twice as loud'
    assert not check.passed and check.framed_rms is not None
    assert pane.rms_check_panel.verdict.text().startswith(
        'The spectrum reads 6.0')


def test_the_stage_stands_down_while_the_check_is_up(window, pump):
    _name, _spectrum, pane = _checked(window, pump)
    pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    assert window._rms_check is not None, 'the words are on the flat plot'
    assert not (pane._waterfall_page is not None
                and pane._waterfall_page.isVisible())


def test_putting_the_check_away_takes_the_label_off(window, pump):
    _name, _spectrum, pane = _checked(window, pump)
    label = window._rms_check['label'].label
    pane.rms_check_action.trigger()
    pump()
    assert window._rms_check is None
    assert not pane.rms_check_panel.isVisible()
    assert label.parentItem() is None


def test_the_check_is_journaled_as_its_plot_call(window, pump):
    name, _spectrum, pane = _checked(window, pump)

    def lines():
        return [line for line in window.project.journal
                if line.startswith('visualdynamics.plot.plot_rms_check(')]

    assert lines() == [(
        f"visualdynamics.plot.plot_rms_check(project[{name!r}], "
        f"project['Record'], record=0, averaging=project.averaging_of("
        f"{name!r}), threshold_db=0.05, path='rms_check.png', show=False)")]
    pane.rms_check_panel.threshold_box.setValue(0.1)
    pump()
    assert len(lines()) == 1 and 'threshold_db=0.1,' in lines()[0], \
        'settled in place, not added to'


def test_two_undeclared_channels_are_each_checked_against_their_own(window,
                                                                    pump):
    """An acceleration and a force at 1Z+, neither declared, told apart
    by their hints: the check is offered on each record of their PSD and
    reads each against its own channel, not refused as two channels."""
    accel = _record(seconds=4.0, seed=7).ordinate[0] / G
    force = 1e-6 * _record(seconds=4.0, seed=8).ordinate[0] / G
    n = accel.size
    pair = TimeHistory(np.arange(n) / FS, np.vstack([accel, force]),
                       response_dof=['1Z+', '1Z+'],
                       dimension_hint=['acceleration', 'force'])
    pair.averaging = Averaging(frame_length=1024, overlap=0.0,
                               window='boxcar').filled(n, FS)
    window.add_object('Pair', pair)
    name = window.project.compute_psds('Pair')
    pane = _show(window, pump, name)
    for record, samples in ((0, accel), (1, force)):
        window._select_records(name, [record])
        window.render_current()
        pump()
        assert pane.rms_check_action.isVisible(), f'record {record}'
        if not pane.rms_check_action.isChecked():
            pane.rms_check_action.trigger()
            pump()
        state = window._rms_check
        assert state['record'] == record and state['history'] == 'Pair'
        check = state['check']
        assert check.time_rms == pytest.approx(np.sqrt(np.mean(samples ** 2)))
        assert check.gap_db == pytest.approx(0.0, abs=1e-9)
        assert check.passed


def test_a_long_channel_is_checked_on_the_worker(window, pump, monkeypatch):
    """Past `RMS_CHECK_INLINE_SAMPLES` the check runs through `_run_long`
    with the bar up, and its result is written as an inline one is; under
    it the window's thread reads it directly."""
    from visualdynamics.gui import main_window as module

    ran = []
    original = window._run_long

    def spy(status, verb, *args, **kwargs):
        ran.append(status)
        return original(status, verb, *args, **kwargs)

    monkeypatch.setattr(window, '_run_long', spy)
    name, _spectrum, pane = _checked(window, pump)
    assert ran == [] and window._rms_check['check'].passed
    monkeypatch.setattr(module, 'RMS_CHECK_INLINE_SAMPLES', 1000)
    pane.rms_check_panel.threshold_box.setValue(0.2)
    pump()
    assert ran == [f'Checking the rms of {name}…']
    check = window._rms_check['check']
    assert check.passed and check.threshold_db == pytest.approx(0.2)
    assert pane.rms_check_panel.verdict.text().startswith('Agree within 0.2')
