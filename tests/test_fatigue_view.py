"""The fatigue reading in the window: a toggle on an acceleration record's
bar, the extreme response and fatigue damage spectra drawn with the
time-compression rule, and the settings and the comparison with a
candidate test in the pane beside them.

Offered for one record of an acceleration PSD or specification, or one
channel of an acceleration time history, and nowhere else; a refusal is
said in the pane with nothing drawn; heavy work goes to the window's
worker with only records and numbers handed to it, and is kept so a
redraw with nothing changed does not count again.
"""

from __future__ import annotations

import numpy as np

from visualdynamics.core import fatigue
from visualdynamics.core.data import Psd, Specification, TimeHistory

G = 9.80665
LEVEL = 0.04 * G ** 2


def _spec(level=LEVEL, low=20.0, high=500.0):
    return Specification([low, high], [[level, level]], response_dof=['1X+'],
                         ordinate_dim=[fatigue.DENSITY])


def _history(seconds=4.0, rate=4096.0, seed=2):
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / rate)
    spectrum[(f < 20.0) | (f > 150.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x *= 2.0 * G / np.sqrt(np.mean(x ** 2))
    return TimeHistory(np.arange(n) / rate, x[None], response_dof=['1X+'],
                       ordinate_dim=['acceleration'], ordinate_unit=['g'])


def _show(window, pump, name):
    window.show_object(name)
    pump()
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
    window.render_current()
    pump()
    return pane


def _plots(pane):
    import pyqtgraph as pg

    return [item for item in pane.graphics.ci.items
            if isinstance(item, pg.PlotItem)]


def _set(panel, pump, **values):
    """Type values into the pane's boxes, each emitting as typed."""
    for key, value in values.items():
        box = {'duration': panel.duration_box,
               'test_duration': panel.test_duration_box,
               'b': panel.b_box, 'q': panel.q_box}[key]
        box.setValue(value)
        pump()


def test_the_reading_is_offered_on_an_acceleration_record_alone(window,
                                                                pump):
    window.add_object('Spec', _spec())
    pane = _show(window, pump, 'Spec')
    assert pane.fatigue_action.isVisible()
    window.add_object('Record', _history())
    assert _show(window, pump, 'Record').fatigue_action.isVisible()
    f = np.arange(0.0, 100.0)
    window.add_object('Bare', Psd(f, np.ones((1, f.size)),
                                  response_dof=['1X+']))
    assert not _show(window, pump, 'Bare').fatigue_action.isVisible(), \
        'a spectrum whose units are not declared cannot be read'
    window.add_object('Velocity', Psd(
        f, np.ones((1, f.size)), response_dof=['1X+'],
        ordinate_dim=['velocity**2/frequency']))
    assert not _show(window, pump, 'Velocity').fatigue_action.isVisible()


def test_a_spectrum_with_no_duration_is_refused_in_the_pane(window, pump):
    window.add_object('Spec', _spec())
    pane = _show(window, pump, 'Spec')
    pane.fatigue_action.trigger()
    pump()
    panel = pane.fatigue_panel
    assert panel.isVisible()
    assert 'duration is not given' in panel.verdict.text()
    assert panel.derived['field'][1].text() == '—'
    assert _plots(pane) == [], 'nothing drawn that could read as a spectrum'


def test_a_duration_typed_draws_the_two_spectra_and_the_rule(window, pump):
    window.add_object('Spec', _spec())
    pane = _show(window, pump, 'Spec')
    pane.fatigue_action.trigger()
    pump()
    panel = pane.fatigue_panel
    _set(panel, pump, duration=600.0, b=6.0)
    assert len(_plots(pane)) == 3
    expected = fatigue.psd_spectra(window.objects['Spec'], b=6.0,
                                   duration=600.0)
    _key, field, pair = window._fatigue_cache
    assert pair is None
    assert np.allclose(field.fds, expected.fds, rtol=1e-12)
    assert panel.derived['field'][1].text() == 'spectrum, 10 min'
    assert 'b = 6' in panel.rests.text()
    assert window.project.journal[-1] == (
        "visualdynamics.plot.plot_fatigue(project['Spec'], 0, b=6.0, "
        "q=10.0, method='narrowband', duration=600.0, path='fatigue.png', "
        'show=False)')
    assert not panel.names['test_duration'].isVisible()


def test_a_candidate_test_is_compared_in_the_pane(window, pump):
    window.add_object('Field', _spec())
    window.add_object('Test', _spec(level=1.5 * LEVEL))
    pane = _show(window, pump, 'Field')
    pane.fatigue_action.trigger()
    pump()
    panel = pane.fatigue_panel
    _set(panel, pump, duration=3600.0)
    index = panel.against_box.findText('Test')
    assert index > 0
    panel.against_box.setCurrentIndex(index)
    pump()
    assert panel.names['test_duration'].isVisible()
    _set(panel, pump, test_duration=360.0)
    pair = window._fatigue_cache[2]
    expected = fatigue.compared(window.objects['Field'],
                                window.objects['Test'],
                                field_duration=3600.0, test_duration=360.0,
                                b=7.5, q=10.0, method='narrowband')
    assert np.allclose(pair.damage_ratio, expected.damage_ratio, rtol=1e-12)
    assert panel.derived['least'][1].text().startswith(
        f'{10 * np.log10(expected.least_ratio):+.2f} dB at')
    assert panel.derived['factor'][1].text() == (
        f'{expected.factor_needed:.4g} on the ASD')
    assert 'falls short' in panel.verdict.text()
    assert "against=project['Test'], against_record=0, " \
           'against_duration=360.0' in window.project.journal[-1]


def test_a_long_record_goes_to_the_worker_once(window, pump, monkeypatch):
    """Over the inline budget the verb runs through `_run_long`, handed
    the record and numbers only; a redraw with nothing changed draws the
    kept result without counting again."""
    window.add_object('Record', _history())
    pane = _show(window, pump, 'Record')
    handed = []
    real = window._run_long

    def spy(status, verb, *args, **kwargs):
        handed.append((verb, args, kwargs))
        return real(status, verb, *args, **kwargs)

    monkeypatch.setattr(window, '_run_long', spy)
    monkeypatch.setattr(type(window), 'FATIGUE_INLINE', 0)
    pane.fatigue_action.trigger()
    pump()
    assert len(handed) == 1
    verb, args, kwargs = handed[0]
    assert verb.__module__ == 'visualdynamics.core.fatigue'
    assert all(isinstance(a, (TimeHistory, int)) for a in args)
    assert all(isinstance(v, (float, int, type(None)))
               for v in kwargs.values())
    assert len(_plots(pane)) == 3
    assert pane.fatigue_panel.derived['field'][1].text() == \
        'time history, 4 s'
    window.render_current()
    pump()
    assert len(handed) == 1, 'nothing changed, nothing counted again'
    _set(pane.fatigue_panel, pump, q=20.0)
    assert len(handed) == 2


def test_settings_are_kept_per_record_and_the_pane_follows(window, pump):
    window.add_object('One', _spec())
    window.add_object('Two', _spec(level=3 * LEVEL))
    pane = _show(window, pump, 'One')
    pane.fatigue_action.trigger()
    pump()
    _set(pane.fatigue_panel, pump, duration=60.0)
    _show(window, pump, 'Two')
    assert pane.fatigue_panel.duration_box.value() == 0.0
    assert 'duration is not given' in pane.fatigue_panel.verdict.text()
    _show(window, pump, 'One')
    assert pane.fatigue_panel.duration_box.value() == 60.0
    assert window._fatigue_cache[1].label == window.objects['One'] \
        .record_label(0)
    window.add_object('Bare', Psd(np.arange(10.0), np.ones((1, 10)),
                                  response_dof=['1X+']))
    _show(window, pump, 'Bare')
    assert not pane.fatigue_panel.isVisible()
