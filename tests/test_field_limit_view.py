"""The field limit in the window: a toggle on a plain PSD's or SRS's bar,
the set drawn with its limit and envelope, the settings and the reading
in the pane beside it, and Make Specification writing what it shows.
"""

from __future__ import annotations

import threading

import numpy as np

from visualdynamics.core import field_limit as fl
from visualdynamics.core.data import Psd, ShockSpecification, Specification, Srs

G = 9.80665
DENSITY = 'acceleration**2/frequency'


def _flights(count=6, seed=12, lines=43):
    """`count` flights on sixth-octave lines from 20 to 2000 Hz: a
    profile with 2 dB of log-normal scatter, in m/s^2 squared per Hz."""
    rng = np.random.default_rng(seed)
    f = np.geomspace(20.0, 2000.0, lines)
    base = 0.04 * G * G * np.minimum(1.0, f / 150.0) * np.minimum(
        1.0, (800.0 / f) ** 2)
    levels = base * 10.0 ** (rng.normal(0.0, 2.0, (count, f.size)) / 10.0)
    return Psd(f, levels, '101Z+', ordinate_dim=DENSITY, ordinate_unit='g')


def _show(window, pump, name):
    window.show_object(name)
    pump()
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
    window.render_current()
    pump()
    return pane


def _reading(window, pump, data=None, name='Flights'):
    window.add_object(name, _flights() if data is None else data)
    pane = _show(window, pump, name)
    pane.field_limit_action.trigger()
    pump()
    return pane


def test_offered_on_plain_spectra_and_nowhere_else(window, pump):
    window.add_object('Flights', _flights())
    pane = _show(window, pump, 'Flights')
    assert pane.field_limit_action.isVisible()
    flights = _flights()
    spec = Specification(flights.abscissa, flights.ordinate[:1], '101Z+',
                         ordinate_dim=DENSITY)
    window.add_object('Spec', spec)
    _show(window, pump, 'Spec')
    assert not pane.field_limit_action.isVisible(), \
        'a specification is a requirement, not a measurement'
    cross = Psd(flights.abscissa, flights.ordinate[:1] * (1 + 1j), '101Z+',
                reference_dof='102Z+', ordinate_dim=DENSITY)
    window.add_object('Cross', cross)
    _show(window, pump, 'Cross')
    assert not pane.field_limit_action.isVisible()
    srs = Srs(flights.abscissa, np.sqrt(flights.ordinate), '101Z+',
              ordinate_dim='acceleration')
    window.add_object('Shocks', srs)
    _show(window, pump, 'Shocks')
    assert pane.field_limit_action.isVisible()


def test_the_reading_draws_the_fit_the_project_gives(window, pump):
    pane = _reading(window, pump)
    panel = pane.field_limit_panel
    assert panel.isVisible()
    state = window._field_limit
    assert state is not None and state['sources'] == 'Flights'
    fit = state['fit']
    again = window.project.field_limit('Flights')
    assert np.array_equal(fit.levels, again.levels)
    assert fit.limit.statistic == 'tolerance'
    derived = panel.derived
    assert derived['records'].text() == '6'
    assert derived['factor'].text() == f'{fit.limit.factor:.3f}'
    assert derived['breakpoints'].text().startswith(str(fit.breakpoints))
    assert derived['limit_rms'].text() == f'{fit.limit_rms / G:.4g} g'
    assert derived['added'].text() == f'{fit.added_db:+.2f} dB'
    assert 'log-normal' in panel.rests_on.text()
    assert panel.make_button.isEnabled()
    assert window.statusBar().currentMessage().startswith(
        'normal tolerance limit P95/50')
    plot = next(item for item in pane.graphics.ci.items
                if hasattr(item, 'addItem'))
    assert 'normal tolerance limit' in plot.titleLabel.text


def test_a_setting_redraws_and_rows_follow_the_statistic(window, pump):
    pane = _reading(window, pump)
    panel = pane.field_limit_panel
    assert panel.editors['confidence'].isVisibleTo(panel)
    assert not panel.editors['k'].isVisibleTo(panel)
    panel.statistic_box.setCurrentIndex(panel.statistic_box.findData('mean_k'))
    pump()
    assert window._field_limit['fit'].limit.statistic == 'mean_k'
    assert panel.editors['k'].isVisibleTo(panel)
    assert not panel.editors['confidence'].isVisibleTo(panel)
    panel.k_box.setValue(1.0)
    pump()
    assert window._field_limit['fit'].limit.factor == 1.0
    assert panel.derived['factor'].text() == '1.000'


def test_a_refusal_is_said_and_nothing_can_be_made(window, pump):
    pane = _reading(window, pump)
    panel = pane.field_limit_panel
    panel.statistic_box.setCurrentIndex(
        panel.statistic_box.findData('distribution_free'))
    pump()
    assert window._field_limit is None
    assert 'needs at least 14 records' in panel.rests_on.text()
    assert not panel.make_button.isEnabled()
    assert window.statusBar().currentMessage().startswith('No field limit:')
    assert all(label.text() == '—' for label in panel.derived.values())


def test_a_nan_in_a_flight_is_named(window, pump):
    flights = _flights()
    flights.ordinate[3, 10] = np.nan
    pane = _reading(window, pump, flights)
    text = pane.field_limit_panel.rests_on.text()
    assert 'record 4' in text and 'NaN' in text
    assert window._field_limit is None


def test_make_specification_writes_what_the_reading_shows(window, pump):
    pane = _reading(window, pump)
    panel = pane.field_limit_panel
    panel.margin_box.setValue(3.0)
    pump()
    fit = window._field_limit['fit']
    panel.make_button.click()
    pump()
    added = 'Flights Derived Specification'
    spec = window.objects[added]
    assert isinstance(spec, Specification)
    assert np.allclose(spec.abscissa, fit.frequencies)
    assert np.allclose(np.real(spec.ordinate[0]),
                       fit.levels * 10 ** (3.0 / 10))
    assert window.project.provenance[added]['verb'] == 'derive_specification'
    assert any(line.startswith("project.derive_specification('Flights'")
               and 'margin_db=3.0' in line
               for line in window.project.journal)
    assert not pane.field_limit_action.isChecked(), 'apply ends the reading'
    assert window.statusBar().currentMessage().startswith(added)


def test_an_srs_set_makes_a_shock_specification(window, pump):
    f = np.geomspace(10.0, 2000.0, 12 * 7 + 1)
    rng = np.random.default_rng(5)
    level = np.minimum(f, 100.0) * G
    srs = Srs(f, level * 10 ** (rng.normal(0, 1.0, (5, f.size)) / 20),
              '101Z+', ordinate_dim='acceleration', q=10.0)
    pane = _reading(window, pump, srs, 'Shocks')
    panel = pane.field_limit_panel
    assert panel.derived['limit_rms'].text() == '—', 'an SRS has no area'
    panel.make_button.click()
    pump()
    spec = window.objects['Shocks Derived Specification']
    assert isinstance(spec, ShockSpecification) and spec.q == 10.0


def test_the_reading_is_journaled_as_its_plot_call(window, pump):
    pane = _reading(window, pump)

    def lines():
        return [line for line in window.project.journal
                if line.startswith('visualdynamics.plot.plot_field_limit(')]

    assert lines() == [(
        "visualdynamics.plot.plot_field_limit(project['Flights'], "
        "statistic='tolerance', coverage=0.95, max_overshoot_db=3.0, "
        "max_breakpoints=12, rms='exceed', confidence=0.5, margin_db=0.0, "
        "path='field_limit.png', show=False)")]
    pane.field_limit_panel.margin_box.setValue(2.0)
    pump()
    assert len(lines()) == 1 and 'margin_db=2.0,' in lines()[0], \
        'settled in place, not added to'


def test_a_large_set_is_worked_out_on_the_worker(window, pump, monkeypatch):
    seen = []
    derive = fl.derive

    def watched(*args, **kwargs):
        seen.append(threading.current_thread() is threading.main_thread())
        return derive(*args, **kwargs)

    monkeypatch.setattr(fl, 'derive', watched)
    monkeypatch.setattr(type(window), 'FIELD_INLINE', 0)
    _reading(window, pump)
    assert seen and not any(seen), 'off the GUI thread'
    assert window._field_limit is not None
    seen.clear()
    monkeypatch.setattr(type(window), 'FIELD_INLINE', 10 ** 9)
    window.render_current()
    pump()
    assert seen == [True], 'a small set is read in place'


def test_putting_the_reading_away_hides_the_panel(window, pump):
    pane = _reading(window, pump)
    pane.field_limit_action.trigger()
    pump()
    assert not pane.field_limit_panel.isVisible()
    assert window._field_limit is None


def test_the_journaled_line_draws_the_reading_headless(window, pump,
                                                       tmp_path):
    import visualdynamics

    _reading(window, pump)
    line = next(line for line in window.project.journal
                if line.startswith('visualdynamics.plot.plot_field_limit('))
    target = tmp_path / 'replayed.png'
    room = {'visualdynamics': visualdynamics, 'project': window.project}
    exec(line.replace("'field_limit.png'", repr(str(target))), room)  # noqa: S102
    assert target.stat().st_size > 0
