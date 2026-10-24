"""The drive prediction in the window: an FRF and a specification
selected together offer a Drive Prediction toggle, which draws the
drives, one channel against its target and every quantity against its
rating, with the settings and the verdict in the panel beside it.

A reading of two objects that exist, so a toggle with no act, offered
for exactly one FRF and one specification and for nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

from visualdynamics.core.data import Frf, Psd, Specification

G = 9.80665


def _frf(gain=2.0, dof='1Z+'):
    f = np.arange(0.0, 2001.0, 5.0)
    return Frf(f, np.full((1, len(f)), gain * G, dtype=complex),
               response_dof=dof, reference_dof='1Z+',
               ordinate_dim='acceleration/voltage')


def _spec(dof='1Z+', cls=Specification):
    return cls(np.array([20.0, 2000.0]), np.array([[0.04, 0.04]]) * G ** 2,
               response_dof=[dof], ordinate_dim='acceleration**2/frequency')


def _selected(window, pump, frf=None, spec=None):
    window.add_object('Plant', _frf() if frf is None else frf)
    window.add_object('Profile', _spec() if spec is None else spec)
    select_objects(window, pump, 'Plant', 'Profile')
    window.render_current()
    pump()
    return window.data_pane


def _predicted(window, pump, **objects):
    pane = _selected(window, pump, **objects)
    pane.prediction_action.trigger()
    pump()
    return pane


def _plots(pane):
    import pyqtgraph as pg

    return [item for item in pane.graphics.ci.items
            if isinstance(item, pg.PlotItem)]


def test_it_is_offered_for_an_frf_beside_a_specification_only(window, pump):
    pane = _selected(window, pump)
    assert pane.prediction_action.isVisible()
    assert not pane.prediction_action.isChecked()
    # the FRF alone has nothing to predict for
    window.show_object('Plant')
    pump()
    window.render_current()
    pump()
    assert not pane.prediction_action.isVisible()


def test_a_measured_psd_is_not_a_specification_and_is_not_offered(window,
                                                                  pump):
    pane = _selected(window, pump, spec=_spec(cls=Psd))
    assert not pane.prediction_action.isVisible()


def test_the_toggle_draws_three_rows_and_the_verdict(window, pump):
    pane = _predicted(window, pump)
    assert pane.prediction_panel.isVisible()
    drives, middle, _shares = _plots(pane)
    assert drives.titleLabel.text.startswith('drive at 1Z+')
    assert middle.titleLabel.text == ('1Z+: the predicted response meets '
                                      'the target within 1 dB')
    assert pane.prediction_panel.verdict.text() == (
        '1 drive for 1 control channel: no ratings given.')
    assert window.statusBar().currentMessage() == (
        '1 drive for 1 control channel: no ratings given')
    # put away, the panel goes with it
    pane.prediction_action.trigger()
    pump()
    assert not pane.prediction_panel.isVisible()


def test_a_limit_typed_in_the_panel_is_read_against_at_once(window, pump):
    """10 V against a drive of 4.46 V rms: about 21 V expected over a
    minute, over its limit, said in the verdict and kept on the pane in
    SI. A moving mass typed in the window's units is kept in SI too."""
    pane = _predicted(window, pump)
    panel = pane.prediction_panel
    panel.editors['drive_limit'].setValue(10.0)
    pump()
    assert pane.prediction_settings['drive_limit'] == pytest.approx(10.0)
    assert panel.verdict.text().startswith(
        '1 drive for 1 control channel: the drive at 1Z+ runs out first')
    assert 'font-weight: 600' in panel.verdict.styleSheet()
    assert 'against 10 V' in panel.readout.text()
    window._units_changed('m-kg-N-s')
    pump()
    assert panel.names['moving_mass'].text() == 'Moving mass [kg]'
    panel.editors['moving_mass'].setValue(50.0)
    pump()
    assert pane.prediction_settings['moving_mass'] == pytest.approx(50.0)
    assert 'Force at 1Z+:' in panel.readout.text()


def test_the_inverse_shows_only_the_setting_it_uses(window, pump):
    pane = _predicted(window, pump)
    panel = pane.prediction_panel
    assert panel.editors['tolerance'].isVisibleTo(panel)
    assert not panel.editors['weight'].isVisibleTo(panel)
    panel.method_box.setCurrentIndex(1)
    pump()
    assert pane.prediction_settings['method'] == 'tikhonov'
    assert panel.editors['weight'].isVisibleTo(panel)
    assert not panel.editors['tolerance'].isVisibleTo(panel)
    # the default weight on a flat plant stays within 1 dB of the
    # target; a weight of a half does not
    _drives, middle, _shares = _plots(pane)
    assert 'meets the target' in middle.titleLabel.text
    panel.editors['weight'].setValue(0.5)
    pump()
    _drives, middle, _shares = _plots(pane)
    assert 'misses the target by more than 1 dB' in middle.titleLabel.text


def test_a_refusal_is_said_in_the_panel_and_draws_nothing(window, pump):
    pane = _predicted(window, pump, spec=_spec(dof='5Z+'))
    assert pane.prediction_panel.verdict.text() == (
        'Not predicted: the specification asks for 5Z+, and the FRF has '
        'no response there.')
    assert _plots(pane) == []
    assert window.statusBar().currentMessage().startswith(
        'No drive prediction: the specification asks for 5Z+')


def test_the_prediction_is_journaled_as_its_plot_call(window, pump):
    pane = _predicted(window, pump)
    pane.prediction_panel.editors['drive_limit'].setValue(10.0)
    pump()
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_drive_prediction(')]
    assert lines == [(
        "visualdynamics.plot.plot_drive_prediction(project['Plant'], "
        "project['Profile'], method='truncated', tolerance=0.01, "
        "weight=0.001, duration=60.0, drive_limit=10.0, "
        "path='drive_prediction.png', show=False)")]


def test_the_settings_survive_clicking_away_and_back(window, pump):
    pane = _predicted(window, pump)
    pane.prediction_panel.editors['drive_limit'].setValue(7.5)
    pump()
    window.show_object('Plant')
    pump()
    select_objects(window, pump, 'Plant', 'Profile')
    window.render_current()
    pump()
    assert pane.prediction_action.isChecked()
    assert pane.prediction_panel.editors['drive_limit'].value() == \
        pytest.approx(7.5)
