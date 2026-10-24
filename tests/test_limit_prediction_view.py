"""The notch prediction in the window and on its own: a transmissibility
from a control channel selected with the specification written there
offers a Notch Prediction toggle, which draws the control with its
predicted notch hatched and each response channel against its limit
profile, with the verdict, the notches in a table and the reading in
words in the panel beside it.

A reading of objects that exist, so a toggle with no act, offered for a
transmissibility from the channel a selected specification is written
at, and for nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

from visualdynamics.core.data import Frf, Psd, Specification

G = 9.80665
F = np.arange(0.0, 2001.0, 1.0)


def _frf(gain=4.0, dim='acceleration/acceleration', reference='1Z+'):
    t = np.ones(len(F), dtype=complex)
    t[(F >= 240) & (F <= 260)] = gain
    return Frf(F, t[None], response_dof=['2Z+'], reference_dof=[reference],
               ordinate_dim=dim)


def _spec(profile=0.18, control='1Z+'):
    level = 0.04 * G ** 2
    values = np.array([[level, level], [profile * G ** 2] * 2])
    nan = np.full(2, np.nan)
    return Specification(
        np.array([20.0, 2000.0]), values, response_dof=[control, '2Z+'],
        ordinate_dim='acceleration**2/frequency',
        abort_lower=np.array([values[0] / 2, nan]),
        abort_upper=np.array([values[0] * 2, nan]))


def _selected(window, pump, frf=None, spec=None, profiles=None):
    window.add_object('Low level H1', _frf() if frf is None else frf)
    window.add_object('Profile', _spec() if spec is None else spec)
    names = ['Low level H1', 'Profile']
    if profiles is not None:
        window.add_object('Limits', profiles)
        names.append('Limits')
    select_objects(window, pump, *names)
    window.render_current()
    pump()
    return window.data_pane


def _predicted(window, pump, **objects):
    pane = _selected(window, pump, **objects)
    pane.notch_action.trigger()
    pump()
    return pane


def _plots(pane):
    import pyqtgraph as pg

    return [item for item in pane.graphics.ci.items
            if isinstance(item, pg.PlotItem)]


VERDICT = ('1 notch in the control at full level, the deepest 5.5 dB at '
           '240 Hz for 2Z+; it takes the control outside the abort '
           'tolerance')


def test_it_is_offered_for_a_transmissibility_beside_its_specification(
        window, pump):
    pane = _selected(window, pump)
    assert pane.notch_action.isVisible()
    assert not pane.notch_action.isChecked()
    # the FRF alone has nothing to predict against
    window.show_object('Low level H1')
    pump()
    window.render_current()
    pump()
    assert not pane.notch_action.isVisible()


def test_nothing_selected_offers_nothing(window):
    """The rule the button follows, read on the selections that are not
    a prediction: none, the FRF alone, two FRFs."""
    frf, spec = _frf(), _spec()
    assert window._notch_source([]) is None
    assert window._notch_source([('H1', frf, None)]) is None
    assert window._notch_source([('H1', frf, None),
                                 ('Other', frf, None)]) is None
    assert window._notch_source([('H1', frf, None), ('Profile', spec, None)]
                                ) == ('H1', frf, 'Profile', spec, None, None)


def test_an_frf_from_the_drives_is_not_a_transmissibility_from_control(
        window, pump):
    """An FRF against volts is the drive prediction's input, not this
    one's: not offered."""
    pane = _selected(window, pump, frf=_frf(dim='acceleration/voltage'))
    assert not pane.notch_action.isVisible()


def test_a_specification_elsewhere_or_a_plain_psd_is_not_offered(window,
                                                                 pump):
    pane = _selected(window, pump, spec=_spec(control='7Z+'))
    assert not pane.notch_action.isVisible()


def test_a_measured_psd_alone_is_not_the_specification(window, pump):
    spec = _spec()
    plain = Psd(spec.abscissa, spec.ordinate, response_dof=spec.response_dof,
                ordinate_dim=spec.ordinate_dim)
    pane = _selected(window, pump, spec=plain)
    assert not pane.notch_action.isVisible()


def test_the_toggle_draws_the_control_and_each_channel_with_the_verdict(
        window, pump):
    pane = _predicted(window, pump)
    panel = pane.notch_panel
    assert panel.isVisible()
    control, channel = _plots(pane)
    assert control.titleLabel.text == VERDICT
    assert channel.titleLabel.text == ('2Z+: 5.5 dB over its limit profile '
                                       'at 240 Hz, from 240 to 260 Hz')
    assert panel.verdict.text() == VERDICT[0].upper() + VERDICT[1:] + '.'
    assert 'font-weight: 600' in panel.verdict.styleSheet()
    assert window.statusBar().currentMessage() == VERDICT
    assert panel.table.rowCount() == 1
    assert [panel.table.item(0, c).text() for c in range(3)] == [
        '240–260', '5.5 (0.28)', 'outside']
    assert ('Notch from 240 to 260 Hz (21 lines): 5.5 dB deep at 240 Hz'
            in panel.table.item(0, 0).toolTip())
    assert 'the control left at 0.01125 g²/Hz' in panel.readout.text()
    # the notch is hatched and labeled in words on the control row
    import pyqtgraph as pg
    words = [item.textItem.toPlainText()
             for item in control.items
             if isinstance(item, pg.TextItem)]
    assert words == ['notch 5.5 dB, 2Z+']
    regions = [item for item in control.items
               if isinstance(item, pg.LinearRegionItem)]
    assert len(regions) == 1
    # put away, the panel goes with it
    pane.notch_action.trigger()
    pump()
    assert not panel.isVisible()


def test_the_tolerance_chosen_in_the_panel_is_read_at_once(window, pump):
    spec = _spec()
    spec.limits['warning_lower'] = spec.limits['abort_lower'] * 0.25
    spec.limits['warning_upper'] = spec.limits['abort_upper']
    pane = _predicted(window, pump, spec=spec)
    panel = pane.notch_panel
    panel.tolerance_box.setCurrentIndex(1)
    pump()
    assert pane.notch_settings == {'tolerance': 'warning'}
    assert panel.verdict.text().endswith(
        'the control stays inside the warning tolerance.')
    assert panel.table.item(0, 2).text() == 'inside'


def test_profiles_selected_as_a_third_object_are_read(window, pump):
    spec = _spec()
    control_only = Specification(
        spec.abscissa, spec.ordinate[:1], response_dof=['1Z+'],
        ordinate_dim='acceleration**2/frequency',
        abort_lower=spec.limits['abort_lower'][:1],
        abort_upper=spec.limits['abort_upper'][:1])
    profiles = Specification(spec.abscissa, spec.ordinate[1:] * 4.0,
                             response_dof=['2Z+'],
                             ordinate_dim='acceleration**2/frequency')
    pane = _predicted(window, pump, spec=control_only, profiles=profiles)
    # a profile four times higher: 0.64 against 0.72, no notch
    assert pane.notch_panel.verdict.text().startswith(
        'No notch: the limit channel stays under its limit profile')
    assert pane.notch_panel.table.rowCount() == 0
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_limit_prediction(')]
    assert lines == [(
        "visualdynamics.plot.plot_limit_prediction(project['Low level H1'], "
        "project['Profile'], project['Limits'], "
        "path='limit_prediction.png', show=False)")]


def test_a_refusal_is_said_in_the_panel_and_draws_nothing(window, pump):
    spec = _spec()
    bare = Specification(spec.abscissa, spec.ordinate[:1],
                         response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency')
    pane = _predicted(window, pump, spec=bare)
    assert pane.notch_panel.verdict.text().startswith(
        'Not predicted: no response channel the transmissibilities reach '
        '(2Z+) has a limit profile in the specification')
    assert _plots(pane) == []
    assert window.statusBar().currentMessage().startswith(
        'No notch prediction: no response channel')


def test_a_dead_channel_reads_as_nothing_judged(window, pump):
    dead = Frf(F, np.zeros((1, len(F)), dtype=complex), response_dof=['2Z+'],
               reference_dof=['1Z+'],
               ordinate_dim='acceleration/acceleration')
    pane = _predicted(window, pump, frf=dead)
    assert pane.notch_panel.verdict.text().startswith('Nothing judged:')
    assert 'zero at every predicted line' in pane.notch_panel.readout.text()


def test_the_prediction_is_journaled_as_its_plot_call(window, pump):
    _predicted(window, pump)
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_limit_prediction(')]
    assert lines == [(
        "visualdynamics.plot.plot_limit_prediction(project['Low level H1'], "
        "project['Profile'], path='limit_prediction.png', show=False)")]


def test_the_choice_survives_clicking_away_and_back(window, pump):
    spec = _spec()
    spec.limits['warning_lower'] = spec.limits['abort_lower']
    pane = _predicted(window, pump, spec=spec)
    pane.notch_panel.tolerance_box.setCurrentIndex(1)
    pump()
    window.show_object('Low level H1')
    pump()
    select_objects(window, pump, 'Low level H1', 'Profile')
    window.render_current()
    pump()
    assert pane.notch_action.isChecked()
    assert pane.notch_panel.tolerance_box.currentIndex() == 1


def test_the_standalone_plot_writes_the_same_rows(tmp_path):
    from visualdynamics.plot import plot_limit_prediction

    path = plot_limit_prediction(_frf(), _spec(), show=False,
                                 path=str(tmp_path / 'notch.png'))
    assert (tmp_path / 'notch.png').stat().st_size > 0
    assert path.endswith('notch.png')
    with pytest.raises(ValueError, match='no autospectrum at 1Z\\+'):
        plot_limit_prediction(_frf(), _spec(control='7Z+'), show=False,
                              path=str(tmp_path / 'never.png'))


def test_the_plot_names_every_curve_and_draws_the_tolerance(qt_app):
    import pyqtgraph as pg

    from visualdynamics.core.limit_prediction import predict_limits
    from visualdynamics.plot.limit_prediction import build_limit_prediction

    layout = pg.GraphicsLayoutWidget()
    control, channel = build_limit_prediction(
        layout, predict_limits(_frf(), _spec()))
    names = [label.text for _sample, label in control.legend.items]
    assert names == ['reference at 1Z+', 'control, predicted',
                     'abort tolerance, lower edge']
    assert [label.text for _s, label in channel.legend.items] == [
        '2Z+ predicted at full level', 'limit profile']
    assert channel.getAxis('bottom').labelText == 'frequency [Hz]'
