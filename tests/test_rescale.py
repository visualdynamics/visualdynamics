"""Rescaling a time history's records: the correction for data recorded
with the wrong sensitivity, or any other factor.

Brandon, 2026-11-12: data recorded with the wrong sensitivity needed
rescaling, and there was no way to do it. A button rather than a
rescale on editing the channel table's sensitivity, because an edited
sensitivity can mean "the data was taken with this" or "the next test
will use this", and only the first should touch the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.rescale import rescaled, sensitivity_factor


def record(channels=3, samples=1024, rate=1024.0):
    t = np.arange(samples) / rate
    rows = np.vstack([(k + 1) * np.sin(2 * np.pi * (10 + 5 * k) * t)
                      for k in range(channels)])
    history = TimeHistory(t, rows,
                          response_dof=[f'{k + 1}Z+' for k in range(channels)],
                          ordinate_dim='acceleration')
    return history


def test_the_factor_is_the_recorded_sensitivity_over_the_correct_one():
    """Volts over the sensitivity: data taken with 100 mV/g from a
    10 mV/g sensor reads a tenth of the true level."""
    assert sensitivity_factor(100.0, 10.0) == pytest.approx(10.0)
    assert sensitivity_factor(9.5, 10.0) == pytest.approx(0.95)
    with pytest.raises(ValueError, match='correct sensitivity'):
        sensitivity_factor(100.0, 0.0)
    with pytest.raises(ValueError, match='recorded sensitivity'):
        sensitivity_factor(float('nan'), 10.0)


def test_only_the_records_picked_are_multiplied():
    history = record()
    made = rescaled(history, 2.5, [2, 0])
    np.testing.assert_array_equal(made.ordinate[0], 2.5 * history.ordinate[0])
    np.testing.assert_array_equal(made.ordinate[1], history.ordinate[1])
    np.testing.assert_array_equal(made.ordinate[2], 2.5 * history.ordinate[2])
    np.testing.assert_array_equal(made.abscissa, history.abscissa)
    assert made.response_dof == history.response_dof
    assert list(made.ordinate_dim) == list(history.ordinate_dim)
    whole = history.rescale(-1.0)
    np.testing.assert_array_equal(whole.ordinate, -history.ordinate)


def test_the_source_is_left_as_it_was():
    history = record()
    before = history.ordinate.copy()
    rescaled(history, 3.0)
    np.testing.assert_array_equal(history.ordinate, before)


def test_what_is_set_on_the_record_is_kept():
    from visualdynamics.core.averaging import Averaging

    history = record()
    history.averaging = Averaging(frame_length=256)
    history.shocks = ((0.1, 0.2),)
    history.truncation = (0.1, 0.9)
    made = rescaled(history, 2.0, [1])
    assert made.averaging == history.averaging
    assert made.shocks == history.shocks
    assert made.truncation == history.truncation


@pytest.mark.parametrize('factor', [0.0, float('inf'), float('nan')])
def test_a_factor_that_would_lose_the_data_is_refused(factor):
    with pytest.raises(ValueError, match='finite number other than zero'):
        rescaled(record(), factor)


def test_records_the_history_does_not_have_are_refused():
    with pytest.raises(ValueError, match='record 3 is not in a history of 3'):
        rescaled(record(), 2.0, [0, 3])
    with pytest.raises(ValueError, match='no records'):
        rescaled(record(), 2.0, [])


# ---- the verb ---------------------------------------------------------------


def test_the_project_rescales_and_records_how():
    import visualdynamics

    project = visualdynamics.Project()
    project.add('Run', record())
    made = project.rescale('Run', recorded_sensitivity=100.0,
                           correct_sensitivity=10.0, records=[1])
    assert made == 'Run Rescaled'
    np.testing.assert_allclose(project[made].ordinate[1],
                               10.0 * project['Run'].ordinate[1])
    np.testing.assert_array_equal(project[made].ordinate[0],
                                  project['Run'].ordinate[0])
    assert project.provenance[made]['verb'] == 'rescale'
    assert project.provenance[made]['params'] == {'factor': 10.0,
                                                  'records': [1]}
    assert made in project.object_group_of('Run')
    assert project.journal[-1] == (
        "project.rescale('Run', recorded_sensitivity=100.0, "
        "correct_sensitivity=10.0, records=[1])")


def test_the_verb_takes_a_factor_or_two_sensitivities_not_both():
    import visualdynamics

    project = visualdynamics.Project()
    project.add('Run', record())
    with pytest.raises(ValueError, match='a factor or two sensitivities'):
        project.rescale('Run')
    with pytest.raises(ValueError, match='a factor or two sensitivities'):
        project.rescale('Run', 2.0, recorded_sensitivity=100.0,
                        correct_sensitivity=10.0)
    with pytest.raises(ValueError, match='a factor or two sensitivities'):
        project.rescale('Run', recorded_sensitivity=100.0)
    assert list(project) == ['Run'], 'nothing added'


def test_a_refresh_rescales_the_source_again():
    """The recipe holds the factor and the records, so a source changed
    upstream gives the same correction again."""
    import visualdynamics

    project = visualdynamics.Project()
    project.add('Run', record())
    made = project.rescale('Run', 4.0, records=[0])
    project['Run'].ordinate[0] *= 2.0
    project.refresh(made)
    np.testing.assert_allclose(project[made].ordinate[0],
                               4.0 * project['Run'].ordinate[0])
    np.testing.assert_array_equal(project[made].ordinate[1],
                                  project['Run'].ordinate[1]), 'still only record 0'


# ---- the window ---------------------------------------------------------------


def _rescale_reading(window, pump, history=None):
    name = window.add_object('Run', history or record())
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    pane = window.data_pane
    assert pane.rescale_action.isVisible()
    pane.rescale_action.trigger()
    pump()
    return pane


def _curves(layout):
    import pyqtgraph as pg

    return [item for item in layout.ci.items
            if isinstance(item, pg.PlotItem)
            for item in item.listDataItems()]


def test_a_time_history_offers_the_rescale_reading(window, pump):
    pane = _rescale_reading(window, pump)
    panel = pane.rescale_panel
    assert pane.rescale_action.text() == 'Rescale'
    assert panel.isVisible()
    assert panel.values['factor'].text() == '1'
    assert panel.values['records'].text() == '3 of 3'
    assert len(_curves(pane.graphics)) == 6, 'each record recorded and rescaled'


def test_the_way_shows_only_the_rows_it_reads(window, pump):
    pane = _rescale_reading(window, pump)
    panel = pane.rescale_panel
    assert panel.factor_box.isVisibleTo(panel)
    assert not panel.recorded_box.isVisibleTo(panel)
    panel.way_box.setCurrentIndex(panel.way_box.findData('sensitivity'))
    pump()
    assert not panel.factor_box.isVisibleTo(panel)
    assert panel.recorded_box.isVisibleTo(panel)
    assert panel.correct_box.isVisibleTo(panel)
    panel.recorded_box.setValue(100.0)
    panel.correct_box.setValue(10.0)
    pump()
    assert panel.values['factor'].text() == '10'
    assert pane.rescale_settings['way'] == 'sensitivity'


def test_the_preview_draws_the_rescaled_records_over_the_recorded(window, pump):
    pane = _rescale_reading(window, pump)
    panel = pane.rescale_panel
    panel.factor_box.setValue(3.0)
    pump()
    peaks = sorted(float(np.max(np.abs(curve.yData)))
                   for curve in _curves(pane.graphics))
    # drawn in the display unit, as every trace is
    recorded = sorted(float(window.unit_system.from_si(np.max(np.abs(row)),
                                                       'acceleration'))
                      for row in window.objects['Run'].ordinate)
    np.testing.assert_allclose(
        peaks, sorted(recorded + [3.0 * value for value in recorded]),
        rtol=1e-3)
    assert panel.values['before'].text() == '0.3059 g', 'the largest, 3 m/s²'
    assert panel.values['after'].text() == '0.9177 g'
    assert window._status_text == '3 of 3 records \N{MULTIPLICATION SIGN} 3'


def test_apply_rescale_scales_the_picked_records(window, pump):
    pane = _rescale_reading(window, pump)
    window._select_records('Run', [2])
    window.render_current()
    pump()
    assert pane.rescale_panel.values['records'].text() == '1 of 3'
    pane.rescale_panel.factor_box.setValue(0.5)
    pump()
    pane.rescale_panel.apply_button.click()
    pump()
    made = window.project['Run Rescaled']
    source = window.project['Run']
    np.testing.assert_allclose(made.ordinate[2], 0.5 * source.ordinate[2])
    np.testing.assert_array_equal(made.ordinate[:2], source.ordinate[:2])
    assert window.project.journal[-1] == (
        "project.rescale('Run', records=[2], factor=0.5)")
