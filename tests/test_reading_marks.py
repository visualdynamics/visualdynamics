"""The readings the plot bar's toggles draw — averaging frames, shock
windows, the cut, the filter's preview, the octave steps — from a
script, flat and on the 3-D stage (Principle 4: anything the interface
can do the API can do; found missing 2026-10-09).

Each is read from the object as the window reads it, through the same
drawing code: `plot.octave_preview`, `viz.marks.octave_steps`,
`stage_filter_preview` and `add_stage_reading` are what the window's
own previews call.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.viz import marks as stage_marks


@pytest.fixture(scope='module')
def run():
    return visualdynamics.random_vibration_run(fixture_path('plate',
                                                            'random.nc4'))


@pytest.fixture(scope='module')
def psd():
    return visualdynamics.import_file(fixture_path('plate', 'psd.npz'))


@pytest.mark.parametrize('reading', ['averaging', 'shocks', 'truncation',
                                     'filter'])
def test_a_history_s_readings_draw_flat(qt_app, run, tmp_path, reading):
    path = tmp_path / f'{reading}.png'
    visualdynamics.plot.plot_data(run.time_history, marks=reading,
                                  path=str(path), show=False)
    assert path.stat().st_size > 2000


def test_the_octave_steps_draw_flat_one_per_record(qt_app, psd):
    import pyqtgraph as pg

    from visualdynamics.plot import build_plot, octave_preview
    from visualdynamics.theme import theme

    layout = pg.GraphicsLayoutWidget()
    build_plot(layout, psd)
    steps = stage_marks.octave_steps(psd, 3)
    drawn = octave_preview(layout.ci.items, steps, theme('light'))
    assert len(drawn) == psd.num_records
    drawn = octave_preview(layout.ci.items, steps, theme('light'),
                           records=[1])
    assert len(drawn) == 1, 'the steps say what the plot says'
    _banded, _x, rows = steps
    # a log plot hands back the data as it maps it
    assert np.allclose(drawn[0][1].getData()[1], np.log10(rows[1]))


def test_a_reading_on_the_wrong_object_is_refused(qt_app, run, psd):
    with pytest.raises(TypeError, match='octave bands read a PSD'):
        visualdynamics.plot.plot_data(run.time_history, marks='octave',
                                      path='never.png', show=False)
    with pytest.raises(TypeError, match='belong on a time history'):
        visualdynamics.plot.plot_data(psd, marks='filter',
                                      path='never.png', show=False)
    with pytest.raises(ValueError, match='not a reading'):
        visualdynamics.plot.plot_data(psd, marks='frames',
                                      path='never.png', show=False)


@pytest.mark.parametrize('reading, primitive', [
    ('averaging', 'add_averaging_marks'),
    ('shocks', 'add_shock_marks'),
    ('truncation', 'add_truncation_marks'),
    ('filter', 'add_filter_preview')])
def test_a_history_s_readings_go_on_the_stage(run, tmp_path, monkeypatch,
                                              reading, primitive):
    """Each reaches the stage's own drawing function, with what the
    history carries or would suggest — the window's numbers."""
    called = []
    real = getattr(stage_marks, primitive)
    monkeypatch.setattr(stage_marks, primitive,
                        lambda *a, **k: (called.append(a), real(*a, **k))[1])
    history = run.time_history
    path = tmp_path / f'{reading}-3d.png'
    visualdynamics.viz.waterfall.plot_waterfall(
        history, [0, 1], marks=reading, screenshot=str(path))
    assert path.stat().st_size > 2000
    assert len(called) == 1, f'{primitive} drew it'
    if reading == 'averaging':
        from visualdynamics.core.averaging import Averaging

        assert called[0][1] == (history.averaging or Averaging.for_records(
            len(history.abscissa))), "the history's own frames"
        assert called[0][2] == history.sample_rate


def test_the_octave_steps_go_on_the_stage(psd, tmp_path, monkeypatch):
    called = []
    real = stage_marks.add_octave_preview
    monkeypatch.setattr(stage_marks, 'add_octave_preview',
                        lambda *a, **k: (called.append(a), real(*a, **k))[1])
    visualdynamics.viz.waterfall.plot_waterfall(
        psd, [0, 2], marks='octave', per_octave=6,
        screenshot=str(tmp_path / 'octave-3d.png'))
    curves = called[0][1]
    assert len(curves) == 2, 'a step trace per drawn record'
    _banded, _x, rows = stage_marks.octave_steps(psd, 6, stage=True)
    assert np.allclose(curves[1][1], rows[2], equal_nan=True)


def test_the_stage_steps_are_on_the_stage_s_own_log_axes(psd):
    """The stage draws log10 of a log-scaled level and of a log
    frequency axis; steps left linear would stand orders of magnitude
    off the ribbons they preview. Nothing caught that until this."""
    banded, x, rows = stage_marks.octave_steps(psd, 3)
    _banded, x_stage, rows_stage = stage_marks.octave_steps(psd, 3,
                                                            stage=True)
    assert banded.log_scaled(), 'a PSD reads on a log level'
    assert np.allclose(rows_stage, np.log10(np.where(rows > 0, rows,
                                                     np.nan)),
                       equal_nan=True)
    expected_x = (np.log10(x) if getattr(banded, 'log_abscissa', False)
                  else x)
    assert np.allclose(x_stage, expected_x, equal_nan=True)
