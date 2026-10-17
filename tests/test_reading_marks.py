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



@pytest.mark.parametrize('reading', ['averaging', 'truncation'])
def test_a_standalone_plot_s_marks_cannot_be_dragged(qt_app, run, reading):
    """Nothing in a standalone plot keeps a dragged span. The cut's
    region could be dragged there once, and the drag went nowhere."""
    import pyqtgraph as pg

    from visualdynamics.plot import _mark, build_plot

    layout = pg.GraphicsLayoutWidget()
    build_plot(layout, run.time_history)
    overlays = _mark(layout, run.time_history, reading)
    assert overlays
    assert not any(overlay.region.movable for overlay in overlays)

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


# ---- a PSD's confidence band, drawn flat --------------------------------


@pytest.fixture(scope='module')
def averaged(run):
    """A PSD computed here, so it carries the degrees of freedom of the
    frames it was averaged over."""
    return run.time_history.compute_psds()


def test_the_confidence_band_draws_flat(qt_app, averaged, tmp_path):
    path = tmp_path / 'confidence.png'
    visualdynamics.plot.plot_data(averaged, marks='confidence',
                                  records=[0], path=str(path), show=False)
    assert path.stat().st_size > 2000


def test_the_band_is_its_spectrum_s_steps_moved_by_its_factors(qt_app,
                                                               averaged):
    """Each edge is the record's own steps times the chi-square factor
    at its degrees of freedom, in the plot's log space, and the plot's
    limits are widened to take the top edge in."""
    import pyqtgraph as pg

    from visualdynamics.core.averaging import chi_square_factors
    from visualdynamics.plot import (
        build_plot,
        confidence_preview,
        data_curves,
        step_outline,
    )
    from visualdynamics.theme import theme
    from visualdynamics.units import DEFAULT_SYSTEM

    layout = pg.GraphicsLayoutWidget()
    build_plot(layout, averaged, records=[1])
    drawn = confidence_preview(layout.ci.items, averaged, theme('light'),
                               records=[1], levels=[0.9])
    edges = [item for _plot, item in drawn
             if isinstance(item, pg.PlotDataItem)]
    fills = [item for _plot, item in drawn
             if isinstance(item, pg.FillBetweenItem)]
    assert len(edges) == 2 and len(fills) == 1, 'one band, two edges'
    low, high = chi_square_factors(averaged.statistical_dof[1], 0.9)
    shown = averaged.display_ordinate(DEFAULT_SYSTEM, [1])[0]
    _x, steps = step_outline(averaged.abscissa, shown)
    assert np.allclose(edges[0].getData()[1], np.log10(steps * low))
    assert np.allclose(edges[1].getData()[1], np.log10(steps * high))
    plot = drawn[0][0]
    assert len(data_curves(plot)) == 1, 'the band is not counted as data'


def test_a_wide_band_widens_the_plot_s_limits(qt_app, averaged):
    """One frame's band reaches nearly twenty times over the estimate,
    well past the margin the plot was locked to; the top edge over the
    highest peak must still be in reach of the view."""
    import copy

    import pyqtgraph as pg

    from visualdynamics.core.averaging import chi_square_factors
    from visualdynamics.plot import build_plot, confidence_preview
    from visualdynamics.theme import theme
    from visualdynamics.units import DEFAULT_SYSTEM

    one = copy.copy(averaged)
    one.statistical_dof = np.full(averaged.num_records, 2.0)
    layout = pg.GraphicsLayoutWidget()
    build_plot(layout, one, records=[1])
    plot = next(item for item in layout.ci.items
                if isinstance(item, pg.PlotItem))
    before = plot.getViewBox().state['limits']['yLimits'][1]
    confidence_preview(layout.ci.items, one, theme('light'), records=[1],
                       levels=[0.9])
    _low, high = chi_square_factors(2.0, 0.9)
    peak = np.max(one.display_ordinate(DEFAULT_SYSTEM, [1])[0]) * high
    assert before < np.log10(peak), 'the case is one the lock cut off'
    assert plot.getViewBox().state['limits']['yLimits'][1] >= np.log10(peak)


def test_each_level_is_named_and_dashed_its_own_way(qt_app, averaged):
    import pyqtgraph as pg

    from visualdynamics.plot import build_plot, confidence_preview
    from visualdynamics.theme import theme

    layout = pg.GraphicsLayoutWidget()
    build_plot(layout, averaged, records=[0])
    drawn = confidence_preview(layout.ci.items, averaged, theme('light'),
                               records=[0])
    plot = drawn[0][0]
    names = [label.text for _sample, label in plot.legend.items]
    assert '90 % confidence' in names and '95 % confidence' in names
    styles = {item.opts['pen'].style() for _plot, item in drawn
              if isinstance(item, pg.PlotDataItem)}
    assert len(styles) == 2, 'told apart by more than the shading'


def test_a_zero_line_leaves_the_rest_of_the_band_shaded(qt_app, averaged):
    """A line at zero has no place on a log axis; the band either side
    of it is still shaded, and a record that is zero throughout draws
    nothing and warns of nothing."""
    import copy
    import warnings

    import pyqtgraph as pg

    from visualdynamics.plot import build_plot, confidence_preview
    from visualdynamics.theme import theme

    holed = copy.copy(averaged)
    holed.ordinate = averaged.ordinate.copy()
    middle = holed.ordinate.shape[1] // 2
    holed.ordinate[1, middle] = 0.0
    holed.ordinate[2] = 0.0
    layout = pg.GraphicsLayoutWidget()
    build_plot(layout, holed, records=[1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        drawn = confidence_preview(layout.ci.items, holed, theme('light'),
                                   records=[1, 2], levels=[0.9])
    fills = [item for _plot, item in drawn
             if isinstance(item, pg.FillBetweenItem)]
    assert len(fills) == 2, 'shaded on each side of the hole'
    edges = [item for _plot, item in drawn
             if isinstance(item, pg.PlotDataItem)]
    assert len(edges) == 4
    assert all(np.isfinite(edge.getData()[1]).all() for edge in edges)


def test_levels_belong_to_the_confidence_mark(qt_app, psd):
    with pytest.raises(ValueError, match="levels are the confidence of "
                                         "marks='confidence'"):
        visualdynamics.plot.plot_data(psd, marks='octave', levels=[0.8],
                                      path='never.png', show=False)


def test_a_band_needs_a_psd_that_knows_its_average(qt_app, run, psd):
    with pytest.raises(TypeError, match='drawn on a PSD'):
        visualdynamics.plot.plot_data(run.time_history, marks='confidence',
                                      path='never.png', show=False)
    with pytest.raises(ValueError, match='does not say what it was '
                                         'averaged over'):
        visualdynamics.plot.plot_data(psd, marks='confidence',
                                      path='never.png', show=False)
