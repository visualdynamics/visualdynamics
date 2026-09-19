"""The frequency axis is the viewer's: decades or hertz, one switch.

By convention an SRS reads in decades and everything else in hertz;
some readers of a specification want the other (Brandon, 2026-09-05).
*Log f* on the plot bar switches every frequency plot, the stage and
the report's figures at once, through `core.data.frequency_axis`.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import Psd, Srs, TimeHistory, frequency_axis


@pytest.fixture(autouse=True)
def default_axis():
    frequency_axis('default')
    yield
    frequency_axis('default')


def _psd():
    f = np.arange(0.0, 2050.0, 2.0)
    return Psd(f, np.ones((2, len(f))), response_dof=['101Z+', '104Z+'],
               reference_dof=['101Z+', '104Z+'], ordinate_dim='acceleration**2/frequency')


def _plot(window):
    return next(item for item in window.data_pane.graphics.ci.items
                if hasattr(item, 'listDataItems'))


def _show(window, pump, name, obj):
    window.add_object(name, obj)
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()


# ---- the switch itself ----------------------------------------------------


def test_the_choice_overrides_the_convention_for_frequency_data_only():
    assert frequency_axis() == 'default'
    assert Srs.log_abscissa and not Psd.log_abscissa
    assert frequency_axis('log') == 'log'
    assert Psd.log_abscissa and Srs.log_abscissa
    assert not TimeHistory.log_abscissa, 'time is never in decades'
    assert _psd().log_abscissa, 'on the instance as on the class'
    assert frequency_axis('linear') == 'linear'
    assert not Srs.log_abscissa, "the viewer's choice beats the convention"
    assert frequency_axis(True) == 'log' and frequency_axis(False) == 'linear'
    assert frequency_axis(None) == 'default'
    assert Srs.log_abscissa and not Psd.log_abscissa
    with pytest.raises(ValueError, match="'log', 'linear' or 'default'"):
        frequency_axis('decades')
    assert visualdynamics.frequency_axis is frequency_axis


# ---- the plot bar --------------------------------------------------------------


def test_the_toggle_is_offered_for_frequency_data_and_reads_the_axis(window,
                                                                      pump):
    action = window.data_pane.log_frequency_action
    _show(window, pump, 'PSD', _psd())
    assert action.isVisible() and not action.isChecked()
    assert not _plot(window).getAxis('bottom').logMode
    f = np.arange(0.0, 1.0, 1 / 256)
    _show(window, pump, 'Time', TimeHistory(f, np.ones((1, len(f))),
                                             response_dof='101Z+',
                                             ordinate_dim='acceleration'))
    assert not action.isVisible(), 'nothing over frequency to switch'
    srs = Srs(np.geomspace(20.0, 2000.0, 50), np.ones((1, 50)),
              response_dof='101Z+', ordinate_dim='acceleration')
    _show(window, pump, 'SRS', srs)
    assert action.isVisible() and action.isChecked(), 'decades by convention'
    assert _plot(window).getAxis('bottom').logMode


def test_the_toggle_switches_every_frequency_plot_and_says_about_0_hz(window,
                                                                      pump):
    action = window.data_pane.log_frequency_action
    _show(window, pump, 'PSD', _psd())
    action.trigger()          # a checkable action toggles on trigger
    pump()
    assert frequency_axis() == 'log'
    assert _plot(window).getAxis('bottom').logMode
    assert action.isChecked()
    assert 'the 0 Hz line is off the log axis' in \
        window.statusBar().currentMessage()
    # the stage reads the same switch
    window.data_pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    assert action.isVisible() and action.isChecked()
    assert window._waterfall_object == 'PSD'
    from visualdynamics.viz.waterfall import waterfall_arrays
    assert waterfall_arrays(window.objects['PSD'],
                            unit_system=window.unit_system)['log_abscissa']
    # and back to hertz, the note gone
    window.data_pane.waterfall_action.setChecked(False)
    action.trigger()
    pump()
    assert frequency_axis() == 'linear'
    assert not _plot(window).getAxis('bottom').logMode
    assert '0 Hz' not in window.statusBar().currentMessage()
    # a specification opens the same way
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    window.show_object('Specification')
    pump()
    assert action.isVisible() and not action.isChecked()


def test_the_report_figure_follows_the_switch(tmp_path):
    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html

    project = visualdynamics.Project('t')
    project.add('PSD', _psd())
    report = Report()
    report.blocks.append({'kind': 'plot', 'source': 'PSD'})
    linear = render_html(report, project, visualdynamics.SI)
    frequency_axis('log')
    decades = render_html(report, project, visualdynamics.SI)
    assert '"logx": false' in linear and '"logx": true' in decades


def test_the_window_fixture_hands_the_next_test_the_default(window):
    frequency_axis('log')
    assert Psd.log_abscissa
    # the conftest teardown resets it; the autouse fixture above does
    # too, and test_srs_log_axis still finds the convention


def test_steps_and_power_laws_are_built_in_hertz_then_put_in_decades():
    """The bug the toggle exposed: the stage and the report figure took
    log10 of the lines first and then worked bin edges and power laws
    out of the exponents, so an octave PSD's steps and a specification's
    law landed in the wrong places under a log axis."""
    from visualdynamics.core.author import SpecificationDraft
    from visualdynamics.plot import as_power_law, bin_edges, step_outline
    from visualdynamics.report import _build_block
    from visualdynamics.viz.waterfall import waterfall_arrays

    banded = _psd().to_octave(3)
    spec = SpecificationDraft.at_dofs(['1Z+']).with_all_pairs(0.0).make()
    frequency_axis('log')
    us = visualdynamics.SI
    # the stage: a banded PSD's steps
    real_x, _values = step_outline(banded.abscissa, banded.ordinate,
                                   banded.bin_widths())
    cx, _cz = waterfall_arrays(banded, unit_system=us)['curves'][0]
    assert np.allclose(cx, np.log10(real_x)), \
        'edges from hertz, then decades — not edges from exponents'
    assert cx[0] == pytest.approx(np.log10(bin_edges(
        banded.abscissa, banded.bin_widths())[0]))
    # the stage: a specification's power law between two breakpoints
    law_x, _law = as_power_law(spec.abscissa, np.real(spec.ordinate[0]))
    assert len(law_x) > 2, 'filled in between 20 and 2000 Hz'
    cx, _cz = waterfall_arrays(spec, unit_system=us)['curves'][0]
    assert np.allclose(cx, np.log10(law_x))
    # the report's figure, both shapes
    built = _build_block({'kind': 'plot', 'source': 'Banded',
                          'mode': 'curves', 'caption': 'c'},
                         {'Banded': banded}, us, [])
    assert built['logx'] and built['steps']
    assert np.allclose(built['x'], np.log10(banded.abscissa))
    assert np.allclose(built['edges'], np.log10(bin_edges(
        banded.abscissa, banded.bin_widths())))
    built = _build_block({'kind': 'plot', 'source': 'Spec',
                          'mode': 'curves', 'caption': 'c'},
                         {'Spec': spec}, us, [])
    assert built['logx']
    assert np.allclose(built['curves'][0]['x'], np.log10(law_x))
    # and in hertz everything is where it was
    frequency_axis('linear')
    cx, _cz = waterfall_arrays(banded, unit_system=us)['curves'][0]
    assert np.allclose(cx, real_x)


# ---- the stage's decades -----------------------------------------------------


def test_the_decades_are_labeled_as_the_flat_plot_labels_them():
    from visualdynamics.viz.marks import decade_labels

    assert decade_labels(1.301, 3.311) == [(2.0, '10²'), (3.0, '10³')]
    assert decade_labels(-0.3, 3.0) == [(0.0, '1'), (1.0, '10¹'),
                                         (2.0, '10²'), (3.0, '10³')]
    assert decade_labels(2.0, 3.0) == [(2.0, '10²'), (3.0, '10³')], \
        'a decade on the edge of the range is on the axis'
    assert decade_labels(2.1, 2.9) == []


def test_the_stage_axis_shows_decades_not_even_divisions(window, pump):
    """Brandon, 2026-09-05: the 3-D axis read −0.6021, 0.4516, 1.505 …
    where the 2-D plot reads 1, 10¹, 10², 10³."""
    _show(window, pump, 'PSD', _psd())
    window.data_pane.log_frequency_action.trigger()
    pump()
    window.data_pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    plotter = window.data_pane.waterfall_plotter
    axes = plotter.renderer.cube_axes_actor

    def drawn(axis):
        strings = axes.GetAxisLabels(axis)
        return [strings.GetValue(i) for i in range(strings.GetNumberOfValues())]

    assert not axes.x_label_visibility and drawn(0) == [''], \
        'the exponents are put away'
    assert not axes.GetXAxisTickVisibility()
    assert not axes.GetDrawXGridlines()
    assert axes.z_label_visibility and drawn(2) != [''], \
        'the level axis keeps its numbers'
    assert 'decade-lines' in plotter.actors
    assert 'decade-labels-labels' in plotter.actors, \
        "pyvista's name for a point-label actor"
    mesh = plotter.actors['decade-lines'].mapper.dataset
    xs = sorted(set(np.round(mesh.points[:, 0], 6)))
    # the steps of a 2 Hz line reach down to its bin edge at 1 Hz, so
    # the stage runs from 10⁰ to just past 10³: four decades
    from visualdynamics.plot import bin_edges
    from visualdynamics.viz.waterfall import STAGE
    edges = bin_edges(window.objects['PSD'].abscissa)
    x0, x1 = np.log10(edges[edges > 0][0]), np.log10(edges[-1])
    assert (x0, round(x1, 3)) == (0.0, round(np.log10(2049.0), 3))
    assert len(xs) == 4, '1 Hz to 2049 Hz spans 1, 10¹, 10², 10³'
    expected = [(n - x0) / (x1 - x0) * STAGE[0] for n in (0, 1, 2, 3)]
    assert np.allclose(xs, expected, atol=1e-5), \
        'each line stands where its decade is on the stage'
    # back to hertz: the cube axes take the axis back
    window.data_pane.log_frequency_action.trigger()
    pump()
    window.render_current()
    pump()
    axes = plotter.renderer.cube_axes_actor
    assert axes.x_label_visibility and axes.GetDrawXGridlines()
    assert drawn(0) != ['']
    assert 'decade-lines' not in plotter.actors


def test_the_banded_stage_draws_the_same_decades(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    frequency_axis('log')
    window.data_pane.waterfall_action.setChecked(True)
    window.show_object('Specification')
    pump()
    plotter = window.data_pane.waterfall_plotter
    assert 'decade-lines' in plotter.actors
    assert not plotter.renderer.cube_axes_actor.x_label_visibility



def test_a_curve_in_decades_is_drawn_whole_not_peak_downsampled(window, pump):
    """The white specification line was jagged on the log axis (Brandon,
    2026-09-06). pyqtgraph's automatic downsampling buckets points in
    index space with one bucket size taken from the mean spacing, then
    draws each bucket's max and min at one x. That presumes evenly
    spaced x. In decades the low end has far fewer lines per pixel than
    the mean, so each bucket there spans several pixels and its
    max-then-min pair becomes a tooth. A spectrum is thousands of lines,
    not the hundreds of thousands of a record: drawn whole on the log
    axis. Pinned on what the curve item is handed to draw, in a plot
    narrow enough (the sheet takes the rest) that the bucketing runs."""
    from visualdynamics.core.data import Specification
    # the drone target's grid: 3981 lines at 0.5 Hz, one power-law skirt
    f = np.arange(10.0, 2000.5, 0.5)
    level = np.atleast_2d(np.where(f < 40, 1e-3 * (f / 40) ** 2, 1e-3))
    spec = Specification(abscissa=f, ordinate=level, response_dof=['101Z+'],
                         ordinate_dim=['acceleration**2/frequency'])
    window.resize(700, 800)
    pump()
    _show(window, pump, 'Spec', spec)

    def white_line():
        window.data_pane.graphics.grab()   # a paint, so the curve is built
        pump()
        [curve] = [c for c in _plot(window).listDataItems() if c.name()]
        return curve, curve.curve.getData()[0]

    curve, x = white_line()
    assert curve.opts['autoDownsample'] and len(x) < len(f), \
        'in hertz the spacing is even and the bucketing is welcome'
    window.data_pane.log_frequency_action.trigger()
    pump()
    assert _plot(window).getAxis('bottom').logMode
    curve, x = white_line()
    assert len(x) == len(f) and np.all(np.diff(x) > 0), \
        'every line drawn, once, in order — no bucket, no tooth'
    assert curve.opts['clipToView'], 'clipping does not bucket; it stays'


def test_the_decade_axis_turns_the_ticks_off_through_vtk():
    """The major ticks off by VTK's own call. The line assigned
    `x_axis_tick_visibility`, a name pyvista does not define on its
    actor, and reached VTK only through the wrappers' snake-case
    alias — which one Windows install refused outright, taking every
    banded stage down (Kevin Cross, 2026-09-19). Ticks on beforehand,
    so the test measures the call rather than the actor's default."""
    import pyvista as pv

    from visualdynamics.viz.marks import add_decade_axis

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(pv.Cube())
    plotter.show_bounds()
    axes = plotter.renderer.cube_axes_actor
    axes.XAxisTickVisibilityOn()
    assert axes.GetXAxisTickVisibility()
    add_decade_axis(plotter, (1.0, 3.0, -3.0, 1.0))
    assert not axes.GetXAxisTickVisibility()
    assert 'x_axis_tick_visibility' not in vars(axes), \
        'nothing assigned under a name the actor does not define'
    plotter.close()
