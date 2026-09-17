"""Each curve drawn the way its own points mean their area.

A PSD line is a density over its bin, so it is flat across the bin and
the area drawn is the area that gets summed. A specification's points
are breakpoints of a power law, so the curve between two of them is
that law and not the straight line a polyline would take. Anything that
is a value at a frequency rather than an area — an FRF, a coherence, a
phase — is drawn as the line it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import Coherence, Psd, Specification, TimeHistory

_ALIVE = []


def drawn(qt_app, data, theme='light'):
    """The curves a plot of `data` ends up carrying, as (x, y)."""
    import pyqtgraph as pg

    from visualdynamics.plot import build_plots, data_curves

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    build_plots(layout, [('D', data, None)], theme=theme)
    plot = next(item for item in layout.ci.items
                if hasattr(item, 'listDataItems'))
    return [tuple(np.asarray(a) for a in curve.getData())
            for curve in data_curves(plot)], plot


def a_psd(lines=None, level=1e-3):
    lines = np.linspace(10.0, 100.0, 10) if lines is None else lines
    return Psd(abscissa=np.asarray(lines, dtype=float),
               ordinate=np.full((1, len(lines)), float(level)),
               response_dof=['101Z+'],
               ordinate_dim=['acceleration**2/frequency'])


# ---- a density per bin --------------------------------------------------


def test_a_psd_is_drawn_flat_across_each_bin(qt_app):
    """What is summed when it is integrated, so the area drawn is the
    area counted. A line between bin centers draws a ramp inside each
    bin that the estimate does not contain."""
    curves, _plot = drawn(qt_app, a_psd())
    x, y = curves[0]
    assert len(x) == len(y) + 1, 'bin edges, one more than the values'


def test_the_bins_meet_at_the_midpoints_between_lines(qt_app):
    lines = np.linspace(10.0, 100.0, 10)
    curves, _plot = drawn(qt_app, a_psd(lines))
    x, _y = curves[0]
    step = lines[1] - lines[0]
    assert x[0] == pytest.approx(lines[0] - step / 2)
    assert x[-1] == pytest.approx(lines[-1] + step / 2)
    assert np.allclose(np.diff(x), step)


def test_a_cross_spectrum_is_a_density_too(qt_app):
    """A CPSD is a PSD with a reference, and its records are bins the
    same way."""
    lines = np.linspace(10.0, 100.0, 8)
    cpsd = Psd(abscissa=lines, ordinate=np.full((1, 8), 1e-3),
               response_dof=['101Z+'], reference_dof=['102Z+'],
               ordinate_dim=['acceleration**2/frequency'])
    curves, _plot = drawn(qt_app, cpsd)
    assert len(curves[0][0]) == len(curves[0][1]) + 1


def test_a_single_line_is_not_a_bin(qt_app):
    """Nothing says how wide it would be."""
    curves, _plot = drawn(qt_app, a_psd(lines=[50.0]))
    x, y = curves[0]
    assert len(x) == len(y)


# ---- a power law between breakpoints ------------------------------------


def test_a_breakpoint_specification_is_filled_in(qt_app):
    """The frequency axis is linear, so a straight line drawn between
    two breakpoints is not the curve they mean — over a decade it runs
    2.5 times over it through the middle."""
    spec = Specification(
        abscissa=np.array([10.0, 100.0, 1000.0]),
        ordinate=np.array([[1e-4, 1e-2, 1e-4]]), response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'])
    curves, _plot = drawn(qt_app, spec)
    x, y = curves[0]
    assert len(x) > 100, 'three points would draw two straight segments'
    # and what is drawn is the power law. The ordinate axis is in log
    # mode, so the drawn y is already log10 of the level in display
    # units — which makes the check a unit-free one: halfway in log
    # frequency between two breakpoints is halfway between their logs.
    def at(frequency):
        return y[np.argmin(np.abs(x - frequency))]

    assert at(31.6227766) == pytest.approx((at(10.0) + at(100.0)) / 2,
                                           abs=0.01)
    # and a straight line in linear frequency would be well above it
    linear = at(10.0) + (31.6227766 - 10.0) / 90.0 * (at(100.0) - at(10.0))
    assert linear < at(31.6227766) + 0.5, 'in log the two differ'
    assert 10.0 ** (linear - at(31.6227766)) == pytest.approx(0.4, abs=0.1)


def test_a_dense_specification_is_left_alone(qt_app):
    """A controller writes its specification on the same line grid as
    everything else, and there is nothing between two adjacent lines to
    fill in."""
    lines = np.linspace(10.0, 1000.0, 991)
    spec = Specification(abscissa=lines, ordinate=np.full((1, 991), 1e-3),
                         response_dof=['101Z+'],
                         ordinate_dim=['acceleration**2/frequency'])
    curves, _plot = drawn(qt_app, spec)
    assert len(curves[0][0]) == 991


def test_a_specification_is_not_drawn_as_steps(qt_app):
    """It is a continuous requirement. Stepped, it would claim the level
    asked for jumps at bin edges, and a three-breakpoint specification
    would come out as three flat shelves instead of two slopes."""
    spec = Specification(
        abscissa=np.array([10.0, 100.0, 1000.0]),
        ordinate=np.array([[1e-4, 1e-2, 1e-4]]), response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'])
    curves, _plot = drawn(qt_app, spec)
    x, y = curves[0]
    assert len(x) == len(y), 'not bin edges'


def test_a_limit_is_shaded_on_the_same_curve(qt_app):
    """A limit is part of the specification and means what it means the
    same way, so the zone filling it is bounded by the same power law —
    even though the limit itself is no longer drawn as a line."""
    import pyqtgraph as pg

    from visualdynamics.plot import data_curves

    spec = Specification(
        abscissa=np.array([10.0, 100.0, 1000.0]),
        ordinate=np.array([[1e-4, 1e-2, 1e-4]]),
        abort_upper=np.array([[4e-4, 4e-2, 4e-4]]),
        response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'])
    _curves, plot = drawn(qt_app, spec)
    assert len(data_curves(plot)) == 1, 'the specification, and no bounds'
    zones = [i for i in plot.items if isinstance(i, pg.FillBetweenItem)]
    assert zones, 'the limit is a zone'
    edges = [len(np.asarray(c.getData()[0])) for z in zones for c in z.curves]
    assert all(n > 100 for n in edges), 'filled between power laws'


# ---- everything else is a value at a frequency --------------------------


def test_a_coherence_is_a_line(qt_app):
    """Not an area under anything: it is a ratio at each frequency."""
    lines = np.linspace(10.0, 100.0, 20)
    data = Coherence(abscissa=lines, ordinate=np.full((1, 20), 0.9),
                     response_dof=['101Z+'], reference_dof=['102Z+'])
    curves, _plot = drawn(qt_app, data)
    assert len(curves[0][0]) == len(curves[0][1])


def test_a_time_history_is_a_line(qt_app):
    t = np.arange(64) / 64.0
    data = TimeHistory(abscissa=t, ordinate=np.sin(2 * np.pi * 4 * t)[None],
                       response_dof=['101Z+'], ordinate_dim=['acceleration'])
    curves, _plot = drawn(qt_app, data)
    assert len(curves[0][0]) == len(curves[0][1])


def test_a_psds_phase_is_a_line(qt_app):
    """A phase is not an area under anything, whatever it is a phase
    of."""
    import pyqtgraph as pg

    from visualdynamics.plot import build_plots, data_curves

    lines = np.linspace(10.0, 100.0, 12)
    cpsd = Psd(abscissa=lines,
               ordinate=np.full((1, 12), 1e-3 + 1e-4j),
               response_dof=['101Z+'], reference_dof=['102Z+'],
               ordinate_dim=['acceleration**2/frequency'])
    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    build_plots(layout, [('D', cpsd, None)], theme='light',
                component='phase')
    plot = next(item for item in layout.ci.items
                if hasattr(item, 'listDataItems'))
    x, y = data_curves(plot)[0].getData()
    assert len(x) == len(y)


# ---- and the report draws by the same rule -------------------------------


def _report_block(source, **extra):
    """The plot block the report would build for this object."""
    from visualdynamics.report import _build_block
    from visualdynamics.units import DEFAULT_SYSTEM

    block = {'kind': 'plot', 'source': 'X', 'caption': '', **extra}
    return _build_block(block, {'X': source}, DEFAULT_SYSTEM)


def test_the_report_asks_the_same_shape_question_as_the_plot():
    """One rule, three readings: the flat plot, the 3-D waterfall and
    the exported figure all draw by `drawing_shape`. The report used to
    decide for itself — `interpolation == 'bin'` and nothing else — so
    a specification exported as straight segments where the app drew
    the power law."""
    from visualdynamics.plot import drawing_shape

    psd = Psd(np.linspace(0.0, 100.0, 33), np.ones((1, 33)),
              response_dof=['1Z+'], ordinate_dim=['acceleration**2/frequency'],
              ordinate_unit=['(m/s**2)**2/Hz'])
    assert drawing_shape(psd) == 'steps'
    assert _report_block(psd).get('steps') is True

    spec = Specification(np.array([10.0, 100.0]), np.array([[1e-4, 1e-2]]),
                         response_dof=['1Z+'],
                         ordinate_dim=['acceleration**2/frequency'],
                         ordinate_unit=['(m/s**2)**2/Hz'])
    assert drawing_shape(spec) == 'law'
    assert not _report_block(spec).get('steps'), \
        'breakpoints are a power law, never a staircase'


def test_the_reports_specification_follows_the_power_law():
    """The frequency axis is linear in the report as in the app, so a
    straight segment between breakpoints runs ~5 dB off mid-decade."""
    spec = Specification(np.array([10.0, 100.0]), np.array([[1e-4, 1e-2]]),
                         response_dof=['1Z+'],
                         ordinate_dim=['acceleration**2/frequency'],
                         ordinate_unit=['(m/s**2)**2/Hz'])
    built = _report_block(spec)
    assert built['logy'] is True
    curve = built['curves'][0]
    assert curve['x'] is not None and len(curve['x']) > 2, \
        'filled in, not two breakpoints joined straight'
    # Halfway along the segment in *frequency*, the drawn value must
    # sit on the law rather than on the chord. Both readings are
    # derived from the drawn endpoints, because the report converts to
    # the display system — asserting SI numbers here was a bad test,
    # and it failed for that rather than for the code.
    xs = np.asarray(curve['x'], dtype=float)
    ys = np.asarray([np.nan if v is None else v for v in curve['y']],
                    dtype=float)
    f0, f1 = xs[0], xs[-1]
    l0, l1 = ys[0], ys[-1]          # already log10 — logy is on
    at = int(np.argmin(np.abs(xs - 0.5 * (f0 + f1))))
    f = xs[at]
    law = l0 + (l1 - l0) * np.log(f / f0) / np.log(f1 / f0)
    chord = l0 + (l1 - l0) * (f - f0) / (f1 - f0)
    assert abs(law - chord) > 0.2, 'the two readings must differ here'
    assert abs(ys[at] - law) < abs(ys[at] - chord), \
        'the drawn point sits on the law, not on the chord'


def test_the_report_steps_on_the_bands_own_edges():
    """An octave band's center is the geometric mean of its edges, so
    midpoints between centers — which the JS used to guess — miss them.
    The payload carries the real edges now."""
    from visualdynamics.core.octave import bin_bounds

    centers = np.array([100.0, 125.0, 160.0])
    widths = np.array([23.0, 29.0, 36.0])
    banded = Psd(centers, np.ones((1, 3)), response_dof=['1Z+'],
                 ordinate_dim=['acceleration**2/frequency'],
                 ordinate_unit=['(m/s**2)**2/Hz'], bandwidth=widths)
    built = _report_block(banded)
    assert built.get('steps') is True
    lower, upper = bin_bounds(centers, widths)
    expected = np.concatenate([lower, [upper[-1]]])
    assert np.allclose(built['edges'], expected), \
        'the bands, not the midpoints between their centers'
