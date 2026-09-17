"""Never hand pyqtgraph a curve with holes in it.

pyqtgraph builds a `QPainterPath` two ways. With no gaps it fills a
`QPolygonF` and lets Qt build the path — Qt's own API. With gaps, and
`arrayToQPath` draws the line at 2% of the points being non-finite, it
packs a byte buffer by hand and has Qt deserialize it, against a
private binary layout its own docstring warns "may change in future
versions of Qt".

That second route is where the intermittent Bus error was raised from,
and on a plot of a specification against its response exactly one curve
was taking it: the specification, 404 non-finite points in 1083,
because a controller writes it to zero outside its band and zero on a
log axis is -inf.

These tests pin the invariant rather than the crash. A Bus error at one
run in eight cannot be tested for directly — a green run proves
nothing — but "no curve is drawn with gaps in it" is exactly checkable,
and it is the condition that keeps the process off that code path.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import Psd, Specification, TimeHistory
from visualdynamics.plot import as_power_law, build_plots, gapless

#: the fraction of non-finite points at which pyqtgraph stops using
#: QPolygonF and starts packing the path's bytes itself
HAND_PACKED = 0.02

BREAKS = np.array([20.0, 80.0, 350.0, 2000.0])
LEVELS = np.array([0.01, 0.16, 0.16, 0.005])

_ALIVE: list = []


def paths_built(series, theme='dark'):
    """Every (points, non-finite) pyqtgraph was asked to build a path from."""
    import pyqtgraph as pg
    import pyqtgraph.functions as fn

    seen: list[tuple[int, int]] = []
    original = fn.arrayToQPath

    def probe(x, y, connect='all', finiteCheck=True):
        n = len(x)
        if n and connect == 'finite' and finiteCheck:
            seen.append(
                (n, int(n - np.sum(np.isfinite(x) & np.isfinite(y)))))
        return original(x, y, connect, finiteCheck)

    fn.arrayToQPath = probe
    try:
        layout = pg.GraphicsLayoutWidget()
        layout.resize(1000, 700)
        _ALIVE.append(layout)
        build_plots(layout, series, theme=theme)
        # a curve builds its path when it is painted, not when it is
        # given its data, so nothing is under test until something
        # paints. grab() renders the widget outright, which needs no
        # event loop and cannot be skipped for being off screen.
        layout.grab()
    finally:
        fn.arrayToQPath = original
    return seen


def hand_packed(series, theme='dark'):
    return [(n, bad) for n, bad in paths_built(series, theme)
            if bad and bad / n >= HAND_PACKED]


def spec(**limits):
    """A specification written to zero outside its band, as a controller
    writes one: the shape that put 404 holes in a 1083-point curve."""
    axis = np.linspace(0.0, 4096.0, 1025)
    values = np.zeros((1, axis.size))
    inside = (axis >= BREAKS[0]) & (axis <= BREAKS[-1])
    values[0, inside] = np.interp(np.log10(axis[inside]),
                                  np.log10(BREAKS), np.log10(LEVELS))
    values[0, inside] = 10.0 ** values[0, inside]
    return Specification(
        abscissa=axis, ordinate=values, response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'],
        **{name: values * factor for name, factor in limits.items()})


def response(target):
    return Psd(abscissa=target.abscissa,
               ordinate=np.where(target.ordinate > 0, target.ordinate, 1e-9),
               response_dof=['101Z+'],
               ordinate_dim=['acceleration**2/frequency'])


# ---- the invariant ------------------------------------------------------


def test_a_specification_is_drawn_without_holes(qt_app):
    target = spec()
    assert hand_packed([('S', target, None)]) == []


def test_a_specification_and_its_response_are_drawn_without_holes(qt_app):
    target = spec(warning_upper=2.0, warning_lower=0.5,
                  abort_upper=4.0, abort_lower=0.25)
    assert hand_packed([('S', target, None),
                        ('M', response(target), None)]) == []


def test_the_zone_shading_is_drawn_without_holes(qt_app):
    """The zone edges are invisible curves, but a `FillBetweenItem`
    still builds a path from them — a corrupt path is corrupt whoever
    paints it."""
    target = spec(warning_upper=2.0, abort_upper=4.0)
    built = paths_built([('S', target, None), ('M', response(target), None)])
    assert built, 'the probe saw nothing, so it is testing nothing'
    assert max(bad / n for n, bad in built) < HAND_PACKED


def test_a_breakpoint_specification_is_drawn_without_holes(qt_app):
    """The other way a specification arrives: a handful of breakpoints,
    filled in to follow its power law."""
    target = Specification(
        abscissa=BREAKS, ordinate=np.atleast_2d(LEVELS),
        response_dof=['101Z+'], ordinate_dim=['acceleration**2/frequency'],
        abort_upper=np.atleast_2d(LEVELS * 4.0))
    assert hand_packed([('S', target, None)]) == []


def test_a_time_history_is_drawn_without_holes(qt_app):
    t = np.linspace(0.0, 1.0, 4096)
    history = TimeHistory(t, np.sin(2 * np.pi * 25.0 * t)[None, :],
                          response_dof=['101Z+'], ordinate_dim=['acceleration'])
    assert hand_packed([('T', history, None)]) == []


# ---- the two pieces that keep it true -----------------------------------


def test_a_specification_is_filled_in_only_over_its_own_band():
    """`as_power_law` used to densify across the whole axis, so every
    point outside the band came back NaN. That single curve was the one
    taking the hand-packed route."""
    axis = np.linspace(0.0, 4096.0, 1025)
    values = np.zeros(axis.size)
    inside = (axis >= 20.0) & (axis <= 2000.0)
    values[inside] = 0.1
    x, y = as_power_law(axis, values)
    assert np.isfinite(y).all(), 'no holes'
    assert x.min() >= 20.0 and x.max() <= 2000.0, 'and only the band'


def test_gapless_cuts_to_the_run_that_says_something():
    x = np.arange(10.0)
    y = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, np.nan, np.nan,
                  np.nan, np.nan])
    cut_x, cut_y = gapless(x, y)
    assert list(cut_x) == [2.0, 3.0, 4.0, 5.0]
    assert list(cut_y) == [1.0, 2.0, 3.0, 4.0]


def test_gapless_leaves_a_whole_curve_alone():
    x, y = np.arange(5.0), np.arange(5.0) ** 2
    cut_x, cut_y = gapless(x, y)
    assert cut_x is x and cut_y is y


def test_gapless_does_not_bridge_a_hole_in_the_middle():
    """It cuts the ends off; it does not join two stretches that were
    never joined. A specification with a genuine gap in the middle is
    still drawn across it — that is a different problem, and inventing
    a line through it would be worse than either."""
    x = np.arange(6.0)
    y = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    cut_x, cut_y = gapless(x, y)
    assert cut_x.size == 6, 'the ends are already finite, so nothing is cut'
    assert np.isnan(cut_y).sum() == 2


@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_neither_theme_draws_a_gapped_curve(qt_app, theme):
    target = spec(warning_upper=2.0, abort_upper=4.0)
    assert hand_packed([('S', target, None),
                        ('M', response(target), None)], theme) == []
