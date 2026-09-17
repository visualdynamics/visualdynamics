"""A specification says where the response may not go, so the plot shades
it rather than leaving four lines to be read against each other.

Yellow between the warning limit and the abort limit, red beyond abort,
above the upper pair and below the lower pair alike. A specification may
carry any of the four limits, so each side is worked out on its own: with
no abort limit the yellow runs to the edge of the plot, and with no
warning limit the red starts at abort.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import Specification
from visualdynamics.theme import theme as resolve_theme


def spec(**limits):
    frequencies = np.logspace(1, 3, 48)
    level = np.full((1, 48), 0.01)
    return Specification(abscissa=frequencies, ordinate=level,
                         response_dof=['101X+'],
                         ordinate_dim=['acceleration**2/frequency'],
                         **{name: level * factor
                            for name, factor in limits.items()})


# the widgets are kept alive for the run: a GraphicsLayoutWidget that
# goes out of scope takes its items' C++ objects with it, and touching a
# fill afterwards raises "already deleted" rather than failing an assert
_ALIVE = []


def zones(qt_app, data, theme='light'):
    """The shaded regions a plot of `data` ends up carrying."""
    import pyqtgraph as pg

    from visualdynamics.plot import build_plots

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    build_plots(layout, [('Spec', data, None)], theme=theme)
    plot = next(item for item in layout.ci.items
                if hasattr(item, 'listDataItems'))
    return [item for item in plot.items
            if isinstance(item, pg.FillBetweenItem)]


def color_of(fill):
    return fill.brush().color().name()


def test_both_limits_give_four_zones(qt_app):
    """Warning-to-abort and beyond-abort, each way — and past abort the
    color says which way.

    Both were red, which said "out of tolerance" and left the direction
    to be read off the geometry, where every other mark on this plot
    already says over in red and under in blue.
    """
    filled = zones(qt_app, spec(warning_lower=0.5, warning_upper=2.0,
                                abort_lower=0.25, abort_upper=4.0))
    assert len(filled) == 4
    colors = sorted(color_of(fill) for fill in filled)
    light = resolve_theme('light')
    assert colors == sorted([light['exceed_over'], light['exceed_under'],
                              light['limit_warning'],
                              light['limit_warning']])


def test_warning_alone_shades_to_the_edge(qt_app):
    """Nothing says where the yellow stops, so it does not stop."""
    filled = zones(qt_app, spec(warning_lower=0.5, warning_upper=2.0))
    assert len(filled) == 2
    warning = resolve_theme('light')['limit_warning']
    assert {color_of(fill) for fill in filled} == {warning}


def test_abort_alone_shades_over_in_red_and_under_in_blue(qt_app):
    filled = zones(qt_app, spec(abort_lower=0.25, abort_upper=4.0))
    assert len(filled) == 2
    light = resolve_theme('light')
    assert {color_of(fill) for fill in filled} == \
        {light['exceed_over'], light['exceed_under']}


def test_a_specification_with_no_limits_shades_nothing(qt_app):
    assert zones(qt_app, spec()) == []


def test_the_zones_are_translucent(qt_app):
    """Solid fills would hide the curve they are describing."""
    filled = zones(qt_app, spec(warning_lower=0.5, warning_upper=2.0,
                                abort_lower=0.25, abort_upper=4.0))
    assert all(0 < fill.brush().color().alpha() < 128 for fill in filled)


def test_the_zones_sit_behind_the_curves(qt_app):
    filled = zones(qt_app, spec(warning_lower=0.5, warning_upper=2.0))
    assert all(fill.zValue() < 0 for fill in filled)


def test_a_record_with_no_limits_of_its_own_is_not_shaded(qt_app):
    """Limits are per control channel, so a cross-spectral record's are
    NaN — a fill of NaN is not a gap, it is a mess."""
    frequencies = np.logspace(1, 3, 48)
    level = np.vstack([np.full(48, 0.01), np.full(48, 0.01)])
    limits = np.vstack([np.full(48, 0.02), np.full(48, np.nan)])
    data = Specification(
        abscissa=frequencies, ordinate=level,
        response_dof=['101X+', '101X+'], reference_dof=['101X+', '102X+'],
        ordinate_dim=['acceleration**2/frequency'] * 2,
        warning_upper=limits, warning_lower=limits / 4)
    filled = zones(qt_app, data)
    assert len(filled) == 2, 'the control channel only'


@pytest.mark.parametrize('name', ['light', 'dark'])
def test_each_theme_brings_its_own_zone_colors(qt_app, name):
    filled = zones(qt_app, spec(warning_lower=0.5, warning_upper=2.0),
                   theme=name)
    assert {color_of(fill)
            for fill in filled} == {resolve_theme(name)['limit_warning']}


def bounds_of(fill):
    """(lower, upper) curves a fill spans, as arrays."""
    return tuple(np.asarray(curve.getData()[1]) for curve in fill.curves)


def shown(data, bound, at=None):
    """A limit as the plot's curves hold it, at the x they are drawn on.

    Two transforms sit between a limit and the numbers in a curve: the
    conversion to display units, and the log the ordinate axis is in —
    a PSD plots on a log axis, and pyqtgraph keeps its items' data in
    that space, so 31 (in/s^2)^2/Hz reads back as 1.491.

    And a third, when the specification is written at breakpoints: its
    zone is filled between power laws rather than straight lines, so the
    fill carries more points than the specification does and the limit
    has to be evaluated where they fall.
    """
    from visualdynamics.core.compliance import log_interpolate
    from visualdynamics.units import DEFAULT_SYSTEM

    values = np.asarray(data.display_limit(bound, DEFAULT_SYSTEM))[0].real
    if at is not None:
        values = log_interpolate(at, data.abscissa, values)
    return np.log10(values)


def edge_x(fill):
    return np.asarray(fill.curves[0].getData()[0])


def test_the_yellow_stops_at_the_abort_line(qt_app):
    """The whole point of having both: the warning zone runs from the
    warning limit up to the abort limit and no further, so the red
    beyond it is still visible as its own zone."""
    data = spec(warning_lower=0.5, warning_upper=2.0,
                abort_lower=0.25, abort_upper=4.0)
    warning = resolve_theme('light')['limit_warning']
    upper = [fill for fill in zones(qt_app, data)
             if color_of(fill) == warning
             and np.allclose(bounds_of(fill)[0],
                             shown(data, 'warning_upper', edge_x(fill)))]
    assert len(upper) == 1, 'one yellow zone starts at the upper warning'
    assert np.allclose(bounds_of(upper[0])[1],
                       shown(data, 'abort_upper', edge_x(upper[0]))), (
        'and stops at abort rather than running past it')


def test_the_lower_yellow_stops_at_the_lower_abort_line(qt_app):
    data = spec(warning_lower=0.5, warning_upper=2.0,
                abort_lower=0.25, abort_upper=4.0)
    warning = resolve_theme('light')['limit_warning']
    lower = [fill for fill in zones(qt_app, data)
             if color_of(fill) == warning
             and np.allclose(bounds_of(fill)[1],
                             shown(data, 'warning_lower', edge_x(fill)))]
    assert len(lower) == 1
    assert np.allclose(bounds_of(lower[0])[0],
                       shown(data, 'abort_lower', edge_x(lower[0])))


def test_without_an_abort_line_the_yellow_runs_past_everything(qt_app):
    """Nothing bounds it, so it has to leave the top of the plot rather
    than stopping at the warning limit and shading nothing."""
    data = spec(warning_lower=0.5, warning_upper=2.0)
    tops = [bounds_of(fill)[1].max() for fill in zones(qt_app, data)]
    # decades, not a factor: the curves are in log space
    assert max(tops) > shown(data, 'warning_upper').max() + 5
