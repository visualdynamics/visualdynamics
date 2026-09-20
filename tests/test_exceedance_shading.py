"""Marking the lines that went outside an abort limit.

A percentage says how much of a run was out of tolerance. This says
which lines, and which way: over its own bin, from the limit to where
the line actually got to — red where it went over, blue where it fell
under.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.compliance import log_interpolate
from visualdynamics.core.data import Psd, Specification
from visualdynamics.theme import theme as resolve_theme

_ALIVE = []
BREAKS = np.array([20.0, 80.0, 350.0, 2000.0])
LEVELS = np.array([0.01, 0.16, 0.16, 0.005])


def spec(**limits):
    return Specification(
        abscissa=BREAKS, ordinate=np.atleast_2d(LEVELS),
        response_dof=['101Z+'], ordinate_dim=['acceleration**2/frequency'],
        **{name: np.atleast_2d(LEVELS * factor)
           for name, factor in limits.items()})


def measured(lines, tweak=None):
    values = log_interpolate(lines, BREAKS, LEVELS)
    if tweak is not None:
        values = tweak(values)
    out = Psd(abscissa=lines, ordinate=np.atleast_2d(values),
              response_dof=['101Z+'],
              ordinate_dim=['acceleration**2/frequency'])
    # held unscaled: these tests aim responses far outside the limits
    # on purpose, and comparison scaling would first lay them back on
    out.scale_db = 0
    return out


def shaded(qt_app, specification, psd, theme='light'):
    """The exceedance fills a plot of the two ends up carrying."""
    import pyqtgraph as pg

    from visualdynamics.plot import build_plots

    layout = pg.GraphicsLayoutWidget()
    _ALIVE.append(layout)
    build_plots(layout, [('S', specification, None), ('M', psd, None)],
                theme=theme)
    plot = next(item for item in layout.ci.items
                if hasattr(item, 'listDataItems'))
    fills = [item for item in plot.items
             if getattr(item, 'is_exceedance', False)]
    return fills, plot


def color_of(fill):
    return fill.brush().color().name()


def span(fill):
    """(low, high) in x over which the fill has any height at all."""
    lower, upper = (np.asarray(curve.getData()[1]) for curve in fill.curves)
    edges = np.asarray(fill.curves[0].getData()[0])
    open_at = np.flatnonzero(~np.isclose(lower, upper))
    if not len(open_at):
        return None
    return float(edges[open_at.min()]), float(edges[open_at.max() + 1])


# ---- which lines, and which way -----------------------------------------


def test_a_line_over_the_abort_limit_is_boxed_in_red(qt_app):
    lines = np.linspace(20.0, 2000.0, 120)
    def over(v):
        v = v.copy()
        v[30:36] *= 9.0
        return v

    fills, _plot = shaded(qt_app, spec(abort_upper=4.0, abort_lower=0.25),
                          measured(lines, over))
    reds = [f for f in fills
            if color_of(f) == resolve_theme('light')['exceed_over']]
    assert len(reds) == 1
    assert span(reds[0]) is not None


def test_a_line_under_the_abort_limit_is_boxed_in_blue(qt_app):
    """An under-test is as much a failure as an over-test, and which way
    it went is the first thing to know."""
    lines = np.linspace(20.0, 2000.0, 120)
    def under(v):
        v = v.copy()
        v[70:74] *= 0.05
        return v

    fills, _plot = shaded(qt_app, spec(abort_upper=4.0, abort_lower=0.25),
                          measured(lines, under))
    blues = [f for f in fills
             if color_of(f) == resolve_theme('light')['exceed_under']]
    assert len(blues) == 1
    assert span(blues[0]) is not None


def test_a_banded_response_on_its_target_is_not_shaded(qt_app):
    """The app's plot judged a banded comparison with `outside`'s
    defaults too (Brandon, 2026-09-19: the last octave band blue under
    a response in the middle of its zone). It reads the pair the way
    the table does now, and a response on its banded target draws no
    box at all."""
    from conftest import banded_pair_on_target

    banded, on_target = banded_pair_on_target()
    fills, _plot = shaded(qt_app, banded, on_target)
    assert fills == []


def test_a_run_inside_its_limits_is_not_shaded(qt_app):
    lines = np.linspace(20.0, 2000.0, 120)
    fills, _plot = shaded(qt_app, spec(abort_upper=4.0, abort_lower=0.25),
                          measured(lines))
    assert fills == []


def test_the_box_covers_the_line_s_own_bin(qt_app):
    """Half a width either side of it, which is exactly the bin the step
    plot draws — the box and the step it marks are the same rectangle."""
    lines = np.linspace(20.0, 2000.0, 100)
    step = lines[1] - lines[0]
    def over(v):
        v = v.copy()
        v[40] *= 9.0
        return v

    fills, _plot = shaded(qt_app, spec(abort_upper=4.0), measured(lines, over))
    low, high = span(fills[0])
    assert low == pytest.approx(lines[40] - step / 2, rel=1e-9)
    assert high == pytest.approx(lines[40] + step / 2, rel=1e-9)


def test_a_box_at_the_edge_stops_where_the_specification_does(qt_app):
    """The bin at the end of the band hangs over the edge, and it is
    judged on the part that is covered — so it is shaded over that part
    and no further. A stripe running past the last breakpoint claims a
    line was out of a tolerance that was never written there."""
    lines = np.linspace(15.0, 2100.0, 120)      # past both breakpoints
    fills, _plot = shaded(qt_app, spec(abort_upper=4.0),
                          measured(lines, lambda v: np.full(v.shape, 1e3)))
    low, high = span(fills[0])
    assert low == pytest.approx(BREAKS[0], rel=1e-9), 'starts at 20 Hz'
    assert high == pytest.approx(BREAKS[-1], rel=1e-9), 'stops at 2000 Hz'


def test_the_shading_marks_the_bins_the_table_counts(qt_app):
    """One rule, asked once. The plot used to decide `out` for itself
    with a comparison of its own, which at the edges gave a different
    answer from the table beside it."""
    from visualdynamics.core.compliance import compare

    lines = np.linspace(15.0, 2100.0, 120)
    def over(v):
        v = np.where(np.isfinite(v), v, 1e-6).copy()
        v[60:] *= 9.0
        return v

    target, psd = spec(abort_upper=4.0), measured(lines, over)
    fills, _plot = shaded(qt_app, target, psd)
    lower, upper = (np.asarray(c.getData()[1]) for c in fills[0].curves)
    assert int(np.sum(~np.isclose(lower, upper))) == \
        compare(target, psd)['abort_lines']


def test_the_stripe_runs_from_the_limit_off_the_top(qt_app):
    """A box from the limit to the line is a few pixels tall when a line
    is barely over, which is exactly when it most needs seeing. The
    stripe runs to the plot's own edge instead. Exactly to it, and not
    a millionfold past: a polygon reaching that far is a thing Qt has to
    rasterize, and it crashed doing so about one run in three."""
    lines = np.linspace(20.0, 2000.0, 100)

    def over(v):
        v = v.copy()
        v[40] *= 4.05          # barely past the fourfold abort limit
        return v

    fills, plot = shaded(qt_app, spec(abort_upper=4.0), measured(lines, over))
    lower, upper = (np.asarray(c.getData()[1]) for c in fills[0].curves)
    at = np.flatnonzero(~np.isclose(lower, upper))[0]
    top = plot.getViewBox().state['limits']['yLimits'][1]
    assert upper[at] == pytest.approx(top), 'the stripe reaches the top'
    assert lower[at] < top, 'and starts at the limit, well below it'


def test_a_line_barely_out_is_marked_as_plainly_as_one_far_out(qt_app):
    """Which is the point of going to the edge: how far out it went is
    the table's job, and being out at all is what has to be seen. A box
    from the limit to the line is a few pixels tall for the first and
    unmissable for the second, which is backwards."""
    lines = np.linspace(20.0, 2000.0, 100)

    def by(factor):
        def tweak(v):
            v = v.copy()
            v[40] *= factor
            return v
        fills, plot = shaded(qt_app, spec(abort_upper=4.0),
                             measured(lines, tweak))
        lower, upper = (np.asarray(c.getData()[1]) for c in fills[0].curves)
        at = np.flatnonzero(~np.isclose(lower, upper))[0]
        top = plot.getViewBox().state['limits']['yLimits'][1]
        return upper[at] - top

    # both reach the plot's own edge, whatever that edge happens to be
    assert by(4.05) == pytest.approx(0.0, abs=1e-9)
    assert by(40.0) == pytest.approx(0.0, abs=1e-9)


def test_only_the_abort_limits_are_marked(qt_app):
    """A warning is a warning. Boxing those too would leave most of a
    real run shaded and say nothing."""
    lines = np.linspace(20.0, 2000.0, 120)
    def past_warning(v):
        v = v.copy()
        v[30:60] *= 2.5          # past warning at 2x, inside abort at 4x
        return v

    fills, _plot = shaded(
        qt_app, spec(warning_upper=2.0, abort_upper=4.0),
        measured(lines, past_warning))
    assert fills == []


def test_nothing_is_marked_where_the_specification_says_nothing(qt_app):
    """Past the end of the band there is no limit to be outside of."""
    lines = np.linspace(20.0, 6000.0, 200)
    def high_outside(v):
        v = v.copy()
        v[np.asarray(lines) > 2000.0] = 100.0
        return np.nan_to_num(v, nan=1e-3)

    fills, _plot = shaded(qt_app, spec(abort_upper=4.0),
                          measured(lines, high_outside))
    for fill in fills:
        reach = span(fill)
        assert reach is None or reach[1] <= 2000.0 + 40.0


# ---- how they sit with everything else ----------------------------------


def test_the_boxes_are_not_counted_as_data(qt_app):
    """Two invisible edges apiece; anything reading the curves off a
    plot must not find them."""
    from visualdynamics.plot import data_curves

    lines = np.linspace(20.0, 2000.0, 120)
    def over(v):
        v = v.copy()
        v[30:36] *= 9.0
        return v

    fills, plot = shaded(qt_app, spec(abort_upper=4.0), measured(lines, over))
    assert fills, 'there were boxes to skip'
    assert len(data_curves(plot)) == 2, 'the psd and the spec'


def test_the_boxes_sit_over_the_zone_shading_and_under_the_curves(qt_app):
    """They are inside the red the abort zone already paints, so they
    have to be visible against it — and still under the lines they
    describe."""
    lines = np.linspace(20.0, 2000.0, 120)
    def over(v):
        v = v.copy()
        v[30:36] *= 9.0
        return v

    fills, plot = shaded(qt_app, spec(abort_upper=4.0), measured(lines, over))
    import pyqtgraph as pg

    zones = [i for i in plot.items
             if isinstance(i, pg.FillBetweenItem)
             and not getattr(i, 'is_exceedance', False)]
    assert zones, 'the zone shading is there too'
    assert all(f.zValue() > z.zValue() for f in fills for z in zones)
    assert all(f.zValue() < 0 for f in fills)


@pytest.mark.parametrize('name', ['light', 'dark'])
def test_each_theme_brings_its_own(qt_app, name):
    lines = np.linspace(20.0, 2000.0, 120)
    def over(v):
        v = v.copy()
        v[30:36] *= 9.0
        return v

    fills, _plot = shaded(qt_app, spec(abort_upper=4.0),
                          measured(lines, over), theme=name)
    assert color_of(fills[0]) == resolve_theme(name)['exceed_over']
