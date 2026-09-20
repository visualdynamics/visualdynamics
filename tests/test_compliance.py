"""How a measured response compares with what it was controlled to.

The numbers a random vibration report is written from: how much energy
was asked for, how much arrived, and how much of the band strayed
outside the limits the controller was told to warn and abort on.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import compliance
from visualdynamics.core.data import Psd, Specification


def spec(frequencies=None, level=1e-3, **limits):
    """A flat specification over 10-1000 Hz, with whatever limits."""
    frequencies = (np.logspace(1, 3, 21) if frequencies is None
                   else np.asarray(frequencies, dtype=float))
    values = np.full((1, len(frequencies)), float(level))
    return Specification(
        abscissa=frequencies, ordinate=values, response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'],
        **{name: values * factor for name, factor in limits.items()})


def psd(frequencies, values):
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.atleast_2d(np.asarray(values, dtype=float))
    return Psd(abscissa=frequencies, ordinate=values, response_dof=['101Z+'],
               ordinate_dim=['acceleration**2/frequency'])


# ---- putting the two on one frequency axis ------------------------------


def test_a_slope_interpolates_as_a_power_law():
    """A specification is drawn in log-log and its breakpoints are read
    that way, so the segment between two of them is a power law. Read as
    a straight line in level against frequency instead, a decade-wide
    segment runs well above what was written through its own middle."""
    breakpoints = np.array([10.0, 100.0])
    values = np.array([1e-4, 1e-2])          # two decades over one
    got = compliance.log_interpolate([31.6227766], breakpoints, values)
    # halfway in log-log is the geometric mean of the levels
    assert got[0] == pytest.approx(1e-3, rel=1e-6)
    straight = np.interp(31.6227766, breakpoints, values)
    # 2.5 times the level written there, which is 4 dB
    assert straight / got[0] == pytest.approx(2.48, rel=0.02)


def test_it_says_nothing_outside_the_band():
    """Extending the end segments would invent a requirement nobody
    wrote, and a response out there is neither passing nor failing."""
    got = compliance.log_interpolate([5.0, 50.0, 5000.0],
                                     [10.0, 1000.0], [1e-3, 1e-3])
    assert np.isnan(got[0]) and np.isnan(got[2])
    assert got[1] == pytest.approx(1e-3)


def test_a_specification_written_to_zero_says_nothing_there():
    """Which is how a controller writes one outside its band."""
    frequencies = np.array([1.0, 10.0, 100.0, 1000.0])
    values = np.array([0.0, 1e-3, 1e-3, 0.0])
    got = compliance.log_interpolate([2.0, 50.0, 500.0], frequencies, values)
    assert np.isnan(got[0]), 'below the band it was written over'
    assert got[1] == pytest.approx(1e-3)
    assert np.isnan(got[2])


def test_too_little_to_interpolate_is_not_guessed():
    assert np.all(np.isnan(compliance.log_interpolate([10.0], [10.0], [1e-3])))
    assert np.all(np.isnan(compliance.log_interpolate([10.0], [], [])))


# ---- the levels ---------------------------------------------------------


def test_the_rms_is_every_line_times_its_own_bin():
    """A discrete spectrum is a density per bin, so its total is the sum
    of the bins — Parseval's, exactly. A thousand and one lines a hertz
    apart hold a thousand and one hertz of bin, not the thousand between
    the first line and the last, and a trapezoid reads that difference
    as two half-bins missing."""
    frequencies = np.linspace(10.0, 1010.0, 1001)      # df = 1 Hz
    values = np.full(1001, 4e-3)
    assert compliance.rms(frequencies, values) == pytest.approx(
        np.sqrt(4e-3 * 1001.0), rel=1e-9)
    trapezoid = np.sqrt(np.trapezoid(values, frequencies))
    assert trapezoid < compliance.rms(frequencies, values)


def test_an_uneven_axis_gets_bins_of_its_own():
    """Nothing says a frequency axis is evenly spaced — a specification
    written at breakpoints is not — so the bin is the spacing either
    side of each line rather than one number for the lot."""
    frequencies = np.array([10.0, 20.0, 40.0, 80.0])
    got = compliance.rms(frequencies, np.full(4, 1.0))
    widths = np.gradient(frequencies)
    assert got == pytest.approx(np.sqrt(widths.sum()), rel=1e-12)


def test_a_line_that_says_nothing_contributes_nothing():
    """Not bridged. Widening a bin onto its neighbors to cover a gap
    is assuming what is in the gap, and a spectrum with a hole in it
    holds less than one without."""
    frequencies = np.linspace(10.0, 110.0, 101)        # df = 1 Hz
    values = np.full(101, 1e-3)
    values[40:60] = np.nan                             # 20 lines missing
    got = compliance.rms(frequencies, values)
    assert got == pytest.approx(np.sqrt(1e-3 * 81.0), rel=1e-9)
    assert got < compliance.rms(frequencies, np.full(101, 1e-3))


# ---- what the table shows -----------------------------------------------


def test_a_response_that_matches_is_nothing_over():
    """Exactly nothing, at any resolution. The specification is
    integrated from its own points and the measurement from its bins —
    two different rules — so they have to be given one band or the
    difference between the rules shows up as a difference between the
    curves."""
    frequencies = np.linspace(10.0, 1000.0, 200)
    target = spec()
    matched = psd(frequencies,
                  compliance.log_interpolate(frequencies, target.abscissa,
                                             target.ordinate[0]))
    got = compliance.compare(target, matched)
    assert got['difference_percent'] == pytest.approx(0.0, abs=1e-6)
    assert got['specification_rms'] == pytest.approx(got['measured_rms'])


def test_twice_the_specification_is_forty_one_percent_over():
    """Power doubles; RMS is its square root, so the level a report
    quotes moves by root two and not by two."""
    frequencies = np.linspace(10.0, 1000.0, 200)
    target = spec()
    doubled = psd(frequencies, 2.0 * compliance.log_interpolate(
        frequencies, target.abscissa, target.ordinate[0]))
    # held unscaled: left to detect, the comparison reads a doubled
    # measurement as a run commanded 3 dB up and lays it back on the
    # specification — which is the feature, and the second half checks it
    got = compliance.compare(target, doubled, scale_db=0)
    assert got['difference_percent'] == pytest.approx(
        100.0 * (np.sqrt(2.0) - 1.0), rel=1e-6)
    detected = compliance.compare(target, doubled)
    assert detected['scale_db'] == -3.0, 'auto-detected, and said so'
    assert detected['difference_percent'] == pytest.approx(0.0, abs=0.5)


def test_only_the_lines_inside_the_band_are_counted():
    """The measurement runs to 2000 Hz and the specification stops at
    1000; the half above it is not a pass and not a failure."""
    frequencies = np.linspace(10.0, 2000.0, 400)
    target = spec()
    measured = psd(frequencies, np.full(400, 1e-3))
    got = compliance.compare(target, measured)
    # the band is what the compared bins cover, and a bin reaches half a
    # width either side of its line
    assert got['band'] == pytest.approx((10.0, 1000.0), abs=6.0)
    assert got['lines'] < 400
    assert got['lines'] >= int(np.sum(frequencies <= 1000.0)) - 2


def test_the_two_need_not_share_a_frequency_axis():
    """A specification written at breakpoints, against an analysis at
    whatever resolution its frames gave."""
    target = spec(frequencies=[10.0, 100.0, 1000.0])
    frequencies = np.linspace(10.0, 1000.0, 991)
    measured = psd(frequencies, np.full(991, 1e-3))
    got = compliance.compare(target, measured)
    assert got['lines'] >= 989, 'all but the bins straddling the two ends'
    assert got['difference_percent'] == pytest.approx(0.0, abs=1e-6)


# ---- the limits ---------------------------------------------------------


def test_lines_outside_each_pair_are_counted_separately():
    """Warning and abort are different questions, and a report answers
    both: a run can sit inside abort throughout and still spend a third
    of its band on warning."""
    frequencies = np.linspace(20.0, 900.0, 100)     # well inside the band
    target = spec(warning_lower=0.5, warning_upper=2.0,
                  abort_lower=0.25, abort_upper=4.0)
    values = np.full(100, 1e-3)
    values[:30] = 3e-3          # past warning, inside abort
    values[30:40] = 5e-3        # past both
    got = compliance.compare(target, psd(frequencies, values))
    assert got['warning_lines'] == 40
    assert got['abort_lines'] == 10
    assert got['warning_percent'] == pytest.approx(40.0)
    assert got['abort_percent'] == pytest.approx(10.0)


def test_under_the_lower_limit_counts_too():
    """An under-test is as much a failure as an over-test, and reading
    only the upper limit is how one goes unreported."""
    frequencies = np.linspace(20.0, 900.0, 100)     # well inside the band
    target = spec(abort_lower=0.25, abort_upper=4.0)
    values = np.full(100, 1e-3)
    values[:20] = 1e-5          # far under
    got = compliance.compare(target, psd(frequencies, values))
    assert got['abort_lines'] == 20


def test_a_limit_that_was_never_written_is_absent():
    """Not zero: nothing was exceeded because nothing was asked, and a
    table saying '0 % outside abort' for a specification with no abort
    limit is a claim it cannot support."""
    frequencies = np.linspace(20.0, 900.0, 100)
    got = compliance.compare(spec(warning_upper=2.0),
                             psd(frequencies, np.full(100, 1e-3)))
    assert 'warning_percent' in got
    assert 'abort_percent' not in got and 'abort_lines' not in got


def test_which_lines_went_out_and_which_way():
    """What the plot marks. A percentage says how much; this says
    where."""
    frequencies = np.linspace(20.0, 900.0, 100)     # well inside the band
    target = spec(abort_lower=0.25, abort_upper=4.0)
    values = np.full(100, 1e-3)
    values[10:15] = 9e-3        # over
    values[80:83] = 1e-5        # under
    over, under = compliance.exceedances(target, psd(frequencies, values))
    assert list(np.flatnonzero(over)) == list(range(10, 15))
    assert list(np.flatnonzero(under)) == list(range(80, 83))


def test_nothing_in_common_is_no_comparison():
    """A specification over 10-1000 Hz and an analysis over 2000-3000
    share no line, and inventing a number from that would be worse than
    saying there is none."""
    got = compliance.compare(spec(),
                             psd(np.linspace(2000.0, 3000.0, 50),
                                 np.full(50, 1e-3)))
    assert got == {'lines': 0}


# ---- the real pair ------------------------------------------------------


def test_the_plate_run_against_its_own_specification():
    """The comparison the fixture exists to make, end to end."""
    from conftest import fixture_path

    import visualdynamics

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    target = loaded['Random_specification']
    measured = loaded['time_data'].compute_psds()
    si = next(i for i in range(target.num_records)
              if target.response_dof[i] == '101Z+')
    mi = next(i for i in range(measured.num_records)
              if measured.response_dof[i] == '101Z+')
    got = compliance.compare(target, measured, si, mi)
    assert got['lines'] > 100
    # the band is the outer edges of the written lines' bins: 50 Hz
    # lines two hertz apart reach a hertz either side
    assert got['band'][0] >= 49.0 and got['band'][1] <= 1501.0
    assert got['specification_rms'] > 0
    assert got['measured_rms'] > 0
    assert np.isfinite(got['difference_percent'])
    assert 0.0 <= got['abort_percent'] <= 100.0


def test_a_limit_reaching_past_the_specification_counts_only_inside():
    """A specification written over 100-1000 Hz whose limit arrays were
    filled in over the whole axis. What is counted has to be what was
    compared, or the percentage has a denominator it does not share with
    its numerator and can exceed a hundred."""
    frequencies = np.array([10.0, 100.0, 1000.0, 5000.0])
    ordinate = np.array([[0.0, 1e-3, 1e-3, 0.0]])       # band is 100-1000
    limits = np.array([[1e-6, 2e-3, 2e-3, 1e-6]])       # written everywhere
    target = Specification(
        abscissa=frequencies, ordinate=ordinate, response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'], abort_upper=limits)
    lines = np.linspace(20.0, 4000.0, 400)              # evenly, across it all
    measured = psd(lines, np.full(400, 1.0))            # over everything
    # scale held at zero: what is under test is the band geometry, and
    # detection would first lay this flat line back onto the target
    got = compliance.compare(target, measured, scale_db=0)
    inside = int(np.sum((lines > 100.0) & (lines < 1000.0)))
    assert abs(got['lines'] - inside) <= 2, (got['lines'], inside)
    assert got['abort_lines'] == got['lines'], 'all of them are out'
    assert got['abort_percent'] == pytest.approx(100.0)


def test_a_bin_the_band_edge_cuts_is_judged_on_the_part_inside():
    """At the two ends a measured bin hangs over the edge, covered by
    the specification for part of its width and by nothing for the
    rest. Reading the limit at the bin's center answers the wrong
    question twice: a bin centered just outside is dropped though most
    of it is in, and a bin centered just inside is judged over width the
    specification never covered.

    So a cut bin is compared over the covered part alone — the power
    the measurement holds there against the power the limit asks for
    over the same stretch. Here a response 1.5x over its abort limit is
    1.5x over it on every bin the band touches, which is the answer
    that does not depend on where the lines happened to land.
    """
    target = spec(frequencies=[10.0, 20.0], level=1.0, abort_upper=2.0)
    lines = np.arange(9.1, 21.0, 1.0)      # centers straddle both edges
    measured = psd(lines, np.full(lines.size, 3.0))
    _start, _stop, width, cut = compliance.covered(lines, 10.0, 20.0)
    assert cut.sum() == 2, 'one bin cut at each end'
    assert width[cut] == pytest.approx([0.6, 0.4])

    # scale held at zero: the subject is the cut bins, and detection
    # would first lay the 1.5x response back onto its target
    got = compliance.compare(target, measured, scale_db=0)
    assert got['abort_lines'] == got['lines'], (
        'every bin the band reaches is 1.5x over the limit over the part '
        'of it that is covered, cut bins included')
    assert got['abort_percent'] == pytest.approx(100.0)


def test_the_cut_bin_at_the_edge_is_counted_where_it_is_compared():
    """The failure the rule above fixes: the line past the edge is in
    the denominator either way — its bin is part of the compared band —
    so leaving it unjudgeable put it in `lines` and never in
    `abort_lines`, and the percentage read low by construction."""
    target = spec(frequencies=[10.0, 20.0], level=1.0, abort_upper=2.0)
    lines = np.arange(9.1, 21.0, 1.0)
    over = compliance.outside(lines, np.full(lines.size, 3.0),
                              target.abscissa,
                              target.limits['abort_upper'][0], over=True)
    past = int(np.flatnonzero(lines > 20.0)[0])
    assert lines[past] == pytest.approx(20.1)
    assert not np.isfinite(compliance.log_interpolate(
        lines[past:past + 1], target.abscissa,
        target.limits['abort_upper'][0])[0]), 'the center is outside'
    assert over[past], 'but 0.4 Hz of its bin is inside, and that part is out'
    assert not over[0], 'a bin wholly outside stays out of it'


def test_a_cut_bin_under_the_limit_is_not_flagged():
    """The rule has to be able to say no at an edge too, or it is just
    marking the ends of every comparison."""
    target = spec(frequencies=[10.0, 20.0], level=1.0, abort_upper=2.0)
    lines = np.arange(9.1, 21.0, 1.0)
    over = compliance.outside(lines, np.full(lines.size, 1.5),
                              target.abscissa,
                              target.limits['abort_upper'][0], over=True)
    assert not over.any(), 'under the limit everywhere, edges included'


def test_a_specifications_level_does_not_move_with_the_analysis():
    """The whole reason it is worked out from its own points. An
    analysis that covers the specification compares over the
    specification's own band whatever its resolution, so the level
    asked for is the same number every time — and equal to the exact
    area under the breakpoint curve."""
    target = spec(frequencies=[20.0, 80.0, 350.0, 2000.0], level=1.0)
    target.ordinate[0] = [0.01, 0.16, 0.16, 0.005]     # +6, flat, -6 dB/oct
    exact = np.sqrt(compliance.log_log_area(target.abscissa,
                                            target.ordinate[0]))
    for n in (60, 400, 4000, 20000):
        lines = np.linspace(5.0, 3000.0, n)            # right past both ends
        measured = psd(lines, np.full(n, 0.1))
        got = compliance.compare(target, measured)
        assert got['specification_rms'] == pytest.approx(exact, rel=1e-12)
        assert got['band'] == pytest.approx((20.0, 2000.0), rel=1e-12)


def test_a_band_cut_short_by_the_analysis_says_so():
    """It is not invariant when it cannot be: an analysis stopping at
    500 Hz has compared a specification only up to 500 Hz, and the level
    it quotes is the level over what was actually compared."""
    target = spec(frequencies=[20.0, 80.0, 350.0, 2000.0], level=1.0)
    target.ordinate[0] = [0.01, 0.16, 0.16, 0.005]
    lines = np.linspace(100.0, 500.0, 2000)
    got = compliance.compare(target, psd(lines, np.full(2000, 0.1)))
    assert got['band'][1] == pytest.approx(500.1, abs=0.2)
    assert got['specification_rms'] < np.sqrt(compliance.log_log_area(
        target.abscissa, target.ordinate[0]))


def test_the_exact_area_under_a_power_law():
    """Closed form, not a rule on a grid. A segment falling as 1/f is
    the case a naive integral gets wrong, and its area is analytic."""
    # W = C/f from 10 to 1000: area is C * ln(100)
    got = compliance.log_log_area([10.0, 1000.0], [1.0, 0.01])
    assert got == pytest.approx(10.0 * np.log(100.0), rel=1e-12)
    # flat: just the width
    assert compliance.log_log_area([10.0, 1000.0], [2.0, 2.0]) == (
        pytest.approx(2.0 * 990.0, rel=1e-12))


def test_a_band_that_cuts_a_segment_cuts_it_on_the_same_curve():
    """Half of a power-law segment is not half of its area, and the
    value where the band cuts comes from the law itself."""
    whole = compliance.log_log_area([10.0, 1000.0], [1.0, 0.01])
    lower = compliance.log_log_area([10.0, 1000.0], [1.0, 0.01],
                                    low=10.0, high=100.0)
    upper = compliance.log_log_area([10.0, 1000.0], [1.0, 0.01],
                                    low=100.0, high=1000.0)
    assert lower + upper == pytest.approx(whole, rel=1e-12)
    assert lower == pytest.approx(10.0 * np.log(10.0), rel=1e-12)


def test_the_area_matches_a_fine_numerical_integral():
    """Held against something that does not share its reasoning."""
    breaks = np.array([20.0, 80.0, 350.0, 2000.0])
    levels = np.array([0.01, 0.16, 0.16, 0.005])
    fine = np.geomspace(20.0, 2000.0, 200001)
    dense = compliance.log_interpolate(fine, breaks, levels)
    assert compliance.log_log_area(breaks, levels) == pytest.approx(
        np.trapezoid(dense, fine), rel=1e-6)


def test_the_measurement_is_integrated_over_the_specifications_band_only():
    """Both sides cover one identical stretch of frequency.

    A measurement usually runs far wider than the specification it
    answers — to 4 kHz against a band that stops at 1 kHz — and
    counting that into its RMS while the specification's own RMS stops
    at 1 kHz would report every channel as over-tested.

    The only out-of-band data that enters is the part of a *straddling*
    bin that lies inside the band, which is the endpoint rule: a bin
    half in and half out is counted for the half that is in. On a run
    whose band edges land on measured lines, not even that.
    """
    target = spec(frequencies=[100.0, 1000.0])
    lines = np.linspace(10.0, 4000.0, 2000)
    inside = compliance.log_interpolate(lines, target.abscissa,
                                        target.ordinate[0])
    matched = np.where(np.isfinite(inside), inside, 0.0)

    quiet = compliance.compare(target, psd(lines, matched))
    assert quiet['band'] == pytest.approx((100.0, 1000.0), abs=2.0)

    # ten times the specification everywhere outside its band
    loud = matched.copy()
    loud[~np.isfinite(inside)] = 1e-2
    got = compliance.compare(target, psd(lines, loud))
    assert got['specification_rms'] == pytest.approx(
        quiet['specification_rms'], rel=1e-12), 'the target never moves'
    # only the two straddling bins carry any of it, so the level barely
    # moves — where counting the whole 3 kHz outside would double it
    assert abs(got['difference_db'] - quiet['difference_db']) < 0.2


def test_a_band_edge_on_a_measured_line_takes_nothing_from_outside():
    """The usual case, and then the guarantee is exact."""
    target = spec(frequencies=[100.0, 1000.0])
    lines = np.arange(0.0, 4000.0 + 1.0, 1.0)      # 100 and 1000 are lines
    inside = compliance.log_interpolate(lines, target.abscissa,
                                        target.ordinate[0])
    matched = np.where(np.isfinite(inside), inside, 0.0)
    outside = matched.copy()
    outside[~np.isfinite(inside)] = 1.0            # a thousand times over
    assert compliance.compare(target, psd(lines, matched))['measured_rms'] == \
        pytest.approx(
            compliance.compare(target, psd(lines, outside))['measured_rms'],
            rel=1e-12)


def test_the_end_bins_of_the_measurement_count_for_half():
    """Brandon's own reading of it, and the right one: the two sides
    are areas under a curve over one shared band, so a bin sitting
    astride the band edge brings the half of it that is inside.

    With the first and last lines *on* the edges, that is exactly half
    a bin each — and the two RMS levels then agree to the last digit
    for a response that matches its specification.
    """
    target = spec(frequencies=[20.0, 2000.0])
    lines = np.arange(20.0, 2000.0 + 0.25, 0.5)     # ends on both edges
    values = np.full(lines.size, 1e-3)

    low, high = compliance.band_of(target, 0)
    widths = np.gradient(lines)
    left, right = lines - widths / 2, lines + widths / 2
    overlap = np.clip(np.minimum(right, high) - np.maximum(left, low), 0, None)
    assert overlap[0] == pytest.approx(widths[0] / 2)
    assert overlap[-1] == pytest.approx(widths[-1] / 2)
    assert overlap[100] == pytest.approx(widths[100]), 'and the rest whole'
    assert overlap.sum() == pytest.approx(high - low), 'exactly the band'

    got = compliance.compare(target, psd(lines, values))
    assert got['measured_rms'] == pytest.approx(got['specification_rms'],
                                                rel=1e-12)
    assert got['difference_db'] == pytest.approx(0.0, abs=1e-12)


def test_counting_the_end_bins_whole_would_read_high():
    """Which is why they are not: two half-bins of extra width is a
    small error, and it is one that never cancels — it is always over.
    """
    target = spec(frequencies=[20.0, 2000.0])
    lines = np.arange(20.0, 2000.0 + 0.25, 0.5)
    values = np.full(lines.size, 1e-3)
    got = compliance.compare(target, psd(lines, values))
    whole = np.sqrt(1e-3 * (np.gradient(lines).sum()))
    assert whole > got['measured_rms'], 'the whole-bin reading is higher'


# --- how far the view may be pulled back -----------------------------

def _chart(rows, low, high=None):
    """(widget, plot) — the widget is handed back so the caller keeps it
    alive. Dropped, Qt deletes the C++ side and every later call on the
    plot raises 'Internal C++ object already deleted'."""
    import pyqtgraph as pg

    from visualdynamics.plot.bars import BarChart
    from visualdynamics.theme import theme

    widget = pg.GraphicsLayoutWidget()
    plot = widget.addPlot(row=0, col=0)
    BarChart(plot, rows, theme('dark'), low=low, high=high)
    return widget, plot


def test_a_bar_chart_cannot_be_zoomed_out_into_empty_ground(qt_app):
    """There is nothing to find by pulling back from a bar chart.

    Every bar starts at zero and the longest one is the whole story, so
    the far side of the axis says only how much nothing there is —
    unfenced, a scroll wheel takes the bars down to a row of specks
    against a decade of blank and getting back means guessing.
    """
    from visualdynamics.plot.bars import ZOOM_MARGIN

    rows = [('101Z+', 31.7), ('121Z+', 41.2), ('223Z+', 56.9)]
    _held, plot = _chart(rows, low=20.0)
    view = plot.getViewBox()
    view.setRange(xRange=(-10_000, 10_000), padding=0)
    low, high = view.viewRange()[0]
    assert high == pytest.approx(56.9 * (1 + ZOOM_MARGIN), rel=1e-6)
    assert low > -56.9 * ZOOM_MARGIN, 'a one-sided reading needs no room below'
    assert low <= 0.0, 'and zero must stay inside — a bar is drawn from it'


def test_zooming_in_stays_free(qt_app):
    """Reading a cluster of near-identical bars apart is a real thing
    to want, so only pulling *back* is fenced."""
    rows = [('101Z+', 31.7), ('121Z+', 32.0), ('223Z+', 32.2)]
    _held, plot = _chart(rows, low=20.0)
    view = plot.getViewBox()
    view.setRange(xRange=(31.5, 32.5), padding=0)
    low, high = view.viewRange()[0]
    assert (low, high) == pytest.approx((31.5, 32.5), abs=1e-6)


def test_a_two_sided_reading_is_fenced_both_ways(qt_app):
    """A dB error runs either side of zero, so the fence does too."""
    rows = [('101Z+', -7.8), ('223Z+', 4.0)]
    _held, plot = _chart(rows, low=-3.0, high=3.0)
    view = plot.getViewBox()
    view.setRange(xRange=(-10_000, 10_000), padding=0)
    low, high = view.viewRange()[0]
    assert low < 0.0 < high
    assert low >= -7.8 * 1.2 and high <= 7.8 * 1.2


def test_the_threshold_counts_toward_the_reach(qt_app):
    """A threshold dragged past every bar must stay on screen — it is
    the thing the bars are being read against."""
    rows = [('101Z+', 2.0), ('121Z+', 3.0)]
    _held, plot = _chart(rows, low=80.0)
    view = plot.getViewBox()
    view.setRange(xRange=(-10_000, 10_000), padding=0)
    assert view.viewRange()[0][1] >= 80.0
    assert _held is not None


# --- a dense specification -------------------------------------------

def test_the_area_survives_a_steep_segment():
    """A specification is not always a handful of breakpoints.

    The PSD of a transient target is a Specification with thousands of
    lines a fraction of a hertz apart, and neighboring lines two
    decades apart over a frequency ratio of 1.00025 give a power-law
    exponent near twenty thousand. Worked out as `a**n` that overflowed,
    the subtraction went inf - inf, and every channel's RMS error came
    back NaN.
    """
    from visualdynamics.core.compliance import log_log_area

    frequencies = np.array([1000.0, 1000.25, 1000.5])
    values = np.array([1e-6, 1e-2, 1e-6])
    area = log_log_area(frequencies, values)
    assert np.isfinite(area), 'a steep segment is not an infinite one'
    assert 0.0 < area < 1.0
    # and the closed form is still exact where it always was: a flat
    # segment is a rectangle
    flat = log_log_area(np.array([10.0, 20.0]), np.array([2.0, 2.0]))
    assert flat == pytest.approx(2.0 * 10.0)
    # a -1 slope integrates to W1·f1·ln(f2/f1)
    decade = log_log_area(np.array([10.0, 100.0]), np.array([1.0, 0.1]))
    assert decade == pytest.approx(1.0 * 10.0 * np.log(10.0))


def test_a_dense_specification_gives_a_finite_rms():
    """What the transient path actually hit."""
    from visualdynamics.core.compliance import specification_rms
    from visualdynamics.core.data import Specification

    rng = np.random.default_rng(0)
    frequencies = np.arange(1, 2049) * 0.25
    # four decades of scatter, line to line — a measured spectrum
    values = 10.0 ** rng.uniform(-8, -2, size=frequencies.size)
    spec = Specification(frequencies, values[None, :],
                         response_dof=['101Z+'],
                         ordinate_dim='acceleration**2/frequency')
    assert np.isfinite(specification_rms(spec, 0))


# --- one field, one integral, one picture ----------------------------

def test_a_spectrum_is_integrated_the_way_it_is_drawn():
    """The property the structure exists to guarantee.

    A spectrum drawn one way and integrated the other is a picture of a
    number nobody computed. They used to be chosen separately — the
    drawing by class, the integral by whichever function the caller
    reached for — so nothing made them agree, and the PSD of a
    transient target was drawn as a power law through four thousand
    density lines while its level came from their area.
    """
    from visualdynamics.plot import bin_edges

    frequencies = np.arange(10.0, 21.0)
    values = np.linspace(1.0, 4.0, len(frequencies))
    psd = Psd(frequencies, values[None, :], response_dof=['1Z+'],
              ordinate_dim='acceleration**2/frequency')

    # read as bins, the picture is a staircase and its area is the sum
    # of the rectangles the plot actually paints
    assert psd.interpolation == 'bin'
    edges = bin_edges(psd.abscissa, psd.bandwidth)
    painted = float(np.sum(np.diff(edges) * values))
    assert psd.area(0) == pytest.approx(painted)

    # read as a power law, it is the curve drawn between the points
    psd.interpolation = 'log_log'
    assert psd.area(0) == pytest.approx(
        compliance.log_log_area(frequencies, values))
    assert psd.area(0) != pytest.approx(painted), (
        'the two readings are different numbers, which is the point'
    )


def test_the_object_owns_the_integral_and_no_caller_chooses():
    """`specification_rms` used to reach straight for `log_log_area`,
    which is right for a specification written at breakpoints and wrong
    for one computed from a record."""
    from visualdynamics.core.compliance import specification_rms

    frequencies = np.arange(10.0, 21.0)
    values = np.full(len(frequencies), 2.0)
    spec = Specification(frequencies, values[None, :],
                         response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency')
    assert spec.interpolation == 'log_log'
    # flat 2.0 from 10 to 20 Hz, as a curve between breakpoints: 2 x 10
    assert specification_rms(spec) == pytest.approx(np.sqrt(20.0))
    # the same numbers read as bins are eleven of them, not ten
    spec.interpolation = 'bin'
    assert specification_rms(spec) == pytest.approx(np.sqrt(22.0))


def test_a_band_narrows_both_readings():
    frequencies = np.arange(10.0, 21.0)
    psd = Psd(frequencies, np.ones((1, len(frequencies))),
              response_dof=['1Z+'],
              ordinate_dim='acceleration**2/frequency')
    assert psd.area(0, 12.0, 18.0) == pytest.approx(6.0)
    psd.interpolation = 'log_log'
    assert psd.area(0, 12.0, 18.0) == pytest.approx(6.0)


def test_a_spectrum_that_is_not_a_density_has_no_reading():
    """An SRS is a value at a frequency, not an area under anything, so
    the field is absent rather than holding some third value — and the
    plot draws it as the line it is."""
    from visualdynamics.core.data import Coherence, Frf, Srs

    for cls in (Srs, Frf, Coherence):
        assert getattr(cls, 'interpolation', None) is None


def test_the_exported_report_steps_where_the_app_steps(tmp_path):
    """The deliverable has to show the number it would compute too.

    `steps` used to be set only for a specification *with limits*,
    which was wrong both ways round: a measured PSD drew as a polyline
    through its own bin centers, and a written specification — whose
    points are breakpoints of a curve — drew as a staircase it is not.
    """
    from conftest import fixture_path

    import visualdynamics
    from visualdynamics.report import _build_block

    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'random.nc4'))
    history = next(n for n, o in project.items()
                   if type(o).__name__ == 'TimeHistory')
    measured = project.compute_psds(history)
    written = next(n for n, o in project.items()
                   if type(o) is Specification)

    def block(name):
        return _build_block({'kind': 'plot', 'source': name,
                             'mode': 'curves', 'caption': ''},
                            dict(project), visualdynamics.SI, [])

    assert project[measured].interpolation == 'bin'
    assert block(measured)['steps'] is True
    # the controller's target on its lines is a density per line and
    # steps too (2026-09-19); a written breakpoint curve is the shape
    # that does not
    assert project[written].interpolation == 'bin'
    assert block(written)['steps'] is True
    breakpoints = project.add('Written', Specification(
        np.array([20.0, 80.0, 800.0, 2000.0]),
        np.array([[1e-3, 4e-3, 4e-3, 1e-3]]), response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'], ordinate_unit=['m/s**2']))
    assert project[breakpoints].interpolation == 'log_log'
    assert block(breakpoints).get('steps', False) is False


def test_a_limit_is_read_the_way_its_specification_is():
    """A limit belongs to a specification and is written the way it is,
    so it is integrated the same way — reaching straight for the
    log-log form is the assumption that had a density drawn as a power
    law through its own lines."""
    frequencies = np.arange(10.0, 21.0)
    spec = Specification(frequencies, np.full((1, len(frequencies)), 2.0),
                         response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency',
                         abort_upper=np.full((1, len(frequencies)), 4.0))
    assert spec.limit('abort_upper').interpolation == spec.interpolation
    spec.interpolation = 'bin'
    assert spec.limit('abort_upper').interpolation == 'bin'


def test_a_banded_comparison_is_judged_on_its_bins_own_edges():
    """The blue last band (Brandon, 2026-09-19): a response squarely on
    its banded target was marked under the lower abort limit in the
    last octave band. Two readings were wrong at once. The written
    band was taken from the band centers, so the last band was "cut"
    at its own center; and a cut bin's covered stretch was center
    minus half a width, which for a geometric bin reaches a couple of
    hertz into the band below — whose limit was far higher — and that
    sliver flipped the verdict. A banded comparison is bin against
    bin: the written band runs to the outer edges of the written
    bins, and a bin is its own edges."""
    from conftest import banded_pair_on_target

    from visualdynamics.core.compliance import (
        cells,
        covered,
        exceedances,
        matched_records,
        outside,
    )
    from visualdynamics.core.octave import bin_bounds

    banded, measured = banded_pair_on_target()
    for label, si, mi in matched_records(banded, measured):
        over, under = exceedances(banded, measured, si, mi, 'abort')
        assert not over.any() and not under.any(), label
    # the cells are the bands themselves — the two grids are one —
    # each whole, the two ends included: an octave band is a defined
    # band and is judged as one
    centers = np.asarray(banded.abscissa)
    lower = np.real(banded.limits['abort_lower'][0])
    written = np.flatnonzero(np.isfinite(lower) & (lower > 0))
    left, right = bin_bounds(centers, banded.bandwidth)
    found = cells(banded, measured, 0, matched_records(banded, measured)[0][2])
    assert len(found) == written.size
    for cell, k in zip(found, written):
        assert cell['pieces'] == [pytest.approx((left[k], right[k]))]
    # a bin cut by a band edge is cut at its own geometric edges
    band = (left[written[0]], right[written[-1]])
    start, stop, _width, cut = covered(centers, *band, banded.bandwidth)
    assert not cut.any(), 'the band ends on bin edges: nothing is cut'
    inside = slice(written[0], written[-1] + 1)
    assert start[inside] == pytest.approx(left[inside])
    assert stop[inside] == pytest.approx(right[inside])
    # and the array form, told the reading — a density per band of
    # these widths — is clean on the same bins too
    _label, _si, mi = matched_records(banded, measured)[0]
    plain = outside(np.asarray(measured.abscissa), np.real(measured.ordinate[mi]),
                    centers, lower, over=False, reading='bin',
                    spec_widths=banded.bandwidth, widths=measured.bandwidth)
    assert not plain.any()
