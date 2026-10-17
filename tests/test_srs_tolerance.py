"""A shock response spectrum judged point by point against its tolerance.

A shock test is accepted on a band in dB around the required SRS and a
rule about how many of the spectrum's points lie inside it. These pin
that judgement by counts that are known from the construction of each
case (a spectrum that is its requirement, one shifted a known number of
decibels, one with a known number of points moved out), and pin the
case the RMS reading cannot see: a narrow, deep dip that leaves the
RMS deviation well inside its threshold while the points rule fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.compliance import (
    SRS_ERROR_DB,
    SRS_INNER_DB,
    srs_errors,
    srs_points,
    srs_tolerance,
    srs_tolerance_all,
)
from visualdynamics.core.data import ShockSpecification, Srs

#: twelfth-octave points from 10 Hz to the last at or below 2 kHz
POINTS = 92


def requirement(**bands):
    """A slope to a knee at 100 Hz and a plateau to 2 kHz, 10 to 100 g
    at Q = 10, on its twelfth-octave points."""
    return ShockSpecification.from_breakpoints(
        [10.0, 100.0, 2000.0], [10.0, 100.0, 100.0], '101Z+',
        ordinate_unit='g', **bands)


def measured(spec, db=0.0, kind='maximax', q=None, dof='101Z+',
             block=None):
    """The requirement shifted by `db` at every point, or by an array
    of `db`, one per point, as an Srs."""
    values = spec.ordinate[0] * 10.0 ** (np.asarray(db, dtype=float) / 20.0)
    return Srs(spec.abscissa, values[None], response_dof=[dof],
               ordinate_dim=['acceleration'], kind=kind,
               q=spec.q if q is None else q, block=block)


# ---- the four cases the rule is written for ------------------------------


def test_a_spectrum_equal_to_its_requirement_is_wholly_inside():
    spec = requirement()
    result = srs_tolerance(spec, measured(spec))
    assert result['points'] == POINTS
    assert result['inside_count'] == POINTS
    assert result['inside_percent'] == 100.0
    assert result['passed'] is True
    assert result['worst_margin_db'] < 0.0       # inside, with room


def test_two_db_over_everywhere_is_inside_the_inner_band():
    """+2 dB is under the inner band's +3 dB ceiling at every point."""
    spec = requirement()
    result = srs_tolerance(spec, measured(spec, 2.0))
    assert result['inside_count'] == POINTS
    assert result['over_count'] == 0
    assert result['passed'] is True
    assert result['worst_db'] == pytest.approx(2.0)
    assert result['worst_margin_db'] == pytest.approx(-1.0)


def test_two_db_under_everywhere_is_out_at_every_point_and_fails():
    """-2 dB is past the inner band's -1.5 dB floor at every point."""
    spec = requirement()
    result = srs_tolerance(spec, measured(spec, -2.0))
    assert result['inside_count'] == 0
    assert result['under_count'] == POINTS
    assert result['under'].sum() == POINTS
    assert result['passed'] is False
    assert result['worst_db'] == pytest.approx(-2.0)
    assert result['worst_margin_db'] == pytest.approx(0.5)


def test_ten_percent_outside_passes_at_ninety_and_fails_at_ninety_five():
    """Forty points, four of them moved 2 dB under: exactly 90 percent
    inside, which meets a 90 percent rule and not a 95."""
    f = np.geomspace(10.0, 2000.0, 40)
    level = np.full(40, 100.0)
    shifted = level.copy()
    shifted[[3, 11, 22, 37]] *= 10.0 ** (-2.0 / 20.0)
    at_ninety = srs_points(f, shifted, f, level, percent=90.0)
    assert at_ninety['inside_count'] == 36
    assert at_ninety['inside_percent'] == pytest.approx(90.0)
    assert at_ninety['passed'] is True
    assert srs_points(f, shifted, f, level, percent=95.0)['passed'] is False
    # and one point fewer inside is under ninety
    shifted[5] *= 10.0 ** (-2.0 / 20.0)
    assert srs_points(f, shifted, f, level, percent=90.0)['passed'] is False


def test_a_narrow_deep_dip_passes_the_rms_reading_and_fails_the_points_rule():
    """Ten neighboring points 5 dB under, the rest on the requirement:
    the signed RMS is under 2 dB, inside the 3 dB the deviation bars
    shade at, while ten of 92 points out is 89 percent inside, under
    the 90 percent rule. The two readings answer different questions,
    and only the count is the tolerance check."""
    spec = requirement()
    db = np.zeros(POINTS)
    db[40:50] = -5.0
    dipped = measured(spec, db)
    (_dof, _event, rms_db), = srs_errors(spec, dipped)
    assert abs(rms_db) < SRS_ERROR_DB
    assert rms_db == pytest.approx(-np.sqrt(10 * 25.0 / POINTS))
    result = srs_tolerance(spec, dipped)
    assert result['inside_count'] == POINTS - 10
    assert result['passed'] is False
    assert result['worst_db'] == pytest.approx(-5.0)
    assert 40 <= np.flatnonzero(spec.abscissa == result['worst_frequency'])[0] < 50


# ---- the edges of the rule ------------------------------------------------


def test_a_point_on_a_bands_edge_is_inside():
    """+3 dB and -1.5 dB exactly, as a spectrum written as the
    requirement times the band's factor reads in floating point."""
    spec = requirement()
    for db in SRS_INNER_DB:
        assert srs_tolerance(spec, measured(spec, db))['inside_count'] == POINTS


def test_a_point_on_the_outer_bands_edge_is_inside_it():
    spec = requirement()
    db = np.zeros(POINTS)
    db[5], db[50] = -3.0, 6.0
    result = srs_tolerance(spec, measured(spec, db), outer=(-3.0, 6.0))
    assert result['outer_count'] == 0
    assert result['inside_count'] == POINTS - 2
    assert result['passed'] is True


def test_the_outer_band_holds_every_point_whatever_the_share():
    """91 of 92 inside the inner band, and the one outside past the
    outer band: the share is met and the spectrum still fails."""
    spec = requirement()
    db = np.zeros(POINTS)
    db[60] = 7.0
    result = srs_tolerance(spec, measured(spec, db), outer=(-3.0, 6.0))
    assert result['inside_count'] == POINTS - 1
    assert result['outer_count'] == 1
    assert result['beyond_outer'][60]
    assert result['passed'] is False
    assert srs_tolerance(spec, measured(spec, db))['passed'] is True


def test_the_above_lower_rule_puts_no_ceiling_on_the_band():
    """+5 dB everywhere is above the inner band and fails a share of
    it, yet every point is above the lower edge; one point under the
    lower edge fails that rule outright."""
    spec = requirement()
    high = measured(spec, 5.0)
    assert srs_tolerance(spec, high)['passed'] is False
    above = srs_tolerance(spec, high, rule='above_lower')
    assert above['passed'] is True
    # the band has no ceiling under this rule, so every point at or
    # above the lower edge is inside it and none is over
    assert above['inside_count'] == POINTS
    assert above['over_count'] == 0
    assert above['inner'] == (-1.5, np.inf)
    assert srs_points(spec.abscissa, high.ordinate[0], spec.abscissa,
                      spec.ordinate[0], rule='above_lower')[
        'inside_count'] == POINTS
    db = np.zeros(POINTS)
    db[0] = -1.6
    one_under = srs_tolerance(spec, measured(spec, db), rule='above_lower')
    assert one_under['passed'] is False
    assert one_under['worst_frequency'] == pytest.approx(10.0)
    assert one_under['worst_margin_db'] == pytest.approx(0.1)


def test_a_share_above_the_requirement_is_the_share_rule_from_zero_up():
    """Half the points above the requirement, read as the share rule
    with a band from 0 dB up."""
    spec = requirement()
    db = np.where(np.arange(POINTS) % 2 == 0, 0.5, -0.5)
    result = srs_tolerance(spec, measured(spec, db), inner=(0.0, np.inf),
                           percent=50.0)
    assert result['inside_count'] == POINTS // 2
    assert result['passed'] is True
    assert srs_tolerance(spec, measured(spec, -0.5), inner=(0.0, np.inf),
                         percent=50.0)['passed'] is False


def test_the_worst_point_is_the_farthest_outside_not_the_farthest_away():
    """Under -1.5/+3 dB, a point 2.5 dB over is inside and one 2 dB
    under is not; the worst is the one under."""
    spec = requirement()
    db = np.zeros(POINTS)
    db[10] = 2.5
    db[70] = -2.0
    result = srs_tolerance(spec, measured(spec, db))
    assert result['worst_frequency'] == pytest.approx(spec.abscissa[70])
    assert result['worst_db'] == pytest.approx(-2.0)


def test_only_points_inside_the_requirement_are_judged():
    """A measurement that starts a decade below the requirement and
    runs past its end is counted only where the requirement is
    written; the points outside are neither in nor out."""
    f = np.geomspace(1.0, 5000.0, 60)
    level = np.full(60, 100.0)
    level[f < 10.0] = 1e-3              # far under, but not required
    result = srs_points(f, level, [10.0, 2000.0], [100.0, 100.0])
    written = (f >= 10.0) & (f <= 2000.0)
    assert result['points'] == int(written.sum())
    assert np.array_equal(result['judged'], written)
    assert result['passed'] is True
    assert np.isnan(result['db'][~written]).all()


def test_a_point_measured_at_nothing_is_not_judged():
    """As the RMS reading skips it: a zero peak is no reading at all,
    not a point an infinite number of decibels under."""
    f = np.geomspace(10.0, 2000.0, 20)
    level = np.full(20, 100.0)
    got = level.copy()
    got[7] = 0.0
    result = srs_points(f, got, f, level)
    assert result['points'] == 19
    assert not result['judged'][7]
    assert result['under_count'] == 0


def test_the_whole_range_of_shares_can_be_asked_for():
    """Every point (100 percent) and none (0 percent) are both rules."""
    f = np.geomspace(10.0, 2000.0, 20)
    level = np.full(20, 100.0)
    got = level.copy()
    got[3] = 50.0
    assert srs_points(f, level, f, level, percent=100.0)['passed'] is True
    assert srs_points(f, got, f, level, percent=100.0)['passed'] is False
    assert srs_points(f, got / 10.0, f, level, percent=0.0)['passed'] is True


def test_nothing_to_judge_gives_no_verdict():
    result = srs_points([1.0, 2.0], [1.0, 1.0], [10.0, 20.0], [1.0, 1.0])
    assert result['points'] == 0
    assert result['passed'] is None
    assert np.isnan(result['worst_db'])


@pytest.mark.parametrize('kwargs, message', [
    ({'rule': 'majority'}, 'not a rule'),
    ({'inner': (3.0, -1.5)}, 'not a band'),
    ({'inner': (0.0, 0.0)}, 'not a band'),
    ({'percent': -1.0}, 'not a share'),
    ({'outer': (6.0,)}, 'dB below, dB above'),
    ({'percent': 120.0}, 'not a share'),
])
def test_a_rule_or_band_that_cannot_be_read_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        srs_points([10.0, 20.0], [1.0, 1.0], [10.0, 20.0], [1.0, 1.0],
                   **kwargs)


# ---- the objects ------------------------------------------------------------


def test_a_spectrum_at_another_q_is_not_judged():
    spec = requirement()
    result = srs_tolerance(spec, measured(spec, q=50.0))
    assert result['points'] == 0
    assert result['passed'] is None
    assert 'Q=50' in result['refused']


def test_each_polarity_is_judged_as_it_is():
    """A maximax requirement, a positive spectrum on it and a negative
    one 2 dB under: judged one at a time, each keeps its own kind."""
    spec = requirement()
    positive = srs_tolerance(spec, measured(spec, kind='positive'))
    negative = srs_tolerance(spec, measured(spec, -2.0, kind='negative'))
    assert (positive['kind'], positive['passed']) == ('positive', True)
    assert (negative['kind'], negative['passed']) == ('negative', False)


def test_every_channel_and_event_is_judged_as_the_rms_reading_pairs_them():
    spec = requirement()
    values = np.vstack([spec.ordinate[0], spec.ordinate[0] * 10 ** -0.1,
                        spec.ordinate[0]])
    run = Srs(spec.abscissa, values, response_dof=['101Z+', '101Z+', '999X+'],
              ordinate_dim='acceleration', block=['shock 1', 'shock 2',
                                                  'shock 1'])
    rows = srs_tolerance_all(spec, run)
    assert [(dof, event) for dof, event, _r in rows] == [
        ('101Z+', 'shock 1'), ('101Z+', 'shock 2')]
    assert [(dof, event) for dof, event, _db in srs_errors(spec, run)] == [
        (dof, event) for dof, event, _r in rows]
    assert [r['passed'] for _dof, _event, r in rows] == [True, False]
    assert rows[1][2]['event'] == 'shock 2'


def test_a_refused_pair_is_reported_rather_than_dropped():
    """A run at the wrong Q is a row that says why, not a channel that
    silently vanished from the judgement."""
    spec = requirement()
    (_dof, _event, result), = srs_tolerance_all(spec,
                                                measured(spec, q=50.0))
    assert 'refused' in result
    assert srs_tolerance_all(spec, measured(spec, dof='202X+')) == []


# ---- a requirement from breakpoints --------------------------------------


def test_a_requirement_is_read_onto_its_spacings_points():
    """Twelfth-octave points from the first breakpoint, the last at or
    under the last breakpoint, on the power law between breakpoints."""
    spec = requirement()
    f = spec.abscissa
    assert spec.num_records == 1
    assert f.size == POINTS
    assert f[0] == 10.0
    assert f[-1] <= 2000.0 < f[-1] * 2.0 ** (1.0 / 12.0)
    assert np.allclose(np.log2(f[1:] / f[:-1]), 1.0 / 12.0)
    g = 9.80665
    at = np.argmin(np.abs(f - 100.0))
    assert spec.ordinate[0, 0] == pytest.approx(10.0 * g)
    assert spec.ordinate[0, at] == pytest.approx(100.0 * g)
    # half a decade up the slope from 10 to 100 g over 10 to 100 Hz
    # is the geometric mean, 31.6 g at 31.6 Hz
    mid = 10.0 * 2.0 ** (20 / 12)
    assert spec.ordinate[0, 20] == pytest.approx(10.0 * g * mid / 10.0)


def test_a_band_of_whole_octaves_ends_on_its_last_breakpoint():
    """Seven octaves at twelve to the octave: 85 points, the last on
    2560 Hz itself and written, not a rounding past it."""
    spec = ShockSpecification.from_breakpoints([20.0, 2560.0], [1.0, 1.0],
                                               '1X+')
    assert spec.abscissa.size == 85
    assert spec.abscissa[-1] == 2560.0
    assert np.isfinite(spec.ordinate).all()


def test_a_requirement_carries_its_q_type_unit_and_spacing():
    spec = ShockSpecification.from_breakpoints(
        [20.0, 2000.0], [5.0, 5.0], ['1X+', '1Y+'], per_octave=6, q=20.0,
        kind='positive', ordinate_unit='g')
    assert (spec.q, spec.kind) == (20.0, 'positive')
    assert spec.ordinate_unit == ['g', 'g']
    assert spec.response_dof == ['1X+', '1Y+']
    assert np.array_equal(spec.ordinate[0], spec.ordinate[1])
    assert np.allclose(np.log2(spec.abscissa[1:] / spec.abscissa[:-1]), 1 / 6)
    assert 'Q=20' in spec.comment[0] and '6 to the octave' in spec.comment[0]


def test_a_requirements_bands_become_its_limits():
    spec = requirement(warning=(-1.5, 3.0), abort=(-3.0, 6.0))
    target = spec.ordinate
    for name, db in (('warning_lower', -1.5), ('warning_upper', 3.0),
                     ('abort_lower', -3.0), ('abort_upper', 6.0)):
        assert np.allclose(spec.limits[name], target * 10.0 ** (db / 20.0))
    assert requirement().limits == {}


@pytest.mark.parametrize('frequencies, levels, message', [
    ([10.0], [1.0], 'two breakpoints'),
    ([100.0, 10.0], [1.0, 1.0], 'increasing'),
    ([10.0, 10.0], [1.0, 1.0], 'increasing'),
    ([0.0, 10.0], [1.0, 1.0], 'positive and increasing'),
    ([10.0, 100.0], [1.0, 0.0], 'positive'),
    ([10.0, 100.0], [1.0, 1.0, 1.0], 'levels for'),
])
def test_a_requirement_that_cannot_be_drawn_is_refused(frequencies, levels,
                                                       message):
    with pytest.raises(ValueError, match=message):
        ShockSpecification.from_breakpoints(frequencies, levels, '1X+')


@pytest.mark.parametrize('levels, dofs, per_octave, message', [
    ([[1.0, 1.0], [2.0, 2.0]], '1X+', None, '2 rows of levels for 1 DOF'),
    ([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], ['1X+', '1Y+'], None,
     '3 rows of levels for 2 DOFs'),
    ([1.0, 1.0], '1X+', 6.5, 'not a whole number'),
    ([1.0, 1.0], '1X+', 0, 'not a whole number'),
])
def test_a_requirement_whose_rows_or_spacing_cannot_be_read_is_refused(
        levels, dofs, per_octave, message):
    """Two rows at one DOF would leave the second with no channel, and
    6.5 to the octave would be read as 6 without a word."""
    with pytest.raises(ValueError, match=message):
        ShockSpecification.from_breakpoints([20.0, 2000.0], levels, dofs,
                                            per_octave=per_octave)
    assert ShockSpecification.from_breakpoints(
        [20.0, 2000.0], [[1.0, 1.0], [2.0, 2.0]], ['1X+', '1Y+'],
        per_octave=6.0).num_records == 2


# ---- the quantity -----------------------------------------------------------


def test_a_spectrum_of_undeclared_units_is_refused_not_failed():
    """Held as it came, a spectrum with no units is not in SI, and read
    against a requirement in g it put every point 20 dB and more under."""
    spec = requirement()
    raw = Srs(spec.abscissa, (spec.ordinate[0] / 9.80665)[None],
              response_dof=['101Z+'], ordinate_dim=['unknown'], q=spec.q)
    result = srs_tolerance(spec, raw)
    assert result['points'] == 0
    assert result['passed'] is None
    assert 'units of the spectrum are not declared' in result['refused']
    (_dof, _event, row), = srs_tolerance_all(spec, raw)
    assert 'not declared' in row['refused']


def test_a_spectrum_of_another_quantity_is_refused():
    """A velocity SRS whose numbers happen to sit on an acceleration
    requirement is not inside its band."""
    spec = requirement()
    velocity = Srs(spec.abscissa, spec.ordinate, response_dof=['101Z+'],
                   ordinate_dim=['velocity'], q=spec.q)
    result = srs_tolerance(spec, velocity)
    assert result['passed'] is None
    assert result['refused'] == ('the requirement is of acceleration and '
                                 'the spectrum of velocity; judge a '
                                 'spectrum of the quantity required')


# ---- the requirement's own limits ------------------------------------------


def test_a_requirements_limits_are_its_bands_when_none_is_given():
    """Warning limits at -3/+6 dB and abort at -6/+9: +5 dB everywhere is
    inside them, where the default -1.5/+3 dB would fail every point,
    and one point at +10 dB is past the abort limits."""
    spec = requirement(warning=(-3.0, 6.0), abort=(-6.0, 9.0))
    db = np.full(POINTS, 5.0)
    result = srs_tolerance(spec, measured(spec, db))
    assert result['limits'] is True
    assert result['inner'] == pytest.approx((-3.0, 6.0))
    assert result['outer'] == pytest.approx((-6.0, 9.0))
    assert (result['inside_count'], result['passed']) == (POINTS, True)
    db[30] = 10.0
    result = srs_tolerance(spec, measured(spec, db))
    assert (result['outer_count'], result['passed']) == (1, False)
    # a band given wins over the limits, and limits=False ignores them
    given = srs_tolerance(spec, measured(spec, 5.0), inner=(-1.5, 3.0))
    assert (given['inside_count'], given['outer'][1]) == (0, pytest.approx(9.0))
    plain = srs_tolerance(spec, measured(spec, db), limits=False)
    assert (plain['limits'], plain['outer'], plain['inside_count']) == (
        False, None, 0)
    assert srs_tolerance(requirement(), measured(spec))['limits'] is False


def _stepped_requirement():
    """A requirement whose warning limits are 0.5 times it below and
    1.2 times it above under 100 Hz, 3 times it above from 100 Hz up:
    not a fixed number of dB."""
    plain = requirement()
    f = plain.abscissa
    target = plain.ordinate
    upper = target * np.where(f < 100.0, 1.2, 3.0)
    return ShockSpecification(f, target, response_dof=['101Z+'],
                              ordinate_dim=['acceleration'], q=plain.q,
                              warning_lower=target * 0.5,
                              warning_upper=upper)


def test_limits_that_vary_along_the_band_are_read_at_each_point():
    """+5 dB everywhere is over 1.2 times (1.6 dB) and under 3 times
    (9.5 dB): outside the band below 100 Hz and inside it above."""
    spec = _stepped_requirement()
    result = srs_tolerance(spec, measured(spec, 5.0))
    above = spec.abscissa >= 100.0
    assert result['inside_count'] == int(above.sum())
    assert np.array_equal(result['over'], ~above)
    upper = result['inner'][1]
    assert np.ndim(upper) == 1
    assert upper[0] == pytest.approx(20 * np.log10(1.2))
    assert result['inner'][0] == pytest.approx(20 * np.log10(0.5))


def test_the_overlay_draws_the_limits_it_judged_against(qt_app):
    """A varying limit is drawn as the curve itself, in the plot's
    units, and named for it."""
    import pyqtgraph as pg

    from visualdynamics.plot.srs_tolerance import build_srs_tolerance

    spec = _stepped_requirement()
    layout = pg.GraphicsLayoutWidget()
    try:
        build_srs_tolerance(layout, spec, measured(spec, 5.0))
        plot = layout.getItem(0, 0)
        edges = _zone_edges(plot)
        assert _has_curve(edges, spec.limits['warning_upper'][0] / 9.80665)
        names = [label.text for _sample, label in plot.legend.items]
        assert 'outside the warning limits' in names
        assert plot.titleLabel.text.startswith(
            '101Z+ against the warning limits;')
    finally:
        layout.close()


# ---- the overlay -----------------------------------------------------------


def test_the_overlay_draws_headless(tmp_path):
    from visualdynamics.plot import plot_srs_tolerance

    spec = requirement()
    path = tmp_path / 'tolerance.png'
    plot_srs_tolerance([measured(spec, kind='positive'),
                        measured(spec, -2.0, kind='negative')], spec,
                       outer=(-3.0, 6.0), path=str(path), show=False)
    assert path.exists() and path.stat().st_size > 2000


def test_the_overlay_marks_the_points_outside_and_states_the_count(qt_app):
    """Each spectrum named with its count, share and verdict; the
    points the rule counts against it filled, and only those."""
    import pyqtgraph as pg

    from visualdynamics.plot.srs_tolerance import SYMBOLS, build_srs_tolerance

    spec = requirement()
    db = np.zeros(POINTS)
    db[40:50] = -5.0
    layout = pg.GraphicsLayoutWidget()
    try:
        results = build_srs_tolerance(
            layout, spec, [measured(spec, db),
                           measured(spec, 2.0, kind='positive')],
            outer=(-3.0, 6.0))
        plot = layout.getItem(0, 0)
        names = [label.text for _sample, label in plot.legend.items]
        assert names[:3] == ['requirement',
                             'maximax: 82/92 inside (89.1 %), fails',
                             'positive: 92/92 inside (100.0 %), passes']
        assert 'outside \u22121.5/+3 dB' in names
        assert 'below \u22123 dB' in names
        assert 'above +6 dB' in names
        marks = [item for item in plot.listDataItems()
                 if getattr(item, 'is_outside_mark', False)]
        assert len(marks) == 1                # the positive has none
        x = marks[0].xData                    # as given, not in decades
        assert np.allclose(x, spec.abscissa[40:50])
        assert marks[0].opts['symbol'] == SYMBOLS[0]
        assert [r['passed'] for r in results] == [False, True]
    finally:
        layout.close()


def test_the_above_lower_overlay_shades_no_ceiling(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.srs_tolerance import build_srs_tolerance

    spec = requirement()
    layout = pg.GraphicsLayoutWidget()
    try:
        build_srs_tolerance(layout, spec, measured(spec, 5.0),
                            rule='above_lower')
        plot = layout.getItem(0, 0)
        names = [label.text for _sample, label in plot.legend.items]
        assert 'maximax: 92/92 at or above \u22121.5 dB (100.0 %), passes' \
            in names
        assert 'below \u22121.5 dB' in names
        assert not any(name.startswith('outside') for name in names)
        assert not [item for item in plot.listDataItems()
                    if getattr(item, 'is_outside_mark', False)]
    finally:
        layout.close()


def test_the_overlay_refuses_a_channel_it_cannot_find(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.srs_tolerance import build_srs_tolerance

    spec = requirement()
    layout = pg.GraphicsLayoutWidget()
    try:
        with pytest.raises(ValueError, match='no record at 101Z'):
            build_srs_tolerance(layout, spec, measured(spec, dof='202X+'))
        with pytest.raises(ValueError, match='Q=50'):
            build_srs_tolerance(layout, spec, measured(spec, q=50.0))
    finally:
        layout.close()


def _zone_edges(plot):
    """The y values of every zone edge drawn, as given."""
    return [np.asarray(item.yData) for item in plot.listDataItems()
            if getattr(item, 'is_zone_edge', False)]


def _has_curve(curves, values):
    return any(c.shape == values.shape and np.allclose(c, values)
               for c in curves)


def test_the_overlay_shades_the_bands_at_their_decibels(qt_app):
    """The zones' edges are the requirement times each band's factor,
    in the plot's units (g by default), and the requirement is drawn
    in the same units."""
    import pyqtgraph as pg

    from visualdynamics.plot.srs_tolerance import build_srs_tolerance

    spec = requirement()
    in_g = spec.ordinate[0] / 9.80665
    layout = pg.GraphicsLayoutWidget()
    try:
        build_srs_tolerance(layout, spec, measured(spec), outer=(-3.0, 6.0))
        plot = layout.getItem(0, 0)
        edges = _zone_edges(plot)
        for db in (-1.5, 3.0, -3.0, 6.0):
            assert _has_curve(edges, in_g * 10.0 ** (db / 20.0)), db
        drawn = [item for item in plot.listDataItems()
                 if item.name() == 'requirement']
        assert np.allclose(drawn[0].yData, in_g)
        assert plot.titleLabel.text.startswith(
            '101Z+ against \u22121.5/+3 dB within \u22123/+6 dB; '
            '90 % of points inside')
    finally:
        layout.close()
    layout = pg.GraphicsLayoutWidget()
    try:
        build_srs_tolerance(layout, spec, measured(spec), rule='above_lower')
        edges = _zone_edges(layout.getItem(0, 0))
        assert _has_curve(edges, in_g * 10.0 ** (-1.5 / 20.0))
        assert not _has_curve(edges, in_g * 10.0 ** (3.0 / 20.0))
        assert 'every point at or above' in layout.getItem(0, 0).titleLabel.text
    finally:
        layout.close()


def test_the_overlay_draws_the_event_named(qt_app):
    """Two events at one channel: the one named is the one judged and
    drawn, and its legend line says so by its count."""
    import pyqtgraph as pg

    from visualdynamics.plot.srs_tolerance import build_srs_tolerance

    spec = requirement()
    values = np.vstack([spec.ordinate[0], spec.ordinate[0] * 10 ** -0.1])
    run = Srs(spec.abscissa, values, response_dof=['101Z+', '101Z+'],
              ordinate_dim='acceleration', block=['shock 1', 'shock 2'])
    layout = pg.GraphicsLayoutWidget()
    try:
        (result,) = build_srs_tolerance(layout, spec, run, event='shock 2')
        assert result['event'] == 'shock 2'
        assert result['inside_count'] == 0
        plot = layout.getItem(0, 0)
        assert plot.titleLabel.text.startswith('101Z+, shock 2 against')
    finally:
        layout.close()
    layout = pg.GraphicsLayoutWidget()
    try:
        with pytest.raises(ValueError, match='101Z\\+, shock 3'):
            build_srs_tolerance(layout, spec, run, event='shock 3')
    finally:
        layout.close()


def test_the_legend_counts_what_the_rule_counts():
    """Under 'above_lower' the count is the points at or above the
    lower edge, whatever is above the band."""
    from visualdynamics.plot.srs_tolerance import describe

    spec = requirement()
    db = np.full(POINTS, 5.0)
    db[:3] = -2.0
    result = srs_tolerance(spec, measured(spec, db), rule='above_lower')
    assert describe(result) == (
        'maximax: 89/92 at or above \u22121.5 dB (96.7 %), fails')


# ---- the reading in the window -----------------------------------------------


def _shock_pair(window, pump, *spectra, bands=None):
    """The requirement and these spectra in the window, selected, on
    the Points reading. The requirement's limits are -1.5/+3 dB and
    -3/+6 dB unless `bands` gives others, or {} for none."""
    from conftest import select_objects

    if bands is None:
        bands = {'warning': (-1.5, 3.0), 'abort': (-3.0, 6.0)}
    spec = requirement(**bands)
    window.add_object('Requirement', spec)
    names = []
    for k, (db, kind) in enumerate(spectra):
        name = f'SRS {k + 1}'
        window.add_object(name, measured(spec, db, kind=kind))
        names.append(name)
    select_objects(window, pump, 'Requirement', *names)
    window.data_pane.srs_actions['points'].trigger()
    pump()
    return spec


def test_the_points_rule_is_a_reading_of_a_shock_comparison(window, pump):
    """Offered beside the spectra and the RMS bars, and choosing it
    puts the panel up and the judgement in it and on the status line."""
    db = np.zeros(POINTS)
    db[40:50] = -5.0
    spec = _shock_pair(window, pump, (db, 'maximax'))
    worst = srs_tolerance(spec, measured(spec, db))['worst_frequency']
    pane = window.data_pane
    assert pane.srs_actions['points'].isVisible()
    assert pane.srs_actions['points'].text() == 'Points Rule'
    assert pane.srs_view == 'points'
    panel = pane.srs_tolerance_panel
    assert panel.isVisible()
    assert panel.derived['points'].text() == '92'
    assert panel.readout.text().splitlines() == [
        'maximax: 82/92 inside (89.1 %), fails',
        f'worst \u22125.0 dB at {worst:.4g} Hz']
    assert 'maximax: 82/92 inside (89.1 %), fails' in window._status_text
    plot = pane.graphics.getItem(0, 0)
    assert [item for item in plot.listDataItems()
            if getattr(item, 'is_outside_mark', False)]


def test_each_polarity_selected_is_judged(window, pump):
    """A positive and a negative spectrum selected with the requirement
    are one measurement read two ways, a line each."""
    _shock_pair(window, pump, (0.0, 'positive'), (-2.0, 'negative'))
    lines = window.data_pane.srs_tolerance_panel.readout.text().splitlines()
    assert lines[0] == 'positive: 92/92 inside (100.0 %), passes'
    assert lines[2] == 'negative: 0/92 inside (0.0 %), fails'
    # every point is 2 dB under, so the worst is any of them
    assert lines[3].startswith('worst \u22122.0 dB at ')


def test_the_panels_band_and_rule_redraw_the_judgement(window, pump):
    """+5 dB everywhere fails a share of -1.5/+3 dB, passes it with the
    band widened to +6, and passes 'All above lower' outright; the
    rows that do not apply to a rule are not shown."""
    _shock_pair(window, pump, (5.0, 'maximax'), bands={})
    panel = window.data_pane.srs_tolerance_panel
    assert not panel.limits_box.isVisibleTo(panel)    # nothing to offer
    assert panel.readout.text().startswith('maximax: 0/92 inside')
    panel.inner_high.setValue(6.0)
    pump()
    assert panel.readout.text().startswith(
        'maximax: 92/92 inside (100.0 %), passes')
    panel.inner_high.setValue(3.0)
    panel.rule_box.setCurrentIndex(1)
    pump()
    assert panel.settings()['rule'] == 'above_lower'
    assert panel.readout.text().startswith(
        'maximax: 92/92 at or above \u22121.5 dB (100.0 %), passes')
    assert not panel.percent_box.isVisibleTo(panel)
    assert not panel.inner_high.isVisibleTo(panel)
    assert not panel.outer_low.isVisibleTo(panel)
    panel.outer_box.setChecked(True)
    pump()
    assert panel.outer_high.isVisibleTo(panel)
    assert panel.settings()['outer'] == (-3.0, 6.0)
    assert 'passes' in panel.readout.text()       # 5 dB is inside +6...
    panel.outer_high.setValue(4.0)
    pump()
    assert 'fails' in panel.readout.text()        # ...and outside +4


def test_the_points_reading_is_journaled_as_its_plot_call(window, pump):
    """Against the requirement's limits the call names no band; with
    them set aside it names the panel's bands and says so."""
    _shock_pair(window, pump, (0.0, 'positive'), (-2.0, 'negative'))
    panel = window.data_pane.srs_tolerance_panel

    def journaled():
        return [line for line in window.project.journal
                if line.startswith('visualdynamics.plot.plot_srs_tolerance(')]

    assert journaled() == [
        ("visualdynamics.plot.plot_srs_tolerance([project['SRS 1'], "
         "project['SRS 2']], project['Requirement'], channel='101Z+', "
         "event=None, inner=None, outer=None, rule='share', percent=90.0, "
         "path='srs_tolerance.png', show=False)")]
    panel.set_settings(rule='above_lower', outer=(-3.0, 6.0), limits=False)
    pump()
    assert journaled()[-1] == (
        "visualdynamics.plot.plot_srs_tolerance([project['SRS 1'], "
        "project['SRS 2']], project['Requirement'], channel='101Z+', "
        "event=None, inner=(-1.5, 3.0), outer=(-3.0, 6.0), "
        "rule='above_lower', percent=90.0, limits=False, "
        "path='srs_tolerance.png', show=False)")


def test_the_panel_judges_against_the_requirements_limits(window, pump):
    """A requirement with -3/+6 dB warning limits: the panel starts on
    them, the band fields put away and seeded with them, so +5 dB
    passes as the Spectra reading's shading says it does; set aside,
    the fields hold -3/+6 and the same verdict follows."""
    _shock_pair(window, pump, (5.0, 'maximax'),
                bands={'warning': (-3.0, 6.0)})
    panel = window.data_pane.srs_tolerance_panel
    assert panel.limits_box.isVisibleTo(panel)
    assert panel.limits_box.isChecked()
    assert not panel.inner_low.isVisibleTo(panel)
    assert panel.settings()['inner'] is None
    assert panel.readout.text().startswith(
        'maximax: 92/92 inside (100.0 %), passes')
    assert (panel.inner_low.value(), panel.inner_high.value()) == (-3.0, 6.0)
    panel.limits_box.setChecked(False)
    pump()
    assert panel.settings()['inner'] == (-3.0, 6.0)
    assert panel.readout.text().startswith('maximax: 92/92 inside')
    # redrawn for the same requirement, the choice made is kept
    window.render_current()
    pump()
    assert not panel.limits_box.isChecked()


def test_several_polarities_are_read_by_the_points_rule_alone(window, pump):
    """The spectra and the bars read one measurement, so with a positive
    and a negative selected they are not offered: the points rule is
    shown whatever was chosen last, judges both, and the table holds
    both polarities' RMS deviations. The choice itself is kept."""
    from conftest import select_objects

    spec = requirement(warning=(-1.5, 3.0))
    window.add_object('Requirement', spec)
    window.add_object('Positive', measured(spec, kind='positive'))
    window.add_object('Negative', measured(spec, -2.0, kind='negative'))
    pane = window.data_pane
    pane.srs_view = 'error'
    select_objects(window, pump, 'Requirement', 'Positive', 'Negative')
    assert not pane.srs_actions['curves'].isVisible()
    assert not pane.srs_actions['error'].isVisible()
    assert pane.srs_actions['points'].isChecked()
    assert pane.srs_view == 'error'
    lines = pane.srs_tolerance_panel.readout.text().splitlines()
    assert lines[0] == 'positive: 92/92 inside (100.0 %), passes'
    assert lines[2] == 'negative: 0/92 inside (0.0 %), fails'
    held = window._replication_holder
    assert held.channels == ['101Z+ positive', '101Z+ negative']
    assert held.values[('101Z+ negative', 0)] == pytest.approx(-2.0)
    # one spectrum with the requirement is offered every reading again
    select_objects(window, pump, 'Requirement', 'Positive')
    assert pane.srs_actions['error'].isVisible()
    assert pane.srs_actions['error'].isChecked()


def test_the_points_rule_is_not_offered_for_sine_levels(qt_app):
    """It is a shock's: the sine comparison keeps its two readings, and
    a points choice carried over reads as its curves."""
    from visualdynamics.gui.panes import DataPane

    pane = DataPane('light', offscreen=True)
    try:
        pane.srs_view = 'points'
        pane.show_srs_views(True, kind='sine')
        assert not pane.srs_actions['points'].isVisible()
        assert pane.srs_reading('sine') == 'curves'
        assert pane.srs_actions['curves'].isChecked()
        assert not pane.srs_tolerance_panel.isVisibleTo(pane)
        pane.show_srs_views(True)
        assert pane.srs_actions['points'].isVisible()
        assert pane.srs_reading() == 'points'
    finally:
        pane.close()
