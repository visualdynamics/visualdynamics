"""The stroke reading: a pulse integrated raw against the shaker's
shock ratings.

Pinned against arithmetic that does not depend on the implementation
and against the worked first checks the reading exists for. A
half-sine of peak A and duration T has a velocity change of 2AT/pi;
compensated symmetrically at a share s of its peak, each compensation
pulse lasts T/(2s), so the event lasts T(1 + 1/s): six durations at a
fifth, eleven at a tenth. Its peak to peak is close to A T^2 / 2, about
193 A T^2 inches with A in g. The worked values: a 100 g, 6 ms pulse
peaks at 73.7 in/s and needs 0.694 in against the rule's 0.695; a
20 g, 30 ms one needs 3.47 in; compensating at a tenth instead needs
1.80 times the stroke; and a 50 g, 11 ms pulse with a moving mass a
quarter of the body's needs 1.46 in of travel relative to the body,
and moves 84.5 in/s against it where the table moves 67.6.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.filters import integrate
from visualdynamics.core.stroke import (
    COMPENSATION_SHARE,
    _number,
    compensated_half_sine,
    read_stroke,
    rule_stroke,
)

G = 9.80665
INCH = 0.0254
LBF = 4.4482216152605
LB = 0.45359237
RATE = 102400.0


def pulse(peak, duration, share=COMPENSATION_SHARE, rate=RATE):
    return compensated_half_sine(peak, duration, rate, share=share,
                                 unit='g')


def lasts(history, duration):
    """How many main-pulse durations the record spans."""
    return (history.abscissa[-1] - history.abscissa[0]) / duration


# ---- the worked first checks ------------------------------------------


def test_a_100_g_6_ms_half_sine_needs_0_694_in_against_the_rules_0_695():
    """Compensated at a fifth it lasts six durations, peaks at half its
    velocity change, 73.7 in/s, and needs 0.694 in peak to peak, where
    193 A T^2 gives 0.695."""
    record = pulse(100.0, 0.006)
    reading = read_stroke(record)
    assert lasts(record, 0.006) == pytest.approx(6.0, abs=1e-3)
    assert reading.peak_velocity / INCH == pytest.approx(73.7, abs=0.05)
    assert reading.peak_to_peak / INCH == pytest.approx(0.694, abs=5e-4)
    assert reading.rule / INCH == pytest.approx(0.695, abs=5e-4)
    assert 193.0 * 100.0 * 0.006 ** 2 == pytest.approx(0.695, abs=5e-4)


def test_a_20_g_30_ms_half_sine_needs_3_47_in():
    """A fifth the peak of the 100 g pulse and five times the duration:
    the same velocity, five times the stroke."""
    reading = read_stroke(pulse(20.0, 0.030))
    assert reading.peak_to_peak / INCH == pytest.approx(3.47, abs=5e-3)
    assert reading.peak_velocity / INCH == pytest.approx(73.7, abs=0.05)


def test_compensating_at_a_tenth_takes_eleven_durations_and_1_80_times_the_stroke():
    """Half the compensation height doubles each compensation pulse's
    length, and the table travels further before it is brought back."""
    fifth = pulse(50.0, 0.011)
    tenth = pulse(50.0, 0.011, share=0.1)
    assert lasts(fifth, 0.011) == pytest.approx(6.0, abs=1e-3)
    assert lasts(tenth, 0.011) == pytest.approx(11.0, abs=1e-3)
    ratio = (read_stroke(tenth).peak_to_peak
             / read_stroke(fifth).peak_to_peak)
    assert ratio == pytest.approx(1.80, abs=5e-3)


def test_a_moving_mass_a_quarter_of_the_bodys_needs_1_46_in_of_relative_travel():
    """The 50 g, 11 ms pulse's table moves 1.17 in; the body moves the
    other way by a quarter of that, so the stops see 1.46 in, and the
    coil moves against the body a quarter faster than the table."""
    reading = read_stroke(pulse(50.0, 0.011), **BODY)
    assert reading.peak_to_peak / INCH == pytest.approx(1.1661, abs=1e-4)
    assert reading.rating.body_share == 0.25
    assert reading.travel / INCH == pytest.approx(1.46, abs=5e-3)
    assert reading.peak_velocity / INCH == pytest.approx(67.593, abs=1e-3)
    assert reading.relative_velocity / INCH == pytest.approx(
        1.25 * 67.593, abs=1e-3)
    # the table alone without it, and without the moving mass beside it
    for alone in ({}, {'body_mass': 400.0 * LB}):
        plain = read_stroke(pulse(50.0, 0.011), **alone)
        assert plain.travel / INCH == pytest.approx(1.1661, abs=1e-4)
        assert plain.relative_velocity == plain.peak_velocity


@pytest.mark.parametrize('peak,duration', [
    (20.0, 0.011), (50.0, 0.011), (100.0, 0.011), (30.0, 0.018),
    (10.0 * np.pi, 0.011), (100.0, 0.006), (20.0, 0.030)])
def test_the_rule_reads_a_sixth_of_a_percent_high(peak, duration):
    """A T^2 / 2 against the double integration, case by case: high by
    the same 0.159 % on every pulse, since the ratio is the arrangement's
    and not the pulse's (193 A T^2, rounded down from 193.04, lands
    within 0.14 %). The rule the reading reports for the record is the
    one for its own peak and duration."""
    reading = read_stroke(pulse(peak, duration))
    assert reading.rule == pytest.approx(rule_stroke(peak * G, duration),
                                         rel=1e-4)
    assert reading.rule / reading.peak_to_peak - 1.0 == pytest.approx(
        0.00159, abs=1e-5)
    assert reading.equivalent_duration == pytest.approx(duration, rel=1e-4)


def test_the_rule_is_193_a_t_squared_in_inches_with_a_in_g():
    assert rule_stroke(G, 1.0) / INCH == pytest.approx(193.04, abs=0.01)


def test_the_peak_velocity_is_half_the_velocity_change():
    """The pre-pulse takes the table to minus half the change, the main
    pulse to plus half, and the post-pulse back to zero."""
    reading = read_stroke(pulse(50.0, 0.011))
    change = 2.0 * 50.0 * G * 0.011 / np.pi
    assert reading.velocity_change == pytest.approx(change, rel=1e-5)
    assert reading.peak_velocity == pytest.approx(change / 2.0, rel=1e-5)
    assert reading.velocity.min() == pytest.approx(-change / 2.0, rel=1e-5)


def test_the_compensated_pulse_starts_and_ends_at_rest():
    """No net velocity change and no net displacement, and the table
    travels one way only: down to its full peak to peak at the middle
    of the main pulse and back."""
    reading = read_stroke(pulse(50.0, 0.011))
    assert abs(reading.velocity[-1]) < 1e-5 * reading.peak_velocity
    assert abs(reading.displacement[-1]) < 1e-5 * reading.peak_to_peak
    assert reading.highest == pytest.approx(0.0, abs=1e-9)
    assert -reading.lowest == pytest.approx(reading.peak_to_peak, rel=1e-9)
    lowest_at = reading.time[np.argmin(reading.displacement)]
    middle = 0.011 / (2.0 * 0.2) + 0.011 / 2.0
    assert lowest_at == pytest.approx(middle, abs=2.0 / RATE)


def test_the_stroke_is_integrated_raw_not_through_the_drift_high_pass():
    """The default 2 Hz drift high-pass reads the 50 g pulse's stroke
    3.9 % high, 1.212 in, and moves it off rest at both ends; the
    reading is the raw 1.1661 in."""
    record = pulse(50.0, 0.011)
    filtered = integrate(integrate(record))
    drifted = np.ptp(filtered.ordinate[0].real)
    assert drifted / INCH == pytest.approx(1.212, abs=1e-3)
    reading = read_stroke(record)
    assert reading.peak_to_peak / INCH == pytest.approx(1.1661, abs=1e-4)
    raw = integrate(integrate(record, None), None)
    np.testing.assert_allclose(reading.displacement, raw.ordinate[0].real,
                               rtol=0, atol=1e-15)


# ---- the ratings -------------------------------------------------------


#: 100 lb moving on a body four times as heavy: a quarter
BODY = {'moving_mass': 100.0 * LB, 'body_mass': 400.0 * LB}


def ratings(**overrides):
    given = {'stroke': 2.5 * INCH, 'velocity': 70.0 * INCH,
             'force': 6000.0 * LBF, **BODY}
    given.update(overrides)
    return given


def test_each_demand_is_a_share_of_its_rating():
    """The travel with the body each way against half the stroke, the
    peak velocity with the body against the shock velocity, and the
    moving mass times the peak against the force."""
    reading = read_stroke(pulse(50.0, 0.011), **ratings())
    shares = reading.shares()
    assert shares['stroke'] == pytest.approx(1.4576 / 1.25, abs=1e-4)
    assert shares['velocity'] == pytest.approx(1.25 * 67.593 / 70.0,
                                               abs=1e-4)
    assert reading.force / LBF == pytest.approx(5000.0, rel=1e-6)
    assert shares['force'] == pytest.approx(5000.0 / 6000.0, rel=1e-6)
    # every share within with room each way and a faster shaker
    roomy = read_stroke(pulse(50.0, 0.011),
                        **ratings(stroke=4.0 * INCH, velocity=90.0 * INCH))
    assert roomy.fits is True


def test_the_stroke_share_is_the_one_the_replication_fit_reads():
    """One rule: the same travel against the same rating is the same
    share in both readings, the larger travel each way over half the
    stroke, not the peak to peak over the stroke."""
    from visualdynamics.core.shaker import ShakerRating

    reading = read_stroke(pulse(50.0, 0.011), **ratings())
    rating = ShakerRating(**ratings())
    assert reading.rating == rating
    assert reading.shares()['stroke'] == rating.stroke_share(
        reading.travel_down, reading.travel_up)
    assert reading.shares()['stroke'] != pytest.approx(
        reading.travel / rating.stroke)


def test_a_rating_exceeded_is_said_over():
    reading = read_stroke(pulse(50.0, 0.011), velocity=60.0 * INCH)
    assert reading.fits is False
    assert 'Peak velocity 67.6 in/s against 60 in/s: OVER, 113 %' in \
        reading.describe()
    counted = read_stroke(pulse(50.0, 0.011), **ratings())
    assert ('Peak velocity with the body 84.5 in/s against 70 in/s: OVER, '
            '121 %') in counted.describe()


def test_a_rating_not_given_is_not_judged():
    reading = read_stroke(pulse(50.0, 0.011))
    assert reading.shares() == {'stroke': None, 'velocity': None,
                                'force': None}
    assert reading.fits is None
    assert reading.fits_stroke is None
    assert reading.force is None
    assert reading.describe()[0] == 'One-sided travel 1.17 in'


def test_a_force_rating_without_a_moving_mass_is_not_judged():
    reading = read_stroke(pulse(50.0, 0.011),
                          **ratings(moving_mass=None))
    assert reading.shares()['force'] is None
    assert not any(line.startswith('Force') for line in reading.describe())


def test_a_one_sided_travel_reaches_a_stop_inside_the_stroke():
    """The symmetric arrangement travels one way only: 1.46 in down
    with the body counted, inside a 2.5 in stroke peak to peak and past
    the 1.25 in to the stop below center, where the armature rests. The
    stops see the travel each way, so the pulse does not fit, and the
    words say so once: 117 %, never 'within, 58 %' beside it."""
    reading = read_stroke(pulse(50.0, 0.011), **ratings())
    assert reading.travel / (2.5 * INCH) < 1.0
    assert reading.shares()['stroke'] == pytest.approx(1.166, abs=1e-3)
    assert reading.fits_stroke is False
    assert reading.fits is False
    assert reading.travel_down / INCH == pytest.approx(1.4576, abs=1e-4)
    assert reading.travel_up == 0.0
    assert not hasattr(reading, 'offset')
    lines = reading.describe()
    assert lines[0] == ('One-sided travel with the body 1.46 in against '
                        '1.25 in to the stop: OVER, 117 %')
    assert lines[1] == ('Centered, the travel would fit: 1.46 in peak to '
                        'peak against 2.5 in. Centering the travel is the '
                        'controller\'s compensation, an asymmetric pre- and '
                        'post-pulse')
    assert 'Each way from center: 1.46 in down, 0 in up' in lines
    assert not any('within, 58' in line or '58 %' in line for line in lines)
    assert reading.to_the_stop() == ('one-sided travel with the body 1.46 in '
                                     'against 1.25 in to the stop')
    assert not any('rest' in line for line in lines)
    # with room each way, the travel is within and no centering is said
    roomy = read_stroke(pulse(50.0, 0.011), **ratings(stroke=4.0 * INCH))
    assert roomy.fits_stroke is True
    assert roomy.describe()[0].endswith('to the stop: within, 73 %')
    assert not any('Centered' in line for line in roomy.describe())
    # and too little stroke even peak to peak: centering is not offered
    short = read_stroke(pulse(50.0, 0.011), **ratings(stroke=1.0 * INCH))
    assert short.fits_stroke is False
    assert not any('Centered' in line for line in short.describe())


def test_a_pulse_the_other_way_reaches_the_upper_stop():
    """Negated, the same pulse runs 1.46 in up from center."""
    record = pulse(50.0, 0.011)
    record.ordinate[0] = -record.ordinate[0]
    reading = read_stroke(record, **ratings())
    assert reading.travel_down == 0.0
    assert reading.travel_up / INCH == pytest.approx(1.4576, abs=1e-4)
    assert reading.fits_stroke is False
    assert reading.to_the_stop().startswith(
        'one-sided travel with the body 1.46 in')


def test_a_side_never_reached_is_zero_not_minus_zero():
    """The table never rises above rest on the nominal pulse; that side
    reads 0.0, and is written '0', never '-0'."""
    import math

    reading = read_stroke(pulse(50.0, 0.011), **ratings())
    assert math.copysign(1.0, reading.travel_up) == 1.0
    assert math.copysign(1.0, reading.highest) == 1.0
    record = pulse(50.0, 0.011)
    record.ordinate[0] = -record.ordinate[0]
    assert math.copysign(1.0, read_stroke(record).travel_down) == 1.0


def test_travel_both_ways_names_the_side_that_reaches_the_stop():
    """The pulse, then the same pulse at half height the other way:
    1.17 in down and 0.58 in up, so the down side is named and the
    travel is not one-sided."""
    rows = pulse(50.0, 0.011).ordinate[0].real
    both = np.concatenate([rows, -0.5 * rows[1:]])
    record = TimeHistory(np.arange(both.size) / RATE, both[None],
                         response_dof=['1Z+'], ordinate_dim=['acceleration'])
    reading = read_stroke(record, stroke=2.0 * INCH)
    assert reading.travel_down / INCH == pytest.approx(1.1661, abs=1e-4)
    assert reading.travel_up / INCH == pytest.approx(0.5830, abs=1e-4)
    assert reading.to_the_stop() == ('travel down 1.17 in against 1 in to '
                                     'the stop')


def test_a_demand_exactly_at_its_rating_is_within_it():
    """At the rating is inside it, each way and against the body."""
    first = read_stroke(pulse(50.0, 0.011), **BODY)
    at = read_stroke(pulse(50.0, 0.011), **BODY,
                     stroke=2.0 * first.travel_down,
                     velocity=first.relative_velocity)
    assert at.shares()['stroke'] == 1.0
    assert at.shares()['velocity'] == 1.0
    assert at.describe()[0].endswith('within, 100 %')
    assert at.fits_stroke is True
    assert at.fits is True
    record = pulse(50.0, 0.011)
    record.ordinate[0] = -record.ordinate[0]
    upward = read_stroke(record, **BODY, stroke=2.0 * first.travel_down)
    assert upward.travel_up == first.travel_down
    assert upward.fits_stroke is True


def test_a_rating_of_zero_is_no_rating():
    """Zero is accepted and judges nothing, as the panel's empty box
    does."""
    reading = read_stroke(pulse(50.0, 0.011), stroke=0.0, velocity=0.0)
    assert reading.shares()['stroke'] is None
    assert reading.shares()['velocity'] is None
    assert reading.rating.stroke is None
    assert reading.fits_stroke is None


def test_the_numbers_are_written_in_the_unit_system_given():
    from visualdynamics.units import SI

    reading = read_stroke(pulse(50.0, 0.011), **ratings())
    lines = reading.describe(SI)
    assert lines[0] == ('One-sided travel with the body 0.037 m against '
                        '0.0318 m to the stop: OVER, 117 %')
    assert lines[3].startswith('Force 22,241 N against 26,689 N')


def test_a_reading_number_is_three_figures_or_whole_with_commas():
    assert _number(5000.0) == '5,000'
    assert _number(1000.0) == '1,000'
    assert _number(0.69393) == '0.694'
    assert _number(67.593) == '67.6'


# ---- the record --------------------------------------------------------


def two_channel(start=0.0):
    record = pulse(50.0, 0.011)
    rows = np.vstack([record.ordinate[0].real / G,
                      record.ordinate[0].real * 0.5])
    return TimeHistory(start + record.abscissa, rows,
                       response_dof=['1X+', '2Z+'],
                       ordinate_dim=['force', 'acceleration'])


def test_the_first_acceleration_channel_is_read_by_default():
    reading = read_stroke(two_channel())
    assert reading.dof == '2Z+'
    assert reading.peak_acceleration == pytest.approx(25.0 * G, rel=1e-6)


def test_a_channel_is_named_by_its_dof_or_its_index():
    assert read_stroke(two_channel(), '2Z+').dof == '2Z+'
    assert read_stroke(two_channel(), 1).dof == '2Z+'


def test_a_channel_that_is_not_acceleration_is_refused():
    with pytest.raises(ValueError, match='1X\\+ is not an acceleration'):
        read_stroke(two_channel(), '1X+')
    with pytest.raises(ValueError, match='record 0 is not'):
        read_stroke(two_channel(), 0)


def test_a_record_with_no_acceleration_is_refused():
    record = two_channel()
    record.delete_records([1])
    with pytest.raises(ValueError, match='holds none'):
        read_stroke(record)


def test_a_negative_rating_is_refused():
    with pytest.raises(ValueError, match='stroke'):
        read_stroke(pulse(50.0, 0.011), stroke=-1.0)


def test_a_rating_that_is_not_a_number_is_refused_as_such():
    with pytest.raises(ValueError, match='the stroke has to be above zero '
                       'and finite, not nan'):
        read_stroke(pulse(50.0, 0.011), stroke=float('nan'))
    with pytest.raises(ValueError, match='the velocity has to be above zero '
                       'and finite, not inf'):
        read_stroke(pulse(50.0, 0.011), velocity=float('inf'))


def test_a_record_with_a_sample_that_is_not_a_number_is_refused():
    """One NaN integrates to a travel of NaN, which would read 'Travel
    nan in ... OVER, nan %'; the record is refused, saying where."""
    record = pulse(50.0, 0.011)
    record.ordinate[0, 100] = np.nan
    with pytest.raises(ValueError, match='1Z\\+ holds 1 sample that is '
                       'not finite, the first at index 100'):
        read_stroke(record, **ratings())


def test_an_acceleration_of_undeclared_unit_is_not_read():
    """Named acceleration by its source and holding the file's raw
    numbers: integrated as if SI, its travel would be read against the
    stroke in a scale nobody chose."""
    record = pulse(50.0, 0.011)
    hinted = TimeHistory(record.abscissa, record.ordinate / G,
                         response_dof=['1Z+'],
                         dimension_hint=['acceleration'])
    assert hinted.known_dim(0) == 'acceleration'
    with pytest.raises(ValueError, match='undeclared unit'):
        read_stroke(hinted)


def test_a_record_that_starts_late_is_read_on_its_own_clock():
    reading = read_stroke(two_channel(start=2.0))
    assert reading.start == 2.0
    assert reading.time[0] == 2.0
    assert reading.time[1] == pytest.approx(2.0 + 1.0 / RATE)
    assert reading.duration == pytest.approx(0.066, abs=1.0 / RATE)


def test_the_pulse_keeps_the_unit_it_was_written_in():
    record = pulse(50.0, 0.011)
    assert record.ordinate_unit == ['g']
    assert record.response_dof == ['1Z+']
    assert record.ordinate[0].real.max() == pytest.approx(50.0 * G,
                                                          rel=1e-6)
    # SI when no unit is given
    si = compensated_half_sine(50.0 * G, 0.011, RATE)
    assert si.ordinate[0].real.max() == pytest.approx(50.0 * G, rel=1e-6)


def test_a_pulse_of_four_samples_is_built():
    """Four samples across the main pulse is the fewest taken."""
    record = compensated_half_sine(50.0, 0.004, 1000.0)
    assert record.ordinate[0].real.max() > 0.0


def test_the_record_closes_on_the_post_pulses_last_sample():
    """Six durations to within half a sample, and named for what it
    is."""
    record = pulse(50.0, 0.011)
    assert abs(record.abscissa[-1] - 6.0 * 0.011) <= 0.5 / RATE
    assert record.abscissa[0] == 0.0
    assert record.comment == [('half-sine, 11 ms, compensated at 0.2 of '
                               'its peak')]


def test_the_compensation_pulses_are_a_share_of_the_peak():
    record = pulse(50.0, 0.011, share=0.1)
    assert record.ordinate[0].real.min() == pytest.approx(-5.0 * G,
                                                          rel=1e-6)


@pytest.mark.parametrize('arguments,match', [
    ((50.0, 0.011, RATE, 1.0), 'between zero and one'),
    ((50.0, 0.011, RATE, 0.0), 'between zero and one'),
    ((-50.0, 0.011, RATE, 0.2), 'positive'),
    ((0.0, 0.011, RATE, 0.2), 'positive'),
    ((50.0, 0.0, RATE, 0.2), 'positive'),
    ((50.0, 0.011, 0.0, 0.2), 'positive'),
    ((50.0, 0.011, 200.0, 0.2), 'a 11 ms pulse at 200 samples a second'),
])
def test_a_pulse_that_cannot_be_built_is_refused(arguments, match):
    peak, duration, rate, share = arguments
    with pytest.raises(ValueError, match=match):
        compensated_half_sine(peak, duration, rate, share=share)


# ---- the view ----------------------------------------------------------


def _plots(layout):
    import pyqtgraph as pg

    return [item for item in layout.ci.items
            if isinstance(item, pg.PlotItem)]


def test_the_view_draws_three_rows_with_each_rating_on_its_row(qt_app):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.stroke import build_stroke

    reading = read_stroke(pulse(50.0, 0.011), **ratings())
    layout = pg.GraphicsLayoutWidget()
    try:
        build_stroke(layout, reading)
        acceleration, velocity, displacement = _plots(layout)
        assert 'force 5,000 lbf against 6,000 lbf (83 %)' in \
            acceleration.titleLabel.text
        assert ('peak velocity with the body 84.5 in/s against 70 in/s '
                '(121 %)') in velocity.titleLabel.text
        title = displacement.titleLabel.text
        assert title.startswith('one-sided travel with the body 1.46 in '
                                'against 1.25 in to the stop (117 %)')
        assert 'stroke (58 %)' not in title
        assert 'table 1.17 in integrated, 1.17 in by the half-sine rule' \
            in title
        # the velocity against the body dashed beside the table's
        assert [curve.opts['pen'].style()
                for curve in velocity.listDataItems()] == [
            Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine]

        def lines(plot):
            return sorted(item.value() for item in plot.items
                          if isinstance(item, pg.InfiniteLine))

        assert lines(velocity) == pytest.approx([-70.0, 70.0])
        assert lines(displacement) == pytest.approx([-1.25, 1.25])
        # the table solid and the travel with the body dashed: told
        # apart by more than their color
        styles = [curve.opts['pen'].style()
                  for curve in displacement.listDataItems()]
        assert styles == [Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine]
        names = [label.text for _sample, label in
                 displacement.legend.items]
        assert names == ['table', 'relative to the body',
                         'stops, half the stroke from center']
    finally:
        layout.close()


def test_without_ratings_the_view_draws_no_limits(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.stroke import build_stroke

    layout = pg.GraphicsLayoutWidget()
    try:
        build_stroke(layout, read_stroke(pulse(100.0, 0.006)))
        plots = _plots(layout)
        assert not [item for plot in plots for item in plot.items
                    if isinstance(item, pg.InfiniteLine)]
        assert len(plots[2].listDataItems()) == 1
        assert plots[2].titleLabel.text.startswith('peak to peak 0.694 in')
    finally:
        layout.close()


def test_the_view_draws_to_a_file(qt_app, tmp_path):
    from visualdynamics.plot import plot_stroke

    path = tmp_path / 'stroke.png'
    plot_stroke(pulse(50.0, 0.011), stroke=2.5 * INCH, **BODY,
                path=str(path), show=False)
    assert path.exists() and path.stat().st_size > 2000


def test_the_summary_says_whether_it_fits():
    from visualdynamics.plot.stroke import summary

    record = pulse(50.0, 0.011)
    assert summary(read_stroke(record, **ratings())) == (
        '1Z+: One-sided travel with the body 1.46 in against 1.25 in to '
        'the stop: OVER, 117 %; does not fit')
    assert summary(read_stroke(record, **ratings(stroke=4.0 * INCH,
                                                 velocity=90.0 * INCH))) \
        .endswith('within, 73 %; fits')
    assert summary(read_stroke(record, **ratings(stroke=4.0 * INCH))) \
        .endswith('; does not fit')
    assert summary(read_stroke(record)).endswith('; no ratings given')


# ---- the window --------------------------------------------------------


def _stroke_reading(window, pump, history=None):
    """A record in the window, shown, on the Stroke reading."""
    name = window.add_object('Pulse', history or pulse(50.0, 0.011))
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    pane = window.data_pane
    assert pane.stroke_action.isVisible()
    if not pane.stroke_action.isChecked():
        pane.stroke_action.trigger()
    pump()
    return pane


def test_a_time_history_offers_the_stroke_reading(window, pump):
    """Choosing it puts the ratings beside three rows and the reading
    in the panel and on the status line, with no rating yet given."""
    pane = _stroke_reading(window, pump)
    panel = pane.stroke_panel
    assert pane.stroke_action.text() == 'Stroke'
    assert panel.isVisible()
    assert len(_plots(pane.graphics)) == 3
    assert panel.derived['peak'].text() == '50 g'
    assert panel.derived['peak_velocity'].text() == '67.6 in/s'
    assert panel.derived['peak_to_peak'].text() == '1.17 in'
    assert panel.derived['rule'].text() == '1.17 in'
    assert panel.derived['each_way'].text() == '1.17 / 0 in'
    assert panel.derived['force'].text() == '—'
    assert panel.readout.text() == ('No ratings given: the reading stands '
                                    'on its own.')
    assert window._status_text == ('1Z+: One-sided travel 1.17 in; no '
                                   'ratings given')


def test_the_ratings_redraw_the_reading(window, pump):
    """Typed in the window's units (inches here), kept in SI, and read
    at once: the 50 g pulse with a quarter-mass body fits a 2.5 in
    stroke peak to peak, and reaches the stop 1.25 in below center."""
    pane = _stroke_reading(window, pump)
    panel = pane.stroke_panel
    mass = window.unit_system.label_text('mass')
    panel.boxes['stroke'].setValue(2.5)
    panel.boxes['moving_mass'].setValue(
        float(window.unit_system.from_si(100.0 * LB, 'mass')))
    panel.boxes['body_mass'].setValue(
        float(window.unit_system.from_si(400.0 * LB, 'mass')))
    pump()
    assert panel.names['body_mass'].text() == f'Body mass [{mass}]'
    assert pane.stroke_settings['stroke'] == pytest.approx(2.5 * INCH)
    # to the box's four places in the window's mass unit
    assert pane.stroke_settings['body_mass'] == pytest.approx(400.0 * LB,
                                                              rel=1e-4)
    assert panel.derived['travel'].text() == '1.46 in'
    lines = panel.readout.text().splitlines()
    assert lines[0] == ('One-sided travel with the body 1.46 in against '
                        '1.25 in to the stop: OVER, 117 %')
    assert lines[1].startswith('Centered, the travel would fit: 1.46 in '
                               'peak to peak against 2.5 in.')
    assert panel.readout.styleSheet() == 'font-weight: 600;'
    title = _plots(pane.graphics)[2].titleLabel.text
    assert 'to the stop (117 %)' in title
    # zero takes the rating away again
    panel.boxes['stroke'].setValue(0.0)
    pump()
    assert pane.stroke_settings['stroke'] is None
    assert panel.readout.text().startswith('No ratings given')


def test_the_ratings_are_restated_in_new_units(window, pump):
    """A change of units shows the same rating in the new unit, not the
    same number in it."""
    from visualdynamics.units import SYSTEMS

    pane = _stroke_reading(window, pump)
    pane.stroke_panel.boxes['stroke'].setValue(2.5)
    pump()
    window._units_changed('m-kg-N-s')
    pump()
    panel = pane.stroke_panel
    assert window.unit_system is SYSTEMS['m-kg-N-s']
    assert panel.names['stroke'].text() == 'Stroke [m]'
    assert panel.boxes['stroke'].value() == pytest.approx(0.0635)
    assert pane.stroke_settings['stroke'] == pytest.approx(2.5 * INCH)
    assert panel.derived['peak_to_peak'].text() == '0.0296 m'


def test_the_stroke_reading_is_journaled_as_its_plot_call(window, pump):
    pane = _stroke_reading(window, pump)
    pane.stroke_panel.boxes['stroke'].setValue(2.5)
    pump()
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_stroke(')]
    assert lines == [
        ("visualdynamics.plot.plot_stroke(project['Pulse'], channel=0, "
         f"stroke={2.5 * INCH!r}, path='stroke.png', show=False)")]


def _shown(window, pump, history, name):
    window.add_object(name, history)
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    return window.data_pane


@pytest.mark.parametrize('kind', ['velocity', 'force', 'undeclared'])
def test_a_record_with_no_acceleration_does_not_offer_it(window, pump, kind):
    """Nothing to integrate, so nothing to offer (principle 3): a
    velocity, a force, or an acceleration whose unit was never declared.
    The record still offers its other readings."""
    record = pulse(50.0, 0.011)
    history = {
        'velocity': lambda: integrate(record, None),
        'force': lambda: TimeHistory(record.abscissa, record.ordinate,
                                     response_dof=['1Z+'],
                                     ordinate_dim=['force']),
        'undeclared': lambda: TimeHistory(record.abscissa, record.ordinate,
                                          response_dof=['1Z+'],
                                          dimension_hint=['acceleration']),
    }[kind]()
    pane = _shown(window, pump, history, 'Not acceleration')
    assert pane.kurtosis_action.isVisible()
    assert not pane.stroke_action.isVisible()
    assert not pane.stroke_panel.isVisibleTo(pane)


def test_a_record_with_a_sample_that_is_not_a_number_says_so(window, pump):
    record = pulse(50.0, 0.011)
    record.ordinate[0, 100] = np.nan
    pane = _stroke_reading(window, pump, record)
    assert 'not finite' in window._status_text
    assert pane.stroke_panel.derived['peak'].text() == '—'


def test_the_traces_are_drawn_at_the_pictures_width(window, pump):
    """Every record-length trace is decimated to its peaks and clipped
    to the view: drawn whole, a million-sample record cost 1.4 s a
    repaint, on every hover and resize."""
    pane = _stroke_reading(window, pump)
    traces = [item for plot in _plots(pane.graphics)
              for item in plot.listDataItems()
              if item.xData is not None and len(item.xData) > 100]
    assert len(traces) >= 3, 'acceleration, velocity and displacement'
    for item in traces:
        assert item.opts['autoDownsample']
        assert item.opts['downsampleMethod'] == 'peak'
        assert item.opts['clipToView']


def test_the_stroke_reading_stands_the_trace_marks_down(window, pump):
    """It replaces the trace, as the scalogram does, so the averaging
    and shock panels are put away and the 3-D toggle is not offered."""
    pane = _stroke_reading(window, pump)
    assert not pane.averaging_panel.isVisibleTo(pane)
    assert not pane.shock_panel.isVisibleTo(pane)
    assert not pane.waterfall_action.isVisible()
    pane.shocks_action.trigger()
    pump()
    assert not pane.stroke_action.isChecked()
    assert not pane.stroke_panel.isVisibleTo(pane)
