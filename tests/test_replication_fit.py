"""The replication fit: whether a shaker can play a reference record.

Pinned two ways. Against arithmetic that does not depend on the
implementation: a sine of acceleration A at frequency f moves the table
A/(2πf)² at a peak velocity of A/(2πf); a second-order Butterworth run
forward and back passes it at (f/fc)⁴/(1 + (f/fc)⁴), so the mean square
it removes is one less that squared; an offset c held for a time T
integrates to cT and cT²/2; and the force at a peak is the moving mass
times it. And against a worked example: sixty seconds of rough road at
a bracket, 2 grms with a 14 g pothole, logged with an offset of a
thousandth of a g, whose displacement as logged walks off to hundreds
of inches, which with its mean taken out still spans 48.6 in, and which
high-passed at 2, 3, 5 and 8 Hz spans 1.40, 0.71, 0.43 and 0.20 in for
1.25, 2.02, 3.33 and 5.10 percent of its mean square. A record that
does not start at rest is tapered before it is filtered, so a tone at
full amplitude from its first sample still moves the table A/(2πf)².
"""

from __future__ import annotations

from functools import cache
from itertools import pairwise

import numpy as np
import pytest

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.filters import integrate
from visualdynamics.core.replication_fit import (
    CUTOFFS,
    SHORTEST,
    FitCase,
    ShakerRating,
    _percent,
    replication_fit,
    taper,
)

G = 9.80665
INCH = 0.0254
LB = 0.45359237
LBF = 4.4482216152605
RATE = 2048.0


def rating(**changed):
    """A 2 in stroke shaker carrying 150 lb, with room to spare on
    force and velocity unless a test says otherwise."""
    fields = {'force': 4000.0 * LBF, 'stroke': 2.0 * INCH,
              'velocity': 70.0 * INCH, 'moving_mass': 150.0 * LB}
    fields.update(changed)
    return ShakerRating(**fields)


def record(values, rate=RATE, dofs=None, dims=None):
    rows = np.atleast_2d(values)
    n = len(rows)
    return TimeHistory(np.arange(rows.shape[1]) / rate, rows,
                       response_dof=dofs or [f'{k + 1}Z+' for k in range(n)],
                       ordinate_dim=dims or ['acceleration'] * n)


def shaped(breakpoints, levels, seconds, seed):
    """A Gaussian record in g whose one-sided ASD follows the
    breakpoints, straight on log-log axes and nothing outside them:
    white noise, its transform scaled line by line from its own flat
    density of 2/rate to the profile."""
    samples = round(seconds * RATE)
    noise = np.random.default_rng(seed).standard_normal(samples)
    frequency = np.fft.rfftfreq(samples, 1.0 / RATE)
    density = np.zeros_like(frequency)
    inside = (frequency >= breakpoints[0]) & (frequency <= breakpoints[-1])
    density[inside] = np.exp(np.interp(np.log(frequency[inside]),
                                       np.log(breakpoints), np.log(levels)))
    gain = np.sqrt(density * RATE / 2.0)
    return np.fft.irfft(np.fft.rfft(noise) * gain, samples)


#: the impacts on the course, as (seconds, peak g); the largest is the
#: 14 g pothole at 32 s
IMPACTS = ((5.2, 9.0), (9.7, 7.5), (13.1, 10.0), (17.8, 8.0),
           (21.4, 11.0), (26.0, 7.0), (32.0, 14.0), (36.5, 9.5),
           (39.9, 8.5), (52.3, 10.5), (55.6, 7.5), (57.1, 9.0))


@cache
def road(seconds=60.0):
    """Sixty seconds at a bracket while a vehicle crosses rough ground,
    in g: ride motion from 0.7 to 3.5 Hz on a floor to 400 Hz, onto and
    off the course over 2 s, a braking swell at 46 s, and a dozen
    impacts each ringing the bracket's modes at 60 and 250 Hz (each
    ring's velocity starts and ends at zero). Its mean is taken out, so
    the only offset is the one a test adds."""
    x = shaped((0.7, 3.5, 6.0, 150.0, 400.0),
               (0.02, 0.02, 0.015, 0.015, 0.0015), seconds, 3404)
    t = np.arange(x.size) / RATE
    x = x * np.clip(np.minimum(t, seconds - t) / 2.0, 0.0, 1.0)
    x = x * (1.0 + 0.8 * np.exp(-((t - 46.0) / 2.5) ** 2))
    for hit, peak in IMPACTS:
        since = np.clip(t - hit, 0.0, None)
        for frequency, share in ((60.0, 0.4), (250.0, 0.6)):
            w = 2.0 * np.pi * frequency
            x = x + (t >= hit) * share * peak * np.exp(-0.06 * w * since) \
                * (np.cos(w * since) - 0.06 * np.sin(w * since))
    return x - x.mean()


def logged():
    """The road record as the logger wrote it, 0.001 g high, in SI."""
    return record((road() + 0.001) * G)


@pytest.fixture(scope='module')
def road_fit():
    return replication_fit(logged(), rating())


def inches(value):
    return value / INCH


# ---- the worked example ------------------------------------------------


def test_the_record_is_the_one_the_example_works(road_fit):
    """2 grms with its largest peak, a 14 g pothole, 32 s in: a crest
    factor of seven, where a Gaussian record checked at three sigma
    would suggest three."""
    assert road_fit.rms / G == pytest.approx(1.98, abs=0.01)
    assert road_fit.peak_acceleration / G == pytest.approx(13.98, abs=0.01)
    assert road_fit.peak_time == pytest.approx(32.0, abs=0.01)
    assert road_fit.crest_factor == pytest.approx(7.05, abs=0.01)


def test_the_force_is_the_moving_mass_times_the_largest_peak(road_fit):
    """150 lb at the 14 g pothole asks about 2,100 lbf, where three
    sigma of the rms would have said 900."""
    assert road_fit.force == pytest.approx(
        150.0 * LB * road_fit.peak_acceleration)
    assert road_fit.force / LBF == pytest.approx(2098.0, abs=1.0)
    assert road_fit.force_share == pytest.approx(2098.0 / 4000.0, abs=1e-3)
    assert road_fit.fits_force
    small = replication_fit(logged(), rating(force=2000.0 * LBF),
                            cutoffs=(3.0,))
    assert not small.fits_force
    assert 'over it' in small.describe()


def test_as_logged_the_offset_walks_the_travel_off_the_stroke(road_fit):
    """The logger's thousandth of a g, held for the minute, is 23.2 in/s
    and 695 in on its own; the record walks 31.4 in the other way, so
    the travel ends 663.5 in out, and it leaves the 1 in of room either
    side of the middle 0.65 s in."""
    as_logged = road_fit.as_logged
    assert road_fit.offset / G == pytest.approx(0.001, abs=1e-9)
    assert inches(road_fit.offset_velocity) == pytest.approx(
        0.001 * G * 60.0 / INCH, rel=1e-4)
    assert inches(road_fit.offset_velocity) == pytest.approx(23.17, abs=0.01)
    assert inches(road_fit.offset_displacement) == pytest.approx(
        694.9, abs=0.1)
    seconds = (road().size - 1) / RATE
    assert road_fit.duration == seconds
    assert road_fit.offset_displacement == pytest.approx(
        0.5 * road_fit.offset * seconds ** 2, rel=1e-12)
    assert inches(as_logged.travel_up) == pytest.approx(663.53, abs=0.01)
    assert inches(as_logged.displacement[-1]) == pytest.approx(663.53,
                                                               abs=0.01)
    assert inches(as_logged.peak_velocity) == pytest.approx(61.66, abs=0.01)
    assert as_logged.first_past_stroke() == pytest.approx(0.65, abs=0.01)
    assert not as_logged.fits
    assert as_logged.mean_square_removed == 0.0
    assert as_logged.peak_change == 0.0
    assert 'times the room' in as_logged.describe()


def test_taking_the_offset_out_alone_is_not_enough():
    """With the mean taken out and nothing else, the ride motion below
    the shaker's reach still spans 48.6 in, from -33.5 to +15.1."""
    centered = record(road() * G)
    fit = replication_fit(centered, rating())
    assert abs(fit.offset) < 1e-12
    assert inches(fit.as_logged.peak_to_peak) == pytest.approx(48.62,
                                                               abs=0.01)
    assert inches(fit.as_logged.travel_down) == pytest.approx(33.52,
                                                              abs=0.01)
    assert inches(fit.as_logged.travel_up) == pytest.approx(15.10, abs=0.01)
    assert fit.as_logged.stroke_share == pytest.approx(
        fit.as_logged.travel_down / fit.rating.room)
    assert not fit.as_logged.fits
    assert 'the travel down is 34 times the room' in \
        fit.as_logged.describe()


@pytest.mark.parametrize('cutoff,span,removed,peak', [
    (2.0, 1.40, 1.25, -0.44), (3.0, 0.71, 2.02, -1.34),
    (5.0, 0.43, 3.33, -2.69), (8.0, 0.20, 5.10, -4.27)])
def test_each_cutoff_has_the_examples_span_and_price(road_fit, cutoff, span,
                                                     removed, peak):
    """With the offset out by the high-pass the record fits a 2.0 in
    stroke at every candidate, and each one's span, mean square removed
    and change in the largest peak are the example's."""
    case = road_fit.case(cutoff)
    assert inches(case.peak_to_peak) == pytest.approx(span, abs=0.01)
    assert 100.0 * case.mean_square_removed == pytest.approx(removed,
                                                             abs=0.01)
    assert 100.0 * case.peak_change == pytest.approx(peak, abs=0.01)
    assert case.fits_stroke
    assert case.fits


def test_the_largest_excursion_falls_as_the_cutoff_rises(road_fit):
    """Each step up the band saves stroke and costs mean square: the
    price curve, in both directions."""
    spans = [case.peak_to_peak for case in road_fit.cases]
    ups = [case.travel_up for case in road_fit.cases]
    downs = [case.travel_down for case in road_fit.cases]
    removed = [case.mean_square_removed for case in road_fit.cases]
    velocities = [case.peak_velocity for case in road_fit.cases]
    for values in (spans, ups, downs, velocities):
        assert all(a > b for a, b in pairwise(values))
    assert all(a < b for a, b in pairwise(removed))
    assert [case.cutoff for case in road_fit.cases] == list(CUTOFFS)


def test_the_velocity_rating_can_rule_out_the_lowest_cutoff():
    """At 2 Hz the table peaks at 23.85 in/s, at 3 Hz at 19.95: a
    22 in/s rating passes the second and fails the first, and the
    lowest cutoff that fits moves up to 3 Hz."""
    fit = replication_fit(logged(), rating(velocity=22.0 * INCH))
    two, three = fit.case(2.0), fit.case(3.0)
    assert inches(two.peak_velocity) == pytest.approx(23.85, abs=0.01)
    assert inches(three.peak_velocity) == pytest.approx(19.95, abs=0.01)
    assert two.fits_stroke and not two.fits_velocity and not two.fits
    assert three.fits
    assert fit.lowest_fitting is three
    assert 'the velocity is' in two.describe()
    assert fit.describe().endswith(
        'The lowest cutoff that fits is 3 Hz, removing 2.02 % of the '
        'mean square.')


def test_the_2_hz_candidate_is_the_default_drift_control(road_fit):
    """The cutoff's filter is the drift high-pass `integrate` applies by
    default, second order, so away from the ends the candidate tapers
    the two readings of the record at 2 Hz agree to a thousandth of an
    inch at every sample (they differ by 0.00013 in); in the last
    second, where the default has no taper and the record has not come
    back to rest, they part by 0.056 in."""
    default = integrate(integrate(logged()))
    drift = np.real(default.ordinate[0])
    apart = np.abs(drift - road_fit.case(2.0).displacement)
    inside = slice(round(2.0 * RATE), round(58.0 * RATE))
    assert np.max(apart[inside]) < 0.001 * INCH
    assert np.max(apart) / INCH == pytest.approx(0.056, abs=0.001)


def test_a_record_cut_short_is_not_failed_on_its_own_edges():
    """The road record cut to 10-40 s starts and ends mid-motion.
    Untapered, the filters' start-up transients ran its 2 Hz travel to
    2.07 in down and failed a 2 in stroke at 2 and 3 Hz; tapered, every
    candidate fits, the 2 Hz span 1.09 in."""
    whole = logged()
    cut = slice(round(10.0 * RATE), round(40.0 * RATE))
    piece = TimeHistory(whole.abscissa[cut], whole.ordinate[:, cut],
                        response_dof=['1Z+'], ordinate_dim=['acceleration'])
    fit = replication_fit(piece, rating())
    assert all(case.fits_stroke for case in fit.cases)
    assert inches(fit.case(2.0).peak_to_peak) == pytest.approx(1.09,
                                                               abs=0.01)
    assert inches(fit.case(2.0).travel_down) == pytest.approx(0.56,
                                                              abs=0.01)


# ---- hand values --------------------------------------------------------


def tone(frequency=20.0, peak=2.0 * G, seconds=10.0, taper=1.0):
    """A sine of acceleration, eased in and out over `taper` seconds as
    a reference is, so the filter's edge transients stay small."""
    t = np.arange(round(seconds * RATE)) / RATE
    ease = np.clip(np.minimum(t, t[-1] - t) / taper, 0.0, 1.0)
    ease = 0.5 - 0.5 * np.cos(np.pi * ease)
    return record(peak * ease * np.sin(2.0 * np.pi * frequency * t))


def butterworth_twice(frequency, cutoff):
    """A second-order Butterworth high-pass's gain, run forward and
    back."""
    ratio = (frequency / cutoff) ** 4
    return ratio / (1.0 + ratio)


def test_a_tone_moves_the_table_a_over_omega_squared():
    """2 g at 20 Hz: 1.24 mm peak at 0.156 m/s, the filter's gain at a
    tenth of the tone counted twice (acceleration and velocity), read
    in the steady middle of the record."""
    fit = replication_fit(tone(), rating(), cutoffs=(2.0,))
    case = fit.case(2.0)
    w = 2.0 * np.pi * 20.0
    gain = butterworth_twice(20.0, 2.0)
    middle = slice(round(3.0 * RATE), round(7.0 * RATE))
    assert np.ptp(case.displacement[middle]) / 2.0 == pytest.approx(
        2.0 * G / w ** 2 * gain ** 2, rel=1e-3)
    assert case.peak_velocity == pytest.approx(2.0 * G / w * gain ** 2,
                                               rel=1e-3)


@pytest.mark.parametrize('cutoff', [1.0, 2.0])
def test_a_tone_from_its_first_sample_still_moves_the_table_a_over_omega_squared(
        cutoff):
    """1 g at 10 Hz for a minute, at full amplitude from the first
    sample: 2.48 mm of travel each way. Untapered, the high-pass's start-up
    transient read 18.2 mm at 1 Hz and 8.7 mm at 2 Hz, and a negative
    mean square removed at 1 Hz; tapered, the travel is A/ω² to a fifth
    of a percent and the price is the filter's own."""
    t = np.arange(round(60.0 * RATE)) / RATE
    w = 2.0 * np.pi * 10.0
    fit = replication_fit(record(G * np.sin(w * t)), rating(),
                          cutoffs=(cutoff,))
    case = fit.case(cutoff)
    assert G / w ** 2 * 1e3 == pytest.approx(2.484, abs=0.001)
    assert max(case.travel_up, case.travel_down) == pytest.approx(
        G / w ** 2, rel=2e-3)
    assert case.mean_square_removed == pytest.approx(
        1.0 - butterworth_twice(10.0, cutoff) ** 2, rel=0.02)


def test_the_taper_is_a_half_cosine_over_each_end():
    window = taper(10, 4.0)
    assert window[:4] == pytest.approx(0.5 - 0.5 * np.cos(
        np.pi * np.arange(4) / 4))
    assert window[4:6] == pytest.approx([1.0, 1.0])
    assert window[6:] == pytest.approx(window[:4][::-1])
    # never more than half the record at each end
    assert taper(10, 100.0)[:5] == pytest.approx(taper(10, 5.0)[:5])
    assert taper(10, 0.0) == pytest.approx(np.ones(10))


def test_the_mean_square_removed_is_one_less_the_gain_squared():
    """At a tenth of the tone the filter passes (10⁴/(1 + 10⁴)), so it
    removes two parts in ten thousand of the mean square; at half the
    tone, 16/17, and it removes 11.4 percent."""
    for cutoff in (2.0, 10.0):
        fit = replication_fit(tone(), rating(), cutoffs=(cutoff,))
        gain = butterworth_twice(20.0, cutoff)
        assert fit.case(cutoff).mean_square_removed == pytest.approx(
            1.0 - gain ** 2, rel=2e-3)


def test_the_body_adds_its_share_of_the_travel():
    """A 150 lb load on a 1,500 lb body: the body moves a tenth of the
    table's displacement the other way, so the travel is 1.1 times it."""
    plain = replication_fit(tone(), rating(), cutoffs=(2.0,)).case(2.0)
    counted = replication_fit(tone(), rating(body_mass=1500.0 * LB),
                              cutoffs=(2.0,)).case(2.0)
    assert counted.rating.body_share == pytest.approx(0.1)
    assert np.allclose(counted.travel, 1.1 * counted.displacement)
    assert counted.travel_up == pytest.approx(1.1 * plain.travel_up)
    # the table's velocity is the table's; the coil's against the body,
    # which the rating is judged by, counts the body the same way
    assert counted.peak_velocity == plain.peak_velocity
    assert plain.relative_velocity == plain.peak_velocity
    assert counted.relative_velocity == pytest.approx(
        1.1 * plain.peak_velocity)
    assert counted.velocity_share == pytest.approx(
        1.1 * plain.velocity_share)


def lift(height=1.4 * INCH, seconds=10.0, sign=1.0):
    """A slow one-sided excursion, a raised cosine of displacement of
    `height` and back over the record, as its acceleration: from rest
    the table follows height/2 (1 - cos ωt) exactly."""
    t = np.arange(round(seconds * RATE)) / RATE
    w = 2.0 * np.pi / seconds
    return record(sign * height / 2.0 * w ** 2 * np.cos(w * t))


#: when 0.7 (1 - cos ωt) first passes 1.0 on a ten-second lift
CROSSING = np.arccos(1.0 - 1.0 / 0.7) / (2.0 * np.pi / 10.0)


def test_the_stroke_is_read_each_way_from_the_middle():
    """A lift of 1.4 in and back. Its span is inside a 2 in stroke, but
    the armature starts from the middle, so the room up is 1 in and the
    lift is 140 percent of it. It leaves the room where
    0.7 (1 - cos ωt) first passes 1.0, at the first sample after
    3.205 s."""
    fit = replication_fit(lift(), rating(), cutoffs=(0.01,))
    logged = fit.as_logged
    assert fit.rating.room == pytest.approx(INCH)
    assert logged.peak_to_peak == pytest.approx(1.4 * INCH, rel=1e-3)
    assert logged.peak_to_peak < fit.rating.stroke
    assert logged.travel_up == pytest.approx(1.4 * INCH, rel=1e-3)
    assert logged.up_share == logged.stroke_share
    assert logged.stroke_share == pytest.approx(1.4, rel=1e-3)
    assert not logged.fits_stroke
    assert 'the travel up is 140 % of the room' in logged.describe()
    assert CROSSING == pytest.approx(3.205, abs=0.001)
    assert logged.first_past_stroke() == pytest.approx(
        np.ceil(CROSSING * RATE) / RATE, abs=0.25 / RATE)
    assert fit.describe().startswith(
        'The force at the largest peak is 0.00 % of the rating. As '
        'logged, the travel runs to 140 % of the room and leaves it '
        '3.21 s in.')
    small = replication_fit(lift(0.8 * INCH), rating(), cutoffs=(0.01,))
    assert small.as_logged.fits_stroke
    assert small.as_logged.first_past_stroke() is None


def test_a_dip_is_read_against_the_room_below():
    """The same excursion downward: the verdict names that side."""
    low = replication_fit(lift(sign=-1.0), rating(), cutoffs=(0.01,))
    assert low.as_logged.travel_up == 0.0
    assert low.as_logged.travel_down == pytest.approx(1.4 * INCH, rel=1e-3)
    assert low.as_logged.up_share == 0.0
    assert low.as_logged.down_share == low.as_logged.stroke_share
    assert low.as_logged.stroke_share == pytest.approx(1.4, rel=1e-3)
    assert 'the travel down is 140 % of the room' in \
        low.as_logged.describe()
    assert low.as_logged.first_past_stroke() == pytest.approx(
        np.ceil(CROSSING * RATE) / RATE, abs=0.25 / RATE)


def test_a_rating_met_exactly_is_met():
    """At the rating is inside it: the force, the stroke and the
    velocity each set to exactly what the record asks."""
    fit = replication_fit(tone(), rating(), cutoffs=(2.0,))
    case = fit.case(2.0)
    exact = rating(force=fit.force, velocity=case.peak_velocity,
                   stroke=2.0 * max(case.travel_up, case.travel_down))
    met = replication_fit(tone(), exact, cutoffs=(2.0,))
    assert met.force_share == 1.0 and met.fits_force
    assert met.case(2.0).velocity_share == 1.0
    assert met.case(2.0).stroke_share == 1.0
    assert met.case(2.0).fits


def test_each_cutoff_judges_the_force_at_its_own_peak(road_fit):
    """A force rating 2 % under the record's largest peak fails the
    cutoffs that leave the peak within 2 % (2 and 3 Hz take 0.44 and
    1.34 % off it) and passes the ones that take more, so the lowest
    cutoff that fits moves up to 5 Hz, and the panel's verdict says so."""
    fit = replication_fit(logged(), rating(force=0.98 * road_fit.force))
    assert [case.fits_force for case in fit.cases] == [False, False,
                                                       True, True]
    assert all(case.fits_stroke and case.fits_velocity
               for case in fit.cases)
    assert fit.case(2.0).force == pytest.approx(
        150.0 * LB * fit.case(2.0).peak_acceleration)
    assert fit.case(2.0).force_share == pytest.approx(1.0159, abs=1e-4)
    assert fit.lowest_fitting is fit.case(5.0)
    assert fit.case(2.0).describe() == ('2 Hz: does not fit; the force is '
                                        '102 % of the rating.')


def test_the_verdicts_are_worded_in_full(road_fit):
    assert road_fit.case(2.0).describe() == (
        '2 Hz: fits, using 71 % of the room, 34 % of the velocity rating '
        'and 52 % of the force rating.')
    assert road_fit.as_logged.describe() == (
        'As logged: does not fit; the travel up is 664 times the room.')
    assert road_fit.describe() == (
        'The force at the largest peak is 52 % of the rating. As logged, '
        'the travel runs to 664 times the room and leaves it 0.653 s in. '
        'The lowest cutoff that fits is 2 Hz, removing 1.25 % of the mean '
        'square.')


def test_a_travel_equal_both_ways_is_said_each_way():
    case = FitCase(cutoff=None, rating=rating(), start=0.0, sample_rate=1.0,
                   velocity=np.zeros(3),
                   displacement=np.array([0.0, 1.5, -1.5]) * INCH,
                   mean_square_removed=0.0, peak_change=0.0,
                   peak_acceleration=0.0)
    assert case.describe() == ('As logged: does not fit; the travel each '
                               'way is 150 % of the room.')


def test_a_record_that_fails_only_on_velocity_says_so():
    """As logged inside the stroke but too fast: the words name the
    velocity, not a travel that never left the room."""
    fit = replication_fit(lift(0.8 * INCH), rating(velocity=1e-4),
                          cutoffs=(0.01,))
    assert fit.as_logged.fits_stroke and not fit.as_logged.fits_velocity
    assert 'As logged, the travel stays inside the room and the velocity ' \
        'is' in fit.describe()


def test_a_record_is_read_on_its_own_clock():
    """A record that starts 5 s in keeps its clock: the peak, the
    travel and the first sample are all where the record says."""
    history = tone()
    late = TimeHistory(history.abscissa + 5.0, history.ordinate,
                       response_dof=['1Z+'], ordinate_dim=['acceleration'])
    fit = replication_fit(late, rating(), cutoffs=(2.0,))
    early = replication_fit(history, rating(), cutoffs=(2.0,))
    assert fit.case(2.0).time[0] == 5.0
    assert fit.peak_time == pytest.approx(early.peak_time + 5.0)
    assert fit.case(2.0).excursion_time == pytest.approx(
        early.case(2.0).excursion_time + 5.0)


def test_a_silent_record_fits_and_costs_nothing():
    fit = replication_fit(record(np.zeros(4096)), rating(), cutoffs=(2.0,))
    case = fit.case(2.0)
    assert case.mean_square_removed == 0.0
    assert case.peak_change == 0.0
    assert fit.crest_factor == 0.0
    assert case.fits and fit.fits_force


# ---- the arguments ------------------------------------------------------


def test_the_channel_is_chosen_by_dof_or_index_and_must_be_acceleration():
    quiet = tone(peak=0.1 * G).ordinate[0].real
    loud = tone(peak=4.0 * G).ordinate[0].real
    force = np.ones_like(quiet)
    history = record(np.vstack([force, quiet, loud]),
                     dofs=['9X+', '1Z+', '2Z+'],
                     dims=['force', 'acceleration', 'acceleration'])
    assert replication_fit(history, rating()).dof == '1Z+'
    assert replication_fit(history, rating(), channel='2Z+').dof == '2Z+'
    assert replication_fit(history, rating(), channel=2).dof == '2Z+'
    with pytest.raises(ValueError, match='not an acceleration channel'):
        replication_fit(history, rating(), channel='9X+')
    with pytest.raises(ValueError, match='not an acceleration channel'):
        replication_fit(history, rating(), channel=0)
    with pytest.raises(ValueError, match='holds none'):
        replication_fit(record(force, dims=['force']), rating())


def test_the_cutoffs_are_worked_lowest_first_and_once_each():
    fit = replication_fit(tone(), rating(), cutoffs=(5, 2.0, 5.0))
    assert [case.cutoff for case in fit.cases] == [2.0, 5.0]
    with pytest.raises(KeyError, match='not one of the candidate'):
        fit.case(3.0)
    assert fit.case(None) is fit.as_logged


def test_a_record_with_a_sample_that_is_not_a_number_is_refused():
    """A NaN would integrate to a travel of NaN, which is neither inside
    the room nor over it; the record is refused, saying where."""
    values = tone().ordinate[0].real.copy()
    values[300] = np.nan
    with pytest.raises(ValueError, match='1Z\\+ holds 1 sample that is '
                       'not finite, the first at index 300'):
        replication_fit(record(values), rating())


def test_an_acceleration_of_undeclared_unit_is_not_checked():
    """Named acceleration by its source and holding the file's raw
    numbers, say in g: integrated as if SI, the travel would be judged
    against the stroke at a tenth of its size."""
    history = tone()
    hinted = TimeHistory(history.abscissa, history.ordinate / G,
                         response_dof=['1Z+'], dimension_hint=['acceleration'])
    assert hinted.known_dim(0) == 'acceleration'
    with pytest.raises(ValueError, match='undeclared unit'):
        replication_fit(hinted, rating())


def test_a_record_too_short_to_filter_says_so():
    with pytest.raises(ValueError, match=f'a record of {SHORTEST - 1} '
                       'samples is too short to high-pass'):
        replication_fit(record(np.ones(SHORTEST - 1)), rating(),
                        cutoffs=(100.0,))
    replication_fit(record(np.ones(SHORTEST)), rating(), cutoffs=(100.0,))


def test_a_rating_the_fit_needs_is_asked_for():
    with pytest.raises(ValueError, match='the replication fit needs the '
                       'force and the velocity as well'):
        replication_fit(tone(), ShakerRating(stroke=0.05, moving_mass=10.0))
    with pytest.raises(ValueError, match='needs the force, the stroke, the '
                       'velocity and the moving mass as well'):
        replication_fit(tone(), ShakerRating())


@pytest.mark.parametrize('cutoffs', [(), (0.0,), (RATE / 2.0,)])
def test_a_cutoff_outside_the_band_is_refused(cutoffs):
    with pytest.raises(ValueError):
        replication_fit(tone(), rating(), cutoffs=cutoffs)


@pytest.mark.parametrize('changed,message', [
    ({'force': 0.0}, 'force'), ({'stroke': -1.0}, 'stroke'),
    ({'velocity': 0.0}, 'velocity'), ({'moving_mass': 0.0}, 'moving mass'),
    ({'body_mass': 0.0}, 'body mass')],
    ids=['force', 'stroke', 'velocity', 'moving-mass', 'body-mass'])
def test_a_rating_out_of_range_is_refused(changed, message):
    with pytest.raises(ValueError, match=message):
        rating(**changed)


def test_without_a_body_mass_the_body_is_not_counted():
    assert rating().body_share == 0.0
    assert rating().room == pytest.approx(INCH)


def test_none_fitting_says_so():
    fit = replication_fit(logged(), rating(stroke=0.1 * INCH))
    assert fit.lowest_fitting is None
    assert fit.describe().endswith('No candidate cutoff fits.')


def test_the_shares_are_written_to_suit_their_size():
    assert _percent(0.0125) == '1.25 %'
    assert _percent(0.71) == '71 %'
    assert _percent(663.5) == '664 times'
    assert _percent(10.0) == '10 times'
    assert _percent(0.0999) == '9.99 %'


# ---- the view -----------------------------------------------------------


def test_the_view_states_the_verdict_and_the_drift(qt_app):
    """The top row is titled with the cutoff's verdict and what the
    record does as logged; the band is the stroke and the drawn travel
    is the case's, in inches."""
    import pyqtgraph as pg

    from visualdynamics.plot.replication_fit import build_replication_fit
    from visualdynamics.units import IN_LBF_S

    us = IN_LBF_S.with_units(acceleration='g')
    fit = replication_fit(logged(), rating(body_mass=3000.0 * LB))
    layout = pg.GraphicsLayoutWidget()
    try:
        case = build_replication_fit(layout, fit, unit_system=us)
        assert case.cutoff == 2.0, 'the lowest cutoff that fits'
        travel = layout.getItem(0, 0)
        title = travel.titleLabel.text
        assert '2 Hz: fits' in title
        assert 'As logged the travel spans' in title
        assert '0.001 g alone integrates to 694.9 in' in title
        assert 'body counted' in travel.getAxis('left').labelText
        band, = [item for item in travel.items
                 if isinstance(item, pg.LinearRegionItem)]
        assert band.getRegion() == pytest.approx((-1.0, 1.0))
        curve = travel.listDataItems()[0]
        assert np.max(curve.yData) == pytest.approx(
            inches(case.travel_up), rel=1e-6)
        for col, words in ((0, 'travel'), (1, 'peak velocity'),
                           (2, 'mean square removed')):
            assert words in layout.getItem(1, col).titleLabel.text
        layout.clear()
        logged_case = build_replication_fit(layout, fit, None,
                                            unit_system=us)
        assert logged_case is fit.as_logged
        assert 'As logged: does not fit' in \
            layout.getItem(0, 0).titleLabel.text
    finally:
        layout.close()
        layout.deleteLater()


def test_the_view_draws_to_a_file(qt_app, tmp_path):
    from visualdynamics.plot import plot_replication_fit

    path = tmp_path / 'fit.png'
    plot_replication_fit(logged(), {'force': 4000.0 * LBF,
                                    'stroke': 2.0 * INCH,
                                    'velocity': 70.0 * INCH,
                                    'moving_mass': 150.0 * LB},
                         cutoff=3.0, path=str(path), show=False)
    assert path.stat().st_size > 2000


def test_the_view_refuses_a_cutoff_it_was_not_given(qt_app):
    from visualdynamics.plot import plot_replication_fit

    with pytest.raises(KeyError):
        plot_replication_fit(tone(), rating(), cutoffs=(2.0,), cutoff=4.0,
                             show=False, path='unused.png')


# ---- the window ---------------------------------------------------------


def _looking_at(window, pump, history, name='Road'):
    window.add_object(name, history)
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()


def _reading(window, pump):
    window.data_pane.fit_action.trigger()
    pump()
    return window.data_pane.replication_fit_panel


def test_the_reading_is_offered_for_acceleration_only(window, pump):
    _looking_at(window, pump, logged())
    assert window.data_pane.fit_action.isVisible()
    _looking_at(window, pump, record(np.ones(4096), dims=['force']),
                name='Force')
    assert not window.data_pane.fit_action.isVisible(), \
        'nothing to integrate, so nothing to offer'
    hinted = TimeHistory(np.arange(4096) / RATE, np.ones((1, 4096)),
                         response_dof=['1Z+'], dimension_hint=['acceleration'])
    _looking_at(window, pump, hinted, name='Undeclared')
    assert not window.data_pane.fit_action.isVisible(), \
        'raw numbers, so nothing to judge in SI'


def test_the_panel_tabulates_the_check(window, pump):
    """A column per candidate after the record as logged, every number
    the core's, in the window's units."""
    _looking_at(window, pump, logged())
    panel = _reading(window, pump)
    assert panel.isVisible()
    table = panel.table
    headers = [table.horizontalHeaderItem(c).text()
               for c in range(table.columnCount())]
    assert headers == ['Logged', '2 Hz', '3 Hz', '5 Hz', '8 Hz']
    fits = [table.item(5, c).text() for c in range(table.columnCount())]
    assert fits == ['no', 'yes', 'yes', 'yes', 'yes']
    assert table.item(3, 1).text() == '1.25'
    assert table.item(3, 2).text() == '2.02'
    assert table.item(1, 0).text() == '664', 'the drift, in inches'
    assert panel.derived['force'].text().startswith('2098 lbf')
    assert 'starting point' in panel.note.text()
    assert table.selectedIndexes()[0].column() == 1, \
        'the lowest cutoff that fits is the one drawn'
    assert not window.data_pane.waterfall_action.isVisible()


def test_a_rating_edited_redraws_the_verdict(window, pump):
    _looking_at(window, pump, logged())
    panel = _reading(window, pump)
    panel.boxes['velocity'].setValue(22.0)                  # in/s
    pump()
    assert window.data_pane.fit_settings['velocity'] == pytest.approx(
        22.0 * INCH)
    assert panel.table.item(5, 1).text() == 'no'
    assert panel.table.selectedIndexes()[0].column() == 2, \
        'the lowest cutoff that still fits'
    assert 'starting point' not in panel.note.text()
    title = window.data_pane.graphics.getItem(0, 0).titleLabel.text
    assert title.startswith('3 Hz: fits')


def test_a_column_clicked_draws_that_cutoff(window, pump):
    _looking_at(window, pump, logged())
    panel = _reading(window, pump)
    panel.table.selectColumn(0)
    pump()
    assert window.data_pane.fit_settings['cutoff'] is None
    title = window.data_pane.graphics.getItem(0, 0).titleLabel.text
    assert title.startswith('As logged: does not fit')
    panel.table.selectColumn(4)
    pump()
    assert window.data_pane.fit_settings['cutoff'] == 8.0
    assert window.data_pane.graphics.getItem(0, 0).titleLabel.text \
        .startswith('8 Hz: fits')


def test_the_cutoffs_are_typed_and_a_bad_list_is_put_back(window, pump):
    _looking_at(window, pump, logged())
    panel = _reading(window, pump)
    panel.cutoffs_edit.setText('4, 1.5')
    panel.cutoffs_edit.editingFinished.emit()
    pump()
    assert window.data_pane.fit_settings['cutoffs'] == (1.5, 4.0)
    assert panel.table.columnCount() == 3
    panel.cutoffs_edit.setText('two')
    panel.cutoffs_edit.editingFinished.emit()
    pump()
    assert panel.cutoffs_edit.text() == '1.5, 4'
    assert window.data_pane.fit_settings['cutoffs'] == (1.5, 4.0)


def test_the_ratings_stay_with_the_window_across_records(window, pump):
    """A shaker typed once stays typed: the ratings are the pane's, not
    the record's."""
    _looking_at(window, pump, logged())
    panel = _reading(window, pump)
    panel.boxes['stroke'].setValue(3.0)
    pump()
    _looking_at(window, pump, tone(), name='Tone')
    assert window.data_pane.fit_action.isChecked()
    assert panel.boxes['stroke'].value() == pytest.approx(3.0)
    assert window.data_pane.fit_settings['stroke'] == pytest.approx(
        3.0 * INCH)


def test_a_rating_at_zero_says_why_and_draws_no_verdict(window, pump):
    _looking_at(window, pump, logged())
    panel = _reading(window, pump)
    panel.boxes['force'].setValue(0.0)
    pump()
    assert panel.table.columnCount() == 0
    assert 'force has to be above zero' in window.statusBar().currentMessage()


def test_the_reading_is_a_call_a_session_replays(window, pump, tmp_path,
                                                 monkeypatch):
    from conftest import fixture_path

    window.import_paths([fixture_path('plate', 'modal.nc4')])
    item = window._item_for_object('Time History')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    _reading(window, pump)
    line = window.project.journal[-1]
    assert line.startswith("visualdynamics.plot.plot_replication_fit("
                           "project['Time History'], {'force': ")
    assert "path='replication_fit.png'" in line
    # by its record index, as the stroke's is: a DOF can repeat
    history = window.project['Time History']
    first = next(i for i, dim in enumerate(history.ordinate_dim)
                 if dim == 'acceleration')
    assert f', channel={first}, cutoffs=' in line
    monkeypatch.chdir(tmp_path)
    exec(window.project.session_script(), {})              # noqa: S102
    assert (tmp_path / 'replication_fit.png').stat().st_size > 2000


def test_a_transient_target_alone_offers_the_fit(window, pump):
    """The reference is often the target itself, checked before the
    first pass is played."""
    from visualdynamics.core.data import TransientSpecification

    history = tone()
    target = TransientSpecification(history.abscissa, history.ordinate,
                                    response_dof=['1Z+'],
                                    ordinate_dim=['acceleration'])
    _looking_at(window, pump, target, name='Target')
    assert window.data_pane.fit_action.isVisible()
    panel = _reading(window, pump)
    assert panel.table.columnCount() == 1 + len(CUTOFFS)


def test_the_reading_wears_its_own_glyph(qt_app):
    from visualdynamics.gui.icons import control_icon

    ours = control_icon('replication_fit').pixmap(32).toImage()
    assert ours != control_icon('no-such-glyph').pixmap(32).toImage()


def test_a_spectrum_after_the_fit_keeps_its_3d_form(window, pump):
    """The fit is drawn flat, and its choice is sticky; a spectrum
    selected next is not the fit's, and keeps its 2D/3D toggle."""
    _looking_at(window, pump, tone(), name='Tone')
    _reading(window, pump)
    assert not window.data_pane.waterfall_action.isVisible()
    psds = window.project.compute_psds('Tone')
    window.show_object(psds)
    item = window._item_for_object(psds)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    assert not window.data_pane.fit_action.isVisible()
    assert window.data_pane.waterfall_action.isVisible()
    assert not window.data_pane.replication_fit_panel.isVisible()
