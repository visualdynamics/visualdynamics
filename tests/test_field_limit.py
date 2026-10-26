"""Field data to specification statistics: a set of spectra read onto
common lines, the limit taken over them, the straight-line envelope
written over that limit, and the specification it writes.

Pinned against hand values (an envelope and a mean plus k standard
deviations worked on three numbers; the tolerance factor against the
published one-sided values and against an independent quadrature over
the chi distribution; one record with a stated 3 dB scatter, which is
the 4.9 and 10.8 dB of a single flight), against sets drawn from a
known distribution (a limit at 95 percent coverage holds for its
stated share of fresh draws), against white noise averaged with the
default window (the distribution-free claim of the envelope), at the
grid's edges, and against every input that cannot be read one way,
each refused with its cause named.
"""

from __future__ import annotations

import copy
import itertools

import numpy as np
import pytest
from scipy import integrate, stats

from visualdynamics.core import field_limit as fl
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.compliance import log_interpolate, log_log_area
from visualdynamics.core.data import (
    Psd,
    ShockSpecification,
    Specification,
    Srs,
    TimeHistory,
)

DENSITY = 'acceleration**2/frequency'


def _psd(f, levels, dof='101Z+', dim=DENSITY, **kw):
    return Psd(np.asarray(f, float), np.atleast_2d(levels), dof,
               ordinate_dim=dim, **kw)


def _from_db(db, scale=10.0):
    return 10.0 ** (np.asarray(db, float) / scale)


def _profile(f):
    """A written random profile read onto lines: +3 dB/oct from 20 Hz to
    150 Hz, 0.04 to 800 Hz, -6 dB/oct to 2000 Hz (in SI, g's squared
    scaled)."""
    f = np.asarray(f, float)
    level = np.full(f.shape, 0.04)
    level = np.where(f < 150.0, 0.04 * (f / 150.0) ** (3 / 3.0103), level)
    level = np.where(f > 800.0, 0.04 * (f / 800.0) ** (-6 / 3.0103), level)
    return level


# ---- the set -------------------------------------------------------------


def test_one_objects_records_are_the_set_and_the_0_hz_line_is_left_out():
    f = np.array([0.0, 10.0, 20.0, 40.0])
    data = _psd(f, [[0.0, 1.0, 2.0, 4.0], [0.0, 3.0, 1.0, 2.0]])
    fieldset = fl.gather(data)
    assert fieldset.count == 2
    assert np.array_equal(fieldset.frequencies, [10.0, 20.0, 40.0])
    assert np.array_equal(fieldset.levels, [[1, 2, 4], [3, 1, 2]])
    assert any('0 Hz' in note for note in fieldset.notes)
    assert fieldset.kind == 'psd' and fieldset.db_scale == 10.0


def test_a_record_starting_above_0_hz_carries_no_note_about_it():
    data = _psd([10.0, 20.0, 40.0], [[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
    assert fl.gather(data).notes == []


def test_records_are_chosen_and_pairs_span_objects():
    f = [10.0, 20.0, 40.0]
    a = _psd(f, [[1.0, 2.0, 4.0], [5.0, 5.0, 5.0]])
    b = _psd(f, [[3.0, 1.0, 2.0]])
    assert fl.gather(a, [1]).count == 1
    both = fl.gather([(a, 0), (b, 0)])
    assert np.array_equal(both.levels, [[1, 2, 4], [3, 1, 2]])
    assert fl.gather([a, b]).count == 3
    with pytest.raises(ValueError, match='per object'):
        fl.gather([a, b], [0])


def test_records_on_other_lines_are_read_onto_the_first_records():
    a = _psd([10.0, 20.0, 40.0, 80.0], [[1.0, 1.0, 1.0, 1.0]])
    # the second spans 15 to 80 Hz on its own lines, a power law
    fb = np.array([15.0, 30.0, 60.0, 80.0])
    b = _psd(fb, [fb / 15.0])
    fieldset = fl.gather([a, b])
    assert np.array_equal(fieldset.frequencies, [20.0, 40.0, 80.0])
    assert np.allclose(fieldset.levels[1], [20 / 15, 40 / 15, 80 / 15])
    assert ('records on other lines were read onto the 3 lines of Psd '
            'record 1 (101Z+) from 20 to 80 Hz, log-log between their own'
            in fieldset.notes)


def test_a_line_between_a_bands_edge_and_the_next_line_is_read():
    """The first record has a line at 26 Hz, inside a band from 25 Hz;
    the second's lines are at 15 and 30 Hz, so 26 Hz is read between
    them, the one outside the band included."""
    a = _psd([10.0, 26.0, 40.0, 80.0], [[1.0, 1.0, 1.0, 1.0]])
    fb = np.array([15.0, 30.0, 60.0, 80.0])
    b = _psd(fb, [fb / 15.0])
    fieldset = fl.gather([a, b], band=(25.0, 80.0))
    assert np.array_equal(fieldset.frequencies, [26.0, 40.0, 80.0])
    assert np.allclose(fieldset.levels[1], [26 / 15, 40 / 15, 80 / 15])
    # with the line below the band a NaN, 26 Hz has nothing to be read
    # from: refused at the line, naming the record
    holed = _psd(fb, [np.where(fb == 15.0, np.nan, fb / 15.0)],
                 dof='102Z+')
    with pytest.raises(ValueError, match=r'record 1 \(102Z\+\) cannot be '
                       r'read at 26 Hz'):
        fl.gather([a, holed], band=(25.0, 80.0))


def test_a_band_reaching_past_the_shared_range_is_refused_by_its_gap():
    data = _psd([10.0, 20.0, 40.0, 80.0], [[1.0, 2.0, 3.0, 4.0]])
    assert np.array_equal(fl.gather(data, band=(20.0, 40.0)).frequencies,
                          [20.0, 40.0])
    with pytest.raises(ValueError, match=r'no data from 80 to 100 Hz'):
        fl.gather(data, band=(20.0, 100.0))
    with pytest.raises(ValueError, match=r'no data from 5 to 10 Hz'):
        fl.gather(data, band=(5.0, 40.0))
    with pytest.raises(ValueError, match='lower first'):
        fl.gather(data, band=(40.0, 20.0))
    with pytest.raises(ValueError, match='fewer than two lines'):
        fl.gather(data, band=(25.0, 35.0))
    with pytest.raises(ValueError, match='two positive frequencies'):
        fl.gather(data, band=(0.0, 40.0))
    with pytest.raises(ValueError, match='two positive frequencies'):
        fl.gather(data, band=(20.0, 20.0))
    # a band exactly on the range is the whole range, its edge lines in
    whole = fl.gather(data, band=(10.0, 80.0))
    assert np.array_equal(whole.frequencies, [10.0, 20.0, 40.0, 80.0])


def test_ranges_with_nothing_in_common_are_refused_naming_each():
    a = _psd([10.0, 20.0], [[1.0, 1.0]])
    b = _psd([30.0, 40.0], [[1.0, 1.0]], dof='102Z+')
    with pytest.raises(ValueError, match='share no frequency range') as caught:
        fl.gather([a, b])
    assert '10 to 20 Hz' in str(caught.value)
    assert '30 to 40 Hz' in str(caught.value)
    # ranges that touch at one frequency share no range either
    touching = _psd([20.0, 30.0], [[1.0, 1.0]], dof='102Z+')
    with pytest.raises(ValueError, match='share no frequency range'):
        fl.gather([a, touching])


def test_a_nan_is_refused_at_its_record_and_line():
    data = _psd([10.0, 20.0, 40.0], [[1.0, 2.0, 4.0], [3.0, np.nan, 2.0]])
    with pytest.raises(ValueError, match=r'record 2 \(101Z\+\) has no level '
                       r'at 20 Hz \(NaN\)'):
        fl.gather(data)


@pytest.mark.parametrize('at', [10.0, 40.0])
def test_a_nan_on_the_bands_edge_lines_is_refused(at):
    f = np.array([10.0, 20.0, 40.0])
    data = _psd(f, [[1.0, 2.0, 4.0], np.where(f == at, np.nan, 1.0)])
    with pytest.raises(ValueError, match=f'no level at {at:g} Hz'):
        fl.gather(data)


def test_a_dead_line_and_a_dead_channel_are_named():
    data = _psd([10.0, 20.0, 40.0], [[1.0, 2.0, 4.0], [3.0, 0.0, 2.0]])
    with pytest.raises(ValueError, match=r'record 2 .* level of 0 at 20 Hz'):
        fl.gather(data)
    dead = _psd([10.0, 20.0, 40.0], [[1.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match='zero at every line .* dead channel'):
        fl.gather(dead)


def test_a_constant_record_is_a_measurement_with_no_scatter():
    data = _psd([10.0, 20.0, 40.0], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    limit = fl.field_limit(fl.gather(data), coverage=0.95, confidence=0.95)
    assert np.allclose(limit.levels, 2.0)
    assert np.allclose(limit.scatter_db, 0.0)


def test_an_empty_selection_is_refused():
    data = _psd([10.0, 20.0], [[1.0, 1.0]])
    with pytest.raises(ValueError, match='no records chosen'):
        fl.gather(data, [])
    with pytest.raises(ValueError, match='no records chosen'):
        fl.gather([])


def test_quantities_and_units_must_agree():
    f = [10.0, 20.0]
    a = _psd(f, [[1.0, 1.0]])
    velocity = _psd(f, [[1.0, 1.0]], dim='velocity**2/frequency')
    raw = _psd(f, [[1.0, 1.0]], dim='unknown')
    with pytest.raises(ValueError, match='cannot be compared'):
        fl.gather([a, velocity])
    with pytest.raises(ValueError, match='not declared'):
        fl.gather([a, raw])
    both_raw = fl.gather([raw, raw])
    assert any('not declared' in note for note in both_raw.notes)


def test_kinds_q_and_cross_spectra_are_refused():
    f = [10.0, 20.0, 40.0]
    psd = _psd(f, [[1.0, 1.0, 1.0]])
    srs = Srs(f, [[1.0, 2.0, 3.0]], '101Z+', ordinate_dim='acceleration')
    other_q = Srs(f, [[1.0, 2.0, 3.0]], '101Z+', ordinate_dim='acceleration',
                  q=25.0)
    positive = Srs(f, [[1.0, 2.0, 3.0]], '101Z+', ordinate_dim='acceleration',
                   kind='positive')
    with pytest.raises(ValueError, match='mixes PSDs and SRSs'):
        fl.gather([psd, srs])
    with pytest.raises(ValueError, match='different Q'):
        fl.gather([srs, other_q])
    with pytest.raises(ValueError, match='different kinds'):
        fl.gather([srs, positive])
    cross = Psd(np.asarray(f), [[1.0 + 1.0j, 1.0, 1.0]], '101Z+',
                reference_dof='102Z+', ordinate_dim=DENSITY)
    with pytest.raises(ValueError, match='cross spectrum'):
        fl.gather(cross)
    spec = Specification(np.asarray(f), [[1.0, 1.0, 1.0]], '101Z+',
                         ordinate_dim=DENSITY)
    with pytest.raises(TypeError, match='a requirement already'):
        fl.gather(spec)
    history = TimeHistory(np.arange(3.0), [[1.0, 2.0, 3.0]], '101Z+')
    with pytest.raises(TypeError, match='not a PSD or an SRS'):
        fl.gather(history)
    with pytest.raises(ValueError, match='no record 3'):
        fl.gather(psd, [2])
    with pytest.raises(ValueError, match='no record 2'):
        fl.gather(psd, [1])
    with pytest.raises(ValueError, match='no record 0'):
        fl.gather(psd, [-1])
    real_cross = Psd(np.asarray(f), [[1.0, 1.0, 1.0]], '101Z+',
                     reference_dof='102Z+', ordinate_dim=DENSITY)
    with pytest.raises(ValueError, match='cross spectrum'):
        fl.gather(real_cross)
    # one DOF on both sides but two quantities: a drive point's
    # acceleration against its force gauge is a cross spectrum too
    drive_point = Psd(np.asarray(f), [[1.0, 1.0, 1.0]], '101Z+',
                      reference_dof='101Z+',
                      ordinate_dim='acceleration*force/frequency')
    with pytest.raises(ValueError, match='cross spectrum'):
        fl.gather(drive_point)


def test_a_set_spanning_dofs_says_it_is_one_population():
    f = [10.0, 20.0]
    data = Psd(np.asarray(f), [[1.0, 1.0], [2.0, 2.0]], ['101Z+', '102Z+'],
               ordinate_dim=DENSITY)
    assert any('2 DOFs' in note for note in fl.gather(data).notes)


# ---- the limits ----------------------------------------------------------


def _three_at_a_line():
    """Three records whose levels at their one line pair are 0, 2 and 4
    dB re 1 (a density): mean 2 dB, standard deviation 2 dB."""
    return _psd([10.0, 20.0], [_from_db([0.0, 0.0]), _from_db([2.0, 2.0]),
                               _from_db([4.0, 4.0])])


def test_the_envelope_is_the_largest_record_with_its_confidence():
    data = _psd([10.0, 20.0, 40.0], [[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
    limit = fl.field_limit(fl.gather(data), 'envelope', coverage=0.9)
    assert np.allclose(limit.levels, [3.0, 2.0, 4.0])
    assert limit.confidence == pytest.approx(1 - 0.9 ** 2)
    assert 'no distribution assumed' in limit.rests_on
    assert '19% confidence' in limit.rests_on


def test_mean_plus_k_is_worked_in_decibels():
    limit = fl.field_limit(fl.gather(_three_at_a_line()), 'mean_k', k=1.5)
    assert np.allclose(limit.mean_db, 2.0)
    assert np.allclose(limit.scatter_db, 2.0)
    assert np.allclose(limit.levels_db, 2.0 + 1.5 * 2.0)
    assert np.allclose(limit.levels, _from_db(5.0))
    assert limit.confidence is None and limit.factor == 1.5


def test_an_srs_is_taken_in_20_log10():
    f = [10.0, 20.0]
    srs = Srs(f, [_from_db([0.0, 0.0], 20), _from_db([2.0, 2.0], 20),
                  _from_db([4.0, 4.0], 20)], '101Z+',
              ordinate_dim='acceleration')
    limit = fl.field_limit(fl.gather(srs), 'mean_k', k=1.0)
    assert np.allclose(limit.levels, _from_db(4.0, 20))
    assert limit.fieldset.db_scale == 20.0


@pytest.mark.parametrize('count,expected', [
    (3, 7.656), (5, 4.203), (10, 2.911), (20, 2.396)])
def test_the_tolerance_factor_matches_the_one_sided_tables(count, expected):
    """The one-sided normal tolerance factors at 95 percent coverage and
    95 percent confidence printed in the statistical tables."""
    assert fl.tolerance_factor(count, 0.95, 0.95) == pytest.approx(
        expected, abs=1e-3)


@pytest.mark.parametrize('count,coverage,confidence', [
    (2, 0.95, 0.5), (4, 0.99, 0.9), (8, 0.95, 0.95), (30, 0.9, 0.75)])
def test_the_tolerance_factor_holds_by_quadrature(count, coverage,
                                                  confidence):
    """An independent truth: the probability that mean + k s stands above
    the population's quantile, integrated over the chi distribution of
    s, is the confidence asked."""
    k = fl.tolerance_factor(count, coverage, confidence)
    z = stats.norm.ppf(coverage)
    dof = count - 1

    def density(s):
        # s is the sample standard deviation over sigma
        return stats.chi2.pdf(dof * s * s, dof) * 2.0 * dof * s

    def covered(s):
        return stats.norm.cdf(np.sqrt(count) * (k * s - z)) * density(s)

    chance, _err = integrate.quad(covered, 0.0, np.inf, limit=200)
    assert chance == pytest.approx(confidence, abs=1e-6)


def test_a_known_scatter_gives_the_single_flight_numbers():
    one = _psd([10.0, 20.0], [[1.0, 1.0]])
    p95 = fl.field_limit(fl.gather(one), sigma_db=3.0, coverage=0.95,
                         confidence=0.5)
    assert np.allclose(p95.levels_db, 1.6449 * 3.0, atol=1e-3)
    assert np.allclose(p95.levels_db, 4.93, atol=0.01)
    p99 = fl.field_limit(fl.gather(one), sigma_db=3.0, coverage=0.99,
                         confidence=0.9)
    assert np.allclose(p99.levels_db, (2.3263 + 1.2816) * 3.0, atol=1e-3)
    assert 'stated, not measured' in p99.rests_on
    assert p99.scatter_db is None
    # four records, the known-scatter factor shrinks with the root of n
    assert fl.tolerance_factor(4, 0.99, 0.9, sigma_known=True) == \
        pytest.approx(2.3263 + 1.2816 / 2.0, abs=1e-4)


def test_a_stated_scatter_beside_a_measured_one_is_noted():
    two = _psd([10.0, 20.0], [_from_db([0.0, 0.0]), _from_db([2.0, 2.0])])
    noted = fl.field_limit(fl.gather(two), sigma_db=3.0).notes
    assert any('1.41 dB at the median line' in note for note in noted)
    limit = fl.field_limit(fl.gather(_three_at_a_line()), sigma_db=3.0)
    assert any('2.00 dB at the median line; 3 dB was stated' in note
               for note in limit.notes)


def test_at_50_percent_confidence_the_factor_falls_toward_z():
    z = stats.norm.ppf(0.95)
    factors = [fl.tolerance_factor(n, 0.95, 0.5) for n in (3, 10, 100, 1000)]
    assert all(a > b for a, b in itertools.pairwise(factors))
    assert factors[-1] == pytest.approx(z, abs=2e-3)
    assert factors[0] > z


@pytest.mark.parametrize('confidence', [0.5, 0.9])
def test_a_tolerance_limit_holds_for_its_share_of_fresh_draws(confidence):
    """Levels log-normal at every line with a known spread: each line is
    one trial. The limit stands above the population's 95th percentile
    on the stated share of lines, and the median line's limit covers
    95 percent of fresh draws when the confidence is one half."""
    rng = np.random.default_rng(1710)
    lines, count, sigma = 4000, 6, 2.5
    f = np.linspace(10.0, 2000.0, lines)
    truth_db = 3.0 * np.sin(np.log(f))
    draws = truth_db + rng.normal(0.0, sigma, (count, lines))
    limit = fl.field_limit(fl.gather(_psd(f, _from_db(draws))),
                           coverage=0.95, confidence=confidence)
    share = stats.norm.cdf((limit.levels_db - truth_db) / sigma)
    held = np.mean(share >= 0.95)
    spread = np.sqrt(confidence * (1 - confidence) / lines)
    assert abs(held - confidence) < 4 * spread
    fresh = truth_db + rng.normal(0.0, sigma, lines)
    under = np.mean(fresh <= limit.levels_db)
    if confidence == 0.5:
        assert float(np.median(share)) == pytest.approx(0.95, abs=0.01)
        assert under == pytest.approx(np.mean(share), abs=0.02)


def test_the_envelopes_claim_holds_on_white_noise_under_the_defaults():
    """The largest of n records stands above a share p of the population
    with probability 1 - p**n, whatever the distribution: checked on
    autospectra of white noise averaged with the default window (Hann),
    where a line's estimate is chi-square-like and neighboring lines
    are correlated. The population's 95th percentile is read off many
    more records pooled over lines, which are alike for white noise."""
    rng = np.random.default_rng(2610)
    fs, n = 1024.0, 4096
    averaging = Averaging(frame_length=256).filled(n, fs)

    def spectra(count):
        x = rng.standard_normal((count, n))
        history = TimeHistory(np.arange(n) / fs, x,
                              [f'{i + 1}Z+' for i in range(count)],
                              ordinate_dim='acceleration')
        psd = history.compute_psds(averaging)
        inside = (psd.abscissa > 20.0) & (psd.abscissa < 480.0)
        return psd, psd.ordinate[:, inside]

    psd, _values = spectra(14)
    _pool, pooled = spectra(120)
    quantile = np.quantile(pooled, 0.95)
    limit = fl.field_limit(fl.gather(psd, band=(21.0, 479.0)), 'envelope')
    held = np.mean(limit.levels >= quantile)
    assert limit.confidence == pytest.approx(1 - 0.95 ** 14)
    assert held == pytest.approx(1 - 0.95 ** 14, abs=0.1)


def test_a_distribution_free_limit_needs_its_count():
    assert fl.records_needed(0.95, 0.5) == 14
    assert fl.records_needed(0.95, 0.95) == 59
    assert fl.records_needed(0.5, 0.5) == 1
    f = [10.0, 20.0]
    rng = np.random.default_rng(3)
    small = _psd(f, rng.uniform(1, 2, (13, 2)))
    with pytest.raises(ValueError, match='needs at least 14 records'):
        fl.field_limit(fl.gather(small), 'distribution_free')
    enough = _psd(f, rng.uniform(1, 2, (14, 2)))
    limit = fl.field_limit(fl.gather(enough), 'distribution_free')
    assert np.allclose(limit.levels, enough.ordinate.max(axis=0))
    assert limit.confidence == 0.5
    assert '14 needed' in limit.rests_on


def test_what_a_limit_cannot_be_asked_is_refused():
    fieldset = fl.gather(_three_at_a_line())
    one = fl.gather(_psd([10.0, 20.0], [[1.0, 1.0]]))
    with pytest.raises(ValueError, match='unknown statistic'):
        fl.field_limit(fieldset, 'median')
    with pytest.raises(ValueError, match='strictly between 0 and 1'):
        fl.field_limit(fieldset, coverage=1.0)
    with pytest.raises(ValueError, match='strictly between 0 and 1'):
        fl.field_limit(fieldset, confidence=0.0)
    for bad in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError, match='strictly between 0 and 1'):
            fl.tolerance_factor(5, bad, 0.5)
        with pytest.raises(ValueError, match='strictly between 0 and 1'):
            fl.records_needed(0.95, bad)
    with pytest.raises(ValueError, match='needs k'):
        fl.field_limit(fieldset, 'mean_k')
    with pytest.raises(ValueError, match='sets its own'):
        fl.field_limit(fieldset, 'tolerance', k=2.0)
    with pytest.raises(ValueError, match='finite'):
        fl.field_limit(fieldset, 'mean_k', k=np.inf)
    with pytest.raises(ValueError, match='takes no standard deviation'):
        fl.field_limit(fieldset, 'envelope', sigma_db=3.0)
    with pytest.raises(ValueError, match='positive number of decibels'):
        fl.field_limit(fieldset, sigma_db=0.0)
    with pytest.raises(ValueError, match='no scatter to estimate'):
        fl.field_limit(one)
    with pytest.raises(ValueError, match='no scatter to estimate'):
        fl.field_limit(one, 'mean_k', k=1.0)
    with pytest.raises(ValueError, match='at least one record'):
        fl.tolerance_factor(0)
    with pytest.raises(ValueError, match='strictly between'):
        fl.records_needed(1.0, 0.5)
    # one record makes an envelope, and says how little it claims
    lone = fl.field_limit(one, 'envelope')
    assert lone.confidence == pytest.approx(0.05)
    assert lone.rests_on.startswith('1 record; ')
    assert lone.describe() == 'maximum envelope over 1 record'
    with pytest.raises(ValueError, match='the largest of 1 record stands'):
        fl.field_limit(one, 'distribution_free')


def test_a_count_takes_its_plural():
    assert fl._count(1, 'line') == '1 line'
    assert fl._count(0, 'line') == '0 lines'
    assert fl._count(2, 'record') == '2 records'


def test_the_limit_describes_itself():
    limit = fl.field_limit(fl.gather(_three_at_a_line()), coverage=0.95,
                           confidence=0.9)
    said = limit.describe()
    assert said.startswith('normal tolerance limit P95/90, k = ')
    assert said.endswith('over 3 records')
    assert 'log-normal' in limit.rests_on
    assert 'estimated from them' in limit.rests_on
    assert fl.field_limit(fl.gather(_three_at_a_line()), 'mean_k',
                          k=2).describe() == (
        'mean plus k standard deviations, k = 2.000 over 3 records')


# ---- the envelope written over the limit ----------------------------------


def test_a_written_profile_read_onto_lines_is_fitted_back_to_its_breakpoints():
    f = np.arange(20.0, 2001.0, 1.0)
    data = _psd(f, [_profile(f)])
    limit = fl.field_limit(fl.gather(data), 'envelope')
    fit = fl.fit_envelope(limit, max_overshoot_db=0.05)
    assert fit.met
    assert np.allclose(fit.frequencies, [20.0, 150.0, 800.0, 2000.0])
    assert np.allclose(fit.levels, _profile(fit.frequencies), rtol=1e-6)
    assert fit.overshoot_db < 1e-6
    assert fit.added_db == pytest.approx(0.0, abs=1e-6)
    assert fit.limit_rms == pytest.approx(
        np.sqrt(log_log_area(f, _profile(f), 20.0, 2000.0)))


def test_the_envelope_covers_every_line_within_its_allowance():
    rng = np.random.default_rng(44)
    f = np.geomspace(20.0, 2000.0, 60)
    ragged = _profile(f) * _from_db(rng.normal(0.0, 1.5, (5, f.size)))
    limit = fl.field_limit(fl.gather(_psd(f, ragged)), coverage=0.95)
    fit = fl.fit_envelope(limit, max_overshoot_db=2.0, max_breakpoints=None)
    over = 10 * np.log10(fit.at(f) / limit.levels)
    assert over.min() >= -1e-9
    assert over.max() <= 2.0 + 1e-9
    assert fit.met and fit.overshoot_db == pytest.approx(over.max())
    assert fit.frequencies[0] == f[0] and fit.frequencies[-1] == f[-1]
    assert fit.added_db > 0.0
    # removing any breakpoint breaks the allowance: none is spare
    x, y = np.log10(f), limit.levels_db
    index = [int(np.argmin(np.abs(f - b))) for b in fit.frequencies]
    for drop in range(1, len(index) - 1):
        trial = index[:drop] + index[drop + 1:]
        levels = fl._least_cover(x, y, trial)
        assert np.max(np.interp(x, x[trial], levels) - y) > 2.0


def test_the_breakpoints_allowed_stop_the_fit_and_it_says_where():
    rng = np.random.default_rng(45)
    f = np.geomspace(20.0, 2000.0, 200)
    ragged = _profile(f) * _from_db(rng.normal(0.0, 3.0, (4, f.size)))
    limit = fl.field_limit(fl.gather(_psd(f, ragged)), 'envelope')
    fit = fl.fit_envelope(limit, max_overshoot_db=0.5, max_breakpoints=6)
    assert fit.breakpoints <= 6
    assert not fit.met
    assert fit.overshoot_db > 0.5
    assert any('the most allowed' in note and 'past the 0.5 dB asked' in note
               for note in fit.notes)
    assert '(past the 0.5 dB asked)' in fit.describe()
    over = 10 * np.log10(fit.at(f) / limit.levels)
    assert over.min() >= -1e-9
    worst = f[int(np.argmax(over))]
    assert any(f'over the limit at {worst:g} Hz' in note for note in fit.notes)


def test_a_peak_on_the_first_and_last_lines_is_covered():
    f = np.geomspace(20.0, 2000.0, 40)
    level = np.full(f.size, 0.01)
    level[0] = level[-1] = 1.0
    limit = fl.field_limit(fl.gather(_psd(f, [level])), 'envelope')
    fit = fl.fit_envelope(limit, max_overshoot_db=1.0)
    assert fit.at([20.0])[0] >= 1.0 - 1e-9
    assert fit.at([2000.0])[0] >= 1.0 - 1e-9
    assert fit.met


def test_an_exact_cover_is_fitted_with_no_allowance_at_all():
    """With no overshoot allowed, a written profile read onto lines is
    fitted back to its four breakpoints and called met: the solver's own
    tolerance is not read as an overshoot."""
    f = np.arange(20.0, 2001.0, 5.0)
    limit = fl.field_limit(fl.gather(_psd(f, [_profile(f)])), 'envelope')
    fit = fl.fit_envelope(limit, max_overshoot_db=0.0)
    assert fit.met
    assert np.allclose(fit.frequencies, [20.0, 150.0, 800.0, 2000.0])


@pytest.mark.parametrize('seed', range(6))
def test_the_cover_is_never_under_the_curve(seed):
    """On the fitting axes, to rounding: the lift past the solver's
    tolerance holds every point under the lines."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(1.0, 3.0, 50))
    y = rng.normal(0.0, 3.0, 50)
    points, levels = fl.cover(x, y, 1.0, 8)
    assert np.min(fl._lines_at(x, points, levels) - y) >= -1e-12


def test_the_span_that_over_tests_most_is_split_first():
    """After the first split, the next breakpoint goes into the span
    with the most area between its line and the curve, counted over its
    width (a wide gap outweighs a tall narrow one)."""
    x = np.array([0.008, 0.345, 0.538, 0.568, 0.692, 0.752, 1.05, 1.624,
                  2.011, 2.574, 2.689, 2.84])
    y = np.array([-5.732, 0.441, -2.721, 5.326, 2.661, 2.848, -0.174,
                  1.839, 1.974, -1.033, -1.492, -0.344])
    three, levels = fl.cover(x, y, 0.5, 3)
    over = fl._lines_at(x, three, levels) - y
    areas = [np.trapezoid(over[p:q + 1], x[p:q + 1])
             for p, q in itertools.pairwise(three)]
    four, _levels = fl.cover(x, y, 0.5, 4)
    added = (set(four) - set(three)).pop()
    widest = int(np.argmax(areas))
    assert three[widest] < added < three[widest + 1]
    assert four == [0, 3, 6, 11]


def test_a_spike_one_line_wide_takes_its_own_breakpoint():
    limit = fl.field_limit(fl.gather(_psd([10.0, 20.0, 40.0],
                                          [[1.0, 100.0, 1.0]])), 'envelope')
    fit = fl.fit_envelope(limit, max_overshoot_db=0.5)
    assert fit.met and np.allclose(fit.frequencies, [10.0, 20.0, 40.0])
    assert np.allclose(fit.levels, [1.0, 100.0, 1.0])


def test_two_breakpoints_may_be_asked_for():
    f = np.geomspace(20.0, 2000.0, 30)
    limit = fl.field_limit(fl.gather(_psd(f, [_profile(f)])), 'envelope')
    fit = fl.fit_envelope(limit, max_overshoot_db=0.0, max_breakpoints=2)
    assert fit.breakpoints == 2 and not fit.met


def test_two_lines_are_their_own_envelope():
    limit = fl.field_limit(fl.gather(_psd([10.0, 20.0], [[1.0, 4.0]])),
                           'envelope')
    fit = fl.fit_envelope(limit)
    assert np.allclose(fit.frequencies, [10.0, 20.0])
    assert np.allclose(fit.levels, [1.0, 4.0])


def test_conserving_the_rms_lowers_the_cover_to_the_limits_area():
    rng = np.random.default_rng(46)
    f = np.geomspace(20.0, 2000.0, 80)
    ragged = _profile(f) * _from_db(rng.normal(0.0, 2.0, (6, f.size)))
    limit = fl.field_limit(fl.gather(_psd(f, ragged)), coverage=0.95)
    exceed = fl.fit_envelope(limit, 3.0)
    keep = fl.fit_envelope(limit, 3.0, 'conserve')
    assert np.array_equal(keep.frequencies, exceed.frequencies)
    assert np.allclose(keep.levels, exceed.levels * _from_db(-exceed.added_db))
    area = log_log_area(keep.frequencies, keep.levels, f[0], f[-1])
    assert area == pytest.approx(keep.limit_area, rel=1e-9)
    assert keep.envelope_rms == pytest.approx(keep.limit_rms)
    under = 10 * np.log10(limit.levels / keep.at(f))
    assert keep.below == int(np.sum(under > 1e-9)) > 0
    assert keep.deepest_db == pytest.approx(under.max())
    assert f'{fl._count(keep.below, "line")} under the limit' in \
        keep.describe()
    assert exceed.below == 0 and exceed.deepest_db == 0.0
    assert 'of area added' in exceed.describe()


def test_what_a_fit_cannot_be_asked_is_refused():
    limit = fl.field_limit(fl.gather(_psd([10.0, 20.0, 40.0],
                                          [[1.0, 2.0, 1.0]])), 'envelope')
    with pytest.raises(ValueError, match='zero or more'):
        fl.fit_envelope(limit, -1.0)
    with pytest.raises(ValueError, match='at least two breakpoints'):
        fl.fit_envelope(limit, max_breakpoints=1)
    with pytest.raises(ValueError, match='unknown rms rule'):
        fl.fit_envelope(limit, rms='keep')
    srs = Srs([10.0, 20.0, 40.0], [[1.0, 2.0, 1.0]], '101Z+',
              ordinate_dim='acceleration')
    srs_limit = fl.field_limit(fl.gather(srs), 'envelope')
    with pytest.raises(ValueError, match='no area to conserve'):
        fl.fit_envelope(srs_limit, rms='conserve')
    assert np.isnan(fl.fit_envelope(srs_limit).added_db)


def test_progress_is_told():
    told = []
    fl.derive(_psd([10.0, 20.0, 40.0], [[1.0, 2.0, 1.0], [2.0, 1.0, 2.0]]),
              progress=lambda done, total: told.append((done, total)))
    assert told and told[-1] == (1, 1)


# ---- the specification -----------------------------------------------------


def test_a_psd_envelope_writes_a_breakpoint_specification_with_margin():
    f = np.arange(20.0, 2001.0, 1.0)
    data = _psd(f, [_profile(f), 0.5 * _profile(f)], ordinate_unit='g')
    fit = fl.derive(data, statistic='envelope', max_overshoot_db=0.05)
    spec = fit.specification(margin_db=3.0)
    assert isinstance(spec, Specification)
    assert spec.interpolation == 'log_log'
    assert np.allclose(spec.abscissa, [20.0, 150.0, 800.0, 2000.0])
    assert np.allclose(np.real(spec.ordinate[0]),
                       _profile(spec.abscissa) * _from_db(3.0))
    assert spec.response_dof == ['101Z+']
    assert spec.ordinate_dim == [DENSITY]
    assert 'derived from 2 records' in spec.comment[0]
    assert '+3 dB margin' in spec.comment[0]


def test_an_srs_envelope_writes_a_shock_specification_at_its_q():
    f = 10.0 * 2.0 ** (np.arange(6 * 7 + 1) / 6.0)
    level = np.minimum(f, 100.0)
    srs = Srs(f, [level, 0.8 * level], '101Z+', ordinate_dim='acceleration',
              q=25.0, kind='positive')
    fit = fl.derive(srs, statistic='envelope', max_overshoot_db=0.05)
    assert np.allclose(fit.frequencies[[0, -1]], [f[0], f[-1]])
    spec = fit.specification(margin_db=6.0)
    assert isinstance(spec, ShockSpecification)
    assert spec.q == 25.0 and spec.kind == 'positive'
    expect = log_interpolate(spec.abscissa, fit.frequencies,
                             fit.levels * _from_db(6.0, 20))
    assert np.allclose(np.real(spec.ordinate[0]), expect, rtol=1e-3)
    assert not spec.has_limits
    from visualdynamics.core.octave import per_octave_of

    assert per_octave_of(spec.abscissa) == 6, 'the spectra\'s own spacing'


def test_the_specification_refuses_what_it_cannot_write():
    f = [10.0, 20.0, 40.0]
    raw = _psd(f, [[1.0, 2.0, 1.0]], dim='unknown')
    with pytest.raises(ValueError, match='units are not declared'):
        fl.derive(raw, statistic='envelope').specification()
    two = Psd(np.asarray(f), [[1.0, 2.0, 1.0], [2.0, 1.0, 2.0]],
              ['101Z+', '102Z+'], ordinate_dim=DENSITY)
    fit = fl.derive(two, statistic='envelope')
    with pytest.raises(ValueError, match='spans 101Z\\+, 102Z\\+'):
        fit.specification()
    assert fit.specification(dof='103X+').response_dof == ['103X+']
    with pytest.raises(ValueError, match='finite'):
        fit.specification(margin_db=np.nan, dof='101Z+')


def test_the_defaults_take_a_p95_50_tolerance_limit_and_a_3_db_cover():
    rng = np.random.default_rng(47)
    f = np.geomspace(20.0, 2000.0, 37)
    data = _psd(f, _profile(f) * _from_db(rng.normal(0, 2.0, (5, f.size))))
    fit = fl.derive(data)
    assert fit.limit.statistic == 'tolerance'
    assert (fit.limit.coverage, fit.limit.confidence) == (0.95, 0.5)
    assert fit.limit.factor == pytest.approx(fl.tolerance_factor(5, 0.95, 0.5))
    assert fit.max_overshoot_db == 3.0 and fit.rule == 'exceed'
    assert fit.max_breakpoints == fl.MAX_BREAKPOINTS == 12
    assert fl.settings_of(fit) == {
        'statistic': 'tolerance', 'coverage': 0.95, 'confidence': 0.5,
        'max_overshoot_db': 3.0, 'max_breakpoints': fl.MAX_BREAKPOINTS,
        'rms': 'exceed'}
    again = fl.derive(data, **fl.settings_of(fit))
    assert np.array_equal(again.levels, fit.levels)


# ---- the project ------------------------------------------------------------


def _flights_project():
    from visualdynamics.project import Project

    rng = np.random.default_rng(48)
    f = np.geomspace(20.0, 2000.0, 37)
    project = Project()
    project.add('Flights', _psd(f, _profile(f) * _from_db(
        rng.normal(0, 2.0, (5, f.size))), ordinate_unit='g'))
    return project


def test_the_project_derives_a_specification_and_refreshes_it():
    project = _flights_project()
    name = project.derive_specification('Flights', [0, 1, 2, 3],
                                        coverage=0.9, margin_db=3.0)
    assert name == 'Flights Derived Specification'
    fit = fl.derive(project['Flights'], [0, 1, 2, 3], coverage=0.9)
    assert np.allclose(project[name].abscissa, fit.frequencies)
    assert np.allclose(np.real(project[name].ordinate[0]),
                       fit.margined(3.0))
    record = project.provenance[name]
    assert record['verb'] == 'derive_specification'
    assert record['params']['records'] == [0, 1, 2, 3]
    assert project.journal[-1] == (
        "project.derive_specification('Flights', [0, 1, 2, 3], "
        'coverage=0.9, margin_db=3.0)')
    assert project.stale() == {}
    project['Flights'].ordinate[0] *= 2.0
    assert name in project.stale()
    project.refresh(name)
    again = fl.derive(project['Flights'], [0, 1, 2, 3], coverage=0.9)
    assert np.allclose(np.real(project[name].ordinate[0]),
                       again.margined(3.0))


def test_the_project_reads_a_set_across_objects_by_pairs():
    project = _flights_project()
    project.add('More', copy.deepcopy(project['Flights']))
    fit = project.field_limit([('Flights', 0), ('More', 1)],
                              statistic='envelope')
    assert fit.limit.count == 2
    name = project.derive_specification([('Flights', 0), ('More', 1)],
                                        statistic='envelope')
    assert project.provenance[name]['params']['sources'] == [
        ['Flights', 0], ['More', 1]]
    assert fit.limit.fieldset.labels[1] == 'Psd record 2 (101Z+)'
    project.refresh(name)
    with pytest.raises(ValueError, match='per object'):
        project.field_limit(['Flights', 'More'], [0])


def test_a_refused_derivation_adds_nothing_and_journals_nothing():
    project = _flights_project()
    before = list(project.journal)
    with pytest.raises(ValueError, match='needs at least 14 records'):
        project.derive_specification('Flights',
                                     statistic='distribution_free')
    assert list(project) == ['Flights'] and project.journal == before


def test_the_guides_example_reads_as_the_guide_prints_it():
    """The flights of the guide page and its figure, and what the page
    says of them."""
    g = 9.80665
    f = np.geomspace(20.0, 2000.0, 41)
    profile = (0.02 * np.minimum(1.0, f / 120.0)
               * np.minimum(1.0, (900.0 / f) ** 2)
               * (1.0 + 3.0 / (1.0 + ((f - 300.0) / 40.0) ** 2)))
    rng = np.random.default_rng(2608)
    flights = _psd(f, profile * _from_db(rng.normal(0.0, 1.5, (8, f.size)))
                   * g * g, ordinate_unit='g')
    fit = fl.derive(flights, statistic='tolerance', coverage=0.95,
                    confidence=0.5)
    assert fit.limit.describe() == ('normal tolerance limit P95/50, '
                                    'k = 1.719 over 8 records')
    assert fit.describe() == ('6 breakpoints, at most 2.87 dB over the '
                              'limit, +1.07 dB of area added')
    assert np.allclose(fit.frequencies, [20.0, 178.3, 317.0, 447.7, 1588.7,
                                         2000.0], atol=0.05)
    assert fit.limit.confidence == 0.5
    envelope = fl.field_limit(fit.limit.fieldset, 'envelope')
    assert round(envelope.confidence, 2) == 0.34
    with pytest.raises(ValueError, match='only 34% confidence.*at least 14'):
        fl.field_limit(fit.limit.fieldset, 'distribution_free')
    assert fl.tolerance_factor(8, 0.95, 0.5) == pytest.approx(1.719,
                                                              abs=5e-4)
