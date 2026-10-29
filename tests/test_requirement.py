"""A random test's requirement, and the run against it criterion by
criterion (`core.requirement`).

What a requirement refuses, how it is stored and said, and each row
read on runs whose numbers are known from their construction. The
reference is flat; a run is the reference times a known factor, so a
stretch eight decibels high stands two decibels over a plus or minus
six band, and a line a tenth of the reference four decibels under it.
The decision rules are checked against simulated scatter: a run held
exactly on its reference, each line a chi-square about it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from visualdynamics.core import requirement as rq
from visualdynamics.core.chance import chance_outside
from visualdynamics.core.data import Psd, Specification

#: the reference level, (m/s²)²/Hz, flat from 20 Hz to 2 kHz
LEVEL = 0.4
PSD_DIM = 'acceleration**2/frequency'
LINES = np.arange(0.0, 2001.0, 5.0)
#: eight decibels, two over a plus or minus six band
HOT = 10.0 ** 0.8
#: a stepped band: plus or minus 3 dB to 500 Hz, plus or minus 6 above
STEPPED = [(None, 500, -3, 3), (500, None, -6, 6)]


def reference(channels=('1Z+',), upper=2000.0):
    return Specification([20.0, upper], np.full((len(channels), 2), LEVEL),
                         response_dof=list(channels),
                         ordinate_dim=[PSD_DIM] * len(channels))


def run(*factors, hot=(), nan_at=None, lines=LINES, dims=None):
    """Channel k the reference times `factors[k]`; each (k, low, high,
    factor) of `hot` multiplies channel k's lines from low to high."""
    values = np.array([np.full(lines.size, LEVEL * f) for f in factors])
    for k, low, high, factor in hot:
        values[k, (lines >= low) & (lines <= high)] *= factor
    if nan_at is not None:
        values[nan_at[0], nan_at[1]] = np.nan
    out = Psd(lines, values,
              response_dof=[f'{k + 1}Z+' for k in range(len(factors))],
              ordinate_dim=dims or [PSD_DIM] * len(factors))
    out.scale_db = 0
    return out


def single(**kwargs):
    kwargs.setdefault('composite_bands', [(None, None, -6, 6)])
    return rq.TestRequirement('Test plan TP-12', **kwargs)


def average(**kwargs):
    kwargs.setdefault('composite_bands', [(None, None, -3, 3)])
    kwargs.setdefault('individual_bands', [(None, None, -6, 6)])
    kwargs.setdefault('channels', ('1Z+', '2Z+', '3Z+', '4Z+'))
    return rq.TestRequirement('Test plan TP-12', strategy='average',
                              **kwargs)


def rows(results, key=None):
    return [row for row in results.rows if key is None or row.key == key]


# ---- the requirement ------------------------------------------------------


@pytest.mark.parametrize(('kwargs', 'said'), [
    ({'document': ' '}, 'names its governing document'),
    ({'composite_bands': []}, 'states its control band'),
    ({'composite_bands': [(None, 500, -3, 3), (400, None, -6, 6)]},
     'overlap'),
    ({'composite_bands': [(None, None, 1, 3)]}, 'about the reference'),
    ({'composite_bands': [(None, None, -math.inf, math.inf)]},
     'no edge on either side'),
    ({'composite_bands': [(500, 400, -3, 3)]}, 'low edge is below'),
    ({'composite_bands': [(0, 400, -3, 3)]}, 'above 0 Hz'),
    ({'composite_bands': [(None, None, float('nan'), 3)]}, 'not NaN'),
    ({'strategy': 'median'}, 'the strategy is one of'),
    ({'individual_bands': [(None, None, -6, 6)]}, 'for a multi-point'),
    ({'channels': ('1Z+', '2Z+')}, 'controls one channel'),
    ({'channels': ('1Z+', '1Z+')}, 'named twice'),
    ({'weights': (1.0,)}, 'weights are for the weighted strategy'),
    ({'grms_percent': 0}, 'above 0'),
    ({'grms_percent': -5}, 'above 0'),
    ({'duration': 180}, 'with its tolerance'),
    ({'duration_tolerance': (5, 5)}, 'needs the duration'),
    ({'duration': 180, 'duration_tolerance': (100, 5)}, 'under 100'),
    ({'duration': 180, 'duration_tolerance': (-1, 5)}, 'at least 0'),
    ({'min_dof': 0}, 'above 0'),
    ({'max_line_spacing': math.inf}, 'above 0'),
    ({'rule': 'eyeball'}, 'decision rule is one of'),
    ({'combine': 4}, 'only under the combined rule'),
    ({'rule': 'combined'}, 'two or more adjacent lines'),
    ({'sources': {'level': ('customer', 'Prime')}}, 'stated for one of'),
    ({'sources': {'grms_percent': ('supplier', '')}}, 'a source is one of'),
    ({'cross_axis': [(None, 500, 0)]}, 'percentage above 0'),
    ({'cross_axis': [(None, 500, math.inf)]}, 'percentage above 0'),
    ({'composite_bands': [(500, 500, -3, 3)]}, 'low edge is below'),
    ({'composite_bands': [(20, math.inf, -3, 3)]}, 'above 0 Hz'),
    ({'duration': 180, 'duration_tolerance': (5, math.inf)}, 'at least 0'),
    ({'duration': 180, 'duration_tolerance': (5, -1)}, 'at least 0'),
])
def test_what_the_plan_could_not_have_meant_is_refused(kwargs, said):
    kwargs = {'document': 'Test plan TP-12',
              'composite_bands': [(None, None, -6, 6)], **kwargs}
    with pytest.raises(ValueError, match=said):
        rq.TestRequirement(**kwargs)


@pytest.mark.parametrize(('kwargs', 'said'), [
    ({'strategy': 'average', 'channels': ('1Z+',)}, 'two or more'),
    ({'strategy': 'average'}, "each control channel's band"),
    ({'strategy': 'weighted', 'individual_bands': [(None, None, -6, 6)]},
     'needs a weight per channel'),
    ({'strategy': 'weighted', 'channels': ('1Z+', '2Z+'),
      'weights': (1.0, 2.0, 3.0), 'individual_bands': [(None, None, -6, 6)]},
     '3 weights for 2 channels'),
])
def test_a_multi_point_requirement_states_what_it_combines(kwargs, said):
    with pytest.raises(ValueError, match=said):
        rq.TestRequirement('Test plan TP-12', [(None, None, -3, 3)],
                           **kwargs)


def test_a_value_of_the_wrong_kind_is_a_type_error():
    with pytest.raises(TypeError, match='list of bands'):
        rq.TestRequirement('TP', 'plus or minus 3 dB')
    with pytest.raises(TypeError, match='a number, not True'):
        rq.TestRequirement('TP', [(None, None, -3, 3)], grms_percent=True)
    with pytest.raises(TypeError, match='whole number'):
        rq.TestRequirement('TP', [(None, None, -3, 3)], rule='combined',
                           combine=2.5)


def test_a_requirement_is_stored_and_read_back_whole():
    requirement = rq.TestRequirement(
        'Test plan TP-12', STEPPED + [], revision='B', clause='±3 dB',
        strategy='weighted', channels=('1Z+', '2Z+'), weights=(3, 1),
        individual_bands=[(None, None, -math.inf, 6)], grms_percent=10,
        duration=180, duration_tolerance=(0, 10), min_dof=120,
        max_line_spacing=2.5, cross_axis=[(None, 500, 50)],
        rule='combined', combine=4,
        sources={'composite_bands': ('customer', 'Prime'),
                 'rule': 'laboratory'})
    stored = json.loads(json.dumps(requirement.as_record()))
    assert stored['individual_bands'] == [[None, None, 'down', 6.0]]
    assert rq.TestRequirement.from_record(stored) == requirement


def test_a_stored_requirement_with_a_field_it_does_not_take_is_refused():
    stored = single().as_record() | {'level_db': -6}
    with pytest.raises(ValueError, match="no field 'level_db'"):
        rq.TestRequirement.from_record(stored)
    with pytest.raises(TypeError, match='mapping'):
        rq.TestRequirement.from_record(['TP'])


def test_the_requirement_in_words_names_each_number_and_its_owner():
    lines = rq.TestRequirement(
        'Test plan TP-12', STEPPED, revision='B',
        clause='Within ±3 dB below 500 Hz', grms_percent=10, duration=180,
        duration_tolerance=(5, 5), min_dof=120, max_line_spacing=2.5,
        sources={'composite_bands': ('customer', 'Prime'),
                 'min_dof': ('laboratory', 'Test engineer')}).describe()
    assert lines[0] == 'Governing document: Test plan TP-12, revision B.'
    assert lines[1] == 'Tolerance clause: "Within ±3 dB below 500 Hz"'
    assert lines[2] == 'Control strategy: single control channel.'
    assert lines[3] == (
        'Control band: −3/+3 dB up to 500 Hz; −6/+6 dB from 500 Hz, '
        'supplied by the customer (Prime); the laboratory has not verified '
        'its derivation.')
    assert 'Duration at level: 3 min, −5/+5 %.' in lines
    assert ('Degrees of freedom: at least 120, from the laboratory (Test '
            'engineer).') in lines
    assert lines[-1] == ('Decision rule: any line outside its band is a '
                         'miss.')


def test_a_multi_point_requirement_says_its_strategy_and_both_bands():
    lines = average(weights=None).describe()
    assert ('Control strategy: average of the control channels 1Z+, 2Z+, '
            '3Z+, 4Z+.') in lines
    assert 'Composite band: −3/+3 dB over the whole band.' in lines
    assert ('Individual band, each control channel: −6/+6 dB over the '
            'whole band.') in lines


# ---- a single control channel ---------------------------------------------


def test_a_run_on_its_reference_meets_every_criterion():
    results = rq.evaluate(single(grms_percent=10), reference(), run(1.0))
    assert [(row.name, row.state) for row in results.rows] == [
        ('Channel 1Z+', 'met'), ('Overall rms', 'met')]
    assert results.rows[0].measured == 'inside the band at all 397 lines'
    assert results.state == 'met' and results.met == 2


def test_a_stretch_over_its_band_is_given_by_its_edges_size_and_lines():
    """Eight dB over 600 to 700 Hz: two over plus or minus 6, on the 21
    lines whose bins run 597.5 to 702.5 Hz."""
    results = rq.evaluate(single(), reference(),
                          run(1.0, hot=[(0, 600, 700, HOT)]))
    (row,) = results.rows
    assert row.state == 'not met'
    (miss,) = row.excursions
    assert (miss.side, miss.low, miss.high, miss.lines) == (
        'above', 597.5, 702.5, 21)
    assert miss.worst_db == pytest.approx(2.0, abs=1e-9)
    assert row.measured == ('above by up to 2 dB from 597.5 to 702.5 Hz '
                            '(21 lines)')


def test_a_stretch_under_its_band_is_given_as_below():
    results = rq.evaluate(single(), reference(),
                          run(1.0, hot=[(0, 800, 820, 0.1)]))
    (miss,) = results.rows[0].excursions
    assert (miss.side, miss.low, miss.high, miss.lines) == (
        'below', 797.5, 822.5, 5)
    assert miss.worst_db == pytest.approx(4.0, abs=1e-9)


def test_a_stepped_band_is_a_row_per_range():
    results = rq.evaluate(single(composite_bands=STEPPED), reference(),
                          run(1.0, hot=[(0, 400, 450, HOT)]))
    low, high = results.rows
    assert (low.required, low.state) == ('−3/+3 dB up to 500 Hz', 'not met')
    assert low.excursions[0].worst_db == pytest.approx(5.0, abs=1e-9)
    assert (high.required, high.state) == ('−6/+6 dB from 500 Hz', 'met')
    assert low.band == (0.0, 500.0) and high.band == (500.0, math.inf)


def test_a_line_across_a_step_is_judged_on_each_side_over_its_own_part():
    """5 dB at the line on 500 Hz: out of the plus or minus 3 band over
    its lower half bin, inside plus or minus 6 over its upper half."""
    results = rq.evaluate(single(composite_bands=STEPPED), reference(),
                          run(1.0, hot=[(0, 500, 500, 10 ** 0.5)]))
    low, high = results.rows
    (miss,) = low.excursions
    assert (miss.low, miss.high, miss.lines) == (497.5, 500.0, 1)
    assert miss.worst_db == pytest.approx(2.0, abs=1e-9)
    assert high.state == 'met'


def test_a_miss_on_the_first_line_of_the_band_starts_at_the_band():
    results = rq.evaluate(single(composite_bands=STEPPED), reference(),
                          run(1.0, hot=[(0, 20, 20, 10 ** 0.5)]))
    (miss,) = results.rows[0].excursions
    assert (miss.low, miss.high) == (20.0, 22.5)


def test_a_dead_channel_is_not_met_and_named():
    results = rq.evaluate(single(grms_percent=10), reference(), run(0.0))
    band, grms = results.rows
    assert band.state == grms.state == 'not met'
    assert band.measured.startswith('below reading zero from 20 to 2000 Hz')
    assert 'dead or disconnected sensor' in grms.measured
    assert results.notes == (('1Z+ reads zero at 397 lines, as a dead or '
                              'disconnected sensor does'),)


def test_a_sample_that_is_not_a_number_leaves_its_row_not_judged():
    """Never met: one line was not read, and the words say how many
    were."""
    results = rq.evaluate(single(), reference(), run(1.0, nan_at=(0, 200)))
    (row,) = results.rows
    assert row.state == 'not judged'
    assert row.measured == ('inside the band at 396 lines; 1 line not '
                            'judged (not a finite density)')
    assert results.state == 'not judged'


def test_a_miss_beside_a_line_not_read_is_still_not_met():
    results = rq.evaluate(single(), reference(),
                          run(1.0, nan_at=(0, 200),
                              hot=[(0, 600, 600, HOT)]))
    assert results.rows[0].state == 'not met'
    assert results.rows[0].measured.endswith('; 1 line not judged')


def test_a_band_the_specification_does_not_reach_is_not_judged():
    results = rq.evaluate(single(composite_bands=[(3000, 4000, -3, 3)]),
                          reference(), run(1.0))
    (row,) = results.rows
    assert row.state == 'not judged'
    assert 'holds nothing the specification writes' in row.measured
    assert 'runs from 20 to 2000 Hz' in row.measured


def test_a_run_of_another_quantity_is_not_judged_and_says_why():
    results = rq.evaluate(single(), reference(),
                          run(1.0, dims=['velocity**2/frequency']))
    assert results.rows[0].state == 'not judged'
    assert 'not judged: ' in results.rows[0].measured


def test_the_one_channel_is_named_found_or_refused():
    specification = reference(('1Z+', '2Z+'))
    measured = run(1.0, 1.0)
    with pytest.raises(ValueError, match='2 autospectra are candidates'):
        rq.evaluate(single(), specification, measured)
    with pytest.raises(ValueError, match='no autospectrum at 9Z'):
        rq.evaluate(single(channels=('9Z+',)), specification, measured)
    results = rq.evaluate(single(channels=('2Z+',)), specification, measured)
    assert results.channels == ('2Z+',)


def test_the_reference_and_the_run_are_refused_by_kind():
    with pytest.raises(TypeError, match='random specification'):
        rq.evaluate(single(), run(1.0), run(1.0))
    with pytest.raises(TypeError, match='measured PSD'):
        rq.evaluate(single(), reference(), reference())
    with pytest.raises(TypeError, match='TestRequirement'):
        rq.evaluate({'document': 'TP'}, reference(), run(1.0))


# ---- a multi-point run ----------------------------------------------------


def test_a_dead_channel_in_an_average_is_caught_on_its_own_row():
    """Three live channels and a dead one average to three quarters of
    the reference, inside plus or minus 3 dB; the dead channel is out of
    its own band at every line, and the reading names it."""
    results = rq.evaluate(average(), reference(), run(1.0, 1.0, 0.0, 1.0))
    composite = rows(results, 'composite')[0]
    assert composite.name == 'Composite (average of 1Z+, 2Z+, 3Z+, 4Z+)'
    assert composite.state == 'met'
    channels = {row.channel: row.state for row in rows(results, 'channel')}
    assert channels == {'1Z+': 'met', '2Z+': 'met', '3Z+': 'not met',
                        '4Z+': 'met'}
    assert any('3Z+ reads zero at every line' in note
               for note in results.notes)
    assert results.state == 'not met'


def test_a_hot_spot_diluted_by_the_average_is_caught_on_its_channel():
    """One channel eight dB high over 600 to 700 Hz lifts the average of
    four by 10 log10((3 + 6.31) / 4) = 3.67 dB there: over plus or minus
    3 by 0.67 dB; the channel is two over its own plus or minus 6."""
    results = rq.evaluate(average(), reference(),
                          run(1.0, 1.0, 1.0, 1.0, hot=[(1, 600, 700, HOT)]))
    composite = rows(results, 'composite')[0]
    want = 10 * math.log10((3 + HOT) / 4) - 3
    assert composite.excursions[0].worst_db == pytest.approx(want, abs=1e-9)
    hot = [row for row in rows(results, 'channel') if row.channel == '2Z+']
    assert hot[0].excursions[0].worst_db == pytest.approx(2.0, abs=1e-9)


def test_the_overall_rms_is_of_the_composite_under_multi_point():
    results = rq.evaluate(average(grms_percent=10), reference(),
                          run(1.21, 1.21, 1.21, 1.21))
    (grms,) = rows(results, 'grms')
    assert grms.name == 'Overall rms (composite)'
    assert grms.measured == '+10.0 % of the reference rms'
    assert grms.state == 'met'


def test_weights_that_do_not_fit_the_channels_found_are_refused():
    """Three weights, and the four channels that pair with the
    specification: refused by the multi-point reading's own rule."""
    requirement = rq.TestRequirement(
        'TP', [(None, None, -3, 3)], strategy='weighted',
        weights=(1.0, 2.0, 3.0), individual_bands=[(None, None, -6, 6)])
    with pytest.raises(ValueError, match='3 weights for 4 channels'):
        rq.evaluate(requirement, reference(('1Z+', '2Z+', '3Z+', '4Z+')),
                    run(1.0, 1.0, 1.0, 1.0))


# ---- the other criteria ---------------------------------------------------


@pytest.mark.parametrize(('factor', 'said', 'state'), [
    (1.21, '+10.0 %', 'met'),            # on the tolerance: met
    (1.25, '+11.8 %', 'not met'),
    (0.81, '−10.0 %', 'met'),
    (0.80, '−10.6 %', 'not met'),
])
def test_the_overall_rms_against_its_tolerance(factor, said, state):
    results = rq.evaluate(single(grms_percent=10), reference(), run(factor))
    (grms,) = rows(results, 'grms')
    assert grms.measured.replace('-', '−') == f'{said} of the reference rms'
    assert grms.state == state


@pytest.mark.parametrize(('run_s', 'state'), [
    (179.9, 'not met'), (180.0, 'met'), (198.0, 'met'), (198.1, 'not met')])
def test_the_duration_against_an_asymmetric_tolerance(run_s, state):
    """Minus 0, plus 10 percent of 3 minutes: 180 to 198 seconds."""
    results = rq.evaluate(single(duration=180, duration_tolerance=(0, 10)),
                          reference(), run(1.0), duration_run=run_s)
    (row,) = rows(results, 'duration')
    assert (row.required, row.state) == ('3 min, −0/+10 %', state)


def test_a_duration_not_stated_for_the_run_is_not_judged():
    results = rq.evaluate(single(duration=180, duration_tolerance=(5, 5)),
                          reference(), run(1.0))
    (row,) = rows(results, 'duration')
    assert row.state == 'not judged'
    assert 'no duration at level stated' in row.measured
    with pytest.raises(ValueError, match='seconds above 0'):
        rq.evaluate(single(), reference(), run(1.0), duration_run=0)


def test_the_degrees_of_freedom_are_the_fewest_channels():
    measured = run(1.0, 1.0, 1.0, 1.0)
    measured.statistical_dof = np.array([120.0, 90.0, 120.0, 120.0])
    results = rq.evaluate(average(min_dof=100), reference(), measured)
    (row,) = rows(results, 'dof')
    assert (row.measured, row.state) == (
        '90, the fewest of the control channels', 'not met')
    measured.statistical_dof = None
    (row,) = rows(rq.evaluate(average(min_dof=100), reference(), measured),
                  'dof')
    assert row.state == 'not judged'
    assert 'does not carry its degrees of freedom' in row.measured


@pytest.mark.parametrize(('widest', 'state'), [(5.0, 'met'),
                                               (2.5, 'not met')])
def test_the_line_spacing(widest, state):
    results = rq.evaluate(single(max_line_spacing=widest), reference(),
                          run(1.0))
    (row,) = rows(results, 'line_spacing')
    assert (row.measured, row.state) == ('5 Hz', state)


def test_lines_not_evenly_spaced_have_no_line_spacing():
    lines = np.geomspace(20.0, 2000.0, 60)
    results = rq.evaluate(single(max_line_spacing=5.0), reference(),
                          run(1.0, lines=lines))
    (row,) = rows(results, 'line_spacing')
    assert row.state == 'not judged'


def test_a_cross_axis_limit_stated_and_not_read_keeps_the_result_open():
    results = rq.evaluate(single(cross_axis=[(None, 500, 50)]), reference(),
                          run(1.0))
    (row,) = rows(results, 'cross_axis')
    assert (row.required, row.state) == (
        '50 % of in-axis up to 500 Hz', 'not judged')
    assert results.state == 'not judged', 'nothing judged is never met'


def test_a_scale_detected_from_the_data_is_said():
    measured = run(10 ** -0.6)
    measured.scale_db = None
    results = rq.evaluate(single(), reference(), measured)
    assert results.scale_db == pytest.approx(6.0)
    assert any('detected from the data as a run at -6 dB' in note
               for note in results.notes)


# ---- the decision rule ----------------------------------------------------


def _scatter(dof, seed=1):
    """A run whose every line is its true level, the reference, times a
    chi-square over `dof`: an averaged estimate of a run held exactly on
    its reference, on 5 Hz lines to 20 kHz, its lines independent."""
    lines = np.arange(0.0, 20001.0, 5.0)
    rng = np.random.default_rng(seed)
    values = LEVEL * rng.chisquare(dof, lines.size) / dof
    out = Psd(lines, values[None], response_dof=['1Z+'],
              ordinate_dim=[PSD_DIM])
    out.scale_db = 0
    out.statistical_dof = np.array([float(dof)])
    out.line_correlation = np.array([1.0])
    return out


def test_scatter_alone_puts_lines_outside_a_tight_band_as_chance_says():
    """At 120 DOF about one line in 13 reads outside plus or minus 1 dB
    of its true level: a run held on its reference fails a one-line
    rule by chance, and the count is the one chance predicts."""
    measured = _scatter(120)
    results = rq.evaluate(single(composite_bands=[(None, None, -1, 1)]),
                          reference(upper=20000.0), measured)
    (row,) = results.rows
    outside = sum(e.lines for e in row.excursions)
    judged = int(((measured.abscissa >= 20) & (measured.abscissa <= 20000))
                 .sum())
    expected = judged * chance_outside(120, -1.0, 1.0).total
    assert row.state == 'not met'
    assert abs(outside - expected) < 4 * math.sqrt(expected)


def test_combined_lines_steady_the_scatter_and_keep_a_real_excursion():
    """Eight lines combined at 120 DOF each: about 960 a group, and
    chance alone puts almost no group outside plus or minus 1 dB; a
    stretch 3 dB high over 40 lines still is."""
    combined = single(composite_bands=[(None, None, -1, 1)],
                      rule='combined', combine=8)
    quiet = rq.evaluate(combined, reference(upper=20000.0), _scatter(120))
    (row,) = quiet.rows
    assert row.state == 'met'
    assert '8 lines a group at about 960 degrees of freedom' in row.measured
    loud = _scatter(120)
    keep = (loud.abscissa >= 5000) & (loud.abscissa < 5200)
    loud.ordinate[0, keep] *= 2.0
    (row,) = rq.evaluate(combined, reference(upper=20000.0), loud).rows
    assert row.state == 'not met'
    (miss,) = row.excursions
    # its edges are the edges of the groups of 8 lines (40 Hz) it lies in
    assert 4997.5 - 40 < miss.low <= 4997.5
    assert 5197.5 <= miss.high < 5197.5 + 40
    assert miss.lines == round((miss.high - miss.low) / 40)
    assert miss.worst_db == pytest.approx(2.0, abs=0.6)


def test_combining_without_degrees_of_freedom_is_not_judged():
    combined = single(rule='combined', combine=4)
    (row,) = rq.evaluate(combined, reference(), run(1.0)).rows
    assert row.state == 'not judged'
    assert 'degrees of freedom' in row.measured


def test_the_result_is_not_met_before_not_judged_and_met_only_whole():
    met = rq.Criterion('grms', 'a', '', '', rq.MET)
    open_ = rq.Criterion('dof', 'b', '', '', rq.NOT_JUDGED)
    missed = rq.Criterion('duration', 'c', '', '', rq.NOT_MET)
    requirement = single()

    def state(*found):
        return rq.Results(requirement, found, 0.0, ()).state

    assert state(met) == 'met'
    assert state(met, open_) == 'not judged'
    assert state(met, open_, missed) == 'not met'
    assert state() == 'not judged'


def test_many_excursions_lead_with_the_worst_and_count_the_rest():
    hot = [(0, 100 + 100 * k, 100 + 100 * k, 10 ** ((7 + k / 10) / 10))
           for k in range(4)]
    (row,) = rq.evaluate(single(), reference(), run(1.0, hot=hot)).rows
    assert row.measured.startswith('4 excursions, 4 lines outside; the '
                                   'worst: above by up to 1.3 dB from 397.5')
    assert row.measured.endswith('and 1 more')
    assert [e.low for e in row.excursions] == sorted(
        e.low for e in row.excursions), 'kept in frequency order'


# ---- the edges the mutation run found -------------------------------------


def test_a_band_may_touch_the_reference_on_one_side():
    """Never above the reference, or never below it: an edge at 0 dB is
    a band, and so is an open edge."""
    assert rq.Band(None, None, -3, 0).db_text == '−3/+0 dB'
    assert rq.Band(None, None, 0, 3).db_text == '+0/+3 dB'
    assert rq.Band(None, None, -math.inf, 3).db_text == 'down/+3 dB'


def test_a_duration_may_have_no_tolerance_above():
    requirement = single(duration=180, duration_tolerance=(5, 0))
    assert requirement.duration_tolerance == (5.0, 0.0)


def test_two_lines_combined_is_the_fewest_the_rule_takes():
    assert single(rule='combined', combine=2).combine == 2


def test_a_value_of_the_wrong_kind_in_a_list_field_is_a_type_error():
    with pytest.raises(TypeError, match='cross-axis limits are a list'):
        single(cross_axis='50 %')
    with pytest.raises(TypeError, match='list of response DOFs'):
        single(channels='1Z+')


def test_a_stored_requirement_names_its_document_and_band():
    stored = single().as_record()
    del stored['composite_bands']
    with pytest.raises(ValueError, match='names its document and its '
                                         'control band'):
        rq.TestRequirement.from_record(stored)


def test_the_words_name_the_weights_and_the_cross_axis_limits():
    lines = rq.TestRequirement(
        'TP', [(None, None, -3, 3)], strategy='weighted',
        channels=('1Z+', '2Z+'), weights=(3, 1),
        individual_bands=[(None, None, -6, 6)],
        cross_axis=[(None, 500, 50), (500, None, 100)],
        sources={'cross_axis': ('derived', 'Analysis')}).describe()
    assert ('Control strategy: weighted of the control channels 1Z+, 2Z+, '
            'weights 3, 1.') in lines
    assert ('Cross-axis: at most 50 % of in-axis up to 500 Hz; 100 % of '
            'in-axis from 500 Hz, derived (Analysis).') in lines


@pytest.mark.parametrize(('seconds', 'said'), [
    (45.0, '45 s'), (45.5, '45.5 s'), (125.5, '2 min 5.5 s'),
    (180.0, '3 min'), (3600.0, '1 h'), (3601.0, '1 h 1 s'),
    (3723.0, '1 h 2 min 3 s')])
def test_a_duration_in_words(seconds, said):
    assert rq.duration_text(seconds) == said


@pytest.mark.parametrize(('worst', 'said'), [
    (1.4, 'by up to 1.4 dB'), (0.25, 'by up to 0.25 dB'),
    (12.3, 'by up to 12 dB'), (123.4, 'by up to 123 dB'),
    (math.inf, 'reading zero')])
def test_an_excursion_in_words(worst, said):
    one = rq.Excursion('above', 600.0, 605.0, worst, 1)
    assert one.text == f'above {said} from 600 to 605 Hz (1 line)'


def test_a_band_of_one_line_says_one_line():
    (row,) = rq.evaluate(single(composite_bands=[(998, 1002, -6, 6)]),
                         reference(), run(1.0)).rows
    assert row.measured == 'inside the band at all 1 line'


def test_three_excursions_are_all_named_with_no_count():
    hot = [(0, 100 + 100 * k, 100 + 100 * k, HOT) for k in range(3)]
    (row,) = rq.evaluate(single(), reference(), run(1.0, hot=hot)).rows
    assert row.measured.count('above by up to 2 dB') == 3
    assert 'more' not in row.measured and 'excursions' not in row.measured


def test_one_zero_line_is_said_in_the_singular():
    results = rq.evaluate(single(), reference(),
                          run(1.0, hot=[(0, 1000, 1000, 0.0)]))
    assert results.notes == (('1Z+ reads zero at 1 line, as a dead or '
                              'disconnected sensor does'),)
    assert 'not judged' not in results.rows[0].measured, (
        'a zero is a reading')


def test_a_negative_density_is_counted_not_judged_never_sized():
    """A negative line is no reading: counted among the lines not judged,
    and never given a size (the log of a negative power). The lines
    either side, read across it, may still fall out of the band."""
    results = rq.evaluate(single(), reference(),
                          run(1.0, hot=[(0, 1000, 1000, -1.0)]))
    (row,) = results.rows
    assert '1 line not judged' in row.measured
    assert not any(math.isnan(e.worst_db) for e in row.excursions)
    assert all(e.high <= 997.5 or e.low >= 1002.5 for e in row.excursions)


def test_a_line_not_read_on_the_edge_of_a_band_is_counted():
    (low, high) = rq.evaluate(single(composite_bands=STEPPED), reference(),
                              run(1.0, nan_at=(0, 4))).rows
    assert low.state == 'not judged' and '1 line not judged' in low.measured
    assert high.state == 'met'
    (low, high) = rq.evaluate(single(composite_bands=STEPPED), reference(),
                              run(1.0, nan_at=(0, 100))).rows
    assert low.state == high.state == 'not judged', \
        'the line on 500 Hz is in both ranges'


def test_a_scale_set_on_the_run_is_not_said_as_detected():
    measured = run(10 ** -0.6)
    measured.scale_db = 6
    results = rq.evaluate(single(), reference(), measured)
    assert results.scale_db == 6.0
    assert results.notes == ()
    assert results.rows[0].state == 'met'


def test_a_cross_term_is_no_control_channel():
    measured = Psd(LINES, np.full((2, LINES.size), LEVEL),
                   response_dof=['1Z+', '2Z+'], reference_dof=['1Z+', '1Z+'],
                   ordinate_dim=[PSD_DIM] * 2)
    measured.scale_db = 0
    with pytest.raises(ValueError, match='no autospectrum at 2Z'):
        rq.evaluate(single(channels=('2Z+',)), reference(), measured)
    assert rq.evaluate(single(), reference(), measured).channels == ('1Z+',)


def test_a_drive_points_force_against_its_acceleration_is_no_control():
    """One DOF on both sides is not enough: the acceleration against the
    force at the drive point is a cross term (`core.data.autospectrum`),
    so the acceleration's own autospectrum is the one control channel."""
    measured = Psd(LINES, np.full((2, LINES.size), LEVEL),
                   response_dof=['1Z+', '1Z+'], reference_dof=['1Z+', '1Z+'],
                   ordinate_dim=[PSD_DIM, 'acceleration*force/frequency'])
    measured.scale_db = 0
    results = rq.evaluate(single(), reference(), measured)
    assert results.channels == ('1Z+',)
    assert results.rows[0].state == 'met'


class _Table:
    """The two answers a channel table gives the reading."""

    def __init__(self, dofs, controls):
        self._dofs, self._controls = dofs, controls

    def dof_strings(self):
        return list(self._dofs)

    def controls_for(self, dofs):
        return list(self._controls)


def test_the_one_channel_the_channel_table_flags_is_the_control():
    specification = reference(('1Z+', '2Z+'))
    measured = run(1.0, HOT)
    results = rq.evaluate(single(), specification, measured,
                          table=_Table(['1Z+', '2Z+'], [False, True]))
    assert results.channels == ('2Z+',)
    with pytest.raises(ValueError, match='none of the measurement.s '
                                         'autospectra the channel table'):
        rq.evaluate(single(), specification, measured,
                    table=_Table(['9Z+'], [True]))


def test_a_combined_run_takes_the_specifications_own_loops_control():
    """A random loop controlling 1Z+ and a sine loop 2Z+ on one table:
    the random reference's single control channel is 1Z+, where the
    Control column's union of both loops offered two candidates."""
    from visualdynamics.core.channel_table import ChannelTable, Environment

    table = ChannelTable({'channel': [1, 2], 'node': ['1', '2'],
                          'direction': ['Z+'] * 2, 'unit': ['g'] * 2})
    table.set_environment('Random', Environment(controls=[1]))
    table.set_environment('Sine', Environment(controls=[2]))
    results = rq.evaluate(single(), reference(('1Z+',)), run(1.0, HOT),
                          table=table)
    assert results.channels == ('1Z+',)


def test_degrees_of_freedom_of_zero_or_nan_are_not_carried():
    measured = run(1.0)
    for counts in ([0.0], [np.nan]):
        measured.statistical_dof = np.array(counts)
        (row,) = rows(rq.evaluate(single(min_dof=10), reference(), measured),
                      'dof')
        assert row.state == 'not judged'
    measured.statistical_dof = np.array([120.0])
    (row,) = rows(rq.evaluate(single(min_dof=10), reference(), measured),
                  'dof')
    assert row.measured == '120', 'one channel: not the fewest of several'


def test_two_control_channels_are_the_fewest_of_several():
    measured = run(1.0, 1.0)
    measured.statistical_dof = np.array([120.0, 90.0])
    (row,) = rows(rq.evaluate(average(channels=('1Z+', '2Z+'), min_dof=10),
                              reference(), measured), 'dof')
    assert row.measured == '90, the fewest of the control channels'


def test_the_band_s_chance_share_is_not_a_note_of_the_result():
    """The multi-point reading's chance share is of one band; a
    requirement of several bands does not repeat it as its own."""
    measured = run(1.0, 1.0, 1.0, 1.0)
    measured.statistical_dof = np.full(4, 10.0)
    results = rq.evaluate(average(), reference(), measured)
    assert not [note for note in results.notes if 'by chance' in note]


def test_combined_lines_under_a_band_are_given_their_size():
    combined = single(composite_bands=[(None, None, -1, 1)],
                      rule='combined', combine=8)
    low = _scatter(120)
    keep = (low.abscissa >= 5000) & (low.abscissa < 5200)
    low.ordinate[0, keep] *= 0.5
    (row,) = rq.evaluate(combined, reference(upper=20000.0), low).rows
    (miss,) = row.excursions
    assert miss.side == 'below'
    assert miss.worst_db == pytest.approx(2.0, abs=0.6)


def test_combined_lines_with_nothing_to_read_are_not_judged():
    combined = single(rule='combined', combine=4)
    measured = run(1.0)
    measured.ordinate[:] = np.nan
    measured.statistical_dof = np.array([120.0])
    (row,) = rq.evaluate(combined, reference(), measured).rows
    assert row.state == 'not judged'


def test_an_rms_of_nothing_is_not_judged():
    measured = run(1.0)
    measured.ordinate[:] = np.nan
    (row,) = rows(rq.evaluate(single(grms_percent=10), reference(),
                              measured), 'grms')
    assert row.state == 'not judged'
    assert 'no finite rms' in row.measured


def test_the_duration_against_a_tolerance_both_ways():
    """Minus 5, plus 10 percent of 3 minutes: 171 to 198 seconds, the
    difference from nominal said in percent."""
    requirement = single(duration=180, duration_tolerance=(5, 10))
    said = {}
    for seconds in (170.9, 171.0, 198.0, 198.1):
        (row,) = rows(rq.evaluate(requirement, reference(), run(1.0),
                                  duration_run=seconds), 'duration')
        said[seconds] = (row.state, row.measured)
    assert said[170.9] == ('not met', '2 min 50.9 s (−5.1 %)'.replace(
        '−', '-'))
    assert said[171.0][0] == 'met' and said[198.0][0] == 'met'
    assert said[198.1] == ('not met', '3 min 18.1 s (+10.1 %)')
    with pytest.raises(ValueError, match='seconds above 0'):
        rq.evaluate(requirement, reference(), run(1.0),
                    duration_run=math.inf)


def test_a_banded_run_or_a_run_of_one_line_has_no_line_spacing():
    banded = run(1.0, lines=np.array([25.0, 31.5, 40.0]))
    banded.bandwidth = np.array([5.8, 7.3, 9.2])
    one = run(1.0, lines=np.array([1000.0]))
    for measured in (banded, one):
        row = rq._spacing_row(single(max_line_spacing=5.0), measured)
        assert row.state == 'not judged'


def test_lines_evenly_spaced_but_for_rounding_have_their_spacing():
    # every other line a micro-hertz high: well inside the tolerance on
    # the spacing, as a frequency written to a few figures leaves it
    lines = LINES + 1e-6 * (np.arange(LINES.size) % 2)
    (row,) = rows(rq.evaluate(single(max_line_spacing=10.0), reference(),
                              run(1.0, lines=lines)), 'line_spacing')
    assert (row.measured, row.state) == ('5 Hz', rq.MET)


def test_degrees_of_freedom_on_the_fewest_allowed_are_met():
    measured = run(1.0)
    measured.statistical_dof = np.array([120.0])
    (row,) = rows(rq.evaluate(single(min_dof=120), reference(), measured),
                  'dof')
    assert (row.measured, row.state) == ('120', rq.MET)


def test_a_band_cut_from_breakpoints_gains_its_edges_once():
    stepped = Specification([20.0, 500.0, 2000.0], np.full((1, 3), LEVEL),
                            response_dof=['1Z+'], ordinate_dim=[PSD_DIM])
    assert list(rq.within(stepped, 0, 500.0, None).abscissa) == [500.0,
                                                                2000.0]
    assert list(rq.within(stepped, 0, 100.0, 500.0).abscissa) == [100.0,
                                                                 500.0]
    with pytest.raises(ValueError, match='holds nothing'):
        rq.within(stepped, 0, 2000.0, 3000.0)


def test_a_band_cut_from_lines_keeps_the_lines_inside_with_their_widths():
    lines = Specification(LINES, np.full((1, LINES.size), LEVEL),
                          response_dof=['1Z+'], ordinate_dim=[PSD_DIM])
    lines.interpolation = 'bin'
    cut = rq.within(lines, 0, 100.0, 200.0)
    assert cut.abscissa[0] == 100.0 and cut.abscissa[-1] == 200.0
    assert cut.abscissa.size == 21 and cut.interpolation == 'bin'
    with pytest.raises(ValueError, match='holds no line'):
        rq.within(lines, 0, 101.0, 104.0)
    centers = np.array([31.5, 40.0, 50.0, 63.0, 80.0])
    bands = Specification(centers, np.full((1, 5), LEVEL),
                          response_dof=['1Z+'], ordinate_dim=[PSD_DIM],
                          bandwidth=centers * 0.23)
    bands.interpolation = 'bin'
    cut = rq.within(bands, 0, 40.0, 63.0)
    assert list(cut.bandwidth) == pytest.approx(list(centers[1:4] * 0.23))


# ---- the measurement's uncertainty against a third of the tolerance --------

#: a budget of the size certificates and data sheets give, each at k = 2:
#: calibration, conditioner and converter, transverse sensitivity times a
#: cross-axis share of a half, and the mount and chain's response
BUDGET = [('calibration', 2.0, 2, 'certificate C-1'),
          ('conditioner and converter', 0.5), ('transverse', 1.5),
          ('mount and chain', 1.0)]


def test_a_budget_is_the_root_sum_square_in_percent_asd_and_db():
    held = single(channels=('1Z+',), composite_bands=STEPPED,
                  grms_percent=10, uncertainty={'1Z+': BUDGET})
    budget, = rq.uncertainty_budget(held)
    assert budget.acceleration_percent == pytest.approx(math.sqrt(7.5))
    assert budget.db == pytest.approx(-20 * math.log10(1 - math.sqrt(7.5)
                                                       / 100))
    assert budget.db == pytest.approx(0.2412, abs=1e-4)
    assert budget.asd_percent == pytest.approx(
        100 * (10 ** (budget.db / 10) - 1))
    assert (budget.band_db, budget.band_limit_db) == pytest.approx((3, 1))
    assert budget.grms_limit_percent == pytest.approx(10 / 3)
    assert (budget.state, budget.words) == (rq.MET, '')


def test_each_contribution_is_taken_at_its_own_coverage_factor():
    held = single(channels=('1Z+',),
                  uncertainty={'1Z+': [('calibration', 1.0, 1),
                                       ('chain', 2.0, 2)]})
    budget, = rq.uncertainty_budget(held)
    assert budget.acceleration_percent == pytest.approx(2 * math.sqrt(2))
    assert rq.combined_percent([rq.Contribution('a', 3.0, 3)]) == \
        pytest.approx(2.0)


def test_a_budget_over_a_third_of_the_band_or_the_rms_is_not_met():
    tight = single(channels=('1Z+',), composite_bands=[(None, None, -0.5, 6)],
                   uncertainty={'1Z+': BUDGET})
    budget, = rq.uncertainty_budget(tight)
    assert budget.state == rq.NOT_MET
    assert budget.band_db == 0.5
    assert budget.words == ('0.24 dB over one third of the 0.5 dB band edge, '
                            '0.17 dB')
    rms = single(channels=('1Z+',), grms_percent=6,
                 uncertainty={'1Z+': BUDGET})
    budget, = rq.uncertainty_budget(rms)
    assert budget.state == rq.NOT_MET
    assert budget.words == ('2.7 % over one third of the 6 % rms tolerance, '
                            '2 %')


def test_a_budget_on_its_third_exactly_is_met():
    # one contribution worth exactly 1 dB, a third of plus or minus 3 dB,
    # and exactly a third of the rms tolerance
    percent = 100 * (1 - 10 ** (-1 / 20))
    held = single(channels=('1Z+',), composite_bands=[(None, None, -3, 3)],
                  grms_percent=3 * percent,
                  uncertainty={'1Z+': [('chain', percent)]})
    budget, = rq.uncertainty_budget(held)
    assert budget.db == pytest.approx(1.0)
    assert budget.state == rq.MET


def test_the_tightest_edge_of_every_band_sets_the_third():
    held = average(composite_bands=[(None, 500, -1.5, 3), (500, None, -6, 6)],
                   individual_bands=[(None, None, 'down', 9)],
                   uncertainty={'1Z+': BUDGET})
    budget = rq.uncertainty_budget(held)[0]
    assert budget.band_db == 1.5 and budget.band_limit_db == 0.5
    touching = single(channels=('1Z+',), composite_bands=[(None, None, 0, 3)],
                      uncertainty={'1Z+': BUDGET})
    budget, = rq.uncertainty_budget(touching)
    assert (budget.band_limit_db, budget.state) == (0.0, rq.NOT_MET), \
        'a band edge on the reference leaves the measurement nothing'


def test_a_channel_with_nothing_entered_is_not_judged_never_met():
    held = average(uncertainty={'1Z+': BUDGET})
    budgets = rq.uncertainty_budget(held)
    assert [b.channel for b in budgets] == ['1Z+', '2Z+', '3Z+', '4Z+']
    assert [b.state for b in budgets] == [rq.MET] + [rq.NOT_JUDGED] * 3
    empty = budgets[1]
    assert empty.words == 'not judged: no contributions are entered'
    assert math.isnan(empty.acceleration_percent) and math.isnan(empty.db)
    assert math.isnan(empty.asd_percent)
    assert rq.uncertainty_budget(single()) == ()
    assert [b.state for b in rq.uncertainty_budget(single(), ['1Z+'])] == [
        rq.NOT_JUDGED]
    nothing = single(uncertainty={'2Z+': []})
    assert [b.state for b in rq.uncertainty_budget(nothing)] == [
        rq.NOT_JUDGED], 'an empty list is nothing entered'


def test_the_channels_are_those_named_else_those_with_contributions():
    held = single(uncertainty={'2Z+': BUDGET})
    assert [b.channel for b in rq.uncertainty_budget(held)] == ['2Z+']
    assert [b.channel for b in rq.uncertainty_budget(held, ['1Z+'])] == [
        '1Z+']


def test_an_uncertainty_of_the_whole_signal_is_infinite_in_db():
    assert rq.uncertainty_db(100.0) == math.inf
    assert rq.uncertainty_db(0.0) == 0.0
    assert math.isnan(rq.combined_percent([]))
    held = single(channels=('1Z+',),
                  uncertainty={'1Z+': [('a', 90.0, 1), ('b', 90.0, 1)]})
    budget, = rq.uncertainty_budget(held)
    assert budget.db == math.inf and budget.state == rq.NOT_MET
    assert budget.words.startswith('infinite dB over one third')


@pytest.mark.parametrize(('entered', 'error', 'said'), [
    ({'1Z+': [('calibration', 0.0)]}, ValueError, 'above 0 and under 100'),
    ({'1Z+': [('calibration', float('nan'))]}, ValueError, 'not NaN'),
    ({'1Z+': [('calibration', 100.0)]}, ValueError, 'above 0 and under 100'),
    ({'1Z+': [('calibration', -1.0)]}, ValueError, 'above 0 and under 100'),
    ({'1Z+': [('calibration', 2.0, 0)]}, ValueError, 'coverage factor'),
    ({'1Z+': [('calibration', 2.0, math.inf)]}, ValueError,
     'coverage factor'),
    ({'1Z+': [(' ', 2.0)]}, ValueError, 'is named'),
    ({'1Z+': [('calibration', 2.0), ('calibration', 1.0)]}, ValueError,
     "'calibration' twice"),
    ({'5Z+': [('calibration', 2.0)]}, ValueError,
     'not a control channel of the requirement'),
    ({'1Z+': [('calibration',)]}, ValueError, 'each uncertainty contribution'),
    ({'1Z+': ['calibration']}, ValueError, 'each uncertainty contribution'),
    ({'1Z+': [('calibration', 2.0, 2, 'C-1', 'extra')]}, ValueError,
     'each uncertainty contribution'),
    ({'1Z+': 'calibration 2 %'}, TypeError, 'list of contributions'),
    ([('calibration', 2.0)], TypeError, 'mapping of control channel'),
    ({'1Z+': [('calibration', 2.0, 2, 7)]}, TypeError, 'source is text'),
    ({'1Z+': [('calibration', 'two')]}, ValueError, 'a number'),
])
def test_an_uncertainty_contribution_the_plan_cannot_mean_is_refused(
        entered, error, said):
    with pytest.raises(error, match=said):
        single(channels=('1Z+',), uncertainty=entered)


def test_an_uncertainty_is_stored_said_and_read_back():
    held = single(channels=('1Z+',), uncertainty={'1Z+': BUDGET},
                  sources={'uncertainty': ('laboratory', 'Metrology')})
    record = json.loads(json.dumps(held.as_record()))
    assert record['uncertainty']['1Z+'][0] == ['calibration', 2.0, 2.0,
                                               'certificate C-1']
    assert rq.TestRequirement.from_record(record) == held
    assert held.describe()[-1] == (
        'Measurement uncertainty: entered for 1Z+, held against one third '
        'of the tolerance, from the laboratory (Metrology).')
    assert 'uncertainty' not in ' '.join(single().describe()).lower()
    assert held.uncertainty['1Z+'][0].text == (
        'calibration 2 % (k = 2, certificate C-1)')
    assert held.uncertainty['1Z+'][1].text == (
        'conditioner and converter 0.5 % (k = 2)')
    assert held.uncertainty['1Z+'][0].standard == 1.0


# ---- cross-axis -----------------------------------------------------------------


#: a cross-axis channel at a fifth of its in-axis channel's ASD
FIFTH = 0.2


def crossed(**kwargs):
    """A single-channel requirement on 1Z+ with 2Z+ its cross-axis."""
    kwargs.setdefault('cross_axis', [(None, 500, 50), (500, None, 100)])
    kwargs.setdefault('cross_axis_channels', {'2Z+': '1Z+'})
    return single(channels=('1Z+',), **kwargs)


def test_cross_axis_is_a_percentage_of_the_in_axis_asd_per_band():
    measured = run(1.0, FIFTH, hot=[(1, 440, 460, 3.0)])
    results = rq.evaluate(crossed(), reference(('1Z+', '2Z+')), measured)
    low, high = rows(results, 'cross_axis')
    assert low.name == 'Cross-axis 2Z+ (in-axis 1Z+)'
    assert (low.required, low.state) == ('50 % of in-axis up to 500 Hz',
                                         rq.NOT_MET)
    assert low.measured == 'worst 60 % at 440 Hz, 5 of 97 lines over'
    assert low.band == (20.0, 500.0), 'open below: from where the spec starts'
    assert low.channel == '2Z+'
    assert low.finding == rq.Offer(
        '2Z+ read up to 60 % of 1Z+ at 440 Hz, over its 50 % cross-axis '
        'limit on 5 of 97 lines from 440 to 460 Hz', '2Z+', (440.0, 460.0)), \
        'the lines over, offered to a findings register'
    assert (high.state, high.measured) == (rq.MET, 'worst 20 % at 500 Hz')
    assert high.finding is None, 'a row met offers no finding'
    assert high.band == (500.0, 2000.0)
    assert results.state == rq.NOT_MET


def test_cross_axis_on_its_limit_exactly_is_met():
    results = rq.evaluate(crossed(cross_axis=[(None, None, 20)]),
                          reference(('1Z+', '2Z+')), run(1.0, FIFTH))
    (row,) = rows(results, 'cross_axis')
    assert row.state == rq.MET


def test_lines_on_the_cross_axis_limit_are_not_in_the_reach_offered():
    measured = run(1.0, FIFTH, hot=[(1, 440, 460, 3.0)])
    (row,) = rows(rq.evaluate(crossed(cross_axis=[(None, None, 20)]),
                              reference(('1Z+', '2Z+')), measured),
                  'cross_axis')
    assert row.state == rq.NOT_MET
    assert row.finding.band == (440.0, 460.0), 'a line on the limit is met'


def test_one_line_over_is_counted_and_offered_in_the_singular():
    measured = run(1.0, FIFTH, hot=[(1, 440, 440, 3.0)])
    (row,) = rows(rq.evaluate(crossed(cross_axis=[(438, 442, 50)]),
                              reference(('1Z+', '2Z+')), measured),
                  'cross_axis')
    assert row.measured == 'worst 60 % at 440 Hz, 1 of 1 line over'
    assert row.finding.text == (
        '2Z+ read up to 60 % of 1Z+ at 440 Hz, over its 50 % cross-axis '
        'limit on 1 of 1 line from 440 to 440 Hz')


def test_cross_axis_names_the_degrees_of_freedom_and_refuses_unlike_ones():
    measured = run(1.0, FIFTH)
    measured.statistical_dof = np.array([120.0, 120.0])
    (row, _high) = rows(rq.evaluate(crossed(), reference(('1Z+', '2Z+')),
                                    measured), 'cross_axis')
    assert row.measured == 'worst 20 % at 20 Hz, at 120 degrees of freedom'
    measured.statistical_dof = np.array([120.0, 120.0 * (1.0 + 1e-11)])
    (row, _high) = rows(rq.evaluate(crossed(), reference(('1Z+', '2Z+')),
                                    measured), 'cross_axis')
    assert row.state == rq.MET, 'the same but for rounding'
    measured.statistical_dof = np.array([120.0, 60.0])
    (row, _high) = rows(rq.evaluate(crossed(), reference(('1Z+', '2Z+')),
                                    measured), 'cross_axis')
    assert row.state == rq.NOT_JUDGED
    assert row.measured == ('not judged: the two are read at different '
                            'degrees of freedom (60 and 120), so their ratio '
                            'is not of like estimates')


def test_a_dead_cross_axis_channel_is_not_judged_never_met():
    (row, _high) = rows(rq.evaluate(crossed(), reference(('1Z+', '2Z+')),
                                    run(1.0, 0.0)), 'cross_axis')
    assert row.state == rq.NOT_JUDGED
    assert row.measured == ('not judged: 2Z+ reads zero over the band, as a '
                            'dead or disconnected sensor does')


def test_a_cross_axis_line_not_read_leaves_its_row_not_judged_unless_over():
    measured = run(1.0, FIFTH, nan_at=(1, 30))
    (row, _high) = rows(rq.evaluate(crossed(), reference(('1Z+', '2Z+')),
                                    measured), 'cross_axis')
    assert row.state == rq.NOT_JUDGED
    assert row.measured == 'worst 20 % at 20 Hz; 1 line not judged'
    measured = run(1.0, FIFTH, nan_at=(0, 30), hot=[(1, 440, 460, 3.0)])
    (row, _high) = rows(rq.evaluate(crossed(), reference(('1Z+', '2Z+')),
                                    measured), 'cross_axis')
    assert row.state == rq.NOT_MET
    assert row.measured.endswith('5 of 96 lines over; 1 line not judged')
    zero = run(0.0, FIFTH)
    (row, _high) = rows(rq.evaluate(
        crossed(), reference(('1Z+', '2Z+')), zero, scale_db=0.0),
        'cross_axis')
    assert row.measured == ('not judged: no line of the band has both '
                            'densities finite and the in-axis one above '
                            'zero')


def test_a_cross_axis_channel_not_measured_or_of_another_quantity_says_why():
    (row, _high) = rows(rq.evaluate(
        crossed(cross_axis_channels={'7Z+': '1Z+'}),
        reference(('1Z+', '2Z+')), run(1.0, FIFTH)), 'cross_axis')
    assert row.measured == ('not judged: the measurement has no '
                            'autospectrum at 7Z+')
    other = run(1.0, FIFTH, dims=[PSD_DIM, 'velocity**2/frequency'])
    (row, _high) = rows(rq.evaluate(crossed(), reference(('1Z+', '2Z+')),
                                    other), 'cross_axis')
    assert row.state == rq.NOT_JUDGED
    assert 'velocity' in row.measured


def test_a_cross_axis_band_the_run_does_not_reach_is_not_judged():
    (row,) = rows(rq.evaluate(crossed(cross_axis=[(3000, 4000, 50)]),
                              reference(('1Z+', '2Z+')), run(1.0, FIFTH)),
                  'cross_axis')
    assert row.measured == ('not judged: the measurement has no line from '
                            '3000 to 4000 Hz')


def test_cross_axis_limits_with_no_channel_named_are_not_judged():
    (row,) = rows(rq.evaluate(single(cross_axis=[(None, 500, 50)]),
                              reference(), run(1.0)), 'cross_axis')
    assert row.measured == 'not judged: no cross-axis channel is named'
    assert row.state == rq.NOT_JUDGED


@pytest.mark.parametrize(('kwargs', 'error', 'said'), [
    ({'cross_axis': ()}, ValueError, 'state them'),
    ({'cross_axis_channels': {'1Z+': '1Z+'}}, ValueError,
     'its own in-axis channel'),
    ({'cross_axis_channels': {'2Z+': '3Z+'}}, ValueError,
     'not a control channel of the requirement'),
    ({'cross_axis_channels': {'1Z+': '9Z+'}}, ValueError,
     'not a control channel of the requirement'),
    ({'cross_axis_channels': {'2Z+': ' '}}, ValueError, 'response DOF'),
    ({'cross_axis_channels': [('2Z+', '1Z+')]}, TypeError, 'a mapping'),
])
def test_cross_axis_channels_the_plan_cannot_mean_are_refused(kwargs, error,
                                                              said):
    with pytest.raises(error, match=said):
        crossed(**kwargs)


def test_a_control_channel_is_not_a_cross_axis_one():
    with pytest.raises(ValueError, match='is a control channel, not a '
                                         'cross-axis one'):
        average(cross_axis=[(None, None, 50)],
                cross_axis_channels={'2Z+': '1Z+'})


# ---- limit channels, the as-run notch -------------------------------------------


def notched(*, limit=1.0, elsewhere=0.25, control=0.5, band=(400, 480)):
    """1Z+ the control, under its reference by `control` over `band`;
    2Z+ the limit channel, `limit` of its profile (the specification's
    record there) over `band` and `elsewhere` of it everywhere else."""
    low, high = band
    return run(1.0, elsewhere, hot=[(0, low, high, control),
                                    (1, low, high, limit / elsewhere)])


def limited(*channels, **kwargs):
    return single(channels=('1Z+',), limit_channels=list(channels),
                  **kwargs)


def test_a_limit_channel_on_its_profile_while_the_control_dips_is_a_notch():
    results = rq.evaluate(limited('2Z+'), reference(('1Z+', '2Z+')),
                          notched())
    (row,) = rows(results, 'limit')
    assert row.name == 'Limit channel 2Z+'
    assert row.required == ('held at or under its limit profile in the '
                            'specification; no notch planned')
    (notch,) = row.excursions
    assert (notch.side, notch.low, notch.high, notch.lines) == (
        'below', 400.0, 480.0, 17)
    assert notch.worst_db == pytest.approx(10 * math.log10(2))
    assert row.measured == ('held its limit from 400 to 480 Hz (17 lines) '
                            'while the control fell up to 3 dB under the '
                            'reference, a notch the plan does not name; +0 '
                            'dB from its limit profile at its closest, '
                            '400 Hz')
    assert row.state == rq.NOT_MET
    assert row.band == (400.0, 480.0)
    assert row.finding == rq.Offer(
        '2Z+ held its limit profile in a notch the plan does not name: from '
        '400 to 480 Hz (17 lines), the control up to 3 dB under the '
        'reference', '2Z+', (400.0, 480.0)), 'offered to a findings register'
    assert results.state == rq.NOT_MET


def test_a_notch_the_plan_names_is_met_and_one_beyond_it_is_not():
    spec = reference(('1Z+', '2Z+'))
    (row,) = rows(rq.evaluate(limited(('2Z+', '', [(380, 500)])), spec,
                              notched()), 'limit')
    assert row.state == rq.MET
    assert row.required.endswith('notches planned 380 to 500 Hz')
    assert 'does not name' not in row.measured
    assert row.finding is None, 'a planned notch offers no finding'
    (row,) = rows(rq.evaluate(limited(('2Z+', '', [(420, 460)])), spec,
                              notched()), 'limit')
    assert row.state == rq.NOT_MET
    assert row.finding.band == (400.0, 480.0)
    for planned in ((380, 460), (420, 500)):
        (row,) = rows(rq.evaluate(limited(('2Z+', '', [planned])), spec,
                                  notched()), 'limit')
        assert row.state == rq.NOT_MET, f'running out of {planned}'
    (row,) = rows(rq.evaluate(limited(('2Z+', '', [(400, 480)])), spec,
                              notched()), 'limit')
    assert row.state == rq.MET, 'a notch on the planned edges is inside'


def test_a_limit_channel_under_its_profile_cuts_no_notch():
    (row,) = rows(rq.evaluate(limited('2Z+'), reference(('1Z+', '2Z+')),
                              notched(limit=0.25)), 'limit')
    assert row.state == rq.MET
    assert row.excursions == ()
    assert row.measured == ('no notch; -6 dB from its limit profile at its '
                            'closest, 20 Hz')


def test_the_control_on_its_reference_cuts_no_notch_whatever_the_limit():
    (row,) = rows(rq.evaluate(limited('2Z+'), reference(('1Z+', '2Z+')),
                              notched(control=1.0, limit=2.0)), 'limit')
    assert row.excursions == ()
    assert row.measured.startswith('no notch; +3 dB from its limit profile')


def test_a_dip_of_two_lines_is_not_a_notch_nor_a_dip_within_a_decibel():
    spec = reference(('1Z+', '2Z+'))
    (row,) = rows(rq.evaluate(limited('2Z+'), spec,
                              notched(band=(400, 405))), 'limit')
    assert row.excursions == (), 'narrower than a controller cuts'
    (row,) = rows(rq.evaluate(limited('2Z+'), spec,
                              notched(band=(400, 410))), 'limit')
    assert row.excursions[0].lines == 3, 'three lines is the narrowest'
    (row,) = rows(rq.evaluate(limited('2Z+'), spec,
                              notched(control=0.85)), 'limit')
    assert row.excursions == (), 'under a decibel is the scatter'
    (row,) = rows(rq.evaluate(limited('2Z+'), spec,
                              notched(limit=0.85)), 'limit')
    assert len(row.excursions) == 1, 'within a decibel of its profile holds'
    (row,) = rows(rq.evaluate(limited('2Z+'), spec,
                              notched(limit=0.75)), 'limit')
    assert row.excursions == (), 'more than a decibel under it does not'


def _scattered_pair(seed, limit_db):
    """The control on its reference and the limit channel `limit_db`
    from its profile, each line a chi-square at 120 DOF about its true
    level."""
    control, limit = _scatter(120, seed), _scatter(120, seed + 100)
    both = Psd(control.abscissa,
               np.vstack([control.ordinate,
                          limit.ordinate * 10.0 ** (limit_db / 10.0)]),
               response_dof=['1Z+', '2Z+'], ordinate_dim=[PSD_DIM] * 2)
    both.scale_db = 0
    return both


def test_scatter_alone_cuts_no_notch():
    """A correct run with no notch: the control on its reference and the
    limit channel 3 dB under its profile, at 120 DOF. The control dips
    a decibel at about one line in 20 by chance, but the limit channel
    is almost never within a decibel of its profile at the same three
    adjacent lines, so no notch is read in 20 runs of 4000 lines."""
    spec = reference(('1Z+', '2Z+'), upper=20000.0)
    found = 0
    for seed in range(20):
        (row,) = rows(rq.evaluate(limited('2Z+'), spec,
                                  _scattered_pair(seed, -3.0)), 'limit')
        found += len(row.excursions)
    assert found == 0


def test_a_limit_channel_on_its_profile_throughout_reads_a_rare_notch():
    """The edge the guide states: a limit channel sitting on its profile
    across the band while the control holds its reference, at 120 DOF,
    reads a three-line notch by chance about once in 10,000 lines."""
    spec = reference(('1Z+', '2Z+'), upper=20000.0)
    found = lines = 0
    for seed in range(20):
        both = _scattered_pair(seed, 0.0)
        (row,) = rows(rq.evaluate(limited('2Z+'), spec, both), 'limit')
        found += len(row.excursions)
        lines += int(((both.abscissa >= 20) & (both.abscissa <= 20000)).sum())
    assert 0 < found / lines < 3e-4


def test_the_control_under_a_multi_point_strategy_is_the_composite():
    measured = run(1.0, 1.0, 0.25, hot=[(0, 400, 480, 0.5), (1, 400, 480, 0.5),
                                        (2, 400, 480, 4.0)])
    requirement = rq.TestRequirement(
        'TP', [(None, None, -3, 3)], strategy='average',
        channels=('1Z+', '2Z+'), individual_bands=[(None, None, -6, 6)],
        limit_channels=['3Z+'])
    (row,) = rows(rq.evaluate(requirement, reference(('1Z+', '2Z+', '3Z+')),
                              measured), 'limit')
    assert row.excursions[0].low == 400.0
    assert row.excursions[0].worst_db == pytest.approx(10 * math.log10(2))


def test_the_comparison_s_scale_applies_to_the_limit_channel_and_control():
    quarter = notched()
    quarter.ordinate[:] = quarter.ordinate / 4.0
    quarter.scale_db = None
    (row,) = rows(rq.evaluate(limited('2Z+'), reference(('1Z+', '2Z+')),
                              quarter, scale_db=10 * math.log10(4)), 'limit')
    assert (row.excursions[0].low, row.excursions[0].lines) == (400.0, 17)


def test_a_limit_profile_from_another_object_is_read_by_its_name():
    spec = reference(('1Z+',))
    profile = Psd(LINES, np.full((1, LINES.size), LEVEL),
                  response_dof=['2Z+'], ordinate_dim=[PSD_DIM])
    (row,) = rows(rq.evaluate(limited(('2Z+', 'Limits')), spec, notched(),
                              profiles={'Limits': profile}), 'limit')
    assert row.required.startswith('held at or under its limit profile in '
                                   'Limits')
    assert len(row.excursions) == 1
    (row,) = rows(rq.evaluate(limited(('2Z+', 'Limits')), spec, notched()),
                  'limit')
    assert row.measured == ("not judged: its limit profile 'Limits' is not "
                            'there to read')
    (row,) = rows(rq.evaluate(limited('2Z+'), spec, notched()), 'limit')
    assert row.measured == 'not judged: no limit profile is written at 2Z+'
    velocity = Psd(LINES, np.full((1, LINES.size), LEVEL),
                   response_dof=['2Z+'],
                   ordinate_dim=['velocity**2/frequency'])
    (row,) = rows(rq.evaluate(limited(('2Z+', 'Limits')), spec, notched(),
                              profiles={'Limits': velocity}), 'limit')
    assert row.state == rq.NOT_JUDGED and 'velocity' in row.measured


def test_a_limit_channel_dead_missing_or_unread_is_not_judged():
    spec = reference(('1Z+', '2Z+'))
    (row,) = rows(rq.evaluate(limited('2Z+'), spec, run(1.0, 0.0)), 'limit')
    assert row.measured == ('not judged: 2Z+ reads zero, as a dead or '
                            'disconnected sensor does')
    assert row.state == rq.NOT_JUDGED
    (row,) = rows(rq.evaluate(limited('5Z+'), spec, run(1.0, 1.0)), 'limit')
    assert row.measured == ('not judged: the measurement has no '
                            'autospectrum at 5Z+')
    unread = run(1.0, 1.0)
    unread.ordinate[1] = np.nan
    (row,) = rows(rq.evaluate(limited('2Z+'), spec, unread), 'limit')
    assert row.measured == ('not judged: no line has the limit channel, its '
                            'profile, the control and the reference all '
                            'read')


def test_one_line_not_read_splits_no_notch_it_is_not_in():
    measured = notched()
    measured.ordinate[1, 10] = np.nan
    (row,) = rows(rq.evaluate(limited('2Z+'), reference(('1Z+', '2Z+')),
                              measured), 'limit')
    assert (row.excursions[0].low, row.excursions[0].lines) == (400.0, 17)


@pytest.mark.parametrize(('channels', 'error', 'said'), [
    ([' '], ValueError, 'named by its response DOF'),
    (['1Z+'], ValueError, 'is a control channel'),
    (['2Z+', '2Z+'], ValueError, 'named twice'),
    ([('2Z+', '', [(500, 400)])], ValueError, 'low edge is below'),
    ([('2Z+', '', [(300, 500), (400, 600)])], ValueError, 'overlap'),
    ([('2Z+', '', [(None, 500)])], ValueError, 'both its edges'),
    ([('2Z+', '', [(400,)])], ValueError, r'is \(low, high\)'),
    ([('2Z+', '', 'notch')], TypeError, 'are a list'),
    ([('2Z+', 7)], TypeError, 'named by text'),
    ([('2Z+', '', (), 'extra')], ValueError, 'each limit channel is'),
    ('2Z+', TypeError, 'the limit channels are a list'),
])
def test_limit_channels_the_plan_cannot_mean_are_refused(channels, error,
                                                        said):
    with pytest.raises(error, match=said):
        single(channels=('1Z+',), limit_channels=channels)


def test_roles_are_stored_said_and_read_back():
    held = crossed(limit_channels=[('3Z+', 'Limits', [(400, 480)]), '4Z+'],
                   sources={'limit_channels': ('customer', 'Analyst')})
    record = json.loads(json.dumps(held.as_record()))
    assert record['cross_axis_channels'] == {'2Z+': '1Z+'}
    assert record['limit_channels'] == [['3Z+', 'Limits', [[400.0, 480.0]]],
                                        ['4Z+', '', []]]
    assert rq.TestRequirement.from_record(record) == held
    said = held.describe()
    assert 'Cross-axis channels: 2Z+ against 1Z+.' in said
    assert next(line for line in said if line.startswith('Limit')) == (
        'Limit channels: 3Z+, held at or under its limit profile in Limits; '
        'notches planned 400 to 480 Hz; 4Z+, held at or under its limit '
        'profile in the specification; no notch planned, supplied by the '
        'customer (Analyst); the laboratory has not verified its '
        'derivation.')


def test_the_cross_axis_rows_read_against_the_first_record_by_default():
    rows_read = rq.cross_axis_rows(crossed(), reference(('1Z+', '2Z+')),
                                   run(1.0, FIFTH))
    assert [row.state for row in rows_read] == [rq.MET, rq.MET]
    one = reference(('1Z+',))
    assert rq.cross_axis_rows(crossed(), one, run(1.0, FIFTH))[0].band == (
        20.0, 500.0)


def test_two_autospectra_at_a_role_s_dof_are_refused_by_name():
    measured = run(1.0, FIFTH, FIFTH,
                   dims=[PSD_DIM, PSD_DIM, 'velocity**2/frequency'])
    measured.response_dof = np.array(['1Z+', '2Z+', '2Z+'])
    (row, _high) = rows(rq.evaluate(crossed(), reference(('1Z+', '2Z+')),
                                    measured), 'cross_axis')
    assert row.measured == ('not judged: the measurement holds 2 autospectra '
                            'at 2Z+; which is meant cannot be told')
    (row,) = rows(rq.evaluate(limited('2Z+'), reference(('1Z+', '2Z+')),
                              measured), 'limit')
    assert row.state == rq.NOT_JUDGED and '2 autospectra' in row.measured


def test_a_notch_is_as_deep_as_its_deepest_line():
    measured = notched()
    x = measured.abscissa
    for at in (400.0, 480.0):
        deeper = measured.ordinate.copy()
        deeper[0, x == at] *= 0.5
        dipped = Psd(x, deeper, response_dof=['1Z+', '2Z+'],
                     ordinate_dim=[PSD_DIM] * 2)
        dipped.scale_db = 0
        (row,) = rows(rq.evaluate(limited('2Z+'), reference(('1Z+', '2Z+')),
                                  dipped), 'limit')
        assert row.excursions[0].worst_db == pytest.approx(
            10 * math.log10(4)), f'the deepest line at {at:g} Hz'
    # the control far under its reference just past the notch, where the
    # limit channel is not on its profile, is no part of it
    beyond = measured.ordinate.copy()
    beyond[0, x == 485.0] *= 0.1
    past = Psd(x, beyond, response_dof=['1Z+', '2Z+'],
               ordinate_dim=[PSD_DIM] * 2)
    past.scale_db = 0
    (row,) = rows(rq.evaluate(limited('2Z+'), reference(('1Z+', '2Z+')),
                              past), 'limit')
    assert (row.excursions[0].high, row.excursions[0].worst_db) == (
        480.0, pytest.approx(10 * math.log10(2)))
