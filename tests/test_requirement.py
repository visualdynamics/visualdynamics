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

    def controls(self):
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
