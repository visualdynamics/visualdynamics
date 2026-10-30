"""Findings register (report feature `findings`).

The rules a finding is checked against (`core.findings`): kinds and
numbers, what a finding holds and refuses, its owed parts, the
sentences that say a kind has none, and the one part a program can
offer, a fact read off a measurement, checked against hand values and
against the lines the comparison figure marks. Then the register on a
report (`report.findings`): entering, editing and removing, the
one-step None, the fact draft offered, accepted and discarded through
its one seam, the check before export, the page, the project verbs and
their journal, persistence, and the editor's pane. And the promise the
feature rests on: off, the page is the page it was. Last, the findings
a run log brought into the project offers: drafts beside the register,
nothing entered and the section owed until the author accepts one.
"""

from __future__ import annotations

import html
import json
import math

import numpy as np
import pytest
from conftest import fixture_path
from test_run_log_extract import HEADER, two_tests, writer

import visualdynamics
from visualdynamics.core import compliance
from visualdynamics.core import findings as cf
from visualdynamics.core import run_log_model as model
from visualdynamics.core.data import Psd, Specification
from visualdynamics.core.report import Report
from visualdynamics.report import features as rf
from visualdynamics.report import findings as fd
from visualdynamics.report import page, render_html
from visualdynamics.report import skeleton as sk

BOTH = ['skeleton_sections', 'findings']


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


def _render(report, objects=None, **kwargs):
    return render_html(report, objects or {}, unit_system=visualdynamics.SI,
                       **kwargs)


def _head(html, number):
    """The html of the skeleton head numbered `number` on the page."""
    for block in _payload(html)['blocks']:
        if block.get('html', '').startswith(
                f'<h2 class="skeleton">{number}.'):
            return block['html']
    raise AssertionError(f'no head {number}')


def _report(*features):
    return Report('R', [], features=list(features or BOTH))


# ---- a flat specification and a measurement with known excursions ---------

#: lines 2 Hz apart from 20 Hz: not starting at zero
LINES = np.arange(20.0, 102.0, 2.0)


def spec(level=1.0, up=2.0, down=0.5, dim='acceleration**2/frequency',
         limits=True, band=(10.0, 1000.0)):
    """Flat over `band` (10 to 1000 Hz), abort at twice and half the
    level (plus and minus 3.01 dB)."""
    frequencies = np.array(band, dtype=float)
    values = np.full((1, 2), float(level))
    extra = ({'abort_upper': values * up, 'abort_lower': values * down}
             if limits else {})
    return Specification(abscissa=frequencies, ordinate=values,
                         response_dof=['1X+'], ordinate_dim=[dim], **extra)


def measured(values, dim='acceleration**2/frequency', lines=LINES):
    return Psd(abscissa=np.asarray(lines, dtype=float),
               ordinate=np.atleast_2d(np.asarray(values, dtype=float)),
               response_dof=['1X+'], ordinate_dim=[dim])


def bumped():
    """Flat at the level, 4 times it on lines 30 to 34 Hz (3.01 dB over
    the upper limit) and a tenth of it at 60 Hz (6.99 dB under the
    lower)."""
    values = np.ones(LINES.size)
    values[(LINES >= 30) & (LINES <= 34)] = 4.0
    values[LINES == 60] = 0.1
    return values


# ---- kinds and numbers ------------------------------------------------------


def test_numbers_are_a_kinds_letters_and_a_whole_number_from_one():
    assert cf.number('D-1') == ('deviation', 1)
    assert cf.number('A-12') == ('anomaly', 12)
    assert cf.number('NC-3') == ('nonconformance', 3)
    for bad in ('D-0', 'd-1', 'D1', 'D-01', 'X-1', 'NC-', ' D-1'):
        with pytest.raises(ValueError, match='is not a finding number'):
            cf.number(bad)
    with pytest.raises(TypeError, match='is a str'):
        cf.number(1)
    with pytest.raises(ValueError, match='unknown kind of finding'):
        cf.kind_of('waiver')


def test_the_next_number_is_one_past_the_highest_of_its_kind():
    entries = [{'number': 'D-1'}, {'number': 'D-3'}, {'number': 'A-1'}]
    assert cf.next_number(entries, 'deviation') == 'D-4', 'a gap is kept'
    assert cf.next_number(entries, 'anomaly') == 'A-2'
    assert cf.next_number(entries, 'nonconformance') == 'NC-1'
    assert cf.next_number([], 'deviation') == 'D-1'


def test_a_finding_is_cleaned_and_what_cannot_be_read_one_way_is_refused():
    entry = cf.clean({'number': 'A-2', 'fact': '  M1 clipped. ',
                      'band': (600, 660.5), 'status': 'open'})
    assert entry == {'number': 'A-2', 'kind': 'anomaly', 'run': '',
                     'channel': '', 'band': [600.0, 660.5],
                     'fact': 'M1 clipped.', 'cause': '', 'effect': '',
                     'approval': '', 'status': 'open', 'draft': None}
    assert cf.clean({'number': 'D-1', 'band': [640, 640]})['band'] == [
        640.0, 640.0], 'one frequency'
    refusals = [
        ({'number': 'D-1', 'kind': 'anomaly'}, 'is a deviation by its number'),
        ({'number': 'D-1', 'shade': 'red'}, 'holds no shade'),
        ({'number': 'D-1', 'band': [1, 2, 3]}, '3 were given'),
        ({'number': 'D-1', 'band': [float('nan'), 2]}, 'is not a frequency'),
        ({'number': 'D-1', 'band': [-1, 2]}, 'is negative'),
        ({'number': 'D-1', 'band': [660, 600]},
         'the low edge 660 Hz is above the high edge 600 Hz'),
        ({'number': 'D-1', 'status': 'pending'}, 'is not open or closed'),
        ({'number': 'D-1', 'draft': {'text': ' '}}, 'offers nothing'),
        ({'number': 'D-1', 'draft': {'text': 'x', 'by': 'me'}},
         'not by'),
    ]
    for entry, message in refusals:
        with pytest.raises(ValueError, match=message):
            cf.clean(entry)
    for entry in ({'number': 'D-1', 'cause': 3},
                  {'number': 'D-1', 'band': 'wide'},
                  {'number': 'D-1', 'band': ['a', 2]},
                  {'number': 'D-1', 'band': [True, 2]},
                  {'number': 'D-1', 'draft': 'text'}):
        with pytest.raises(TypeError):
            cf.clean(entry)


def test_a_register_reads_by_kind_then_number_and_refuses_a_number_twice():
    entries = [{'number': n} for n in ('NC-1', 'A-1', 'D-10', 'D-2')]
    assert [e['number'] for e in cf.read(entries)] == [
        'D-2', 'D-10', 'A-1', 'NC-1']
    assert cf.read(None) == []
    with pytest.raises(ValueError, match='D-2 is given to two findings'):
        cf.read([{'number': 'D-2'}, {'number': 'D-2'}])
    with pytest.raises(TypeError, match='list of findings'):
        cf.read({'number': 'D-1'})


def test_a_blank_part_is_owed_and_none_is_never_filled_in():
    assert cf.owed({'fact': 'x', 'cause': ' ', 'effect': '',
                    'approval': 'y'}) == ['cause', 'effect']
    assert cf.owed({}) == list(cf.PARTS)


def test_the_none_sentences_are_the_sections_none_and_read_back():
    assert ' '.join(cf.NONE_SENTENCES[k] for k in cf.KINDS) == \
        sk.BY_KEY['deviations'].none, 'one wording, not two'
    words = 'Seen on the day.\n\nNo anomalies. No nonconformances.'
    assert cf.declared(words) == {'anomaly', 'nonconformance'}
    assert cf.declared('') == set()
    assert cf.withdraw(words, ['anomaly']) == ('Seen on the day.\n\n'
                                               'No nonconformances.')
    assert cf.withdraw(sk.BY_KEY['deviations'].none, cf.KINDS) == ''
    assert cf.declaration(['nonconformance', 'deviation']) == (
        'No deviations. No nonconformances.')


# ---- the fact read off a measurement ----------------------------------------


def test_excursions_against_hand_values():
    reading = cf.excursions(spec(), measured(bumped()))
    above, below = reading.excursions
    assert (above.side, above.low, above.high, above.lines) == (
        'above', 30.0, 34.0, 3)
    assert above.worst_db == pytest.approx(10 * math.log10(4 / 2))
    assert (below.side, below.low, below.high, below.lines) == (
        'below', 60.0, 60.0, 1)
    assert below.worst_db == pytest.approx(10 * math.log10(0.5 / 0.1))
    assert reading.judged == LINES.size and reading.unjudged == 0
    assert reading.low == pytest.approx(19.0)
    assert reading.high == pytest.approx(101.0)
    assert reading.scale_db == 0.0


def test_an_excursion_on_the_first_and_the_last_line():
    values = np.ones(LINES.size)
    values[0] = values[-1] = 3.0
    first, last = cf.excursions(spec(), measured(values)).excursions
    assert (first.low, first.high, first.lines) == (20.0, 20.0, 1)
    assert (last.low, last.high, last.lines) == (100.0, 100.0, 1)
    assert first.worst_db == pytest.approx(10 * math.log10(1.5))


def test_a_band_reads_only_what_reaches_into_it():
    reading = cf.excursions(spec(), measured(bumped()), band=(29, 33))
    (above,) = reading.excursions
    assert (above.low, above.high, above.lines) == (30.0, 32.0, 2), \
        'the cell at 34 Hz, from 33 to 35, only touches the band'
    reading = cf.excursions(spec(), measured(bumped()), band=(60, 60))
    assert [e.side for e in reading.excursions] == ['below'], \
        'a band of one frequency reads the cell holding it'
    assert reading.judged == 1


def test_a_band_of_one_frequency_on_a_cells_edge_reads_both_cells():
    reading = cf.excursions(spec(), measured(bumped()), band=(61.0, 61.0))
    assert reading.judged == 2, 'the cells from 59 to 61 and 61 to 63'
    assert [e.side for e in reading.excursions] == ['below']


def test_lines_on_the_specifications_own_ends_are_counted():
    """The first and last cells end exactly on a line when the
    specification starts and stops on one."""
    values = np.ones(LINES.size)
    values[0] = values[-1] = 3.0
    first, last = cf.excursions(spec(band=(20.0, 100.0)), measured(values)
                                ).excursions
    assert (first.low, first.high, first.lines) == (20.0, 20.0, 1)
    assert (last.low, last.high, last.lines) == (100.0, 100.0, 1)


def test_lines_with_no_value_are_counted_in_the_band_edges_included():
    values = bumped()
    values[LINES == 32] = np.nan
    for band, count in (((25, 50), 1), ((32, 50), 1), ((20, 32), 1),
                        ((34, 50), 0), ((20, 30), 0)):
        assert cf.excursions(spec(), measured(values), band=band
                             ).unjudged == count, band


def test_a_band_only_partly_covered_by_the_data_says_what_was_judged():
    reading = cf.excursions(spec(), measured(bumped()), band=(5.0, 25.0))
    assert reading.excursions == ()
    assert reading.low == pytest.approx(19.0), 'judged from the data on'
    text_band = (5.0, 25.0)
    assert reading.low > text_band[0]


def test_a_constant_record_above_its_limit_is_one_stretch():
    (stretch,) = cf.excursions(spec(), measured(np.full(LINES.size, 3.0))
                               ).excursions
    assert (stretch.low, stretch.high, stretch.lines) == (20.0, 100.0,
                                                          LINES.size)


def test_dead_nan_and_empty_inputs_each_name_their_cause():
    with pytest.raises(ValueError, match='holds no power.*all-zero'):
        cf.excursions(spec(), measured(np.zeros(LINES.size)))
    values = bumped()
    values[LINES == 32] = np.nan
    reading = cf.excursions(spec(), measured(values))
    assert reading.unjudged == 1, 'the NaN line is said, not judged'
    first, second, _below = reading.excursions
    assert (first.low, first.high, second.low, second.high) == (
        30.0, 30.0, 34.0, 34.0), 'a gap in the data is a gap in the stretch'
    assert cf.excursions(spec(), measured(values), band=(40, 80)
                         ).unjudged == 0, 'counted only in the band'
    text = fd.fact_text('1X+', reading, 'abort', "PSD 'psd'")
    assert text.endswith('; 1 line with no value, not judged.')
    with pytest.raises(ValueError, match='nothing judged from 2000 to 3000'):
        cf.excursions(spec(), measured(bumped()), band=(2000, 3000))
    with pytest.raises(ValueError, match='nothing judged'):
        cf.excursions(spec(), measured(np.full(LINES.size, np.nan)))


def test_a_line_below_its_limit_with_no_power_is_infinitely_far():
    values = np.ones(LINES.size)
    values[LINES == 60] = 0.0
    (below,) = cf.excursions(spec(), measured(values)).excursions
    assert below.side == 'below' and math.isinf(below.worst_db)


def test_what_the_comparison_refuses_the_reading_refuses():
    with pytest.raises(ValueError, match='abort or warning|warning or abort'):
        cf.excursions(spec(), measured(bumped()), pair='red')
    with pytest.raises(ValueError, match='carries no warning limits'):
        cf.excursions(spec(), measured(bumped()), pair='warning')
    with pytest.raises(ValueError, match='carries no abort limits'):
        cf.excursions(spec(limits=False), measured(bumped()))
    with pytest.raises(ValueError, match='velocity|quantit|units'):
        cf.excursions(spec(), measured(bumped(),
                                       dim='velocity**2/frequency'))


def test_records_are_paired_as_the_comparison_pairs_them():
    assert cf.pair_records(spec(), measured(bumped()), '1X+') == (0, 0)
    with pytest.raises(ValueError, match='2Y\\+ is not a channel.*share: 1X\\+'):
        cf.pair_records(spec(), measured(bumped()), '2Y+')
    with pytest.raises(ValueError, match='name the channel'):
        cf.pair_records(spec(), measured(bumped()), ' ')


def test_the_scale_the_comparison_uses_is_the_scale_read():
    held = measured(bumped() / 4.0)       # the bump at the level, -6 dB
    held.scale_db = 6
    (above, *_rest) = cf.excursions(spec(), held).excursions
    assert above.worst_db == pytest.approx(10 * math.log10(4 / 2), abs=0.03)
    assert cf.excursions(spec(), held).scale_db == 6.0


@pytest.fixture(scope='module')
def run():
    """A random run worked up with the default settings (Hann), and its
    report."""
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'))
    name = project.generate_report('random')
    return project, name


def test_the_stretches_are_the_lines_the_comparison_figure_marks(run):
    """Read under the default window off a worked-up run, the stretches
    cover exactly the lines `compliance.exceedances` marks, which is
    what the figure draws, on every control channel."""
    project, _name = run
    specification = project['Specification']
    held = project['Time History PSDs']
    for label, s, m in compliance.matched_records(specification, held):
        over, under = compliance.exceedances(specification, held, s, m)
        try:
            reading = cf.excursions(specification, held, s, m)
        except ValueError:
            assert not over.any() and not under.any(), label
            continue
        lines = np.asarray(held.abscissa)
        marked = np.zeros(lines.size, dtype=bool)
        for excursion in reading.excursions:
            reach = (lines >= excursion.low) & (lines <= excursion.high)
            assert reach.sum() == excursion.lines, label
            marked |= reach
        assert np.array_equal(marked, over | under), label


# ---- the register on a report ------------------------------------------------


def test_the_feature_needs_the_skeleton_and_is_off_by_default():
    feature = rf.FEATURES['findings']
    assert feature.available and not feature.default
    assert feature.requires == ('skeleton_sections',)
    assert rf.reason('findings') == 'needs Skeleton sections'
    report = _report('skeleton_sections')
    with pytest.raises(ValueError, match='does not have the Findings register'):
        fd.add(report, 'deviation')
    assert report.blocks == []


def test_findings_are_numbered_by_kind_and_a_number_is_never_reused():
    report = _report()
    assert fd.add(report, 'deviation') == 'D-1'
    assert fd.add(report, 'anomaly', fact='Seen.') == 'A-1'
    assert fd.add(report, 'deviation') == 'D-2'
    fd.remove(report, 'D-2')
    assert fd.add(report, 'deviation') == 'D-3'
    assert fd.add(report, 'nonconformance', number='NC-4') == 'NC-4'
    assert [e['number'] for e in fd.register(report)] == [
        'D-1', 'D-3', 'A-1', 'NC-4']
    # the register lives in the deviations section's own block
    (index,) = sk.blocks_of(report, 'deviations')
    assert len(report.blocks[index]['findings']) == 4


def test_a_refused_entry_changes_nothing():
    report = _report()
    fd.add(report, 'deviation', channel='2Y+')
    before = json.dumps(report.blocks)
    for call, message in (
            (lambda: fd.add(report, 'deviation', number='A-2'),
             'is a anomaly number|is an? anomaly number'),
            (lambda: fd.add(report, 'deviation', number='D-1'),
             'already entered'),
            (lambda: fd.add(report, 'waiver'), 'unknown kind'),
            (lambda: fd.add(report, 'anomaly', band=[5, 1]), 'above the high'),
            (lambda: fd.add(report, 'anomaly', shade='x'), 'not shade'),
            (lambda: fd.edit(report, 'D-1', status='done'), 'open or closed'),
            (lambda: fd.edit(report, 'D-9', fact='x'), 'no finding D-9'),
            (lambda: fd.edit(report, 'D-1', kind='anomaly'), 'not kind'),
            (lambda: fd.remove(report, 'A-1'), 'entered: D-1'),
            (lambda: fd.accept(report, 'D-1'), 'no fact draft waiting'),
            (lambda: fd.discard(report, 'D-1'), 'no fact draft waiting')):
        with pytest.raises(ValueError, match=message):
            call()
        assert json.dumps(report.blocks) == before
    with pytest.raises(TypeError):
        fd.add(report, 'anomaly', cause=1)
    with pytest.raises(TypeError, match='FactDraft'):
        fd.offer(report, 'D-1', {'text': 'x', 'source': 'y'})
    assert json.dumps(report.blocks) == before


def test_editing_leaves_what_is_not_given_and_an_empty_band_clears_it():
    report = _report()
    fd.add(report, 'deviation', channel='2Y+', band=(600, 660), fact='F.')
    fd.edit(report, 'D-1', cause='The fixture.', channel=None)
    entry = fd.find(report, 'D-1')
    assert entry['channel'] == '2Y+' and entry['cause'] == 'The fixture.'
    assert entry['band'] == [600.0, 660.0] and entry['fact'] == 'F.'
    fd.edit(report, 'D-1', band=())
    assert fd.find(report, 'D-1')['band'] is None
    fd.remove(report, 'D-1')
    (index,) = sk.blocks_of(report, 'deviations')
    assert 'findings' not in report.blocks[index]
    assert report.blocks[index][fd.ISSUED] == {'D': 1}, \
        'the number issued is remembered'
    assert fd.add(report, 'deviation') == 'D-2'
    with pytest.raises(ValueError, match='D-1 was issued before'):
        fd.add(report, 'deviation', number='D-1')


def test_none_with_nothing_entered_is_the_skeletons_none_exactly():
    report, plain = _report(), _report('skeleton_sections')
    sk.write(plain, 'deviations', none=True)
    assert fd.none(report) == sk.BY_KEY['deviations'].none
    assert report.blocks == plain.blocks


def test_none_with_findings_entered_says_none_only_of_the_empty_kinds():
    report = _report()
    sk.write(report, 'deviations', 'Read with the log.')
    fd.add(report, 'deviation')
    assert fd.none(report) == ('Read with the log.\n\nNo anomalies. '
                               'No nonconformances.')
    # a finding of a kind said to be none withdraws the sentence
    fd.add(report, 'anomaly')
    assert sk.words(report, 'deviations') == ('Read with the log.\n\n'
                                              'No nonconformances.')
    fd.add(report, 'nonconformance')
    assert sk.words(report, 'deviations') == 'Read with the log.'
    with pytest.raises(ValueError, match='every kind has a finding entered'):
        fd.none(report)


def test_a_draft_is_offered_through_one_seam_and_accepted_or_discarded():
    report = _report()
    fd.add(report, 'deviation', channel='C2 as typed')
    draft = fd.FactDraft('C2 over by 1.22 dB.', 'a measurement', run='Y run 11',
                         channel='2Y+', band=(617.5, 660.0))
    fd.offer(report, 'D-1', draft)
    entry = fd.find(report, 'D-1')
    assert entry['fact'] == '' and entry['draft']['text'] == draft.text, \
        'a draft fills nothing'
    fd.accept(report, 'D-1')
    entry = fd.find(report, 'D-1')
    assert entry['fact'] == 'C2 over by 1.22 dB.' and entry['draft'] is None
    assert entry['run'] == 'Y run 11' and entry['band'] == [617.5, 660.0]
    assert entry['channel'] == 'C2 as typed', "the author's field stands"
    fd.offer(report, 'D-1', draft)
    fd.accept(report, 'D-1', '  As edited.  ')
    assert fd.find(report, 'D-1')['fact'] == 'As edited.'
    fd.offer(report, 'D-1', draft)
    with pytest.raises(ValueError, match='blank fact is not accepted'):
        fd.accept(report, 'D-1', ' ')
    fd.discard(report, 'D-1')
    entry = fd.find(report, 'D-1')
    assert entry['draft'] is None and entry['fact'] == 'As edited.'


def _objects():
    return ({'psd': measured(bumped()), 'spec': spec(), 'th': object()},
            {'psd': {'source': 'th'}})


def test_the_fact_draft_in_words():
    objects, provenance = _objects()
    draft = fd.fact_from_comparison(objects, 'psd', 'spec', '1X+',
                                    provenance=provenance)
    assert draft.text == (
        "1X+ above its abort limit from 30.0 to 34.0 Hz by up to 3.01 dB "
        "(3 lines); below its abort limit at 60.0 Hz by up to 6.99 dB "
        "(1 line); PSD 'psd', from 'th'.")
    assert draft.run == 'th' and draft.channel == '1X+'
    assert draft.band == (30.0, 60.0)
    assert draft.source == 'psd against spec, 1X+, abort limits'
    partial = fd.fact_from_comparison(objects, 'psd', 'spec', '1X+',
                                      band=(25.0, 200.0))
    assert partial.text.endswith(
        "PSD 'psd'; judged from 25.0 to 101.0 Hz of the 25.0 to 200.0 Hz "
        'asked.')


def test_many_stretches_are_said_once_a_side():
    values = np.ones(LINES.size)
    values[::4] = 2.5                    # every fourth line, 0.97 dB over
    values[LINES == 40] = 8.0            # the worst, 6.02 dB over
    objects = {'psd': measured(values), 'spec': spec()}
    text = fd.fact_from_comparison(objects, 'psd', 'spec', '1X+').text
    assert text.startswith('1X+ above its abort limit in 12 stretches from '
                           '20.0 to 100.0 Hz, 12 lines in all, by up to '
                           '6.02 dB (at 40.0 Hz)')


def test_no_draft_from_a_channel_inside_its_limits_or_a_wrong_name():
    objects, _provenance = _objects()
    with pytest.raises(ValueError, match='inside its abort limits wherever '
                                         'it was judged.*no excursion'):
        fd.fact_from_comparison(objects, 'psd', 'spec', '1X+',
                                band=(70, 90))
    with pytest.raises(ValueError, match="'nope' is not an object"):
        fd.fact_from_comparison(objects, 'nope', 'spec', '1X+')
    with pytest.raises(ValueError, match="'spec' is a Specification, not a "
                                         'measured spectrum'):
        fd.fact_from_comparison(objects, 'spec', 'spec', '1X+')
    with pytest.raises(ValueError, match="'psd' is a Psd, not a "
                                         'specification'):
        fd.fact_from_comparison(objects, 'psd', 'psd', '1X+')


def test_drafting_reads_the_findings_channel_and_band_and_offers_it():
    objects, provenance = _objects()
    report = _report()
    fd.add(report, 'deviation')
    with pytest.raises(ValueError, match='D-1: name the channel'):
        fd.draft_from_comparison(report, 'D-1', objects, 'psd', 'spec')
    fd.edit(report, 'D-1', channel='1X+', band=(50, 70))
    draft = fd.draft_from_comparison(report, 'D-1', objects, 'psd', 'spec',
                                     provenance=provenance)
    assert draft.text.startswith('1X+ below its abort limit at 60.0 Hz')
    assert fd.find(report, 'D-1')['draft']['text'] == draft.text


# ---- the check before export ------------------------------------------------


def _owed(report):
    return [str(item) for item in fd.check(report, sk.check(report, {}))
            if item.key == 'deviations']


def test_the_check_reads_section_nine_off_the_register():
    report = _report()
    assert _owed(report) == [
        '9. Deviations, Anomalies and Nonconformances: not yet written']
    fd.add(report, 'deviation', fact='F.', cause='C.')
    fd.add(report, 'anomaly', fact='F.', cause='C.', effect='E.')
    assert _owed(report) == [
        ('9. Deviations, Anomalies and Nonconformances: nonconformances '
        'neither entered nor set to None'),
        ('9. Deviations, Anomalies and Nonconformances: D-1: effect, '
        'approval not yet written'),
        ('9. Deviations, Anomalies and Nonconformances: A-1: disposition not '
        'yet written')]
    fd.none(report)
    fd.edit(report, 'D-1', effect='E.', approval='The customer, Day 2.')
    fd.edit(report, 'A-1', approval='Closed by the analyst.')
    assert _owed(report) == []
    # the author's own words saying none of a kind that has findings
    sk.write(report, 'deviations', 'No deviations. No nonconformances.')
    assert _owed(report) == [
        ('9. Deviations, Anomalies and Nonconformances: deviations entered '
        'and also said to be none')]


def test_the_check_is_the_skeletons_with_the_feature_off_and_in_order():
    report = _report('skeleton_sections')
    owed = sk.check(report, {})
    assert fd.check(report, owed) == owed
    report = _report()
    fd.add(report, 'deviation')
    numbers = [item.number for item in fd.check(report, sk.check(report, {}))]
    assert numbers == sorted(numbers)


def test_issue_reads_section_nine_off_the_register():
    """With Identity and issue on, the issue check owes what the export
    check owes: section 9 read off the register, not off its words."""
    from visualdynamics.report import identity as ri

    report = _report(*BOTH, ri.FEATURE)
    ri.identify(report, **{
        field.key: ['A. Engineer, test engineer'] if field.key == 'authorized'
        else 'given' for field in ri.FIELDS if field.required})
    fd.add(report, 'deviation', fact='F.', cause='C.')
    with pytest.raises(ValueError, match='D-1: effect, approval not yet '
                                         'written'):
        ri.issue_ready(report, {})
    fd.none(report)
    fd.edit(report, 'D-1', effect='E.', approval='The customer, Day 2.')
    with pytest.raises(ValueError,
                       match='cannot be issued while it owes') as refused:
        ri.issue_ready(report, {})
    assert '9. Deviations' not in str(refused.value), (
        'a register with nothing owed fills section 9 with no words in it')


# ---- the page -----------------------------------------------------------------


def test_off_the_page_is_the_page_it_was_even_with_findings_entered():
    report = _report()
    fd.add(report, 'deviation', fact='F.')
    report.features = ['skeleton_sections']
    skeleton_only = Report('R', [], features=['skeleton_sections'])
    sk.write(skeleton_only, 'deviations')
    assert _render(report) == _render(skeleton_only), \
        'the register is not drawn while the feature is off'
    report.features = []
    assert _render(report) == _render(Report('R', [])), \
        'nor anything of the section with nothing enabled'


def test_the_stylesheet_is_added_only_with_the_feature():
    report = _report('skeleton_sections')
    assert page._FINDINGS_CSS not in _render(report)
    report.features = BOTH
    assert page._FINDINGS_CSS in _render(report)


def test_an_untouched_section_stays_owed_and_the_summary_says_so():
    html = _render(_report())
    assert sk.OWED in _head(html, 9) and 'class="finding' not in _head(html, 9)
    assert _head(html, 2).endswith(
        '<p class="findings-summary">Deviations: <span class="owed">not yet '
        'written</span>. Anomalies: <span class="owed">not yet written'
        '</span>. Nonconformances: <span class="owed">not yet written'
        '</span>.</p>')


def test_section_nine_draws_each_finding_in_four_parts_in_order():
    report = _report()
    fd.add(report, 'nonconformance', fact='Under-test at the interface.',
           status='open')
    fd.add(report, 'anomaly', fact='M1 clipped.\nC1 reached 32.7 g.',
           cause='<autorange>', status='closed')
    fd.add(report, 'deviation', run='Y run 11', channel='2Y+',
           band=(617.5, 660.0), fact='Over.', cause='Fixture mode.',
           effect='Over-test.', approval='The customer.')
    fd.offer(report, 'D-1', fd.FactDraft('A draft never drawn.', 's'))
    html = _head(_render(report), 9)
    assert html.index('D-1.') < html.index('A-1.') < html.index('NC-1.')
    assert ('<p class="finding-head"><strong>D-1.</strong> Deviation '
            '<span class="finding-where">(run Y run 11; 2Y+; 617.5 to 660.0 '
            'Hz)</span></p>') in html
    assert '<strong>Approval:</strong> The customer.' in html
    assert ('<strong>Fact:</strong> M1 clipped.<br>C1 reached 32.7 g.'
            in html), 'a line break, and nothing the page would draw as a gap'
    assert '&lt;autorange&gt;' in html and '<autorange>' not in html
    assert ('<strong>Disposition:</strong> <span class="owed">NOT YET '
            'WRITTEN</span>') in html, 'a blank part is owed, not hidden'
    assert 'A draft never drawn.' not in html
    assert '\n' not in html
    assert sk.OWED + '</strong>' not in html, 'no owed card'
    summary = _head(_render(report), 2)
    assert ('Deviations: D-1. Anomalies: A-1 (closed). Nonconformances: '
            'NC-1 (open).') in summary


def test_none_sentences_are_drawn_in_their_kinds_places():
    report = _report()
    sk.write(report, 'deviations', 'Read with the log.')
    fd.add(report, 'anomaly', fact='Seen.')
    html = _head(_render(report), 9)
    assert html.index('Read with the log.') < html.index('A-1.')
    assert html.endswith('<p class="finding-none"><span class="owed">NOT YET '
                         'WRITTEN</span> Nonconformances: enter each in its '
                         'four parts, or set None.</p>')
    assert ('<span class="owed">NOT YET WRITTEN</span> Deviations: enter '
            'each') in html
    fd.none(report)
    html = _head(_render(report), 9)
    assert html.count('No deviations.') == 1
    assert html.index('No deviations.') < html.index('A-1.') < html.index(
        'No nonconformances.'), 'each kind in its place'
    assert 'Deviations: none. Anomalies: A-1. Nonconformances: none.' in \
        _head(_render(report), 2)


def test_the_none_wording_alone_draws_three_kinds_of_none():
    report = _report()
    fd.none(report)
    html = _head(_render(report), 9)
    for sentence in cf.NONE_SENTENCES.values():
        assert f'<p class="finding-none">{sentence}</p>' in html


def test_a_contradiction_stays_on_the_page():
    report = _report()
    fd.add(report, 'deviation', fact='F.')
    sk.write(report, 'deviations', 'No deviations.')
    html = _head(_render(report), 9)
    assert 'No deviations.' in html and 'D-1.' in html


def test_the_edit_page_head_names_the_sections_block(run):
    project, name = run
    report = project[name]
    blocks = [dict(block) for block in report.blocks]
    try:
        project.report_features(name, enable=BOTH)
        number = project.report_finding(name, kind='deviation')
        data = _payload(_render(report, project, edit=True,
                                object_groups=project.object_groups))
        (head,) = [block for block in data['blocks']
                   if block.get('html', '').startswith(
                       '<h2 class="skeleton">9.')]
        assert head['index'] == sk.blocks_of(report, 'deviations')[0]
        assert f'<strong>{number}.</strong>' in head['html']
    finally:
        report.blocks = blocks
        report.features = []


# ---- the project verbs ---------------------------------------------------------


def test_the_verbs_are_journaled_and_refuse_in_the_core(run):
    project, name = run
    report = project[name]
    blocks = [dict(block) for block in report.blocks]
    try:
        with pytest.raises(ValueError, match='Findings register'):
            project.report_finding(name, kind='deviation')
        project.report_features(name, enable=BOTH)
        start = len(project.journal)
        assert project.report_finding(name, kind='deviation',
                                      channel='101Z+') == 'D-1'
        text = project.draft_finding_fact(name, 'D-1', 'Time History PSDs',
                                          'Specification', band=(140, 210))
        assert text.startswith('101Z+ above its abort limit ')
        assert "PSD 'Time History PSDs', from 'Time History'." in text
        project.report_finding(name, 'D-1', accept=True, cause='C.')
        entry = fd.find(report, 'D-1')
        assert entry['fact'] == text and entry['cause'] == 'C.'
        assert entry['run'] == 'Time History'
        project.report_section(name, 'deviations', none=True)
        assert sk.words(report, 'deviations') == ('No anomalies. '
                                                  'No nonconformances.')
        assert project.journal[start:] == [
            (f"project.report_finding({name!r}, kind='deviation', "
            "channel='101Z+')"),
            (f"project.draft_finding_fact({name!r}, 'D-1', "
            "'Time History PSDs', 'Specification', band=(140, 210))"),
            (f"project.report_finding({name!r}, 'D-1', accept=True, "
            "cause='C.')"),
            f"project.report_section({name!r}, 'deviations', none=True)"]
        before = list(project.journal)
        for call in (
                lambda: project.report_finding(name),
                lambda: project.report_finding(name, 'D-1', kind='anomaly'),
                lambda: project.report_finding(name, 'D-1', remove=True,
                                               fact='x'),
                lambda: project.report_finding(name, 'D-1', accept=True,
                                               discard=True),
                lambda: project.report_finding(name, accept=True),
                lambda: project.report_finding(name, 'D-7'),
                lambda: project.draft_finding_fact(
                    name, 'D-1', 'Time History PSDs', 'Specification',
                    band=(600, 700)),
                lambda: project.draft_finding_fact(
                    name, 'D-1', 'Time History', 'Specification')):
            with pytest.raises(ValueError):
                call()
        assert project.journal == before, 'a refusal is not journaled'
        owed = project.report_check(name)
        assert ('9. Deviations, Anomalies and Nonconformances: D-1: effect, '
                'approval not yet written') in owed
        project.report_finding(name, 'D-1', remove=True)
        assert fd.register(report) == []
    finally:
        report.blocks = blocks
        report.features = []


def test_the_register_rides_in_the_project_file_and_the_template(tmp_path):
    from visualdynamics import io
    from visualdynamics.io import report_template

    report = _report()
    fd.add(report, 'deviation', band=(617.5, 660), fact='F.')
    fd.offer(report, 'D-1', fd.FactDraft('D.', 's', band=(1.0, 2.0)))
    path = str(tmp_path / 'r.vdyn')
    io.save(report, path)
    back = io.load(path)
    assert back.blocks == report.blocks and fd.register(back) == \
        fd.register(report)
    report_template.save(report, tmp_path / 'r.vdreport')
    again = report_template.load(tmp_path / 'r.vdreport')
    assert fd.register(again) == fd.register(report)


# ---- the editor's pane ---------------------------------------------------------


@pytest.fixture
def editing(window, pump, project, run):
    for name, obj in project.items():
        window.add_object(name, obj)
    measured_project, _name = run
    for name in ('Time History PSDs', 'Specification'):
        window.add_object(name, measured_project[name])
    name = window.generate_report('modal')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    report = window.objects[name]
    editor = window.report_editor
    editor._operate({'op': 'features', 'enable': BOTH})
    editor._operate({'op': 'select', 'at': 'section:deviations'})
    return window, editor, report, name


def test_the_pane_holds_the_register_only_with_the_feature(editing):
    _window, editor, _target, _name = editing
    assert editor.findings_pane is not None
    editor._operate({'op': 'features', 'disable': ['findings']})
    editor._operate({'op': 'select', 'at': None})
    editor._operate({'op': 'select',
                     'at': sk.blocks_of(editor.report, 'deviations')[0]})
    assert editor.findings_pane is None and editor.none_button is not None


def test_entering_a_finding_in_the_pane(editing):
    window, editor, report, name = editing
    pane = editor.findings_pane
    pane.kind_box.setCurrentIndex(pane.kind_box.findData('anomaly'))
    pane.add_button.click()
    assert [e['number'] for e in fd.register(report)] == ['A-1']
    pane = editor.findings_pane
    assert pane.entry['number'] == 'A-1', 'the new finding is chosen'
    pane.parts['fact'].setPlainText('M1 clipped.')
    pane.parts['cause'].setPlainText('Autorange.')
    editor.flush_text()
    entry = fd.find(report, 'A-1')
    assert (entry['fact'], entry['cause']) == ('M1 clipped.', 'Autorange.')
    pane.lines['channel'].setText('M1')
    pane.lines['channel'].editingFinished.emit()
    pane.lines['low'].setText('600')
    pane.lines['high'].setText('nope')
    pane.lines['high'].editingFinished.emit()
    assert fd.find(report, 'A-1')['channel'] == 'M1'
    assert fd.find(report, 'A-1')['band'] is None
    assert 'is not a frequency' in editor.findings_pane.note.text()
    journaled = [line for line in window.project.journal
                 if line.startswith(f'project[{name!r}].blocks = ')]
    assert len(journaled) == 1 and 'M1 clipped.' in journaled[0]


def test_drafting_accepting_and_none_in_the_pane(editing):
    window, editor, report, name = editing
    editor.findings_pane.add_button.click()          # D-1
    pane = editor.findings_pane
    pane.draft_button.click()
    assert 'D-1: name the channel' in editor.findings_pane.note.text(), \
        'the refusal is shown, and nothing changed'
    assert fd.find(report, 'D-1')['draft'] is None
    pane = editor.findings_pane
    pane.lines['channel'].setText('101Z+')
    pane.lines['channel'].editingFinished.emit()
    pane.measured_box.setCurrentIndex(
        pane.measured_box.findData('Time History PSDs'))
    pane.specification_box.setCurrentIndex(
        pane.specification_box.findData('Specification'))
    pane.draft_button.click()
    draft = fd.find(report, 'D-1')['draft']
    assert draft['text'].startswith('101Z+ above its abort limit')
    assert any(line.startswith(f"project.draft_finding_fact({name!r}, 'D-1'")
               for line in window.project.journal), \
        "read on the window's worker through the journaled verb"
    pane = editor.findings_pane
    assert pane.accept_button is not None and pane.discard_button is not None
    pane.accept_button.click()
    entry = fd.find(report, 'D-1')
    assert entry['fact'] == draft['text'] and entry['draft'] is None
    editor.none_button.click()
    assert sk.words(report, 'deviations') == ('No anomalies. '
                                              'No nonconformances.')
    editor.findings_pane.remove_button.click()
    assert fd.register(report) == []


def test_a_waiting_draft_leaves_the_fact_owed_and_marked_as_drafted():
    report = _report()
    fd.add(report, 'deviation', cause='C.', effect='E.', approval='A.')
    fd.none(report)
    fd.offer(report, 'D-1', fd.FactDraft('Drafted words.', 'a measurement'))
    html = _head(_render(report), 9)
    assert ('<strong>Fact:</strong> <span class="owed">NOT YET WRITTEN'
            f'</span> <span class="finding-drafted">{fd.DRAFTED}</span>'
            ) in html
    assert 'Drafted words.' not in html, 'never filled in silently'
    assert _owed(report) == [
        ('9. Deviations, Anomalies and Nonconformances: D-1: fact not yet '
        f'written ({fd.DRAFTED})')]
    fd.accept(report, 'D-1')
    assert _owed(report) == []
    assert fd.DRAFTED not in _head(_render(report), 9)


def test_the_numbers_issued_are_read_and_refused_one_way():
    assert cf.issued_of(None) == {}
    assert cf.issued_of({'D': 3, 'NC': 1}) == {'D': 3, 'NC': 1}
    assert cf.next_number([{'number': 'D-1'}], 'deviation', {'D': 3}) == 'D-4'
    assert cf.next_number([{'number': 'D-5'}], 'deviation', {'D': 3}) == 'D-6'
    for bad in ([('D', 3)], {'X': 1}, {'D': 0}, {'D': True}, {'D': 1.0}):
        with pytest.raises(ValueError, match='numbers issued'):
            cf.issued_of(bad)


def test_a_band_edge_at_zero_is_a_frequency():
    assert cf.band_of([0, 10]) == [0.0, 10.0]
    assert cf.band_of(None) is None


def test_a_register_that_is_not_a_list_is_refused_by_name():
    for bad in ('D-1', b'D-1', {'number': 'D-1'}, 3):
        with pytest.raises(TypeError, match='list of findings'):
            cf.read(bad)


def test_a_channel_compared_as_two_quantities_is_refused():
    frequencies = np.array([10.0, 1000.0])
    values = np.ones((2, 2))
    two = Specification(abscissa=frequencies, ordinate=values,
                        response_dof=['1X+', '1X+'],
                        ordinate_dim=['acceleration**2/frequency',
                                      'force**2/frequency'],
                        abort_upper=values * 2, abort_lower=values / 2)
    held = Psd(abscissa=LINES, ordinate=np.ones((2, LINES.size)),
               response_dof=['1X+', '1X+'],
               ordinate_dim=['acceleration**2/frequency',
                             'force**2/frequency'])
    with pytest.raises(ValueError, match='compared as 2 quantities'):
        cf.pair_records(two, held, '1X+')


def test_three_stretches_on_a_side_are_listed_and_four_are_said_once():
    def text(count):
        values = np.ones(LINES.size)
        values[[4 * k + 2 for k in range(count)]] = 3.0
        return fd.fact_from_comparison({'psd': measured(values),
                                        'spec': spec()},
                                       'psd', 'spec', '1X+').text
    assert text(3).count('above its abort limit at ') == 3
    assert 'in 4 stretches from 24.0 to 48.0 Hz, 4 lines in all' in text(4)


def test_a_band_judged_whole_is_not_said_to_be_part_judged():
    reading = cf.excursions(spec(), measured(bumped()))
    text = fd.fact_text('1X+', reading, 'abort', "PSD 'psd'",
                        band=(reading.low, reading.high))
    assert text.endswith("PSD 'psd'.")


def test_frequencies_are_written_with_as_few_decimals_as_say_them():
    assert fd._hz(617.5) == '617.5' and fd._hz(660.0) == '660.0'
    assert fd._hz(100.25) == '100.25'
    assert fd._hz(1.23456) == '1.2346', 'four decimals at most'
    assert fd._hz(1000.00001) == '1000.0', 'a last digit of float noise'


def test_a_number_is_refused_again_only_once_issued():
    report = _report()
    assert fd.add(report, 'deviation', number='D-1') == 'D-1'
    fd.remove(report, 'D-1')
    with pytest.raises(ValueError, match='D-1 was issued before'):
        fd.add(report, 'deviation', number='D-1')
    assert fd.add(report, 'deviation', number='D-2') == 'D-2'


def test_a_band_short_of_the_data_at_its_low_end_is_said():
    reading = cf.excursions(spec(), measured(bumped()), band=(5.0, 101.0))
    text = fd.fact_text('1X+', reading, 'abort', "PSD 'psd'",
                        band=(5.0, 101.0))
    assert text.endswith('; judged from 19.0 to 101.0 Hz of the 5.0 to '
                         '101.0 Hz asked.')


def test_only_a_blank_fact_with_a_draft_waiting_says_so():
    report = _report()
    fd.add(report, 'deviation')
    fd.add(report, 'anomaly', fact='Seen.')
    fd.offer(report, 'A-1', fd.FactDraft('Another reading.', 's'))
    fd.none(report)
    assert _owed(report) == [
        ('9. Deviations, Anomalies and Nonconformances: D-1: fact, cause, '
         'effect, approval not yet written'),
        ('9. Deviations, Anomalies and Nonconformances: A-1: cause, effect, '
         'disposition not yet written')]


# ---- findings offered from a run log ----------------------------------------


def _logged(tmp_path, *features):
    """A project holding a report with `features` (the register's by
    default) and one run log extract of a random test with a deviation
    (D-1) and an anomaly (A-1) whose disposition the next test logged."""
    path = tmp_path / 'article.vdlog'
    ids = two_tests(path)
    project = visualdynamics.Project('Article 7')
    name = project.add('R', _report(*features))
    held = project.attach_run_log(path, test=1)
    return project, name, held, path, ids


def test_without_a_run_log_the_register_and_page_are_as_before(tmp_path):
    project, name, held, _path, _ids = _logged(tmp_path)
    project.detach_run_log(held)
    report = project[name]
    before = _render(report, project)
    edit = _render(report, project, edit=True)
    with pytest.raises(ValueError, match='holds no run log extract; bring one '
                       'in with attach_run_log'):
        project.draft_findings_from_run_log(name)
    assert fd.register(report) == [] and fd.log_drafts(report) == []
    assert _render(report, project) == before
    assert _render(report, project, edit=True) == edit


def test_the_log_findings_are_offered_and_nothing_enters_the_register(
        tmp_path):
    project, name, held, _path, ids = _logged(tmp_path)
    report = project[name]
    extract = project[held]
    deviation = next(found for found in extract.findings()
                     if found.number == 'D-1')
    assert project.draft_findings_from_run_log(name) == [
        ('D-1: offered as a draft from the run log, not in the register until '
         'accepted.'),
        ('A-1: offered as a draft from the run log, not in the register until '
         'accepted.')]
    # nothing reaches the register, and section 9 keeps its owed card
    assert fd.register(report) == []
    assert sk.words(report, fd.SECTION) == ''
    assert _owed(report) == [
        '9. Deviations, Anomalies and Nonconformances: not yet written',
        ('9. Deviations, Anomalies and Nonconformances: D-1, A-1 offered from '
         'the run log, not yet accepted')]
    reading = _render(report, project)
    assert sk.OWED in _head(reading, 9) and fd.FROM_LOG not in reading
    assert 'D-1' not in _head(reading, 2)
    head = _head(_render(report, project, edit=True), 9)
    assert sk.OWED in head and fd.FROM_LOG in head
    assert ('<strong>Proposed D-1.</strong> Deviation <span class="finding-'
            'where">(' + html.escape(f'{extract.cited()}, entry '
                                     f'#{deviation.entry.id}')
            + ')</span>: a notch at 180 Hz not in the plan') in head
    drafts = fd.log_drafts(report)
    assert [draft['number'] for draft in drafts] == ['D-1', 'A-1']
    assert drafts[1]['source'].endswith(f'entry #{ids["anomaly"]}')
    # the disposition the log holds for A-1 is never offered
    assert 'use as is' not in json.dumps(drafts)
    # accepting is what enters the finding: its fact, the rest owed
    assert project.accept_logged_finding(name, 'D-1') == 'D-1'
    entry = fd.find(report, 'D-1')
    assert entry['fact'] == 'a notch at 180 Hz not in the plan'
    assert [entry[part] for part in ('cause', 'effect', 'approval')] == [
        '', '', '']
    assert entry['draft'] is None
    assert [draft['number'] for draft in fd.log_drafts(report)] == ['A-1']
    owed = _owed(report)
    assert ('9. Deviations, Anomalies and Nonconformances: D-1: cause, '
            'effect, approval not yet written') in owed
    assert ('9. Deviations, Anomalies and Nonconformances: A-1 offered from '
            'the run log, not yet accepted') in owed
    # accepted as edited
    assert project.accept_logged_finding(
        name, 'A-1', fact='A rattle at 300 Hz on the cable tie.') == 'A-1'
    assert fd.find(report, 'A-1')['fact'] == ('A rattle at 300 Hz on the '
                                              'cable tie.')
    assert fd.log_drafts(report) == []
    assert not any('offered from the run log' in item
                   for item in _owed(report))
    # offered again: a written fact is not offered over
    assert project.draft_findings_from_run_log(name)[0] == (
        "D-1: the register holds it with its fact written; the log's is not "
        'offered over it.')
    assert project.journal[-4:-1] == [
        f'project.draft_findings_from_run_log({name!r})',
        f"project.accept_logged_finding({name!r}, 'D-1')",
        (f"project.accept_logged_finding({name!r}, 'A-1', fact='A rattle at "
         "300 Hz on the cable tie.')")]


def test_the_badge_shows_in_the_draft_source_when_the_log_grew(tmp_path):
    project, name, _held, path, _ids = _logged(tmp_path)
    with writer(path) as log:
        log.append('note', author='AB', test=1, text='later')
    project.draft_findings_from_run_log(name)
    assert fd.log_drafts(project[name])[1]['source'].split(', entry')[
        0].endswith('[newer entries in the log]')


def test_an_accepted_finding_takes_the_logged_number_only_when_free(
        tmp_path):
    project, name, _held, _path, _ids = _logged(tmp_path)
    report = project[name]
    # D-1 issued and removed: the log's D-1 enters as D-2
    project.report_finding(name, kind='deviation')
    project.report_finding(name, 'D-1', remove=True)
    # A-1 entered by the author with its fact blank: it takes the fact
    assert project.report_finding(name, kind='anomaly',
                                  cause='Not determined.') == 'A-1'
    project.draft_findings_from_run_log(name)
    assert project.accept_logged_finding(name, 'D-1') == 'D-2'
    assert fd.find(report, 'D-2')['fact'] == ('a notch at 180 Hz not in the '
                                              'plan')
    assert project.accept_logged_finding(name, 'A-1') == 'A-1'
    assert fd.find(report, 'A-1')['fact'] == 'a rattle at 300 Hz'
    assert fd.find(report, 'A-1')['cause'] == 'Not determined.'
    assert [entry['number'] for entry in fd.register(report)] == ['D-2',
                                                                  'A-1']


def test_a_logged_finding_with_no_fact_is_not_offered(tmp_path):
    path = tmp_path / 'article.vdlog'
    with writer(path) as log:
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header=HEADER)
        model.append(log, 'anomaly_opened', author='AB', test=1, axis='Z',
                     run=2, fields={'owner': 'CD'})
        model.append(log, 'anomaly_opened', author='AB', test=1, axis='Z',
                     run=3, fields={'fact': 'a crack at the foot'})
    project = visualdynamics.Project()
    name = project.add('R', _report())
    project.attach_run_log(path)
    assert project.draft_findings_from_run_log(name) == [
        ('A-1: the log holds no fact for it, so nothing is offered; enter it '
         'in the register.'),
        ('A-2: offered as a draft from the run log, not in the register until '
         'accepted.')]
    # the run the log names fills the finding's run on acceptance
    assert project.accept_logged_finding(name, 'A-2') == 'A-2'
    assert fd.find(project[name], 'A-2')['run'] == '3'


def test_a_log_with_no_finding_offers_no_none(tmp_path):
    path = tmp_path / 'article.vdlog'
    with writer(path) as log:
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header=HEADER)
        model.append(log, 'level_row', author='AB', test=1, axis='Z', run=1,
                     fields={'level_db': 0.0})
    project = visualdynamics.Project()
    name = project.add('R', _report())
    project.attach_run_log(path)
    report = project[name]
    assert project.draft_findings_from_run_log(name) == [
        ('The run log holds no deviation or anomaly of test 1 by entry #2; '
         "nothing is offered, and whether there were none is the author's to "
         'say (set the section to None).')]
    assert report.blocks == [] and fd.register(report) == []
    assert _owed(report) == [
        '9. Deviations, Anomalies and Nonconformances: not yet written']


def test_offering_and_accepting_are_refused_by_name(tmp_path):
    project, name, held, path, _ids = _logged(tmp_path, 'skeleton_sections')
    with pytest.raises(ValueError, match='does not have the Findings register'):
        project.draft_findings_from_run_log(name)
    with pytest.raises(ValueError, match='does not have the Findings register'):
        project.accept_logged_finding(name, 'D-1')
    project.report_features(name, enable=['findings'])
    project.attach_run_log(path, test=2)
    with pytest.raises(ValueError, match=r"holds 2 run log extracts \('Run "
                       r"Log, Test 1', 'Run Log, Test 2'\); name the one"):
        project.draft_findings_from_run_log(name)
    with pytest.raises(TypeError, match="'R' is a Report, not a run log "
                       'extract'):
        project.draft_findings_from_run_log(name, run_log='R')
    with pytest.raises(ValueError, match='no finding D-1 offered from the run '
                       'log; waiting: none'):
        project.accept_logged_finding(name, 'D-1')
    project.draft_findings_from_run_log(name, run_log=held)
    report = project[name]
    with pytest.raises(ValueError, match='D-1: a blank fact is not accepted'):
        project.accept_logged_finding(name, 'D-1', fact='  ')
    with pytest.raises(ValueError, match='a discarded draft takes no fact'):
        project.accept_logged_finding(name, 'D-1', fact='x', discard=True)
    with pytest.raises(ValueError, match='waiting: D-1, A-1'):
        project.accept_logged_finding(name, 'D-9')
    assert fd.register(report) == []
    assert project.accept_logged_finding(name, 'D-1', discard=True) is None
    assert [draft['number'] for draft in fd.log_drafts(report)] == ['A-1']
    assert fd.register(report) == []
