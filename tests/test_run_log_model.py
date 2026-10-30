"""The run log read as tests, axes and runs: states and the roll, holds
raised and cleared, owed lists, derived views, and the projection to
full level against hand values.

Every log is written under `tmp_path` with a fixed login and machine
name, on a clock that moves one minute per entry, so event order and
the late mark are pinned by value.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from visualdynamics.core import run_log_model as model_module
from visualdynamics.core.report import PROJECT_TYPES
from visualdynamics.core.run_log import (
    HEADER_FIELDS,
    FileRef,
    RunLog,
    RunLogError,
    band_hz,
)
from visualdynamics.core.run_log_model import (
    AXIS_CHANGE,
    BEFORE_LOW_LEVEL,
    DRIVE_WINDOW,
    HOLD_RULES,
    RELEASE,
    TEST_KINDS,
    RunLogModel,
    append,
    change_header,
    close_test,
    confirm,
    full_level_factor,
    open_test,
    parse_counter,
    project,
    release,
    roll,
)

WHO = ('tester', 'bench')
ZONE = timezone(timedelta(hours=-6))
START = datetime(2026, 10, 30, 8, 0, tzinfo=ZONE)


class Ticking:
    """A clock that moves one minute every time it is read."""

    def __init__(self, now=START):
        self.now = now

    def __call__(self):
        self.now += timedelta(minutes=1)
        return self.now


def new_log(tmp_path, name='a.vdlog'):
    return RunLog.create(tmp_path / name, clock=Ticking(), who=WHO)


def at(minutes):
    return START + timedelta(minutes=minutes)


def rules(holds):
    return [hold.rule for hold in holds]


HEADER = {'title': 'Random, Z', 'article': 'Unit 7', 'configuration': 'flight',
          'steps_db': [-12, -6, -3, 0], 'sigma': 3.0,
          'rated_current_peak': 800.0, 'rail_voltage': 250.0,
          'duration_s': 600.0, 'duration_minus_pct': 0.0,
          'duration_plus_pct': 10.0}


def random_test(log, header=None):
    return open_test(log, author='AB', test_kind='Random Vibration',
                     header={**HEADER, **(header or {})})


# ---- the projection ---------------------------------------------------------


@pytest.mark.parametrize(('level', 'hand'), [
    (-12.0, 3.98107170553497),     # 10**0.6
    (-6.0, 1.99526231496888),      # 10**0.3
    (-3.0, 1.41253754462275),      # 10**0.15
    (0.0, 1.0),
])
def test_the_factor_to_full_level_is_exactly_ten_to_minus_l_over_twenty(
        level, hand):
    assert full_level_factor(level) == pytest.approx(hand, rel=1e-13)


def test_63_amps_at_minus_12_projects_to_250_8_not_252():
    projection = project(-12.0, current=63.0, voltage=20.0,
                         drive_fraction=0.12, sigma=3.0,
                         rated_current_peak=800.0, rail_voltage=250.0)
    assert projection.factor == pytest.approx(3.98107170553497, rel=1e-13)
    peak = projection.check('peak current')
    # 63 x 3.98107 = 250.8075 A rms, x 3 sigma = 752.4226 A peak
    assert peak.at_full == pytest.approx(752.422552346, rel=1e-11)
    assert peak.verdict == 'within'
    assert '752.4 A at full level, within the peak current rating of 800 A' \
        in peak.message
    voltage = projection.check('peak voltage')
    assert voltage.at_full == pytest.approx(238.864302332, rel=1e-11)
    assert voltage.verdict == 'within'
    drive = projection.check('drive fraction')
    assert drive.at_full == pytest.approx(0.477728604664, rel=1e-11)
    assert drive.verdict == 'within' and 'the default window' in drive.message
    assert projection.verdict == 'within'


def test_the_projection_at_minus_6_and_minus_3_against_hand_values():
    six = project(-6.0, current=100.0, sigma=2.0, rated_current_peak=300.0)
    assert six.check('peak current').at_full == pytest.approx(
        399.052462993776, rel=1e-12)              # 100 x 1.99526 x 2
    three = project(-3.0, voltage=40.0, sigma=3.0, rail_voltage=150.0)
    assert three.check('peak voltage').at_full == pytest.approx(
        169.504505354730, rel=1e-12)              # 40 x 1.41254 x 3
    assert three.check('peak voltage').verdict == 'over'
    assert 'above the rail of 150 V' in three.check('peak voltage').message


def test_a_reading_exactly_at_its_limit_is_within_and_just_past_is_over():
    at_limit = project(0.0, current=100.0, voltage=50.0, sigma=2.0,
                       rated_current_peak=200.0, rail_voltage=100.0)
    assert at_limit.check('peak current').verdict == 'within'
    assert at_limit.check('peak voltage').verdict == 'within'
    past = project(0.0, current=100.001, voltage=50.001, sigma=2.0,
                   rated_current_peak=200.0, rail_voltage=100.0)
    assert past.check('peak current').verdict == 'over'
    assert past.check('peak voltage').verdict == 'over'
    assert past.verdict == 'outside'


def test_the_drive_fraction_is_judged_against_its_window_at_both_edges():
    low, high = DRIVE_WINDOW
    assert project(0.0, drive_fraction=low).check(
        'drive fraction').verdict == 'within'
    assert project(0.0, drive_fraction=high).check(
        'drive fraction').verdict == 'within'
    under = project(-12.0, drive_fraction=0.05).check('drive fraction')
    assert under.verdict == 'under' and 'below the default window' in \
        under.message
    over = project(-12.0, drive_fraction=0.2).check('drive fraction')
    assert over.verdict == 'over' and over.at_full == pytest.approx(
        0.796214341107, rel=1e-11)
    given = project(-6.0, drive_fraction=0.2, drive_window=(0.2, 0.5))
    assert given.check('drive fraction').verdict == 'within'
    assert 'within the window 0.2 to 0.5' in given.check(
        'drive fraction').message


def test_the_continuous_rating_is_checked_only_when_the_header_gives_it():
    assert [check.name for check in project(0.0).checks] == [
        'drive fraction', 'peak current', 'peak voltage']
    sine = project(-6.0, current=50.0, rated_current_rms=99.0)
    rms = sine.check('rms current')
    assert rms.at_full == pytest.approx(99.763115748444, rel=1e-12)
    assert rms.verdict == 'over'
    with pytest.raises(KeyError, match="no 'rms' check"):
        sine.check('rms')


def test_what_cannot_be_judged_says_why_and_is_never_a_pass():
    projection = project(-12.0, current=63.0)
    peak = projection.check('peak current')
    assert not peak.judged and peak.at_full is None
    assert peak.message == ('not judged: the header gives no sigma; the '
                            'header gives no peak current rating')
    assert projection.check('peak voltage').message == (
        'not judged: no voltage was read; the header gives no sigma; the '
        'header gives no rail')
    assert projection.check('drive fraction').message == (
        'not judged: no drive fraction was read')
    assert projection.verdict == 'not judged'
    # judged within everywhere it could be judged is still not a pass
    partly = project(-12.0, current=63.0, sigma=3.0, rated_current_peak=800.0)
    assert partly.check('peak current').verdict == 'within'
    assert partly.verdict == 'not judged'


def test_an_all_zero_reading_is_not_judged_rather_than_passed():
    zero = project(-12.0, current=0.0, voltage=0.0, drive_fraction=0.0,
                   sigma=3.0, rated_current_peak=800.0, rail_voltage=250.0)
    assert {check.verdict for check in zero.checks} == {'not judged'}
    assert 'the current reads zero' in zero.check('peak current').message
    assert 'the voltage reads zero' in zero.check('peak voltage').message
    assert 'the drive fraction reads zero' in zero.check(
        'drive fraction').message
    assert zero.verdict == 'not judged'


@pytest.mark.parametrize(('kwargs', 'message'), [
    ({'level_db': 3.0}, r'\+3 dB is above full level'),
    ({'level_db': math.nan}, 'not a finite number'),
    ({'level_db': True}, 'dB relative to full level'),
    ({'level_db': '-12'}, 'dB relative to full level'),
    ({'level_db': -12.0, 'current': math.nan}, 'a current is not a finite'),
    ({'level_db': -12.0, 'voltage': -1.0}, 'a voltage of -1 is negative'),
    ({'level_db': -12.0, 'drive_fraction': 1.5},
     'a drive fraction of 1.5 is above 1'),
    ({'level_db': -12.0, 'current': 'a lot'}, 'a current is a number'),
    ({'level_db': -12.0, 'sigma': 0.0}, 'a sigma of zero is not a rating'),
    ({'level_db': -12.0, 'rail_voltage': math.inf}, 'not a finite number'),
    ({'level_db': -12.0, 'drive_window': (0.7, 0.3)}, 'low then high'),
    ({'level_db': -12.0, 'drive_window': (0.2,)}, 'low then high'),
    ({'level_db': -12.0, 'drive_window': (0.0, 1.5)}, 'low then high'),
])
def test_what_cannot_be_projected_one_way_is_refused_naming_it(kwargs,
                                                                 message):
    level = kwargs.pop('level_db')
    with pytest.raises(RunLogError, match=message):
        project(level, **kwargs)


# ---- counters and bands ------------------------------------------------------


@pytest.mark.parametrize(('text', 'seconds'), [
    ('750', 750.0), ('750 s', 750.0), ('12:30', 750.0), ('1:02:30', 3750.0),
    ('0:59.5', 59.5), (90, 90.0), ('10:00', 600.0),
])
def test_a_counter_is_read_as_seconds(text, seconds):
    assert parse_counter(text) == seconds


@pytest.mark.parametrize('text', ['12:60', 'ten', '1:2:3:4', '', '1.5:00',
                                  '-5', True, None])
def test_a_counter_that_is_not_a_time_is_refused(text):
    with pytest.raises(RunLogError, match='counter'):
        parse_counter(text)


def test_clock_text_reads_as_the_counter_does():
    assert model_module.clock_text(750) == '12:30'
    assert model_module.clock_text(3750) == '1:02:30'
    assert model_module.clock_text(59.6) == '1:00'


@pytest.mark.parametrize(('value', 'band'), [
    ('180-220 Hz', (180.0, 220.0)), ('180 to 220', (180.0, 220.0)),
    ('1.2-2 kHz', (1200.0, 2000.0)), ('200 Hz', (200.0, 200.0)),
    ([180, 220], (180.0, 220.0)), ('180 – 220 hz', (180.0, 220.0)),
])
def test_a_band_is_read_from_text_or_two_frequencies(value, band):
    assert band_hz(value) == band


@pytest.mark.parametrize(('value', 'message'), [
    ('around 200', 'not read as a band'), ('220-180', 'low edge 220 Hz'),
    ([1, 2, 3], 'two frequencies'), ('1..2-3', 'not read as a band'),
    (None, 'two frequencies'),
])
def test_a_band_that_cannot_be_read_one_way_is_refused(value, message):
    with pytest.raises(RunLogError, match=message):
        band_hz(value)


# ---- what the log file now carries -------------------------------------------


def test_a_header_change_holds_a_value_of_its_fields_type(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        entry = log.append('header_change', author='AB', test=1,
                           fields={'field': 'steps_db', 'value': [-9, 0],
                                   'was': [-12, -6, -3, 0]})
        assert entry.fields['value'] == [-9, 0]
        corrected = log.correct(entry, author='AB',
                                fields={'value': [-10, 0]})
        assert corrected.fields == {'value': [-10, 0]}
        cases = [
            ({'field': 'steps_db', 'value': 'loud'}, 'list of finite numbers'),
            ({'field': 'sigma', 'value': [3]}, 'value is a number'),
            ({'field': 'colour', 'value': 'red'},
             "'colour' is not a field of a test's header"),
        ]
        for fields, message in cases:
            with pytest.raises(RunLogError, match=message):
                log.append('header_change', author='AB', test=1, fields=fields)
        with pytest.raises(RunLogError, match='field is never changed'):
            log.correct(entry, author='AB', fields={'field': 'sigma'})
        with pytest.raises(RunLogError, match='list of finite numbers'):
            log.correct(entry, author='AB', fields={'value': 'loud'})


def test_the_new_vocabularies_and_types_are_refused_naming_the_field(
        tmp_path):
    log = new_log(tmp_path)
    cases = [
        (('level_row', {'answer': 'fine'}), "'fine' is not an answer"),
        (('level_row', {'channel_answers': {'4': 'odd'}}),
         "channel 4: 'odd' is not an answer"),
        (('level_row', {'channel_answers': ['4']}), 'answer by channel name'),
        (('level_row', {'channel_answers': {' ': 'explained'}}),
         'answer by channel name'),
        (('level_row', {'channel_answers': {'4': 3}}),
         'answer by channel name'),
        (('level_row', {'channel_answers': {4: 'explained'}}),
         'answer by channel name'),
        (('setting_change', {'setting': 'gain', 'governs': 'mood'}),
         "'mood' is not what a setting governs"),
        (('sweep', {'direction': 'sideways'}),
         "'sideways' is not a sweep direction"),
        (('test_opened', {'notches_planned': 'one'}),
         'list of planned notches'),
        (('test_opened', {'notches_planned': [{'axis': 'Z'}]}),
         'notch 1 is a mapping with its band'),
        (('test_opened', {'notches_planned': [{'band': 'wide'}]}),
         "notch 1: band 'wide' is not read as a band"),
        (('test_opened', {'notches_planned': [{'band': '1-2',
                                               'width': 3}]}),
         'has no width'),
        (('test_opened', {'notches_planned': [{'band': '1-2',
                                               'depth_db': math.nan}]}),
         'depth_db is a finite number'),
        (('test_opened', {'steps_db': [-12, math.inf]}),
         'list of finite numbers'),
        (('shot', {'pulses': 0}), 'whole number from 1'),
    ]
    for (kind, fields), message in cases:
        with pytest.raises(RunLogError, match=message):
            log.append(kind, author='AB', test=1, fields=fields)
    notches = log.append('test_opened', author='AB', test=1, fields={
        'notches_planned': [{'band': [180, 220], 'axis': ' Z ',
                             'depth_db': -6}, {'band': '1-2 kHz'}]})
    assert notches.fields['notches_planned'] == [
        {'axis': 'Z', 'band': [180.0, 220.0], 'depth_db': -6},
        {'band': '1-2 kHz'}]
    log.close()


def test_every_header_field_has_a_type_and_test_kinds_extend_project_types():
    from visualdynamics.core.run_log import FIELD_TYPES, KINDS
    assert set(HEADER_FIELDS) <= set(FIELD_TYPES)
    assert KINDS['test_opened'].optional == (*HEADER_FIELDS, 'carried')
    assert TEST_KINDS == (*PROJECT_TYPES, 'Functional', 'Other')


# ---- tests, states and the roll ----------------------------------------------


def test_an_empty_log_has_no_tests_holds_owed_lines_or_views(tmp_path):
    with new_log(tmp_path) as log:
        model = RunLogModel(log)
        assert model.tests() == [] and model.current is None
        assert model.holds() == [] and model.owed() == []
        assert model.runs() == [] and model.credited() == []
        assert model.findings() == [] and model.chronology() == []
        assert model.files() == [] and model.late() == []
        assert model.interruptions() == []
        assert model.next_test() == 1
        with pytest.raises(RunLogError, match='the log has no test 1'):
            model.test(1)
        with pytest.raises(RunLogError, match='selected by its number'):
            model.holds(test=0)


def test_a_test_with_no_runs_is_open_with_its_setup_owed(tmp_path):
    with new_log(tmp_path) as log:
        opened = random_test(log, {'title': ''})
        model = RunLogModel(log)
        test = model.test(1)
        assert (test.state, test.kind, test.shape) == (
            'open', 'Random Vibration', 'levels')
        assert test.title == 'Test 1'                 # a blank title is left out
        assert test.opened == opened and test.runs == ()
        assert test.header.get('sigma') == 3.0
        assert test.header.set_by['sigma'] == opened.id
        assert model.current is test and model.next_run(1) == 1
        assert model.credited(1) == [] and model.holds(1) == []
        owed = model.owed(1)
        assert [found.name for found in owed] == ['before low level', 'clock']
        assert [line.key for line in owed[0].owed] == [
            key for key, _ in BEFORE_LOW_LEVEL]


def test_a_test_opened_by_hand_is_refused_while_another_is_open(tmp_path):
    with new_log(tmp_path) as log:
        with pytest.raises(RunLogError, match="'Vibe' is not a kind of test"):
            open_test(log, author='AB', test_kind='Vibe')
        with pytest.raises(RunLogError, match='carries nothing'):
            open_test(log, author='AB', test_kind='Shock',
                      header={'carried': ['title']})
        random_test(log)
        with pytest.raises(RunLogError, match='test 1 is open; roll'):
            open_test(log, author='AB', test_kind='Shock')
        close_test(log, 1, author='AB', reason='done')
        with pytest.raises(RunLogError, match='already closed to runs'):
            close_test(log, 1, author='AB')
        assert open_test(log, author='AB', test_kind='Shock').test == 2


def test_the_roll_carries_the_header_marked_until_confirmed(tmp_path):
    with new_log(tmp_path) as log:
        with pytest.raises(RunLogError, match='no test is open to roll'):
            roll(log, author='AB')
        random_test(log)
        closing, opening = roll(log, author='AB', test_kind='Shock',
                                header={'title': 'Release shock',
                                        'pulse_budget': 12})
        assert closing.fields == {'reason': 'rolled to test 2'}
        assert (closing.test, opening.test) == (1, 2)
        assert closing.event_time == opening.event_time
        model = RunLogModel(log)
        first, second = model.test(1), model.test(2)
        assert first.state == 'closed to runs' and first.closed == closing
        assert second.state == 'open' and second.kind == 'Shock'
        assert second.header.get('title') == 'Release shock'
        assert second.header.get('sigma') == 3.0
        assert set(second.header.carried) == set(HEADER) - {'title'}
        assert not second.header.is_carried('test_kind')
        assert not second.header.is_carried('pulse_budget')
        assert [found.name for found in model.owed(1)] == [
            'before low level', 'release', 'clock']
        confirm(log, 2, ['sigma'], author='AB')
        change_header(log, 2, 'article', 'Unit 8', author='AB')
        header = RunLogModel(log).test(2).header
        assert not header.is_carried('sigma') and header.get('sigma') == 3.0
        assert not header.is_carried('article')
        assert header.get('article') == 'Unit 8'
        with pytest.raises(RunLogError, match="'sigma' is not carried"):
            confirm(log, 2, ['sigma'], author='AB')
        rest = confirm(log, 2, author='AB')
        assert {entry.fields['field'] for entry in rest} == set(
            HEADER) - {'title', 'sigma', 'article'}
        assert RunLogModel(log).test(2).header.carried == ()
        # a confirmation, once running, is not a change to the test
        assert RunLogModel(log).holds() == []


def test_the_roll_refuses_a_bad_header_before_writing_anything(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        count = len(log.entries)
        with pytest.raises(RunLogError, match='sigma is a number'):
            roll(log, author='AB', header={'sigma': 'three'})
        with pytest.raises(RunLogError, match='worked out by the roll'):
            roll(log, author='AB', header={'carried': []})
        with pytest.raises(RunLogError, match="'Vibe' is not a kind"):
            roll(log, author='AB', test_kind='Vibe')
        assert len(log.entries) == count


def test_a_closed_test_takes_findings_until_released_but_no_runs(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        append(log, 'level_row', author='AB', test=1, axis='Z', run=1,
               fields={'level_db': 0.0})
        with pytest.raises(RunLogError, match='open to runs; close it'):
            release(log, 1, author='AB')
        roll(log, author='AB', test_kind='Shock')
        with pytest.raises(RunLogError, match='test 1 is closed to runs; it '
                           'still takes findings'):
            append(log, 'level_row', author='AB', test=1, run=2)
        anomaly = append(log, 'anomaly_opened', author='AB', test=1,
                         fields={'fact': 'bracket channel noisy',
                                 'owner': 'CD'})
        append(log, 'post_test_step', author='AB', test=1,
               fields={'step': 'Data check'})
        model = RunLogModel(log)
        assert model.open_anomalies(1)[0].number == anomaly.fields['number']
        release_list = model.checklist(1, 'release')
        assert release_list.lines[2].done is not None
        assert release_list.lines[2].done.author == 'AB'
        released = release(log, 1, author='AB', text='released: CD, EF')
        assert RunLogModel(log).test(1).state == 'released'
        assert RunLogModel(log).test(1).released == released
        with pytest.raises(RunLogError, match='already released'):
            release(log, 1, author='AB')
        for kind in ('deviation', 'anomaly_opened', 'level_row'):
            with pytest.raises(RunLogError, match='test 1 is released'):
                append(log, kind, author='AB', test=1)
        with pytest.raises(RunLogError, match='test 1 is released'):
            change_header(log, 1, 'title', 'x', author='AB')
        # an open anomaly goes to the board, which disposes of it later
        append(log, 'anomaly_dispositioned', author='AB', test=1,
               fields={'number': anomaly.fields['number'],
                       'disposition': 'sensor rebonded', 'by': 'board'})
        append(log, 'post_test_step', author='AB', test=1,
               fields={'step': 'flash report'})
        model = RunLogModel(log)
        assert model.open_anomalies() == []
        assert model.test(1).problems == ()


def test_entries_that_do_not_fit_a_tests_state_are_named(tmp_path):
    with new_log(tmp_path) as log:
        with pytest.raises(RunLogError, match='test 1 has not been opened'):
            append(log, 'note', author='AB', test=1)
        with pytest.raises(RunLogError, match='belongs to a run of a test'):
            append(log, 'level_row', author='AB')
        with pytest.raises(RunLogError, match='opened with open_test or roll'):
            append(log, 'test_opened', author='AB', test=1)
        assert append(log, 'note', author='AB').test is None
        log.append('note', author='AB', test=3)              # by hand
        log.append('test_opened', author='AB', test=1,
                   fields={'test_kind': 'Vibe'})
        log.append('test_opened', author='AB', test=1)
        log.append('test_closed', author='AB', test=1)
        log.append('level_row', author='AB', test=1, run=1)
        model = RunLogModel(log)
        assert model.test(3).problems == (
            'test 3 has entries but was never opened',)
        # a test never opened sorts by its first entry
        assert [found.number for found in model.tests()] == [3, 1]
        problems = model.test(1).problems
        assert problems[0] == '#4 opens test 1 again; the first opening, #3, ' \
                              'stands'
        assert "test 1's kind 'Vibe' is not one of" in problems[1]
        assert problems[2] == ('#6 (Level) was written after test 1 was '
                               'closed to runs')
        assert model.test(1).kind == 'Other'


def test_a_test_is_closed_to_runs_by_whichever_comes_first(tmp_path):
    with new_log(tmp_path) as log:
        log.append('test_opened', author='AB', test=1,
                   fields={'test_kind': 'Shock'})
        log.append('test_opened', author='AB', test=2,
                   fields={'test_kind': 'Shock'})
        log.append('test_closed', author='AB', test=1)
        log.append('test_closed', author='AB', test=2)
        log.append('test_opened', author='AB', test=3,
                   fields={'test_kind': 'Shock'})
        model = RunLogModel(log)
        assert model.test(1).closed.id == 2      # the next test's opening
        assert model.test(2).closed.id == 4      # its own closing, first
        assert model.test(3).state == 'open'


def test_a_hand_edited_header_change_or_notch_list_is_read_past(tmp_path):
    path = tmp_path / 'a.vdlog'
    with new_log(tmp_path) as log:
        random_test(log, {'notches_planned': [{'band': '100-110 Hz'},
                                              {'band': '180-220 Hz'}]})
        log.append('header_change', author='AB', test=1,
                   fields={'field': 'title', 'value': 'kept'})
        log.append('setup_step', author='AB', test=1, axis='Z',
                   fields={'step': 'surveys'})
        log.append('notch', author='AB', test=1, axis='Z', run=1,
                   fields={'band': '190-200 Hz'})
    text = path.read_text(encoding='utf-8')
    text = text.replace('"100-110 Hz"', '"wide"')
    text = text.replace('"field": "title"', '"field": "colour"')
    text = text.replace('"step": "surveys"', '"step": 5')
    path.write_bytes(text.encode('utf-8'))
    model = RunLogModel(RunLog.read(path))
    assert model.test(1).header.get('title') == 'Random, Z'
    assert 'colour' not in model.test(1).header.values
    assert model.holds() == []                    # the second notch plans it
    assert model.checklist(1, 'before low level').lines[5].owed


def test_a_roll_from_a_test_of_no_known_kind_asks_for_its_kind(tmp_path):
    with new_log(tmp_path) as log:
        log.append('test_opened', author='AB', test=1,
                   fields={'test_kind': 'Vibe'})
        with pytest.raises(RunLogError, match="'Vibe' is not a kind of test"):
            roll(log, author='AB')
        assert roll(log, author='AB', test_kind='Shock')[1].test == 2


def test_a_test_opened_after_release_entries_flags_late_findings(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        roll(log, author='AB', test_kind='Shock')
        release(log, 1, author='AB')
        log.append('deviation', author='AB', test=1)          # by hand
        assert RunLogModel(log).test(1).problems == (
            '#5 (Deviation) was written after test 1 was released',)


# ---- runs by kind ------------------------------------------------------------


def test_random_runs_hold_level_rows_ending_at_full_level(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        for level in (-12.0, -6.0, -3.0, 0.0):
            log.append('level_row', author='AB', test=1, axis='Z', run=1,
                       fields={'level_db': level, 'go': True})
        log.append('run_stop', author='AB', test=1, run=1,
                   fields={'counter': '10:00'})
        log.append('level_row', author='AB', test=1, axis='Z', run=2,
                   fields={'level_db': -12.0})
        model = RunLogModel(log)
        first, second = model.runs(1)
        assert [row.fields['level_db'] for row in first.levels] == [
            -12.0, -6.0, -3.0, 0.0]
        assert (first.axis, first.outcome, first.counter) == (
            'Z', 'completed', '10:00')
        assert first.reached_full and not second.reached_full
        assert second.outcome == 'running' and second.stop is None
        assert str(first) == 'test 1, run 1, axis Z: completed'
        assert model.test(1).run(2) == second
        with pytest.raises(KeyError, match='no run 3'):
            model.test(1).run(3)
        assert model.next_run(1) == 3


def test_sine_runs_hold_sweeps_and_survey_runs_a_result(tmp_path):
    with new_log(tmp_path) as log:
        open_test(log, author='AB', test_kind='Sine Sweep')
        for n, direction in ((1, 'up'), (2, 'down')):
            log.append('sweep', author='AB', test=1, axis='X', run=1,
                       fields={'direction': direction, 'n': n, 'of': 2})
        roll(log, author='AB', test_kind='Modal Test')
        log.append('run_start', author='AB', test=2, axis='X', run=1)
        log.append('run_stop', author='AB', test=2, run=1,
                   fields={'result': 'first mode 212 Hz', 'by': 'CD'})
        model = RunLogModel(log)
        sweeps = model.test(1).runs[0].sweeps
        assert [(s.fields['direction'], s.fields['n']) for s in sweeps] == [
            ('up', 1), ('down', 2)]
        survey = model.test(2).runs[0]
        assert model.test(2).shape == 'result'
        assert (survey.result, survey.by) == ('first mode 212 Hz', 'CD')
        assert survey.start is not None
        assert len(model.runs()) == 2


def test_shock_shots_count_pulses_against_the_budget_with_their_checks(
        tmp_path):
    with new_log(tmp_path) as log:
        open_test(log, author='AB', test_kind='Shock',
                  header={'pulse_budget': 4})
        full = {'in_tolerance': True, 'within_stroke': True,
                'peak_current': 410.0, 'peak_voltage': 120.0,
                'prediction': 'next level within the rail'}
        log.append('shot', author='AB', test=1, axis='Z', run=1,
                   fields={'polarity': '+', 'n': 1, **full})
        partial = log.append('shot', author='AB', test=1, axis='Z', run=1,
                             fields={'polarity': '-', 'n': 2,
                                     'in_tolerance': False,
                                     'peak_current': 400.0})
        log.append('shot', author='AB', test=1, axis='X', run=2,
                   fields={'pulses': 2})
        model = RunLogModel(log)
        count = model.pulses(1)
        assert (count.count, count.budget, count.verdict) == (4, 4, 'within')
        assert dict(count.by_axis) == {'Z': 2, 'X': 2}
        assert count.message == '4 of 4 pulses, 0 left'
        checks = model.shot_checks(1)
        assert [line.owed for line in checks[0].lines] == [False] * 4
        assert [line.key for line in checks[1].owed] == [
            'stroke', 'amplifier', 'prediction']     # one peak is not both
        assert checks[1].lines[0].done == partial
        assert model.test(1).runs[0].shots[1] == partial
        assert model.test(1).shape == 'shots'
        log.append('shot', author='AB', test=1, axis='X', run=2)
        over = RunLogModel(log).pulses(1)
        assert over.verdict == 'over'
        assert over.message == '5 pulses, over the budget of 4 by 1'
        roll(log, author='AB', header={'pulse_budget': None})
        assert RunLogModel(log).pulses(2).message == (
            '0 pulses; not judged: the header gives no pulse budget')


# ---- holds -------------------------------------------------------------------


def test_every_hold_rule_is_named():
    assert set(HOLD_RULES) == {'abort-cause', 'over-test', 'restart-cause',
                               'not-explained', 'unplanned-notch',
                               'plan-change', 'hold-point'}


def test_an_abort_holds_until_its_cause_and_over_test_answer(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        abort = log.append('abort', author='AB', test=1, axis='Z', run=1,
                           fields={'stop': 'amplifier trip'})
        holds = RunLogModel(log).holds()
        assert rules(holds) == ['abort-cause', 'over-test']
        assert str(holds[0]).startswith('HOLD (test 1, #2): the amplifier '
                                        'trip at #2 has no cause written')
        log.append('cause_found', author='AB', test=1,
                   fields={'cause': 'loose drive cable'})
        assert rules(RunLogModel(log).holds()) == ['over-test']
        log.correct(abort, author='AB',
                    fields={'over_test': 'no: control and monitor below the '
                            'abort band', 'records_read': 'drive, control'})
        assert RunLogModel(log).holds() == []


def test_a_cause_answers_the_abort_it_names_or_the_latest_open_one(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        first = log.append('abort', author='AB', test=1, run=1,
                           fields={'stop': 'stop word', 'over_test': 'no'})
        second = log.append('abort', author='AB', test=1, run=2,
                            fields={'stop': 'power loss', 'over_test': 'no'})
        log.append('cause_found', author='AB', test=1,
                   fields={'cause': 'mains dropped'})
        holds = RunLogModel(log).holds()
        assert [(hold.rule, hold.entry.id) for hold in holds] == [
            ('abort-cause', first.id)]
        log.append('cause_found', author='AB', fields={
            'cause': 'stop called on a sound', 'refers_to': first.id})
        assert RunLogModel(log).holds() == []
        assert second.id not in [h.entry.id for h in RunLogModel(log).holds()]


def test_a_restart_holds_until_its_cause_is_written(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        restart = log.append('restart', author='AB', test=1, run=1,
                             fields={'resume': True})
        assert rules(RunLogModel(log).holds()) == ['restart-cause']
        log.correct(restart, author='AB', fields={'cause': 'cable reseated'})
        assert RunLogModel(log).holds() == []


def test_not_explained_holds_at_that_level_until_a_later_answer(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -12.0, 'answer': 'as predicted',
                           'channel_answers': {'7': 'not explained'}})
        holds = RunLogModel(log).holds()
        assert rules(holds) == ['not-explained']
        assert '"not explained" (channel 7)' in holds[0].message
        log.append('level_row', author='AB', test=1, axis='X', run=2,
                   fields={'level_db': -12.0, 'answer': 'explained'})
        assert rules(RunLogModel(log).holds()) == ['not-explained']
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -12.0})              # no answer yet
        assert rules(RunLogModel(log).holds()) == ['not-explained']
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -12.0, 'answer': 'explained'})
        assert RunLogModel(log).holds() == []


def test_a_notch_not_in_the_planned_list_holds_until_a_deviation(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log, {'notches_planned': [
            {'axis': 'Z', 'band': '180-220 Hz', 'depth_db': -6}]})
        planned = log.append('notch', author='AB', test=1, axis='Z', run=1,
                             fields={'band': '190-210 Hz', 'depth_db': -4})
        edge = log.append('notch', author='AB', test=1, axis='Z', run=1,
                          fields={'band': '180-220', 'depth_db': -6.04})
        model = RunLogModel(log)
        assert model.notch_planned(planned) == (True,
                                                'inside planned notch 1.')
        assert model.notch_planned(edge)[0]
        cases = [
            ({'band': '190-210 Hz', 'depth_db': -4}, 'X',
             '190 to 210 Hz on axis X is in no planned notch'),
            ({'band': '190-210 Hz', 'depth_db': -8}, 'Z',
             'deeper than its 6 dB'),
            ({'band': '175-210 Hz', 'depth_db': -3}, 'Z',
             'in no planned notch'),
            ({'band': '190-210 Hz'}, 'Z', ('which has a depth, but it gives '
             'none')),
            ({'depth_db': -3}, 'Z', 'gives no band'),
            ({'band': 'low', 'depth_db': -3}, 'Z', 'not read as a band'),
            ({'band': '300 Hz', 'depth_db': -3, 'planned': True}, 'Z',
             'marked planned, but the header\'s list decides'),
        ]
        unplanned = []
        for fields, axis, why in cases:
            entry = log.append('notch', author='AB', test=1, axis=axis, run=1,
                               fields=fields)
            assert why in RunLogModel(log).notch_planned(entry)[1]
            unplanned.append(entry)
        holds = RunLogModel(log).holds()
        assert rules(holds) == ['unplanned-notch'] * len(cases)
        assert [hold.entry for hold in holds] == unplanned
        for entry in unplanned:
            log.append('deviation', author='AB', test=1,
                       fields={'fact': 'notch added', 'agreed_by': 'CD',
                               'refers_to': entry.id})
        assert RunLogModel(log).holds() == []


def test_a_planned_notch_is_judged_against_the_header_as_it_stood(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -12.0})
        notch = log.append('notch', author='AB', test=1, axis='Z', run=1,
                           fields={'band': '200 Hz', 'depth_db': -3})
        assert RunLogModel(log).notch_planned(notch) == (
            False, 'the header plans no notches.')
        change = change_header(log, 1, 'notches_planned', [{'band': '200 Hz'}],
                               author='AB')
        holds = RunLogModel(log).holds()
        # adding the notch to the plan afterwards does not plan it, and is
        # itself a change to the test
        assert [(hold.rule, hold.entry.id) for hold in holds] == [
            ('unplanned-notch', notch.id), ('plan-change', change.id)]
        later = log.append('notch', author='AB', test=1, axis='X', run=2,
                           fields={'band': '200 Hz'})
        assert RunLogModel(log).notch_planned(later)[0]   # no axis: every axis


def test_a_limit_or_tolerance_change_once_running_holds_until_a_deviation(
        tmp_path):
    with new_log(tmp_path) as log:
        random_test(log, {'limits': 'response 4 g rms'})
        typed = change_header(log, 1, 'limits', 'response 5 g rms',
                              author='AB')                # before running
        log.append('run_start', author='AB', test=1, axis='Z', run=1)
        changed = change_header(log, 1, 'tolerance', '+/-3 dB', author='AB')
        same = log.append('header_change', author='AB', test=1, fields={
            'field': 'limits', 'value': 'response 5 g rms'})
        other = change_header(log, 1, 'title', 'Random, Z axis', author='AB')
        abort_band = log.append('setting_change', author='AB', test=1,
                                fields={'setting': 'band 3', 'value': 9.0,
                                        'governs': 'abort band'})
        worded = log.append('setting_change', author='AB', test=1,
                            fields={'setting': 'Notch limit, channel 7'})
        log.append('setting_change', author='AB', test=1,
                   fields={'setting': 'limit 2', 'governs': 'other'})
        log.append('setting_change', author='AB', test=1,
                   fields={'setting': 'display scale'})
        holds = RunLogModel(log).holds()
        assert [(hold.rule, hold.entry.id) for hold in holds] == [
            ('plan-change', changed.id), ('plan-change', abort_band.id),
            ('plan-change', worded.id)]
        assert "the change of the header's tolerance" in holds[0].message
        assert typed.id not in [h.entry.id for h in holds]
        assert same.id not in [h.entry.id for h in holds]
        assert other.id not in [h.entry.id for h in holds]
        for hold in holds:
            log.append('deviation', author='AB', test=1,
                       fields={'refers_to': hold.entry.id})
        assert RunLogModel(log).holds() == []


def test_a_go_at_a_hold_point_waits_for_the_customers_agreement(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log, {'hold_points_db': [-3.0]})
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -6.0, 'go': True})
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -3.0, 'go': False})
        go = log.append('level_row', author='AB', test=1, axis='Z', run=1,
                        fields={'level_db': -3.02, 'go': True})
        log.append('level_row', author='AB', test=1, axis='Z', run=2,
                   fields={'level_db': -3.0, 'go': True,
                           'agreed_by': 'customer, GH'})
        holds = RunLogModel(log).holds()
        assert [(hold.rule, hold.entry.id) for hold in holds] == [
            ('hold-point', go.id)]
        assert 'hold point -3.02 dB' in holds[0].message
        log.correct(go, author='AB', fields={'agreed_by': 'customer, GH'})
        assert RunLogModel(log).holds() == []


def test_holds_select_by_test(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('restart', author='AB', test=1, run=1)
        roll(log, author='AB')
        log.append('restart', author='AB', test=2, run=1)
        model = RunLogModel(log)
        assert [hold.test for hold in model.holds()] == [1, 2]
        assert [hold.test for hold in model.holds(2)] == [2]
        assert model.holds(None) == []


# ---- owed lists --------------------------------------------------------------


def test_a_partial_abort_is_accepted_and_its_first_minute_owed(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        abort = append(log, 'abort', author='AB', test=1, axis='Z', run=1,
                       fields={'stop': 'controller ramp'})
        correction = log.correct(abort, author='AB',
                                 fields={'channel': '4', 'counter': '3:10'})
        model = RunLogModel(log)
        owed, = model.abort_fields(1)
        assert owed.title == f'The first minute of abort #{abort.id}'
        assert owed.axis == 'Z'
        assert [line.key for line in owed.owed] == [
            'band', 'panel_fault', 'called_by', 'heard', 'records_read',
            'over_test', 'level_db']
        done = {line.key: line.done for line in owed.lines if not line.owed}
        assert done == {'channel': correction, 'counter': correction}
        assert 'abort' in [found.name for found in model.owed(1)]
        log.correct(abort, author='AB', fields={'channel': None})
        owed, = RunLogModel(log).abort_fields(1)
        assert 'channel' in [line.key for line in owed.owed]


def test_the_setup_is_owed_before_the_first_low_level(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        steps = ['Receiving inspection', 'configuration',
                 "The article's baseline operational checkout",
                 'mounting_record', 'END TO END check', 'surveys']
        for step in steps:
            log.append('setup_step', author='CD', test=1, axis='Z',
                       fields={'step': step})
        log.append('setup_step', author='CD', test=1,
                   fields={'step': 'tea break'})
        first = log.append('level_row', author='AB', test=1, axis='Z', run=1,
                           fields={'level_db': -12.0})
        late = log.append('setup_step', author='CD', test=1, axis='Z',
                          fields={'step': 'photographs'})
        owed = RunLogModel(log).checklist(1, 'before low level')
        assert (owed.axis, owed.due) == ('Z', first)
        assert [line.key for line in owed.owed] == ['second person']
        photographs = owed.lines[6]
        assert photographs.done == late and photographs.late
        assert not owed.lines[0].late and owed.lines[0].done.author == 'CD'
        assert str(photographs).endswith('(after it was due)')
        assert str(owed.lines[7]).startswith('[ ] A second person')
        assert '1 of 8 owed' in str(owed)


def test_each_later_axis_owes_the_axis_change(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': 0.0})
        log.append('setup_step', author='CD', test=1, axis='Z',
                   fields={'step': 'axis closed'})
        for key, _ in AXIS_CHANGE[1:-1]:
            log.append('setup_step', author='CD', test=1, axis='X',
                       fields={'step': key})
        log.append('setup_step', author='CD', test=1, axis='Y',
                   fields={'step': 'second person'})
        log.append('level_row', author='AB', test=1, axis='X', run=2,
                   fields={'level_db': -12.0})
        model = RunLogModel(log)
        change = model.checklist(1, 'axis change', 'X')
        assert change.title == 'The change to axis X'
        assert [line.key for line in change.owed] == ['second person']
        assert change.lines[0].done.axis == 'Z'      # it names the axis closed
        assert [found.title for found in model.owed(1)] == [
            'Before the first low level', 'The change to axis X',
            'The change to axis Y', 'A clock check each day']
        with pytest.raises(RunLogError, match="no axis change to 'Z'; its "
                           'axes are: Z, X, Y'):
            model.checklist(1, 'axis change', 'Z')
        with pytest.raises(RunLogError, match="'teardown' is not a checklist"):
            model.checklist(1, 'teardown')


def test_the_release_checklist_is_owed_once_the_test_is_closed(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        assert 'release' not in [o.name for o in RunLogModel(log).owed(1)]
        close_test(log, 1, author='AB')
        release_list = RunLogModel(log).checklist(1, 'release')
        assert [line.key for line in release_list.lines] == [
            key for key, _ in RELEASE]
        assert release_list.due is None
        for key, _ in RELEASE:
            log.append('post_test_step', author='AB', test=1,
                       fields={'step': key})
        assert RunLogModel(log).checklist(1, 'release').complete
        assert 'release' not in [o.name for o in RunLogModel(log).owed(1)]


def test_a_day_with_no_clock_check_is_owed(tmp_path):
    with new_log(tmp_path) as log:
        check = log.append('clock_check', author='AB',
                           fields={'offset_s': 1.0})
        random_test(log)
        log.append('note', author='AB', test=1,
                   event_time=at(24 * 60))                  # the next day
        model = RunLogModel(log)
        days = model.clock_days()
        assert [(line.key, line.done) for line in days.lines] == [
            ('2026-10-30', check), ('2026-10-31', None)]
        assert [line.label for line in days.owed] == [
            'A clock check on 2026-10-31']
        assert model.clock_days(1).test == 1
        assert [line.key for line in model.clock_days(None).lines] == [
            '2026-10-30']


# ---- derived views -----------------------------------------------------------


def credit_log(tmp_path, *, final='10:00', resume=True):
    log = new_log(tmp_path)
    random_test(log)
    log.append('run_start', author='AB', test=1, axis='Z', run=1,
               fields={'level_db': -12.0})
    log.append('level_row', author='AB', test=1, axis='Z', run=1,
               fields={'level_db': 0.0})
    log.append('abort', author='AB', test=1, axis='Z', run=1,
               fields={'stop': 'amplifier trip', 'over_test': 'no'})
    log.append('run_stop', author='AB', test=1, axis='Z', run=1,
               fields={'counter': '4:00'})
    log.append('cause_found', author='AB', test=1,
               fields={'cause': 'coolant flow'})
    log.append('restart', author='AB', test=1, run=1,
               fields={'resume': resume, 'counter_at_stop': '4:00',
                       'counter_at_restart': '4:00' if resume else '0:00',
                       'cause': 'coolant flow restored'})
    log.append('run_stop', author='AB', test=1, axis='Z', run=1,
               fields={'counter': final})
    return log


def test_credited_duration_across_an_interruption(tmp_path):
    with credit_log(tmp_path) as log:
        credit, = RunLogModel(log).credited(1)
        assert (credit.test, credit.axis) == (1, 'Z')
        assert credit.seconds == 600.0 and credit.complete
        assert (credit.duration, credit.lower, credit.upper) == (
            600.0, 600.0, pytest.approx(660.0))
        assert credit.verdict == 'within'
        assert credit.message == '10:00 credited, within 10:00 to 11:00'
        assert str(credit).startswith('test 1, axis Z: 10:00 credited')


@pytest.mark.parametrize(('final', 'verdict', 'message'), [
    ('9:59', 'short', ('9:59 credited is short of 10:00, the least the '
     'tolerance allows on 10:00')),
    ('11:01', 'over', ('11:01 credited is over 11:00, the most the tolerance '
     'allows on 10:00')),
    ('11:00', 'within', '11:00 credited, within 10:00 to 11:00'),
])
def test_credited_duration_is_judged_against_the_tolerance(
        tmp_path, final, verdict, message):
    with credit_log(tmp_path, final=final) as log:
        credit, = RunLogModel(log).credited(1)
        assert (credit.verdict, credit.message) == (verdict, message)


def test_a_restart_that_starts_again_discards_what_was_credited(tmp_path):
    with credit_log(tmp_path, final='10:00', resume=False) as log:
        credit, = RunLogModel(log).credited(1)
        assert credit.seconds == 600.0 and credit.complete
        assert credit.notes == (
            ('the restart at #7 starts the test again; the 4:00 credited '
            'before it is not counted'),)


def test_credited_time_that_cannot_be_read_is_a_lower_bound(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('run_start', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': 0.0, 'counter': '0:30'})
        log.append('run_stop', author='AB', test=1, run=1,
                   fields={'counter': '2:30'})
        log.append('run_start', author='AB', test=1, axis='Z', run=2,
                   fields={'level_db': 0.0})
        log.append('abort', author='AB', test=1, axis='Z', run=2,
                   fields={'stop': 'stop word'})
        log.append('restart', author='AB', test=1, run=2)
        log.append('run_stop', author='AB', test=1, run=2,
                   fields={'counter': '5:00'})
        log.append('run_start', author='AB', test=1, axis='Z', run=3,
                   fields={'level_db': 0.0})
        log.append('run_stop', author='AB', test=1, run=3,
                   fields={'counter': 'later'})
        log.append('run_start', author='AB', test=1, axis='Z', run=4,
                   fields={'level_db': -12.0})
        log.append('run_stop', author='AB', test=1, run=4,
                   fields={'counter': '1:00'})
        log.append('run_start', author='AB', test=1, axis='Z', run=5,
                   fields={'level_db': 0.0, 'counter': '3:00'})
        log.append('run_stop', author='AB', test=1, run=5,
                   fields={'counter': '2:00'})
        log.append('run_stop', author='AB', test=1, axis='Z', run=6,
                   fields={'counter': '1:00'})
        log.append('run_start', author='AB', test=1, axis='Z', run=7,
                   fields={'level_db': 0.0})
        log.append('run_start', author='AB', test=1, axis='Z', run=8,
                   fields={'level_db': 0.0})
        log.append('abort', author='AB', test=1,
                   fields={'stop': 'stop word', 'counter': '0:10'})
        log.append('run_stop', author='AB', test=1, axis='Z', run=8,
                   fields={'counter': '0:30'})
        log.append('run_start', author='AB', test=1, axis='Z', run=9,
                   fields={'level_db': 0.0})
        model = RunLogModel(log)
        credit, loose = model.credited(1)
        assert loose.axis is None and loose.notes == (
            '#17 stops a run with no start logged; its time is not credited',)
        assert credit.seconds == 150.0 and not credit.complete
        assert credit.verdict == 'not judged'
        assert credit.message.startswith('at least 2:30 credited; not judged')
        assert credit.notes == (
            ('the abort or trip at #5 gives no counter, so the time before it '
            'is not credited'),
            ('the restart at #6 does not say whether it resumes or starts '
            'again'),
            ("#9: the counter 'later' is not read as a time: give seconds, "
            'minutes:seconds or hours:minutes:seconds'),
            'run 4 did not reach full level; its time is not credited',
            ("the counter at #13 (2:00) is below the counter at the segment's "
            'start (3:00)'),
            '#14 stops a run with no start logged; its time is not credited',
            'run 7 has no stop logged before #16; its time is not credited',
            'run 9 has no stop logged yet; its time so far is not credited')


def test_credited_time_without_a_duration_or_tolerance_is_not_judged(
        tmp_path):
    with new_log(tmp_path) as log:
        open_test(log, author='AB', test_kind='Random Vibration',
                  header={'duration_s': 60.0})
        log.append('run_start', author='AB', test=1, run=1,
                   fields={'level_db': 0.0})
        log.append('run_stop', author='AB', test=1, run=1,
                   fields={'counter': '60'})
        credit, = RunLogModel(log).credited(1)
        assert (credit.axis, credit.verdict) == (None, 'not judged')
        assert credit.message == ('1:00 credited of 1:00; not judged: the '
                                  'header gives no duration tolerance (minus '
                                  'and plus)')
        roll(log, author='AB', header={'duration_s': None})
        log.append('run_start', author='AB', test=2, run=1,
                   fields={'level_db': 0.0})
        log.append('run_stop', author='AB', test=2, run=1,
                   fields={'counter': '30'})
        credit, = RunLogModel(log).credited(2)
        assert credit.message == ('0:30 credited; not judged: the header '
                                  'gives no duration')


def test_a_resume_without_a_counter_credits_nothing_after_it(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('run_start', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': 0.0})
        log.append('run_stop', author='AB', test=1, run=1,
                   fields={'counter': '6:00', 'scheduled': True})
        log.append('restart', author='AB', test=1, run=1,
                   fields={'resume': True, 'cause': 'scheduled'})
        log.append('run_stop', author='AB', test=1, run=1,
                   fields={'counter': '10:00'})
        log.append('restart', author='AB', test=1, run=1,
                   fields={'resume': True, 'counter_at_stop': '10:00',
                           'cause': 'x'})
        log.append('run_stop', author='AB', test=1, run=1,
                   fields={'counter': '10:30'})
        credit, = RunLogModel(log).credited(1)
        assert credit.seconds == 390.0 and not credit.complete
        assert credit.notes == (('the restart at #4 gives no counter, so the '
                                'time after it is not credited'),)
        model = RunLogModel(log)
        assert model.runs(1)[0].outcome == 'completed'
        stops = model.interruptions(1)
        assert [(stop.kind, stop.restart.id) for stop in stops] == [
            ('scheduled', 4)]


def test_interruptions_carry_their_cause_path_and_restart(tmp_path):
    with credit_log(tmp_path) as log:
        log.append('abort', author='AB', test=1, axis='Z', run=2,
                   fields={'stop': 'stop word'})
        path = log.append('path_decided', author='AB', test=1,
                          fields={'path': 'inspect fixture', 'refers_to': 4})
        model = RunLogModel(log)
        trip, word = model.interruptions(1)
        assert (trip.kind, trip.entry.id) == ('trip', 4)
        assert trip.cause.fields['cause'] == 'coolant flow'
        assert trip.restart.id == 7 and trip.path == path
        assert str(trip) == '#4 trip: coolant flow'
        assert (word.kind, word.cause, word.restart) == ('abort', None, None)
        assert str(word) == '#9 abort: no cause'
        assert model.runs(1)[0].outcome == 'completed'
        assert model.runs(1)[1].outcome == 'aborted'


def test_the_abort_and_run_stop_written_together_are_one_stop(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('run_start', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': 0.0})
        log.append('run_stop', author='AB', test=1, run=1)
        log.append('abort', author='AB', test=1, run=1,
                   fields={'stop': 'controller ramp', 'counter': '2:00'})
        model = RunLogModel(log)
        credit, = model.credited(1)
        assert (credit.seconds, credit.complete) == (120.0, True)
        assert model.runs(1)[0].outcome == 'aborted'


def test_findings_by_number_with_open_anomalies_and_their_owners(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('deviation', author='AB', test=1,
                   fields={'fact': 'notch at 200 Hz', 'agreed_by': 'CD'})
        log.append('anomaly_opened', author='AB', test=1,
                   fields={'fact': 'bracket noisy', 'owner': 'EF'})
        roll(log, author='AB', test_kind='Shock')
        log.append('anomaly_opened', author='AB', test=2,
                   fields={'fact': 'connector loose', 'owner': 'GH'})
        log.append('anomaly_dispositioned', author='AB', test=2,
                   fields={'number': 'A-1', 'disposition': 'use as is'})
        model = RunLogModel(log)
        found = model.findings()
        assert [(f.number, f.kind, f.test) for f in found] == [
            ('D-1', 'deviation', 1), ('A-1', 'anomaly', 1),
            ('A-2', 'anomaly', 2)]
        assert not found[0].open and found[0].fact == 'notch at 200 Hz'
        assert found[1].dispositions[0].fields['disposition'] == 'use as is'
        assert [(f.number, f.owner) for f in model.open_anomalies()] == [
            ('A-2', 'GH')]
        assert [f.number for f in model.findings(1)] == ['D-1', 'A-1']
        assert model.open_anomalies(1) == []


def test_files_late_entries_and_the_chronology(tmp_path):
    with new_log(tmp_path) as log:
        log.append('clock_check', author='AB', fields={'offset_s': 2.0})
        random_test(log)
        ref = FileRef('run1.csv', '0' * 64, 10)
        named = log.append('note', author='AB', test=1, files=[ref])
        early = log.append('note', author='AB', test=1, event_time=at(-30),
                           text='written late')
        model = RunLogModel(log)
        assert [(row.entry, row.file) for row in model.files()] == [
            (named, ref)]
        assert model.files(None) == []
        assert model.late(1) == [early]
        rows = model.chronology()
        assert [row.entry.id for row in rows] == [4, 1, 2, 3]
        assert rows[0].late and not rows[1].late
        assert rows[0].recorder_time is None             # before the check
        assert rows[1].recorder_time == rows[1].entry.event_time + timedelta(
            seconds=2)
        assert [row.entry.id for row in model.chronology(None)] == [1]


# ---- readings ----------------------------------------------------------------


def test_a_level_row_is_projected_against_its_tests_header(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -12.0, 'voltage': 20.0,
                           'current': 63.0, 'drive_fraction': 0.12})
        row = log.append('level_row', author='AB', test=1, axis='Z', run=1,
                         fields={'level_db': -6.0, 'voltage': 40.0,
                                 'current': 110.0, 'drive_fraction': 0.24})
        reading = RunLogModel(log).reading(row)
        assert reading.verdict == 'within'
        assert reading.projection.check('peak current').at_full == \
            pytest.approx(110.0 * 1.99526231496888 * 3.0, rel=1e-12)
        assert reading.message.startswith('-6 dB, factor 1.995 to full level: '
                                          'within')
        previous = reading.previous
        assert previous.what == 'the previous level'
        assert previous.ratios['volts per amp'] == pytest.approx(
            (40 / 110) / (20 / 63))
        assert previous.flagged == ('volts per amp',)     # +15 percent
        assert 'FLAGGED: volts per amp' in str(previous)
        assert reading.last_run is None


def test_the_volts_per_amp_change_is_flagged_only_past_its_threshold(
        tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -12.0, 'voltage': 20.0,
                           'current': 50.0})
        steady = log.append('level_row', author='AB', test=1, axis='Z', run=1,
                            fields={'level_db': -6.0, 'voltage': 43.9,
                                    'current': 100.0})
        assert RunLogModel(log).reading(steady).previous.flagged == ()


def test_a_level_row_is_compared_with_the_last_run_of_the_same_setup(
        tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        first = log.append('level_row', author='AB', test=1, axis='Z', run=1,
                           fields={'level_db': -12.0, 'voltage': 20.0,
                                   'current': 63.0})
        log.append('level_row', author='AB', test=1, axis='X', run=2,
                   fields={'level_db': -12.0, 'voltage': 25.0,
                           'current': 70.0})
        log.append('level_row', author='AB', test=1, axis='Z',
                   fields={'level_db': -12.0, 'current': 10.0})   # no run
        again = log.append('level_row', author='AB', test=1, axis='Z', run=3,
                           fields={'level_db': -12.0, 'voltage': 20.5,
                                   'current': 80.0})
        last = RunLogModel(log).reading(again).last_run
        assert last.against == first
        assert last.what == 'the last run of the same setup (test 1, run 1)'
        assert last.ratios['current'] == pytest.approx(80 / 63)
        assert last.flagged == ('current',)
        roll(log, author='AB')
        later = log.append('level_row', author='AB', test=2, axis='Z', run=1,
                           fields={'level_db': -12.0, 'current': 63.0})
        assert RunLogModel(log).reading(later).last_run.against == again
        roll(log, author='AB', header={'configuration': 'dummy mass'})
        other = log.append('level_row', author='AB', test=3, axis='Z', run=1,
                           fields={'level_db': -12.0, 'current': 63.0})
        assert RunLogModel(log).reading(other).last_run is None
        assert 'voltage not compared' in str(
            RunLogModel(log).reading(later).last_run)


def test_a_reading_that_cannot_be_projected_says_why(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        bare = log.append('level_row', author='AB', test=1, run=1)
        above = log.append('level_row', author='AB', test=1, run=1,
                           fields={'level_db': 2.0, 'current': 9.0})
        loose = log.append('level_row', author='AB', run=1,
                           fields={'level_db': -12.0})
        empty = log.append('level_row', author='AB', test=1, run=1,
                           fields={'level_db': -12.0})
        note = log.append('note', author='AB')
        model = RunLogModel(log)
        assert model.reading(bare.id).message == ('not judged: the row gives '
                                                  'no level')
        assert model.reading(bare).verdict == 'not judged'
        assert 'above full level' in model.reading(above).message
        assert model.reading(above).projection is None
        assert 'belongs to no test' in model.reading(loose).message
        nothing = model.reading(empty)
        assert nothing.verdict == 'not judged'
        assert {c.verdict for c in nothing.projection.checks} == {
            'not judged'}
        with pytest.raises(RunLogError, match='is a Note entry, not a level'):
            model.reading(note)
        with pytest.raises(RunLogError, match='no entry #99'):
            model.reading(99)


def test_a_reading_uses_the_header_as_it_stood_at_the_row(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log, {'sigma': None})
        row = log.append('level_row', author='AB', test=1, axis='Z', run=1,
                         fields={'level_db': 0.0, 'current': 100.0})
        change_header(log, 1, 'sigma', 3.0, author='AB')
        model = RunLogModel(log)
        assert 'no sigma' in model.reading(row).projection.check(
            'peak current').message
        assert model.header_at(row).get('sigma') is None
        assert model.test(1).header.get('sigma') == 3.0
        loose = log.append('note', author='AB')
        assert RunLogModel(log).header_at(loose).values == {}


# ---- edges the mutation run found unpinned -----------------------------------


def test_the_edges_of_what_is_projected():
    with pytest.raises(RunLogError, match=r'\+0.5 dB is above full level'):
        full_level_factor(0.5)
    assert project(0.0, drive_fraction=1.0).check(
        'drive fraction').verdict == 'over'               # 1.0 is a reading
    for window in ((0.0, 0.5), (0.5, 1.0)):
        assert project(0.0, drive_fraction=0.5, drive_window=window).check(
            'drive fraction').verdict == 'within'
    for window in ((0.5, 0.5), (False, True)):
        with pytest.raises(RunLogError, match='low then high'):
            project(0.0, drive_window=window)
    peak = project(-12.0, current=63.0, sigma=3.0, rated_current_peak=800.0)
    assert peak.check('peak current').message.endswith('(94%)')


@pytest.mark.parametrize(('text', 'message'), [
    (True, 'a counter is a time such as 12:30, not True'),
    (None, 'a counter is a time such as 12:30, not None'),
])
def test_a_counter_that_is_not_text_or_a_number_is_named(text, message):
    with pytest.raises(RunLogError, match=message):
        parse_counter(text)
    assert parse_counter('750s') == 750.0


def test_a_level_within_the_match_of_full_is_full(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -0.05})
        assert RunLogModel(log).runs(1)[0].reached_full


def test_tests_are_ordered_by_their_opening_not_their_first_entry(tmp_path):
    with new_log(tmp_path) as log:
        log.append('note', author='AB', test=2)
        log.append('test_opened', author='AB', test=1,
                   fields={'test_kind': 'Shock'})
        log.append('test_opened', author='AB', test=2,
                   fields={'test_kind': 'Shock'})
        assert [t.number for t in RunLogModel(log).tests()] == [1, 2]


def test_a_header_change_written_before_its_test_opened_does_not_apply(
        tmp_path):
    with new_log(tmp_path) as log:
        log.append('header_change', author='AB', test=1,
                   fields={'field': 'title', 'value': 'early'})
        log.append('test_opened', author='AB', test=1,
                   fields={'test_kind': 'Shock', 'title': 'Shock'})
        assert RunLogModel(log).test(1).header.get('title') == 'Shock'


def test_only_a_deviation_naming_the_entry_answers_its_hold(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        notch = log.append('notch', author='AB', test=1, axis='Z', run=1,
                           fields={'band': '200 Hz'})
        other = log.append('note', author='AB', test=1)
        log.append('deviation', author='AB', test=1,
                   fields={'refers_to': other.id})
        log.append('cause_found', author='AB', test=1,
                   fields={'cause': 'x', 'refers_to': notch.id})
        assert rules(RunLogModel(log).holds()) == ['unplanned-notch']


def test_a_notch_exactly_at_its_planned_depth_and_match_is_planned(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log, {'notches_planned': [{'band': '180-220 Hz',
                                               'depth_db': -6}]})
        notch = log.append('notch', author='AB', test=1, axis='Z', run=1,
                           fields={'band': '200 Hz', 'depth_db': -6.05})
        assert RunLogModel(log).notch_planned(notch)[0]


def test_a_setup_step_at_the_instant_it_was_due_is_not_late(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('setup_step', author='CD', test=1, axis='Z',
                   fields={'step': 'surveys'}, event_time=at(30))
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -12.0}, event_time=at(30))
        line = RunLogModel(log).checklist(1, 'before low level').lines[5]
        assert line.done is not None and not line.late


def test_the_axis_change_keeps_all_eight_lines(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('level_row', author='AB', test=1, axis='Z', run=1)
        log.append('setup_step', author='CD', test=1, axis='X',
                   fields={'step': 'hardware'})
        change = RunLogModel(log).checklist(1, 'axis change', 'X')
        assert [line.key for line in change.lines] == [
            key for key, _ in AXIS_CHANGE]
        assert not change.lines[1].owed


def test_an_abort_with_its_counter_and_a_bare_stop_beside_it(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('run_start', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': 0.0})
        log.append('abort', author='AB', test=1, run=1,
                   fields={'stop': 'controller ramp', 'counter': '2:00'})
        log.append('run_stop', author='AB', test=1, run=1)
        credit, = RunLogModel(log).credited(1)
        assert (credit.seconds, credit.complete) == (120.0, True)
        # both giving a counter: the first written stands
        log.append('run_start', author='AB', test=1, axis='Z', run=2,
                   fields={'level_db': 0.0})
        log.append('abort', author='AB', test=1, run=2,
                   fields={'stop': 'controller ramp', 'counter': '1:00'})
        log.append('run_stop', author='AB', test=1, run=2,
                   fields={'counter': '1:05'})
        credit, = RunLogModel(log).credited(1)
        assert credit.seconds == 180.0


def test_an_unreadable_start_counter_alone_leaves_the_credit_open(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('run_start', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': 0.0, 'counter': 'soon'})
        log.append('run_stop', author='AB', test=1, run=1,
                   fields={'counter': '10:00'})
        credit, = RunLogModel(log).credited(1)
        assert not credit.complete and credit.verdict == 'not judged'


def test_a_restart_from_its_own_counter_and_a_resume_in_a_new_run(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('run_start', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': 0.0})
        log.append('abort', author='AB', test=1, run=1,
                   fields={'stop': 'stop word', 'counter': '3:00'})
        log.append('restart', author='AB', test=1, axis='Z', run=2,
                   fields={'resume': True, 'counter_at_restart': '3:00',
                           'cause': 'x'})
        log.append('run_stop', author='AB', test=1, run=2,
                   fields={'counter': '3:00'})            # a zero segment
        log.append('restart', author='AB', test=1, axis='Z', run=3,
                   fields={'resume': True, 'counter_at_restart': '3:00',
                           'cause': 'x'})
        log.append('run_stop', author='AB', test=1, run=3,
                   fields={'counter': '8:00'})
        credit, = RunLogModel(log).credited(1)
        assert (credit.seconds, credit.complete) == (480.0, True)
        log.append('restart', author='AB', test=1, axis='Z', run=4,
                   fields={'resume': False, 'counter_at_restart': '0:30',
                           'cause': 'x'})
        log.append('level_row', author='AB', test=1, axis='Z', run=4,
                   fields={'level_db': 0.0})
        log.append('run_stop', author='AB', test=1, run=4,
                   fields={'counter': '10:30'})
        credit, = RunLogModel(log).credited(1)
        assert (credit.seconds, credit.complete) == (600.0, True)


@pytest.mark.parametrize(('final', 'verdict'), [('9:30', 'within'),
                                                ('9:29', 'short'),
                                                ('10:30', 'within'),
                                                ('10:31', 'over')])
def test_a_tolerance_on_both_sides_bounds_the_credit(tmp_path, final,
                                                     verdict):
    with new_log(tmp_path) as log:
        random_test(log, {'duration_minus_pct': 5.0,
                          'duration_plus_pct': 5.0})
        log.append('run_start', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': 0.0})
        log.append('run_stop', author='AB', test=1, run=1,
                   fields={'counter': final})
        credit, = RunLogModel(log).credited(1)
        assert (credit.lower, credit.upper) == (pytest.approx(570.0),
                                                pytest.approx(630.0))
        assert credit.verdict == verdict


def test_a_tolerance_on_one_side_only_is_not_judged(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log, {'duration_plus_pct': None})
        log.append('run_start', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': 0.0})
        log.append('run_stop', author='AB', test=1, run=1,
                   fields={'counter': '10:00'})
        assert RunLogModel(log).credited(1)[0].verdict == 'not judged'


def test_a_zero_current_reading_is_never_compared(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -12.0, 'voltage': 5.0,
                           'current': 0.0})
        row = log.append('level_row', author='AB', test=1, axis='Z', run=1,
                         fields={'level_db': -6.0, 'voltage': 10.0,
                                 'current': 30.0})
        assert RunLogModel(log).reading(row).previous is None


def test_change_header_records_what_the_field_was(tmp_path):
    with new_log(tmp_path) as log:
        random_test(log)
        changed = change_header(log, 1, 'sigma', 4.0, author='AB')
        assert changed.fields == {'field': 'sigma', 'value': 4.0, 'was': 3.0}
        new = change_header(log, 1, 'purpose', 'qualification', author='AB')
        assert new.fields == {'field': 'purpose', 'value': 'qualification'}
        with pytest.raises(RunLogError, match='the carried fields are: none'):
            confirm(log, 1, ['sigma'], author='AB')
