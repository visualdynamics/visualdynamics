"""Calibration in date: each sensor against the test date (report feature
`calibration_in_date`).

The check itself (`core.calibration`): dates read one way only, the test
span, the four answers at their edges, a cell that is not a date named,
and a check that judged nothing saying so. Then the report
(`report.calibration`): the Calibration column and its marks, the
paragraph under the table, the setup and method offer, the dates kept
with the report and left out of a template, the project verbs, the
editor's pane, and the page left as it was with the feature off.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core import calibration as cal
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.report import Report
from visualdynamics.report import calibration as rc
from visualdynamics.report import features as rf
from visualdynamics.report import render_html

D = datetime.date

#: one expiration per channel of the plate's random run: in date,
#: expired, lapsing on the middle day, blank, and the rest in date,
#: with channel 10 due on the last test day
EXPIRATIONS = ['2027-01-01', '2026-09-30', '2026-10-25', '', '2027-01-01',
               '2027-01-01', '2027-01-01', '2027-01-01', '2027-01-01',
               '2026-10-26', '', '2027-01-01']


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


def _table(rows):
    """A channel table from (expiration, unit, node, control) rows."""
    return ChannelTable({
        'channel': list(range(1, len(rows) + 1)),
        'node': [row[2] for row in rows],
        'direction': ['Z+'] * len(rows),
        'unit': [row[1] for row in rows],
        'control': [row[3] for row in rows],
        'expiration': [row[0] for row in rows]})


SPAN = cal.DateSpan(D(2026, 10, 24), D(2026, 10, 26))


# ---- reading a date one way ------------------------------------------------


def test_a_date_is_read_in_one_form_only():
    assert cal.parse_date('2026-10-24') == D(2026, 10, 24)
    assert cal.parse_date(' 2026-10-24 ') == D(2026, 10, 24)
    assert cal.parse_date(D(2026, 10, 24)) == D(2026, 10, 24)
    assert cal.parse_date(datetime.datetime(2026, 10, 24, 23, 59)) \
        == D(2026, 10, 24)
    # a slashed date reads two ways; the compact and week forms that
    # fromisoformat would take are not the form the table stores
    for text in ('10/24/2026', '04/05/2026', '24.10.2026', '20261024',
                 '2026-W43-6', '2026-13-01', '2026-02-30', '', 'soon'):
        with pytest.raises(ValueError, match='cannot be read one way'):
            cal.parse_date(text)
    with pytest.raises(ValueError, match="the last test day '10/26/2026'"):
        cal.parse_date('10/26/2026', 'last test day')
    with pytest.raises(TypeError, match='not int'):
        cal.parse_date(20261024)


def test_the_span_is_entered_or_recorded_and_refused_when_it_reads_two_ways():
    assert cal.span_of() is None
    assert cal.span_of('', '') is None
    one = cal.span_of('2026-10-24')
    assert one == cal.DateSpan(D(2026, 10, 24), D(2026, 10, 24))
    assert one.days == 1 and str(one) == '2026-10-24'
    assert cal.span_of('2026-10-24', '') == one
    assert SPAN.days == 3 and str(SPAN) == '2026-10-24 to 2026-10-26'
    assert cal.span_of('2026-10-24', '2026-10-26') == SPAN
    # from the records alone, the earliest to the latest
    assert cal.span_of(recorded=['2026-10-26', D(2026, 10, 24)]) == SPAN
    # an entry that covers what the records say is kept
    assert cal.span_of('2026-10-24', '2026-10-26',
                       recorded=['2026-10-25']) == SPAN
    # a record on the first or the last day is inside the span
    assert cal.span_of('2026-10-24', '2026-10-26',
                       recorded=['2026-10-24', '2026-10-26']) == SPAN
    with pytest.raises(ValueError, match='does not cover .* 2026-10-23'):
        cal.span_of('2026-10-24', '2026-10-26', recorded=['2026-10-23'])
    with pytest.raises(ValueError, match='does not cover .* 2026-10-27'):
        cal.span_of('2026-10-24', '2026-10-26', recorded=['2026-10-27'])
    with pytest.raises(ValueError, match='before the first, 2026-10-24'):
        cal.span_of('2026-10-24', '2026-10-23')
    with pytest.raises(ValueError, match='without a first'):
        cal.span_of(None, '2026-10-26')
    with pytest.raises(ValueError, match="test date '24/10/2026'"):
        cal.span_of('24/10/2026')


# ---- the four answers ------------------------------------------------------


def test_the_answers_at_their_edges():
    """Through the expiration date in calibration: due on the first day
    is in date on a one-day test, due the day before is expired by one
    day, due on the last day is in date, due on the first of three days
    lapses for two."""
    day = cal.DateSpan(D(2026, 10, 24), D(2026, 10, 24))
    assert cal.judge(D(2026, 10, 24), day) == (cal.IN_DATE, None)
    assert cal.judge(D(2026, 10, 23), day) == (cal.EXPIRED, 1)
    assert cal.judge(D(2025, 10, 24), day) == (cal.EXPIRED, 365)
    assert cal.judge(None, day) == (cal.NO_DATE, None)
    assert cal.judge(D(2026, 10, 26), SPAN) == (cal.IN_DATE, None)
    assert cal.judge(D(2026, 10, 25), SPAN) == (cal.LAPSES, 1)
    assert cal.judge(D(2026, 10, 24), SPAN) == (cal.LAPSES, 2)
    assert cal.judge(D(2026, 10, 23), SPAN) == (cal.EXPIRED, 1)


def test_each_channel_is_named_by_its_identity_and_judged():
    table = _table([('2027-01-01', 'g', 101, True),
                    ('2026-10-20', 'lbf', 101, False),
                    ('2026-10-24', 'g', 102, False),
                    ('', '', 103, False)])
    found = cal.check(table, SPAN)
    assert [c.status for c in found.channels] == [
        cal.IN_DATE, cal.EXPIRED, cal.LAPSES, cal.NO_DATE]
    # a force and an acceleration at one DOF are two sensors
    assert [c.key for c in found.channels] == [
        ('101Z+', 'acceleration', None), ('101Z+', 'force', None),
        ('102Z+', 'acceleration', None), ('103Z+', 'unknown', None)]
    assert [c.name for c in found.channels] == [
        'channel 1 (101Z+ acceleration, control)',
        'channel 2 (101Z+ force)', 'channel 3 (102Z+ acceleration)',
        'channel 4 (103Z+)']
    assert [c.words for c in found.channels] == [
        'in date', 'expired 4 days before the test',
        'lapses during the test, in date through 2026-10-24',
        'no date given, not checked']
    assert found.judged == 3
    assert [c.channel for c in found.lapsed] == [2, 3]
    assert found.summary() == [
        ('Calibration on the test dates, 2026-10-24 to 2026-10-26: 1 of 4 '
        'channels in date.'),
        'Channel 2 (101Z+ force) expired 2026-10-20, 4 days before the test.',
        ('Channel 3 (102Z+ acceleration) lapses during the test: in date '
        'through 2026-10-24, not on the last 2 test days.'),
        'No expiration date given, so not checked: channel 4 (103Z+).']


def test_singular_days_and_a_channel_without_a_dof():
    table = _table([('2026-10-23', 'g', '', False),
                    ('2026-10-25', 'g', 7, False)])
    table.set_cell('direction', 0, '')
    found = cal.check(table, SPAN)
    assert found.channels[0].name == 'channel 1 (acceleration)'
    assert found.channels[0].words == 'expired 1 day before the test'
    assert found.summary()[1:] == [
        'Channel 1 (acceleration) expired 2026-10-23, 1 day before the test.',
        ('Channel 2 (7Z+ acceleration) lapses during the test: in date '
        'through 2026-10-25, not on the last test day.')]
    blank = _table([('2027-01-01', '', '', True)])
    blank.set_cell('direction', 0, '')
    assert cal.check(blank, SPAN).channels[0].name == 'channel 1 (control)'
    bare = _table([('2027-01-01', '', '', False)])
    bare.set_cell('direction', 0, '')
    assert cal.check(bare, SPAN).channels[0].name == 'channel 1'


def test_a_unit_typed_later_still_names_the_quantity():
    """A unit typed into a table after it was made leaves the channel
    type blank; the unit is the same claim, and names the quantity."""
    table = _table([('2027-01-01', '', 5, False)])
    assert table.types() == ['']
    table.set_cell('unit', 0, 'g')
    assert table.types() == [''], 'the table leaves the type to its owner'
    found = cal.check(table, SPAN).channels[0]
    assert found.key == ('5Z+', 'acceleration', None)
    assert found.name == 'channel 1 (5Z+ acceleration)'


def test_a_record_finds_its_sensor_by_channel_key(run):
    project, _name = run
    history = next(obj for obj in project.values()
                   if type(obj).__name__ == 'TimeHistory')
    table = _table([('2027-01-01', 'g', 101, True),
                    ('2026-10-01', 'lbf', 101, False)])
    found = cal.check(table, SPAN)
    keys = {history.channel_key(i) for i in range(history.num_records)}
    accel = next(key for key in keys if key[:2] == ('101Z+', 'acceleration'))
    force = next(key for key in keys if key[:2] == ('101Z+', 'force'))
    assert found.of(accel).channel == 1
    assert found.of(force).status == cal.EXPIRED
    assert found.of(('999Z+', 'acceleration', None)) is None
    twice = cal.check(_table([('2027-01-01', 'g', 101, True),
                              ('2027-01-01', 'g', 101, False)]), SPAN)
    with pytest.raises(ValueError, match='channel 1 .* and channel 2 .* are '
                                         'both 101Z\\+ acceleration'):
        twice.of(accel)


def test_all_in_date_is_said_only_when_every_channel_had_a_date():
    table = _table([('2027-01-01', 'g', 1, False),
                    ('2026-10-26', 'g', 2, False)])
    assert cal.check(table, SPAN).summary() == [
        ('Calibration on the test dates, 2026-10-24 to 2026-10-26: all 2 '
        'channels in date.')]
    table.set_cell('expiration', 1, '')
    words = cal.check(table, SPAN).summary()
    assert words[0].endswith('1 of 2 channels in date.')
    assert 'all' not in words[0]


def test_a_check_that_judged_nothing_says_so():
    """Dead and empty inputs: a table with no expiration at all, and a
    table with no channels. Neither is ever 'in date'."""
    blank = _table([('', 'g', 1, True), ('', 'g', 2, False)])
    found = cal.check(blank, cal.span_of('2026-10-24'))
    assert found.judged == 0
    assert found.summary() == [
        ('Calibration on the test date, 2026-10-24: not checked, no channel '
        'has an expiration date.')]
    assert all(c.words == 'no date given, not checked'
               for c in found.channels)
    empty = ChannelTable({'channel': [], 'node': [], 'direction': [],
                          'unit': []})
    nothing = cal.check(empty, SPAN)
    assert nothing.channels == () and nothing.judged == 0
    assert nothing.summary() == [
        ('Calibration on the test dates, 2026-10-24 to 2026-10-26: not '
        'checked, the channel table has no channels.')]
    for words in (found.summary(), nothing.summary()):
        assert 'in date' not in ' '.join(words)


def test_one_blank_among_dated_channels_is_named_not_passed():
    """The one-NaN case: one channel without a date among dated ones is
    named as not checked, and the headline does not count it in."""
    table = _table([('2027-01-01', 'g', n, False) for n in range(1, 6)])
    table.set_cell('expiration', 2, '')
    words = cal.check(table, SPAN).summary()
    assert words == [
        ('Calibration on the test dates, 2026-10-24 to 2026-10-26: 4 of 5 '
        'channels in date.'),
        ('No expiration date given, so not checked: channel 3 (3Z+ '
        'acceleration).')]


def test_a_cell_that_is_not_a_date_is_refused_by_name():
    table = _table([('2027-01-01', 'g', 1, False),
                    ('2027-01-01', 'g', 2, False)])
    # the table's own entry refuses it; a file edited by hand can still
    # carry one
    with pytest.raises(ValueError, match='not a date'):
        table.set_cell('expiration', 1, 'next spring')
    table.frame.loc[1, 'expiration'] = '01/02/2027'
    with pytest.raises(ValueError, match="channel table row 2 \\(channel 2\\), "
                                         "Expiration: '01/02/2027' is not a "
                                         'date'):
        cal.check(table, SPAN)


def test_check_refuses_what_is_not_a_table_or_a_span():
    with pytest.raises(TypeError, match='channel table, not dict'):
        cal.check({}, SPAN)
    with pytest.raises(TypeError, match='DateSpan, not str'):
        cal.check(_table([('', 'g', 1, False)]), '2026-10-24')


# ---- the report -------------------------------------------------------------


@pytest.fixture(scope='module')
def run():
    """A random run worked up, its channel table given expirations."""
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'))
    table = next(obj for obj in project.values()
                 if isinstance(obj, ChannelTable))
    for row, when in enumerate(EXPIRATIONS):
        table.set_cell('expiration', row, when)
    name = project.generate_report('random')
    return project, name


@pytest.fixture
def calibrating(run):
    """The run's report with the feature on, put back afterwards."""
    project, name = run
    report = project[name]
    blocks = [dict(block) for block in report.blocks]
    project.report_features(name, enable=['calibration_in_date'])
    yield project, name, report
    report.blocks = blocks
    report.features = []


def _render(project, name, **kwargs):
    return render_html(project[name], project, unit_system=visualdynamics.SI,
                       object_groups=project.object_groups, **kwargs)


def _channel_table(data):
    tables = [block for block in data['blocks'] if block['kind'] == 'table']
    assert len(tables) == 1
    return tables[0], data['blocks'][data['blocks'].index(tables[0]) + 1]


def test_the_feature_is_built_and_off_by_default():
    feature = rf.FEATURES['calibration_in_date']
    assert feature.available and not feature.default and not feature.requires
    assert feature.group == 'Records'
    assert 'calibration_in_date' in rf.PRESETS['test_report'].features
    assert rf.FEATURES['calibration_in_date'].key == rc.FEATURE


def test_off_the_page_is_the_page_it_was_even_with_dates_kept(run):
    project, name = run
    report = project[name]
    plain = _render(project, name)
    held = rc.table_blocks(report, project, project.object_groups)
    try:
        for index in held:
            report.blocks[index][rc.START] = '2026-10-24'
            report.blocks[index][rc.END] = '2026-10-26'
        assert _render(project, name) == plain
        assert _render(project, name, edit=True) == _render(
            project, name, edit=True)
    finally:
        for index in held:
            report.blocks[index].pop(rc.START, None)
            report.blocks[index].pop(rc.END, None)
    assert 'Calibration' not in json.dumps(_payload(plain))


def test_without_a_date_the_table_says_it_was_not_checked(calibrating):
    project, name, _report = calibrating
    table, under = _channel_table(_payload(_render(project, name)))
    assert 'Calibration' not in table['headers'] and 'marks' not in table
    assert under['kind'] == 'text'
    assert under['html'] == ('<p><strong>Calibration not checked: no test '
                             'date is entered.</strong></p>')


def test_the_column_the_marks_and_the_paragraph(calibrating):
    project, name, _report = calibrating
    project.report_test_date(name, '2026-10-24', '2026-10-26')
    table, under = _channel_table(_payload(_render(project, name)))
    assert table['headers'][-1] == 'Calibration'
    words = [row[-1] for row in table['rows']]
    assert words[:4] == ['in date', 'expired 24 days before the test',
                         'lapses during the test, in date through 2026-10-25',
                         'no date given, not checked']
    assert words[9] == 'in date', 'due on the last test day'
    # the expired and the lapsing channels' Expiration and Calibration
    # cells, and nothing else
    expiration = table['headers'].index('Expiration')
    marked = {(r, c) for r, row in enumerate(table['marks'])
              for c, mark in enumerate(row) if mark}
    last = len(table['headers']) - 1
    assert marked == {(1, expiration), (1, last), (2, expiration), (2, last)}
    assert {mark for row in table['marks'] for mark in row} == {None, 'over'}
    assert under['html'].startswith(
        '<p><strong>Calibration on the test dates, 2026-10-24 to 2026-10-26: '
        '8 of 12 channels in date.</strong></p>')
    assert ('<li>Channel 2 (104Z+ acceleration, control) expired 2026-09-30, '
            '24 days before the test.</li>') in under['html']
    assert ('No expiration date given, so not checked: channel 4 (113Z+ '
            'acceleration, control), channel 11 (1304Z+ force).') \
        in under['html']


def test_no_marks_when_nothing_is_out_of_date(calibrating):
    project, name, _report = calibrating
    project.report_test_date(name, '2026-09-01')
    table, under = _channel_table(_payload(_render(project, name)))
    assert 'marks' not in table
    assert 'expired' not in under['html']
    # a headline and one line: the line is an item under it
    assert under['html'].endswith(
        '<ul>\n<li>No expiration date given, so not checked: channel 4 '
        '(113Z+ acceleration, control), channel 11 (1304Z+ force).</li>\n'
        '</ul>')


def test_a_lone_headline_is_a_paragraph_and_nothing_more():
    assert rc._paragraph(['All in date.']) == {'kind': 'text',
                                               'text': '**All in date.**'}
    assert rc._paragraph(['Head.', 'One.']) == {
        'kind': 'text', 'text': '**Head.**\n\n- One.'}


def test_the_offer_on_one_day_with_a_sensor_of_no_stated_quantity():
    table = _table([('2026-10-01', '', 7, False)])
    report = Report('R', [{'kind': 'table', 'source': 'Channels',
                           'caption': ''}],
                    features=['calibration_in_date'])
    objects = {'Channels': table}
    rc.set_dates(report, objects, None, '2026-10-24')
    lines = rc.draft(report, objects).splitlines()
    assert lines[0] == ('Sensors and their calibration status on the test '
                        'date, 2026-10-24, from the channel table:')
    assert lines[4] == ('| 1 | 7Z+ |  |  |  |  | 2026-10-01 | expired 23 days '
                        'before the test |')


def test_a_kept_date_or_cell_that_cannot_be_read_is_said_on_the_page(
        calibrating):
    project, name, report = calibrating
    index = rc.table_blocks(report, project, project.object_groups)[0]
    report.blocks[index][rc.START] = '10/24/2026'
    _table_block, under = _channel_table(_payload(_render(project, name)))
    assert "Calibration not checked: the test date '10/24/2026' cannot be " \
           'read one way' in under['html']
    report.blocks[index][rc.START] = '2026-10-24'
    table = next(obj for obj in project.values()
                 if isinstance(obj, ChannelTable))
    kept = table.frame.loc[5, 'expiration']
    try:
        table.frame.loc[5, 'expiration'] = 'June'
        built, under = _channel_table(_payload(_render(project, name)))
        assert 'Calibration' not in built['headers']
        assert ("Calibration not checked: channel table row 6 (channel 6), "
                "Expiration: 'June' is not a date") in under['html']
    finally:
        table.frame.loc[5, 'expiration'] = kept


def test_with_the_skeleton_it_falls_under_instrumentation(calibrating):
    project, name, _report = calibrating
    project.report_features(name, enable=['skeleton_sections'])
    project.report_test_date(name, '2026-10-24')
    blocks = _payload(_render(project, name))['blocks']
    heads = [i for i, block in enumerate(blocks)
             if block.get('html', '').startswith('<h2 class="skeleton">')]
    titles = [blocks[i]['html'].split('>', 1)[1].split('<')[0] for i in heads]
    start = heads[titles.index('6. Instrumentation')]
    stop = heads[titles.index('7. Chronology')]
    inside = blocks[start:stop]
    assert any(block['kind'] == 'table' for block in inside)
    assert any('Calibration on the test date, 2026-10-24'
               in block.get('html', '') for block in inside)


def test_set_dates_refuses_before_changing_anything(run):
    project, name = run
    report = project[name]
    with pytest.raises(ValueError, match='does not have Calibration in date'):
        project.report_test_date(name, '2026-10-24')
    try:
        project.report_features(name, enable=['calibration_in_date'])
        before = [dict(block) for block in report.blocks]
        for bad in (('10/24/2026', None), ('2026-10-24', '2026-10-20'),
                    (None, '2026-10-24')):
            with pytest.raises(ValueError):
                project.report_test_date(name, *bad)
            assert report.blocks == before, bad
        with pytest.raises(TypeError, match='not int'):
            project.report_test_date(name, 20261024)
        bare = Report('Bare', [{'kind': 'text', 'text': 'x'}],
                      features=['calibration_in_date'])
        with pytest.raises(ValueError, match='no channel table block'):
            rc.set_dates(bare, project, project.object_groups, '2026-10-24')
        # one day keeps no last day; clearing takes both away
        project.report_test_date(name, '2026-10-24', '2026-10-26')
        project.report_test_date(name, '2026-10-24', '2026-10-24')
        index = rc.table_blocks(report, project, project.object_groups)[0]
        assert report.blocks[index][rc.START] == '2026-10-24'
        assert rc.END not in report.blocks[index]
        project.report_test_date(name, None)
        assert rc.START not in report.blocks[index]
        assert rc.span_for(report.blocks[index]) is None
    finally:
        report.features = []


def test_the_verbs_are_journaled_and_the_reading_refuses_without_a_date(
        calibrating):
    project, name, _report = calibrating
    with pytest.raises(ValueError, match='has no test date'):
        project.report_calibration(name)
    before = list(project.journal)
    with pytest.raises(ValueError):
        project.report_test_date(name, '2026/10/24')
    assert project.journal == before, 'a refusal is not journaled'
    project.report_test_date(name, '2026-10-24', end='2026-10-26')
    assert project.journal[-1] == (f"project.report_test_date({name!r}, "
                                   "'2026-10-24', end='2026-10-26')")
    found = project.report_calibration(name)
    assert found.span == SPAN
    assert [c.status for c in found.channels].count(cal.IN_DATE) == 8
    assert len(project.journal) == len(before) + 1, 'a reading is not'


def test_the_setup_section_is_offered_the_equipment_list(calibrating):
    project, name, _report = calibrating
    project.report_test_date(name, '2026-10-24', '2026-10-26')
    text = project.report_calibration_text(name)
    lines = text.splitlines()
    assert lines[0] == ('Sensors and their calibration status on the test '
                        'dates, 2026-10-24 to 2026-10-26, from the channel '
                        'table:')
    assert lines[2] == ('| Channel | DOF | Quantity | Make | Model | Serial '
                        'Number | Expiration | Calibration |')
    assert lines[5] == ('| 2 | 104Z+ | acceleration |  |  |  | 2026-09-30 | '
                        'expired 24 days before the test |')
    assert lines[7] == ('| 4 | 113Z+ | acceleration |  |  |  |  | no date '
                        'given, not checked |')
    assert lines[-4].startswith('Calibration on the test dates')
    # offered, never written: the setup section is still owed
    project.report_features(name, enable=['skeleton_sections'])
    assert any(line.startswith('5. Setup and Method')
               for line in project.report_check(name))
    project.report_section(name, 'setup', text)
    assert not any(line.startswith('5. Setup and Method')
                   for line in project.report_check(name))
    page = json.dumps(_payload(_render(project, name)))
    assert '<td>expired 24 days before the test</td>' in page


def test_a_pipe_in_a_sensor_cell_stays_in_its_cell(calibrating):
    project, name, _report = calibrating
    table = next(obj for obj in project.values()
                 if isinstance(obj, ChannelTable))
    table.set_cell('model', 0, '356A|16\nrev B')
    try:
        project.report_test_date(name, '2026-10-24')
        row = project.report_calibration_text(name).splitlines()[4]
        assert '| 356A\\|16 rev B |' in row
    finally:
        table.set_cell('model', 0, '')


def test_the_dates_ride_in_the_project_file_but_not_the_template(
        calibrating, tmp_path):
    from visualdynamics import io
    from visualdynamics.io import report_template

    project, name, report = calibrating
    project.report_test_date(name, '2026-10-24', '2026-10-26')
    index = rc.table_blocks(report, project, project.object_groups)[0]
    io.save(report, str(tmp_path / 'r.vdyn'))
    back = io.load(str(tmp_path / 'r.vdyn'))
    assert back.blocks[index][rc.START] == '2026-10-24'
    assert back.blocks[index][rc.END] == '2026-10-26'
    report_template.save(report, tmp_path / 'house.vdreport')
    house = report_template.load(tmp_path / 'house.vdreport')
    assert rc.START not in house.blocks[index]
    assert rc.END not in house.blocks[index]
    assert house.features == ['calibration_in_date']
    assert house.blocks[index]['source'] == '@basis:ChannelTable'
    # the report itself keeps its dates
    assert report.blocks[index][rc.START] == '2026-10-24'


def test_a_table_block_of_something_else_is_left_alone(run):
    project, _name = run
    shapes_like = {'kind': 'table', 'source': 'nothing', 'caption': ''}
    built = {'kind': 'table', 'headers': ['a'], 'rows': [['1']]}
    assert rc.apply(built, shapes_like, project) is built
    assert rc.channel_table_of({'kind': 'text'}, project) is None


# ---- the editor ---------------------------------------------------------------


@pytest.fixture
def editing(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = window.generate_report('random')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    report = window.objects[name]
    return window, window.report_editor, report, name


def test_the_editor_sets_the_dates_on_the_channel_table(editing):
    window, editor, report, name = editing
    held = rc.table_blocks(report, window.project, window.project.object_groups)
    assert held, 'the random report shows its channel table'
    editor._operate({'op': 'select', 'at': held[0]})
    assert editor.date_edit is None, 'no dates while the feature is off'
    editor._operate({'op': 'features', 'enable': ['calibration_in_date']})
    editor._operate({'op': 'select', 'at': None})
    editor._operate({'op': 'select', 'at': held[0]})
    assert editor.date_edit is not None and editor.end_edit is not None
    editor.date_edit.setText('10/24/2026')
    editor.date_edit.editingFinished.emit()
    assert 'cannot be read one way' in editor.calibration_note.text()
    assert rc.START not in report.blocks[held[0]]
    editor.date_edit.setText('2026-10-24')
    editor.date_edit.editingFinished.emit()
    assert report.blocks[held[0]][rc.START] == '2026-10-24'
    assert editor.calibration_note.text() == ''
    assert window.project.journal[-1] == (
        f"project.report_test_date({name!r}, '2026-10-24', None)")
    # a pane of another block has no dates
    editor._operate({'op': 'select', 'at': 0})
    assert editor.date_edit is None


def test_the_setup_pane_offers_the_calibration_status(editing):
    _window, editor, report, _name = editing
    editor._operate({'op': 'features',
                     'enable': ['calibration_in_date', 'skeleton_sections']})
    editor._operate({'op': 'select', 'at': 'section:article'})
    assert editor.calibration_button is None, 'only on the setup section'
    editor._operate({'op': 'select', 'at': 'section:setup'})
    assert editor.calibration_button is not None
    editor.calibration_button.click()
    assert 'has no test date' in editor.calibration_note.text()
    assert editor.text_editor.toPlainText() == ''
    rc.set_dates(report, editor.objects(), editor.object_groups(),
                 '2026-10-24')
    editor.text_editor.setPlainText('Shaker and fixture as planned.\n\n')
    cursor = editor.text_editor.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    editor.text_editor.setTextCursor(cursor)
    editor.calibration_button.click()
    text = report.blocks[editor.selected]['text']
    assert text.startswith('Shaker and fixture as planned.\n\nSensors and '
                           'their calibration status on the test date, '
                           '2026-10-24')
    assert editor.calibration_note.text() == ''


def test_the_values_are_unchanged_by_the_check(run):
    """The check reads the table and writes nothing to it."""
    project, _name = run
    table = next(obj for obj in project.values()
                 if isinstance(obj, ChannelTable))
    before = table.frame.copy()
    cal.check(table, SPAN)
    assert table.frame.equals(before)
    assert np.array_equal(table['expiration'], np.array(EXPIRATIONS))
