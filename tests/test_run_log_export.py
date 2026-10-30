"""The run log's export views: plain text and CSV of the entries, with
corrections and late entries marked, and the machine log."""

import csv
import io
import json
from datetime import datetime, timedelta, timezone

import pytest

from visualdynamics.core import run_log_export as export
from visualdynamics.core.run_log import FileRef, RunLog, RunLogError

WHO = ('tester', 'bench')
MOUNTAIN = timezone(timedelta(hours=-6))
STANDARD = timezone(timedelta(hours=-7))
START = datetime(2026, 10, 30, 8, 0, tzinfo=MOUNTAIN)


class Clock:
    def __init__(self, now=START):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def log(tmp_path):
    """Two tests' worth of floor: a clock check, a test opened, an abort
    later corrected, a late note between tests with a file, a setting
    changed, a person in, and a second test."""
    clock = Clock()
    log = RunLog.create(tmp_path / 'a.vdlog', title='Article 7', clock=clock,
                        who=WHO)
    log.append('clock_check', author='AB',
               fields={'offset_s': 1.5, 'recorder': 'rack'})
    log.append('test_opened', author='AB', test=1,
               fields={'title': 'Random, Z'})
    clock.now = START + timedelta(minutes=5)
    abort = log.append('abort', author='AB', test=1, axis='Z', run=2,
                       fields={'stop': 'amplifier trip', 'heard': 'bang'},
                       text='trip on rise')
    clock.now = START + timedelta(minutes=30)
    log.correct(abort, author='CD',
                fields={'stop': 'power loss', 'heard': None, 'channel': '3'},
                text='read off the panel')
    log.append('note', author='AB', event_time=START + timedelta(minutes=10),
               text='late note', files=[FileRef('run2.csv', 'a' * 64, 12)])
    log.append('setting_change', author='AB', test=1,
               fields={'setting': 'gain', 'value': 2.5, 'was': 2})
    log.append('person_in', author='AB', fields={'person': 'EF'})
    log.append('test_opened', author='AB', test=2, fields={'title': 'Shock'})
    log.append('level_row', author='AB', test=2, fields={'level_db': -12.0})
    yield log
    log.close()


def rows(text):
    return list(csv.reader(io.StringIO(text)))


def test_the_text_lists_every_entry_with_its_time_author_and_place(log):
    lines = export.to_text(log).splitlines()
    assert lines[0] == 'Run log: Article 7'
    assert lines[1] == (f'Log {log.header.log_id}, run log format 1, created '
                        '2026-10-30 08:00:00 -06:00')
    assert lines[2] == '9 entries'
    assert lines[4:7] == [
        '#1  2026-10-30 08:00:00 -06:00  AB (tester on bench)  log',
        '    Clock check: offset_s: 1.5; recorder: rack',
        "    [recorders' clock 2026-10-30 08:00:01 -06:00]"]
    assert ('#3  2026-10-30 08:05:00 -06:00  AB (tester on bench)  test 1 / '
            'axis Z / run 2') in lines


def test_the_text_marks_a_corrected_entry_and_says_what_changed(log):
    lines = export.to_text(log).splitlines()
    at = lines.index('    Abort or trip: stop: amplifier trip; heard: bang')
    assert lines[at + 1] == '    "trip on rise"'
    assert lines[at + 2] == ("    [recorders' clock 2026-10-30 08:05:01 "
                             '-06:00] [CORRECTED by #4]')
    assert ('    Correction of #3: stop: power loss (was amplifier trip); '
            'heard: struck (was bang); channel: 3 (was not given)') in lines


def test_a_second_correction_shows_the_value_the_first_left(log):
    log.correct(3, author='AB', fields={'channel': '4'})
    text = export.to_text(log)
    assert 'Correction of #3: channel: 4 (was 3)' in text
    assert '[CORRECTED by #4, #10]' in text


def test_a_correction_of_its_event_time_says_the_time_it_replaced(log):
    log.correct(2, author='AB', fields={'event_time': START + timedelta(
        minutes=1), 'run': 1})
    assert ('Correction of #2: event_time: 2026-10-30T08:01:00.000-06:00 '
            '(was 2026-10-30T08:00:00.000-06:00); run: 1 (was not given)'
            ) in export.to_text(log)


def test_the_text_marks_a_late_entry_and_names_its_file(log):
    lines = export.to_text(log).splitlines()
    at = lines.index('    "late note"')
    assert lines[at + 1] == ('    file: run2.csv (SHA-256 ' + 'a' * 64
                             + ', 12 bytes)')
    assert lines[at + 2] == ("    [recorders' clock 2026-10-30 08:10:01 "
                             '-06:00] [LATE: entered 2026-10-30 08:30:00 '
                             '-06:00]')
    assert 'LATE' not in export.to_text(log, late_after=timedelta(hours=1))


def test_the_text_closes_with_the_heads_and_what_verifying_found(log):
    lines = export.to_text(log).splitlines()
    at = lines.index('Chain heads (SHA-256):')
    assert lines[at + 1:at + 4] == [f'  log: {log.head()}',
                                    f'  test 1: {log.head(1)}',
                                    f'  test 2: {log.head(2)}']
    assert lines[-1] == ('Integrity: every chain checks: 9 entries in 3 '
                         'chains. This detects accidental damage and casual '
                         'edits; it is not proof against deliberate '
                         'alteration.')


def test_the_text_of_a_damaged_log_says_where_it_broke(log, tmp_path):
    path = tmp_path / 'a.vdlog'
    lines = path.read_bytes().split(b'\n')
    lines[2] = lines[2].replace(b'Random, Z', b'Random, X')
    lines[-1] = b'{"id": 10, "kin'
    path.write_bytes(b'\n'.join(lines))
    text = export.to_text(RunLog.read(path))
    assert ('Lines not read:\n  line 11: the last line is torn: the file was '
            'cut off in the middle of an entry, as a crash leaves it. It was '
            'skipped.') in text
    assert text.splitlines()[-1].startswith(
        'Integrity: broken at test 1, entry #2 (line 3): its hash does not '
        'match')


def test_a_correction_of_an_entry_not_in_the_log_is_still_shown(tmp_path):
    path = tmp_path / 'b.vdlog'
    with RunLog.create(path, clock=Clock(), who=WHO) as log:
        log.append('note', author='AB', run=1)
        log.correct(1, author='AB', fields={'run': 2})
    lines = path.read_bytes().split(b'\n')
    lines[2] = lines[2].replace(b'"corrects": 1', b'"corrects": 9')
    path.write_bytes(b'\n'.join(lines))
    assert 'Correction of #9 (not in the log): run: 2' in export.to_text(
        RunLog.read(path))


def test_the_text_of_one_test_holds_only_that_test(log):
    one = export.to_text(log, test=1)
    assert one.splitlines()[2] == '4 entries, test 1'
    assert '#1 ' not in one and '#3 ' in one and '#8 ' not in one
    assert f'  test 1: {log.head(1)}' in one
    assert '  log: ' not in one
    between = export.to_text(log, test=None)
    assert between.splitlines()[2] == '3 entries, between tests'
    assert f'  log: {log.head()}' in between
    for wrong in (0, True, 'one', 1.0):
        with pytest.raises(RunLogError, match='selected by its number'):
            export.to_text(log, test=wrong)


def test_the_csv_has_a_row_per_stored_entry_with_its_marks(log):
    table = rows(export.to_csv(log))
    assert tuple(table[0]) == export.CSV_COLUMNS
    assert [row[0] for row in table[1:]] == [str(n) for n in range(1, 10)]
    by_id = {row[0]: dict(zip(table[0], row, strict=True))
             for row in table[1:]}
    abort, fix, note = by_id['3'], by_id['4'], by_id['5']
    assert abort['corrected_by'] == '4' and abort['corrects'] == ''
    assert fix['corrects'] == '3' and fix['kind'] == 'correction'
    assert json.loads(abort['fields']) == {'stop': 'amplifier trip',
                                           'heard': 'bang'}
    assert json.loads(fix['fields']) == {'stop': 'power loss', 'heard': None,
                                         'channel': '3'}
    assert (abort['test'], abort['axis'], abort['run']) == ('1', 'Z', '2')
    assert note['late'] == 'late' and abort['late'] == ''
    assert note['files'] == 'run2.csv ' + 'a' * 64
    assert note['event_time'] == '2026-10-30T08:10:00.000-06:00'
    assert note['recorder_time'] == '2026-10-30T08:10:01.500-06:00'
    assert note['entered_at'] == '2026-10-30T08:30:00.000-06:00'
    assert (note['author'], note['login'], note['computer']) == (
        'AB', 'tester', 'bench')
    assert note['hash'] == log.entry(5).hash
    assert note['label'] == 'Note' and note['test'] == ''
    assert [row[0] for row in rows(export.to_csv(log, test=2))[1:]] == ['8',
                                                                        '9']


def test_the_machine_log_holds_the_trips_and_machine_events_corrected(log):
    table = rows(export.machine_log_csv(log))
    assert tuple(table[0]) == export.MACHINE_COLUMNS
    events = [dict(zip(table[0], row, strict=True)) for row in table[1:]]
    assert [event['entry'] for event in events] == ['1', '2', '3', '6', '7',
                                                    '8']
    trip = events[2]
    assert trip == {
        'date': '2026-10-30', 'time': '08:05:00', 'utc_offset': '-06:00',
        'recorder_time': '2026-10-30T08:05:01.500-06:00',
        'event': 'Abort or trip',
        'detail': 'stop: power loss; channel: 3; trip on rise',
        'test': '1', 'test_title': 'Random, Z', 'axis': 'Z', 'run': '2',
        'author': 'AB', 'entry': '3', 'corrected': 'corrected'}
    assert events[3]['detail'] == 'setting: gain; value: 2.5; was: 2'
    assert events[3]['corrected'] == ''
    assert (events[4]['event'], events[4]['test'], events[4]['test_title']) \
        == ('Person in', '', '')
    assert events[5]['test_title'] == 'Shock'
    assert [row[-2] for row in rows(export.machine_log_csv(log, test=1))[1:]
            ] == ['2', '3', '6']


def test_the_machine_log_keeps_each_events_own_offset_across_a_change(
        tmp_path):
    before = datetime(2026, 11, 1, 1, 30, tzinfo=MOUNTAIN)
    after = datetime(2026, 11, 1, 1, 10, tzinfo=STANDARD)
    with RunLog.create(tmp_path / 'b.vdlog', clock=Clock(after), who=WHO) \
            as log:
        log.append('abort', author='AB', event_time=before,
                   fields={'stop': 'stop word'})
        log.append('abort', author='AB', event_time=after,
                   fields={'stop': 'power loss'})
        table = rows(export.machine_log_csv(log))
    assert [(row[0], row[1], row[2], row[3]) for row in table[1:]] == [
        ('2026-11-01', '01:30:00', '-06:00', ''),
        ('2026-11-01', '01:10:00', '-07:00', '')]


def test_an_empty_log_exports_its_headings_and_nothing_else(tmp_path):
    with RunLog.create(tmp_path / 'b.vdlog', clock=Clock(), who=WHO) as log:
        text = export.to_text(log)
        assert text.splitlines()[2] == '0 entries'
        assert 'Chain heads' not in text
        assert rows(export.to_csv(log)) == [list(export.CSV_COLUMNS)]
        assert rows(export.machine_log_csv(log)) == [
            list(export.MACHINE_COLUMNS)]
        log.append('note', author='AB')
        lines = export.to_text(log).splitlines()
        assert lines[0] == 'Run log: (untitled)'
        assert lines[2] == '1 entry'


def test_each_view_is_written_to_a_file_beside_the_log(log, tmp_path):
    written = export.export(log, tmp_path / 'out', 'text')
    assert written == str(tmp_path / 'out.txt')
    with open(written, encoding='utf-8', newline='') as handle:
        assert handle.read() == export.to_text(log)
    table = export.export(log, tmp_path / 'table', 'csv', test=1)
    assert table.endswith('table.csv')
    with open(table, encoding='utf-8', newline='') as handle:
        assert handle.read() == export.to_csv(log, test=1)
    machine = export.export(log, tmp_path / 'machine.log.csv', 'machine')
    with open(machine, encoding='utf-8', newline='') as handle:
        assert handle.read() == export.machine_log_csv(log)
    with pytest.raises(RunLogError, match="'pdf' is not a view"):
        export.export(log, tmp_path / 'out', 'pdf')
    with pytest.raises(RunLogError, match='never over it'):
        export.export(log, tmp_path / 'a.vdlog', 'csv')
    assert RunLog.read(tmp_path / 'a.vdlog').verify().ok
