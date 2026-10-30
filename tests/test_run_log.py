"""The run log file and its entries: written durably, read leniently,
corrected without editing, chained per test, and opened by one writer.

Every log here is written under `tmp_path` with a fixed login and
machine name, and most on a fixed clock, so entered-at times and the
late mark are pinned by value.
"""

import glob
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone

import pytest
from conftest import fixture_path

from visualdynamics.core import run_log
from visualdynamics.core.run_log import (
    FORMAT,
    FORMAT_VERSION,
    KINDS,
    STOPS,
    FileRef,
    RunLog,
    RunLogError,
    RunLogLocked,
)

WHO = ('tester', 'bench')
MOUNTAIN = timezone(timedelta(hours=-6))
STANDARD = timezone(timedelta(hours=-7))
START = datetime(2026, 10, 30, 8, 0, tzinfo=MOUNTAIN)


class Clock:
    """A clock that reads `now`, which a test moves."""

    def __init__(self, now=START):
        self.now = now

    def __call__(self):
        return self.now


def new_log(tmp_path, name='a.vdlog', clock=None, **kwargs):
    return RunLog.create(tmp_path / name, clock=clock or Clock(), who=WHO,
                         **kwargs)


def lines_of(path):
    return open(path, 'rb').read().split(b'\n')


def write_lines(path, lines):
    with open(path, 'wb') as handle:
        handle.write(b'\n'.join(lines))


def stored(path, line):
    """The JSON of `line` of the file (the header is line 1)."""
    return json.loads(lines_of(path)[line - 1])


def three_entries(tmp_path):
    log = new_log(tmp_path)
    log.append('test_opened', author='AB', test=1, fields={'title': 'Random'})
    log.append('level_row', author='AB', test=1, run=1,
               fields={'level_db': -12.0, 'current': 63.0})
    log.append('note', author='AB', text='between tests')
    log.close()
    return tmp_path / 'a.vdlog'


# ---- the file ---------------------------------------------------------------


def test_a_new_log_is_one_header_line_stamped_with_format_and_version(tmp_path):
    with new_log(tmp_path, title='Article 7') as log:
        assert log.entries == ()
    lines = lines_of(tmp_path / 'a.vdlog')
    assert lines[1:] == [b'']
    header = json.loads(lines[0])
    assert header['format'] == FORMAT == 'visualdynamics-run-log'
    assert header['version'] == FORMAT_VERSION == 1
    assert header['created'] == '2026-10-30T08:00:00.000-06:00'
    assert (header['title'], header['login'], header['computer']) == (
        'Article 7', 'tester', 'bench')


def test_each_entry_is_on_the_disk_and_synced_before_append_returns(
        tmp_path, monkeypatch):
    log = new_log(tmp_path)
    path = tmp_path / 'a.vdlog'
    synced = []
    real = os.fsync

    def watch(fd):
        synced.append(path.read_bytes().count(b'\n'))
        real(fd)

    monkeypatch.setattr(os, 'fsync', watch)
    log.append('note', author='AB', text='one')
    assert synced == [2]
    log.append('note', author='AB', text='two')
    assert synced == [2, 3]
    log.close()


def test_the_file_round_trips_every_value_it_holds(tmp_path):
    clock = Clock()
    with new_log(tmp_path, clock=clock) as log:
        log.append('brief', author='AB', fields={'attendees': ['AB', 'CD'],
                                                 'topics': ['stop words']})
        clock.now += timedelta(minutes=1)
        log.append('level_row', author='AB', test=1, axis='Z', run=2,
                   fields={'level_db': -6, 'voltage': 41.5, 'go': False,
                           'answer': 'as predicted'},
                   text='line one\nline two, "quoted", ünïcode',
                   files=[FileRef('run2.csv', 'a' * 64, 120)])
        log.append('clock_check', author='AB', fields={'offset_s': -0.75})
        written = log.entries
    read = RunLog.read(tmp_path / 'a.vdlog')
    assert read.entries == written
    assert read.problems == ()
    level = read.entry(2)
    assert level.fields == {'level_db': -6, 'voltage': 41.5, 'go': False,
                            'answer': 'as predicted'}
    assert isinstance(level.fields['level_db'], int)
    assert level.text == 'line one\nline two, "quoted", ünïcode'
    assert level.files == (FileRef('run2.csv', 'a' * 64, 120),)
    assert (level.test, level.axis, level.run) == (1, 'Z', 2)
    assert level.entered_at == START + timedelta(minutes=1)
    assert [entry.line for entry in read.entries] == [2, 3, 4]
    assert read.verify().ok


def test_every_entry_records_the_login_and_machine_beside_the_initials(
        tmp_path):
    with RunLog.create(tmp_path / 'a.vdlog', who=('alice', 'bench-2')) as log:
        entry = log.append('note', author=' KC ')
    assert (entry.author, entry.login, entry.computer) == (
        'KC', 'alice', 'bench-2')
    login, computer = run_log.identity()
    assert isinstance(login, str) and isinstance(computer, str)


def test_a_file_is_named_by_its_sha256(tmp_path):
    data = tmp_path / 'run.csv'
    data.write_bytes(b'time,a\n0,1\n')
    with new_log(tmp_path) as log:
        entry = log.append('note', author='AB', files=[data])
    assert entry.files == (FileRef('run.csv', hashlib.sha256(
        data.read_bytes()).hexdigest(), 11),)
    with pytest.raises(RunLogError, match='not a SHA-256'):
        FileRef('run.csv', 'ABC')
    with pytest.raises(RunLogError, match='whole number of bytes'):
        FileRef('run.csv', '0' * 64, -1)
    with pytest.raises(RunLogError, match='its name'):
        FileRef(' ', '0' * 64)
    with new_log(tmp_path, 'b.vdlog') as log, pytest.raises(
            RunLogError, match='path or a FileRef'):
        log.append('note', author='AB', files=[3])


def test_a_log_is_never_created_over_a_file(tmp_path):
    (tmp_path / 'a.vdlog').write_text('kept')
    with pytest.raises(RunLogError, match='already at'):
        new_log(tmp_path)
    assert (tmp_path / 'a.vdlog').read_text() == 'kept'
    assert not os.path.exists(tmp_path / 'a.vdlog.lock')


# ---- a crash, and a file read leniently ---------------------------------------


def test_a_torn_last_line_is_named_and_the_rest_is_read(tmp_path):
    path = three_entries(tmp_path)
    whole = lines_of(path)[3]
    with open(path, 'ab') as handle:
        handle.write(whole[:len(whole) // 2])       # killed mid-append
    read = RunLog.read(path)
    assert [entry.id for entry in read.entries] == [1, 2, 3]
    assert len(read.problems) == 1
    assert read.problems[0].line == 5
    assert 'torn' in read.problems[0].reason
    assert read.verify().ok


def test_a_log_killed_mid_append_reopens_and_goes_on(tmp_path):
    path = three_entries(tmp_path)
    with open(path, 'ab') as handle:
        handle.write(b'{"id": 4, "kind": "no')
    with RunLog.open(path, who=WHO, clock=Clock()) as log:
        assert 'torn' in log.problems[0].reason
        entry = log.append('note', author='AB', text='after the crash')
    assert entry.id == 4 and entry.line == 6
    read = RunLog.read(path)
    assert [entry.id for entry in read.entries] == [1, 2, 3, 4]
    assert [(p.line, 'not an entry' in p.reason) for p in read.problems] == [
        (5, True)]
    assert read.verify().ok


def test_a_complete_last_line_without_its_line_end_is_read(tmp_path):
    path = three_entries(tmp_path)
    path.write_bytes(path.read_bytes()[:-1])
    read = RunLog.read(path)
    assert [entry.id for entry in read.entries] == [1, 2, 3]
    assert 'no line end' in read.problems[0].reason
    with RunLog.open(path, who=WHO, clock=Clock()) as log:
        log.append('note', author='AB')
    assert RunLog.read(path).problems == ()
    assert len(lines_of(path)) == 6


def test_a_process_killed_while_writing_leaves_a_log_that_opens(tmp_path):
    path = tmp_path / 'a.vdlog'
    new_log(tmp_path).close()
    child = textwrap.dedent('''
        import sys
        from visualdynamics.core.run_log import RunLog
        log = RunLog.open(sys.argv[1], who=('child', 'bench'))
        written = 0
        while True:
            log.append('note', author='CH', text='x' * 2000)
            written += 1
            if written == 20:
                print('ready', flush=True)
    ''')
    process = subprocess.Popen([sys.executable, '-c', child, str(path)],
                               stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
    finally:
        process.kill()
        process.wait()
        process.stdout.close()
    deadline = time.monotonic() + 20
    while True:
        # the system releases a dead process's lock, though not always
        # at the instant it ends
        try:
            log = RunLog.open(path, who=WHO)
            break
        except RunLogLocked:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)
    with log:
        assert len(log.entries) >= 20
        assert log.verify().ok
        log.append('note', author='AB', text='after the kill')
    read = RunLog.read(path)
    assert read.verify().ok
    assert read.entries[-1].text == 'after the kill'


def test_a_write_that_fails_part_way_is_not_taken_and_the_next_is(
        tmp_path, monkeypatch):
    log = new_log(tmp_path)
    log.append('note', author='AB', text='first')
    real = run_log._durable_append
    calls = []

    def half(handle, data):
        if calls:
            return real(handle, data)
        calls.append(data)
        handle.write(data[:len(data) // 2])
        handle.flush()
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(run_log, '_durable_append', half)
    with pytest.raises(OSError, match='No space'):
        log.append('note', author='AB', text='lost')
    monkeypatch.undo()
    entry = log.append('note', author='AB', text='second')
    log.close()
    assert [e.text for e in log.entries] == ['first', 'second']
    assert entry.id == 2
    read = RunLog.read(tmp_path / 'a.vdlog')
    assert [e.text for e in read.entries] == ['first', 'second']
    assert len(read.problems) == 1
    assert read.verify().ok


def test_a_line_that_is_not_an_entry_is_skipped_and_named(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    unknown = json.loads(lines[2])
    unknown['kind'] = 'teleport'
    lines[2] = json.dumps(unknown).encode()
    lines.insert(3, b'[1, 2]')
    lines.insert(3, b'   ')
    write_lines(path, lines)
    read = RunLog.read(path)
    assert [entry.id for entry in read.entries] == [1, 3]
    reasons = {problem.line: problem.reason for problem in read.problems}
    assert "'teleport' is not one this version knows" in reasons[3]
    assert 'not one JSON object' in reasons[5]
    assert 4 not in reasons                       # a blank line is passed over
    # the unreadable entry is still on its chain, and its edit is found
    assert read.verify().first.entry == 2


def test_a_second_entry_with_an_id_already_read_is_named(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    lines.insert(4, lines[3])
    write_lines(path, lines)
    read = RunLog.read(path)
    assert [entry.id for entry in read.entries] == [1, 2, 3]
    assert 'a second entry #3' in read.problems[0].reason
    assert not read.verify().ok


def test_a_file_that_is_not_a_run_log_is_refused(tmp_path):
    other = tmp_path / 'other.vdlog'
    other.write_text('time,a\n0,1\n')
    with pytest.raises(RunLogError, match='not a Visual Dynamics run log'):
        RunLog.read(other)
    other.write_text('{"format": "something else"}\n')
    with pytest.raises(RunLogError, match="no 'visualdynamics-run-log' stamp"):
        RunLog.read(other)
    other.write_text('{"format": "visualdynamics-run-log", "ver')
    with pytest.raises(RunLogError, match='first line is torn'):
        RunLog.read(other)
    other.write_text('{"format": "visualdynamics-run-log"}\n')
    with pytest.raises(RunLogError, match='no format version'):
        RunLog.read(other)
    with pytest.raises(FileNotFoundError):
        RunLog.read(tmp_path / 'missing.vdlog')
    with pytest.raises(FileNotFoundError):
        RunLog.open(tmp_path / 'missing.vdlog')


# ---- empty logs -------------------------------------------------------------


def test_an_empty_log_has_no_entries_and_checks(tmp_path):
    log = new_log(tmp_path)
    assert log.entries == () and log.view() == [] and log.tests() == []
    assert log.chains() == []
    assert log.verify().ok
    assert str(log.verify()).startswith('every chain checks: 0 entries in 0 '
                                        'chains')
    assert log.recorder_time(START) is None
    log.close()
    read = RunLog.read(tmp_path / 'a.vdlog')
    assert read.entries == () and read.problems == ()
    assert read.head() == log.head()


def test_a_zero_byte_log_is_refused_read_only_and_given_its_header_to_write(
        tmp_path):
    path = tmp_path / 'a.vdlog'
    path.write_bytes(b'')
    with pytest.raises(RunLogError, match='is empty'):
        RunLog.read(path)
    with RunLog.open(path, who=WHO, clock=Clock()) as log:
        assert 'header was written' in log.problems[0].reason
        log.append('note', author='AB')
    assert [entry.id for entry in RunLog.read(path).entries] == [1]


def test_an_empty_test_has_its_own_starting_head(tmp_path):
    with new_log(tmp_path) as log:
        heads = {log.head(None), log.head(5), log.head(6)}
        assert len(heads) == 3
        assert all(len(head) == 64 for head in heads)
        empty = log.head(5)
        first = log.append('note', author='AB', test=5)
        assert first.prev == empty
        assert log.head(5) == first.hash
        assert log.head(6) in heads


# ---- the chain ----------------------------------------------------------------


def test_each_test_and_the_log_between_tests_have_their_own_chain(tmp_path):
    with new_log(tmp_path) as log:
        a = log.append('note', author='AB', test=1)
        b = log.append('note', author='AB')
        start = log.head(2)
        c = log.append('note', author='AB', test=2)
        assert c.prev == start != a.hash
        d = log.append('note', author='AB', test=1)
        e = log.append('note', author='AB')
        assert d.prev == a.hash and e.prev == b.hash
        assert log.chains() == ['test 1', 'log', 'test 2']
        assert log.tests() == [1, 2]
        assert (a.chain, b.chain) == ('test 1', 'log')
        assert log.head(1) == d.hash and log.head() == e.hash
        assert d.hash == hashlib.sha256(run_log.canonical(
            d.to_dict()).encode()).hexdigest()


def test_an_edited_entry_is_found_by_test_and_entry(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    lines[2] = lines[2].replace(b'-12.0', b'-6.0')
    write_lines(path, lines)
    found = RunLog.read(path).verify()
    assert not found.ok
    assert (found.first.chain, found.first.test, found.first.entry,
            found.first.line) == ('test 1', 1, 2, 3)
    assert 'changed after it was written' in found.first.reason
    assert str(found) == ('broken at test 1, entry #2 (line 3): its hash does '
                          'not match its contents: it was changed after it '
                          'was written')
    assert len(found.breaks) == 1                    # no cascade


def test_a_removed_entry_is_found_by_the_entry_after_it(tmp_path):
    path = three_entries(tmp_path)
    with RunLog.open(path, who=WHO, clock=Clock()) as log:
        log.append('note', author='AB', test=1)
    lines = lines_of(path)
    del lines[2]                                     # entry #2, test 1
    write_lines(path, lines)
    found = RunLog.read(path).verify()
    reasons = [(b.chain, b.entry, b.reason) for b in found.breaks]
    assert ('log', 3, 'the ids jump: entry #2 is missing') in reasons
    assert any(chain == 'test 1' and entry == 4 and 'previous hash' in reason
               for chain, entry, reason in reasons)


def test_ids_out_of_order_are_found(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    lines[2], lines[3] = lines[3], lines[2]
    write_lines(path, lines)
    found = RunLog.read(path).verify()
    assert 'the ids jump: entry #2 is missing' in found.breaks[0].reason
    assert any(b.entry == 2 and 'out of order (#4 was next)' in b.reason
               for b in found.breaks)


def test_an_edited_header_breaks_every_chain(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    lines[0] = lines[0].replace(b'"title": ""', b'"title": "another"')
    write_lines(path, lines)
    found = RunLog.read(path).verify()
    assert {b.chain for b in found.breaks} == {'test 1', 'log'}


def test_a_file_cut_at_an_entry_end_is_found_only_against_a_recorded_head(
        tmp_path):
    path = three_entries(tmp_path)
    log = RunLog.read(path)
    heads = {1: log.head(1), None: log.head()}
    lines = lines_of(path)
    write_lines(path, lines[:3] + [b''])             # entry #3 cut off
    cut = RunLog.read(path)
    assert cut.problems == ()
    assert cut.verify().ok                           # the claim's limit
    found = cut.verify(heads=heads)
    assert [(b.chain, b.entry) for b in found.breaks] == [('log', None)]
    assert 'cut short' in found.first.reason
    # a head recorded before the log grew is still found
    with RunLog.open(path, who=WHO, clock=Clock()) as log:
        log.append('note', author='AB', test=1)
        assert log.verify(heads={1: heads[1]}).ok
        # the starting head of a chain with nothing in it yet is in it
        assert log.verify(heads={7: log.head(7)}).ok


def test_a_file_cut_mid_entry_is_reported_on_opening(tmp_path):
    path = three_entries(tmp_path)
    path.write_bytes(path.read_bytes()[:-20])
    read = RunLog.read(path)
    assert 'torn' in read.problems[0].reason
    assert [entry.id for entry in read.entries] == [1, 2]


def test_an_edit_whose_hashes_are_computed_again_is_not_detected(tmp_path):
    # The documented limit: the chain is not proof against deliberate
    # alteration.
    path = three_entries(tmp_path)
    lines = lines_of(path)
    entry = json.loads(lines[2])
    entry['fields']['level_db'] = -3.0
    entry['hash'] = hashlib.sha256(
        run_log.canonical(entry).encode()).hexdigest()
    lines[2] = json.dumps(entry).encode()
    write_lines(path, lines)
    read = RunLog.read(path)
    assert read.entry(2).fields['level_db'] == -3.0
    assert read.verify().ok


def test_one_tests_chain_checks_alone(tmp_path):
    path = three_entries(tmp_path)
    with RunLog.open(path, who=WHO, clock=Clock()) as log:
        log.append('note', author='AB', test=2)
        log.append('note', author='AB', test=1)
    lines = lines_of(path)
    extract = tmp_path / 'test1.vdlog'
    kept = [line for line in lines[1:]
            if line and json.loads(line)['test'] == 1]
    write_lines(extract, [lines[0], *kept, b''])
    alone = RunLog.read(extract)
    assert alone.verify(tests=[1]).ok
    assert not alone.verify().ok                     # the ids jump


def test_verify_says_what_the_check_is_and_is_not(tmp_path):
    with new_log(tmp_path) as log:
        log.append('note', author='AB')
        words = str(log.verify())
    assert words == ('every chain checks: 1 entry in 1 chain. This detects '
                     'accidental damage and casual edits; it is not proof '
                     'against deliberate alteration.')


# ---- one writer ---------------------------------------------------------------


def test_a_second_writer_is_refused_naming_who_holds_the_log(tmp_path):
    first = RunLog.create(tmp_path / 'a.vdlog', who=('alice', 'bench-2'))
    first.append('note', author='AL')
    with pytest.raises(RunLogLocked, match='open for writing by alice on '
                       'bench-2'):
        RunLog.open(tmp_path / 'a.vdlog', who=WHO)
    reader = RunLog.read(tmp_path / 'a.vdlog')
    assert [entry.author for entry in reader.entries] == ['AL']
    with pytest.raises(RunLogError, match='not open for writing'):
        reader.append('note', author='AB')
    first.close()
    assert not os.path.exists(tmp_path / 'a.vdlog.lock')
    with RunLog.open(tmp_path / 'a.vdlog', who=WHO) as second:
        second.append('note', author='AB')
    with pytest.raises(RunLogError, match='not open for writing'):
        second.append('note', author='AB')


def test_a_lock_file_left_by_a_crash_does_not_hold_the_log(tmp_path):
    path = three_entries(tmp_path)
    (tmp_path / 'a.vdlog.lock').write_text(
        '{"login": "gone", "computer": "x", "pid": 1, "since": "then"}')
    with RunLog.open(path, who=WHO) as log:
        assert len(log.entries) == 3


def test_a_lock_that_does_not_say_who_holds_it_is_still_refused(tmp_path):
    with new_log(tmp_path):
        held = tmp_path / 'a.vdlog.lock'
        with open(held, 'r+b') as handle:
            handle.write(b'not json')
        with pytest.raises(RunLogLocked, match='does not say who'):
            RunLog.open(tmp_path / 'a.vdlog', who=WHO)


# ---- the mirror -------------------------------------------------------------


def test_every_entry_is_mirrored_to_the_second_path(tmp_path):
    mirror = tmp_path / 'share' / 'a.vdlog'
    mirror.parent.mkdir()
    with new_log(tmp_path, mirror=mirror) as log:
        assert mirror.read_bytes() == (tmp_path / 'a.vdlog').read_bytes()
        log.append('note', author='AB', text='one')
        log.append('note', author='AB', text='two')
        assert log.mirror_problem == ''
    assert mirror.read_bytes() == (tmp_path / 'a.vdlog').read_bytes()
    assert RunLog.read(mirror).verify().ok


def test_a_mirror_that_fails_is_reported_and_the_entry_is_kept(tmp_path):
    mirror = tmp_path / 'share' / 'a.vdlog'
    log = new_log(tmp_path, mirror=mirror)
    assert 'was not written' in log.mirror_problem
    entry = log.append('note', author='AB', text='kept')
    assert str(mirror) in log.mirror_problem
    assert str(tmp_path / 'a.vdlog') in log.mirror_problem
    assert RunLog.read(tmp_path / 'a.vdlog').entries == (entry,)
    mirror.parent.mkdir()                            # the share comes back
    log.append('note', author='AB', text='caught up')
    assert log.mirror_problem == ''
    assert mirror.read_bytes() == (tmp_path / 'a.vdlog').read_bytes()
    log.close()


def test_a_mirror_that_drops_out_between_entries_catches_up(tmp_path):
    mirror = tmp_path / 'share' / 'a.vdlog'
    mirror.parent.mkdir()
    log = new_log(tmp_path, mirror=mirror)
    log.append('note', author='AB', text='one')
    mirror.unlink()                                  # the share goes away
    os.rmdir(mirror.parent)
    log.append('note', author='AB', text='two')
    assert 'was not written' in log.mirror_problem
    mirror.parent.mkdir()
    log.append('note', author='AB', text='three')
    assert log.mirror_problem == ''
    assert [e.text for e in RunLog.read(mirror).entries] == ['one', 'two',
                                                             'three']
    # a mirror changed behind the writer's back is written again whole
    with open(mirror, 'ab') as handle:
        handle.write(b'junk\n')
    log.append('note', author='AB', text='four')
    assert mirror.read_bytes() == (tmp_path / 'a.vdlog').read_bytes()
    log.close()


def test_a_mirror_is_never_the_log_itself(tmp_path):
    with pytest.raises(RunLogError, match='second path'):
        new_log(tmp_path, mirror=tmp_path / 'a.vdlog')
    assert not os.path.exists(tmp_path / 'a.vdlog.lock')


# ---- versions -----------------------------------------------------------------

CORPUS = fixture_path('vdlog_corpus')


@pytest.mark.parametrize('name', sorted(
    os.path.basename(p) for p in glob.glob(str(CORPUS) + '/*.vdlog')))
def test_every_frozen_version_still_opens_and_checks(name):
    log = RunLog.read(os.path.join(str(CORPUS), name))
    assert log.version == int(name[1:-len('.vdlog')])
    assert len(log.entries) > 0
    assert log.problems == ()
    assert log.verify().ok


def test_every_version_has_a_reader_and_a_frozen_file():
    versions = range(1, FORMAT_VERSION + 1)
    assert sorted(run_log._READERS) == list(versions)
    for version in versions:
        assert os.path.isfile(os.path.join(str(CORPUS), f'v{version}.vdlog'))


def test_version_1_values_are_intact():
    log = RunLog.read(os.path.join(str(CORPUS), 'v1.vdlog'))
    assert log.header.title == 'Corpus article'
    assert log.header.created == datetime(2026, 10, 30, 8, 0,
                                          tzinfo=MOUNTAIN)
    assert [entry.kind for entry in log.entries] == [
        'clock_check', 'person_in', 'test_opened', 'level_row', 'abort',
        'correction', 'deviation', 'note', 'test_closed', 'test_opened',
        'shot']
    level = log.entry(4)
    assert level.fields == {'level_db': -12.0, 'voltage': 20.0,
                            'current': 63.0, 'drive_fraction': 0.25,
                            'answer': 'as predicted', 'go': True}
    assert (level.test, level.axis, level.run) == (1, 'Z', 1)
    assert log.current(5).fields == {'stop': 'amplifier trip',
                                     'channel': '4'}
    assert log.entry(7).fields['number'] == 'D-1'
    assert log.entry(8).text == 'Résumé: ünïcode kept'
    assert log.entry(8).files == (FileRef('run2.csv', '0' * 64, 12),)
    assert log.recorder_time(log.entry(4)) == (
        log.entry(4).event_time + timedelta(seconds=1.25))
    assert (log.entry(1).login, log.entry(1).computer) == ('tester', 'bench')
    assert log.chains() == ['log', 'test 1', 'test 2']


def test_a_log_from_a_newer_version_is_refused_with_a_message_to_update(
        tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    lines[0] = lines[0].replace(b'"version": 1', b'"version": 2')
    write_lines(path, lines)
    with pytest.raises(RunLogError, match='newer Visual Dynamics.*format 2; '
                       'this build reads up to 1.*Update'):
        RunLog.read(path)
    lines[0] = lines[0].replace(b'"version": 2', b'"version": 0')
    write_lines(path, lines)
    with pytest.raises(RunLogError, match='format 0 is not a version'):
        RunLog.read(path)


def test_an_older_version_is_read_but_never_added_to(tmp_path, monkeypatch):
    path = three_entries(tmp_path)
    monkeypatch.setattr(run_log, 'FORMAT_VERSION', 2)
    monkeypatch.setitem(run_log._READERS, 2, run_log._READERS[1])
    assert [e.id for e in RunLog.read(path).entries] == [1, 2, 3]
    with pytest.raises(RunLogError, match='version 1; this build adds '
                       'entries only to version 2'):
        RunLog.open(path, who=WHO)
    assert not os.path.exists(str(path) + '.lock')


def test_a_header_without_its_id_is_refused(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    header = json.loads(lines[0])
    del header['log_id']
    lines[0] = json.dumps(header).encode()
    write_lines(path, lines)
    with pytest.raises(RunLogError, match='no log id'):
        RunLog.read(path)


# ---- the kinds of entry -----------------------------------------------------


def test_the_registry_names_a_type_for_every_field_and_a_label_for_every_kind():
    for kind in KINDS.values():
        assert kind.label and kind.name in KINDS
        for name in kind.fields:
            assert name in run_log.FIELD_TYPES, (kind.name, name)
        assert not set(kind.required) & set(kind.optional)
    assert {name for name, kind in KINDS.items() if kind.machine} == {
        'test_opened', 'test_closed', 'clock_check', 'person_in',
        'person_out', 'abort', 'restart', 'setting_change'}
    assert {name for name, kind in KINDS.items() if kind.needs_test} == {
        'test_opened', 'test_closed'}


def test_an_abort_needs_only_its_time_kind_of_stop_and_author(tmp_path):
    with new_log(tmp_path) as log:
        entry = log.append('abort', author='AB',
                           fields={'stop': 'amplifier trip', 'channel': '',
                                   'heard': None})
        assert entry.fields == {'stop': 'amplifier trip'}
        assert (entry.test, entry.run) == (None, None)
        assert KINDS['abort'].required == ('stop',)
        with pytest.raises(RunLogError, match='kind of stop: one of '
                           'controller ramp, amplifier trip, power loss, '
                           'stop word'):
            log.append('abort', author='AB')
        with pytest.raises(RunLogError, match="'bang' is not a kind of stop"):
            log.append('abort', author='AB', fields={'stop': 'bang'})
    assert STOPS == ('controller ramp', 'amplifier trip', 'power loss',
                     'stop word')


def test_what_an_entry_cannot_hold_is_refused_naming_the_field(tmp_path):
    log = new_log(tmp_path)
    cases = [
        (('teleport', {}), "'teleport' is not a kind of entry"),
        (('note', {'stop': 'x'}), ("a Note entry has no field 'stop'; its "
                                  'fields are: none')),
        (('clock_check', {'offset_s': '1.5'}), ('Clock check: offset_s is a '
                                               'number')),
        (('clock_check', {'offset_s': float('nan')}), 'not a finite number'),
        (('clock_check', {'offset_s': True}), 'is a number'),
        (('clock_check', {}), 'a Clock check entry needs its offset_s'),
        (('sweep', {'n': 0}), 'Sweep: n is a whole number from 1'),
        (('sweep', {'of': 2.0}), 'Sweep: of is a whole number from 1'),
        (('level_row', {'go': 'yes'}), 'Level: go is yes or no'),
        (('brief', {'attendees': 'AB'}), ('Brief: attendees is a list of '
                                         'texts')),
        (('person_in', {'person': 3}), 'Person in: person is text'),
        (('setting_change', {'setting': 'gain', 'value': [1]}),
         'Setting changed: value is text, a number or yes or no'),
        (('setting_change', {'setting': 'gain', 'value': float('inf')}),
         'not a finite number'),
        (('cause_found', {'cause': 'x', 'refers_to': 9}),
         'refers_to names entry #9, which the log does not hold'),
    ]
    for (kind, fields), message in cases:
        with pytest.raises(RunLogError, match=message):
            log.append(kind, author='AB', fields=fields)
    with pytest.raises(RunLogError, match='fields are a mapping'):
        log.append('note', author='AB', fields=['x'])
    with pytest.raises(RunLogError, match="author's initials"):
        log.append('note', author=' ')
    with pytest.raises(RunLogError, match='free text is text'):
        log.append('note', author='AB', text=3)
    with pytest.raises(RunLogError, match='has no time zone'):
        log.append('note', author='AB', event_time=datetime(2026, 10, 30))
    with pytest.raises(RunLogError, match='Test opened entry needs the test'):
        log.append('test_opened', author='AB')
    with pytest.raises(RunLogError, match='number from 1'):
        log.append('note', author='AB', test=0)
    with pytest.raises(RunLogError, match='number from 1'):
        log.append('note', author='AB', run=True)
    with pytest.raises(RunLogError, match='axis is referred to by its name'):
        log.append('note', author='AB', axis=' ')
    with pytest.raises(RunLogError, match='written with correct'):
        log.append('correction', author='AB')
    assert log.entries == ()                         # nothing half-written
    log.close()


def test_a_value_of_each_type_is_kept_as_given(tmp_path):
    with new_log(tmp_path) as log:
        first = log.append('cause_found', author='AB', fields={'cause': 'x'})
        entry = log.append('path_decided', author='AB',
                           fields={'path': 'restart', 'refers_to': first.id})
        assert entry.fields == {'path': 'restart', 'refers_to': 1}
        reading = log.append('checklist_reading', author='AB',
                             fields={'item': 'coolant', 'reading': 31.5})
        assert reading.fields['reading'] == 31.5
        flagged = log.append('notch', author='AB', fields={'planned': False,
                                                           'depth_db': -8})
        assert flagged.fields == {'planned': False, 'depth_db': -8}


def test_an_entry_can_be_entered_after_its_event_and_is_then_late(tmp_path):
    clock = Clock(START + timedelta(minutes=30))
    with new_log(tmp_path, clock=clock) as log:
        late = log.append('note', author='AB', event_time=START)
        prompt = log.append('note', author='AB',
                            event_time=START + timedelta(minutes=20))
        exactly = log.append('note', author='AB',
                             event_time=START + timedelta(minutes=20))
    assert late.is_late() and not prompt.is_late()
    assert not exactly.is_late(timedelta(minutes=10))
    assert exactly.is_late(timedelta(minutes=9, seconds=59))
    assert late.entered_at - late.event_time == timedelta(minutes=30)


# ---- corrections --------------------------------------------------------------


def test_a_correction_leaves_the_entry_and_the_view_shows_it_corrected(
        tmp_path):
    with new_log(tmp_path) as log:
        abort = log.append('abort', author='AB', test=1, run=3,
                           fields={'stop': 'amplifier trip', 'heard': 'bang'})
        fix = log.correct(abort, author='CD',
                          fields={'stop': 'power loss', 'heard': None,
                                  'channel': '2'},
                          text='read off the panel')
        assert log.entry(abort.id) == abort           # stored, unchanged
        current = log.current(abort.id)
        assert current.fields == {'stop': 'power loss', 'channel': '2'}
        assert current.corrected_by == (fix.id,)
        assert (fix.kind, fix.corrects, fix.test, fix.run) == (
            'correction', abort.id, 1, 3)
        assert fix.prev == abort.hash                 # in test 1's chain
        assert [entry.id for entry in log.view()] == [abort.id]
        assert log.corrections_of(abort.id) == (fix,)
    read = RunLog.read(tmp_path / 'a.vdlog')
    assert read.current(abort.id).fields == current.fields
    assert read.entry(abort.id).fields == {'stop': 'amplifier trip',
                                           'heard': 'bang'}


def test_corrections_apply_in_order_and_may_move_time_text_and_references(
        tmp_path):
    with new_log(tmp_path) as log:
        entry = log.append('level_row', author='AB', test=1, axis='Z', run=4,
                           fields={'level_db': -12.0}, text='first')
        log.correct(entry.id, author='AB', fields={'level_db': -9.0})
        later = START + timedelta(minutes=5)
        log.correct(entry.id, author='AB',
                    fields={'level_db': -6.0, 'event_time': later,
                            'text': 'second', 'run': 5, 'axis': 'X',
                            'test': 2})
        current = log.current(entry.id)
    assert current.fields == {'level_db': -6.0}
    assert (current.event_time, current.text, current.test, current.axis,
            current.run) == (later, 'second', 2, 'X', 5)
    assert current.corrected_by == (2, 3)


def test_what_a_correction_cannot_do_is_refused(tmp_path):
    log = new_log(tmp_path)
    abort = log.append('abort', author='AB', test=1,
                       fields={'stop': 'stop word'})
    opened = log.append('test_opened', author='AB', test=1)
    deviation = log.append('deviation', author='AB')
    fix = log.correct(abort, author='AB', fields={'channel': '1'})
    cases = [
        ((99, {'channel': '1'}), 'no entry #99 to correct'),
        ((fix.id, {'channel': '2'}), ('is a correction; correct the original, '
                                     f'#{abort.id}, again')),
        ((abort.id, {}), 'needs the values it corrects'),
        ((abort.id, {'stop': None}), 'kind of stop'),
        ((abort.id, {'stop': 'bang'}), 'not a kind of stop'),
        ((abort.id, {'wing': 'x'}), ("no field 'wing'; a correction may "
                                    'change: stop, channel')),
        ((abort.id, {'event_time': datetime(2026, 1, 1)}), 'no time zone'),
        ((abort.id, {'text': 3}), 'free text is text'),
        ((abort.id, {'run': 0}), 'number from 1'),
        ((abort.id, {'channel': 3}), 'Abort or trip: channel is text'),
        ((deviation.id, {'number': 'D-5'}), "number is never changed"),
        ((opened.id, {'test': None}), 'needs the test'),
        ((abort.id, {'refers_to': 1}), "no field 'refers_to'"),
    ]
    for (target, fields), message in cases:
        with pytest.raises(RunLogError, match=message):
            log.correct(target, author='AB', fields=fields)
    with pytest.raises(RunLogError, match='mapping'):
        log.correct(abort.id, author='AB', fields=[('channel', '1')])
    cause = log.append('cause_found', author='AB', fields={'cause': 'x'})
    with pytest.raises(RunLogError, match='names entry #77'):
        log.correct(cause, author='AB', fields={'refers_to': 77})
    assert log.correct(cause, author='AB',
                       fields={'refers_to': abort.id}).fields == {
        'refers_to': abort.id}
    log.close()


def test_a_correction_with_an_unreadable_value_leaves_the_stored_one(tmp_path):
    path = tmp_path / 'a.vdlog'
    with new_log(tmp_path) as log:
        entry = log.append('note', author='AB', run=2)
        log.correct(entry, author='AB', fields={'run': 3})
    lines = lines_of(path)
    lines[2] = lines[2].replace(b'"run": 3', b'"run": "three"', 1)
    write_lines(path, lines)
    read = RunLog.read(path)
    assert read.current(1).run == 2
    assert read.current(1).corrected_by == (2,)


# ---- numbering ----------------------------------------------------------------


def test_deviations_and_anomalies_are_numbered_across_the_log(tmp_path):
    with new_log(tmp_path) as log:
        assert log.append('deviation', author='AB', test=1).fields == {
            'number': 'D-1'}
        assert log.append('deviation', author='AB', test=2).fields[
            'number'] == 'D-2'
        assert log.append('anomaly_opened', author='AB', test=2).fields[
            'number'] == 'A-1'
        assert log.append('anomaly_opened', author='AB',
                          fields={'number': 'A-4'}).fields['number'] == 'A-4'
        assert log.next_number('anomaly') == 'A-5'
        with pytest.raises(RunLogError, match='A-4 is already taken in this '
                           'log; the next is A-5'):
            log.append('anomaly_opened', author='AB', fields={'number': 'A-4'})
        with pytest.raises(RunLogError, match='not the number of a deviation'):
            log.append('deviation', author='AB', fields={'number': 'A-9'})
        with pytest.raises(RunLogError, match='not the number of an anomaly'):
            log.append('anomaly_opened', author='AB', fields={'number': 'D-9'})
        with pytest.raises(RunLogError, match='not a finding number'):
            log.append('deviation', author='AB', fields={'number': 'D-0'})
        done = log.append('anomaly_dispositioned', author='AB',
                          fields={'number': 'A-1', 'disposition': 'use as is'})
        assert done.fields['number'] == 'A-1'
        for number in ('A-2', 'D-1'):
            with pytest.raises(RunLogError, match=f'{number} is not an anomaly '
                               'opened'):
                log.append('anomaly_dispositioned', author='AB',
                           fields={'number': number})


def test_a_number_spoiled_in_the_file_is_passed_over_when_numbering(tmp_path):
    path = tmp_path / 'a.vdlog'
    with new_log(tmp_path) as log:
        log.append('deviation', author='AB')
        log.append('deviation', author='AB')
    lines = lines_of(path)
    lines[2] = lines[2].replace(b'"D-2"', b'"D-x"')
    write_lines(path, lines)
    assert RunLog.read(path).next_number('deviation') == 'D-2'


# ---- times and the clock --------------------------------------------------------


def test_event_times_keep_their_offset_across_a_daylight_saving_change(
        tmp_path):
    # 01:30 on the first Sunday of November happens twice on a mountain
    # clock: once at UTC-6, and again an hour later at UTC-7.
    before = datetime(2026, 11, 1, 1, 30, tzinfo=MOUNTAIN)
    after = datetime(2026, 11, 1, 1, 10, tzinfo=STANDARD)
    clock = Clock(datetime(2026, 11, 1, 1, 15, tzinfo=STANDARD))
    with new_log(tmp_path, clock=clock) as log:
        log.append('clock_check', author='AB', event_time=before,
                   fields={'offset_s': 2.0})
        first = log.append('note', author='AB', event_time=before)
        second = log.append('note', author='AB', event_time=after)
    read = RunLog.read(tmp_path / 'a.vdlog')
    first, second = read.entry(first.id), read.entry(second.id)
    assert first.event_time.utcoffset() == timedelta(hours=-6)
    assert second.event_time.utcoffset() == timedelta(hours=-7)
    assert second.event_time - first.event_time == timedelta(minutes=40)
    # entered 01:15 standard is 45 minutes after 01:30 daylight
    assert first.is_late() and not second.is_late()
    assert read.recorder_time(second) == after + timedelta(seconds=2)
    assert read.clock_check_for(second).id == 1


def test_the_clock_check_in_force_is_the_latest_before_across_days(tmp_path):
    day1 = datetime(2026, 10, 30, 23, 59, 50, tzinfo=MOUNTAIN)
    day2 = datetime(2026, 10, 31, 7, 0, tzinfo=MOUNTAIN)
    with new_log(tmp_path, clock=Clock(day2 + timedelta(hours=2))) as log:
        early = log.append('note', author='AB', event_time=day1 -
                           timedelta(hours=1))
        log.append('clock_check', author='AB', event_time=day1,
                   fields={'offset_s': 1.5, 'recorder': 'rack'})
        log.append('clock_check', author='AB', event_time=day2,
                   fields={'offset_s': -0.25, 'recorder': 'rack'})
        log.append('clock_check', author='AB',
                   event_time=day2 + timedelta(minutes=1),
                   fields={'offset_s': 9.0, 'recorder': 'scope'})
        late_night = log.append('note', author='AB',
                                event_time=day1 + timedelta(seconds=20))
        morning = log.append('note', author='AB',
                             event_time=day2 + timedelta(hours=1))
        assert log.recorder_time(early) is None
        assert log.clock_check_for(early) is None
        assert log.recorder_time(late_night) == (
            late_night.event_time + timedelta(seconds=1.5))
        assert log.recorder_time(morning) == (
            morning.event_time + timedelta(seconds=9.0))
        assert log.recorder_time(morning, recorder='rack') == (
            morning.event_time - timedelta(seconds=0.25))
        assert log.recorder_time(morning, recorder='nobody') is None
        # a check is in force at its own time
        assert log.recorder_time(day2) == day2 - timedelta(seconds=0.25)
        assert log.recorder_time(day2 - timedelta(microseconds=1)) == (
            day2 - timedelta(microseconds=1) + timedelta(seconds=1.5))
        with pytest.raises(RunLogError, match='has no time zone'):
            log.recorder_time(datetime(2026, 10, 31, 8, 0))


def test_a_corrected_clock_check_maps_with_its_corrected_offset(tmp_path):
    with new_log(tmp_path, clock=Clock(START + timedelta(hours=1))) as log:
        check = log.append('clock_check', author='AB', event_time=START,
                           fields={'offset_s': 1.0})
        later = log.append('note', author='AB',
                           event_time=START + timedelta(minutes=1))
        log.correct(check, author='AB', fields={'offset_s': 3.0})
        assert log.recorder_time(later) == later.event_time + timedelta(
            seconds=3)
        # moved after the note, it no longer covers it
        log.correct(check, author='AB',
                    fields={'event_time': START + timedelta(minutes=2)})
        assert log.recorder_time(later) is None
        # an entry from elsewhere is mapped by its own time
        stranger = run_log.Entry(id=99, kind='note', event_time=START +
                                 timedelta(minutes=3), entered_at=START,
                                 author='AB')
        assert log.recorder_time(stranger) == stranger.event_time + timedelta(
            seconds=3)


def test_two_clock_checks_at_one_time_use_the_later_entry(tmp_path):
    with new_log(tmp_path) as log:
        log.append('clock_check', author='AB', event_time=START,
                   fields={'offset_s': 1.0})
        log.append('clock_check', author='AB', event_time=START,
                   fields={'offset_s': 2.0})
        assert log.clock_check_for(START).id == 2


# ---- edges the mutation run found ---------------------------------------------


def test_entries_written_in_a_session_carry_their_lines_in_the_file(tmp_path):
    with new_log(tmp_path) as log:
        written = [log.append('note', author='AB') for _ in range(3)]
    assert [entry.line for entry in written] == [2, 3, 4]
    assert repr(log).endswith(', 3 entries)')
    with new_log(tmp_path, 'b.vdlog') as one:
        one.append('note', author='AB')
        assert repr(one).endswith(', 1 entry)')


def test_a_write_that_fails_before_writing_leaves_no_blank_line(
        tmp_path, monkeypatch):
    log = new_log(tmp_path)
    log.append('note', author='AB', text='first')
    real = run_log._durable_append
    calls = []

    def refuse(handle, data):
        if calls:
            return real(handle, data)
        calls.append(data)
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(run_log, '_durable_append', refuse)
    with pytest.raises(OSError):
        log.append('note', author='AB', text='lost')
    monkeypatch.undo()
    entry = log.append('note', author='AB', text='second')
    log.close()
    assert entry.line == 3
    assert lines_of(tmp_path / 'a.vdlog')[3] == b''
    assert len(lines_of(tmp_path / 'a.vdlog')) == 4


def test_after_a_torn_write_the_next_entry_has_its_true_line(
        tmp_path, monkeypatch):
    log = new_log(tmp_path)
    log.append('note', author='AB')
    real = run_log._durable_append
    calls = []

    def half(handle, data):
        if calls:
            return real(handle, data)
        calls.append(data)
        handle.write(data[:10])
        handle.flush()
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(run_log, '_durable_append', half)
    with pytest.raises(OSError):
        log.append('note', author='AB')
    monkeypatch.undo()
    entry = log.append('note', author='AB')
    log.close()
    assert entry.line == 4
    assert RunLog.read(tmp_path / 'a.vdlog').entry(entry.id).line == 4


def test_a_closed_log_lets_go_of_its_file(tmp_path):
    log = new_log(tmp_path)
    log.append('note', author='AB')
    assert log.writable
    log.close()
    log.close()                                       # twice does nothing
    assert not log.writable
    os.remove(tmp_path / 'a.vdlog')                  # no handle holds it
    assert not os.path.exists(tmp_path / 'a.vdlog')


def test_an_entry_ten_and_a_half_minutes_late_is_late_by_default(tmp_path):
    with new_log(tmp_path, clock=Clock(START + timedelta(minutes=10,
                                                          seconds=30))) as log:
        entry = log.append('note', author='AB', event_time=START)
    assert run_log.LATE_AFTER == timedelta(minutes=10)
    assert entry.is_late()


def test_a_file_of_no_bytes_may_be_named():
    assert FileRef('empty.txt', '0' * 64, 0).size == 0


def test_lines_with_ids_or_references_that_cannot_be_read_are_named(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    base = json.loads(lines[3])
    bad = []
    for change in ({'id': 'seven'}, {'id': True}, {'id': 0},
                   {'corrects': 'x'}, {'corrects': 0}):
        entry = dict(base, **change)
        bad.append(json.dumps(entry).encode())
    write_lines(path, lines[:4] + bad + [b''])
    read = RunLog.read(path)
    assert [entry.id for entry in read.entries] == [1, 2, 3]
    reasons = [problem.reason for problem in read.problems]
    assert len(reasons) == 5
    assert "its id 'seven' is not a number from 1" in reasons[0]
    assert 'its id True is not a number from 1' in reasons[1]
    assert 'its id 0 is not a number from 1' in reasons[2]
    assert 'corrected entry' in reasons[3] and 'corrected entry' in reasons[4]


def test_an_entry_or_header_with_no_login_reads_as_blank(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    header = json.loads(lines[0])
    del header['login'], header['computer']
    lines[0] = json.dumps(header).encode()
    entry = json.loads(lines[1])
    del entry['login'], entry['computer']
    lines[1] = json.dumps(entry).encode()
    write_lines(path, lines)
    read = RunLog.read(path)
    assert (read.header.login, read.header.computer) == ('', '')
    assert (read.entry(1).login, read.entry(1).computer) == ('', '')


def test_a_header_with_a_blank_id_is_refused(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    header = json.loads(lines[0])
    for log_id in ('', 7):
        header['log_id'] = log_id
        lines[0] = json.dumps(header).encode()
        write_lines(path, lines)
        with pytest.raises(RunLogError, match='no log id'):
            RunLog.read(path)


def test_a_one_line_file_that_is_not_a_header_is_not_called_torn(tmp_path):
    other = tmp_path / 'other.vdlog'
    other.write_text('garbage\n')
    with pytest.raises(RunLogError, match='not a Visual Dynamics run log'):
        RunLog.read(other)


def test_an_entry_that_is_not_a_correction_corrects_nothing(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    note = json.loads(lines[3])
    note['corrects'] = 2
    lines[3] = json.dumps(note).encode()
    write_lines(path, lines)
    read = RunLog.read(path)
    assert read.corrections_of(2) == ()
    assert read.current(2).corrected_by == ()


def test_a_log_opened_again_records_who_opened_it(tmp_path):
    path = three_entries(tmp_path)
    with RunLog.open(path, who=('carol', 'bench-3')) as log:
        entry = log.append('note', author='CA')
    assert (entry.login, entry.computer) == ('carol', 'bench-3')


def test_a_zero_byte_log_is_named_on_its_first_line(tmp_path):
    path = tmp_path / 'a.vdlog'
    path.write_bytes(b'')
    with RunLog.open(path, who=WHO) as log:
        assert log.problems[0].line == 1


def test_a_numbered_findings_other_parts_can_be_corrected(tmp_path):
    with new_log(tmp_path) as log:
        deviation = log.append('deviation', author='AB')
        log.correct(deviation, author='AB', fields={'fact': 'notch at 180 Hz'})
        assert log.current(deviation.id).fields == {
            'number': 'D-1', 'fact': 'notch at 180 Hz'}
        note = log.append('note', author='AB')
        with pytest.raises(RunLogError, match="no field 'number'"):
            log.correct(note, author='AB', fields={'number': 'D-2'})


def test_two_missing_entries_are_named_together(tmp_path):
    path = three_entries(tmp_path)
    with RunLog.open(path, who=WHO, clock=Clock()) as log:
        log.append('note', author='AB')
    lines = lines_of(path)
    del lines[2:4]
    write_lines(path, lines)
    found = RunLog.read(path).verify()
    assert 'the ids jump: entries #2 to #3 are missing' in [
        b.reason for b in found.breaks]


def test_a_missing_head_is_named_after_the_breaks_in_the_file(tmp_path):
    path = three_entries(tmp_path)
    lines = lines_of(path)
    lines[3] = lines[3].replace(b'between tests', b'between the tests')
    write_lines(path, lines)
    head = '9' * 64
    found = RunLog.read(path).verify(heads={1: head})
    assert [(b.entry, b.chain) for b in found.breaks] == [(3, 'log'),
                                                          (None, 'test 1')]
    assert f'({head[:12]}...)' in found.breaks[1].reason


def test_checking_one_test_ignores_damage_to_another(tmp_path):
    path = three_entries(tmp_path)
    with RunLog.open(path, who=WHO, clock=Clock()) as log:
        log.append('note', author='AB', test=2, text='second test')
    lines = lines_of(path)
    lines[4] = lines[4].replace(b'second test', b'other test')
    write_lines(path, lines)
    read = RunLog.read(path)
    assert read.verify(tests=[1]).ok
    assert not read.verify(tests=[2]).ok
    assert read.verify(tests=[None]).ok
