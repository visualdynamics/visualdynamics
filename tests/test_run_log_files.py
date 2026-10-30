"""Controller files: a watched folder's files proposed once they settle,
accepted into the run log by SHA-256 with their format and holdings,
compared with the run's counter, and put into the data index.

Settling is timed on a clock each test moves, so a file is settled or
not by value, never by how fast the machine is; only the tests of the
watch's own thread wait, and they wait for the thing, with a deadline.
"""

import gc
import hashlib
import os
import re
import shutil
import threading
import time
import weakref
from datetime import datetime, timedelta, timezone

import pytest
from conftest import destroy_window, fixture_path

from visualdynamics import io
from visualdynamics.core import data_index, run_log_files
from visualdynamics.core.data_index import MATCHES, DataIndex, IndexEntry
from visualdynamics.core.run_log import FileRef, RunLog, RunLogError
from visualdynamics.core.run_log_files import (
    LOG_FORMAT,
    SOURCE,
    UNKNOWN,
    ControllerFilesError,
    ControllerWatch,
    Place,
    Proposal,
    WatchProblem,
    accept,
    check_stream,
    counted_seconds,
    counter_seconds,
    describe,
    index_run_log,
    place_at,
)

WHO = ('tester', 'bench')
ZONE = timezone(timedelta(hours=-6))
START = datetime(2026, 10, 30, 8, 0, tzinfo=ZONE)
#: when the watch first sees a file: after run 1 of test 1 stopped
SEEN = START + timedelta(minutes=6)


class Ticks:
    """A monotonic clock that reads `t`, which a test moves."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def watch_of(folders, **settings):
    ticks = Ticks()
    settings.setdefault('rate', None)
    watch = ControllerWatch(folders, clock=ticks, now=lambda: SEEN,
                            **settings)
    return watch, ticks


def settled(watch, ticks, seconds=10.0):
    """One look, then another `seconds` later."""
    first = watch.scan()
    ticks.t += seconds
    return first + watch.scan()


def proposals(found):
    return [item for item in found if isinstance(item, Proposal)]


def problems(found):
    return [item for item in found if isinstance(item, WatchProblem)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def new_log(tmp_path, start=None, stop='0:02'):
    """A log holding test 1, run 1 on axis Z, started with counter `start`
    and stopped with counter `stop`."""
    log = RunLog.create(tmp_path / 'a.vdlog', clock=lambda: START
                        + timedelta(minutes=30), who=WHO)
    log.append('test_opened', author='AB', test=1, event_time=START,
               fields={'title': 'Shock'})
    log.append('run_start', author='AB', test=1, axis='Z', run=1,
               event_time=START + timedelta(minutes=1),
               fields={'counter': start})
    log.append('run_stop', author='AB', test=1, axis='Z', run=1,
               event_time=START + timedelta(minutes=5),
               fields={'counter': stop})
    return log


def output(tmp_path):
    folder = tmp_path / 'controller'
    folder.mkdir()
    return folder


# ---- the folder ---------------------------------------------------------------


def test_a_file_appears_settles_is_proposed_and_accepted(tmp_path):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    assert watch.scan() == [], 'an empty folder proposes nothing'
    (folder / 'run1.dat').write_bytes(b'abc')
    assert watch.scan() == [], 'just appeared: not settled'
    assert watch.pending == (str(folder / 'run1.dat'),)
    ticks.t += 4.9
    assert watch.scan() == [], 'not yet still for the settling time'
    ticks.t += 0.1
    found = watch.scan()
    assert len(found) == 1
    proposal = found[0]
    assert proposal.name == 'run1.dat'
    assert proposal.sha256 == sha(b'abc')
    assert proposal.file.size == 3
    assert proposal.file.source == SOURCE
    assert proposal.path == str(folder / 'run1.dat')
    assert proposal.folder == str(folder)
    assert proposal.seen == SEEN
    assert not proposal.archived
    assert watch.pending == ()
    ticks.t += 10
    assert watch.scan() == [], 'proposed once'

    with new_log(tmp_path) as log:
        entry = accept(log, proposal, author='AB')
    assert (entry.test, entry.axis, entry.run) == (1, 'Z', 1)
    assert entry.event_time == SEEN
    assert entry.kind == 'note'
    ref = entry.files[0]
    assert (ref.name, ref.sha256, ref.size) == ('run1.dat', sha(b'abc'), 3)
    assert ref.format == UNKNOWN and ref.path == proposal.path
    back = RunLog.read(tmp_path / 'a.vdlog').entry(entry.id).files[0]
    assert back == ref, 'format, holdings and path are read back'
    assert RunLog.read(tmp_path / 'a.vdlog').verify().ok


def test_nothing_is_attached_without_acceptance(tmp_path):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    with new_log(tmp_path) as log:
        before = len(log.entries)
        (folder / 'run1.dat').write_bytes(b'abc')
        assert len(proposals(settled(watch, ticks))) == 1
        assert len(log.entries) == before


def test_a_growing_file_is_not_hashed_until_it_settles(tmp_path, monkeypatch):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    hashed = []
    real = data_index.digest
    monkeypatch.setattr(data_index, 'digest',
                        lambda path, progress=None: (hashed.append(path),
                                                     real(path, progress))[1])
    target = folder / 'stream.dat'
    target.write_bytes(b'a')
    watch.scan()
    for k in range(5):
        ticks.t += 10
        with open(target, 'ab') as handle:
            handle.write(b'b' * (k + 1))
        assert watch.scan() == [], f'still growing at look {k}'
    assert hashed == [], 'a file still being written is never hashed'
    ticks.t += 10
    found = watch.scan()
    data = target.read_bytes()
    assert [p.sha256 for p in found] == [sha(data)]
    assert hashed == [str(target)]


def test_a_file_changed_while_it_is_hashed_waits_to_settle_again(
        tmp_path, monkeypatch):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    target = folder / 'stream.dat'
    target.write_bytes(b'abc')
    real = data_index.digest
    calls = []

    def written_during(path, progress=None):
        result = real(path, progress)
        if not calls:
            with open(path, 'ab') as handle:
                handle.write(b'more')
        calls.append(path)
        return result

    monkeypatch.setattr(data_index, 'digest', written_during)
    assert settled(watch, ticks) == []
    assert watch.pending == (str(target),)
    found = settled(watch, ticks)
    assert [p.sha256 for p in found] == [sha(b'abcmore')]


def test_a_file_removed_before_it_settles_is_forgotten(tmp_path):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    (folder / 'scratch.tmp').write_bytes(b'abc')
    watch.scan()
    os.remove(folder / 'scratch.tmp')
    ticks.t += 10
    assert watch.scan() == []
    assert watch.pending == ()
    ticks.t += 10
    assert watch.scan() == []


def test_the_files_already_there_are_earlier_runs(tmp_path):
    folder = output(tmp_path)
    (folder / 'old.dat').write_bytes(b'old')
    watch, ticks = watch_of(folder)
    assert settled(watch, ticks) == [], 'a file there before is not proposed'
    (folder / 'old.dat').write_bytes(b'rewritten')
    assert [p.name for p in settled(watch, ticks)] == ['old.dat'], \
        'one that changes afterwards is'
    everything, ticks = watch_of(folder, existing=True)
    assert [p.name for p in settled(everything, ticks)] == ['old.dat']


def test_two_folders_and_the_log_itself_ignored(tmp_path):
    first, second = output(tmp_path), tmp_path / 'second'
    second.mkdir()
    watch, ticks = watch_of([first, second, first])
    assert watch.folders == (str(first), str(second))
    (first / 'a.dat').write_bytes(b'a')
    (second / 'b.dat').write_bytes(b'b')
    for name in ('a.vdlog', 'a.vdlog.lock', 'a.vdlog.partial'):
        (first / name).write_bytes(b'{}')
    (first / 'sub').mkdir()
    found = settled(watch, ticks)
    assert sorted(p.name for p in found) == ['a.dat', 'b.dat']


def test_a_missing_folder_is_refused_by_name(tmp_path):
    gone = tmp_path / 'nowhere'
    with pytest.raises(ControllerFilesError,
                       match=re.escape(f'no folder at {gone}')):
        ControllerWatch(gone)
    afile = tmp_path / 'f.txt'
    afile.write_text('x')
    with pytest.raises(ControllerFilesError,
                       match=re.escape(f'{afile} is a file')):
        ControllerWatch(afile)
    with pytest.raises(ControllerFilesError, match='at least one'):
        ControllerWatch([])


@pytest.mark.parametrize('setting, value, match', [
    ('settle_s', -1, 'settling time'), ('settle_s', float('nan'), 'settling'),
    ('poll_s', 0, 'polling interval'), ('poll_s', True, 'polling'),
    ('rate', 0, 'hashing rate'), ('rate', float('inf'), 'hashing rate')])
def test_settings_it_cannot_take_are_refused(tmp_path, setting, value, match):
    with pytest.raises(ControllerFilesError, match=match):
        ControllerWatch(output(tmp_path), **{setting: value})


def test_no_settling_time_still_looks_twice(tmp_path):
    folder = output(tmp_path)
    watch, _ticks = watch_of(folder, settle_s=0)
    (folder / 'a.dat').write_bytes(b'a')
    assert watch.scan() == [], 'seen once is never settled'
    with open(folder / 'a.dat', 'ab') as handle:
        handle.write(b'b')
    assert watch.scan() == [], 'seen changed is seen once again'
    assert [p.sha256 for p in watch.scan()] == [sha(b'ab')]


def test_an_unreadable_file_is_reported_once_and_tried_again(
        tmp_path, monkeypatch):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    (folder / 'locked.dat').write_bytes(b'abc')
    real = data_index.digest
    locked = [True]

    def refusing(path, progress=None):
        if locked[0]:
            raise PermissionError(13, 'Permission denied')
        return real(path, progress)

    monkeypatch.setattr(data_index, 'digest', refusing)
    found = settled(watch, ticks)
    assert [str(p) for p in found] == [
        (f'{folder / "locked.dat"}: locked.dat could not be read '
         '(Permission denied); it is tried again at the next look')]
    ticks.t += 10
    assert watch.scan() == [], 'reported once while it lasts'
    locked[0] = False
    ticks.t += 10
    assert [p.sha256 for p in watch.scan()] == [sha(b'abc')]


def test_a_folder_that_goes_away_is_reported_and_looked_for_again(tmp_path):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    shutil.rmtree(folder)
    found = watch.scan()
    assert len(found) == 1 and isinstance(found[0], WatchProblem)
    assert found[0].path == str(folder)
    assert 'cannot be read' in found[0].reason
    assert watch.scan() == [], 'reported once while it lasts'
    folder.mkdir()
    (folder / 'back.dat').write_bytes(b'b')
    assert [p.name for p in settled(watch, ticks)] == ['back.dat']


def test_archive_hashes_what_has_not_settled(tmp_path):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    (folder / 'early.dat').write_bytes(b'e')
    watch.scan()
    (folder / 'last.dat').write_bytes(b'l')
    found = watch.archive()
    assert sorted(p.name for p in found) == ['early.dat', 'last.dat']
    assert all(p.archived for p in found)
    assert watch.pending == ()
    ticks.t += 10
    assert watch.scan() == []


def test_archive_reports_a_file_still_changing(tmp_path, monkeypatch):
    folder = output(tmp_path)
    watch, _ticks = watch_of(folder)
    target = folder / 'live.dat'
    target.write_bytes(b'abc')
    real = data_index.digest

    def written_during(path, progress=None):
        result = real(path, progress)
        with open(path, 'ab') as handle:
            handle.write(b'x')
        return result

    monkeypatch.setattr(data_index, 'digest', written_during)
    found = watch.archive()
    assert [type(item) for item in found] == [WatchProblem]
    assert 'changed while it was being hashed' in found[0].reason
    assert watch.pending == (str(target),)


# ---- format and holdings ------------------------------------------------------


def test_an_unknown_format_is_kept_by_name_size_and_hash(tmp_path):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    (folder / 'odd.qqq').write_bytes(b'\x00\x01')
    proposal, = settled(watch, ticks)
    assert proposal.file.format == UNKNOWN
    assert proposal.file.size == 2
    assert proposal.sha256 == sha(b'\x00\x01')
    assert proposal.file.holds.startswith('not read: no importer recognizes')
    assert proposal.channels == () and proposal.stream is None
    assert io.known_format(folder / 'odd.qqq') is None
    assert io.format_of(folder / 'odd.qqq') == 'QQQ file (.qqq)', \
        'the data index still names it by its suffix'


def test_a_rattlesnake_run_gives_its_channels_and_stream(tmp_path):
    target = tmp_path / 'shock_Z_run1.nc4'
    shutil.copy(fixture_path('plate', 'shock.nc4'), target)
    said = describe(target)
    assert said.format == 'Rattlesnake controller output (.nc4)'
    assert len(said.channels) == 7
    assert said.stream.sample_rate == 8192.0
    assert said.stream.seconds == pytest.approx(2.8157958984375)
    assert said.stream.channels == 7
    assert said.run == {'axis': 'Z', 'run': '1'}
    # the guide's table prints this row
    assert said.holds == ('Rattlesnake transient run; 7 channels, 101Z+ to '
                          '1Z+; 8192 Hz; time stream 2.8158 s; run 1, axis Z '
                          '(from its name)')
    assert (said.channels[0], said.channels[-1]) == ('101Z+', '1Z+')


def test_a_file_an_importer_reads_gives_its_channels(tmp_path):
    import numpy as np

    from visualdynamics.core.data import TimeHistory

    path = tmp_path / 'run2_m6dB.vdyn'
    TimeHistory(np.arange(64) / 64.0, np.ones((2, 64)),
                response_dof=['1X+', '2Y+'],
                ordinate_dim=['acceleration'] * 2,
                ordinate_unit=['g'] * 2).save(str(path))
    said = describe(path)
    assert said.format == 'Visual Dynamics file (.vdyn)'
    assert said.channels == ('1X+', '2Y+')
    assert 'channels 1X+ to 2Y+' in said.holds
    assert said.holds.endswith('run 2, -6 dB (from its name)')
    assert describe(path, limit=os.path.getsize(path)).channels == (
        '1X+', '2Y+'), 'a file of the limit itself is read'
    big = describe(path, limit=10)
    assert big.channels == ()
    assert big.notes == (
        ('larger than 10 bytes, so it was not read for its channels'),)
    assert big.holds.startswith('run 2'), 'its name is still read'


def test_a_reader_that_fails_is_a_note_not_a_refusal(tmp_path):
    broken = tmp_path / 'broken.vdyn'
    broken.write_bytes(b'not a project')
    said = describe(broken)
    assert said.format == 'Visual Dynamics file (.vdyn)'
    assert said.notes[0].startswith('its holdings could not be read')
    assert said.holds.startswith('not read: its holdings could not be read')


# ---- the stream and the counter -----------------------------------------------


def stream_proposal(tmp_path, source='shock.nc4'):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    shutil.copy(fixture_path('plate', source), folder / source)
    proposal, = settled(watch, ticks)
    return proposal


@pytest.mark.parametrize('start, stop, ok, words', [
    (None, '0:02', True, 'within the 1 s allowed'),
    ('5', '7.5 s', True, 'against 2.5 s counted'),
    (None, '0:10', False, '7.2 s shorter, more than the 1 s allowed'),
    (None, '0:00', False, '2.8 s longer, more than the 1 s allowed'),
])
def test_the_stream_is_compared_with_the_counter(tmp_path, start, stop, ok,
                                                 words):
    proposal = stream_proposal(tmp_path)
    with new_log(tmp_path, start=start, stop=stop) as log:
        found = check_stream(log, proposal, tolerance_s=1.0)
    assert found.ok is ok
    assert words in found.message
    assert found.stream_s == pytest.approx(2.8157958984375)
    assert found.sample_rate == 8192.0
    assert found.message.startswith('shock.nc4: the stream holds 2.8 s at '
                                    '8192 Hz against ')


def test_the_tolerance_is_the_edge(tmp_path):
    proposal = stream_proposal(tmp_path)
    with new_log(tmp_path, stop='2') as log:
        over = proposal.stream.seconds - 2.0
        assert check_stream(log, proposal, tolerance_s=over).ok is True
        assert check_stream(log, proposal, tolerance_s=over - 1e-9).ok is False
        assert check_stream(log, proposal).tolerance_s == \
            run_log_files.STREAM_TOLERANCE_S
        with pytest.raises(ControllerFilesError, match='tolerance'):
            check_stream(log, proposal, tolerance_s=-1)
    os.remove(tmp_path / 'a.vdlog')
    with new_log(tmp_path, stop='10') as log:
        short = 10.0 - proposal.stream.seconds
        assert check_stream(log, proposal, tolerance_s=short).ok is True
        assert check_stream(log, proposal, tolerance_s=short - 1e-9).ok is False


def test_nothing_compared_is_never_a_pass(tmp_path):
    spectra = stream_proposal(tmp_path, 'random_spectra.nc4')
    with new_log(tmp_path, stop=None) as log:
        assert spectra.stream is None
        found = check_stream(log, spectra)
        assert found.ok is None and 'holds no time stream' in found.message
    os.remove(tmp_path / 'a.vdlog')
    shutil.rmtree(tmp_path / 'controller')
    shock = stream_proposal(tmp_path)
    with new_log(tmp_path, stop=None) as log:
        found = check_stream(log, shock)
        assert found.ok is None
        assert 'has no counter recorded at its stop' in found.message
        assert found.stream_s == pytest.approx(2.8157958984375)
        found = check_stream(log, shock, place=Place(1, 'Z', 9))
        assert found.ok is None and 'no entries for run 9' in found.message
        found = check_stream(log, shock, place=Place(None, None, None))
        assert found.ok is None and 'is not in a run' in found.message
        found = check_stream(log, shock, place=Place(1, None, None))
        assert found.ok is None and 'is not in a run' in found.message
        log.correct(3, author='AB', fields={'counter': 'soon'})
        found = check_stream(log, shock)
        assert found.ok is None and "entry #3: counter 'soon'" in found.message


@pytest.mark.parametrize('text, seconds', [
    ('1:02:03', 3723.0), ('12:30', 750.0), ('0:00:02.5', 2.5), ('750', 750.0),
    ('750 s', 750.0), ('12.5 min', 750.0), ('1 h', 3600.0), (' 2 SEC ', 2.0),
    ('.5', 0.5), ('90:00', 5400.0)])
def test_a_counter_is_read_in_the_forms_typed(text, seconds):
    assert counter_seconds(text) == seconds


@pytest.mark.parametrize('text', ['1:60:00', '1:00:60', '12:60', 'soon',
                                  '5 parsecs', '', '-3', '1::2'])
def test_a_counter_that_is_not_a_time_is_refused(text):
    with pytest.raises(ControllerFilesError, match=re.escape(repr(text))):
        counter_seconds(text)


def test_the_counter_reads_the_last_stop_less_the_start(tmp_path):
    with new_log(tmp_path, start='1:00', stop='3:00') as log:
        assert counted_seconds(log, 1, 1) == (120.0, '')
        log.append('abort', author='AB', test=1, axis='Z', run=1,
                   fields={'stop': 'amplifier trip', 'counter': '4:00'})
        assert counted_seconds(log, 1, 1) == (180.0, '')
        log.append('note', author='AB', test=1, axis='Z', run=1,
                   text='counter not read here')
        assert counted_seconds(log, 1, 1) == (180.0, '')


# ---- the log ------------------------------------------------------------------


def test_a_file_goes_to_the_run_current_when_it_appeared(tmp_path):
    with new_log(tmp_path) as log:
        assert place_at(log, START - timedelta(minutes=1)) == \
            Place(None, None, None)
        assert place_at(log, START) == Place(1, None, None), \
            'a test open, no run yet'
        assert place_at(log, START + timedelta(minutes=1)) == Place(1, 'Z', 1)
        assert place_at(log, SEEN) == Place(1, 'Z', 1), \
            'after its stop, still the run that wrote it'
        log.append('test_opened', author='AB', test=2,
                   event_time=START + timedelta(minutes=10))
        assert place_at(log, START + timedelta(minutes=10)) == \
            Place(2, None, None)
        assert place_at(log, SEEN) == Place(1, 'Z', 1)
        log.correct(2, author='AB', fields={'axis': 'X'})
        assert place_at(log, SEEN) == Place(1, 'X', 1), 'the corrected view'
        at = START + timedelta(minutes=10)
        log.append('run_start', author='AB', test=2, axis='Y', run=1,
                   event_time=at)
        assert place_at(log, at) == Place(2, 'Y', 1), (
            'at the same time, the one written later')


def test_accept_takes_a_place_and_refuses_a_second_time(tmp_path):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    (folder / 'a.dat').write_bytes(b'a')
    proposal, = settled(watch, ticks)
    with new_log(tmp_path) as log:
        entry = accept(log, proposal, author='AB', place=Place(None, None, None),
                       kind='observation', text='from the controller')
        assert (entry.test, entry.run, entry.kind, entry.text) == (
            None, None, 'observation', 'from the controller')
        with pytest.raises(ControllerFilesError,
                           match=f'in the log already, entry #{entry.id}'):
            accept(log, proposal, author='AB')
        (folder / 'a.dat').write_bytes(b'rewritten')
        rewritten, = settled(watch, ticks)
        assert rewritten.name == proposal.name
        written = accept(log, rewritten, author='AB')
        assert written.files[0].sha256 == sha(b'rewritten'), (
            'the same name with new contents is a new file')
        (folder / 'b.dat').write_bytes(b'a')
        copy, = settled(watch, ticks)
        assert copy.sha256 == proposal.sha256
        copied = accept(log, copy, author='AB')
        assert copied.files[0].name == 'b.dat', (
            'the same contents under another name is another file')
        other = Proposal(IndexEntry('b.dat', proposal.path, 1, '1' * 64, 'x'),
                         str(folder), SEEN)
        with pytest.raises(RunLogError, match='initials'):
            accept(log, other, author=' ')


def test_a_file_ref_carries_format_holdings_and_path(tmp_path):
    ref = FileRef('a.nc4', 'a' * 64, 3, 'Rattlesnake', '7 channels', '/x/a.nc4')
    assert ref.to_dict() == {'name': 'a.nc4', 'sha256': 'a' * 64, 'size': 3,
                             'format': 'Rattlesnake', 'holds': '7 channels',
                             'path': '/x/a.nc4'}
    assert FileRef('a', 'a' * 64).to_dict() == {'name': 'a', 'sha256': 'a' * 64}
    with pytest.raises(RunLogError, match='its holds is text'):
        FileRef('a', 'a' * 64, holds=7)
    log = RunLog.create(tmp_path / 'b.vdlog', who=WHO)
    log.append('note', author='AB', files=[ref])
    log.close()
    assert RunLog.read(tmp_path / 'b.vdlog').entries[0].files == (ref,)


# ---- the data index -----------------------------------------------------------


def test_the_index_holds_the_logs_files_and_its_head(tmp_path):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    (folder / 'run1.dat').write_bytes(b'abc')
    proposal, = settled(watch, ticks)
    index = DataIndex()
    with new_log(tmp_path) as log:
        accept(log, proposal, author='AB')
        log.append('note', author='AB', text='between tests')
        indexed = index_run_log(index, log)
        head = log.head(1)
        between = log.head(None)
    assert len(index) == 2
    file_entry, log_entry = index
    assert indexed.files == (file_entry,) and indexed.log == log_entry
    assert indexed.missing == ()
    assert (file_entry.name, file_entry.sha256, file_entry.size) == (
        'run1.dat', sha(b'abc'), 3)
    assert file_entry.format == UNKNOWN and file_entry.source == SOURCE
    assert file_entry.holds == proposal.file.holds
    assert log_entry.format == LOG_FORMAT and log_entry.source == SOURCE
    assert log_entry.holds == (f'run log; test 1 head {head}; between tests '
                               f'head {between}')
    assert log_entry.sha256 == sha((tmp_path / 'a.vdlog').read_bytes())
    checked = index.verify()
    assert [entry.status for entry in checked] == [MATCHES, MATCHES]


def test_the_index_takes_the_logs_checksum_not_a_second_hash(
        tmp_path, monkeypatch):
    target = tmp_path / 'kept.dat'
    target.write_bytes(b'abc')
    log = RunLog.create(tmp_path / 'a.vdlog', who=WHO, title='Article 7')
    log.append('test_opened', author='AB', test=1)
    log.append('note', author='AB', test=1, files=[FileRef(
        'kept.dat', sha(b'abc'), 3, path=str(target))])
    hashed = []
    real = data_index.digest
    monkeypatch.setattr(data_index, 'digest',
                        lambda path, progress=None: (hashed.append(path),
                                                     real(path, progress))[1])
    index = DataIndex()
    indexed = index_run_log(index, log, tests=[1])
    assert hashed == [os.path.abspath(log.path)], 'only the log is hashed'
    assert indexed.log.holds == f'run log, Article 7; test 1 head {log.head(1)}'
    target.write_bytes(b'abd')
    assert index.verify()[0].status == 'changed'
    log.close()


def test_a_file_is_found_by_name_and_one_found_nowhere_is_listed(tmp_path):
    archive = tmp_path / 'archive'
    archive.mkdir()
    (archive / 'moved.dat').write_bytes(b'm')
    (tmp_path / 'beside.dat').write_bytes(b'b')
    with RunLog.create(tmp_path / 'a.vdlog', who=WHO) as log:
        log.append('test_opened', author='AB', test=1)
        log.append('note', author='AB', test=1, files=[
            FileRef('moved.dat', sha(b'm'), 1, path=str(tmp_path / 'gone')),
            FileRef('beside.dat', sha(b'b')),
            FileRef('lost.dat', sha(b'l'), 1, path=str(tmp_path / 'lost.dat'))])
        index = DataIndex()
        indexed = index_run_log(index, log, search=[archive])
    assert [e.path for e in indexed.files] == [str(archive / 'moved.dat'),
                                               str(tmp_path / 'beside.dat')]
    assert indexed.files[1].size == 1, 'a size not recorded is the file\'s'
    assert indexed.files[1].holds == 'named in the run log, entry #2'
    assert indexed.files[1].format == 'DAT file (.dat)'
    (ref, why), = indexed.missing
    assert ref.name == 'lost.dat'
    assert why.startswith(f'lost.dat (entry #2) was not found at '
                          f'{tmp_path / "lost.dat"} or by its name in '
                          f'{archive}, {tmp_path}')


def test_indexing_again_replaces_the_logs_entry(tmp_path):
    index = DataIndex()
    with RunLog.create(tmp_path / 'a.vdlog', who=WHO) as log:
        log.append('test_opened', author='AB', test=1)
        first = index_run_log(index, log)
        log.append('note', author='AB', test=1)
        again = index_run_log(index, log)
        assert again.log.sha256 != first.log.sha256
        assert [entry.sha256 for entry in index] == [again.log.sha256]
        assert again.log.holds.endswith(log.head(1))
        settings = tmp_path / 'settings.txt'
        settings.write_text('x')
        index.add(settings, 'settings list', source='author')
        index_run_log(index, log)
        assert [entry.source for entry in index] == ['author', SOURCE]


def test_a_test_not_in_the_log_is_refused(tmp_path):
    with new_log(tmp_path) as log:
        with pytest.raises(ControllerFilesError, match='holds no test 4'):
            index_run_log(DataIndex(), log, tests=[4])
        empty = RunLog.create(tmp_path / 'e.vdlog', who=WHO)
        assert index_run_log(DataIndex(), empty).log.holds == \
            'run log; no test yet'
        empty.close()


def test_a_reports_data_index_lists_them(tmp_path):
    import visualdynamics
    from visualdynamics import Project
    from visualdynamics.report import render_html

    project = Project('P')
    name = project.generate_report('modal')
    with new_log(tmp_path) as log:
        index_run_log(project.data_index, log)
        head = log.head(1)
    project[name].features = ['data_index']
    html = render_html(project[name], project,
                       unit_system=visualdynamics.SI,
                       object_groups=project.object_groups)
    assert f'test 1 head {head}' in html
    assert 'a.vdlog' in html


# ---- the watch's own thread -----------------------------------------------------


def wait_for(condition, seconds=20.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_a_share_is_polled_with_no_notice(tmp_path):
    """Nothing tells the watch the folder changed, as on a network
    share: the poll alone finds the file."""
    folder = output(tmp_path)
    watch = ControllerWatch(folder, settle_s=0.05, poll_s=0.02, rate=None)
    watch.start()
    try:
        (folder / 'run1.dat').write_bytes(b'abc')
        taken = []
        assert wait_for(lambda: taken.extend(watch.take()) or taken)
        assert [p.sha256 for p in taken] == [sha(b'abc')]
        with pytest.raises(ControllerFilesError, match='running already'):
            watch.start()
    finally:
        assert watch.stop()
    assert not watch.running


def test_a_poke_looks_at_once(tmp_path):
    folder = output(tmp_path)
    watch = ControllerWatch(folder, settle_s=0, poll_s=3600, rate=None)
    got = []
    watch.start(on_proposal=got.append)
    try:
        (folder / 'a.dat').write_bytes(b'a')
        watch.poke()
        # the thread's own first look can land after the write, and then
        # this poke settles the file at once: pending or already proposed
        assert wait_for(lambda: watch.pending or got), 'a poke, a look'
        watch.poke()
        assert wait_for(lambda: got), 'another, and it is settled'
        (folder / 'b.dat').write_bytes(b'b')
        watch.ask_archive()
        assert wait_for(lambda: len(got) == 2)
        assert got[1].archived
    finally:
        watch.stop()


def test_hashing_is_throttled_and_stops_with_the_watch(tmp_path):
    folder = output(tmp_path)
    watch, ticks = watch_of(folder, rate=8 * data_index.CHUNK)
    big = folder / 'big.dat'
    big.write_bytes(b'x' * (4 * data_index.CHUNK))
    watch.scan()
    ticks.t += 10
    began = time.monotonic()
    proposal, = watch.scan()
    took = time.monotonic() - began
    assert took >= 0.45, 'four chunks at eight a second'
    assert took < 2.0, f'held to the rate, not past it ({took:.2f} s)'
    assert proposal.sha256 == sha(big.read_bytes())

    slow = ControllerWatch(folder, settle_s=0, poll_s=0.01, rate=1024,
                           existing=True)
    hashing = threading.Event()
    real = data_index.digest

    def watched(path, progress=None):
        hashing.set()
        return real(path, progress)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data_index, 'digest', watched)
        slow.start()
        # at a kilobyte a second the hash would take hours
        assert wait_for(hashing.is_set)
    began = time.monotonic()
    assert slow.stop()
    assert time.monotonic() - began < 2, 'a hash in progress is abandoned'
    assert slow.take() == [], 'nothing half-hashed is proposed'


def test_a_callback_whose_receiver_is_gone_ends_the_thread(tmp_path):
    folder = output(tmp_path)
    watch = ControllerWatch(folder, settle_s=0, poll_s=0.01, rate=None)

    def deleted(_item):
        raise RuntimeError('Internal C++ object already deleted.')

    watch.start(on_proposal=deleted)
    (folder / 'a.dat').write_bytes(b'a')
    assert wait_for(lambda: not watch.running)
    assert watch.stop()


def test_a_failure_inside_a_look_is_reported_and_the_watch_goes_on(
        tmp_path, monkeypatch):
    folder = output(tmp_path)
    watch = ControllerWatch(folder, settle_s=0, poll_s=0.01, rate=None)
    broken = [True]
    real = run_log_files.describe

    def failing(path, limit):
        if broken[0]:
            broken[0] = False
            raise MemoryError('out of memory')
        return real(path, limit=limit)

    monkeypatch.setattr(run_log_files, 'describe',
                        lambda path, limit: failing(path, limit))
    watch.start()
    try:
        (folder / 'a.dat').write_bytes(b'a')
        taken = []
        assert wait_for(lambda: taken.extend(watch.take()) or any(
            isinstance(item, Proposal) for item in taken))
    finally:
        watch.stop()
    problem = next(item for item in taken if isinstance(item, WatchProblem))
    assert 'the watch failed (out of memory)' in problem.reason


# ---- in the window --------------------------------------------------------------


def test_the_window_gets_proposals_on_its_own_thread(qt_app, tmp_path):
    from PySide6.QtWidgets import QWidget

    from visualdynamics.gui.run_log_watch import ControllerFilesWatch

    folder = output(tmp_path)
    parent = QWidget()
    relay = ControllerFilesWatch(folder, parent, settle_s=0.05, poll_s=0.02,
                                 rate=None)
    got = []
    relay.proposed.connect(lambda p: got.append(
        (p, threading.current_thread() is threading.main_thread())))
    relay.problem.connect(lambda p: got.append((p, None)))
    try:
        assert relay.running
        (folder / 'a.dat').write_bytes(b'a')
        assert wait_for(lambda: qt_app.processEvents() or got)
        proposal, on_gui_thread = got[0]
        assert proposal.sha256 == sha(b'a') and on_gui_thread is True
        shutil.rmtree(folder)
        assert wait_for(lambda: qt_app.processEvents() or len(got) == 2)
        assert isinstance(got[1][0], WatchProblem)
    finally:
        assert relay.stop() and relay.stop(), 'stopping twice is harmless'
        parent.deleteLater()


def test_a_missing_folder_makes_no_relay(qt_app, tmp_path):
    from PySide6.QtWidgets import QWidget

    from visualdynamics.gui.run_log_watch import ControllerFilesWatch

    parent = QWidget()
    with pytest.raises(ControllerFilesError,
                       match=re.escape(str(tmp_path / 'nowhere'))):
        ControllerFilesWatch(tmp_path / 'nowhere', parent)
    assert parent.children() == []
    parent.deleteLater()


def test_the_worker_never_holds_up_input(qt_app, tmp_path):
    """A slow hash runs while the keeper types: every key lands at once,
    no input gate or wait cursor is put up, and the proposal follows."""
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLineEdit

    from visualdynamics.gui.run_log_watch import ControllerFilesWatch

    folder = output(tmp_path)
    field = QLineEdit()
    field.show()
    relay = ControllerFilesWatch(folder, field, settle_s=0, poll_s=0.01,
                                 rate=data_index.CHUNK)
    got = []
    relay.proposed.connect(got.append)
    hashing = threading.Event()
    real = data_index.digest

    def watched(path, progress=None):
        hashing.set()
        return real(path, progress)

    try:
        from unittest import mock

        with mock.patch.object(data_index, 'digest', watched):
            (folder / 'big.dat').write_bytes(b'x' * (2 * data_index.CHUNK))
            assert wait_for(hashing.is_set)
            began = time.monotonic()
            QTest.keyClicks(field, 'AB level -12')
            qt_app.processEvents()
            typed = time.monotonic() - began
            assert field.text() == 'AB level -12'
            assert typed < 1.0, f'typing took {typed:.2f} s during the hash'
            assert QApplication.overrideCursor() is None
            assert not got, 'the hash takes about two seconds'
            assert wait_for(lambda: qt_app.processEvents() or got)
    finally:
        relay.stop()
        field.deleteLater()
    assert got[0].name == 'big.dat'


def test_the_watch_stops_with_the_window(qt_app, tmp_path):
    from visualdynamics.gui.main_window import MainWindow
    from visualdynamics.gui.run_log_watch import ControllerFilesWatch

    folder = output(tmp_path)
    window = MainWindow(offscreen_3d=True)
    window.show()
    qt_app.processEvents()
    relay = ControllerFilesWatch(folder, window, settle_s=0.01, poll_s=0.01,
                                 rate=None)
    watch = relay.watch
    assert watch.running
    ghost = weakref.ref(window)
    if getattr(window, 'report_editor', None) is not None:
        window.report_editor.view.setPage(None)
    del relay
    destroy_window(window, qt_app)
    assert not watch.running, 'the window closed and the thread ended'
    (folder / 'late.dat').write_bytes(b'l')
    time.sleep(0.1)
    assert watch.take() == []
    del window
    gc.collect()
    assert ghost() is None, 'the watch holds nothing of the window'


def test_a_file_is_timed_by_its_own_stat_not_the_listings(tmp_path,
                                                          monkeypatch):
    """Windows keeps a folder's copy of a file's size and time up to
    date lazily, so the listing can disagree with the file's own stat
    taken after the hash, and a file judged by the listing never
    settles. The listing here is stale on purpose."""
    folder = output(tmp_path)
    watch, ticks = watch_of(folder)
    real = os.scandir

    class Stale:
        def __init__(self, item):
            self._item = item
            self.name, self.path = item.name, item.path

        def is_file(self):
            return self._item.is_file()

        def stat(self):
            return os.stat_result((0,) * 10)

    class Listing:
        def __init__(self, folder):
            self._found = real(folder)

        def __enter__(self):
            return (Stale(item) for item in self._found)

        def __exit__(self, *exc):
            self._found.close()

    monkeypatch.setattr(os, 'scandir', Listing)
    (folder / 'a.dat').write_bytes(b'abc')
    assert [p.sha256 for p in settled(watch, ticks)] == [sha(b'abc')]


def test_a_run_of_one_channel_names_it(tmp_path, monkeypatch):
    from visualdynamics.io import rattlesnake

    target = tmp_path / 'one.nc4'
    shutil.copy(fixture_path('plate', 'shock.nc4'), target)
    real = rattlesnake.stream_summary

    def one(path):
        summary = real(path)
        return {**summary, 'channels': summary['channels'][-1:]}

    monkeypatch.setattr(rattlesnake, 'stream_summary', one)
    said = describe(target)
    assert said.holds.startswith('Rattlesnake transient run; 1 channel, 1Z+;')


def test_a_file_with_no_records_names_no_channels(tmp_path):
    target = tmp_path / 'geometry.unv'
    shutil.copy(fixture_path('plate', 'geometry.unv'), target)
    said = describe(target)
    assert said.channels == () and said.notes == ()
    assert said.holds == 'Geometry: 169 nodes, 144 elements'
