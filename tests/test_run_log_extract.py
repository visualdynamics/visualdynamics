"""A run log brought into a project: one test frozen at a stated entry,
read as the run log model reads the log, saved in the project file, and
the project's verbs that attach, detach and refresh it.

Every log is written under `tmp_path` with a fixed login and machine
name, on a clock that moves one minute per entry.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import h5py
import pytest

from visualdynamics import Project
from visualdynamics.core import run_log_extract as extract_module
from visualdynamics.core import run_log_model as model
from visualdynamics.core.run_log import RunLog, RunLogError
from visualdynamics.core.run_log_extract import (
    DRAFTED,
    RunLogExtract,
    resolved_path,
    stored_path,
    take,
)
from visualdynamics.io import native
from visualdynamics.report import features as rf

WHO = ('tester', 'bench')
ZONE = timezone(timedelta(hours=-6))
START = datetime(2026, 10, 30, 8, 0, tzinfo=ZONE)
TAKEN = datetime(2026, 10, 30, 12, 0, tzinfo=ZONE)

HEADER = {'title': 'Random, Z', 'article': 'Unit 7', 'configuration': 'flight',
          'steps_db': [-12, -6, -3, 0], 'sigma': 3.0,
          'rated_current_peak': 800.0, 'rail_voltage': 250.0,
          'duration_s': 600.0, 'duration_minus_pct': 0.0,
          'duration_plus_pct': 10.0}


class Ticking:
    """A clock that moves one minute every time it is read."""

    def __init__(self, now=START):
        self.now = now

    def __call__(self):
        self.now += timedelta(minutes=1)
        return self.now


def writer(path):
    """The log at `path`, open for writing: created when it is not there."""
    if os.path.exists(path):
        return RunLog.open(path, clock=Ticking(START + timedelta(hours=1)),
                           who=WHO)
    return RunLog.create(path, title='Article 7', clock=Ticking(), who=WHO)


def two_tests(path):
    """A random test that ran to full level and tripped, with a deviation
    and an anomaly, rolled into a shock test on whose entries the trip's
    cause and the anomaly's disposition were written; a clock check and
    a person between tests. Returns the ids that matter."""
    with writer(path) as log:
        log.append('clock_check', author='AB', fields={'offset_s': 2.0})
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header=HEADER)
        log.append('person_in', author='AB', fields={'person': 'CD'})
        model.append(log, 'level_row', author='AB', test=1, axis='Z', run=1,
                     fields={'level_db': -12.0, 'voltage': 20.0,
                             'current': 63.0, 'go': True})
        model.append(log, 'level_row', author='AB', test=1, axis='Z', run=1,
                     fields={'level_db': 0.0, 'voltage': 80.0,
                             'current': 250.0, 'go': True})
        trip = model.append(log, 'abort', author='AB', test=1, axis='Z',
                            run=1, fields={'stop': 'amplifier trip',
                                           'counter': '4:00'})
        model.append(log, 'deviation', author='AB', test=1,
                     fields={'fact': 'a notch at 180 Hz not in the plan',
                             'agreed_by': 'EF'})
        anomaly = model.append(log, 'anomaly_opened', author='AB', test=1,
                               fields={'fact': 'a rattle at 300 Hz',
                                       'owner': 'CD'},
                               files=[os.fspath(path)])
        model.roll(log, author='AB', test_kind='Shock',
                   header={'title': 'Release shock', 'pulse_budget': 6})
        model.append(log, 'cause_found', author='AB', test=2,
                     fields={'cause': 'a loose bolt on the slip table',
                             'refers_to': trip.id})
        model.append(log, 'anomaly_dispositioned', author='AB', test=2,
                     fields={'number': anomaly.fields['number'],
                             'disposition': 'use as is'})
        model.append(log, 'shot', author='AB', test=2, axis='Z', run=1,
                     fields={'polarity': 'positive', 'n': 1, 'of': 3})
        return {'trip': trip.id, 'anomaly': anomaly.id, 'last': log.entries[-1].id}


def one_test(path):
    """A log of one random test with one level row."""
    with writer(path) as log:
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header=HEADER)
        model.append(log, 'level_row', author='AB', test=1, axis='Z', run=1,
                     fields={'level_db': -12.0, 'voltage': 20.0,
                             'current': 63.0, 'go': True})


def append_note(path, test=1, text='later'):
    with writer(path) as log:
        log.append('note', author='AB', test=test, text=text)


def views(source, test):
    """What a report reads of one test, as comparable values, from an
    extract or (with `test`) from a live model."""
    get = ((lambda name: getattr(source, name)()) if test is None
           else (lambda name: getattr(source, name)(test)))
    return {
        'runs': [(r.number, r.axis, r.outcome, [e.id for e in r.entries])
                 for r in get('runs')],
        'interruptions': [(i.entry.id, i.kind,
                           i.cause and i.cause.id, i.path and i.path.id,
                           i.restart and i.restart.id)
                          for i in get('interruptions')],
        'credited': [(c.axis, c.seconds, c.verdict, c.message)
                     for c in get('credited')],
        'findings': [(f.number, f.kind, f.fact, f.open,
                      [d.id for d in f.dispositions]) for f in get('findings')],
        'files': [(row.entry.id, row.file.sha256) for row in get('files')],
        'chronology': [(row.entry.id, row.recorder_time, row.late)
                       for row in get('chronology')],
        'holds': [(h.rule, h.entry.id) for h in get('holds')],
    }


# ---- the snapshot ------------------------------------------------------------


def test_the_extract_reads_as_the_model_read_the_log_when_it_was_taken(
        tmp_path):
    """The views of the test from the snapshot are the live model's,
    including what was written under the next test and answers this one:
    the trip's cause, the anomaly's disposition, and the next test's
    opening, which closed this one to runs."""
    path = tmp_path / 'a.vdlog'
    ids = two_tests(path)
    extract = take(path, test=1, taken=TAKEN)
    live = model.RunLogModel(RunLog.read(path))
    assert views(extract, None) == views(live, 1)
    assert extract.state == live.test(1).state == 'closed to runs'
    cause, = extract.interruptions()
    assert cause.cause.fields['cause'] == 'a loose bolt on the slip table'
    anomaly = next(f for f in extract.findings() if f.kind == 'anomaly')
    assert not anomaly.open, 'the disposition written under test 2 counts'
    assert extract.upto == ids['last']
    assert extract.head == live.log.head(1)
    assert extract.kind == 'Random Vibration'
    assert extract.title == 'Random, Z'
    assert extract.header.get('sigma') == 3.0
    assert extract.log_title == 'Article 7'
    assert extract.verify().ok


def test_the_extract_holds_one_test_not_the_others(tmp_path):
    """Of the shock test only the entries that answer the random test
    are kept: the cause and the disposition, not its shot, the shot's
    correction, or its own anomaly and that anomaly's disposition."""
    path = tmp_path / 'a.vdlog'
    two_tests(path)
    with writer(path) as log:
        shot = next(entry for entry in log.entries if entry.kind == 'shot')
        log.correct(shot, author='AB', fields={'of': 4})
        own = model.append(log, 'anomaly_opened', author='AB', test=2,
                           fields={'fact': 'a crack in the fixture'})
        model.append(log, 'anomaly_dispositioned', author='AB', test=2,
                     fields={'number': own.fields['number']})
    extract = take(path, test=1)
    kept = [entry.kind for entry in extract.log.entries if entry.test == 2]
    assert kept == ['cause_found', 'anomaly_dispositioned'], (
        'the roll closed the test with an entry of its own')
    assert all(entry.test == 1 for entry in extract.entries)


def test_a_test_closed_by_the_next_opening_reads_closed(tmp_path):
    """A test opened while another was open (written past the model's
    refusal) closes the first to runs; the opening comes with it."""
    path = tmp_path / 'a.vdlog'
    one_test(path)
    with writer(path) as log:
        log.append('test_opened', author='AB', test=2,
                   fields={'test_kind': 'Shock'})
    extract = take(path, test=1)
    assert extract.state == 'closed to runs'
    assert [entry.kind for entry in extract.log.entries
            if entry.test == 2] == ['test_opened']
    assert extract.verify().ok


def test_a_report_never_changes_because_the_log_grew(tmp_path):
    """The views and the drafts stay as taken; the badge says the log
    holds newer entries, and how many are of the test."""
    path = tmp_path / 'a.vdlog'
    two_tests(path)
    extract = take(path, test=1)
    before = (views(extract, None), extract.draft_chronology().text,
              extract.draft_findings().text, extract.head)
    assert (extract.status().state, extract.status().newer,
            extract.status().newer_in_test) == ('current', 0, 0)
    assert extract.status().badge == ''
    append_note(path, test=1, text='a second rattle')
    append_note(path, test=1, text='a third')
    append_note(path, test=2)
    assert (views(extract, None), extract.draft_chronology().text,
            extract.draft_findings().text, extract.head) == before
    status = extract.status()
    assert (status.state, status.newer, status.newer_in_test) == ('newer', 3, 2)
    assert status.badge == 'newer entries in the log'
    assert 'newer entries in the log: 3 after entry #' in status.message


def test_an_extract_to_an_earlier_entry_knows_the_entries_after_it(tmp_path):
    """Taken short of the log's end, the file's length is unchanged, and
    the badge still counts what was after the cut."""
    path = tmp_path / 'a.vdlog'
    ids = two_tests(path)
    extract = take(path, test=1, upto=ids['anomaly'])
    assert extract.upto == ids['anomaly']
    assert extract.state == 'open', 'the roll came after the cut'
    assert all(entry.id <= ids['anomaly'] for entry in extract.log.entries)
    status = extract.status()
    assert status.state == 'newer'
    assert status.newer == ids['last'] - ids['anomaly']
    assert status.newer_in_test == 1, 'the roll closing it'


def test_an_entry_moved_into_the_test_by_correction_comes_with_it(tmp_path):
    path = tmp_path / 'a.vdlog'
    two_tests(path)
    with writer(path) as log:
        stray = log.append('note', author='AB', test=2, text='of the random')
        log.correct(stray, author='AB', fields={'test': 1})
    extract = take(path, test=1)
    assert stray.id in [entry.id for entry in extract.entries]
    live = model.RunLogModel(RunLog.read(path))
    assert views(extract, None) == views(live, 1)


def test_a_test_with_nothing_but_its_opening(tmp_path):
    path = tmp_path / 'a.vdlog'
    with writer(path) as log:
        model.open_test(log, author='AB', test_kind='Functional',
                        header={'title': 'Baseline'})
    extract = take(path)
    assert extract.test == 1
    assert extract.runs() == extract.findings() == extract.files() == []
    assert len(extract.entries) == 1
    chronology = extract.draft_chronology()
    assert not chronology and chronology.text == ''
    assert chronology.notes == ('The log holds no runs of test 1 by entry #1.',)
    assert not extract.draft_findings()
    assert 'whether there were none is the author' in \
        extract.draft_findings().notes[0]


# ---- refusals ------------------------------------------------------------------


def test_a_file_that_is_not_a_run_log_is_refused(tmp_path):
    other = tmp_path / 'notes.vdlog'
    other.write_text('{"hello": 1}\n', encoding='utf-8')
    with pytest.raises(RunLogError, match='not a Visual Dynamics run log'):
        take(other)
    with pytest.raises(FileNotFoundError, match='no run log at'):
        take(tmp_path / 'absent.vdlog')
    empty = tmp_path / 'empty.vdlog'
    RunLog.create(empty, who=WHO).close()
    with pytest.raises(RunLogError, match='holds no entries yet'):
        take(empty)


def test_no_such_test_or_entry_is_refused_by_name(tmp_path):
    path = tmp_path / 'a.vdlog'
    ids = two_tests(path)
    with pytest.raises(RunLogError, match=r'no test 3 by entry #\d+; its '
                       r'tests are 1, 2'):
        take(path, test=3)
    with pytest.raises(RunLogError, match=r'holds 2 tests by entry #\d+ '
                       r'\(1, 2\); say which'):
        take(path)
    with pytest.raises(RunLogError, match=r'no test 2 by entry #3; its tests '
                       r'are 1'):
        take(path, test=2, upto=3)
    with pytest.raises(RunLogError, match=rf'no entry #99; its entries run '
                       rf'from #1 to #{ids["last"]}'):
        take(path, test=1, upto=99)
    for bad in (0, True, 1.5):
        with pytest.raises(RunLogError, match='whole number from 1'):
            take(path, test=bad)


def test_a_broken_chain_is_refused_with_where(tmp_path):
    """An entry of the test edited in the file: the refusal names the
    chain, the entry and the line."""
    path = tmp_path / 'a.vdlog'
    two_tests(path)
    lines = path.read_bytes().split(b'\n')
    number = next(k for k, line in enumerate(lines)
                  if b'"current": 63.0' in line)
    entry = json.loads(lines[number])['id']
    lines[number] = lines[number].replace(b'"current": 63.0',
                                          b'"current": 36.0')
    path.write_bytes(b'\n'.join(lines))
    with pytest.raises(RunLogError, match=(
            rf'chain does not verify at test 1, entry #{entry} \(line '
            rf'{number + 1}\): its hash does not match its contents')):
        take(path, test=1)
    # the cut before the damage is whole, and each test's chain checks
    # alone: the shock test's own entries do not rest on the random's
    assert take(path, test=1, upto=entry - 1).upto == entry - 1


def test_a_broken_chain_of_another_test_does_not_refuse_this_one(tmp_path):
    path = tmp_path / 'a.vdlog'
    two_tests(path)
    lines = path.read_bytes().split(b'\n')
    number = next(k for k, line in enumerate(lines) if b'"shot"' in line)
    lines[number] = lines[number].replace(b'"n": 1', b'"n": 2')
    path.write_bytes(b'\n'.join(lines))
    assert take(path, test=1).verify().ok
    with pytest.raises(RunLogError, match='chain does not verify at test 2'):
        take(path, test=2)


def test_a_torn_last_line_is_refused_until_an_entry_is_named(tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    with open(path, 'ab') as handle:
        handle.write(b'{"id": 3, "kind": "no')
    lines = len(path.read_bytes().split(b'\n'))
    with pytest.raises(RunLogError, match=rf'line {lines}: the last line is '
                       'torn'):
        take(path)
    assert take(path, upto=2).upto == 2


def test_a_fragment_the_writer_ended_is_skipped_and_named(tmp_path):
    """A crash fragment ended by the next writer is not an entry: the log
    skips it, the chain checks across it, and the extract says so."""
    path = tmp_path / 'a.vdlog'
    one_test(path)
    with open(path, 'ab') as handle:
        handle.write(b'{"id": 3, "kind": "no')
    append_note(path)
    extract = take(path)
    assert extract.upto == 3
    assert len(extract.problems) == 1
    assert extract.problems[0].startswith('line 4: ')


def test_an_entry_the_log_cannot_read_before_the_cut_is_refused(tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    append_note(path)
    lines = path.read_bytes().split(b'\n')
    lines[2] = lines[2].replace(b'"kind": "level_row"', b'"kind": "dance"')
    path.write_bytes(b'\n'.join(lines))
    with pytest.raises(RunLogError, match='line 3: an entry the log cannot '
                       'read'):
        take(path)
    assert take(path, upto=1).upto == 1


# ---- the live file ---------------------------------------------------------------


def test_a_missing_or_changed_log_is_said_not_raised(tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    extract = take(path)
    os.replace(path, tmp_path / 'moved.vdlog')
    status = extract.status()
    assert (status.state, status.badge) == ('missing', 'log not found')
    assert (status.newer, status.newer_in_test) == (0, 0)
    assert str(path) in status.message
    assert extract.status(tmp_path / 'moved.vdlog').state == 'current'
    # another log at the same path
    one_test(path)
    append_note(path)
    assert (extract.status().state, extract.status().newer,
            extract.status().newer_in_test) == ('changed', 0, 0)
    assert 'another run log' in extract.status().message
    # the same log cut short: its head is gone
    os.replace(tmp_path / 'moved.vdlog', path)
    append_note(path)
    whole = path.read_bytes()
    lines = whole.split(b'\n')
    path.write_bytes(b'\n'.join(lines[:2]) + b'\n')
    status = extract.status()
    assert (status.state, status.newer, status.newer_in_test) == (
        'changed', 0, 0)
    assert 'no longer holds the last entry of test 1' in status.message
    path.write_bytes(b'not a log at all\n')
    assert (extract.status().state, extract.status().newer,
            extract.status().newer_in_test) == ('unreadable', 0, 0)
    assert extract.status().badge == 'log unreadable'


# ---- drafts ------------------------------------------------------------------------


def test_the_drafts_are_marked_and_offer_only_the_fact(tmp_path):
    path = tmp_path / 'a.vdlog'
    two_tests(path)
    extract = take(path, test=1)
    chronology = extract.draft_chronology()
    assert chronology.mark == DRAFTED == ('DRAFTED FROM THE RUN LOG, NOT YET '
                                          'ACCEPTED')
    assert chronology.section == 'chronology'
    lines = chronology.text.splitlines()
    assert lines[2] == ('| 1 | Z | 2026-10-30 08:05:02 | levels -12 dB, 0 dB '
                        '| 4:00 | aborted |'), 'on the recorders\' clock'
    assert ('| #6 | 2026-10-30 08:07:02 | amplifier trip | a loose bolt on '
            'the slip table | not logged | none logged |') in lines
    assert chronology.text.endswith('is the author\'s to say.')
    findings = extract.draft_findings()
    assert findings.section == 'deviations'
    disposition, = [entry.id for entry in extract.log.entries
                    if entry.kind == 'anomaly_dispositioned']
    assert findings.text.splitlines()[:2] == [
        ('- **D-1**, deviation (log entry #7). Fact: a notch at 180 Hz not in '
        'the plan'),
        ('- **A-1**, anomaly (log entry #8), owner CD, a disposition is logged '
        f'(#{disposition}). Fact: a rattle at 300 Hz')]
    assert 'loose bolt' not in findings.text, 'a cause is the author\'s'
    assert 'use as is' not in findings.text, 'a disposition is the author\'s'
    assert 'EF' not in findings.text
    assert findings.source == (f"the run log 'Article 7', test 1, to entry "
                               f'#{extract.upto} (test head '
                               f'{extract.head[:12]})')
    assert "Times are on the recorders' clock." in chronology.text


def test_a_run_with_no_interruption_offers_no_none(tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    draft = take(path).draft_chronology()
    assert draft
    assert 'none' not in draft.text.lower()
    assert '| Entry |' not in draft.text
    assert 'whether the test ran without one is the author' in draft.notes[0]
    assert 'log computer\'s clock' in draft.text, 'no clock check was logged'


# ---- the project ---------------------------------------------------------------------


def test_attach_detach_and_refresh_are_journaled_and_undone(tmp_path):
    path = tmp_path / 'a.vdlog'
    ids = two_tests(path)
    project = Project('t')
    name = project.attach_run_log(path, test=1, upto=ids['anomaly'])
    assert name == 'Run Log, Test 1'
    assert project.journal[-1] == (f'project.attach_run_log({str(path)!r}, '
                                   f"test=1, upto={ids['anomaly']})")
    first = project[name]
    assert project.run_log_extract is first
    assert project.run_log_extracts == [first]
    taken = project.refresh_run_log(name)
    assert project.journal[-1] == f'project.refresh_run_log({name!r})'
    assert project[name] is taken and taken.upto == ids['last']
    # undone: refreshed back to the entry it had
    again = project.refresh_run_log(name, upto=first.upto)
    assert project.journal[-1] == (f'project.refresh_run_log({name!r}, '
                                   f'upto={first.upto})')
    assert again.text == first.text and again.head == first.head
    removed = project.detach_run_log(name)
    assert project.journal[-1] == f'project.detach_run_log({name!r})'
    assert name not in project and removed is again
    # undone: attached again with the same test and entry, even after the
    # log grew
    append_note(path)
    back = project.attach_run_log(path, test=1, upto=first.upto)
    assert project[back].text == first.text


def test_the_verbs_refuse_what_is_not_an_extract_and_record_nothing(tmp_path):
    from visualdynamics.core.report import Report

    path = tmp_path / 'a.vdlog'
    one_test(path)
    project = Project('t')
    project.add('Report', Report('R'))
    lines = list(project.journal)
    with pytest.raises(TypeError, match="'Report' is a Report, not a run log "
                       'extract'):
        project.detach_run_log('Report')
    with pytest.raises(TypeError, match='not a run log extract'):
        project.refresh_run_log('Report')
    with pytest.raises(RunLogError, match='no test 4'):
        project.attach_run_log(path, test=4)
    assert project.journal == lines
    assert list(project) == ['Report']


def test_a_refresh_refuses_a_log_that_changed_under_it(tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    project = Project('t')
    name = project.attach_run_log(path)
    held = project[name]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        project.refresh_run_log(name)
    one_test(path)                                  # another log, same path
    append_note(path)
    with pytest.raises(RunLogError, match='another run log now'):
        project.refresh_run_log(name)
    assert project[name] is held


def test_the_project_saves_and_reopens_with_the_extract(tmp_path):
    path = tmp_path / 'a.vdlog'
    two_tests(path)
    project = Project('t')
    name = project.attach_run_log(path, test=1)
    saved = tmp_path / 'project.vdyn'
    project.save(saved)
    with h5py.File(saved, 'r') as f:
        assert f.attrs['visualdynamics_schema'] == 3
        group = next(f['objects'][key] for key in f['objects'])
        assert group.attrs['kind'] == 'run_log_extract'
        assert group.attrs['path'] == 'a.vdlog', 'beside the project: relative'
    reopened = Project.open(saved)
    assert reopened[name] == project[name]
    assert reopened[name].path == str(path)
    assert views(reopened[name], None) == views(project[name], None)
    assert 'test 1, Random Vibration' in repr(reopened)


def test_the_path_is_relative_beside_the_project_and_absolute_otherwise(
        tmp_path):
    inner = tmp_path / 'job' / 'logs'
    inner.mkdir(parents=True)
    away = tmp_path / 'share'
    away.mkdir()
    project = Project('t')
    for folder in (inner, away):
        one_test(folder / 'a.vdlog')
        project.attach_run_log(folder / 'a.vdlog', name=folder.name)
    saved = tmp_path / 'job' / 'project.vdyn'
    project.save(saved)
    with h5py.File(saved, 'r') as f:
        stored = {group.attrs['name']: group.attrs['path']
                  for group in f['objects'].values()}
    assert stored == {'logs': 'logs/a.vdlog',
                      'share': str(away / 'a.vdlog')}
    # moved with its log, the project still finds it; the other stays
    moved = tmp_path / 'moved'
    os.replace(tmp_path / 'job', moved)
    reopened = Project.open(moved / 'project.vdyn')
    assert reopened['logs'].path == str(moved / 'logs' / 'a.vdlog')
    assert reopened['logs'].status().state == 'current'
    assert reopened['share'].path == str(away / 'a.vdlog')


def test_a_missing_log_still_opens_the_project_and_says_so(tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    project = Project('t')
    name = project.attach_run_log(path)
    project.save(tmp_path / 'p.vdyn')
    path.unlink()
    reopened = Project.open(tmp_path / 'p.vdyn')
    extract = reopened[name]
    assert extract.status().state == 'missing'
    assert len(extract.runs()) == 1, 'the snapshot reads without the log'
    assert extract.draft_chronology()


def test_a_file_without_an_extract_keeps_its_old_stamp(tmp_path):
    """Only a file holding an extract needs schema 3, so a build that
    reads schema 2 still opens every other file this build writes."""
    from visualdynamics.core.report import Report

    project = Project('t')
    project.add('Report', Report('R'))
    project.save(tmp_path / 'plain.vdyn')
    with h5py.File(tmp_path / 'plain.vdyn', 'r') as f:
        assert f.attrs['visualdynamics_schema'] == 1


def test_a_build_reading_schema_2_refuses_a_file_with_an_extract(
        tmp_path, monkeypatch):
    """What an older Visual Dynamics does with a project holding a log:
    it refuses the whole file with its message to update, and loads
    nothing of it."""
    path = tmp_path / 'a.vdlog'
    one_test(path)
    project = Project('t')
    project.attach_run_log(path)
    project.save(tmp_path / 'held.vdyn')
    monkeypatch.setattr(native, 'SCHEMA_VERSION', 2)
    with pytest.raises(ValueError, match=(
            r'held\.vdyn was written by a newer Visual Dynamics \(schema 3; '
            r'this build reads up to 2\)\. Update to open it\.')):
        Project.open(tmp_path / 'held.vdyn')


def test_an_extract_saved_alone_round_trips(tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    extract = take(path, taken=TAKEN)
    native.save(extract, tmp_path / 'alone.vdyn')
    again = native.load(tmp_path / 'alone.vdyn')
    assert again == extract
    assert native.schema_for(extract) == 3


def test_the_window_shows_a_project_holding_an_extract(window, pump,
                                                       tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    project = Project('t')
    name = project.attach_run_log(path)
    project.save(tmp_path / 'p.vdyn')
    window.import_paths([str(tmp_path / 'p.vdyn')])
    assert isinstance(window.objects[name], RunLogExtract)
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()


# ---- the paths, and the report's hook ----------------------------------------------


def test_stored_and_resolved_paths(tmp_path):
    project = tmp_path / 'p.vdyn'
    beside = tmp_path / 'logs' / 'a.vdlog'
    assert stored_path(str(beside), str(project)) == 'logs/a.vdlog'
    assert resolved_path('logs/a.vdlog', str(project)) == str(beside)
    above = tmp_path.parent / 'a.vdlog'
    assert stored_path(str(above), str(project)) == str(above)
    assert stored_path(str(beside), None) == str(beside)
    assert resolved_path('logs/a.vdlog', None) == 'logs/a.vdlog'
    # an absolute path from either platform is never joined to a folder
    for absolute in ('C:\\logs\\a.vdlog', '/srv/logs/a.vdlog'):
        assert resolved_path(absolute, str(project)) == absolute
    assert stored_path('logs/a.vdlog', str(project)) == 'logs/a.vdlog'


def test_a_report_feature_can_require_the_extract():
    capability = rf.CAPABILITIES['run_log_extract']
    assert capability.probe()
    assert capability.label == 'the run log brought into a project'
    assert extract_module.CHRONOLOGY_SECTION == 'chronology'
    assert extract_module.FINDINGS_SECTION == 'deviations'


def test_an_extract_refuses_a_snapshot_that_does_not_hold_what_it_says(
        tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    extract = take(path, taken=TAKEN)
    fields = {'path': extract.path, 'taken': TAKEN, 'log_size': 1}
    with pytest.raises(RunLogError, match='no test 2'):
        RunLogExtract(extract.text, test=2, upto=2, **fields)
    with pytest.raises(RunLogError, match='an extract to entry #1 holds entry '
                       '#2'):
        RunLogExtract(extract.text, test=1, upto=1, **fields)
    with pytest.raises(RunLogError, match='with its zone'):
        RunLogExtract(extract.text, test=1, upto=2, path=extract.path,
                      taken=datetime(2026, 10, 30), log_size=1)
    with pytest.raises(RunLogError, match='not a Visual Dynamics run log'):
        RunLogExtract('{}\n', test=1, upto=2, **fields)


def test_a_refresh_refuses_a_log_cut_short_under_it(tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    append_note(path)
    project = Project('t')
    name = project.attach_run_log(path)
    held = project[name]
    lines = path.read_bytes().split(b'\n')
    path.write_bytes(b'\n'.join(lines[:3]) + b'\n')
    with pytest.raises(RunLogError, match='no longer holds the last entry of '
                       'test 1.*Attach it again'):
        project.refresh_run_log(name)
    assert project[name] is held


def test_the_file_is_read_only_when_its_length_changed(tmp_path,
                                                       monkeypatch):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    extract = take(path)
    read = []
    monkeypatch.setattr(RunLog, 'read', classmethod(
        lambda cls, where: read.append(where)))
    assert extract.status().state == 'current'
    assert read == [], 'the same length: not read'


def test_a_last_line_with_no_line_end_that_was_read_is_kept(tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    path.write_bytes(path.read_bytes().rstrip(b'\n'))
    extract = take(path)
    assert (extract.upto, extract.problems) == (2, ())


def test_a_damaged_entry_of_another_test_that_answers_this_one_refuses(
        tmp_path):
    """The cause written under the shock test is kept in the random
    test's extract, so it is checked too, though its chain is another."""
    path = tmp_path / 'a.vdlog'
    two_tests(path)
    lines = path.read_bytes().split(b'\n')
    number = next(k for k, line in enumerate(lines)
                  if b'"cause_found"' in line)
    lines[number] = lines[number].replace(b'loose bolt', b'loose nut')
    path.write_bytes(b'\n'.join(lines))
    with pytest.raises(RunLogError, match=rf'at test 2, entry #\d+ \(line '
                       rf'{number + 1}\): its hash does not match'):
        take(path, test=1)


def test_what_an_extract_says_of_itself(tmp_path):
    path = tmp_path / 'a.vdlog'
    one_test(path)
    extract = take(path, taken=TAKEN)
    assert repr(extract) == "RunLogExtract(test 1 of 'Article 7', to entry #2)"
    plain = RunLogExtract(extract.text, path=extract.path, test=1, upto=2,
                          taken=TAKEN.isoformat(), log_size=extract.log_size)
    assert (plain.after, plain.after_in_test, plain.problems) == (0, 0, ())
    assert plain.taken == TAKEN
    untitled = tmp_path / 'untitled.vdlog'
    with RunLog.create(untitled, clock=Ticking(), who=WHO) as log:
        model.open_test(log, author='AB', test_kind='Other')
    assert repr(take(untitled)) == ("RunLogExtract(test 1 of "
                                    "'untitled.vdlog', to entry #1)")


def test_the_chronology_draft_says_what_each_run_held(tmp_path):
    """Shots counted, and each restart worded by whether it resumed."""
    path = tmp_path / 'a.vdlog'
    with writer(path) as log:
        log.append('clock_check', author='AB', fields={'offset_s': 0.0})
        model.open_test(log, author='AB', test_kind='Shock',
                        header={'title': 'Release shock'})
        model.append(log, 'shot', author='AB', test=1, axis='Z', run=1,
                     fields={'n': 1, 'of': 3})
        first = model.append(log, 'abort', author='AB', test=1, axis='Z',
                             run=1, fields={'stop': 'stop word'})
        model.append(log, 'restart', author='AB', test=1, axis='Z', run=1,
                     fields={'resume': True, 'counter_at_restart': '0:30',
                             'refers_to': first.id})
        for n in (2, 3):
            model.append(log, 'shot', author='AB', test=1, axis='Z', run=2,
                         fields={'n': n, 'of': 3})
        second = model.append(log, 'abort', author='AB', test=1, axis='Z',
                              run=2, fields={'stop': 'power loss'})
        model.append(log, 'restart', author='AB', test=1, axis='Z', run=2,
                     fields={'resume': False, 'refers_to': second.id})
        third = model.append(log, 'run_stop', author='AB', test=1, axis='Z',
                             run=2, fields={'scheduled': True})
        model.append(log, 'restart', author='AB', test=1, axis='Z', run=2,
                     fields={'refers_to': third.id})
    rows = take(path).draft_chronology().text.splitlines()
    assert rows[2].split(' | ')[3] == '1 shot'
    assert rows[3].split(' | ')[3] == '2 shots'
    stops = [row.split(' | ') for row in rows if row.startswith('| #')]
    assert [cells[2] for cells in stops] == ['stop word', 'power loss',
                                            'scheduled stop']
    assert [cells[-1] for cells in stops] == ['resumed at 0:30 |',
                                             'started again |', 'restarted |']


def test_a_project_saved_before_schema_3_opens_and_takes_a_log(tmp_path):
    from conftest import fixture_path

    project = Project.open(fixture_path('vdyn_corpus', 'schema2.vdyn'))
    assert 'Predictions' in project
    path = tmp_path / 'a.vdlog'
    one_test(path)
    name = project.attach_run_log(path)
    project.save(tmp_path / 'later.vdyn')
    with h5py.File(tmp_path / 'later.vdyn', 'r') as f:
        assert f.attrs['visualdynamics_schema'] == 3
    again = Project.open(tmp_path / 'later.vdyn')
    assert again[name] == project[name]
    assert again['Predictions'].comment == 'schema exercise predictions'


def test_a_report_cites_the_extract_with_its_badge(tmp_path):
    path = tmp_path / 'article.vdlog'
    two_tests(path)
    project = Project('Article 7')
    name = project.attach_run_log(path, test=1)
    extract = project[name]
    assert extract_module.held(project) == [(name, extract)]
    assert extract_module.held({'other': 1}) == []
    plain = (f"the run log 'Article 7', test 1, to entry #{extract.upto} "
             f'(test head {extract.head[:12]})')
    assert extract.cited() == plain
    append_note(path, test=2)
    assert extract.cited() == f'{plain} [newer entries in the log]'
    assert extract.cited(extract.status()) == extract.cited()
    os.remove(path)
    assert extract.cited() == f'{plain} [log not found]'
    # the pieces of the chronology draft a report feature reuses
    assert extract.interruption_lines()[0] == ('| Entry | Time | Stop | Cause '
                                               '| Path | Restart |')
    assert extract.credit_lines() == [f'- Axis Z: {credit.message}'
                                      for credit in extract.credited()]
    for line in extract.interruption_lines() + extract.credit_lines():
        assert line in extract.draft_chronology().text
