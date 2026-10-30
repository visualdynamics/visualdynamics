"""Runs and chronology (report feature `run_sections`).

The run identity a record carries (`core.runs`: read from a name,
entered, refused when it cannot be read one way), what a record shows
about its run (the time at level and the interruptions, from the
running RMS, checked against hand values and against noise at the
default settings), and the feature on a project: the identity with the
comparison's level, the chronology offered as a draft the author
accepts, the Results grouped by run and repeated for every run, the
editor's panels, and the page unchanged with the feature off; and a
run log brought into the project as a second source of outcomes,
causes and credited duration, the draft unchanged without one.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import warnings

import numpy as np
import pytest
from conftest import fixture_path
from test_run_log_extract import HEADER, writer

import visualdynamics
from visualdynamics.core import compliance
from visualdynamics.core import run_log_model as model
from visualdynamics.core import runs as cr
from visualdynamics.core.data import Psd, TimeHistory
from visualdynamics.core.report import Report
from visualdynamics.io import ImportNote
from visualdynamics.report import page, render_html
from visualdynamics.report import runs as rr
from visualdynamics.report import skeleton as sk

FEATURES = ['skeleton_sections', 'run_sections']
RATE = 1024.0


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


def _render(project, name, **kwargs):
    return render_html(project[name], project, unit_system=visualdynamics.SI,
                       object_groups=project.object_groups, **kwargs)


# ---- reading a name -----------------------------------------------------------


@pytest.mark.parametrize(('name', 'expected'), [
    ('Z_run07_-6dB_2026-10-11T111105.nc4',
     {'axis': 'Z', 'run': '7', 'level_db': -6.0,
      'start': '2026-10-11T11:11:05'}),
    ('x-axis R3 full 20261011.nc4',
     {'axis': 'X', 'run': '3', 'level_db': 0.0, 'start': '2026-10-11'}),
    ('Yaxis_run_12_m12dB_2026-10-11_0930.nc4',
     {'axis': 'Y', 'run': '12', 'level_db': -12.0,
      'start': '2026-10-11T09:30:00'}),
    ('C:\\tests\\Z_r2_minus4.5dB', {'axis': 'Z', 'run': '2', 'level_db': -4.5}),
    ('Z_run2_+3dB_FL.nc4', None),
    ('random.nc4', {}),
    ('random_30-60s.nc4', {}),
    ('serial_12345678_rerun.nc4', {}),
    ('Z_run1_0dB_2026-10-11.nc4', {'axis': 'Z', 'run': '1', 'level_db': 0.0,
                                   'start': '2026-10-11'}),
])
def test_a_name_gives_what_it_says_and_nothing_else(name, expected):
    values, refused = cr.read_name(name)
    if expected is None:
        assert 'level_db' in refused, '+3 dB and full level are two levels'
        assert values == {'axis': 'Z', 'run': '2'}
        return
    assert values == expected
    assert refused == {}


def test_what_a_name_says_two_ways_is_refused_and_why():
    values, refused = cr.read_name('Z_X_run3_run4.nc4')
    assert values == {}
    assert refused == {'axis': 'the axis is read two ways (X, Z)',
                       'run': 'the run number is read two ways (3, 4)'}
    values, refused = cr.read_name('Z_run3_6dB.nc4')
    assert values == {'axis': 'Z', 'run': '3'}
    assert refused['level_db'].startswith("'6dB' has no sign")
    # across several names: a file name and an environment's
    values, refused = cr.read_name('Z_run7.nc4', 'Random X')
    assert values == {'run': '7'} and 'axis' in refused
    values, refused = cr.read_name('run7.nc4', 'Random Z')
    assert values == {'run': '7', 'axis': 'Z'} and refused == {}
    # a date alone that a date and time also give is the same start
    values, _refused = cr.read_name('2026-10-11_run1_2026-10-11T0930')
    assert values['start'] == '2026-10-11T09:30:00'


def test_an_entry_is_checked_and_refused_by_name():
    assert cr.entered('TH', axis=' z ', run=7, level_db=-6, start='2026-10-11',
                      comment='after the axis change') == {
        'axis': 'Z', 'run': '7', 'level_db': -6.0, 'start': '2026-10-11',
        'comment': 'after the axis change'}
    assert cr.entered('TH', axis='', level_db=None) == {'axis': None,
                                                        'level_db': None}
    import datetime as dt

    assert cr.entered('TH', start=dt.datetime(2026, 10, 11, 9, 30)) == {
        'start': '2026-10-11T09:30:00'}
    with pytest.raises(ValueError, match="TH: the level is a finite"):
        cr.entered('TH', level_db=float('nan'))
    with pytest.raises(TypeError, match='TH: the level is a number'):
        cr.entered('TH', level_db=True)
    with pytest.raises(TypeError, match='TH: the level is a number'):
        cr.entered('TH', level_db='-6')
    with pytest.raises(ValueError, match="TH: the start 'Tuesday' is not"):
        cr.entered('TH', start='Tuesday')
    with pytest.raises(ValueError, match='TH: the axis is one line'):
        cr.entered('TH', axis='Z\nX')
    with pytest.raises(ValueError, match="TH: 'speed' is not a field"):
        cr.entered('TH', speed=3)
    with pytest.raises(TypeError, match='TH: the comment is text'):
        cr.entered('TH', comment=3)


def test_an_identity_labels_itself_and_round_trips():
    identity = cr.RunIdentity(axis='Z', run='7', level_db=-6.0,
                              start='2026-10-11T11:11:05',
                              sources=(('axis', cr.FROM_NAME),),
                              refused=(('comment', 'a reason'),))
    assert identity.label() == 'run 7, Z axis, -6 dB, 2026-10-11 11:11:05'
    assert cr.RunIdentity(level_db=0.0).label() == '0 dB'
    assert cr.RunIdentity(level_db=3.0).label() == '+3 dB'
    assert cr.RunIdentity(comment='only a note').known is False
    assert cr.RunIdentity().label() == ''
    back = cr.RunIdentity.from_json(json.loads(json.dumps(identity.to_json())))
    assert back.label() == identity.label()
    assert back.source('axis') == cr.FROM_NAME
    assert back.source('run') == cr.ENTERED, 'a field with no source: entered'
    assert back.reason('comment') == 'a reason'
    cleared = identity.with_values({'axis': '', 'level_db': None}, '')
    assert cleared.axis == '' and cleared.level_db is None
    assert cleared.source('axis') == ''
    with pytest.raises(ValueError, match='TH: the refusals'):
        cr.RunIdentity.from_json({'refused': {'speed': 'x'}}, 'TH')
    with pytest.raises(TypeError, match='TH: a run identity is a mapping'):
        cr.RunIdentity.from_json(['axis'], 'TH')


def test_runs_are_ordered_by_start_then_by_run_or_refused_by_name():
    a = cr.RunIdentity(run='10', start='2026-10-12T08:00')
    b = cr.RunIdentity(run='2', start='2026-10-11T09:00')
    c = cr.RunIdentity(run='3', start='2026-10-11')
    assert cr.order([('a', a), ('b', b), ('c', c)]) == ['c', 'b', 'a']
    # some without a start: by run number, numerically
    assert cr.order([('a', a), ('d', cr.RunIdentity(run='9'))]) == ['d', 'a']
    with pytest.raises(ValueError, match="'e' has no start and no run"):
        cr.order([('a', a), ('e', cr.RunIdentity(axis='Z'))])
    assert cr.order([('e', cr.RunIdentity())]) == ['e'], 'a lone run'


# ---- reading a record -----------------------------------------------------------


def _alternating(seconds, channels=2):
    """+1, -1, +1 ... on every channel: a mean square of exactly one in
    any window, so the half-power crossing falls on a sample."""
    count = int(seconds * RATE)
    return np.tile(np.where(np.arange(count) % 2, -1.0, 1.0), (channels, 1))


def _moved(ratio):
    """How far, in seconds, the window (an odd number of samples) moves
    an edge between full level and a level `ratio` of its mean square."""
    share = (0.5 - ratio) / (1.0 - ratio)
    return (0.5 - share) * (2 * round(cr.WINDOW_S * RATE / 2) + 1) / RATE


def test_a_stop_mid_run_is_an_interruption_timed_to_the_sample():
    values = _alternating(30.0)
    values[:, int(10 * RATE):int(14 * RATE)] = 0.0
    found = cr.read_levels(values, RATE, ['1Z+', '2Z+'])
    assert found.not_read == ''
    assert found.channels == ('1Z+', '2Z+')
    assert found.seconds == pytest.approx(30.0)
    assert found.at_level_s == pytest.approx(26.0, abs=0.5 / RATE)
    (stop,) = found.interruptions
    assert stop.start_s == pytest.approx(10.0, abs=0.5 / RATE)
    assert stop.seconds == pytest.approx(4.0, abs=0.5 / RATE)
    assert stop.depth_db is None and str(stop) == 'at 10.0 s for 4.0 s, to zero'
    # on the record's own clock
    later = cr.read_levels(values, RATE, start_s=100.0)
    assert later.interruptions[0].start_s == pytest.approx(110.0,
                                                            abs=0.5 / RATE)


def test_a_deep_drop_reads_its_depth_and_a_shallow_or_short_one_is_not_listed():
    deep = 3.0 * _alternating(30.0)
    deep[:, int(10 * RATE):int(14 * RATE)] *= 0.01
    (stop,) = cr.read_levels(deep, RATE).interruptions
    assert stop.depth_db == pytest.approx(-40.0, abs=0.1)
    assert str(stop) == 'at 10.0 s for 4.0 s, 40 dB down'
    shallow = _alternating(30.0)
    shallow[:, int(10 * RATE):int(14 * RATE)] *= 10 ** (-4 / 20)
    found = cr.read_levels(shallow, RATE)
    assert found.interruptions == (), 'a 4 dB dip is not 6 dB down'
    # but it is out of level, more than 3 dB down. Between two levels
    # that are not silence the window moves each edge: a window with a
    # share f of the louder level reads f + (1 - f) r of it, and is at
    # level from f = (1/2 - r)/(1 - r), so each edge moves in by
    # (1/2 - f) of a window, never more than half of one
    assert found.at_level_s == pytest.approx(
        26.0 + 2 * _moved(10 ** (-4 / 10)), abs=2 / RATE)
    short = _alternating(30.0)
    short[:, int(10 * RATE):int(10.5 * RATE)] = 0.0
    found = cr.read_levels(short, RATE)
    assert found.interruptions == (), 'half a second is under the least'
    assert found.at_level_s == pytest.approx(29.5, abs=2 / RATE)
    assert cr.read_levels(short, RATE, min_s=0.25).interruptions, \
        'the least duration is a setting'


def test_the_lead_in_and_the_tail_are_not_interruptions():
    values = _alternating(30.0)
    values[:, :int(5 * RATE)] = 0.0
    values[:, int(25 * RATE):] = 0.0
    found = cr.read_levels(values, RATE)
    assert found.interruptions == ()
    assert found.at_level_s == pytest.approx(20.0, abs=0.5 / RATE)
    assert cr.read_levels(values, RATE, min_s=0).interruptions == (), \
        'not even a sample of the tail'
    # a step-up: the lower steps are not the run's level, and not stops
    stepped = _alternating(30.0)
    stepped[:, :int(10 * RATE)] *= 10 ** (-12 / 20)
    stepped[:, int(10 * RATE):int(20 * RATE)] *= 10 ** (-6 / 20)
    found = cr.read_levels(stepped, RATE)
    assert found.interruptions == ()
    assert found.at_level_s == pytest.approx(
        10.0 + _moved(10 ** (-6 / 10)), abs=2 / RATE)


def test_white_noise_at_the_defaults_is_at_level_throughout():
    """The rule against simulated noise: forty records of white noise,
    every one read at level from end to end, with no interruption."""
    rng = np.random.default_rng(27)
    for _ in range(40):
        values = rng.standard_normal((4, int(30 * RATE)))
        found = cr.read_levels(values, RATE)
        assert found.interruptions == ()
        assert found.at_level_s == pytest.approx(30.0)
    # and a tone of a few hertz, which ripples the running RMS
    t = np.arange(int(30 * RATE)) / RATE
    found = cr.read_levels(np.sin(2 * np.pi * 5.0 * t), RATE)
    assert found.interruptions == () and found.at_level_s == \
        pytest.approx(30.0)


def test_dead_nan_and_empty_inputs_say_what_was_read_on():
    values = _alternating(10.0, channels=4)
    values[1] = 0.0
    values[2] = 3.0
    values[3, 100] = np.nan
    found = cr.read_levels(values, RATE, ['a', 'b', 'c', 'd'])
    assert found.channels == ('a',)
    assert found.left_out == ('b: all zero', 'c: constant',
                              'd: holds a sample that is not a number')
    assert found.at_level_s == pytest.approx(10.0)
    dead = cr.read_levels(np.zeros((2, 2048)), RATE, ['a', 'b'])
    assert dead.at_level_s is None and dead.interruptions == ()
    assert dead.not_read == 'no channel varies: a: all zero; b: all zero'
    empty = cr.read_levels(np.empty((0, 2048)), RATE)
    assert empty.not_read == 'no channel to read the level on'
    short = cr.read_levels(np.ones((1, 100)), RATE)
    assert short.not_read.startswith('the record (0.0977 s) is shorter than')
    for kwargs, match in (({'window_s': 0}, 'longer than zero'),
                          ({'drop_db': 3.0}, 'further below the level'),
                          ({'min_s': -1}, 'zero seconds or more'),
                          ({'window_s': float('nan')}, 'finite number')):
        with pytest.raises(ValueError, match=match):
            cr.read_levels(values, RATE, **kwargs)
    with pytest.raises(ValueError, match='sample rate is positive'):
        cr.read_levels(values, 0.0)


# ---- the comparison's level ---------------------------------------------------


def test_the_detection_says_when_it_read_nothing():
    """`read_scale_db` is `detect_scale_db` with its 'not clearly' kept
    apart from 0 dB, a reading of full level."""
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'))
    spec = project['Specification']
    measured = project['Time History PSDs']
    assert compliance.read_scale_db(spec, measured) is None
    assert compliance.detect_scale_db(spec, measured) == 0
    # the specification's own autospectra, held 6 dB low
    held = Psd(spec.abscissa, spec.ordinate * 10 ** (-6 / 10),
               **spec.record_fields())
    assert compliance.read_scale_db(spec, held) == 6
    assert compliance.detect_scale_db(spec, held) == 6
    same = Psd(spec.abscissa, spec.ordinate.copy(), **spec.record_fields())
    assert compliance.read_scale_db(spec, same) == 0


# ---- the feature on a project ---------------------------------------------------


@pytest.fixture
def two_runs(tmp_path):
    """Two random runs of one test, named as a lab names them: run 1 at
    full level, then run 2 at half amplitude with a 4 s stop mid-run
    and no level in its name."""
    first = tmp_path / 'Z_run1_0dB_2026-10-11T0930.nc4'
    second = tmp_path / 'Z_run2_2026-10-11T1000.nc4'
    shutil.copy(fixture_path('plate', 'random.nc4'), first)
    shutil.copy(fixture_path('plate', 'random.nc4'), second)
    project = visualdynamics.random_vibration_run(first)
    added = project.import_file(second)
    other = next(name for name in added if name.startswith('Time History'))
    history = project[other]
    rate = history.sample_rate
    history.ordinate *= 0.5
    history.ordinate[:, int(10 * rate):int(14 * rate)] = 0.0
    psds = project.compute_psds(other)
    project.compute_octave(psds)
    name = project.generate_report('random')
    project.report_features(name, enable=FEATURES)
    return project, name, 'Time History', other


def test_an_import_reads_the_name_and_notes_what_it_refused(tmp_path):
    from visualdynamics import io

    plain = io.import_file(fixture_path('plate', 'random.nc4'))
    assert plain['time_data'].run_identity is None, 'a name that says nothing'
    named = tmp_path / 'Z_X_run3_-6dB.nc4'
    shutil.copy(fixture_path('plate', 'random.nc4'), named)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = io.import_file(named)
    identity = result['time_data'].run_identity
    assert (identity.run, identity.level_db, identity.axis) == ('3', -6.0, '')
    assert identity.source('run') == cr.FROM_NAME
    assert identity.reason('axis') == 'the axis is read two ways (X, Z)'
    notes = [str(w.message) for w in caught if w.category is ImportNote]
    assert notes == [('Z_X_run3_-6dB.nc4: the axis is read two ways (X, Z), '
                     'so the axis is not read; enter it')]


def test_the_identity_travels_in_the_project_file(tmp_path, two_runs):
    import h5py

    from visualdynamics import io

    project, _name, first, other = two_runs
    project.set_run_identity(other, comment='restarted after the stop')
    project.save(str(tmp_path / 'p.vdyn'))
    back = io.load(str(tmp_path / 'p.vdyn'))
    assert back[first].run_identity == project[first].run_identity
    assert back[other].run_identity.comment == 'restarted after the stop'
    assert back[other].run_identity.source('comment') == cr.ENTERED
    # a history with no identity is stored as it always was
    bare = TimeHistory(np.arange(4.0), np.ones((1, 4)), '1X+')
    io.save(bare, str(tmp_path / 'bare.vdyn'))
    with h5py.File(tmp_path / 'bare.vdyn') as handle:
        attributes = set()
        handle.visititems(lambda _n, item: attributes.update(item.attrs))
    assert 'run_identity' not in attributes
    assert io.load(str(tmp_path / 'bare.vdyn')).run_identity is None


def test_an_entry_stands_over_what_was_read_and_is_refused_by_name(two_runs):
    project, _name, first, other = two_runs
    read = project.run_identity(first)
    assert (read.axis, read.run, read.level_db) == ('Z', '1', 0.0)
    assert read.source('level_db') == cr.FROM_NAME
    entered = project.set_run_identity(first, axis='X', level_db='')
    assert entered.axis == 'X' and entered.source('axis') == cr.ENTERED
    assert entered.level_db is None, 'cleared, back to what the record shows'
    assert f"project.set_run_identity({first!r}, axis='X', level_db='')" \
        in project.journal
    with pytest.raises(ValueError, match=re.escape(f'{other}: the level is a finite')):
        project.set_run_identity(other, level_db=float('inf'))
    with pytest.raises(TypeError, match="'Time History PSDs' is a Psd"):
        project.set_run_identity('Time History PSDs', run=4)
    cut = project.add('Cut', project[first].truncate(
        project[first].suggest_truncation()))
    project.provenance[cut] = {'verb': 'truncate', 'source': first,
                               'params': {}}
    with pytest.raises(ValueError, match="'Cut' was computed from "
                       "'Time History'"):
        project.set_run_identity(cut, run=9)
    assert project.run_identity(cut).axis == 'X', "the run it came from"
    assert rr.histories(project) == [first, other], 'a cut is not a run'


def test_the_level_comes_from_the_comparison_when_nothing_else_gives_it(
        two_runs, monkeypatch):
    project, _name, first, other = two_runs
    assert project.run_identity(other).level_db is None, 'not clearly read'
    monkeypatch.setattr(compliance, 'read_scale_db', lambda *_a, **_k: 6)
    found = project.run_identity(other)
    assert found.level_db == -6.0
    assert found.source('level_db') == ("the comparison of 'Time History (2) "
                                        "PSDs' with 'Specification (2)'")
    # the name says 0 dB, the comparison -6: two readings, refused
    clash = project.run_identity(first)
    assert clash.level_db is None
    assert clash.reason('level_db') == (
        'the level is read two ways (0 dB from the file name, -6 dB from the '
        "comparison of 'Time History PSDs' with 'Specification')")
    # an entry settles it
    project.set_run_identity(first, level_db=-9)
    assert project.run_identity(first).level_db == -9.0, \
        'an entry stands over the comparison, even where they differ'
    # a scale held on the comparison is a reading too
    monkeypatch.undo()
    project.set_comparison_scale('Time History (2) PSDs', 3)
    assert project.run_identity(other).source('level_db') == \
        "the comparison scale set on 'Time History (2) PSDs'"
    assert project.run_identity(other).level_db == -3.0


def test_two_specifications_could_give_the_level_and_are_refused(two_runs):
    project, _name, _first, other = two_runs
    group = project.object_group_of(other)
    spare = project.add('Spare Specification',
                        project['Specification (2)'].__class__(
                            project['Specification (2)'].abscissa,
                            project['Specification (2)'].ordinate,
                            **project['Specification (2)'].record_fields()))
    project.link(group[0], spare)
    assert project.run_identity(other).reason('level_db').startswith(
        "2 specifications could give the level of 'Time History (2) PSDs'")


def test_the_chronology_is_offered_and_the_section_stays_owed(two_runs):
    project, name, first, other = two_runs
    report = project[name]
    told = []
    text = project.report_chronology(name, progress=lambda *a: told.append(a))
    assert told == [(1, 2), (2, 2)]
    lines = text.splitlines()
    assert lines[1] == '|---|---|---|---|---|---:|---|---|'
    assert lines[0] == ('| Run | Axis | Level | Start | Record | At level | '
                        'Interruptions found | Outcome |')
    assert lines[2] == (f'| 1 | Z | 0 dB | 2026-10-11 09:30:00 | {first} | '
                        '28.0 s | none found |  |')
    assert lines[3] == (f'| 2 | Z |  | 2026-10-11 10:00:00 | {other} | '
                        '24.0 s | at 10.0 s for 4.0 s, to zero |  |')
    assert 'The runs are in the order of their start.' in text
    assert 'for the author to enter' in text
    # offered, not written: the section is still owed, and the page
    # being edited shows the draft, marked; the reading page does not
    assert rr.drafted(report) == text
    assert sk.words(report, 'chronology') == ''
    owed = [str(item) for item in sk.check(report, project,
                                           project.object_groups)]
    assert '7. Chronology: not yet written' in owed
    edit = _render(project, name, edit=True)
    assert rr.DRAFTED in edit and 'div class=\\"drafted\\"' in edit
    reading = _render(project, name)
    assert rr.DRAFTED not in reading
    assert f'project.report_chronology({name!r})' in project.journal
    # accepted: the section's words, and no draft on offer
    project.accept_chronology(name)
    assert sk.words(report, 'chronology') == text
    assert rr.drafted(report) == ''
    assert '7. Chronology' not in ' '.join(
        str(item) for item in sk.check(report, project, project.object_groups))
    with pytest.raises(ValueError, match='no drafted chronology is on offer'):
        project.accept_chronology(name)
    # an edited draft is accepted as edited
    project.report_chronology(name)
    project.accept_chronology(name, 'Run 1 then run 2; the stop was a trip.')
    assert sk.words(report, 'chronology') == ('Run 1 then run 2; the stop was '
                                              'a trip.')


def test_a_written_section_hides_a_draft_left_on_offer(two_runs):
    project, name, _first, _other = two_runs
    project.report_chronology(name)
    project.report_section(name, 'chronology', 'In my own words.')
    assert rr.DRAFTED not in _render(project, name, edit=True)


def test_the_chronology_is_refused_when_it_cannot_be_read(two_runs):
    project, name, first, other = two_runs
    project[first].run_identity = None
    with pytest.raises(ValueError, match="'Time History' has no start and no "
                       'run number; enter one'):
        project.report_chronology(name)
    project.report_features(name, disable=['run_sections'])
    with pytest.raises(ValueError, match='does not have Runs and chronology'):
        project.report_chronology(name)
    with pytest.raises(ValueError, match='does not have Runs and chronology'):
        project.report_runs(name)
    empty = visualdynamics.Project()
    with pytest.raises(ValueError, match='holds no time history'):
        rr.chronology(empty)
    assert other


def test_a_record_of_events_is_not_read_as_a_run_at_level():
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'shock.nc4'))
    history = rr.histories(project)[0]
    found = rr.chronology(project)
    assert found.rows[0].reading.not_read.startswith(
        'a transient record is not one run at level')
    text = rr.draft(found)
    assert f'|  |  |  |  | {history} | not read | not read |  |' in text
    assert f'- {history}: a transient record is not one run at level' in text


def test_the_results_are_grouped_by_run_in_the_order_they_were_run(two_runs):
    project, name, first, other = two_runs
    added = project.report_runs(name)
    assert added == [(f"{other}: 8 figures added; not added: nothing computed "
                     f"from {other!r} stands for 'Time History Multiple "
                     "Coherence'; compute it, then repeat")]
    assert project.report_runs(name) == [('every run is drawn on already; '
                                         'nothing added')]
    blocks = project[name].blocks
    assert [block['kind'] for block in blocks].count('verdict') == 1, \
        'the pass/fail box heads the Summary, and is not a Results section'
    assert blocks[-1]['text'].startswith('## Conclusions'), \
        'the closing text stays last'
    copies = [block for block in blocks if block.get('source') == f'{other} '
              'PSDs']
    assert copies and all(block['caption'].endswith(
        '(run 2, Z axis, 2026-10-11 10:00:00)') for block in copies)
    # each copy bound to the object computed from run 2 the same way
    bound = {block[key] for block in blocks for key in ('source', 'measured')
             if str(block.get(key, '')).startswith(other)}
    assert bound == {other, f'{other} PSDs', f'{other} PSDs 1/6 Octave'}
    octave = next(block for block in blocks if block.get('caption', '')
                  .startswith('Control against specification, octave bands (')
                  )
    assert octave['source'] == f'{other} PSDs 1/6 Octave'
    narrow = next(block for block in blocks if block.get('caption', '')
                  .startswith('Control against specification ('))
    assert narrow['source'] == f'{other} PSDs'
    # placed right after run 1's last section, its kurtosis bars
    starts = [k for k, block in enumerate(blocks)
              if str(block.get('text', '')).startswith('## Measured Data')]
    assert len(starts) == 2 and blocks[starts[1] - 1]['mode'] == 'kurtosis'
    page_html = _render(project, name)
    data = _payload(page_html)
    heads = [block['html'] for block in data['blocks']
             if 'class="run"' in block.get('html', '')]
    assert heads == [
        (f'<h3 class="run">Run 1, Z axis, 0 dB, 2026-10-11 09:30:00 ({first})'
        '</h3>'),
        f'<h3 class="run">Run 2, Z axis, 2026-10-11 10:00:00 ({other})</h3>']
    texts = [block.get('html', '') for block in data['blocks']]
    second = texts.index(heads[1])
    assert any('Data Quality' in text for text in texts[:second]), \
        "run 1's sections under run 1"
    first_head = texts.index(heads[0])
    labeled = [str(block.get('caption', '')).endswith('10:00:00)')
               for block in data['blocks']]
    assert not any(labeled[first_head:second]), 'no run 2 figure under run 1'
    assert any(labeled[second:]), "run 2's figures under run 2"
    # nothing lost in the grouping: the skeleton's page, and a heading
    # for each run
    project.report_features(name, disable=['run_sections'])
    plain = _payload(_render(project, name))['blocks']
    project.report_features(name, enable=['run_sections'])
    assert len(data['blocks']) == len(plain) + 2
    assert page._RUNS_CSS in page_html
    # run 2 first in time: the headings follow
    project.set_run_identity(other, start='2026-10-11T08:00')
    data = _payload(_render(project, name))
    heads = [block['html'] for block in data['blocks']
             if 'class="run"' in block.get('html', '')]
    assert heads[0].startswith('<h3 class="run">Run 2')
    # the feature off: no headings, and the skeleton's page as it was
    project.report_features(name, disable=['run_sections'])
    off = _render(project, name)
    assert 'class=\\"run\\"' not in off and page._RUNS_CSS not in off


def test_the_checks_still_close_the_results_grouped_by_run(two_runs):
    project, name, _first, _other = two_runs
    project.report_runs(name)
    project.report_features(name, enable=['check_blocks'])
    texts = [block.get('html', '')
             for block in _payload(_render(project, name))['blocks']]
    runs = [k for k, text in enumerate(texts) if 'class="run"' in text]
    checks = texts.index('<h3>Checks</h3>')
    closing = next(k for k, text in enumerate(texts)
                   if k > checks and '9. Deviations' in text)
    assert len(runs) == 2 and checks > runs[-1], \
        'the checks after the last run, not first in the Results'
    assert not any('class="run"' in text or '<h3>' in text
                   for text in texts[checks + 1:closing]), \
        'nothing of a run under the Checks heading'


def test_a_check_of_its_own_is_not_repeated_for_every_run(two_runs):
    project, name, first, _other = two_runs
    project.report_features(name, enable=['check_blocks'])
    report = project[name]
    # the last section a Results one every figure of which repeats, so
    # the check appended after it falls inside it
    while report.blocks[-1]['kind'] == 'text':
        report.blocks.pop()
    report.blocks[:] = [block for block in report.blocks
                        if 'Coherence' not in str(block.get('source', ''))]
    project.report_check_block(name, 'time_statistics', source=first)
    assert project.report_runs(name)[0].endswith('figures added')
    assert [block['source'] for block in report.blocks
            if 'check' in block] == [first]


def test_a_report_on_no_run_has_nothing_to_repeat(two_runs):
    project, _name, _first, _other = two_runs
    lone = project.add('Notes', Report('Notes', [{'kind': 'text',
                                                  'text': '## Notes'}],
                                       features=FEATURES))
    with pytest.raises(ValueError, match="'Notes' draws on no run"):
        project.report_runs(lone)


def test_one_run_nothing_identifies_gets_no_heading():
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'))
    name = project.generate_report('random')
    project.report_features(name, enable=['skeleton_sections'])
    skeleton_only = _render(project, name)
    project.report_features(name, enable=['run_sections'])
    with_runs = _render(project, name)
    assert 'class=\\"run\\"' not in with_runs
    # the page differs only by the feature's style
    assert with_runs.replace(page._RUNS_CSS, '') == skeleton_only
    project.set_run_identity('Time History', run=1)
    assert 'Run 1 (Time History)' in _render(project, name)


def test_the_seam_for_a_later_source_of_outcomes(two_runs, monkeypatch):
    """A later source feeds `read_log`; the chronology carries what it
    says where the record and the source count the same interruptions,
    and nothing else changes."""
    project, _name, first, other = two_runs
    monkeypatch.setattr(rr, 'read_log', lambda _objects, _groups=None:
                        rr.LogSource({other: rr.Outcome(
                            'completed after a restart',
                            ('amplifier trip',))}))
    text = rr.draft(rr.chronology(project))
    assert ('at 10.0 s for 4.0 s, to zero: amplifier trip | completed after a '
            'restart |') in text
    assert f'| {first} | 28.0 s | none found |  |' in text


# ---- the editor ---------------------------------------------------------------


@pytest.fixture
def editing(window, pump, tmp_path):
    named = tmp_path / 'Z_run4_-6dB_2026-10-11T1300.nc4'
    shutil.copy(fixture_path('plate', 'random.nc4'), named)
    window.import_paths([str(named)])
    pump()
    name = window.generate_report('random')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    editor = window.report_editor
    editor._operate({'op': 'features', 'enable': FEATURES})
    return window, editor, window.objects[name], name


def test_the_runs_panel_lists_and_enters_identities(editing):
    window, editor, _report, _name = editing
    panel = editor.runs_panel
    assert panel is not None, 'with the feature on, under the features'
    history = next(n for n, o in window.objects.items()
                   if isinstance(o, TimeHistory))
    assert panel.names == [history]
    cells = [panel.table.item(0, c).text() for c in range(6)]
    assert cells == [history, 'Z', '4', '-6', '2026-10-11T13:00:00', '']
    assert panel.table.item(0, 3).toolTip() == 'from the file name'
    panel.table.item(0, 1).setText('X')
    assert window.project[history].run_identity.axis == 'X'
    assert editor.runs_panel.table.item(0, 1).toolTip() == 'entered'
    assert f"project.set_run_identity({history!r}, axis='X')" in \
        window.project.journal
    editor.runs_panel.table.item(0, 3).setText('loud')
    assert 'the level is a number of decibels' in \
        editor.runs_panel.message.text()
    assert window.project[history].run_identity.level_db == -6.0, \
        'refused, nothing changed'
    editor.runs_panel.repeat_button.click()
    assert editor.runs_panel.message.text() == ('every run is drawn on '
                                                'already; nothing added')
    editor._operate({'op': 'features', 'disable': ['run_sections']})
    assert editor.runs_panel is None


def test_a_cell_typed_in_does_not_delete_its_table_under_it(editing):
    """The cell's change rebuilds the pane from inside the table's own
    signal; the pane it replaces, the table with it, is deleted once
    that signal has returned, not while the table is still in it."""
    import shiboken6

    window, editor, _report, _name = editing
    table = editor.runs_panel.table
    table.item(0, 1).setText('X')
    assert editor.runs_panel.table is not table, 'the pane was rebuilt'
    assert shiboken6.isValid(table), 'still there while its signal returns'
    assert not table.isVisible()
    window._run_long('Reading…', lambda progress=None: None)
    assert not shiboken6.isValid(table), 'and deleted after'


def test_the_chronology_pane_drafts_and_accepts(editing):
    _window, editor, report, _name = editing
    editor._operate({'op': 'select', 'at': 'section:chronology'})
    panel = editor.chronology_panel
    assert panel is not None and panel.draft_view is None
    panel.draft_button.click()
    panel = editor.chronology_panel
    assert panel.draft_view is not None, 'the draft on offer'
    assert panel.draft_view.toPlainText().startswith('| Run | Axis |')
    assert panel.message.text().startswith('Drafted from the records.')
    assert sk.words(report, 'chronology') == '', 'still owed'
    with open(editor._page_path, encoding='utf-8') as page_file:
        assert rr.DRAFTED in page_file.read()
    panel.accept_button.click()
    assert sk.words(report, 'chronology').startswith('| Run | Axis |')
    assert editor.text_editor.toPlainText().startswith('| Run | Axis |')
    assert editor.chronology_panel.draft_view is None


def test_the_edges_of_the_reading():
    """Where a mutant of the reading would move a result by a sample or a
    boundary: each case pinned to its hand value."""
    # two stops after a lead-in, the second exactly the least duration
    values = _alternating(40.0)
    values[:, :int(2 * RATE)] = 0.0
    values[:, int(10 * RATE):int(14 * RATE)] = 0.0
    values[:, int(20 * RATE):int(21 * RATE)] = 0.0
    found = cr.read_levels(values, RATE)
    assert [(round(stop.start_s, 4), round(stop.seconds, 4))
            for stop in found.interruptions] == [(10.0, 4.0), (20.0, 1.0)]
    assert found.at_level_s == pytest.approx(33.0, abs=0.5 / RATE)
    # a record with an offset: the mean is taken off, so a stop at the
    # offset is a stop
    offset = 5.0 + _alternating(30.0)
    offset[:, int(10 * RATE):int(14 * RATE)] = 5.0
    assert len(cr.read_levels(offset, RATE).interruptions) == 1
    # the settings at their edges
    with pytest.raises(ValueError, match='further below the level'):
        cr.read_levels(values, RATE, drop_db=10 * math.log10(2))
    assert cr.read_levels(values, RATE, min_s=0).not_read == ''
    assert cr.read_levels(values, 0.5).not_read.startswith(
        'the 0.5 s running-RMS window holds fewer than three samples')
    # the shortest record a window reads: exactly one window long
    assert cr.read_levels(np.array([[1.0, -1.0, 1.0]]), 4.0).not_read == ''
    assert cr.read_levels(np.array([[1.0, -1.0]]), 4.0).not_read.startswith(
        'the record (0.5 s) is shorter')
    assert cr.running_mean_square(np.array([[1.0, -1.0, 1.0, -1.0]]),
                                  2).tolist() == [1.0, 1.0, 1.0]


def test_the_edges_of_an_identity():
    assert cr.RunIdentity(axis='Z').known
    assert cr.RunIdentity(level_db=0.0).known, 'full level is a level'
    assert cr.RunIdentity(axis='Z', run='1').with_values(
        {'axis': ''}, cr.ENTERED).source('axis') == ''
    assert cr.parse_start('2026-10-11T11:11:05+02:00').isoformat() == \
        '2026-10-11T09:11:05'
    assert cr.read_name('X_Y_Z')[1]['axis'].startswith(
        'the axis is read three ways')
    assert cr.read_name('r1_r2_r3_r4')[1]['run'].startswith(
        'the run number is read four ways')
    assert cr.read_name('19800101')[0] == {'start': '1980-01-01'}
    assert cr.read_name('19791231')[0] == {}
    assert cr.read_name('22001231')[0] == {'start': '2200-12-31'}
    assert cr.read_name('22010101')[0] == {}
    assert math.copysign(1.0, cr.entered('TH', level_db=-0.0)['level_db']) \
        == 1.0, 'no negative zero'
    assert cr.entered('TH', comment='x', run='a') == {'comment': 'x',
                                                      'run': 'a'}
    assert cr.order([('a', cr.RunIdentity(run='A')),
                     ('b', cr.RunIdentity(run='10')),
                     ('c', cr.RunIdentity(run='2'))]) == ['c', 'b', 'a']
    with pytest.raises(ValueError, match="cannot be read: 'e' has no start"):
        cr.order([('d', cr.RunIdentity(run='9')), ('e', cr.RunIdentity())])


def test_the_feature_needs_the_skeleton():
    from visualdynamics.report import features as rf

    assert rf.FEATURES['run_sections'].available
    assert rf.unmet('run_sections') == ['needs Skeleton sections']
    report = Report()
    rf.configure(report, enable=FEATURES)
    selection = rf.configure(report, disable=['skeleton_sections'])
    assert report.features == []
    assert selection.notes == (('Runs and chronology turned off with Skeleton '
                               'sections, which it needs'),)


def test_the_channels_the_level_is_read_on(two_runs):
    """The control channels (the channel table's Control flag), never a
    drive; without controls the response channels; one quantity."""
    project, _name, first, _other = two_runs
    groups = project.object_groups
    accelerations = ('101Z+', '104Z+', '110Z+', '113Z+', '1301Z+', '1304Z+',
                     '1310Z+', '1313Z+')
    assert rr.read_record(project, first, groups).channels == accelerations
    table = project['Channel Table']
    table.frame.loc[0, 'control'] = ''
    assert rr.read_record(project, first, groups).channels == \
        accelerations[1:], 'the control channels only'
    assert rr.read_record(dict(project.items()), first, groups).channels == \
        accelerations[1:], 'the groups given, not only the project\'s own'
    # a control channel used as a reference is a drive, never read on
    table.frame['control'] = ''
    table.frame.loc[0, 'control'] = 'True'
    project[first].roles = {('101Z+', 'acceleration'): 'reference'}
    assert rr.read_record(project, first, groups).channels == \
        accelerations[1:], 'no control left but a drive: the responses'
    table.frame['control'] = ''
    project[first].roles = {('104Z+', 'acceleration'): 'monitor'}
    assert rr.read_record(project, first, groups).channels == tuple(
        dof for dof in accelerations if dof != '104Z+'), \
        'the response channels, a monitor left out'
    project[first].roles = {(dof, 'acceleration'): 'monitor'
                            for dof in accelerations}
    assert rr.read_record(project, first, groups).channels == \
        accelerations, 'every channel, the most common quantity'


def test_a_record_read_on_its_own_clock_and_its_refusals():
    rate = 64.0
    values = np.tile(np.where(np.arange(int(30 * rate)) % 2, -1.0, 1.0),
                     (1, 1))
    values[:, int(10 * rate):int(14 * rate)] = 0.0
    history = TimeHistory(100.0 + np.arange(values.shape[1]) / rate, values,
                          '')
    found = rr.read_record({'h': history}, 'h')
    assert found.channels == ('record 1',), 'a channel with no DOF, named'
    assert found.interruptions[0].start_s == pytest.approx(110.0)
    stacked = TimeHistory(5.0 + np.arange(64.0) / rate, np.ones((2, 64)),
                          '1X+', block=['avg 1', 'avg 2'])
    reading = rr.read_record({'s': stacked}, 's')
    assert reading.not_read.startswith('a stack of 2 captures')
    assert (reading.start_s, reading.seconds) == (5.0, 1.0)
    single = TimeHistory(np.arange(64.0) / rate, np.ones((1, 64)), '1X+',
                         block=['avg 1'])
    assert rr.read_record({'s': single}, 's').not_read.startswith(
        'no channel varies'), 'one capture is one record'
    pair = TimeHistory([2.0, 2.5], [[1.0, -1.0]], '1X+')
    reading = rr.read_record({'p': pair}, 'p')
    assert reading.not_read.startswith('the 0.5 s running-RMS window holds')
    assert (reading.start_s, reading.seconds) == (2.0, 1.0)
    lone = TimeHistory([3.0], [[1.0]], '1X+')
    reading = rr.read_record({'l': lone}, 'l')
    assert reading.not_read == 'the record has fewer than two samples'
    assert (reading.start_s, reading.seconds) == (3.0, 0.0)


def test_a_refused_level_stays_refused_and_two_readings_are_refused(
        two_runs, monkeypatch):
    project, _name, first, other = two_runs
    project[first].run_identity = cr.RunIdentity(
        refused=(('level_db', "'6dB' has no sign"),))
    monkeypatch.setattr(compliance, 'read_scale_db', lambda *_a, **_k: 6)
    assert project.run_identity(first).level_db is None, \
        'what the name refused, the comparison does not settle'
    # a second PSD of run 2 that reads another level
    more = project.compute_psds(other)
    readings = {f'{other} PSDs': 6, more: 3}
    monkeypatch.setattr(compliance, 'read_scale_db',
                        lambda _s, measured, *_a, **_k: next(
                            value for name, value in readings.items()
                            if project[name] is measured))
    assert project.run_identity(other).reason('level_db').startswith(
        'the level is read two ways (-6 dB from')
    readings[project.compute_psds(other)] = 0
    assert project.run_identity(other).reason('level_db').startswith(
        'the level is read three ways (-6 dB from')


def test_one_figure_is_one_figure(two_runs):
    project, _name, _first, other = two_runs
    lone = project.add('Short', Report('Short', [
        {'kind': 'text', 'text': '## Time'},
        {'kind': 'plot', 'source': '@basis:TimeHistory', 'mode': 'curves',
         'caption': 'Time'}], features=FEATURES))
    assert project.report_runs(lone) == [f'{other}: 1 figure added']


# ---- a run log as a second source -----------------------------------------


def _random_log(path, *, abort=True):
    """A random test on Z: run 1 to full level and a stop at 10:00; run 2
    at full level with an abort at 2:00 (its cause, path and a resume
    logged) and a stop at 8:00, or with no abort. Returns the abort's
    id (None without one)."""
    with writer(path) as log:
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header=HEADER)
        model.append(log, 'run_start', author='AB', test=1, axis='Z', run=1,
                     fields={'counter': '0:00'})
        for level in (-12.0, 0.0):
            model.append(log, 'level_row', author='AB', test=1, axis='Z',
                         run=1, fields={'level_db': level})
        model.append(log, 'run_stop', author='AB', test=1, axis='Z', run=1,
                     fields={'counter': '10:00'})
        model.append(log, 'run_start', author='AB', test=1, axis='Z', run=2,
                     fields={'counter': '0:00'})
        model.append(log, 'level_row', author='AB', test=1, axis='Z', run=2,
                     fields={'level_db': 0.0})
        stop = None
        if abort:
            stop = model.append(log, 'abort', author='AB', test=1, axis='Z',
                                run=2, fields={'stop': 'controller ramp',
                                               'counter': '2:00'})
            model.append(log, 'cause_found', author='AB', test=1,
                         fields={'cause': 'a loose cable',
                                 'refers_to': stop.id})
            model.append(log, 'path_decided', author='AB', test=1,
                         fields={'path': 'resume', 'refers_to': stop.id})
            model.append(log, 'restart', author='AB', test=1, axis='Z', run=2,
                         fields={'resume': True, 'counter_at_restart': '2:00',
                                 'refers_to': stop.id})
        model.append(log, 'run_stop', author='AB', test=1, axis='Z', run=2,
                     fields={'counter': '8:00'})
        return stop.id if stop is not None else None


def test_without_a_run_log_the_chronology_and_page_are_as_before(two_runs,
                                                                 tmp_path):
    project, name, _first, _other = two_runs
    assert rr.read_log(project) == rr.LogSource()
    text = project.report_chronology(name)
    edit = _render(project, name, edit=True)
    assert 'run log' not in text.lower()
    assert text.endswith('for the author to enter.')
    # attached and taken out again: the draft and the page byte for byte
    path = tmp_path / 'article.vdlog'
    _random_log(path)
    attached = project.attach_run_log(path)
    project.detach_run_log(attached)
    assert project.report_chronology(name) == text
    assert _render(project, name, edit=True) == edit
    assert 'draft_logs' not in str(project[name].blocks)


def test_the_run_log_gives_outcomes_causes_and_the_credited_time(two_runs,
                                                                 tmp_path):
    project, name, first, other = two_runs
    report = project[name]
    path = tmp_path / 'article.vdlog'
    abort = _random_log(path)
    attached = project.attach_run_log(path)
    extract = project[attached]
    text = project.report_chronology(name)
    lines = text.splitlines()
    assert lines[2].endswith(f'| {first} | 28.0 s | none found | completed '
                             'at 10:00 |')
    assert lines[3].endswith(
        f'| {other} | 24.0 s | at 10.0 s for 4.0 s, to zero: controller ramp '
        f'(log entry #{abort}), cause: a loose cable | completed at 8:00 |')
    assert 'The outcome of each run is as the run log records it' in text
    assert f'From {extract.cited()}:' in text
    assert "the run log 'Article 7', test 1, to entry #" in text
    assert (f'| #{abort} | 2026-10-30 08:09:00 | controller ramp | a loose cable '
            '| resume | resumed at 2:00 |') in text
    assert [line for line in lines if line.startswith('- Axis Z: ')] == [
        f'- Axis Z: {extract.credited()[0].message}']
    # offered, never placed: the section stays owed until accepted
    assert sk.words(report, 'chronology') == ''
    assert '7. Chronology: not yet written' in [
        str(item) for item in sk.check(report, project, project.object_groups)]
    edit = _render(project, name, edit=True)
    assert rr.DRAFTED_WITH_LOG in edit and rr.DRAFTED not in edit
    assert rr.DRAFTED_WITH_LOG not in _render(project, name)
    project.accept_chronology(name)
    assert sk.words(report, 'chronology') == text
    assert rr.DRAFTED_WITH_LOG not in _render(project, name, edit=True)
    assert 'draft_logs' not in str(report.blocks)


def test_the_badge_shows_when_the_log_grew_after_the_draft(two_runs,
                                                           tmp_path):
    project, name, _first, _other = two_runs
    path = tmp_path / 'article.vdlog'
    _random_log(path)
    attached = project.attach_run_log(path)
    project.report_chronology(name)
    assert f'{rr.DRAFTED_WITH_LOG}<' in _render(project, name, edit=True)
    with writer(path) as log:
        log.append('note', author='AB', test=1, text='later')
    edit = _render(project, name, edit=True)
    assert (f'{rr.DRAFTED_WITH_LOG} [{attached}: newer entries in the log]'
            in edit)
    # drafted again, the draft names the badge where it cites its source
    assert '[newer entries in the log]:' in project.report_chronology(name)
    project.detach_run_log(attached)
    assert f'[{attached}: {rr.GONE}]' in _render(project, name, edit=True)


def test_a_record_the_log_cannot_answer_for_is_named(two_runs, tmp_path):
    project, name, first, other = two_runs
    path = tmp_path / 'article.vdlog'
    _random_log(path)
    project.attach_run_log(path)
    project.set_run_identity(other, run=3)
    text = project.report_chronology(name)
    assert (f'- {other}: the run log holds no run 3 on Z, so its outcome is '
            'left to the author.') in text
    project.set_run_identity(other, run='2b')
    assert (f"- {other}: run '2b' is not a run number the log keeps, so the "
            'run log cannot say its outcome; enter it.') in rr.draft(
                rr.chronology(project))
    # run 1 in two tests, on two axes, the record's axis not known
    with writer(path) as log:
        model.roll(log, author='AB')
        model.append(log, 'level_row', author='AB', test=2, axis='X', run=1,
                     fields={'level_db': 0.0})
    second = project.attach_run_log(path, test=2)
    found = rr.read_log(project)
    assert first in found.outcomes, 'its axis tells the two apart'
    project.set_run_identity(first, axis='')
    found = rr.read_log(project)
    assert first not in found.outcomes
    assert (f'{first}: the run log holds run 1 2 times (Run Log, Test 1, axis '
            f'Z; {second}, axis X), so its outcome is left to the author; '
            'enter the axis.') in found.notes
    assert found.logs == ('Run Log, Test 1', second)


def test_the_logged_causes_stay_off_a_record_that_counts_otherwise(
        two_runs, tmp_path):
    project, _name, first, _other = two_runs
    path = tmp_path / 'article.vdlog'
    with writer(path) as log:
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header=HEADER)
        model.append(log, 'level_row', author='AB', test=1, axis='Z', run=1,
                     fields={'level_db': 0.0})
        model.append(log, 'abort', author='AB', test=1, axis='Z', run=1,
                     fields={'stop': 'amplifier trip', 'counter': '1:00'})
    project.attach_run_log(path)
    found = rr.chronology(project)
    row = next(row for row in found.rows if row.name == first)
    assert row.outcome == rr.Outcome('aborted at 1:00')
    assert (f'{first}: the record shows 0 interruptions and the run log 1 for '
            "its run, so the log's are listed under the runs rather than "
            "beside the record's.") in found.notes
    text = rr.draft(found)
    assert '| amplifier trip | not logged | not logged | none logged |' in text


def test_a_log_with_no_abort_offers_no_none(two_runs, tmp_path):
    project, name, _first, _other = two_runs
    path = tmp_path / 'article.vdlog'
    _random_log(path, abort=False)
    project.attach_run_log(path)
    text = project.report_chronology(name)
    assert ('The log holds no abort, trip or scheduled stop of test 1 by entry '
            "#8; whether the test ran without one is the author's to say."
            in text)
    assert 'No interruptions' not in text and 'no aborts' not in text.lower()


def test_a_log_and_no_record_offers_the_log_chronology(tmp_path):
    from visualdynamics.core.run_log_extract import DRAFTED as FROM_LOG

    project = visualdynamics.Project()
    name = project.add('Report', Report('Report', [], features=FEATURES))
    empty = tmp_path / 'empty.vdlog'
    with writer(empty) as log:
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header=HEADER)
    held = project.attach_run_log(empty)
    with pytest.raises(ValueError, match='no run log extract holds a run: The '
                       'log holds no runs of test 1 by entry #1'):
        project.report_chronology(name)
    assert rr.drafted(project[name]) == ''
    project.detach_run_log(held)
    path = tmp_path / 'article.vdlog'
    _random_log(path)
    attached = project.attach_run_log(path)
    text = project.report_chronology(name)
    assert text == project[attached].draft_chronology().text
    assert FROM_LOG in _render(project, name, edit=True)
    assert sk.words(project[name], 'chronology') == ''


def test_two_logged_stops_against_one_found_stay_in_the_log_table(two_runs,
                                                                   tmp_path):
    project, _name, _first, other = two_runs
    path = tmp_path / 'article.vdlog'
    with writer(path) as log:
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header=HEADER)
        model.append(log, 'level_row', author='AB', test=1, axis='Z', run=2,
                     fields={'level_db': 0.0})
        for counter in ('1:00', '3:00'):
            model.append(log, 'abort', author='AB', test=1, axis='Z', run=2,
                         fields={'stop': 'controller ramp',
                                 'counter': counter})
            model.append(log, 'restart', author='AB', test=1, axis='Z',
                         run=2, fields={'resume': True})
    project.attach_run_log(path)
    found = rr.chronology(project)
    row = next(row for row in found.rows if row.name == other)
    assert row.outcome.explanations == ()
    assert (f'{other}: the record shows 1 interruption and the run log 2 for '
            "its run, so the log's are listed under the runs rather than "
            "beside the record's.") in found.notes


def test_a_record_not_read_keeps_the_logged_stops_in_the_log_table(
        tmp_path):
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'shock.nc4'))
    history = rr.histories(project)[0]
    project.set_run_identity(history, run=1)
    path = tmp_path / 'article.vdlog'
    with writer(path) as log:
        model.open_test(log, author='AB', test_kind='Shock',
                        header={'title': 'Release shock'})
        model.append(log, 'shot', author='AB', test=1, axis='Z', run=1,
                     fields={'polarity': 'positive', 'n': 1, 'of': 3})
        model.append(log, 'abort', author='AB', test=1, axis='Z', run=1,
                     fields={'stop': 'stop word'})
    project.attach_run_log(path)
    found = rr.chronology(project)
    assert found.rows[0].outcome == rr.Outcome('aborted')
    assert (f"{history}: the record's interruptions are not read, so the run "
            "log's are listed under the runs.") in found.notes
    text = rr.draft(found)
    assert '| not read | not read | aborted |' in text
    assert '| stop word | not logged | not logged | none logged |' in text
