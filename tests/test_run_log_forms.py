"""The Run Log's forms: the test header in full, carried fields marked
until confirmed, the readings of each kind of test (a sweep against the
continuous rating, a shot's four checks and its pulses against the
budget, a survey's result), the notch against the plan and the
findings.

Every log is written under `tmp_path`, with a fixed login and machine
name, on a clock the test sets. Each form is driven the way the keeper
drives it: boxes typed, Enter (or `write`) pressed, and the entry read
back from the file.
"""

from __future__ import annotations

import gc
import hashlib
import threading
import weakref
from datetime import datetime, timedelta, timezone

import pytest
from conftest import destroy_window, release_modifiers

from visualdynamics.core.run_log import HEADER_FIELDS, RunLog

WHO = ('tester', 'bench')
ZONE = timezone(timedelta(hours=-6))
START = datetime(2026, 10, 30, 8, 0, tzinfo=ZONE)

RANDOM = {'test_kind': 'Random Vibration', 'title': 'Random, Z',
          'steps_db': '-12, -6, -3, 0', 'sigma': '3',
          'rated_current_peak': '800', 'rail_voltage': '250'}


class Clock:
    """A clock that reads what the test sets, moved by hand."""

    def __init__(self, now=START):
        self.now = now

    def __call__(self):
        return self.now

    def move(self, **step):
        self.now += timedelta(**step)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def panel(window, clock):
    dock = window.show_run_log()
    panel = dock.panel
    panel.clock = clock
    panel.log_clock = clock
    panel.who = WHO
    return panel


def _new(panel, tmp_path, name='article.vdlog'):
    path = str(tmp_path / name)
    panel.new_log(path, 'Article 7')
    assert panel.settle()
    panel.initials_box.setText('AB')
    return path


def _form(panel, name, **boxes):
    form = panel.open_form(name)
    assert form is not None, panel.message.text()
    for field, value in boxes.items():
        form.set_box(field, value)
    return form


def _write(panel, form):
    form.write()
    assert panel.settle()
    assert panel.form is None, (panel.message.text(), form.message.text())


def _refused(panel, form):
    """Write the form, which the log refuses: the form stays up and says
    why. Returns what it says."""
    form.write()
    assert panel.settle()
    assert panel.form is form
    return form.message.text()


def _opened(panel, tmp_path, **header):
    path = _new(panel, tmp_path)
    _write(panel, _form(panel, 'open test', **header))
    panel.axis_box.setText('Z')
    return path


def _entries(path, kind=None):
    entries = RunLog.read(path).view()
    return [e for e in entries if kind is None or e.kind == kind]


def _key(widget, letter):
    """Ctrl+Shift+`letter` pressed in `widget`, then let go."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    widget.setFocus()
    QApplication.processEvents()
    QTest.keyClick(widget, getattr(Qt.Key, f'Key_{letter}'),
                   Qt.KeyboardModifier.ControlModifier
                   | Qt.KeyboardModifier.ShiftModifier)
    release_modifiers(widget)
    QApplication.processEvents()


# ---- typed planned notches ------------------------------------------------


def test_planned_notches_are_read_as_typed_and_written_back():
    from visualdynamics.core.run_log import RunLogError
    from visualdynamics.gui.run_log_dock import notches_text, planned_notches

    typed = 'Z: 180-220 Hz, 6 dB; 400-450 Hz; X: 1.2-2 kHz, -3'
    notches = planned_notches(typed)
    assert notches == [{'axis': 'Z', 'band': '180-220 Hz', 'depth_db': 6.0},
                       {'band': '400-450 Hz'},
                       {'axis': 'X', 'band': '1.2-2 kHz', 'depth_db': -3.0}]
    assert planned_notches(notches_text(notches)) == notches
    assert notches_text([{'band': [180.0, 220.0], 'depth_db': 6.25}]) == \
        '180-220 Hz, 6.25 dB'
    assert planned_notches('  ') is None
    with pytest.raises(RunLogError, match="'Z: wide': its band 'wide' is "
                                          'not read as a band'):
        planned_notches('Z: wide')
    with pytest.raises(RunLogError, match="its depth: 'deep' is not a "
                                          'number'):
        planned_notches('180-220 Hz, deep')
    with pytest.raises(RunLogError, match='the depth after the comma is '
                                          'blank'):
        planned_notches('180-220 Hz,')


# ---- the header -----------------------------------------------------------


def test_the_header_form_holds_every_header_field_once_in_its_groups(
        panel, tmp_path):
    from visualdynamics.gui.run_log_forms import HEADER_GROUPS

    grouped = [field for _group, rows in HEADER_GROUPS
               for field, _label, _kind in rows]
    assert sorted(grouped) == sorted(HEADER_FIELDS)
    assert len(grouped) == len(set(grouped))
    _new(panel, tmp_path)
    form = panel.open_form('open test')
    assert set(form.boxes) == {*HEADER_FIELDS, 'files'}
    assert form.headings == [group for group, _rows in HEADER_GROUPS] + [
        'Attached']
    assert form.labels['test_kind'].text() == 'Kind *'
    form.cancelled.emit()
    _write(panel, _form(panel, 'open test', **RANDOM))
    form = panel.open_form('header')
    # the kind is the test's, set when it opened; the rest is the header
    assert set(form.boxes) == ({*HEADER_FIELDS} - {'test_kind'}) | {'confirm'}
    assert form.box_text('sigma') == '3'
    assert form.box_text('steps_db') == '-12, -6, -3, 0'


def test_a_test_opened_with_its_whole_header_and_files_hashed_on_the_worker(
        panel, tmp_path, monkeypatch):
    from visualdynamics.core import data_index

    plan = tmp_path / 'plan.txt'
    plan.write_bytes(b'the plan, revision B')
    profile = tmp_path / 'profile.csv'
    profile.write_bytes(b'20,0.01\n2000,0.01\n')
    threads = []
    digest = data_index.digest

    def watched(path, *args, **kwargs):
        threads.append(threading.current_thread().name)
        return digest(path, *args, **kwargs)

    monkeypatch.setattr(data_index, 'digest', watched)
    path = _new(panel, tmp_path)
    form = _form(
        panel, 'open test', test_kind='Shock', title='Release shock, Z',
        test_class='qualification', purpose='release event',
        article='Unit 7', serial='SN 0042', configuration='flight',
        document='Test document', document_revision='B', plan_revision='C',
        procedure='Procedure 12', procedure_revision='A',
        profile_source='the plan, table 3', steps_db='-12, -6, 0',
        hold_points_db='-6', notches_planned='Z: 180-220 Hz, 6 dB; 400-450 Hz',
        limits='bracket limit profile', tolerance='+/-3 dB',
        abort_bands='+/-6 dB', duration_s='600', duration_minus_pct='0',
        duration_plus_pct='10', sigma='3', rated_current_peak='800',
        rated_current_rms='400', rail_voltage='250', drive_window='0.3, 0.7',
        projected_current='700', projected_voltage='200', pulse_budget='18',
        files=f'{plan}; {profile}')
    _write(panel, form)
    (opened,) = _entries(path, 'test_opened')
    assert dict(opened.fields) == {
        'test_kind': 'Shock', 'title': 'Release shock, Z',
        'test_class': 'qualification', 'purpose': 'release event',
        'article': 'Unit 7', 'serial': 'SN 0042', 'configuration': 'flight',
        'document': 'Test document', 'document_revision': 'B',
        'plan_revision': 'C', 'procedure': 'Procedure 12',
        'procedure_revision': 'A', 'profile_source': 'the plan, table 3',
        'steps_db': [-12.0, -6.0, 0.0], 'hold_points_db': [-6.0],
        'notches_planned': [{'axis': 'Z', 'band': '180-220 Hz',
                             'depth_db': 6.0}, {'band': '400-450 Hz'}],
        'limits': 'bracket limit profile', 'tolerance': '+/-3 dB',
        'abort_bands': '+/-6 dB', 'duration_s': 600.0,
        'duration_minus_pct': 0.0, 'duration_plus_pct': 10.0, 'sigma': 3.0,
        'rated_current_peak': 800.0, 'rated_current_rms': 400.0,
        'rail_voltage': 250.0, 'drive_window': [0.3, 0.7],
        'projected_current': 700.0, 'projected_voltage': 200.0,
        'pulse_budget': 18}
    assert [(ref.name, ref.sha256, ref.size) for ref in opened.files] == [
        ('plan.txt', hashlib.sha256(b'the plan, revision B').hexdigest(), 20),
        ('profile.csv', hashlib.sha256(b'20,0.01\n2000,0.01\n').hexdigest(),
         18)]
    assert len(threads) == 2 and all(name.startswith('run-log')
                                     for name in threads), threads


def test_a_file_that_cannot_be_read_is_refused_naming_it(panel, tmp_path):
    path = _new(panel, tmp_path)
    gone = tmp_path / 'missing.pdf'
    form = _form(panel, 'open test', test_kind='Shock', files=str(gone))
    said = _refused(panel, form)
    assert said.startswith(f'{gone} could not be read to attach it')
    assert not _entries(path, 'test_opened')


def test_the_roll_carries_what_is_left_marked_until_it_is_confirmed(
        panel, tmp_path):
    path = _opened(panel, tmp_path, **RANDOM)
    form = panel.open_form('roll')
    assert form.box_text('sigma') == '3'
    assert form.box_text('test_kind') == 'Random Vibration'
    assert form.labels['sigma'].text() == 'Sigma (carried)'
    form.set_box('title', 'Random, X')
    _write(panel, form)
    opened = _entries(path, 'test_opened')[-1]
    assert opened.test == 2
    assert opened.fields['title'] == 'Random, X'
    assert sorted(opened.fields['carried']) == sorted(
        set(RANDOM) - {'title'})
    assert set(panel.snapshot.carried) == set(RANDOM) - {'title'}

    form = panel.open_form('header')
    assert form.labels['sigma'].text() == 'Sigma (carried)'
    assert form.labels['title'].text() == 'Title'
    said = _refused(panel, form)
    assert said == ('Nothing was changed. Tick Confirm to confirm the carried '
                    'fields as they are.')
    form.set_box('rail_voltage', '260')
    form.set_box('confirm', 'yes')
    _write(panel, form)
    changes = [e for e in _entries(path, 'header_change') if e.test == 2]
    rail = [e for e in changes if e.fields['field'] == 'rail_voltage']
    assert [dict(e.fields) for e in rail] == [
        {'field': 'rail_voltage', 'value': 260.0, 'was': 250.0}]
    confirmed = {e.fields['field'] for e in changes
                 if e.text == 'confirmed as carried'}
    assert confirmed == set(RANDOM) - {'title', 'rail_voltage'}
    assert panel.snapshot.carried == ()
    form = panel.open_form('header')
    assert form.labels['sigma'].text() == 'Sigma'


def test_a_header_field_emptied_or_refused_writes_nothing(panel, tmp_path):
    path = _opened(panel, tmp_path, **RANDOM)
    form = _form(panel, 'header', sigma='')
    assert _refused(panel, form).startswith('Sigma: a header field is '
                                            'changed to its new value, not '
                                            'emptied')
    form.set_box('sigma', '3')
    form.set_box('title', 'Random, Z, again')
    form.set_box('pulse_budget', '0')
    # the core's refusal, shown on the form; the title is not written
    # without the budget
    assert _refused(panel, form) == ('Header changed: value is a whole '
                                     'number from 1, not 0')
    assert not _entries(path, 'header_change')


# ---- the readings ---------------------------------------------------------


def test_a_sweep_writes_its_fields_and_reads_the_continuous_rating(
        panel, tmp_path):
    """At -12 dB the factor is 3.981, so 100 A rms is 398.1 A at full
    level against a 400 A continuous rating; 420 A at full level is
    over it."""
    path = _opened(panel, tmp_path, test_kind='Sine Sweep',
                   rated_current_rms='400')
    _key(panel.level_row.voltage_box, 'W')
    form = panel.form
    assert form is not None and form.name == 'sweep'
    assert form.box_text('n') == '1'
    form.set_box('direction', 'up')
    form.set_box('of', '3')
    form.set_box('rate', '1 oct/min')
    form.set_box('span', '5-2000 Hz')
    form.set_box('level_db', '-12')
    form.set_box('current', '100')
    assert form.live_label.text() == (
        'Sweep 1 of 3. Continuous rating: 398.1 A at full level, within the '
        'continuous current rating of 400 A (100%)')
    form.set_box('level_db', '0')
    form.set_box('current', '420')
    assert form.live_label.text().endswith(
        '420 A at full level, above the continuous current rating of 400 A '
        '(105%)')
    _write(panel, form)
    (sweep,) = _entries(path, 'sweep')
    assert dict(sweep.fields) == {'direction': 'up', 'n': 1, 'of': 3,
                                  'rate': '1 oct/min', 'span': '5-2000 Hz',
                                  'level_db': 0.0, 'current': 420.0}
    assert (sweep.test, sweep.axis, sweep.run) == (1, 'Z', 1)
    form = panel.open_form('sweep')
    assert [form.box_text(name) for name in ('n', 'of', 'rate', 'span',
                                             'level_db', 'direction')] == [
        '2', '3', '1 oct/min', '5-2000 Hz', '0', '']
    form.set_box('span', 'wide')
    assert _refused(panel, form) == ("a sweep's span 'wide' is not read as a "
                                     'band: give it as low-high in Hz, such '
                                     'as 180-220 Hz')
    assert len(_entries(path, 'sweep')) == 1


def test_a_sweep_with_no_rating_or_reading_says_why_it_is_not_judged(
        panel, tmp_path):
    _opened(panel, tmp_path, test_kind='Sine Sweep')
    form = _form(panel, 'sweep', level_db='0', current='300')
    assert form.live_label.text() == (
        'Continuous rating: not judged: the header gives no continuous '
        'current rating')
    form.cancelled.emit()
    _write(panel, _form(panel, 'header', rated_current_rms='400'))
    form = _form(panel, 'sweep', level_db='0')
    assert form.live_label.text() == ('Continuous rating: not judged: no '
                                      'current was read')
    form.set_box('level_db', '')
    form.set_box('current', '300')
    assert form.live_label.text() == ('Continuous rating: not judged: no '
                                      'level is given')
    form.set_box('level_db', '3')
    assert 'above full level' in form.live_label.text()


def test_a_shot_counts_its_pulses_against_the_budget_as_it_is_typed(
        panel, tmp_path):
    path = _opened(panel, tmp_path, test_kind='Shock', pulse_budget='3')
    _key(panel.level_row.voltage_box, 'P')
    form = panel.form
    assert form is not None and form.name == 'shot'
    lines = form.live_label.text().splitlines()
    assert lines[0].startswith('Checks answered: 0 of 4; owed: the pulse or '
                               'spectrum against its tolerance')
    assert lines[1:] == ['Pulses so far: 0 of 3 pulses, 3 left',
                         'With this shot: 1 of 3 pulses, 2 left']
    for field, value in (('polarity', 'positive'), ('level_db', '-6'),
                         ('in_tolerance', 'yes'), ('within_stroke', 'yes'),
                         ('peak_current', '210'), ('peak_voltage', '120'),
                         ('prediction', 'next level fits the stroke')):
        form.set_box(field, value)
    assert form.live_label.text().splitlines()[0] == 'Checks answered: 4 of 4'
    _write(panel, form)
    (shot,) = _entries(path, 'shot')
    assert dict(shot.fields) == {
        'polarity': 'positive', 'level_db': -6.0, 'in_tolerance': True,
        'within_stroke': True, 'peak_current': 210.0, 'peak_voltage': 120.0,
        'prediction': 'next level fits the stroke'}
    assert (shot.axis, shot.run) == ('Z', 1)
    form = _form(panel, 'shot', pulses='3', in_tolerance='no')
    lines = form.live_label.text().splitlines()
    assert lines[0].startswith('Checks answered: 1 of 4; owed: ')
    assert lines[1:] == ['Pulses so far: 1 of 3 pulses, 2 left',
                         'With this shot: 4 pulses, over the budget of 3 by 1']
    _write(panel, form)
    assert panel.snapshot.pulses.message == '4 pulses, over the budget of 3 ' \
                                            'by 1'
    second = _entries(path, 'shot')[-1]
    assert any(text == f'The checks after shot #{second.id}: 3 of 4 owed'
               for text in panel.owed_texts())


def test_a_survey_result_is_written_as_its_runs_stop(panel, tmp_path):
    from visualdynamics.core.run_log_model import RunLogModel

    path = _opened(panel, tmp_path, test_kind='Functional')
    _write(panel, _form(panel, 'result', result='passed, all channels',
                        by='R. Analyst'))
    with RunLog.read(path) as log:
        (run,) = RunLogModel(log).runs(1)
    assert (run.result, run.by, run.axis) == ('passed, all channels',
                                             'R. Analyst', 'Z')
    form = _form(panel, 'result')
    assert _refused(panel, form) == 'Result is needed'


def test_only_the_readings_a_test_takes_are_offered(panel, tmp_path):
    acts = panel.actions_by_name

    def offered():
        return {name for name in ('sweep', 'shot', 'result')
                if acts[name].isVisible()}

    _new(panel, tmp_path)
    assert offered() == set()
    _write(panel, _form(panel, 'open test', test_kind='Random Vibration'))
    assert offered() == set()
    for kind, shown in (('Shock', {'shot'}), ('Sine Sweep', {'sweep'}),
                        ('Random and Sine', {'sweep'}),
                        ('Modal Test', {'result'}),
                        ('Other', {'sweep', 'shot', 'result'})):
        _write(panel, _form(panel, 'roll', test_kind=kind))
        assert offered() == shown, kind


def test_every_form_has_its_key(panel, tmp_path):
    from visualdynamics.gui.run_log_dock import KEYS

    _opened(panel, tmp_path, test_kind='Other')
    wanted = ('header', 'sweep', 'shot', 'result', 'notch', 'deviation',
              'anomaly', 'disposition', 'setting change', 'opinion')
    letters = {name: KEYS[name].rsplit('+', 1)[1] for name in wanted}
    assert all(KEYS[name] == f'Ctrl+Shift+{letters[name]}' for name in wanted)
    assert len(set(KEYS.values())) == len(KEYS), 'no key is taken twice'
    for name in wanted:
        panel.focus_level_row()
        _key(panel.level_row.voltage_box, letters[name])
        assert panel.form is not None and panel.form.name == name, name
    panel.focus_level_row()


# ---- notches and deviations -----------------------------------------------


def test_an_unplanned_notch_holds_and_a_deviation_naming_it_clears_it(
        panel, tmp_path):
    path = _opened(panel, tmp_path, **RANDOM,
                   notches_planned='Z: 180-220 Hz, 6 dB')
    form = _form(panel, 'notch', channel='12', band='190-210', depth_db='4')
    assert form.box_text('axis') == 'Z'
    assert form.live_label.text().splitlines()[0] == \
        'Planned: inside planned notch 1.'
    _write(panel, form)
    assert panel.snapshot.holds == ()
    form = _form(panel, 'notch', channel='12', band='400-450', depth_db='3')
    assert form.live_label.text().splitlines()[0] == (
        'Not planned: 400 to 450 Hz on axis Z is in no planned notch. '
        'Written, it holds the test until a deviation names it.')
    _write(panel, form)
    notches = _entries(path, 'notch')
    assert [e.fields['planned'] for e in notches] == [True, False]
    unplanned = notches[1]
    assert [hold.rule for hold in panel.snapshot.holds] == ['unplanned-notch']
    assert panel.message.text() == (f'Notch #{unplanned.id} is not planned: '
                                    'the test holds until a deviation names '
                                    'it (Ctrl+Shift+D).')
    _key(panel.level_row.voltage_box, 'D')
    form = panel.form
    assert form.name == 'deviation'
    assert form.box_text('refers_to') == str(unplanned.id)
    assert form.live_label.text().startswith(
        f'Answers the hold at #{unplanned.id}: the notch at '
        f'#{unplanned.id} is not planned')
    for field, value in (
            ('fact', 'limit channel 12 notched 400-450 Hz by 3 dB at -6 dB'),
            ('cause', 'bracket mode, not in the prediction'),
            ('effect', 'the article sees 3 dB less over 400-450 Hz'),
            ('agreed_by', 'C. Rep, R. Analyst')):
        form.set_box(field, value)
    _write(panel, form)
    (deviation,) = _entries(path, 'deviation')
    assert dict(deviation.fields) == {
        'number': 'D-1',
        'fact': 'limit channel 12 notched 400-450 Hz by 3 dB at -6 dB',
        'cause': 'bracket mode, not in the prediction',
        'effect': 'the article sees 3 dB less over 400-450 Hz',
        'agreed_by': 'C. Rep, R. Analyst', 'refers_to': unplanned.id}
    assert deviation.test == 1 and deviation.run is None
    assert panel.snapshot.holds == ()
    assert panel.status.holds_button.text() == 'Holds: 0'
    # the approval written later, when it is known, by correction
    item = next(panel.day_list.item(i) for i in range(panel.day_list.count())
                if panel.day_list.item(i).text().startswith(
                    f'#{deviation.id} '))
    panel.day_list.setCurrentItem(item)
    _write(panel, _form(panel, 'correct', approval='the lab form, signed'))
    assert _entries(path, 'deviation')[0].fields['approval'] == \
        'the lab form, signed'


def test_a_deviation_naming_nothing_says_it_answers_no_hold(panel, tmp_path):
    _opened(panel, tmp_path, **RANDOM)
    form = _form(panel, 'deviation')
    assert form.box_text('refers_to') == ''
    assert form.live_label.text().startswith('Names no entry')
    form.set_box('refers_to', '1')
    assert form.live_label.text() == '#1 raises no hold that a deviation ' \
                                     'answers'


def test_an_anomaly_opened_and_dispositioned(panel, tmp_path):
    path = _opened(panel, tmp_path, **RANDOM)
    _key(panel.level_row.voltage_box, 'M')
    form = panel.form
    form.set_box('fact', 'connector backshell loose after run 2')
    form.set_box('owner', 'R. Analyst')
    _write(panel, form)
    assert panel.snapshot.open_anomalies == ('A-1',)
    form = _form(panel, 'disposition', number='A-9')
    assert _refused(panel, form) == 'A-9 is not an anomaly opened in this log'
    form.cancelled.emit()
    form = panel.open_form('disposition')
    assert form.box_text('number') == 'A-1'
    form.set_box('disposition', 'retorqued, functional passed')
    form.set_box('by', 'R. Analyst')
    _write(panel, form)
    assert panel.snapshot.open_anomalies == ()
    (opened,) = _entries(path, 'anomaly_opened')
    (closed,) = _entries(path, 'anomaly_dispositioned')
    assert dict(opened.fields) == {
        'number': 'A-1', 'fact': 'connector backshell loose after run 2',
        'owner': 'R. Analyst'}
    assert dict(closed.fields) == {'number': 'A-1',
                                   'disposition': 'retorqued, functional '
                                                  'passed',
                                   'by': 'R. Analyst'}
    assert opened.test == closed.test == 1


def test_a_limit_changed_holds_and_an_opinion_is_written(panel, tmp_path):
    path = _opened(panel, tmp_path, **RANDOM)
    _write(panel, _form(panel, 'setting change', setting='channel 12 limit',
                        value='+3 dB', was='0 dB', governs='limit'))
    (change,) = _entries(path, 'setting_change')
    assert dict(change.fields) == {'setting': 'channel 12 limit',
                                   'value': '+3 dB', 'was': '0 dB',
                                   'governs': 'limit'}
    assert [hold.rule for hold in panel.snapshot.holds] == ['plan-change']
    assert panel.snapshot.deviation_wanted == (change.id,)
    _write(panel, _form(panel, 'opinion', to='C. Rep',
                        opinion='the bracket response looks like a loose '
                                'joint'))
    (opinion,) = _entries(path, 'opinion_given')
    assert dict(opinion.fields) == {
        'to': 'C. Rep',
        'opinion': 'the bracket response looks like a loose joint'}


# ---- drawn, and torn down -------------------------------------------------


def test_a_form_with_a_live_line_stays_inside_the_dock(panel, tmp_path, pump):
    """The line a form computes is placed in the form before it is
    shown, so it never opens as a window of its own and takes the
    keyboard from the dock."""
    _opened(panel, tmp_path, test_kind='Shock')
    form = panel.open_form('shot')
    pump()
    assert not form.live_label.isWindow()
    assert form.live_label.isVisibleTo(form)
    assert panel.window().isActiveWindow()
    assert not panel.open_form('note').live_label.isVisibleTo(panel)


def test_the_header_form_renders_headless_and_scrolls(panel, tmp_path, pump):
    _new(panel, tmp_path)
    form = panel.open_form('open test')
    pump()
    image = form.grab().toImage()
    colors = {image.pixel(x, y) for x in range(0, image.width(), 3)
              for y in range(0, image.height(), 3)}
    assert len(colors) > 2
    holder = panel.entry_stack.currentWidget()
    assert holder.widget() is form


def test_the_window_closed_with_a_form_up_releases_the_log_and_dies(
        qt_app, tmp_path, no_swallowed_errors):
    from visualdynamics.gui.main_window import MainWindow

    window = MainWindow(offscreen_3d=True)
    window.show()
    qt_app.processEvents()
    panel = window.show_run_log().panel
    panel.who = WHO
    path = _new(panel, tmp_path)
    panel.open_form('open test').set_box('test_kind', 'Shock')
    panel.form.write()
    panel.open_form('shot')
    ghost = weakref.ref(window)
    destroy_window(window, qt_app)
    del window, panel
    gc.collect()
    assert ghost() is None, 'something in the process still holds the window'
    with RunLog.open(path, who=WHO) as log:
        assert [e.kind for e in log.view()] == ['test_opened']
