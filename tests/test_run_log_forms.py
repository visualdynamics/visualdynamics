"""The Run Log's header forms: the test header in full, its files
hashed on the log's worker, and carried fields marked until confirmed.

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


def test_the_header_form_has_its_key(panel, tmp_path):
    from visualdynamics.gui.run_log_dock import KEYS

    _opened(panel, tmp_path, **RANDOM)
    assert KEYS['header'] == 'Ctrl+Shift+H'
    assert len(set(KEYS.values())) == len(KEYS), 'no key is taken twice'
    _key(panel.level_row.voltage_box, 'H')
    assert panel.form is not None and panel.form.name == 'header'


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


# ---- drawn, and torn down -------------------------------------------------


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
    panel.open_form('header')
    ghost = weakref.ref(window)
    destroy_window(window, qt_app)
    del window, panel
    gc.collect()
    assert ghost() is None, 'something in the process still holds the window'
    with RunLog.open(path, who=WHO) as log:
        assert [e.kind for e in log.view()] == ['test_opened']
