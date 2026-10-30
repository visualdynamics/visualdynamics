"""The Run Log dock: opened from the menu and by `--run-log`, floated and
put back where it was left, the log alone in the window, the level row
and the abort key, the holds, a log read-only when another program
writes it, corrections struck through, and the window's teardown.

Every log is written under `tmp_path`, with a fixed login and machine
name, on a clock the test sets: the event time an act takes and the
entered-at time the log writes are both read off it, so a time "taken
at the key" is pinned by value.

What the launch imports is read in a fresh interpreter, since the
suite's own process has imported VTK long before any test here runs.
"""

from __future__ import annotations

import gc
import json
import os
import pathlib
import subprocess
import sys
import threading
import weakref
from datetime import datetime, timedelta, timezone

import pytest
from conftest import destroy_window, fixture_path, release_modifiers

import visualdynamics
from visualdynamics.core.run_log import RunLog
from visualdynamics.core.run_log_model import RunLogModel

ROOT = pathlib.Path(__file__).resolve().parent.parent
WHO = ('tester', 'bench')
ZONE = timezone(timedelta(hours=-6))
START = datetime(2026, 10, 30, 8, 0, tzinfo=ZONE)

HEADER = {'test_kind': 'Random Vibration', 'title': 'Random, Z',
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
    """The dock's panel in a fresh window, on the test's clock and with a
    fixed login and machine; no log open yet."""
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
    assert panel.snapshot is not None and panel.snapshot.writable
    panel.initials_box.setText('AB')
    return path


def _form(panel, name, **boxes):
    form = panel.open_form(name)
    for field, value in boxes.items():
        form.set_box(field, value)
    return form


def _write(panel, form):
    form.write()
    assert panel.settle()
    assert panel.form is None, (panel.message.text(), form.message.text())


def _opened(panel, tmp_path, **header):
    path = _new(panel, tmp_path)
    _write(panel, _form(panel, 'open test', **{**HEADER, **header}))
    panel.axis_box.setText('Z')
    return path


def _entries(path):
    """The log's corrected view, read from its file."""
    return RunLog.read(path).view()


def _keys(widget, text):
    from PySide6.QtTest import QTest

    widget.setFocus()
    QTest.keyClicks(widget, text)


def _press(widget, key, modifiers=None):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    QTest.keyClick(widget, key, modifiers or Qt.KeyboardModifier.NoModifier)
    if modifiers:
        # a chord leaves its modifiers held on the offscreen platform
        release_modifiers(widget)


# ---- the dock -------------------------------------------------------------


def test_the_file_menu_opens_the_dock(window, pump):
    from visualdynamics.gui.run_log_dock import RunLogDock

    assert window.run_log_dock is None, 'built before it was asked for'
    file_menu = next(a.menu() for a in window.menuBar().actions()
                     if a.text().replace('&', '') == 'File')
    action = next(a for a in file_menu.actions()
                  if a.text().replace('&', '') == 'Run Log')
    action.trigger()
    pump()
    dock = window.run_log_dock
    assert isinstance(dock, RunLogDock)
    assert dock.isVisible() and not dock.isFloating()
    assert window.dockWidgetArea(dock) == \
        window.dockWidgetArea(dock).RightDockWidgetArea
    assert window.centralWidget().isVisible(), 'the window is still whole'


def test_the_dock_floats_and_is_put_back_where_it_was_left(window_factory,
                                                         qt_app, pump):
    """Floated to a place of its own, closed with the window, and found
    floating there by the next window; docked on the left, found on the
    left."""
    from PySide6.QtCore import QRect, Qt

    first = window_factory()
    dock = first.show_run_log()
    dock.setFloating(True)
    dock.setGeometry(QRect(140, 120, 700, 520))
    pump()
    assert dock.isFloating()
    placed = dock.geometry()
    first.close()                   # where it is is remembered on closing
    second = window_factory()
    dock = second.show_run_log()
    pump()
    assert dock.isFloating(), 'not floating where it was left'
    assert abs(dock.width() - placed.width()) <= 2
    assert abs(dock.height() - placed.height()) <= 2
    dock.setFloating(False)
    second.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    pump()
    second.close()
    third = window_factory()
    dock = third.show_run_log()
    pump()
    assert not dock.isFloating()
    assert third.dockWidgetArea(dock) == Qt.DockWidgetArea.LeftDockWidgetArea


def test_the_log_alone_fills_the_window_and_the_whole_window_comes_back(
        window, pump):
    window.set_log_only(True)
    pump()
    dock = window.run_log_dock
    assert window.run_log_only
    assert not window.centralWidget().isVisible()
    assert not window.project_dock.isVisible()
    assert dock.width() >= window.width() - 20, 'the log fills the window'
    assert dock.panel.layout_action.text() == 'Whole Window'
    dock.panel.layout_action.trigger()
    pump()
    assert not window.run_log_only
    assert window.centralWidget().isVisible() and window.project_dock.isVisible()
    assert dock.isVisible() and dock.width() < window.width() - 20
    assert dock.panel.layout_action.text() == 'Log Only'


def test_a_log_not_yet_written_is_offered_to_the_new_log_form(window, pump,
                                                              tmp_path):
    """Named on the command line but not there: nothing is written until
    the keeper confirms the form."""
    path = str(tmp_path / 'next.vdlog')
    dock = window.show_run_log(path)
    pump()
    form = dock.panel.form
    assert form is not None and form.name == 'new log'
    assert form.box_text('path') == path
    assert not os.path.exists(path)


def test_the_command_line_flag():
    from visualdynamics.gui.run_log_dock import log_argument

    assert log_argument(['a.vdyn']) == (['a.vdyn'], False, None)
    rest, wanted, path = log_argument(['--run-log', 'day.vdlog', 'b.nc4'])
    assert rest == ['b.nc4'] and wanted
    assert path == os.path.abspath('day.vdlog')
    assert log_argument(['--run-log', 'b.nc4']) == (['b.nc4'], True, None)
    assert log_argument(['--run-log']) == ([], True, None)


# ---- the launch -----------------------------------------------------------

# The real launcher (`gui.main`) in a fresh interpreter, as
# test_first_3d_view runs it, with `--run-log`: the check and its message
# box replaced by recorders and the event loop by a function that reads
# the window, then goes to the whole window and opens a project the way
# File > Open does.
_LAUNCH = '''
import json, os, sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from visualdynamics import gui
from visualdynamics.gui import opengl

ROOTS = {'vtk', 'vtkmodules', 'pyvista', 'pyvistaqt'}


def vtk_loaded():
    return sorted(m for m in sys.modules if m.split('.')[0] in ROOTS)


log, path = json.loads(sys.argv[1])
asked, said, result = [], [], {}


def missing():
    asked.append(bool(vtk_loaded()))     # was VTK in before the check?
    return None


opengl.missing = missing
opengl.say = said.append


def run(app):
    from visualdynamics.gui.main_window import MainWindow

    window = next(w for w in app.topLevelWidgets()
                  if isinstance(w, MainWindow))
    dock = window.run_log_dock
    dock.panel.settle()
    for _ in range(5):
        app.processEvents()
    seen = dock.panel.snapshot
    result['log_only'] = window.run_log_only
    result['dock_shown'] = dock.isVisible()
    result['panes_shown'] = window.centralWidget().isVisible()
    result['tree_shown'] = window.project_dock.isVisible()
    result['fills'] = dock.width() >= window.width() - 20
    result['log'] = None if seen is None else seen.path
    result['writable'] = None if seen is None else seen.writable
    result['vtk_in_log'] = vtk_loaded()
    result['asked_in_log'] = len(asked)
    window.set_log_only(False)
    for _ in range(5):
        app.processEvents()
    result['asked_whole'] = len(asked)
    if path:
        window.import_path(path)
        for _ in range(5):
            app.processEvents()
        plotter = window.scene.built_plotter
        result['view'] = None if plotter is None else type(plotter).__name__
        result['actors'] = (0 if plotter is None
                            else len(plotter.renderer.actors))
    result['asked'] = asked
    result['said'] = said
    window.close()
    return 0


QApplication.exec = run
code = gui.main(['--no-disclaimer', '--run-log', log])
print('RESULT ' + json.dumps(result))
sys.exit(code)
'''


def _launch(tmp_path, log, path=None):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen',
               PYVISTA_OFF_SCREEN='true',
               VISUALDYNAMICS_SETTINGS=str(tmp_path / 'preferences.ini'))
    result = subprocess.run(
        [sys.executable, '-c', _LAUNCH, json.dumps([log, path])],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
        check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    line = next(line for line in result.stdout.splitlines()
                if line.startswith('RESULT '))
    return json.loads(line[len('RESULT '):])


@pytest.fixture
def day_log(tmp_path):
    path = tmp_path / 'day.vdlog'
    RunLog.create(path, title='Article 7', who=WHO).close()
    return str(path)


@pytest.fixture
def plate(tmp_path):
    """A project holding one geometry: opening it draws it in the scene."""
    project = visualdynamics.Project('Plate')
    project.import_file(fixture_path('plate', 'test_geometry.npz'))
    project.geometry.define_units('m')
    return str(project.save(tmp_path / 'plate.vdyn'))


def test_the_log_only_launch_imports_no_vtk_and_runs_no_check(tmp_path,
                                                            day_log):
    """`--run-log day.vdlog`: the window with the log alone in it, the log
    open for writing, and nothing of VTK in the process; going to the
    whole window with nothing to draw asks nothing either."""
    seen = _launch(tmp_path, day_log)
    assert seen['log_only'] and seen['dock_shown'] and seen['fills']
    assert not seen['panes_shown'] and not seen['tree_shown']
    assert seen['log'] == day_log and seen['writable']
    assert seen['vtk_in_log'] == [], 'VTK imported for the log alone'
    assert seen['asked_in_log'] == 0, 'the check ran for the log alone'
    assert seen['asked_whole'] == 0, 'asked with no 3-D view to build'


def test_the_whole_window_checks_before_its_first_3d_view(tmp_path, day_log,
                                                         plate):
    """From the log alone to the whole window in one session: the first
    3-D view still asks the check once, before VTK is imported, and the
    scene is built and drawn."""
    seen = _launch(tmp_path, day_log, plate)
    assert seen['vtk_in_log'] == [] and seen['asked_in_log'] == 0
    assert seen['asked'] == [False], 'asked once, before any VTK'
    assert seen['view'] == 'QtInteractor'
    assert seen['actors'] > 0, 'the geometry was drawn'
    assert seen['said'] == []


# ---- the level row --------------------------------------------------------


def test_a_level_row_typed_and_written_with_enter(panel, tmp_path, clock):
    """The level offered from the step schedule, the readings typed in tab
    order, Enter writes one entry with all of it, and the next step is
    offered. The event time is the row's first keystroke."""
    from PySide6.QtCore import Qt

    path = _opened(panel, tmp_path)
    row = panel.level_row
    assert row.level_box.text() == '-12'
    clock.move(minutes=1)
    typed_at = clock.now
    _keys(row.voltage_box, '50')
    clock.move(seconds=20)
    _keys(row.current_box, '63')
    _keys(row.drive_box, '0.12')
    row.answer_box.setCurrentText('as predicted')
    _keys(row.channels_box, '12: explained')
    row.go_box.setChecked(True)
    _press(row.drive_box, Qt.Key.Key_Return)
    assert panel.settle()
    level = [e for e in _entries(path) if e.kind == 'level_row']
    assert len(level) == 1, panel.message.text()
    entry = level[0]
    assert dict(entry.fields) == {
        'level_db': -12.0, 'voltage': 50.0, 'current': 63.0,
        'drive_fraction': 0.12, 'answer': 'as predicted',
        'channel_answers': {'12': 'explained'}, 'go': True}
    assert (entry.test, entry.axis, entry.run) == (1, 'Z', 1)
    assert entry.event_time == typed_at
    assert row.level_box.text() == '-6', 'the next step is offered'
    assert row.voltage_box.text() == '' and not row.go_box.isChecked()
    assert panel.status.place_label.text() == 'axis Z, run 1, -12 dB'
    assert panel.status.state_label.text() == 'RUNNING'


def test_a_level_row_that_cannot_be_read_is_not_written(panel, tmp_path):
    from PySide6.QtCore import Qt

    path = _opened(panel, tmp_path)
    row = panel.level_row
    _keys(row.voltage_box, 'fifty')
    _press(row.voltage_box, Qt.Key.Key_Return)
    assert panel.settle()
    assert "voltage: 'fifty' is not a number" in row.message.text()
    _keys(row.channels_box, '12 not explained')
    assert not any(e.kind == 'level_row' for e in _entries(path))
    row.voltage_box.setText('50')
    row.write()
    assert panel.settle()
    assert 'is not "channel: answer"' in row.message.text()


def test_the_projection_is_shown_as_the_row_is_typed(panel, tmp_path):
    """At -12 dB the factor is 3.981, so 63 A at sigma 3 is 752.4 A at
    full level against the 800 A rating; a check with no reading says
    why it was not judged."""
    _opened(panel, tmp_path)
    row = panel.level_row
    _keys(row.current_box, '63')
    said = row.projection_label.text()
    assert '-12 dB, factor 3.981 to full level: not judged' in said
    assert ('peak current: 752.4 A at full level, within the peak current '
            'rating of 800 A (94%)') in said
    assert 'peak voltage: not judged: no voltage was read' in said
    assert 'drive fraction: not judged: no drive fraction was read' in said
    _keys(row.voltage_box, '50')
    _keys(row.drive_box, '0.12')
    said = row.projection_label.text()
    assert 'peak voltage: 597.2 V at full level, above the rail of 250 V' \
        in said
    assert 'to full level: outside' in said
    row.level_box.setText('3')
    assert 'not judged: a level of +3 dB is above full level' in \
        row.projection_label.text()


# ---- the abort ------------------------------------------------------------


def test_the_abort_key_writes_the_stop_and_the_abort_at_the_keys_time(
        panel, tmp_path, clock, pump):
    """Ctrl+Shift+A, pressed while typing in the level row, takes the time
    at the key; written a minute later, both entries carry it, and what
    the first minute still owes is listed."""
    from PySide6.QtCore import Qt

    path = _opened(panel, tmp_path)
    row = panel.level_row
    row.voltage_box.setText('50')
    row.write()
    assert panel.settle()
    clock.move(minutes=5)
    pressed = clock.now
    _keys(row.current_box, '6')
    _press(row.current_box, Qt.Key.Key_A,
           Qt.KeyboardModifier.ControlModifier
           | Qt.KeyboardModifier.ShiftModifier)
    pump()
    form = panel.form
    assert form is not None and form.name == 'abort'
    assert form.title_label.text() == 'ABORT at 08:05:00'
    clock.move(minutes=1)
    form.set_box('stop', 'controller ramp')
    form.set_box('channel', '12')
    _press(form.boxes['channel'], Qt.Key.Key_Return)
    assert panel.settle()
    written = _entries(path)[-2:]
    assert [e.kind for e in written] == ['run_stop', 'abort']
    assert [e.event_time for e in written] == [pressed, pressed]
    assert all((e.test, e.axis, e.run) == (1, 'Z', 1) for e in written)
    assert dict(written[1].fields) == {'stop': 'controller ramp',
                                       'channel': '12'}
    said = panel.message.text()
    assert said.startswith(f'Abort #{written[1].id} written. Owed: ')
    for owed in ('band', 'panel fault', 'counter', 'called by', 'heard',
                 'records read', 'over test'):
        assert owed in said
    assert 'channel' not in said.split('Owed: ')[1].split(', ')
    assert f'The first minute of abort #{written[1].id}: 8 of 9 owed' in \
        panel.owed_texts()
    assert panel.status.state_label.text() == 'STOPPED: aborted'


def test_a_partial_abort_is_accepted_and_one_with_no_stop_is_not(panel,
                                                                tmp_path):
    path = _opened(panel, tmp_path)
    form = panel.abort()
    form.write()
    assert panel.settle()
    assert form.message.text() == 'Kind of stop is needed'
    assert not any(e.kind == 'abort' for e in _entries(path))
    form.set_box('stop', 'stop word')
    form.write()
    assert panel.settle()
    aborts = [e for e in _entries(path) if e.kind == 'abort']
    assert len(aborts) == 1 and dict(aborts[0].fields) == {'stop':
                                                           'stop word'}


def test_an_abort_with_no_initials_asks_for_them(panel, tmp_path):
    path = _opened(panel, tmp_path)
    panel.initials_box.clear()
    form = panel.abort()
    form.set_box('stop', 'power loss')
    form.write()
    assert panel.settle()
    assert 'initials' in panel.message.text()
    assert not any(e.kind == 'abort' for e in _entries(path))


# ---- the holds ------------------------------------------------------------


def test_the_holds_count_follows_a_hold_raised_and_cleared(panel, tmp_path,
                                                          pump):
    """An abort raises two (no cause, the over-test question); a cause
    found clears one, the over-test answer added by correction the
    other. The list is a click away."""
    _opened(panel, tmp_path)
    strip = panel.status
    assert strip.holds_button.text() == 'Holds: 0'
    form = panel.abort()
    form.set_box('stop', 'amplifier trip')
    form.write()
    assert panel.settle()
    assert strip.holds_button.text() == 'Holds: 2'
    strip.holds_button.click()
    pump()
    assert panel.holds_list.isVisible()
    held = [panel.holds_list.item(i).text()
            for i in range(panel.holds_list.count())]
    assert any('has no cause written' in text for text in held)
    assert any('over-test question' in text for text in held)
    abort_id = panel.snapshot.open_aborts[-1]
    cause = panel.open_form('cause found')
    assert cause.box_text('refers_to') == str(abort_id)
    cause.set_box('cause', 'loose connector on the drive cable')
    _write(panel, cause)
    assert strip.holds_button.text() == 'Holds: 1'
    item = next(panel.day_list.item(i) for i in range(panel.day_list.count())
                if panel.day_list.item(i).text().startswith(f'#{abort_id} '))
    panel.day_list.setCurrentItem(item)
    correct = panel.open_form('correct')
    correct.set_box('over_test', 'no: drive and control steady, records read')
    _write(panel, correct)
    assert strip.holds_button.text() == 'Holds: 0'
    assert panel.holds_list.item(0).text() == 'No holds.'


# ---- the day --------------------------------------------------------------


def test_a_correction_is_shown_struck_through_with_what_corrected_it(
        panel, tmp_path):
    path = _opened(panel, tmp_path)
    _write(panel, _form(panel, 'person in', person='C. Analyst',
                        role='analyst'))
    item = next(panel.day_list.item(i) for i in range(panel.day_list.count())
                if 'Person in' in panel.day_list.item(i).text())
    panel.day_list.setCurrentItem(item)
    form = panel.open_form('correct')
    assert form.box_text('person') == 'C. Analyst'
    form.set_box('role', 'customer')
    _write(panel, form)
    rows = [panel.day_list.item(i) for i in range(panel.day_list.count())]
    struck = [r for r in rows if r.font().strikeOut()]
    assert len(struck) == 1
    assert 'role: analyst' in struck[0].text(), 'shown as written'
    beneath = rows[rows.index(struck[0]) + 1]
    assert not beneath.font().strikeOut()
    assert 'corrected: Correction of #' in beneath.text()
    assert 'role: customer (was analyst)' in beneath.text()
    person = next(e for e in _entries(path) if e.kind == 'person_in')
    assert person.fields['role'] == 'customer'
    assert RunLog.read(path).entry(person.id).fields['role'] == 'analyst'


def test_a_late_entry_is_marked(panel, tmp_path, clock):
    """A note whose event time is typed half an hour back is written with
    that time, and the day marks it late with when it was entered."""
    path = _opened(panel, tmp_path)
    clock.move(hours=1)
    form = _form(panel, 'note')
    form.time_box.setText('08:30')
    form.text_box.setText('fixture bolt found finger tight')
    _write(panel, form)
    note = next(e for e in _entries(path) if e.kind == 'note')
    assert note.event_time == START.replace(minute=30)
    line = next(text for text in panel.day_texts() if 'Note' in text)
    assert line.endswith('LATE: entered 09:00:00')


def test_earlier_tests_are_a_click_away(panel, tmp_path):
    _opened(panel, tmp_path)
    _write(panel, _form(panel, 'note'))
    _write(panel, _form(panel, 'roll', test_kind='Shock', title='Shock, Z'))
    bar = panel.tests_bar
    tabs = [bar.tabText(i) for i in range(bar.count())]
    assert tabs == ['Whole log', 'Test 1: Random Vibration (closed to runs)',
                    'Test 2: Shock']
    assert panel.status.test_label.text() == 'Test 2: Shock, Z (Shock)'
    panel.show_test(1)
    shown = [text for text in panel.day_texts() if text.startswith('#')]
    assert shown and all('test 1' in text for text in shown)
    assert any('Release checklist' in text or 'release checklist' in text
               for text in panel.owed_texts())
    panel.show_test('all')
    assert any('test 2' in text for text in panel.day_texts())
    # looking at the current test, the keeper follows it on the roll;
    # looking at the whole log or an earlier test, stays
    panel.show_test(2)
    _write(panel, _form(panel, 'roll', test_kind='Sine Sweep'))
    assert bar.tabData(bar.currentIndex()) == 3
    panel.show_test(1)
    _write(panel, _form(panel, 'roll', test_kind='Shock'))
    assert bar.tabData(bar.currentIndex()) == 1


def test_esc_puts_a_form_away_and_nothing_is_written(panel, tmp_path, pump):
    from PySide6.QtCore import Qt

    path = _opened(panel, tmp_path)
    form = panel.open_form('note')
    form.text_box.setText('not this one')
    _press(form.text_box, Qt.Key.Key_Escape)
    pump()
    assert panel.form is None
    assert panel.entry_stack.currentWidget() is panel.level_row
    assert not any(e.kind == 'note' for e in _entries(path))


# ---- the file -------------------------------------------------------------


def test_a_log_held_by_another_writer_opens_read_only_saying_who(
        panel, tmp_path):
    path = str(tmp_path / 'held.vdlog')
    with RunLog.create(path, title='Article 7', who=('other', 'console')) \
            as held:
        held.append('note', author='CD', text='kept elsewhere')
        panel.open_log(path)
        assert panel.settle()
        seen = panel.snapshot
        assert seen is not None and not seen.writable
        notice = panel.status.notice_label.text()
        assert notice.startswith('Read-only: ')
        assert 'open for writing by other on console' in notice
        assert any('kept elsewhere' in text for text in panel.day_texts())
        assert not panel.level_row.isEnabled()
        assert not panel.actions_by_name['abort'].isEnabled()
        assert panel.actions_by_name['reload'].isEnabled()
        held.append('note', author='CD', text='added since')
        panel.reload()
        assert panel.settle()
        assert any('added since' in text for text in panel.day_texts())
    panel.close_log()
    assert panel.settle()


def test_a_mirror_problem_is_shown_in_the_strip(panel, tmp_path):
    away = tmp_path / 'share'
    away.mkdir()
    panel.new_log(str(tmp_path / 'm.vdlog'), 'Article 7',
                  str(away / 'm.vdlog'))
    assert panel.settle()
    panel.initials_box.setText('AB')
    assert not panel.status.notice_label.isVisibleTo(panel)
    (away / 'm.vdlog').unlink()
    away.rmdir()
    _write(panel, _form(panel, 'note'))
    notice = panel.status.notice_label.text()
    assert notice.startswith('Mirror: the mirror at ')
    assert 'was not written' in notice


def test_writing_with_no_log_open_says_so(panel):
    panel.initials_box.setText('AB')
    panel.level_row.voltage_box.setText('50')
    panel.level_row.write()
    assert panel.settle()
    assert panel.message.text().startswith('No log is open')


def test_the_worker_never_holds_the_keys(panel, tmp_path):
    """While a job is still running on the log's worker, keys typed into
    the level row land: the worker gates nothing."""
    _opened(panel, tmp_path)
    release = threading.Event()
    panel.worker.submit(lambda: release.wait(10), lambda ok, value: None)
    try:
        assert panel.worker.pending == 1
        _keys(panel.level_row.voltage_box, '48.5')
        assert panel.level_row.voltage_box.text() == '48.5'
        assert panel.worker.pending == 1, 'the job is still running'
    finally:
        release.set()
    assert panel.settle()


def test_the_window_closed_releases_the_log_and_dies(qt_app, tmp_path,
                                                    no_swallowed_errors):
    """The window's close writes what was given, releases the lock, and
    leaves nothing in the process holding the window."""
    from visualdynamics.gui.main_window import MainWindow

    window = MainWindow(offscreen_3d=True)
    window.show()
    qt_app.processEvents()
    panel = window.show_run_log().panel
    panel.who = WHO
    path = _new(panel, tmp_path)
    panel.open_form('note').write()
    ghost = weakref.ref(window)
    destroy_window(window, qt_app)
    del window, panel
    gc.collect()
    assert ghost() is None, 'something in the process still holds the window'
    with RunLog.open(path, who=WHO) as log:     # the lock was released
        assert [e.kind for e in log.view()] == ['note']


# ---- drawn without the window ---------------------------------------------


def _drawn(widget):
    """A grab with something on it: more than one color."""
    image = widget.grab().toImage()
    assert image.width() > 50 and image.height() > 10
    colors = {image.pixel(x, y) for x in range(0, image.width(), 3)
              for y in range(0, image.height(), 3)}
    return len(colors) > 2


def test_the_day_the_holds_and_the_strip_render_headless(panel, tmp_path,
                                                         pump):
    _opened(panel, tmp_path)
    form = panel.abort()
    form.set_box('stop', 'power loss')
    form.write()
    assert panel.settle()
    panel.status.holds_button.click()
    pump()
    for widget in (panel.status, panel.day_list, panel.holds_list,
                   panel.owed_tree):
        assert _drawn(widget), type(widget).__name__


def test_the_snapshot_is_the_models_reading(tmp_path):
    """What the dock shows is the model's, read once on the worker: the
    holds, the owed lists and the current run, by value."""
    from visualdynamics.core import run_log_model as model
    from visualdynamics.gui.run_log_dock import next_level, snapshot

    with RunLog.create(tmp_path / 's.vdlog', who=WHO) as log:
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header={'steps_db': [-12, -6, -3, 0]})
        model.append(log, 'level_row', author='AB', test=1, axis='X', run=1,
                     fields={'level_db': -12.0})
        seen = snapshot(log)
        reading = RunLogModel(log)
        assert seen.holds == tuple(reading.holds())
        assert seen.owed == tuple(reading.owed())
        assert (seen.run, seen.levels) == (1, (-12.0,))
        assert seen.status.outcome == 'running' and seen.status.axis == 'X'
    assert next_level([-12, -6, -3, 0], [-12.0]) == -6.0
    assert next_level([-12, -6, -3, 0], [0.0]) == 0.0
    assert next_level([-12, -6, -3, 0], []) == -12.0
    assert next_level(None, [-9.0]) == -9.0
    assert next_level(None, []) is None
