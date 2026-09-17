"""Launching the app from a session that already chose a Qt.

`import sdynpy` binds pyqtgraph to PyQt5 and sets QT_API=pyqt5, and
neither can be undone in a running process — so a notebook or VS Code
session that has been using sdynpy cannot host visualdynamics's window. Rather
than explain that, `launch_gui` starts the app in a fresh interpreter
with both choices pinned to PySide6, which is what this holds it to.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The probe below poisons a subprocess the way `import sdynpy` poisons a
# session, which needs the Qt it names to actually be installed —
# pyqtgraph refuses to import at all when PYQTGRAPH_QT_LIB points at a
# module that is not there. sdynpy brings PyQt5 with it, so a working
# machine has it; a bare CI runner does not, and the test would be
# reporting on the runner rather than on visualdynamics.
needs_pyqt5 = pytest.mark.skipif(
    importlib.util.find_spec('PyQt5') is None,
    reason='no PyQt5 to bind pyqtgraph to, so the clash cannot be staged')

# poison this interpreter the way sdynpy poisons one, then ask visualdynamics
# for the app and inspect what it would start
PROBE = '''
import os, subprocess, sys
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'
os.environ['QT_API'] = 'pyqt5'
import pyqtgraph, visualdynamics
from visualdynamics.gui import qt_binding

print('parent:', qt_binding())
started = []
visualdynamics._spawn_gui(('one.vdyn', 'two.unv'),
                spawn=lambda cmd, env: started.append((cmd, env)))
command, environment = started[0]
print('pins:', environment['PYQTGRAPH_QT_LIB'], environment['QT_API'])
print('paths:', command[3:])
check = ('from visualdynamics.gui import check_qt_binding, qt_binding; '
         'check_qt_binding(); print("child:", qt_binding())')
result = subprocess.run([sys.executable, '-c', check], env=environment,
                        capture_output=True, text=True, timeout=120)
print((result.stdout or result.stderr).strip().splitlines()[-1])
'''


def _bound_to(monkeypatch, binding):
    """Stage a session that has already chosen `binding`.

    Only what visualdynamics reads has to be real: `qt_binding()` asks
    `sys.modules['pyqtgraph'].Qt.QT_LIB` and nothing else, so a stub
    stages the poisoned session without PyQt5 being installed at all.
    Installing a GPL Qt to test a package that never imports one would
    be staging the third party's behavior, not ours — that is what the
    end-to-end probe below is for, on a machine that has sdynpy.
    """
    import types

    monkeypatch.setitem(sys.modules, 'pyqtgraph', types.SimpleNamespace(
        Qt=types.SimpleNamespace(QT_LIB=binding)))


def _record_both_routes(monkeypatch):
    """(spawned, hosted): which way `launch_gui` went, without a window."""
    import visualdynamics
    import visualdynamics.gui

    spawned, hosted = [], []
    monkeypatch.setattr(visualdynamics, '_spawn_gui',
                        lambda paths=(): spawned.append(list(paths)))
    monkeypatch.setattr(visualdynamics.gui, 'main', hosted.append)
    return spawned, hosted


def test_the_spawned_app_is_pinned_to_our_qt_and_carries_the_files(monkeypatch):
    """The child's environment is the decision: it inherits a session
    that chose another Qt, so both variables are overridden rather than
    defaulted — QT_API too, or pyvistaqt follows the parent into a
    mixed-binding crash.

    The parent's variables are set to what sdynpy sets first, which is
    the only state where this bites: `import visualdynamics` puts
    pyside6 in an empty environment, so a test run from one passes with
    the pinning deleted.
    """
    import visualdynamics

    monkeypatch.setenv('PYQTGRAPH_QT_LIB', 'PyQt5')
    monkeypatch.setenv('QT_API', 'pyqt5')
    started = []
    visualdynamics._spawn_gui(
        ('one.vdyn', 'two.unv'),
        spawn=lambda command, env: started.append((command, env)))
    (command, environment), = started
    assert environment['PYQTGRAPH_QT_LIB'] == 'PySide6'
    assert environment['QT_API'] == 'pyside6'
    assert command[0] == sys.executable, 'this interpreter, freshly'
    assert 'launch_gui' in command[2] and 'in_process=True' in command[2], (
        'the child must not decide again and spawn a third')
    assert command[3:] == ['one.vdyn', 'two.unv'], 'files ride along'


def test_a_session_bound_to_another_qt_spawns_rather_than_hosting(monkeypatch):
    import visualdynamics

    _bound_to(monkeypatch, 'PyQt5')
    spawned, hosted = _record_both_routes(monkeypatch)
    visualdynamics.launch_gui('one.vdyn')
    assert spawned == [['one.vdyn']] and hosted == [], (
        'a window built out of two Qt bindings is the thing being avoided')


def test_a_session_already_on_our_qt_keeps_the_window(monkeypatch):
    """pyqtgraph bound to PySide6 is the arrangement visualdynamics wants;
    spawning there would cost a second interpreter for nothing."""
    import visualdynamics

    _bound_to(monkeypatch, 'PySide6')
    spawned, hosted = _record_both_routes(monkeypatch)
    visualdynamics.launch_gui('one.vdyn')
    assert hosted == [['one.vdyn']] and spawned == []


def test_in_process_overrides_the_decision_either_way(monkeypatch):
    import visualdynamics

    _bound_to(monkeypatch, 'PyQt5')
    spawned, hosted = _record_both_routes(monkeypatch)
    visualdynamics.launch_gui(in_process=True)
    assert hosted == [[]] and spawned == []
    _bound_to(monkeypatch, 'PySide6')
    visualdynamics.launch_gui(in_process=False)
    assert spawned == [[]]


@needs_pyqt5
def test_a_taken_binding_sends_the_app_to_its_own_process():
    """The premise the tests above stub out, end to end: that a real
    poisoned session really does report PyQt5, and that the child really
    does pass the binding check. It stages the clash with the Qt sdynpy
    brings, so it runs on a machine that has sdynpy and is skipped where
    the clash cannot be staged — visualdynamics itself never imports
    PyQt5, and nothing that ships names it."""
    result = subprocess.run([sys.executable, '-c', PROBE], cwd=ROOT,
                            capture_output=True, text=True, timeout=300,
                            check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    out = result.stdout
    assert 'parent: PyQt5' in out, 'the session was poisoned as intended'
    assert 'pins: PySide6 pyside6' in out, (
        'both choices are overridden for the child — QT_API too, or '
        'pyvistaqt follows the parent into a mixed-binding crash')
    assert "paths: ['one.vdyn', 'two.unv']" in out, 'files ride along'
    assert 'child: PySide6' in out, (
        'the fresh process passes the binding check the parent fails')


def test_a_clean_session_hosts_the_window_itself():
    """Nothing has chosen a Qt yet, so the app runs in this process —
    spawning would cost a second interpreter for nothing."""
    probe = ('import visualdynamics\n'
             'from visualdynamics.gui import qt_binding\n'
             'print("binding:", qt_binding())\n'
             'import visualdynamics.plot, pyqtgraph\n'
             'print("after ours:", qt_binding())\n')
    result = subprocess.run([sys.executable, '-c', probe], cwd=ROOT,
                            capture_output=True, text=True, timeout=300,
                            check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'binding: None' in result.stdout, (
        'importing visualdynamics must not drag pyqtgraph in with it')
    assert 'after ours: PySide6' in result.stdout, (
        'and when it does arrive it is ours')


def test_known_qt_noise_is_filtered_out():
    """Three messages Qt and Chromium emit that nobody can act on. Each
    is suppressed with its reason recorded beside it, not swept up by a
    catch-all — anything else still reaches the terminal.
    """
    import contextlib
    import io as _io

    from visualdynamics.gui import _quiet_message_handler

    noisy = [
        'QWidget::activateWindow: window must be a top level window',
        ('Texture 0x1 () belongs to QRhi 0x2, but client code attempted '
         'to use it with QRhi 0x3. This is wrong.'),
        'ResizeObserver loop completed with undelivered notifications.',
    ]
    captured = _io.StringIO()
    with contextlib.redirect_stderr(captured):
        for message in noisy:
            _quiet_message_handler(None, None, message)
        _quiet_message_handler(None, None, 'something worth reading')
    assert captured.getvalue().strip() == 'something worth reading'
