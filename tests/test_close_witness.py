"""The close witness: how far a close got, and every thread's stack while
it stalls, written to a file a smoke test names.

The Windows release smoke test asked the window to close and found it
still running 15 s later in four of seven releases, with nothing on
screen but the window (2026-11-03). These hold the witness to what it
has to say when that happens next: the steps in order, the stacks once
the interval passes, and nothing at all when no file is named.
"""

from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
import time

import pytest

from visualdynamics import gui


@pytest.fixture
def witness_file(tmp_path, monkeypatch):
    """A file named for the witness, and the witness disarmed after."""
    path = tmp_path / 'close.log'
    monkeypatch.setenv(gui.CLOSE_WITNESS, str(path))
    yield path
    # the dump timer is process-wide (pytest's own faulthandler_timeout
    # uses it too, and arms it again for the next test)
    faulthandler.cancel_dump_traceback_later()
    log = gui._close_witness.pop('file', None)
    if log is not None:
        log.close()


def _steps(path):
    return [line.split(' ', 1)[1] for line in path.read_text().splitlines()
            if line[:1].isdigit()]


def test_unset_it_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(gui.CLOSE_WITNESS, raising=False)
    gui.close_witness('close asked')
    assert 'file' not in gui._close_witness
    assert list(tmp_path.iterdir()) == []


def test_a_stall_is_dumped_with_every_thread(witness_file, monkeypatch):
    monkeypatch.setattr(gui, 'CLOSE_WITNESS_INTERVAL', 0.2)
    gui.close_witness('close asked')
    time.sleep(0.7)
    text = witness_file.read_text()
    assert _steps(witness_file) == ['close asked']
    # faulthandler's own heading for the thread that was stalled here,
    # and this very function on its stack
    assert 'Current thread' in text or 'Thread 0x' in text
    assert 'test_a_stall_is_dumped_with_every_thread' in text


def test_a_prompt_close_writes_the_steps_alone(witness_file):
    gui.close_witness('close asked')
    gui.close_witness('views closed')
    assert _steps(witness_file) == ['close asked', 'views closed']
    assert 'Thread' not in witness_file.read_text()


def test_closing_the_window_is_witnessed(window, pump, witness_file):
    window.close()
    pump()
    assert _steps(witness_file) == ['close asked', 'views closed']


def test_an_exit_past_the_thread_joins_says_so(tmp_path):
    # in a process of its own: 'threads joined' is written by atexit,
    # which runs once the interpreter has joined its non-daemon threads
    path = tmp_path / 'close.log'
    env = dict(os.environ, **{gui.CLOSE_WITNESS: str(path)})
    subprocess.run([sys.executable, '-c',
                    ('from visualdynamics.gui import close_witness; '
                     'close_witness("windows released")')],
                   env=env, check=True, timeout=60)
    assert _steps(path) == ['windows released', 'threads joined']
