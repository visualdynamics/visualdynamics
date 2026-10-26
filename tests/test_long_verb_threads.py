"""A long verb's worker makes, frees and collects no Qt object.

`MainWindow._run_long` runs a verb on a worker thread while this thread
turns the event loop. Qt objects belong to the thread that made them and
are deleted there; one deleted on the worker, under the loop turning
here, crashes the loop. PySide deletes a Python-owned Qt object wherever
its last reference goes, and Python's cyclic collector runs on whichever
thread happens to allocate. So the runner leaves nothing of Qt's for the
collector, and the collector does not run on the worker at all. Both are
checked here directly, so the unsafe pattern fails on any platform, not
only where the crash shows.
"""

from __future__ import annotations

import gc
import threading

import shiboken6
from conftest import left_for_the_collector

#: what `_run_long` makes for a call: other parts of the window (a plot's
#: context menu, for one) leave cycles of their own, which are not the
#: runner's and are freed on the window thread when the window closes
RUNNER_KINDS = ('_LongVerbRelay', 'QEventLoop', '_InputGate')


def _runners(found):
    return [name for name in found if name in RUNNER_KINDS]


def test_a_verb_that_raises_leaves_no_qt_object_for_the_collector(window):
    def refused(progress=None):
        progress(0, 1)
        raise ValueError('refused')

    gc.collect()
    said = []
    try:
        window._run_long('Refusing…', refused)
    except ValueError as failure:
        said.append(str(failure))
    assert said == ['refused'], 'raised here, on the window thread'
    # the traceback holds the runner's frame in a cycle: nothing the
    # runner made may be the collector's to delete
    assert _runners(left_for_the_collector()) == []


def test_a_verb_that_answers_leaves_no_qt_object_for_the_collector(window):
    gc.collect()
    assert window._run_long('Reading…', lambda progress=None: 7) == 7
    assert _runners(left_for_the_collector()) == []


def test_the_collector_does_not_run_on_the_worker(window):
    started = []

    def watch(phase, _info):
        if phase == 'start':
            started.append(threading.current_thread().name)

    def churn(progress=None):
        # cycles enough to cross every generation's threshold
        for _ in range(50000):
            held = []
            held.append(held)
        return gc.isenabled()

    gc.callbacks.append(watch)
    try:
        assert window._run_long('Churning…', churn) is False, \
            'automatic collection is off while the verb runs'
    finally:
        gc.callbacks.remove(watch)
    assert gc.isenabled(), 'and on again after'
    assert [name for name in started if name.startswith('long-verb')] == []


def test_one_relay_for_the_windows_life(window):
    from PySide6.QtCore import QEventLoop

    relay = window._long_relay
    assert not shiboken6.ownedByPython(relay), "the window's, not Python's"
    for _ in range(3):
        window._run_long('Reading…', lambda progress=None: progress(1, 1))
    assert window._long_relay is relay
    assert window.findChildren(QEventLoop) == [], 'no loop left behind'
