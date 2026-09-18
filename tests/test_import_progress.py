"""A progress bar for imports, honest about what it knows.

A project file is minutes of someone's day, and its reader is the only
thing that knows how far along it is — so `native.load` reports per
object, a multi-file drop reports per file, and a single foreign file
shows no bar at all: it is one unreported read, and a busy-bar over a
deliberately blocked event loop cannot animate, so it would read as a
hang. The bar repaints synchronously, never via processEvents — the
loop stays blocked on purpose, and reopening it mid-import is the
re-entrancy that once crashed a drop.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics import io


@pytest.fixture
def project_file(tmp_path):
    """A three-object project on disk."""
    geometry = visualdynamics.Geometry(
        node_id=[1, 2, 3], node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        length_unit='m')
    time = visualdynamics.TimeHistory(
        np.arange(8) / 256.0, np.ones((1, 8)), response_dof=['101Z+'])
    from visualdynamics.core.channel_table import ChannelTable

    table = ChannelTable({'channel': [1], 'node': ['101'],
                          'direction': ['Z+'], 'unit': ['g']})
    path = tmp_path / 'survey.vdyn'
    io.save_test(path, 'Survey', {'Geometry': geometry, 'Time': time,
                                  'Channels': table})
    return str(path)


def test_the_loader_reports_per_object(project_file):
    ticks = []
    io.load(project_file, progress=lambda done, total: ticks.append(
        (done, total)))
    assert ticks == [(0, 3), (1, 3), (2, 3), (3, 3)], (
        'once up front, once per object')


def test_a_single_object_file_reports_nothing(tmp_path):
    """One object is one step, and a bar with one step is a light bulb."""
    geometry = visualdynamics.Geometry(
        node_id=[1], node_xyz=[[0, 0, 0]], length_unit='m')
    path = tmp_path / 'geometry.vdyn'
    io.save(geometry, path)
    ticks = []
    io.load(str(path), progress=lambda *call: ticks.append(call))
    assert ticks == []


def _bar_history(window, monkeypatch):
    """(value, maximum, visible) at every synchronous repaint."""
    from PySide6.QtWidgets import QProgressBar

    seen = []
    monkeypatch.setattr(
        QProgressBar, 'repaint',
        lambda self: seen.append((self.value(), self.maximum(),
                                  self.isVisible())),
        raising=True)
    return seen


def test_a_project_import_walks_the_bar(window, monkeypatch, project_file,
                                        pump):
    window.show()
    seen = _bar_history(window, monkeypatch)
    window.import_paths([project_file])
    pump()
    assert (3, 3, True) in seen, 'the bar reached the end, visibly'
    assert not window._import_progress.isVisible(), 'and was put away'


def test_a_single_foreign_file_shows_no_bar(window, monkeypatch, pump):
    window.show()
    seen = _bar_history(window, monkeypatch)
    window.import_paths([fixture_path('plate', 'geometry.unv')])
    pump()
    assert seen == [], 'no honest progress to show: no bar'
    assert not window._import_progress.isVisible()


def test_a_multi_file_drop_ticks_per_file(window, monkeypatch, pump):
    window.show()
    seen = _bar_history(window, monkeypatch)
    window.import_paths([fixture_path('plate', 'geometry.unv'),
                         fixture_path('plate', 'time.npz')])
    pump()
    values = [(value, maximum) for value, maximum, _shown in seen]
    assert (0, 2) in values and (1, 2) in values and (2, 2) in values
    assert not window._import_progress.isVisible()


def test_the_bar_walks_the_tree_building_too(window, monkeypatch,
                                             project_file, pump):
    """Half a second of reading, four of building rows: most of a big
    project import happens *after* the load, and a bar that filled and
    froze there read as no bar at all. The add phase re-walks it, one
    object at a time."""
    seen = _bar_history(window, monkeypatch)
    window.show()
    window.import_paths([project_file])
    pump()
    full = [entry for entry in seen if entry == (3, 3, True)]
    assert len(full) >= 2, (
        'the bar reaches its end twice: once loaded, once in the tree')


def test_paints_flush_through_the_event_loop_without_user_input(
        window, monkeypatch, project_file, pump):
    """repaint() alone painted into the backing store and no further:
    macOS flushes layer-backed views only from the event loop, so the
    bar moved and nobody could see it — reported twice as 'no loading
    bar'. Every tick must spin the loop with user input excluded."""
    from PySide6.QtCore import QEventLoop
    from PySide6.QtWidgets import QApplication

    flags = []
    real = QApplication.processEvents

    def spy(*args):
        flags.append(args[-1] if args else None)
        # QApplication.processEvents is static under PySide6

    monkeypatch.setattr(QApplication, 'processEvents',
                        staticmethod(spy), raising=True)
    try:
        window.show()
        window.import_paths([project_file])
    finally:
        monkeypatch.setattr(QApplication, 'processEvents',
                            staticmethod(real), raising=True)
    assert flags, 'the import never flushed a paint'
    assert all(
        flag == QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        for flag in flags), (
        'flushing paints must not let a click or a second drop in')


def test_a_second_import_mid_import_waits_its_turn(
        window, monkeypatch, project_file, tmp_path, pump):
    """The paint flush pumps the queue, and a drop landing mid-import
    queues a second import the pump would start *inside* the first —
    the re-entrancy all the drop deferral exists to prevent. It is
    refused on the spot and re-queued for afterwards."""
    import visualdynamics
    from visualdynamics import io as vio

    second = tmp_path / 'second.vdyn'
    vio.save(visualdynamics.Geometry(node_id=[1], node_xyz=[[0, 0, 0]],
                                     length_unit='m'), second)
    nested = []
    original = window._add_result

    def drop_arrives_mid_import(path, result, tick=None, options=None):
        if not nested:
            nested.append(window.import_paths([str(second)]))
        return original(path, result, tick=tick, options=options)

    monkeypatch.setattr(window, '_add_result', drop_arrives_mid_import)
    window.import_paths([project_file])
    assert nested == [[]], 'the nested call declined to import anything'
    # the declined import waits on a real 100 ms timer, so real time
    # has to pass before it runs — a sleepless pump never fires it
    import time
    deadline = time.monotonic() + 3.0
    while ('Geometry (2)' not in window.objects
           and time.monotonic() < deadline):
        time.sleep(0.02)
        pump()
    assert 'Geometry (2)' in window.objects, 'the queued import ran after'


def test_the_bar_paints_on_the_left_beside_the_words(
        window, monkeypatch, project_file, pump):
    """The bar lived on the right as a permanent widget, because a
    temporary status message obscures the left ones — and a bar a
    window's width away from the 'Importing …' text it belongs to
    reads as furniture. The import brings its own left-hand strip
    instead, so while it paints there must be no temporary message up
    (it would hide the strip), the words must ride the strip's label,
    and the bar must sit in the status bar's left half."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QProgressBar

    seen = []
    monkeypatch.setattr(
        QProgressBar, 'repaint',
        lambda self: seen.append((
            self.mapTo(window.statusBar(), QPoint(0, 0)).x(),
            window.statusBar().currentMessage(),
            window._import_label.text(),
            window._import_label.mapTo(window.statusBar(),
                                       QPoint(0, 0)).x())),
        raising=True)
    window.show()
    pump()
    window._show_status('Importing survey.vdyn…')
    window.import_paths([project_file])
    assert seen, 'the bar painted at all'
    for x, message, label, label_x in seen:
        assert x < window.statusBar().width() / 2, 'left half, always'
        assert x < label_x, 'the bar first, its caption after'
        assert message == '', 'no temporary message to obscure the strip'
        assert 'Importing' in label or 'switched' in label or label, (
            'the words ride the strip')
    assert 'Importing' in seen[0][2]
