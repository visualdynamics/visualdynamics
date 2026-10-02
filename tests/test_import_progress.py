"""A lone file's import moves the strip's bar, off the loop thread, and
can be cancelled (Brandon, 2026-10-01: "I don't ever really see a
progress bar on imports"). The nc4 reader counts its samples a slab at
a time; the dispatcher hands the callback to a reader that takes one;
the window runs the read through `_run_long`.
"""

from __future__ import annotations

import threading

from conftest import fixture_path

import visualdynamics


def test_the_nc4_reader_counts_its_samples():
    heard = []
    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'),
                                        progress=lambda d, t: heard.append((d, t)))
    history = loaded['time_data']
    assert heard, 'nothing heard'
    assert heard[-1] == (history.ordinate.shape[1], history.ordinate.shape[1])
    done = [d for d, _t in heard]
    assert done == sorted(done) and all(d <= t for d, t in heard)


def test_a_reader_without_a_progress_hook_is_not_handed_one():
    """A UFF file is one parse; the dispatcher must not pass it a
    keyword it does not take."""
    loaded = visualdynamics.import_file(fixture_path('plate', 'geometry.unv'),
                                        progress=lambda d, t: None)
    assert loaded is not None


def test_a_lone_import_runs_off_the_loop_and_moves_the_bar(window, pump, monkeypatch):
    from visualdynamics.gui import main_window as window_module

    seen, threads = [], []
    real_import = window_module.io.import_file
    real_tick = window._strip_tick

    def spy_import(path, **options):
        threads.append(threading.current_thread() is threading.main_thread())
        return real_import(path, **options)

    def spy_tick(done, total):
        real_tick(done, total)
        seen.append((done, total, window._import_strip.isVisibleTo(window),
                     window._import_progress.isVisibleTo(window)))

    monkeypatch.setattr(window_module.io, 'import_file', spy_import)
    monkeypatch.setattr(window, '_strip_tick', spy_tick)
    added = window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    assert added and threads == [False], 'the read ran on the loop thread'
    assert seen and seen[-1][0] == seen[-1][1] > 0
    assert all(strip and bar for _d, _t, strip, bar in seen)
    assert not window._import_strip.isVisibleTo(window), 'put away after'
    assert not window._importing


def test_cancelling_an_import_says_so_and_raises_no_failure(window, pump, monkeypatch):
    import time

    from PySide6.QtCore import QTimer

    from visualdynamics.gui import main_window as window_module

    warned = []
    monkeypatch.setattr(window_module.QMessageBox, 'warning',
                        lambda *a, **k: warned.append(a))

    def slow_import(path, progress=None, **options):
        for k in range(100):
            progress(k, 100)
            time.sleep(0.02)
        return {}

    monkeypatch.setattr(window_module.io, 'import_file', slow_import)
    QTimer.singleShot(150, window._cancel_button.click)
    added = window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    assert added == []
    assert 'import cancelled' in window._status_text
    assert warned == [], 'a cancel is not a failure to report'
