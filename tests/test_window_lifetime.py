"""A closed window dies.

Every test that asks for the `window` fixture makes a MainWindow, and
until 2026-09-13 every one of them outlived its test: `deleteLater`
only posts a deferred delete that `processEvents` never delivers,
and the application's style hints held a lambda capturing the window
for as long as the process ran. Fifteen megabytes a window empty,
far more with a project loaded, until an xdist worker reached 2.7 GB
and CI was killed for memory. Both halves are pinned here: the
fixture's teardown destroys the C++ window, and nothing left in the
process holds the Python one.
"""

from __future__ import annotations

import gc
import weakref

from conftest import destroy_window, fixture_path


def test_a_destroyed_window_is_gone(qt_app):
    from visualdynamics.gui.main_window import MainWindow

    window = MainWindow(offscreen_3d=True)
    window.show()
    qt_app.processEvents()
    ghost = weakref.ref(window)
    if getattr(window, 'report_editor', None) is not None:
        window.report_editor.view.setPage(None)
    destroy_window(window, qt_app)
    del window
    gc.collect()
    assert ghost() is None, 'something in the process still holds the window'


def test_a_window_closed_mid_load_leaves_no_timer_behind(qt_app,
                                                         no_swallowed_errors):
    """The report editor scrolls the page back 50 ms after it loads,
    from a single-shot timer. A window destroyed inside those 50 ms —
    every test's teardown, now that teardown destroys — took the view
    with it, and a bare timer then fired into a deleted
    QWebEngineView (Linux, 2026-09-13). The view is the timer's
    context, so a dead view means no call."""
    from PySide6.QtTest import QTest

    from visualdynamics.gui.main_window import MainWindow

    window = MainWindow(offscreen_3d=True)
    window.show()
    qt_app.processEvents()
    window.import_paths([fixture_path('plate', 'modal.nc4')])
    qt_app.processEvents()
    window.project.generate_report('modal')
    loaded = []
    window.show_object('Report')      # the editor connects its own
    view = window.report_editor.view  # loadFinished slot first
    view.loadFinished.connect(lambda _ok: loaded.append(True))
    for _ in range(200):              # up to 20 s: the first page of a
        if loaded:                    # process starts a browser engine
            break
        QTest.qWait(100)
    assert loaded, 'the report page never finished loading'
    destroy_window(window, qt_app)    # inside the timer's 50 ms
    del window
    QTest.qWait(300)                  # past the 50 ms the timer waits


def test_a_window_closed_mid_navigation_stops_the_page(qt_app,
                                                       no_swallowed_errors):
    """A window closed with a live report page destroyed the view while
    Chromium's teardown waited on its render process for good — the
    gate's stall at 98 %, sampled on 2026-09-18 with the worker parked
    inside QtWebEngineCore, then named by a faulthandler dump in the
    fixture's deferred delete after the page had loaded. Closing stops
    any navigation, drops the slot waiting on it, hides the view and
    discards the page, so the destructor finds nothing to wait for."""
    from visualdynamics.gui.main_window import MainWindow

    window = MainWindow(offscreen_3d=True)
    window.show()
    qt_app.processEvents()
    window.import_paths([fixture_path('plate', 'modal.nc4')])
    qt_app.processEvents()
    window.project.generate_report('modal')
    window.show_object('Report')          # a load in flight
    editor = window.report_editor
    assert editor._restore is not None, 'a navigation is waiting to finish'
    window.close()
    assert editor._restore is None, 'closing dropped the waiting slot'
    for _ in range(20):
        qt_app.processEvents()
    assert not editor.view.page().isLoading(), 'closing stopped the load'
    # the page's render process is gone before the destructor runs —
    # the faulthandler dump of 2026-09-18 put the hang there
    assert editor.view.page().lifecycleState().name == 'Discarded'
    assert not editor.view.isVisible()
    destroy_window(window, qt_app)

