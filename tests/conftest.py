"""Shared fixtures.

The GUI ones matter most: a `MainWindow` built here runs in-process and
headless, so a GUI behavior can have a named test that runs in
milliseconds instead of living inside `gui_smoke_script.py` — which is one
subprocess, all-or-nothing, and reports failures as a line number.

`offscreen_3d=True` is what makes in-process safe: it keeps the Qt/VTK
render widget from ever being created, and that widget is what makes tearing
a window down mid-suite unsafe.
"""

from __future__ import annotations

import os
import tempfile

import pytest

TESTDATA = os.path.join(os.path.dirname(__file__), '..', 'testdata')

# The preferences the window remembers (File → Appearance) go to a
# folder of the session's own, never the user's real store — and this
# is done here, at import, not in a fixture: a fixture only guards the
# tests that ask for it, and a preferences test that did not ask for
# the app wrote 'light' into Brandon's own settings file (2026-09-14),
# which is exactly the kind of thing a test must never do.
SETTINGS_STORE = tempfile.mkdtemp(prefix='visualdynamics-settings-')
os.environ['VISUALDYNAMICS_SETTINGS'] = os.path.join(SETTINGS_STORE,
                                                     'preferences.ini')


def fixture_path(*parts):
    """A path into testdata/. Not named test* — pytest would collect it."""
    return os.path.join(TESTDATA, *parts)


def banded_pair_on_target(cut_at=1000.0, per_octave=6):
    """(banded specification, banded measurement sitting exactly on its
    target): the plate's target cut off above `cut_at` — mid-band, so
    the last written band is only partly covered — banded, and a
    measurement whose matched records carry the banded target itself.
    Any mark on it is a false one. The case behind the blue last band
    of 2026-09-19."""
    import copy

    import numpy as np

    import visualdynamics
    from visualdynamics.core.compliance import matched_records

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = copy.deepcopy(loaded['Random_specification'])
    beyond = np.asarray(spec.abscissa) > cut_at
    spec.ordinate = np.where(beyond[None, :], np.nan, np.asarray(spec.ordinate))
    spec.limits = {name: np.where(beyond[None, :], np.nan, np.asarray(values))
                   for name, values in spec.limits.items()}
    # the measurement carries the narrowband target on the lines it is
    # written, and nothing beyond — so the two band onto the very same
    # bins, the end band of each cut at the same place
    narrow = loaded['time_data'].compute_psds()
    narrow.ordinate[:, ~spec.written()] = np.nan   # every record speaks where the target does
    for _label, si, mi in matched_records(spec, narrow):
        target = np.real(spec.ordinate[si])
        narrow.ordinate[mi] = np.where(np.isfinite(target) & (target > 0),
                                       target, np.nan)
    banded = spec.to_octave(per_octave)
    measured = narrow.to_octave(per_octave)
    measured.scale_db = 0
    return banded, measured


def plate_geometry_and_shapes():
    """The plate's geometry in meters and its truth shapes — the pair
    every animation and deflection test starts from."""
    import visualdynamics

    return (visualdynamics.import_file(fixture_path('plate', 'geometry.npz'),
                                       length_unit='m'),
            visualdynamics.import_file(fixture_path('plate', 'shapes.npy')))


def select_objects(window, pump, *names):
    """Select these tree rows, the first one current. Current first:
    `setCurrentItem` clears a multi-selection."""
    window.tree.setCurrentItem(window._item_for_object(names[0]))
    window.tree.clearSelection()
    for name in names:
        window._item_for_object(name).setSelected(True)
    pump()


def edit_category(window, pump, label):
    """Open a geometry category's edit table, the way the pencil does,
    and return its row."""
    item = window._item_for_object('Geometry')
    item.setExpanded(True)
    child = next(item.child(i) for i in range(item.childCount())
                 if item.child(i).text(0).startswith(label))
    window.tree.clearSelection()
    window.tree.setCurrentItem(child)
    child.setSelected(True)
    window.edit_entities()
    pump()
    return child


def overlay_selection(window, pump, modes=None):
    """Mode picks beside the FRF: the synthesis-overlay selection.

    Selecting both whole objects opens the fit screen instead, so the
    overlay is reached the way it reads best anyway — by picking which
    modes to synthesize from.
    """
    shapes = window.objects['Shapes']
    window.tree.clearSelection()
    item = window._item_for_object('Shapes')
    item.setExpanded(True)
    pump()
    window.record_grids['Shapes'].select_records(
        list(modes) if modes is not None else list(range(shapes.num_shapes)))
    item.setSelected(True)
    window._item_for_object('FRF').setSelected(True)
    window.render_current()
    pump()


def prepared_comparison(window, pump):
    """A project holding a specification and PSDs of the same channels:
    the random run imported and its PSDs computed. Returns the two
    names, specification first."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    history = next(n for n, o in window.objects.items()
                   if type(o).__name__ == 'TimeHistory')
    item = window._item_for_object(history)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.compute_psds()
    pump()
    spec = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'Specification')
    psd = next(n for n, o in window.objects.items()
               if type(o).__name__ == 'Psd' and n != spec)
    return spec, psd


def menu_entries(window, qt_app, item):
    """What the context menu would show for this row: grab the visible
    menu from a zero-timer and close it. Texts keep their '&'."""
    from PySide6.QtCore import QPoint, QTimer
    from PySide6.QtWidgets import QMenu

    window.tree.scrollToItem(item)
    qt_app.processEvents()
    rect = window.tree.visualItemRect(item)
    position = next(
        (QPoint(rect.center().x(), y)
         for y in range(rect.top(), rect.bottom() + 1)
         if window.tree.itemAt(QPoint(rect.center().x(), y)) is item),
        None)
    assert position is not None, f'cannot aim at {item.text(0)!r} in {rect}'
    shown = []

    def grab():
        for widget in qt_app.topLevelWidgets():
            if isinstance(widget, QMenu) and widget.isVisible():
                shown.extend(a.text() for a in widget.actions() if a.text())
                widget.close()

    QTimer.singleShot(0, grab)
    window._show_tree_menu(position)
    qt_app.processEvents()
    return shown


@pytest.fixture(scope='session')
def qt_app():
    """One QApplication for the whole session; Qt allows exactly one."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # before the app exists: WebEngine (the report editor) refuses to
    # share GL contexts declared any later, and hangs under offscreen
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication([])
    # the report editor's witness log: the page's state at every
    # discard, appended per worker to a gitignored file, so the next
    # stall in the gate (PLAN.md "The 98 % stall, named") comes with
    # the state the faulthandler dump cannot give
    import logging

    witness = logging.getLogger('visualdynamics.gui.report_editor')
    witness.setLevel(logging.DEBUG)
    handler = logging.FileHandler(
        os.path.join(os.path.dirname(__file__), '.witness.log'))
    handler.setFormatter(logging.Formatter(
        f'%(asctime)s pid={os.getpid()} %(message)s'))
    witness.addHandler(handler)
    yield app
    # nothing of Python's left for Qt's static destructors to delete
    # after the interpreter is gone: a clipboard mime object from a
    # tree-copy test aborted a worker at exit (`gui.main` does the
    # same on the way out; a crash report, 2026-09-19)
    app.clipboard().clear()
    app.processEvents()


@pytest.fixture
def no_swallowed_errors():
    """Fail the test if Qt ate an exception.

    An exception raised inside a slot — a selection change, a signal —
    does not propagate: Qt prints it through sys.excepthook and carries
    on. So the app can be spraying tracebacks at the terminal while the
    suite stays green, which is exactly what happened when a pane
    attribute was read before __init__ had set it.
    """
    import sys

    caught = []
    previous = sys.excepthook
    sys.excepthook = lambda *info: caught.append(info)
    try:
        yield caught
    finally:
        sys.excepthook = previous
    if caught:
        import traceback

        first = ''.join(traceback.format_exception(*caught[0]))
        raise AssertionError(
            f'Qt swallowed {len(caught)} exception(s); the first:\n{first}')


def destroy_window(window, qt_app):
    """Close a window and actually destroy it.

    `deleteLater` only posts a deferred delete, and `processEvents`
    does not deliver those — so every window a test made outlived its
    test with its project, its plots and its tree, about 15 MB empty
    and far more loaded, until an xdist worker reached 2.7 GB and CI
    was killed for memory (2026-09-13, measured: 15 MB a window
    before this, 1.3 MB after). The posted delete is sent by hand.
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    # work a test queued for the next loop turn — a drop's deferred
    # import, a settled drag — lands on a live window first, not on a
    # destroyed one from inside the teardown
    for _ in range(3):
        qt_app.processEvents()
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    for _ in range(10):
        qt_app.processEvents()


@pytest.fixture
def window(qt_app, no_swallowed_errors):
    """A fresh headless main window, torn down after the test.

    A modal message box under a headless run is a hang: nobody clicks
    it, the event loop waits forever, and CI stalled 25 minutes on
    exactly that before being killed (2026-09-13 — a failed import's
    warning box). So every static QMessageBox door raises here,
    naming the box and its text; a test that expects one patches the
    door itself, as test_import_feedback does."""
    from PySide6.QtWidgets import QMessageBox

    from visualdynamics.gui.main_window import MainWindow

    def refuse(kind):
        def opened(parent, title='', text='', *args, **kwargs):
            raise AssertionError(
                f'a modal QMessageBox.{kind} opened under a headless '
                f'test — {title!r}: {text!r}')
        return staticmethod(opened)

    # patched by hand, not through the monkeypatch fixture: asking for
    # that fixture here would set it up before the window and tear it
    # down after, leaving a test's own patches (a fake report editor,
    # say) in place while the window closes
    doors = {kind: getattr(QMessageBox, kind)
             for kind in ('warning', 'critical', 'information', 'question')}
    for kind in doors:
        setattr(QMessageBox, kind, refuse(kind))

    window = MainWindow(offscreen_3d=True)
    window.resize(1400, 800)
    window.show()
    qt_app.processEvents()
    yield window
    # the frequency-axis choice is the process's, not the window's; a
    # test that flipped it must not hand decades to the next one
    from visualdynamics.core.data import frequency_axis
    frequency_axis('default')
    # a WebEnginePage still alive at interpreter exit segfaults the
    # process — detach the report editor's page before the window goes
    if getattr(window, 'report_editor', None) is not None:
        window.report_editor.view.setPage(None)
    destroy_window(window, qt_app)
    for kind, door in doors.items():
        setattr(QMessageBox, kind, door)


@pytest.fixture
def flat_reading(request):
    """Stand the 3-D stage down so a module tests the flat overlays'
    contracts — the drags, the rail, the table sync — which live on
    the 2-D plot; the stage's own marks have their tests
    (test_stage_marks). Opted into per module with
    `pytestmark = pytest.mark.usefixtures('flat_reading')`, never
    autouse: the toggle tests need the stage up."""
    if 'window' in request.fixturenames:
        window = request.getfixturevalue('window')
        window.data_pane.waterfall_action.setChecked(False)
    yield


@pytest.fixture
def flat_grid(request):
    """Ask for the flat MAC grid first: the bars are the default
    reading, and a module testing the grid's own machinery — aspect,
    ticks, zoom, visibility — opts in with `usefixtures`."""
    if 'window' in request.fixturenames:
        request.getfixturevalue('window').mac_bars_action.setChecked(False)
    yield


@pytest.fixture
def project(survey):
    """The plate's geometry, shapes and FRFs as a project dict — what a
    report is generated from."""
    import visualdynamics

    shapes, frfs = survey
    return {
        'Geometry': visualdynamics.import_file(fixture_path('plate',
                                                            'geometry.npz')),
        'Shape Set': shapes,
        'FRF': frfs,
    }


@pytest.fixture
def survey():
    """(truth shapes, matching FRFs): fixtures born from one model, so a
    fit or a synthesis can be checked against exact truth. Fresh objects
    per test — several tests define units on them."""
    import visualdynamics

    return (visualdynamics.import_file(fixture_path('plate', 'shapes.npy')),
            visualdynamics.import_file(fixture_path('plate', 'frfs.npz')))


@pytest.fixture
def window_factory(qt_app):
    """Extra fresh windows, for tests that move things between projects."""
    from visualdynamics.gui.main_window import MainWindow

    windows = []

    def make():
        window = MainWindow(offscreen_3d=True)
        window.resize(1400, 800)
        window.show()
        qt_app.processEvents()
        windows.append(window)
        return window

    yield make
    for window in windows:
        destroy_window(window, qt_app)


@pytest.fixture
def pump(qt_app):
    """Let queued work run — deferred re-renders, drop-down popups."""
    def pump(times=8):
        for _ in range(times):
            qt_app.processEvents()
    return pump


def web_read(view, expression, ready=bool, timeout=60.0, interval=0.25):
    """Ask a page `expression` until `ready(answer)`; return the answer.

    For every test that reads a rendered report back out of
    `QWebEngineView`. Three rules, each learned by a failure:

    - **Polled, never read once after a fixed delay.** A single reading
      1.5 s after load came back empty under a full four-worker run
      because the canvas did not exist yet (2026-09-23).
    - **Driven here, never through `app.exec()`.** An `exec()` returns
      at once on a `quit()` left pending by an earlier test in the same
      worker, which on CI looked like a page that never loaded at all
      (2026-09-21).
    - **One question in flight at a time.** `runJavaScript` is
      asynchronous; asking again every `interval` while a slow answer
      is still pending stacks up whole-canvas scans under load, the one
      condition in which they are slowest.

    `expression` should evaluate to a string (usually `JSON.stringify`)
    or to null while the page is not ready — and "ready" must be a
    fact about *this* page: the view's opening about:blank answers
    `document.readyState === 'complete'` too. Fails naming the last
    answer seen, so a timeout says what the page *did* have.
    """
    import time

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    state = {'pending': False, 'last': None, 'ok': False}

    def answered(value):
        state['pending'] = False
        state['last'] = value
        if ready(value):
            state['ok'] = True

    deadline = time.monotonic() + timeout
    while not state['ok'] and time.monotonic() < deadline:
        app.processEvents()
        if not state['pending']:
            state['pending'] = True
            view.page().runJavaScript(expression, 0, answered)
            asked = time.monotonic()
        elif time.monotonic() - asked > interval * 40:
            state['pending'] = False     # an answer lost, not merely late
        time.sleep(0.01)
    assert state['ok'], (
        f'the page never answered ready within {timeout:g} s; '
        f'last answer: {state["last"]!r}'
    )
    return state['last']


def web_close(view):
    """Tear a view down while the application still runs.

    A page alive at interpreter exit segfaults the process — two crash
    reports on 2026-09-20 were exactly that — so every test that makes
    a view ends by calling this.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    view.setPage(None)
    view.deleteLater()
    for _ in range(10):
        app.processEvents()
