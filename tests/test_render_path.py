"""Rendering is synchronous, on the thread that asked for it.

On macOS pyvistaqt threads ``render()`` and signals back to the GUI
thread, so every render becomes a queued metacall — and a queued call
can arrive after its window has died, which is a SIGSEGV in
``vtkCocoaRenderWindow`` (seen in the field, 2026-08-20). ``undeferred``
removes the indirection at construction. These tests cannot reproduce
the race — it needs a real Cocoa window dying under a queued call — so
they pin the two things that close it: the helper really is synchronous,
and every plotter this application constructs goes through it.
"""

from __future__ import annotations

import ast
import pathlib
import threading

from visualdynamics.viz import undeferred

SOURCES = pathlib.Path(__file__).parent.parent / 'src' / 'visualdynamics'


class _ThreadedLikePyvistaqt:
    """The shape of QtInteractor's macOS render path, in miniature."""

    def __init__(self):
        self.calls: list[str] = []
        self._rendered = False

    def _render(self):
        self.calls.append(threading.current_thread().name)

    def render(self):
        thread = threading.Thread(target=self._render)
        thread.start()
        thread.join()


def test_an_undeferred_plotter_renders_on_the_calling_thread():
    plotter = undeferred(_ThreadedLikePyvistaqt())
    plotter.render()
    assert plotter.calls == [threading.current_thread().name], \
        'no worker thread, no queued delivery — the render ran here'
    assert plotter._rendered, 'the flag pyvistaqt calls crucial'


def test_every_qt_plotter_this_application_builds_is_undeferred():
    """The helper only helps at the construction site, and the fixture
    windows are offscreen so no test ever runs those sites — an AST
    walk is what holds them, the way the license boundary is held."""
    makers = {'QtInteractor', 'BackgroundPlotter'}
    bare = []
    for path in SOURCES.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        wrapped = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'undeferred'):
                for inner in ast.walk(node):
                    wrapped.add(id(inner))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in makers
                    and id(node) not in wrapped):
                bare.append(f'{path.relative_to(SOURCES)}:{node.lineno}')
    assert not bare, ('Qt plotters built without undeferred() — each is '
                      f'a queued-render crash waiting for a dead window: '
                      f'{bare}')


def test_undeferred_stops_the_auto_render_timer(qt_app):
    """QtInteractor defaults to auto_update=5.0: a QTimer calling
    render() five times a second for the life of the widget. Every
    draw here is explicit, so those ticks bought nothing — and the
    first to fire after the native window died was the field crash of
    2026-08-30, the same dead-QPlatformWindow SIGSEGV as the threaded
    render, arriving by timer instead of by queued metacall."""
    from PySide6.QtCore import QTimer

    plotter = _ThreadedLikePyvistaqt()
    plotter.render_timer = QTimer()
    plotter.render_timer.start(200)
    assert plotter.render_timer.isActive(), 'the default really ticks'
    undeferred(plotter)
    assert not plotter.render_timer.isActive(), \
        "nothing renders behind the app's back"


def test_closing_the_window_shuts_every_3d_view_down(window_factory, pump):
    """The scene-only closeEvent left the stage and the MAC bars to
    take their chances at teardown — same interactor, same GL context,
    same timers (Brandon's crash report, 2026-08-30)."""
    import numpy as np

    import visualdynamics

    window = window_factory()
    t = np.arange(2048) / 1024.0
    window.add_object('Run', visualdynamics.TimeHistory(
        t, np.random.default_rng(1).standard_normal((2, len(t))),
        response_dof=['101Z+', '104Z+'], ordinate_dim='acceleration'))
    item = window._item_for_object('Run')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert window.data_pane.waterfall_plotter is not None, \
        'the stage was built, so there is something to shut down'
    window.close()
    pump()
    assert window.data_pane.waterfall_plotter is None, \
        'the stage went down with the window'
    assert window._mac_bars_plotter_obj is None
