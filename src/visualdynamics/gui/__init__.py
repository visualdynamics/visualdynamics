"""The desktop application.

Start it with `visualdynamics [files...]` (the older `visualdynamics-gui`
still works) or `python -m visualdynamics`. Files named on the command
line are imported once the window is open, `--theme dark` or
`--theme light` sets the theme, and `--no-disclaimer` skips the alpha
notice. On Windows, a machine without the OpenGL the 3-D view needs is
told so before the window is built (`opengl`).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

# Qt messages that say nothing anyone can act on.
#
# 'must be a top level window': embedding VTK forces sibling widgets to
# become native windows on macOS, so activating the window while focus
# sits on one of them makes Qt ask to activate a non-top-level window.
# The request is ignored; only the warning reaches the terminal.
#
# 'belongs to QRhi': floating the dock that holds the report editor moves
# a QtWebEngine view into a native window of its own, and it goes on
# using the texture it was built with. It fires only on that float —
# hiding, closing and re-docking are all silent — and the report renders
# correctly before, during and after, checked by grabbing the view at
# each step. Setting AA_ShareOpenGLContexts before the QApplication and
# cycling the view's visibility after the reparent were both tried and
# changed nothing. It is Qt talking to itself about its own texture
# bookkeeping.
#
# 'ResizeObserver loop': Chromium's own console noise, from the same view.
_IGNORED_QT_MESSAGES = ('must be a top level window',
                        'belongs to QRhi',
                        'ResizeObserver loop')


def _quiet_message_handler(_mode: Any, _context: Any,
                           message: str) -> None:
    """Pass Qt messages through, minus the ones that are pure noise."""
    if any(text in message for text in _IGNORED_QT_MESSAGES):
        return
    sys.stderr.write(message + '\n')


def qt_binding() -> str | None:
    """The Qt binding pyqtgraph uses, or None if it has not chosen one.

    None means nothing has imported pyqtgraph yet, so importing the
    Visual Dynamics widgets first will make it use PySide6.
    """
    import sys

    module = sys.modules.get('pyqtgraph')
    return None if module is None else module.Qt.QT_LIB


def check_qt_binding() -> None:
    """Make pyqtgraph use the same Qt binding as the rest of the GUI, or
    raise an error that says why it cannot.

    pyqtgraph binds to one Qt for the life of the process, chosen when it
    is first imported: whichever binding is already in `sys.modules`, and
    otherwise its own order of preference, which puts PyQt6 ahead of
    PySide6. Having PySide6 installed is not enough.

    Visual Dynamics is a PySide6 application, and a PySide6 layout will
    not accept plot widgets from another binding. The failure otherwise
    comes from deep inside the main window, as `QSplitter.addWidget
    called with wrong argument types`, and does not name the cause.

    So this imports PySide6 first, which settles the binding whenever
    pyqtgraph has not been imported yet. If it has (sdynpy, for example,
    imports it with PyQt5 at `import sdynpy`), the choice cannot be
    undone, and this raises RuntimeError before a window is built.
    """
    import importlib

    import PySide6.QtWidgets  # noqa: F401  — imported for its side effect

    # deliberately not an import statement: these two are ordered, and an
    # import sorter would put pyqtgraph first and undo the whole point
    binding = importlib.import_module('pyqtgraph').Qt.QT_LIB
    if binding == 'PySide6':
        return
    raise RuntimeError(
        f'pyqtgraph is using {binding}, but visualdynamics is a PySide6 application, '
        f'and widgets from two Qt bindings cannot share a window.\n'
        f'\n'
        f'pyqtgraph picks its binding when it is first imported, and '
        f'something imported it before visualdynamics — sdynpy does this, with PyQt5, '
        f'at "import sdynpy". The choice cannot be undone in a running '
        f'process.\n'
        f'\n'
        f'`visualdynamics.launch_gui()` handles this by starting the app in a '
        f'fresh process; this path is for running the window inside the '
        f'current one, which needs an interpreter that has not imported '
        f'sdynpy — or PYQTGRAPH_QT_LIB=PySide6 set before any import, '
        f'which then breaks sdynpy\'s own plotting in that process.')


def theme_flag(argv: list[str]) -> tuple[list[str], str | None]:
    """Remove `--theme dark`, `--theme light` or `--theme=dark` from the
    arguments. Returns the remaining arguments and the theme named
    (lower case), or None when there is none; raises SystemExit for a
    theme other than dark or light."""
    rest: list[str] = []
    theme = None
    skip = False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg.startswith('--theme='):
            theme = arg.split('=', 1)[1]
        elif arg == '--theme' and i + 1 < len(argv):
            theme = argv[i + 1]
            skip = True
        else:
            rest.append(arg)
    if theme is not None:
        theme = theme.strip().lower()
        if theme not in ('dark', 'light'):
            raise SystemExit(f"--theme takes dark or light, not {theme!r}")
    return rest, theme


def before_application() -> None:
    """What has to be set before the QApplication exists: the
    application's name, and one graphics system for Qt Quick and the
    3-D views (`opengl.one_graphics_system`)."""
    from PySide6.QtWidgets import QApplication

    from .opengl import one_graphics_system

    one_graphics_system()

    # what the machine calls this while it is running — and it must be
    # said *before* the QApplication exists. Cocoa's application menu
    # ('Hide …', 'Quit …') is built during construction, titled from
    # the application name as it stands at that instant, which defaults
    # to argv[0]'s basename: '__main__.py'. Setting the name afterwards
    # renames the menu bar's title but never those already-built items.
    QApplication.setApplicationName('Visual Dynamics')
    QApplication.setApplicationDisplayName('Visual Dynamics')


def main(argv: Sequence[str] | None = None) -> int:
    import os

    from PySide6.QtCore import qInstallMessageHandler

    # silence a harmless Qt-internal warning (QStyleHints unique connections)
    # that PySide6/pyvistaqt trigger on startup
    os.environ.setdefault('QT_LOGGING_RULES', 'qt.core.qobject.connect=false')
    qInstallMessageHandler(_quiet_message_handler)

    argv = list(sys.argv[1:] if argv is None else argv)
    # Before anything Qt. The GL check is Win32 calls, and a machine
    # that fails it must never build a QApplication: built on a
    # Windows machine with no OpenGL, the application crashed on the
    # way out behind Windows' crash dialog (the 0.1.0a56 release run,
    # 2026-10-18)
    from . import opengl

    if '--check-gl' in argv:
        # the packaged build's other self-check: can this machine draw
        # the 3-D view? (gui.opengl.check — 0 yes, 2 no, 1 unknown)
        code, message = opengl.check()
        print(message, flush=True)
        return code
    reason = opengl.missing()
    if reason is not None:
        opengl.say(reason)
        return 2

    import signal

    from PySide6.QtWidgets import QApplication

    check_qt_binding()      # before a window is built out of the mismatch

    from ..theme import OVERRIDE as THEME_OVERRIDE
    from .main_window import MainWindow

    # let Ctrl-C in the launching terminal kill the GUI immediately instead
    # of landing as a KeyboardInterrupt inside some Qt callback
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    from PySide6.QtCore import QTimer

    # a script that opens the window has read the notice; the flag is
    # its own and is not a file to import
    with_disclaimer = '--no-disclaimer' not in argv
    argv = [a for a in argv if a != '--no-disclaimer']
    # --theme dark|light: said into the environment, which is the one
    # place the theme module looks first, so the window and every
    # later `apply_theme` agree (2026-09-14)
    argv, theme = theme_flag(argv)
    if theme is not None:
        os.environ[THEME_OVERRIDE] = theme
    before_application()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    if '--check-movie' in argv:
        # the packaged build's self-check, no window: can this install
        # save an animation? (gui.movie.check — 0 yes, 2 no H.264 here)
        from .movie import check

        where = argv[argv.index('--check-movie') + 1:][:1] or ['check.mp4']
        code, message = check(where[0])
        print(message, flush=True)
        return code
    from .icons import app_icon

    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.show()
    # macOS: a window from a terminal python opens behind the terminal
    # unless explicitly raised and activated
    window.raise_()
    window.activateWindow()
    if with_disclaimer:
        # the alpha notice, acknowledged before the window takes input,
        # on the first launch of each version (Brandon, 2026-10-21; every
        # launch from 2026-09-11)
        from . import disclaimer

        if disclaimer.needed():
            if not disclaimer.acknowledged(window):
                return 0
            disclaimer.remember()
    if argv:
        # Import once the event loop is running: an import may open a modal
        # units dialog, which cannot be displayed before app.exec() starts.
        QTimer.singleShot(0, lambda: [window.import_path(p) for p in argv])
    code = app.exec()
    # A mime object handed to the clipboard (copying a tree item) is
    # Qt's to delete, and Qt deletes it from its static destructors —
    # after the interpreter has gone, where the Python-made object's
    # wrapper cannot run and the process aborts on the way out (a
    # macOS crash report from a test worker, 2026-09-19). Cleared
    # while both still stand.
    app.clipboard().clear()
    app.processEvents()
    release_widgets(app)
    return code


def release_widgets(app: Any) -> int:
    """Delete the windows Python still owns, one by one, while the
    application stands, and return how many there were.

    Left to the interpreter's exit, PySide deletes every widget it still
    owns in one sweep over its own table, in no particular order. A
    window deleted there takes its children with it, and the sweep then
    reaches a child it still lists and deletes it again: the process
    segfaults on its way out. Deleting each window here, through its
    wrapper, leaves the sweep nothing to do.
    """
    # a gate worker segfaulted at exit three times on 2026-10-22, in
    # PySide's moduleShutdown sweep, on a QListWidget, a QComboBox and a
    # QRadioButton of dialogs the tests had left open; PLAN.md, "Nothing
    # of Python's left for PySide's sweep"
    import gc

    import shiboken6
    from PySide6.QtCore import QCoreApplication, QEvent

    # the cycles first: a window held by nothing but its own connections
    # goes the way Python owns it, not the sweep's
    gc.collect()
    released = 0
    for widget in app.topLevelWidgets():
        # through shiboken, not deleteLater: a window Qt deletes leaves
        # its children's wrappers standing, and the sweep still crashed
        if (shiboken6.isValid(widget) and shiboken6.ownedByPython(widget)
                and widget.parent() is None):
            shiboken6.delete(widget)
            released += 1
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    return released
