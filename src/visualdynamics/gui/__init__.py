"""Desktop GUI. Start with `visualdynamics-gui [files...]` or `python -m visualdynamics`."""

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
    """Which Qt pyqtgraph is bound to, or None if it is still free.

    None means nothing has imported pyqtgraph yet, so importing visualdynamics's
    own widgets first will decide it in our favor.
    """
    import sys

    module = sys.modules.get('pyqtgraph')
    return None if module is None else module.Qt.QT_LIB


def check_qt_binding() -> None:
    """Point pyqtgraph at the same Qt as the rest of the GUI, or explain.

    pyqtgraph binds to one Qt for the life of the process, chosen when it is
    first imported: whichever binding is already in `sys.modules`, and
    failing that its own order of preference, which puts **PyQt6 ahead of
    PySide6**. Having PySide6 installed is not enough.

    visualdynamics is a PySide6 application, so a pyqtgraph on any other binding hands
    back plot widgets a PySide6 layout will not accept. That surfaced as
    `QSplitter.addWidget called with wrong argument types` from deep inside
    the main window, naming nothing useful.

    So: import PySide6 first, which decides it in our favor whenever
    pyqtgraph has not yet been imported. When it has — sdynpy imports it
    with PyQt5 at `import sdynpy` — the choice cannot be undone, and all
    that is left is to say so before a window is built out of the mismatch.
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
    """Take `--theme dark`, `--theme light` or `--theme=dark` out of
    the arguments: the rest, and the theme named, or None."""
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


def main(argv: Sequence[str] | None = None) -> int:
    import os

    from PySide6.QtCore import qInstallMessageHandler

    # silence a harmless Qt-internal warning (QStyleHints unique connections)
    # that PySide6/pyvistaqt trigger on startup
    os.environ.setdefault('QT_LOGGING_RULES', 'qt.core.qobject.connect=false')
    qInstallMessageHandler(_quiet_message_handler)

    import signal

    from PySide6.QtWidgets import QApplication

    check_qt_binding()      # before a window is built out of the mismatch

    from ..theme import OVERRIDE as THEME_OVERRIDE
    from .main_window import MainWindow

    # let Ctrl-C in the launching terminal kill the GUI immediately instead
    # of landing as a KeyboardInterrupt inside some Qt callback
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    from PySide6.QtCore import QTimer

    argv = list(sys.argv[1:] if argv is None else argv)
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
    # what the machine calls this while it is running — and it must be
    # said *before* the QApplication exists. Cocoa's application menu
    # ('Hide …', 'Quit …') is built during construction, titled from
    # the application name as it stands at that instant, which defaults
    # to argv[0]'s basename: '__main__.py'. Setting the name afterwards
    # renames the menu bar's title but never those already-built items.
    QApplication.setApplicationName('Visual Dynamics')
    QApplication.setApplicationDisplayName('Visual Dynamics')
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from .icons import app_icon

    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.show()
    # macOS: a window from a terminal python opens behind the terminal
    # unless explicitly raised and activated
    window.raise_()
    window.activateWindow()
    if with_disclaimer:
        # the alpha notice, acknowledged before the window takes input
        # — every launch, by decision (Brandon, 2026-09-11)
        from .disclaimer import acknowledged

        if not acknowledged(window):
            return 0
    if argv:
        # Import once the event loop is running: an import may open a modal
        # units dialog, which cannot be displayed before app.exec() starts.
        QTimer.singleShot(0, lambda: [window.import_path(p) for p in argv])
    return app.exec()
