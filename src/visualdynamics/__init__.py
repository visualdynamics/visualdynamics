"""visualdynamics: units-aware structural dynamics analysis toolset."""

import os as _os
from typing import Any

# visualdynamics's widgets are PySide6, so every library that picks a Qt for
# itself must pick the same one. Left alone in an environment that
# also has PyQt5/PyQt6, pyqtgraph binds to one of those and every
# brush and widget visualdynamics hands it is a foreign object — which is how
# `data.plot()` from a plain script died on a color it could not
# read. `QT_API` is the same question asked by qtpy, which is how
# pyvistaqt picks. Both are set *only if unset*: a session that has
# already chosen (sdynpy sets QT_API=pyqt5 at import) is left alone
# rather than half-converted, and `launch_gui` starts the app in its
# own process instead.
_os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PySide6')
_os.environ.setdefault('QT_API', 'pyside6')

# Qt Data Visualization is offered under GPLv3 or a commercial license
# and nothing else — no LGPL (its own licensing page, read 2026-08-20).
# Nothing here uses it, but qtpy imports it at startup for a
# Windows-Qt5 compatibility alias, inside a suppress(ImportError) — so
# it was loaded into the process by accident of a dependency. A
# GPL-only module in the address space is exactly what the sanctioned
# Qt set exists to keep out: this project's licensing is deliberately
# undecided, and shipping combined with a GPL-only Qt add-on would
# decide it by accident. `setdefault`, not assignment: a host process
# that loaded the module itself before importing visualdynamics made
# its own licensing choice and is left alone. `None` in sys.modules
# makes any later import raise ImportError, which qtpy's suppress
# swallows and moves on from.
import sys as _sys

_sys.modules.setdefault('PySide6.QtDataVisualization', None)

from .compatibility import Issue, Report, check_compatibility
from .core import (
    ChannelTable,
    DataArray,
    Frf,
    Geometry,
    Psd,
    ShapeSet,
    ShockSpecification,
    Specification,
    Spectrum,
    Srs,
    TimeHistory,
    TransientSpecification,
    fem,
)
from .core.data import frequency_axis
from .io import (
    export_file,
    from_sep005,
    import_file,
    importers,
    load,
    register_importer,
    save,
)
from .project import Project, random_vibration_report, random_vibration_run
from .units import (
    DEFAULT_SYSTEM,
    FT_LBF_S,
    IN_LBF_S,
    MMKS,
    SI,
    SYSTEMS,
    UnitsRequired,
    UnitSystem,
    convert,
    si_factor,
)

#: alpha, said in the version itself so the dialog, the About text, the
#: installers and latest.json all say the same thing, and the first
#: non-alpha release is offered as an upgrade (Brandon, 2026-09-11)
__version__ = '0.1.0a4'


def launch_gui(*paths: str, in_process: bool | None = None) -> Any:
    """Launch the app, optionally importing files on the way in.

        visualdynamics.launch_gui('test.vdyn', 'frfs.unv')

    Normally this runs the window in *this* interpreter and blocks
    until it closes, and returns nothing — the window is gone, there is
    nothing to hand back. If pyqtgraph has already been bound to another
    Qt — `import sdynpy` does that, with PyQt5, and the choice cannot be
    undone in a running process — the app is started in a fresh process
    instead, and the `Popen` handle comes back straight away because
    that one is still running. So a session that has been using sdynpy
    still gets a window, and sdynpy's own plotting in that session is
    left alone.

    `in_process` overrides the decision either way.
    """
    from .gui import qt_binding

    if in_process is None:
        in_process = qt_binding() in (None, 'PySide6')
    if in_process:
        from .gui import main

        # `main` hands back Qt's exit status, which `visualdynamics-gui` wants as
        # its own and nothing else does — it is always 0, since nothing
        # here ever calls exit() with anything else. Swallowed, so
        # closing the window from a prompt does not answer with a bare
        # `0` from the REPL echoing it.
        main(list(paths))
        return None
    return _spawn_gui(paths)


def _spawn_gui(paths=(), spawn=None):
    """Start the app in a fresh interpreter, pinned to PySide6.

    `spawn` is the process starter, injectable so the decision can be
    tested without opening a window.
    """
    import subprocess
    import sys

    # the child is a fresh interpreter, but it inherits this one's
    # environment — including whatever chose a Qt here. Both are
    # overridden, not defaulted: sdynpy sets QT_API=pyqt5 at import,
    # and a child that honors it crashes mixing bindings.
    environment = dict(_os.environ, PYQTGRAPH_QT_LIB='PySide6',
                       QT_API='pyside6')
    launcher = ('import sys, visualdynamics; '
                'visualdynamics.launch_gui(*sys.argv[1:], in_process=True)')
    command = [sys.executable, '-c', launcher,
               *[str(path) for path in paths]]
    print('visualdynamics: pyqtgraph in this session is bound to another Qt '
          '(sdynpy does that) — starting the app in its own process.')
    return (spawn or subprocess.Popen)(command, env=environment)

__all__ = [
    'DEFAULT_SYSTEM',
    'FT_LBF_S',
    'IN_LBF_S',
    'MMKS',
    'SI',
    'SYSTEMS',
    'ChannelTable',
    'DataArray',
    'Frf',
    'Geometry',
    'Issue',
    'Project',
    'Psd',
    'Report',
    'ShapeSet',
    'ShockSpecification',
    'Specification',
    'Spectrum',
    'Srs',
    'TimeHistory',
    'TransientSpecification',
    'UnitSystem',
    'UnitsRequired',
    '__version__',
    'check_compatibility',
    'convert',
    'export_file',
    'fem',
    'frequency_axis',
    'from_sep005',
    'import_file',
    'importers',
    'launch_gui',
    'load',
    'random_vibration_report',
    'random_vibration_run',
    'register_importer',
    'save',
    'si_factor',
]
