"""File dialogs for the one-call workflows.

The scripted entry points take paths. When a path is left out, they ask
for it here with a file dialog instead of refusing, so
`visualdynamics.random_vibration_report()` with no arguments works.

This module is in `gui/` because it needs Qt. The workflows import it
inside the function that asks, so `project.py` stays free of Qt and a
script that passes every path never loads Qt at all.
"""

from __future__ import annotations

from typing import Any

# A dialog in place of a refusal for a missing path: Brandon, 2026-09-22.

#: what the import dialog offers for a controller's run
RUN_FILTER = ('Controller runs (*.nc4 *.nc);;All files (*)')

#: and for a geometry, the formats a geometry may arrive in
GEOMETRY_FILTER = (
    'Geometry files (*.npz *.unv *.uff *.uf *.exo *.e *.exo2 *.g *.gen '
    '*.bdf *.dat *.nas *.neu *.3mf *.stl *.step *.stp *.iges *.igs'
    ');;All files (*)')


def _application() -> Any:
    """The running QApplication, or one made for the occasion.

    A dialog with no application behind it aborts the process, and a
    machine with no display cannot show one at all — both say so here
    rather than in a stack trace from inside Qt.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:      # pragma: no cover - PySide6 is required
        raise RuntimeError(
            'asking for a file needs the desktop application installed; '
            'pass the path instead') from exc
    existing = QApplication.instance()
    if existing is not None:
        return existing
    try:
        return QApplication([])
    except Exception as exc:        # pragma: no cover - a headless machine
        raise RuntimeError(
            'no display to ask on; pass the path instead') from exc


def for_runs(title: str = 'Select the runs to report on') -> list[str]:
    """The controller runs the user picks in a file dialog, or an empty
    list when the dialog is canceled.

    The dialog allows several files at once, because a test campaign is
    a folder of runs and each run gets its own report.
    """
    from PySide6.QtWidgets import QFileDialog

    _application()
    paths, _ = QFileDialog.getOpenFileNames(None, title, '', RUN_FILTER)
    return [str(p) for p in paths]


def for_geometry(title: str = 'Select a geometry (Cancel for none)'
                 ) -> str | None:
    """The geometry file the user picks in a file dialog, or None when
    the dialog is canceled.

    Canceling is allowed: a report is still written without a geometry,
    and the sections that need one are left out.
    """
    from PySide6.QtWidgets import QFileDialog

    _application()
    path, _ = QFileDialog.getOpenFileName(None, title, '', GEOMETRY_FILTER)
    return str(path) or None
