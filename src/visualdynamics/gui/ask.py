"""File dialogs for the one-call workflows.

The scripted entry points take paths. When one is left out, this is
where the question gets asked — a dialog rather than a refusal, so
`visualdynamics.random_vibration_report()` on its own is a working
thing to type (Brandon, 2026-09-22).

It lives in `gui/` because Qt does, and the workflows reach it by a
function-local import: `project.py` stays Qt-free, and a script that
passes every path never loads a widget toolkit at all.
"""

from __future__ import annotations

from typing import Any

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
    """Controller runs to work up, chosen by the user. Empty if none.

    Several at once on purpose: a test campaign is a folder of runs and
    the answer to each is its own report.
    """
    from PySide6.QtWidgets import QFileDialog

    _application()
    paths, _ = QFileDialog.getOpenFileNames(None, title, '', RUN_FILTER)
    return [str(p) for p in paths]


def for_geometry(title: str = 'Select a geometry (Cancel for none)'
                 ) -> str | None:
    """One geometry, or None when the user cancels.

    Cancel is an answer, not a failure: a report without a geometry is
    a report, and its sections stand down on their own.
    """
    from PySide6.QtWidgets import QFileDialog

    _application()
    path, _ = QFileDialog.getOpenFileName(None, title, '', GEOMETRY_FILTER)
    return str(path) or None
