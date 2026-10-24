"""Save a report on its own, as a template another project can use.

A report is a list of blocks with symbolic bindings, such as
`@basis:Frf` or `@other:ShapeSet`, so a report written without its
project is a template: the same figures and text, bound again to
whatever project loads it. The file is JSON with one marker key and the
fields the project file keeps for a report (title, marking, marking
color, blocks, and the report features when any are enabled; see
`native.save_report`), under its own suffix, `.vdreport`. A lab's house
template therefore carries its choice of report features. A block whose binding does not resolve in the new
project is an unbound card there, to be pointed at an object in the
editor's pane, as an unbound block of a built-in template is.

Templates saved in `templates_folder()` are offered by Generate Report
beside the built-in ones in every project.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.report import Report

# Brandon, 2026-09-08: "Is there a way for the user to save a report
# template or load a saved report template?"

SUFFIX = '.vdreport'
MARKER = 'visualdynamics_report_template'


def handles(obj: Any) -> bool:
    from ..core.report import Report

    return isinstance(obj, Report)


def sniff(path: str | os.PathLike) -> bool:
    path = str(path)
    if path.endswith(SUFFIX):
        return True
    if not path.endswith('.json'):
        return False
    try:
        with open(path, encoding='utf-8') as handle:
            return MARKER in handle.read(4096)
    except OSError:
        return False


def save(report: Report, path: str | os.PathLike, unit_system: Any = None,
         **_ignored: Any) -> None:
    """Write the report as a template (JSON).

    `unit_system` is accepted because every exporter takes it, and is not
    used: a template holds no values, only what to draw and how to bind
    it.
    """
    out = {MARKER: 1, 'title': report.title, 'marking': report.marking,
           'marking_color': report.marking_color,
           'blocks': [dict(block) for block in report.blocks]}
    if report.features:
        # only when there are any: a template with none is the file it
        # always was, and opens in a version that has no features
        out['features'] = list(report.features)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(out, handle, indent=1)
        handle.write('\n')


def load(path: str | os.PathLike, **_ignored: Any) -> Report:
    from ..core.report import Report

    with open(path, encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or MARKER not in data:
        raise ValueError(f'{path} is not a Visual Dynamics report template')
    features = data.get('features', [])
    if (not isinstance(features, list)
            or not all(isinstance(key, str) for key in features)):
        raise ValueError(f'{path}: features must be a list of feature '
                         f'keys, not {features!r}')
    return Report(title=str(data.get('title', 'Report')),
                  blocks=data.get('blocks', []),
                  marking=str(data.get('marking', 'UNCLASSIFIED')),
                  marking_color=str(data.get('marking_color', 'ink')),
                  features=features)


def templates_folder() -> Path:
    """The folder saved templates are kept in.

    It is `Visual Dynamics/Report Templates` in the user's application
    data folder for the platform (`~/Library/Application Support` on
    macOS, `APPDATA` on Windows, `XDG_DATA_HOME` or `~/.local/share`
    elsewhere), or the folder `VISUALDYNAMICS_TEMPLATES` names when that
    environment variable is set. The folder is not created here; whatever
    writes to it creates it.
    """
    override = os.environ.get('VISUALDYNAMICS_TEMPLATES')
    if override:
        return Path(override)
    if sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    elif sys.platform.startswith('win'):
        base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    else:
        base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
    return base / 'Visual Dynamics' / 'Report Templates'


def saved_templates() -> list[tuple[str, Path]]:
    """(name, path) for every `.vdreport` file in `templates_folder()`,
    sorted by name; empty when the folder does not exist."""
    folder = templates_folder()
    if not folder.is_dir():
        return []
    return sorted((path.name[:-len(SUFFIX)], path)
                  for path in folder.iterdir()
                  if path.is_file() and path.name.endswith(SUFFIX))
