"""File import and export.

Each importer and exporter is added to a registry here, so a new format
is added without changing existing code.

Every object that can be imported can also be saved, though not always
in the format it came from. `.vdyn` (HDF5) is the native format and the
one Save and Load use; it keeps everything, including units, and so
does `.mat`, which is the same layout in MATLAB's container. The other
formats are for exchanging data with other tools, and each loses
whatever it has no field for; see each module.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import (
    adf,
    delimited,
    escdf_objects,
    excel,
    exodus,
    femap,
    matlab,
    nastran,
    native,
    photo_files,
    punch,
    rattlesnake,
    rattlesnake_sine,
    rattlesnake_specification,
    report_template,
    sdynpy_data,
    sdynpy_npz,
    sdynpy_shapes,
    step,
    stl,
    threemf,
    unv,
)
from .exporters import export_file, exporters, register_exporter
from .native import TestContents, load, save, save_test
from .notes import ImportNote
from .sep005 import from_sep005


@dataclass(frozen=True)
class Importer:
    """One format Visual Dynamics can read.

    `sniff(path)` says whether a file is in this format, from its
    content where the content identifies the format, and `load(path)`
    returns the object, the dict of objects, or the whole Project the
    file holds. `project_type(path)`, for a format that records one,
    gives the kind of project the file is a run of.
    """

    name: str
    description: str
    sniff: Callable
    load: Callable
    #: what kind of project this file is a run of, if the format says.
    #: Most do not — a geometry is a geometry whatever it is for.
    project_type: Callable | None = None


_IMPORTERS: list[Importer] = []


def register_importer(name: str, description: str, sniff: Callable,
                      load: Callable,
                      project_type: Callable | None = None) -> None:
    """Add a format to the importers. Registered importers are tried in
    order, so one added later is tried last.

    Parameters
    ----------
    name : str
        The importer's name, which `import_file(format=)` takes.
    description : str
        What the format is, as a file dialog lists it.
    sniff : callable
        ``sniff(path) -> bool``: whether a file is this format.
    load : callable
        ``load(path, **kwargs)``: the file read into objects.
    project_type : callable, optional
        ``project_type(path) -> str or None``: the project type the file
        declares, for a format that carries one.

    Returns
    -------
    None
    """
    _IMPORTERS.append(Importer(name, description, sniff, load, project_type))


def importers() -> list[Importer]:
    """Every format Visual Dynamics can read, in the order they are
    tried."""
    return list(_IMPORTERS)


def project_type_of(path: str | os.PathLike) -> str | None:
    """The kind of project a file records itself as a run of, or None.

    A controller's own save records whether it was a modal test or a
    random vibration run, and this reads it, before or after the file
    is imported. Anything else (a geometry, a photo, a file no importer
    recognizes, or one that cannot be read) gives None rather than
    raising, because a file with no project type is the ordinary case,
    not an error.
    """
    path = str(path)
    for imp in _IMPORTERS:
        if imp.project_type is None:
            continue
        try:
            if imp.sniff(path):
                return imp.project_type(path)
        except (ValueError, OSError):
            return None
    return None


#: what a file no importer reads is called in a data index, by suffix:
#: the documents a test's record carries beside its data
_DOCUMENT_FORMATS = {
    '.csv': 'comma-separated text (.csv)',
    '.tsv': 'tab-separated text (.tsv)',
    '.txt': 'text (.txt)',
    '.md': 'Markdown text (.md)',
    '.json': 'JSON (.json)',
    '.jsonl': 'JSON Lines (.jsonl)',
    '.pdf': 'PDF document (.pdf)',
    '.docx': 'Word document (.docx)',
    '.png': 'PNG image (.png)',
    '.jpg': 'JPEG image (.jpg)',
    '.jpeg': 'JPEG image (.jpeg)',
    '.html': 'HTML document (.html)',
}


def format_of(path: str | os.PathLike) -> str:
    """The name of a file's format, as a data index prints it.

    A Visual Dynamics file is named as one. A document the record carries
    beside its data (a settings list, a log, a scanned procedure) is
    named by its kind, from its suffix. Any other file an importer
    recognizes is named by that importer's description ('Rattlesnake
    controller output (.nc4)'), and an unknown file by its suffix. A
    sniffer that trips over the file is not its format, so this does
    not raise for a readable path.
    """
    path = str(path)
    suffix = os.path.splitext(path)[1].lower()
    if suffix == '.vdyn':
        return 'Visual Dynamics file (.vdyn)'
    if suffix in _DOCUMENT_FORMATS:
        return _DOCUMENT_FORMATS[suffix]
    for imp in _IMPORTERS:
        try:
            if imp.sniff(path):
                return imp.description
        except Exception:  # noqa: BLE001, S112 — a sniffer that trips says no
            continue
    return f'{suffix[1:].upper()} file ({suffix})' if suffix else 'file'


def identify_runs(path: str, result: Any) -> Any:
    """Give each time history `result` holds the run identity its file's
    name says, and return `result`.

    The name is the one place a run's axis, number, level and start
    often are, and it is not kept after the import, so it is read here
    (`core.runs.read_name`), with a Rattlesnake run's environment names
    beside it. A name that says nothing gives nothing, and the history
    is left as it was. A field the names give two ways, or a level
    without a sign, is not read: an `ImportNote` names the file and the
    field, to be entered. A history that already carries an identity (a
    reader's own) keeps it.
    """
    import warnings

    from ..core.data import TimeHistory
    from ..core.runs import (
        FIELD_NAMES,
        FROM_ENVIRONMENT,
        FROM_NAME,
        RunIdentity,
        read_name,
    )

    objects = (list(result.values()) if isinstance(result, dict)
               else [result])
    histories = [obj for obj in objects if isinstance(obj, TimeHistory)
                 and getattr(obj, 'run_identity', None) is None]
    if not histories:
        return result
    base = os.path.basename(path)
    environments: list[str] = []
    if rattlesnake.sniff(path):
        try:
            environments = list(rattlesnake.environment_kinds(path))
        except Exception:  # noqa: BLE001 - names are a convenience
            environments = []
    from_name, refused = read_name(base, *environments)
    if not from_name and not refused:
        return result
    # where each field came from: the file's name when it alone gives
    # it, an environment's name otherwise
    own, _ = read_name(base)
    identity = RunIdentity(
        **from_name,
        sources=tuple((name, FROM_NAME if own.get(name) == value
                       else FROM_ENVIRONMENT)
                      for name, value in from_name.items()),
        refused=tuple(refused.items()))
    for name, why in refused.items():
        warnings.warn(ImportNote(
            f'{base}: {why}, so the {FIELD_NAMES[name]} is not read; enter '
            'it'), stacklevel=3)
    for history in histories:
        history.run_identity = identity
    return result


def import_file(path: str | os.PathLike, format: str | None = None,
                progress: Any | None = None, **kwargs: Any) -> Any:
    """Import a file, returning the Visual Dynamics object it contains.

    Units may be declared here (for example `length_unit='m'`) for
    sources that do not record them; without a declaration the object
    imports without units, holding the file's values as they are until
    `define_units()` is called. `format` chooses an importer by name
    instead of by recognizing the file; a `.vdyn` file is always read
    by the native loader. `progress` is a (done, total) callable, used
    where the reader can count (a project file's objects, a Rattlesnake
    run's samples) and ignored where it cannot: a UFF file is read in
    one parse with no partial progress to report. A leading `~` in the
    path is expanded.

    Raises FileNotFoundError when there is no file at `path`,
    IsADirectoryError for a folder, and ValueError for an unknown
    `format` or a file no importer recognizes (the message says what
    the file appears to be, where that can be told).

    Parameters
    ----------
    path : str or os.PathLike
        The file to read.
    format : str, optional
        An importer by name, rather than the one the file is recognized as.
    progress : callable, optional
        Called as ``progress(done, total)`` where the reader can count.
    **kwargs
        Passed to the importer: declared units (``length_unit='m'``), a
        window of a long run (``start=``, ``stop=``), channels, and so on.

    Returns
    -------
    object
        The object the file holds; a dict of them by name for a file that
        holds several, and a `Project` for a project file.
    """
    # a Rattlesnake run's reader reports its samples as progress
    # (2026-10-02); a reader without a progress parameter is not passed
    # one
    import inspect

    def load_with(imp):
        kwargs_for = dict(kwargs)
        if progress is not None and 'progress' in inspect.signature(
                imp.load).parameters:
            kwargs_for['progress'] = progress
        return imp.load(path, **kwargs_for)

    # a person types a path where the window hands one over: `~` is
    # theirs to write and Python's to expand, and a path that is not
    # there has to say so. Both came back as "No importer recognizes",
    # which reads as "your file is the wrong kind" and is how a typo
    # looked like an unsupported format (Brandon, 2026-09-20).
    path = os.path.expanduser(str(path))
    if not os.path.exists(path):
        raise FileNotFoundError(f'no file at {path}')
    if os.path.isdir(path):
        raise IsADirectoryError(f'{path} is a folder, not a file')
    if path.endswith('.vdyn'):
        return load(path, progress=progress)
    if format is not None:
        for imp in _IMPORTERS:
            if imp.name == format:
                return identify_runs(path, load_with(imp))
        raise ValueError(f"No importer named {format!r}; "
                         f"available: {[i.name for i in _IMPORTERS]}")
    for imp in _IMPORTERS:
        if imp.sniff(path):
            return identify_runs(path, load_with(imp))
    # a refusal that says what the file *is* saves a round trip: a
    # container whose contents decide the reader, and whose name does
    # not, is the case that brought this up (Brandon, 2026-09-20)
    from .sniffing import describe

    said = describe(path)
    raise ValueError(f"No importer recognizes {path}"
                     + (f" — {said}" if said else ''))


register_importer('sdynpy_geometry', 'sdynpy native geometry (.npz)',
                  sdynpy_npz.sniff, sdynpy_npz.load)
register_importer('exodus', 'Exodus finite element file (.exo/.e)',
                  exodus.sniff, exodus.load)
register_importer('nastran', 'Nastran bulk data (.bdf/.dat/.nas)',
                  nastran.sniff, nastran.load)
register_importer('punch', 'Nastran punch eigenvectors (.pch)',
                  punch.sniff, punch.load)
register_importer('femap', 'Femap Neutral file (.neu)',
                  femap.sniff, femap.load)
register_importer('unv', 'Universal file (.unv/.uff)',
                  unv.sniff, unv.load)
register_importer('adf', 'I-DEAS Associated Data File (.afu/.ati/.ash)',
                  adf.sniff, adf.load)
register_importer('sdynpy_data', 'sdynpy data array (.npz)',
                  sdynpy_data.sniff, sdynpy_data.load)
register_importer('report_template',
                  'Visual Dynamics report template (.vdreport)',
                  report_template.sniff, report_template.load)
register_importer('rattlesnake_specification',
                  'Rattlesnake random specification (.npz)',
                  rattlesnake_specification.sniff,
                  rattlesnake_specification.load)
register_importer('rattlesnake_sine',
                  'Rattlesnake sine specification (.npz)',
                  rattlesnake_sine.sniff, rattlesnake_sine.load)
register_importer('rattlesnake', 'Rattlesnake controller output (.nc4)',
                  rattlesnake.sniff, rattlesnake.load,
                  rattlesnake.project_type)
register_importer('sdynpy_shapes', 'sdynpy mode shapes (.npy)',
                  sdynpy_shapes.sniff, sdynpy_shapes.load)
register_importer('excel', 'Channel table spreadsheet (.xlsx)',
                  excel.sniff, excel.load)
register_importer('threemf', 'CAD mesh, parts as blocks (.3mf)',
                  threemf.sniff, threemf.load)
register_importer('step',
                  'STEP/IGES CAD, tessellated (.step/.stp/.iges/.igs)',
                  step.sniff, step.load)
register_importer('stl', 'CAD triangle mesh (.stl)',
                  stl.sniff, stl.load)
register_importer('matlab', 'MATLAB file, the project layout (.mat)',
                  matlab.sniff, matlab.load)
register_importer('escdf', 'Engineering Sciences Common Data Format (.h5)',
                  escdf_objects.sniff, escdf_objects.load)
# last on purpose: a text file another importer recognizes by its
# content is that importer's, and only numbers in columns are left here
register_importer('delimited',
                  'Delimited text time history (.csv/.tsv/.txt)',
                  delimited.sniff, delimited.load)

register_exporter('sdynpy_geometry', 'sdynpy native geometry (.npz)', '.npz',
                  sdynpy_npz.handles, sdynpy_npz.save)
register_exporter('sdynpy_data', 'sdynpy data array (.npz)', '.npz',
                  sdynpy_data.handles, sdynpy_data.save)
register_exporter('sdynpy_shapes', 'sdynpy mode shapes (.npy)', '.npy',
                  sdynpy_shapes.handles, sdynpy_shapes.save)
register_exporter('exodus', 'Exodus finite element file (.exo)', '.exo',
                  exodus.handles, exodus.save)
register_exporter('unv', 'Universal file (.unv)', '.unv',
                  unv.handles, unv.save)
register_exporter('threemf', 'CAD mesh, blocks as parts (.3mf)', '.3mf',
                  threemf.handles, threemf.save)
register_exporter('stl', 'CAD triangle mesh (.stl)', '.stl',
                  stl.handles, stl.save)
# the same format, the functions written as raw floats instead of as
# text: a third smaller and exact, where 13.5E loses the sixth digit.
# Second in the list on purpose — picking by suffix takes the ASCII
# form, which is the one every reader takes.
register_exporter('unv_binary', 'Universal file, binary 58b (.unv)', '.unv',
                  unv.handles_binary, unv.save_binary)
register_exporter('nastran', 'Nastran bulk data (.bdf)', '.bdf',
                  nastran.handles, nastran.save)
register_exporter('adf_functions', 'I-DEAS function ADF (.afu)', '.afu',
                  adf.handles_functions, adf.save)
register_exporter('adf_time', 'I-DEAS time history ADF (.ati)', '.ati',
                  adf.handles_time, adf.save)
register_exporter('adf_shapes', 'I-DEAS shape ADF (.ash)', '.ash',
                  adf.handles_shapes, adf.save)


def _is_channel_table(obj):
    from ..core.channel_table import ChannelTable
    return isinstance(obj, ChannelTable)


register_exporter('excel', 'Excel workbook (.xlsx)', '.xlsx',
                  _is_channel_table, excel.save)
# a folder rather than a file: the object is several pictures, and a
# picture format holds one
register_exporter('photos', 'Photographs (a folder of images)', '',
                  photo_files.handles, photo_files.save)
# rattlesnake .nc4 is deliberately read-only: it is the recording of a
# controller run, carrying hardware and environment settings visualdynamics never
# holds, so a file written from visualdynamics would be a controller run that never
# happened. The controller's *target* — the specification file its
# Random environment loads before a test — is another matter: a
# specification edited here goes back out as one
register_exporter('rattlesnake_specification',
                  'Rattlesnake random specification (.npz)', '.npz',
                  rattlesnake_specification.handles,
                  rattlesnake_specification.save)
# a report on its own is a template: the blocks and their bindings,
# no values, to be bound afresh in whatever project imports it
register_exporter('report_template', 'Report template (.vdreport)',
                  report_template.SUFFIX, report_template.handles,
                  report_template.save)
# the project file in MATLAB's container: every object and a whole
# project, with the units named. With no display system the values are
# SI, exactly as .vdyn holds them; exported in a display system they
# are in that system's units (matlab's module docstring)
register_exporter('matlab', 'MATLAB file (.mat)', matlab.SUFFIX,
                  matlab.handles, matlab.save)
register_exporter('escdf', 'Engineering Sciences Common Data Format (.h5)',
                  escdf_objects.SUFFIX, escdf_objects.handles, escdf_objects.save)
# a time history back out as the text it most often arrives as, read
# back by the delimited importer with its units from the header
register_exporter('delimited', 'Delimited text time history (.csv)', '.csv',
                  delimited.handles, delimited.save)

__all__ = [
    'ImportNote',
    'TestContents',
    'export_file',
    'exporters',
    'format_of',
    'from_sep005',
    'import_file',
    'importers',
    'load',
    'native',
    'project_type_of',
    'register_exporter',
    'register_importer',
    'save',
    'save_test',
]
