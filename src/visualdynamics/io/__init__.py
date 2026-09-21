"""File import and export.

Importers and exporters each register themselves in a small registry, so a
new format is added without touching existing code.

Anything visualdynamics can read, it can write. `.vdyn` (HDF5) is the native format
and the one Save and Load use — it is the only one that keeps everything,
including units. The foreign formats are for getting data to other tools,
and each loses whatever it has no way to record; see each module.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import (
    adf,
    excel,
    exodus,
    femap,
    matlab,
    nastran,
    native,
    photo_files,
    punch,
    rattlesnake,
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
from .sep005 import from_sep005


@dataclass(frozen=True)
class Importer:
    """One format visualdynamics can read.

    `sniff(path)` says whether this is that format — by content where
    the content says, never by the extension alone — and `load(path)`
    returns the object, the dict of objects, or the whole Project the
    file holds.
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
    """Teach visualdynamics a format. Registered ones are tried in order, so a
    reader added later is asked last."""
    _IMPORTERS.append(Importer(name, description, sniff, load, project_type))


def importers() -> list[Importer]:
    """Every format visualdynamics can read, in the order they are tried."""
    return list(_IMPORTERS)


def project_type_of(path: str | os.PathLike) -> str | None:
    """The kind of project a file says it is a run of, or None.

    A controller's own save knows whether it was a modal test or a
    random vibration run; asked before or after importing it, this is
    how it says so. Anything else — a geometry, a photo, a file no
    importer recognizes — answers None rather than raising: not knowing
    is the ordinary case, not a failure.
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


def import_file(path: str | os.PathLike, format: str | None = None,
                progress: Any | None = None, **kwargs: Any) -> Any:
    """Import a foreign file, returning the visualdynamics object it contains.

    Units may be declared here (e.g. length_unit='m') for sources that do not
    carry them; without a declaration the object imports unit-less, holding
    the file's raw values until `define_units()` is called.
    `format` forces a specific importer by name. `progress` is a
    (done, total) callable, honored where the reader can count — a
    project file's objects — and quietly unused where it cannot: a
    foreign file is one read, and nothing inside netCDF or UFF parsing
    reports fractions worth relaying.
    """
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
                return imp.load(path, **kwargs)
        raise ValueError(f"No importer named {format!r}; "
                         f"available: {[i.name for i in _IMPORTERS]}")
    for imp in _IMPORTERS:
        if imp.sniff(path):
            return imp.load(path, **kwargs)
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
# project, SI with the units named, exactly as .vdyn holds them
register_exporter('matlab', 'MATLAB file (.mat)', matlab.SUFFIX,
                  matlab.handles, matlab.save)

__all__ = ['TestContents', 'export_file', 'exporters', 'from_sep005',
           'import_file', 'importers', 'load', 'native', 'project_type_of',
           'register_exporter', 'register_importer', 'save', 'save_test']
