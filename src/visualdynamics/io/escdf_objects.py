"""A project to and from an ESCDF file: the objects as the standard's
own types, and whole as Visual Dynamics keeps them.

The standard has a type for the things a test produces — `geometry`,
`channel_table`, `data` and `response_spectrum`, `mode` — and every one
of them is written here as that type, so any reader of the format gets
nodes, channels, curves and modes with nothing of ours in the way. It
has no field for much of what a project holds: element blocks and what
they are made of, coordinate systems, the processing record and the
provenance, reports, matched modes, sine and random specifications, a
geometry's view. Those ride in the `attachments` every group inherits
(Brandon, 2026-09-30: nothing that would make their reader fail), one
attachment per object, holding the object in its own `.vdyn` layout —
the one schema owner, `io.native`, run into memory — so a file written
here reads back whole, and a foreign reader sees an attachment it can
ignore. A report rides twice: rendered to its self-contained HTML, for
anyone with the file, and as its definition, so it reopens editable.

An activity is a link group (Brandon: *an Activity should perhaps be
equal to a linked group of objects*): the group's results are the
activity's data, and the geometry and channel tables it reads against
are metadata at the root, linked. A named group is an activity with
that name; an unnamed one takes its basis object's name. Objects linked
to nothing form one activity of their own. Photos, reports, matched
modes and sine specifications are metadata too — parameter sets, linked
to the activity whose group holds them — because the standard puts
anything that is not a result at the root.

Reading a file this program did not write builds each object from the
standard fields: a geometry's nodes, lines, elements and per-node
directions (a node whose axes are not the global ones gets a
coordinate system of its own), a data type's records and units, a mode
set's frequencies and shapes. The unit strings travel as the format
spells them; both spellings parse here.
"""

from __future__ import annotations

import datetime as dt
import getpass
import io
import json
import os
import re
import warnings
from typing import Any

import h5py
import numpy as np

from ..core.channel_table import ChannelTable
from ..core.data import (
    Coherence,
    DataArray,
    Frf,
    MultipleCoherence,
    Psd,
    Specification,
    Spectrum,
    Srs,
    TimeHistory,
    TransientSpecification,
)
from ..core.geometry import Geometry
from ..core.matches import MatchedModes
from ..core.photos import Photos
from ..core.report import Report
from ..core.shapes import ShapeSet
from ..core.sine import SineLevelSet, SineSweepSpecification
from ..project import Project
from . import escdf, native
from .notes import ImportNote

__all__ = ['SUFFIX', 'from_file', 'handles', 'load', 'save', 'sniff', 'to_file']

SUFFIX = '.escdf'
#: the attachment that holds an object whole, in its own layout
WHOLE = 'visualdynamics.vdyn'
#: the root metadata group carrying what belongs to the project itself
PROJECT_RECORD = 'visualdynamics_project'

#: the standard's data types, by the class written and read
_DATA_TYPES: dict[type, str] = {
    TimeHistory: 'time response',
    TransientSpecification: 'time response',
    Spectrum: 'spectrum',
    Frf: 'frequency response function',
    Psd: 'power spectral density',
    Specification: 'power spectral density',
    Coherence: 'coherence',
    MultipleCoherence: 'multiple coherence',
}
_CLASSES: dict[str, type] = {
    'time response': TimeHistory,
    'spectrum': Spectrum,
    'power spectrum': Spectrum,
    'frequency response function': Frf,
    'transmissibility': Frf,
    'power spectral density': Psd,
    'coherence': Coherence,
    'partial coherence': Coherence,
    'multiple coherence': MultipleCoherence,
    'response spectrum': Srs,
    'shock response spectrum': Srs,
}
#: the standard's element names by this program's descriptor, where a
#: descriptor has one; the rest stay in the whole-object attachment
_ELEMENT_NAMES: dict[int, str] = {
    **dict.fromkeys((11, 21, 22, 31, 121, 122, 137, 141, 151), 'bar2'),
    **dict.fromkeys((23, 24, 32), 'bar3'),
    **dict.fromkeys((41, 51, 61, 71, 91), 'tri3'),
    **dict.fromkeys((42, 52, 62, 72, 92), 'tri6'),
    **dict.fromkeys((44, 54, 64, 74, 94), 'quad4'),
    **dict.fromkeys((45, 55, 65, 75, 95), 'quad8'),
    111: 'tet4', 118: 'tet10', 115: 'hex8', 104: 'hex8', 116: 'hex20',
    105: 'hex20', 112: 'wedge6', 101: 'wedge6', 113: 'wedge15',
    102: 'wedge15',
    **dict.fromkeys((136, 138, 139, 142, 152, 161), 'sphere1'),
}
_ELEMENT_CODES: dict[str, int] = {
    'bar2': 21, 'bar3': 23, 'tri3': 41, 'tri6': 42, 'quad4': 44,
    'quad8': 45, 'tet4': 111, 'tet10': 118, 'hex8': 115, 'hex20': 116,
    'wedge6': 112, 'wedge15': 113, 'sphere1': 161,
}
#: the channel table's data types, from this program's channel types
_CHANNEL_TYPES = {
    'acceleration': 'acceleration', 'velocity': 'velocity',
    'displacement': 'displacement', 'force': 'force', 'strain': 'strain',
    'pressure': 'pressure', 'voltage': 'voltage', 'temperature': 'temperature',
}

_DOF = re.compile(r'^\d+(R?[XYZ]{1,2}[+-])?$')


def handles(obj: Any) -> bool:
    """A whole project, or any one object of it."""
    return isinstance(obj, (Project, Geometry, DataArray, ShapeSet, ChannelTable,
                            Report, Photos, MatchedModes, SineSweepSpecification,
                            SineLevelSet))


def sniff(path: str | os.PathLike) -> bool:
    return escdf.sniff(path)


# ---- writing ---------------------------------------------------------------

def _whole(obj: Any) -> np.ndarray:
    """The object in its own layout, as bytes for an attachment."""
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as f:
        native.save_into(obj, f)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def _unit(text: str | None) -> str:
    return '' if text is None else str(text)


def _spelled(unit: str) -> str:
    """A unit as the format's examples spell it: '^' for a power."""
    return unit.replace('**', '^')


def _wrapped(unit: str) -> str:
    """A unit safe to raise or multiply: compound ones in parentheses."""
    return f'({unit})' if any(c in unit for c in '/*^') else unit


def _psd_text(base: str, cross: str | None) -> str:
    """The file's unit of a spectral density from this program's: the
    base quantity squared per hertz, or two quantities' product per
    hertz for a cross term ('g^2/Hz', '(m/s^2)*(N)/Hz')."""
    base = _spelled(base)
    if cross and cross != base:
        return f'{_wrapped(base)}*{_wrapped(_spelled(cross))}/Hz'
    return f'{_wrapped(base)}^2/Hz'


def _psd_units(text: str) -> tuple[str, str | None]:
    """This program's (base, cross) units from a file's spectral density
    unit: 'g^2/Hz' or '(m/s**2)**2/Hz' give ('g', None) and
    ('m/s**2', None); 'g*N/Hz' or '(g)*(N)/Hz' give ('g', 'N'); a
    string in neither form is taken as the base itself, the way a file
    that stored 'g' for a PSD in g^2/Hz would mean it."""
    body = text.strip().replace('^', '**')
    for suffix in ('/Hz', '/hertz', '/ Hz'):
        if body.endswith(suffix):
            body = body[:-len(suffix)].strip()
            break
    else:
        return body, None

    def unwrap(part: str) -> str:
        part = part.strip()
        while part.startswith('(') and part.endswith(')') and _balanced(part[1:-1]):
            part = part[1:-1].strip()
        return part

    if body.endswith('**2'):
        return unwrap(body[:-3]), None
    depth, cut = 0, -1
    for i, ch in enumerate(body):
        depth += (ch == '(') - (ch == ')')
        if ch == '*' and depth == 0 and body[i:i + 2] != '**' and body[i - 1:i] != '*':
            cut = i
            break
    if cut > 0:
        return unwrap(body[:cut]), unwrap(body[cut + 1:])
    return unwrap(body), None


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += (ch == '(') - (ch == ')')
        if depth < 0:
            return False
    return depth == 0


def _colors(indices: Any) -> np.ndarray:
    from ..viz.geometry import color_rgb

    return np.array([[round(255 * c) for c in color_rgb(int(i))]
                     for i in np.asarray(indices).ravel()],
                    dtype=np.uint64).reshape(-1, 3)


def _geometry_dataset(name: str, geometry: Geometry) -> escdf.Dataset:
    nodes = [int(n) for n in geometry.node_id]
    values: dict[str, Any] = {
        'node_id': np.asarray(nodes, dtype=np.uint64),
        'node_position': np.asarray(geometry.node_xyz, dtype=np.float64).reshape(-1, 3),
        'position_units': geometry.length_unit or '',
    }
    for axis in 'xyz':
        values[f'node_{axis}_direction'] = np.array(
            [geometry.dof_direction(f'{n}{axis.upper()}+') for n in nodes],
            dtype=np.float64).reshape(-1, 3)
    if len(geometry.traceline_conn):
        values['line_connection'] = [np.asarray(c, dtype=np.uint64)
                                     for c in geometry.traceline_conn]
        values['line_color'] = _colors(geometry.traceline_color)
    rows = [i for i, code in enumerate(geometry.elem_type)
            if int(code) in _ELEMENT_NAMES]
    if rows:
        values['element_connection'] = [np.asarray(geometry.elem_conn[i], dtype=np.uint64)
                                        for i in rows]
        values['element_type'] = np.array([_ELEMENT_NAMES[int(geometry.elem_type[i])]
                                           for i in rows], dtype=object)
        values['element_color'] = _colors([geometry.elem_color[i] for i in rows])
    return escdf.Dataset(name, 'geometry', name, values)


def _channel_table_dataset(name: str, table: ChannelTable) -> escdf.Dataset:
    frame = table.frame
    count = len(frame)
    types = [_CHANNEL_TYPES.get(str(t).lower(), 'voltage')
             for t in frame['channel_type']]
    units = [str(u) for u in frame['unit']]
    values: dict[str, Any] = {
        'node_id': np.array([int(n) if str(n).strip() not in ('', 'nan') else 0
                             for n in frame['node']], dtype=np.uint64),
        'node_direction': np.array([str(d) for d in frame['direction']], dtype=object),
        'data_type': np.array(types, dtype=object),
        'make': np.array([str(v) for v in frame['make']], dtype=object),
        'model': np.array([str(v) for v in frame['model']], dtype=object),
        'serial_number': np.array([str(v) for v in frame['serial_number']], dtype=object),
        'sensitivity': np.array([float(v) if str(v).strip() not in ('', 'nan') else 0.0
                                 for v in frame['sensitivity']], dtype=np.float64),
        'sensitivity_unit': np.array([f'mV/{u}' if u else 'mV' for u in units],
                                     dtype=object),
        'daq': np.array([str(c) for c in frame['channel']], dtype=object),
    }
    comments = [str(v) for v in frame['comment']]
    if any(comments):
        values['description'] = np.array(comments, dtype=object)
    del count
    return escdf.Dataset(name, 'channel_table', name, values)


def _data_dataset(name: str, data: DataArray) -> escdf.Dataset:
    kind = 'response_spectrum' if isinstance(data, Srs) else 'data'
    references = data.reference_dof
    if references is not None:
        channel = np.array([[r, q] for r, q in zip(data.response_dof, references)],
                           dtype=object)
    else:
        channel = np.array([[r] for r in data.response_dof], dtype=object)
    ordinate_units = [_unit(u) for u in data.ordinate_unit]
    reference_units = [_unit(u) for u in (data.reference_unit or [None] * data.num_records)]
    if isinstance(data, Frf):
        # the standard's unit of a ratio is the ratio, spelled whole
        ordinate_units = [f'{_wrapped(_spelled(o))}/{_wrapped(_spelled(r))}' if r
                          else _spelled(o)
                          for o, r in zip(ordinate_units, reference_units)]
    elif isinstance(data, Psd):
        # a density's unit here is the base quantity's; the file's is
        # the square (or the cross product) per hertz
        ordinate_units = [_psd_text(o, r or None) if o else ''
                          for o, r in zip(ordinate_units, reference_units)]
    else:
        ordinate_units = [_spelled(o) for o in ordinate_units]
    values: dict[str, Any] = {
        'data_type': 'response spectrum' if isinstance(data, Srs)
        else _DATA_TYPES.get(type(data), 'spectrum'),
        'channel': channel,
        'ordinate': np.asarray(data.ordinate),
        'abscissa_unit': _unit(getattr(data, 'abscissa_unit', None)
                               or ('s' if data.abscissa_dim == 'time' else 'Hz')),
    }
    values['ordinate_unit'] = (ordinate_units[0] if len(set(ordinate_units)) == 1
                               else np.array(ordinate_units, dtype=object))
    abscissa = np.asarray(data.abscissa, dtype=np.float64)
    steps = np.diff(abscissa)
    if len(abscissa) > 1 and np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
        values['abscissa_start'] = float(abscissa[0])
        values['abscissa_step'] = float(steps[0])
    else:
        values['abscissa'] = abscissa
    if isinstance(data, Srs):
        values['model'] = 'absolute acceleration'
        values['percent_damping'] = float(100.0 / (2.0 * data.q)) if data.q else 5.0
        values['response_type'] = {'maximax': 'maximax'}.get(
            str(getattr(data, 'kind', 'maximax')), 'maximax')
        values['response_variable'] = 'acceleration'
    return escdf.Dataset(name, kind, name, values)


def _mode_dataset(name: str, shapes: ShapeSet) -> escdf.Dataset:
    values: dict[str, Any] = {
        'frequency': np.asarray(shapes.frequency, dtype=np.float64),
        'damping_ratio': np.asarray(shapes.damping, dtype=np.float64),
        'shape': np.asarray(shapes.shape_matrix).T,
        'dof_name': np.array([str(c) for c in shapes.coordinate], dtype=object),
    }
    if shapes.modal_mass is not None:
        values['modal_mass'] = np.asarray(shapes.modal_mass, dtype=np.float64)
    if shapes.mass_unit:
        values['modal_mass_unit'] = str(shapes.mass_unit)
    if shapes.comment:
        values['description'] = np.array([str(shapes.comment)], dtype=object)
    return escdf.Dataset(name, 'mode', name, values)


def _attach(dataset: escdf.Dataset, named: list[tuple[str, np.ndarray]],
            note: str) -> None:
    dataset.values['attachment_names'] = np.array([n for n, _ in named], dtype=object)
    dataset.values['attachments'] = [b for _, b in named]
    dataset.values['notes'] = np.array([note], dtype=object)


def _note(kind: str) -> str:
    from .. import __version__

    return (f'Written by Visual Dynamics {__version__}. The attachment '
            f'{WHOLE} holds this {kind} as Visual Dynamics keeps it, '
            'every field included; the properties beside it are the same '
            'object as the standard defines it.')


def to_file(project: Project, created_by: str | None = None,
            unit_system: Any = None) -> escdf.File:
    """A project as an ESCDF `File`: the objects as the standard's
    types, each whole in an attachment, an activity per link group.

    Parameters
    ----------
    project : Project
        What to write.
    created_by : str, optional
        The file's creator; the login name when left out.
    unit_system : UnitSystem, optional
        The system the reports are rendered in; SI when left out.

    Returns
    -------
    escdf.File
    """
    from .. import SI

    file = escdf.File(created_by=created_by or getpass.getuser(),
                      created_date=dt.datetime.now(dt.UTC))
    taken: set[str] = {PROJECT_RECORD}

    def identifier(name: str) -> str:
        base = escdf.valid_identifier(name)
        candidate, k = base, 2
        while candidate in taken:
            candidate, k = f'{base}_{k}', k + 1
        taken.add(candidate)
        return candidate

    names = {name: identifier(name) for name in project}
    metadata: dict[str, escdf.Dataset] = {}
    results: dict[str, escdf.Dataset] = {}
    for name, obj in project.items():
        key = names[name]
        if isinstance(obj, Geometry):
            dataset = _geometry_dataset(key, obj)
            _attach(dataset, [(WHOLE, _whole(obj))], _note('geometry'))
            metadata[key] = dataset
        elif isinstance(obj, ChannelTable):
            dataset = _channel_table_dataset(key, obj)
            _attach(dataset, [(WHOLE, _whole(obj))], _note('channel table'))
            metadata[key] = dataset
        elif isinstance(obj, DataArray):
            dataset = _data_dataset(key, obj)
            _attach(dataset, [(WHOLE, _whole(obj))], _note(type(obj).__name__))
            results[key] = dataset
        elif isinstance(obj, ShapeSet):
            dataset = _mode_dataset(key, obj)
            _attach(dataset, [(WHOLE, _whole(obj))], _note('shape set'))
            results[key] = dataset
        elif isinstance(obj, Report):
            from ..report import render_html

            html = render_html(obj, dict(project.items()), unit_system or SI,
                               links=project.links)
            dataset = escdf.Dataset(key, 'parameter_set', name)
            _attach(dataset, [(f'{key}.html', np.frombuffer(html.encode('utf-8'),
                                                            dtype=np.uint8)),
                              (WHOLE, _whole(obj))],
                    _note('report') + f' {key}.html is the report rendered, '
                    'complete in one page.')
            metadata[key] = dataset
        elif isinstance(obj, Photos):
            dataset = escdf.Dataset(key, 'parameter_set', name)
            _attach(dataset, [(f'{n}.{f}', np.frombuffer(bytes(b), dtype=np.uint8))
                              for n, f, b in zip(obj.names, obj.formats, obj.images)],
                    'Photographs, one attachment each, named as taken.')
            metadata[key] = dataset
        else:
            dataset = escdf.Dataset(key, 'parameter_set', name)
            _attach(dataset, [(WHOLE, _whole(obj))], _note(type(obj).__name__))
            metadata[key] = dataset
        dataset.descriptive_name = name
    # the project's own record: what the standard has no field for
    record = escdf.Dataset(PROJECT_RECORD, 'parameter_set', project.name)
    record.values['notes'] = np.array([(
        'Visual Dynamics project record: attachments hold the link groups, '
        'the provenance of derived objects and the project settings, as JSON.')],
        dtype=object)
    settings = {'name': project.name, 'project_type': project.project_type,
                'active_geometry': project.active_geometry,
                'names': names}
    record.values['attachment_names'] = np.array(
        ['links.json', 'provenance.json', 'project.json'], dtype=object)
    record.values['attachments'] = [
        np.frombuffer(json.dumps(project.links).encode('utf-8'), dtype=np.uint8),
        np.frombuffer(json.dumps(project.provenance).encode('utf-8'), dtype=np.uint8),
        np.frombuffer(json.dumps(settings).encode('utf-8'), dtype=np.uint8)]
    file.metadata[PROJECT_RECORD] = record
    file.metadata.update(metadata)
    # activities: one per link group, its results inside, its metadata linked
    placed: set[str] = set()
    activity_names: set[str] = set()
    for group in project.links:
        members = [m for m in group['members'] if m in names]
        if not members:
            continue
        basis = next((m for m in members if isinstance(project[m], Geometry)), members[0])
        label = group.get('name') or basis
        key = escdf.valid_identifier(label, 'activity_')
        k = 2
        while key in activity_names:
            key, k = f'{escdf.valid_identifier(label, "activity_")}_{k}', k + 1
        activity_names.add(key)
        activity = escdf.Activity(key, label, file.created_date)
        for member in members:
            dataset_name = names[member]
            if dataset_name in results:
                activity.data[dataset_name] = results[dataset_name]
            else:
                activity.links.append(dataset_name)
            placed.add(member)
        file.activities[key] = activity
    loose = [name for name in project if name not in placed]
    if loose:
        key = escdf.valid_identifier(project.name or 'project', 'activity_')
        while key in activity_names:
            key += '_'
        activity = escdf.Activity(key, project.name or 'Unlinked objects',
                                  file.created_date)
        for member in loose:
            dataset_name = names[member]
            if dataset_name in results:
                activity.data[dataset_name] = results[dataset_name]
            else:
                activity.links.append(dataset_name)
        file.activities[key] = activity
    return file


def save(obj: Any, path: str | os.PathLike, unit_system: Any = None,
         created_by: str | None = None, **_ignored: Any) -> None:
    """Write a project, or one object as a project of one, as an ESCDF
    file.

    Parameters
    ----------
    obj : Project or object
        What to write.
    path : path-like
        Where; `.escdf` is added when the suffix is not there.
    unit_system : UnitSystem, optional
        The system reports are rendered in.
    created_by : str, optional
        The creator recorded in the file.
    """
    path = str(path)
    if not path.endswith(SUFFIX):
        path += SUFFIX
    if not isinstance(obj, Project):
        project = Project(getattr(obj, 'name', '') or type(obj).__name__)
        project.add(type(obj).__name__, obj)
        obj = project
    escdf.write(to_file(obj, created_by, unit_system), path)


# ---- reading ---------------------------------------------------------------

def _attached(dataset: escdf.Dataset, name: str) -> bytes | None:
    names = dataset.values.get('attachment_names')
    blobs = dataset.values.get('attachments')
    if names is None or blobs is None:
        return None
    for label, blob in zip(np.asarray(names, dtype=object).ravel(), blobs):
        if str(label) == name:
            return bytes(np.asarray(blob, dtype=np.uint8))
    return None


def _whole_object(dataset: escdf.Dataset) -> Any | None:
    blob = _attached(dataset, WHOLE)
    if blob is None:
        return None
    try:
        with h5py.File(io.BytesIO(blob), 'r') as f:
            return native.load_from(f, path=f'{dataset.name}/{WHOLE}')
    except (OSError, ValueError, KeyError) as failure:
        warnings.warn(ImportNote(f'{dataset.name}: the {WHOLE} attachment could '
                                 f'not be read ({failure}); the object was built '
                                 'from the standard fields instead'), stacklevel=2)
        return None


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in np.asarray(value, dtype=object).ravel()]


def _geometry_from(dataset: escdf.Dataset) -> Geometry:
    v = dataset.values
    node_id = [int(n) for n in np.asarray(v['node_id']).ravel()]
    xyz = np.asarray(v['node_position'], dtype=np.float64).reshape(-1, 3)
    kwargs: dict[str, Any] = {}
    # per-node axes: a node whose frame is not the global one gets a
    # coordinate system of its own, and is displaced in it
    axes = np.stack([np.asarray(v[f'node_{a}_direction'], dtype=np.float64).reshape(-1, 3)
                     for a in 'xyz'], axis=1)              # (nodes, 3 axes, 3)
    differs = [i for i, frame in enumerate(axes) if not np.allclose(frame, np.eye(3))]
    if differs:
        cs_id, cs_type, cs_matrix, disp = [0], [0], [np.vstack([np.eye(3), np.zeros(3)])], []
        by_frame: dict[bytes, int] = {}
        for i, frame in enumerate(axes):
            if i not in differs:
                disp.append(0)
                continue
            key = np.round(frame, 9).tobytes()
            if key not in by_frame:
                by_frame[key] = len(cs_id)
                cs_id.append(len(cs_id))
                cs_type.append(0)
                cs_matrix.append(np.vstack([frame, np.zeros(3)]))
            disp.append(by_frame[key])
        kwargs.update(cs_id=cs_id, cs_type=cs_type, cs_matrix=np.array(cs_matrix),
                      cs_name=[''] * len(cs_id), node_disp_cs=disp,
                      node_def_cs=[0] * len(node_id))
    if v.get('line_connection') is not None:
        lines = [np.asarray(c, dtype=np.int64) for c in v['line_connection']]
        kwargs.update(traceline_id=list(range(1, len(lines) + 1)),
                      traceline_conn=lines,
                      traceline_color=_indices(v.get('line_color'), len(lines)))
    if v.get('element_connection') is not None:
        conn = [np.asarray(c, dtype=np.int64) for c in v['element_connection']]
        names = _strings(v.get('element_type')) or ['bar2'] * len(conn)
        kwargs.update(elem_id=list(range(1, len(conn) + 1)), elem_conn=conn,
                      elem_type=[_ELEMENT_CODES.get(n, 44) for n in names],
                      elem_color=_indices(v.get('element_color'), len(conn)),
                      elem_block=[1] * len(conn), block_id=[1],
                      block_name=[dataset.descriptive_name or dataset.name])
    if v.get('line_connection') is None and v.get('element_connection') is None:
        # said, because a geometry that arrives as bare nodes looks like
        # a reader that dropped its elements (Brandon, 2026-09-30)
        warnings.warn(ImportNote(f'{dataset.name}: the file\'s geometry carries '
                                 f'{len(node_id)} nodes and no line_connection or '
                                 'element_connection, so nothing joins them'),
                      stacklevel=2)
    geometry = Geometry(node_id, xyz, **kwargs)
    unit = str(v.get('position_units') or '').strip()
    if unit and unit.lower() not in ('unknown', 'none'):
        try:
            geometry.define_units(unit)
        except Exception as failure:  # noqa: BLE001 — a unit nobody spells
            warnings.warn(ImportNote(f'{dataset.name}: position_units {unit!r} was '
                                     f'not understood ({failure}); units left '
                                     'undefined'), stacklevel=2)
    return geometry


def _indices(colors: Any, count: int) -> list[int]:
    """The nearest of this program's color indices to each RGB triple."""
    if colors is None:
        return [1] * count
    from ..viz.geometry import color_rgb

    palette = np.array([color_rgb(i) for i in range(1, 17)]) * 255.0
    out = []
    for rgb in np.asarray(colors, dtype=np.float64).reshape(-1, 3):
        out.append(int(np.argmin(np.linalg.norm(palette - rgb, axis=1))) + 1)
    return out or [1] * count


def _split_ratio(unit: str) -> tuple[str, str | None]:
    """'(m/s^2)/(N)' or 'm/s^2/N' as (response unit, reference unit)
    when the last top-level division splits into two units the unit
    registry knows; otherwise the whole string and None."""
    from ..units import normalize_unit

    text = unit.strip().replace('^', '**')

    def unwrap(part: str) -> str:
        part = part.strip()
        while part.startswith('(') and part.endswith(')') and _balanced(part[1:-1]):
            part = part[1:-1].strip()
        return part

    depth, cut = 0, -1
    for i, ch in enumerate(text):
        depth += (ch == '(') - (ch == ')')
        if ch == '/' and depth == 0:
            cut = i
    if cut < 0:
        return unwrap(text), None
    left, right = unwrap(text[:cut]), unwrap(text[cut + 1:])
    try:
        normalize_unit(left)
        normalize_unit(right)
    except Exception:  # noqa: BLE001 — not two units, then
        return unwrap(text), None
    return left, right


def _data_from(dataset: escdf.Dataset) -> DataArray:
    v = dataset.values
    data_type = str(v.get('data_type', 'spectrum'))
    cls = _CLASSES.get(data_type, Spectrum)
    channel = np.asarray(v['channel'], dtype=object).reshape(len(np.asarray(v['ordinate'])), -1)
    response = [str(c) for c in channel[:, 0]]
    reference = [str(c) for c in channel[:, 1]] if channel.shape[1] > 1 else None
    ordinate = np.asarray(v['ordinate'])
    if 'abscissa' in v:
        abscissa = np.asarray(v['abscissa'], dtype=np.float64)
        if abscissa.ndim == 2:
            if not np.allclose(abscissa, abscissa[0]):
                warnings.warn(ImportNote(f'{dataset.name}: each row has its own '
                                         'abscissa; the first row\'s is used for '
                                         'all'), stacklevel=2)
            abscissa = abscissa[0]
    else:
        start, step = float(v['abscissa_start']), float(v['abscissa_step'])
        abscissa = start + step * np.arange(ordinate.shape[1])
    units = [u.replace('^', '**') for u in _strings(v.get('ordinate_unit'))]
    units = units * ordinate.shape[0] if len(units) == 1 else units
    reference_units: list[str | None] | None = None
    if cls is Frf:
        pairs = [_split_ratio(u) for u in units]
        units = [p[0] for p in pairs]
        reference_units = [p[1] for p in pairs]
    elif cls is Psd:
        pairs = [_psd_units(u) if u else ('', None) for u in units]
        units = [p[0] for p in pairs]
        reference_units = [p[1] for p in pairs]
    elif cls in (Coherence, MultipleCoherence):
        units = ['' for _ in units]              # dimensionless
    # the file's abscissa is in whatever it says; here time is seconds
    # and frequency hertz
    abscissa_unit = str(v.get('abscissa_unit') or '').strip()
    if abscissa_unit:
        from ..units import UnitError, si_transform

        dimension = 'time' if cls in (TimeHistory,) else 'frequency'
        try:
            scale, offset = si_transform(abscissa_unit, dimension)
            abscissa = abscissa * scale + offset
        except UnitError as failure:
            warnings.warn(ImportNote(f'{dataset.name}: abscissa_unit '
                                     f'{abscissa_unit!r} was not understood '
                                     f'({failure}); the abscissa is taken as '
                                     f'{"seconds" if dimension == "time" else "hertz"}'),
                          stacklevel=2)
    kwargs: dict[str, Any] = {}
    if cls is Srs:
        damping = float(v.get('percent_damping', 5.0))
        kwargs['q'] = 100.0 / (2.0 * damping) if damping else None
    if cls in (Frf, Coherence, MultipleCoherence, Psd) and reference is None:
        reference = list(response)
    if cls in (TimeHistory, Spectrum, Srs) and reference is not None \
            and all(r == q for r, q in zip(response, reference)):
        reference = None
    if reference is not None:
        kwargs['reference_dof'] = reference
    if cls is MultipleCoherence:
        kwargs.pop('reference_dof', None)
    try:
        data = cls(abscissa, ordinate, response, **kwargs)
    except (TypeError, ValueError) as failure:
        warnings.warn(ImportNote(f'{dataset.name}: read as a spectrum, not '
                                 f'{data_type!r} ({failure})'), stacklevel=2)
        data = Spectrum(abscissa, ordinate, response)
    # declared, not merely named: the values are in the file's units,
    # and declaring them is what converts them to SI and shows them
    # (a first cut named them and the app showed them undefined —
    # Brandon, 2026-09-30). A unit the registry does not know leaves
    # that record undefined, with a note.
    declared = [u or None for u in units]
    if any(declared):
        try:
            data.define_units(declared, reference_units if reference_units
                              and any(reference_units) else None)
        except Exception as failure:  # noqa: BLE001 — a unit nobody spells
            spelled = sorted({u for u in declared if u})
            warnings.warn(ImportNote(f'{dataset.name}: units {spelled} '
                                     f'were not understood ({failure}); left '
                                     'undefined'), stacklevel=2)
    return data


def _shapes_from(dataset: escdf.Dataset) -> ShapeSet:
    v = dataset.values
    frequency = np.asarray(v['frequency'], dtype=np.float64)
    damping = (np.asarray(v['damping_ratio'], dtype=np.float64)
               if v.get('damping_ratio') is not None else np.zeros_like(frequency))
    description = _strings(v.get('description'))
    return ShapeSet(frequency, damping, _strings(v['dof_name']),
                    np.asarray(v['shape']).T,
                    modal_mass=(np.asarray(v['modal_mass'], dtype=np.float64)
                                if v.get('modal_mass') is not None else None),
                    mass_unit=str(v['modal_mass_unit']) if v.get('modal_mass_unit') else None,
                    comment=' '.join(description) if description else None)


def _channel_table_from(dataset: escdf.Dataset) -> ChannelTable:
    v = dataset.values
    count = len(np.asarray(v['node_id']).ravel())
    types = {t: k for k, t in _CHANNEL_TYPES.items()}
    sensitivity_units = _strings(v.get('sensitivity_unit')) or [''] * count
    columns = {
        'channel': list(range(1, count + 1)),
        'node': [int(n) for n in np.asarray(v['node_id']).ravel()],
        'direction': _strings(v.get('node_direction')) or [''] * count,
        'channel_type': [types.get(t, '') for t in _strings(v.get('data_type'))]
        or [''] * count,
        'unit': [u.split('/', 1)[1] if '/' in u else '' for u in sensitivity_units],
        'sensitivity': [float(s) for s in np.asarray(v['sensitivity']).ravel()],
        'serial_number': _strings(v.get('serial_number')) or [''] * count,
        'make': _strings(v.get('make')) or [''] * count,
        'model': _strings(v.get('model')) or [''] * count,
        'comment': _strings(v.get('description')) or [''] * count,
    }
    daq = _strings(v.get('daq'))
    if daq and all(d.strip().isdigit() for d in daq):
        columns['channel'] = [int(d) for d in daq]
    return ChannelTable(columns)


def _photos_from(dataset: escdf.Dataset) -> Photos:
    names = _strings(dataset.values.get('attachment_names'))
    blobs = dataset.values.get('attachments') or []
    stems, formats, images = [], [], []
    for label, blob in zip(names, blobs):
        stem, _, suffix = label.rpartition('.')
        stems.append(stem or label)
        formats.append(suffix.lower() or 'png')
        images.append(bytes(np.asarray(blob, dtype=np.uint8)))
    return Photos(stems, formats, images)


def _object_from(dataset: escdf.Dataset) -> Any | None:
    whole = _whole_object(dataset)
    if whole is not None:
        return whole
    spec = dataset.specification
    if spec is None:
        warnings.warn(ImportNote(f'{dataset.name}: type {dataset.kind!r} is not one '
                                 'the specifications define; left out'), stacklevel=2)
        return None
    if spec.is_a('geometry'):
        return _geometry_from(dataset)
    if spec.is_a('channel_table'):
        return _channel_table_from(dataset)
    if spec.is_a('data'):
        return _data_from(dataset)
    if spec.is_a('mode'):
        return _shapes_from(dataset)
    names = _strings(dataset.values.get('attachment_names'))
    if names and all(n.lower().rpartition('.')[2] in ('png', 'jpg', 'jpeg', 'gif', 'bmp')
                     for n in names):
        return _photos_from(dataset)
    return None


def from_file(file: escdf.File) -> Project:
    """A project from an ESCDF `File`: every dataset an object, each
    activity a link group named for it, and the project's own record
    when the file carries one.

    Parameters
    ----------
    file : escdf.File
        As `escdf.read` returns it.

    Returns
    -------
    Project
    """
    record = file.metadata.get(PROJECT_RECORD)
    settings: dict[str, Any] = {}
    links_record: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    if record is not None:
        for label, target in (('project.json', settings), ('links.json', links_record),
                              ('provenance.json', provenance)):
            blob = _attached(record, label)
            if blob is not None:
                loaded = json.loads(blob.decode('utf-8'))
                (target.update if isinstance(target, dict) else target.extend)(loaded)
    display = {v: k for k, v in settings.get('names', {}).items()}
    objects: dict[str, Any] = {}
    kept: dict[str, str] = {}                 # dataset name -> object name
    for name, dataset in file.metadata.items():
        if name == PROJECT_RECORD:
            continue
        obj = _object_from(dataset)
        if obj is not None:
            label = display.get(name) or dataset.descriptive_name or name
            objects[label] = obj
            kept[name] = label
    for activity in file.activities.values():
        for name, dataset in activity.data.items():
            obj = _object_from(dataset)
            if obj is not None:
                label = display.get(name) or dataset.descriptive_name or name
                while label in objects:
                    label += ' (2)'
                objects[label] = obj
                kept[name] = label
    # in the order the project held them, when the file says; a
    # foreign file's order is its own
    order = [display[k] for k in settings.get('names', {}).values() if k in display]
    objects = {**{name: objects[name] for name in order if name in objects},
               **objects}
    project = Project(settings.get('name') or 'ESCDF import', objects,
                      settings.get('active_geometry'), settings.get('project_type'),
                      provenance=provenance)
    if links_record and all(m in objects for g in links_record for m in g['members']):
        project.links = Project('x', links=links_record).links
    else:
        for activity in file.activities.values():
            members = [kept[n] for n in list(activity.data) + list(activity.links)
                       if n in kept]
            if len(members) >= 2:
                project.link(*members, name=activity.descriptive_name or activity.name)
    return project


def load(path: str | os.PathLike, **_ignored: Any) -> Project:
    """Read an ESCDF file as a project.

    Parameters
    ----------
    path : path-like
        The file.

    Returns
    -------
    Project
    """
    return from_file(escdf.read(path))

