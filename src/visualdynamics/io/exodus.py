"""Reader and writer for Exodus finite element files (.exo/.e).

Exodus is netCDF underneath; both directions go straight through
netCDF4. Read: nodes, element blocks, the file's coordinate frames
(as coordinate systems), the node sets that carry which node is placed
or measured in which frame, and results. Write (`save`): a geometry as
named blocks with its systems as frames and its assignments as node
sets, and shapes or records riding that geometry as nodal and global
variables — damping and modal mass as `Damping`/`ModalMass`, a complex
part as `<name>_IM` beside the real one — so a modal model round-trips
lossless (2026-09-12). Tracelines have no exodus form and go out as
beam elements. The format carries no units; `length_unit` may be
given to declare them at import, otherwise the geometry arrives
unit-less.

Beyond the mesh, exodus carries results: `time_whole` is a step axis,
nodal variables are (steps x nodes) records, global variables scalars
per step. What the step axis *means* the file cannot say — a modal run
stores one mode per step with frequency as time, a transient stores
time, and spectral conventions store frequency — so `load(steps=...)`
declares it rather than guessing, the same principle as declaring
units. See "Exodus beyond the mesh" in PLAN.md.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem

import itertools
import re

import numpy as np

from ..core.geometry import Geometry
from ..units import si_factor

# exodus elem_type attribute -> UFF 2412 descriptor code
_ELEM_TYPES = {
    'BAR': 21, 'BAR2': 21, 'BEAM': 21, 'BEAM2': 21, 'TRUSS': 21, 'TRUSS2': 21,
    'BAR3': 23, 'BEAM3': 23, 'TRUSS3': 23,
    'TRI': 91, 'TRI3': 91, 'TRIANGLE': 91, 'TRISHELL': 91, 'TRISHELL3': 91,
    'TRI6': 92, 'TRISHELL6': 92,
    'QUAD': 94, 'QUAD4': 94, 'SHELL': 94, 'SHELL4': 94,
    'QUAD8': 95, 'QUAD9': 95, 'SHELL8': 95, 'SHELL9': 95,
    'TET': 111, 'TET4': 111, 'TETRA': 111, 'TETRA4': 111,
    'TET10': 118, 'TETRA10': 118,
    'WEDGE': 112, 'WEDGE6': 112, 'WEDGE15': 113,
    'HEX': 115, 'HEX8': 115,
    'HEX20': 116, 'HEX27': 117,
    'PYRAMID': 201, 'PYRAMID5': 201, 'PYRAMID13': 202,
    'SPHERE': 161, 'SPRING': 136,
}

EXTENSIONS = ('.exo', '.e', '.exo2', '.g', '.gen')


def sniff(path: str | os.PathLike) -> bool:
    return str(path).lower().endswith(EXTENSIONS)


def _block_names(variables, count):
    """The `eb_names` strings, padded out to `count` and NUL-trimmed."""
    names = [''] * count
    if 'eb_names' not in variables:
        return names
    for i, row in enumerate(np.asarray(variables['eb_names'][()])[:count]):
        text = b''.join(bytes(c) for c in row if c not in (b'', b'\x00'))
        names[i] = text.decode('ascii', 'replace').strip('\x00').strip()
    return names


def _export_blocks(geometry, count):
    """(id, name) per element — what block each one goes out in."""
    block = np.asarray(getattr(geometry, 'elem_block', []), dtype=np.int64)
    ids = np.asarray(getattr(geometry, 'block_id', []), dtype=np.int64)
    names = list(getattr(geometry, 'block_name', []))
    named = {int(b): (int(b), names[i] if i < len(names) else '')
             for i, b in enumerate(ids)}
    return [named.get(int(block[i]), (0, '')) if i < len(block) else (0, '')
            for i in range(count)]


def _unique_block_labels(keys):
    """A distinct (id, name) for each exodus block being written.

    Exodus holds one element type per block, so a geometry block that
    mixes quads and tris has to go out as two of them — and a file whose
    blocks share an id, or share a name, is not readable: ParaView's IOSS
    reader fails the whole file at REQUEST_INFORMATION (no message, and
    the pipeline reports zero cells), and this package's own importer
    refuses duplicate ids outright. The drone went out as 34 blocks under
    22 ids and would load in nothing, including here.

    The first piece of a block keeps what the block declared, so an
    ordinary mesh goes out with the ids and names it came in with; the
    pieces after it take the lowest free id. A name already used gains
    the block's element type — 'canopy' and 'canopy TRI3' — and a blank
    name stays blank, since exodus names those blocks from their ids and
    inventing one would name a block nobody named. A split block is the
    one thing exodus cannot round-trip: it comes back as two.
    """
    declared = [int(key[0][0]) for key in keys]
    # every declared id is spoken for before any is handed out, so a
    # freshly allocated one cannot land on a number a later block will
    # claim — and the *first* piece of a split block keeps what the
    # block declared, so an ordinary mesh goes out with its own numbers
    reserved = {i for i in declared if i > 0}
    free = itertools.count(1)
    used_ids, used_names, labels = set(), set(), []
    for index, ((_id, declared_name), _code, type_name, _count) in \
            enumerate(keys):
        block_id = declared[index]
        if block_id <= 0 or block_id in used_ids:
            block_id = next(i for i in free
                            if i not in reserved and i not in used_ids)
        used_ids.add(block_id)
        name = str(declared_name)
        if name and name in used_names:
            name = f'{name} {type_name}'
            for suffix in itertools.count(2):
                if name not in used_names:
                    break
                name = f'{declared_name} {type_name} {suffix}'
        used_names.add(name)
        labels.append((block_id, name[:32]))
    return labels


def load(path: str | os.PathLike, length_unit: str | None = None,
         blocks: dict[int, str] | None = None, steps: str | None = None,
         nodes: Any | None = None) -> Any:
    """Read an exodus file: the mesh, and whatever results it carries.

    `steps` declares what the file's step axis means, because the file
    cannot say: `'modes'` (the default) reads displacement variables as
    a mode per step with frequency as the step's time; `'time'` reads
    every nodal and global variable as a `TimeHistory`; `'frequency'`
    reads them as a `Spectrum`, with `_RE`/`_IM` variable pairs merged
    into complex records. `nodes` limits the result records to those
    node ids — a large transient is records x steps for every node, and
    nobody wants all of a 200k-node mesh as channels.
    """
    import netCDF4

    if steps not in (None, 'modes', 'time', 'frequency'):
        raise ValueError(
            f"steps={steps!r}: choose 'modes' (one mode per step, the "
            "default), 'time' or 'frequency'")

    scale = 1.0 if length_unit is None else si_factor(length_unit, 'length')

    with netCDF4.Dataset(path) as ds:
        v = ds.variables
        if 'coord' in v:
            xyz = np.asarray(v['coord'][()]).T
        else:
            xyz = np.column_stack([np.asarray(v[c][()])
                                   for c in ('coordx', 'coordy', 'coordz') if c in v])
        num_nodes = xyz.shape[0]
        node_ids = (np.asarray(v['node_num_map'][()])
                    if 'node_num_map' in v
                    else np.arange(1, num_nodes + 1))

        block_ids = (np.asarray(v['eb_prop1'][()])
                     if 'eb_prop1' in v else [])
        # eb_status 0 is the format's own 'not really here' — the one
        # block an element-less file writes as a placeholder has no
        # connect variable to read
        status = (np.asarray(v['eb_status'][()]) if 'eb_status' in v
                  else np.ones(len(block_ids)))
        names = _block_names(v, len(block_ids))
        elem_type, elem_conn, elem_block = [], [], []
        kept_blocks, kept_names = [], []
        for i, block_id in enumerate(block_ids, start=1):
            if not status[i - 1]:
                continue
            if blocks is not None and block_id not in blocks:
                continue
            conn_var = v[f'connect{i}']
            type_name = str(conn_var.elem_type).upper()
            try:
                code = _ELEM_TYPES[type_name]
            except KeyError:
                raise ValueError(
                    f"Unsupported exodus element type {type_name!r} "
                    f"in block {block_id} of {path}")
            conn = np.asarray(conn_var[()], dtype=np.int64)  # 1-based local indices
            kept_blocks.append(int(block_id))
            kept_names.append(names[i - 1])
            for row in conn:
                elem_type.append(code)
                elem_conn.append(node_ids[row - 1])
                elem_block.append(int(block_id))

        elem_ids = (np.asarray(v['elem_num_map'][()])
                    if 'elem_num_map' in v and blocks is None
                    else np.arange(1, len(elem_conn) + 1))
        systems = _read_frames(v, scale, path, num_nodes)
        _read_assignments(v, systems)

    geometry = Geometry(
        node_id=node_ids,
        node_xyz=xyz * scale,
        elem_id=elem_ids,
        elem_type=elem_type,
        elem_conn=elem_conn,
        elem_block=elem_block or None,
        block_id=kept_blocks or None,
        block_name=kept_names or None,
        length_unit=length_unit,
        **systems,
    )
    if steps in ('time', 'frequency'):
        result = {'geometry': geometry}
        result.update(_read_results(path, node_ids, steps, nodes))
        return result
    shapes = _read_shapes(path, node_ids)
    if shapes is None:
        return geometry
    return {'geometry': geometry, 'shapes': shapes}


#: exodus frame tags against the geometry's `cs_type`
_FRAME_TAGS = {0: 'R', 1: 'C', 2: 'S'}
_FRAME_TYPES = {tag: code for code, tag in _FRAME_TAGS.items()}


def _read_frames(v, scale, path, num_nodes):
    """The file's coordinate frames as the geometry's coordinate systems,
    or nothing when it has none.

    Exodus defines *coordinate frames* — an id, three points (the
    origin, one on the local Z axis, one in the local XZ plane) and a
    tag for rectangular, cylindrical or spherical — and nothing in the
    format refers to them: nodal coordinates and nodal variables are
    always global Cartesian, and an analysis code names a frame from
    its own input for a load or a boundary condition. So the
    *definitions* travel here and the *assignment* cannot: every node
    reads back placed and measured in the global system. Frames have
    no names either.

    The nodes need a global system to refer to. A frame that is the
    identity at the origin serves; a file whose frames are all local
    gets one added, under id 1 when that is free and the next id
    otherwise.
    """
    if 'frame_ids' not in v:
        return {}
    ids = [int(i) for i in np.asarray(v['frame_ids'][()])]
    coords = np.asarray(v['frame_coordinates'][()],
                        dtype=np.float64).reshape(len(ids), 9)
    tags = [t.decode() if isinstance(t, bytes) else str(t)
            for t in np.asarray(v['frame_tags'][()]).ravel()]
    matrices, types = [], []
    for frame, row, tag in zip(ids, coords, tags):
        origin, on_z, in_xz = row[0:3], row[3:6], row[6:9]
        z = on_z - origin
        x = in_xz - origin
        if np.linalg.norm(z) == 0.0 or np.linalg.norm(x) == 0.0:
            raise ValueError(
                f'coordinate frame {frame} in {path} has a point on its '
                'origin, so its axes cannot be made')
        z = z / np.linalg.norm(z)
        x = x - np.dot(x, z) * z          # the XZ-plane point off the axis
        if np.linalg.norm(x) == 0.0:
            raise ValueError(
                f'coordinate frame {frame} in {path} puts its XZ-plane '
                'point on its Z axis, so its X axis cannot be made')
        x = x / np.linalg.norm(x)
        matrices.append(np.vstack([x, np.cross(z, x), z, origin * scale]))
        types.append(_FRAME_TYPES.get(tag.upper(), 0))
    matrices = np.asarray(matrices)
    identity = np.vstack([np.eye(3), np.zeros(3)])
    is_global = [np.allclose(m, identity) and t == 0
                 for m, t in zip(matrices, types)]
    if any(is_global):
        home = ids[is_global.index(True)]
    else:
        home = 1 if 1 not in ids else max(ids) + 1
        ids.insert(0, home)
        matrices = np.concatenate([identity[np.newaxis], matrices])
        types.insert(0, 0)
    placed = np.full(num_nodes, home, dtype=np.int64)
    return {'cs_id': ids, 'cs_name': [''] * len(ids), 'cs_type': types,
            'cs_matrix': matrices, 'node_def_cs': placed,
            'node_disp_cs': placed.copy()}


def _write_frames(ds, geometry, matrices):
    """The geometry's coordinate systems as exodus coordinate frames.

    Each system's origin row and its X and Z direction rows become the
    frame's three points, in the coordinates' own unit, and its type
    the tag. What is *not* written is which node is placed or measured
    in which system — see `_read_frames`; values at those nodes are
    rotated into the global frame on the way out (`_to_global`,
    `_rotate_records`).
    """
    count = len(geometry.cs_id)
    if not count:
        return
    ds.createDimension('num_cframes', count)
    ds.createDimension('num_cframes9', 9 * count)
    ids = ds.createVariable('frame_ids', 'i4', ('num_cframes',))
    ids[:] = np.asarray(geometry.cs_id, dtype=np.int32)
    points = np.empty((count, 9))
    for i, matrix in enumerate(matrices):
        origin = matrix[3]
        points[i, 0:3] = origin
        points[i, 3:6] = origin + matrix[2]       # on the local Z axis
        points[i, 6:9] = origin + matrix[0]       # in the local XZ plane
    ds.createVariable('frame_coordinates', 'f8',
                      ('num_cframes9',))[:] = points.ravel()
    tags = ds.createVariable('frame_tags', 'S1', ('num_cframes',))
    tags[:] = np.array([_FRAME_TAGS.get(int(t), 'R')
                        for t in geometry.cs_type], dtype='S1')


#: the node sets that carry which node is placed (`def`) or measured
#: (`disp`) in which local coordinate system — ours, since exodus has no
#: node-to-frame reference of its own; standard node sets, named so any
#: exodus tool shows them harmlessly and this reader knows them
_ASSIGNMENT_SET = re.compile(r'^(def|disp)_cs_(\d+)$')


def _read_assignments(v, systems):
    """Restore `node_def_cs` / `node_disp_cs` from the marker node sets,
    for the systems the file's frames defined."""
    if not systems or 'ns_names' not in v:
        return
    names = _char_names(v['ns_names'])
    known = set(systems['cs_id'])
    for i, name in enumerate(names, start=1):
        found = _ASSIGNMENT_SET.match(name)
        if found is None or int(found.group(2)) not in known \
                or f'node_ns{i}' not in v:
            continue
        local = np.asarray(v[f'node_ns{i}'][()], dtype=np.int64) - 1
        systems[f'node_{found.group(1)}_cs'][local] = int(found.group(2))


def _write_assignments(ds, geometry, row_of_node):
    """One node set per local system that any node is placed or measured
    in, named `def_cs_<id>` / `disp_cs_<id>`, holding those nodes as the
    format's 1-based indices. The global system needs none: a node not
    in any set reads back global, which is what it was.
    """
    identity = np.vstack([np.eye(3), np.zeros(3)])
    is_global = {int(cs): bool(np.allclose(m, identity) and t == 0)
                 for cs, m, t in zip(geometry.cs_id, geometry.cs_matrix,
                                     geometry.cs_type)}
    sets = []
    for kind, assigned in (('def', geometry.node_def_cs),
                           ('disp', geometry.node_disp_cs)):
        for cs in sorted({int(c) for c in assigned}):
            if is_global.get(cs, True):
                continue
            rows = [row_of_node[int(n)]
                    for n, c in zip(geometry.node_id, assigned) if int(c) == cs]
            sets.append((f'{kind}_cs_{cs}', rows))
    if not sets:
        return
    ds.createDimension('num_node_sets', len(sets))
    ids = ds.createVariable('ns_prop1', 'i4', ('num_node_sets',))
    ids.setncattr('name', 'ID')
    ids[:] = np.arange(1, len(sets) + 1, dtype=np.int32)
    ds.createVariable('ns_status', 'i4', ('num_node_sets',))[:] = 1
    names = ds.createVariable('ns_names', 'S1', ('num_node_sets', 'len_string'))
    for i, (name, rows) in enumerate(sets):
        names[i, :len(name)] = np.array(list(name), dtype='S1')
        ds.createDimension(f'num_nod_ns{i + 1}', len(rows))
        ds.createVariable(f'node_ns{i + 1}', 'i4',
                          (f'num_nod_ns{i + 1}',))[:] = np.asarray(rows, np.int32)


def _read_shapes(path, node_ids):
    """Mode shapes, if the file carries nodal results; else None.

    Exodus records results per time step. A modal run puts one mode in each
    step with its frequency as the step's time, which is what this reads;
    damping and modal mass come back from the global variables this
    package writes them as, and are zero and one when a file from
    elsewhere has none.
    """
    import netCDF4

    from ..core.shapes import ShapeSet

    with netCDF4.Dataset(path) as ds:
        if 'name_nod_var' not in ds.variables:
            return None
        names = [''.join(c.decode() if isinstance(c, bytes) else str(c)
                         for c in np.asarray(row) if c not in (b'', '', None)
                         ).strip('\x00 ')
                 for row in ds.variables['name_nod_var'][:]]
        wanted = [(i, _MODE_COMPONENTS[_MODE_VARIABLES.index(name)])
                  for i, name in enumerate(names) if name in _MODE_VARIABLES]
        if not wanted:
            return None
        frequency = np.asarray(ds.variables['time_whole'][:]) \
            if 'time_whole' in ds.variables else np.arange(1.0)
        columns = {component: np.asarray(ds.variables[f'vals_nod_var{i + 1}'][:])
                   for i, component in wanted}
        # a complex shape's imaginary part rides as <name>_IM beside the
        # real; put the two back together where both are present
        for i, name in enumerate(names):
            stem = name[:-3] if name.endswith('_IM') else None
            if stem in _MODE_VARIABLES:
                component = _MODE_COMPONENTS[_MODE_VARIABLES.index(stem)]
                if component in columns:
                    columns[component] = columns[component] + 1j * np.asarray(
                        ds.variables[f'vals_nod_var{i + 1}'][:])
        modes = len(frequency)
        damping, modal_mass = np.zeros(modes), np.ones(modes)
        if 'name_glo_var' in ds.variables and 'vals_glo_var' in ds.variables:
            labels = _char_names(ds.variables['name_glo_var'])
            values = np.asarray(ds.variables['vals_glo_var'][:],
                                dtype=np.float64)
            if _MODE_GLOBALS[0] in labels:
                damping = values[:modes, labels.index(_MODE_GLOBALS[0])]
            if _MODE_GLOBALS[1] in labels:
                modal_mass = values[:modes, labels.index(_MODE_GLOBALS[1])]

    order = [c for c in _MODE_COMPONENTS if c in columns]
    coordinate = [f'{int(node)}{component}'
                  for node in node_ids for component in order]
    shape_matrix = np.column_stack(
        [columns[component][:modes, i]
         for i in range(len(node_ids)) for component in order])
    return ShapeSet(
        frequency=frequency,
        damping=damping,
        coordinate=coordinate,
        shape_matrix=shape_matrix,
        modal_mass=modal_mass,
    )


# exodus stores modes the way it stores any nodal result: one "time step"
# per mode, with the frequency as the time value
_MODE_COMPONENTS = ('X+', 'Y+', 'Z+', 'RX+', 'RY+', 'RZ+')
_MODE_VARIABLES = ('DispX', 'DispY', 'DispZ', 'RotX', 'RotY', 'RotZ')
#: the global variables a mode's scalars ride as, one per step
_MODE_GLOBALS = ('Damping', 'ModalMass')


# ---- results beyond modes -----------------------------------------------

#: variable-name stem -> the quantity hint it carries. Exodus has no
#: units, so nothing is ever *scaled* by these — a hint narrows the
#: units offered later and labels the axis, exactly as a UNV 58 with no
#: 164 does. One table, matched exactly after the axis is stripped.
_STEM_HINTS = {
    'disp': 'length', 'displ': 'length', 'displacement': 'length',
    'vel': 'velocity', 'veloc': 'velocity', 'velocity': 'velocity',
    'acc': 'acceleration', 'accel': 'acceleration',
    'acceleration': 'acceleration',
    'force': 'force', 'press': 'pressure', 'pressure': 'pressure',
    'temp': 'temperature', 'temperature': 'temperature',
    'volt': 'voltage', 'voltage': 'voltage', 'strain': 'strain',
}

_AXIS_DIRECTIONS = {'x': 'X+', 'y': 'Y+', 'z': 'Z+',
                    'rx': 'RX+', 'ry': 'RY+', 'rz': 'RZ+'}

#: complex-part suffixes, recognized only behind a separator:
#: 'pressure' ends in 're', and a bare-suffix match would read it as
#: the real half of a complex pressure
_PART_SUFFIXES = {'re': 're', 'real': 're', 'im': 'im', 'imag': 'im'}


def _parse_variable(name):
    """(base, part, direction, hint): how a result variable's name reads.

    `part` is '' | 're' | 'im' — a separated complex suffix, stripped
    first so 'DispX_RE' pairs with 'DispX_IM' under one base.
    `direction` is the trailing axis as a DOF direction, '' when the
    name carries none (a scalar like 'pressure'). `hint` is the stem's
    quantity, None for a stem the table does not know — the record
    still imports, it just cannot help name its own axis.
    """
    text = str(name).strip()
    part = ''
    lowered = text.lower()
    for suffix, kind in _PART_SUFFIXES.items():
        if lowered.endswith('_' + suffix):
            part = kind
            text = text[:-(len(suffix) + 1)]
            lowered = text.lower()
            break
    direction = ''
    stem = text
    for axis in ('rx', 'ry', 'rz', 'x', 'y', 'z'):
        if lowered.endswith(axis) and len(lowered) > len(axis):
            head = text[:-len(axis)].rstrip('_-')
            # RotX names a rotation about x, not an x translation
            if axis in ('x', 'y', 'z') and head.lower() in ('rot',
                                                            'rotation'):
                direction = _AXIS_DIRECTIONS['r' + axis]
            else:
                direction = _AXIS_DIRECTIONS[axis]
            stem = head
            break
    hint = _STEM_HINTS.get(stem.lower().strip('_-'))
    return text, part, direction, hint


def _char_names(variable):
    """The strings a netCDF char-matrix variable holds, NUL-trimmed."""
    names = []
    for row in np.asarray(variable[()]):
        text = b''.join(bytes(c) for c in row if c not in (b'', b'\x00'))
        names.append(text.decode('ascii', 'replace').strip('\x00').strip())
    return names


def _merge_complex(columns):
    """[(base, values)] with _RE/_IM pairs merged into complex arrays.

    `columns` is [(base, part, values)]; a base seen with both parts
    becomes one complex column, either part alone stays real exactly as
    it arrived — half a pair is not evidence of the other half.
    """
    by_base, order = {}, []
    for base, part, values in columns:
        if base not in by_base:
            order.append(base)
        by_base.setdefault(base, {})[part or 'value'] = values
    merged = []
    for base in order:
        parts = by_base[base]
        if 're' in parts and 'im' in parts:
            merged.append((base, parts['re'] + 1j * parts['im']))
        else:
            for values in parts.values():
                merged.append((base, values))
    return merged


def _read_results(path, node_ids, steps, nodes=None):
    """Nodal and global variables as data records, the axis declared.

    Returns {'time': ...} or {'spectra': ...} for the nodal records and
    a second entry for the global ones — separate objects because a
    global variable has no node: its name is its whole identity,
    carried in `block` where a nodal record's identity is its DOF.
    """
    import netCDF4

    from ..core.data import Spectrum, TimeHistory
    from ..units import UNKNOWN

    cls = TimeHistory if steps == 'time' else Spectrum
    keys = ('time', 'global') if steps == 'time' else ('spectra',
                                                       'global spectra')
    with netCDF4.Dataset(path) as ds:
        v = ds.variables
        if 'time_whole' not in v:
            raise ValueError(
                f'{path} has no step axis (time_whole): nothing to read '
                f'as {steps} data')
        abscissa = np.asarray(v['time_whole'][:], dtype=np.float64)

        wanted = None
        if nodes is not None:
            asked = {int(n) for n in np.asarray(nodes).ravel()}
            wanted = np.asarray([int(n) in asked for n in node_ids])
            missing = asked - {int(n) for n in node_ids}
            if missing:
                raise ValueError(
                    f'{path} has no node {sorted(missing)}: nothing to '
                    'read there')
        kept_ids = node_ids if wanted is None else node_ids[wanted]

        result = {}
        if 'name_nod_var' in v:
            columns = []
            for i, name in enumerate(_char_names(v['name_nod_var'])):
                values = np.asarray(v[f'vals_nod_var{i + 1}'][()],
                                    dtype=np.float64)
                if wanted is not None:
                    values = values[:, wanted]
                base, part, direction, hint = _parse_variable(name)
                if steps == 'time':
                    part = ''      # a time axis has no complex halves
                    base = name    # so the suffix stays in the label
                columns.append((base, part, (values, direction, hint)))
            records, dofs, hints = [], [], []
            merged = _merge_complex(
                [(base, part, values)
                 for base, part, (values, _d, _h) in columns])
            readings = {base: (direction, hint)
                        for base, _p, (_v, direction, hint) in columns}
            for base, values in merged:
                direction, hint = readings[base]
                for k, node in enumerate(kept_ids):
                    row = values[:, k]
                    # a column of nothing is no record: the writer fills
                    # NaN where a node has no reading, and reading those
                    # back as channels would triple a round trip
                    if np.isnan(row).all():
                        continue
                    records.append(row)
                    dofs.append(f'{int(node)}{direction}')
                    hints.append(hint)
            if records:
                result[keys[0]] = cls(
                    abscissa=abscissa,
                    ordinate=np.asarray(records),
                    response_dof=dofs,
                    ordinate_dim=[UNKNOWN] * len(dofs),
                    dimension_hint=hints,
                )

        if 'name_glo_var' in v and 'vals_glo_var' in v:
            names = _char_names(v['name_glo_var'])
            values = np.asarray(v['vals_glo_var'][()], dtype=np.float64)
            columns = []
            for i, name in enumerate(names):
                base, part, _direction, hint = _parse_variable(name)
                if steps == 'time':
                    part, base = '', name
                columns.append((base, part, (values[:, i], hint)))
            merged = _merge_complex(
                [(base, part, values)
                 for base, part, (values, _h) in columns])
            hints = {base: hint for base, _p, (_v, hint) in columns}
            if merged:
                result[keys[1]] = cls(
                    abscissa=abscissa,
                    ordinate=np.asarray([values for _b, values in merged]),
                    response_dof=[''] * len(merged),
                    ordinate_dim=[UNKNOWN] * len(merged),
                    dimension_hint=[hints[base] for base, _v in merged],
                    block=[base for base, _v in merged],
                )
    if not result:
        raise ValueError(
            f'{path} carries no nodal or global variables: nothing to '
            f'read as {steps} data')
    return result


def handles(obj: Any) -> bool:
    from ..core.data import Spectrum, TimeHistory
    from ..core.shapes import ShapeSet

    return isinstance(obj, (Geometry, ShapeSet, TimeHistory, Spectrum))


# UFF 2412 descriptor code -> (exodus element type, nodes per element)
_EXODUS_TYPES = {
    11: ('BAR2', 2), 21: ('BAR2', 2), 22: ('BAR2', 2),
    23: ('BAR3', 3), 24: ('BAR3', 3),
    41: ('TRI3', 3), 91: ('TRISHELL3', 3),
    42: ('TRI6', 6), 92: ('TRISHELL6', 6),
    44: ('SHELL4', 4), 94: ('SHELL4', 4),
    45: ('SHELL8', 8), 95: ('SHELL8', 8),
    111: ('TETRA4', 4), 112: ('WEDGE6', 6), 113: ('WEDGE15', 15),
    115: ('HEX8', 8), 116: ('HEX20', 20), 117: ('HEX27', 27),
    118: ('TETRA10', 10),
    201: ('PYRAMID5', 5), 202: ('PYRAMID13', 13),
}


def save(obj: Any, path: str | os.PathLike, title: str = 'visualdynamics geometry',
         tracelines_as_beams: bool = True, geometry: Geometry | None = None,
         unit_system: UnitSystem | None = None) -> None:
    """Write a geometry, or mode shapes, as exodus."""
    from ..core.data import Spectrum, TimeHistory
    from ..core.shapes import ShapeSet

    if isinstance(obj, ShapeSet):
        if geometry is None:
            raise ValueError(
                'Exporting mode shapes to exodus needs a geometry: the '
                'file is a mesh with results, and shapes alone give '
                'ParaView no mesh to draw or animate')
        return _save_geometry(geometry, path, title=title,
                              tracelines_as_beams=tracelines_as_beams,
                              unit_system=unit_system, shapes=obj)
    if isinstance(obj, (TimeHistory, Spectrum)):
        if geometry is None:
            raise ValueError(
                'Exporting data to exodus needs a geometry: the file is '
                'a mesh with results, and records alone give it no mesh '
                'to sit on')
        return _save_geometry(geometry, path, title=title,
                              tracelines_as_beams=tracelines_as_beams,
                              unit_system=unit_system, data=obj)
    return _save_geometry(obj, path, title=title,
                          tracelines_as_beams=tracelines_as_beams,
                          unit_system=unit_system)


def _mode_node_order(shapes):
    """(node ids, the components each carries) from the shape's DOFs."""
    from ..core.data import parse_dof

    order, components = [], {}
    for i, dof in enumerate(shapes.coordinate):
        node, direction = parse_dof(dof)
        if node is None or direction.upper() not in _MODE_COMPONENTS:
            raise ValueError(f'{dof} is not a direction exodus can hold')
        if node not in components:
            order.append(node)
            components[node] = {}
        components[node][direction.upper()] = i
    present = [c for c in _MODE_COMPONENTS
               if any(c in components[n] for n in order)]
    return order, components, present


def _to_global(shapes, geometry, order, components, present, values=None):
    """Shape values rotated out of each node's displacement system.

    Exodus defines coordinate frames but nothing refers to them — a
    nodal variable is always global — so a value left in a node's local
    frame would be read back as though it were global. Translations
    are rotated as a triple, and rotations as another; a node whose
    displacement system is already the global one is untouched.

    Needs the geometry — the shapes alone do not say what frame they are
    in. Without it the values go out as they stand.
    """
    from ..deform import _cs_rotations

    values = np.array(shapes.shape_matrix if values is None else values)
    # complex stays complex: a real rotation of a complex vector rotates
    # both parts, and the writer decides what to do with the imaginary
    if not np.iscomplexobj(values):
        values = values.astype(np.float64)
    if geometry is None:
        return values
    rotations = _cs_rotations(geometry)
    if not rotations:
        return values                 # every system is already global

    row_of = {int(node): i for i, node in enumerate(geometry.node_id)}
    for node in order:
        row = row_of.get(int(node))
        if row is None:
            continue
        rotation = rotations.get(int(geometry.node_disp_cs[row]))
        if rotation is None:
            continue
        for triple in (('X+', 'Y+', 'Z+'), ('RX+', 'RY+', 'RZ+')):
            columns = [components[node].get(c) for c in triple]
            if any(c is None for c in columns):
                continue
            values[:, columns] = values[:, columns] @ rotation
    return values


def _save_geometry(geometry, path, title='visualdynamics geometry',
                   tracelines_as_beams=True, unit_system=None, shapes=None,
                   data=None):
    """Write a geometry as exodus — with mode shapes riding along.

    With `shapes`, each mode goes out as one time step of nodal results,
    its frequency as the step's time — how a modal run is stored, and what
    ParaView animates. Values are rotated into the global frame (an
    exodus nodal variable is global, whatever frames the file defines), a
    complex shape goes out as its real part under the plain name with the
    imaginary part beside it as <name>_IM,
    DOFs at nodes the mesh does not have are dropped, and damping and
    modal mass ride as two global variables, one number per step.

    Elements are grouped into one block per type, which is how exodus wants
    them — a block holds elements of a single shape.

    Exodus has no traceline, so by default each one is written as a run of
    two-node beam elements rather than dropped. That is not reversible:
    they read back as elements, because on the way in there is nothing to
    say a beam was ever a traceline. Pass `tracelines_as_beams=False` to
    leave them out instead.

    Coordinate systems go out as the file's coordinate frames, and which
    node is placed or measured in which as named node sets — the format
    has no node-to-frame reference of its own, so the sets are this
    package's convention on a standard feature (`_write_assignments`);
    a frame carries no name. Coordinates go out as stored, since the
    format records no units either.

    With `data` (a TimeHistory or Spectrum), records ride as results:
    the abscissa becomes `time_whole`, each (quantity, direction) a
    nodal variable named the way the reader reads them back (AccX,
    Pressure, DispX_RE/_IM for a complex spectrum), DOF-less records as
    global variables named by their block. A (node, variable) with no
    record is NaN, not zero — an absent reading is not a reading of
    nothing — and values at nodes with a local displacement system are
    rotated to the global frame, exactly as shapes are.
    Records at nodes the mesh does not have are dropped, like shape
    DOFs; a stacked object (averages) refuses, one record per channel
    being the only honest mapping onto one variable per node.
    """
    import netCDF4

    types = list(geometry.elem_type)
    connectivity = [list(conn) for conn in geometry.elem_conn]
    ids = [int(i) for i in geometry.elem_id]
    if tracelines_as_beams:
        next_id = (max(ids) if ids else 0) + 1
        for line in geometry.traceline_conn:
            nodes = [int(n) for n in line]
            for start, end in itertools.pairwise(nodes):
                types.append(21)               # beam2
                connectivity.append([start, end])
                ids.append(next_id)
                next_id += 1

    # Grouped by the geometry's own block first and by element type within
    # it, because exodus needs one type per block but the *block* is what
    # the file was told. Grouping by type alone is what turned a file with
    # 'wing' and 'tail' into one anonymous block of quads on the way out.
    labels = _export_blocks(geometry, len(connectivity))
    blocks = {}
    for index, (code, conn) in enumerate(zip(types, connectivity)):
        code = int(code)
        if code not in _EXODUS_TYPES:
            raise ValueError(
                f'No exodus element type for UFF descriptor {code}')
        name, count = _EXODUS_TYPES[code]
        if len(conn) != count:
            raise ValueError(
                f'{name} takes {count} nodes; an element has {len(conn)}')
        blocks.setdefault((labels[index], code, name, count), []).append(index)

    row_of = {int(node): i + 1 for i, node in enumerate(geometry.node_id)}

    with netCDF4.Dataset(path, 'w', format='NETCDF3_CLASSIC') as ds:
        ds.title = title
        ds.api_version = np.float32(4.98)
        ds.version = np.float32(4.98)
        ds.floating_point_word_size = 8
        ds.file_size = 1
        ds.createDimension('len_string', 33)
        ds.createDimension('len_line', 81)
        ds.createDimension('four', 4)
        ds.createDimension('time_step', None)
        ds.createDimension('num_dim', 3)
        ds.createDimension('num_nodes', geometry.num_nodes)
        # only when there is one: a zero-length dimension is how netCDF3
        # spells *unlimited*, and the file already has time_step — a
        # node-only geometry (data riding a sensor cloud) hit exactly
        # this as 'NC_UNLIMITED size already in use'
        if connectivity:
            ds.createDimension('num_elem', len(connectivity))
        ds.createDimension('num_el_blk', max(len(blocks), 1))

        from .exporters import geometry_values

        points, matrices = geometry_values(geometry, unit_system)
        points = np.asarray(points, dtype=np.float64)
        # file_size = 1 declares the large-model layout, where coordinates
        # are three separate variables. Writing the combined 'coord' array
        # under that declaration gave the exodus library nothing it would
        # read — and segfaulted ParaView on the empty result.
        for axis, name in enumerate(('coordx', 'coordy', 'coordz')):
            ds.createVariable(name, 'f8', ('num_nodes',))[:] = points[:, axis]
        node_map = ds.createVariable('node_num_map', 'i4', ('num_nodes',))
        node_map[:] = np.asarray(geometry.node_id, dtype=np.int32)
        _write_frames(ds, geometry, matrices)
        _write_assignments(ds, geometry, row_of)

        properties = ds.createVariable('eb_prop1', 'i4', ('num_el_blk',))
        properties.setncattr('name', 'ID')   # 'name' is reserved on the
                                             # netCDF4 object itself
        status = ds.createVariable('eb_status', 'i4', ('num_el_blk',))
        block_names = ds.createVariable(
            'eb_names', 'S1', ('num_el_blk', 'len_string'))
        written = _unique_block_labels(list(blocks))
        for i, ((_label, _code, name, count), rows) in enumerate(
                blocks.items(), start=1):
            ds.createDimension(f'num_el_in_blk{i}', len(rows))
            ds.createDimension(f'num_nod_per_el{i}', count)
            connect = ds.createVariable(
                f'connect{i}', 'i4',
                (f'num_el_in_blk{i}', f'num_nod_per_el{i}'))
            connect.elem_type = name
            connect[:] = np.array(
                [[row_of[int(n)] for n in connectivity[i]] for i in rows],
                dtype=np.int32)
            block_id, text = written[i - 1]
            properties[i - 1] = block_id
            status[i - 1] = 1
            if text:
                block_names[i - 1, :len(text)] = np.array(list(text),
                                                          dtype='S1')
        if not blocks:
            properties[0] = 1
            status[0] = 0

        if connectivity:
            elem_map = ds.createVariable('elem_num_map', 'i4',
                                         ('num_elem',))
            # blocks are written grouped by type, so the ids follow that
            # order rather than the order they came in
            elem_map[:] = np.asarray(
                [ids[i] for block in blocks.values() for i in block],
                dtype=np.int32)

        if shapes is not None:
            from .exporters import shape_values

            order, components, present = _mode_node_order(shapes)
            values = _to_global(shapes, geometry, order, components, present,
                                shape_values(shapes, unit_system))
            # the imaginary part rides beside the real as <name>_IM, the
            # pairing the spectrum export uses; DispX stays the real part
            # so ParaView still assembles its Disp vector. A real set
            # writes no _IM at all.
            parts = [('', np.real)]
            if np.iscomplexobj(values) and np.any(np.imag(values) != 0):
                parts.append(('_IM', np.imag))
            ds.createDimension('num_nod_var', len(present) * len(parts))
            # strictly increasing times: ParaView collapses duplicate time
            # values, so six rigid-body modes at exactly 0 Hz would read
            # back as one. A hair of separation keeps every mode.
            times = np.asarray(shapes.frequency, dtype=np.float64).copy()
            nudge = max(1.0, float(times.max())) * 1e-9
            for i in range(1, len(times)):
                if times[i] <= times[i - 1]:
                    times[i] = times[i - 1] + nudge
            ds.createVariable('time_whole', 'f8', ('time_step',))[:] = times
            names = ds.createVariable('name_nod_var', 'S1',
                                      ('num_nod_var', 'len_string'))
            node_row = {int(n): i for i, n in enumerate(geometry.node_id)}
            i = 0
            for suffix, part in parts:
                for component in present:
                    label = _MODE_VARIABLES[_MODE_COMPONENTS.index(component)]
                    label += suffix
                    names[i, :len(label)] = np.array(list(label), dtype='S1')
                    column = np.zeros((shapes.num_shapes, geometry.num_nodes))
                    for node in order:
                        row = node_row.get(int(node))
                        index = components[node].get(component)
                        if row is not None and index is not None:
                            column[:, row] = part(values[:, index])
                    ds.createVariable(f'vals_nod_var{i + 1}', 'f8',
                                      ('time_step', 'num_nodes'))[:] = column
                    i += 1
            # a mode's damping and modal mass: one number per step is
            # what a global variable is, so they ride as two — lost
            # until 2026-09-12. Modal mass in the target system's mass
            # unit, as the shape coefficients are.
            mass = np.asarray(shapes.modal_mass, dtype=np.float64)
            if unit_system is not None and shapes.units_defined:
                mass = unit_system.coherent.from_si(mass, 'mass')
            ds.createDimension('num_glo_var', 2)
            labels = ds.createVariable('name_glo_var', 'S1',
                                       ('num_glo_var', 'len_string'))
            for k, label in enumerate(_MODE_GLOBALS):
                labels[k, :len(label)] = np.array(list(label), dtype='S1')
            ds.createVariable('vals_glo_var', 'f8',
                              ('time_step', 'num_glo_var'))[:] = (
                np.column_stack([np.asarray(shapes.damping, np.float64),
                                 np.real(mass)]))

        if data is not None:
            _write_results(ds, data, geometry, unit_system)


#: dimension (or hint) -> the stem its export variable is named with.
#: The inverse of `_STEM_HINTS`' canonical spellings, so what is written
#: reads back as the same quantity. `Val` is the stem for a record whose
#: quantity nobody knows — it reads back hint-less, which is the truth.
_EXPORT_STEMS = {
    'length': 'Disp', 'velocity': 'Vel', 'acceleration': 'Acc',
    'force': 'Force', 'pressure': 'Pressure', 'temperature': 'Temp',
    'voltage': 'Volt', 'strain': 'Strain',
}


def _rotate_records(values, geometry, entries):
    """Record columns rotated out of their nodes' displacement systems.

    An exodus nodal variable is global, whatever frames the file defines,
    so a value left in a node's local frame would read back as global.
    `entries` is
    [(node, quantity, direction, column)]; a (node, quantity) whose
    records complete a translation or rotation triple is rotated as
    one, and anything short of a triple goes out as it stands — two
    components of a rotated triple cannot be rotated honestly, and
    refusing the whole export over them would be worse. The rotation
    matrix is real, so a complex spectrum's parts rotate together.
    """
    from ..deform import _cs_rotations

    rotations = _cs_rotations(geometry)
    if not rotations:
        return values
    row_of = {int(node): i for i, node in enumerate(geometry.node_id)}
    columns = {}
    for node, quantity, direction, column in entries:
        columns.setdefault((int(node), quantity), {})[direction] = column
    values = np.array(values)
    for (node, _quantity), directions in columns.items():
        row = row_of.get(node)
        if row is None:
            continue
        rotation = rotations.get(int(geometry.node_disp_cs[row]))
        if rotation is None:
            continue
        for triple in (('X+', 'Y+', 'Z+'), ('RX+', 'RY+', 'RZ+')):
            spots = [directions.get(d) for d in triple]
            if any(s is None for s in spots):
                continue
            values[:, spots] = values[:, spots] @ rotation
    return values


def _result_names(data):
    """[(record index, node or None, variable name)] for every record.

    The name is how the reader reads it back: canonical stem for the
    record's quantity plus its direction — `AccX`, `Pressure`, `Val`
    for a quantity nobody knows — and the block's own name for a
    DOF-less (global) record. Two records writing the same variable at
    the same node is refused with the collision named: a stacked
    object's averages all share a channel, and one variable per node
    cannot hold twenty readings.
    """
    from ..core.data import parse_dof

    named, seen = [], {}
    for i, dof in enumerate(data.response_dof):
        node, direction = parse_dof(dof)
        if node is None:
            label = ((data.block[i] if data.block is not None else '')
                     or (data.comment[i] or '').strip()
                     or f'Global{i + 1}')
            named.append((i, None, label[:32]))
            continue
        stem = _EXPORT_STEMS.get(data.known_dim(i), 'Val')
        suffix = direction.rstrip('+-').upper()
        name = f'{stem}{suffix}'
        clash = seen.get((int(node), name))
        if clash is not None:
            raise ValueError(
                f'records {clash} and {i} both write {name} at node '
                f'{node}: one record per (node, direction, quantity) — '
                'a stacked object exports one record per channel, so '
                'pick records first')
        seen[(int(node), name)] = i
        named.append((i, int(node), name))
    return named


def _write_results(ds, data, geometry, unit_system):
    """The data's records into an open exodus dataset, reader-shaped."""
    from .exporters import data_values

    abscissa, ordinate = data_values(data, unit_system)
    named = _result_names(data)
    row_of = {int(node): i for i, node in enumerate(geometry.node_id)}

    # rotation wants columns per record: (steps, records)
    entries = [(node, data.known_dim(i), _direction_of(data, i), i)
               for i, node, _name in named if node is not None
               and int(node) in row_of]
    values = _rotate_records(np.asarray(ordinate).T, geometry,
                             [(n, q, d, i) for n, q, d, i in entries if d])

    ds.createVariable('time_whole', 'f8', ('time_step',))[:] = (
        np.asarray(abscissa, dtype=np.float64))

    # nodal variables: every (name, part) column, NaN where a node has
    # no record — an absent reading is not a reading of nothing
    nodal = [(i, node, name) for i, node, name in named
             if node is not None and int(node) in row_of]
    variables = []
    for name in dict.fromkeys(name for _i, _n, name in nodal):
        rows = [(i, node) for i, node, n in nodal if n == name]
        complex_any = any(np.iscomplexobj(values[:, i])
                          or np.iscomplexobj(np.asarray(ordinate[i]))
                          for i, _node in rows)
        parts = (('_RE', np.real), ('_IM', np.imag)) if complex_any \
            else (('', np.real),)
        for suffix, take in parts:
            column = np.full((len(abscissa), geometry.num_nodes), np.nan)
            for i, node in rows:
                column[:, row_of[int(node)]] = take(values[:, i])
            variables.append((f'{name}{suffix}'[:32], column))
    if variables:
        ds.createDimension('num_nod_var', len(variables))
        names = ds.createVariable('name_nod_var', 'S1',
                                  ('num_nod_var', 'len_string'))
        for k, (label, column) in enumerate(variables):
            names[k, :len(label)] = np.array(list(label), dtype='S1')
            ds.createVariable(f'vals_nod_var{k + 1}', 'f8',
                              ('time_step', 'num_nodes'))[:] = column

    # global variables: one column each, complex as _RE/_IM like a
    # nodal record — same convention, same reader
    entries = [(i, name) for i, node, name in named if node is None]
    columns = []
    for i, name in entries:
        row = np.asarray(ordinate[i])
        if np.iscomplexobj(row):
            columns.append((f'{name}_RE'[:32], np.real(row)))
            columns.append((f'{name}_IM'[:32], np.imag(row)))
        else:
            columns.append((name, np.real(row)))
    if columns:
        ds.createDimension('num_glo_var', len(columns))
        names = ds.createVariable('name_glo_var', 'S1',
                                  ('num_glo_var', 'len_string'))
        for k, (label, _row) in enumerate(columns):
            names[k, :len(label)] = np.array(list(label), dtype='S1')
        ds.createVariable('vals_glo_var', 'f8',
                          ('time_step', 'num_glo_var'))[:] = (
            np.column_stack([row for _label, row in columns]))


def _direction_of(data, i):
    """The record's DOF direction, '' when it has none."""
    from ..core.data import parse_dof

    _node, direction = parse_dof(data.response_dof[i])
    return direction.upper() if direction else ''


def result_summary(path: str | os.PathLike) -> dict[str, Any]:
    """What the file's results are, cheaply — the import dialog's facts.

    {'nodal': [names], 'global': [names], 'steps': n, 'nodes': n,
    'node_sets': [(id, name, count)]} — everything empty or zero for a
    bare mesh, which is how a caller knows there is nothing to ask
    about. Names only; no values are read.
    """
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        v = ds.variables
        nodal = (_char_names(v['name_nod_var'])
                 if 'name_nod_var' in v else [])
        globals_ = (_char_names(v['name_glo_var'])
                    if 'name_glo_var' in v else [])
        steps = len(v['time_whole']) if 'time_whole' in v else 0
        nodes = (ds.dimensions['num_nodes'].size
                 if 'num_nodes' in ds.dimensions else 0)
        sets = []
        if 'ns_prop1' in v:
            ids = np.asarray(v['ns_prop1'][()])
            names = (_char_names(v['ns_names'])
                     if 'ns_names' in v else [''] * len(ids))
            for i, set_id in enumerate(ids, start=1):
                if i - 1 < len(names) and _ASSIGNMENT_SET.match(names[i - 1]):
                    continue        # a coordinate-system marker, not nodes
                count = (ds.dimensions[f'num_nod_ns{i}'].size
                         if f'num_nod_ns{i}' in ds.dimensions else 0)
                sets.append((int(set_id),
                             names[i - 1] if i - 1 < len(names) else '',
                             count))
    return {'nodal': nodal, 'global': globals_, 'steps': steps,
            'nodes': nodes, 'node_sets': sets}


def node_set_nodes(path: str | os.PathLike, set_id: int) -> np.ndarray:
    """The node ids one node set holds — what `load(nodes=...)` wants.

    Exodus stores set members as 1-based local indices; they come back
    mapped through `node_num_map` into the ids everything else speaks.
    """
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        v = ds.variables
        ids = np.asarray(v['ns_prop1'][()]) if 'ns_prop1' in v else []
        for i, found in enumerate(ids, start=1):
            if int(found) == int(set_id):
                local = np.asarray(v[f'node_ns{i}'][()], dtype=np.int64)
                node_ids = (np.asarray(v['node_num_map'][()])
                            if 'node_num_map' in v
                            else np.arange(1, ds.dimensions[
                                'num_nodes'].size + 1))
                return np.asarray(node_ids)[local - 1]
    raise ValueError(f'{path} has no node set {set_id}')
