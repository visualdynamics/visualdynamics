"""Read Femap neutral files (.neu): geometry and normal modes. Neutral
files are not written.

The neutral file is Femap's text interchange format: blocks bracketed
by `-1` lines, each opening with its numeric ID. The layouts here were
checked against real files, a Femap 2020 model export (neutral version
11.1) and a results file written by a translator (version 5.0).

What is read:

- 403, nodes: one line per node with the id, definition and output
  coordinate systems, layer, color, six permanent constraints, x, y, z
  and type. Coordinates are stored in the definition system, so a node
  with a nonzero definition system would need block 405 (coordinate
  systems), whose record layout no sample file has shown. Such a node
  is refused by name instead of being placed in the wrong position.
  Scalar nodes (type 1) are skipped, as everywhere else in this
  package.
- 404, elements: a fixed seven-line record in the version checked, a
  header (14 fields, topology fifth), nodes 1-10, nodes 11-20,
  orientation, two offset lines and a trailing list line. Every line's
  field count is checked, and a mismatch is refused with the neutral
  version named, because a version that changed the record would
  otherwise be misread without an error. Rigid, MultiList, Contact and
  Weld elements add list records that break the fixed record length,
  so they are refused by name; skipping them is not safe when the
  record length changes.
- 450, output sets, with 451 or 1051, output data vectors: only
  normal-modes sets (analysis type 2) become shapes. The set's value is
  the frequency in Hz, and the nodal translation vectors (2, 3, 4) and
  rotation vectors (6, 7, 8) become the ShapeSet's columns. Other
  analyses' displacement vectors and all stress and strain vectors are
  skipped, since this path is for correlating mode shapes. 451 lists
  one `id, value` pair per line; 1051 packs runs of `first, last,
  values...`; both end at a `-1`. The ShapeSet is marked unscaled,
  because the neutral file records min/max statistics but not the
  normalization, and its damping is zero.

A file with both returns a dict keyed 'geometry' and 'shapes'; a file
with one returns it directly; a file with neither is refused. A neutral
file has no units, so `length_unit` declares the unit of the positions,
as for a bulk deck or a universal file without a dataset 164.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .sniffing import text_head

# Phase 3 of the FEM interchange work (PLAN.md). The layouts were checked
# against the two real files on 2026-08-21, written from those files and
# no other tool's source. Writing waits until someone needs it.

# Femap element topology -> visualdynamics element type code
_TOPOLOGY = {
    0: 21,    # Line2 -> beam2
    1: 23,    # Line3 -> beam3
    2: 91,    # Tri3 -> trishell3
    3: 92,    # Tri6 -> trishell6
    4: 94,    # Quad4 -> quadshell4
    5: 95,    # Quad8 -> quadshell8
    6: 111,   # Tetra4
    7: 112,   # Wedge6
    8: 115,   # Brick8
    9: 161,   # Point -> mass
    10: 118,  # Tetra10
    11: 113,  # Wedge15
    12: 116,  # Brick20
}
_STRIDE_BREAKERS = {13: 'Rigid', 15: 'MultiList', 16: 'Contact',
                    17: 'Weld'}

# how many values a modal displacement vector contributes, by Femap
# vector ID: 2/3/4 are T1/T2/T3, 6/7/8 are R1/R2/R3 (1 and 5 are the
# derived totals, recomputable, so they are ignored)
_VECTOR_COMPONENT = {2: 0, 3: 1, 4: 2, 6: 3, 7: 4, 8: 5}
_COMPONENTS = ('X+', 'Y+', 'Z+', 'RX+', 'RY+', 'RZ+')

_MODES = 2          # Femap analysis type for normal modes
_NODAL = 7          # entity type for nodal vectors


def sniff(path: str | os.PathLike) -> bool:
    path = str(path)
    if os.path.splitext(path)[1].lower() != '.neu':
        return False
    head = text_head(path, 2048)
    if head is None:
        return False
    # a Femap neutral opens with a -1 line then a numeric block ID;
    # Gambit also writes .neu but opens with words, not the bracket
    stripped = [line.strip() for line in head.splitlines()[:6]]
    return any(stripped[k] == '-1' and stripped[k + 1].isdigit()
               for k in range(len(stripped) - 1))


def _blocks(lines):
    """Yield (block_id, body_lines) for each -1 ... -1 bracket."""
    k = 0
    while k < len(lines):
        if lines[k].strip() == '-1' and k + 1 < len(lines) \
                and lines[k + 1].strip().isdigit():
            start = k + 2
            end = start
            while end < len(lines) and lines[end].strip() != '-1':
                end += 1
            yield int(lines[k + 1]), lines[start:end]
            k = end + 1
        else:
            k += 1


def _fields(line):
    return [tok.strip() for tok in line.rstrip().rstrip(',').split(',')]


def _parse_403(body, nodes, path):
    for line in body:
        f = _fields(line)
        if len(f) < 15:
            raise ValueError(
                f'{path}: a node record with {len(f)} fields — this '
                'neutral vintage is not the one pinned here')
        node_type = int(f[14])
        if node_type != 0:
            continue        # scalar and extra nodes carry no position
        if int(f[1]) != 0:
            raise ValueError(
                f'{path}: node {f[0]} is defined in coordinate system '
                f'{f[1]}, and block 405 has never been seen with '
                'content here — its coordinates cannot be resolved. '
                'Export with nodes in the basic system, or bring the '
                'file so 405 can be read.')
        nodes.append((int(f[0]), [float(f[11]), float(f[12]),
                                  float(f[13])]))


def _record_line(body, k, count, what, path, version):
    """One line of a 404 record, its field count enforced."""
    if k >= len(body):
        raise ValueError(f'{path}: element block ends inside a record')
    f = _fields(body[k])
    if len(f) != count:
        raise ValueError(
            f'{path}: the {what} line of an element record has '
            f'{len(f)} fields where the pinned layout (neutral '
            f'version {version}) has {count} — refusing rather than '
            'misreading an unknown vintage')
    return f


def _parse_404(body, elements, path, version):
    from ..core.geometry import ELEMENT_TYPES

    k = 0
    while k < len(body):
        head = _record_line(body, k, 14, 'header', path, version)
        topology = int(head[4])
        if topology in _STRIDE_BREAKERS:
            raise ValueError(
                f'{path}: element {head[0]} is a Femap '
                f'{_STRIDE_BREAKERS[topology]} element, whose record '
                'appends list lines that break the fixed layout. Not '
                'supported; remove or convert those elements.')
        if topology not in _TOPOLOGY:
            raise ValueError(f'{path}: element {head[0]} has unknown '
                             f'topology {topology}')
        n1 = _record_line(body, k + 1, 10, 'first node', path, version)
        n2 = _record_line(body, k + 2, 10, 'second node', path, version)
        _record_line(body, k + 3, 9, 'orientation', path, version)
        _record_line(body, k + 4, 3, 'first offset', path, version)
        _record_line(body, k + 5, 3, 'second offset', path, version)
        _record_line(body, k + 6, 16, 'trailing list', path, version)
        k += 7

        code = _TOPOLOGY[topology]
        n_nodes = ELEMENT_TYPES[code][1]
        conn = [int(tok) for tok in (n1 + n2)[:n_nodes]]
        elements.append((int(head[0]), int(head[1]), code, conn))


def _parse_450(body, sets):
    """One output set per 450 block: id, title, program and analysis
    type, the set value (frequency for modes), then notes."""
    set_id = int(_fields(body[0])[0])
    title = body[1].strip()
    kind = _fields(body[2])
    value = float(_fields(body[3])[0])
    sets[set_id] = {'title': title, 'analysis': int(kind[1]),
                    'value': value, 'vectors': {}}


def _parse_451(body, sets, path):
    """Vectors in the one-pair-per-line encoding, several per block."""
    k = 0
    while k < len(body):
        set_id, vec_id = (int(tok) for tok in _fields(body[k])[:2])
        ent_type = int(_fields(body[k + 5])[3])
        k += 7                        # header through the 3-int line
        values = {}
        while True:
            f = _fields(body[k])
            k += 1
            ident = int(f[0])
            if ident == -1:
                break
            values[ident] = float(f[1])
        _keep_vector(sets, set_id, vec_id, ent_type, values, path)


def _parse_1051(body, sets, path):
    """Vectors in the packed-run encoding: `first, last, values...`
    per contiguous id run, flowing across lines, until a -1 run
    header. A run short of its promised values refuses rather than
    borrowing the next run's numbers."""
    k = 0
    while k < len(body):
        set_id, vec_id = (int(tok) for tok in _fields(body[k])[:2])
        ent_type = int(_fields(body[k + 6])[3])
        k += 8                        # one line longer than 451's header
        values: dict[int, float] = {}
        tokens: list[str] = []
        while True:
            while len(tokens) < 2:
                if k >= len(body):
                    raise ValueError(f'{path}: a 1051 vector runs '
                                     'past the end of its block')
                tokens += _fields(body[k])
                k += 1
            first = int(float(tokens[0]))
            if first == -1:
                break
            last = int(float(tokens[1]))
            need = last - first + 1
            if need < 1:
                raise ValueError(
                    f'{path}: a 1051 run claims ids {first}..{last}')
            while len(tokens) < 2 + need:
                if k >= len(body):
                    raise ValueError(
                        f'{path}: a 1051 run promises {need} values '
                        'and the block ends first')
                tokens += _fields(body[k])
                k += 1
            for i in range(need):
                values[first + i] = float(tokens[2 + i])
            tokens = tokens[2 + need:]
        _keep_vector(sets, set_id, vec_id, ent_type, values, path)


def _keep_vector(sets, set_id, vec_id, ent_type, values, path):
    if set_id not in sets:
        raise ValueError(f'{path}: output vectors for set {set_id}, '
                         'which no 450 block declared')
    entry = sets[set_id]
    if entry['analysis'] != _MODES or ent_type != _NODAL \
            or vec_id not in _VECTOR_COMPONENT:
        return                        # skipped knowingly — see module doc
    entry['vectors'][vec_id] = dict(values)


def _build_shapes(sets, path):
    from ..core.shapes import ShapeSet

    modal = [(sid, s) for sid, s in sorted(sets.items())
             if s['analysis'] == _MODES and s['vectors']]
    if not modal:
        return None
    first = modal[0][1]['vectors']
    nodes = sorted(next(iter(first.values())))
    columns = sorted({_VECTOR_COMPONENT[v]
                      for _, s in modal for v in s['vectors']})
    for sid, s in modal:
        for vec_id, values in s['vectors'].items():
            if sorted(values) != nodes:
                raise ValueError(
                    f'{path}: output set {sid} vector {vec_id} covers '
                    'different nodes than the first vector; a '
                    'ShapeSet shares one set of coordinates')
    coordinate = [f'{node}{_COMPONENTS[c]}'
                  for node in nodes for c in columns]
    matrix = np.zeros((len(modal), len(nodes) * len(columns)))
    for row, (_, s) in enumerate(modal):
        for vec_id, values in s['vectors'].items():
            offset = columns.index(_VECTOR_COMPONENT[vec_id])
            for j, node in enumerate(nodes):
                matrix[row, j * len(columns) + offset] = values[node]
    return ShapeSet(
        frequency=[s['value'] for _, s in modal],
        damping=[0.0] * len(modal),
        coordinate=coordinate, shape_matrix=matrix,
        description=[s['title'] for _, s in modal],
        comment=f'normal modes from {os.path.basename(path)}',
        unscaled=True)


def load(path: str | os.PathLike, length_unit: str | None = None,
         **_unused: Any) -> Any:
    from ..core.geometry import Geometry

    path = str(path)
    with open(path, errors='replace') as f:
        lines = f.read().splitlines()

    version = 'unknown'
    nodes: list[tuple[int, list[float]]] = []
    elements: list[tuple[int, int, int, list[int]]] = []
    sets: dict[int, dict[str, Any]] = {}

    for group_id, body in _blocks(lines):
        if not body:
            continue
        if group_id == 100:
            version = _fields(body[-1])[0]
        elif group_id == 403:
            _parse_403(body, nodes, path)
        elif group_id == 404:
            _parse_404(body, elements, path, version)
        elif group_id == 450:
            _parse_450(body, sets)
        elif group_id == 451:
            _parse_451(body, sets, path)
        elif group_id == 1051:
            _parse_1051(body, sets, path)

    out: dict[str, Any] = {}
    shapes = _build_shapes(sets, path)
    if shapes is not None:
        out['shapes'] = shapes
    if nodes:
        geometry = Geometry(
            node_id=[n for n, _ in nodes],
            node_xyz=[xyz for _, xyz in nodes],
            elem_id=[e[0] for e in elements],
            elem_type=[e[2] for e in elements],
            elem_color=[e[1] for e in elements],
            elem_conn=[np.array(e[3]) for e in elements])
        if length_unit:
            geometry.define_units(length_unit)
        out['geometry'] = geometry
    if not out:
        raise ValueError(f'{path}: no nodes and no normal-modes '
                         'output in this neutral file')
    return next(iter(out.values())) if len(out) == 1 else out
