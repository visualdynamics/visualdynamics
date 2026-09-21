"""Nastran bulk data (.bdf/.dat/.nas) — geometry, read and written.

The bulk deck is the lingua franca of the FEM world this tool's
measurements get correlated against, and its card formats are public
knowledge documented in every Nastran vendor's reference guide and
decades of literature. This reader is written from those card layouts;
no other tool's source was consulted.

What is read is the *geometry*: GRID points with their definition and
displacement systems, CORD2R/C/S coordinate systems (chained
references resolved), the connection elements the vocabulary knows,
and PLOTEL — Nastran's own display-only line — as tracelines. What is
deliberately not read, and why:

- **Properties, materials, loads, constraints, control decks** — a
  bulk file describes an analysis; only its mesh is geometry. These
  are skipped by card name, silently, the way a UNV reader skips the
  datasets it was not asked about.
- **Rigid elements (RBE2/RBE3/RBAR/RROD) and MPCs** are constraints
  wearing element names — they carry no mesh and are skipped.
- **CORD1R/C/S** (systems defined by grid points) refuse loudly: a
  guessed frame places every node in it wrongly, and the cards are
  rare enough that a refusal names the fix (redefine as CORD2).
- Any other `C`-prefixed connection card refuses loudly with its
  name, because an element silently dropped is a mesh that lies.

A deck is unitless by construction, so imports arrive raw and
`length_unit=` declares — exactly the universal-file-without-164
convention.

Writing is the mirror: coordinate systems as CORD2, grids as
large-field GRID* (full precision — small-field's eight characters
are why so many decks in the wild have five-digit coordinates),
elements on their own cards with PID 1 throughout, and tracelines as
PLOTEL chains. Properties are an analyst's statement about the
structure, not the geometry's to invent, so the written deck is
interchange geometry: it meshes viewers and preprocessors, and it
would need PSHELL/PSOLID cards added before Nastran itself would run
it — the docstring of `save` says so rather than leaving it to be
discovered.
"""

from __future__ import annotations

import os
import re
from itertools import pairwise
from typing import Any

import numpy as np

from ..core.geometry import ELEMENT_TYPES, Geometry, placed, to_local
from .sniffing import text_head

EXTENSIONS = ('.bdf', '.dat', '.nas')

#: connection card -> (UFF descriptor by node count). One card name can
#: mean two vocabulary entries — CTETRA is tet4 or tet10 by how many
#: grids follow — which is exactly how Nastran itself distinguishes
#: them.
_ELEMENT_CARDS = {
    'CROD': {2: 11},
    'CBAR': {2: 21},
    'CBEAM': {2: 21},
    'CTRIA3': {3: 91},
    'CTRIA6': {6: 92},
    'CQUAD4': {4: 94},
    'CQUAD8': {8: 95},
    'CTETRA': {4: 111, 10: 118},
    'CPENTA': {6: 112, 15: 113},
    'CHEXA': {8: 115, 20: 116},
    'CPYRAM': {5: 201, 13: 202},
}

#: cards that look like elements and are not mesh: rigid links and
#: multipoint constraints. Constraints belong to the analysis, so they
#: are skipped the way a property card is — knowingly, not silently
#: in the bad sense: the set is written down here.
_CONSTRAINT_CARDS = {'RBE2', 'RBE3', 'RBAR', 'RBAR1', 'RROD', 'RBE1',
                     'RTRPLT', 'RSPLINE'}

#: how many leading fields to drop to reach the grid list, per card:
#: every element card is NAME, EID, PID, grids... except the sprung
#: and massy ones handled by hand below
_GRIDS_START = 2

_FLOAT_EXPONENT = re.compile(r'(?<=[0-9.])([+-]\d+)$')


def sniff(path: str | os.PathLike) -> bool:
    path = str(path)
    if os.path.splitext(path)[1].lower() not in EXTENSIONS:
        return False
    head = text_head(path)
    if head is None:
        return False
    upper = head.upper()
    return ('BEGIN BULK' in upper
            or any(re.search(rf'^{name}[\s,*]', upper, re.MULTILINE)
                   for name in ('GRID', 'CEND', 'SOL ')))


def _value(token: str) -> Any:
    """One field: int, float (Nastran's bare-exponent form included),
    or the stripped string. '1.-3' means 1.0e-3 — the E was dropped to
    fit eight characters, and the sign carries the exponent."""
    token = token.strip()
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    mended = _FLOAT_EXPONENT.sub(r'E\1', token)
    if mended != token:
        try:
            return float(mended)
        except ValueError:
            pass
    return token


def _fields(line: str) -> list[Any]:
    """One physical line's fields, in whichever of the three formats
    it is written."""
    line = line.split('$', 1)[0].rstrip('\n')
    if ',' in line:
        return [_value(tok) for tok in line.split(',')]
    name = line[:8]
    if '*' in name:
        # large field: 8-character name, then four 16-character fields
        out = [_value(name.replace('*', ''))]
        body = line[8:72]
        out.extend(_value(body[k:k + 16]) for k in range(0, len(body), 16))
        return out
    return [_value(line[k:k + 8]) for k in range(0, min(len(line), 72), 8)]


def _cards(lines: list[str]) -> list[list[Any]]:
    """Logical cards: each starts on a line whose first column is a
    letter, and swallows the continuation lines after it (leading
    blank, '+' or '*'). The optional trailing continuation markers in
    field 10 are dropped — they only name the next line."""
    out: list[list[Any]] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith('$'):
            continue
        first = line[0] if line else ' '
        fields = _fields(line)
        if first.isalpha():
            out.append(list(fields))
        elif out:
            # continuation: drop its leading marker field
            out[-1].extend(fields[1:])
    # strip string continuation markers (+A1 and kin) that rode along
    for card in out:
        card[:] = [f for f in card
                   if not (isinstance(f, str) and f[:1] in '+*')]
    return out


def _pad(card: list[Any], length: int) -> list[Any]:
    return card + [None] * (length - len(card))


def _cs_matrix(a, b, c):
    """A CORD2 card's three points as the vocabulary's (4, 3) matrix:
    A is the origin, A->B the z axis, and C fixes the x-z plane."""
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    z = b - a
    z = z / np.linalg.norm(z)
    x = c - a
    x = x - z * (x @ z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z, a])


def load(path: str | os.PathLike, length_unit: str | None = None) -> Any:
    path = str(path)
    lines = _bulk_lines(path)
    cards = _cards(lines)

    nodes: list[tuple[int, int, np.ndarray, int]] = []
    systems: dict[int, dict[str, Any]] = {}
    elements: list[tuple[int, int, list[int]]] = []
    tracelines: list[list[int]] = []

    for card in cards:
        name = str(card[0]).upper()
        if name == 'GRID':
            card = _pad(card, 7)
            nodes.append((int(card[1]), int(card[2] or 0),
                          np.array([float(card[3] or 0.0),
                                    float(card[4] or 0.0),
                                    float(card[5] or 0.0)]),
                          int(card[6] or 0)))
        elif name in ('CORD2R', 'CORD2C', 'CORD2S'):
            card = _pad(card, 12)
            systems[int(card[1])] = {
                'reference': int(card[2] or 0),
                'kind': {'CORD2R': 0, 'CORD2C': 1, 'CORD2S': 2}[name],
                'points': [[float(v or 0.0) for v in card[3:6]],
                           [float(v or 0.0) for v in card[6:9]],
                           [float(v or 0.0) for v in card[9:12]]],
            }
        elif name in ('CORD1R', 'CORD1C', 'CORD1S'):
            raise ValueError(
                f'{path}: {name} defines a coordinate system by grid '
                'points, which this reader does not resolve — redefine '
                'it as the equivalent CORD2 card')
        elif name == 'PLOTEL':
            # Nastran's own display-only line: exactly a traceline
            tracelines.append([int(card[2]), int(card[3])])
        elif name == 'CONM2':
            elements.append((int(card[1]), 161, [int(card[2])]))
        elif name in ('CELAS1', 'CELAS2'):
            # the grounded spring keeps its one live grid; a
            # two-grid spring keeps the first, the way the
            # vocabulary's point element does
            grid = card[4] if name == 'CELAS1' else card[3]
            elements.append((int(card[1]), 136, [int(grid)]))
        elif name in _ELEMENT_CARDS:
            grids = [int(g) for g in card[_GRIDS_START + 1:]
                     if isinstance(g, (int, float)) and g]
            by_count = _ELEMENT_CARDS[name]
            if len(grids) not in by_count:
                raise ValueError(
                    f'{path}: {name} {card[1]} carries {len(grids)} '
                    f'grids; this card means {sorted(by_count)} '
                    'of them')
            elements.append((int(card[1]), by_count[len(grids)], grids))
        elif name.startswith('C') and name not in _CONSTRAINT_CARDS \
                and _looks_like_connection(card):
            raise ValueError(
                f'{path}: unsupported connection card {name} — an '
                'element silently dropped is a mesh that lies')

    geometry = Geometry(
        node_id=[n[0] for n in nodes],
        node_xyz=_resolve_positions(nodes, systems, path),
        node_def_cs=[n[1] for n in nodes],
        node_disp_cs=[n[3] for n in nodes],
        node_color=[1] * len(nodes),
        cs_id=[0] + sorted(systems),
        cs_name=[''] * (1 + len(systems)),
        cs_type=[0] + [systems[k]['kind'] for k in sorted(systems)],
        cs_matrix=np.concatenate(
            [np.vstack([np.eye(3), np.zeros(3)])[None]]
            + [_resolved_matrix(systems, k, path)[None]
               for k in sorted(systems)]),
        traceline_id=list(range(1, len(tracelines) + 1)),
        traceline_color=[1] * len(tracelines),
        traceline_conn=[np.array(t) for t in tracelines],
        elem_id=[e[0] for e in elements],
        elem_type=[e[1] for e in elements],
        elem_color=[1] * len(elements),
        elem_conn=[np.array(e[2]) for e in elements],
    )
    if length_unit:
        geometry.define_units(length_unit)
    return geometry


def _looks_like_connection(card) -> bool:
    """A C-card shaped like an element: id, property, then grids."""
    return (len(card) >= 4 and isinstance(card[1], int)
            and all(isinstance(f, (int, float)) or f is None
                    for f in card[1:]))


def _bulk_lines(path: str) -> list[str]:
    """The bulk section's lines, INCLUDEs inlined relative to the deck.

    Everything before BEGIN BULK is executive and case control — an
    analysis's business — and everything after ENDDATA is nothing at
    all. A file with neither marker is taken as all bulk, which is how
    partial decks travel."""
    def read(one: str) -> list[str]:
        with open(one, errors='replace') as f:
            raw = f.readlines()
        out: list[str] = []
        for line in raw:
            stripped = line.strip()
            if stripped.upper().startswith('INCLUDE'):
                target = stripped[7:].strip().strip("'\"")
                out.extend(read(os.path.join(os.path.dirname(one), target)))
            else:
                out.append(line)
        return out

    lines = read(path)
    upper = [line.upper() for line in lines]
    start = next((i + 1 for i, line in enumerate(upper)
                  if line.startswith('BEGIN BULK')), 0)
    stop = next((i for i, line in enumerate(upper)
                 if line.startswith('ENDDATA')), len(lines))
    return lines[start:stop]


def _resolved_matrix(systems, cid, path, _seen=None):
    """One CORD2 system's (4, 3) matrix in the basic frame, chains of
    references walked — B defined in A defined in basic."""
    _seen = _seen or set()
    if cid in _seen:
        raise ValueError(f'{path}: coordinate system {cid} refers to '
                         'itself through its references')
    entry = systems[cid]
    matrix = _cs_matrix(*entry['points'])
    reference = entry['reference']
    if reference == 0:
        return matrix
    if reference not in systems:
        raise ValueError(f'{path}: coordinate system {cid} references '
                         f'{reference}, which the deck does not define')
    if systems[reference]['kind'] != 0:
        raise ValueError(
            f'{path}: coordinate system {cid} is defined in the '
            f'curvilinear system {reference}; only cartesian '
            'references are resolved')
    outer = _resolved_matrix(systems, reference, path, _seen | {cid})
    axes, origin = outer[:3], outer[3]
    return np.vstack([matrix[:3] @ axes, origin + matrix[3] @ axes])


def _resolve_positions(nodes, systems, path):
    """Every grid's basic-frame position: CP = 0 is already there, a
    cartesian CP rotates and shifts, a cylindrical or spherical CP
    converts first — the standard meanings of the CORD2C/S columns."""
    out = np.zeros((len(nodes), 3))
    for k, (label, cp, xyz, _cd) in enumerate(nodes):
        if cp == 0:
            out[k] = xyz
            continue
        # a deck that places a grid in a frame it never defines is
        # wrong about itself, and saying which grid beats drawing it
        # somewhere plausible
        if cp not in systems:
            raise ValueError(f'{path}: grid {label} is defined in '
                             f'coordinate system {cp}, which the deck '
                             'does not define')
        out[k] = placed(xyz, systems[cp]['kind'],
                        _resolved_matrix(systems, cp, path))
    return out


# ---- writing -----------------------------------------------------------

#: vocabulary code -> bulk card name (node count picks the variant on
#: the way back in)
_CARD_NAMES = {11: 'CROD', 21: 'CBAR', 22: 'CBAR', 23: 'CBAR',
               24: 'CBAR',
               41: 'CTRIA3', 91: 'CTRIA3', 42: 'CTRIA6', 92: 'CTRIA6',
               44: 'CQUAD4', 94: 'CQUAD4', 45: 'CQUAD8', 95: 'CQUAD8',
               111: 'CTETRA', 118: 'CTETRA', 112: 'CPENTA',
               113: 'CPENTA', 115: 'CHEXA', 116: 'CHEXA', 117: 'CHEXA',
               201: 'CPYRAM', 202: 'CPYRAM',
               136: 'CELAS2', 161: 'CONM2'}


def _small(value) -> str:
    if value is None:
        return ' ' * 8
    if isinstance(value, str):
        return f'{value:<8.8s}'
    if isinstance(value, (int, np.integer)):
        return f'{int(value):8d}'
    return f'{float(value):8.6G}'[:8].rjust(8)


def _card(name: str, *fields) -> str:
    """A small-field card, continued every eight data fields."""
    out, row = [f'{name:<8s}'], 0
    for field in fields:
        if row == 8:
            out.append('\n+       ')
            row = 0
        out.append(_small(field))
        row += 1
    return ''.join(out) + '\n'


def handles(obj: Any) -> bool:
    return isinstance(obj, Geometry)


def save(geometry: Geometry, path: str | os.PathLike,
         unit_system: Any = None) -> None:
    """Write a geometry as a bulk deck.

    Interchange geometry, not a runnable analysis: elements carry
    PID 1 and no PSHELL/PSOLID/MAT cards are written, because
    properties are the analyst's statement about the structure and
    inventing them here would put made-up stiffness in a real deck.
    Grids go out large-field for full precision. Tracelines become
    PLOTEL chains — Nastran's own display-only line. Values are
    written in SI, the geometry's storage.
    """
    from ..core.geometry import ELEMENT_TYPES as VOCABULARY

    lines = ['$ written by visualdynamics\n', 'BEGIN BULK\n']
    for i, cid in enumerate(geometry.cs_id):
        if int(cid) == 0:
            continue
        matrix = np.asarray(geometry.cs_matrix[i], dtype=float)
        name = {0: 'CORD2R', 1: 'CORD2C', 2: 'CORD2S'}[
            int(geometry.cs_type[i])]
        a = matrix[3]
        b = a + matrix[2]          # the z axis point
        c = a + matrix[0]          # in the x-z plane
        lines.append(_card(name, int(cid), 0, *a, *b, *c))
    # each grid is stated in the frame its CP field names, or the deck
    # would carry basic-frame values under a local CP and every reader
    # of it, this one included, would resolve them a second time
    written = to_local(np.asarray(geometry.node_xyz, dtype=float),
                       geometry.node_def_cs, geometry.cs_id,
                       geometry.cs_type,
                       np.asarray(geometry.cs_matrix, dtype=float))
    for i, node in enumerate(geometry.node_id):
        x, y, z = (float(v) for v in written[i])
        lines.append(
            f'GRID*   {int(node):>16d}'
            f'{int(geometry.node_def_cs[i]):>16d}'
            f'{x:16.9E}{y:16.9E}*\n'
            f'*       {z:16.9E}'
            f'{int(geometry.node_disp_cs[i]):>16d}\n')
    for i, code in enumerate(geometry.elem_type):
        code = int(code)
        name = _CARD_NAMES.get(code)
        if name is None:
            raise ValueError(
                f'no bulk card for a {VOCABULARY[code][0]} '
                f'(element {int(geometry.elem_id[i])})')
        conn = [int(n) for n in geometry.elem_conn[i]]
        if name == 'CONM2':
            lines.append(_card(name, int(geometry.elem_id[i]),
                               conn[0], 0, 0.0))
        elif name == 'CELAS2':
            lines.append(_card(name, int(geometry.elem_id[i]),
                               0.0, conn[0], 1))
        else:
            lines.append(_card(name, int(geometry.elem_id[i]), 1, *conn))
    plotel = 1 + (max((int(e) for e in geometry.elem_id), default=0))
    for i, conn in enumerate(geometry.traceline_conn):
        chain = [int(n) for n in conn]
        for a, b in pairwise(chain):
            lines.append(_card('PLOTEL', plotel, a, b))
            plotel += 1
    lines.append('ENDDATA\n')
    with open(str(path), 'w') as f:
        f.writelines(lines)


assert set(_ELEMENT_CARDS) <= {name for name in _CARD_NAMES.values()} | \
    {'CROD', 'CBEAM'}, 'every readable card is writable or aliased'
assert all(code in ELEMENT_TYPES for by_count in _ELEMENT_CARDS.values()
           for code in by_count.values()), \
    'the card map speaks the vocabulary'
