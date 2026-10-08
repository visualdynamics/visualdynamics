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

from ..core.fem import AXES
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
    elements: list[tuple[int, int, list[int], int]] = []
    tracelines: list[list[int]] = []
    # the property cards, for the element groups: an element group per property id, and
    # what a solid or shell property says its element group is made of
    materials: dict[int, dict[str, float | None]] = {}
    solids: dict[int, int] = {}
    shells: dict[int, tuple[int, float | None]] = {}
    # each CONM2's mass, by element, when the card is a plain point
    # mass: one element group per distinct mass afterwards
    point_masses: dict[int, float] = {}
    # each CELAS: (eid, g1, component, g2 or None, its stiffness or the
    # PELAS id that holds it), one group per stiffness and direction
    # afterwards; the PELAS cards; each SPC'd grid's components
    springs: list[tuple[int, int, int, int | None, Any]] = []
    pelas: dict[int, float] = {}
    supports: dict[int, set[int]] = {}
    spc_sets: set[int] = set()

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
            # Nastran's own display-only line: exactly a drawn line
            tracelines.append([int(card[2]), int(card[3])])
        elif name == 'CONM2':
            elements.append((int(card[1]), 161, [int(card[2])], 0))
            # EID G CID M X1 X2 X3, then the inertias: a mass carried
            # off its grid or with inertia is not a point mass, and its
            # element group is left without properties for the analyst to state
            # rather than given a mass that would put it in the wrong
            # place
            card = _pad(card, 5)
            mass, rest = card[4], card[5:]
            if (isinstance(mass, (int, float)) and mass > 0
                    and not any(isinstance(v, (int, float)) and v
                                for v in rest)):
                point_masses[int(card[1])] = float(mass)
        elif name in ('CELAS1', 'CELAS2'):
            # EID PID|K G1 C1 G2 C2: a CELAS1 names the PELAS holding its
            # stiffness, a CELAS2 carries it (2026-10-08: the reader kept
            # the first grid alone and dropped the stiffness, and read a
            # CELAS1's grid out of its component field)
            card = _pad(card, 7)
            springs.append(_spring_card(path, name, card))
        elif name == 'PELAS':
            # up to two properties to a card: PID K GE S, twice
            card = _pad(card, 9)
            for at in (1, 5):
                if isinstance(card[at], int) and isinstance(
                        card[at + 1], (int, float)):
                    pelas[int(card[at])] = float(card[at + 1])
        elif name in ('SPC', 'SPC1'):
            spc_sets.add(int(card[1]))
            for grid, component in _support_card(path, name, card):
                supports.setdefault(grid, set()).update(component)
        elif name in _ELEMENT_CARDS:
            grids = [int(g) for g in card[_GRIDS_START + 1:]
                     if isinstance(g, (int, float)) and g]
            by_count = _ELEMENT_CARDS[name]
            if len(grids) not in by_count:
                raise ValueError(
                    f'{path}: {name} {card[1]} carries {len(grids)} '
                    f'grids; this card means {sorted(by_count)} '
                    'of them')
            pid = card[_GRIDS_START] if len(card) > _GRIDS_START else 0
            elements.append((int(card[1]), by_count[len(grids)], grids,
                             int(pid) if isinstance(pid, (int, float)) else 0))
        elif name == 'MAT1':
            card = _pad(card, 6)
            materials[int(card[1])] = {
                key: (float(value) if isinstance(value, (int, float))
                      else None)
                for key, value in zip(('E', 'G', 'nu', 'rho'), card[2:6])}
        elif name == 'PSOLID':
            solids[int(card[1])] = int(card[2])
        elif name == 'PSHELL':
            card = _pad(card, 4)
            thickness = card[3]
            shells[int(card[1])] = (int(card[2]), float(thickness)
                                    if isinstance(thickness, (int, float))
                                    else None)
        elif name.startswith('C') and name not in _CONSTRAINT_CARDS \
                and _looks_like_connection(card):
            raise ValueError(
                f'{path}: unsupported connection card {name} — an '
                'element silently dropped is a mesh that lies')

    # an element group per distinct CONM2 mass, numbered after every property
    # id so none is taken, carrying its mass (2026-10-07)
    mass_blocks: dict[float, int] = {}
    if point_masses:
        after = max((e[3] for e in elements), default=0) + 1
        for mass in sorted(set(point_masses.values())):
            mass_blocks[mass] = after + len(mass_blocks)
        elements = [(eid, code, grids, mass_blocks[point_masses[eid]])
                    if code == 161 and eid in point_masses
                    else (eid, code, grids, pid)
                    for eid, code, grids, pid in elements]
    mass_names = {group: f'CONM2 {mass:g}'
                  for mass, group in mass_blocks.items()}
    properties = _block_properties(materials, solids, shells)
    properties.update({group: _mass_properties(mass)
                       for mass, group in mass_blocks.items()})
    if len(spc_sets) > 1:
        raise ValueError(
            f'{path}: the deck holds SPC sets '
            + ', '.join(str(s) for s in sorted(spc_sets))
            + '; a geometry holds one set of supports — keep the one the '
            'case control selects')
    # a group per distinct CELAS stiffness and direction, and per
    # distinct set of SPC components, numbered after every group so far
    # (2026-10-08)
    after = max([e[3] for e in elements] + list(mass_names), default=0) + 1
    next_eid = max([e[0] for e in elements], default=0) + 1
    made: dict[Any, int] = {}
    for eid, g1, component, g2, stiffness in springs:
        if isinstance(stiffness, tuple):
            stiffness = pelas.get(stiffness[1])
        code = ((136 if component <= 3 else 137) if g2
                else (138 if component <= 3 else 139))
        grids = [g1, g2] if g2 else [g1]
        if stiffness is None:
            # its PELAS is not in the deck: where it is, with nothing said
            elements.append((eid, code, grids, 0))
            continue
        key = ('spring', stiffness, component)
        if key not in made:
            made[key] = after + len(made)
            values = [None] * 6
            values[component - 1] = stiffness
            properties[made[key]] = _spring_properties(tuple(values))
            mass_names[made[key]] = (f'CELAS {stiffness:g} '
                                     f'{AXES[component - 1]}')
        elements.append((eid, code, grids, made[key]))
    for grid in sorted(supports):
        held = tuple(sorted(supports[grid]))
        key = ('ground', held)
        if key not in made:
            made[key] = after + len(made)
            properties[made[key]] = _ground_properties(
                [AXES[c - 1] for c in held])
            mass_names[made[key]] = 'SPC ' + ''.join(str(c) for c in held)
        elements.append((next_eid, 161, [grid], made[key]))
        next_eid += 1
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
        elem_id=[e[0] for e in elements],
        elem_type=[e[1] for e in elements],
        elem_color=[1] * len(elements),
        elem_conn=[np.array(e[2]) for e in elements],
        elem_group=[e[3] for e in elements],
        group_id=sorted({e[3] for e in elements}),
        group_name=[mass_names.get(pid) or (f'property {pid}' if pid
                                            else '')
                    for pid in sorted({e[3] for e in elements})],
        group_properties=properties,
    )
    # the PLOTELs are one drawn line — an element group of two-node line
    # elements with no properties — chained where they join
    if tracelines:
        geometry.attach_drawn_lines([('PLOTEL', 1, tracelines)])
    if length_unit:
        geometry.define_units(length_unit)
    return geometry


def _block_properties(materials: dict, solids: dict, shells: dict) -> dict:
    """What the deck's own property cards say each element group is made of:
    a PSOLID names a material, a PSHELL a material and a thickness, a
    MAT1 the numbers. An element group whose cards are not all there carries
    nothing, and the Element Groups table asks (2026-09-30). The deck is taken
    to be in SI: Nastran has no units, and a deck in inches and pounds
    reads the same as one in meters and kilograms."""
    from ..core.fem import GroupProperties, Material

    def material(mid: int) -> Material | None:
        found = materials.get(mid)
        if not found or found['E'] is None or found['rho'] is None:
            return None
        nu = found['nu']
        if nu is None:
            # E and G given: nu follows from them, the way Nastran fills it
            if found['G'] is None:
                return None
            nu = found['E'] / (2.0 * found['G']) - 1.0
        return Material(f'MAT1 {mid}', found['E'], found['rho'], nu,
                        found['G'])

    out = {}
    for pid, mid in solids.items():
        made = material(mid)
        if made is not None:
            out[pid] = GroupProperties(made)
    for pid, (mid, thickness) in shells.items():
        made = material(mid)
        if made is not None and thickness is not None:
            out[pid] = GroupProperties(made, thickness=thickness)
    return out


def _spring_card(path, name, card) -> tuple:
    """(eid, g1, component, g2 or None, stiffness or ('PELAS', pid)).
    A spring joins one direction at both ends here: one that joins two
    different components is refused by name rather than read as half of
    what it says."""
    eid = int(card[1])
    g1, c1, g2, c2 = card[3], card[4], card[5], card[6]
    if not isinstance(g1, int) or g1 == 0:
        raise ValueError(f'{path}: {name} {eid} acts on a scalar point; '
                         'only grid points are read')
    if not isinstance(c1, int) or not 1 <= c1 <= 6:
        raise ValueError(f'{path}: {name} {eid} names component {c1!r}; '
                         'a grid point has components 1 to 6')
    if isinstance(g2, int) and g2:
        if c2 != c1:
            raise ValueError(
                f'{path}: {name} {eid} joins component {c1} of grid {g1} '
                f'to component {c2} of grid {g2}; a spring here joins one '
                'direction at both ends')
    else:
        g2 = None
    stiffness = (('PELAS', int(card[2])) if name == 'CELAS1'
                 else float(card[2]) if isinstance(card[2], (int, float))
                 else None)
    return eid, int(g1), int(c1), g2, stiffness


def _support_card(path, name, card):
    """(grid, components) for each grid an SPC or SPC1 holds. An SPC
    that enforces a displacement is no support, and is refused."""
    def components(value):
        digits = str(value or '')
        if not digits or not set(digits) <= set('123456'):
            raise ValueError(f'{path}: {name} names components {value!r}; '
                             'a grid point has components 1 to 6')
        return {int(d) for d in digits}

    if name == 'SPC':
        card = _pad(card, 8)
        for at in (2, 5):
            grid, held, enforced = card[at], card[at + 1], card[at + 2]
            if not isinstance(grid, int) or not grid:
                continue
            if isinstance(enforced, (int, float)) and enforced:
                raise ValueError(f'{path}: SPC {card[1]} enforces a '
                                 f'displacement at grid {grid}; only '
                                 'supports are read')
            yield grid, components(held)
        return
    held = components(card[2])
    grids = card[3:]
    if len(grids) >= 3 and str(grids[1]).upper() == 'THRU':
        for grid in range(int(grids[0]), int(grids[2]) + 1):
            yield grid, held
        return
    for grid in grids:
        if isinstance(grid, int) and grid:
            yield grid, held


def _spring_properties(stiffness):
    from ..core.fem import GroupProperties

    return GroupProperties(stiffness=stiffness)


def _ground_properties(axes):
    from ..core.fem import GroupProperties

    return GroupProperties(ground=tuple(axes))


def _mass_properties(mass: float):
    from ..core.fem import GroupProperties

    return GroupProperties(mass=mass)


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
               136: 'CELAS2', 137: 'CELAS2', 138: 'CELAS2',
               139: 'CELAS2', 161: 'CONM2'}


def _small(value) -> str:
    if value is None:
        return ' ' * 8
    if isinstance(value, str):
        return f'{value:<8.8s}'
    if isinstance(value, (int, np.integer)):
        return f'{int(value):8d}'
    return _real8(float(value)).rjust(8)


def _real8(value: float) -> str:
    """A real in eight characters, as many digits as fit, in Nastran's
    own short form where an exponent is needed ('1.2346+7'). Cut to
    eight, '1.23457E+07' was '1.23457E', a number no reader parses back
    (2026-10-08, found writing a stiffness)."""
    if value == 0.0:
        return '0.'
    for digits in range(7, 0, -1):
        text = f'{value:.{digits}G}'
        if 'E' in text:
            mantissa, exponent = text.split('E')
            if '.' not in mantissa:
                mantissa += '.'
            text = mantissa + f'{int(exponent):+d}'
        elif '.' not in text:
            text += '.'
        if len(text) <= 8:
            return text
    raise ValueError(f'{value!r} does not fit a small field')


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
    A point mass is the exception that is no invention: a CONM2 carries
    the mass its element group was given, when it was given one, and 0.0
    otherwise. Springs and supports likewise: a group of springs goes
    out as a CELAS2 per element and direction with its stiffness, and a
    ground group as SPC1 cards of its components (2026-10-08). Grids go
    out large-field for full precision. Drawn lines become
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
    # a drawn line — an element group of two-node line elements with no
    # properties — goes out as PLOTEL chains and not as elements too
    drawn = geometry.drawn_lines()
    drawn_groups = {line['group'] for line in drawn}
    groups = getattr(geometry, 'group_properties', None) or {}
    # a spring of several directions is a CELAS2 per direction, the ones
    # after the first numbered past every element; a ground point is a
    # grid on an SPC1 of its components, and no element (2026-10-08)
    spare = 1 + max((int(e) for e in geometry.elem_id), default=0)
    supports: dict[str, list[int]] = {}
    for i, code in enumerate(geometry.elem_type):
        if int(geometry.elem_group[i]) in drawn_groups:
            continue
        props = groups.get(int(geometry.elem_group[i]))
        conn = [int(n) for n in geometry.elem_conn[i]]
        if props is not None and props.kind == 'ground':
            held = ''.join(str(AXES.index(a) + 1) for a in props.ground)
            supports.setdefault(held, []).append(conn[0])
            continue
        if props is not None and props.kind == 'spring':
            eid = int(geometry.elem_id[i])
            for k, stiffness in enumerate(props.stiffness):
                if not stiffness:
                    continue
                far = (conn[-1], k + 1) if len(conn) == 2 else ()
                lines.append(_card('CELAS2', eid, float(stiffness), conn[0],
                                   k + 1, *far))
                eid, spare = spare, spare + 1
            continue
        code = int(code)
        name = _CARD_NAMES.get(code)
        if name is None:
            raise ValueError(
                f'no bulk card for a {VOCABULARY[code][0]} '
                f'(element {int(geometry.elem_id[i])})')
        if name == 'CONM2':
            mass = props.mass if props is not None and props.mass else 0.0
            lines.append(_card(name, int(geometry.elem_id[i]),
                               conn[0], 0, float(mass)))
        elif name == 'CELAS2':
            # a spring element in no group of springs: where it is, with
            # no stiffness to say
            far = (conn[1], 1) if len(conn) == 2 else ()
            lines.append(_card(name, int(geometry.elem_id[i]),
                               0.0, conn[0], 1, *far))
        else:
            lines.append(_card(name, int(geometry.elem_id[i]), 1, *conn))
    for held, grids in supports.items():
        lines.append(_card('SPC1', 1, int(held), *grids))
    plotel = spare
    for line in drawn:
        for chain in line['chains']:
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
