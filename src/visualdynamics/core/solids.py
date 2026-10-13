"""Solids: the shapes a part is made of, before it is meshed.

A part is described most directly by its shapes: the Frame-Wing's frame
is a block with four windows and 41 holes taken out of it, a wing a
block less eighteen holes. Brandon (2026-10-11) wanted parts built that
way in the app, a hole a cylinder subtracted rather than a parameter of
one meshed block, and meshed once they are right. So a geometry holds a
table of solids beside its nodes, coordinate systems and elements: each
one a primitive — a block or a cylinder, placed by a center and three
angles — or a boolean of others, union, subtract or intersect. They are
not elements. Nothing that solves, animates or exports a geometry reads
them; meshing one is what turns it into elements.

The table holds recipes, the numbers a person typed, so a saved project
reopens editable and needs no geometry kernel to load. Evaluating one —
its volume, its surface for drawing, the boolean itself — is
OpenCASCADE's work (`shape`), through the OCP bindings the STEP importer
already uses: in every packaged build, and `visualdynamics[step]` for a
pip install.

Lengths are held as the geometry holds its coordinates — meters once
its units are defined — and angles in degrees about the geometry's
fixed X, then Y, then Z axes, the rule Add Block's pane turns a block
by (`rotate.frame_from_angles`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import numpy as np

from .fem import SHAPES as BEAM_SHAPES

#: the shapes a section is, along its own Z, and the dimensions each is
#: typed by, in order (2026-10-12): the beam element's shapes, named and
#: dimensioned as its sections are (`fem.SHAPES`), and a tee, an
#: I-beam's dimensions
SECTIONS: dict[str, tuple[str, ...]] = {
    shape: BEAM_SHAPES['I-beam' if shape == 'tee' else shape][1]
    for shape in ('I-beam', 'channel', 'tee', 'angle', 'rectangular tube',
                  'round tube')}
#: a hole, to subtract from a part: entered at its center, running along
#: its own -Z, a depth of zero through the part it is cut from, a
#: counterbore diameter of zero none
HOLE = 'hole'
HOLE_DIMENSIONS = ('diameter', 'depth', 'counterbore diameter', 'counterbore depth')
#: the solids typed by their dimensions, each made of blocks and
#: cylinders (`_pieces`)
SHAPED = (*SECTIONS, HOLE)
#: the primitives — the block and the cylinder, and the shapes made of
#: them, each one row a person places, turns and sizes — the booleans
#: over them, and the pattern: copies of its solids in rows, columns and
#: layers, made of them as a boolean is
BASIC = ('block', 'cylinder')
PRIMITIVES = (*BASIC, *SHAPED)
OPERATIONS = ('union', 'subtract', 'intersect')
PATTERN = 'pattern'

#: how fine a surface is tessellated for drawing, as a fraction of the
#: shape's size: context for the eye, not a mesh to solve — coarse on
#: purpose, matching the guidance on CAD export settings
DEFLECTION = 0.005


@dataclass(frozen=True)
class Solid:
    """One row of a geometry's solids table.

    A primitive is placed by its `center` and turned by `angles`; a
    block is `widths` across along its own axes, a cylinder `radius`
    round its own z axis and `length` along it. A boolean names its
    `operands` by id, in order — a subtraction takes the rest from the
    first — and its own numbers are unused. A pattern names its solids
    as a boolean does, and holds `counts` copies along its own axes,
    `widths` apart, turned by `angles` (Brandon, 2026-10-11: the copies
    follow the original, and the pattern is changed as one). A section
    or a hole is typed by its `dimensions` (`SECTIONS`, `HOLE`), a
    section `length` along its own Z (2026-10-12).

    Parameters
    ----------
    id : int
        Unique within the geometry.
    name : str
        What a person calls it; may be empty.
    kind : str
        One of `PRIMITIVES` or `OPERATIONS`, or `PATTERN`.
    center : tuple of float
        The primitive's middle.
    angles : tuple of float
        Degrees about the fixed X, then Y, then Z axes.
    widths : tuple of float
        A block's three widths along its own axes; a pattern's spacing
        along its own.
    radius, length : float
        A cylinder's.
    operands : tuple of int
        A boolean's solids, by id.
    corner : float
        A block's corner radius: its four edges along its own Z rounded
        (2026-10-11) — a window cut with rounded corners leaves the
        fillets a part has there.
    counts : tuple of int
        A pattern's copies along its own axes, the original counted.
    dimensions : tuple of float
        A section's or a hole's, in the order `SECTIONS` and
        `HOLE_DIMENSIONS` name them.
    """

    id: int
    name: str
    kind: str
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0)
    widths: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    length: float = 0.0
    operands: tuple[int, ...] = ()
    corner: float = 0.0
    counts: tuple[int, int, int] = (1, 1, 1)
    dimensions: tuple[float, ...] = ()

    def scaled(self, factor: float) -> Solid:
        """The same solid with every length multiplied by `factor`:
        what a change of the geometry's length unit does to it."""
        return replace(self, center=tuple(float(c) * factor for c in self.center),
                       widths=tuple(float(w) * factor for w in self.widths),
                       radius=self.radius * factor,
                       length=self.length * factor,
                       corner=self.corner * factor,
                       dimensions=tuple(float(d) * factor for d in self.dimensions))

    def as_record(self) -> dict:
        """The solid as plain numbers, for a file."""
        return {'id': self.id, 'name': self.name, 'kind': self.kind,
                'center': list(self.center), 'angles': list(self.angles),
                'widths': list(self.widths), 'radius': self.radius,
                'length': self.length, 'operands': list(self.operands),
                'corner': self.corner, 'counts': list(self.counts),
                'dimensions': list(self.dimensions)}

    @classmethod
    def from_record(cls, record: dict) -> Solid:
        """The solid a file's record describes (`as_record`)."""
        return cls(int(record['id']), str(record['name']), str(record['kind']),
                   tuple(float(v) for v in record['center']),
                   tuple(float(v) for v in record['angles']),
                   tuple(float(v) for v in record['widths']),
                   float(record['radius']), float(record['length']),
                   tuple(int(v) for v in record['operands']),
                   float(record.get('corner', 0.0)),
                   tuple(int(v) for v in record.get('counts', (1, 1, 1))),
                   tuple(float(v) for v in record.get('dimensions', ())))


def _next_id(geometry: Any) -> int:
    return max((s.id for s in geometry.solids), default=0) + 1


def _numbers(values, count: int, what: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in np.asarray(values, dtype=float).ravel())
    if len(out) != count or not all(np.isfinite(out)):
        raise ValueError(f'{what} takes {count} finite numbers')
    return out


def add_block(geometry: Any, center, widths, angles=(0.0, 0.0, 0.0),
              name: str = '', corner: float = 0.0) -> int:
    """Add a block to `geometry`'s solids: `widths` across along its own
    axes, its middle at `center`, turned by `angles`.

    Parameters
    ----------
    geometry : Geometry
        Where it goes; changed in place.
    center, widths : array_like
        Three numbers each, as the geometry holds its coordinates.
    angles : array_like, default (0, 0, 0)
        Degrees about the fixed X, then Y, then Z axes.
    name : str, optional
        What to call it.
    corner : float, default 0
        The radius its four edges along its own Z are rounded to; at
        most half its narrower width across.

    Returns
    -------
    int
        The new solid's id.
    """
    solid = _block(_next_id(geometry), name, center, widths, angles, corner)
    geometry.solids.append(solid)
    return solid.id


def _block(solid_id: int, name: str, center, widths, angles, corner) -> Solid:
    """A block's row, its numbers checked: what adding one and changing
    one both refuse."""
    widths = _numbers(widths, 3, 'a block\'s widths')
    if min(widths) <= 0.0:
        raise ValueError('a block needs a width along all three axes')
    corner = float(corner)
    if not 0.0 <= corner <= min(widths[:2]) / 2.0:
        raise ValueError('a corner radius from zero to half the block\'s '
                         'narrower width across')
    return Solid(solid_id, name, 'block', _numbers(center, 3, 'a center'),
                 _numbers(angles, 3, 'angles'), widths, corner=corner)


def add_cylinder(geometry: Any, center, radius: float, length: float,
                 angles=(0.0, 0.0, 0.0), name: str = '') -> int:
    """Add a cylinder to `geometry`'s solids: `radius` round its own z
    axis and `length` along it, its middle at `center`, turned by
    `angles` — with none, its axis is the geometry's Z.

    Parameters
    ----------
    geometry : Geometry
        Where it goes; changed in place.
    center : array_like
        Three numbers, as the geometry holds its coordinates.
    radius, length : float
        Likewise.
    angles : array_like, default (0, 0, 0)
        Degrees about the fixed X, then Y, then Z axes.
    name : str, optional
        What to call it.

    Returns
    -------
    int
        The new solid's id.
    """
    solid = _cylinder(_next_id(geometry), name, center, radius, length, angles)
    geometry.solids.append(solid)
    return solid.id


def _cylinder(solid_id: int, name: str, center, radius, length, angles) -> Solid:
    """A cylinder's row, its numbers checked."""
    if not (float(radius) > 0.0 and float(length) > 0.0):
        raise ValueError('a cylinder needs a radius and a length')
    return Solid(solid_id, name, 'cylinder', _numbers(center, 3, 'a center'),
                 _numbers(angles, 3, 'angles'), radius=float(radius),
                 length=float(length))


def add_section(geometry: Any, shape: str, center, dimensions, length: float,
                angles=(0.0, 0.0, 0.0), name: str = '') -> int:
    """Add a section to `geometry`'s solids: an I-beam, a channel, a tee,
    an angle or a tube, `length` along its own Z (2026-10-12) — one row,
    sized by its own dimensions, where a union of blocks would be three
    rows changed one by one. Its middle, the middle of the box it fills,
    is at `center`; with no `angles` it runs along the geometry's Z.

    Across, along its own axes: an I-beam's or a tee's `depth` runs
    along Y, its flange `flange width` along X, a tee's at +Y; a
    channel's flanges run along +X from its web; an angle's long leg
    runs along +X and its short leg along +Y from the heel; a
    rectangular tube is `width` along X and `height` along Y.

    Parameters
    ----------
    geometry : Geometry
        Where it goes; changed in place.
    shape : str
        One of `SECTIONS`.
    center : array_like
        Three numbers, as the geometry holds its coordinates.
    dimensions : sequence of float
        In the order `SECTIONS[shape]` names them, likewise.
    length : float
        Likewise.
    angles : array_like, default (0, 0, 0)
        Degrees about the fixed X, then Y, then Z axes.
    name : str, optional
        What to call it.

    Returns
    -------
    int
        The new solid's id.
    """
    if shape not in SECTIONS:
        raise ValueError(f'{shape!r} is not one of {tuple(SECTIONS)}')
    solid = _shaped(_next_id(geometry), name, shape, center, dimensions, length,
                    angles)
    geometry.solids.append(solid)
    return solid.id


def add_hole(geometry: Any, center, diameter: float, depth: float | None = None,
             angles=(0.0, 0.0, 0.0), name: str = '',
             counterbore=None) -> int:
    """Add a hole to `geometry`'s solids, to subtract from a part
    (2026-10-12): entered at `center`, on the face it is drilled from,
    and running `depth` along its own -Z — down the geometry's Z with no
    `angles` — or, with none, through the part it is cut from, however
    thick that is made. A counterbore is wider at the entry.

    Parameters
    ----------
    geometry : Geometry
        Where it goes; changed in place.
    center : array_like
        Where it enters, three numbers as the geometry holds its
        coordinates.
    diameter : float
        Likewise.
    depth : float, optional
        Likewise; None through.
    angles : array_like, default (0, 0, 0)
        Degrees about the fixed X, then Y, then Z axes.
    name : str, optional
        What to call it.
    counterbore : (float, float), optional
        Its diameter, wider than the hole's, and its depth, shallower.

    Returns
    -------
    int
        The new solid's id.
    """
    wide, deep = (0.0, 0.0) if counterbore is None else counterbore
    solid = _shaped(_next_id(geometry), name, HOLE, center,
                    (diameter, 0.0 if depth is None else depth, wide, deep), 0.0,
                    angles)
    geometry.solids.append(solid)
    return solid.id


def _shaped(solid_id: int, name: str, kind: str, center, dimensions, length,
            angles) -> Solid:
    """A section's or a hole's row, its numbers checked: what adding one
    and changing one both refuse. The beam sections' rules are the beam
    element's (`fem.Section.of_shape`), so a shape refused as a beam's
    section is refused as a solid."""
    from .fem import Section, with_article

    named = SECTIONS.get(kind, HOLE_DIMENSIONS)
    dims = _numbers(dimensions, len(named), f'{with_article(kind)}\'s dimensions')
    length = float(length)
    if kind == HOLE:
        diameter, depth, wide, deep = dims
        if diameter <= 0.0 or depth < 0.0:
            raise ValueError('a hole needs a diameter, and a depth of zero or '
                             'more — zero through')
        if (wide or deep) and (wide <= diameter or deep <= 0.0
                               or (depth and deep >= depth)):
            raise ValueError('a counterbore wider than the hole and shallower '
                             'than it')
    else:
        if not length > 0.0:
            raise ValueError(f'{with_article(kind)} needs a length')
        if kind == 'tee':
            depth, flange, tf, tw = dims
            if min(dims) <= 0.0 or depth - tf <= 0.0 or tw > flange:
                raise ValueError('the tee: the flanges and web do not make a tee')
        else:
            Section.of_shape(f'the {kind}', kind, dims)
        if kind == 'round tube' and 2.0 * dims[1] >= dims[0]:
            # the beam's section takes a wall of the whole radius, a rod;
            # a solid's bore would be a cylinder of no radius
            raise ValueError('the round tube: a wall under half its diameter '
                             '— a rod is a cylinder')
    return Solid(solid_id, name, kind, _numbers(center, 3, 'a center'),
                 _numbers(angles, 3, 'angles'), length=length, dimensions=dims)


#: what each kind's numbers are, to be changed by `edit`
EDITABLE = {'block': ('center', 'angles', 'widths', 'corner'),
            'cylinder': ('center', 'angles', 'radius', 'length'),
            **{shape: ('center', 'angles', 'dimensions', 'length')
               for shape in SECTIONS},
            HOLE: ('center', 'angles', 'dimensions'),
            PATTERN: ('counts', 'spacing', 'angles')}


def edit(geometry: Any, solid_id: int, **changes) -> None:
    """Change a solid after it is made, in place (Brandon, 2026-10-11:
    move it, size it, reshape it). A block's center, angles, widths and
    corner, a cylinder's center, angles, radius and length, a section's
    center, angles, dimensions and length, a hole's center, angles and
    dimensions, a pattern's counts, spacing and angles, and any solid's
    name; the rest are kept. It keeps its id, so a boolean it is
    part of takes the change: a hole made larger is a larger hole in the
    plate it was subtracted from. A mesh already made from it is not
    changed; Mesh Solids again meshes the change.

    Parameters
    ----------
    geometry : Geometry
        Whose solid; changed in place.
    solid_id : int
        Which.
    **changes
        The numbers to change, as `add_block`, `add_cylinder` and
        `add_section` take them — a hole's as its `HOLE_DIMENSIONS`, a
        depth of zero through — and `name`.

    Raises
    ------
    ValueError
        For a number its kind does not have — a boolean's shape is its
        solids', changed one by one, and it is moved whole by `move` —
        or one an add would refuse.
    """
    item = solid(geometry, solid_id)
    allowed = EDITABLE.get(item.kind, ()) + ('name',)
    wrong = [key for key in changes if key not in allowed]
    if wrong:
        if item.kind in OPERATIONS:
            raise ValueError(f'solid {solid_id} is a {item.kind}: change the '
                             'solids it is made of, or move it whole')
        from .fem import with_article

        raise ValueError(f'{with_article(item.kind)} has no {wrong[0]}')
    now = {'name': item.name, 'center': item.center, 'angles': item.angles,
           'widths': item.widths, 'corner': item.corner, 'radius': item.radius,
           'length': item.length, 'counts': item.counts, 'spacing': item.widths,
           'dimensions': item.dimensions, **changes}
    if item.kind == 'block':
        changed = _block(item.id, now['name'], now['center'], now['widths'],
                         now['angles'], now['corner'])
    elif item.kind == 'cylinder':
        changed = _cylinder(item.id, now['name'], now['center'], now['radius'],
                            now['length'], now['angles'])
    elif item.kind in SHAPED:
        changed = _shaped(item.id, now['name'], item.kind, now['center'],
                          now['dimensions'], now['length'], now['angles'])
    elif item.kind == PATTERN:
        changed = _pattern(item.id, now['name'], item.operands, now['counts'],
                           now['spacing'], now['angles'])
    else:
        changed = replace(item, name=now['name'])
    geometry.solids[geometry.solids.index(item)] = changed


def move(geometry: Any, ids, offset=(0.0, 0.0, 0.0),
         turn=(0.0, 0.0, 0.0)) -> int:
    """Move solids whole — a part with every solid it is made of — turned
    by `turn` about the middle of their extent, then slid by `offset`;
    how many primitives moved. A boolean has no place of its
    own to change, so its solids are moved, each center turned about
    that middle and each turned as much: the part is the same shape,
    elsewhere.

    Parameters
    ----------
    geometry : Geometry
        Whose solids; changed in place.
    ids : sequence of int
        The solids, by id.
    offset : array_like, default (0, 0, 0)
        How far, as the geometry holds its coordinates.
    turn : array_like, default (0, 0, 0)
        Degrees about the fixed X, then Y, then Z axes, as a solid's
        angles are.
    """
    from ..rotate import angles_of, frame_from_angles

    ids = [int(i) for i in ids]
    if not ids:
        raise ValueError('pick a solid to move')
    offset = np.asarray(_numbers(offset, 3, 'an offset'))
    turn = _numbers(turn, 3, 'a turn')
    low, high = zip(*(_box(geometry, i) for i in ids))
    middle = (np.min(low, axis=0) + np.max(high, axis=0)) / 2.0
    # a turned vector, as a row: v @ R^T, and frame_from_angles keeps R^T
    rows = frame_from_angles(turn)[:3]

    def turned(item):
        frame = np.zeros((4, 3))
        frame[:3] = _frame(item) @ rows
        return tuple(float(a) for a in angles_of(frame))

    # the solids as stored, each once: a pattern's copies are where its
    # originals are, along its own axes, so the originals move and the
    # pattern's axes turn with them
    stored, stack = {}, list(ids)
    while stack:
        item = solid(geometry, stack.pop())
        if item.id not in stored:
            stored[item.id] = item
            stack.extend(item.operands)
    moved = 0
    for item in stored.values():
        if item.kind in PRIMITIVES:
            center = np.asarray(item.center)
            angles = item.angles
            if any(turn):
                center = middle + (center - middle) @ rows
                angles = turned(item)
            changed = replace(item, center=tuple(float(c) for c in center + offset),
                              angles=angles)
            moved += 1
        elif item.kind == PATTERN and any(turn):
            changed = replace(item, angles=turned(item))
        else:
            continue
        geometry.solids[geometry.solids.index(item)] = changed
    return moved


def combine(geometry: Any, operation: str, ids, name: str = '') -> int:
    """Add a boolean of solids already there: their union, the first
    with the rest subtracted from it, or what they share. Each solid
    goes into one boolean only — the operands are the new solid's now,
    and the table lists only the solids nothing else uses
    (`top_level`).

    Parameters
    ----------
    geometry : Geometry
        Where it goes; changed in place.
    operation : str
        One of `OPERATIONS`.
    ids : sequence of int
        Two solids or more, by id, the first first.
    name : str, optional
        What to call it; the first solid's name when not given, so a
        plate named 'frame' with its holes subtracted is still 'frame',
        and so is the element group it meshes into.

    Returns
    -------
    int
        The new solid's id.
    """
    if operation not in OPERATIONS:
        raise ValueError(f'{operation!r} is not one of {OPERATIONS}')
    ids = tuple(int(i) for i in ids)
    if len(ids) < 2 or len(set(ids)) < len(ids):
        raise ValueError('a boolean takes two solids or more, each once')
    there = {s.id for s in geometry.solids}
    missing = [i for i in ids if i not in there]
    if missing:
        raise ValueError(f'no solid {missing[0]}')
    used = used_by(geometry)
    taken = [i for i in ids if i in used]
    if taken:
        raise ValueError(f'solid {taken[0]} is already part of solid '
                         f'{used[taken[0]]}')
    name = name or solid(geometry, ids[0]).name
    made = Solid(_next_id(geometry), name, operation, operands=ids)
    geometry.solids.append(made)
    return made.id


def _checked(geometry: Any, ids) -> set[int]:
    ids = {int(i) for i in ids}
    missing = sorted(ids - {s.id for s in geometry.solids})
    if missing:
        raise ValueError(f'no solid {missing[0]}')
    used = used_by(geometry)
    held = sorted(i for i in ids if i in used and used[i] not in ids)
    if held:
        raise ValueError(f'solid {held[0]} is part of solid {used[held[0]]}; '
                         'take that apart first')
    return ids


def delete(geometry: Any, ids) -> int:
    """Delete solids, each with everything it is made of: a plate with
    its holes cut goes whole, the cylinders that cut them with it. One a
    boolean still uses is refused, named — take that boolean apart
    first (`separate`).

    Parameters
    ----------
    geometry : Geometry
        Whose solids; changed in place.
    ids : sequence of int
        The solids, by id.

    Returns
    -------
    int
        How many solids went, those inside included.
    """
    going = set()
    stack = list(_checked(geometry, ids))
    while stack:
        current = stack.pop()
        if current not in going:
            going.add(current)
            stack.extend(solid(geometry, current).operands)
    geometry.solids = [s for s in geometry.solids if s.id not in going]
    return len(going)


def separate(geometry: Any, ids) -> int:
    """Take booleans apart: each one goes, and its operands stand as
    parts again — a subtraction made by mistake undone. A primitive has
    nothing to take apart and is refused.

    Parameters
    ----------
    geometry : Geometry
        Whose solids; changed in place.
    ids : sequence of int
        The booleans, by id.

    Returns
    -------
    int
        How many solids stand again.
    """
    ids = _checked(geometry, ids)
    plain = sorted(i for i in ids if solid(geometry, i).kind in PRIMITIVES)
    if plain:
        raise ValueError(f'solid {plain[0]} is a {solid(geometry, plain[0]).kind}, '
                         'with nothing to take apart')
    freed = sum(len(solid(geometry, i).operands) for i in ids)
    geometry.solids = [s for s in geometry.solids if s.id not in ids]
    return freed


#: past this many copies a pattern is refused rather than made
PATTERN_LIMIT = 10_000


def pattern(geometry: Any, ids, counts, spacing) -> int:
    """A pattern of solids: `counts` copies along the geometry's X, Y and
    Z, `spacing` apart, the originals the first of each — 13 holes along
    a rail are one hole and (13, 1, 1) at the insert pitch. One solid,
    made of the originals as a boolean is (Brandon, 2026-10-11): a hole
    changed changes every copy, and the pattern's counts and spacing are
    changed as one (`edit`). It takes its first solid's name, as a
    boolean does. A pattern of a pattern is the grid of a grid: the row
    of 13, two rows across the rails.

    Parameters
    ----------
    geometry : Geometry
        Whose solids; changed in place.
    ids : sequence of int
        The solids to copy, by id: solids nothing else uses.
    counts : sequence of int
        How many along X, Y and Z, the original included; each one or
        more.
    spacing : sequence of float
        How far apart along X, Y and Z, as the geometry holds its
        coordinates; zero along an axis with one only.

    Returns
    -------
    int
        The pattern's id.
    """
    ids = tuple(int(i) for i in ids)
    if not ids or len(set(ids)) < len(ids):
        raise ValueError('a pattern takes one solid or more, each once')
    used = used_by(geometry)
    for i in ids:
        solid(geometry, i)
        if i in used:
            raise ValueError(f'solid {i} is part of solid {used[i]}; pattern '
                             'that one')
    made = _pattern(_next_id(geometry), solid(geometry, ids[0]).name, ids, counts,
                    spacing, (0.0, 0.0, 0.0))
    geometry.solids.append(made)
    return made.id


def _pattern(solid_id: int, name: str, ids, counts, spacing, angles) -> Solid:
    """A pattern's row, its numbers checked: what making one and
    changing one both refuse."""
    counts = tuple(int(c) for c in np.asarray(counts).ravel())
    spacing = _numbers(spacing, 3, 'spacing')
    if len(counts) != 3 or min(counts) < 1:
        raise ValueError('a pattern takes one or more along each of X, Y and Z')
    for axis, (count, step) in enumerate(zip(counts, spacing)):
        if count > 1 and step == 0.0:
            raise ValueError(f'more than one along {"XYZ"[axis]} needs a '
                             'spacing there, or the copies sit on each other')
    total = len(ids) * (counts[0] * counts[1] * counts[2] - 1)
    if total > PATTERN_LIMIT:
        raise ValueError(f'{total:,} copies: more than {PATTERN_LIMIT:,}')
    return Solid(solid_id, name, PATTERN, angles=_numbers(angles, 3, 'angles'),
                 widths=spacing, operands=tuple(ids), counts=counts)


def _offsets(item: Solid) -> np.ndarray:
    """(n, 3): where a pattern puts each copy of its solids, the
    original's (zeros) first."""
    grid = np.stack(np.meshgrid(*(np.arange(c) for c in item.counts), indexing='ij'),
                    axis=-1).reshape(-1, 3)
    return (grid * np.asarray(item.widths)) @ _frame(item)


def used_by(geometry: Any) -> dict[int, int]:
    """{solid: the boolean it is part of}, for every solid that is one's
    operand."""
    return {operand: s.id for s in geometry.solids for operand in s.operands}


def top_level(geometry: Any) -> list[Solid]:
    """The solids no boolean uses: the parts as they stand, the ones a
    person sees, edits and meshes."""
    used = used_by(geometry)
    return [s for s in geometry.solids if s.id not in used]


def solid(geometry: Any, solid_id: int) -> Solid:
    """The solid with this id."""
    for s in geometry.solids:
        if s.id == int(solid_id):
            return s
    raise KeyError(f'no solid {solid_id}')


# ---- the kernel ------------------------------------------------------------

def require_kernel():
    """The OCP bindings, or the sentence that says how to get them."""
    try:
        import OCP
    except ImportError:
        raise ValueError(
            'solids and STEP/IGES need the OpenCASCADE kernel, which is '
            "not installed — pip install 'visualdynamics[step]' (the "
            'packaged application already carries it)') from None
    return OCP


def _placed(shape, item: Solid):
    """`shape`, built about the origin along the geometry's axes, turned
    by the solid's angles and moved to its center."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf

    from ..rotate import frame_from_angles

    frame = frame_from_angles(item.angles, item.center)
    turn = frame[:3].T                     # columns: the solid's own axes
    trsf = gp_Trsf()
    trsf.SetValues(*turn[0], frame[3][0], *turn[1], frame[3][1],
                   *turn[2], frame[3][2])
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def shape(geometry: Any, solid_id: int):
    """The solid as OpenCASCADE's shape: a primitive built and placed,
    a boolean computed from its operands' shapes, as the geometry holds
    its coordinates.

    Parameters
    ----------
    geometry : Geometry
        The geometry whose solids table holds it.
    solid_id : int
        The solid, by id.

    Returns
    -------
    OCP.TopoDS.TopoDS_Shape
    """
    require_kernel()
    from OCP.BRepAlgoAPI import (
        BRepAlgoAPI_Common,
        BRepAlgoAPI_Cut,
        BRepAlgoAPI_Fuse,
    )

    item = solid(geometry, solid_id)
    if item.kind in BASIC:
        return _basic_shape(item)
    if item.kind in SHAPED:
        # the uncut pieces made one, then the cut taken from them
        pieces = _pieces(geometry, item)
        kept = [_basic_shape(p) for cutting, p in pieces if not cutting]
        cut = [_basic_shape(p) for cutting, p in pieces if cutting]
        result = kept[0]
        for operand, algorithm in ([(k, BRepAlgoAPI_Fuse) for k in kept[1:]]
                                   + [(c, BRepAlgoAPI_Cut) for c in cut]):
            step = algorithm(result, operand)
            if not step.IsDone():
                raise ValueError(f'the {item.kind} of solid {item.id} failed '
                                 'in the kernel')
            result = step.Shape()
        return result
    if item.kind == PATTERN:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.gp import gp_Trsf, gp_Vec

        originals = [shape(geometry, o) for o in item.operands]
        copies = []
        for offset in _offsets(item):
            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(*(float(v) for v in offset)))
            copies.extend(BRepBuilderAPI_Transform(original, trsf, True).Shape()
                          for original in originals)
        operands, algorithm = copies, BRepAlgoAPI_Fuse
    else:
        operands = [shape(geometry, o) for o in item.operands]
        algorithm = {'union': BRepAlgoAPI_Fuse, 'subtract': BRepAlgoAPI_Cut,
                     'intersect': BRepAlgoAPI_Common}[item.kind]
    result = operands[0]
    for operand in operands[1:]:
        step = algorithm(result, operand)
        if not step.IsDone():
            raise ValueError(f'the {item.kind} of solid {item.id} failed in '
                             'the kernel')
        result = step.Shape()
    return result


def _basic_shape(item: Solid):
    """A block or a cylinder as OpenCASCADE's shape, placed."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    if item.kind == 'cylinder':
        axis = gp_Ax2(gp_Pnt(0.0, 0.0, -item.length / 2), gp_Dir(0.0, 0.0, 1.0))
        built = BRepPrimAPI_MakeCylinder(axis, item.radius, item.length).Shape()
        return _placed(built, item)
    wx, wy, wz = item.widths
    c = item.corner
    if c <= 0.0:
        built = BRepPrimAPI_MakeBox(gp_Pnt(-wx / 2, -wy / 2, -wz / 2),
                                    wx, wy, wz).Shape()
        return _placed(built, item)
    # rounded: a box narrowed by the radius each way, and a cylinder at
    # each corner, made one
    pieces = [BRepPrimAPI_MakeBox(gp_Pnt(-a / 2, -b / 2, -wz / 2), a, b,
                                  wz).Shape()
              for a, b in ((wx - 2 * c, wy), (wx, wy - 2 * c)) if a > 0 and b > 0]
    for sx in (-1, 1):
        for sy in (-1, 1):
            axis = gp_Ax2(gp_Pnt(sx * (wx / 2 - c), sy * (wy / 2 - c), -wz / 2),
                          gp_Dir(0.0, 0.0, 1.0))
            pieces.append(BRepPrimAPI_MakeCylinder(axis, c, wz).Shape())
    built = pieces[0]
    for piece in pieces[1:]:
        built = BRepAlgoAPI_Fuse(built, piece).Shape()
    return _placed(built, item)


def volume(geometry: Any, solid_id: int) -> float:
    """The solid's volume, in the cube of the geometry's length unit (m³
    once units are defined).

    Parameters
    ----------
    geometry : Geometry
        The geometry whose solids table holds it.
    solid_id : int
        The solid, by id.

    Returns
    -------
    float
    """
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape(geometry, solid_id), props)
    return float(props.Mass())


def shape_triangles(item, deflection: float = DEFLECTION) -> np.ndarray:
    """(n, 3, 3) corners of the triangles OpenCASCADE tessellates a
    shape's faces into, `deflection` a fraction of its size: what draws
    a solid, and what the STEP importer reads a part as.

    Parameters
    ----------
    item : OCP.TopoDS.TopoDS_Shape
        The shape.
    deflection : float, optional
        How far a triangle may stand off the surface, as a fraction of
        the shape's size.

    Returns
    -------
    numpy.ndarray
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRep import BRep_Tool
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    # the 7.9 binding suffixes a static that shares its name with a
    # method; 8.0 drops the suffix for this one and keeps it for the
    # rest (2026-09-13, Linux on 8.0.1 beside the Mac on 7.9.3)
    _as_face = getattr(TopoDS, 'Face_s', None) or TopoDS.Face

    box = Bnd_Box()
    BRepBndLib.Add_s(item, box)
    size = max(box.CornerMax().XYZ().Subtracted(
        box.CornerMin().XYZ()).Modulus(), 1e-9)
    BRepMesh_IncrementalMesh(item, size * deflection, False, 0.5, True)
    triangles: list[np.ndarray] = []
    walker = TopExp_Explorer(item, TopAbs_FACE)
    while walker.More():
        face = _as_face(walker.Current())
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is not None:
            matrix = location.Transformation()
            points = np.asarray(
                [[mesh.Node(i + 1).X(), mesh.Node(i + 1).Y(),
                  mesh.Node(i + 1).Z()] for i in range(mesh.NbNodes())])
            rows = np.asarray(
                [[matrix.Value(r, c) for c in (1, 2, 3, 4)]
                 for r in (1, 2, 3)])
            points = points @ rows[:, :3].T + rows[:, 3]
            faces = np.asarray(
                [mesh.Triangle(i + 1).Get() for i in
                 range(mesh.NbTriangles())], dtype=np.int64) - 1
            if face.Orientation() == TopAbs_REVERSED:
                faces = faces[:, ::-1]
            triangles.append(points[faces])
        walker.Next()
    return (np.concatenate(triangles) if triangles
            else np.empty((0, 3, 3)))


# ---- meshing (2026-10-11) --------------------------------------------------
#
# Mesh Solids turns parts into bricks. Automatic hex meshing of any solid
# is an unsolved problem; every part built so far is an extrusion — an
# outline pushed through a thickness — and that is meshed reliably the
# way Add Block cuts its holes, generalized: a grid laid over the parts,
# cut at every flat face of every block so those fall on grid lines, a
# brick kept where its center is inside a part, and the nodes between a
# cylinder's inside and a part's material pulled onto its circle. Whether
# a point is inside is read off the recipes, not asked of the kernel: a
# block's box and a cylinder's radius, combined as the booleans say —
# exact, and fast enough to preview while the size is typed.

#: past this many bricks the grid is refused rather than built
MESH_LIMIT = 2_000_000


def _frame(item: Solid) -> np.ndarray:
    """The solid's own axes as rows, in the geometry's frame."""
    from ..rotate import frame_from_angles

    return frame_from_angles(item.angles)[:3]


def _pieces(geometry: Any, item: Solid) -> list[tuple[bool, Solid]]:
    """(cut, piece) for each block and cylinder a section or a hole is
    made of, placed where it stands: its material the uncut pieces less
    the cut ones — a tube's bore, a rectangular tube's inside. Each
    piece carries the shape's id, so a refusal names the shape."""
    rows = _frame(item)
    middle = np.asarray(item.center)

    def at(x, y, z):
        return tuple(float(c) for c in middle + np.array([x, y, z]) @ rows)

    def block(x, y, wx, wy):
        return Solid(item.id, item.name, 'block', at(x, y, 0.0), item.angles,
                     (float(wx), float(wy), float(item.length)))

    def cylinder(radius, z=0.0, length=None):
        return Solid(item.id, item.name, 'cylinder', at(0.0, 0.0, z), item.angles,
                     radius=float(radius),
                     length=float(item.length if length is None else length))

    d = item.dimensions
    if item.kind == HOLE:
        diameter, depth, wide, deep = d
        depth = depth or _through(geometry, item)
        pieces = [(False, cylinder(diameter / 2.0, -depth / 2.0, depth))]
        if wide:
            pieces.append((False, cylinder(wide / 2.0, -deep / 2.0, deep)))
        return pieces
    if item.kind in ('I-beam', 'channel', 'tee'):
        depth, flange, tf, tw = d
        if item.kind == 'tee':
            return [(False, block(0.0, (depth - tf) / 2.0, flange, tf)),
                    (False, block(0.0, -tf / 2.0, tw, depth - tf))]
        web_x = 0.0 if item.kind == 'I-beam' else -(flange - tw) / 2.0
        return [(False, block(0.0, (depth - tf) / 2.0, flange, tf)),
                (False, block(0.0, -(depth - tf) / 2.0, flange, tf)),
                (False, block(web_x, 0.0, tw, depth - 2.0 * tf))]
    if item.kind == 'angle':
        long_leg, short_leg, t = d
        return [(False, block(0.0, -(short_leg - t) / 2.0, long_leg, t)),
                (False, block(-(long_leg - t) / 2.0, t / 2.0, t, short_leg - t))]
    if item.kind == 'rectangular tube':
        width, height, wall = d
        return [(False, block(0.0, 0.0, width, height)),
                (True, block(0.0, 0.0, width - 2.0 * wall, height - 2.0 * wall))]
    outer, wall = d
    return [(False, cylinder(outer / 2.0)), (True, cylinder(outer / 2.0 - wall))]


def _through(geometry: Any, item: Solid) -> float:
    """How deep a hole with no depth runs: from where it enters out past
    the far side of the part it is cut from — the first solid of the
    subtraction it is taken by, a pattern of it included — so a plate
    made thicker keeps its hole through. Cut from nothing yet, it is
    drawn a diameter deep."""
    used = used_by(geometry)
    current = item.id
    while current in used:
        parent = solid(geometry, used[current])
        if parent.kind == 'subtract' and parent.operands[0] != current:
            low, high = _box(geometry, parent.operands[0])
            corners = np.array(np.meshgrid(*zip(low, high), indexing='ij')).reshape(3, -1).T
            # along its own -Z, to the farthest corner, and a hundredth of
            # the part past it, so the kernel meets no face on a face
            reach = float(((np.asarray(item.center) - corners) @ _frame(item)[2]).max())
            return max(reach, 0.0) + 0.01 * float(np.linalg.norm(high - low))
        current = parent.id
    return item.dimensions[0]


def _leaves(geometry: Any, solid_id: int) -> list[Solid]:
    """The blocks and cylinders a solid is made of, where they stand: a
    section's or a hole's pieces, a pattern's copies each its own, moved
    — what meshing reads (its grid cut at each copy's faces, a ring
    about each copy's circle)."""
    item = solid(geometry, solid_id)
    if item.kind in BASIC:
        return [item]
    if item.kind in SHAPED:
        return [piece for _cut, piece in _pieces(geometry, item)]
    leaves = [leaf for o in item.operands for leaf in _leaves(geometry, o)]
    if item.kind != PATTERN:
        return leaves
    return [replace(leaf, center=tuple(float(c) for c in np.asarray(leaf.center) + offset))
            for offset in _offsets(item) for leaf in leaves]


def contains(geometry: Any, solid_id: int, points) -> np.ndarray:
    """Which of `points` lie inside the solid, read off its recipe: a
    block's box, a cylinder's radius and length, combined as its
    booleans say. A point on a face counts as inside.

    Parameters
    ----------
    geometry : Geometry
        The geometry whose solids table holds it.
    solid_id : int
        The solid, by id.
    points : array_like
        (n, 3), as the geometry holds its coordinates.

    Returns
    -------
    numpy.ndarray of bool
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    item = solid(geometry, solid_id)
    if item.kind in BASIC:
        return _basic_contains(item, points)
    if item.kind in SHAPED:
        return _pieces_contain(_pieces(geometry, item), points)
    if item.kind == PATTERN:
        # a section's or a hole's pieces worked out once, not once a copy:
        # a through hole's depth is asked of the whole table (`_through`),
        # and a pattern of 400 holes meshed three times slower than one of
        # cylinders that way
        tests = []
        for o in item.operands:
            if solid(geometry, o).kind in SHAPED:
                pieces = _pieces(geometry, solid(geometry, o))
                tests.append(lambda at, pieces=pieces: _pieces_contain(pieces, at))
            else:
                tests.append(lambda at, o=o: contains(geometry, o, at))
        return np.any([test(points - offset) for offset in _offsets(item)
                       for test in tests], axis=0)
    inside = [contains(geometry, o, points) for o in item.operands]
    if item.kind == 'union':
        return np.any(inside, axis=0)
    if item.kind == 'intersect':
        return np.all(inside, axis=0)
    return inside[0] & ~np.any(inside[1:], axis=0)


def _basic_contains(item: Solid, points: np.ndarray) -> np.ndarray:
    """Which points lie inside a block or a cylinder, a face counted."""
    local = (points - np.asarray(item.center)) @ _frame(item).T
    slack = 1e-9 * (max(item.widths) + item.radius + item.length)
    if item.kind == 'block':
        half = np.asarray(item.widths) / 2.0 + slack
        inside = np.all(np.abs(local) <= half, axis=1)
        if item.corner > 0.0:
            # in a corner's square, only within its radius of the arc's
            # center
            arc = np.asarray(item.widths[:2]) / 2.0 - item.corner
            beyond = np.abs(local[:, :2]) - arc
            cornered = np.all(beyond > 0.0, axis=1)
            inside &= ~cornered | (np.hypot(*beyond.T) <= item.corner + slack)
        return inside
    return ((np.hypot(local[:, 0], local[:, 1]) <= item.radius + slack)
            & (np.abs(local[:, 2]) <= item.length / 2.0 + slack))


def _pieces_contain(pieces, points: np.ndarray) -> np.ndarray:
    """Which points lie in a section's or a hole's material: in an uncut
    piece and in no cut one."""
    inside = np.any([_basic_contains(p, points) for cut, p in pieces if not cut], axis=0)
    for cut, p in pieces:
        if cut:
            inside &= ~_basic_contains(p, points)
    return inside


def _basic_box(item: Solid) -> tuple[np.ndarray, np.ndarray]:
    """A block's or a cylinder's extent along the geometry's axes."""
    axes = _frame(item)
    if item.kind == 'block':
        half = np.abs(axes.T) @ (np.asarray(item.widths) / 2.0)
    else:
        along = np.abs(axes[2]) * item.length / 2.0
        across = item.radius * np.sqrt(np.clip(1.0 - axes[2] ** 2, 0.0, 1.0))
        half = along + across
    center = np.asarray(item.center)
    return center - half, center + half


def _box(geometry: Any, solid_id: int) -> tuple[np.ndarray, np.ndarray]:
    """The solid's extent along the geometry's axes, from its recipe."""
    item = solid(geometry, solid_id)
    if item.kind in BASIC:
        return _basic_box(item)
    if item.kind in SHAPED:
        boxes = [_basic_box(p) for cut, p in _pieces(geometry, item) if not cut]
        return np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
    boxes = [_box(geometry, o) for o in item.operands]
    if item.kind == PATTERN:
        offsets = _offsets(item)
        return (np.min([b[0] for b in boxes], axis=0) + offsets.min(axis=0),
                np.max([b[1] for b in boxes], axis=0) + offsets.max(axis=0))
    if item.kind == 'union':
        return (np.min([b[0] for b in boxes], axis=0),
                np.max([b[1] for b in boxes], axis=0))
    if item.kind == 'intersect':
        return (np.max([b[0] for b in boxes], axis=0),
                np.min([b[1] for b in boxes], axis=0))
    return boxes[0]


def extrusion_axis(geometry: Any, ids) -> int:
    """The axis, 0, 1 or 2, the parts are extrusions along — every block
    square to the geometry's axes and every cylinder's axis along it —
    or the sentence that says which solid is not.

    Parameters
    ----------
    geometry : Geometry
        Whose solids.
    ids : sequence of int
        The parts, by id.

    Returns
    -------
    int
    """
    leaves = [leaf for i in ids for leaf in _leaves(geometry, i)]
    for leaf in leaves:
        axes = np.abs(_frame(leaf))
        if not np.allclose(np.sort(axes, axis=1), [[0, 0, 1]] * 3, atol=1e-9):
            raise ValueError(
                f'solid {leaf.id} is turned off the geometry\'s axes: Mesh '
                'Solids meshes extrusions along X, Y or Z')
    along = {int(np.argmax(np.abs(_frame(leaf)[2])))
             for leaf in leaves if leaf.kind == 'cylinder' or leaf.corner > 0.0}
    if len(along) > 1:
        raise ValueError('the cylinders and rounded corners run along '
                         'different axes: Mesh Solids meshes an extrusion '
                         'along one')
    if along:
        return along.pop()
    # blocks alone are an extrusion along any axis: the thinnest one
    low, high = zip(*(_box(geometry, i) for i in ids))
    return int(np.argmin(np.max(high, axis=0) - np.min(low, axis=0)))


def mesh_size(geometry: Any, ids) -> float:
    """The element size Mesh Solids opens on: half the thinnest feature —
    the shortest span between two flat faces, or the narrowest cylinder
    across — so every feature is two bricks across.

    Parameters
    ----------
    geometry : Geometry
        Whose solids.
    ids : sequence of int
        The parts, by id.

    Returns
    -------
    float
    """
    cuts = _cuts(geometry, ids)
    spans = [float(b - a) for axis in cuts for a, b in pairwise(axis)]
    spans += [2 * (leaf.radius if leaf.kind == 'cylinder' else leaf.corner)
              for i in ids for leaf in _leaves(geometry, i)
              if leaf.kind == 'cylinder' or leaf.corner > 0.0]
    spans = [s for s in spans if s > 0]
    return 0.5 * min(spans) if spans else 1.0


def _cuts(geometry: Any, ids) -> list[np.ndarray]:
    """Per axis, where the grid must have a line: every block face, every
    cylinder's end along its axis, and the parts' extent, kept within
    that extent."""
    low, high = zip(*(_box(geometry, i) for i in ids))
    low, high = np.min(low, axis=0), np.max(high, axis=0)
    cuts: list[set] = [{low[k], high[k]} for k in range(3)]
    for i in ids:
        for leaf in _leaves(geometry, i):
            center = np.asarray(leaf.center)
            if leaf.kind == 'block':
                half = np.abs(_frame(leaf).T) @ (np.asarray(leaf.widths) / 2.0)
                for k in range(3):
                    cuts[k].update((center[k] - half[k], center[k] + half[k]))
            else:
                k = int(np.argmax(np.abs(_frame(leaf)[2])))
                cuts[k].update((center[k] - leaf.length / 2,
                                center[k] + leaf.length / 2))
    size = float(np.max(high - low)) or 1.0
    out = []
    for k in range(3):
        kept = sorted(c for c in cuts[k]
                      if low[k] - 1e-9 * size <= c <= high[k] + 1e-9 * size)
        merged = [kept[0]]
        for c in kept[1:]:
            if c - merged[-1] > 1e-9 * size:
                merged.append(c)
        out.append(np.asarray(merged))
    return out


def extrusion_mesh(geometry: Any, ids, size: float,
                   parts_apart: bool = False) -> tuple[Any, dict]:
    """The parts as eight-node bricks, ready to join to a geometry
    (`mesh.join`): one element group per part, named for it, parts that
    touch sharing their nodes there — or, `parts_apart`, each keeping its
    own, on the same grid, so they lie on each other's for bolts to be
    merged at (the BARC's box, channels and beam, 2026-10-11: meshed one
    at a time, their grids did not meet at a quarter inch).

    The grid is cut at every flat face of every block along every axis,
    each span divided into the whole number of bricks nearest `size`,
    so flat faces fall on grid lines; a brick goes to the part whose
    recipe holds its center (`contains`) — the later one where parts
    overlap, as the Frame-Wing's steel screws take the inserts they
    are threaded into — none to the void; and
    a hole, a boss or an insert is round rather than stair-stepped:
    about each circle with room a ring mesh replaces the grid, its nodes
    on the circle evenly spaced (`_plane`), and a circle without room,
    or a rounded corner's arc, has the nodes between its inside and the
    material beside it pulled radially onto it, a pull that would turn
    a brick inside out not made (`_round_onto_circles`). Refused,
    with the solid named, when a part is not an extrusion along X, Y or
    Z (`extrusion_axis`).

    Parameters
    ----------
    geometry : Geometry
        Whose solids.
    ids : sequence of int
        The parts, by id; each a solid nothing else uses. Where two
        overlap, the later takes the bricks.
    size : float
        The element size aimed at, as the geometry holds its
        coordinates.
    parts_apart : bool, default False
        Each part keeps its own nodes where it touches another.

    Returns
    -------
    (Geometry, dict)
        The bricks, in the geometry's length unit; and what was made:
        'bricks', 'nodes', 'axis' (0, 1 or 2), 'rounded' (nodes pulled
        onto a circle), 'rings' (the rings laid, one about each center
        with room), 'shortest' (the shortest brick edge).
    """
    from .geometry import Geometry

    ids = [int(i) for i in ids]
    if not ids:
        raise ValueError('pick a solid to mesh')
    used = used_by(geometry)
    inside_another = [i for i in ids if i in used]
    if inside_another:
        raise ValueError(f'solid {inside_another[0]} is part of solid '
                         f'{used[inside_another[0]]}; mesh that one')
    if not float(size) > 0.0:
        raise ValueError('an element size above zero')
    axis = extrusion_axis(geometry, ids)
    # u, v, w right-handed, w along the extrusion
    order = [(axis + 1) % 3, (axis + 2) % 3, axis]
    cuts = _cuts(geometry, ids)
    lines = []
    for k in order:
        edges = [cuts[k][0]]
        for a, b in pairwise(cuts[k]):
            count = max(1, round((b - a) / size))
            edges.extend(np.linspace(a, b, count + 1)[1:])
        lines.append(np.asarray(edges))
    nu, nv, nw = (len(line) for line in lines)
    cells = (nu - 1) * (nv - 1) * (nw - 1)
    if cells > MESH_LIMIT:
        raise ValueError(f'{cells:,} bricks at that size: use a larger one')
    xyz, conn, groups, made = _mesh_grid(
        lines, order, size, lambda points: _owners(geometry, ids, points),
        _circles(geometry, ids, axis))
    if not len(conn):
        raise ValueError('the parts hold nothing to mesh')
    if parts_apart:
        # a node for each part that uses it: one grid, no node shared
        stride = int(groups.max()) + 1
        key = (conn.ravel() - 1) * stride + np.repeat(groups, 8)
        kept, inverse = np.unique(key, return_inverse=True)
        xyz, conn = xyz[kept // stride], inverse.reshape(-1, 8) + 1
    names, present = [], []
    for slot, part in enumerate(ids, start=1):
        if (groups == slot).any():
            item = solid(geometry, part)
            names.append(item.name.strip() or f'{item.kind.title()} {item.id}')
            present.append(slot)
    part = Geometry(node_id=np.arange(1, len(xyz) + 1), node_xyz=xyz,
                    elem_conn=list(conn), elem_type=[115] * len(conn),
                    elem_group=[int(g) for g in groups], group_id=present,
                    group_name=names, length_unit=geometry.length_unit)
    return part, {'bricks': len(conn), 'nodes': len(xyz), 'axis': axis, **made}


def meshed_bricks(geometry: Any, names) -> np.ndarray:
    """The ids of the bricks in the element groups named `names`: what
    meshing those parts again replaces (`Project.mesh_solids`, and the
    pane's reading before Mesh is pressed).

    Parameters
    ----------
    geometry : Geometry
        Whose elements.
    names : iterable of str
        The element groups' names; an empty one names none.

    Returns
    -------
    numpy.ndarray of int
    """
    names = {name for name in names if name}
    wanted = [int(g) for g, name in zip(geometry.group_id, geometry.group_name)
              if name in names]
    mask = (np.isin(np.asarray(geometry.elem_group, dtype=int), wanted)
            & (np.asarray(geometry.elem_type, dtype=int) == 115))
    return np.asarray(geometry.elem_id, dtype=int)[mask]


def _mesh_grid(lines, order, size, held, circles):
    """Bricks on a grid swept along w, rings laid about the circles:
    the one mesher behind Mesh Solids and Add Block's holes.

    `lines` are the grid's lines along u, v and w, `order` the axes
    (x 0, y 1, z 2) u, v and w are, right-handed. `held(points)` says
    which part holds each point (n, 3): 0 none, else a part's number.
    `circles` are (center, radius, low, high, whole), round w, the
    first two of the center read along u and v: a whole circle is
    ringed where there is room (`_plane`), the rest pulled round from
    the grid (`_round_onto_circles`).

    Returns
    -------
    (xyz, conn, groups, made)
        The nodes that bricks use, (n, 3); each brick's eight, numbered
        from 1 in that order; each brick's part; and 'rounded', 'rings'
        and 'shortest' as `extrusion_mesh` says them. No bricks if
        nothing holds any.
    """
    from .geometry import _CORNER_EDGES

    nu, nv, nw = (len(line) for line in lines)
    mids = [(line[:-1] + line[1:]) / 2.0 for line in lines]
    cu, cv, cw = np.meshgrid(*mids, indexing='ij')
    centers = np.zeros((cu.size, 3))
    for slot, k in enumerate(order):
        centers[:, k] = (cu, cv, cw)[slot].ravel()
    owner = held(centers).reshape(nu - 1, nv - 1, nw - 1)
    if not owner.any():
        return (np.zeros((0, 3)), np.zeros((0, 8), dtype=int), np.zeros(0, dtype=int),
                {'rounded': 0, 'rings': 0, 'shortest': 0.0})
    # the plane's mesh, rings about the circles swapped in for the grid
    # there, then swept along w: node (p, k) is p * nw + k, so where
    # there is no ring the numbering is the grid's own
    plane, quads, known, on_circle, left, ringed = _plane(lines, order, size, owner,
                                                          circles, held)
    xyz = np.zeros((len(plane) * nw, 3))
    xyz[:, order[0]] = np.repeat(plane[:, 0], nw)
    xyz[:, order[1]] = np.repeat(plane[:, 1], nw)
    xyz[:, order[2]] = np.tile(lines[2], len(plane))
    base = quads[:, None, :] * nw + np.arange(nw - 1)[None, :, None]
    # every brick, empty ones too: a hole's inside is what its rim is
    # found against
    corners = np.concatenate([base, base + 1], axis=2).reshape(-1, 8)
    if len(corners) > MESH_LIMIT:
        raise ValueError(f'{len(corners):,} bricks at that size: use a larger one')
    groups = known.ravel().copy()
    rounded = 0
    if left:
        # a brick's center, its long diagonal's midpoint
        middles = (xyz[corners[:, 0]] + xyz[corners[:, 6]]) / 2.0
        cell = float(np.linalg.norm([np.diff(line).max() for line in lines]))
        bounds = np.array([[lines[0][0], lines[1][0]], [lines[0][-1], lines[1][-1]]])
        rounded = _round_onto_circles(left, xyz, corners, groups, middles, order,
                                      cell, bounds)
    corners, groups = corners[groups > 0], groups[groups > 0]
    keep = np.unique(corners)
    rounded += int(np.isin(keep // nw, on_circle).sum())
    renumber = np.zeros(len(xyz), dtype=int)
    renumber[keep] = np.arange(1, len(keep) + 1)
    # all twelve edges, as four opposite corners' three each: a ring's
    # brick has its short edge on the circle, which corner 0's three
    # miss (the pane read 0.1 in for a hole whose circle's were 0.059)
    shortest = min(float(np.linalg.norm(xyz[corners[:, list(_CORNER_EDGES[8][c])]]
                                        - xyz[corners[:, [c]]], axis=2).min())
                   for c in (0, 2, 5, 7))
    return xyz[keep], renumber[corners], groups, {'rounded': rounded, 'rings': ringed,
                                                  'shortest': shortest}


def _owners(geometry: Any, ids, points) -> np.ndarray:
    """Which part holds each point: 0 none, else 1 + its place in
    `ids`. Where parts overlap the later one takes it, as a later layer
    of paint covers an earlier one — a steel screw added over an insert
    makes those bricks steel."""
    owner = np.zeros(len(points), dtype=int)
    for slot, part in enumerate(ids):
        owner[contains(geometry, part, points)] = slot + 1
    return owner


def _plane(lines, order, size, owner, circles, held):
    """The mesh of the plane square to the extrusion, before it is
    swept: the grid's nodes and quads, with a ring mesh laid about each
    circle there is room for (`_ring`).

    Pulling a staircase of grid nodes onto a circle (all Mesh Solids did
    at first) left the hole jagged: a pull that would fold a brick is
    not made, so some rim nodes stayed inside the hole, and where the
    staircase runs at 45 degrees it puts twice the nodes on the circle
    that it does along the grid, so the bricks there came out thin
    (Brandon, 2026-10-11, a two-inch hole). The grid's own layout puts
    the wrong count of nodes on the circle, so no smoothing mends it;
    a ring mesh — the circle's nodes evenly spaced, layers out to the
    grid around it — is how a block-structured mesh is made round.
    Circles about one center are one ring (a hole and its insert, a
    washer about a hole); a rounded corner's arc, a circle with no room
    about it, and one less than a brick across are left to
    `_round_onto_circles`.

    Returns
    -------
    plane : (n, 2)
        Every node's (u, v); node p is the grid's (i, j) for p = i * nv + j.
    quads : (m, 4)
        Node numbers, counterclockwise about w.
    known : (m, nw - 1)
        The part each quad's brick in each layer belongs to: the grid's
        own, and a ring's by the band it lies in.
    on_circle : array
        The plane's nodes on a circle.
    left : list
        The circles to round the old way.
    ringed : int
        How many rings were laid.
    """
    u, v, _w = order
    lu, lv, _lw = lines
    nu, nv = len(lu), len(lv)
    gu, gv = np.meshgrid(lu, lv, indexing='ij')
    plane = [np.stack([gu.ravel(), gv.ravel()], axis=1)]
    count = nu * nv
    alive = np.ones((nu - 1, nv - 1), dtype=bool)
    tol = 1e-9 * float(max(lu[-1] - lu[0], lv[-1] - lv[0]))
    left, centered = [], []
    for circle in circles:
        if not circle[4]:
            left.append(circle)
            continue
        here = circle[0][[u, v]]
        for center, members in centered:
            if np.hypot(*(center - here)) <= tol:
                members.append(circle)
                break
        else:
            centered.append((here, [circle]))
    rings, ring_owners, on_circle = [], [], []
    for center, members in centered:
        radii = []
        for r in sorted(m[1] for m in members):
            if not radii or r - radii[-1] > tol:
                radii.append(r)
        # a circle less than a brick across is no ring's business: its
        # core's bricks would be a fraction of the size asked for
        if radii[-1] < size / 2.0:
            left.extend(members)
            continue
        made = _ring(lines, order, size, owner, alive, center, radii, tol, count,
                     held)
        if made is None:
            left.extend(members)
        elif made:
            (i0, i1, j0, j1, mask), positions, quads, held_by, circle_nodes = made
            alive[i0:i1, j0:j1] &= ~mask
            plane.append(positions)
            count += len(positions)
            rings.append(quads)
            ring_owners.append(held_by)
            on_circle.append(circle_nodes)
    ci, cj = np.nonzero(alive)
    grid = np.stack([ci * nv + cj, (ci + 1) * nv + cj, (ci + 1) * nv + cj + 1,
                     ci * nv + cj + 1], axis=1)
    known = owner[ci, cj]
    if rings:
        grid = np.concatenate([grid, *rings])
        known = np.concatenate([known, *ring_owners])
    on_circle = np.concatenate(on_circle) if on_circle else np.zeros(0, dtype=int)
    return np.concatenate(plane), grid, known, on_circle, left, len(rings)


def _ring(lines, order, size, owner, alive, center, radii, tol, start, held):
    """A ring mesh about circles sharing `center`, `radii` ascending, in
    a rectangle of grid cells: None where there is no room for one,
    False where no circle bounds anything (a cylinder unioned inside a
    plate), else ((i0, i1, j0, j1), new nodes' (u, v), quads, the nodes
    on a circle), new nodes numbered from `start`.

    The rectangle reaches past the largest circle by a quarter of its
    radius or half a brick, whichever is more, out to the next grid
    line, and its edge keeps the grid's nodes, so the ring meets the
    grid around it node for node. It must hold nothing but these
    circles' boundaries: no ring already there, and in every layer one
    material across each band between circles — sampled at a point,
    and held to the grid's own bricks there, so a face crossing the
    rectangle is seen, as is another circle in it. Where the rectangle
    runs past the grid it stops at the grid's edge, and where it holds
    more than the circles (a table's legs under a large hole) the
    largest one inside it that does is taken, a grid line in on every
    side at a time — so long as the circle fits inside it, touching
    it included. A circle nearer the rectangle's edge than a tenth of a
    brick (a leg flush with a table's corner touches two edges) is held
    that far in there, so no layer of bricks has no thickness: exactly
    round elsewhere, flattened by that much where it comes closer.
    Where no rectangle holds with that room, the ring's outline steps:
    the grid's cells inside a disk clear of what is in the way, an 11 in
    hole in a 12 in table's top stepping round its legs (`staircase`),
    a tenth of a brick of room left; a rectangle the circle would be
    held in from is taken only when no outline steps. Each brick of the ring takes its band's part,
    layer by layer — beyond the outer circle, between two, or inside
    the inner — not the part its center lies in: where the circle is
    held in, a thin brick's center can lie inside the true circle, and
    a flush leg grew out to the table's edge that way.

    Each circle gets as many nodes as the rectangle's edge has, spaced
    evenly between the angles of the rectangle's corners, and layers of
    about `size` run between edge and circle and between circles. A
    circle with material inside it gets a core: a grid of the
    rectangle's own counts, a quarter of the circle's width, with
    layers out to it."""
    u, v, w = order
    lu, lv, lw = lines
    nu, nv, nw = len(lu), len(lv), len(lw)
    outer = radii[-1]
    reach = outer + max(0.25 * outer, 0.5 * size)
    i0 = int(np.searchsorted(lu, center[0] - reach + tol, 'right')) - 1
    i1 = int(np.searchsorted(lu, center[0] + reach - tol, 'left'))
    j0 = int(np.searchsorted(lv, center[1] - reach + tol, 'right')) - 1
    j1 = int(np.searchsorted(lv, center[1] + reach - tol, 'left'))
    i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, nu - 1), min(j1, nv - 1)
    middles = (lw[:-1] + lw[1:]) / 2.0
    # a rectangle's circle comes no nearer its edge than a tenth of a
    # brick: a leg flush with a table's corner touches its two edges,
    # and the layer there would have no thickness, so the circle is held
    # that far in where it comes closer — flattened there by that much,
    # and exactly round everywhere else; a stepped outline must leave
    # that much room (Brandon, 2026-10-12)
    clear = 0.1 * size

    def layered(points):
        """(points, layers): the part at each point in each layer."""
        found = np.zeros((len(points) * (nw - 1), 3))
        found[:, u] = np.repeat(points[:, 0], nw - 1)
        found[:, v] = np.repeat(points[:, 1], nw - 1)
        found[:, w] = np.tile(middles, len(points))
        return held(found).reshape(len(points), nw - 1)

    def agreement(i0, i1, j0, j1, bands):
        """(corners, agree): each grid cell's four corners' distances
        from the center, and, layer by layer, whether the grid's brick
        there holds the part of the band it lies in — one across a
        circle that of a band it touches."""
        du, dv = np.meshgrid(lu[i0:i1 + 1] - center[0], lv[j0:j1 + 1] - center[1],
                             indexing='ij')
        far = np.hypot(du, dv)
        corners = np.stack([far[:-1, :-1], far[1:, :-1], far[1:, 1:], far[:-1, 1:]])
        inner = np.searchsorted(radii, corners.min(axis=0) + tol)
        outmost = np.searchsorted(radii, corners.max(axis=0) - tol)
        cells = owner[i0:i1, j0:j1]
        agree = np.zeros(cells.shape, dtype=bool)
        for t, band in enumerate(bands):
            agree |= (((inner <= t) & (t <= outmost))[:, :, None]
                      & (cells == band[None, None, :]))
        return corners, agree

    def fits(i0, i1, j0, j1):
        """The bands' parts in this rectangle, or None where it holds
        more than these circles' boundaries."""
        low, high = np.array([lu[i0], lv[j0]]), np.array([lu[i1], lv[j1]])
        # each band's part, layer by layer, at one point: the middle,
        # each annulus's mid radius, and halfway from the outer circle
        # to the rectangle's farthest corner, where there is room even
        # when the rectangle touches the circle
        corner = np.where(center - low > high - center, low, high) - center
        reach_out = float(np.hypot(*corner))
        across = [0.0] + [(r0 + r1) / 2.0 for r0, r1 in pairwise(radii)]
        across.append((outer + reach_out) / 2.0)
        bands = layered(center + np.outer(across, corner / reach_out))
        _corners, agree = agreement(i0, i1, j0, j1, bands)
        return bands if agree.all() else None

    def staircase(i0, i1, j0, j1):
        """The cells of this rectangle inside a disk about the center wide
        enough for the circles and clear of every cell that holds more
        than them, with their outline; None where even that will not
        hold a ring. Where the rectangle cannot, a stepped outline about
        the hole can: a hole nearly as wide as a table's top, whose every
        rectangle took in the legs at its corners, steps round them
        (Brandon, 2026-10-12)."""
        low, high = np.array([lu[i0], lv[j0]]), np.array([lu[i1], lv[j1]])
        # the band beyond the circles sampled toward the side with the
        # most room, clear of a corner's leg
        gaps = np.array([center[0] - low[0], high[0] - center[0],
                         center[1] - low[1], high[1] - center[1]])
        side = int(np.argmax(gaps))
        toward = np.array([(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)][side])
        across = [0.0] + [(r0 + r1) / 2.0 for r0, r1 in pairwise(radii)]
        across.append((outer + gaps[side]) / 2.0)
        bands = layered(center + np.outer(across, toward))
        corners, agree = agreement(i0, i1, j0, j1, bands)
        bad = ~agree.all(axis=2) | ~alive[i0:i1, j0:j1]
        if not bad.any():
            return None
        # every cell wholly inside the nearest bad cell's nearest corner:
        # a bad cell reaches past that radius, so none is taken
        cap = float(corners.min(axis=0)[bad].min())
        inside = (corners.max(axis=0) <= cap + tol) & ~bad
        path = _outline(inside)
        if path is None:
            return None
        nodes = [(i0 + ci, j0 + cj) for ci, cj in path]
        q = np.array([(lu[i], lv[j]) for i, j in nodes])
        following = np.roll(q, -1, axis=0) - q
        t = np.clip(((center - q) * following).sum(axis=1)
                    / (following * following).sum(axis=1), 0.0, 1.0)
        room = float(np.hypot(*(q + t[:, None] * following - center).T).min()) - outer
        if room < -tol or (bands[-1].any() and room < clear):
            return None
        # star-shaped about the center, or a ring drawn to it would fold;
        # begun at its bottom left, as a rectangle's edge is
        turn = np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0])
        first = int(np.argmin(np.abs(np.angle(np.exp(1j * (turn + 0.75 * np.pi))))))
        nodes, q = nodes[first:] + nodes[:first], np.roll(q, -first, axis=0)
        theta = np.unwrap(np.roll(turn, -first))
        if (np.diff(theta) <= 0).any() or theta[-1] - theta[0] >= 2.0 * np.pi:
            return None
        return bands, inside, nodes, q, theta

    # the rectangle a quarter radius out, else the largest inside it that
    # holds nothing but these circles, a grid line in on every side at a
    # time: a hole of 4.5 in in a table's top reached its legs' columns,
    # 5 in out, and was pulled round from the grid instead (Brandon,
    # 2026-10-12) — while a layer of bricks still fits between
    bands = stepped = touching = None
    preferred = (i0, i1, j0, j1)
    while i1 - i0 >= 2 and j1 - j0 >= 2:
        low, high = np.array([lu[i0], lv[j0]]), np.array([lu[i1], lv[j1]])
        room = float(min((center - low).min(), (high - center).min())) - outer
        if room < -tol:
            break
        if alive[i0:i1, j0:j1].all():
            bands = fits(i0, i1, j0, j1)
            if bands is not None and room < clear and bands[-1].any():
                # one the circle would be held in from: kept for last,
                # since a stepped outline may hold it round — round legs
                # set in a quarter inch left a 9 in hole's rectangle
                # touching it, flattened four times (2026-10-12)
                touching, bands = (bands, i0, i1, j0, j1), None
                break
            if bands is not None:
                break
        i0, i1, j0, j1 = i0 + 1, i1 - 1, j0 + 1, j1 - 1
    if bands is None and preferred[1] - preferred[0] >= 2 \
            and preferred[3] - preferred[2] >= 2:
        # no rectangle holds it with room: the stepped outline, since a
        # rectangle's circle nodes are evenly spaced and its outline's
        # follow the steps
        stepped = staircase(*preferred)
        if stepped is not None:
            bands = stepped[0]
    if bands is None and touching is not None:
        bands, i0, i1, j0, j1 = touching
    if bands is None:
        return None
    kept = [r for t, r in enumerate(radii) if (bands[t] != bands[t + 1]).any()]
    if not kept:
        return False
    if stepped is not None:
        # the stepped outline: each circle node at its outline node's own
        # angle, the radial lines straight; a core of half the count a
        # side, as near square as the count allows
        _bands, mask, edge, q, theta = stepped
        i0, i1, j0, j1 = preferred
        count = len(q)
        a = count // 4
        b = count // 2 - a
    else:
        mask = np.ones((i1 - i0, j1 - j0), dtype=bool)
        a, b = i1 - i0, j1 - j0
        # the rectangle's edge, counterclockwise about w from (i0, j0)
        edge = ([(i, j0) for i in range(i0, i1)] + [(i1, j) for j in range(j0, j1)]
                + [(i, j1) for i in range(i1, i0, -1)]
                + [(i0, j) for j in range(j1, j0, -1)])
        q = np.array([(lu[i], lv[j]) for i, j in edge])
        count = len(q)
        # the circles' nodes: evenly spaced between the corners' angles
        marks = [0, a, a + b, 2 * a + b, count]
        angles = np.arctan2(*(q[marks[:4]] - center).T[::-1])
        angles = np.append(angles[0] + np.mod(angles - angles[0], 2.0 * np.pi),
                           angles[0] + 2.0 * np.pi)
        theta = np.empty(count)
        for side in range(4):
            first, last = marks[side], marks[side + 1]
            along = (np.linalg.norm(q[first:last] - q[first], axis=1)
                     / np.linalg.norm(q[last % count] - q[first]))
            theta[first:last] = angles[side] + along * (angles[side + 1] - angles[side])
    edge_nodes = np.array([i * nv + j for i, j in edge])
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    # held in only where there is material beyond the circle: a lone
    # cylinder touches its own grid's edge, and beyond it is nothing
    if stepped is None and bands[-1].any():
        inward = (np.array([lu[i0], lv[j0]]) + clear, np.array([lu[i1], lv[j1]]) - clear)
    else:
        inward = None

    def placed(radius):
        """A circle's nodes, held `clear` inside a rectangle's edge."""
        points = center + radius * unit
        return points if inward is None else np.clip(points, *inward)
    positions, rings, on_circle = [], [edge_nodes], []
    # the band each pair of rings bounds: `len(radii)` beyond the outer
    # circle, `t` between radii[t - 1] and radii[t], 0 inside
    between: list[int] = []
    made = start

    def ring_at(points):
        nonlocal made
        positions.append(points)
        made += len(points)
        return np.arange(made - len(points), made)

    rim = placed(kept[-1])
    layers = 1 if not bands[-1].any() else max(1, round(float(np.median(
        np.linalg.norm(rim - q, axis=1))) / size))
    for step in range(1, layers + 1):
        rings.append(ring_at(q + step / layers * (rim - q)))
        between.append(len(radii))
    on_circle.append(rings[-1])
    reached = kept[-1]
    for r in reversed(kept[:-1]):
        layers = max(1, round((reached - r) / size))
        band = int(np.searchsorted(radii, (reached + r) / 2.0))
        for step in range(1, layers + 1):
            rings.append(ring_at(placed(reached - step / layers * (reached - r))))
            between.append(band)
        on_circle.append(rings[-1])
        reached = r
    quads = []
    if bands[0].any():
        # a core of the rectangle's own counts, so its edge meets the
        # circle node for node, its bricks square
        half = 0.5 * reached / max(a, b)
        cu = center[0] + half * (2 * np.arange(a + 1) - a)
        cv = center[1] + half * (2 * np.arange(b + 1) - b)
        gu, gv = np.meshgrid(cu, cv, indexing='ij')
        core = ring_at(np.stack([gu.ravel(), gv.ravel()], axis=1)).reshape(a + 1, b + 1)
        rim = placed(reached)
        core_edge = np.array([core[k, 0] for k in range(a)] + [core[a, j] for j in range(b)]
                             + [core[k, b] for k in range(a, 0, -1)]
                             + [core[0, j] for j in range(b, 0, -1)])
        edge_xy = np.concatenate(positions)[core_edge - start]
        layers = max(1, round(float(np.median(np.linalg.norm(rim - edge_xy, axis=1)))
                              / size))
        for step in range(layers - 1, 0, -1):
            rings.append(ring_at(edge_xy + step / layers * (rim - edge_xy)))
            between.append(0)
        rings.append(core_edge)
        between.append(0)
        quads.append(np.stack([core[:-1, :-1].ravel(), core[1:, :-1].ravel(),
                               core[1:, 1:].ravel(), core[:-1, 1:].ravel()], axis=1))
    k = np.arange(count)
    following = (k + 1) % count
    owners = [np.repeat(bands[band][None, :], count, axis=0) for band in between]
    if quads:
        owners.append(np.repeat(bands[0][None, :], len(quads[0]), axis=0))
    quads = [np.stack([outer_ring[k], outer_ring[following], inner_ring[following],
                       inner_ring[k]], axis=1)
             for outer_ring, inner_ring in pairwise(rings)] + quads
    return ((i0, i1, j0, j1, mask), np.concatenate(positions), np.concatenate(quads),
            np.concatenate(owners), np.concatenate(on_circle))


def _outline(inside: np.ndarray) -> list[tuple[int, int]] | None:
    """The outline of a set of grid cells, counterclockwise, as the
    (i, j) of its corners in order — every grid node along it, as a
    rectangle's edge has — or None where it is not one simple loop (two
    cells meeting at a corner only, or a hole in the set)."""
    a, b = inside.shape

    def cell(i, j):
        return 0 <= i < a and 0 <= j < b and bool(inside[i, j])

    step: dict[tuple[int, int], tuple[int, int]] = {}
    for i, j in zip(*np.nonzero(inside)):
        i, j = int(i), int(j)
        for start, end, beyond in (((i, j), (i + 1, j), (i, j - 1)),
                                   ((i + 1, j), (i + 1, j + 1), (i + 1, j)),
                                   ((i + 1, j + 1), (i, j + 1), (i, j + 1)),
                                   ((i, j + 1), (i, j), (i - 1, j))):
            if not cell(*beyond):
                if start in step:
                    return None
                step[start] = end
    if not step:
        return None
    first = min(step)
    path, at = [first], step[first]
    while at != first:
        path.append(at)
        at = step.get(at)
        if at is None or len(path) > len(step):
            return None
    return path if len(path) == len(step) else None


def _circles(geometry, ids, w: int) -> list[tuple[np.ndarray, float, float, float, bool]]:
    """(center, radius, low, high, whole) of every circle the parts'
    outlines draw, `w` their extrusion axis: each cylinder's, whole, and
    each rounded block's four corner arcs — a circle of the corner's
    radius about the arc's center, spanning the block along `w`, a
    quarter of it drawn."""
    out = []
    for i in ids:
        for leaf in _leaves(geometry, i):
            center = np.asarray(leaf.center)
            axes = _frame(leaf)
            if leaf.kind == 'cylinder':
                out.append((center, leaf.radius, center[w] - leaf.length / 2,
                            center[w] + leaf.length / 2, True))
            elif leaf.corner > 0.0:
                wx, wy, wz = leaf.widths
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        arc = center + (sx * (wx / 2 - leaf.corner) * axes[0]
                                        + sy * (wy / 2 - leaf.corner) * axes[1])
                        out.append((arc, leaf.corner, center[w] - wz / 2,
                                    center[w] + wz / 2, False))
    return out


def _round_onto_circles(circles, xyz, corners, groups, centers, order,
                        cell: float, bounds) -> int:
    """Pull the nodes on each circle's rim onto it, in place; how many
    moved. For the circles no ring could be laid about (`_plane`): a
    rounded corner's arc, which its block's faces touch, and a circle
    with no room around it. `cell` is a brick's diagonal, at most;
    `bounds` the grid's lowest and highest (u, v), its outer faces.

    Within the circle's length, a rim node is one the
    bricks inside the circle and the bricks beyond it share, where no
    material — empty counted as one — lies on both sides of it: a hole
    (empty or an insert inside, the part beyond), a boss (the part
    inside, empty beyond). A node with the same material on both sides
    is not on a boundary the circle draws, and one a hole shares with
    an edge's empty space keeps its place. A node on the grid's outer
    face, where a hole breaks out through it, slides along that face to
    where the circle crosses it rather than leaving it (Add Block's
    rule, 2026-09-30). A node two circles claim goes to the nearer.

    The pulls are made one at a time, the node nearest its circle first,
    and one that turns a brick beside it inside out is undone: where
    three corners of a brick land on the circle, the arc carries the
    middle one past the line between the others and folds the brick
    into a dart (a hole six bricks across, 2026-10-11). No distance rule
    caught every case, since whether a pull folds a brick depends on
    its neighbors' pulls; trying each against them does."""
    from .geometry import _CORNER_EDGES

    edges = np.asarray(_CORNER_EDGES[8])

    def volumes(points):
        """Each corner's share of a brick's volume, for bricks (m, 8, 3):
        its sign says the brick is still the right way out."""
        legs = points[:, edges] - points[:, :, None, :]
        return np.einsum('mci,mci->mc', legs[:, :, 0],
                         np.cross(legs[:, :, 1], legs[:, :, 2]))

    u, v, w = order
    best: dict[int, tuple[float, np.ndarray]] = {}
    classes = np.unique(groups)
    for center, radius, low, high, _whole in circles:
        span = (centers[:, w] > low) & (centers[:, w] < high)
        reach = np.hypot(centers[:, u] - center[u], centers[:, v] - center[v])
        # only the bricks about the circle: a rim node's bricks are all
        # within a brick's diagonal of it
        near = np.flatnonzero(span & (reach <= radius + 2 * cell))
        inside = reach[near] <= radius
        if not inside.any() or inside.all():
            continue
        around, owners = corners[near], groups[near]
        rim = np.intersect1d(around[inside], around[~inside])
        for material in classes:
            both = np.intersect1d(around[inside & (owners == material)],
                                  around[~inside & (owners == material)])
            rim = np.setdiff1d(rim, both)
        for n in rim:
            here = xyz[n, [u, v]]
            offset = here - center[[u, v]]
            distance = float(np.hypot(*offset))
            if distance == 0.0:
                continue
            target = center[[u, v]] + offset * (radius / distance)
            # a node on two faces is the grid's corner, one brick's own,
            # never on a rim
            faces = [k for k in (0, 1) if np.isclose(here[k], bounds[0][k], rtol=0, atol=1e-9 * cell)
                     or np.isclose(here[k], bounds[1][k], rtol=0, atol=1e-9 * cell)]
            if faces:
                held, other = faces[0], 1 - faces[0]
                reach = radius ** 2 - offset[held] ** 2
                if reach < 0.0:
                    continue
                target[held] = here[held]
                target[other] = center[[u, v]][other] + (np.sign(offset[other]) or 1.0) * np.sqrt(reach)
            gap = abs(distance - radius)
            if n not in best or gap < best[n][0]:
                best[n] = (gap, target)
    if not best:
        return 0
    material = corners[groups > 0]
    rows = np.flatnonzero(np.isin(material, list(best)).any(axis=1))
    beside: dict[int, list[int]] = {}
    for row in rows:
        for n in material[row]:
            if int(n) in best:
                beside.setdefault(int(n), []).append(int(row))
    grid = xyz.copy()
    moved = 0
    for n in sorted(best, key=lambda m: best[m][0]):
        held = xyz[n].copy()
        xyz[n, [u, v]] = best[n][1]
        bricks = material[beside.get(n, [])]
        if len(bricks) and (volumes(grid[bricks]) * volumes(xyz[bricks]) <= 0).any():
            xyz[n] = held
        else:
            moved += 1
    return moved
