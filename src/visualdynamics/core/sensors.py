"""Where each sensor a channel table names sits on a geometry, and which
way it faces: the placement the 3-D view draws a sensor's body with.

One physical sensor is a node and a model. A triaxial accelerometer is
its rows at one node with one model; a uniaxial accelerometer or a force
transducer is one row. Only acceleration and force rows that name a
model place a sensor: a row with no model says what is measured, not
with what, and its DOF arrow already marks it. A model named but not
in the catalog is drawn as a generic body of its kind.

The node is the center of the body. A triax's frame is its legs: the
Triax DOF column says which leg a row is (blank: the row's own direction
letter), and the row's direction in global coordinates
(`Geometry.dof_direction`, the rule behind the table's Unit X, Y and Z)
is that leg's positive direction. Two legs give the third. Legs that
are not perpendicular, or that make a left-handed frame, are an error in
the table; the sensor is then drawn as a generic body on the node's own
axes and a note names the node. A uniaxial sensor's or a force
transducer's sensing axis is its row's direction, through the node; its
turn about that axis is fixed by one rule, the connector toward the
node's local +X (+Y when the sensing axis is along X), since a table
does not say it.

Orientation comes from the table alone, never from the surface a sensor
is mounted on. A wrong sign in the table shows in the drawing: a
triax's lettered face, or a uniaxial's arrow, points the wrong way.
"""

# Brandon, 2026-10-26 and 2026-10-27 (PLAN.md, "Sensors drawn on the
# geometry"): the table decides the orientation; the node is the body's
# center (the lines through the centers of a triax's X, Y and Z faces
# cross at it, and a uniaxial's axis runs through it), not the base at
# the node of the first plan; the mounting surface is never read.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sensor_catalog import GENERIC, MEASURES, SensorModel, lookup

#: legs this far from perpendicular are a table error, not a sensor
SQUARE_TOLERANCE = np.sin(np.radians(1.0))

_LEGS = ('X', 'Y', 'Z')


@dataclass
class Sensor:
    """One sensor placed on a geometry.

    `center` is the node's coordinates as the geometry holds them.
    `frame`'s columns are the body's x, y and z in global coordinates, a
    right-handed set of unit vectors. `marked` says the body carries its
    marks, a triax's letters or a uniaxial's arrow: not when the frame is
    the node's own axes because the table's legs could not be read, since
    the marks would then say what the table does not.
    """

    node: int
    model: str
    entry: SensorModel
    center: np.ndarray
    frame: np.ndarray
    rows: tuple[int, ...]
    generic: bool
    marked: bool = True

    @property
    def kind(self) -> str:
        """'triax', 'uniaxial' or 'force'."""
        return self.entry.kind


def place(table: Any, geometry: Any) -> tuple[list[Sensor], list[str]]:
    """The sensors a channel table places on a geometry, and notes on
    what could not be placed as the table says.

    Parameters
    ----------
    table : ChannelTable
        The channels: node, direction, channel type, make, model and
        Triax DOF are read.
    geometry : Geometry
        The geometry the nodes are in.

    Returns
    -------
    sensors : list of Sensor
        One per physical sensor, in the order of their first rows.
    notes : list of str
        Rows left out and why, models drawn as generic bodies, and triax
        legs the table gets wrong, each naming the node.
    """
    types = table.types()
    nodes = table['node']
    directions = table['direction']
    models = table['model']
    legs = table['triax_dof']
    known = {int(n) for n in np.asarray(geometry.node_id)}
    left_out = {'no node': 0, 'node not in the geometry': 0,
                'no direction': 0,
                'node in a cylindrical or spherical system': 0}
    groups: dict[tuple, list[int]] = {}
    for row, channel_type in enumerate(types):
        model = str(models[row]).strip()
        if channel_type not in ('acceleration', 'force') or not model:
            continue
        node = str(nodes[row]).strip()
        if not node:
            left_out['no node'] += 1
            continue
        if int(node) not in known:
            left_out['node not in the geometry'] += 1
            continue
        direction = str(directions[row]).strip().upper()
        if direction[:1] not in _LEGS:
            left_out['no direction'] += 1
            continue
        if geometry.dof_direction(f'{node}{direction}') is None:
            left_out['node in a cylindrical or spherical system'] += 1
            continue
        entry = lookup(model)
        if entry is not None and MEASURES[entry.kind] != channel_type:
            entry = None
        if entry is not None:
            kind = entry.kind
        elif channel_type == 'force':
            kind = 'force'
        else:
            same = [r for r in range(len(types))
                    if str(nodes[r]).strip() == node
                    and str(models[r]).strip().upper() == model.upper()
                    and types[r] == 'acceleration']
            kind = 'triax' if any(str(legs[r]).strip() for r in same) \
                else 'uniaxial'
        # a triax is all its rows at the node; anything else is one row
        key = ((int(node), model.upper(), kind) if kind == 'triax'
               else (int(node), model.upper(), kind, row))
        groups.setdefault(key, []).append(row)

    sensors: list[Sensor] = []
    notes: list[str] = []
    unknown: dict[str, list[str]] = {}
    for key, rows in groups.items():
        node, kind = key[0], key[2]
        model = str(models[rows[0]]).strip()
        entry = lookup(model)
        if entry is not None and entry.kind != kind:
            notes.append(f'{model} is a {GENERIC[entry.kind].description}, '
                         f'on a {MEASURES[kind]} row at node {node}: drawn as '
                         f'a generic {GENERIC[kind].description}')
            entry = GENERIC[kind]
        elif entry is None:
            unknown.setdefault(GENERIC[kind].description, []).append(model)
            entry = GENERIC[kind]
        center = np.asarray(
            geometry.node_xyz[int(np.flatnonzero(
                np.asarray(geometry.node_id) == node)[0])], dtype=float)
        if kind == 'triax':
            frame, problem = _triax_frame(geometry, node, rows, directions,
                                          legs)
            if problem is not None:
                notes.append(f'node {node}: {problem}; drawn on the node\'s '
                             'own axes')
                sensors.append(Sensor(node, model, GENERIC['triax'], center,
                                      _local_frame(geometry, node),
                                      tuple(rows), True, marked=False))
                continue
        else:
            direction = str(directions[rows[0]]).strip().upper()
            frame = _axial_frame(geometry, node, direction)
        sensors.append(Sensor(node, model, entry, center, frame, tuple(rows),
                              entry is GENERIC[kind]))
    for what, names in unknown.items():
        listed = ', '.join(sorted(set(names)))
        notes.append(f'not in the catalog, drawn as a generic {what}: '
                     f'{listed}')
    for why, count in left_out.items():
        if count:
            notes.append(f'{count} sensor row{"s" * (count != 1)} not drawn: '
                         f'{why}')
    return sensors, notes


def _direction(geometry: Any, node: int, direction: str) -> np.ndarray:
    return np.asarray(geometry.dof_direction(f'{node}{direction}'),
                      dtype=float)


def _local_frame(geometry: Any, node: int) -> np.ndarray:
    """The node's own X, Y and Z as columns."""
    return np.column_stack([_direction(geometry, node, f'{axis}+')
                            for axis in _LEGS])


def _triax_frame(geometry: Any, node: int, rows: Sequence[int], directions,
                 legs) -> tuple[np.ndarray | None, str | None]:
    """The triax's frame from its rows, or None and what is wrong."""
    found: dict[int, np.ndarray] = {}
    for row in rows:
        direction = str(directions[row]).strip().upper()
        leg = str(legs[row]).strip().upper() or direction[0]
        index = _LEGS.index(leg)
        if index in found:
            return None, f'two rows are the triax\'s {leg} leg'
        found[index] = _direction(geometry, node, direction)
    if len(found) == 1:
        # one leg wired: the next leg is the node's own next axis, which
        # keeps the body upright on the node's axes. The wired leg is one
        # of those axes too, either way round, so the two are square.
        (index, _vector), = found.items()
        after = (index + 1) % 3
        found[after] = _direction(geometry, node, f'{_LEGS[after]}+')
    if len(found) == 2:
        a, b = sorted(found)
        if abs(found[a] @ found[b]) > SQUARE_TOLERANCE:
            return None, 'the triax legs are not perpendicular'
        missing = 3 - a - b
        # X = Y x Z, Y = Z x X, Z = X x Y
        nxt, after = (missing + 1) % 3, (missing + 2) % 3
        found[missing] = np.cross(found[nxt], found[after])
    frame = np.column_stack([found[0], found[1], found[2]])
    gram = frame.T @ frame
    if np.max(np.abs(gram - np.eye(3))) > SQUARE_TOLERANCE:
        return None, 'the triax legs are not perpendicular'
    if np.linalg.det(frame) < 0:
        return None, ('the triax legs make a left-handed frame, so one '
                      'sign in the table is wrong')
    return frame, None


def _axial_frame(geometry: Any, node: int, direction: str) -> np.ndarray:
    """A uniaxial sensor's frame: z along the sensing direction, x the
    node's local +X (+Y when z is along X). The direction is one of the
    node's own axes, so the two are square without adjusting."""
    z = _direction(geometry, node, direction)
    x = _direction(geometry, node, 'Y+' if direction[0] == 'X' else 'X+')
    return np.column_stack([x, np.cross(z, x), z])
