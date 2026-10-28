"""Sensor bodies drawn at their nodes: one mesh for every sensor a
channel table places (`core.sensors.place`), so a model with hundreds of
sensors is one actor rather than hundreds.

Each body is built about its own origin from its catalog size, turned
into its frame and moved to its node, which is therefore the body's
center. A triax's X, Y and Z faces carry their letters in black, as the
sensors do; a uniaxial accelerometer or a force transducer carries a
black arrow on each side along its sensing axis, pointing the positive
way. The body is gray, the connector and the cable darker.

Bodies are true size where the geometry's units are defined. Where they
are not, a millimeter cannot be placed, and the bodies are drawn at a
size proportional to the model's instead. `size` multiplies either,
about the node, so the node stays at the center.

The points are a VTK buffer of their own with a writable view, so an
animation moves the bodies with their nodes by one write per frame
(`GeometryAnimator.follow`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..units import DEFAULT_SYSTEM
from .geometry import shared_points

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.channel_table import ChannelTable
    from ..core.geometry import Geometry
    from ..core.sensor_catalog import SensorModel
    from ..core.sensors import Sensor
    from ..units import UnitSystem

#: the body, its connector and cable, and the letters and arrows on it
BODY_COLOR = '#a7abb1'
CONNECTOR_COLOR = '#5d6166'
CABLE_COLOR = '#3b3f44'
MARK_COLOR = '#1a1a1a'

#: a teardrop's narrow end, as a fraction of its round end's diameter
TAIL = 0.42

#: a mark stands this fraction of the body's smallest side off its face,
#: so the face does not draw through it
LIFT = 0.02

#: with the geometry's units undefined, a millimeter is drawn as this
#: fraction of the model's size (a 10 mm cube is 1.5 % of it)
NOMINAL_MM = 0.0015

#: sides of the circle a puck and a connector are drawn with
ROUND = 24

NOMINAL_NOTE = ("the geometry's units are not defined, so the sensors are "
                'drawn at a size proportional to the model, not true size')


@dataclass
class SensorMesh:
    """The sensors' mesh, and what an animation needs to move it.

    `node_rows` is the geometry's node row of every point, `base` the
    points as drawn with every node at rest, and `points` and `view` the
    VTK buffer the mesh is drawn from and a writable view of it.
    """

    mesh: Any
    node_rows: np.ndarray
    base: np.ndarray
    points: Any
    view: np.ndarray
    sensors: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip('#')
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _prism(outline: np.ndarray, bottom: float, top: float):
    """A prism along z over a convex outline (counterclockwise): its
    points, its triangles, and each triangle's face, ('z', sign) for a
    cap and ('side', outward normal) for a side."""
    n = len(outline)
    low = np.column_stack([outline, np.full(n, bottom)])
    high = np.column_stack([outline, np.full(n, top)])
    points = np.vstack([low, high])
    triangles, faces = [], []
    for k in range(1, n - 1):
        triangles.append((0, k + 1, k))
        faces.append(('z', -1))
        triangles.append((n, n + k, n + k + 1))
        faces.append(('z', 1))
    for k in range(n):
        j = (k + 1) % n
        edge = outline[j] - outline[k]
        normal = np.array([edge[1], -edge[0]])
        triangles += [(k, j, n + j), (k, n + j, n + k)]
        faces += [('side', normal)] * 2
    return points, triangles, faces


def _circle(radius: float, sides: int, turn: float = 0.0) -> np.ndarray:
    angles = turn + 2 * np.pi * np.arange(sides) / sides
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _teardrop(length: float, width: float) -> np.ndarray:
    """A teardrop along x, counterclockwise: a round end `width` across
    at -length/2 narrowing to a flat tail at +length/2."""
    radius, tail = width / 2, TAIL * width / 2
    cx = -length / 2 + radius
    # the flanks run from the tail's corners to where they touch the circle
    dx, dy = length / 2 - cx, tail
    touch = np.arctan2(dy, dx) + np.arccos(radius / np.hypot(dx, dy))
    # an odd count puts a point on the nose itself, so the length is exact
    arc = np.linspace(touch, 2 * np.pi - touch, ROUND + 1)
    nose = np.column_stack([cx + radius * np.cos(arc), radius * np.sin(arc)])
    return np.vstack([[[length / 2, tail]], nose, [[length / 2, -tail]]])


def _rounded(length: float, width: float) -> np.ndarray:
    """A box along x with its -x end a half circle, counterclockwise: the
    nose is round and the cable leaves the flat end."""
    radius = width / 2
    cx = -length / 2 + radius
    arc = np.linspace(np.pi / 2, 3 * np.pi / 2, ROUND // 2 + 1)
    nose = np.column_stack([cx + radius * np.cos(arc), radius * np.sin(arc)])
    return np.vstack([[[length / 2, radius]], nose, [[length / 2, -radius]]])


def _outline(entry: SensorModel) -> np.ndarray:
    sx, sy, _sz = entry.size
    if entry.shape == 'box':
        return np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]]) / 2
    if entry.shape == 'hex':
        # flats facing +-x, so the connector sits on one
        return _circle(sx / np.sqrt(3.0), 6, np.pi / 6)
    if entry.shape == 'teardrop':
        return _teardrop(sx, sy)
    if entry.shape == 'rounded':
        return _rounded(sx, sy)
    return _circle(sx / 2, ROUND)


#: each letter as strokes in a box one unit high, (right, up) from the
#: letter's center: the strokes of X, Y and Z as the sensors etch them
_LETTERS = {
    'X': [((-0.35, 0.5), (0.35, -0.5)), ((-0.35, -0.5), (0.35, 0.5))],
    'Y': [((-0.35, 0.5), (0.0, 0.0)), ((0.35, 0.5), (0.0, 0.0)),
          ((0.0, 0.0), (0.0, -0.5))],
    'Z': [((-0.35, 0.5), (0.35, 0.5)), ((0.35, 0.5), (-0.35, -0.5)),
          ((-0.35, -0.5), (0.35, -0.5))],
}

#: a stroke's width as a fraction of the letter's height
STROKE = 0.16


def _letter(letter: str, height: float):
    """A letter's strokes as flat polygons in the (right, up) plane,
    `height` high and centered on the origin."""
    polygons = []
    for (r0, u0), (r1, u1) in _LETTERS[letter]:
        a, b = np.array([r0, u0]) * height, np.array([r1, u1]) * height
        along = b - a
        across = np.array([-along[1], along[0]]) / np.linalg.norm(along)
        half = STROKE * height / 2 * across
        polygons.append(np.array([a - half, b - half, b + half, a + half]))
    return polygons


def _arrow(height: float):
    """An arrow up the (right, up) plane, `height` high: its shaft and its
    head, the shaft reaching lower than the head stands tall so the
    arrow reads from either side."""
    shaft, head = 0.07 * height, 0.25 * height
    tail, neck, tip = -0.4 * height, 0.1 * height, 0.5 * height
    return ([np.array([[-shaft, tail], [shaft, tail], [shaft, neck],
                       [-shaft, neck]])],
            [np.array([[-head, neck], [head, neck], [0.0, tip]])])


def _on_face(polygons, origin, normal, up, lift):
    """Flat polygons placed on a face: `origin` is the mark's center on
    the face, `normal` the face's outward normal, `up` the mark's up,
    both unit vectors in the body's frame."""
    right = np.cross(up, normal)
    base = np.asarray(origin, dtype=float) + lift * normal
    placed, triangles, start = [], [], 0
    for polygon in polygons:
        points = base + np.outer(polygon[:, 0], right) + np.outer(polygon[:, 1], up)
        placed.append(points)
        triangles += [(start, start + k, start + k + 1)
                      for k in range(1, len(points) - 1)]
        start += len(points)
    return np.vstack(placed), triangles


def _arrow_sides(outline: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Where a uniaxial's arrows go: on each side of the body, the
    longest side facing that way; among equals the one facing it most
    squarely (a puck's facets), then the one farther from the connector
    (a hex's two flats either side of its back)."""
    chosen = []
    n = len(outline)
    for sign in (1.0, -1.0):
        candidates = []
        for k in range(n):
            a, b = outline[k], outline[(k + 1) % n]
            edge = b - a
            normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
            if normal[1] * sign > 0.5:
                candidates.append((-round(float(np.linalg.norm(edge)), 6),
                                   -round(abs(float(normal[1])), 6),
                                   normal[0], (a + b) / 2, normal))
        candidates.sort(key=lambda c: c[:3])
        chosen.append((candidates[0][3], candidates[0][4]))
    return chosen


def _cylinder(radius: float, start: float, length: float, sides: int,
              axis: str = 'x', sign: float = 1.0):
    """A cylinder of `length` along +-x or +-z from `start` along it."""
    ring = _circle(radius, sides)
    points, triangles, _faces = _prism(ring, 0.0, length)
    if axis == 'x':
        points = np.column_stack([sign * (start + points[:, 2]),
                                  points[:, 0], points[:, 1]])
    else:
        points = np.column_stack([points[:, 0], points[:, 1],
                                  sign * (start + points[:, 2])])
    if sign < 0:
        triangles = [(a, c, b) for a, b, c in triangles]
    return points, triangles


def body(entry: SensorModel, marks: bool = True):
    """A sensor's body in its own frame, in millimeters, centered on the
    origin: (points, triangles, parts), where a triangle's part is
    'body', 'connector' (the plug and a force transducer's studs),
    'cable', or a mark on the body: 'mark:X', 'mark:Y' and 'mark:Z' for
    a triax's letters, 'mark:shaft' and 'mark:head' for the arrow of a
    uniaxial accelerometer or a force transducer, left out when
    `marks` is False."""
    sx, sy, sz = entry.size
    outline = _outline(entry)
    pieces = []        # (points, triangles, part)
    points, triangles, _faces = _prism(outline, -sz / 2, sz / 2)
    pieces.append((points, triangles, 'body'))
    # the connector: a plug out of the middle of its face, stepping down
    # to its thread, and the cable beyond it
    sign = 1.0 if entry.connector == '+x' else -1.0
    reach = outline[:, 0].max() if sign > 0 else -outline[:, 0].min()
    diameter, length = entry.plug
    if length > 0:
        base = 0.45 * length
        pieces.append((*_cylinder(diameter / 2, reach, base, 12, 'x', sign),
                       'connector'))
        pieces.append((*_cylinder(0.6 * diameter / 2, reach + base,
                                  length - base, 12, 'x', sign), 'connector'))
        reach += length
    diameter, length = entry.cable
    if length > 0:
        pieces.append((*_cylinder(diameter / 2, reach, length, 8, 'x', sign),
                       'cable'))
    diameter, length = entry.stud
    if length > 0:
        for end in (1.0, -1.0):
            pieces.append((*_cylinder(diameter / 2, sz / 2, length, 12, 'z',
                                      end), 'connector'))
    if marks:
        lift = LIFT * min(entry.size)
        if entry.kind == 'triax':
            # the letter on each positive face, upright as the sensors
            # etch them: Z read with the connector below it
            faces = (('X', (sx / 2, 0, 0), (1, 0, 0), (0, 0, 1), min(sy, sz)),
                     ('Y', (0, sy / 2, 0), (0, 1, 0), (0, 0, 1), min(sx, sz)),
                     ('Z', (0, 0, sz / 2), (0, 0, 1), (1, 0, 0), min(sx, sy)))
            for letter, origin, normal, up, side in faces:
                placed = _on_face(_letter(letter, 0.6 * side), origin,
                                  np.array(normal, float), np.array(up, float),
                                  lift)
                pieces.append((*placed, f'mark:{letter}'))
        else:
            shafts_and_heads = _arrow(0.7 * sz)
            for center, normal in _arrow_sides(outline):
                origin = np.array([center[0], center[1], 0.0])
                normal = np.array([normal[0], normal[1], 0.0])
                for polygons, part in zip(shafts_and_heads, ('shaft', 'head')):
                    placed = _on_face(polygons, origin, normal,
                                      np.array([0.0, 0.0, 1.0]), lift)
                    pieces.append((*placed, f'mark:{part}'))
    all_points, all_triangles, parts = [], [], []
    offset = 0
    for points, triangles, part in pieces:
        all_points.append(points)
        all_triangles += [(a + offset, b + offset, c + offset)
                          for a, b, c in triangles]
        parts += [part] * len(triangles)
        offset += len(points)
    return (np.vstack(all_points), np.asarray(all_triangles, dtype=np.int64),
            parts)


def sensor_mesh(sensors: list[Sensor], geometry: Geometry,
                unit_system: UnitSystem | None = None,
                size: float = 1.0) -> SensorMesh | None:
    """One mesh of every sensor's body at its node, in display units.

    Parameters
    ----------
    sensors : list of Sensor
        From `core.sensors.place`.
    geometry : Geometry
        The geometry they are placed on.
    unit_system : UnitSystem, optional
        The display units; the default system when omitted.
    size : float, default 1.0
        A multiple of true size (or of the proportional size, when the
        geometry's units are not defined), about each node.

    Returns
    -------
    SensorMesh or None
        None when there are no sensors.
    """
    import pyvista as pv

    if not sensors:
        return None
    unit_system = unit_system or DEFAULT_SYSTEM
    notes = []
    if geometry.units_defined:
        per_mm = 1e-3
    else:
        xyz = np.asarray(geometry.node_xyz, dtype=float)
        extent = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))) or 1.0
        per_mm = NOMINAL_MM * extent
        notes.append(NOMINAL_NOTE)
    ids = np.asarray(geometry.node_id)
    colors = {'body': _rgb(BODY_COLOR), 'connector': _rgb(CONNECTOR_COLOR),
              'cable': _rgb(CABLE_COLOR), 'mark': _rgb(MARK_COLOR)}
    all_points, all_triangles, all_colors, all_rows = [], [], [], []
    count = 0
    for sensor in sensors:
        points, triangles, parts = body(sensor.entry, marks=sensor.marked)
        placed = sensor.center + (points * per_mm * size) @ sensor.frame.T
        all_points.append(placed)
        all_triangles.append(triangles + count)
        count += len(points)
        row = int(np.flatnonzero(ids == sensor.node)[0])
        all_rows.append(np.full(len(points), row))
        all_colors += [colors[part.split(':')[0]] for part in parts]
    points = np.vstack(all_points)
    if geometry.units_defined:
        points = np.asarray(unit_system.from_si(points, 'length'), dtype=float)
    triangles = np.vstack(all_triangles)
    faces = np.column_stack([np.full(len(triangles), 3), triangles]).ravel()
    vtk_points, view = shared_points(points)
    mesh = pv.PolyData()
    mesh.SetPoints(vtk_points)
    mesh.faces = faces
    mesh.cell_data['colors'] = np.asarray(all_colors, dtype=np.uint8)
    return SensorMesh(mesh, np.concatenate(all_rows),
                      np.ascontiguousarray(points), vtk_points, view,
                      list(sensors), notes)


def add_sensors(plotter: Any, geometry: Geometry, table: ChannelTable,
                unit_system: UnitSystem | None = None, size: float = 1.0,
                name: str = 'sensors') -> tuple[SensorMesh | None, list[str]]:
    """Draw the sensors a channel table places on a geometry.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Where they are drawn, as one actor named `name`.
    geometry : Geometry
        The geometry the table's nodes are in.
    table : ChannelTable
        The channels.
    unit_system : UnitSystem, optional
        The display units.
    size : float, default 1.0
        A multiple of true size, about each node.
    name : str, default 'sensors'
        The actor's name, so a redraw replaces it.

    Returns
    -------
    mesh : SensorMesh or None
        What was drawn, for an animation to move; None when the table
        places no sensor on the geometry.
    notes : list of str
        What the placement and the drawing say about it.
    """
    from ..core.sensors import place

    sensors, notes = place(table, geometry)
    drawn = sensor_mesh(sensors, geometry, unit_system, size)
    if drawn is None:
        return None, notes
    plotter.add_mesh(drawn.mesh, scalars='colors', rgb=True, name=name,
                     pickable=False, reset_camera=False, render=False,
                     show_scalar_bar=False)
    return drawn, notes + drawn.notes
