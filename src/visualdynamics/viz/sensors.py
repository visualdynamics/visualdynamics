"""Sensor bodies drawn at their nodes: one mesh for every sensor a
channel table places (`core.sensors.place`), so a model with hundreds of
sensors is one actor rather than hundreds.

Each body is built about its own origin from its catalog size, turned
into its frame and moved to its node, which is therefore the body's
center. The positive face of each sensing axis is drawn in that axis's
arrow color; the rest of the body is gray and the connector darker.

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
from .geometry import AXIS_COLORS, shared_points

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.channel_table import ChannelTable
    from ..core.geometry import Geometry
    from ..core.sensor_catalog import SensorModel
    from ..core.sensors import Sensor
    from ..units import UnitSystem

#: the body and its connector
BODY_COLOR = '#a7abb1'
CONNECTOR_COLOR = '#5d6166'

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


def body(entry: SensorModel):
    """A sensor's body in its own frame, in millimeters, centered on the
    origin: (points, triangles, parts), where a triangle's part is the
    face it is on, ('x', 1) for the +x face and so on, or 'connector'.
    A side of a hex or a puck counts as a face of the axis its outward
    normal is nearest, when it faces that axis squarely."""
    sx, sy, sz = entry.size
    if entry.shape == 'box':
        outline = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]]) / 2
        flat = sx / 2
    elif entry.shape == 'hex':
        # flats facing +-x, so the connector sits on one
        outline = _circle(sx / np.sqrt(3.0), 6, np.pi / 6)
        flat = sx / 2
    else:
        outline = _circle(sx / 2, ROUND)
        flat = sx / 2
    points, triangles, faces = _prism(outline, -sz / 2, sz / 2)
    parts = []
    for what, detail in faces:
        if what == 'z':
            parts.append(('z', detail))
            continue
        normal = np.asarray(detail, dtype=float)
        normal /= np.linalg.norm(normal)
        axis = int(np.argmax(np.abs(normal)))
        parts.append(('xy'[axis], int(np.sign(normal[axis])))
                     if abs(normal[axis]) > 0.999 else None)
    # the connector: a short cylinder out of the middle of its face
    sign = 1.0 if entry.connector == '+x' else -1.0
    small = min(sy, sz)
    radius, length = 0.18 * small, 0.4 * small
    ring = _circle(radius, 12)
    stub, stub_triangles, _stub_faces = _prism(ring, 0.0, length)
    # the stub's axis is z; turn it to point along +-x from the face
    stub = np.column_stack([sign * (flat + stub[:, 2]), stub[:, 0], stub[:, 1]])
    if sign < 0:
        stub_triangles = [(a, c, b) for a, b, c in stub_triangles]
    offset = len(points)
    points = np.vstack([points, stub])
    triangles = triangles + [(a + offset, b + offset, c + offset)
                             for a, b, c in stub_triangles]
    parts = parts + ['connector'] * len(stub_triangles)
    return points, np.asarray(triangles, dtype=np.int64), parts


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
              **{letter: _rgb(color) for letter, color
                 in zip('XYZ', AXIS_COLORS)}}
    all_points, all_triangles, all_colors, all_rows = [], [], [], []
    count = 0
    for sensor in sensors:
        points, triangles, parts = body(sensor.entry)
        placed = sensor.center + (points * per_mm * size) @ sensor.frame.T
        all_points.append(placed)
        all_triangles.append(triangles + count)
        count += len(points)
        row = int(np.flatnonzero(ids == sensor.node)[0])
        all_rows.append(np.full(len(points), row))
        for part in parts:
            if part == 'connector':
                all_colors.append(colors['connector'])
                continue
            letter = None
            if part is not None:
                axis = 'xyz'.index(part[0])
                if part[1] > 0:
                    letter = sensor.positive.get(axis)
            all_colors.append(colors[letter] if letter else colors['body'])
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
