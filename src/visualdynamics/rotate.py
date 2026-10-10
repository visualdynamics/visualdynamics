"""Turning a coordinate system by dragging a ring around one of its axes.

The geometry and the angle arithmetic live here, apart from Qt and VTK, so
the fiddly part — which way a drag turns the frame, and by how much — can be
tested without a window.

A coordinate system is stored as four rows: three orthonormal basis vectors
and an origin. Rotation only ever touches the basis; the origin stays put.

Sliding is the other gesture (2026-10-02): an arrow along each axis of the
frame, dragged, moves the origin along that axis and snaps it to a grid in
the geometry's own axes — a tenth of an inch for the inch family of units,
a centimetre for the metric one — so a frame or a block lands on round
coordinates whichever way it is turned. The arrow arithmetic lives here
beside the ring arithmetic, for the same reason.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

RING_STEPS = 96          # points around a ring: enough that picking is smooth


def rotation_about(axis: ArrayLike, angle: float) -> np.ndarray:
    """The 3x3 rotation of `angle` radians about a unit `axis`.

    Rodrigues' formula, written out rather than pulled from a library:
    scipy is not a dependency and this is four lines.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return (np.eye(3) + np.sin(angle) * cross
            + (1.0 - np.cos(angle)) * (cross @ cross))


def ring_points(matrix: ArrayLike, axis: int, radius: float,
                steps: int = RING_STEPS) -> np.ndarray:
    """A closed ring of points around one principal axis of a frame.

    The ring for the X axis lies in the frame's own Y-Z plane, so it is the
    circle you would sweep by turning about X.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    origin = matrix[3]
    first, second = ((axis + 1) % 3, (axis + 2) % 3)
    angle = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)[:, np.newaxis]
    return origin + radius * (np.cos(angle) * matrix[first]
                              + np.sin(angle) * matrix[second])


def rotate_frame(matrix: ArrayLike, axis: int, angle: float) -> np.ndarray:
    """A copy of `matrix` with its basis turned about one of its own axes.

    The axis travels with the frame — turning about X twice by 45 degrees
    is turning about that same X by 90, not about the world's X.
    """
    matrix = np.array(matrix, dtype=np.float64)
    turned = np.array(matrix)
    turned[:3] = matrix[:3] @ rotation_about(matrix[axis], angle).T
    return turned


def angle_in_plane(matrix: ArrayLike, axis: int,
                   point: ArrayLike) -> float:
    """Where a world point sits around one principal axis, in radians.

    Measured in the ring's own plane from the first of the two axes it
    spans, so it pairs with `ring_points`.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(point, dtype=np.float64) - matrix[3]
    first, second = ((axis + 1) % 3, (axis + 2) % 3)
    return float(np.arctan2(offset @ matrix[second], offset @ matrix[first]))


def wrapped(angle: float) -> float:
    """An angle folded into [-pi, pi), so a drag past the seam stays small.

    A drag from 179 to -179 degrees is two degrees, not 358.
    """
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def ring_under_cursor(rings: Sequence[tuple[int, np.ndarray]],
                      cursor: ArrayLike,
                      tolerance: float = 14.0) -> int | None:
    """Which ring a pixel is over, or None.

    `rings` is a list of (axis, screen points), each a polyline — a ring's
    closed back on its first point. The nearest ring within the
    tolerance wins; the rings cross each other, so ties go to whichever is
    genuinely closer rather than to whichever was drawn first.

    Distance is to the line, not to its points. It was to the points
    until 2026-10-10, and a ring 400 pixels across drawn from 96 of them
    has 26 pixels between neighbours — so a click on the ring itself,
    midway between two, could miss a 14-pixel tolerance.
    """
    from .viz.pick import segment_distances

    cursor = np.asarray(cursor, dtype=np.float64)
    best, best_distance = None, tolerance
    for axis, points in rings:
        points = np.asarray(points, dtype=np.float64)
        if not len(points):
            continue
        if len(points) == 1:
            distance = float(np.linalg.norm(points[0] - cursor))
        else:
            distance = float(segment_distances(cursor, points[:-1],
                                               points[1:]).min())
        if distance < best_distance:
            best, best_distance = axis, distance
    return best


def plane_hit(origin: ArrayLike, normal: ArrayLike, eye: ArrayLike,
              direction: ArrayLike) -> np.ndarray | None:
    """Where a ray meets the plane through `origin`, or None if parallel.

    Dragging a ring means following the cursor around the plane the ring
    lies in, which is what this finds.
    """
    normal = np.asarray(normal, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    along = float(normal @ direction)
    if abs(along) < 1e-9:
        return None
    distance = float(normal @ (np.asarray(origin, dtype=np.float64)
                               - np.asarray(eye, dtype=np.float64))) / along
    return np.asarray(eye, dtype=np.float64) + distance * direction


def identity_frame(matrix: ArrayLike) -> np.ndarray:
    """The frame with its rotation undone, keeping where it sits."""
    matrix = np.asarray(matrix, dtype=np.float64)
    reset = np.zeros_like(matrix)
    reset[:3] = np.eye(3)
    reset[3] = matrix[3]
    return reset


# ---- sliding --------------------------------------------------------------

#: the units a tenth of an inch is the grid for; everything else is metric
INCH_FAMILY = frozenset({'in', 'inch', 'inches', 'ft', 'foot', 'feet',
                         'mil', 'mils', 'thou'})

ARROW_STEPS = 24         # points along an arrow's shaft, for picking


def grid_step(unit: str) -> float:
    """The grid a slide lands on, in the display `unit`: a tenth of an
    inch for the inch family, a centimetre for the metric one."""
    from .units import convert

    if unit in INCH_FAMILY:
        return float(convert(0.1, 'in', unit))
    return float(convert(1.0, 'cm', unit))


def arrow_points(matrix: ArrayLike, axis: int, length: float,
                 steps: int = ARROW_STEPS) -> np.ndarray:
    """Points along the arrow for one principal axis of a frame, from
    the origin out to `length` along it — a polyline the way a ring is,
    so `ring_under_cursor` picks an arrow the same way."""
    matrix = np.asarray(matrix, dtype=np.float64)
    along = np.linspace(0.0, length, steps)[:, None]
    return matrix[3] + along * matrix[axis]


def translate_frame(matrix: ArrayLike, axis: int, distance: float) -> np.ndarray:
    """A copy of `matrix` with its origin slid `distance` along one of
    its own axes; the basis stays put, the mirror of `rotate_frame`."""
    moved = np.array(matrix, dtype=np.float64)
    moved[3] = moved[3] + float(distance) * moved[axis]
    return moved


def axis_hit(origin: ArrayLike, direction: ArrayLike, eye: ArrayLike,
             ray: ArrayLike) -> np.ndarray | None:
    """The point on the axis line nearest the cursor's ray, or None when
    the two are parallel.

    Dragging an arrow means following the cursor along a line the cursor
    can only ever be near, never on: the closest approach of the two
    lines is where the drag is read, which is what this finds.
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    ray = np.asarray(ray, dtype=np.float64)
    w0 = origin - eye
    a, b, c = direction @ direction, direction @ ray, ray @ ray
    d, e = direction @ w0, ray @ w0
    denominator = a * c - b * b
    if abs(denominator) < 1e-12 * max(a * c, 1e-300):
        return None
    along = (b * e - c * d) / denominator
    return origin + along * direction


def distance_along(matrix: ArrayLike, axis: int, world: ArrayLike) -> float:
    """How far a world point sits from the frame's origin along one of
    its axes: the drag's reading, before the snap."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return float((np.asarray(world, dtype=np.float64) - matrix[3]) @ matrix[axis])


def snapped(point: ArrayLike, step: float) -> np.ndarray:
    """`point` moved to the nearest grid line in each coordinate — the
    geometry's own axes, so a turned frame still lands on round
    coordinates. A step of zero or less is no snap."""
    point = np.asarray(point, dtype=np.float64)
    if step <= 0.0:
        return point.copy()
    return np.round(point / step) * step


# ---- angles -----------------------------------------------------------------


def frame_from_angles(angles: ArrayLike, origin: ArrayLike = (0.0, 0.0, 0.0)
                      ) -> np.ndarray:
    """A frame turned by `angles`, in degrees about the geometry's fixed
    X, then Y, then Z axes — the order a person types them in — at
    `origin`. The rows are the frame's axes, as a coordinate system's
    are."""
    ax, ay, az = (np.radians(float(a)) for a in angles)
    turn = (rotation_about((0.0, 0.0, 1.0), az) @ rotation_about((0.0, 1.0, 0.0), ay)
            @ rotation_about((1.0, 0.0, 0.0), ax))
    frame = np.zeros((4, 3))
    frame[:3] = turn.T
    frame[3] = np.asarray(origin, dtype=np.float64)
    return frame


def angles_of(matrix: ArrayLike) -> tuple[float, float, float]:
    """The fixed-axis X, Y, Z angles in degrees that turn the geometry's
    axes into the frame's (`frame_from_angles` undone), each in
    (-180, 180]. At a quarter turn about Y the X and Z turns are one
    turn, and it is said as Z with X at zero."""
    turn = np.asarray(matrix, dtype=np.float64)[:3].T
    ay = float(np.arcsin(np.clip(-turn[2, 0], -1.0, 1.0)))
    if abs(np.cos(ay)) > 1e-9:
        ax = float(np.arctan2(turn[2, 1], turn[2, 2]))
        az = float(np.arctan2(turn[1, 0], turn[0, 0]))
    else:
        ax = 0.0
        az = float(np.arctan2(-turn[0, 1], turn[1, 1]))
    return tuple(float(np.round(np.degrees(a), 9)) + 0.0 for a in (ax, ay, az))
