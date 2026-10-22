"""Finding the entity under the cursor.

Picking is done in screen space: the geometry is projected through the
camera once, and the pick is the entity nearest the cursor in pixels.
That answers the question being asked, what is under the pointer, even
where a ray would hit nothing, which matters for a wireframe, where most
of the screen is empty space between thin lines.

The projection is cached and rebuilt only when the camera moves or the
view is resized, so a mouse move costs one vectorized distance
computation. It uses VTK's camera and numpy only, with no other
dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.geometry import Geometry


class ScreenProjector:
    """Projects `points` (a geometry's nodes, as drawn) to pixels in
    `renderer`, caching the result until the camera moves or the
    renderer is resized."""

    def __init__(self, renderer: Any, points: ArrayLike) -> None:
        self.renderer: Any = renderer
        self.points: np.ndarray = np.ascontiguousarray(points, dtype=np.float64)
        self._homogeneous = np.column_stack(
            [self.points, np.ones(len(self.points))])
        self._key = None
        self._screen: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._inverse = None
        self._size = (1, 1)

    def _camera_key(self):
        camera = self.renderer.GetActiveCamera()
        size = tuple(self.renderer.GetSize())
        return (camera.GetMTime(), size)

    def screen(self) -> tuple[np.ndarray, np.ndarray]:
        """(pixels (N, 2), depth (N,)) for every node, projected again
        only when the camera or the size has changed."""
        key = self._camera_key()
        if key != self._key:
            self._project()
            self._key = key
        screen, depth = self._screen, self._depth
        assert screen is not None and depth is not None
        return screen, depth

    def _project(self):
        width, height = self.renderer.GetSize()
        width, height = max(int(width), 1), max(int(height), 1)
        camera = self.renderer.GetActiveCamera()
        matrix = camera.GetCompositeProjectionTransformMatrix(
            width / height, -1.0, 1.0)
        transform = np.array([[matrix.GetElement(r, c) for c in range(4)]
                              for r in range(4)])
        self._size = (width, height)
        self._inverse = np.linalg.inv(transform)
        clip = self._homogeneous @ transform.T
        w = clip[:, 3:4]
        w[np.abs(w) < 1e-12] = 1e-12            # points on the camera plane
        ndc = clip[:, :3] / w
        self._screen = np.column_stack([
            (ndc[:, 0] * 0.5 + 0.5) * width,
            (ndc[:, 1] * 0.5 + 0.5) * height])
        self._depth = ndc[:, 2]

    def invalidate(self) -> None:
        """Drop the cached projection, so the next use projects again."""
        self._key = None

    def reference_depth(self, x: float, y: float) -> float:
        """The depth to place a new point at: that of the node nearest
        (x, y) on screen, or 0 for an empty model.

        A click gives two dimensions, and the third has to come from
        somewhere. Using the nearest node's depth puts the new point on a
        plane through the model near where the click was, which is where
        the person is looking.
        """
        screen, depth = self.screen()
        if not len(screen):
            return 0.0                 # empty model: `place_point` uses the focal plane
        distances = np.linalg.norm(screen - np.array([x, y]), axis=1)
        return float(depth[int(np.argmin(distances))])

    def place_point(self, x: float, y: float) -> np.ndarray:
        """Where to put a new point clicked at (x, y), in world
        coordinates.

        The click gives two dimensions, and the third comes from the
        model. For a flat model (a plate, a panel) the click ray is
        intersected with the model's own plane through the node nearest
        the click, so the new point lands on the model and not above it.
        For any other model, or when the view is edge-on to the plane,
        the point is placed at the nearest node's depth
        (`reference_depth`), which keeps it near what was clicked.
        """
        screen, _ = self.screen()
        if not len(screen):
            return self.on_focal_plane(x, y)

        distances = np.linalg.norm(screen - np.array([x, y]), axis=1)
        anchor = self.points[int(np.argmin(distances))]

        near = self.unproject(x, y, -1.0)
        far = self.unproject(x, y, 1.0)
        direction = far - near
        normal = self._plane_normal()
        if normal is None:
            return self.unproject(x, y)          # not planar: use the depth
        denominator = float(np.dot(direction, normal))
        if abs(denominator) < 1e-9:              # looking along the plane
            return self.unproject(x, y)
        t = float(np.dot(anchor - near, normal)) / denominator
        return near + t * direction

    def on_focal_plane(self, x: float, y: float) -> np.ndarray:
        """Where the ray through pixel (x, y) crosses the plane through
        the camera's focal point, square to the view: where a first point
        goes when the model has none to take a depth from.

        A fixed depth in normalized device coordinates is not that plane:
        in perspective it lies close to the camera, and a first node
        clicked at the middle of an empty view landed about an inch
        from where the camera stood (2026-10-21)."""
        camera = self.renderer.GetActiveCamera()
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        normal = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
        near = self.unproject(x, y, -1.0)
        direction = self.unproject(x, y, 1.0) - near
        denominator = float(np.dot(direction, normal))
        if abs(denominator) < 1e-12:
            return focal
        return near + float(np.dot(focal - near, normal)) / denominator * direction

    def _plane_normal(self, flatness=0.02):
        """The model's plane normal, if it is essentially flat."""
        if len(self.points) < 3:
            return None
        centered = self.points - self.points.mean(axis=0)
        _u, singular, vectors = np.linalg.svd(centered, full_matrices=False)
        if singular[1] <= 0 or singular[2] / singular[1] > flatness:
            return None                          # genuinely three-dimensional
        return vectors[2]

    def unproject(self, x: float, y: float,
                  depth: float | None = None) -> np.ndarray:
        """The world point under a pixel, at `depth` in normalized device
        coordinates (-1 near, 1 far), or at the nearest node's depth
        when None."""
        self.screen()                  # make sure the transform is current
        if depth is None:
            depth = self.reference_depth(x, y)
        width, height = self._size
        ndc = np.array([2.0 * x / width - 1.0,
                        2.0 * y / height - 1.0,
                        depth, 1.0])
        world = self._inverse @ ndc
        return world[:3] / world[3]


#: how close, in points, two entities' distances must be for depth to
#: decide between them
TIE_POINTS = 2.0


def _nearest_candidate(distances, depth, tolerance, tie):
    """Index of the entity under the cursor.

    Screen distance decides; depth only separates entities that are within
    `tie` pixels of each other, so something directly under the pointer is
    never lost to something further away that happens to be nearer the
    camera.
    """
    candidates = np.flatnonzero(distances <= tolerance)
    if not len(candidates):
        return None
    closest = distances[candidates].min()
    contenders = candidates[distances[candidates] <= closest + tie]
    return int(contenders[np.argmin(depth[contenders])])


def _cross2(u, v):
    """Scalar cross product of 2D vectors; its sign says which side."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def segment_distances(point, starts, ends):
    """Distance from a 2-D point to each of many 2-D segments, given as
    arrays of `starts` and `ends`."""
    segment = ends - starts
    length2 = np.einsum('ij,ij->i', segment, segment)
    length2[length2 == 0] = 1e-12
    t = np.clip(np.einsum('ij,ij->i', point - starts, segment) / length2, 0, 1)
    closest = starts + t[:, np.newaxis] * segment
    return np.linalg.norm(point - closest, axis=1)


class EntityPicker:
    """Which node, coordinate system or element is under a pixel.

    `kind` is 'nodes', 'coordinate_systems' or 'elements'. The candidate
    geometry is reduced once to points, or to segments between node rows
    (and, for elements, triangles of the face interiors). A pick then
    projects them, measures the distance in pixels and takes the nearest
    within the tolerance, breaking near ties by depth so the front-most
    wins.
    """

    def __init__(self, geometry: Geometry, kind: str,
                 projector: ScreenProjector) -> None:
        self.geometry: Any = geometry
        self.kind: str = kind
        self.projector: ScreenProjector = projector
        self.rows: np.ndarray | None = None      # for point-like kinds
        self.segments: tuple[np.ndarray, np.ndarray] | None = None
        self.owner: np.ndarray | None = None     # entity per segment
        #: face interiors, so clicking inside a face works
        self.triangles: np.ndarray | None = None
        self.triangle_owner: np.ndarray | None = None
        self._prepare()

    def _prepare(self):
        geometry = self.geometry
        row_of = {int(node): row for row, node in enumerate(geometry.node_id)}

        if self.kind == 'nodes':
            self.rows = np.arange(geometry.num_nodes)
            return
        if self.kind == 'coordinate_systems':
            # origins are not nodes, so they are projected separately
            self.rows = None
            return

        connectivity = geometry.elem_conn
        starts, ends, owners = [], [], []
        for index, nodes in enumerate(connectivity):
            rows = [row_of[int(node)] for node in nodes if int(node) in row_of]
            if len(rows) < 2:
                if rows:                      # a point element: a degenerate
                    starts.append(rows[0])    # segment keeps the math uniform
                    ends.append(rows[0])
                    owners.append(index)
                continue
            closed = (self.kind == 'elements' and len(rows) > 2)
            ordered = rows + rows[:1] if closed else rows
            for a, b in pairwise(ordered):
                starts.append(a)
                ends.append(b)
                owners.append(index)
        self.segments = (np.asarray(starts, dtype=np.int64),
                         np.asarray(ends, dtype=np.int64))
        self.owner = np.asarray(owners, dtype=np.int64)
        if self.kind == 'elements':
            self._prepare_faces(connectivity, row_of)

    def _prepare_faces(self, connectivity, row_of):
        """Fan-triangulate face elements so their interiors are pickable."""
        from ..core.geometry import ELEMENT_TYPES

        triangles, owners = [], []
        for index, nodes in enumerate(connectivity):
            kind = ELEMENT_TYPES.get(int(self.geometry.elem_type[index]))
            if kind is None or kind[2] != 'face':
                continue
            rows = [row_of[int(node)] for node in nodes if int(node) in row_of]
            for a, b in pairwise(rows[1:]):
                triangles.append((rows[0], a, b))
                owners.append(index)
        self.triangles = (np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
                          if triangles else np.empty((0, 3), dtype=np.int64))
        self.triangle_owner = np.asarray(owners, dtype=np.int64)

    def pick(self, x: float, y: float, tolerance: float = 12.0,
             pixel_ratio: float = 1.0) -> int | None:
        """The entity under (x, y), or None.

        `x` and `y` are in the device pixels VTK reports a cursor in and
        the projector projects to. `tolerance` is in points, and
        `pixel_ratio` is device pixels per point (2 on a Retina screen).
        An element whose face contains the cursor is picked first; past
        that, the nearest within the tolerance. Returns a node id, a
        coordinate system id or an element index, which is how the rest
        of the code identifies each kind.
        """
        # Until 2026-10-10 the tolerance was in pixels, which on a Retina
        # screen was a 6-point target.
        screen, depth = self.projector.screen()
        cursor = np.array([float(x), float(y)])
        tolerance = tolerance * pixel_ratio
        tie = TIE_POINTS * pixel_ratio

        if self.kind == 'coordinate_systems':
            return self._pick_origin(cursor, tolerance, tie)

        if self.kind == 'nodes':
            if not len(screen):
                return None
            distances = np.linalg.norm(screen - cursor, axis=1)
            nearest = _nearest_candidate(distances, depth, tolerance, tie)
            if nearest is None:
                return None
            return int(self.geometry.node_id[nearest])

        inside = self._pick_face(cursor, screen, depth)
        if inside is not None:
            return inside

        starts, ends = self.segments
        if not len(starts):
            return None
        distances = segment_distances(cursor, screen[starts], screen[ends])
        midpoint_depth = 0.5 * (depth[starts] + depth[ends])
        nearest = _nearest_candidate(distances, midpoint_depth, tolerance, tie)
        return None if nearest is None else int(self.owner[nearest])

    def _pick_face(self, cursor, screen, depth):
        """The front-most face whose projected interior contains the cursor."""
        if self.triangles is None or not len(self.triangles):
            return None
        a, b, c = (screen[self.triangles[:, i]] for i in range(3))
        d1 = _cross2(b - a, cursor - a)
        d2 = _cross2(c - b, cursor - b)
        d3 = _cross2(a - c, cursor - c)
        inside = (((d1 >= 0) & (d2 >= 0) & (d3 >= 0))
                  | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0)))
        hits = np.flatnonzero(inside)
        if not len(hits):
            return None
        center_depth = depth[self.triangles[hits]].mean(axis=1)
        return int(self.triangle_owner[hits[np.argmin(center_depth)]])

    def _pick_origin(self, cursor, tolerance, tie):
        origins = np.asarray(self.geometry.cs_matrix[:, 3, :], dtype=np.float64)
        projector = ScreenProjector(self.projector.renderer, origins)
        screen, depth = projector.screen()
        if not len(screen):
            return None
        distances = np.linalg.norm(screen - cursor, axis=1)
        nearest = _nearest_candidate(distances, depth, tolerance, tie)
        return None if nearest is None else int(self.geometry.cs_id[nearest])


class FacePicker:
    """Which of a set of polygons is under a pixel: the front-most one
    whose projected interior holds it.

    `polygons` are rows into the projector's points, each listed in
    order around it, and `owner` names what each one belongs to; a pick
    returns the owner. Built once, since fanning tens of thousands of
    faces into triangles is the slow part; a pick projects the points
    again only when the camera has moved.
    """

    def __init__(self, projector: ScreenProjector, polygons: Sequence[Sequence[int]],
                 owner: Sequence[int]) -> None:
        self.projector: ScreenProjector = projector
        triangles, owners = [], []
        for rows, who in zip(polygons, owner, strict=True):
            for a, b in pairwise(rows[1:]):
                triangles.append((rows[0], a, b))
                owners.append(who)
        self.triangles: np.ndarray = (np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
                                      if triangles else np.empty((0, 3), dtype=np.int64))
        self.owner: np.ndarray = np.asarray(owners, dtype=np.int64)

    def pick(self, x: float, y: float) -> int | None:
        """The owner of the front-most polygon under the device pixel
        `(x, y)`, or None."""
        if not len(self.triangles):
            return None
        screen, depth = self.projector.screen()
        cursor = np.array([float(x), float(y)])
        a, b, c = (screen[self.triangles[:, i]] for i in range(3))
        d1 = _cross2(b - a, cursor - a)
        d2 = _cross2(c - b, cursor - b)
        d3 = _cross2(a - c, cursor - c)
        inside = (((d1 >= 0) & (d2 >= 0) & (d3 >= 0))
                  | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0)))
        hits = np.flatnonzero(inside)
        if not len(hits):
            return None
        center_depth = depth[self.triangles[hits]].mean(axis=1)
        return int(self.owner[hits[np.argmin(center_depth)]])
