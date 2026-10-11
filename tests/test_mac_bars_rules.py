"""The MAC bars as solids: each bar a closed box standing where its
cell is, as tall as its value, every outline and checker on it
(2026-10-11).

`tools/mutate.py` over viz/mac_bars.py caught 97 of 229 against
test_mac_bars.py, which checks the bars through pictures, picks and
counts; 132 survivors changed a corner index, a winding or an offset,
and a picture of a field of bars does not say which corner is where.
A plotter that keeps what it is given stands in for the window, and
the meshes are read for their geometry.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from visualdynamics.viz import mac_bars as mb


class _Kept:
    """A plotter that keeps every mesh by name and draws nothing."""

    def __init__(self):
        self.meshes, self.labels, self.camera_position = {}, None, None

    def add_mesh(self, mesh, name=None, **_style):
        self.meshes[name] = mesh

    def add_point_labels(self, points, names, **_style):
        self.labels = (np.asarray(points), list(names))

    def remove_actor(self, name):
        self.meshes.pop(name, None)

    def show_bounds(self, **_style):
        pass


MAC = np.array([[1.0, 0.25, 0.0], [0.1, 0.9, 0.5]])
TALL = mb.HEIGHT * 3
MARGIN = (1.0 - mb.FOOTPRINT) / 2.0


def _drawn(**marks):
    plotter = _Kept()
    mb.add_mac_bars(plotter, [10.0, 20.0], MAC, column_frequencies=[11.0, 22.0, 33.0],
                    **marks)
    return plotter


def _faces(mesh):
    return mesh.faces.reshape(-1, 5)[:, 1:]


def _expected_bounds(r, c, lift=0.0):
    height = max(MAC[r, c], mb.PLINTH) * TALL
    return ((c + MARGIN - lift, c + MARGIN + mb.FOOTPRINT + lift),
            (r + MARGIN - lift, r + MARGIN + mb.FOOTPRINT + lift),
            (0.0, height + lift))


def test_each_bar_is_a_closed_box_on_its_cell_as_tall_as_its_value():
    bars = _drawn().meshes['mac-bars']
    faces = _faces(bars)
    assert len(faces) == 6 * MAC.size
    points = np.asarray(bars.points)
    for box in range(MAC.size):
        r, c = mb.cell_to_pair(6 * box, MAC.shape[1])
        quads = faces[6 * box:6 * box + 6]
        corners = points[np.unique(quads)]
        assert len(corners) == 8
        for axis, (low, high) in enumerate(_expected_bounds(r, c)):
            assert sorted(set(np.round(corners[:, axis], 12))) == pytest.approx(
                [low, high]), (r, c, axis)
        # closed: every edge is shared by exactly two of its faces
        edges = Counter(tuple(sorted((int(q[k]), int(q[(k + 1) % 4]))))
                        for q in quads for k in range(4))
        assert len(edges) == 12 and set(edges.values()) == {2}, (r, c)
        # and each face is a flat rectangle on one side of the box
        for quad in quads:
            spread = np.ptp(points[quad], axis=0)
            assert np.count_nonzero(spread < 1e-12) == 1, (r, c, quad)
    assert np.array_equal(bars.cell_data['mac'], np.repeat(MAC.ravel(), 6))


def test_every_bar_wears_its_twelve_edges_lifted_off_it():
    edges = _drawn().meshes['mac-bars-edges']
    lines = edges.lines.reshape(-1, 3)[:, 1:]
    points = np.asarray(edges.points)
    assert len(lines) == 12 * MAC.size
    for box in range(MAC.size):
        r, c = divmod(box, MAC.shape[1])
        (x0, x1), (y0, y1), (_z0, z1) = _expected_bounds(r, c, lift=mb.EDGE_LIFT)
        mine = lines[12 * box:12 * box + 12]
        lengths = sorted(np.round(np.linalg.norm(points[mine[:, 1]] - points[mine[:, 0]],
                                                 axis=1), 12))
        assert lengths == pytest.approx(sorted([x1 - x0] * 4 + [y1 - y0] * 4 + [z1] * 4))
        degree = Counter(int(n) for n in mine.ravel())
        assert len(degree) == 8 and set(degree.values()) == {3}, (r, c)


def test_a_selection_outlines_only_its_bars_and_the_active_one_again():
    plotter = _drawn(selected=[(0, 1), (5, 5)], active=(0, 1))
    selected = plotter.meshes['mac-bars-selected']
    assert len(selected.lines.reshape(-1, 3)) == 12, 'the pair off the grid is dropped'
    (x0, x1), (y0, y1), (_z0, z1) = _expected_bounds(0, 1, lift=mb.MARK_LIFT)
    assert np.allclose(np.asarray(selected.bounds), [x0, x1, y0, y1, 0.0, z1])
    assert 'mac-bars-active' in plotter.meshes
    assert 'mac-bars-active' not in _drawn(selected=[(0, 1)], active=(1, 1)).meshes


def test_a_matched_bar_is_checkered_on_its_top_and_four_sides():
    """Every face's squares tile half of it — the even ones — and lie in
    that face, so the mark reads from any side."""
    checkers = _drawn(matched=[(1, 2), (9, 9)]).meshes['mac-bars-matched']
    (x0, x1), (y0, y1), (_z0, z1) = _expected_bounds(1, 2, lift=mb.MARK_LIFT)
    quads = _faces(checkers)
    points = np.asarray(checkers.points)
    dx, dy = x1 - x0, y1 - y0
    # face: (the axis it is square to, where, its along side, its up side)
    sides = {'top': (2, z1, dx, dy), 'front': (1, y0, dx, z1), 'back': (1, y1, dx, z1),
             'left': (0, x0, dy, z1), 'right': (0, x1, dy, z1)}
    covered = dict.fromkeys(sides, 0.0)
    for quad in quads:
        corner = points[quad]
        area = float(np.linalg.norm(np.cross(corner[1] - corner[0], corner[3] - corner[0])))
        face = [name for name, (axis, at, _u, _v) in sides.items()
                if np.allclose(corner[:, axis], at)]
        assert len(face) == 1, corner
        covered[face[0]] += area
    for name, (_axis, _at, along, up) in sides.items():
        squares = mb.CHECKER * max(1, round(mb.CHECKER * up / along))
        kept = (squares + 1) // 2
        assert covered[name] == pytest.approx(along * up * kept / squares), name


def test_the_labels_sit_off_the_grid_beside_their_row_and_column():
    points, names = _drawn().labels
    assert names == ['10.0', '20.0', '11.0', '22.0', '33.0']
    assert points.tolist() == [[-0.35, 0.5, 0.0], [-0.35, 1.5, 0.0],
                               [0.5, -0.35, 0.0], [1.5, -0.35, 0.0], [2.5, -0.35, 0.0]]


def test_the_home_view_stands_front_left_of_the_grid_and_looks_into_it():
    plotter = _Kept()
    mb.place_camera(plotter, 2, 3)
    eye, focus, _up = plotter.camera_position
    assert eye == pytest.approx((1.5 - 3.3, -4.2, 3.6))
    assert focus == pytest.approx((1.5, 0.84, 0.36))
