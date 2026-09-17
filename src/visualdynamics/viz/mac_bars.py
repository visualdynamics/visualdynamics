"""The MAC matrix as 3-D bars: height and color are both the value.

The flat grid answers "which pairs match" at a glance; the bars answer
the follow-up the grid is poor at — *how much* the near-misses differ,
which as color alone is a judgment of shade. Height makes a 0.6
against a 0.9 a visible step. The reading is pinned 0..1 exactly like
the flat grid and the coherence map, because a MAC is a bounded ratio.

One mesh whatever the mode count, like the waterfall: a 139×139
cross-MAC is 19 321 boxes and still one actor. Rows are the first
set's modes and columns the second's, labeled by frequency the way
the flat grid's axes are, and thinned past the same label limit.
Row 0 sits nearest the viewer — the flat grid reads downward from the
top-left, and the camera puts the same first mode in front.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..theme import theme as resolve_theme
from .waterfall import LABEL_LIMIT

if TYPE_CHECKING:
    from collections.abc import Sequence

#: bar footprint within its unit cell. The gap between neighbors is
#: what lets a row of near-1.0 bars read as bars rather than a wall.
FOOTPRINT = 0.78

#: a full MAC of 1.0 stands this fraction of the grid's larger side
#: tall — proportional, so a big cross-MAC keeps its skyline instead of
#: flattening into tiles
HEIGHT = 0.4

#: a zero cell still shows a plinth this fraction of full height, so an
#: empty cell reads as measured-and-nothing rather than as a hole
PLINTH = 0.004

#: the committed-match red — the same literal in both readings, so a
#: match reads the same color on the grid and on the bars
MARK_RED = '#e5534b'

#: the picked-pair blue, for the outlines in both readings. It used to
#: be the same red as the checker, and a red outline on a red-checkered
#: bar could not be seen — the two marks answer different questions
#: (picked now vs committed already) and now wear different colors
PICK_BLUE = '#0a84ff'

#: checker squares across a face's short side. Committed matches wear
#: a red checker (Brandon's call, after trying diagonal stripes): the
#: long side takes however many squares keep the checkers near-square,
#: so a tall bar reads checkered rather than banded
CHECKER = 4

#: how far a mark stands off the bar it marks, in cell units — enough
#: that lines never z-fight the faces they trace
MARK_LIFT = 0.02

#: the outline every bar wears. A field of near-1.0 MACs is a field of
#: one color — viridis has nowhere left to go above 0.95 — and the
#: gap between neighbors hides behind the bars themselves at any
#: camera but straight down, so a well-correlated pair read as a solid
#: yellow slab with the answer somewhere inside it (Brandon,
#: 2026-08-26). The outline is what puts the bars back.
#:
#: Black on both themes on purpose: its job is to divide bar from bar,
#: which it does on white and on black alike, and where it meets the
#: background it simply disappears into it — no loss, since the
#: silhouette is not the edge that was missing.
EDGE_INK = '#000000'

#: how far the plain outline stands off its bar — half the marks',
#: so a selected bar's blue frame sits outside the black one instead
#: of fighting it for the same depth
EDGE_LIFT = MARK_LIFT / 2.0

#: the largest grid that gets outlined. Past this a bar is a pixel or
#: two wide and the outlines close over the field into a dark mat —
#: the opposite of the point. Measured on a 139x139 cross-MAC, where
#: even hairlines swallowed the color (Brandon, 2026-08-26).
OUTLINE_LIMIT = 40


def cell_to_pair(cell: int, columns: int) -> tuple[int, int]:
    """Which (row, column) a picked mesh cell belongs to.

    The bars mesh is built box by box, six faces each, in row-major
    order — this is that construction read backwards, and the picking
    depends on it staying true."""
    box = int(cell) // 6
    return box // int(columns), box % int(columns)


def add_mac_bars(plotter: Any, frequencies: Sequence[float],
                 matrix: Any,
                 column_frequencies: Sequence[float] | None = None,
                 theme: Any = None,
                 selected: Sequence[tuple[int, int]] = (),
                 active: tuple[int, int] | None = None,
                 matched: Sequence[tuple[int, int]] = ()
                 ) -> dict[str, Any]:
    """Draw the MAC bars into a plotter: one mesh, labels, the 0..1 axis.

    The marks mirror the flat grid's, in the same red: `selected` pairs
    wear an outline traced around their bar, the `active` (animated)
    pair the boldest one, and `matched` pairs — already committed to
    the matched-modes table — wear the diagonal stripes across their
    bar's top. Returns {'rows', 'columns', 'named_rows',
    'named_columns'} — the named lists say which modes got a frequency
    label, for the same thinning test the waterfall's labels have.
    """
    import pyvista as pv

    colors = resolve_theme(theme)
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    rows, columns = matrix.shape
    row_f = np.asarray(frequencies, dtype=float)
    col_f = (row_f if column_frequencies is None
             else np.asarray(column_frequencies, dtype=float))
    # refused at entry rather than an IndexError from the label loop: a
    # rectangular cross-MAC must bring the second set's frequencies
    if len(row_f) != rows or len(col_f) != columns:
        raise ValueError(
            f'a {rows}x{columns} MAC needs {rows} row and {columns} '
            f'column frequencies, got {len(row_f)} and {len(col_f)}')

    tall = HEIGHT * max(rows, columns)
    heights = np.maximum(matrix, PLINTH) * tall
    margin = (1.0 - FOOTPRINT) / 2.0
    j, i = np.meshgrid(np.arange(columns), np.arange(rows))
    x0 = (j + margin).ravel()
    y0 = (i + margin).ravel()
    z1 = heights.ravel()
    boxes = rows * columns
    corners = np.empty((boxes, 8, 3))
    # bottom ring 0-3 and top ring 4-7, both wound the same way
    corners[:, [0, 3, 4, 7], 0] = x0[:, None]
    corners[:, [1, 2, 5, 6], 0] = (x0 + FOOTPRINT)[:, None]
    corners[:, [0, 1, 4, 5], 1] = y0[:, None]
    corners[:, [2, 3, 6, 7], 1] = (y0 + FOOTPRINT)[:, None]
    corners[:, :4, 2] = 0.0
    corners[:, 4:, 2] = z1[:, None]
    sides = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                      [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    quads = (8 * np.arange(boxes)[:, None, None] + sides[None]).reshape(-1, 4)
    faces = np.hstack([np.full((len(quads), 1), 4), quads]).ravel()
    mesh = pv.PolyData(corners.reshape(-1, 3), faces=faces)
    # the value colors every face of its box — pinned 0..1, the same
    # scale as the flat grid, never the matrix's own extremes
    mesh.cell_data['mac'] = np.repeat(matrix.ravel(), 6)
    plotter.add_mesh(mesh, scalars='mac', cmap='viridis', clim=(0.0, 1.0),
                     show_scalar_bar=False, name='mac-bars')

    named_rows = list(range(0, rows, max(1, -(-rows // LABEL_LIMIT))))
    named_columns = list(range(0, columns,
                               max(1, -(-columns // LABEL_LIMIT))))
    spots, names = [], []
    for i in named_rows:
        spots.append([-0.35, i + 0.5, 0.0])
        names.append(f'{row_f[i]:.1f}')
    for j in named_columns:
        spots.append([j + 0.5, -0.35, 0.0])
        names.append(f'{col_f[j]:.1f}')
    plotter.add_point_labels(
        np.asarray(spots), names, font_size=11, always_visible=True,
        text_color=colors['scene_text'], shape=None, fill_shape=False,
        show_points=False, name='mac-bars-labels')
    def bar_frame(r, c, lift=MARK_LIFT):
        """The eight corners of (r, c)'s bar, inflated by the lift."""
        x0 = c + margin - lift
        x1 = c + margin + FOOTPRINT + lift
        y0 = r + margin - lift
        y1 = r + margin + FOOTPRINT + lift
        z1 = heights[r, c] + lift
        return x0, x1, y0, y1, z1

    def outline_lines(cells, lift=MARK_LIFT):
        """One line mesh tracing the twelve edges of each cell's bar."""
        points, lines = [], []
        for r, c in cells:
            x0, x1, y0, y1, z1 = bar_frame(r, c, lift)
            base = len(points)
            points.extend([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0),
                           (x0, y1, 0), (x0, y0, z1), (x1, y0, z1),
                           (x1, y1, z1), (x0, y1, z1)])
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
                lines.append([2, base + a, base + b])
        return (np.asarray(points, dtype=float),
                np.concatenate(lines) if lines else None)

    def add_lines(name, points, lines, width, color=PICK_BLUE):
        if lines is None:
            plotter.remove_actor(name)
            return
        # pickable=False on every mark: a mark stands in front of the
        # face it marks, and a pickable one swallowed the click meant
        # for the bar underneath — selecting a checkered bar did nothing
        plotter.add_mesh(pv.PolyData(points, lines=lines), color=color,
                         line_width=width, show_scalar_bar=False,
                         pickable=False, name=name)

    # Every bar outlined. Drawn as real lifted lines rather than with
    # `show_edges=True`, which loses the depth test against the very
    # faces it traces: measured here, widths 1 and 2 put nothing on
    # screen at all and 3 showed only the part spilling past the
    # silhouette, with VTK's polygon offset making no difference. The
    # lift these lines already carry is this file's own answer to that,
    # and it is exact rather than a depth-buffer negotiation.
    if max(rows, columns) <= OUTLINE_LIMIT:
        add_lines('mac-bars-edges',
                  *outline_lines([(r, c) for r in range(rows)
                                  for c in range(columns)],
                                 lift=EDGE_LIFT),
                  width=1, color=EDGE_INK)
    else:
        plotter.remove_actor('mac-bars-edges')

    inside = [pair for pair in selected
              if 0 <= pair[0] < rows and 0 <= pair[1] < columns]
    add_lines('mac-bars-selected', *outline_lines(inside), width=3)
    add_lines('mac-bars-active',
              *outline_lines([active] if active in inside else []), width=6)
    # committed matches wear the flat grid's red checker — on every
    # face, so the mark reads from whatever side the camera looks: an
    # outline alone vanished against the viridis field there, and a
    # top-only mark vanished behind a taller neighbor here
    checker_points, checker_faces = [], []

    def checker_face(origin, along, up):
        """The checker mapped onto one face, squares kept near-square.

        `CHECKER` squares across the `along` side; the `up` side takes
        however many keep the aspect, so a tall bar's sides read
        checkered rather than banded.
        """
        origin, along, up = (np.asarray(v, dtype=float)
                             for v in (origin, along, up))
        ku = CHECKER
        u_len = float(np.linalg.norm(along)) or 1.0
        kv = max(1, round(CHECKER * float(np.linalg.norm(up)) / u_len))
        for i in range(ku):
            for j in range(kv):
                if (i + j) % 2:
                    continue
                u0, u1 = i / ku, (i + 1) / ku
                v0, v1 = j / kv, (j + 1) / kv
                base = len(checker_points)
                checker_points.extend([
                    origin + u0 * along + v0 * up,
                    origin + u1 * along + v0 * up,
                    origin + u1 * along + v1 * up,
                    origin + u0 * along + v1 * up])
                checker_faces.append([4, base, base + 1, base + 2,
                                      base + 3])

    for r, c in matched:
        if not (0 <= r < rows and 0 <= c < columns):
            continue
        x0, x1, y0, y1, z1 = bar_frame(r, c)
        dx, dy = x1 - x0, y1 - y0
        checker_face((x0, y0, z1), (dx, 0, 0), (0, dy, 0))   # top
        checker_face((x0, y0, 0), (dx, 0, 0), (0, 0, z1))    # front
        checker_face((x0, y1, 0), (dx, 0, 0), (0, 0, z1))    # back
        checker_face((x0, y0, 0), (0, dy, 0), (0, 0, z1))    # left
        checker_face((x1, y0, 0), (0, dy, 0), (0, 0, z1))    # right
    if checker_faces:
        plotter.add_mesh(
            pv.PolyData(np.asarray(checker_points, dtype=float),
                        faces=np.concatenate(checker_faces)),
            color=MARK_RED, show_scalar_bar=False, pickable=False,
            name='mac-bars-matched')
    else:
        plotter.remove_actor('mac-bars-matched')

    # the one axis with a number worth reading: the value, 0..1. The
    # mode axes are named by the labels on their own edges
    plotter.show_bounds(
        axes_ranges=(0, columns, 0, rows, 0.0, 1.0),
        fmt='%.4g',
        xtitle='', ytitle='', ztitle='MAC',
        show_xlabels=False, show_ylabels=False, grid='back',
        location='outer', use_3d_text=False, color=colors['scene_text'])
    return {'rows': rows, 'columns': columns, 'named_rows': named_rows,
            'named_columns': named_columns}


def place_camera(plotter: Any, rows: int, columns: int) -> None:
    """The bars' home view: row 0 nearest, columns left to right.

    Placed when the comparison changes and at no other time, the same
    standing rule as every scene. Framed from the front-left and high
    enough that a full-height bar clears the far edge.
    """
    extent = max(rows, columns)
    plotter.camera_position = [
        (columns * 0.5 - 1.1 * extent, -1.4 * extent, 1.2 * extent),
        (columns * 0.5, rows * 0.42, 0.12 * extent),
        (0.0, 0.0, 1.0)]


def mac_bars_scene(frequencies: Sequence[float], matrix: Any,
                   column_frequencies: Sequence[float] | None = None,
                   plotter: Any = None, off_screen: bool = False,
                   theme: Any = None) -> Any:
    """Build (or add to) a PyVista plotter showing the MAC bars."""
    import pyvista as pv

    colors = resolve_theme(theme)
    if plotter is None:
        plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background(colors['scene_background'])
    info = add_mac_bars(plotter, frequencies, matrix,
                        column_frequencies=column_frequencies, theme=colors)
    place_camera(plotter, info['rows'], info['columns'])
    return plotter
