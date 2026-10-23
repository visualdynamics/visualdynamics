"""Building the PyVista scene for a geometry.

Coordinates are stored in SI and converted to the display unit system
when the scene is built, so changing the unit system rebuilds the scene
and leaves the data alone.

Line elements are drawn in direct colors, grouped by color index,
except when the scene is colored by value.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

# the solid face table lives in core, where finding a part's skin
# needs it too (Align Meshes, 2026-10-11)
from ..core.geometry import ELEMENT_TYPES, face_corners, solid_faces
from ..theme import theme as resolve_theme
from ..units import DEFAULT_SYSTEM

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.geometry import Geometry
    from ..units import UnitSystem

# An older note here warned that macOS VTK silently drops scalar-mapped
# line cells. Measured against pyvista 0.48.4 / VTK 9.6.2 it did not
# reproduce; the numbers were in docs/colormap.md, since removed (git
# history has it).

# UFF-style color indices -> RGB
PALETTE = [
    (0.55, 0.55, 0.55),  # 0 gray
    (0.12, 0.47, 0.71),  # 1 blue
    (1.00, 0.50, 0.05),  # 2 orange
    (0.17, 0.63, 0.17),  # 3 green
    (0.84, 0.15, 0.16),  # 4 red
    (0.58, 0.40, 0.74),  # 5 purple
    (0.55, 0.34, 0.29),  # 6 brown
    (0.89, 0.47, 0.76),  # 7 pink
    (0.74, 0.74, 0.13),  # 8 olive
    (0.09, 0.75, 0.81),  # 9 cyan
]

COLOR_NAMES = ['gray', 'blue', 'orange', 'green', 'red',
               'purple', 'brown', 'pink', 'olive', 'cyan']

# Coloring by displacement. Perceptually uniform and readable on both
# scene backgrounds; matplotlib-native, so it needs nothing pyvista does
# not already require.
COLORMAP = 'viridis'

#: the opacity a solid's skin is drawn at. It is see-through so a hole
#: or a pocket inside a part shows, and so does the part's far side:
#: drawn two-sided, the skin's far wall blends with its near one in
#: whatever order the cells came, which Brandon looked at against the
#: far wall culled and chose (2026-10-19; PLAN.md "The skin
#: see-through, the nodes that mean something"). Plates draw opaque: a
#: plate seen from behind is still the plate.
SKIN_OPACITY = 0.5


def color_name(index: int) -> str:
    """The name of a color index ('gray', 'blue', ...), wrapping past 9."""
    return COLOR_NAMES[int(index) % len(COLOR_NAMES)]


def color_rgb(index: int) -> tuple[float, float, float]:
    """The RGB triple (0 to 1) of a color index, wrapping past 9."""
    return PALETTE[int(index) % len(PALETTE)]


def axis_unit_text(geometry: Geometry, unit_system: UnitSystem) -> str:
    """The unit text the labeled axes show, such as ``'[mm]'``. A
    geometry whose units are undefined shows ``'[units undefined]'``
    instead of a unit."""
    if not geometry.units_defined:
        return '[units undefined]'
    return f"[{unit_system.label_text('length')}]"


def display_points(geometry: Geometry,
                   unit_system: UnitSystem) -> tuple[np.ndarray, str]:
    """The coordinates to draw for a geometry, and the axis unit text.

    Coordinates are converted from SI to `unit_system`. A geometry whose
    units are undefined is drawn with the file's raw coordinates, because
    converting them would scale values whose unit is unknown, and its
    axis text says the units are undefined.
    """
    if not geometry.units_defined:
        return geometry.node_xyz, axis_unit_text(geometry, unit_system)
    return (unit_system.from_si(geometry.node_xyz, 'length'),
            axis_unit_text(geometry, unit_system))


def _display_length(values: ArrayLike, geometry: Geometry,
                    unit_system: UnitSystem) -> np.ndarray:
    """Convert a stored length to display units, or pass it through when the
    geometry's units are undefined."""
    if not geometry.units_defined:
        return np.asarray(values)
    return unit_system.from_si(np.asarray(values), 'length')


def element_cells(geometry: Geometry,
                  indices: Sequence[int] | None = None,
                  separate_skins: bool = False) -> tuple:
    """How the elements are drawn: (faces, lines, points), each keyed by
    element color, with every entry given as node rows (positions in
    `node_id`), not node ids; with `separate_skins`, (faces, lines,
    points, skins), the solids' skins kept apart from the face elements
    so a scene can draw them see-through.

    A face element is drawn as its corner nodes (`face_corners`), since
    a quad8's midside nodes are not corners of the outline. A solid is
    drawn as the outer surface of its color: each cell's faces
    (`solid_faces`), less every face that two cells share. A line
    element is its first two nodes, and a point element is its nodes.
    The 3-D view, the report's scenes and the HTML export all use this
    one function.

    Parameters
    ----------
    geometry : Geometry
        The geometry.
    indices : sequence of int, optional
        Which elements, by row; every element when omitted.
    separate_skins : bool, default False
        Return the solids' skins as a fourth dict instead of among the
        faces.

    Returns
    -------
    tuple of dict
        Faces and lines as lists of row lists, points as a row list,
        each by element color; and the skins, when asked for.
    """
    # One rule for every scene because the HTML export once drew a brick
    # as one eight-sided polygon in node order, which folds it into a bow
    # tie (Brandon, 2026-10-11, the website's brick examples).
    id_to_row = {int(i): r for r, i in enumerate(geometry.node_id)}

    def rows(node_ids):
        return [id_to_row[int(i)] for i in node_ids]

    faces: dict[int, list[list[int]]] = {}
    lines: dict[int, list[list[int]]] = {}
    points: dict[int, list[int]] = {}
    skins: dict[int, dict[tuple[int, ...], list[int]]] = {}
    for i in (range(len(geometry.elem_conn)) if indices is None else indices):
        code, color = int(geometry.elem_type[i]), int(geometry.elem_color[i])
        conn = geometry.elem_conn[i]
        _name, _nnodes, render = ELEMENT_TYPES[code]
        if render == 'face':
            faces.setdefault(color, []).append(rows(conn[:face_corners(code)]))
        elif render == 'volume':
            # a solid draws as its skin: the faces of its cells that no
            # other cell of the color shares (2026-09-30; a polyline
            # through every corner drew the mesh's insides as a tangle)
            skin = skins.setdefault(color, {})
            for face in solid_faces(len(conn)):
                corners = rows([conn[k] for k in face])
                key = tuple(sorted(corners))
                if key in skin:
                    del skin[key]
                else:
                    skin[key] = corners
        elif render == 'line':
            lines.setdefault(color, []).append(rows(conn[:2]))
        elif render == 'point':
            points.setdefault(color, []).extend(rows(conn))
    if separate_skins:
        return faces, lines, points, {color: list(skin.values())
                                      for color, skin in skins.items()}
    for color, skin in skins.items():
        faces.setdefault(color, []).extend(skin.values())
    return faces, lines, points


def drawn_nodes(geometry: Geometry) -> np.ndarray:
    """A mask over the nodes of the ones a scene draws as points when
    nothing more specific is asked for: the nodes no element uses, which
    nothing else would show, and the nodes of line and point elements,
    where a beam's end or a mass's place is read. A plate's or a solid's
    nodes sit at the corners of the faces drawn and are not drawn again;
    a brick model's are mostly inside it, and drawn they were more than
    half of a frame (PLAN.md "The skin see-through, the nodes that mean
    something"). A pick still draws the nodes picked, and the Nodes row
    draws them all.
    """
    count = geometry.num_nodes
    if not len(geometry.elem_conn):
        return np.ones(count, dtype=bool)
    conn = [np.asarray(c, dtype=np.int64) for c in geometry.elem_conn]
    ids = np.concatenate(conn)
    on_lines = np.concatenate([
        np.full(len(c), ELEMENT_TYPES[int(code)][2] in ('line', 'point'))
        for c, code in zip(conn, geometry.elem_type)])
    order = np.argsort(geometry.node_id)
    rows = order[np.searchsorted(geometry.node_id[order], ids)]
    used = np.zeros(count, dtype=bool)
    used[rows] = True
    shown = np.zeros(count, dtype=bool)
    shown[rows[on_lines]] = True
    return shown | ~used


def _cells(index_lists: Sequence[Sequence[int]]) -> np.ndarray:
    """Flat VTK cell array: [n, i0..in-1, n, i0..in-1, ...]"""
    return np.concatenate([[len(ix), *ix] for ix in index_lists])


def geometry_scene(geometry: Geometry, unit_system: UnitSystem | None = None,
                   plotter: Any = None, node_size: float = 8.0,
                   line_width: float = 2.0, show_edges: bool = True,
                   opacity: float = 1.0,
                   labels: Sequence[str] | None = None,
                   off_screen: bool = False, theme: Any = None,
                   components: Sequence[str] | None = None,
                   scale: float = 1.0, bounds: bool = True,
                   orientation: bool = True) -> Any:
    """Build a PyVista plotter showing the geometry, or add it to `plotter`.

    `theme` is 'light', 'dark' or a colors dict, and sets the scene
    background and the annotation color. `components` limits what is
    drawn to any of 'nodes', 'elements' and 'coordinate_systems' (the
    default is nodes and elements); selecting one of them in the project
    tree shows only that part. `labels` names the kinds to caption with
    their ids (`add_geometry`). `node_size` and `line_width` are in
    pixels, and `scale` multiplies every size given in pixels (nodes,
    lines, the bounds' labels) for a render at print resolution
    (`print_plotter`). `bounds` and `orientation` turn on the labeled
    box and the corner triad (`annotate_scene`); a print figure may want
    neither. A new plotter is made off screen when `off_screen` is true.

    Returns the plotter. Call ``.show()`` on it, or ``.screenshot()``
    when it is off screen.
    """
    import pyvista as pv

    us = unit_system or DEFAULT_SYSTEM
    colors = resolve_theme(theme)
    if plotter is None:
        plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background(colors['scene_background'])
    axis_unit = add_geometry(plotter, geometry, unit_system=us,
                             node_size=node_size * scale,
                             line_width=line_width * scale,
                             show_edges=show_edges, opacity=opacity,
                             labels=labels, components=components,
                             text_color=colors['scene_text'])
    annotate_scene(plotter, axis_unit, colors, bounds=bounds,
                   orientation=orientation, scale=scale)
    return plotter


#: VTK's own dots per inch: a text actor's size in points is its size in
#: pixels at 72
VTK_DPI = 72.0
#: the logical dots per inch a screen image is laid out at, which a print
#: render is scaled from — the 2-D export's reference too
SCREEN_DPI = 96.0
#: the bounds box's label size in a print render, per unit of scale, as
#: the cube axes' screen size. Calibrated so the box prints its labels at
#: the size the scene's other labels print: a title stands about 2.1
#: pixels per unit of screen size, and a 13-point label at a render
#: window DPI of 72 × scale about 9.6 × scale pixels (measured,
#: 2026-09-27). The screen's own 10 is twice that, and multiplied out
#: it printed the box's titles at 14 to 16 pt and stacked the ticks on a
#: thin axis (the band-average paper).
PRINT_BOUNDS_SIZE = 4.6


def print_plotter(size_in: tuple[float, float], dpi: float) -> tuple[Any, float]:
    """An off-screen plotter for a figure `size_in` inches at `dpi`, and
    the scale to draw every pixel size in it at.

    The window is the printed size in pixels, and the render window's
    DPI is set so that a text actor's size in points prints in points.
    The returned scale is how many print pixels one screen pixel
    becomes (`dpi` / 96). Multiply the sizes VTK takes in pixels (nodes,
    lines, the bounds' labels) by it, and the figure prints the way the
    screen shows it.

    Parameters
    ----------
    size_in : (float, float)
        Width and height of the printed figure, in inches.
    dpi : float
        Its resolution.

    Returns
    -------
    (pyvista.Plotter, float)
        The plotter and the pixel scale.
    """
    # A plain 3-D screenshot at print resolution came out with its labels
    # at about 2 pt, and screenshot(scale=) dropped the point labels and
    # enlarged the axis triad (the band-average paper, 2026-09-26).
    import pyvista as pv

    scale = float(dpi) / SCREEN_DPI
    plotter = pv.Plotter(off_screen=True, window_size=(
        max(1, round(size_in[0] * dpi)), max(1, round(size_in[1] * dpi))))
    plotter.ren_win.SetDPI(round(VTK_DPI * scale))
    return plotter, scale


def annotate_scene(plotter: Any, axis_unit: str, colors: Mapping[str, str],
                   bounds: bool = True,
                   orientation: bool = True, scale: float = 1.0) -> None:
    """Add or remove the scene's two annotations.

    `bounds` is the labeled box around the geometry, its titles carrying
    `axis_unit`, and `orientation` is the small triad in the corner; the
    3-D view's toolbar turns each on and off separately. Turned off, an
    annotation already in the scene is removed. A `scale` other than one
    means a print render (`print_plotter`): the box's labels, which are
    sized in screen pixels, are set to print at the size of the scene's
    other labels (`PRINT_BOUNDS_SIZE`).
    """
    if bounds:
        box = plotter.show_bounds(xtitle=f'X {axis_unit}',
                                  ytitle=f'Y {axis_unit}',
                                  ztitle=f'Z {axis_unit}', grid='back',
                                  location='outer', color=colors['scene_text'],
                                  **_label_counts(plotter.bounds))
        if scale != 1.0 and hasattr(box, 'SetScreenSize'):
            # the box's labels are sized in screen pixels by the cube
            # axes' own screen size — not by the font size, not by the
            # render window's DPI (both measured, 2026-09-26) — so a
            # print render sets that, to print as the other labels do
            box.SetScreenSize(PRINT_BOUNDS_SIZE * scale)
    else:
        plotter.remove_bounds_axes()
    if orientation:
        plotter.add_axes(color=colors['scene_text'])
    else:
        plotter.hide_axes()


#: an axis shorter than this share of the model's longest carries its
#: title and no numbers
THIN_AXIS = 0.1


def _label_counts(bounds: Sequence[float]) -> dict[str, bool]:
    """Which axes of the bounds box carry numbers: all of them, except
    one much shorter than the model — a plate's thickness — whose five
    numbers stacked into a smudge, on screen and in print alike, and
    whose two end numbers still overlapped (the band-average paper,
    2026-09-27). It keeps its title."""
    extents = [abs(bounds[2 * k + 1] - bounds[2 * k]) for k in range(3)]
    longest = max(extents) or 1.0
    return {f'show_{axis}labels': bool(extent >= THIN_AXIS * longest)
            for axis, extent in zip('xyz', extents)}


def shared_points(points: ArrayLike) -> tuple[Any, np.ndarray]:
    """A vtkPoints that every mesh can share, and a writable numpy view of it.

    The points are a copy of `points`. Writing through the view and
    calling ``Modified()`` on the vtkPoints moves every mesh at once, so
    a frame is one upload rather than one per mesh.
    """
    import pyvista as pv
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy

    vtk_points = vtk.vtkPoints()
    # copy: convert_array wraps the caller's buffer, and the animator needs
    # VTK's memory to be distinct from the base positions it writes from
    vtk_points.SetData(pv.convert_array(np.array(points, dtype=np.float64)))
    return vtk_points, vtk_to_numpy(vtk_points.GetData())


# What each coordinate system's three directions are called, and which of
# them are angles. A radius is a straight direction; an angle is drawn as an
# arc about the axis it turns around.
# Written as math so the Greek letters appear at all: VTK's own font has
# no glyph for them and draws nothing, while matplotlib's mathtext — which
# pyvista requires anyway — renders them properly.
CS_AXIS_LABELS = {
    0: ('$X$', '$Y$', '$Z$'),
    1: ('$R$', r'$\theta$', '$Z$'),
    2: ('$R$', r'$\theta$', r'$\phi$'),
}
# An angular direction is drawn as an arc: which axis it starts along, and
# which it turns about. Theta on a sphere starts at Z and swings towards X,
# the way the polar angle is measured; the azimuths start at X and swing
# towards Y.
CS_ARCS = {
    1: {1: (0, 2)},                    # cylindrical: theta, X about Z
    2: {1: (2, 1), 2: (0, 2)},         # spherical: theta Z about Y, phi X about Z
}
AXIS_COLORS = ('#e5534b', '#3fb950', '#4c92d9')
# 60 degrees, drawn inside the straight arrows: a wider sweep puts the
# spherical theta arc's head on top of R's, and the two labels collide
ARC_SWEEP = np.pi / 2.6
ARC_RADIUS = 0.85
# pyvista's arrow is a shaft of 0.05 and a head 0.25 long by 0.1 wide, all
# as fractions of its length. An arc is drawn to the same proportions, so a
# curved direction has the same weight on screen as a straight one.
ARROW_SHAFT_RADIUS = 0.05
ARROW_HEAD_LENGTH = 0.25
ARROW_HEAD_RADIUS = 0.1


def _arc_points(origin, start, about, radius, steps=28):
    """Points along an arc that starts along `start` and turns about `about`.

    Both are unit vectors of the coordinate system's own frame, so the arc
    lies in that system's plane rather than the global one.
    """
    sideways = np.cross(about, start)
    angle = np.linspace(0.0, ARC_SWEEP, steps)[:, np.newaxis]
    return origin + radius * (np.cos(angle) * start + np.sin(angle) * sideways)


# DOF arrow direction per direction string; a rotational DOF points
# along its axis and keeps its R label to say what it is
DOF_DIRECTIONS = {
    'X+': (1, 0, 0), 'X-': (-1, 0, 0), 'Y+': (0, 1, 0), 'Y-': (0, -1, 0),
    'Z+': (0, 0, 1), 'Z-': (0, 0, -1),
    'RX+': (1, 0, 0), 'RX-': (-1, 0, 0), 'RY+': (0, 1, 0),
    'RY-': (0, -1, 0), 'RZ+': (0, 0, 1), 'RZ-': (0, 0, -1),
}
DOF_LABEL_LIMIT = 40
#: a geometry shown on its own, with nothing measured to ask about, draws
#: its arrows at every node's three translations — up to this many
#: nodes. Past it the arrows are a solid hedge that says nothing, and
#: the app says why they are missing instead (Brandon, 2026-09-27: a
#: few thousand nodes of them hung the window)
DOF_ARROW_NODE_LIMIT = 300


def dof_axis(direction: str) -> int | None:
    """0, 1 or 2 for the axis of a DOF direction, or None if it has none.

    A rotation counts as its axis: 'RZ-' is 2, and draws blue like 'Z+'.
    """
    letter = str(direction).upper().lstrip('R')[:1]
    return {'X': 0, 'Y': 1, 'Z': 2}.get(letter)


def dof_arrow_vectors(geometry: Geometry, dofs: Sequence[str]
                      ) -> list[tuple[int, np.ndarray, int] | None]:
    """For each DOF, (node row, global direction, axis) for its arrow, or
    None for a DOF at a node the geometry lacks or with no axis.

    The direction is turned by the node's displacement coordinate
    system the same way the animation turns the motion
    (`deform.dof_directions`), so '101X+' on a node measured in a turned
    system points along that system's X. A rotational DOF points along
    its axis, turned the same way. The axis, which picks the color, is
    the DOF's own letter, not the nearest global axis.
    """
    # The arrows were drawn along global X, Y and Z whatever the node's
    # displacement system said, so a turned accelerometer was marked
    # pointing the wrong way while the animation moved it the right one
    # (Brandon, 2026-10-16).
    from ..core.data import parse_dof
    from ..deform import _cs_rotations

    rows = {int(node): i for i, node in enumerate(geometry.node_id)}
    rotations = _cs_rotations(geometry)
    out: list[tuple[int, np.ndarray, int] | None] = []
    for dof in dofs:
        node, direction = parse_dof(dof)
        vector = DOF_DIRECTIONS.get(str(direction).upper())
        axis = dof_axis(direction)
        if node is None or vector is None or axis is None \
                or int(node) not in rows:
            out.append(None)
            continue
        row = rows[int(node)]
        vector = np.asarray(vector, dtype=float)
        rotation = rotations.get(int(geometry.node_disp_cs[row]))
        if rotation is not None:
            vector = vector @ rotation
        out.append((row, vector, axis))
    return out


def dof_arrow_length(points: ArrayLike, extent: float) -> float:
    """One length for every arrow on a plot: 12 % of `extent`, reduced to
    0.9 times the closest spacing of the arrowed `points` so that arrows
    on closely spaced nodes do not overlap."""
    length = 0.12 * extent
    unique = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(unique) > 1:
        from scipy.spatial import cKDTree

        # each node's nearest neighbor, by tree: the all-pairs table this
        # replaced was 600 MB at five thousand nodes (2026-09-27)
        distances, _index = cKDTree(unique).query(unique, k=2)
        nearest = float(distances[:, 1].min())
        if np.isfinite(nearest) and nearest > 0:
            length = min(length, 0.9 * nearest)
    return length


def add_dof_arrows(plotter: Any, geometry: Geometry, dofs: Sequence[str],
                   unit_system: UnitSystem | None = None,
                   incoming: bool | Sequence[bool] = False,
                   name: str | None = None) -> int:
    """Labeled arrows marking DOFs on the geometry.

    One arrow per DOF, colored by its axis: X red, Y green, Z blue, the
    same colors as the orientation triad. It points along that axis of
    the node's displacement coordinate system (`dof_arrow_vectors`). The
    arrows of one color are built into one mesh, so ten thousand arrows
    are three actors. A response arrow starts at the node and points
    outward, with its label at the tip; with `incoming` (forces) the
    arrow ends on the node instead, with its label at the base;
    `incoming` is one flag for every arrow or one per DOF. All
    arrows have one length (`dof_arrow_length`). Labels are drawn only
    when there are at most `DOF_LABEL_LIMIT` arrows. A DOF at a node the
    geometry does not have, or with a direction that is not a
    translation or rotation, is skipped. Given a `name`, the actors are
    named from it, so drawing again replaces them.

    Returns how many arrows were drawn.
    """
    import pyvista as pv

    points, _unit = display_points(geometry, unit_system)
    points = np.asarray(points, dtype=float)
    if not len(points):
        return 0
    center = points.mean(axis=0)
    extent = float(np.sqrt(((points - center) ** 2)
                           .sum(axis=1).max())) or 1.0
    inward = ([bool(incoming)] * len(dofs) if isinstance(incoming, bool)
              else [bool(flag) for flag in incoming])
    entries = [(points[found[0]], found[1], found[2], dof, into)
               for dof, into, found in zip(dofs, inward,
                                           dof_arrow_vectors(geometry, dofs))
               if found is not None]
    if not entries:
        return 0
    length = dof_arrow_length([entry[0] for entry in entries], extent)
    label_spots = {0: [], 1: [], 2: []}
    starts = {0: [], 1: [], 2: []}
    vectors = {0: [], 1: [], 2: []}
    for position, vector, axis, dof, into in entries:
        start = position - vector * length if into else position
        starts[axis].append(start)
        vectors[axis].append(vector)
        spot = (start - vector * length * 0.25 if into
                else position + vector * length * 1.25)
        label_spots[axis].append((spot, dof))
    # One mesh per axis color, every arrow of that color glyphed into
    # it, added without a render. An actor per arrow — each add a
    # synchronous render of the whole scene in the app — made a bare
    # geometry's arrows at every node quadratic, and a model of a few
    # thousand nodes hung the window for minutes (Brandon, 2026-09-27).
    for axis in (0, 1, 2):
        if not starts[axis]:
            continue
        base = pv.PolyData(np.asarray(starts[axis], dtype=float))
        base['direction'] = np.asarray(vectors[axis], dtype=float)
        plotter.add_mesh(
            base.glyph(orient='direction', scale=False, factor=length,
                       geom=pv.Arrow()),
            color=AXIS_COLORS[axis], render=False,
            name=None if name is None else f'{name}-arrows{axis}')
    # labels only while they can be read: past a few dozen arrows the
    # names overprint into noise, and the arrows alone say where
    if len(entries) <= DOF_LABEL_LIMIT:
        for axis, spots in label_spots.items():
            if not spots:
                continue
            plotter.add_point_labels(
                np.asarray([spot for spot, _dof in spots]),
                [dof for _spot, dof in spots],
                font_size=12, always_visible=True,
                text_color=AXIS_COLORS[axis], shape=None,
                fill_shape=False, show_points=False, render=False,
                name=None if name is None else f'{name}-labels{axis}')
    return len(entries)


def add_coordinate_system(plotter: Any, origin: ArrayLike, matrix: ArrayLike,
                          cs_type: int, length: float,
                          text_color: str = '#000000',
                          label: str | None = None,
                          name: str | None = None) -> None:
    """Draw one coordinate system: three arrows, with angular directions
    drawn as arcs.

    `matrix` holds the three direction vectors in its first three rows,
    and `length` is the arrow length. The arrows show the type even
    without names: a cylindrical system has one arc and a spherical one
    has two. When `label` (the system's id) is given, it is written at
    the origin and each direction's name, from the type (X/Y/Z, R/theta/Z
    or R/theta/phi), is written at the tip of its arrow in `text_color`.
    Without a `label`, only the arrows are drawn.

    Given a `name`, every actor is named from it, so drawing again
    replaces the drawing instead of adding another on top. That is what
    lets a frame be rotated live without rebuilding the rest of the
    scene.
    """
    import pyvista as pv

    names = CS_AXIS_LABELS.get(int(cs_type), CS_AXIS_LABELS[0])
    arcs = CS_ARCS.get(int(cs_type), {})
    tips, tip_names = [], []

    def actor_name(part: str) -> str | None:
        return None if name is None else f'{name}-{part}'

    for axis, (direction, axis_label, color) in enumerate(
            zip(matrix[:3], names, AXIS_COLORS)):
        if axis in arcs:
            start, about = (matrix[i] for i in arcs[axis])
            points = _arc_points(origin, start, about, length * ARC_RADIUS)
            # the head takes the last stretch of the arc, so a curved
            # direction reaches as far as a straight one rather than past it
            head = ARROW_HEAD_LENGTH * length
            walked = np.cumsum(
                np.linalg.norm(np.diff(points[::-1], axis=0), axis=1))
            base = len(points) - 1 - int(np.searchsorted(walked, head)) - 1
            base = min(max(base, 0), len(points) - 2)
            plotter.add_mesh(
                pv.lines_from_points(points[:base + 1]).tube(
                    radius=ARROW_SHAFT_RADIUS * length, n_sides=16),
                color=color, name=actor_name(f'arc{axis}'))
            heading = points[-1] - points[base]
            plotter.add_mesh(
                pv.Cone(center=(points[base] + points[-1]) / 2,
                        direction=heading,
                        height=float(np.linalg.norm(heading)),
                        radius=ARROW_HEAD_RADIUS * length, resolution=20),
                color=color, name=actor_name(f'head{axis}'))
            heading = heading / np.linalg.norm(heading)
            tips.append(points[-1] + heading * length * 0.45)
        else:
            arrow = pv.Arrow(start=origin, direction=direction,
                             scale=length)
            plotter.add_mesh(arrow, color=color,
                             name=actor_name(f'axis{axis}'))
            tips.append(origin + direction * length * 1.2)
        tip_names.append(axis_label)

    if label is None:
        # not singled out: the shape of the arrows still says which type it
        # is, without writing over the model
        return
    plotter.add_point_labels(
        np.asarray(tips), tip_names, font_size=13, always_visible=True,
        text_color=text_color, shape=None, fill_shape=False,
        show_points=False, name=actor_name('names'))
    plotter.add_point_labels(
        origin[np.newaxis], [str(label)], font_size=14, always_visible=True,
        text_color=text_color, shape=None, fill_shape=False,
        show_points=False, name=actor_name('id'))


def shared_scalars(count: int, name: str) -> tuple[np.ndarray, Any]:
    """A numpy view of a new VTK point-data array, and the array itself.

    The array holds `count` zeros and is named `name`. It does for color
    what `shared_points` does for position: every mesh in the scene uses
    the one array, so writing it once per frame recolors all of them.
    """
    import pyvista as pv
    from vtkmodules.util.numpy_support import vtk_to_numpy

    array = pv.convert_array(np.zeros(count, dtype=np.float64))
    array.SetName(name)
    return vtk_to_numpy(array), array


#: how many captions one kind may put on a scene at once.
#:
#: Measured, because the obvious reason for a limit turned out not to be
#: one. Captions are *not* an actor apiece: VTK places them through one
#: mapper that decimates whatever overlaps, so a redraw costs about a
#: millisecond whether the scene carries a hundred labels or fifty
#: thousand, and a crowded run thins itself into every third id rather
#: than smearing. Neither the drawing nor the reading is what bites.
#:
#: What bites is `label_spots` itself, and only for the kinds whose
#: caption sits at a centroid — that is a Python loop over ragged
#: connectivity. On a square plate: 1 k elements 5 ms, 10 k 48 ms, 40 k
#: 217 ms, 200 k 1.0 s. Nodes are vectorized and cost 37 ms at 200 k.
#:
#: 5 000 puts the worst of those at about 25 ms, which no one feels on a
#: selection change, and is far above anything worth reading — the
#: demonstration drone is 1 422 nodes and 1 564 elements. `labels_fit`
#: is how a caller finds out before asking; the window says so in the
#: status bar rather than quietly drawing none.
ENTITY_LABEL_LIMIT = 5000


def label_choices(labels: Sequence[str] | Mapping[str, Sequence[int] | None]
                  | None) -> dict[str, Sequence[int] | None]:
    """`labels` in one form: {kind: the ids to caption, or None}.

    Two forms are accepted. A script passes a list of kinds, such as
    ``labels=['nodes']``, meaning the nodes that are drawn. The window,
    which knows what was picked, passes a dict, such as
    ``labels={'groups': [3]}``. None as a value means every entity of
    that kind that is drawn. `labels` of None gives an empty dict.
    """
    if labels is None:
        return {}
    if isinstance(labels, Mapping):
        return dict(labels)
    return {str(kind): None for kind in labels}


def labels_fit(geometry: Geometry, kind: str,
               chosen: Sequence[int] | None = None) -> bool:
    """Whether captioning this kind stays within `ENTITY_LABEL_LIMIT`.

    Counts `chosen` when it is given, else every entity of the kind. The
    window asks before it turns labels on, so a selection too large to
    caption is reported in the status bar instead of silently drawing no
    labels.
    """
    if chosen is not None:
        return len(chosen) <= ENTITY_LABEL_LIMIT
    counts = {'nodes': geometry.num_nodes,
              'coordinate_systems': len(geometry.cs_id),
              'elements': len(geometry.elem_conn),
              'groups': len(geometry.group_id)}
    return counts.get(kind, 0) <= ENTITY_LABEL_LIMIT


def label_spots(geometry: Geometry, points: np.ndarray, kind: str,
                chosen: Sequence[int] | None = None
                ) -> tuple[np.ndarray, list[str]]:
    """(positions, texts) captioning one kind of entity with its id.

    `kind` is 'nodes', 'elements' or 'groups'; any other kind gets no
    captions (a coordinate system's triad writes its own id). `points`
    is the geometry's nodes as drawn, so a caption is in the same units
    and frame as what it names. A node is captioned at the node. An
    element is captioned at the mean of its nodes, and an element group
    at the mean of its elements' nodes, so the number stays off the
    vertices, which carry the node ids when both are shown. A group is
    captioned with its name when it has one, else its id.

    `chosen` is the ids (nodes, element groups) or the indices
    (elements) to caption; None captions every one of the kind. Nothing
    is returned past `ENTITY_LABEL_LIMIT` (see `labels_fit`).
    """
    if not labels_fit(geometry, kind, chosen):
        return np.empty((0, 3)), []
    row_of = {int(node): row for row, node in enumerate(geometry.node_id)}

    def center(node_ids: Sequence[int]) -> np.ndarray | None:
        rows = [row_of[int(n)] for n in node_ids if int(n) in row_of]
        return points[rows].mean(axis=0) if rows else None

    spots: list[np.ndarray] = []
    texts: list[str] = []

    if kind == 'nodes':
        mask = (np.isin(geometry.node_id, list(chosen)) if chosen is not None
                else np.ones(geometry.num_nodes, bool))
        return points[mask], [str(int(i)) for i in geometry.node_id[mask]]

    if kind == 'elements':
        wanted = (list(chosen) if chosen is not None
                  else range(len(geometry.elem_conn)))
        for i in wanted:
            spot = center(geometry.elem_conn[int(i)])
            if spot is not None:
                spots.append(spot)
                texts.append(str(int(geometry.elem_id[int(i)])))

    elif kind == 'groups':
        # an element group holds no coordinates of its own: it is a label on
        # elements, so it is captioned in the middle of the elements
        # that carry its id
        wanted = ({int(b) for b in chosen} if chosen is not None
                  else {int(b) for b in geometry.group_id})
        groups = np.asarray(geometry.elem_group, dtype=np.int64)
        for row, group in enumerate(geometry.group_id):
            if int(group) not in wanted:
                continue
            members = np.flatnonzero(groups == int(group))
            nodes = [n for i in members for n in geometry.elem_conn[int(i)]]
            spot = center(nodes)
            if spot is not None:
                spots.append(spot)
                name = str(geometry.group_name[row]).strip()
                texts.append(name or str(int(group)))

    return (np.asarray(spots) if spots else np.empty((0, 3))), texts


def add_geometry(plotter: Any, geometry: Geometry,
                 unit_system: UnitSystem | None = None,
                 node_size: float = 8.0, line_width: float = 2.0,
                 show_edges: bool = True, opacity: float = 1.0,
                 labels: Sequence[str] |
                 Mapping[str, Sequence[int] | None] | None = None,
                 components: Sequence[str] | None = None,
                 color_override: Any = None, text_color: str = '#000000',
                 entities: Mapping[str, Sequence[int]] | None = None,
                 points_source: Any = None,
                 meshes: list[Any] | None = None, scalars: Any = None,
                 clim: tuple[float, float] | None = None) -> str:
    """Add one geometry's meshes to an existing plotter.

    `color_override` draws the whole geometry in one color, which is how
    several geometries overlaid in one scene are told apart. `entities`
    limits the drawing to specific items, as a dict with any of 'nodes'
    (node ids), 'coordinate_systems' (ids) and 'elements' (indices); that
    is how a single node or element picked in the tree is highlighted.
    `components` limits it to whole kinds instead (`geometry_scene`).

    `labels` names the kinds to caption with their ids, any of 'nodes',
    'coordinate_systems', 'elements' and 'groups' (`label_choices`).
    Captions follow `entities` when it limits the drawing, so labeling a
    picked element names that element only, and a kind with more than
    `ENTITY_LABEL_LIMIT` entities is not captioned (`labels_fit`).
    Captions are drawn in `text_color`.

    `points_source` is a shared vtkPoints (`shared_points`) for the
    meshes to use instead of a new one, and each mesh made is appended to
    `meshes` when a list is given; the animator uses both to move the
    scene. `scalars` is a VTK point-data array (`shared_scalars`) shared
    by every mesh, coloring the geometry by value instead of by its color
    indices, and `clim` fixes the values at the two ends of the color
    map. `node_size` and `line_width` are in pixels.

    Returns the axis unit text.
    """
    import pyvista as pv

    us = unit_system or DEFAULT_SYSTEM
    points, axis_unit = display_points(geometry, us)
    if points_source is None:
        points_source, _ = shared_points(points)

    def new_mesh() -> Any:
        """An empty mesh sharing the scene's points, and their colors."""
        mesh = pv.PolyData()
        mesh.SetPoints(points_source)
        if scalars is not None:
            mesh.GetPointData().SetScalars(scalars)
        if meshes is not None:
            meshes.append(mesh)
        return mesh

    def paint(index: int) -> Any:
        return color_override if color_override else color_rgb(index)

    def painted(index: int) -> dict[str, Any]:
        """How to color one mesh: by value, or by its color index."""
        if scalars is None:
            return {'color': paint(index)}
        return {'scalars': scalars.GetName(), 'cmap': COLORMAP,
                'clim': clim, 'show_scalar_bar': False}

    picked = entities or {}
    # a caption follows what is drawn: told nothing more specific, a
    # kind is captioned over exactly the entities the pick restricted
    # it to, so labeling a picked element names that one alone
    wanted_labels = {kind: (picked.get(kind) if chosen is None else chosen)
                     for kind, chosen in label_choices(labels).items()}
    if components:
        draw = set(components)
    elif picked:
        draw = {kind for kind, values in picked.items() if values}
    else:
        draw = {'nodes', 'elements'}
    id_to_row = {int(i): r for r, i in enumerate(geometry.node_id)}

    def rows(node_ids: Sequence[int]) -> list[int]:
        return [id_to_row[int(i)] for i in node_ids]

    # Nodes, grouped by color: the ones picked, every one when the
    # Nodes row is what is shown, and otherwise the ones that mean
    # something on their own (`drawn_nodes`)
    wanted_nodes = picked.get('nodes')
    if wanted_nodes:
        node_mask = np.isin(geometry.node_id, list(wanted_nodes))
    elif components and 'nodes' in components:
        node_mask = np.ones(geometry.num_nodes, bool)
    else:
        node_mask = drawn_nodes(geometry)
        if 'nodes' in wanted_labels and wanted_labels['nodes'] is None:
            # a caption names a node that is drawn
            wanted_labels['nodes'] = [int(i) for i in geometry.node_id[node_mask]]
    for color in (np.unique(geometry.node_color[node_mask])
                  if 'nodes' in draw and node_mask.any() else []):
        mask = (geometry.node_color == color) & node_mask
        indices = np.flatnonzero(mask)
        mesh = new_mesh()
        mesh.verts = np.column_stack(
            [np.ones(len(indices), dtype=np.int64), indices]).ravel()
        plotter.add_mesh(mesh, **painted(color), point_size=node_size,
                         render_points_as_spheres=True, opacity=opacity)

    # Elements, split by render class and grouped by color (a drawn
    # line is an element group of two-node line elements since 2026-09-30, and
    # draws as its line elements do: direct color per group)
    wanted_elements = picked.get('elements')
    faces, lines, cell_points, skins = (
        element_cells(geometry, None if wanted_elements is None
                      else list(wanted_elements), separate_skins=True)
        if 'elements' in draw else ({}, {}, {}, {}))
    for color, polys in faces.items():
        mesh = new_mesh()
        mesh.faces = _cells(polys)
        plotter.add_mesh(mesh, **painted(color), opacity=opacity,
                         specular=0.3, specular_power=25,
                         show_edges=show_edges)
    # a solid's skin see-through, both sides (`SKIN_OPACITY`); a
    # caller drawing fainter still, the muted context or a merge, is
    # taken at its word
    for color, polys in skins.items():
        mesh = new_mesh()
        mesh.faces = _cells(polys)
        plotter.add_mesh(mesh, **painted(color),
                         opacity=min(opacity, SKIN_OPACITY),
                         specular=0.3, specular_power=25,
                         show_edges=show_edges)
    for color, polylines in lines.items():
        mesh = new_mesh()
        mesh.lines = _cells(polylines)
        plotter.add_mesh(mesh, **painted(color), line_width=line_width,
                         opacity=opacity)
    for color, rows_ in cell_points.items():
        mesh = new_mesh()
        indices = np.asarray(rows_, dtype=np.int64)
        mesh.verts = np.column_stack(
            [np.ones(len(indices), dtype=np.int64), indices]).ravel()
        plotter.add_mesh(mesh, **painted(color), point_size=node_size * 1.5,
                         render_points_as_spheres=True, opacity=opacity)

    if 'coordinate_systems' in draw:
        wanted_cs = picked.get('coordinate_systems')
        span = np.ptp(points, axis=0).max() if len(points) > 1 else 1.0
        length = 0.12 * (span or 1.0)
        for i, cs_id in enumerate(geometry.cs_id):
            if wanted_cs and int(cs_id) not in {int(c) for c in wanted_cs}:
                continue
            matrix = geometry.cs_matrix[i]
            origin = _display_length(matrix[3], geometry, us)
            # axis colors survive a color override — direction identity is
            # the whole point of drawing a triad. Names and the id appear
            # for a system singled out, where they are worth the clutter.
            named = 'coordinate_systems' in wanted_labels and labels_fit(
                geometry, 'coordinate_systems',
                wanted_labels['coordinate_systems'])
            add_coordinate_system(
                plotter, origin, matrix, geometry.cs_type[i], length,
                text_color=text_color,
                label=int(cs_id) if wanted_cs or named else None)

    for kind, chosen in wanted_labels.items():
        if kind == 'coordinate_systems':
            continue                  # the triad writes its own id, above
        spots, texts = label_spots(geometry, points, kind, chosen)
        if not len(spots):
            continue
        plotter.add_point_labels(spots, texts, font_size=14,
                                 always_visible=True, text_color=text_color,
                                 shape=None, fill_shape=False,
                                 show_points=False)
    return axis_unit


def place_view(plotter: Any, view: Any = None, render: bool = True) -> None:
    """Turn the camera to `view` (a geometry's `View`, or the default
    isometric view when None) and fit everything the plotter holds.

    Every 3-D view of a geometry is opened through this function: the
    app when it first shows one and on Reset View, the script windows,
    the animations and every exported figure, so a geometry opens the
    same way wherever it is drawn. The report's scenes open on the same
    view through `View.basis`. `render` renders after the camera moves.
    """
    from ..core.geometry import DEFAULT_VIEW

    view = view or DEFAULT_VIEW
    plotter.view_vector(view.eye, viewup=view.up, render=False)
    plotter.reset_camera(render=render)


def plot_geometry(geometry: Geometry, unit_system: UnitSystem | None = None,
                  screenshot: str | None = None, theme: Any = None,
                  show: bool = True, size_in: tuple[float, float] | None = None,
                  dpi: float | None = None, **kwargs: Any) -> Any:
    """Show the geometry interactively, or render it to the file
    `screenshot` with no window.

    Shown, it opens in the app's own 3-D pane, with the labeled axes and
    the orientation triad as toggles on the bar above it, as in the
    window. Returns the pane (its `.plotter` is the PyVista one), or the
    image array when rendering to a file. For print, `dpi` renders the
    file at its printed size (`print_plotter`), `size_in` inches (6 by
    4.5 by default), with every label and line at the size the screen
    shows it; `size_in` is used only with `dpi`. The camera opens on the
    geometry's own view. Other keyword arguments go to
    `geometry_scene`.
    """
    if screenshot is not None:
        if dpi is not None:
            plotter, scale = print_plotter(size_in or (6.0, 4.5), dpi)
            geometry_scene(geometry, unit_system=unit_system, theme=theme,
                           plotter=plotter, scale=scale, **kwargs)
        else:
            plotter = geometry_scene(geometry, unit_system=unit_system,
                                     theme=theme, off_screen=True, **kwargs)
        place_view(plotter, geometry.opening_view, render=False)
        img = plotter.screenshot(screenshot)
        plotter.close()
        return img
    from ..gui.windows import scene_window

    us = unit_system or DEFAULT_SYSTEM
    return scene_window(
        lambda plotter: geometry_scene(geometry, unit_system=us,
                                       plotter=plotter, theme=theme,
                                       **kwargs),
        theme=theme, axis_unit=axis_unit_text(geometry, us),
        title='Geometry', show=show, view=geometry.opening_view)


def plot_dofs(geometry: Geometry, source: Any, quantity: str,
              unit_system: UnitSystem | None = None,
              screenshot: str | None = None, theme: Any = None,
              show: bool = True, size_in: tuple[float, float] | None = None,
              dpi: float | None = None, **kwargs: Any) -> Any:
    """The geometry with labeled arrows at every DOF that `source`
    measures as `quantity`: the app's DOF-arrows toggle, from a script.

    Force arrows end on their node with the label at the base, and
    response arrows start at it with the label at the tip, as the app
    draws them (`add_dof_arrows`). `source` is a data object, or a list
    or tuple of them. Shown or rendered to `screenshot` as
    `plot_geometry` does, including `size_in` and `dpi` for print; other
    keyword arguments go to `geometry_scene`.
    """
    from ..core.report import EXCITATION_QUANTITIES, series_quantity_dofs

    us = unit_system or DEFAULT_SYSTEM
    series = ([('', source, None)] if not isinstance(source, (list, tuple))
              else [('', obj, None) for obj in source])
    dofs = series_quantity_dofs(series, quantity)
    return _dof_plot(geometry, dofs, quantity in EXCITATION_QUANTITIES, us,
                     f'DOFs — {quantity}', screenshot, theme, show, size_in,
                     dpi, kwargs)


#: what a channel table's arrows can be limited to, in the order the
#: app's drop-down lists them; 'all' is every channel
CHANNEL_ROLES = ('all', 'control', 'reference', 'response', 'monitor')


def channel_dofs(table: Any, role: str = 'all') -> tuple[list[str], list[bool]]:
    """The DOFs of a channel table's channels in `role`, each with
    whether its arrow points in.

    `role` is one of `CHANNEL_ROLES`: 'control' is the channels with
    Control checked, whatever their role. A channel measuring an excitation
    (a force or a drive voltage) points in, as a force record's arrows
    do. Channels are kept even when they cannot be drawn yet (no node,
    no direction, or a node the geometry lacks), so the count of what
    was asked for is still known; `add_dof_arrows` skips them.
    """
    from ..core.report import EXCITATION_QUANTITIES

    if role not in CHANNEL_ROLES:
        raise ValueError(f'{role!r} is not a role; use one of {CHANNEL_ROLES}')
    dofs, types = table.dof_strings(), table.types()
    # the control channels by their flag, which any role can carry
    # (Brandon, 2026-10-22)
    controls = table.controls()
    rows = [i for i, own in enumerate(table.roles())
            if role == 'all' or own == role
            or (role == 'control' and controls[i])]
    return ([dofs[i] for i in rows],
            [types[i] in EXCITATION_QUANTITIES for i in rows])


def plot_channels(geometry: Geometry, table: Any, role: str = 'all',
                  unit_system: UnitSystem | None = None,
                  screenshot: str | None = None, theme: Any = None,
                  show: bool = True, size_in: tuple[float, float] | None = None,
                  dpi: float | None = None, **kwargs: Any) -> Any:
    """The geometry with labeled arrows at a channel table's DOFs: the
    app's view of a channel table selected with a geometry, from a
    script.

    `role` limits the arrows to the references, responses or monitors
    (`channel_dofs`). Shown or rendered to `screenshot` as `plot_dofs`
    is; other keyword arguments go to `geometry_scene`.
    """
    us = unit_system or DEFAULT_SYSTEM
    dofs, incoming = channel_dofs(table, role)
    return _dof_plot(geometry, dofs, incoming, us, f'Channels — {role}',
                     screenshot, theme, show, size_in, dpi, kwargs)


def _dof_plot(geometry, dofs, incoming, us, title, screenshot, theme, show,
              size_in, dpi, kwargs):
    """`plot_dofs` and `plot_channels` once the DOFs are chosen: the
    geometry and its arrows, shown or rendered to a file."""
    def draw(plotter: Any, scale: float = 1.0) -> None:
        geometry_scene(geometry, unit_system=us, plotter=plotter,
                       theme=theme, scale=scale, **kwargs)
        add_dof_arrows(plotter, geometry, dofs, unit_system=us,
                       incoming=incoming)

    if screenshot is not None:
        import pyvista as pv
        if dpi is not None:
            plotter, scale = print_plotter(size_in or (6.0, 4.5), dpi)
            draw(plotter, scale)
        else:
            plotter = pv.Plotter(off_screen=True)
            draw(plotter)
        place_view(plotter, geometry.opening_view, render=False)
        image = plotter.screenshot(str(screenshot))
        plotter.close()
        return image
    from ..gui.windows import scene_window

    return scene_window(draw, theme=theme,
                        axis_unit=axis_unit_text(geometry, us),
                        title=title, show=show,
                        view=geometry.opening_view)
