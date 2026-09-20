"""PyVista scene construction for Geometry.

Coordinates are converted from stored SI to the display unit system at scene
build time — switching unit systems rebuilds the scene, never the data.

Tracelines and line elements are drawn with direct colors, grouped by color
index, except when the scene is colored by value (see docs/colormap.md).
An older note here warned that macOS VTK silently drops scalar-mapped line
cells; that was measured against pyvista 0.48.4 / VTK 9.6.2 and does not
reproduce — see the same doc for the numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.geometry import ELEMENT_TYPES, face_corners
from ..theme import theme as resolve_theme
from ..units import DEFAULT_SYSTEM

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.geometry import Geometry
    from ..units import UnitSystem

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


def color_name(index: int) -> str:
    return COLOR_NAMES[int(index) % len(COLOR_NAMES)]


def color_rgb(index: int) -> tuple[float, float, float]:
    return PALETTE[int(index) % len(PALETTE)]


def axis_unit_text(geometry: Geometry, unit_system: UnitSystem) -> str:
    """What the labeled axes say. A geometry whose units are undefined
    says so rather than naming one."""
    if not geometry.units_defined:
        return '[units undefined]'
    return f"[{unit_system.label_text('length')}]"


def display_points(geometry: Geometry,
                   unit_system: UnitSystem) -> tuple[np.ndarray, str]:
    """(coordinates to draw, axis unit text) for a geometry.

    A geometry whose units are undefined is drawn with the file's raw
    coordinates — converting them would silently scale unknown values — and
    its axes say so rather than naming a unit.
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


def _cells(index_lists: Sequence[Sequence[int]]) -> np.ndarray:
    """Flat VTK cell array: [n, i0..in-1, n, i0..in-1, ...]"""
    return np.concatenate([[len(ix), *ix] for ix in index_lists])


def geometry_scene(geometry: Geometry, unit_system: UnitSystem | None = None,
                   plotter: Any = None, node_size: float = 8.0,
                   line_width: float = 2.0, show_edges: bool = True,
                   opacity: float = 1.0,
                   labels: Sequence[str] | None = None,
                   off_screen: bool = False, theme: Any = None,
                   components: Sequence[str] | None = None) -> Any:
    """Build (or add to) a PyVista plotter showing the geometry.

    `theme` is 'light', 'dark', or a colors dict; it sets the scene
    background and annotation color. `components` limits what is drawn to a
    subset of {'nodes', 'tracelines', 'elements'} — selecting one in the
    project tree shows just that part. Returns the plotter; call .show() on
    it (or .screenshot() if off_screen).
    """
    import pyvista as pv

    us = unit_system or DEFAULT_SYSTEM
    colors = resolve_theme(theme)
    if plotter is None:
        plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background(colors['scene_background'])
    axis_unit = add_geometry(plotter, geometry, unit_system=us,
                             node_size=node_size, line_width=line_width,
                             show_edges=show_edges, opacity=opacity,
                             labels=labels, components=components,
                             text_color=colors['scene_text'])
    annotate_scene(plotter, axis_unit, colors)
    return plotter


def annotate_scene(plotter: Any, axis_unit: str, colors: Mapping[str, str],
                   bounds: bool = True,
                   orientation: bool = True) -> None:
    """Scene annotations, each independently switchable.

    `bounds` is the labeled box drawn around the geometry; `orientation` is
    the small triad in the corner. The 3D view toolbar toggles them
    separately.
    """
    if bounds:
        plotter.show_bounds(xtitle=f'X {axis_unit}', ytitle=f'Y {axis_unit}',
                            ztitle=f'Z {axis_unit}', grid='back',
                            location='outer', color=colors['scene_text'])
    else:
        plotter.remove_bounds_axes()
    if orientation:
        plotter.add_axes(color=colors['scene_text'])
    else:
        plotter.hide_axes()


def shared_points(points: ArrayLike) -> tuple[Any, np.ndarray]:
    """A vtkPoints every mesh can share, plus a writable numpy view of it.

    Writing through the view and calling Modified() moves every mesh at
    once — the difference between one upload per frame and one per mesh.
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


def dof_axis(direction: str) -> int | None:
    """0/1/2 for a DOF direction's axis — 'RZ-' is a Z, so blue."""
    letter = str(direction).upper().lstrip('R')[:1]
    return {'X': 0, 'Y': 1, 'Z': 2}.get(letter)


def dof_arrow_length(points: ArrayLike, extent: float) -> float:
    """One length for every arrow on a plot: 12% of the geometry's
    extent, shrunk to the closest spacing of the arrowed nodes so a
    dense set of arrows never overlaps its neighbors."""
    length = 0.12 * extent
    unique = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(unique) > 1:
        deltas = unique[:, np.newaxis, :] - unique[np.newaxis, :, :]
        distances = np.sqrt((deltas ** 2).sum(axis=2))
        distances[distances == 0] = np.inf
        nearest = float(distances.min())
        if np.isfinite(nearest):
            length = min(length, 0.9 * nearest)
    return length


def add_dof_arrows(plotter: Any, geometry: Geometry, dofs: Sequence[str],
                   unit_system: UnitSystem | None = None,
                   incoming: bool = False,
                   name: str | None = None) -> int:
    """Labeled arrows marking DOFs on the geometry.

    One arrow per DOF, colored by the axis it points along — X red,
    Y green, Z blue, the orientation marker's own convention. Response
    style starts at the node and points outward, label at the tip;
    `incoming` (forces) ends on the node instead, label at the base.
    All arrows share one length: 12% of the geometry's extent, shrunk
    to the closest node spacing so dense sets never overlap. DOFs at
    nodes the geometry does not have are skipped. Returns how many
    were drawn.
    """
    import pyvista as pv

    from ..core.data import parse_dof

    points, _unit = display_points(geometry, unit_system)
    points = np.asarray(points, dtype=float)
    if not len(points):
        return 0
    rows = {int(node): i for i, node in enumerate(geometry.node_id)}
    center = points.mean(axis=0)
    extent = float(np.sqrt(((points - center) ** 2)
                           .sum(axis=1).max())) or 1.0
    entries = []
    for dof in dofs:
        node, direction = parse_dof(dof)
        vector = DOF_DIRECTIONS.get(str(direction).upper())
        axis = dof_axis(direction)
        if node is None or vector is None or axis is None \
                or int(node) not in rows:
            continue
        entries.append((points[rows[int(node)]],
                        np.asarray(vector, dtype=float), axis, dof))
    if not entries:
        return 0
    length = dof_arrow_length([entry[0] for entry in entries], extent)
    label_spots = {0: [], 1: [], 2: []}
    for index, (position, vector, axis, dof) in enumerate(entries):
        start = position - vector * length if incoming else position
        plotter.add_mesh(
            pv.Arrow(start=start, direction=vector, scale=length),
            color=AXIS_COLORS[axis],
            name=None if name is None else f'{name}-arrow{index}')
        spot = (start - vector * length * 0.25 if incoming
                else position + vector * length * 1.25)
        label_spots[axis].append((spot, dof))
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
                fill_shape=False, show_points=False,
                name=None if name is None else f'{name}-labels{axis}')
    return len(entries)


def add_coordinate_system(plotter: Any, origin: ArrayLike, matrix: ArrayLike,
                          cs_type: int, length: float,
                          text_color: str = '#000000',
                          label: str | None = None,
                          name: str | None = None) -> None:
    """One coordinate system: three directions, named, with angles curved.

    `label` (the system's id) is written at the origin. Direction names come
    from the type — X/Y/Z, R/theta/Z, or R/theta/phi — and each is written
    at the tip of its own arrow, so a cylindrical system is tellable from a
    cartesian one at a glance rather than by consulting the table.

    Given a `name`, every actor is named from it, so drawing again replaces
    the drawing instead of piling another one on top — which is what lets a
    frame be turned live without rebuilding the scene around it.
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
    """A VTK point-data array every mesh can share, and a numpy view of it.

    The color twin of `shared_points`: one array, written once per frame,
    recolors every mesh in the scene at once.
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
    """`labels` in one shape: {kind: the ids to caption, or None}.

    Both spellings read naturally where they are used. A script says
    `labels=['nodes']` and means *the nodes that are drawn*; the window,
    which knows exactly what the user picked, says
    `labels={'blocks': [3]}`. None means "whatever this kind draws",
    which is the right answer for both.
    """
    if labels is None:
        return {}
    if isinstance(labels, Mapping):
        return dict(labels)
    return {str(kind): None for kind in labels}


def labels_fit(geometry: Geometry, kind: str,
               chosen: Sequence[int] | None = None) -> bool:
    """Would captioning this kind stay under `ENTITY_LABEL_LIMIT`?

    Asked by the window before it turns labels on, so a selection too
    large to caption says so in the status bar rather than drawing
    nothing and leaving the user to wonder.
    """
    if chosen is not None:
        return len(chosen) <= ENTITY_LABEL_LIMIT
    counts = {'nodes': geometry.num_nodes,
              'coordinate_systems': len(geometry.cs_id),
              'tracelines': len(geometry.traceline_conn),
              'elements': len(geometry.elem_conn),
              'blocks': len(geometry.block_id)}
    return counts.get(kind, 0) <= ENTITY_LABEL_LIMIT


def label_spots(geometry: Geometry, points: np.ndarray, kind: str,
                chosen: Sequence[int] | None = None
                ) -> tuple[np.ndarray, list[str]]:
    """(positions, texts) captioning one kind of entity with its id.

    `points` is the geometry's nodes as drawn, so a caption lands in the
    same units and the same frame as the thing it names. Everything but
    a node is captioned at the **centroid of its own nodes**: the middle
    of a traceline's run, of an element's corners, of a block's
    elements. That is where a reader looks for the name of a shape, and
    it keeps the number off the vertices, which are already carrying
    node ids whenever both are shown.

    A traceline id can name several runs — a UNV dataset-82 line that
    lifts the pen arrives as several polylines under one id — and each
    run is captioned where it is. One centroid for the id would land
    between them, in empty space, naming nothing.

    `chosen` is the ids (nodes, coordinate systems, blocks) or indices
    (tracelines, elements) to caption; None captions every one of the
    kind. Returns nothing past `ENTITY_LABEL_LIMIT`; see `labels_fit`.
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

    if kind == 'tracelines':
        wanted = (list(chosen) if chosen is not None
                  else range(len(geometry.traceline_conn)))
        for i in wanted:
            spot = center(geometry.traceline_conn[int(i)])
            if spot is not None:
                spots.append(spot)
                texts.append(str(int(geometry.traceline_id[int(i)])))

    elif kind == 'elements':
        wanted = (list(chosen) if chosen is not None
                  else range(len(geometry.elem_conn)))
        for i in wanted:
            spot = center(geometry.elem_conn[int(i)])
            if spot is not None:
                spots.append(spot)
                texts.append(str(int(geometry.elem_id[int(i)])))

    elif kind == 'blocks':
        # a block holds no coordinates of its own: it is a label on
        # elements, so it is captioned in the middle of the elements
        # that carry its id
        wanted = ({int(b) for b in chosen} if chosen is not None
                  else {int(b) for b in geometry.block_id})
        blocks = np.asarray(geometry.elem_block, dtype=np.int64)
        for row, block in enumerate(geometry.block_id):
            if int(block) not in wanted:
                continue
            members = np.flatnonzero(blocks == int(block))
            nodes = [n for i in members for n in geometry.elem_conn[int(i)]]
            spot = center(nodes)
            if spot is not None:
                spots.append(spot)
                name = str(geometry.block_name[row]).strip()
                texts.append(name or str(int(block)))

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

    `color_override` paints the whole geometry one color, which is how
    several geometries overlaid in one scene stay tellable apart.
    `entities` restricts drawing to specific items, as a dict with any of
    'nodes' (node ids), 'coordinate_systems' (ids), 'tracelines' (indices)
    and 'elements' (indices) — that is how a single node or traceline picked
    in the tree gets highlighted.

    `labels` names the kinds to caption with their ids — any of 'nodes',
    'coordinate_systems', 'tracelines', 'elements', 'blocks'. Captions
    follow `entities` when it restricts the drawing, so labeling a
    picked traceline names that one and not all of them, and a kind with
    more than `ENTITY_LABEL_LIMIT` of them is left uncaptioned (see
    `labels_fit`).

    `scalars` is a VTK point-data array shared by every mesh, coloring the
    whole geometry by value instead of by the geometry's own color indices;
    `clim` fixes what the ends of the color map mean. One array serves all
    the meshes, so a frame writes it once. Returns the axis unit text.
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
    # it to, so labeling a picked traceline names that one alone
    wanted_labels = {kind: (picked.get(kind) if chosen is None else chosen)
                     for kind, chosen in label_choices(labels).items()}
    if components:
        draw = set(components)
    elif picked:
        draw = {kind for kind, values in picked.items() if values}
    else:
        draw = {'nodes', 'tracelines', 'elements'}
    id_to_row = {int(i): r for r, i in enumerate(geometry.node_id)}

    def rows(node_ids: Sequence[int]) -> list[int]:
        return [id_to_row[int(i)] for i in node_ids]

    # Nodes, grouped by color
    wanted_nodes = picked.get('nodes')
    node_mask = (np.isin(geometry.node_id, list(wanted_nodes))
                 if wanted_nodes else np.ones(geometry.num_nodes, bool))
    for color in (np.unique(geometry.node_color[node_mask])
                  if 'nodes' in draw and node_mask.any() else []):
        mask = (geometry.node_color == color) & node_mask
        indices = np.flatnonzero(mask)
        mesh = new_mesh()
        mesh.verts = np.column_stack(
            [np.ones(len(indices), dtype=np.int64), indices]).ravel()
        plotter.add_mesh(mesh, **painted(color), point_size=node_size,
                         render_points_as_spheres=True, opacity=opacity)

    # Tracelines: direct color per group (see VTK note above)
    line_groups = {}
    wanted_lines = picked.get('tracelines')
    traceline_indices = (list(wanted_lines) if wanted_lines is not None
                         else range(len(geometry.traceline_conn)))
    tracelines = ([(geometry.traceline_color[i], geometry.traceline_conn[i])
                   for i in traceline_indices] if 'tracelines' in draw else [])
    for color, conn in tracelines:
        if len(conn) >= 2:
            line_groups.setdefault(int(color), []).append(rows(conn))
    for color, polylines in line_groups.items():
        mesh = new_mesh()
        mesh.lines = _cells(polylines)
        plotter.add_mesh(mesh, **painted(color), line_width=line_width,
                         opacity=opacity)

    # Elements, split by render class and grouped by color
    faces, lines, cell_points = {}, {}, {}
    wanted_elements = picked.get('elements')
    element_indices = (list(wanted_elements) if wanted_elements is not None
                       else range(len(geometry.elem_conn)))
    elements = ([(geometry.elem_type[i], geometry.elem_color[i],
                  geometry.elem_conn[i]) for i in element_indices]
                if 'elements' in draw else [])
    for code, color, conn in elements:
        _name, _nnodes, render = ELEMENT_TYPES[int(code)]
        if render == 'face':
            faces.setdefault(int(color), []).append(
                rows(conn[:face_corners(int(code))]))
        elif render == 'volume':
            # render outer faces later; M0 draws the wireframe of the cell
            lines.setdefault(int(color), []).append(rows(conn))
        elif render == 'line':
            lines.setdefault(int(color), []).append(rows(conn[:2]))
        elif render == 'point':
            cell_points.setdefault(int(color), []).extend(rows(conn))
    for color, polys in faces.items():
        mesh = new_mesh()
        mesh.faces = _cells(polys)
        plotter.add_mesh(mesh, **painted(color), opacity=opacity,
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


def plot_geometry(geometry: Geometry, unit_system: UnitSystem | None = None,
                  screenshot: str | None = None, theme: Any = None,
                  show: bool = True, **kwargs: Any) -> Any:
    """Show the geometry interactively, or render to `screenshot` headlessly.

    Shown, it comes up in the app's own 3-D pane — the labeled axes and
    the orientation triad are toggles on the bar over it, exactly as in
    the window. Returns the pane (its `.plotter` is the PyVista one), or
    the image array when rendering to a file.
    """
    if screenshot is not None:
        plotter = geometry_scene(geometry, unit_system=unit_system,
                                 theme=theme, off_screen=True, **kwargs)
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
        title='Geometry', show=show)


def plot_dofs(geometry: Geometry, source: Any, quantity: str,
              unit_system: UnitSystem | None = None,
              screenshot: str | None = None, theme: Any = None,
              show: bool = True, **kwargs: Any) -> Any:
    """The geometry with labeled arrows at every DOF `source` measures
    as `quantity` — the GUI's DOF-arrows toggle, from a script.

    Forces end on their node with the label at the base, responses
    leave it with the label at the tip, exactly as the desktop draws
    them. `source` is a data object (or several).
    """
    from ..core.report import EXCITATION_QUANTITIES, series_quantity_dofs

    us = unit_system or DEFAULT_SYSTEM
    series = ([('', source, None)] if not isinstance(source, (list, tuple))
              else [('', obj, None) for obj in source])
    dofs = series_quantity_dofs(series, quantity)
    def draw(plotter: Any) -> None:
        geometry_scene(geometry, unit_system=us, plotter=plotter,
                       theme=theme, **kwargs)
        add_dof_arrows(plotter, geometry, dofs, unit_system=us,
                       incoming=quantity in EXCITATION_QUANTITIES)

    if screenshot is not None:
        import pyvista as pv
        plotter = pv.Plotter(off_screen=True)
        draw(plotter)
        image = plotter.screenshot(str(screenshot))
        plotter.close()
        return image
    from ..gui.windows import scene_window

    return scene_window(draw, theme=theme,
                        axis_unit=axis_unit_text(geometry, us),
                        title=f'DOFs — {quantity}', show=show)
