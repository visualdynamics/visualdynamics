"""Draw time-correlation scores at measured nodes on a geometry scene."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.data import parse_dof
from .geometry import add_geometry, display_points, shared_scalars


def draw_time_correlation(plotter: Any, geometries: list, rows: list, *,
                          metric: str, contour: bool, unit_system: Any,
                          text_color: str = 'black') -> int:
    """Draw selectable scores and an optional interpolated surface."""
    import pyvista as pv
    from matplotlib import colormaps
    from scipy.spatial import cKDTree

    from .geometry import COLORMAP

    primary = geometries[0]
    points, _ = display_points(primary, unit_system)
    positions = {int(node): i for i, node in enumerate(primary.node_id)}
    stations = [(i, positions[node], row[metric]) for i, row in enumerate(rows)
                if row[metric] is not None
                and (node := parse_dof(row['dof'])[0]) in positions]
    extent = max((abs(score - 1) for _, _, score in stations), default=1)
    clim = (0, 1) if metric == 'trac' else (1 - max(extent, .05), 1 + max(extent, .05))
    palette = colormaps[COLORMAP]
    by_node = {}
    for _, index, score in stations:
        by_node.setdefault(index, []).append(score)
    surfaces = [geometry for geometry in geometries if len(geometry.elem_type)]
    for geometry in surfaces:
        scalars = None
        if contour and by_node:
            vertices, _ = display_points(geometry, unit_system)
            # The nearest scored station gives a piecewise-constant map.
            # Markers still carry each actual measurement.
            _, nearest = cKDTree(points[list(by_node)]).query(vertices)
            values, scalars = shared_scalars(len(vertices), metric)
            values[:] = np.array([np.mean(scores) for scores in by_node.values()])[nearest]
        add_geometry(plotter, geometry, unit_system, opacity=.45,
                     color_override='#8795a1' if scalars is None else None,
                     scalars=scalars, clim=clim)
    for index, scores in by_node.items():
        at_node = [(row, score) for row, point, score in stations if point == index]
        marker = pv.PolyData(points[index:index+1])
        marker.field_data['comparison_row'] = [row for row, _ in at_node]
        plotter.add_mesh(marker, point_size=15, render_points_as_spheres=True,
                         color=palette((np.mean(scores)-clim[0])/(clim[1]-clim[0]))[:3])
    if stations:
        plotter.add_text(f'{metric.upper() if metric == "trac" else "sTRAC"}: '
                         f'{clim[0]:.3g} to {clim[1]:.3g}', font_size=10, color=text_color)
    else:
        plotter.add_text('No scored DOFs match the linked geometry', font_size=10, color=text_color)
    return len(stations)


def matching_surfaces(primary: Any, candidates: Any) -> list:
    """Find meshes whose node IDs and positions agree with a point survey."""
    from ..core.geometry import Geometry

    matched = []
    for geometry in candidates:
        if not isinstance(geometry, Geometry) or geometry is primary or not len(geometry.elem_type):
            continue
        if primary.num_nodes == 0 or not np.all(geometry.contains_nodes(primary.node_id)):
            continue
        positions = geometry.node_xyz[geometry.node_index(primary.node_id)]
        # A scan may be projected onto the XY plane. Keep that geometry as
        # measured and check the axes on which it actually spans a distance.
        axes = np.ptp(primary.node_xyz, axis=0) > 1e-8
        if axes.any() and np.allclose(positions[:, axes], primary.node_xyz[:, axes],
                                     atol=1e-4, rtol=0):
            matched.append(geometry)
    return matched
