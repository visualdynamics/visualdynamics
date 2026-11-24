"""Draw time-correlation scores at measured nodes on a geometry scene."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.data import parse_dof
from .geometry import add_geometry, display_points, shared_scalars


def draw_time_correlation(plotter: Any, geometries: list, rows: list, *,
                          metric: str, contour: bool, unit_system: Any,
                          text_color: str = 'black',
                          smooth: bool = False) -> tuple[int, str]:
    """Draw each station's score at its node, and optionally a surface
    colored by the scores, on `plotter`.

    The stations are placed on the first geometry, by the node of each
    row's DOF; a geometry with elements is drawn translucent under them,
    colored by the nearest station's score with `contour`, or by an
    inverse-distance-squared blend of all of them with `smooth` as well.
    Returns the number of stations drawn and the axis unit text the
    scene's bounds are labeled with."""
    import pyvista as pv
    from pyvista import LookupTable
    from scipy.spatial import cKDTree

    from .geometry import COLORMAP

    primary = geometries[0]
    points, axis_unit = display_points(primary, unit_system)
    positions = {int(node): i for i, node in enumerate(primary.node_id)}
    stations = [(i, positions[node], row[metric]) for i, row in enumerate(rows)
                if row[metric] is not None
                and (node := parse_dof(row['dof'])[0]) in positions]
    extent = max((abs(score - 1) for _, _, score in stations), default=1)
    clim = (0, 1) if metric == 'trac' else (1 - max(extent, .05), 1 + max(extent, .05))
    palette = LookupTable(cmap=COLORMAP, scalar_range=clim)
    by_node = {}
    for _, index, score in stations:
        by_node.setdefault(index, []).append(score)
    surfaces = [geometry for geometry in geometries if len(geometry.elem_type)]
    for geometry in surfaces:
        scalars = None
        if contour and by_node:
            vertices, _ = display_points(geometry, unit_system)
            station_points = points[list(by_node)]
            station_scores = np.array([np.mean(scores) for scores in by_node.values()])
            values, scalars = shared_scalars(len(vertices), metric)
            if smooth:
                # Normalize by the closest distance before squaring to keep
                # weights finite. Coincident vertices retain the measured score.
                distances = np.linalg.norm(vertices[:, None] - station_points, axis=2)
                closest = distances.argmin(axis=1)
                minimum = distances.min(axis=1)
                weights = np.divide(minimum[:, None], distances,
                                    out=np.zeros_like(distances), where=distances > 0)**2
                exact = minimum == 0
                weights[exact, closest[exact]] = 1
                values[:] = (weights @ station_scores) / weights.sum(axis=1)
            else:
                _, nearest = cKDTree(station_points).query(vertices)
                values[:] = station_scores[nearest]
        axis_unit = add_geometry(plotter, geometry, unit_system, opacity=.45,
                     color_override='#8795a1' if scalars is None else None,
                     scalars=scalars, clim=clim)
    for index, scores in by_node.items():
        marker = pv.PolyData(points[index:index+1])
        plotter.add_mesh(marker, point_size=15, render_points_as_spheres=True,
                         color=palette.map_value(float(np.mean(scores)))[:3])
    if stations:
        label = 'TRAC' if metric == 'trac' else 'sTRAC'
        average = np.mean([score for _, _, score in stations])
        plotter.add_text(f'Average {label}: {average:.3f}', position='upper_left',
                         font_size=12, color=text_color, name='correlation-average')
        mapper = pv.DataSetMapper()
        mapper.lookup_table = palette
        plotter.add_scalar_bar(label,
                               mapper=mapper, color=text_color, vertical=True,
                               position_x=.87, position_y=.18, width=.09, height=.52)
    else:
        plotter.add_text('No scored DOFs match the linked geometry', font_size=10, color=text_color)
    return len(stations), axis_unit
