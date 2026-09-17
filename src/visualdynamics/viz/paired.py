"""Two densities on one stage: channels receding, paired per station.

The 3-D reading of two selected PSDs (Brandon, 2026-08-25), in the
banded stage's own idiom: each shared channel is a station along the
depth axis, and at each station the two objects meet — the louder
drawn level-colored, the quieter stood back in the muted gray a
compared reference wears. The Ratio reading divides instead: one
curve per station, in decibels, linear — the number being read *is*
the decibel.

Rides `waterfall_arrays` for decimation, paging and labels, and
`density_ratio` for the division, so the stage, the flat plots and
the report cannot disagree about a value.

Kept free of Qt, like the rest of `viz`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.data import density_ratio
from ..theme import theme as resolve_theme
from ..units import DEFAULT_SYSTEM, UnitSystem
from ._quiet import one_render
from .waterfall import (
    POINT_BUDGET,
    STAGE,
    finish_scene,
    place_camera,
    waterfall_arrays,
)

__all__ = ['add_paired_stage', 'paired_arrays', 'plot_paired_stage']

#: the quieter object's opacity — present, never competing
QUIET_OPACITY = 0.55


def paired_arrays(loud: Any, quiet: Any,
                  loud_records: Sequence[int] | None = None,
                  quiet_records: Sequence[int] | None = None,
                  unit_system: UnitSystem | None = None,
                  budget: int = POINT_BUDGET,
                  quantity: tuple[str, str | None] | None = None,
                  page: int = 0) -> dict[str, Any]:
    """The two objects' curves, matched into stations by label.

    A station exists where both objects hold a record of the same
    label (DOF and quantity, as `record_label` spells it); records
    only one side holds are left out and counted in ``unmatched``.
    """
    us = unit_system or DEFAULT_SYSTEM
    half = max(budget // 2, 1)
    arrays_loud = waterfall_arrays(loud, loud_records, us, 'magnitude',
                                   half, quantity, page)
    arrays_quiet = waterfall_arrays(quiet, quiet_records, us,
                                    'magnitude', half, quantity, page)
    by_label = dict(zip(arrays_quiet['labels'], arrays_quiet['curves']))
    stations = []
    for label, curve in zip(arrays_loud['labels'],
                            arrays_loud['curves']):
        other = by_label.get(label)
        if other is None:
            continue
        stations.append({'label': label, 'loud': curve, 'quiet': other})
    unmatched = (len(arrays_loud['labels']) + len(arrays_quiet['labels'])
                 - 2 * len(stations))
    return {'stations': stations, 'unmatched': unmatched,
            'log_scaled': arrays_loud['log_scaled'],
            'xlabel': arrays_loud['xlabel'],
            'zlabel': arrays_loud['zlabel'],
            'page': arrays_loud['page'], 'pages': arrays_loud['pages']}


def _extent(values, fallback):
    finite = [v for v in values if np.isfinite(v)]
    return (min(finite), max(finite)) if finite else fallback


def paired_stage_curves(loud: Any, quiet: Any,
                        loud_records: Sequence[int] | None = None,
                        quiet_records: Sequence[int] | None = None,
                        mode: str = 'overlay',
                        unit_system: UnitSystem | None = None,
                        budget: int = POINT_BUDGET,
                        quantity: tuple[str, str | None] | None = None,
                        page: int = 0) -> dict[str, Any]:
    """The pair placed on the unit stage — overlaid, or divided.

    The paired counterpart of `waterfall.stage_curves`, and here for
    the same reason: the app draws this stage in VTK and the report
    draws it in a canvas, and two normalizations would be two
    pictures of one measurement. Returns ``runs`` — each
    ``{'station', 'points', 'levels', 'quiet', 'label'}``, the quiet
    ones being the stood-back object in an overlay — plus the real
    ``extents`` (x0, x1, z0, z1) the stage stands in for, the station
    ``labels``, and the axis titles.

    In 'ratio' the division is `density_ratio` at full resolution —
    the same call the flat plot and the report's flat figure divide
    through — so no reading of a pair can disagree with another.
    """
    us = unit_system or DEFAULT_SYSTEM
    arrays = paired_arrays(loud, quiet, loud_records, quiet_records,
                           us, budget, quantity, page)
    sx, sy, sz = STAGE
    runs: list[dict[str, Any]] = []

    if mode == 'ratio':
        abscissa, rows, dofs, dims = density_ratio(loud, quiet)
        with np.errstate(divide='ignore', invalid='ignore'):
            decibels = 10.0 * np.log10(np.real(rows))
        drawn = [(dof, decibels[k]) for k, dof in enumerate(dofs)
                 if np.isfinite(decibels[k]).any()
                 and (quantity is None or dims[k] == quantity[0])]
        x0, x1 = float(abscissa.min()), float(abscissa.max())
        z_all = np.concatenate([row[np.isfinite(row)]
                                for _dof, row in drawn]) \
            if drawn else np.array([0.0])
        z0, z1 = float(z_all.min()), float(z_all.max())
        xspan, zspan = (x1 - x0) or 1.0, (z1 - z0) or 1.0
        for k, (dof, row) in enumerate(drawn):
            station = (k / max(len(drawn) - 1, 1)) * sy
            finite = np.isfinite(row)
            if finite.sum() < 2:
                continue
            xn = (abscissa[finite] - x0) / xspan * sx
            zn = (row[finite] - z0) / zspan * sz
            runs.append({
                'station': float(station), 'quiet': False, 'label': dof,
                'points': np.column_stack(
                    [xn, np.full(len(xn), station), zn]),
                'levels': np.asarray(row[finite], dtype=float)})
        arrays['runs'] = runs
        arrays['extents'] = (x0, x1, z0, z1)
        arrays['drawn'] = len(drawn)
        arrays['zlabel'] = 'ratio [dB]'
        return arrays

    stations = arrays['stations']
    n = len(stations)
    xs = [float(np.nanmin(c[which][0])) for c in stations
          for which in ('loud', 'quiet') if len(c[which][0])]
    xe = [float(np.nanmax(c[which][0])) for c in stations
          for which in ('loud', 'quiet') if len(c[which][0])]
    zs = np.concatenate(
        [c[which][1][np.isfinite(c[which][1])] for c in stations
         for which in ('loud', 'quiet')]) if stations else np.array([0.0, 1.0])
    x0, x1 = _extent(xs, (0.0,) * 2)[0], _extent(xe, (1.0,) * 2)[1]
    z0, z1 = float(zs.min()), float(zs.max())
    xspan, zspan = (x1 - x0) or 1.0, (z1 - z0) or 1.0
    for k, station_curves in enumerate(stations):
        station = (k / max(n - 1, 1)) * sy
        for which in ('loud', 'quiet'):
            cx, cz = station_curves[which]
            finite = np.isfinite(cz)
            if finite.sum() < 2:
                continue
            xn = (cx[finite] - x0) / xspan * sx
            zn = (cz[finite] - z0) / zspan * sz
            runs.append({
                'station': float(station), 'quiet': which == 'quiet',
                'label': station_curves['label'],
                'points': np.column_stack(
                    [xn, np.full(len(xn), station), zn]),
                'levels': np.asarray(cz[finite], dtype=float)})
    arrays['runs'] = runs
    arrays['extents'] = (x0, x1, z0, z1)
    arrays['drawn'] = n
    return arrays


@one_render
def add_paired_stage(plotter: Any, loud: Any, quiet: Any,
                     loud_records: Sequence[int] | None = None,
                     quiet_records: Sequence[int] | None = None,
                     mode: str = 'overlay',
                     unit_system: UnitSystem | None = None,
                     theme: Any = None,
                     budget: int = POINT_BUDGET,
                     quantity: tuple[str, str | None] | None = None,
                     page: int = 0) -> dict[str, Any]:
    """Draw the pair into a plotter — overlaid, or divided in dB.

    The numbers come from `paired_stage_curves`, which the report's
    canvas figure reads too; this places them in VTK and labels the
    axes with the real ranges the stage stands in for.
    """
    import pyvista as pv

    colors = resolve_theme(theme)
    us = unit_system or DEFAULT_SYSTEM
    arrays = paired_stage_curves(loud, quiet, loud_records, quiet_records,
                                 mode, us, budget, quantity, page)
    x0, x1, z0, z1 = arrays['extents']
    sx, sy, sz = STAGE
    del sx, sy, sz

    def mesh_of(runs):
        """One PolyData over a set of runs, and the levels along them."""
        points, lines, scalars, total = [], [], [], 0
        for run in runs:
            count = len(run['points'])
            points.append(run['points'])
            lines.append(np.concatenate([[count],
                                         np.arange(total, total + count)]))
            scalars.append(run['levels'])
            total += count
        if not total:
            return None, 0
        built = pv.PolyData(np.vstack(points).astype(np.float32),
                            lines=np.concatenate(lines))
        built.point_data['level'] = np.concatenate(scalars)
        return built, total

    louds = [run for run in arrays['runs'] if not run['quiet']]
    quiets = [run for run in arrays['runs'] if run['quiet']]
    loud_mesh, total = mesh_of(louds)
    quiet_mesh, quiet_total = mesh_of(quiets)
    if loud_mesh is not None:
        plotter.add_mesh(loud_mesh, scalars='level', cmap='viridis',
                         line_width=2, clim=(z0, z1),
                         show_scalar_bar=False,
                         name='paired-ratio' if mode == 'ratio'
                         else 'paired-loud')
    if quiet_mesh is not None:
        # the quieter object stood back in the muted gray a compared
        # reference wears — present, never competing
        plotter.add_mesh(quiet_mesh, color=colors['specification_curve'],
                         opacity=QUIET_OPACITY, line_width=2,
                         name='paired-quiet')
    spots = [[run['points'][0][0] - 0.03 * STAGE[0], run['station'],
              run['points'][0][2]] for run in louds]
    names = [run['label'] for run in louds]
    if names:
        plotter.add_point_labels(
            np.asarray(spots), names, font_size=12, always_visible=True,
            text_color=colors['scene_text'], shape=None,
            fill_shape=False, show_points=False, name='paired-labels')
    if total:
        plotter.show_bounds(
            axes_ranges=(x0, x1, 0, max(len(louds) - 1, 1), z0, z1),
            fmt='%.4g',
            xtitle=arrays['xlabel'], ytitle=' ',
            ztitle=arrays['zlabel'], show_ylabels=False, grid='back',
            location='outer', use_3d_text=False,
            color=colors['scene_text'])
    arrays['points'] = total + quiet_total
    return arrays


def plot_paired_stage(loud: Any, quiet: Any, mode: str = 'overlay',
                      unit_system: UnitSystem | None = None,
                      theme: Any = None,
                      screenshot: str | None = None,
                      show: bool = True) -> Any:
    """The headless call for the stage's paired readings."""
    import pyvista as pv

    colors = resolve_theme(theme)
    plotter = pv.Plotter(off_screen=screenshot is not None or not show)
    plotter.set_background(colors['scene_background'],
                           top=colors['scene_background_top'])
    add_paired_stage(plotter, loud, quiet, mode=mode,
                     unit_system=unit_system, theme=theme)
    place_camera(plotter)
    return finish_scene(plotter, screenshot, show)
