"""Specifications with limits on a 3-D stage: frequency, channel, level.

A random or shock specification is one curve per channel, so the third
axis is the channel: frequency across, channels receding (the
waterfall's depth axis), level up. Alone, a specification shows every
channel's requirement with its warning and abort zones on one stage.
With its measurement, every channel's comparison shows at once, and the
frequency lines that went outside an abort limit are shaded as the 2-D
comparison shades them: one rectangle per line outside, as wide as its
own bin, red up to the top of the stage above the upper abort limit and
blue down to the floor below the lower one.

The measured curves come from `waterfall_arrays`, with the same
decimation, log mapping and labels the plain waterfall uses, so the two
3-D views cost and behave alike. The exceedances are judged at full
resolution with `core.compliance.outside`, the same call the 2-D shading
and the compliance table use, so what is shaded here and what is
counted there agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..theme import theme as resolve_theme
from ..units import DEFAULT_SYSTEM, UnitSystem
from ._quiet import one_render
from .sinespec import _TONE_COLORS, EXCEED_OPACITY, ZONE_OPACITY, limit_zones
from .waterfall import (
    POINT_BUDGET,
    STAGE,
    drawn_span,
    finish_scene,
    place_camera,
    ribbon,
    waterfall_arrays,
)

# The sine stage put a specification in its own three dimensions first;
# this is the same view for every other banded spectrum (Brandon,
# 2026-08-22).


def banded_stage_arrays(specification: Any, measured: Any = None,
                        records: Any = None,
                        unit_system: UnitSystem | None = None,
                        budget: int = POINT_BUDGET,
                        specification_records: Any = None
                        ) -> dict[str, Any]:
    """The stage's numbers, before VTK is involved.

    One station per channel that the specification and the `measured`
    records (`records` of it) share, or per specification channel when
    nothing is measured, in the specification's order. The measured
    record at a station is the one the comparison pairs with the
    station's autospectrum (`compliance.matched_records`), so a
    measured cross-spectral matrix draws its autospectra. `specification_records` limits the stations
    to the channels of those specification records; with nothing
    measured, each picked record is its own station instead, and a cross
    term is drawn as its magnitude and labeled by its pair. A station's
    target is its channel's autospectrum.

    Each station holds the specification's target and limit curves (for
    drawing, log10 when the level is log-scaled, and as written) and,
    when measured, that channel's drawn curves (decimated to `budget`
    as the waterfall decimates) and the abort exceedances, judged at
    full resolution. Returns a dict of ``stations`` and the shared
    values: ``spec_x`` and ``spec_x_drawn`` (the specification's lines,
    and the abscissa its curves are drawn along), ``log_scaled``,
    ``xlabel``, ``zlabel``, ``compared`` (whether anything is
    measured), ``left_out`` (measured records paired with a station
    but not drawn, past the one scene the stage holds), and the two
    transforms ``logged`` (level) and ``spread``
    (abscissa) into drawing values.
    """
    from ..core.compliance import outside

    us = unit_system or DEFAULT_SYSTEM
    spec_x = np.asarray(specification.display_abscissa(us), dtype=float)
    spec_dofs = [str(dof) for dof in specification.response_dof]

    drawn_by_dof: dict[str, list[dict[str, Any]]] = {}
    log_scaled = True
    xlabel = f'frequency [{us.label_text("frequency")}]'
    if getattr(specification, 'log_abscissa', False):
        # VTK prints the coordinates as they are, so the title says so
        # — the same honesty as the vertical axis's own prefix
        xlabel = f'log10 {xlabel}'
    zlabel = None
    left_out = 0
    references = specification.reference_dof
    if measured is not None:
        # the measured record at each station is the one the comparison
        # pairs with the station's target (`matched_records`, by response,
        # reference and quantity): every record wearing the channel was
        # drawn, so a measured cross-spectral matrix put each channel's
        # cross terms on its station, judged against the autospectrum's
        # abort limits, and its channels past the waterfall's first page
        # were not drawn at all (2026-10-14, found by the docstring pass)
        from ..core.compliance import matched_records

        station_specs = [i for i, dof in enumerate(spec_dofs)
                         if references is None or str(references[i]) == dof]
        if specification_records is not None:
            chosen = {spec_dofs[int(i)] for i in specification_records}
            station_specs = [i for i in station_specs if spec_dofs[i] in chosen]
        paired = [m for _label, _s, m in matched_records(
            specification, measured, station_specs, records)]
        arrays = waterfall_arrays(measured, paired, us, budget=budget)
        # paired but not drawn: another quantity, or past the first page
        # of a set too large for one scene; the window says how many
        left_out = len(paired) - len(arrays['drawn'])
        log_scaled = arrays['log_scaled']
        xlabel, zlabel = arrays['xlabel'], arrays['zlabel']
        full_x = np.asarray(measured.display_abscissa(us), dtype=float)
        for (cx, cz), index in zip(arrays['curves'], arrays['drawn']):
            dof = str(measured.response_dof[index])
            full = np.real(np.asarray(
                measured.display_ordinate(us, [index])[0]))
            drawn_by_dof.setdefault(dof, []).append(
                {'x': cx, 'z': cz, 'full_x': full_x, 'full': full,
                 'label': measured.record_label(index)})

    # a station's target is its channel's autospectrum. The first
    # record wearing the channel is not that when the target holds its
    # cross terms — a virtual point's target with every pair stated
    # put M1/M2, all zeros, on M2's station, and the stage drew one
    # channel of three (Brandon, 2026-09-07)
    autos: dict[str, int] = {}
    for i, dof in enumerate(spec_dofs):
        if references is None or str(references[i]) == dof:
            autos.setdefault(dof, i)
    picked = (None if specification_records is None
              else sorted({int(i) for i in specification_records}))
    if measured is None and picked is not None:
        # the records picked, each its own station: an auto with its
        # bands, a cross term as the magnitude it is, labeled by its
        # pair — nine picked of a virtual point's target showed three
        # (Brandon, 2026-09-07: "never a cross-term label"). Compared,
        # the stations stay the channels, since a measurement is
        # judged against a channel's bands and a cross term has none.
        station_rows = [
            (dof if references is None or str(references[i]) == dof
             else f'{dof}/{references[i]}', i)
            for i in picked for dof in [spec_dofs[i]]]
    else:
        wanted = None if picked is None else {spec_dofs[i] for i in picked}
        stations = ([dof for dof in dict.fromkeys(spec_dofs)
                     if dof in drawn_by_dof] if measured is not None
                    else list(dict.fromkeys(spec_dofs)))
        if wanted is not None:
            stations = [dof for dof in stations if dof in wanted]
        station_rows = [(dof, autos.get(dof, spec_dofs.index(dof)))
                        for dof in stations]

    def logged(values):
        values = np.asarray(values, dtype=float)
        if not log_scaled:
            return values
        values = np.where(values > 0, values, np.nan)
        with np.errstate(invalid='ignore'):
            return np.log10(values)

    # The x axis's own transform, the same class flag every renderer
    # reads: an SRS lays natural frequency out in decades. Kept as a
    # function rather than applied to `spec_x`, because this module
    # works in two spaces at once — the exceedances are *judged* in
    # real frequency (`outside`, the log_interpolate of a limit onto
    # a record's own lines) and only *drawn* in the log one. The
    # measured curves arrive from `waterfall_arrays` already in
    # drawing space; `spread` is how everything of the
    # specification's joins them there. Mixing the two was found on
    # sight (Brandon, 2026-08-25): ribbons in decades over a band in
    # hertz put the whole measurement in the left margin of its own
    # requirement.
    log_x = bool(getattr(specification, 'log_abscissa', False))

    def spread(values):
        values = np.asarray(values, dtype=float)
        if not log_x:
            return values
        values = np.where(values > 0, values, np.nan)
        with np.errstate(invalid='ignore'):
            return np.log10(values)

    # A banded specification (octave bands: `bandwidth` set) is a
    # density per bin and draws flat across each — the 2-D plot's
    # stepMode and the waterfall's outline, `drawing_shape`'s one rule
    # — where this stage drew it as the line through its bin centers
    # (Brandon, 2026-09-18). The outline is for drawing only: the
    # exceedances are judged against the written limits on the
    # specification's own lines, as before.
    from ..plot import drawing_shape, step_outline

    # by the one drawing rule: a density per bin steps, whether its
    # bins are octave bands or a controller's lines (2026-09-19); a
    # breakpoint curve is the law through its points
    widths = (specification.bin_edges()
              if drawing_shape(specification) == 'steps' else None)

    def drawn(values):
        if widths is None:
            return np.asarray(values, dtype=float)
        return np.atleast_2d(step_outline(spec_x, values, widths)[1])[0]

    spec_x_drawn = (spec_x if widths is None
                    else step_outline(spec_x, np.zeros(len(spec_x)), widths)[0])

    out = []
    for dof, row in station_rows:
        # a cross term is complex; its magnitude is what the 2-D plot
        # and the waterfall draw, and the stage draws the same
        target = np.abs(np.asarray(
            specification.display_ordinate(us, [row])[0]))
        limits = {}
        for name in specification.LIMITS:
            values = specification.display_limit(name, us)
            if values is None or not np.isfinite(values[row]).any():
                continue
            limits[name] = np.real(np.asarray(values[row]))
        exceed = []
        for curve in drawn_by_dof.get(dof, []):
            for bound, over in (('abort_upper', True),
                                ('abort_lower', False)):
                written = limits.get(bound)
                if written is None:
                    continue
                mask = outside(curve['full_x'], curve['full'],
                               spec_x, written, over)
                if mask.any():
                    exceed.append({'x': curve['full_x'],
                                   'out': mask, 'bound': bound,
                                   'written': written, 'over': over})
        out.append({'dof': dof, 'row': row, 'target': logged(drawn(target)),
                    'cross': references is not None
                    and str(references[row]) != spec_dofs[row],
                    'limits': {name: logged(drawn(values))
                               for name, values in limits.items()},
                    'raw_limits': limits,
                    'curves': [{'x': c['x'], 'z': c['z'],
                                'label': c['label']}
                               for c in drawn_by_dof.get(dof, [])],
                    'exceed': exceed})
    if zlabel is None:
        word = getattr(specification, 'ordinate_dim', [''])
        word = word[0] if word else ''
        zlabel = (f'log10 {word}' if log_scaled else word) or 'level'
    return {'stations': out, 'spec_x': spec_x, 'spec_x_drawn': spec_x_drawn,
            'log_scaled': log_scaled,
            'xlabel': xlabel, 'zlabel': zlabel,
            'compared': measured is not None, 'left_out': left_out,
            'logged': logged, 'spread': spread}


@one_render
def add_banded_stage(plotter: Any, specification: Any,
                     measured: Any = None, records: Any = None,
                     unit_system: UnitSystem | None = None,
                     theme: Any = None,
                     budget: int = POINT_BUDGET,
                     specification_records: Any = None
                     ) -> dict[str, Any]:
    """Draw the stage into a plotter. The arguments are those of
    `banded_stage_arrays`.

    Alone, every channel's requirement is drawn in the 2-D plot's
    response-curve color, with its warning and abort zones. Compared,
    the requirements are drawn in the specification gray and each
    station's measured curves in a color of their own, as the 2-D plot
    pairs them, with the exceedance rectangles over their own bins. A
    log abscissa is labeled in decades (`marks.add_decade_axis`).

    Returns the dict from `banded_stage_arrays` with ``points`` (the
    vertices drawn) added and, when there are stations, ``extents``
    ((x0, x1, z0, z1), the real ranges the stage stands for, which a
    preview drawn over it normalizes against), and for a log abscissa
    ``decades``, the decade labels drawn.
    """
    import pyvista as pv

    from ..plot import bin_edges

    colors = resolve_theme(theme)
    arrays = banded_stage_arrays(specification, measured, records,
                                 unit_system, budget,
                                 specification_records)
    stations = arrays['stations']
    if not stations:
        arrays['points'] = 0
        return arrays
    sx, sy, sz = STAGE
    spec_x = arrays['spec_x']
    logged = arrays['logged']
    spread = arrays['spread']
    compared = arrays['compared']

    # the targets and limits are drawn along `spec_x_drawn` — the
    # lines, or a banded specification's bin edges — while the
    # exceedances below are judged against `spec_x`
    drawn_x = arrays['spec_x_drawn']
    zs = ([s['target'] for s in stations]
          + [v for s in stations for v in s['limits'].values()]
          + [c['z'] for s in stations for c in s['curves']])
    finite_z = [np.asarray(z)[np.isfinite(z)] for z in zs]
    finite_z = [z for z in finite_z if len(z)]
    # where something is drawn, as the waterfall spans (`drawn_span`):
    # a target written to zero outside its band drew 50 to 1500 Hz as a
    # sliver of a 0 to 2048 Hz floor (Brandon, 2026-10-12)
    across = spread(drawn_x)
    lines = across[np.isfinite(across)]
    x0, x1 = drawn_span(
        [(across, s['target']) for s in stations]
        + [(across, v) for s in stations for v in s['limits'].values()]
        + [(c['x'], c['z']) for s in stations for c in s['curves']],
        default=(float(lines.min()), float(lines.max())))
    z0 = min((float(z.min()) for z in finite_z), default=0.0)
    z1 = max((float(z.max()) for z in finite_z), default=1.0)
    xspan = (x1 - x0) or 1.0
    zspan = (z1 - z0) or 1.0

    # the stage's own extents, for whatever is drawn over it after —
    # the octave preview normalizes its steps against these, the
    # waterfall's rule (`info['extents']`), so the banding lands on
    # the bands it was read from rather than nowhere (2026-09-18)
    arrays['extents'] = (x0, x1, z0, z1)

    def nx(values):
        return (np.asarray(values, dtype=float) - x0) / xspan * sx

    def nz(values):
        return (np.asarray(values, dtype=float) - z0) / zspan * sz

    total = 0
    label_spots, label_names = [], []
    spec_xn = nx(spread(drawn_x))
    for k, station in enumerate(stations):
        y = (k / max(len(stations) - 1, 1)) * sy
        limits = station['limits']

        def level(name, limits=limits):
            values = limits.get(name)
            if values is None:
                return None
            values = nz(values)
            return values if np.isfinite(values).any() else None

        zones = limit_zones(level, len(spec_xn), sz)
        for j, (inner, outer, key) in enumerate(zones):
            good = np.isfinite(inner) & np.isfinite(outer)
            if good.sum() < 2:
                continue
            plotter.add_mesh(
                ribbon(spec_xn[good], y, inner[good], outer[good]),
                color=colors[key], opacity=ZONE_OPACITY,
                name=f'banded-{k}-zone-{j}')

        target = station['target']
        good = np.isfinite(target)
        if good.sum() >= 2:
            # alone, the requirement wears the 2-D reading's own pen —
            # the flat plot draws it in response_curve, and a stage
            # that recolored it blue read as a different object
            # (Brandon, 2026-08-23); compared, it stands back in gray
            # behind the tone-colored measurements, as the sine stage
            # does
            color = (colors['specification_curve'] if compared
                     else colors['response_curve'])
            plotter.add_mesh(
                pv.lines_from_points(np.column_stack(
                    [spec_xn[good], np.full(int(good.sum()), y),
                     nz(station['target'])[good]]).astype(np.float32)),
                color=color, line_width=2 if compared else 4,
                name=f'banded-{k}-target')
            total += int(good.sum())

        for m, curve in enumerate(station['curves']):
            good = np.isfinite(curve['z'])
            if good.sum() < 2:
                continue
            plotter.add_mesh(
                pv.lines_from_points(np.column_stack(
                    [nx(curve['x'])[good],
                     np.full(int(good.sum()), y),
                     nz(curve['z'])[good]]).astype(np.float32)),
                color=_TONE_COLORS[k % len(_TONE_COLORS)],
                line_width=4, name=f'banded-{k}-measured-{m}')
            total += int(good.sum())

        for m, hit in enumerate(station['exceed']):
            from ..core.compliance import log_interpolate

            edges = nx(np.clip(spread(bin_edges(hit['x'])), x0, x1))
            limit_full = logged(log_interpolate(
                hit['x'], spec_x, hit['written']))
            z_limit = nz(limit_full)
            z_edge = sz if hit['over'] else 0.0
            key = 'exceed_over' if hit['over'] else 'exceed_under'
            hits = np.flatnonzero(hit['out'] & np.isfinite(z_limit))
            if not len(hits):
                continue
            corners, faces = [], []
            for q, line in enumerate(hits):
                corners.extend([
                    [edges[line], y, z_limit[line]],
                    [edges[line + 1], y, z_limit[line]],
                    [edges[line + 1], y, z_edge],
                    [edges[line], y, z_edge]])
                faces.append([4, 4 * q, 4 * q + 1, 4 * q + 2,
                              4 * q + 3])
            plotter.add_mesh(
                pv.PolyData(np.asarray(corners, dtype=np.float32),
                            faces=np.concatenate(faces)),
                color=colors[key], opacity=EXCEED_OPACITY,
                name=f'banded-{k}-{m}-{key}')

        anchor = station['target']
        start_z = (nz(anchor)[np.isfinite(anchor)][0]
                   if np.isfinite(anchor).any() else 0.0)
        label_spots.append([spec_xn[0] - 0.03 * sx, y,
                            start_z + 0.04 * sz])
        label_names.append(station['dof'])

    if label_names:
        plotter.add_point_labels(
            np.asarray(label_spots), label_names, font_size=12,
            always_visible=True, text_color=colors['scene_text'],
            shape=None, fill_shape=False, show_points=False,
            name='banded-labels')
    if total:
        plotter.show_bounds(
            axes_ranges=(x0, x1, 0, max(len(stations) - 1, 1), z0, z1),
            fmt='%.4g',
            xtitle=arrays['xlabel'], ytitle=' ',
            ztitle=arrays['zlabel'], show_ylabels=False,
            grid='back', location='outer', use_3d_text=False,
            color=colors['scene_text'])
        if getattr(specification, 'log_abscissa', False):
            from .marks import add_decade_axis

            arrays['decades'] = add_decade_axis(
                plotter, (x0, x1, z0, z1), theme=theme)['labels']
    arrays['points'] = total
    return arrays


def plot_banded_stage(specification: Any, measured: Any = None,
                      records: Any = None, *,
                      screenshot: str | None = None,
                      unit_system: UnitSystem | None = None,
                      theme: Any = None, show: bool = True) -> Any:
    """Show the stage interactively, or render it to `screenshot`: the
    script version of the 3-D specification and comparison views the
    plot bar offers for every banded spectrum. `measured` and `records`
    are those of `banded_stage_arrays`. Returns the image array when
    rendering to a file, and otherwise the plotter (`finish_scene`)."""
    import pyvista as pv

    colors = resolve_theme(theme)
    off_screen = screenshot is not None or not show
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background(colors['scene_background'],
                           top=colors['scene_background_top'])
    add_banded_stage(plotter, specification, measured, records,
                     unit_system, theme)
    place_camera(plotter)
    return finish_scene(plotter, screenshot, show)
