"""The sine specification on a 3-D stage: frequency, time, amplitude.

A sine sweep specification *is* three-dimensional (Brandon,
2026-08-22): each tone is a path through (time, frequency) with an
amplitude along it, and its warning and abort limits ride the same
path above and below. Flattened onto amplitude-versus-frequency the
timing disappears — which tones overlap, which cross, which are done
before the others begin — so the specification's own view is the
stage: frequency across, time receding, amplitude up, one polyline
per curve.

The same stage the waterfall stands on (`STAGE`, `place_camera`), so
the two 3-D readings feel like one room. Amplitudes draw in the
display unit system; a specification with undeclared units draws its
raw values and the axis says so by giving no unit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..theme import theme as resolve_theme
from ..units import DEFAULT_SYSTEM, UNKNOWN, UnitSystem
from ._quiet import one_render
from .waterfall import STAGE, finish_scene, place_camera, ribbon

#: matplotlib's tab10, the same cycle the 2-D curves use — one color
#: per tone, the limit curves in the zone colors whatever the tone
_TONE_COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')

#: how solid an exceedance stripe is — the 2-D's EXCEED_ALPHA (150 of
#: 255), stronger than the zone shading it sits inside
EXCEED_OPACITY = 150 / 255

#: how many points each tone's curves get — a straight sweep needs few,
#: a log sweep bends, and 400 draws either without weight
LINES = 400


def sine_stage_arrays(specification: Any, channel: int = 0,
                      unit_system: UnitSystem | None = None,
                      lines: int = LINES,
                      tones: Any = None) -> dict[str, Any]:
    """The curves of the stage, as plain arrays.

    One entry per tone: `t` (environment seconds), `f` (Hz), and a
    dict of amplitude curves — `'target'` always, each written limit
    beside it — in display units. Pure numpy, so a test holds the
    numbers without a scene.
    """
    us = unit_system or DEFAULT_SYSTEM
    dim = specification.ordinate_dim
    convert = (specification.ordinate_unit is not None
               and dim and dim != UNKNOWN)

    wanted = (None if tones is None else set(tones))
    built = []
    for tone in specification.tones:
        if wanted is not None and tone.name not in wanted:
            continue
        t, f = tone.trajectory(tone.duration() / lines)
        curves = {'target': tone.target(f)[:, channel]}
        for limit in tone.limits:
            curves[limit] = tone.target(f, curve=limit)[:, channel]
        if convert:
            curves = {name: us.from_si(values, dim)
                      for name, values in curves.items()}
        built.append({'name': tone.name, 't': t + tone.start_time,
                      'f': f, 'curves': curves})
    zlabel = (f'amplitude [{us.label_text(dim)}]' if convert
              else 'amplitude')
    return {'tones': built, 'zlabel': zlabel,
            'channel': str(specification.response_dof[channel])}


@one_render
def add_sine_specification(plotter: Any, specification: Any,
                           channel: int = 0,
                           unit_system: UnitSystem | None = None,
                           theme: Any = None,
                           lines: int = LINES,
                           tones: Any = None) -> dict[str, Any]:
    """The specification alone — `add_sine_stage` with no levels."""
    dof = (str(specification.response_dof[channel])
           if specification.response_dof else None)
    return add_sine_stage(plotter, specification=specification, dof=dof,
                          unit_system=unit_system, theme=theme,
                          lines=lines, tones=tones)


@one_render
def add_sine_stage(plotter: Any, specification: Any = None,
                   levels: Any = None, dof: str | None = None,
                   unit_system: UnitSystem | None = None,
                   theme: Any = None, lines: int = LINES,
                   tones: Any = None) -> dict[str, Any]:
    """Draw the specification's stage into a plotter.

    The stage takes either half or both: a specification's tones
    (target curves, one color per tone, the warning and abort limits
    as thin lines *and* as the translucent zone ribbons the 2-D
    comparison shades, with its fallbacks), and measured levels —
    each tone's extracted path in the tone's color, timed by its own
    per-line clock. With both drawn the targets step back to the
    specification gray the 2-D comparison uses, because the
    measurement is what is being looked at. A level's clock is
    aligned to the specification's (its onset to the tone's start
    time) when the tone is known, and left in recording seconds
    otherwise. Returns the arrays dict with ``points`` added — the
    vertex count a size test can pin.
    """
    import pyvista as pv

    colors = resolve_theme(theme)
    us = unit_system or DEFAULT_SYSTEM
    spec_tones = []
    zlabel = 'amplitude'
    spec_dim = getattr(specification, 'ordinate_dim', None)
    convert_spec = (specification is not None
                    and specification.ordinate_unit is not None
                    and spec_dim and spec_dim != UNKNOWN)
    if specification is not None:
        channel = 0
        if dof is not None and dof in specification.response_dof:
            channel = list(specification.response_dof).index(dof)
        arrays = sine_stage_arrays(specification, channel, unit_system,
                                   lines, tones)
        spec_tones = arrays['tones']
        zlabel = arrays['zlabel']
    else:
        arrays = {'tones': [], 'zlabel': zlabel, 'channel': dof or ''}

    starts = ({tone.name: tone.start_time
               for tone in specification.tones}
              if specification is not None else {})
    spec_by_name = ({tone.name: tone for tone in specification.tones}
                    if specification is not None else {})
    spec_channel = 0
    if (specification is not None and dof is not None
            and dof in specification.response_dof):
        spec_channel = list(specification.response_dof).index(dof)
    paths = []
    for level in (levels or []):
        if level.seconds is None:
            continue
        rows = [i for i, d in enumerate(level.response_dof)
                if dof is None or str(d) == dof]
        if not rows:
            continue
        amplitude = np.abs(np.asarray(
            level.display_ordinate(us, rows[:1])[0]))
        seconds = np.asarray(level.seconds, dtype=float)
        if level.tone in starts:
            seconds = seconds - level.onset + starts[level.tone]
        frequencies = np.asarray(level.abscissa, dtype=float)
        # the abort limits at this path's own frequencies, in the same
        # display units, so the exceedance marks land where the 2-D
        # comparison would box them
        aborts = {}
        tone = spec_by_name.get(level.tone)
        if tone is not None:
            for limit in ('abort_upper', 'abort_lower'):
                if limit not in tone.limits:
                    continue
                values = tone.target(frequencies,
                                     curve=limit)[:, spec_channel]
                if convert_spec:
                    values = us.from_si(values, spec_dim)
                aborts[limit] = values
        paths.append({'name': level.tone, 't': seconds,
                      'f': frequencies, 'amplitude': amplitude,
                      'aborts': aborts})
    arrays['levels'] = paths
    sx, sy, sz = STAGE

    finite = [values[np.isfinite(values)]
              for tone in spec_tones
              for values in tone['curves'].values()]
    finite += [path['amplitude'][np.isfinite(path['amplitude'])]
               for path in paths]
    finite = [values for values in finite if len(values)]
    everything = spec_tones + paths
    if not everything:
        arrays['points'] = 0
        return arrays
    t0 = min(float(entry['t'].min()) for entry in everything)
    t1 = max(float(entry['t'].max()) for entry in everything)
    f0 = min(float(entry['f'].min()) for entry in everything)
    f1 = max(float(entry['f'].max()) for entry in everything)
    z0 = min((float(values.min()) for values in finite), default=0.0)
    z1 = max((float(values.max()) for values in finite), default=1.0)
    # amplitude starts at zero unless a limit dips below it: height on
    # this stage is level, and a floor at the quietest limit would
    # draw the bottom of every band as the ground
    z0 = min(z0, 0.0)
    fspan = (f1 - f0) or 1.0
    tspan = (t1 - t0) or 1.0
    zspan = (z1 - z0) or 1.0

    total = 0
    label_spots, label_names = [], []
    for k, tone in enumerate(spec_tones):
        xn = (tone['f'] - f0) / fspan * sx
        yn = (tone['t'] - t0) / tspan * sy
        curves = tone['curves']

        def level(name, curves=curves):
            values = curves.get(name)
            if values is None or not np.isfinite(values).all():
                return None
            return (values - z0) / zspan * sz

        # the zones, exactly as the 2-D comparison shades them:
        # yellow between warning and abort, and past abort *which
        # way* — red above, blue below, the same over/under colors
        # every other mark uses
        zones = []
        for warning, abort, edge, beyond in (
                ('warning_upper', 'abort_upper', sz, 'exceed_over'),
                ('warning_lower', 'abort_lower', 0.0, 'exceed_under')):
            w, a = level(warning), level(abort)
            if w is not None:
                outer = a if a is not None else np.full(len(xn), edge)
                zones.append((w, outer, 'limit_warning'))
            if a is not None:
                zones.append((a, np.full(len(xn), edge), beyond))
        for j, (inner, outer, key) in enumerate(zones):
            plotter.add_mesh(
                ribbon(xn, yn, inner, outer), color=colors[key],
                opacity=0.22, name=f'sine-{k}-zone-{j}')

        for name, values in curves.items():
            good = np.isfinite(values)
            if not good.any():
                continue
            zn = (values - z0) / zspan * sz
            run = np.column_stack([xn[good], yn[good], zn[good]])
            if name == 'target':
                # with measurements on the stage the target steps back
                # to the specification gray; alone it carries the
                # tone's own color, exactly the 2-D pairing rule
                color = (colors['specification_curve'] if paths
                         else _TONE_COLORS[k % len(_TONE_COLORS)])
                width = 2 if paths else 4
            else:
                color = colors['limit_warning' if 'warning' in name
                               else 'limit_abort']
                width = 1
            plotter.add_mesh(
                pv.lines_from_points(run.astype(np.float32)),
                color=color, line_width=width,
                name=f'sine-{k}-{name}')
            total += int(good.sum())
        if not paths:
            first = np.column_stack([xn, yn]).astype(float)
            target = tone['curves']['target']
            start_z = ((target[0] - z0) / zspan * sz
                       if np.isfinite(target[0]) else 0.0)
            label_spots.append([first[0, 0], first[0, 1] - 0.03 * sy,
                                start_z + 0.04 * sz])
            label_names.append(tone['name'])

    order = {tone['name']: k for k, tone in enumerate(spec_tones)}
    for j, path in enumerate(paths):
        good = np.isfinite(path['amplitude'])
        if not good.any():
            continue
        xn = (path['f'][good] - f0) / fspan * sx
        yn = (path['t'][good] - t0) / tspan * sy
        zn = (path['amplitude'][good] - z0) / zspan * sz
        k = order.get(path['name'], j)
        plotter.add_mesh(
            pv.lines_from_points(
                np.column_stack([xn, yn, zn]).astype(np.float32)),
            color=_TONE_COLORS[k % len(_TONE_COLORS)], line_width=4,
            name=f'sine-level-{j}')
        total += int(good.sum())
        # exceedances read exactly as the 2-D comparison draws them:
        # one rectangle per spectral line that went outside, the width
        # of that line's own bin along the sweep, from the abort limit
        # to the stage's edge — red to the ceiling, blue to the floor.
        # Rectangles, not a ribbon: a fill following the path point to
        # point drew wedges at every on/off transition. Stronger than
        # the zone shading it sits in, for the 2-D's own reason.
        amplitude = path['amplitude'][good]
        # bin edges along the path, in stage coordinates: midpoints
        # between neighboring lines, the ends extended half a bin
        def edges_of(values):
            mids = (values[:-1] + values[1:]) / 2.0
            first = values[0] - (mids[0] - values[0])
            last = values[-1] + (values[-1] - mids[-1])
            return np.concatenate([[first], mids, [last]])

        ex, ey = edges_of(xn), edges_of(yn)
        for limit, key, over in (('abort_upper', 'exceed_over', True),
                                 ('abort_lower', 'exceed_under',
                                  False)):
            edge = path['aborts'].get(limit)
            if edge is None:
                continue
            edge = np.asarray(edge)[good]
            finite = np.isfinite(edge)
            out = finite & (amplitude > edge if over
                            else amplitude < edge)
            if not out.any():
                continue
            hits = np.flatnonzero(out)
            z_limit = (edge[hits] - z0) / zspan * sz
            z_edge = sz if over else 0.0
            corners = []
            for m, k in enumerate(hits):
                corners.extend([
                    [ex[k], ey[k], z_limit[m]],
                    [ex[k + 1], ey[k + 1], z_limit[m]],
                    [ex[k + 1], ey[k + 1], z_edge],
                    [ex[k], ey[k], z_edge]])
            base = 4 * np.arange(len(hits))[:, None]
            faces = np.column_stack([
                np.full(len(hits), 4),
                base + 0, base + 1, base + 2, base + 3]).ravel()
            plotter.add_mesh(
                pv.PolyData(np.asarray(corners, dtype=np.float32),
                            faces=faces),
                color=colors[key], opacity=EXCEED_OPACITY,
                name=f'sine-level-{j}-{key}')
        label_spots.append([xn[0], yn[0] - 0.03 * sy,
                            zn[0] + 0.04 * sz])
        label_names.append(path['name'])

    if label_names:
        plotter.add_point_labels(
            np.asarray(label_spots), label_names, font_size=12,
            always_visible=True, text_color=colors['scene_text'],
            shape=None, fill_shape=False, show_points=False,
            name='sine-labels')
    if total:
        plotter.show_bounds(
            axes_ranges=(f0, f1, t0, t1, z0, z1),
            fmt='%.4g',
            xtitle='frequency [Hz]', ytitle='time [s]',
            ztitle=arrays['zlabel'],
            grid='back', location='outer', use_3d_text=False,
            color=colors['scene_text'])
    arrays['points'] = total
    return arrays


def sine_specification_scene(specification: Any, channel: int = 0,
                             unit_system: UnitSystem | None = None,
                             plotter: Any = None,
                             off_screen: bool = False,
                             theme: Any = None) -> Any:
    """Build (or add to) a PyVista plotter showing the stage.

    Same shape as `waterfall_scene`: returns the plotter — `.show()`
    it, or `.screenshot()` if off_screen.
    """
    import pyvista as pv

    colors = resolve_theme(theme)
    if plotter is None:
        plotter = pv.Plotter(off_screen=off_screen)
        plotter.set_background(colors['scene_background'],
                               top=colors['scene_background_top'])
    add_sine_specification(plotter, specification, channel,
                           unit_system, theme)
    place_camera(plotter)
    return plotter


def plot_sine_specification(specification: Any, channel: int = 0, *,
                            screenshot: str | None = None,
                            unit_system: UnitSystem | None = None,
                            theme: Any = None, show: bool = True) -> Any:
    """Show the stage interactively, or render it to `screenshot` —
    the scriptable face of the plot bar's 3-D specification view."""
    off_screen = screenshot is not None or not show
    plotter = sine_specification_scene(
        specification, channel, unit_system,
        off_screen=off_screen, theme=theme)
    return finish_scene(plotter, screenshot, show)
