"""The scalogram as a surface: time across, frequency back, amplitude up.

The flat picture puts magnitude in colour, which is the honest way to
draw three quantities on two axes and is also the way a level is hardest
to read — a viridis square is a number only as accurately as the eye can
match it against a bar. A surface puts the third quantity on a third
axis, where a ridge is a ridge and a floor is flat, and keeps the colour
as well, so the same value is said twice.

That is principle 1's rule, and it is why 3-D is what this reading opens
in (Brandon, 2026-08-27): where a reading has a natural 3-D form, it
gets one, and a scalogram's is about as natural as they come — it is
already a function of two variables.

The depth axis is **log frequency**, spaced as the transform's own rows
are: a wavelet's bandwidth is a constant fraction of its centre
frequency, so the rows are evenly spaced by octave and drawing them
evenly spaced in linear frequency would put the surface's own rows where
they do not belong. The axis is labelled back in Hz.

The cone of influence is drawn here too, as a translucent wall standing
where the edges reach in. On the flat picture it is a veil over the
suspect region; here it is a surface the ridge passes behind, which is
the same statement in the geometry the eye is already reading.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import wavelet
from ..theme import theme as resolve_theme
from ._quiet import one_render
from .waterfall import STAGE


@one_render
def add_scalogram(plotter: Any, magnitude: np.ndarray, times: np.ndarray,
                  frequencies: np.ndarray, *, theme: Any = None,
                  omega0: float = wavelet.OMEGA0,
                  time_label: str = 'time [s]',
                  level_label: str = 'amplitude') -> dict[str, Any]:
    """Draw one channel's scalogram as a surface, and say what it stands for.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Cleared by the caller, as the waterfall's is.
    magnitude : numpy.ndarray
        ``(frequencies, times)``. Already magnitude: this draws.
    times : numpy.ndarray
        Seconds, one per column.
    frequencies : numpy.ndarray
        Hz, one per row, ascending and spaced evenly by octave.
    theme : optional
        The theme name or mapping.
    omega0 : float
        The wavelet's width, for the cone.
    time_label, level_label : str
        Axis titles, units included.

    Returns
    -------
    dict
        ``extents`` as ``(t0, t1, f0, f1, z0, z1)`` in real units — the
        ranges the unit stage stands in for, so anything drawn after
        this can land at a data position — and ``points``, the vertex
        count, which is what a size-ceiling test pins.
    """
    import pyvista as pv

    colors = resolve_theme(theme)
    sx, sy, sz = STAGE

    values = np.asarray(magnitude, dtype=float)
    rows = np.log10(np.asarray(frequencies, dtype=float))
    clock = np.asarray(times, dtype=float)

    t0, t1 = float(clock[0]), float(clock[-1])
    f0, f1 = float(rows[0]), float(rows[-1])
    finite = np.isfinite(values)
    z0 = 0.0
    z1 = float(values[finite].max()) if finite.any() else 1.0
    if z1 <= z0:
        z1 = z0 + 1.0

    # the stage is a unit box whatever the data's ranges, exactly as the
    # waterfall's is, so a scene does not change shape with its units
    xn = (clock - t0) / (t1 - t0) * sx if t1 > t0 else np.zeros_like(clock)
    yn = (rows - f0) / (f1 - f0) * sy if f1 > f0 else np.zeros_like(rows)
    zn = (values - z0) / (z1 - z0) * sz

    # x fastest, which is the order a StructuredGrid of dimensions
    # (nx, ny, 1) reads its points in. `meshgrid` gives (ny, nx) arrays
    # and `values` is already (frequencies, times) = (ny, nx), so a
    # plain C-order ravel of all three agrees. Getting this wrong does
    # not fail — it draws a surface, fanned and spiked, that looks like
    # a scalogram of something.
    mesh_x, mesh_y = np.meshgrid(xn, yn)
    grid = pv.StructuredGrid()
    grid.points = np.column_stack([mesh_x.ravel(), mesh_y.ravel(),
                                   zn.ravel()]).astype(np.float32)
    grid.dimensions = (len(clock), len(rows), 1)
    grid.point_data['level'] = values.ravel()
    plotter.add_mesh(grid, scalars='level', cmap='viridis',
                     clim=(z0, z1), show_scalar_bar=False,
                     name='scalogram')

    _add_cone(plotter, clock, frequencies, rows, colors, omega0,
              sx, sy, sz, t0, t1, f0, f1)
    _label_frequencies(plotter, frequencies, rows, colors, sy, f0, f1)

    # **Last**, and that is not a style choice: every actor added after
    # `show_bounds` refits the cube axes and rewrites their labelled
    # ranges back to the raw bounds, so a scene built in the other order
    # reads its stage coordinates out as seconds — 0 to 1.6 s on a
    # four-second record (STATUS.md records the same trap for the
    # stage's marks). The cone and the frequency labels go on first.
    #
    # The depth axis keeps its grid and loses its numbers, the way the
    # waterfall's does: `axes_ranges` interpolates labels *linearly*
    # between the ends, and this axis is log-spaced, so its numbers
    # would be wrong everywhere except at the two corners — a reader
    # would take a ridge at 100 Hz for one at 210.
    plotter.show_bounds(
        axes_ranges=(t0, t1, 0.0, 1.0, z0, z1),
        xtitle=time_label, ytitle='frequency [Hz]', ztitle=level_label,
        show_ylabels=False, grid='back', location='outer',
        use_3d_text=False, color=colors['scene_text'])
    return {'extents': (t0, t1, float(frequencies[0]),
                        float(frequencies[-1]), z0, z1),
            'points': int(grid.n_points)}


def _add_cone(plotter, clock, frequencies, rows, colors, omega0,
              sx, sy, sz, t0, t1, f0, f1) -> None:
    """Two translucent walls where the record's ends reach in.

    Drawn rather than left to be remembered, for the same reason the
    flat picture shades it: inside the cone the surface is a picture of
    where the record was cut, and it rises and falls exactly as if it
    were a picture of the record.
    """
    import pyvista as pv

    reach = wavelet.cone_of_influence(frequencies, 1.0, omega0)
    if f1 <= f0 or t1 <= t0:
        return
    yn = (rows - f0) / (f1 - f0) * sy

    for side in ('left', 'right'):
        edge = (np.minimum(t0 + reach, t1) if side == 'left'
                else np.maximum(t1 - reach, t0))
        xn = (edge - t0) / (t1 - t0) * sx
        # a wall from the floor to the stage's ceiling at each row
        low = np.column_stack([xn, yn, np.zeros_like(xn)])
        high = np.column_stack([xn, yn, np.full_like(xn, sz)])
        points = np.vstack([low, high])
        count = len(xn)
        faces = []
        for i in range(count - 1):
            faces.extend([4, i, i + 1, count + i + 1, count + i])
        if not faces:
            continue
        wall = pv.PolyData(points.astype(np.float32),
                           faces=np.asarray(faces))
        plotter.add_mesh(wall, color=colors['scene_text'], opacity=0.18,
                         show_scalar_bar=False, name=f'cone-{side}')


def _label_frequencies(plotter, frequencies, rows, colors, sy, f0, f1) -> None:
    """Hz along the depth axis, at the values a person would choose.

    1, 2, 5 per decade, placed where they really are on a log axis
    rather than spread evenly along it. The axis's own numbers are off
    (`show_ylabels=False` above) because they would be spread evenly,
    which is the same axis saying two different things.
    """
    import numpy as np

    if f1 <= f0:
        return
    spots, names = [], []
    for value in wavelet.decade_values(frequencies[0], frequencies[-1]):
        station = (np.log10(value) - f0) / (f1 - f0) * sy
        spots.append([-0.04, float(station), 0.0])
        names.append(f'{value:g}')
    if names:
        actor = plotter.add_point_labels(
            np.asarray(spots), names, font_size=11, always_visible=True,
            text_color=colors['scene_text'], shape=None, fill_shape=False,
            show_points=False, name='scalogram-frequencies')
        # out of the bounds the axes fit themselves to: these sit just
        # off the edge of the stage, and counting them would stretch the
        # box a little every time one was drawn
        try:
            actor.GetProperty()
            actor.UseBoundsOff()
        except AttributeError:                       # pragma: no cover
            pass
