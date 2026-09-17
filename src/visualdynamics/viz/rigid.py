"""The reference point of a rigid-body preview, drawn on the scene.

A small sphere with three axis ticks at the point the rotations pivot
on, so the pane's numbers have a place on the model. Sized from the
model, never from the point's coordinates; excluded from the scene's
bounds so it never resizes the axes, and unpickable so a click near it
goes to the geometry — the same rules the stage marks live by.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

#: the sphere's radius as a fraction of the model's size, and the
#: ticks' reach in radii
RADIUS_FRACTION = 0.015
TICK_RADII = 3.0

NAME = 'rigid-point'


def add_reference_point(plotter: Any, point: ArrayLike, model_size: float,
                        color: Any) -> None:
    """Draw the marker at `point`, in the scene's display units."""
    import pyvista as pv

    center = np.asarray(point, dtype=np.float64)
    radius = max(float(model_size) * RADIUS_FRACTION, 1e-9)
    reach = radius * TICK_RADII
    ends = np.array([center - reach * axis for axis in np.eye(3)]
                    + [center + reach * axis for axis in np.eye(3)])
    ticks = pv.PolyData(ends, lines=np.array([[2, 0, 3], [2, 1, 4],
                                              [2, 2, 5]]).ravel())
    for name, mesh, extra in (
            (NAME, pv.Sphere(radius=radius, center=center), {}),
            (NAME + '-axes', ticks, {'line_width': 2.0})):
        actor = plotter.add_mesh(mesh, color=color, name=name,
                                 render=False, pickable=False, **extra)
        actor.UseBoundsOff()

