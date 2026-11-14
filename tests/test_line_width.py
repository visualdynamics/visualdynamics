"""How wide a geometry's lines are drawn: tracelines and bar elements.

Brandon, 2026-11-13: a test geometry of lines alone was hard to see
with a finite element model's mode shape drawn over the test's, at the
2 px they were drawn at. They are 3 px now (`LINE_WIDTH`), in a still
scene and in an animated shape alike, and a caller drawing fainter
context still gets the width it asks for.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.geometry import Geometry
from visualdynamics.deform import ShapeDeflection
from visualdynamics.viz.geometry import LINE_WIDTH, add_geometry


def lines_only() -> Geometry:
    """Four nodes joined by three bars: a geometry of lines, no surface."""
    return Geometry(node_id=[1, 2, 3, 4],
                    node_xyz=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                    elem_conn=[[1, 2], [2, 3], [3, 4]], elem_type=[21] * 3,
                    length_unit='m')


def line_widths(plotter) -> list[float]:
    """The width of every actor whose mesh holds lines."""
    return [actor.prop.line_width
            for actor in plotter.renderer.actors.values()
            if getattr(actor, 'mapper', None) is not None
            and actor.mapper.dataset.n_lines]


@pytest.fixture
def plotter():
    import pyvista as pv

    made = pv.Plotter(off_screen=True)
    yield made
    made.close()


def test_lines_are_drawn_three_pixels_wide(plotter):
    assert LINE_WIDTH == 3.0
    add_geometry(plotter, lines_only())
    assert line_widths(plotter) == [LINE_WIDTH]


def test_an_animated_shape_draws_its_lines_as_wide(plotter):
    """The path a mode shape takes, overlaid in one color the way a
    second geometry is told apart."""
    from visualdynamics.viz.animate import GeometryAnimator

    geometry = lines_only()
    deflection = ShapeDeflection(geometry, ['2Z+', '3Z+'],
                                 np.array([1.0, -1.0]))
    GeometryAnimator(plotter, geometry, deflection, color_override='#ff8000')
    assert line_widths(plotter) == [LINE_WIDTH]


def test_a_caller_asking_for_thinner_lines_gets_them(plotter):
    add_geometry(plotter, lines_only(), line_width=1.0)
    assert line_widths(plotter) == [1.0]
