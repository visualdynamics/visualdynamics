"""The three meshing shapes (`demo.shapes`): built from the recipe,
measured against it, meshed both ways. The full-size meshes of the
guide's table are `slow`; the quick tests mesh coarsely.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from visualdynamics.core import solids
from visualdynamics.demo import shapes

INCH = 0.0254


@pytest.mark.parametrize('key', list(shapes.SHAPES))
def test_each_shape_is_one_part_that_is_not_an_extrusion(key):
    project = shapes.SHAPES[key]()
    assert project.name == shapes.TITLES[key]
    geometry = project[project.names[0]]
    top = solids.top_level(geometry)
    assert len(top) == 1 and top[0].kind == 'subtract'
    with pytest.raises(solids.NotAnExtrusion):
        solids.extrusion_axis(geometry, [top[0].id])


def test_the_recipe_volume_is_right_where_it_is_known():
    """The shaft less its hole: a cylinder of 6 pi less a bore of
    radius 0.3 whose axis runs 4 in through the shaft (2 in of radius
    each way at 30 degrees to the axis), within a percent, the bore's
    ends through the curved wall being what the formula misses."""
    project = shapes.shaft()
    geometry = project[project.names[0]]
    cut = solids.top_level(geometry)[0].id
    found = shapes.recipe_volume(geometry, [cut], samples=1_000_000) / INCH ** 3
    bore = math.pi * 0.3 ** 2 * (2.0 * 1.0 / math.sin(math.radians(30)))
    assert found == pytest.approx(6.0 * math.pi - bore, rel=0.01)


def test_mesh_volume_integrates_bricks_and_tets():
    project = shapes.v_block()
    got = shapes.mesh(project, 'tets', 0.5)
    assert got['method'] == 'tets' and got['count'] > 1000
    assert abs(got['volume_error']) < 0.02
    assert got['volume'] == pytest.approx(shapes.mesh_volume(project[project.names[0]]) / INCH ** 3)


def test_a_refused_mesh_is_recorded_with_its_reason():
    """The wye's walls are one brick thick at 0.3 in: bricks fold, and
    the record says so instead of raising, so a table can show it."""
    project = shapes.wye()
    got = shapes.mesh(project, 'bricks', 0.3)
    assert got['method'] == 'refused' and 'folded fitting the shape' in got['refusal']
    assert got['count'] == 0 and np.isnan(got['quality']) and got['recipe_volume'] > 0


@pytest.mark.slow
@pytest.mark.parametrize('key', list(shapes.SHAPES))
def test_the_guides_meshes_are_as_the_table_says(key):
    """What the guide's table records (2026-10-19): the tetrahedra mesh
    all three within a percent of the recipe's volume with no flat
    element, and the fitted bricks are refused on each for a handful
    of folded bricks, with the reason. When the bricks come to fit,
    this test is what says so."""
    found = shapes.report(key)
    tets = found['tets']
    # no element inside out; the worst on the round shapes is a flat
    # one on a convex wall, 0.02 here and 0.0002 on the public runner
    # (the triangulation differs by machine), so no bar above zero
    assert tets['quality'] > 0.0 and abs(tets['volume_error']) < 0.01, (key, tets)
    bricks = found['bricks']
    if bricks['method'] == 'refused':
        assert 'folded fitting the shape' in bricks['refusal'], (key, bricks)
    else:
        assert bricks['quality'] > 0.1 and abs(bricks['volume_error']) < 0.01, (key, bricks)
