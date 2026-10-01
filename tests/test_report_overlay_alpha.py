"""The matched-pair overlay in a report is drawn the way the GUI draws it.

The basis is the set being looked *at*; the other is drawn *through* it,
because a finite element model has a skin where a test set has a
wireframe and an opaque FEM simply hides the thing it is being compared
against. The comparison screen has always done that. The report drew both
sets solid, so the near mesh covered the far one — a difference nobody
could see until the two were put side by side.

The opacity is one constant now, in `theme`, read by the window and by
the report's canvas alike.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.matches import MatchedModes
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.report import _overlay_block
from visualdynamics.theme import OVERLAY_ALPHA


def _geometry():
    geometry = visualdynamics.Geometry(
        node_id=[1, 2, 3], node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        length_unit='m')
    geometry.add_beams([1, 2, 3])
    return geometry


def _shapes(scale=1.0):
    return ShapeSet(np.array([10.0, 20.0]), np.array([0.01, 0.01]),
                    ['1Z+', '2Z+', '3Z+'],
                    scale * np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]))


@pytest.fixture
def objects():
    return {'Test Geometry': _geometry(), 'Test Modes': _shapes(),
            'FEM Geometry': _geometry(), 'FEM Modes': _shapes(2.0),
            'Matched Modes': MatchedModes(
                'Test Modes', 'FEM Modes', pairs=[[0, 0], [1, 1]],
                macs=[1.0, 1.0], first_geometry='Test Geometry',
                second_geometry='FEM Geometry')}


def _built(objects, links):
    from visualdynamics.units import DEFAULT_SYSTEM

    return _overlay_block({'kind': 'pairs', 'source': 'Matched Modes',
                           'caption': ''}, objects, DEFAULT_SYSTEM,
                          links=links)


def _alphas(block):
    """Every alpha the payload carries, by the side it belongs to."""
    first = {round(line[3], 3) for line in block['lines']
             if line[2] == '#4c92d9'}
    second = {round(line[3], 3) for line in block['lines']
              if line[2] == '#ff8c2b'}
    return first, second


def test_the_other_set_is_drawn_through_the_basis(objects):
    """The basis here is the first-named set, so the second is the one
    drawn through it."""
    links = [{'members': ['Test Modes', 'Test Geometry'], 'role': 'Basis'},
             {'members': ['FEM Modes', 'FEM Geometry'], 'role': 'FEM'}]
    first, second = _alphas(_built(objects, links))
    assert first == {1.0}
    assert second == {OVERLAY_ALPHA}
    assert OVERLAY_ALPHA < 1.0, 'or none of this means anything'


def test_which_set_is_translucent_follows_the_links(objects):
    """Which of the pair is the basis depends on the project, not on
    the order they were matched in — the same question the window asks
    before it builds the comparison scene."""
    links = [{'members': ['Test Modes', 'Test Geometry'], 'role': None},
             {'members': ['FEM Modes', 'FEM Geometry'], 'role': 'Basis'}]
    first, second = _alphas(_built(objects, links))
    assert first == {OVERLAY_ALPHA}, 'the non-basis set gives way'
    assert second == {1.0}


def test_with_no_basis_declared_the_first_set_stays_solid(objects):
    first, second = _alphas(_built(objects, []))
    assert first == {1.0}
    assert second == {OVERLAY_ALPHA}


def test_the_nodes_and_faces_carry_it_too(objects):
    """A translucent wireframe with solid dots on it is not translucent."""
    links = [{'members': ['FEM Modes', 'FEM Geometry'], 'role': 'Basis'}]
    block = _built(objects, links)
    assert len(block['node_alphas']) == len(block['points'])
    assert set(block['node_alphas']) == {OVERLAY_ALPHA, 1.0}
    for face in block['faces']:
        assert face['alpha'] in (OVERLAY_ALPHA, 1.0)


def test_an_ordinary_scene_says_nothing_about_opacity():
    """Only the overlay carries it; every other scene draws solid, and
    the canvas defaults to that when the key is absent."""
    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html

    project = {'Geometry': _geometry(), 'Shape Set': _shapes()}
    report = Report('R', [{'kind': 'scene', 'geometry': 'Geometry',
                           'shapes': 'Shape Set', 'caption': ''}])
    import json
    html = render_html(report, project)
    payload = json.loads(
        html.split('type="application/json">')[1].split('</script>')[0])
    scene = payload['blocks'][0]
    assert 'node_alphas' not in scene
    assert all(len(line) == 3 for line in scene['lines'])
