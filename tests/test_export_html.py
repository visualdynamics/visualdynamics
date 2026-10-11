"""One figure as one interactive HTML file (`visualdynamics.export_html`).

For an interactive slide deck beside a paper's printed figures (Brandon,
2026-09-27): the report's own figure — zoom, pan and readout on a plot,
turning and animation on a scene — alone in a file that opens offline
and, with `fill`, takes whatever frame holds it.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from conftest import web_close, web_read, web_view

import visualdynamics
from visualdynamics.core.data import Psd, Specification


def _pair():
    f = np.linspace(20.0, 2000.0, 400)
    level = np.full((1, f.size), 1e-3)
    spec = Specification(f, level, response_dof=['101Z+'],
                         ordinate_dim=['acceleration**2/frequency'],
                         warning_upper=level * 2, warning_lower=level / 2,
                         abort_upper=level * 4, abort_lower=level / 4)
    measured = Psd(f, level * 1.2, response_dof=['101Z+'],
                   reference_dof=['101Z+'],
                   ordinate_dim=['acceleration**2/frequency'])
    return measured, spec


def _payload(path):
    with open(path, encoding='utf-8') as page:
        text = page.read()
    found = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                      text, re.DOTALL)
    return text, json.loads(found.group(1))


def test_a_comparison_is_one_self_contained_file(tmp_path):
    measured, spec = _pair()
    path = visualdynamics.export_html(tmp_path / 'spec.html', measured,
                                      specification=spec, channel='101Z+',
                                      theme='light')
    text, data = _payload(path)
    assert not re.search(r'(src|href)="https?://', text), 'opens offline'
    assert data['fill'] is True and data['theme'] == 'light'
    assert data['marking'] == '' and data['title'] == ''
    [block] = data['blocks']
    assert block['kind'] == 'plot' and block['curves'], 'the one figure, drawn'


def test_a_scene_takes_its_shapes(tmp_path):
    from visualdynamics import mesh
    from visualdynamics.core.shapes import ShapeSet

    plate = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, unit=None)
    dofs = [f'{n}Z+' for n in plate.node_id]
    shapes = ShapeSet([10.0], [0.01], dofs, np.ones((1, len(dofs))))
    path = visualdynamics.export_html(tmp_path / 'm.html', geometry=plate,
                                      shapes=shapes)
    _text, data = _payload(path)
    [block] = data['blocks']
    assert block['kind'] == 'scene' and len(block['modes']) == 1
    assert data['theme'] is None, 'unset, it follows the reader'


def test_a_brick_is_drawn_as_its_six_faces_and_a_block_as_its_skin(tmp_path):
    """The HTML scene draws an element as the 3-D view does. It drew
    every element of three nodes or more as one polygon in node order,
    and a brick's eight nodes in that order fold into a bow tie: the
    website's brick examples drew no cubes at all (Brandon, 2026-10-11).
    One brick is six square faces of a unit cube; two bricks sharing a
    face are the ten faces of their skin, the shared one gone."""
    from visualdynamics import mesh

    brick = mesh.block((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1, unit=None)
    _text, data = _payload(visualdynamics.export_html(tmp_path / 'b.html',
                                                      geometry=brick))
    [block] = data['blocks']
    points = np.asarray(block['points'], dtype=float)
    faces = [points[face['nodes']] for face in block['faces']]
    assert len(faces) == 6 and all(len(face) == 4 for face in faces)
    for face in faces:
        # four corners of the cube, all on one side of it, going round
        flat = [axis for axis in range(3) if np.ptp(face[:, axis]) == 0]
        assert len(flat) == 1 and face[0, flat[0]] in (0.0, 1.0)
        sides = np.linalg.norm(np.roll(face, -1, axis=0) - face, axis=1)
        assert np.allclose(sides, 1.0), 'edges, not diagonals'
    two = mesh.block((0, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 1), 1, unit=None)
    _text, data = _payload(visualdynamics.export_html(tmp_path / 't.html',
                                                      geometry=two))
    assert len(data['blocks'][0]['faces']) == 10


def test_a_higher_order_element_is_drawn_by_its_corners(tmp_path):
    """A quad8 is its four corners, not an eight-sided polygon through its
    midside nodes; a three-node beam is a line, not a triangle."""
    geometry = visualdynamics.Geometry(
        np.arange(1, 12),
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, 0, 0], [2, 1, 0],
         [1, 2, 0], [0, 1, 0], [5, 0, 0], [7, 0, 0], [6, 0, 0]])
    geometry.add_elements([list(range(1, 9)), [9, 10, 11]], [95, 23], [1, 2])
    _text, data = _payload(visualdynamics.export_html(tmp_path / 'q.html',
                                                      geometry=geometry))
    [block] = data['blocks']
    assert [face['nodes'] for face in block['faces']] == [[0, 1, 2, 3]]
    assert [line[:2] for line in block['lines']] == [[8, 9]]


def test_nothing_to_draw_is_refused(tmp_path):
    with pytest.raises(ValueError, match='data to plot or a geometry'):
        visualdynamics.export_html(tmp_path / 'x.html')


def test_a_report_page_does_not_fill(tmp_path):
    """The option is the export's; a report keeps its page."""
    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html

    text = render_html(Report('R', []), {})
    data = json.loads(re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        text, re.DOTALL).group(1))
    assert data['fill'] is False and data['theme'] is None


def test_the_figure_fills_its_frame(qt_app, tmp_path):
    """In a 600 by 300 frame the figure is the page: no title, no number,
    and the page as tall as the frame."""
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl

    measured, spec = _pair()
    path = visualdynamics.export_html(tmp_path / 'spec.html', measured,
                                      specification=spec, channel='101Z+')
    view = web_view()
    view.resize(600, 300)
    view.load(QUrl.fromLocalFile(str(path)))
    view.show()
    probe = ("(() => { const c = document.querySelector('section canvas');"
             "if (!c || !c.height) return '';"
             "const shown = e => e && getComputedStyle(e).display !== 'none';"
             "return JSON.stringify({"
             " body: document.body.getBoundingClientRect().height,"
             " inner: window.innerHeight,"
             " title: shown(document.querySelector('h1')),"
             " number: shown(document.querySelector('.caption strong'))}); })()")
    try:
        got = json.loads(web_read(view, probe))
    finally:
        web_close(view)
    assert not got['title'] and not got['number']
    assert abs(got['body'] - got['inner']) <= 2, got


def test_a_comparison_keys_its_shading_as_the_app_does(qt_app, tmp_path):
    """The report's plots always match the app's (Brandon, 2026-09-27):
    the legend names the warning band and the zones past abort in the
    same words, and the one response and its specification are drawn in
    the app's colors and at its widths."""
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl

    from visualdynamics.plot import (
        CURVE_WIDTH,
        MEASURED_ALPHA,
        MEASURED_WIDTH,
        STOOD_BACK_WIDTH,
    )

    measured, spec = _pair()
    path = visualdynamics.export_html(tmp_path / 'spec.html', measured,
                                      specification=spec, channel='101Z+')
    _text, data = _payload(path)
    [block] = data['blocks']
    assert block['channels'][0]['key'] == [
        ['warning', 'warning band (±3 dB)'],
        ['above', 'above abort (+6 dB)'], ['below', 'below abort (−6 dB)']]
    assert [c.get('width') for c in block['curves']] == [STOOD_BACK_WIDTH,
                                                        MEASURED_WIDTH]
    # and colored as the app colors them: the specification gray, the
    # response the first curve color, translucent over it — not the
    # page's ink, which could not be told from the gray in print
    gray, response = block['curves']
    assert gray.get('gray') and not response.get('ink')
    assert response.get('color') == 0
    assert response.get('alpha') == MEASURED_ALPHA
    assert data['curve_width'] == CURVE_WIDTH
    view = web_view()
    view.resize(700, 400)
    view.load(QUrl.fromLocalFile(str(path)))
    view.show()
    probe = ("(() => { const spans = document.querySelectorAll('.legend span');"
             "if (!spans.length) return '';"
             "return JSON.stringify(Array.from(spans, s => s.textContent)); })()")
    try:
        names = json.loads(web_read(view, probe))
    finally:
        web_close(view)
    assert names[-3:] == ['warning band (±3 dB)', 'above abort (+6 dB)',
                          'below abort (−6 dB)'], names
