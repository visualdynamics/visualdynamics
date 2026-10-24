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


def test_a_solids_insides_are_left_out_of_its_scene(tmp_path):
    """A block of bricks draws as its skin, so the nodes inside it are
    never drawn: the scene leaves them out with their share of each mode
    (2026-10-11 — the Frame-Wing's figure was 12.8 MB of them). Every
    node kept is on the skin, the faces point at the kept nodes, and a
    mode's motion is still each node's own."""
    from visualdynamics import mesh
    from visualdynamics.core.shapes import ShapeSet

    block = mesh.block((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), 0.5, 'b', unit=None)
    assert block.num_nodes == 125
    xyz = block.node_xyz
    dofs = [f'{int(n)}Z+' for n in block.node_id]
    shapes = ShapeSet([10.0], [0.01], dofs, [xyz[:, 0] + 1.0])
    _text, data = _payload(visualdynamics.export_html(tmp_path / 'b.html', geometry=block,
                                                      shapes=shapes))
    [scene] = data['blocks']
    points = np.array(scene['points'], dtype=float)
    assert len(points) == 125 - 27, 'the 3 x 3 x 3 inside left out'
    on_skin = np.any((points == points.min(axis=0)) | (points == points.max(axis=0)), axis=1)
    assert on_skin.all()
    used = {n for face in scene['faces'] for n in face['nodes']}
    assert used == set(range(len(points)))
    motion = np.array(scene['modes'][0]['real'], dtype=float)[:, 2]
    assert np.allclose(motion / motion.max(), (points[:, 0] + 1) / (points[:, 0].max() + 1))


def test_a_real_mode_leaves_out_its_imaginary_part_and_still_moves(qt_app, tmp_path):
    """Every normal mode of a model is real, and its imaginary zeros were
    half a scene's mode data (2026-10-11): a real mode carries `real`
    alone, a complex one both, and the page animates a mode without its
    imaginary part — the figure is drawn differently from one moment to
    the next."""
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl

    from visualdynamics import mesh
    from visualdynamics.core.shapes import ShapeSet

    plate = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, unit=None)
    dofs = [f'{n}Z+' for n in plate.node_id]
    x = plate.node_xyz[:, 0]
    shapes = ShapeSet([10.0, 20.0], [0.01, 0.01], dofs,
                      np.array([x + 1.0, (x + 1.0) * (1 + 1j)]))
    path = visualdynamics.export_html(tmp_path / 'm.html', geometry=plate, shapes=shapes)
    _text, data = _payload(path)
    real, complex_ = data['blocks'][0]['modes']
    assert 'imag' not in real and 'imag' in complex_
    view = web_view()
    view.resize(600, 400)
    view.load(QUrl.fromLocalFile(str(path)))
    view.show()
    probe = ("(() => { const c = document.querySelector('section canvas');"
             "if (!c || !c.width) return null;"
             "const now = c.toDataURL();"
             "window.seen = (window.seen || 0) + 1;"
             "if (!window.first) { window.first = now; return null; }"
             "if (now !== window.first) return 'moved';"
             "return window.seen > 40 ? 'still' : null; })()")
    try:
        got = web_read(view, probe)
    finally:
        web_close(view)
    assert got == 'moved'


def _workmanship():
    """The workmanship profile written at its four breakpoints (20 Hz
    0.01, 80 to 500 Hz 0.04, 2000 Hz 0.01 g²/Hz, stored in SI) with the
    default bands: a specification on its own, nothing measured."""
    from visualdynamics.core.author import SpecificationDraft

    levels = [0.01, 0.04, 0.04, 0.01]
    draft = SpecificationDraft(['1Z+'], ['acceleration'],
                               [20.0, 80.0, 500.0, 2000.0],
                               [[g * 9.80665 ** 2 for g in levels]])
    return draft.make('Workmanship')


def test_a_specification_alone_is_drawn_on_the_grid_its_channels_are(
        tmp_path):
    """A specification with limits and nothing measured beside it drew
    an empty plot. Its target was filled in as a power law on a grid of
    its own (the curve's `x`, 800 points), while the channels the page
    draws from, the target and its zones, stayed on the four
    breakpoints; the page puts the picked channel's values on the first
    curve and draws them against that curve's grid, so four values were
    drawn at the first four of 800 frequencies, 20 to 20.3 Hz, and the
    zones with them. Every array the page draws for a channel now has
    the length of the grid it is drawn on, and spans the band."""
    from visualdynamics.units import SYSTEMS

    _text, data = _payload(visualdynamics.export_html(
        tmp_path / 'spec.html', _workmanship(),
        unit_system=SYSTEMS['in-slinch-lbf-s (g)']))
    [block] = data['blocks']
    target = block['curves'][0]
    drawn = target['x'] or block['x']
    [channel] = block['channels']
    assert len(channel['y']) == len(drawn)
    assert channel['zones']
    for zone in channel['zones']:
        for edge in (zone['lower'], zone['upper']):
            assert edge is None or len(edge) == len(drawn)
    written = [f for f, v in zip(drawn, channel['y']) if v is not None]
    assert min(written) == pytest.approx(20.0)
    assert max(written) == pytest.approx(2000.0)
    # the power law, not a straight line between breakpoints: at 1 kHz,
    # halfway down the falling segment in decades, -3 dB/oct from 0.04
    k = int(np.argmin([abs(f - 1000.0) for f in drawn]))
    expected = np.log10(0.04 * (drawn[k] / 500.0) ** (np.log10(0.25) / np.log10(4.0)))
    assert channel['y'][k] == pytest.approx(expected, abs=2e-3)


def test_a_specification_alone_draws_across_its_band(qt_app, tmp_path):
    """In a browser the figure of a specification on its own is drawn
    across its band, where it was an empty frame: nearly every column of
    the plot carries the target, opaque, where the grid and the zones
    are faint."""
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl

    path = visualdynamics.export_html(tmp_path / 'spec.html', _workmanship(),
                                      theme='light')
    view = web_view()
    view.resize(800, 500)
    view.load(QUrl.fromLocalFile(str(path)))
    view.show()
    probe = ("(() => { const c = document.querySelector('section canvas');"
             "if (!c || !c.height || !c.dataset.view) return null;"
             "const r = c.width / c.clientWidth;"
             "const left = Math.ceil(64 * r) + 2, right = c.width - Math.ceil(12 * r) - 2;"
             "const top = Math.ceil(10 * r) + 2, bottom = c.height - Math.ceil(40 * r) - 2;"
             "const px = c.getContext('2d').getImageData(left, top, right - left,"
             "                                          bottom - top);"
             "const w = right - left, h = bottom - top; let inked = 0;"
             "for (let x = 0; x < w; x++) { for (let y = 0; y < h; y++) {"
             "  if (px.data[4 * (y * w + x) + 3] > 60) { inked++; break; } } }"
             "return JSON.stringify(inked / w); })()")
    try:
        inked = json.loads(web_read(view, probe))
    finally:
        web_close(view)
    assert inked > 0.9, f'{inked:.0%} of the plot columns carry ink'
