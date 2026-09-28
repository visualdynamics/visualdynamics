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
from conftest import web_close, web_read

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
    from PySide6.QtWebEngineWidgets import QWebEngineView

    measured, spec = _pair()
    path = visualdynamics.export_html(tmp_path / 'spec.html', measured,
                                      specification=spec, channel='101Z+')
    view = QWebEngineView()
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
    from PySide6.QtWebEngineWidgets import QWebEngineView

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
    view = QWebEngineView()
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
