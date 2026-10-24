"""One figure as a data file and a script (`visualdynamics.export_figure`).

A page whose Content-Security-Policy allows no inline script and no
frames can show neither the one-file page nor an iframe of it. The
split export writes the figure's data as JSON and the viewers as one
same-origin script that mounts a figure on an element the page gives
it, and both exports can keep fewer channels and fewer points.
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pytest
from conftest import web_close, web_read, web_view

import visualdynamics
from visualdynamics.core.data import Psd, Specification
from visualdynamics.decimate import shared_peak_indices
from visualdynamics.report import page

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _pair(lines=400, dofs=('101Z+',), spike=None, dip=None):
    f = np.linspace(20.0, 2000.0, lines)
    level = np.full((len(dofs), f.size), 1e-3)
    spec = Specification(f, level, response_dof=list(dofs),
                         ordinate_dim=['acceleration**2/frequency'] * len(dofs),
                         warning_upper=level * 2, warning_lower=level / 2,
                         abort_upper=level * 4, abort_lower=level / 4)
    wobble = 1.0 + 0.2 * np.sin(np.arange(f.size) * 0.37)
    measured = level * 1.2 * wobble
    if spike is not None:
        measured[0, spike] = 9e-3
    if dip is not None:
        measured[0, dip] = 1e-4
    psd = Psd(f, measured, response_dof=list(dofs), reference_dof=list(dofs),
              ordinate_dim=['acceleration**2/frequency'] * len(dofs))
    return psd, spec


def _inline_payload(path):
    text = pathlib.Path(path).read_text(encoding='utf-8')
    found = re.search(r'<script id="data" type="application/json">(.*?)'
                      r'</script>', text, re.DOTALL)
    return json.loads(found.group(1))


def _arrays_of(block):
    """Every list of numbers in a plot block, by where it hangs."""
    out = {}

    def walk(node, where):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f'{where}.{key}')
        elif isinstance(node, list) and node and all(
                v is None or isinstance(v, (int, float)) for v in node):
            out[where] = node
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f'{where}[{i}]')
    walk(block, 'block')
    return out


def test_the_split_files_carry_the_one_file_figures_data(tmp_path):
    """The JSON is the very payload the HTML file carries inline, plus
    the format number its renderer checks."""
    psd, spec = _pair()
    html = visualdynamics.export_html(tmp_path / 'a.html', psd,
                                      specification=spec, channel='101Z+',
                                      theme='light', fill=False)
    data = visualdynamics.export_figure(tmp_path / 'a.json', psd,
                                        specification=spec, channel='101Z+',
                                        theme='light',
                                        renderer=tmp_path / 'r.js')
    split = json.loads(pathlib.Path(data).read_text(encoding='utf-8'))
    assert split.pop('format') == page.FIGURE_FORMAT
    assert split == _inline_payload(html)
    assert split['fill'] is False, 'a mounted figure keeps its own height'
    assert (tmp_path / 'r.js').exists()


def test_a_scene_splits_as_well(tmp_path):
    from visualdynamics import mesh
    from visualdynamics.core.shapes import ShapeSet

    plate = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, unit=None)
    dofs = [f'{n}Z+' for n in plate.node_id]
    shapes = ShapeSet([10.0], [0.01], dofs, np.ones((1, len(dofs))))
    html = visualdynamics.export_html(tmp_path / 'm.html', geometry=plate,
                                      shapes=shapes, fill=False)
    data = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'm.json', geometry=plate, shapes=shapes)).read_text())
    data.pop('format')
    assert data == _inline_payload(html)
    assert data['blocks'][0]['kind'] == 'scene'


def test_one_renderer_draws_every_figure(tmp_path):
    """The script does not depend on the figure: written beside two
    different figures it is the same bytes, and it reads its data from
    the caller, never from the page."""
    psd, spec = _pair()
    visualdynamics.export_figure(tmp_path / 'a.json', psd, specification=spec,
                                 renderer=tmp_path / 'a.js')
    visualdynamics.export_figure(tmp_path / 'b.json', psd, mode='curves',
                                 renderer=tmp_path / 'b.js')
    a = (tmp_path / 'a.js').read_text(encoding='utf-8')
    assert a == (tmp_path / 'b.js').read_text(encoding='utf-8')
    assert b'\r\n' not in (tmp_path / 'a.js').read_bytes(), \
        'the same bytes on every platform'
    assert 'window.VisualDynamicsFigure' in a
    assert f'const FORMAT = {page.FIGURE_FORMAT};' in a
    assert visualdynamics.__version__ in a
    assert "getElementById('data')" not in a and 'document.body' not in a


def test_the_renderer_carries_the_license_word_for_word():
    """It is a copy of the package's code that travels alone, so it
    carries the license at its head, and that is the repository's
    LICENSE, not a paraphrase of it."""
    license_text = (ROOT / 'LICENSE').read_text(encoding='utf-8')
    assert page._LICENSE_NOTICE.strip() == license_text.strip()
    js = page.renderer_js('0')
    assert js.index(page._LICENSE_NOTICE) < js.index('(function () {')


def test_the_viewers_reach_only_their_host():
    """One body of code draws both pages, and it names the page only
    through the bindings each page gives it, so a mounted figure never
    reaches into the host page around it."""
    assert page._JS == page._JS_PAGE + page._JS_BODY
    for page_wide in ('document.body', 'document.documentElement',
                      'document.getElementById', 'document.querySelector',
                      'window.innerHeight'):
        assert page_wide not in page._JS_BODY, page_wide


def test_every_rule_of_the_mounted_figure_is_scoped():
    """The figure's stylesheet styles nothing outside its element: the
    page's `:root`, `body` and `html.fill` rules become the element
    itself, and every other selector a descendant of it."""
    css = page.scoped_css(page._CSS)
    flat = re.sub(r'@media[^{]*\{', '', css)
    selectors = [s.strip() for rule in re.findall(r'([^{}]+)\{', flat)
                 for s in rule.split(',')]
    assert len(selectors) > 60
    outside = [s for s in selectors if not s.startswith(page.FIGURE_SCOPE)]
    assert outside == []
    assert '.vdfig { --ink: #1f2328;' in css
    assert '.vdfig.fill { padding: 0;' in css
    assert '.vdfig.marked .themetoggle' in css


@pytest.mark.parametrize('css, refused', [
    ('@font-face { font-family: x; }', 'at-rule'),
    ('html body { margin: 0; }', 'selector'),
    ('body > section { margin: 0; }', 'selector'),
])
def test_a_rule_that_cannot_be_scoped_is_refused(css, refused):
    with pytest.raises(ValueError, match=refused):
        page.scoped_css(css)


def test_shared_indices_keep_every_rows_extremes():
    rng = np.random.default_rng(3)
    rows = rng.standard_normal((3, 5000))
    rows[0, 1234] = 40.0
    rows[1, 4321] = -40.0
    rows[2, 2000:2100] = np.nan                # a gap stays a gap
    keep = shared_peak_indices(rows, 300)
    assert len(keep) <= 300 and np.all(np.diff(keep) > 0)
    assert keep[0] == 0 and keep[-1] == 4999, 'the extent is kept'
    assert 1234 in keep and 4321 in keep
    for row in rows:
        kept = row[keep]
        assert np.nanmax(kept) == np.nanmax(row)
        assert np.nanmin(kept) == np.nanmin(row)
    # every slice's extremes, not only the record's: the largest kept
    # point in each tenth of the record is that tenth's largest
    for start in range(0, 5000, 500):
        inside = keep[(keep >= start) & (keep < start + 500)]
        assert rows[0, inside].max() == rows[0, start:start + 500].max()


def test_shared_indices_leave_a_short_record_whole_and_refuse_too_few():
    assert np.array_equal(shared_peak_indices(np.ones((2, 50)), 64),
                          np.arange(50))
    with pytest.raises(ValueError, match='at least 8'):
        shared_peak_indices(np.ones((3, 500)), 7)


def test_a_trimmed_figure_keeps_its_peaks_and_one_grid(tmp_path):
    """Thinned to 200 points, the comparison keeps the spike and the dip
    a uniform stride would step over, every array on the plot's grid is
    still the grid's length, the stepped drawing becomes a line (its bin
    edges no longer describe the bins kept), and the figure says it was
    thinned."""
    psd, spec = _pair(lines=4000, spike=1234, dip=2711)
    full = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'full.json', psd, specification=spec,
        channel='101Z+')).read_text())['blocks'][0]
    trimmed = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'trim.json', psd, specification=spec, channel='101Z+',
        max_points=200)).read_text())['blocks'][0]
    assert full['steps'] and len(full['edges']) == 4001
    assert len(trimmed['x']) <= 200
    lengths = {where: len(values) for where, values in
               _arrays_of(trimmed).items() if len(values) > 2}
    assert set(lengths.values()) == {len(trimmed['x'])}, lengths
    response = next(c for c in trimmed['curves'] if not c.get('gray'))['y']
    original = next(c for c in full['curves'] if not c.get('gray'))['y']
    assert max(response) == max(original), 'the spike'
    assert min(response) == min(original), 'the dip'
    assert trimmed['x'][0] == full['x'][0] and trimmed['x'][-1] == full['x'][-1]
    assert 'edges' not in trimmed and trimmed['steps'] is False
    assert trimmed['note'].startswith('Thinned for the file:')
    assert '4000' in trimmed['note']
    # the readouts are the whole record's, worked out before thinning
    assert trimmed['channels'][0]['readouts'] == full['channels'][0]['readouts']


def test_a_figure_under_the_limit_is_left_whole(tmp_path):
    psd, spec = _pair(lines=400)
    whole = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'a.json', psd, specification=spec,
        channel='101Z+')).read_text())
    kept = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'b.json', psd, specification=spec, channel='101Z+',
        max_points=1000)).read_text())
    assert kept == whole


def test_the_one_file_page_trims_as_well(tmp_path):
    psd, spec = _pair(lines=4000, spike=99)
    path = visualdynamics.export_html(tmp_path / 'a.html', psd,
                                      specification=spec, channel='101Z+',
                                      max_points=300)
    [block] = _inline_payload(path)['blocks']
    assert len(block['x']) <= 300 and 'Thinned' in block['note']


def test_channels_keep_only_those_records(tmp_path):
    psd, spec = _pair(dofs=('101Z+', '102Z+', '103Z+'))
    data = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'a.json', psd, specification=spec,
        channels=['102Z+'])).read_text())
    [block] = data['blocks']
    assert [c['label'] for c in block['channels']] == ['102Z+']
    everything = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'b.json', psd, specification=spec)).read_text())
    assert len(everything['blocks'][0]['channels']) == 3
    assert len(psd.response_dof) == 3, 'the caller\'s object is untouched'


def test_a_channel_the_data_does_not_hold_is_refused(tmp_path):
    psd, _spec = _pair(dofs=('101Z+', '102Z+'))
    with pytest.raises(ValueError, match=r"holds \['101Z\+', '102Z\+'\]"):
        visualdynamics.export_figure(tmp_path / 'a.json', psd,
                                     channels=['999Z+'])
    with pytest.raises(ValueError, match="'999Z\\+'.*not among"):
        visualdynamics.export_figure(tmp_path / 'a.json', psd,
                                     channels=['101Z+', '999Z+'])


def test_channels_without_a_plot_are_refused(tmp_path):
    from visualdynamics import mesh

    plate = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, unit=None)
    with pytest.raises(ValueError, match='channels choose the records'):
        visualdynamics.export_figure(tmp_path / 'a.json', geometry=plate,
                                     channels=['1Z+'])


def test_too_few_points_are_refused(tmp_path):
    psd, _spec = _pair()
    with pytest.raises(ValueError, match='at least 16'):
        visualdynamics.export_figure(tmp_path / 'a.json', psd, max_points=8)


def test_a_figure_with_nothing_to_draw_is_refused(tmp_path):
    """A comparison whose channel the specification does not hold has
    nothing to draw; an empty file is refused rather than written."""
    psd, spec = _pair()
    with pytest.raises(ValueError, match='nothing in the figure'):
        visualdynamics.export_figure(tmp_path / 'a.json', psd,
                                     specification=spec, channel='555Z+')


def test_the_split_pair_draws_in_a_host_page(qt_app, tmp_path):
    """A host page loads the renderer as a script of its own and mounts
    the figure on its element: the plot draws, its legend names its
    curves, the title and figure number stay hidden, the host page's own
    styles are untouched, the wheel is left to the page with
    `pageScroll`, and data of another format is refused."""
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl

    psd, spec = _pair()
    data = visualdynamics.export_figure(tmp_path / 'a.json', psd,
                                        specification=spec, channel='101Z+',
                                        theme='light',
                                        renderer=tmp_path / 'figure.js')
    payload = pathlib.Path(data).read_text(encoding='utf-8')
    host = tmp_path / 'host.html'
    host.write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<style>body { padding: 37px; font-family: serif; }</style></head>'
        '<body><p>Prose before the figure.</p><div id="fig"></div>'
        '<script src="figure.js"></script><script>'
        f'const DATA = {payload};'
        "VisualDynamicsFigure.mount(document.getElementById('fig'), DATA,"
        ' {pageScroll: true});'
        'try { VisualDynamicsFigure.mount(document.createElement("div"),'
        ' Object.assign({}, DATA, {format: 999})); window.refused = "no"; }'
        ' catch (e) { window.refused = e.message; }'
        '</script></body></html>', encoding='utf-8')
    view = web_view()
    view.resize(800, 600)
    view.load(QUrl.fromLocalFile(str(host)))
    view.show()
    probe = ("(() => { const fig = document.getElementById('fig');"
             "const c = fig.querySelector('section canvas');"
             "if (!c || !c.height) return null;"
             "const shown = e => !!e && getComputedStyle(e).display !== 'none';"
             "return JSON.stringify({"
             " legend: Array.from(fig.querySelectorAll('.legend span'),"
             "                    s => s.textContent),"
             " hint: fig.querySelector('.hint').textContent,"
             " title: shown(fig.querySelector('h1')),"
             " toggle: shown(fig.querySelector('.themetoggle')),"
             " theme: fig.dataset.theme, classes: fig.className,"
             " bodyPadding: getComputedStyle(document.body).paddingTop,"
             " bodyFont: getComputedStyle(document.body).fontFamily,"
             " touch: getComputedStyle(c).touchAction,"
             " refused: window.refused}); })()")
    try:
        got = json.loads(web_read(view, probe))
    finally:
        web_close(view)
    assert got['legend'][:2] == ['101Z+', 'measured'], got
    assert 'Ctrl+wheel to zoom' in got['hint']
    assert not got['title'] and not got['toggle']
    assert got['theme'] == 'light' and 'vdfig' in got['classes']
    assert got['bodyPadding'] == '37px' and 'serif' in got['bodyFont']
    assert got['touch'] == 'pan-y pinch-zoom'
    assert 'format 1' in got['refused'] and '999' in got['refused']


def test_a_complex_plot_reads_as_the_component_asked_for(tmp_path):
    """An FRF's phase is a figure of its own, as the app's drop-down
    reads it: in degrees on a linear axis with the app's phase ticks;
    a reading the drop-down does not offer is refused."""
    from visualdynamics.core.data import Frf
    from visualdynamics.plot import PHASE_LIMITS

    f = np.arange(10.0, 1001.0, 5.0)
    h = 1.0 / (1.0 - (f / 300.0) ** 2 + 0.1j * f / 300.0)
    frf = Frf(f, h[None], response_dof=['101Z+'], reference_dof=['1Z+'],
              ordinate_dim=['acceleration/acceleration'])
    magnitude = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'm.json', frf)).read_text())['blocks'][0]
    phase = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'p.json', frf, component='phase')).read_text())['blocks'][0]
    assert magnitude['logy'] and 'yticks' not in magnitude
    assert not phase['logy'] and phase['ylabel'] == 'phase [deg]'
    assert phase['yrange'] == list(PHASE_LIMITS)
    k = int(np.argmin(np.abs(f - 300.0)))
    assert phase['curves'][0]['y'][k] == pytest.approx(-90.0, abs=0.01)
    with pytest.raises(ValueError, match='one of magnitude, phase'):
        visualdynamics.export_figure(tmp_path / 'x.json', frf,
                                     component='angle')
