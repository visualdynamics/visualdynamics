"""One figure as a data file and a script (`visualdynamics.export_figure`).

A page whose Content-Security-Policy allows no inline script and no
frames can show neither the one-file page nor an iframe of it. The
split export writes the figure's data as JSON and the viewers as one
same-origin script that mounts a figure on an element the page gives
it, and both exports can keep fewer channels and fewer points.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re

import numpy as np
import pytest
from conftest import web_close, web_read, web_view

import visualdynamics
from visualdynamics.core.data import Psd, Specification, TimeHistory
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


def test_shared_indices_keep_a_lone_missing_sample_as_a_gap():
    """A slice's extremes skip a missing sample, so a lone NaN would be
    thinned away and the line drawn across it; the start of each gap is
    kept while the gaps fit a quarter of the budget, and past that the
    budget still holds."""
    rows = np.sin(np.arange(5000) / 40.0)[None, :].copy()
    rows[0, 777] = np.nan
    keep = shared_peak_indices(rows, 100)
    assert 777 in keep and len(keep) <= 100
    rows[0, 3000:3010] = np.nan
    keep = shared_peak_indices(rows, 100)
    assert 777 in keep and 3000 in keep and 3001 not in keep
    many = np.sin(np.arange(5000) / 40.0)[None, :].copy()
    many[0, 10::150] = np.nan                  # 34 gaps against 25 allowed
    keep = shared_peak_indices(many, 100)
    assert len(keep) <= 100
    assert not set(range(10, 5000, 150)) <= set(keep.tolist())


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
    from visualdynamics.core.shapes import ShapeSet

    plate = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, unit=None)
    with pytest.raises(ValueError, match='channels choose the records'):
        visualdynamics.export_figure(tmp_path / 'a.json', geometry=plate,
                                     channels=['1Z+'])
    # data that holds no records by DOF (shapes) is refused the same way
    shapes = ShapeSet([10.0], [0.01], ['1Z+'], np.ones((1, 1)))
    with pytest.raises(ValueError, match='channels choose the records'):
        visualdynamics.export_figure(tmp_path / 'a.json', shapes,
                                     channels=['1Z+'])


def test_too_few_points_are_refused(tmp_path):
    psd, _spec = _pair()
    with pytest.raises(ValueError, match='at least 16'):
        visualdynamics.export_figure(tmp_path / 'a.json', psd, max_points=8)


def _like(psd, ordinate):
    return Psd(psd.abscissa, ordinate, response_dof=psd.response_dof,
               reference_dof=psd.reference_dof, ordinate_dim=psd.ordinate_dim)


def test_a_record_with_nothing_to_draw_is_refused_naming_why(tmp_path):
    """A record of zeros on a logarithmic axis, one missing throughout
    and one with no samples would each be written as an empty frame;
    each is refused with the reason."""
    psd, _spec = _pair()
    zero = _like(psd, np.zeros_like(psd.ordinate))
    with pytest.raises(ValueError, match=r'^nothing in the figure has a value '
                       r'to draw: 101Z\+/101Z\+ holds no value above zero, '
                       r'which is all a logarithmic axis draws$'):
        visualdynamics.export_figure(tmp_path / 'a.json', zero)
    gone = _like(psd, np.full_like(psd.ordinate, np.nan))
    with pytest.raises(ValueError, match=r'101Z\+/101Z\+ holds no value above'):
        visualdynamics.export_html(tmp_path / 'a.html', gone, max_points=50)
    empty = Psd(np.array([]), np.zeros((1, 0)), response_dof=['1Z+'],
                reference_dof=['1Z+'], ordinate_dim=psd.ordinate_dim)
    with pytest.raises(ValueError, match=r'1Z\+/1Z\+ holds no value above'):
        visualdynamics.export_figure(tmp_path / 'a.json', empty)
    # a dead channel beside a live one is drawn: the figure has a curve
    both = Psd(psd.abscissa, np.vstack([psd.ordinate, 0 * psd.ordinate]),
               response_dof=['101Z+', '102Z+'], reference_dof=['101Z+', '102Z+'],
               ordinate_dim=psd.ordinate_dim * 2)
    visualdynamics.export_figure(tmp_path / 'b.json', both)


def test_a_lone_missing_sample_stays_a_gap_when_thinned(tmp_path):
    psd, spec = _pair(lines=400)
    ordinate = psd.ordinate.copy()
    ordinate[0, 123] = np.nan
    data = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'a.json', _like(psd, ordinate), specification=spec,
        max_points=50)).read_text())
    [block] = data['blocks']
    x = block['curves'][0]['x'] or block['x']
    [measured] = [c for c in block['curves'] if any(v is None for v in c['y'])]
    assert measured['y'][x.index(pytest.approx(psd.abscissa[123]))] is None
    assert len(measured['y']) <= 50


def test_an_empty_channel_list_is_refused(tmp_path):
    psd, _spec = _pair()
    with pytest.raises(ValueError, match=r"^channels is empty: name at least "
                       r"one record to keep, by its response DOF \('101Z\+'\)$"):
        visualdynamics.export_figure(tmp_path / 'a.json', psd, channels=[])


def test_a_limit_in_other_units_at_a_chosen_channel_is_refused(tmp_path):
    """Records at one DOF can be different quantities; a specification
    record kept by `channels` must be the quantity the data's record at
    that DOF is, or the limit would be drawn against a curve in other
    units."""
    psd, spec = _pair(dofs=('101Z+', '102Z+'))
    force = copy.deepcopy(spec)
    force.ordinate_dim[0] = 'force**2/frequency'
    with pytest.raises(ValueError, match=r"specification's 101Z\+ is "
                       r"force\*\*2/frequency and the data's is "
                       r"acceleration\*\*2/frequency"):
        visualdynamics.export_figure(tmp_path / 'a.json', psd,
                                     specification=force, channels=['101Z+'])
    data = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'b.json', psd, specification=force,
        channels=['102Z+'])).read_text())
    assert [c['label'] for c in data['blocks'][0]['channels']] == ['102Z+']


@pytest.mark.parametrize('max_points', [None, 100])
def test_a_specification_alone_splits_on_the_grid_its_channels_are(
        tmp_path, max_points):
    """The split export of a specification on its own, whole and
    trimmed: every array the page draws for a channel has the length of
    the grid it is drawn on, and spans the band."""
    from test_export_html import _workmanship

    from visualdynamics.units import SYSTEMS

    data = visualdynamics.export_figure(
        tmp_path / 'spec.json', _workmanship(), max_points=max_points,
        unit_system=SYSTEMS['in-slinch-lbf-s (g)'])
    [block] = json.loads(pathlib.Path(data).read_text())['blocks']
    drawn = block['curves'][0]['x'] or block['x']
    [channel] = block['channels']
    assert len(channel['y']) == len(drawn)
    assert max_points is None or len(drawn) <= max_points
    for zone in channel['zones']:
        for edge in (zone['lower'], zone['upper']):
            assert edge is None or len(edge) == len(drawn)
    written = [f for f, v in zip(drawn, channel['y'], strict=True)
               if v is not None]
    assert min(written) == pytest.approx(20.0)
    assert max(written) == pytest.approx(2000.0)


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


# ---- the mutation run's boundaries, each pinned ----------------------------------

def test_shared_indices_fill_the_budget_and_keep_a_record_that_fits_whole():
    """A rising record keeps each slice's first and last sample, two a
    slice, so the most slices that fit fill the budget exactly; a record
    of exactly the budget is kept whole."""
    rising = np.arange(1000.0)[None, :]
    assert len(shared_peak_indices(rising, 100)) == 100
    assert np.array_equal(shared_peak_indices(np.ones((2, 64)), 64),
                          np.arange(64))


def test_shared_indices_find_the_most_slices_that_fit():
    """Two records whose extremes fall apart: the bisection finds the
    largest slicing whose points fit, the one an exhaustive search
    finds (the count of points kept rises with the slicing here)."""
    from visualdynamics.decimate import _extreme_indices

    n = np.arange(1000.0)
    rows = np.array([n, (n * 37.0) % 101.0])
    sizes = [len(np.unique(np.concatenate(
        [[0, 999], *_extreme_indices(rows, bins)]))) for bins in range(1, 51)]
    assert sizes == sorted(sizes), 'the premise: more slices keep more points'
    assert len(shared_peak_indices(rows, 100)) == max(
        s for s in sizes if s <= 100)


def test_shared_indices_at_the_fewest_points_a_budget_may_hold():
    """Three records need eight points, two a record and the two ends:
    eight are taken, never more, with the ends kept; seven are refused."""
    rows = np.random.default_rng(5).standard_normal((3, 500))
    keep = shared_peak_indices(rows, 8)
    assert len(keep) <= 8 and keep[0] == 0 and keep[-1] == 499
    with pytest.raises(ValueError, match='at least 8'):
        shared_peak_indices(rows, 7)


def test_gaps_are_kept_up_to_a_quarter_of_the_budget_and_the_room_left():
    """25 gaps fit a budget of 100 and are all kept. Six records at a
    budget of 16 leave room beside the coarsest slicing's twelve
    extremes and two ends for two gaps but not three: two are kept, and
    with three the budget still holds."""
    rising = np.arange(5000.0)[None, :].copy()
    starts = list(range(100, 5000, 196))[:25]
    rising[0, starts] = np.nan
    assert set(starts) <= set(shared_peak_indices(rising, 100).tolist())
    six = np.random.default_rng(9).standard_normal((6, 400))
    six[:, [100, 300]] = np.nan
    keep = shared_peak_indices(six, 16)
    assert {100, 300} <= set(keep.tolist()) and len(keep) <= 16
    six[:, 200] = np.nan
    assert len(shared_peak_indices(six, 16)) <= 16


def test_a_curve_of_exactly_max_points_is_left_whole(tmp_path):
    """The plot's shared abscissa at exactly `max_points` is not thinned
    again, and the note does not say it was. A curve on an abscissa of
    its own (a spectrum the page already thinned, which leaves the
    plot's own abscissa drawn on by nothing) is trimmed with that
    abscissa thinned once, on its own values."""
    psd, spec = _pair(lines=400)
    whole = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'a.json', psd, specification=spec)).read_text())
    kept = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'b.json', psd, specification=spec,
        max_points=400)).read_text())
    assert kept == whole
    f = np.linspace(1.0, 2000.0, 90_000)
    long = Psd(f, (1.0 + 0.1 * np.sin(f))[None, :] * 1e-3,
               response_dof=['1Z+'], reference_dof=['1Z+'],
               ordinate_dim=['acceleration**2/frequency'])
    [own] = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'c.json', long)).read_text())['blocks']
    length = len(own['curves'][0]['x'])
    assert length < 90_000, 'the page thinned it onto an abscissa of its own'
    [trimmed] = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'e.json', long, max_points=1000)).read_text())['blocks']
    curve = trimmed['curves'][0]
    assert len(curve['x']) == len(curve['y']) <= 1000
    assert len(trimmed['x']) <= 1000
    assert trimmed['x'][0] == own['x'][0] and trimmed['x'][-1] == own['x'][-1]


def test_a_note_already_on_a_figure_is_kept_when_it_is_trimmed(tmp_path):
    """A record long enough for the page to thin it already carries that
    note; trimming it further leaves one sentence, counting from the
    record (two sentences stood here, the second understating the
    record, 2026-10-28)."""
    history = TimeHistory(np.arange(100_000) / 1000.0,
                          np.random.default_rng(0).standard_normal((1, 100_000)),
                          response_dof=['1Z+'])
    [own] = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'a.json', history)).read_text())['blocks']
    [trimmed] = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'b.json', history, max_points=1000)).read_text())['blocks']
    first = own['note']
    assert first.startswith('Thinned for the file:')
    assert trimmed['note'].count('Thinned for the file') == 1
    assert 'of 100000,' in trimmed['note'] and 'every extreme kept' in trimmed['note']
    assert len(trimmed['x']) <= 1000


def test_sixteen_points_is_the_fewest_a_curve_may_keep(tmp_path):
    psd, _spec = _pair(lines=400)
    [block] = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'a.json', psd, max_points=16)).read_text())['blocks']
    assert len(block['x']) <= 16


def test_a_scene_carries_its_shapes_only_when_given(tmp_path):
    from visualdynamics import mesh
    from visualdynamics.core.shapes import ShapeSet

    plate = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, unit=None)
    dofs = [f'{n}Z+' for n in plate.node_id]
    shapes = ShapeSet([10.0], [0.01], dofs, np.ones((1, len(dofs))))
    [moving] = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'a.json', geometry=plate, shapes=shapes)).read_text())['blocks']
    [still] = json.loads(pathlib.Path(visualdynamics.export_figure(
        tmp_path / 'b.json', geometry=plate)).read_text())['blocks']
    assert len(moving['modes']) == 1 and not still['modes']


def test_the_split_pair_draws_under_a_strict_content_security_policy(
        qt_app, tmp_path):
    """Served over HTTP to a page whose policy allows scripts and styles
    from its own origin only (styles also by nonce) and no inline script
    at all, the renderer mounts the figure, fetched with `load`, with no
    violation of the policy reported."""
    import functools
    import http.server
    import threading

    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl

    psd, spec = _pair()
    visualdynamics.export_figure(tmp_path / 'a.json', psd, specification=spec,
                                 channel='101Z+', theme='light',
                                 renderer=tmp_path / 'figure.js')
    (tmp_path / 'page.js').write_text(
        "window.violations = [];"
        "document.addEventListener('securitypolicyviolation',"
        " e => window.violations.push(e.violatedDirective));"
        "VisualDynamicsFigure.load(document.getElementById('fig'), 'a.json',"
        " {pageScroll: true}).then(() => { window.mounted = true; },"
        " e => { window.mounted = String(e); });", encoding='utf-8')
    (tmp_path / 'host.html').write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<meta http-equiv="Content-Security-Policy" content="default-src '
        "'self'; script-src 'self'; style-src 'self' 'nonce-n1'\">"
        '</head><body><div id="fig"></div>'
        '<script src="figure.js" nonce="n1"></script>'
        '<script src="page.js"></script></body></html>', encoding='utf-8')
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(tmp_path))
    handler.log_message = lambda *_a: None
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    view = web_view()
    view.resize(800, 600)
    view.load(QUrl(f'http://127.0.0.1:{server.server_address[1]}/host.html'))
    view.show()
    probe = ("(() => { if (!window.mounted) return null;"
             "const c = document.querySelector('#fig section canvas');"
             "if (window.mounted === true && (!c || !c.height)) return null;"
             "return JSON.stringify({mounted: window.mounted,"
             " violations: window.violations,"
             " legend: Array.from(document.querySelectorAll("
             "   '#fig .legend span'), s => s.textContent)}); })()")
    try:
        got = json.loads(web_read(view, probe))
    finally:
        web_close(view)
        server.shutdown()
        server.server_close()
    assert got['mounted'] is True, got
    assert got['violations'] == [], got
    assert got['legend'][:2] == ['101Z+', 'measured']


def test_a_setting_that_would_do_nothing_is_refused(tmp_path):
    from visualdynamics import mesh

    plate = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, unit=None)
    with pytest.raises(ValueError, match='max_points thins a plot.s curves, '
                                         'and a scene has none'):
        visualdynamics.export_figure(tmp_path / 'a.json', geometry=plate,
                                     max_points=100)
    psd, _spec = _pair()
    with pytest.raises(ValueError, match="component reads complex data, and "
                                         "this data is real: 'phase'"):
        visualdynamics.export_figure(tmp_path / 'a.json', psd,
                                     component='phase')


def test_a_figure_thinned_twice_carries_one_sentence_counting_the_record(tmp_path):
    """A time history's builder thins it and the export thins it again
    to `max_points`: one note, from the record's own count, not two
    with the second understating it (2026-10-28)."""
    rate, seconds = 4096.0, 28.0
    t = np.arange(int(rate * seconds)) / rate
    history = TimeHistory(t, np.sin(2 * np.pi * 7.0 * t)[None],
                          response_dof=['1Z+'], ordinate_dim='acceleration')
    from visualdynamics.report import export_figure

    path = tmp_path / 'thinned.json'
    export_figure(str(path), history, max_points=4000)
    block = json.loads(path.read_text())['blocks'][0]
    note = block['note']
    assert note.count('Thinned for the file') == 1
    assert f'of {t.size},' in note
