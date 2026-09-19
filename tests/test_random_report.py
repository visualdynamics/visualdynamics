"""The random vibration report, against a real run.

The template is only worth anything if it binds to what an imported
Rattlesnake run actually contains and renders every block. These build
it from `random.nc4` and check the page that comes out, rather than
checking the template's own dict.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.report import random_template
from visualdynamics.report import render_html

#: the report's bar charts: RMS error and band-outside-abort, on the
#: narrowband spectra and again on the octave bands, plus the kurtosis
#: reading of the record every report carries (Brandon, 2026-08-24)
BAR_CHARTS = 5


@pytest.fixture
def run(window, pump):
    """A project holding an imported random vibration run and its PSDs."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    history = next(obj for obj in window.objects.values()
                   if isinstance(obj, visualdynamics.TimeHistory))
    window.add_object('Control PSDs', history.compute_psds())
    pump()
    # the octave section reads the banded objects, both of them
    window.project.compute_octave('Control PSDs', 6)
    spec = next(name for name, obj in window.objects.items()
                if isinstance(obj, visualdynamics.Specification))
    window.project.compute_octave(spec, 6)
    pump()
    return window


def report_of(window):
    return random_template(window.project, links=window.project.links)
def test_every_block_binds_to_a_real_run(run):
    """An unbound block renders as a gray slot. On a run that carries a
    specification, its PSDs, coherence, time data and a channel table,
    none of them should be."""
    page = render_html(report_of(run), run.project, unit_system=visualdynamics.SI)
    assert 'unbound' not in page.lower() or page.count('unbound') < 3
    assert 'Random Vibration Test Report' in page
def blocks_of(template):
    return [b for b in template.blocks]


def test_the_front_matter_is_the_modal_reports():
    """A reader has to know what was tested and where it was
    instrumented before any spectrum means anything."""
    from visualdynamics.core.report import random_template

    kinds = [(b.get('kind'), b.get('caption', '')) for b in
             random_template({}).blocks]
    scenes = [c for k, c in kinds if k == 'scene']
    assert any('geometry' in c.lower() for c in scenes)
    assert any('Excitation degrees of freedom' in c for c in scenes)
    assert any('Response degrees of freedom' in c for c in scenes)
    assert any(k == 'photo' for k, _c in kinds)


def test_coherence_is_never_a_thicket_of_stacked_lines():
    """A dozen channels of coherence stacked on one 2-D axis says
    nothing. The map answered that first — frequency across, channel
    down — and the stage answers it the way the app now does, the
    channels spread in depth with each trace still readable (Brandon,
    2026-08-23). Either reading is right; flat curves are not."""
    from visualdynamics.core.report import random_template

    coherence = next(b for b in random_template({}).blocks
                     if b.get('source') == '@basis:MultipleCoherence')
    assert coherence['mode'] in ('stage', 'map')
    assert coherence['mode'] != 'curves'


def test_the_comparison_figure_carries_every_channel(run):
    """One drawn and the rest a pick away — the app's reading, and for
    the same reason: six responses against six targets with their
    limits is a thicket with no comparison visible in it."""
    from visualdynamics.report import _plot_block

    spec = next(o for o in run.objects.values()
                if isinstance(o, visualdynamics.Specification))
    psds = run.objects['Control PSDs']
    built = _plot_block(
        {'kind': 'plot', 'source': 'P', 'specification': 'S',
         'mode': 'curves', 'caption': 'Control'},
        psds, {'S': spec, 'P': psds}, visualdynamics.SI)
    assert len(built['curves']) == 2, 'the target and the response'
    assert len(built['channels']) > 1
    first = built['channels'][0]
    assert 'response' in first, 'the measurement that answered it'
    assert [z['severity'] for z in first['zones']] == [
        'warning', 'abort', 'warning', 'abort']
def test_the_time_history_carries_the_frames_it_was_averaged_over(run):
    """A time history in a report is not read for its values. It is
    read for what part of the run was analyzed and how, which a bare
    trace leaves out."""
    from visualdynamics.report import _plot_block

    history = next(o for o in run.objects.values()
                   if isinstance(o, visualdynamics.TimeHistory))
    built = _plot_block({'kind': 'plot', 'source': 'T', 'mode': 'curves',
                         'select': 'dim:acceleration'}, history, {}, visualdynamics.SI)
    marks = built['averaging']
    assert len(marks['frames']) == history.averaging.frames
    assert max(marks['window']) == pytest.approx(1.0), 'normalized'
    assert history.averaging.window in marks['label']


def test_the_figure_is_told_the_rail_rather_than_knowing_it(run):
    """Figure and screen must agree in proportion, not only in spirit
    (Brandon, 2026-08-23). The page used to restate
    `plot/averaging.py`'s constants in its JavaScript and was held to
    them by matching strings — a weak seam, since a string can match
    while nothing draws it. The rail now arrives worked out, from the
    averaging the app's own overlay asks, and the page draws what it
    is handed."""
    from visualdynamics.report import _plot_block
    from visualdynamics.report.page import _JS

    history = next(o for o in run.objects.values()
                   if isinstance(o, visualdynamics.TimeHistory))
    built = _plot_block({'kind': 'plot', 'source': 'T', 'mode': 'curves',
                         'select': 'dim:acceleration'}, history, {},
                        visualdynamics.SI)
    rail = built['averaging']['rail']
    theirs = history.averaging.rail(history.sample_rate)
    for key in ('baselines', 'glyph', 'cap', 'foot',
                'band_alpha', 'glyph_alpha'):
        assert rail[key] == theirs[key], key
    assert len(rail['baselines']) == history.averaging.frames
    assert 0.25 <= rail['foot'] < 1.0, 'the trace keeps its share'

    # and nothing restates those numbers in the page any more
    for literal in ('0.97 -', '0.34 / count', '* 0.78', '0.016'):
        assert literal not in _JS, \
            f'{literal!r} is the rail knowing what it should be told'


def test_the_page_is_told_its_colors_too():
    """One color scale and one pair of mark colors, from the app's
    own theme — carried in the payload rather than copied into the
    JavaScript, where a change to the theme could not reach them."""
    import json

    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html
    from visualdynamics.report.page import _JS
    from visualdynamics.theme import DARK, LIGHT, VIRIDIS

    html = render_html(Report('T', []), {})
    payload = json.loads(html.split('type="application/json">')[1]
                         .split('</script>')[0])
    assert payload['viridis'] == [list(stop) for stop in VIRIDIS]
    assert payload['marks_color'] == {
        'light': {'band': LIGHT['averaging_band'],
                  'window': LIGHT['averaging_window']},
        'dark': {'band': DARK['averaging_band'],
                 'window': DARK['averaging_window']}}
    assert '68,1,84' not in _JS.replace(' ', ''), \
        'the viridis stops are handed over, not carried'
    for theme in (LIGHT, DARK):
        for key in ('averaging_band', 'averaging_window'):
            assert theme[key].lstrip('#') not in _JS, \
                f'{theme["name"]} {key} is baked into the page'


def test_a_history_with_no_averaging_carries_no_frames(run):
    from visualdynamics.report import _plot_block

    history = next(o for o in run.objects.values()
                   if isinstance(o, visualdynamics.TimeHistory))
    history.averaging = None
    built = _plot_block({'kind': 'plot', 'source': 'T', 'mode': 'curves',
                         'select': 'dim:acceleration'}, history, {}, visualdynamics.SI)
    assert 'averaging' not in built


def test_the_page_builds_the_channel_picker_and_the_marks(run, tmp_path):
    """Executed in a browser engine, not searched for strings.

    Searching the markup proves nothing here: every one of these words
    is in the script whether or not it ever runs, so a syntax error in
    the drawing would leave a text search perfectly happy and the page
    blank.
    """
    import json
    import os

    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QTimer, QUrl
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    path = tmp_path / 'random.html'
    path.write_text(render_html(report_of(run), run.project,
                                unit_system=visualdynamics.SI), encoding='utf-8')
    app = QApplication.instance() or QApplication(['x'])
    view = QWebEngineView()
    view.resize(1000, 800)
    outcome = {}

    # What the drawing itself says it did, not what the pixels say.
    # `getImageData` hands back a fully transparent buffer on a machine
    # with no working GPU path — every canvas on this page reads as
    # blank under the offscreen platform, however well it drew — so
    # counting ink was measuring the renderer and not the report. Two
    # versions of this test passed on that measurement by luck.
    #
    # Each bar chart stamps how many bars it laid down, at the end of
    # its own drawing. That is the fault this test exists to catch: a
    # syntax or runtime error in the script leaves the markup perfectly
    # searchable and the page blank, and an unreached stamp says so.
    drawn = (
        "const cs = Array.from(document.querySelectorAll('canvas.bars'));"
        "JSON.stringify({canvases: cs.length,"
        " bars: cs.map(c => +(c.dataset.bars || 0))})")
    picked = (
        "const pick = document.querySelector('select.channel');"
        "const was = pick ? pick.options[pick.selectedIndex].text : null;"
        "if (pick) { pick.selectedIndex = 1;"
        "  pick.dispatchEvent(new Event('change')); }"
        "JSON.stringify({"
        " pickers: document.querySelectorAll('select.channel').length,"
        " channels: pick ? pick.options.length : 0,"
        " was: was,"
        " now: pick ? pick.options[pick.selectedIndex].text : null,"
        " figures: Array.from(document.querySelectorAll('.caption'),"
        "   d => d.textContent).filter(t => t.startsWith('Figure '))"
        "   .length})")

    def finish(value):
        outcome.update(picked=value)
        app.quit()

    def read_drawing(value, left):
        """Wait for the page to have built itself, rather than for a
        fixed moment. A single 1500 ms delay was enough on an idle
        machine and not on a busy one, which made this the suite's
        flakiest test the day it started running in parallel.

        An empty answer is the script not having run at all yet, which
        is a reason to keep waiting and not a reading to keep.
        """
        if value:
            outcome.update(drawn=value)
        ready = json.loads(value) if value else {}
        if left and len(ready.get('bars') or []) < BAR_CHARTS:
            QTimer.singleShot(200, lambda: view.page().runJavaScript(
                drawn, 0, lambda v: read_drawing(v, left - 1)))
            return
        view.page().runJavaScript(picked, 0, finish)

    def probe(_ok):
        QTimer.singleShot(200, lambda: view.page().runJavaScript(
            drawn, 0, lambda value: read_drawing(value, 40)))

    view.loadFinished.connect(probe)
    view.load(QUrl.fromLocalFile(str(path)))
    view.show()
    QTimer.singleShot(60000, app.quit)
    app.exec()
    view.deleteLater()

    assert outcome.get('picked'), 'the page never reported back'
    got = json.loads(outcome['picked'])
    # no drop-down anywhere in the random report (Brandon, 2026-09-18):
    # one figure per control channel, the specification's own included
    assert got['pickers'] == 0, 'nothing to pick from'
    assert got['figures'] >= 4, 'and the figures carry their numbers'
    assert outcome.get('drawn'), 'the page never reported what it had drawn'
    drew = json.loads(outcome['drawn'])
    assert drew['canvases'] > 0, 'the figures drew'
    # the two bar charts are the last canvases the page builds, and a
    # blank one would mean the drawing threw where reading the markup
    # would have looked perfectly fine
    assert drew['canvases'] == BAR_CHARTS, 'every bar chart is on the page'
    assert min(drew['bars']) > 1, (
        'and every one of them drew its bars through to the end')


def test_a_binding_never_answers_with_what_a_narrower_one_claims():
    """Specification subclasses Psd, so '@basis:Psd' answered with the
    specification — and the report compared the control PSDs against
    their specification by comparing it with itself: one curve where
    there should be two, and a perfect match whatever the run had done.
    """
    from visualdynamics.core.report import resolve_binding

    project = {'Spec': visualdynamics.Specification(
        abscissa=np.array([10.0, 100.0]),
        ordinate=np.atleast_2d([1e-3, 1e-3]), response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'])}
    assert resolve_binding('@basis:Psd', project) is None, (
        'a specification is not the answer to a request for a PSD')
    assert resolve_binding('@basis:Specification', project) == 'Spec'

    project['Measured'] = visualdynamics.Psd(
        abscissa=np.array([10.0, 100.0]),
        ordinate=np.atleast_2d([1e-3, 1e-3]), response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'])
    assert resolve_binding('@basis:Psd', project) == 'Measured'


def test_a_banded_object_answers_only_to_its_own_token():
    """An octave-band PSD is a Psd and an octave-band specification a
    Specification, and each is what the report's octave section
    compares (Brandon, 2026-09-18): the plain token never answers
    with the banded object, and the banded token never with the
    plain one."""
    from visualdynamics.core.report import binding_label, resolve_binding

    f = np.linspace(10.0, 2000.0, 200)
    level = np.full((1, 200), 1e-3)
    spec = visualdynamics.Specification(
        f, level, response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'], ordinate_unit=['m/s**2'],
        abort_upper=level * 4, abort_lower=level / 4)
    psd = visualdynamics.Psd(f, level, response_dof=['101Z+'],
                             ordinate_dim=['acceleration**2/frequency'],
                             ordinate_unit=['m/s**2'])
    project = {'Octave Spec': spec.to_octave(6), 'Octave PSD': psd.to_octave(6),
               'Spec': spec, 'PSD': psd}
    assert resolve_binding('@basis:Psd', project) == 'PSD'
    assert resolve_binding('@basis:OctavePsd', project) == 'Octave PSD'
    assert resolve_binding('@basis:Specification', project) == 'Spec'
    assert resolve_binding('@basis:OctaveSpecification', project) == 'Octave Spec'
    del project['PSD'], project['Spec']
    assert resolve_binding('@basis:Psd', project) is None, \
        'the banded PSD is not an answer for the narrowband one'
    assert resolve_binding('@basis:Specification', project) is None
    assert binding_label('@basis:OctavePsd') == 'Basis Octave PSD (auto)'
    assert binding_label('@basis:OctaveSpecification') == \
        'Basis Octave Specification (auto)'


def test_the_octave_section_reads_the_banded_pair_from_a_real_run(run):
    """The banded specification and the banded PSDs the run fixture
    made are what the octave blocks bind, and the page draws them as
    a comparison of steps against steps."""
    from visualdynamics.core.report import resolve_binding

    objects, links = run.project, run.project.links
    template = report_of(run)
    octave = [b for b in template.blocks
              if b.get('source') in ('@basis:OctavePsd',
                                      '@basis:OctaveSpecification')]
    from visualdynamics.core.report import control_channel_labels

    per_channel = len(control_channel_labels(objects, links,
                                             '@basis:OctaveSpecification'))
    assert per_channel > 1
    assert len(octave) == 2 + 2 * per_channel, \
        'a specification figure and a comparison per control channel, two bar charts'
    for block in octave:
        for field in ('source', 'specification', 'measured'):
            if block.get(field):
                name = resolve_binding(block[field], objects, links)
                assert name is not None, (field, block[field])
                assert objects[name].bandwidth is not None, 'the banded object'
    assert not [i for i in template.unbound(objects, links)
                if template.blocks[i].get('source', '').startswith('@basis:Octave')]
    page = render_html(template, objects, unit_system=visualdynamics.SI)
    assert 'Control against specification, octave bands' in page
    assert 'Test specification, octave bands' in page


def test_the_script_workflow_bands_the_specification_too():
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'), per_octave=6)
    banded = [name for name, obj in project.items()
              if isinstance(obj, visualdynamics.Specification)
              and obj.bandwidth is not None]
    assert len(banded) == 1
    assert 'to_octave' in ''.join(project.journal) or any(
        'compute_octave' in line for line in project.journal)


def test_the_comparison_is_drawn_on_the_measurements_own_axis():
    """The two rarely share one — a specification is written at
    breakpoints or on the controller's lines — so the specification is
    interpolated onto the measurement rather than the other way about."""
    from visualdynamics.report import _plot_block

    spec = visualdynamics.Specification(
        abscissa=np.array([10.0, 100.0, 1000.0]),
        ordinate=np.atleast_2d([1e-3, 1e-3, 1e-3]), response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'],
        abort_upper=np.atleast_2d([4e-3, 4e-3, 4e-3]),
        abort_lower=np.atleast_2d([2.5e-4, 2.5e-4, 2.5e-4]))
    lines = np.linspace(10.0, 1000.0, 400)
    # held unscaled: the response is over abort on purpose, and
    # detection would lay it back onto the specification first
    measured = visualdynamics.Psd(abscissa=lines,
                         ordinate=np.atleast_2d(np.full(400, 9e-3)),
                         response_dof=['101Z+'],
                         ordinate_dim=['acceleration**2/frequency'])
    measured.scale_db = 0
    built = _plot_block(
        {'kind': 'plot', 'source': 'P', 'specification': 'S',
         'mode': 'curves'}, measured, {'S': spec, 'P': measured}, visualdynamics.SI)
    assert len(built['x']) == 400, "the measurement's grid, not the spec's"
    channel = built['channels'][0]
    assert len(channel['response']) == 400
    assert len(channel['y']) == 400, 'the target, interpolated onto it'
    marked = sum(1 for v in channel['over'] if v is not None)
    assert marked > 300, 'well over abort everywhere, and marked everywhere'


def test_the_marked_lines_are_the_lines_the_table_counts():
    """One rule, asked once. The marks were computed against the raw
    limits while the response came back converted, so the comparison
    was m/s^2 against whatever the report was set to — it agreed with
    the table only in SI, and by luck."""
    from visualdynamics.core.compliance import compare_all
    from visualdynamics.report import _plot_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    psds = loaded['time_data'].compute_psds()
    counted = dict(compare_all(spec, psds))
    for units in (visualdynamics.SI, visualdynamics.IN_LBF_S):
        built = _plot_block(
            {'kind': 'plot', 'source': 'P', 'specification': 'S',
             'mode': 'curves'}, psds, {'S': spec, 'P': psds}, units)
        for channel in built['channels']:
            marked = sum(1 for key in ('over', 'under')
                         for v in channel.get(key, []) if v is not None)
            assert marked == counted[channel['label']]['abort_lines'], (
                f'{channel["label"]} in {units.name}')


def test_a_density_is_drawn_flat_across_its_own_bin():
    """A polyline through the line centers draws a slope that is not in
    the data. The app steps a density; so does the report.

    And only a density. A written specification's points are
    breakpoints of a continuous curve — that is what its area is taken
    under — so a staircase is a shape it does not have. The report used
    to step it anyway, because the flag was set for a specification
    *with limits*, which meant "the zones are being drawn too" and not
    "this is a density".
    """
    from visualdynamics.report import _plot_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    psds = loaded['time_data'].compute_psds()
    comparison = _plot_block(
        {'kind': 'plot', 'source': 'P', 'specification': 'S',
         'mode': 'curves'}, psds, {'S': spec, 'P': psds}, visualdynamics.SI)
    alone = _plot_block({'kind': 'plot', 'source': 'S', 'mode': 'curves'},
                        spec, {}, visualdynamics.SI)
    assert psds.interpolation == 'bin'
    assert comparison['steps'] is True
    assert spec.interpolation == 'log_log'
    assert alone.get('steps', False) is False


def test_the_reference_is_gray_and_the_measurement_is_the_page_ink():
    """The app's own reading of the pair: gray behind because it is the
    reference, the page's ink in front because it is what is being
    looked at."""
    from visualdynamics.report import _plot_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    psds = loaded['time_data'].compute_psds()
    built = _plot_block(
        {'kind': 'plot', 'source': 'P', 'specification': 'S',
         'mode': 'curves'}, psds, {'S': spec, 'P': psds}, visualdynamics.SI)
    target, measured = built['curves']
    assert target.get('gray') and not target.get('ink')
    assert measured.get('ink') and not measured.get('gray')


# ---- the comparison as bars ---------------------------------------------


def test_the_report_carries_both_bar_charts():
    """The table these replaced is gone rather than sitting beside
    them: a table of six channels is read, a table of sixty is
    scanned, and the same numbers as bars answer which channel is
    worst before anything is read at all."""
    from visualdynamics.core.report import random_template

    blocks = random_template({}).blocks
    charts = [b for b in blocks if b.get('kind') == 'bars']
    assert [b['mode'] for b in charts] == ['error', 'lines',
                                           'error', 'lines',
                                           'kurtosis'], (
        'both readings, narrowband and then on octave bands')
    assert [b.get('octave') for b in charts] == [None] * 5, (
        'no chart bands for itself: the octave pair reads the banded objects')
    assert [(b['source'], b.get('measured')) for b in charts[2:4]] == [
        ('@basis:OctaveSpecification', '@basis:OctavePsd')] * 2, (
        'the banded requirement against the banded measurement')
    assert not any(b.get('kind') == 'compliance' for b in blocks), (
        'the comparison table is gone entirely')


def test_the_error_chart_is_in_dB_and_the_lines_chart_a_share():
    """A tolerance on the level is written in dB and read in dB. A
    share of the band has a ceiling and no floor — no amount of staying
    inside the abort limits is a fault."""
    from visualdynamics.core.compliance import ERROR_DB, LINES_PERCENT
    from visualdynamics.report import _bars_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    psds = loaded['time_data'].compute_psds()
    error = _bars_block({'mode': 'error'}, spec, psds, visualdynamics.SI)
    assert (error['low'], error['high']) == (-ERROR_DB, ERROR_DB)
    assert error['units'].strip() == 'dB'
    lines = _bars_block({'mode': 'lines'}, spec, psds, visualdynamics.SI)
    assert (lines['low'], lines['high']) == (LINES_PERCENT, None)
    assert all(v >= 0 for v in lines['values'])


def test_the_bars_are_the_numbers_the_comparison_gives():
    from visualdynamics.core.compliance import channel_errors, compare_all
    from visualdynamics.report import _bars_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    psds = loaded['time_data'].compute_psds()
    rows = channel_errors(compare_all(spec, psds))
    error = _bars_block({'mode': 'error'}, spec, psds, visualdynamics.SI)
    assert error['labels'] == [label for label, _d, _p in rows]
    assert error['values'] == pytest.approx([d for _l, d, _p in rows],
                                            abs=1e-3)


def test_a_bar_chart_with_nothing_in_common_renders_nothing():
    """Not a chart of zeros, which would read as six channels that
    matched."""
    from visualdynamics.report import _bars_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    elsewhere = visualdynamics.Psd(
        abscissa=spec.abscissa, ordinate=np.ones_like(spec.ordinate.real),
        response_dof=[f'9{i}Z+' for i in range(spec.num_records)],
        ordinate_dim=spec.ordinate_dim)
    assert _bars_block({'mode': 'error'}, spec, elsewhere, visualdynamics.SI) is None


def test_the_bar_charts_are_numbered_figures():
    """They were left out of the numbering, so they had no label at
    all — and the text that refers to them by name had nothing to
    point at."""
    import json
    import re

    from visualdynamics.core.report import random_template

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    psds = loaded['time_data'].compute_psds()
    project = {'Specification': loaded['Random_specification'],
               'Measured': psds,
               'Octave Specification': loaded['Random_specification'].to_octave(6),
               'Octave Measured': psds.to_octave(6)}
    page = render_html(random_template(project), project,
                       unit_system=visualdynamics.SI)
    payload = json.loads(re.search(
        r'<script id="data"[^>]*>(.*?)</script>', page,
        re.DOTALL).group(1))
    bars = [b for b in payload['blocks'] if b['kind'] == 'bars']
    assert len(bars) == 4, 'narrowband and octave, two readings each'
    for block in bars:
        assert block.get('label', '').startswith('Figure ')


def test_a_share_of_a_band_cannot_be_negative():
    """So its axis does not pretend it might be. An error in dB runs
    both ways and gets whatever the data needs."""
    from visualdynamics.report import _bars_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    psds = loaded['time_data'].compute_psds()
    assert _bars_block({'mode': 'lines'}, spec, psds, visualdynamics.SI)['floor'] == 0.0
    assert _bars_block({'mode': 'error'}, spec, psds, visualdynamics.SI)['floor'] is None


def test_the_octave_section_bands_the_measurement_for_itself():
    """The block says which fraction it wants and the report does the
    integration, so a template describes the document rather than
    listing what the project must already contain."""
    from visualdynamics.report import _bars_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    psds = loaded['time_data'].compute_psds()
    plain = _bars_block({'mode': 'error'}, spec, psds, visualdynamics.SI)
    banded = _bars_block({'mode': 'error', 'octave': 6}, spec, psds,
                         visualdynamics.SI)
    assert plain['labels'] == banded['labels'], 'the same control channels'
    # the level each channel came out at cannot depend on how the
    # measurement was arranged — that is why the specification is not
    # banded alongside it
    assert banded['values'] == pytest.approx(plain['values'], abs=0.05)


def test_the_octave_comparison_reads_the_same_channels():
    from visualdynamics.report import _plot_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    psds = loaded['time_data'].compute_psds()
    built = _plot_block(
        {'kind': 'plot', 'source': 'P', 'specification': 'S',
         'mode': 'curves', 'octave': 6}, psds, {'S': spec, 'P': psds},
        visualdynamics.SI)
    assert built['steps'], 'bands are drawn flat across their own width'
    assert len(built['channels']) > 1
    assert len(built['x']) < len(psds.abscissa), 'fewer bands than lines'


# ---- the comparison figure, per channel and on the specification's band ----

def _pair_for_comparison():
    lines = np.linspace(5.0, 2000.0, 800)
    spec = visualdynamics.Specification(
        abscissa=np.array([10.0, 100.0, 1000.0]),
        ordinate=np.atleast_2d([1e-3, 1e-3, 1e-3]), response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'],
        abort_upper=np.atleast_2d([4e-3, 4e-3, 4e-3]),
        abort_lower=np.atleast_2d([2.5e-4, 2.5e-4, 2.5e-4]))
    measured = visualdynamics.Psd(abscissa=lines,
                                  ordinate=np.atleast_2d(np.full(800, 9e-3)),
                                  response_dof=['101Z+'],
                                  ordinate_dim=['acceleration**2/frequency'])
    measured.scale_db = 0
    return spec, measured


def test_the_comparison_opens_on_the_specifications_band():
    """The axis ran to the response's band, well past the requirement
    (Brandon, 2026-09-18): the figure opens on the specification's
    own band — its lines, or a banded one's outer edges — and the
    rest is a zoom away."""
    from visualdynamics.plot import bin_edges
    from visualdynamics.report import _plot_block

    spec, measured = _pair_for_comparison()
    built = _plot_block({'kind': 'plot', 'source': 'P', 'specification': 'S',
                         'mode': 'curves'}, measured, {'S': spec, 'P': measured},
                        visualdynamics.SI)
    assert built['home_x'] == [10.0, 1000.0]
    assert min(built['x']) == 5.0 and max(built['x']) == 2000.0, \
        'the measurement is still there to zoom out to'
    banded_spec = spec.to_octave(3)
    banded = _plot_block({'kind': 'plot', 'source': 'P', 'specification': 'S',
                          'mode': 'curves'}, measured.to_octave(6),
                         {'S': banded_spec, 'P': measured.to_octave(6)},
                         visualdynamics.SI)
    edges = bin_edges(banded_spec.abscissa, banded_spec.bin_widths())
    assert banded['home_x'] == pytest.approx([edges[0], edges[-1]])


def test_a_banded_specification_is_drawn_on_its_own_bins():
    """Its target and zones flat across each of *its* bands, beside a
    response stepped on the measurement's: the requirement used to be
    interpolated onto the measurement's grid and stepped there, a
    staircase of the measurement's making (Brandon, 2026-09-18)."""
    from visualdynamics.plot import bin_edges
    from visualdynamics.report import _plot_block

    spec, measured = _pair_for_comparison()
    banded_spec, banded_psd = spec.to_octave(3), measured.to_octave(6)
    built = _plot_block({'kind': 'plot', 'source': 'P', 'specification': 'S',
                         'mode': 'curves'}, banded_psd,
                        {'S': banded_spec, 'P': banded_psd}, visualdynamics.SI)
    target, response = built['curves']
    assert target['x'] == pytest.approx(list(banded_spec.abscissa))
    assert target['steps'] is True
    assert target['edges'] == pytest.approx(
        list(bin_edges(banded_spec.abscissa, banded_spec.bin_widths())))
    channel = built['channels'][0]
    assert len(channel['y']) == len(banded_spec.abscissa)
    zone = next(z for z in channel['zones'] if z['lower'] is not None)
    assert len(zone['lower']) == len(banded_spec.abscissa)
    assert len(channel['response']) == len(built['x']) == len(banded_psd.abscissa)
    assert len(channel['over']) == len(built['x']), 'the marks on the measurement'
    assert response['x'] is None and 'edges' not in response
    # and a specification on lines stays on the measurement's axis
    plain = _plot_block({'kind': 'plot', 'source': 'P', 'specification': 'S',
                         'mode': 'curves'}, measured, {'S': spec, 'P': measured},
                        visualdynamics.SI)
    assert plain['curves'][0]['x'] is None
    assert len(plain['channels'][0]['y']) == len(plain['x'])


def test_a_block_may_name_its_channel():
    from visualdynamics.report import _plot_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    psds = loaded['time_data'].compute_psds()
    named = _plot_block({'kind': 'plot', 'source': 'P', 'specification': 'S',
                         'mode': 'curves', 'channel': '104Z+',
                         'caption': 'Control against specification — 104Z+'},
                        psds, {'S': spec, 'P': psds}, visualdynamics.SI)
    assert [ch['label'] for ch in named['channels']] == ['104Z+']
    assert named['caption'] == 'Control against specification — 104Z+', \
        'no drop-down note: there is one channel'
    assert _plot_block({'kind': 'plot', 'source': 'P', 'specification': 'S',
                        'mode': 'curves', 'channel': '999Z+'},
                       psds, {'S': spec, 'P': psds}, visualdynamics.SI) is None


def test_the_template_writes_a_figure_per_control_channel():
    """One figure per control channel, narrowband and on octave bands,
    rather than one figure with a drop-down (Brandon, 2026-09-18); a
    project with nothing bound yet gets the one figure that picks."""
    from visualdynamics.core.report import control_channel_labels, random_template

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    psds = loaded['time_data'].compute_psds()
    project = {'Specification': spec, 'Measured': psds,
               'Octave Specification': spec.to_octave(6),
               'Octave Measured': psds.to_octave(6)}
    labels = control_channel_labels(project, None)
    assert len(labels) == 8 and labels[0] == '101Z+'
    blocks = random_template(project).blocks
    narrow = [b for b in blocks if b.get('specification') == '@basis:Specification'
              and b.get('kind') == 'plot']
    octave = [b for b in blocks
              if b.get('specification') == '@basis:OctaveSpecification'
              and b.get('kind') == 'plot']
    assert [b['channel'] for b in narrow] == labels
    assert [b['channel'] for b in octave] == labels
    assert narrow[0]['caption'] == 'Control against specification — 101Z+'
    assert octave[-1]['caption'].endswith(f'octave bands — {labels[-1]}')
    empty = random_template({}).blocks
    lone = [b for b in empty if b.get('specification') == '@basis:Specification']
    assert len(lone) == 1 and 'channel' not in lone[0]
    import json
    import re

    page = render_html(random_template(project), project,
                       unit_system=visualdynamics.SI)
    payload = json.loads(re.search(
        r'<script id="data"[^>]*>(.*?)</script>', page, re.DOTALL).group(1))
    captions = [b.get('caption', '') for b in payload['blocks']
                if b.get('kind') == 'plot']
    comparisons = [c for c in captions if c.startswith('Control against specification')]
    assert len(comparisons) == 16
    specs = [c for c in captions if c.startswith('Test specification')]
    assert len(specs) == 16, 'the specification figures too, one per channel'
    assert not any('drop-down' in c for c in captions), \
        'no drop-down anywhere in the random report'
    lone_specs = [b for b in random_template({}).blocks
                  if b.get('source') == '@basis:Specification' and b.get('kind') == 'plot']
    assert len(lone_specs) == 1 and 'channel' not in lone_specs[0]


def test_a_table_row_is_one_line():
    """A channel table's comments wrapped every row onto four lines
    (Brandon, 2026-09-18): cells never wrap, and a table wider than the
    page scrolls in its own wrap."""
    from visualdynamics.report.page import _CSS, _JS

    rule = _CSS[_CSS.index('th, td {'):]
    rule = rule[:rule.index('}')]
    assert 'white-space: nowrap' in rule
    assert '.tablewrap { overflow-x: auto; }' in _CSS
    assert "wrap.className = 'tablewrap'" in _JS


def test_a_specification_figure_may_name_its_channel():
    """A bounded source's block that names a channel carries that one
    channel and no drop-down note."""
    from visualdynamics.report import _plot_block

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = loaded['Random_specification']
    named = _plot_block({'kind': 'plot', 'source': 'S', 'mode': 'curves',
                         'channel': '104Z+', 'caption': 'Test specification — 104Z+'},
                        spec, {'S': spec}, visualdynamics.SI)
    assert [ch['label'] for ch in named['channels']] == ['104Z+']
    assert named['caption'] == 'Test specification — 104Z+'
    assert _plot_block({'kind': 'plot', 'source': 'S', 'mode': 'curves',
                        'channel': '999Z+'}, spec, {'S': spec},
                       visualdynamics.SI) is None
