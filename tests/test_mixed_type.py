"""The random-and-sine project type: its slots, its report, its import.

A random environment with a sine sweep under it is the one mixed run
that is a project type of its own (Brandon, 2026-09-24): the project
shows both halves' slots, and one report holds both judgments. Nothing
in it is a third copy of either half — the slot list is composed from
the two types' lists, and the report's judgment blocks are the very
functions the random and sine reports call.
"""

from __future__ import annotations

import json
import os

import pytest
from conftest import web_close, web_read
from test_project_type import _placeholders
from test_rattlesnake_sine import _write_run

import visualdynamics
from visualdynamics.core.report import (
    PROJECT_TEMPLATES,
    mixed_template,
    project_expectations,
    random_template,
    sine_template,
)

STRESS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'stressdata', 'plate')
needs_stress = pytest.mark.skipif(
    not os.path.exists(os.path.join(STRESS, 'mixed.nc4')),
    reason='stressdata/plate/mixed.nc4 not generated')


def _labels(project_type):
    return [name for name, *_rest in project_expectations(project_type)]


def test_the_slots_are_the_randoms_plus_what_the_sine_adds():
    random = _labels('Random Vibration')
    mixed = _labels('Random and Sine')
    assert mixed == random[:-1] + ['Sine Sweep Specification',
                                   'Sine Level Set', 'Report']
    # labels repeat by design (the second Specification is the banded
    # one, by ordinal); the icons say which slot is which, and none of
    # the sine's duplicates one the random already had
    icons = [icon for _n, _cls, icon, *_rest
             in project_expectations('Random and Sine')]
    assert len(set(icons)) == len(icons), 'no slot twice'


def test_the_tree_shows_both_halves_slots(window, pump):
    window.set_project_type('Random and Sine')
    pump()
    assert _placeholders(window) == _labels('Random and Sine')


def _headings(report):
    return [block['text'].split('\n', 1)[0] for block in report.blocks
            if block['kind'] == 'text' and block['text'].startswith('## ')]


def test_the_report_holds_both_judgments_once_each():
    """The random's control comparisons and compliance, then the sine's
    levels and deviation, with the front matter, the recording, the
    data quality and the conclusions each once."""
    report = mixed_template(visualdynamics.Project(), links=[])
    assert report.title == 'Random and Sine Test Report'
    assert report.blocks[0]['kind'] == 'verdict', 'the verdict leads'
    headings = _headings(report)
    for once in ('## Test Summary', '## Test Article and Instrumentation',
                 '## Measured Data', '## Control', '## Compliance',
                 '## Octave Band Comparison', '## The Sweep',
                 '## Level Against Requirement', '## Deviation',
                 '## Data Quality', '## Conclusions'):
        assert headings.count(once) == 1, once
    assert headings.index('## Control') < headings.index('## The Sweep') \
        < headings.index('## Data Quality') < headings.index('## Conclusions')
    bars = [(b['mode'], b['caption']) for b in report.blocks
            if b['kind'] == 'bars']
    assert ('error', 'RMS error by control channel') in bars
    assert ('sine', 'Sine level deviation by tone and channel') in bars
    assert sum(1 for b in report.blocks
               if b['kind'] == 'bars' and b.get('mode') == 'kurtosis') == 1


def test_the_judgment_blocks_are_the_single_reports_own():
    """Composed, not copied: every control and compliance block of the
    random report and every level and deviation block of the sine
    report appears in the mixed one as it is."""
    empty = visualdynamics.Project()
    mixed = mixed_template(empty, links=[]).blocks
    random = random_template(empty, links=[]).blocks
    sine = sine_template(empty, links=[]).blocks

    def judgments(blocks, first, last):
        heads = [i for i, b in enumerate(blocks)
                 if b['kind'] == 'text' and b['text'].startswith(first)]
        tails = [i for i, b in enumerate(blocks)
                 if b['kind'] == 'text' and b['text'].startswith(last)]
        return blocks[heads[0]:tails[0]]

    random_part = judgments(random, '## Control', '## Data Quality')
    sine_part = judgments(sine, '## Level Against Requirement',
                          '## The Shape Of The Recording')
    assert random_part and sine_part
    for block in random_part + sine_part:
        assert block in mixed, block.get('caption') or block['text'][:40]


def test_a_mixed_run_imports_as_the_type_and_generates_its_report(
        window, pump, tmp_path):
    path = _write_run(tmp_path / 'mixed.nc4', random=True)
    window.import_paths([path])
    pump()
    assert window.project_type == 'Random and Sine'
    assert PROJECT_TEMPLATES[window.project_type] == 'mixed'
    window.generate_typed_report()
    pump()
    from visualdynamics.core.report import Report

    reports = [o for o in window.objects.values() if isinstance(o, Report)]
    assert [r.title for r in reports] == ['Random and Sine Test Report']


def test_mixed_run_refuses_a_run_with_no_sweep():
    """Half of what was asked for is not what was asked for."""
    from conftest import fixture_path

    with pytest.raises(ValueError, match='no sine sweep specification'):
        visualdynamics.mixed_run(fixture_path('plate', 'random.nc4'))


@needs_stress
def test_the_one_call_writes_the_report_for_a_mixed_run(tmp_path):
    path = visualdynamics.mixed_report(os.path.join(STRESS, 'mixed.nc4'),
                                       tmp_path / 'mixed.html')
    html = (tmp_path / 'mixed.html').read_text(encoding='utf-8')
    assert str(path) == str(tmp_path / 'mixed.html')
    assert 'Random and Sine Test Report' in html
    for caption in ('Control against specification',
                    'extracted level against the requirement',
                    'Sine level deviation by tone and channel'):
        assert caption in html, caption


@needs_stress
def test_the_real_mixed_run_renders_both_halves(tmp_path):
    """The plate's random with the quiet sweep under it, worked up the
    way the random one-call does, the sweep extracted, and the report
    rendered in a real browser engine: the control comparisons and the
    tone's level figure are both on the page."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    from visualdynamics.report import render_html

    project = visualdynamics.random_vibration_run(
        os.path.join(STRESS, 'mixed.nc4'))
    assert project.project_type == 'Random and Sine'
    project.extract_sine(project.time_history)
    name = project.generate_report('mixed')
    path = tmp_path / 'mixed.html'
    path.write_text(render_html(project[name], project, None,
                                links=project.links), encoding='utf-8')
    QApplication.instance() or QApplication(['x'])
    view = QWebEngineView()
    view.resize(1000, 800)
    view.load(QUrl.fromLocalFile(str(path)))
    try:
        answer = web_read(
            view,
            "(() => { if (document.readyState !== 'complete'"
            "    || !document.getElementById('data')"
            "    || !document.querySelector('.themetoggle')) return null;"
            "return JSON.stringify({captions: Array.from("
            "  document.querySelectorAll('.caption'), c => c.textContent),"
            " canvases: document.querySelectorAll('canvas').length,"
            " verdict: document.querySelectorAll('.verdict').length,"
            " level: (document.querySelector('.testlevel') || {}).textContent"
            "});"
            " })()")
    finally:
        web_close(view)
    found = json.loads(answer)
    captions = ' | '.join(found['captions'])
    assert 'Control against specification' in captions
    assert 'extracted level against the requirement' in captions
    assert 'Sine level deviation by tone and channel' in captions
    assert 'RMS error by control channel' in captions
    assert found['verdict'] == 1
    assert found['level'].startswith(('0 dB', '+', '\u2212')), found['level']
    assert 'Random test level' in found['level'], (
        "the level beside the verdict is the random half's")
    assert found['canvases'] > 8


def test_the_random_report_charts_the_margin_to_abort_not_the_share():
    """The second compliance chart is the signed margin to the abort
    limits (Brandon, 2026-10-01: a share of the band cannot go
    negative, and the question behind the chart was how far)."""
    from visualdynamics.core.report import random_template

    blocks = random_template({}, links=[]).blocks
    bars = [b for b in blocks if b.get('kind') == 'bars']
    assert [b['mode'] for b in bars][:4] == ['error', 'margin', 'error', 'margin']
    assert not any(b.get('mode') == 'lines' for b in blocks)
    text = ' '.join(b.get('text', '') for b in blocks if b.get('kind') == 'text')
    assert '{{figure:Margin to the abort limits, by control channel}}' in text


def test_the_sine_comparison_draws_every_control_channel():
    """One figure per control channel, or a grid past four — never a
    drop-down (Brandon, 2026-10-01)."""
    import numpy as np
    from test_extract_sine import _flat_sweep

    from visualdynamics.core.report import sine_template
    from visualdynamics.core.sine import (
        SineLevelSet,
        SineSweepSpecification,
        SineTone,
        extract_sine,
    )

    history, spec = _flat_sweep(noise=1.0)
    levels = extract_sine(history, spec, cycles=40.0, workers=1, refine=False)
    assert isinstance(levels, SineLevelSet)
    few = sine_template({'Sweep': spec, 'Levels': levels}, links=[]).blocks
    figures = [b for b in few if b.get('kind') == 'plot' and b.get('tone')]
    assert [b.get('channel') for b in figures] == list(spec.response_dof)
    assert all(not b.get('grid') for b in figures)
    tone = spec.tones[0]
    wide = SineSweepSpecification(
        [SineTone(tone.name, tone.start_time, tone.frequency,
                  np.repeat(tone.amplitude[:, :1], 6, axis=1),
                  tone.segment_type, tone.segment_rate)],
        [f'{n}Z+' for n in range(101, 107)], ordinate_unit='m/s**2')
    many = sine_template({'Sweep': wide, 'Levels': levels}, links=[]).blocks
    figures = [b for b in many if b.get('kind') == 'plot' and b.get('tone')]
    assert len(figures) == 1 and figures[0]['grid'] is True


def test_a_six_channel_sine_report_renders_the_grid():
    """Past four control channels the sine comparison is one grid
    figure, each cell the channel's own figure: rendered, not merely
    asked for."""
    import json

    import numpy as np
    from test_extract_sine import _flat_sweep

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.report import sine_template
    from visualdynamics.core.sine import SineSweepSpecification, SineTone, extract_sine
    from visualdynamics.report import render_html

    history, spec = _flat_sweep(noise=1.0)
    dofs = [f'{n}Z+' for n in range(101, 107)]
    tone = spec.tones[0]
    wide = SineSweepSpecification(
        [SineTone(tone.name, tone.start_time, tone.frequency,
                  np.repeat(tone.amplitude[:, :1], 6, axis=1),
                  tone.segment_type, tone.segment_rate)],
        dofs, ordinate_unit='m/s**2')
    signal = np.vstack([history.ordinate] * 3)
    six = TimeHistory(history.abscissa, signal, response_dof=dofs,
                      ordinate_dim=['acceleration'] * 6,
                      ordinate_unit=['m/s**2'] * 6)
    levels = extract_sine(six, wide, cycles=40.0, workers=1, refine=False)
    objects = {'Sweep': wide, 'Record': six, 'Levels': levels}
    html = render_html(sine_template(objects, links=[]), objects)
    data = json.loads(html.split('type="application/json">')[1]
                      .split('</script>')[0])
    grids = [b for b in data['blocks'] if b.get('kind') == 'grid']
    assert len(grids) == 1, [b.get('kind') for b in data['blocks']]
    cells = [cell for row in grids[0]['rows'] for column in row['cells'] for cell in column]
    assert len(cells) == 6
    assert all(len(cell['channels']) == 1 for cell in cells)
    assert sorted(cell['channels'][0]['label'] for cell in cells) == dofs


def test_the_sine_report_draws_no_averaging_frames_on_its_time_figures():
    """The sine extraction demodulates the whole record and averages
    nothing, so the frames an averaging setting would draw on the time
    figures are not this reading's (Brandon, 2026-10-02); the random
    report, which averages, keeps them."""
    from test_extract_sine import _flat_sweep

    from visualdynamics.core.report import random_template, sine_template
    from visualdynamics.report import render_html

    history, spec = _flat_sweep(noise=1.0)
    history.averaging = history.suggest_averaging()
    assert history.averaging is not None

    def time_figures(template):
        objects = {'Sweep': spec, 'Record': history}
        html = render_html(template(objects, links=[]), objects)
        data = json.loads(html.split('type="application/json">')[1]
                          .split('</script>')[0])
        return [b for b in data['blocks'] if b.get('kind') == 'plot'
                and 'time histories' in b.get('caption', '')]

    import json

    sine = time_figures(sine_template)
    assert sine and all('averaging' not in b for b in sine)
    random = time_figures(random_template)
    assert random and all('averaging' in b for b in random)


def test_virtual_channels_lie_across_one_row_of_the_grid():
    """Channels with no direction — a transformation's '1', '2', '3' —
    have nothing to stack by, so they are one row, a column each."""
    from visualdynamics.core.report import channel_grid

    grid = channel_grid(['1', '2', '3'])
    assert grid['columns'] == ['1', '2', '3']
    assert len(grid['rows']) == 1
    assert [[c['channel'] for c in col] for col in grid['rows'][0]['cells']] == [['1'], ['2'], ['3']]
    # physical channels keep their node rows and direction columns
    grid = channel_grid(['101X+', '101Z+', '102Z+'])
    assert grid['columns'] == ['X', 'Z'] and len(grid['rows']) == 2


def test_many_sweeps_on_few_channels_tile_into_a_grid_per_sweep():
    """The count that decides tiling is the whole report's: nine tones
    on three channels is twenty-seven figures, so each tone becomes a
    grid of its three (Brandon, 2026-10-02); one tone on two channels
    stays two figures."""
    import json

    import numpy as np
    from test_extract_sine import _flat_sweep

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.report import sine_template
    from visualdynamics.core.sine import SineSweepSpecification, SineTone, extract_sine
    from visualdynamics.report import render_html

    history, spec = _flat_sweep(noise=1.0)
    base = spec.tones[0]
    dofs = ['1', '2', '3']
    tones = [SineTone(f'Tone {k + 1}', base.start_time, base.frequency,
                      np.repeat(base.amplitude[:, :1], 3, axis=1),
                      base.segment_type, base.segment_rate)
             for k in range(9)]
    nine = SineSweepSpecification(tones, dofs, ordinate_unit='m/s**2')
    blocks = sine_template({'Sweep': nine}, links=[]).blocks
    figures = [b for b in blocks if b.get('kind') == 'plot' and b.get('tone')]
    assert not figures, 'nothing extracted yet: the one symbolic figure'
    three = TimeHistory(history.abscissa, np.vstack([history.ordinate, history.ordinate[:1]]),
                        response_dof=dofs, ordinate_dim=['acceleration'] * 3,
                        ordinate_unit=['m/s**2'] * 3)
    # one tone at a time keeps the extraction quick; the levels carry all nine names
    levels = extract_sine(three, nine, tones=[t.name for t in tones], cycles=40.0,
                          workers=1, refine=False)
    objects = {'Sweep': nine, 'Record': three, 'Levels': levels}
    blocks = sine_template(objects, links=[]).blocks
    figures = [b for b in blocks if b.get('kind') == 'plot' and b.get('tone')]
    assert len(figures) == 9 and all(b.get('grid') is True for b in figures)
    html = render_html(sine_template(objects, links=[]), objects)
    data = json.loads(html.split('type="application/json">')[1].split('</script>')[0])
    grids = [b for b in data['blocks'] if b.get('kind') == 'grid']
    assert len(grids) == 9
    assert all(len(g['rows']) == 1 and len(g['rows'][0]['cells']) == 3 for g in grids)
    assert grids[0]['columns'] == dofs
