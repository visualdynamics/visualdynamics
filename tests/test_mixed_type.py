"""The random-and-sine project type: its slots, its reports, its import.

A random environment with a sine sweep under it is the one mixed run
that is a project type of its own (Brandon, 2026-09-24): the project
shows both halves' slots, composed from the two types' lists. It made
one combined report until 2026-10-05; it makes a random report and a
sine report since (Brandon: "retire the combined random sine report"),
each read the way its own test is, and the random one saying what the
sweep does to its figures.
"""

from __future__ import annotations

import json
import os

import pytest
from conftest import web_close, web_read, web_view
from test_project_type import _placeholders
from test_rattlesnake_sine import _write_run

import visualdynamics
from visualdynamics.core.report import (
    PROJECT_TEMPLATES,
    project_expectations,
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
    """The random's slots, what the sine's adds, and two reports."""
    random = _labels('Random Vibration')
    mixed = _labels('Random and Sine')
    assert mixed == random[:-1] + ['Sine Sweep Specification',
                                   'Sine Level Set', 'Report', 'Report']
    # labels repeat by design (the second Specification is the banded
    # one, by ordinal); the icons say which slot is which, and none
    # repeats but the report's, which is two reports
    icons = [icon for _n, _cls, icon, *_rest
             in project_expectations('Random and Sine')]
    assert icons.count('Report') == 2
    rest = [icon for icon in icons if icon != 'Report']
    assert len(set(rest)) == len(rest), 'no other slot twice'


def test_the_tree_shows_both_halves_slots(window, pump):
    window.set_project_type('Random and Sine')
    pump()
    assert _placeholders(window) == _labels('Random and Sine')


def _headings(report):
    return [block['text'].split('\n', 1)[0] for block in report.blocks
            if block['kind'] == 'text' and block['text'].startswith('## ')]


def _text(report):
    return ' '.join(b['text'] for b in report.blocks if b['kind'] == 'text')


def test_the_type_makes_a_random_report_and_a_sine_report():
    assert PROJECT_TEMPLATES['Random and Sine'] == ('random', 'sine')
    assert 'mixed' not in __import__(
        'visualdynamics.core.report', fromlist=['x']).TEMPLATE_BUILDERS


def _worked_up(project_type, sweep=True):
    """A project of `project_type` from a recording, a random
    specification and (with `sweep`) the sweep's, worked up by
    Automatic: every block of its reports has something to bind."""
    import numpy as np
    from test_extract_sine import _recording, _spec

    from visualdynamics.core.data import Specification
    from visualdynamics.core.report import Report

    spec = _spec()
    history = _recording(spec, noise=0.5)
    frequencies = np.logspace(1, 3, 25)
    level = np.full((2, len(frequencies)), 1e-3)
    project = visualdynamics.Project()
    project.add('Time History', history)
    project.add('PSD Specification', Specification(
        abscissa=frequencies, ordinate=level,
        response_dof=list(history.response_dof),
        ordinate_dim=['acceleration**2/frequency'] * 2,
        ordinate_unit=['(m/s**2)**2/Hz'] * 2,
        abort_upper=level * 2.0, abort_lower=level * 0.5))
    if sweep:
        project.add('Sine Specification', spec)
    project.project_type = project_type
    project.work_up()
    return {project[n].title: project[n] for n in project.names
            if isinstance(project[n], Report)}


def test_a_random_report_under_a_sweep_says_what_the_sweep_does():
    """The combined report's two warnings, where the random report is
    read: the tone's power is in the control spectra, and a kurtosis
    under three is the sweep; and the level beside the verdict is the
    random's. A plain random report says none of it."""
    swept = _worked_up('Random and Sine')['Random Vibration Test Report']
    plain = _worked_up('Random Vibration', sweep=False)[
        'Random Vibration Test Report']
    assert '## The Sweep Under the Random' in _headings(swept)
    assert '## The Sweep Under the Random' not in _headings(plain)
    for words in ('may be the tone rather than the random',
                  'a channel under three is the sweep',
                  'the tone is the first suspect'):
        assert words in _text(swept), words
        assert words not in _text(plain), words
    verdict = [b for b in swept.blocks if b['kind'] == 'verdict']
    assert verdict[0]['level_label'] == 'Random test level'
    assert 'level_label' not in next(b for b in plain.blocks
                                     if b['kind'] == 'verdict')


def test_a_sine_report_under_a_random_says_where_the_random_is_judged():
    from test_extract_sine import _spec as _sweep

    under = _worked_up('Random and Sine')['Sine Sweep Test Report']
    alone = sine_template({'Sweep': _sweep()}, links=[])
    assert 'the random is judged in its own report' in _text(under)
    assert 'the random is judged in its own report' not in _text(alone)


def test_a_mixed_run_imports_as_the_type_and_generates_both_reports(
        window, pump, tmp_path):
    path = _write_run(tmp_path / 'mixed.nc4', random=True)
    window.import_paths([path])
    pump()
    assert window.project_type == 'Random and Sine'
    window.generate_typed_report()
    pump()
    from visualdynamics.core.report import Report

    reports = [o for o in window.objects.values() if isinstance(o, Report)]
    assert [r.title for r in reports] == ['Random Vibration Test Report',
                                          'Sine Sweep Test Report']


def test_mixed_run_refuses_a_run_with_no_sweep():
    """Half of what was asked for is not what was asked for."""
    from conftest import fixture_path

    with pytest.raises(ValueError, match='no sine sweep specification'):
        visualdynamics.mixed_run(fixture_path('plate', 'random.nc4'))


def test_the_one_call_writes_two_reports_the_sine_from_the_whole_run(
        tmp_path):
    """`run_report` on a random-and-sine run: `<name>_random.html` and
    `<name>_sine.html`, the random reading only the last `last` seconds
    and the sweep the whole run, since a sweep cut short loses tones."""
    import re

    run = _write_run(tmp_path / 'both.nc4', random=True, seconds=18.0)
    out = tmp_path / 'reports'
    out.mkdir()
    written = visualdynamics.run_report(run, str(out) + '/', last=5.0)
    assert sorted(os.path.basename(p) for p in written) == \
        ['both_random.html', 'both_sine.html']
    pages = {}
    for p in written:
        with open(p, encoding='utf-8') as handle:
            pages[os.path.basename(p)] = handle.read()
    assert 'Random Vibration Test Report' in pages['both_random.html']
    assert 'Sine Sweep Test Report' in pages['both_sine.html']

    def seconds(html):
        return float(re.search(r'over ([0-9.]+) ?s', html).group(1))

    assert seconds(pages['both_random.html']) == pytest.approx(5.0, abs=0.01)
    assert seconds(pages['both_sine.html']) == pytest.approx(18.0, abs=0.01)


@needs_stress
def test_the_real_mixed_run_renders_both_reports(tmp_path):
    """The plate's random with the quiet sweep under it, worked up and
    given its two reports, the random one rendered in a real browser
    engine: the control comparisons, the verdict and the level named as
    the random's; the sine one carrying the tone's level figure."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication

    from visualdynamics.report import render_html

    project = visualdynamics.mixed_run(os.path.join(STRESS, 'mixed.nc4'))
    assert project.project_type == 'Random and Sine'
    random_name, sine_name = [project.generate_report(template)
                              for template in PROJECT_TEMPLATES[
                                  project.project_type]]
    sine_html = render_html(project[sine_name], project, None,
                            links=project.links)
    assert 'extracted level against the requirement' in sine_html
    assert 'Sine level deviation by tone and channel' in sine_html
    path = tmp_path / 'random.html'
    path.write_text(render_html(project[random_name], project, None,
                                links=project.links), encoding='utf-8')
    QApplication.instance() or QApplication(['x'])
    view = web_view()
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
            " verdict: document.querySelectorAll('.verdict').length,"
            " level: (document.querySelector('.testlevel') || {}).textContent"
            "});"
            " })()")
    finally:
        web_close(view)
    found = json.loads(answer)
    captions = ' | '.join(found['captions'])
    assert 'Control against specification' in captions
    assert 'RMS error by control channel' in captions
    assert found['verdict'] == 1
    assert 'Random test level' in found['level'], (
        "the level beside the verdict is the random's")


@pytest.mark.parametrize('template', ['random_template'])
def test_the_random_reports_chart_the_band_outside_abort(template):
    """The second compliance chart is the share of each channel's band
    outside the abort limits, in percent. A signed margin to abort in
    dB replaced it on 2026-10-01 and was taken back out on 2026-10-05
    (Brandon: "I want that back ... I don't want it in the reports")."""
    from visualdynamics.core import report as templates

    blocks = getattr(templates, template)({}, links=[]).blocks
    bars = [b for b in blocks if b.get('kind') == 'bars']
    assert [b['mode'] for b in bars][:4] == ['error', 'lines', 'error', 'lines']
    text = ' '.join(b.get('text', '') for b in blocks if b.get('kind') == 'text')
    assert '{{figure:Band outside the abort limits, by control channel}}' in text
    assert 'margin to abort' not in text.lower()


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
