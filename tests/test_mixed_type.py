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
            " verdict: document.querySelectorAll('.verdict').length});"
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
    assert found['canvases'] > 8
