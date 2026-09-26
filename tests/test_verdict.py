"""The pass/fail box under the random report's title.

Whether the environment passed, read off the octave-band comparison
(Brandon, 2026-09-20): the share of control channels with more than
a tenth of their band outside the abort limits, failing at a fifth
of them, and the share more than 3 dB off in RMS, failing at a
tenth. One word in one color, and the two readings beside it.
"""

from __future__ import annotations

import json
import re

import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core import compliance
from visualdynamics.core.report import random_template
from visualdynamics.report import render_html


def _rows(abort_percents, errors_db):
    return [(f'{101 + i}Z+', {'lines': 10, 'abort_percent': share,
                              'difference_db': db})
            for i, (share, db) in enumerate(zip(abort_percents, errors_db))]


def test_the_verdict_is_two_shares_against_two_thresholds():
    read = compliance.verdict(_rows([0, 0, 12, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert read['channels'] == 10
    assert read['lines_percent'] == 10.0 and read['rms_percent'] == 0.0
    assert read['passed'] is True
    # at the number exactly, failed: a fifth of the channels out on lines
    read = compliance.verdict(_rows([12, 12, 0, 0, 0, 0, 0, 0, 0, 0], [0] * 10))
    assert read['lines_percent'] == 20.0 and read['passed'] is False
    # a tenth of them off in RMS, either way
    read = compliance.verdict(_rows([0] * 10, [-3.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert read['rms_percent'] == 10.0 and read['passed'] is False
    # exactly on a channel threshold is not out: 10% of the band, 3 dB
    read = compliance.verdict(_rows([10.0, 0], [3.0, 0]))
    assert read['lines_percent'] == 0.0 and read['rms_percent'] == 0.0
    assert read['passed'] is True
    # channels with nothing compared are not counted; none at all is no verdict
    rows = _rows([50, 50], [0, 0]) + [('999Z+', {'lines': 0, 'refused': 'no'})]
    assert compliance.verdict(rows)['channels'] == 2
    assert compliance.verdict([('999Z+', {'lines': 0})])['passed'] is None


def _payload(project):
    page = render_html(random_template(project, links=project.links), project,
                       links=project.links, unit_system=visualdynamics.SI)
    return json.loads(re.search(
        r'<script id="data"[^>]*>(.*?)</script>', page, re.DOTALL).group(1))


def test_the_random_report_opens_on_its_verdict():
    project = visualdynamics.random_vibration_run(fixture_path('plate', 'random.nc4'))
    first = random_template(project, links=project.links).blocks[0]
    assert first == {'kind': 'verdict', 'source': '@basis:OctaveSpecification',
                     'measured': '@basis:OctavePsd'}
    blocks = _payload(project)['blocks']
    assert blocks[0]['kind'] == 'verdict', 'under the title, before a word'
    box = blocks[0]
    assert box['channels'] == 8 and 'label' not in box, 'not a numbered figure'
    _spec, octave_spec = project.specifications
    _psds, octave = project.psds
    read = compliance.verdict(compliance.compare_all(octave_spec, octave))
    assert box['passed'] == read['passed']
    assert box['lines_percent'] == pytest.approx(read['lines_percent'])
    assert box['rms_percent'] == pytest.approx(read['rms_percent'])
    assert (box['lines_fail_percent'], box['rms_fail_percent']) == (20.0, 10.0)


def test_a_run_with_no_octave_comparison_has_no_box():
    thin = visualdynamics.Project()
    thin.import_file(fixture_path('plate', 'random.nc4'))
    thin.compute_psds('Time History')
    blocks = _payload(thin)['blocks']
    assert not [b for b in blocks if b['kind'] == 'verdict']
    assert random_template({}).blocks[0]['kind'] == 'verdict', 'the outline keeps its slot'


def test_the_page_paints_the_box_in_its_two_colors():
    from visualdynamics.report.page import _CSS, _JS

    assert "else if (block.kind === 'verdict') verdictBlock(block);" in _JS
    assert "word.textContent = block.passed ? 'PASS' : 'FAIL';" in _JS
    assert "s.className = 'verdict ' + (block.passed ? 'pass' : 'fail');" in _JS
    assert '.verdict.pass { border-color: #2e8b57;' in _CSS
    assert '.verdict.fail { border-color: #c0392b;' in _CSS


def test_the_editor_can_insert_one():
    from visualdynamics.core.report import BLOCK_KINDS, insert_options
    from visualdynamics.gui.report_editor import BLOCK_TEMPLATES

    assert 'verdict' in BLOCK_KINDS
    assert ('verdict', 'Pass/Fail Box') in insert_options({})
    template = BLOCK_TEMPLATES['verdict']
    assert template['kind'] == 'verdict' and template['_bind'] == ('source', 'measured')


def test_octave_band_figures_read_on_a_log_axis_and_narrowband_ones_linear():
    """Brandon, 2026-09-20: linear for the narrowband figures, log for
    the octave-band ones — the specification figure and the comparison
    both, every x they carry in decades."""
    import numpy as np

    project = visualdynamics.random_vibration_run(fixture_path('plate', 'random.nc4'))
    grids = {b['caption']: b for b in _payload(project)['blocks'] if b['kind'] == 'grid'}
    narrow = grids['Control against specification']['rows'][0]['cells'][0][0]
    banded = grids['Control against specification, octave bands']['rows'][0]['cells'][0][0]
    assert not narrow.get('logx') and banded['logx'] is True
    _psds, octave = project.psds
    _spec, octave_spec = project.specifications
    assert banded['x'][1] == pytest.approx(np.log10(octave.abscissa[1]), abs=1e-5)
    assert banded['home_x'] == pytest.approx(
        list(np.log10([octave_spec.bin_edges()[k] for k in
                       (int(np.flatnonzero(octave_spec.written(0))[0]),
                        int(np.flatnonzero(octave_spec.written(0))[-1]) + 1)])), rel=1e-5)
    assert banded['edges'][0] == pytest.approx(np.log10(octave.bin_edges()[0]), abs=1e-5)
    # and the narrowband comparison stays linear
    assert not grids['Control against specification'][
        'rows'][0]['cells'][0][0].get('logx')
    # the specification is no longer drawn on its own, in either form
    assert not [key for key in grids if key.startswith('Test specification')]


def test_the_box_carries_the_test_level_it_was_judged_at():
    """Left of PASS/FAIL, the test level every comparison was judged at
    (Brandon, 2026-09-26) — the negative of the scale added, and
    whether it was detected or set."""
    from test_comparison_scale import _measured, _spec

    from visualdynamics.report import _verdict_block

    channels = tuple(f'{n}Z+' for n in range(101, 109))
    spec = _spec(channels)
    run = _measured(spec, [6.0] * 8, wiggle=0.3)
    box = _verdict_block(spec, run)
    assert box['test_level_db'] == -6.0 and box['level_source'] == 'detected'
    run.scale_db = 3
    box = _verdict_block(spec, run)
    assert box['test_level_db'] == -3.0 and box['level_source'] == 'set'
    full = _verdict_block(spec, _measured(spec, [0.0] * 8, wiggle=0.3))
    assert full['test_level_db'] == 0.0
    assert str(full['test_level_db']) == '0.0', 'never a negative zero'


def test_a_mixed_report_says_the_level_is_the_randoms():
    """The mixed report's verdict and level are its random half's; the
    sine is compared as measured. The box says which (Brandon,
    2026-09-26)."""
    from test_comparison_scale import _measured, _spec

    from visualdynamics.core.report import mixed_template, random_template
    from visualdynamics.report import _verdict_block
    from visualdynamics.report.page import _JS

    mixed = [b for b in mixed_template({}).blocks if b['kind'] == 'verdict']
    assert mixed[0]['level_label'] == 'Random test level'
    plain = [b for b in random_template({}).blocks if b['kind'] == 'verdict']
    assert 'level_label' not in plain[0], 'a random report says Test level'
    spec = _spec(('101Z+', '102Z+'))
    run = _measured(spec, [0.0, 0.0], wiggle=0.3)
    assert _verdict_block(spec, run)['level_label'] == 'Test level'
    assert _verdict_block(spec, run, 'Random test level')['level_label'] == \
        'Random test level'
    assert "const label = block.level_label || 'Test level';" in _JS


def test_the_page_puts_the_level_left_of_the_verdict():
    from visualdynamics.report.page import _CSS, _JS

    body = _JS[_JS.index('function verdictBlock'):]
    body = body[:body.index('\n}\n')]
    assert body.index("level.className = 'testlevel'") < body.index(
        "s.className = 'verdict '"), 'the level box goes in first'
    assert "big.textContent = testLevelText(block.test_level_db);" in body
    assert '.testlevel .word { font-size: 2rem; font-weight: 700;' in _CSS
    assert '.verdictrow { display: flex;' in _CSS
