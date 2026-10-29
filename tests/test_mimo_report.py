"""The MIMO report template, generated from a small multiple-input run.

The run is two drives through a complex plant, simulated in the time
domain and read by the package's own estimators: the full cross-spectral
matrix of the run, the H1 FRF from the drives, the multiple coherence,
and a specification made from the control channels' own matrix, cross
terms included. Each test builds the report from that project, or from
one with a piece taken away, and reads the sections and the page that
come out: the template computes nothing, so what it owes is checked as
words, and what it draws as figures the page actually holds.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.data import Specification, TimeHistory, autospectrum
from visualdynamics.core.report import (
    OWED,
    TEMPLATE_BUILDERS,
    mimo_template,
)
from visualdynamics.report import render_html

DRIVES = ['1X+', '2X+']
CONTROLS = ['1Z+', '2Z+']

#: every heading of the report of a full run, in order
FULL = ['Test Summary', 'Test Article and Instrumentation', 'Transformation',
        'Measured Data', 'Specification Realizability', 'Conditioning',
        'Drives', 'Matrix Comparison', 'Matrix Grid', 'Cross-Term Verdicts',
        'Data Quality', 'Reference and Partial Coherence', 'Conclusions']

#: the sections whose reading no report block draws
OWED_SECTIONS = ['Transformation', 'Specification Realizability', 'Drives',
                 'Matrix Grid', 'Cross-Term Verdicts',
                 'Reference and Partial Coherence']


def _history(dead=None, gap=None):
    """Two drives through a complex plant, as a time history: white
    noise drives, the second carrying the first three samples late, and
    four short FIR filters for the plant, so every entry of H is complex
    and differs from line to line. `dead` names a control channel
    recorded as zeros, and `gap` one with a sample that is not a
    number."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(11)
    fs, frame = 1024.0, 512
    n = int(16 * fs)
    noise = rng.standard_normal((2, n))
    drives = [noise[0], 0.9 * np.roll(noise[0], 3) + 0.5 * noise[1]]
    taps = {(0, 0): [1.0, 0.5], (0, 1): [0.0, 0.0, 0.7],
            (1, 0): [0.0, 0.3, 0.0, 0.2], (1, 1): [0.8, 0.0, -0.4]}
    responses = [sum(lfilter(taps[(i, j)], 1.0, drives[j])
                     for j in range(2)) for i in range(2)]
    if dead is not None:
        responses[CONTROLS.index(dead)] = np.zeros(n)
    if gap is not None:
        responses[CONTROLS.index(gap)][1000] = np.nan
    history = TimeHistory(np.arange(n) / fs, np.array(drives + responses),
                          response_dof=DRIVES + CONTROLS,
                          ordinate_dim=['voltage'] * 2
                          + ['acceleration'] * 2)
    history.averaging = Averaging(frame_length=frame, overlap=0.5,
                                  frames=2 * n // frame - 1)
    return history


def _specification(cpsd, cross=True):
    """The control channels' own matrix as a specification, read per
    line, with abort limits 6 dB either side of each autospectrum; with
    `cross` False, the autospectra alone."""
    keep = [k for k, pair in enumerate(zip(cpsd.response_dof,
                                           cpsd.reference_dof))
            if set(pair) <= set(CONTROLS)
            and (cross or autospectrum(cpsd, k))]
    values = cpsd.ordinate[keep]
    auto = np.array([autospectrum(cpsd, k) for k in keep])[:, None]
    spec = Specification(
        cpsd.abscissa, values,
        response_dof=[cpsd.response_dof[k] for k in keep],
        reference_dof=[cpsd.reference_dof[k] for k in keep],
        ordinate_dim=[cpsd.ordinate_dim[k] for k in keep],
        abort_lower=np.where(auto, values.real * 10 ** -0.6, np.nan),
        abort_upper=np.where(auto, values.real * 10 ** 0.6, np.nan))
    spec.interpolation = 'bin'
    return spec


def _project(*, frf=True, specification='cross', matrix=True, dead=None,
             gap=None):
    """The run as a project: the record, its control spectra (the full
    matrix, or with `matrix` False the autospectra alone), the FRF from
    the drives, the multiple coherence, a channel table, and the
    specification ('cross', 'auto' for one without cross terms, or None
    for none)."""
    project = visualdynamics.Project()
    history = _history(dead, gap)
    project.add('Run', history)
    project.add('Channels', ChannelTable({
        'channel': [1, 2, 3, 4], 'node': [1, 2, 1, 2],
        'direction': ['X+', 'X+', 'Z+', 'Z+'],
        'unit': ['V', 'V', 'g', 'g'],
        'channel_type': ['voltage', 'voltage', 'acceleration',
                         'acceleration'],
        'comment': ['drive 1', 'drive 2', 'control 1', 'control 2'],
        'sensitivity': ['', '', '10', '10'], 'daq_range': ['', '', '', '']}))
    full = history.compute_cpsds()
    if matrix:
        project.compute_cpsds('Run')
    else:
        project.compute_psds('Run')
    if frf:
        project.add('FRF', history.compute_frfs(references=DRIVES,
                                                method='H1'))
    project.add('Coherence',
                history.compute_multiple_coherence(references=DRIVES))
    if specification is not None:
        project.add('Specification',
                    _specification(full, specification == 'cross'))
    return project


def _report(project):
    return mimo_template(project, object_groups=project.object_groups)


def _headings(report):
    return [block['text'].split('\n', 1)[0][3:] for block in report.blocks
            if block['kind'] == 'text' and block['text'].startswith('## ')]


def _section(report, heading):
    return next(block['text'] for block in report.blocks
                if block['kind'] == 'text'
                and block['text'].startswith(f'## {heading}\n'))


def _page(project, report):
    page = render_html(report, project, object_groups=project.object_groups,
                       unit_system=visualdynamics.SI)
    return json.loads(re.search(r'<script id="data"[^>]*>(.*?)</script>',
                                page, re.DOTALL).group(1))


@pytest.fixture(scope='module')
def full():
    project = _project()
    return project, _report(project)


def test_the_template_is_registered_and_owes_in_the_skeletons_word():
    """'mimo' is a built-in a project generates by name, and an owed
    section reads as the skeleton's owed sections read."""
    from visualdynamics.report.skeleton import OWED as SKELETON_OWED

    assert TEMPLATE_BUILDERS['mimo'] is mimo_template
    assert OWED == SKELETON_OWED
    project = _project()
    name = project.generate_report('mimo')
    assert project[name].title == 'MIMO Random Vibration Test Report'
    with pytest.raises(ValueError, match='mimo'):
        project.generate_report('no such template')


def test_a_full_run_has_every_section_in_order(full):
    _, report = full
    assert _headings(report) == FULL
    for heading in OWED_SECTIONS:
        assert f'**{OWED}:**' in _section(report, heading), heading
    assert 'Not judged' not in ' '.join(
        block.get('text', '') for block in report.blocks)


def test_each_owed_section_names_the_reading_that_gives_it(full):
    _, report = full
    readings = {'Transformation': 'Project.read_transformation',
                'Specification Realizability': 'Psd.realizability',
                'Conditioning': 'Frf.conditioning',
                'Drives': 'core.drive_prediction.predict_drives',
                'Matrix Grid': 'core.matrix_comparison.compare_matrix',
                'Cross-Term Verdicts': 'core.matrix_comparison.compare_matrix',
                'Reference and Partial Coherence':
                    'Project.read_partial_coherence'}
    for heading, reading in readings.items():
        assert f'`{reading}`' in _section(report, heading), heading
    assert 'holds no' not in ' '.join(
        _section(report, heading) for heading in readings)


def test_a_full_run_draws_the_blocks_that_exist(full):
    """The diagonal against its specification, one figure per control
    channel with the caption block its record carries, the two
    compliance charts, the FRF's singular values, the coherence map and
    the kurtosis; and every reference in the text resolves."""
    project, report = full
    comparisons = [block for block in report.blocks
                   if block.get('specification') == '@basis:Specification'
                   and block['kind'] == 'plot']
    assert [block['channel'] for block in comparisons] == CONTROLS
    assert all(block['record_caption'] for block in comparisons)
    payload = _page(project, report)
    captions = [block.get('caption', '') for block in payload['blocks']]
    for wanted in ('Control against specification — 1Z+',
                   'Control against specification — 2Z+',
                   'RMS error by control channel',
                   'Band outside the abort limits, by control channel',
                   'Singular values of the FRF matrix (CMIF)',
                   'Multiple coherence'):
        assert wanted in captions, wanted
    assert any(caption.startswith('Pearson kurtosis') for caption in captions)
    drawn = {block['caption']: block for block in payload['blocks']
             if block.get('caption')}
    facts = dict(drawn['Control against specification — 1Z+']['facts'])
    assert facts['Run'].startswith('Run'), 'the caption block reads the run'
    cmif = drawn['Singular values of the FRF matrix (CMIF)']
    assert len(cmif['curves']) == 2, 'one singular value per drive'
    text = ' '.join(block['html'] for block in payload['blocks']
                    if block['kind'] == 'text')
    assert '{{' not in text, 'every reference resolves'
    assert 'Figure' in text


def test_the_bars_judge_the_diagonal_and_never_a_cross_term(full):
    """A cross term is refused by the level comparison, so the RMS error
    chart has a bar per control channel and none per pair."""
    project, report = full
    payload = _page(project, report)
    bars = next(block for block in payload['blocks']
                if block.get('caption') == 'RMS error by control channel')
    assert sorted(bars['labels']) == CONTROLS


def test_an_empty_project_keeps_the_whole_outline():
    """With nothing to bind, the template itself is read: every
    heading, every figure as a slot to fill, and every owed section
    owed without a word about what is missing."""
    report = mimo_template({})
    assert _headings(report) == FULL
    kinds = [(block['kind'], block.get('mode'), block.get('source'))
             for block in report.blocks]
    assert ('plot', 'cmif', '@basis:Frf') in kinds
    assert ('plot', 'map', '@basis:MultipleCoherence') in kinds
    text = ' '.join(block.get('text', '') for block in report.blocks)
    assert 'holds no' not in text and 'Not judged' not in text
    for heading in OWED_SECTIONS:
        assert f'**{OWED}:**' in _section(report, heading), heading


def test_without_an_frf_conditioning_and_drives_say_what_is_missing():
    project = _project(frf=False)
    report = _report(project)
    assert _headings(report) == FULL
    conditioning = _section(report, 'Conditioning')
    assert f'**{OWED}:**' in conditioning
    assert ('The project holds no FRF from the drives to the control '
            'channels to read it from.') in conditioning
    assert 'The project holds no FRF to read it from.' in \
        _section(report, 'Drives')
    assert not [block for block in report.blocks
                if block.get('mode') == 'cmif']


def test_without_a_specification_nothing_is_judged_against_one():
    project = _project(specification=None)
    report = _report(project)
    assert 'Matrix Comparison' not in _headings(report), \
        'no specification, no diagonal figures and no section over nothing'
    assert ('The project holds no specification to read it from.'
            in _section(report, 'Specification Realizability'))
    assert 'The project holds no specification to read it from.' in \
        _section(report, 'Drives')
    verdicts = _section(report, 'Cross-Term Verdicts')
    assert verdicts == ('## Cross-Term Verdicts\n\n**Not judged:** the '
                        'project holds no specification. No cross term is '
                        'counted as passing.')
    assert _section(report, 'Matrix Grid') == (
        '## Matrix Grid\n\n**Not drawn:** the project holds no '
        'specification.')


def test_control_spectra_with_no_cross_terms_are_not_judged_as_a_matrix():
    """A run analyzed channel by channel holds no cross term: the grid
    is not drawn and the verdicts say why, never that they passed. The
    diagonal is still judged."""
    project = _project(matrix=False)
    report = _report(project)
    for heading in ('Matrix Grid', 'Cross-Term Verdicts'):
        section = _section(report, heading)
        assert OWED not in section
        assert 'the control spectra hold no cross terms' in section
        assert '`TimeHistory.compute_cpsds`' in section
    assert 'No cross term is counted as passing.' in \
        _section(report, 'Cross-Term Verdicts')
    assert 'Matrix Comparison' in _headings(report)


def test_a_specification_with_no_cross_terms_is_not_judged_as_a_matrix():
    project = _project(specification='auto')
    report = _report(project)
    assert _section(report, 'Specification Realizability') == (
        '## Specification Realizability\n\n**Not judged:** the '
        'specification holds no cross terms, so there are no coherences '
        'and phases to be inconsistent.')
    verdicts = _section(report, 'Cross-Term Verdicts')
    assert 'the specification holds no cross terms' in verdicts
    assert 'No cross term is counted as passing.' in verdicts


def test_without_a_coherence_the_partial_coherence_says_so():
    project = _project()
    del project['Coherence']
    report = _report(project)
    assert ('The project holds no multiple coherence to read it from.'
            in _section(report, 'Reference and Partial Coherence'))
    assert 'Multiple coherence (' not in _section(report, 'Data Quality')


def test_a_dead_control_channel_still_renders_and_judges_no_cross_term():
    """A control channel recorded as zeros: the page renders, every
    figure the run can fill is drawn, and the cross terms stay owed
    rather than read as passing."""
    project = _project(dead='2Z+')
    report = _report(project)
    payload = _page(project, report)
    assert any(block.get('caption') == 'Control against specification — 2Z+'
               for block in payload['blocks'])
    assert f'**{OWED}:**' in _section(report, 'Cross-Term Verdicts')


def test_a_sample_that_is_not_a_number_still_renders_and_judges_nothing_more():
    """One sample that is not a number on a control channel: the page
    renders, and the template judges no cross term on its own account
    whatever the spectra read."""
    project = _project(gap='1Z+')
    report = _report(project)
    payload = _page(project, report)
    assert any(block.get('caption') == 'Control against specification — 1Z+'
               for block in payload['blocks'])
    assert 'Not judged' not in _section(report, 'Cross-Term Verdicts')
    assert f'**{OWED}:**' in _section(report, 'Cross-Term Verdicts')
    cmif = next(block for block in payload['blocks']
                if block.get('caption') == 'Singular values of the FRF '
                'matrix (CMIF)')
    drawn = [value for curve in cmif['curves'] for value in curve['y']
             if value is not None]
    assert not drawn, ('every line of the FRF is not a number, and none is '
                       'drawn as one')


def test_many_control_channels_read_as_one_grid(monkeypatch):
    from visualdynamics.core import report as core_report

    monkeypatch.setattr(core_report, 'GRID_ABOVE', 1)
    project = _project()
    report = _report(project)
    grids = [block for block in report.blocks if block.get('grid')]
    assert [block['caption'] for block in grids] == [
        'Control against specification']
    assert 'a row per node and a column per direction' in \
        _section(report, 'Matrix Comparison')
