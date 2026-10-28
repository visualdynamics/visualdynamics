"""A figure or table reference never reaches a reader unresolved.

A report printed '({{figure:Excitation degrees of freedom (force)}})'
for a run with no force channels (Brandon, 2026-10-27): the paragraph
named a figure the run could not have. The templates now name what they
built, and the renderer is the guard behind them: on the reading page a
sentence citing a figure that is not there is left out, and on the edit
page the reference is marked where it stands.
"""

from __future__ import annotations

import re

import pytest
from conftest import fixture_path
from test_rattlesnake_sine import _write_run

import visualdynamics
from visualdynamics.core.report import TEMPLATE_BUILDERS
from visualdynamics.report import _resolve_figures, render_html

LABELED = [('Figure 1', 'Test geometry and measurement locations'),
           ('Figure 2', 'Response degrees of freedom (acceleration)'),
           ('Table 1', 'Instrumentation')]


def test_a_sentence_citing_a_missing_figure_is_left_out():
    text = ('The test geometry ({{figure:Test geometry}}) shows where. '
            'The drive forces ({{figure:Drive forces}}) are what it made. '
            'The response ({{figure:Response}}) reads 1.0 at most.')
    assert _resolve_figures(text, LABELED) == (
        'The test geometry (Figure 1) shows where. '
        'The response (Figure 2) reads 1.0 at most.')


def test_a_heading_or_a_table_row_keeps_its_words():
    text = ('## Results {{figure:Nothing}}\n'
            '| channel | ({{table:Gone}}) |\n'
            'See {{table:Instrumentation}}.')
    assert _resolve_figures(text, LABELED) == (
        '## Results\n| channel | |\nSee Table 1.')


def test_the_editor_marks_a_missing_reference_where_it_stands():
    """A caption typed wrong in the editor is shown, not deleted."""
    text = 'The drive forces ({{figure:Drive forces}}) are what it made.'
    assert _resolve_figures(text, LABELED, edit=True) == (
        'The drive forces ([no figure starts "Drive forces"]) are what '
        'it made.')


def _runs(state, tmp_path):
    """(template, project) for each run a template is written for: a
    random run, a modal run, a shock run and a sine run, as recorded,
    with their force channels removed, or with no geometry."""
    geometry = state != 'no geometry'
    if geometry:
        random = visualdynamics.random_vibration_run(
            fixture_path('plate', 'random.nc4'),
            geometry=fixture_path('plate', 'geometry.npz'), length_unit='m')
    else:
        random = visualdynamics.random_vibration_run(
            fixture_path('plate', 'random.nc4'))
    yield 'random', random
    for kind, project_type, path in (
            ('modal', 'Modal Test', fixture_path('plate', 'modal.nc4')),
            ('shock', 'Shock', fixture_path('plate', 'shock.nc4')),
            ('sine', 'Sine Sweep', _write_run(tmp_path / 'sine.nc4'))):
        project = visualdynamics.Project(project_type=project_type)
        project.import_file(str(path))
        if geometry:
            project.import_file(fixture_path('plate', 'geometry.npz'))
        project.work_up()
        yield kind, project


def _without_force(project):
    for obj in list(project.values()):
        dims = set(map(str, getattr(obj, 'ordinate_dim', ())))
        if 'force' in dims and dims != {'force'} and hasattr(
                obj, 'delete_records'):
            obj.delete_records(dim='force')
    return project


@pytest.mark.parametrize('state', ['as run', 'no force', 'no geometry'])
def test_no_reference_reaches_the_page_as_written(state, tmp_path):
    """Every template on the run it is written for, in each state."""
    for kind, project in _runs(state, tmp_path):
        if state == 'no force':
            _without_force(project)
        report = TEMPLATE_BUILDERS[kind](project, project.object_groups)
        page = render_html(report, project,
                           object_groups=project.object_groups,
                           unit_system=visualdynamics.SI)
        assert not re.search(r'\{\{\s*(figure|table)\s*:', page), \
            f'{kind}, {state}: a reference reached the page as written'


def test_a_quantity_off_the_geometry_draws_no_figure_and_is_not_cited(
        tmp_path):
    """The shock run's voltage channels are on nodes the plate does not
    have: no voltage DOF figure, and the DOF sentence names the response
    figure rather than being dropped for citing the voltage one."""
    [(_kind, project)] = [(k, p) for k, p in _runs('as run', tmp_path)
                          if k == 'shock']
    report = TEMPLATE_BUILDERS['shock'](project, project.object_groups)
    captions = [b['caption'] for b in report.blocks if b['kind'] == 'scene']
    assert not any('(voltage)' in caption for caption in captions)
    page = render_html(report, project, object_groups=project.object_groups,
                       unit_system=visualdynamics.SI)
    assert 'The response degrees of freedom (Figure' in page


def test_a_field_reference_the_page_cannot_fill_never_reaches_a_reader():
    """{{@basis:TimeHistory.sample_rate}} on a project with no time
    history was printed as written (2026-10-28); it goes the way a
    missing figure's citation goes."""
    text = ('Sampled at {{@basis:TimeHistory.sample_rate}} Hz. '
            'Second sentence. See {{figure:Response degrees}}.')
    assert _resolve_figures(text, LABELED) == 'Second sentence. See Figure 2.'
    assert _resolve_figures(text, LABELED, edit=True) == (
        'Sampled at [no value for @basis:TimeHistory.sample_rate] Hz. '
        'Second sentence. See Figure 2.')
    assert _resolve_figures('## Rate {{Run.sample_rate}}', LABELED) == '## Rate'


def test_a_modal_report_from_files_quotes_no_unfilled_field(tmp_path):
    project = visualdynamics.Project(project_type='Modal Test')
    project.import_file(fixture_path('plate', 'frfs.npz'))
    project.import_file(fixture_path('plate', 'geometry.npz'))
    project.work_up()
    report = TEMPLATE_BUILDERS['modal'](project, project.object_groups)
    page = render_html(report, project, object_groups=project.object_groups,
                       unit_system=visualdynamics.SI)
    assert '{{' not in page.split('<script id="data"')[0]
    assert '@basis:TimeHistory' not in page
