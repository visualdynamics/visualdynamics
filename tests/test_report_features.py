"""Report features: optional additions to a report, off until enabled.

The framework (`report.features`): the registry and its refusals, the
presets, the requirement check, persistence with the report and the
template, the Report Features panel in the editor, and the one
promise everything rests on: a report with nothing enabled renders
byte for byte as it did before features existed. Then the one feature
built on it, the plot conventions (`report.plot_conventions`), on and
off.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import fixture_path, web_close, web_read, web_view

import visualdynamics
from visualdynamics.core.data import Psd, ShockSpecification, Specification
from visualdynamics.core.report import Report
from visualdynamics.report import features as rf
from visualdynamics.report import page, plot_conventions, render_html

LIVE = ('skeleton_sections', 'plot_conventions', 'shared_decades')
#: the plot conventions and their option
CONVENTIONS = ('plot_conventions', 'shared_decades')

#: the catalogue the plan registers: one feature per report gap, plus
#: the shared-decades option of the plot conventions
CATALOGUE = {
    'test_requirement', 'skeleton_sections', 'report_identity',
    'data_index', 'findings', 'record_captions', 'plot_conventions',
    'shared_decades', 'check_blocks', 'prediction', 'cross_axis',
    'run_sections', 'calibration_in_date', 'flash_report',
    'control_uncertainty'}


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


def _cells(block):
    if block['kind'] == 'plot':
        return [block]
    if block['kind'] == 'grid':
        return [cell for row in block['rows'] for column in row['cells']
                for cell in column]
    return []


@pytest.fixture(scope='module')
def run():
    """A random run worked up, with its report: the report every test
    of the rendered page reads."""
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'))
    name = project.generate_report('random')
    return project, name


def _render(project, name):
    return render_html(project[name], project,
                       unit_system=visualdynamics.SI,
                       object_groups=project.object_groups)


# ---- the registry ---------------------------------------------------------


def test_every_feature_is_off_by_default():
    """Off by default is the contract: a new report starts with none,
    and the registry's defaults say so. A feature switched on by
    default has to change this test on purpose."""
    assert rf.defaults() == ()
    assert not any(feature.default for feature in rf.FEATURES.values())
    assert Report().features == []


def test_the_catalogue_is_the_road_map_with_the_built_features_live():
    assert set(rf.FEATURES) == CATALOGUE
    live = tuple(key for key, f in rf.FEATURES.items() if f.available)
    assert live == LIVE, 'the built features, nothing else'
    for feature in rf.FEATURES.values():
        assert feature.group in rf.GROUPS
        assert feature.title and feature.description.endswith('.')
    assert {f.group for f in rf.FEATURES.values()} == set(rf.GROUPS), \
        'every group has a feature in it'


def test_a_requirement_is_a_feature_listed_first_or_a_capability():
    """`enabled` walks the registry in order, so a feature must come
    after every feature it needs."""
    order = list(rf.FEATURES)
    for key, feature in rf.FEATURES.items():
        for need in feature.requires:
            assert need in rf.FEATURES or need in rf.CAPABILITIES, need
            if need in rf.FEATURES:
                assert order.index(need) < order.index(key), (key, need)


def test_a_group_outside_the_list_is_refused():
    with pytest.raises(ValueError, match='group'):
        rf.ReportFeature('x', 'X', 'Appendices', 'Y.')


def test_an_unknown_feature_is_refused_by_name():
    report = Report()
    with pytest.raises(ValueError, match="unknown report feature 'plot_convention'"):
        rf.configure(report, enable=['plot_convention'])
    with pytest.raises(ValueError, match="unknown report feature 'nope'"):
        rf.configure(report, disable=['nope'])
    with pytest.raises(TypeError, match='key, a str'):
        rf.configure(report, enable=[3])
    with pytest.raises(ValueError, match='unknown preset'):
        rf.configure(report, preset='qualification')
    with pytest.raises(ValueError, match='unknown report feature'):
        rf.unmet('nope')
    assert report.features == []


def test_a_lone_string_is_one_key_not_its_letters():
    report = Report()
    assert rf.configure(report, enable='plot_conventions').enabled == (
        'plot_conventions',)


def test_a_feature_not_yet_built_is_refused_and_says_so():
    report = Report()
    with pytest.raises(ValueError, match='Findings register.*not yet available'):
        rf.configure(report, enable=['findings'])
    assert rf.reason('findings') == 'not yet available'
    # and an unbuilt feature names what it will wait on
    assert rf.reason('cross_axis') == (
        'not yet available; waits on Requirement and results')


def test_an_unmet_requirement_is_refused_and_named():
    report = Report()
    with pytest.raises(ValueError,
                       match='Shared decades.*needs Plot conventions'):
        rf.configure(report, enable=['shared_decades'])
    assert rf.reason('shared_decades') == 'needs Plot conventions'
    assert rf.reason('shared_decades', ['plot_conventions']) == ''
    # asked for together, the one needed comes first whatever the order
    assert rf.configure(report, enable=['shared_decades',
                                        'plot_conventions']).enabled == CONVENTIONS


def test_a_missing_capability_is_named(monkeypatch):
    """A built feature can wait on something the package itself must
    have; the reason names it."""
    built = rf.ReportFeature('probe', 'Probe', 'Results', 'Probe.',
                             requires=('multi_point',))
    monkeypatch.setitem(rf.FEATURES, 'probe', built)
    monkeypatch.setitem(rf.CAPABILITIES, 'multi_point', rf.Capability(
        'multi_point', 'the multi-point composite', lambda: False))
    assert rf.unmet('probe') == ['needs the multi-point composite']
    with pytest.raises(ValueError, match='needs the multi-point composite'):
        rf.configure(Report(), enable=['probe'])
    monkeypatch.setitem(rf.CAPABILITIES, 'multi_point', rf.Capability(
        'multi_point', 'the multi-point composite', lambda: True))
    assert rf.unmet('probe') == []


def test_a_capability_is_probed_by_its_module():
    assert rf._has_module('visualdynamics.core.data')() is True
    assert rf._has_module('visualdynamics.core.no_such_module')() is False
    assert rf._has_module('visualdynamics.no_such_package.module')() is False


def test_a_refused_call_changes_nothing():
    report = Report(features=['plot_conventions'])
    for bad in ({'enable': ['plot_conventions', 'findings']},
                {'disable': ['plot_conventions'], 'enable': ['findings']},
                {'enable': ['shared_decades'], 'disable': ['shared_decades']}):
        with pytest.raises(ValueError):
            rf.configure(report, **bad)
        assert report.features == ['plot_conventions'], bad


def test_turning_a_feature_off_takes_what_needs_it_with_a_note():
    report = Report()
    rf.configure(report, enable=CONVENTIONS)
    selection = rf.configure(report, disable=['plot_conventions'])
    assert selection.enabled == () and report.features == []
    assert selection.notes == (
        'Shared decades turned off with Plot conventions, which it needs',)


def test_the_presets_enable_what_they_can_and_say_what_they_skipped():
    report = Report()
    selection = rf.configure(report, preset='test_report')
    assert selection.enabled == LIVE
    skipped = [key for key in rf.PRESETS['test_report'].features
               if key not in LIVE]
    assert len(selection.notes) == len(skipped) == 8
    assert selection.notes[1] == ('Test report: Findings register skipped, '
                                  'not yet available')
    flash = rf.configure(report, preset='flash_report')
    assert flash.enabled == () and report.features == []
    assert len(flash.notes) == 4
    assert rf.configure(report, preset='default') == rf.FeatureSelection((), ())


def test_a_preset_naming_an_unknown_feature_skips_it(monkeypatch):
    monkeypatch.setitem(rf.PRESETS, 'odd', rf.Preset(
        'odd', 'Odd', 'Odd.', ('plot_conventions', 'tea_break')))
    selection = rf.configure(Report(), preset='odd')
    assert selection.enabled == ('plot_conventions',)
    assert selection.notes == ("Odd: 'tea_break' skipped, not a known feature",)


def test_a_preset_replaces_and_the_set_is_read_back_as_a_preset():
    report = Report(features=['plot_conventions'])
    rf.configure(report, preset='test_report')
    assert rf.preset_of(report.features) == 'test_report'
    assert rf.preset_of([]) == 'default'
    # one feature off the test report's set is a custom choice
    assert rf.preset_of(['plot_conventions']) is None


def test_enabled_reads_only_what_can_act():
    """A key stored on the report that the registry does not know, an
    unbuilt one, or one whose requirement is off does nothing."""
    report = Report(features=['from_a_later_version', 'findings',
                              'shared_decades'])
    assert rf.enabled(report) == ()
    report.features = [*report.features, 'plot_conventions']
    assert rf.enabled(report) == CONVENTIONS
    # a later version's key is kept through a change made here
    rf.configure(report, disable=['shared_decades'])
    assert 'from_a_later_version' in report.features


def test_the_models_features_are_keys_kept_once():
    report = Report(features=['plot_conventions', 'plot_conventions'])
    assert report.features == ['plot_conventions']
    with pytest.raises(TypeError, match='not one string'):
        report.features = 'plot_conventions'
    with pytest.raises(TypeError, match='key, a str'):
        report.features = [None]


# ---- the project verb -------------------------------------------------------


def test_the_project_verb_is_journaled_and_refuses_in_the_core(run):
    project, name = run
    try:
        selection = project.report_features(name, preset='test_report')
        assert selection.enabled == LIVE
        assert project.journal[-1] == (
            f"project.report_features({name!r}, preset='test_report')")
        before = list(project.journal)
        with pytest.raises(ValueError, match='not yet available'):
            project.report_features(name, enable=['findings'])
        assert project.journal == before, 'a refusal is not journaled'
        with pytest.raises(TypeError, match='not a report'):
            project.report_features('Specification', enable=['plot_conventions'])
    finally:
        project[name].features = []


# ---- persistence ------------------------------------------------------------


def test_features_round_trip_through_the_project_file(tmp_path):
    from visualdynamics import io

    report = Report('R', [{'kind': 'text', 'text': 'x'}],
                    features=['plot_conventions', 'from_a_later_version'])
    io.save(report, str(tmp_path / 'r.vdyn'))
    back = io.load(str(tmp_path / 'r.vdyn'))
    assert back.features == ['plot_conventions', 'from_a_later_version'], \
        'a key this version does not know survives the round trip'


def test_a_report_with_none_is_stored_as_it_always_was(tmp_path):
    """No `features` attribute is written for a report with none, so
    the file is the one written before features existed, and a file
    without the attribute loads with none."""
    import h5py

    from visualdynamics import io

    io.save(Report('R'), str(tmp_path / 'r.vdyn'))
    with h5py.File(tmp_path / 'r.vdyn') as handle:
        attributes = set()
        handle.visititems(lambda _n, item: attributes.update(item.attrs))
    assert 'features' not in attributes
    assert io.load(str(tmp_path / 'r.vdyn')).features == []


def test_the_template_carries_the_choice(tmp_path):
    from visualdynamics.io import report_template

    report = Report('House', [{'kind': 'text', 'text': '# Results'}],
                    features=list(LIVE))
    report_template.save(report, tmp_path / 'house.vdreport')
    assert report_template.load(tmp_path / 'house.vdreport').features == list(LIVE)


def test_a_template_with_none_is_the_file_it_always_was(tmp_path):
    from visualdynamics.io import report_template

    report = Report('House', [{'kind': 'text', 'text': '# Results'}])
    path = tmp_path / 'house.vdreport'
    report_template.save(report, path)
    expected = json.dumps({report_template.MARKER: 1, 'title': 'House',
                           'marking': 'UNCLASSIFIED', 'marking_color': 'ink',
                           'blocks': report.blocks}, indent=1) + '\n'
    assert path.read_text(encoding='utf-8') == expected
    assert report_template.load(path).features == [], 'an old template: none'


def test_a_template_whose_features_are_not_keys_is_refused(tmp_path):
    from visualdynamics.io import report_template

    path = tmp_path / 'bad.vdreport'
    path.write_text(json.dumps({report_template.MARKER: 1,
                                'features': 'plot_conventions'}),
                    encoding='utf-8')
    with pytest.raises(ValueError, match='list of feature keys'):
        report_template.load(path)


# ---- nothing enabled: the page it always was ------------------------------


def test_nothing_enabled_is_byte_identical(run):
    """The promise: with no feature enabled the page carries the page
    script and stylesheet untouched and nothing a feature adds, and
    turning features on and off again leaves no trace."""
    project, name = run
    report = project[name]
    assert report.features == []
    plain = _render(project, name)
    assert f'<style>{page._CSS}</style>' in plain
    assert f'<script>{page._JS}</script>' in plain
    data = _payload(plain)
    assert 'line_patterns' not in data and 'end_labels' not in data
    assert not any('home' in cell for block in data['blocks']
                   for cell in _cells(block))
    try:
        rf.configure(report, enable=LIVE)
        assert _render(project, name) != plain
        # keys that cannot act are inert
        report.features = ['findings', 'from_a_later_version']
        assert _render(project, name) == plain
    finally:
        report.features = []
    assert _render(project, name) == plain


def test_the_page_code_is_spliced_only_where_anchored():
    css, js = page.page_parts(())
    assert css is page._CSS and js is page._JS
    for key, (splices, functions, _style) in page.FEATURE_PAGE.items():
        for anchor, _replacement in splices:
            assert page._JS.count(anchor) == 1, (key, anchor)
        # a feature adds a script, a stylesheet, or both
        assert functions.strip() or _style.strip()
    css, js = page.page_parts(LIVE)
    for call in ('legendStyle(block, curve, i, item)',
                 'lineDash(block, curve, index)', 'endLabels(g, block',
                 'fixedWindow(block, home, full)'):
        assert call in js
    assert css.startswith(page._CSS) and '.legend svg.swatch' in css


def test_an_anchor_that_moved_is_refused(monkeypatch):
    monkeypatch.setitem(page.FEATURE_PAGE, 'plot_conventions',
                        ((('  no such line\n', 'x'),), '', ''))
    with pytest.raises(RuntimeError, match='found 0 times'):
        page.page_parts(['plot_conventions'])


# ---- the plot conventions ---------------------------------------------------


def _narrowband():
    """The plate's specification and the PSDs measured against it."""
    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = next(o for o in loaded.values() if isinstance(o, Specification))
    history = next(o for o in loaded.values()
                   if isinstance(o, visualdynamics.TimeHistory))
    return {'Spec': spec, 'PSDs': history.compute_psds(),
            'History': history}


def _comparison_report(**features):
    return Report('R', [
        {'kind': 'plot', 'source': 'PSDs', 'specification': 'Spec',
         'mode': 'curves', 'caption': 'comparison'},
        {'kind': 'plot', 'source': 'History', 'mode': 'curves',
         'caption': 'time'}], **features)


def test_an_asd_comparison_opens_log_log_only_with_the_feature():
    objects = _narrowband()
    off = _payload(render_html(_comparison_report(), objects))['blocks']
    on = _payload(render_html(_comparison_report(
        features=['plot_conventions']), objects))['blocks']
    assert not off[0].get('logx'), 'a narrowband comparison opens linear today'
    assert on[0]['logx'] is True and on[0]['logy'] is True
    # the same grid, in decades; the line at 0 Hz has no place on a log
    # axis and is a gap
    linear = np.array(off[0]['x'], dtype=float)
    decades = np.array([np.nan if v is None else v for v in on[0]['x']])
    assert linear[0] == 0.0 and np.isnan(decades[0])
    assert np.allclose(decades[1:], np.log10(linear[1:]), atol=1e-6)
    assert np.allclose(on[0]['home_x'], np.log10(off[0]['home_x']))
    # the ordinates are not touched
    assert on[0]['channels'][0]['response'] == off[0]['channels'][0]['response']
    # a time history is not an ASD: its axes stay
    assert on[1]['logx'] is False and on[1]['x'] == off[1]['x']


def test_an_srs_comparison_stays_on_its_log_axes():
    frequencies = np.geomspace(20.0, 2000.0, 13)
    required = 10.0 * (frequencies / 20.0) ** 0.5
    spec = ShockSpecification(
        abscissa=frequencies, ordinate=np.atleast_2d(required),
        response_dof=['101Z+'], ordinate_dim=['acceleration'],
        abort_upper=np.atleast_2d(required * 2.0),
        abort_lower=np.atleast_2d(required * 0.5))
    measured = visualdynamics.Srs(
        abscissa=frequencies, ordinate=np.atleast_2d(required * 1.2),
        response_dof=['101Z+'], ordinate_dim=['acceleration'])
    objects = {'Spec': spec, 'SRS': measured}
    block = {'kind': 'plot', 'source': 'SRS', 'specification': 'Spec'}
    off = _payload(render_html(Report('R', [block]), objects))['blocks'][0]
    on = _payload(render_html(Report('R', [block],
                                     features=['plot_conventions']),
                              objects))['blocks'][0]
    assert off['logx'] is True, 'an SRS is drawn in decades already'
    assert on['x'] == off['x'], 'and is not put into decades twice'
    assert plot_conventions.figure_kind(block, objects) == 'srs'


def test_the_kinds_the_conventions_know():
    objects = _narrowband()
    kind = plot_conventions.figure_kind
    assert kind({'kind': 'plot', 'source': 'PSDs'}, objects) == 'asd'
    assert kind({'kind': 'plot', 'source': 'Spec'}, objects) == 'asd'
    assert kind({'kind': 'plot', 'source': 'History'}, objects) is None
    assert kind({'kind': 'plot', 'source': 'PSDs', 'mode': 'stage'},
                objects) is None
    assert kind({'kind': 'bars', 'source': 'Spec'}, objects) is None
    assert kind({'kind': 'plot', 'source': 'missing'}, objects) is None


def test_the_page_carries_the_line_patterns_with_the_feature():
    from visualdynamics.plot import END_LABEL_CURVES, LINE_PATTERNS

    objects = _narrowband()
    data = _payload(render_html(_comparison_report(
        features=['plot_conventions']), objects))
    assert data['line_patterns'] == [list(p) for p in LINE_PATTERNS]
    assert data['end_labels'] == END_LABEL_CURVES == 8
    assert len({tuple(p) for p in LINE_PATTERNS}) == len(LINE_PATTERNS) == 8, \
        'eight curves labeled at their ends, eight different styles'
    assert LINE_PATTERNS[0] == (), 'the first curve is solid'


def _figure(xs, ys, ylabel='acceleration PSD [g²/Hz]', home_x=None):
    built = {'kind': 'plot', 'logx': True, 'logy': True, 'x': xs,
             'xlabel': 'frequency [Hz]', 'ylabel': ylabel,
             'curves': [{'label': 'a', 'x': None, 'y': ys}]}
    if home_x is not None:
        built['home_x'] = home_x
    return built


def test_shared_decades_cover_every_figure_of_a_kind_in_whole_decades():
    first = _figure([1.0, 1.5, 2.5], [-3.2, -1.1, -2.0])
    second = _figure([1.3, 2.0, 3.3], [-4.5, -2.0, None])
    force = _figure([1.0, 2.0], [0.5, 1.5], ylabel='force PSD [lbf²/Hz]')
    # nothing finite: it widens nothing, and opens where its kind opens
    blank = _figure([0.5, 9.0], [None, float('nan')])
    alone = _figure([1.0, 2.0], [None, None], ylabel='velocity PSD [(in/s)²/Hz]')
    srs = _figure([1.0, 3.0], [0.2, 2.7], ylabel='acceleration [g]')
    figures = [(first, 'asd'), (second, 'asd'), (force, 'asd'),
               (blank, 'asd'), (alone, 'asd'), (srs, 'srs'),
               (_figure([1.0], [1.0]), None)]
    plot_conventions.share_decades(figures)
    shared = {'x0': 1.0, 'x1': 4.0, 'y0': -5.0, 'y1': -1.0}
    assert first['home'] == shared and second['home'] == shared
    assert blank['home'] == shared
    # another quantity is never forced onto the acceleration's scale
    assert force['home'] == {'x0': 1.0, 'x1': 2.0, 'y0': 0.0, 'y1': 2.0}
    assert 'home' not in alone, 'a kind with nothing finite has no window'
    assert srs['home'] == {'x0': 1.0, 'x1': 3.0, 'y0': 0.0, 'y1': 3.0}
    assert 'home' not in figures[-1][0], 'not a kind the decades apply to'


def test_shared_decades_read_a_comparison_on_its_opening_band():
    """A comparison opens on its specification's band, not on the
    measurement's lines to Nyquist; values outside the band do not
    widen the shared window, and one decade is the least it spans."""
    figure = _figure([1.0, 1.5, 2.0, 3.7], [-2.5, -2.4, -2.2, -9.0],
                     home_x=[1.0, 2.0])
    plot_conventions.share_decades([(figure, 'asd')])
    assert figure['home'] == {'x0': 1.0, 'x1': 2.0, 'y0': -3.0, 'y1': -2.0}
    flat = _figure([2.0, 2.0], [-3.0, -3.0])
    plot_conventions.share_decades([(flat, 'asd')])
    assert flat['home'] == {'x0': 2.0, 'x1': 3.0, 'y0': -3.0, 'y1': -2.0}


def test_a_target_on_its_own_grid_is_read_on_that_grid():
    """A banded specification is drawn on its own bands, so its limits
    are read on its own abscissa, not the measurement's."""
    figure = _figure([1.0, 1.5, 2.0, 3.0], [-2.0, -2.0, -2.0, -2.0],
                     home_x=[1.0, 2.0])
    figure['curves'] = [{'label': 'spec', 'x': [1.0, 3.0], 'y': [-2.2, -2.2],
                         'gray': True}]
    figure['channels'] = [{'y': [-2.2, -2.2], 'response': [-2.0] * 4,
                           'responses': [{'y': [-1.5, -2.0, -2.0, -6.0]}],
                           'zones': [{'lower': [-3.4, -9.0], 'upper': None}]}]
    plot_conventions.share_decades([(figure, 'asd')])
    # the limit at 3.0 decades and the event at 3.0 are outside the
    # band; the limit at 1.0 decade and the event at 1.0 are in it
    assert figure['home'] == {'x0': 1.0, 'x1': 2.0, 'y0': -4.0, 'y1': -1.0}


def test_only_log_log_figures_share_decades():
    on_log_x = _figure([1.0, 2.0], [-1.0, 1.0])
    on_log_x['logy'] = False
    plot_conventions.share_decades([(on_log_x, 'asd')])
    assert 'home' not in on_log_x, 'a ratio in decibels keeps its own scale'


def test_a_grid_and_its_mode_marks_go_into_decades():
    cell = {'kind': 'plot', 'logy': True, 'x': [10.0, 100.0],
            'curves': [{'label': 'a', 'x': None, 'y': [0.0, 1.0]}],
            'marks': [10.0, 1000.0, 0.0, 0.5]}
    grid = {'kind': 'grid', 'rows': [{'cells': [[cell]]}]}
    payload = {}
    plot_conventions.apply(payload, [(grid, 'asd')])
    assert cell['logx'] is True and cell['x'] == [1.0, 2.0]
    assert cell['marks'] == pytest.approx([1.0, 3.0, -0.30103]), \
        'a bookmark at 0 Hz has no place; one under 1 Hz does'
    assert payload['end_labels'] == 8


def test_dead_nan_and_empty_inputs_under_the_conventions():
    """An all-zero channel has nothing to draw on a log axis, so its
    figure goes to log frequency and takes no part in the shared
    window; a channel with one NaN line still opens on the shared
    decades; an empty report and an empty request change nothing."""
    frequencies = np.linspace(10.0, 2000.0, 200)
    nan = np.full((1, 200), 0.01)
    nan[0, 50] = np.nan
    objects = {
        'Dead': Psd(abscissa=frequencies, ordinate=np.zeros((1, 200)),
                    response_dof=['1X+'],
                    ordinate_dim=['acceleration**2/frequency']),
        'Gap': Psd(abscissa=frequencies, ordinate=nan, response_dof=['1Y+'],
                   ordinate_dim=['acceleration**2/frequency'])}
    report = Report('R', [{'kind': 'plot', 'source': 'Dead'},
                          {'kind': 'plot', 'source': 'Gap'}],
                    features=list(CONVENTIONS))
    dead, gap = _payload(render_html(report, objects,
                                     unit_system=visualdynamics.SI))['blocks']
    assert dead['logx'] is True and gap['logx'] is True
    assert all(v is None for v in dead['curves'][0]['y'])
    assert gap['home'] == {'x0': 1.0, 'x1': 4.0, 'y0': -2.0, 'y1': -1.0}
    assert dead['home'] == gap['home'], 'it opens where its kind opens'
    empty = Report('R', features=list(CONVENTIONS))
    assert _payload(render_html(empty, {}))['blocks'] == []
    assert rf.configure(empty, enable=[]) == rf.FeatureSelection(CONVENTIONS, ())


def test_the_random_report_shares_its_decades(run):
    project, name = run
    try:
        project.report_features(name, enable=CONVENTIONS)
        data = _payload(_render(project, name))
    finally:
        project[name].features = []
    homes = {}
    for block in data['blocks']:
        for cell in _cells(block):
            if cell.get('home'):
                homes.setdefault(cell['ylabel'], set()).add(
                    tuple(cell['home'].values()))
                assert cell['logx'] and cell['logy']
                assert all(float(v).is_integer() for v in cell['home'].values())
    assert homes, 'the control figures carry a shared window'
    assert all(len(windows) == 1 for windows in homes.values()), homes


def test_the_figures_draw_their_styles_in_a_real_browser_engine(tmp_path):
    """The page with the conventions on: three PSD curves take three
    patterns, the legend draws each one, the end labels are written
    for a figure of three curves and not for one of more than eight,
    and the shared window is the window the figure opens on."""
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication

    frequencies = np.linspace(10.0, 2000.0, 200)
    values = np.vstack([np.full(200, 0.01 * (k + 1)) for k in range(3)])
    psds = Psd(abscissa=frequencies, ordinate=values,
               response_dof=['1X+', '1Y+', '1Z+'],
               ordinate_dim=['acceleration**2/frequency'] * 3)
    report = Report('R', [{'kind': 'plot', 'source': 'PSDs'}],
                    features=list(CONVENTIONS))
    path = tmp_path / 'styled.html'
    html = render_html(report, {'PSDs': psds})
    path.write_text(html, encoding='utf-8')
    QApplication.instance() or QApplication(['x'])
    view = web_view()
    view.resize(1000, 800)
    view.load(QUrl.fromLocalFile(str(path)))
    try:
        js = web_read(
            view,
            "(() => { const c = document.querySelector('canvas');"
            "if (!c || !c.dataset.home) return null;"
            "const block = DATA.blocks[0];"
            "const texts = [];"
            "const g = {strokeText() {}, fillText(t) { texts.push(t); }};"
            "const view = JSON.parse(c.dataset.home);"
            "const id = v => v;"
            "endLabels(g, block, view, id, id,"
            "  {left: 0, top: 0}, 1000, 1000);"
            "const many = Object.assign({}, block, {curves:"
            "  Array.from({length: 9}, (_, k) => block.curves[k % 3])});"
            "const none = [];"
            "endLabels({strokeText() {}, fillText(t) { none.push(t); }},"
            "  many, view, id, id, {left: 0, top: 0}, 1000, 1000);"
            "return JSON.stringify({"
            " dashes: block.curves.map((cv, k) => lineDash(block, cv, k)),"
            " swatches: Array.from(document.querySelectorAll("
            "   '.legend svg.swatch line'),"
            "   l => l.getAttribute('stroke-dasharray')),"
            " labels: texts, crowded: none, home: view}); })()")
    finally:
        web_close(view)
    got = json.loads(js)
    assert got['dashes'] == [[], [7, 4], [1.5, 3]]
    assert got['swatches'] == [None, '7 4', '1.5 3']
    assert sorted(got['labels']) == sorted(c for c in
                                           ('1X+', '1Y+', '1Z+'))
    assert got['crowded'] == [], 'nine curves: the legend names them'
    assert got['home'] == _payload(html)['blocks'][0]['home'], \
        'the figure opens on its whole decades'
    assert got['home']['x0'] == 1.0 and got['home']['x1'] == 4.0, \
        '10 Hz to 2 kHz in whole decades'


# ---- the panel --------------------------------------------------------------


@pytest.fixture
def editing(window, pump, project):
    for name, obj in project.items():
        window.add_object(name, obj)
    name = window.generate_report('modal')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    return window, window.report_editor, window.objects[name], name


def test_the_panel_shows_every_feature_by_group_with_its_reason(editing):
    from visualdynamics.gui.tooltips import plain

    _window, editor, report, _name = editing
    panel = editor.features_panel
    assert panel is not None, "in the pane with the report's own settings"
    assert list(panel.boxes) == [key for group in rf.GROUPS
                                 for key, f in rf.FEATURES.items()
                                 if f.group == group]
    assert panel.preset_box.currentData() == 'default'
    assert [panel.preset_box.itemText(i)
            for i in range(panel.preset_box.count())] == [
        'Visual Dynamics default', 'Test report', 'Flash report', 'Custom']
    live = panel.boxes['plot_conventions']
    assert live.isEnabled() and not live.isChecked()
    assert plain(live.toolTip()).startswith(
        rf.FEATURES['plot_conventions'].description)
    shared = panel.boxes['shared_decades']
    assert not shared.isEnabled()
    assert panel.reasons['shared_decades'].text() == 'needs Plot conventions'
    for key, feature in rf.FEATURES.items():
        if not feature.available:
            assert not panel.boxes[key].isEnabled(), key
            assert panel.reasons[key].text().startswith('not yet available')
            assert 'Not yet available' in plain(panel.boxes[key].toolTip())
    assert report.features == []


def test_a_box_changes_the_report_through_the_verb(editing):
    window, editor, report, name = editing
    panel = editor.features_panel
    panel.boxes['plot_conventions'].setChecked(True)
    assert report.features == ['plot_conventions']
    # journaled as the verb a script would call; the editor's own
    # settling lines for the report may follow it
    assert (f"project.report_features({name!r}, enable=['plot_conventions'])"
            in window.project.journal)
    assert panel.boxes['shared_decades'].isEnabled(), 'its requirement is met'
    assert not panel.reasons['shared_decades'].isVisibleTo(panel)
    with open(editor._page_path, encoding='utf-8') as page_file:
        assert '"line_patterns"' in page_file.read(), 'the page re-rendered'
    panel.boxes['shared_decades'].setChecked(True)
    assert report.features == list(CONVENTIONS)
    panel.boxes['plot_conventions'].setChecked(False)
    assert report.features == []
    assert not panel.boxes['shared_decades'].isChecked()
    assert 'Shared decades turned off' in panel.notes.text()
    assert panel.preset_box.currentData() == 'default'


def test_the_preset_selector_applies_and_lists_what_it_skipped(editing):
    _window, editor, report, _name = editing
    panel = editor.features_panel
    panel.preset_box.setCurrentIndex(panel.preset_box.findData('test_report'))
    assert report.features == list(LIVE)
    assert panel.boxes['plot_conventions'].isChecked()
    assert 'Findings register skipped, not yet available' in panel.notes.text()
    assert panel.preset_box.currentData() == 'test_report'
    panel.boxes['shared_decades'].setChecked(False)
    assert panel.preset_box.currentText() == 'Custom', 'a set no preset names'
    panel.preset_box.setCurrentIndex(panel.preset_box.findData('default'))
    assert report.features == []


def test_a_refusal_shows_on_the_panel_and_changes_nothing(editing):
    _window, editor, report, _name = editing
    edits = []
    editor.edited.connect(lambda: edits.append(1))
    editor._operate({'op': 'features', 'enable': ['findings']})
    assert report.features == []
    assert 'not yet available' in editor.features_panel.notes.text()
    assert edits == [], 'nothing changed: no edit to journal, no rebuild'
    editor._operate({'op': 'features', 'enable': ['plot_conventions']})
    assert edits == [1] and editor.features_panel.notes.text() == ''
