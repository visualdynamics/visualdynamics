"""The test requirement and the result against it in a report (report
feature `test_requirement`, `report.requirement`).

Off, the page is unchanged; on, the requirement's words, the result
table and the pass/fail box read from the requirement, or labeled a
screening; the reading kept and made on the window's worker.
"""

from __future__ import annotations

import json

import pytest
from conftest import fixture_path, web_close, web_read, web_view
from test_requirement import STEPPED, reference, run, single

import visualdynamics
from visualdynamics.core import requirement as rq
from visualdynamics.core.report import Report
from visualdynamics.report import page, render_html
from visualdynamics.report import requirement as rr

# ---- the report -----------------------------------------------------------


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


@pytest.fixture(scope='module')
def plate():
    """A random run worked up, with its generated report."""
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'))
    name = project.generate_report('random')
    return project, name


@pytest.fixture
def bound(plate):
    """The plate's report with a requirement bound, removed after."""
    project, name = plate
    report = project[name]
    blocks, features = list(report.blocks), list(report.features)
    index = project.report_requirement(
        name, 'Specification', 'Time History PSDs', 'Test plan TP-12',
        STEPPED, revision='B', strategy='average',
        individual_bands=[(None, None, -6, 6)], grms_percent=10,
        sources={'composite_bands': ('customer', 'Prime')})
    yield project, name, index
    report.blocks, report.features = blocks, features


def _render(project, name, **kwargs):
    return render_html(project[name], project, unit_system=visualdynamics.SI,
                       object_groups=project.object_groups, **kwargs)


def test_off_a_requirement_block_changes_nothing(plate):
    project, name = plate
    before = _render(project, name)
    report = project[name]
    blocks = list(report.blocks)
    try:
        project.report_requirement(name, 'Specification', 'Time History PSDs',
                                   'TP', [(None, None, -6, 6)],
                                   strategy='average',
                                   individual_bands=[(None, None, -9, 9)])
        assert _render(project, name) == before
    finally:
        report.blocks = blocks


def test_on_the_words_the_table_and_the_box_read_from_the_requirement(
        bound):
    project, name, _index = bound
    project.report_features(name, enable=['test_requirement'])
    drawn = _payload(_render(project, name))['blocks']
    box = next(block for block in drawn if block['kind'] == 'verdict')
    said = box['requirement']
    assert said['document'] == 'Test plan TP-12, revision B'
    assert said['rule'] == 'any line outside its band is a miss'
    results = rr.read(project[name].blocks[_index], project,
                      project.object_groups)
    assert (said['state'], said['met'], said['rows']) == (
        results.state, results.met, len(results.rows))
    assert said['missed'] == [f'{row.name}, {row.required}'
                              for row in results.rows
                              if row.state == 'not met']
    words = next(block for block in drawn if 'Test requirement' in
                 block.get('html', ''))
    assert ('supplied by the customer (Prime); the laboratory has not '
            'verified its derivation') in words['html']
    table = next(block for block in drawn if block['kind'] == 'table'
                 and block['caption'].startswith(rr.CAPTION))
    assert table['caption'] == (f'{rr.CAPTION}: Time History PSDs, Test '
                                'plan TP-12, revision B')
    assert table['headers'] == ['Criterion', 'Requirement', 'Measured',
                                'Result']
    assert [row[3] for row in table['rows']] == [
        row.state for row in results.rows]
    assert [mark[3] for mark in table['marks']] == [
        rr.MARKS.get(row.state, '') for row in results.rows]
    assert table['label'].startswith('Table ')


def test_on_a_box_with_no_requirement_is_a_screening(plate):
    project, name = plate
    report = project[name]
    try:
        project.report_features(name, enable=['test_requirement'])
        drawn = _payload(_render(project, name))['blocks']
        box = next(block for block in drawn if block['kind'] == 'verdict')
        assert box['screening'] == rr.SCREENING
        assert 'requirement' not in box
    finally:
        report.features = []
    box = next(block for block in _payload(_render(project, name))['blocks']
               if block['kind'] == 'verdict')
    assert 'screening' not in box, 'off: the box it was'


def test_a_box_reads_from_the_requirement_of_its_own_pair_first():
    report = Report(blocks=[
        {'kind': 'verdict', 'source': 'B', 'measured': 'RB'},
        {'kind': 'text', 'text': '', 'requirement': {}, 'source': 'A',
         'measured': 'RA'},
        {'kind': 'text', 'text': '', 'requirement': {}, 'source': 'B',
         'measured': 'RB'}])
    objects = dict.fromkeys(('A', 'B', 'RA', 'RB'))
    assert rr.matching(report, report.blocks[0], objects) is report.blocks[2]
    del report.blocks[2]
    assert rr.matching(report, report.blocks[0], objects) is report.blocks[1]
    report.blocks.append(dict(report.blocks[1], source='C'))
    assert rr.matching(report, report.blocks[0],
                       dict(objects, C=None)) is None, \
        'two requirements, neither of its pair: no guess'


def test_under_the_skeleton_the_words_and_the_table_find_their_sections(
        bound):
    project, name, _index = bound
    project.report_features(name, enable=['test_requirement',
                                          'skeleton_sections'])
    drawn = _payload(_render(project, name))['blocks']
    heads = [k for k, block in enumerate(drawn)
             if 'class="skeleton"' in block.get('html', '')]
    titles = [drawn[k]['html'] for k in heads]
    requirement = heads[next(k for k, t in enumerate(titles)
                             if '3. Requirement' in t)]
    results = heads[next(k for k, t in enumerate(titles)
                         if '8. Results' in t)]
    assert 'Test requirement' in drawn[requirement + 1]['html']
    assert drawn[results + 1]['kind'] == 'table'
    assert drawn[results + 1]['caption'].startswith(rr.CAPTION)
    assert drawn[results + 1]['label'] == 'Table 2', \
        'numbered in the order the page shows'
    assert drawn[results + 2]['html'].startswith('<ul>\n<li>')
    assert drawn[results + 3]['html'].startswith('<h3>Measured Data'), \
        'the section goes on after the note'


def test_the_page_carries_the_feature_code_only_when_on():
    css, js = page.page_parts(())
    assert 'requirementVerdict' not in js and 'mark-notmet' not in css
    css, js = page.page_parts(('test_requirement',))
    assert 'function requirementVerdict' in js
    assert 'if (block.requirement)' in js and 'block.screening' in js
    assert 'td.mark-notmet' in css


def test_the_box_draws_the_result_in_words(bound, tmp_path, qt_app):
    from PySide6.QtCore import QUrl

    project, name, _index = bound
    project.report_features(name, enable=['test_requirement'])
    path = tmp_path / 'page.html'
    path.write_text(_render(project, name), encoding='utf-8')
    view = web_view()
    view.load(QUrl.fromLocalFile(str(path)))
    try:
        said = web_read(
            view, "(() => { const v = document.querySelector('.verdict');"
                  "return v ? v.innerText : null; })()")
    finally:
        web_close(view)
    assert said.startswith('NOT MET')
    assert 'criteria met against Test plan TP-12, revision B' in said
    assert 'Decision rule: any line outside its band is a miss.' in said


def test_the_box_says_screening_on_the_page(plate, tmp_path, qt_app):
    from PySide6.QtCore import QUrl

    project, name = plate
    report = project[name]
    project.report_features(name, enable=['test_requirement'])
    try:
        path = tmp_path / 'page.html'
        path.write_text(_render(project, name), encoding='utf-8')
    finally:
        report.features = []
    view = web_view()
    view.load(QUrl.fromLocalFile(str(path)))
    try:
        said = web_read(
            view, "(() => { const v = document.querySelector('.verdict');"
                  "return v ? v.innerText : null; })()")
    finally:
        web_close(view)
    assert said.upper().startswith(rr.SCREENING.upper())
    assert 'not a statement of conformity' in said


def test_a_requirement_that_cannot_be_read_says_so_in_results(plate):
    project, _name = plate
    report = Report(features=['test_requirement'])
    rr.add(report, single(channels=('nowhere',)), source='Specification',
           measured='Time History PSDs')
    drawn = _payload(render_html(report, project))['blocks']
    texts = [block['html'] for block in drawn if block['kind'] == 'text']
    assert any('Test requirement' in text for text in texts)
    assert any('nothing judged; the measurement has no autospectrum at '
               'nowhere' in text for text in texts)
    assert not [block for block in drawn if block['kind'] == 'table']


def test_a_requirement_unbound_is_not_drawn(plate):
    project, _name = plate
    report = Report(features=['test_requirement'])
    rr.add(report, single(), source='Missing', measured='Time History PSDs')
    assert _payload(render_html(report, project))['blocks'] == []


def test_on_the_edit_page_its_blocks_select_the_requirement(bound):
    project, name, index = bound
    project.report_features(name, enable=['test_requirement'])
    drawn = _payload(_render(project, name, edit=True))['blocks']
    mine = [block for block in drawn if block.get('index') == index]
    assert [block['kind'] for block in mine] == ['text', 'table', 'text']
    project.report_features(name, enable=['control_uncertainty'])
    drawn = _payload(_render(project, name, edit=True))['blocks']
    mine = [block for block in drawn if block.get('index') == index]
    assert [block['kind'] for block in mine] == ['text', 'text', 'table',
                                                 'text'], \
        'the words, the uncertainty, the results and their note'


def test_the_block_goes_after_the_box_and_keeps_through_a_template(
        bound, tmp_path):
    from visualdynamics.io import report_template

    project, name, index = bound
    report = project[name]
    assert report.blocks[index - 1]['kind'] == 'verdict'
    report_template.save(report, tmp_path / 'house.vdreport')
    loaded = report_template.load(tmp_path / 'house.vdreport')
    assert loaded.blocks[index]['requirement'] == \
        report.blocks[index]['requirement']


def test_the_verb_is_journaled_and_refuses_before_changing_anything(plate):
    project, name = plate
    report = project[name]
    blocks = list(report.blocks)
    journal = list(project.journal)
    try:
        with pytest.raises(ValueError, match='each control channel'):
            project.report_requirement(name, 'Specification',
                                       'Time History PSDs', 'TP',
                                       [(None, None, -3, 3)],
                                       strategy='average')
        with pytest.raises(ValueError, match='autospectra are candidates'):
            project.report_requirement(name, 'Specification',
                                       'Time History PSDs', 'TP',
                                       [(None, None, -3, 3)])
        assert report.blocks == blocks and project.journal == journal
        index = project.report_requirement(
            name, '@basis:Specification', 'Time History PSDs', 'TP',
            [(None, None, -3, 3)], strategy='maximum',
            individual_bands=[(None, None, -6, 6)], duration=180,
            duration_tolerance=(5, 5), duration_run=184)
        assert report.blocks[index]['source'] == '@basis:Specification'
        assert report.blocks[index]['duration_run'] == 184.0
        assert project.journal[-1].startswith(
            f'project.report_requirement({name!r}, '
            "'@basis:Specification', 'Time History PSDs', 'TP'")
    finally:
        report.blocks = blocks


def test_a_reading_is_kept_and_made_again_when_its_run_changes(bound,
                                                               monkeypatch):
    project, name, index = bound
    from visualdynamics.report.check_blocks import forget

    project.report_features(name, enable=['test_requirement'])
    report = project[name]
    block = report.blocks[index]
    forget()
    assert rr.pending(report, project, project.object_groups) == [block]
    told = []
    assert rr.prepare(report, project, project.object_groups,
                      progress=lambda done, total: told.append(
                          (done, total))) == 1
    assert told == [(0, 1), (1, 1)]
    assert rr.pending(report, project, project.object_groups) == []

    def no(*args, **kwargs):
        raise AssertionError('read again')

    monkeypatch.setattr(rq, 'evaluate', no)
    _render(project, name)          # drawn from what was kept
    monkeypatch.undo()
    block['duration_run'] = 120.0
    assert rr.pending(report, project, project.object_groups) == [block]


def test_a_canceled_reading_is_drawn_as_nothing_judged_and_offered_again(
        bound):
    from visualdynamics.core.progress import Cancelled
    from visualdynamics.report.check_blocks import forget

    project, name, index = bound
    project.report_features(name, enable=['test_requirement'])
    report = project[name]
    forget()

    def cancel(done, total):
        raise Cancelled('Reading the results against the requirement')

    with pytest.raises(Cancelled):
        rr.prepare(report, project, project.object_groups, progress=cancel)
    drawn = _payload(_render(project, name))['blocks']
    assert any('nothing judged; the reading was canceled' in
               block.get('html', '') for block in drawn)
    assert rr.pending(report, project, project.object_groups) == [
        report.blocks[index]]
    forget()


def test_off_nothing_is_pending(bound):
    project, name, _index = bound
    assert rr.pending(project[name], project, project.object_groups) == []


# ---- the window -----------------------------------------------------------


def test_the_window_reads_the_requirement_on_its_worker_and_shows_it(
        window, pump):
    from PySide6.QtWidgets import QLabel

    from visualdynamics.report.check_blocks import forget

    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = window.generate_report('random')
    window.project.add('Ref', reference())
    window.project.add('Run', run(1.0, 1.0, 0.0, 1.0))
    index = window.project.report_requirement(
        name, 'Ref', 'Run', 'Test plan TP-12',
        [(None, None, -3, 3)], strategy='average',
        channels=('1Z+', '2Z+', '3Z+', '4Z+'),
        individual_bands=[(None, None, -6, 6)])
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    forget()
    ran = []
    real = window._run_long

    def spy(status, verb, *args, **kwargs):
        ran.append((status, verb))
        return real(status, verb, *args, **kwargs)

    window._run_long = spy
    window.report_editor._operate({'op': 'features',
                                   'enable': ['test_requirement']})
    pump()
    assert ran == [('Reading the results against the requirement…',
                    rr.prepare)]
    editor = window.report_editor
    editor.selected = index
    editor._show_selection()
    shown = [label.text() for label in editor.pane.widget().findChildren(
        QLabel)]
    assert '<b>Test requirement</b> — requirement and results' in shown
    assert any('Governing document: Test plan TP-12.' in text
               and 'Result: ' in text for text in shown)
    forget()


# ---- the edges the mutation run found -------------------------------------


def test_a_block_names_what_it_reads_and_a_duration_above_zero():
    report = Report()
    for source, measured in (('', 'Run'), ('Ref', '')):
        with pytest.raises(ValueError, match='names its specification'):
            rr.add(report, single(), source=source, measured=measured)
    with pytest.raises(ValueError, match='seconds above 0'):
        rr.add(report, single(), source='Ref', measured='Run',
               duration_run=0)
    assert report.blocks == []
    at = rr.add(report, single(), source='Ref', measured='Run',
                duration_run=30, caption='Y axis')
    assert report.blocks[at]['caption'] == 'Y axis'


def test_a_block_whose_run_is_missing_is_not_read():
    report = Report()
    rr.add(report, single(), source='Ref', measured='Missing')
    assert rr.read(report.blocks[0], {'Ref': reference()}) is None
    assert rr.inputs(report.blocks[0], {'Ref': reference()}) is None


def test_the_project_s_one_channel_table_names_the_control_channel():
    from visualdynamics.core.channel_table import ChannelTable

    table = ChannelTable({'channel': [1, 2], 'node': [1, 2],
                          'direction': ['Z+', 'Z+'],
                          'unit': ['m/s**2', 'm/s**2'],
                          'control': ['False', 'True']})
    objects = {'Table': table, 'Ref': reference(('1Z+', '2Z+')),
               'Run': run(1.0, 1.0)}
    assert rr.channel_table(objects) is table
    assert rr.channel_table({**objects, 'Other': table}) is None, \
        'two tables: neither is the one'
    report = Report()
    rr.add(report, single(), source='Ref', measured='Run')
    assert rr.read(report.blocks[0], objects).channels == ('2Z+',)


def test_the_caption_and_the_note_say_what_the_block_and_results_say():
    report = Report()
    rr.add(report, single(duration=180, duration_tolerance=(5, 5)),
           source='Ref', measured='Run', caption='Y axis, full level')
    objects = {'Ref': reference(), 'Run': run(1.0)}
    made = rr.build(report.blocks[0], objects)
    table, note = made.results
    assert table['caption'] == 'Y axis, full level: Run, Test plan TP-12'
    assert note['text'] == '- 1 of 2 criteria met; the result is not judged.'
    one = rq.Results(single(), (rq.Criterion('grms', 'a', '', '', rq.MET),),
                     -6.0, (), ('the measurement was scaled by -6 dB',))
    assert rr._note(one).split('\n') == [
        '- 1 of 1 criterion met; the result is met.',
        '- Every comparison is of the measurement scaled by -6 dB.',
        '- The measurement was scaled by -6 dB.']


def test_the_box_lists_what_was_not_judged_and_what_was_refused():
    report = Report(blocks=[{'kind': 'verdict', 'source': 'Ref',
                             'measured': 'Run'}])
    rr.add(report, single(duration=180, duration_tolerance=(5, 5)),
           source='Ref', measured='Run')
    objects = {'Ref': reference(), 'Run': run(1.0)}
    box = rr.verdict({'kind': 'verdict'}, report.blocks[0], report, objects)
    assert box['requirement']['state'] == 'not judged'
    assert box['requirement']['unjudged'] == [
        'Duration at level, 3 min, −5/+5 %']
    report.blocks[1]['requirement'] = single(channels=('9Z+',)).as_record()
    box = rr.verdict({'kind': 'verdict'}, report.blocks[0], report, objects)
    assert box['requirement'] == {
        'state': 'not judged', 'met': 0, 'rows': 0, 'document': '',
        'missed': [],
        'unjudged': ['the measurement has no autospectrum at 9Z+'],
        'rule': ''}
    assert rr.verdict(None, report.blocks[0], report, objects) is None


# ---- the control channels' uncertainty ----------------------------------------


def test_the_uncertainty_table_falls_under_the_requirement_s_words():
    from test_requirement import BUDGET

    report = Report()
    rr.add(report, single(channels=('1Z+',), grms_percent=10,
                          uncertainty={'1Z+': BUDGET}),
           source='Ref', measured='Run')
    objects = {'Ref': reference(), 'Run': run(1.0)}
    assert len(rr.build(report.blocks[0], objects).words) == 1, \
        'the feature off, the words alone'
    made = rr.build(report.blocks[0], objects, uncertainty_on=True)
    words, table, note = made.words
    assert 'Measurement uncertainty: entered for 1Z+' in words['text']
    assert table['caption'] == rr.UNCERTAINTY
    assert table['headers'] == ['Channel', 'Contributions',
                                'Combined, acceleration', 'On the ASD',
                                'In dB', 'One third of the tolerance',
                                'Result']
    assert table['rows'] == [[
        '1Z+', ('calibration 2 % (k = 2, certificate C-1); conditioner and '
        'converter 0.5 % (k = 2); transverse 1.5 % (k = 2); mount and chain '
        '1 % (k = 2)'), '2.7 %', '5.7 %', '0.24 dB',
        '2 dB (±6 dB band); 3.3 % (±10 % rms)', 'met']]
    assert 'marks' not in table
    assert note['text'].startswith('- Each contribution is divided by its '
                                   'coverage factor')
    assert 'expanded at k = 2' in note['text']


def test_an_uncertainty_over_its_third_or_not_entered_is_marked():
    from test_requirement import BUDGET

    report = Report()
    rr.add(report, single(composite_bands=[(None, None, -0.5, 0.5)],
                          uncertainty={'1Z+': BUDGET, '7Z+': BUDGET}),
           source='Ref', measured='Run')
    objects = {'Ref': reference(), 'Run': run(1.0)}
    _words, table, note = rr.build(report.blocks[0], objects,
                                   uncertainty_on=True).words
    (row,) = table['rows']
    assert row[0] == '1Z+' and row[-1].startswith('not met: 0.24 dB over')
    assert row[5] == '0.17 dB (±0.5 dB band)', 'no rms tolerance, no third'
    assert table['marks'] == [[''] * 6 + ['notmet']]
    assert note['text'].split('\n')[1] == (
        '- Contributions are entered for 7Z+, which is not a control '
        'channel of the run.')
    two = Report()
    rr.add(two, single(uncertainty={'8Z+': BUDGET, '9Z+': BUDGET}),
           source='Ref', measured='Run')
    _words, table, note = rr.build(two.blocks[0], objects,
                                   uncertainty_on=True).words
    assert table['rows'][0][1:5] == ['—'] * 4
    assert len(table['rows'][0]) == len(table['headers'])
    assert table['rows'][0][-1] == 'not judged: no contributions are entered'
    assert table['marks'] == [[''] * 6 + ['notjudged']]
    assert 'for 8Z+, 9Z+, which are not' in note['text']


def test_with_nothing_entered_the_uncertainty_says_not_judged():
    report = Report()
    rr.add(report, single(), source='Ref', measured='Run')
    made = rr.build(report.blocks[0], {'Ref': reference(), 'Run': run(1.0)},
                    uncertainty_on=True)
    assert made.words[1] == {'kind': 'text', 'text': (
        f'**{rr.UNCERTAINTY}:** not judged; no contributions are entered '
        'for the control channels.')}
    refused = Report()
    rr.add(refused, single(channels=('9Z+',)), source='Ref', measured='Run')
    made = rr.build(refused.blocks[0], {'Ref': reference(), 'Run': run(1.0)},
                    uncertainty_on=True)
    assert len(made.words) == 2, 'a run not read still says its uncertainty'
    assert len(rr.build(refused.blocks[0], {'Ref': reference(),
                                            'Run': run(1.0)}).words) == 1
    assert made.results[0]['text'].startswith(f'**{rr.CAPTION}:** nothing')


def test_the_verb_takes_the_uncertainty_and_the_skeleton_places_it(bound):
    from test_requirement import BUDGET

    project, name, index = bound
    report = project[name]
    del report.blocks[index]
    project.report_features(name, enable=['test_requirement',
                                          'skeleton_sections',
                                          'control_uncertainty'])
    found = rq.evaluate(rq.TestRequirement('TP', STEPPED, strategy='average',
                                           individual_bands=[(None, None,
                                                              -6, 6)]),
                        project['Specification'],
                        project['Time History PSDs'])
    at = project.report_requirement(
        name, 'Specification', 'Time History PSDs', 'TP', STEPPED,
        strategy='average', individual_bands=[(None, None, -6, 6)],
        uncertainty={found.channels[0]: BUDGET})
    drawn = _payload(_render(project, name))['blocks']
    heads = [k for k, block in enumerate(drawn)
             if 'class="skeleton"' in block.get('html', '')]
    requirement = next(k for k in heads if '3. Requirement' in drawn[k]['html'])
    table = drawn[requirement + 2]
    assert table['caption'] == rr.UNCERTAINTY
    assert table['label'] == 'Table 1'
    assert [row[0] for row in table['rows']] == list(found.channels)
    assert table['rows'][0][-1] == 'met'
    del report.blocks[at]


# ---- cross-axis and limit channels ---------------------------------------------


def test_the_cross_axis_and_limit_rows_are_in_the_table_with_the_profile():
    import numpy as np
    from test_requirement import LEVEL, LINES, PSD_DIM, notched

    from visualdynamics.core.data import Psd

    profile = Psd(LINES, np.full((1, LINES.size), LEVEL), response_dof=['2Z+'],
                  ordinate_dim=[PSD_DIM])
    objects = {'Ref': reference(('1Z+',)), 'Run': notched(),
               'Limits': profile}
    report = Report()
    rr.add(report, single(channels=('1Z+',), cross_axis=[(None, None, 50)],
                          cross_axis_channels={'2Z+': '1Z+'},
                          limit_channels=[('2Z+', 'Limits', [(380, 500)])]),
           source='Ref', measured='Run')
    off, _note = rr.build(report.blocks[0], objects).results
    assert [row[2] for row in off['rows'][-2:]] == [rr.ROLES_OFF] * 2, \
        'the feature off, each role read not judged'
    assert [row[0] for row in off['rows'][-2:]] == [
        'Cross-axis', 'Limit channel 2Z+']
    table, _note = rr.build(report.blocks[0], objects, roles_on=True).results
    assert off['rows'][0] == table['rows'][0], \
        'a band row is read whatever the roles'
    names = [row[0] for row in table['rows']]
    assert names[-2:] == ['Cross-axis 2Z+ (in-axis 1Z+)', 'Limit channel 2Z+']
    limit = table['rows'][-1]
    assert limit[3] == 'met'
    assert limit[2].startswith('held its limit from 400 to 480 Hz')
    cross = table['rows'][-2]
    assert cross[3] == 'not met', 'the limit channel is the cross-axis here'
    assert 'Finding' not in table['headers']
    words = rr.build(report.blocks[0], objects,
                     roles_on=True).words[0]['text']
    assert 'Limit channels: 2Z+, held at or under its limit profile in ' \
        'Limits' in words
    del objects['Limits']
    kept = rr.read(report.blocks[0], objects, roles=True)
    assert kept.rows[-1].measured == ("not judged: its limit profile "
                                      "'Limits' is not there to read"), \
        'the profile gone is a reading of its own, not the one kept'


def _offering(planned=()):
    """A report with the requirement, the roles and the findings on, its
    limit channel 2Z+ cutting a notch from 400 to 480 Hz, the notches
    `planned` named, and finding D-1 entered."""
    from test_requirement import notched

    from visualdynamics.report import findings

    objects = {'Ref': reference(('1Z+', '2Z+')), 'Run': notched()}
    report = Report('R', [], features=['skeleton_sections', 'test_requirement',
                                       'findings', 'cross_axis'])
    rr.add(report, single(channels=('1Z+',),
                          limit_channels=[('2Z+', '', list(planned))]),
           source='Ref', measured='Run')
    findings.add(report, 'deviation')
    return report, objects


def test_an_unplanned_notch_offers_its_fact_through_the_register():
    from visualdynamics.report import findings

    report, objects = _offering()
    (entry,) = findings.register(report)
    assert entry['draft'] is None
    draft = rr.draft_finding(report, 'D-1', objects, 'Run', 'Ref',
                             'Limit channel 2Z+',
                             provenance={'Run': {'source': 'Run 3'}})
    assert draft.text == (
        '2Z+ held its limit profile in a notch the plan does not name: from '
        '400 to 480 Hz (17 lines), the control up to 3 dB under the '
        "reference; PSD 'Run', from 'Run 3'.")
    assert (draft.channel, draft.band, draft.run) == ('2Z+', (400.0, 480.0),
                                                      'Run 3')
    assert draft.source == 'Run against Ref, Test plan TP-12, Limit channel 2Z+'
    (entry,) = findings.register(report)
    assert entry['draft']['text'] == draft.text, 'offered, through offer'
    assert entry['fact'] == '', 'a draft fills nothing'
    table, _note = rr.build(report.blocks[0], objects, roles_on=True,
                            register=findings.register(report)).results
    assert table['headers'][-1] == 'Finding'
    assert table['rows'][-1][-1] == 'D-1', 'the finding its fact went to'
    assert {row[-1] for row in table['rows'][:-1]} == {''}
    findings.accept(report, 'D-1')
    table, _note = rr.build(report.blocks[0], objects, roles_on=True,
                            register=findings.register(report)).results
    assert table['rows'][-1][-1] == 'D-1', 'and keeps it once accepted'


def test_a_planned_notch_offers_nothing_and_adds_no_column():
    from visualdynamics.report import findings

    report, objects = _offering(planned=[(380, 500)])
    with pytest.raises(ValueError, match='offers no finding'):
        rr.draft_finding(report, 'D-1', objects, 'Run', 'Ref', 'Limit channel 2Z+')
    assert findings.register(report)[0]['draft'] is None, 'nothing changed'
    table, _note = rr.build(report.blocks[0], objects, roles_on=True,
                            register=findings.register(report)).results
    assert 'Finding' not in table['headers'], 'no row carries one'


def test_an_offer_with_no_finding_entered_says_so_and_the_row_is_found():
    from visualdynamics.report import findings

    report, objects = _offering()
    table, _note = rr.build(report.blocks[0], objects, roles_on=True,
                            register=findings.register(report)).results
    assert table['rows'][-1][-1] == rr.UNENTERED
    without, _note = rr.build(report.blocks[0], objects, roles_on=True).results
    assert 'Finding' not in without['headers'], 'no register, no column'
    count = len(table['rows'])
    draft = rr.draft_finding(report, 'D-1', objects, 'Run', 'Ref', count)
    assert draft.channel == '2Z+', 'a row named by its number'
    for criterion, said in ((count + 1, f'rows 1 to {count}'),
                            ('Limit channel 9Z+', 'no row'),
                            (1, 'offers no finding')):
        with pytest.raises(ValueError, match=said):
            rr.draft_finding(report, 'D-1', objects, 'Run', 'Ref', criterion)
    with pytest.raises(ValueError, match='no requirement is bound'):
        rr.draft_finding(report, 'D-1', objects, 'Ref', 'Run', count)
    report.features = ['test_requirement', 'cross_axis']
    with pytest.raises(ValueError, match='does not have'):
        rr.draft_finding(report, 'D-1', objects, 'Run', 'Ref', count)


def test_a_name_two_rows_share_finds_the_one_that_offers():
    from visualdynamics.report import findings

    # 2Z+ against two cross-axis limits: one row per limit, both named
    # for the channel, the one below 500 Hz over its limit
    objects = {'Ref': reference(('1Z+', '2Z+')),
               'Run': run(1.0, 0.2, hot=[(1, 440, 460, 3.0)])}
    report = Report('R', [], features=['skeleton_sections', 'test_requirement',
                                       'findings', 'cross_axis'])
    rr.add(report, single(channels=('1Z+',),
                          cross_axis=[(None, 500, 50), (500, None, 100)],
                          cross_axis_channels={'2Z+': '1Z+'}),
           source='Ref', measured='Run')
    findings.add(report, 'deviation')
    name = 'Cross-axis 2Z+ (in-axis 1Z+)'
    draft = rr.draft_finding(report, 'D-1', objects, 'Run', 'Ref', name)
    assert draft.band == (440.0, 460.0)
    objects['Run'] = run(1.0, 0.2, hot=[(1, 440, 460, 3.0),
                                        (1, 600, 620, 6.0)])
    with pytest.raises(ValueError, match='2 rows .* offer a finding'):
        rr.draft_finding(report, 'D-1', objects, 'Run', 'Ref', name)


def test_the_verb_is_journaled_and_drafts_through_the_register(bound):
    from visualdynamics.report import findings

    project, name, _index = bound
    report = project[name]
    project.report_features(name, enable=['skeleton_sections', 'findings',
                                          'test_requirement', 'cross_axis'])
    del report.blocks[_index]
    at_block = project.report_requirement(
        name, 'Specification', 'Time History PSDs', 'TP', STEPPED,
        strategy='average', individual_bands=[(None, None, -6, 6)],
        cross_axis=[(None, None, 1)],
        cross_axis_channels={'113Z+': '104Z+'})
    number = project.report_finding(name, kind='anomaly')
    before = list(project.journal)
    with pytest.raises(ValueError, match='offers no finding'):
        project.draft_finding_from_requirement(
            name, number, 'Time History PSDs', 'Specification', 1)
    assert project.journal == before, 'a refusal is not journaled'
    rows_ = rr.read(report.blocks[at_block], project,
                    project.object_groups).rows
    at = next(k for k, row in enumerate(rows_, 1)
              if row.key == 'cross_axis')
    text = project.draft_finding_from_requirement(
        name, number, 'Time History PSDs', 'Specification', at)
    assert text.startswith('113Z+ read up to ')
    assert 'over its 1 % cross-axis limit' in text
    assert project.journal[-1] == (
        f'project.draft_finding_from_requirement({name!r}, {number!r}, '
        f"'Time History PSDs', 'Specification', {at})")
    assert findings.find(report, number)['draft']['text'] == text


def test_the_verb_takes_the_roles_and_reads_a_named_profile(bound):
    project, name, _index = bound
    report = project[name]
    # DOFs the plate's spectra hold once each
    at = project.report_requirement(
        name, 'Specification', 'Time History PSDs', 'TP', STEPPED,
        strategy='average', individual_bands=[(None, None, -6, 6)],
        cross_axis=[(None, None, 50)],
        cross_axis_channels={'113Z+': '104Z+'},
        limit_channels=[('1301Z+', '@basis:Psd')])
    stored = report.blocks[at]['requirement']
    assert stored['cross_axis_channels'] == {'113Z+': '104Z+'}
    assert stored['limit_channels'] == [['1301Z+', '@basis:Psd', []]]
    results = rr.read(report.blocks[at], project, project.object_groups)
    limit = next(row for row in results.rows if row.key == 'limit')
    assert limit.channel == '1301Z+'
    assert not limit.measured.startswith('not judged'), \
        'the symbolic profile resolves to the project object'
    cross = next(row for row in results.rows if row.key == 'cross_axis')
    assert cross.channel == '113Z+'
    assert cross.measured.startswith('worst ')


def test_the_roles_are_read_on_the_worker_as_the_page_reads_them():
    from test_requirement import notched

    objects = {'Ref': reference(('1Z+', '2Z+')), 'Run': notched()}
    report = Report(features=['test_requirement', 'cross_axis'])
    rr.add(report, single(channels=('1Z+',), limit_channels=['2Z+']),
           source='Ref', measured='Run')
    assert rr.roles_on(report)
    assert rr.prepare(report, objects) == 1
    assert rr.pending(report, objects) == [], 'kept under the key the page reads'
    found = rr.read(report.blocks[0], objects, roles=True)
    assert found.rows[-1].key == 'limit' and found.rows[-1].excursions
    report.features = ['test_requirement']
    assert not rr.roles_on(report)
    assert rr.pending(report, objects) == report.blocks, \
        'read again with the feature off'
    rr.prepare(report, objects)
    assert rr.read(report.blocks[0], objects,
                   roles=False).rows[-1].measured == rr.ROLES_OFF


def test_a_stored_limit_channel_of_one_or_two_values_reads_its_profile():
    import numpy as np
    from test_requirement import LEVEL, LINES, PSD_DIM, notched

    from visualdynamics.core.data import Psd

    profile = Psd(LINES, np.full((1, LINES.size), LEVEL), response_dof=['2Z+'],
                  ordinate_dim=[PSD_DIM])
    objects = {'Ref': reference(('1Z+',)), 'Run': notched(),
               'Limits': profile}
    report = Report()
    rr.add(report, single(channels=('1Z+',)), source='Ref', measured='Run')
    block = report.blocks[0]
    block['requirement']['limit_channels'] = [['2Z+', 'Limits']]
    found = rr.read(block, objects, roles=True)
    assert found.rows[-1].excursions, 'two values: the channel and profile'
    block['requirement']['limit_channels'] = [['2Z+']]
    found = rr.read(block, objects, roles=True)
    assert found.rows[-1].measured == ('not judged: no limit profile is '
                                       'written at 2Z+')
