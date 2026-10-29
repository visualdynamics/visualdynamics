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
