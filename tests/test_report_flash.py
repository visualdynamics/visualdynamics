"""The flash report (report feature `flash_report`, `report.flash`).

One preliminary page drawn from what the project holds: the identity
marked Preliminary, the headline read off the result table, the
findings by number with their owners, and owed lines for what only the
author writes. Off, the page is the page it was. Then the empty
project, the report with no requirement and the empty register; the
typed parts and their refusals; issue, the check and the verb; the
project file and the template; and a real PDF of one page. Last, what a
run log brought into the project adds, read from its extract, saying
that the log holds none of a kind and never that there was none, and
the page unchanged without one.
"""

from __future__ import annotations

import html
import json
import os

import numpy as np
import pytest
from test_requirement import reference, run
from test_run_log_extract import HEADER, writer

import visualdynamics
from visualdynamics.core import run_log_model as model
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.report import Report
from visualdynamics.report import features as rf
from visualdynamics.report import findings, render_html
from visualdynamics.report import flash as fl
from visualdynamics.report import identity as ri
from visualdynamics.report import requirement as rr

#: every field the identification needs, entered
COMPLETE = {
    'number': 'FR-0037-1', 'revision': '0', 'laboratory': 'Test Lab',
    'address': '1 Main Street', 'location': 'Bay 2', 'customer': 'Acme',
    'contact': 'J. Doe, jdoe@example.com', 'received': '2026-10-01',
    'tested': '2026-10-02', 'issued': '2026-10-02',
    'authorized': ['A. Engineer, test engineer', 'Q. Signer, quality']}

#: the parts only the author writes, typed
TYPED = {'condition': 'No damage seen; released to the article engineer.',
         'next_steps': 'Full data review; comparison with the predictions.',
         'report_date': 'within 30 days',
         'opinions': [('analyst', 'The Z peak looks like the bracket.')]}

#: the features the flash report preset turns on
FLASH = ('skeleton_sections', 'test_requirement', 'findings',
         'report_identity', 'flash_report')


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


def _render(project, name):
    return render_html(project[name], project,
                       object_groups=project.object_groups)


def _texts(project, name):
    return [block['html'] for block in _payload(_render(project, name))['blocks']]


def _project(hot=(), requirement=True):
    """A run against its specification, a report on the flash preset and,
    unless asked otherwise, a requirement bound to the pair. `hot`
    raises a band of the run, so a criterion is not met."""
    project = visualdynamics.Project()
    project.add('Specification', reference())
    project.add('Run PSDs', run(1.0, hot=hot))
    name = project.add('Flash', Report('Flash Report', [
        {'kind': 'text', 'text': '## Results'},
        {'kind': 'plot', 'source': 'Run PSDs', 'caption': 'The ASD'}]))
    project.report_features(name, preset='flash_report')
    if requirement:
        project.report_requirement(name, 'Specification', 'Run PSDs',
                                   'Test plan TP-12', [(None, None, -3, 3)],
                                   revision='C')
    return project, name


def _complete(project, name):
    """Everything the page owes, entered: the identification, the
    register said to be none, and the typed parts."""
    project.report_identity(name, **COMPLETE)
    project.report_section(name, 'deviations', none=True)
    project.report_flash(name, **TYPED)


# ---- the feature ------------------------------------------------------------


def test_the_feature_is_built_and_the_preset_turns_it_on():
    assert rf.FEATURES[fl.FEATURE].available
    assert rf.reason(fl.FEATURE) == ('needs Requirement and results; needs '
                                     'Identity and issue; needs Findings '
                                     'register')
    report = Report()
    selection = rf.configure(report, preset='flash_report')
    assert selection.enabled == FLASH and selection.notes == ()
    assert rf.preset_of(report.features) == 'flash_report'


def test_the_page_is_one_preliminary_page_in_the_readers_order():
    project, name = _project()
    texts = _texts(project, name)
    assert len(texts) == 7, "the report's own figure is not on the page"
    banner, identity, head, deviations, open_, opinions, nxt = texts
    assert identity.startswith('<div class="identity">')
    assert 'Preliminary (draft)' in identity
    assert banner == ('<p class="flash-preliminary"><strong>PRELIMINARY'
                      '</strong> Results are subject to data review; the '
                      'test report supersedes this page.</p>')
    for text, title in ((head, 'Headline'),
                        (deviations, 'As Run Against the Plan'),
                        (open_, 'What Is Open'),
                        (opinions, 'Opinions Recorded as Opinions'),
                        (nxt, 'What Happens Next')):
        assert text.startswith(f'<h2 class="flash">{title}</h2>'), title
    assert nxt.endswith('<p class="endmark">End of report.</p>')
    assert all('\n' not in text for text in texts), 'the page keeps white space'


def test_every_printed_page_says_preliminary():
    project, name = _project()
    project.report_identity(name, number='FR-0037-1', revision='0')
    page = _render(project, name)
    assert '"Report FR-0037-1, Revision 0, Preliminary (draft)"' in page
    assert 'p.flash-preliminary' in page, 'its style comes with the feature'


def test_off_the_page_is_the_page_it_was():
    """The typed parts are kept with the identity and do nothing while
    the feature is off: byte for byte the page without them."""
    project, name = _project()
    project.report_identity(name, **COMPLETE)
    report = project[name]
    rf.configure(report, disable=[fl.FEATURE])
    before = _render(project, name)
    rf.configure(report, enable=[fl.FEATURE])
    project.report_flash(name, **TYPED)
    assert _render(project, name) != before
    rf.configure(report, disable=[fl.FEATURE])
    assert _render(project, name) == before
    # and a report with no features at all, the typed parts held
    plain = Report('R', [{'kind': 'text', 'text': 'x'}])
    objects = {}
    first = render_html(plain, objects)
    plain.identity = {'flash': {'condition': 'Released.'}}
    assert render_html(plain, objects) == first


# ---- the headline -----------------------------------------------------------


def test_the_headline_is_read_off_the_result_rows():
    project, name = _project(hot=[(0, 100.0, 200.0, 10.0)])
    block = next(block for block in project[name].blocks
                 if 'requirement' in block)
    results = rr.read(block, project, project.object_groups)
    missed = [row for row in results.rows if row.state == 'not met']
    assert missed, 'the hot band misses its tolerance'
    [line] = fl.headline(project[name], project, project.object_groups)
    assert (results.met, len(results.rows)) == (0, 1), 'one band, one row'
    assert line.startswith('Run PSDs: 0 of 1 criterion met against ')
    assert 'met against Test plan TP-12, revision C' in line
    assert '; not met: ' + '; '.join(
        f'{row.name}, {row.required}' for row in missed) in line
    head = _texts(project, name)[2]
    assert fl.NOT_CONFORMITY in head
    for word in ('PASS', 'FAIL', 'conforms', 'MET'):
        assert word not in head, word


def test_a_requirement_that_judged_nothing_says_why():
    project, name = _project()
    project['Run PSDs'].ordinate[:] = np.nan
    [line] = fl.headline(project[name], project, project.object_groups)
    assert line.startswith('Run PSDs: ')
    assert 'met against' in line or 'nothing judged' in line
    assert 'not judged' in line or 'nothing judged' in line


def test_with_runs_on_each_line_is_labeled_by_its_run_in_run_order():
    project, name = _project()
    t = np.arange(2048) / 1024.0
    for history, axis, number in (('Y run', 'Y', '11'), ('X run', 'X', '3')):
        project.add(history, TimeHistory(
            t, np.random.default_rng(1).standard_normal((1, t.size)),
            response_dof=['1Z+'], ordinate_dim='acceleration'))
        project.set_run_identity(history, axis=axis, run=number, level_db=0)
    project.add('X PSDs', run(1.0, hot=[(0, 100.0, 200.0, 10.0)]))
    project.provenance['Run PSDs'] = {'verb': 'compute_psds',
                                      'source': 'Y run'}
    project.provenance['X PSDs'] = {'verb': 'compute_psds',
                                    'source': 'X run'}
    project.report_requirement(name, 'Specification', 'X PSDs',
                               'Test plan TP-12', [(None, None, -3, 3)],
                               grms_percent=10)
    report = project[name]
    plain = fl.headline(report, project, project.object_groups)
    assert [line.split(':')[0] for line in plain] == ['Run PSDs', 'X PSDs']
    project.report_features(name, enable=['run_sections'])
    lines = fl.headline(report, project, project.object_groups)
    assert lines[0].startswith('Run 3, X axis, 0 dB (X run): 0 of 2 '
                               'criteria met')
    assert lines[1].startswith('Run 11, Y axis, 0 dB (Y run): 1 of 1 '
                               'criterion met')
    # each run's line carries its own reading
    assert 'not met' in lines[0] and 'not met' not in lines[1]


# ---- what the project does not hold is owed ----------------------------------


def test_an_empty_project_owes_every_part_and_invents_none():
    project = visualdynamics.Project()
    name = project.add('Flash', Report('Flash Report'))
    project.report_features(name, preset='flash_report')
    texts = _texts(project, name)
    assert len(texts) == 7
    assert texts[2].count('NOT YET WRITTEN') == 1
    assert fl.NO_HEADLINE in texts[2]
    assert 'not entered' in texts[1]
    for at, key in ((4, 'condition'), (6, 'next_steps'),
                    (6, 'report_date')):
        assert html.escape(f'{fl.LABELS[key]}: {fl.WHAT[key]}') in texts[at], key
    assert fl.check(project[name], project) == [
        'Identification: ' + ', '.join(ri.missing(project[name]))
        + ' not entered',
        'Headline: no requirement is read against a run',
        'Deviations: neither entered nor set to None',
        'Anomalies: neither entered nor set to None',
        'Nonconformances: neither entered nor set to None',
        'Condition on release: not yet written',
        'What happens next: not yet written',
        'Report date: not yet written',
        'Opinions: neither recorded nor set to None']
    assert project.report_check(name) == fl.check(project[name], project)


def test_with_no_requirement_the_headline_is_owed():
    project, name = _project(requirement=False)
    assert fl.headline(project[name], project) == []
    assert 'Headline: no requirement is read against a run' in fl.check(
        project[name], project)
    assert fl.NO_HEADLINE in _texts(project, name)[2]
    # a requirement whose run is not in the project is left out, as the
    # page leaves out a block that does not resolve
    project.report_requirement(name, 'Specification', 'Run PSDs', 'TP',
                               [(None, None, -3, 3)])
    project.remove('Run PSDs')
    assert fl.headline(project[name], project) == []


def test_with_an_empty_register_each_kind_is_owed_until_said():
    project, name = _project()
    texts = _texts(project, name)
    assert 'NOT YET WRITTEN</strong> Deviations: enter each' in texts[3]
    assert 'NOT YET WRITTEN</strong> Anomalies: enter each' in texts[4]
    assert 'NOT YET WRITTEN</strong> Nonconformances: enter each' in texts[4]
    project.report_section(name, 'deviations', none=True)
    texts = _texts(project, name)
    assert 'No deviations.' in texts[3] and 'NOT YET' not in texts[3]
    assert 'No anomalies.' in texts[4] and 'No nonconformances.' in texts[4]


def test_the_findings_are_listed_by_number_with_their_owners():
    project, name = _project()
    project.report_finding(name, kind='deviation', channel='1Z+',
                           band=(600, 680), fact='C2 above its band.',
                           approval="The customer's representative, Day 2.")
    project.report_finding(name, kind='anomaly', status='open',
                           fact='M1 clipped at its 17 g range.')
    project.report_section(name, 'deviations', none=True)
    texts = _texts(project, name)
    assert ('<strong>D-1</strong> <span class="finding-where">(1Z+; 600.0 to '
            '680.0 Hz)</span>: C2 above its band. <em>Approval:</em> The '
            'customer&#x27;s representative, Day 2.') in texts[3]
    assert ('<strong>A-1</strong> <span class="finding-where">(open)</span>: '
            'M1 clipped at its 17 g range. <em>Disposition:</em> '
            '<span class="owed">NOT YET WRITTEN</span>') in texts[4]
    assert 'No nonconformances.' in texts[4]
    def register_owes():
        return [line for line in fl.check(project[name], project)
                if line.startswith(('D-', 'A-', 'NC-', 'Deviations',
                                    'Anomalies', 'Nonconformances'))]

    assert register_owes() == ['A-1: disposition not yet written']
    project.report_finding(name, kind='nonconformance')
    assert register_owes() == [
        'A-1: disposition not yet written',
        'NC-1: fact and disposition not yet written']


# ---- the typed parts --------------------------------------------------------


def test_the_typed_parts_are_drawn_as_typed_and_opinions_as_opinions():
    project, name = _project()
    project.report_flash(name, **TYPED)
    texts = _texts(project, name)
    assert ('<p class="flash-typed"><strong>Condition on release:</strong> '
            'No damage seen; released to the article engineer.</p>'
            ) in texts[4]
    assert texts[5].endswith(
        '<p class="flash-opinion"><em>Opinion, by analyst:</em> The Z peak '
        'looks like the bracket.</p>')
    assert 'within 30 days' in texts[6] and 'Full data review' in texts[6]
    assert fl.held(project[name])['opinions'] == [
        {'by': 'analyst', 'text': 'The Z peak looks like the bracket.'}]
    project.report_flash(name, opinions=[], opinions_none=True)
    assert fl.NO_OPINIONS in _texts(project, name)[5]
    project.report_flash(name, condition='', opinions_none=False)
    assert 'Condition on release:' in _texts(project, name)[4]
    assert 'condition' not in fl.held(project[name])


@pytest.mark.parametrize(('fields', 'error', 'match'), [
    ({'opinions': [('', 'Looks fine.')]}, ValueError, 'who gave it'),
    ({'opinions': [('analyst', ' ')]}, ValueError, 'opinion is not entered'),
    ({'opinions': [{'by': 'a', 'text': 'b', 'when': 'c'}]}, ValueError,
     "unknown part 'when'"),
    ({'opinions': 'Looks fine.'}, TypeError, 'list of opinions'),
    ({'opinions': [('analyst',)]}, TypeError, 'opinion 1'),
    ({'opinions': [(3, 'Looks fine.')]}, TypeError, 'each a str'),
    ({'opinions': [('analyst', 3)]}, TypeError, 'each a str'),
    ({'opinions': [('a', 'b')], 'opinions_none': True}, ValueError,
     'also said to be none'),
    ({'opinions_none': True}, ValueError, 'also said to be none'),
    ({'condition': 3}, TypeError, 'Condition on release is a str'),
    ({'opinions_none': 'yes'}, TypeError, 'True or False'),
])
def test_a_refused_part_changes_nothing(fields, error, match):
    project, name = _project()
    project.report_flash(name, opinions=[('analyst', 'Looks fine.')])
    before = dict(project[name].identity)
    with pytest.raises(error, match=match):
        project.report_flash(name, **fields)
    assert project[name].identity == before


def test_the_feature_off_refuses_the_typed_parts_and_the_check():
    project, name = _project()
    rf.configure(project[name], disable=[fl.FEATURE])
    with pytest.raises(ValueError, match='Flash report enabled'):
        project.report_flash(name, condition='Released.')
    with pytest.raises(ValueError, match='Flash report enabled'):
        fl.check(project[name], project)


# ---- issue, the check and the verb -------------------------------------------


def test_issue_waits_on_what_the_page_owes_not_the_skeleton(tmp_path):
    project, name = _project()
    project.report_identity(name, **COMPLETE)
    with pytest.raises(ValueError, match='flash report owes 7 parts: '
                       'Deviations: neither entered'):
        project.issue_report(name, tmp_path / 'FR.pdf')
    _complete(project, name)
    assert fl.check(project[name], project) == []
    assert ri.issue_ready(project[name], project) == 'issued'
    project._issue(project[name], None, lambda _page: b'%PDF stand-in',
                   'FR.pdf')
    assert ri.status_words(project[name]) == 'Preliminary (issued)'
    assert 'Preliminary (issued)' in _texts(project, name)[1]


def test_the_verb_is_journaled_and_a_refusal_is_not():
    project, name = _project()
    project.report_flash(name, report_date='2026-11-01')
    assert project.journal[-1] == (
        f"project.report_flash({name!r}, report_date='2026-11-01')")
    before = list(project.journal)
    with pytest.raises(ValueError):
        project.report_flash(name, opinions=[('', 'x')])
    assert project.journal == before


# ---- persistence --------------------------------------------------------------


def test_the_typed_parts_round_trip_and_never_reach_a_template(tmp_path):
    from visualdynamics import io
    from visualdynamics.io import report_template

    project, name = _project()
    project.report_identity(name, **COMPLETE)
    project.report_flash(name, **TYPED)
    io.save(project[name], str(tmp_path / 'r.vdyn'))
    assert io.load(str(tmp_path / 'r.vdyn')).identity == project[name].identity
    report_template.save(project[name], tmp_path / 'house.vdreport')
    back = report_template.load(tmp_path / 'house.vdreport')
    assert fl.held(back) == {}, "one test's page never passes to the next"
    assert back.features == list(FLASH)


# ---- the PDF ------------------------------------------------------------------


def test_the_flash_report_prints_on_one_page(tmp_path, qt_app):
    from test_report_identity import _pdf_pages

    project, name = _project(hot=[(0, 100.0, 200.0, 10.0)])
    project.report_finding(name, kind='deviation', fact='C2 above its band.',
                           approval='Approved at the post-test review.')
    project.report_finding(name, kind='anomaly', status='closed',
                           fact='M1 clipped.', approval='Closed by the analyst.')
    _complete(project, name)
    path = tmp_path / 'FR-0037-1.pdf'
    project.issue_report(name, path)
    pages = _pdf_pages(path)
    assert len(pages) == 1
    [text] = pages
    assert 'PRELIMINARY' in text and 'Page 1 of 1' in text
    assert 'Report FR-0037-1, Revision 0, Preliminary (issued)' in text
    assert findings.register(project[name])[0]['number'] == 'D-1'
    assert 'End of report.' in text


# ---- what a run log brought into the project adds ---------------------------


def _floor_log(path):
    """A random test on Z: run 1 completed at 10:00, run 2 aborted at 2:00
    with no cause logged (a hold), an anomaly open with its owner, an
    opinion given to the customer, and the setup before the first low
    level never logged (owed)."""
    with writer(path) as log:
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header=HEADER)
        model.append(log, 'run_start', author='AB', test=1, axis='Z', run=1,
                     fields={'counter': '0:00'})
        model.append(log, 'level_row', author='AB', test=1, axis='Z', run=1,
                     fields={'level_db': 0.0})
        model.append(log, 'run_stop', author='AB', test=1, axis='Z', run=1,
                     fields={'counter': '10:00'})
        model.append(log, 'run_start', author='AB', test=1, axis='Z', run=2,
                     fields={'counter': '0:00'})
        model.append(log, 'abort', author='AB', test=1, axis='Z', run=2,
                     fields={'stop': 'amplifier trip', 'counter': '2:00'})
        model.append(log, 'anomaly_opened', author='AB', test=1,
                     fields={'fact': 'a rattle at 300 Hz', 'owner': 'CD'})
        model.append(log, 'opinion_given', author='AB', test=1,
                     fields={'to': 'EF', 'opinion': 'the rattle looks like '
                             'the cable tie, not the article'})


def _flash_with_log(tmp_path, log=_floor_log):
    project, name = _project()
    path = tmp_path / 'article.vdlog'
    log(path)
    held = project.attach_run_log(path)
    return project, name, held, path


def _log_part(project, name):
    return next(text for text in _texts(project, name)
                if text.startswith(f'<h2 class="flash">{fl.FROM_LOG}</h2>'))


def test_without_a_run_log_the_flash_page_is_as_before(tmp_path):
    project, name = _project()
    _complete(project, name)
    before = _render(project, name)
    path = tmp_path / 'article.vdlog'
    _floor_log(path)
    held = project.attach_run_log(path)
    with_log = _render(project, name)
    assert fl.FROM_LOG in with_log and fl.LOG_CSS in with_log
    project.detach_run_log(held)
    assert _render(project, name) == before
    assert fl.LOG_CSS not in before
    assert fl.log_html(project[name], project) == ''


def test_what_the_log_adds_and_what_the_page_already_says_is_not_repeated(
        tmp_path):
    project, name, held, _path = _flash_with_log(tmp_path)
    extract = project[held]
    texts = _texts(project, name)
    assert len(texts) == 8
    assert texts[-2].startswith(f'<h2 class="flash">{fl.FROM_LOG}</h2>')
    assert texts[-1].startswith('<h2 class="flash">What Happens Next</h2>')
    part = texts[-2]
    assert ('<p class="flash-log-source">'
            + html.escape(f'As recorded in {extract.cited()}.') + '</p>'
            in part)
    assert _part('Runs so far', 'run 1 (Z) completed at 10:00; run 2 (Z) '
                 'aborted at 2:00') in part
    holds = extract.holds()
    assert len(holds) == 2
    assert _part('Holds open', 'an abort with no cause written (#6); an '
                 'abort whose over-test question is unanswered (#6)') in part
    assert _part('Anomalies open', 'A-1, owner CD: a rattle at 300 Hz') in part
    assert _part('Opinions logged', 'by AB to EF: the rattle looks like the '
                 'cable tie, not the article') in part
    assert html.escape('before the first low level, 8 of 8') in part
    # the page's own parts stay the author's: the logged opinion fills
    # nothing, and the check still owes the page's opinions
    assert 'Opinions: neither recorded nor set to None' in fl.check(
        project[name], project, project.object_groups)
    assert fl.OPINIONS_OWED in ''.join(texts)
    # what the page already says is left out: the anomaly once the
    # register holds it, the opinion once the page records it
    project.report_finding(name, kind='anomaly', fact='A rattle at 300 Hz.')
    project.report_flash(name, opinions=[
        ('AB', 'the rattle looks like the cable tie, not the article')])
    part = _log_part(project, name)
    assert 'Anomalies open' not in part and 'Opinions logged' not in part
    assert ('<p class="flash-log">The log holds no open anomaly or opinion by '
            'entry #8.</p>') in part
    assert all('\n' not in text for text in _texts(project, name))


def _part(title, text):
    return (f'<p class="flash-log"><strong>{html.escape(title)}:</strong> '
            f'{html.escape(text)}</p>')


def test_a_long_list_is_capped_with_what_the_log_holds_more(tmp_path):
    def many(path):
        with writer(path) as log:
            model.open_test(log, author='AB', test_kind='Random Vibration',
                            header=HEADER)
            for run in range(1, 6):
                model.append(log, 'run_stop', author='AB', test=1, axis='Z',
                             run=run, fields={'counter': f'{run}:00'})

    project, name, _held, _path = _flash_with_log(tmp_path, many)
    assert fl.LISTED == 3
    assert _part('Runs so far', 'run 1 (Z) completed at 1:00; run 2 (Z) '
                 'completed at 2:00; run 3 (Z) completed at 3:00; and 2 more, '
                 'in the run log.') in _log_part(project, name)
    assert fl._capped(['a', 'b', 'c']) == ['a', 'b', 'c']


def test_the_log_part_carries_the_badge_when_the_log_grew(tmp_path):
    project, name, _held, path = _flash_with_log(tmp_path)
    with writer(path) as log:
        log.append('note', author='AB', test=1, text='later')
    assert '[newer entries in the log].' in _log_part(project, name)
    os.remove(path)
    assert '[log not found].' in _log_part(project, name)


def test_an_empty_test_says_the_log_holds_none_never_that_there_was_none(
        tmp_path):
    def opened(path):
        with writer(path) as log:
            model.open_test(log, author='AB', test_kind='Random Vibration',
                            header=HEADER)

    project, name, _held, _path = _flash_with_log(tmp_path, opened)
    part = _log_part(project, name)
    for title in ('Runs so far', 'Holds open', 'Anomalies open',
                  'Opinions logged'):
        assert f'<strong>{title}:' not in part
    assert html.escape('The log holds no run, open hold, open anomaly or '
                       'opinion by entry #1.') in part
    assert fl.NO_OPINIONS not in part and 'No anomalies' not in part
    assert 'Anomalies: neither entered nor set to None' in fl.check(
        project[name], project, project.object_groups)
    assert fl._nothing(['run'], 4) == 'The log holds no run by entry #4.'


def test_a_typical_day_with_a_run_log_still_prints_on_one_page(tmp_path,
                                                               qt_app):
    """The flash report is a day on one sheet, and a run log attached
    keeps it there: the log's part says only what the page does not, and
    the page is set tighter while the part is on it."""
    from test_report_identity import _pdf_pages

    project, name, _held, _path = _flash_with_log(tmp_path, _typical_day)
    project.report_finding(name, kind='anomaly', status='open',
                           fact='M1 clipped.', approval='Open, owner CD.')
    _complete(project, name)
    path = tmp_path / 'FR-0037-1.pdf'
    project.issue_report(name, path)
    pages = _pdf_pages(path)
    assert len(pages) == 1
    [text] = pages
    assert 'Page 1 of 1' in text and 'From the Run Log' in text
    assert 'End of report.' in text
    assert 'Runs so far: run 1 (Z) completed at 10:00' in text
    # what the page says already is not said twice
    assert 'Anomalies open' not in text and 'Opinions logged' not in text


def test_the_runs_so_far_are_in_the_order_they_were_started(tmp_path):
    def interleaved(path):
        # run 1 starts, run 2 starts and stops, then run 1 stops: the
        # runs are listed by their first entry
        with writer(path) as log:
            model.open_test(log, author='AB', test_kind='Random Vibration',
                            header=HEADER)
            for run, counter in ((1, None), (2, None), (2, '1:00'),
                                 (1, '5:00')):
                if counter is None:
                    model.append(log, 'run_start', author='AB', test=1,
                                 axis='Z', run=run)
                else:
                    model.append(log, 'run_stop', author='AB', test=1,
                                 axis='Z', run=run,
                                 fields={'counter': counter})

    project, name, _held, _path = _flash_with_log(tmp_path, interleaved)
    assert _part('Runs so far', 'run 1 (Z) completed at 5:00; run 2 (Z) '
                 'completed at 1:00') in _log_part(project, name)


def _typical_day(path):
    """A typical day on one axis: the setup logged, three runs (the second
    tripped, its cause, path and restart logged), an anomaly the author
    has also entered in the register, and an opinion the page records in
    the same words; the release checklist still to do."""
    with writer(path) as log:
        log.append('clock_check', author='AB', fields={'offset_s': 0.0})
        model.open_test(log, author='AB', test_kind='Random Vibration',
                        header=HEADER)
        for step, _label in model.BEFORE_LOW_LEVEL:
            model.append(log, 'setup_step', author='AB', test=1, axis='Z',
                         fields={'step': step})
        for run, ends in ((1, 'run_stop'), (2, 'abort'), (3, 'run_stop')):
            model.append(log, 'run_start', author='AB', test=1, axis='Z',
                         run=run, fields={'counter': '0:00'})
            for level in (-12.0, -6.0, 0.0):
                model.append(log, 'level_row', author='AB', test=1, axis='Z',
                             run=run, fields={'level_db': level})
            if ends == 'abort':
                stop = model.append(log, 'abort', author='AB', test=1,
                                    axis='Z', run=run,
                                    fields={'stop': 'amplifier trip',
                                            'counter': '3:00',
                                            'called_by': 'AB',
                                            'heard': 'a clunk',
                                            'records_read': 'all',
                                            'over_test': 'no',
                                            'channel': 'amplifier',
                                            'band': 'all',
                                            'panel_fault': 'fan'})
                model.append(log, 'cause_found', author='AB', test=1,
                             fields={'cause': 'a fan fault',
                                     'refers_to': stop.id})
                model.append(log, 'path_decided', author='AB', test=1,
                             fields={'path': 'restart',
                                     'refers_to': stop.id})
                model.append(log, 'restart', author='AB', test=1, axis='Z',
                             run=run, fields={'resume': False,
                                              'cause': 'a fan fault',
                                              'refers_to': stop.id})
            model.append(log, 'run_stop', author='AB', test=1, axis='Z',
                         run=run, fields={'counter': '10:00'})
        model.append(log, 'anomaly_opened', author='AB', test=1,
                     fields={'fact': 'M1 clipped.', 'owner': 'CD'})
        model.append(log, 'opinion_given', author='AB', test=1,
                     fields={'to': 'EF',
                             'opinion': 'The Z peak looks like the bracket.'})
