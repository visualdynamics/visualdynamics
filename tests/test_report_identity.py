"""Identity and issue (report feature `report_identity`).

The identification the author enters, and nothing the program makes
up; the identity on every printed page with page n of N, the
identification block on the first page and the end mark on the last;
the status from draft to issued, an issued report never written again
under its identity once it has changed, and an amendment with its own
revision and a table of what changed. Then persistence, the editor's
fields and Amend, and a real PDF whose page numbers are read back.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.data import Psd
from visualdynamics.core.report import Report
from visualdynamics.report import features as rf
from visualdynamics.report import identity as ri
from visualdynamics.report import page, render_html
from visualdynamics.report import skeleton as sk

#: every field the identification needs, entered
COMPLETE = {
    'number': 'R-0037', 'revision': '0', 'laboratory': 'Test Lab',
    'address': '1 Main Street, Albuquerque', 'location': 'Bay 2',
    'customer': 'Acme', 'contact': 'J. Doe, jdoe@example.com',
    'received': '2026-10-01', 'tested': '2026-10-02',
    'issued': '2026-10-05',
    'authorized': ['A. Engineer, test engineer', 'Q. Signer, quality']}

#: what the fake print returns in place of a PDF
PDF = b'%PDF-1.4 stand-in'


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


def _psd(values=None):
    frequencies = np.linspace(10.0, 2000.0, 60)
    ordinate = (np.full((1, 60), 0.01) if values is None
                else np.asarray(values, dtype=float).reshape(1, 60))
    return Psd(abscissa=frequencies, ordinate=ordinate, response_dof=['1X+'],
               ordinate_dim=['acceleration**2/frequency'])


def _report(features=(ri.FEATURE,), **fields):
    report = Report('Vibration Test Report',
                    [{'kind': 'text', 'text': '## Results'},
                     {'kind': 'plot', 'source': 'P', 'caption': 'The ASD'}],
                    features=list(features))
    if fields:
        ri.identify(report, **fields)
    return report


def _project(report, name='Report'):
    project = visualdynamics.Project()
    project.add('P', _psd())
    return project, project.add(name, report)


def _issue(project, name, file='R-0037.pdf'):
    """Issue through the project's one sequence, with a stand-in for the
    print: everything but the web engine."""
    project._issue(project[name], None, lambda _page: PDF, file)


def _page(project, name):
    return render_html(project[name], project,
                       object_groups=project.object_groups)


# ---- entering the identification ------------------------------------------


def test_fields_are_entered_and_a_blank_clears():
    report = _report(**COMPLETE)
    assert ri.value(report, 'number') == 'R-0037'
    assert ri.value(report, 'authorized') == COMPLETE['authorized']
    assert ri.missing(report) == []
    ri.identify(report, number='  ', distribution='Customer; file')
    assert 'number' not in report.identity
    assert ri.missing(report) == ['Report number']
    assert ri.value(report, 'distribution') == 'Customer; file'
    ri.identify(report, authorized='B. One, engineer\n\n C. Two, quality ')
    assert ri.value(report, 'authorized') == ['B. One, engineer',
                                              'C. Two, quality']


def test_nothing_is_entered_for_the_author():
    """A new report has no number, revision, date or name, and the
    statement is on: nothing stored at all."""
    report = Report(features=[ri.FEATURE])
    assert report.identity == {}
    assert ri.value(report, 'status') == 'draft'
    assert ri.value(report, 'full_only') is True
    assert ri.missing(report) == [field.label for field in ri.FIELDS
                                  if field.required]
    ri.identify(report, full_only=False)
    assert report.identity == {'full_only': False}
    ri.identify(report, full_only=True)
    assert report.identity == {}, 'the default is not stored'


@pytest.mark.parametrize(('fields', 'error', 'match'), [
    ({'colour': 'red'}, ValueError, "unknown identification field 'colour'"),
    ({'status': 'issued'}, ValueError, 'status is not entered'),
    ({'record': []}, ValueError, 'record is not entered'),
    ({'number': 37}, TypeError, 'Report number is a str'),
    ({'full_only': 'yes'}, TypeError, 'full_only is True or False'),
    ({'authorized': 3}, TypeError, 'authorized is a list'),
    ({'authorized': ['A', 4]}, TypeError, 'each name is a str'),
])
def test_a_refused_field_changes_nothing(fields, error, match):
    report = _report(number='R-1', laboratory='Lab')
    before = dict(report.identity)
    with pytest.raises(error, match=match):
        ri.identify(report, laboratory='Other', **fields)
    assert report.identity == before


def test_the_model_holds_a_mapping():
    with pytest.raises(TypeError, match='identity is a mapping'):
        Report(identity=['number'])
    report = Report(identity={'number': 'X'})
    held = report.identity
    report.identity = {'number': 'Y'}
    assert held == {'number': 'X'}, 'assigning stores a copy'


# ---- the page --------------------------------------------------------------


def test_off_the_page_is_the_page_it_always_was():
    """Identification entered, issued even, but the feature off: the
    page is byte for byte the page with no identification at all."""
    plain = render_html(_report(features=()), {'P': _psd()})
    held = _report(features=(), **COMPLETE)
    held.identity = {**held.identity, 'status': 'issued',
                     'record': [{'number': 'R-0037', 'revision': '0'}]}
    assert render_html(held, {'P': _psd()}) == plain
    assert '@page' not in plain and 'identity' not in plain
    ri.guard(held, 'anything')                  # nothing to guard: off


def test_on_the_identification_opens_and_the_end_mark_closes():
    report = _report(**COMPLETE)
    html = render_html(report, {'P': _psd()})
    blocks = _payload(html)['blocks']
    first, last = blocks[0], blocks[-1]
    assert first['kind'] == 'text' and 'class="identity"' in first['html']
    for given in ('R-0037', 'Test Lab', '1 Main Street, Albuquerque',
                  'Bay 2', 'Acme', 'J. Doe, jdoe@example.com', '2026-10-01',
                  '2026-10-02', '2026-10-05', 'A. Engineer, test engineer'):
        assert given in first['html'], given
    assert first['html'].count('class="signature"') == 2
    assert first['html'].count('>Signature<') == 2, 'each line printed blank'
    assert ri.FULL_ONLY in first['html']
    assert last['html'] == '<p class="endmark">End of report.</p>'
    assert [b['kind'] for b in blocks[1:-1]] == ['text', 'plot'], \
        "the report's own blocks between, in order"
    assert blocks[2]['label'] == 'Figure 1', 'numbering untouched'


def test_a_field_not_entered_says_so_and_is_never_filled():
    report = _report(number='R-9')
    first = _payload(render_html(report, {'P': _psd()}))['blocks'][0]['html']
    assert first.count(ri.NOT_ENTERED) == len(ri.missing(report)), \
        'one caution per required field missing, authorized included'
    assert 'Distribution' not in first, 'an optional field left out'
    ri.identify(report, full_only=False)
    first = _payload(render_html(report, {'P': _psd()}))['blocks'][0]['html']
    assert ri.FULL_ONLY not in first


def test_the_printed_margins_carry_the_identity_and_page_n_of_n():
    report = _report(**COMPLETE)
    report.marking = 'CUI'
    report.marking_color = 'red'
    css = ri.page_css(report)
    assert '@top-left { content: "Test Lab";' in css
    assert '@top-right { content: "Report R-0037, Revision 0, Draft";' in css
    assert '@bottom-left { content: "Vibration Test Report";' in css
    assert 'content: "Page " counter(page) " of " counter(pages);' in css
    assert css.count('content: "CUI"') == 2 and '#d02b2b' in css
    html = render_html(report, {'P': _psd()})
    assert css in html, 'in the page itself, so a browser prints it too'
    report.marking = ''
    assert 'top-center' not in ri.page_css(report)
    assert '(laboratory not entered)' in ri.page_css(_report())
    assert '(number not entered)' in ri.page_css(_report())


def test_entered_text_cannot_escape_the_page():
    report = _report(laboratory='Lab "A" \\ </style><script>x()</script>',
                     customer='<b>Acme</b> & Sons')
    css = ri.page_css(report)
    assert '</style>' not in css and '<' not in css
    assert '"Lab \\"A\\" \\\\ \\3c /style>' in css
    html = render_html(report, {'P': _psd()})
    assert '<script>x()' not in html and '&lt;b&gt;Acme' in html
    assert ri._css_string('a\nb\x7f') == '"a\\a b\\7f "'


def test_the_page_code_is_its_own_entry():
    css, js = page.page_parts([ri.FEATURE])
    assert 'printScale()' in js and 'function printScale()' in js
    assert '@media print' in css and 'div.identity' in css
    assert page.page_parts(())[1] is page._JS


def test_dead_nan_and_empty_inputs_still_carry_the_identity():
    """An all-zero channel, a record with a NaN, a constant channel and
    a report with no blocks: each page opens with the identification
    and closes with the end mark; the identity judges no data."""
    nan = np.full(60, 0.01)
    nan[7] = np.nan
    for values in (np.zeros(60), nan, np.full(60, 0.5)):
        blocks = _payload(render_html(_report(**COMPLETE),
                                      {'P': _psd(values)}))['blocks']
        assert 'class="identity"' in blocks[0]['html']
        assert blocks[-1]['html'].endswith('End of report.</p>')
    empty = Report('Empty', features=[ri.FEATURE])
    blocks = _payload(render_html(empty, {}))['blocks']
    assert len(blocks) == 2 and 'class="identity"' in blocks[0]['html']


# ---- with skeleton sections -------------------------------------------------


def test_with_the_skeleton_the_identification_section_is_the_block():
    report = _report(features=(sk.FEATURE, ri.FEATURE), **COMPLETE)
    blocks = _payload(render_html(report, {'P': _psd()}))['blocks']
    assert blocks[0]['html'] == '<h2 class="skeleton">1. Identification</h2>'
    assert 'class="identity"' in blocks[1]['html']
    assert sk.OWED not in blocks[0]['html']
    assert blocks[-1]['html'] == '<p class="endmark">End of report.</p>'
    assert sum('endmark' in b.get('html', '') for b in blocks) == 1
    assert 'identification' not in [o.key for o in sk.check(report,
                                                           {'P': _psd()})]
    ri.identify(report, customer='', issued='')
    owed = {o.key: o.reason for o in sk.check(report, {'P': _psd()})}
    assert owed['identification'] == \
        'Customer, Date of issue not entered'
    # the author's own words in section 1 stay, above the block
    sk.write(report, 'identification', 'Prepared under contract 12.')
    blocks = _payload(render_html(report, {'P': _psd()}))['blocks']
    assert 'Prepared under contract 12.' in blocks[0]['html']
    assert 'class="identity"' in blocks[1]['html']


def test_the_skeleton_alone_is_unchanged():
    report = _report(features=(sk.FEATURE,), **COMPLETE)
    blocks = _payload(render_html(report, {'P': _psd()}))['blocks']
    assert sk.OWED in blocks[0]['html'], 'still owed: identity is off'
    assert not any('class="identity"' in b.get('html', '') for b in blocks)


# ---- issue -------------------------------------------------------------------


def test_issue_records_the_revision_and_says_issued():
    project, name = _project(_report(**COMPLETE))
    _issue(project, name)
    report = project[name]
    assert ri.value(report, 'status') == 'issued'
    [entry] = ri.record(report)
    page_now = _page(project, name)
    assert entry == {'number': 'R-0037', 'revision': '0', 'status': 'issued',
                     'issued': '2026-10-05',
                     'sha256': hashlib.sha256(PDF).hexdigest(),
                     'content': ri.content_hash(page_now),
                     'file': 'R-0037.pdf'}
    assert '<td>Issued</td>' in _payload(page_now)['blocks'][0]['html']
    assert 'Revision 0";' in ri.page_css(report), 'no Draft beside it'


@pytest.mark.parametrize(('change', 'match'), [
    ({'features': []}, 'does not have Identity and issue enabled'),
    ({'missing': ('customer', 'authorized')},
     'Customer, Authorized by are not entered'),
    ({'missing': ('issued',)}, 'Date of issue is not entered'),
])
def test_issue_is_refused_before_anything_changes(change, match, tmp_path):
    report = _report(**COMPLETE)
    if 'features' in change:
        report.features = change['features']
    for key in change.get('missing', ()):
        ri.identify(report, **{key: [] if key == 'authorized' else ''})
    project, name = _project(report)
    before = dict(report.identity)
    target = tmp_path / 'never.pdf'
    with pytest.raises(ValueError, match=match):
        project.issue_report(name, target)
    assert report.identity == before and not target.exists()
    assert not any('issue_report' in line for line in project.journal)


def test_issue_is_refused_while_the_skeleton_owes_sections(tmp_path):
    project, name = _project(_report(features=(sk.FEATURE, ri.FEATURE),
                                     **COMPLETE))
    with pytest.raises(ValueError, match='cannot be issued while it owes '
                                         r'\d+ sections: 2\. Summary'):
        project.issue_report(name, tmp_path / 'never.pdf')
    assert ri.value(project[name], 'status') == 'draft'


def test_a_report_is_issued_once(tmp_path):
    project, name = _project(_report(**COMPLETE))
    _issue(project, name)
    with pytest.raises(ValueError, match='already issued; a change after '
                                         'issue is an amendment'):
        project.issue_report(name, tmp_path / 'again.pdf')
    assert len(ri.record(project[name])) == 1


def test_a_failed_print_leaves_the_report_a_draft():
    project, name = _project(_report(**COMPLETE))

    def fail(_page):
        raise OSError('the printer caught fire')

    with pytest.raises(OSError, match='caught fire'):
        project._issue(project[name], None, fail, 'x.pdf')
    assert ri.value(project[name], 'status') == 'draft'
    assert ri.record(project[name]) == []


def test_an_unknown_page_size_is_refused(tmp_path):
    project, name = _project(_report(**COMPLETE))
    with pytest.raises(ValueError, match="unknown page size 'tabloid'"):
        project.issue_report(name, tmp_path / 'x.pdf', page_size='tabloid')
    from visualdynamics.report.pdf import print_pdf

    with pytest.raises(ValueError, match="unknown page size 'b5'"):
        print_pdf('<html></html>', tmp_path / 'x.pdf', page_size='b5')


def test_the_number_and_revision_of_an_issued_report_are_fixed():
    project, name = _project(_report(**COMPLETE))
    _issue(project, name)
    for key in ('number', 'revision'):
        with pytest.raises(ValueError, match='cannot change: the report was '
                                             'issued as Report R-0037'):
            project.report_identity(name, **{key: 'X'})
    project.report_identity(name, number='R-0037')        # unchanged: fine


# ---- no silent reissue -----------------------------------------------------


def test_an_issued_report_is_exported_again_only_unchanged(tmp_path):
    project, name = _project(_report(**COMPLETE))
    _issue(project, name)
    out = project.export_report(name, tmp_path / 'copy.html')
    with open(out, encoding='utf-8') as handle:
        assert ri.content_hash(handle.read()) == \
            ri.record(project[name])[0]['content'], 'a copy of the issued'
    changes = [
        lambda r: setattr(r, 'title', 'Another Title'),
        lambda r: r.blocks.append({'kind': 'text', 'text': 'A late note.'}),
        lambda r: ri.identify(r, customer='Someone Else'),
        lambda r: setattr(r, 'marking', 'SECRET'),
    ]
    for change in changes:
        report = project[name]
        saved = (report.title, [dict(b) for b in report.blocks],
                 dict(report.identity), report.marking)
        change(report)
        for export in (project.export_report, project.export_report_pdf):
            with pytest.raises(ValueError, match='was issued on 2026-10-05 '
                                                 'and has changed since'):
                export(name, tmp_path / 'changed.out')
        assert not (tmp_path / 'changed.out').exists()
        report.title, report.blocks, report.identity, report.marking = saved
    # the data under a figure changed is a change too
    project['P'] = _psd(np.full(60, 0.02))
    with pytest.raises(ValueError, match='has changed since'):
        project.export_report(name, tmp_path / 'changed.html')


def test_a_draft_exports_as_often_as_wanted(tmp_path):
    project, name = _project(_report(**COMPLETE))
    project.export_report(name, tmp_path / 'a.html')
    project[name].title = 'Retitled'
    project.export_report(name, tmp_path / 'b.html')
    assert 'Report R-0037, Revision 0, Draft' in (
        tmp_path / 'b.html').read_text(encoding='utf-8')


# ---- amendment ---------------------------------------------------------------

CHANGE = {'where': 'Page 3, Table 1', 'was': 'Sensitivity 10.2 mV/g',
          'now': 'Sensitivity 12.0 mV/g', 'reason': 'Digits transposed'}


def test_an_amendment_has_its_own_revision_and_a_table_of_changes(tmp_path):
    project, name = _project(_report(**COMPLETE))
    _issue(project, name)
    project.amend_report(name, [CHANGE], revision='1')
    report = project[name]
    assert project.journal[-1] == (
        f'project.amend_report({name!r}, [{CHANGE!r}], revision=\'1\')')
    assert ri.value(report, 'status') == 'draft'
    assert ri.value(report, 'revision') == '1'
    assert ri.value(report, 'issued') == '', 'its own date, to be entered'
    assert report.identity['amends'] == {'number': 'R-0037', 'revision': '0',
                                         'issued': '2026-10-05'}
    assert ri.amendment(report) == 1
    assert ri.status_words(report) == 'Draft amendment'
    blocks = _payload(_page(project, name))['blocks']
    first = blocks[0]['html']
    assert ('Amendment 1 to Report R-0037, Revision 0, issued 2026-10-05.'
            in first)
    assert ('<tr><td>Page 3, Table 1</td><td>Sensitivity 10.2 mV/g</td>'
            '<td>Sensitivity 12.0 mV/g</td><td>Digits transposed</td></tr>'
            in first)
    assert blocks[-1]['html'] == '<p class="endmark">End of amendment.</p>'
    with pytest.raises(ValueError, match='Date of issue is not entered'):
        project.issue_report(name, tmp_path / 'a.pdf')
    project.report_identity(name, issued='2026-10-20')
    _issue(project, name, 'R-0037-1.pdf')
    assert ri.value(report, 'status') == 'amended'
    assert ri.status_words(report) == 'Amendment 1'
    original, amended = ri.record(report)
    assert original['revision'] == '0' and original['status'] == 'issued'
    assert amended['revision'] == '1' and amended['status'] == 'amended'
    assert amended['changes'] == [CHANGE]
    assert amended['amends']['revision'] == '0'
    # and a second, against the first amendment, under its own number
    project.amend_report(name, [{'where': 'Page 1', 'was': 'Bay 2',
                                 'now': 'Bay 3'}], revision='0',
                         number='R-0041')
    assert report.identity['amends'] == {'number': 'R-0037', 'revision': '1',
                                         'issued': '2026-10-20'}
    assert ri.amendment(report) == 2
    assert report.identity['changes'][0]['reason'] == ''


@pytest.mark.parametrize(('args', 'error', 'match'), [
    (([], '1'), ValueError, 'at least one change'),
    (([{'where': 'p1', 'was': 'a'}], '1'), ValueError,
     'change 1: now is not entered'),
    (([{'where': 'p1', 'was': '', 'now': ''}], '1'), ValueError,
     'change 1: was, now are not entered'),
    (([{**CHANGE, 'page': 3}], '1'), ValueError, "unknown part 'page'"),
    (([{**CHANGE, 'now': 3}], '1'), TypeError, 'now is a str'),
    ((CHANGE, '1'), TypeError, 'changes is a list'),
    (('p1', '1'), TypeError, 'changes is a list'),
    (([3], '1'), TypeError, 'change 1 is a mapping'),
    (([CHANGE], ' '), ValueError, 'needs its own revision'),
    (([CHANGE], '0'), ValueError, 'Revision 0 has already been issued'),
    (([CHANGE], 1), TypeError, 'revision is a str'),
])
def test_an_amendment_is_refused_before_anything_changes(args, error, match):
    project, name = _project(_report(**COMPLETE))
    _issue(project, name)
    before = dict(project[name].identity)
    changes, revision = args
    with pytest.raises(error, match=match):
        project.amend_report(name, changes, revision=revision)
    assert project[name].identity == before


def test_a_draft_needs_no_amendment():
    project, name = _project(_report(**COMPLETE))
    with pytest.raises(ValueError, match='has not been issued; a draft is '
                                         'changed in place'):
        project.amend_report(name, [CHANGE], revision='1')
    project[name].features = []
    with pytest.raises(ValueError, match='nothing issued to amend'):
        project.amend_report(name, [CHANGE], revision='1')
    _issue_anyway = _report(**COMPLETE)
    with pytest.raises(ValueError, match='needs a report number'):
        project2, name2 = _project(_issue_anyway)
        _issue(project2, name2)
        project2.amend_report(name2, [CHANGE], revision='1', number=' ')


def test_the_end_mark_follows_the_amendment():
    report = _report(**COMPLETE)
    assert ri.end_mark(report) == sk.END
    report.identity = {**report.identity, 'amends': {'number': 'R'}}
    assert ri.end_mark(report) == ri.END_AMENDMENT


# ---- persistence ---------------------------------------------------------------


def test_the_identity_and_record_round_trip_through_the_project_file(
        tmp_path):
    from visualdynamics import io

    project, name = _project(_report(**COMPLETE))
    _issue(project, name)
    io.save(project[name], str(tmp_path / 'r.vdyn'))
    back = io.load(str(tmp_path / 'r.vdyn'))
    assert back.identity == project[name].identity


def test_a_report_with_no_identity_is_stored_as_it_always_was(tmp_path):
    import h5py

    from visualdynamics import io

    io.save(Report('R'), str(tmp_path / 'r.vdyn'))
    with h5py.File(tmp_path / 'r.vdyn') as handle:
        attributes = set()
        handle.visititems(lambda _n, item: attributes.update(item.attrs))
    assert 'identity' not in attributes
    assert io.load(str(tmp_path / 'r.vdyn')).identity == {}


def test_a_template_carries_the_laboratory_and_never_the_issue(tmp_path):
    from visualdynamics.io import report_template

    project, name = _project(_report(**COMPLETE, full_only=False))
    _issue(project, name)
    report_template.save(project[name], tmp_path / 'house.vdreport')
    saved = json.loads((tmp_path / 'house.vdreport').read_text('utf-8'))
    assert saved['identity'] == {'laboratory': 'Test Lab',
                                 'address': '1 Main Street, Albuquerque',
                                 'full_only': False}
    back = report_template.load(tmp_path / 'house.vdreport')
    assert back.identity == saved['identity']
    assert ri.value(back, 'status') == 'draft' and ri.record(back) == []
    report_template.save(Report('R'), tmp_path / 'plain.vdreport')
    assert 'identity' not in (tmp_path / 'plain.vdreport').read_text('utf-8')
    (tmp_path / 'bad.vdreport').write_text(json.dumps(
        {report_template.MARKER: 1, 'identity': ['x']}), 'utf-8')
    with pytest.raises(ValueError, match='identity must be a mapping'):
        report_template.load(tmp_path / 'bad.vdreport')
    # a forged number in a template file does not reach the report
    (tmp_path / 'forged.vdreport').write_text(json.dumps(
        {report_template.MARKER: 1, 'identity': {'number': 'R-1',
                                                 'status': 'issued'}}),
        'utf-8')
    assert report_template.load(tmp_path / 'forged.vdreport').identity == {}


def test_the_project_verbs_are_journaled():
    project, name = _project(_report())
    project.report_identity(name, number='R-1')
    assert project.journal[-1] == (
        f"project.report_identity({name!r}, number='R-1')")
    before = list(project.journal)
    with pytest.raises(ValueError):
        project.report_identity(name, colour='red')
    assert project.journal == before, 'a refusal is not journaled'
    assert rf.FEATURES[ri.FEATURE].available


# ---- the editor --------------------------------------------------------------


@pytest.fixture
def editing(window, pump, project):
    for name, obj in project.items():
        window.add_object(name, obj)
    name = window.generate_report('modal')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    return window, window.report_editor, window.objects[name], name


def test_the_fields_show_with_the_feature_and_are_journaled(editing):
    window, editor, report, name = editing
    assert editor.identity_box is not None
    assert editor.identity_box.isHidden(), 'off: no identity fields'
    editor._operate({'op': 'features', 'enable': [ri.FEATURE]})
    assert not editor.identity_box.isHidden()
    edit = editor.identity_edits['number']
    edit.setText('R-0050')
    edit.editingFinished.emit()
    assert ri.value(report, 'number') == 'R-0050'
    assert (f"project.report_identity({name!r}, number='R-0050')"
            in window.project.journal)
    people = editor.identity_edits['authorized']
    people.setText('A. One, engineer; B. Two, quality')
    people.editingFinished.emit()
    assert ri.value(report, 'authorized') == ['A. One, engineer',
                                              'B. Two, quality']
    editor.identity_edits['full_only'].setChecked(False)
    assert ri.value(report, 'full_only') is False
    with open(editor._page_path, encoding='utf-8') as page_file:
        assert 'R-0050' in page_file.read(), 'the page re-rendered'
    editor._operate({'op': 'features', 'disable': [ri.FEATURE]})
    assert editor.identity_box.isHidden()


def test_an_issued_report_offers_amend_and_fixes_its_number(editing,
                                                             monkeypatch):
    from visualdynamics.gui.amend_dialog import AmendDialog

    window, editor, report, name = editing
    window.project.report_features(name, enable=[ri.FEATURE])
    window.project.report_identity(name, **COMPLETE)
    window.project._issue(report, None, lambda _page: PDF, 'x.pdf')
    editor._show_selection()
    assert editor.identity_edits['number'].isReadOnly()
    assert not editor.identity_edits['laboratory'].isReadOnly()
    assert editor.amend_button is not None
    # a refused amendment is said under the fields and changes nothing
    editor._operate({'op': 'amend', 'revision': '0', 'changes': [CHANGE]})
    assert 'already been issued' in editor.identity_notes.text()
    assert ri.value(report, 'status') == 'issued'

    def answered(dialog):
        dialog.fill('1', [CHANGE])
        return True

    monkeypatch.setattr(AmendDialog, 'exec', answered)
    editor.amend_button.click()
    assert ri.value(report, 'revision') == '1'
    assert report.identity['changes'] == [CHANGE]
    assert (f"project.amend_report({name!r}, [{CHANGE!r}], revision='1', "
            'number=None)' in window.project.journal)


def test_the_dialog_reads_back_what_was_typed(qt_app):
    from visualdynamics.gui.amend_dialog import AmendDialog

    dialog = AmendDialog(_report(**COMPLETE))
    dialog.fill('2', [CHANGE, {'where': 'p4', 'was': 'a', 'now': 'b'}],
                number='R-9')
    dialog.changes.insertRow(2)                 # a blank row is skipped
    assert dialog.amendment() == {
        'revision': '2', 'number': 'R-9',
        'changes': [CHANGE, {'where': 'p4', 'was': 'a', 'now': 'b',
                             'reason': ''}]}
    dialog.fill('2', [CHANGE])
    assert dialog.amendment()['number'] is None
    dialog.deleteLater()


def test_the_menu_offers_pdf_and_issue_only_with_the_feature(editing,
                                                             qt_app):
    from conftest import menu_entries

    window, name = editing[0], editing[3]
    item = window._item_for_object(name)
    entries = menu_entries(window, qt_app, item)
    assert 'Export &Report...' in entries
    assert 'Export Report as P&DF...' not in entries
    assert '&Issue Report...' not in entries
    window.project.report_features(name, enable=[ri.FEATURE])
    entries = menu_entries(window, qt_app, item)
    assert 'Export Report as P&DF...' in entries
    assert '&Issue Report...' in entries


def test_the_window_refuses_issue_before_asking_for_a_file(editing,
                                                          monkeypatch):
    """A field not entered is said before the save dialog opens, and
    nothing is written or recorded."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    window, _editor, report, name = editing
    window.project.report_features(name, enable=[ri.FEATURE])
    told, asked = [], []
    monkeypatch.setattr(QMessageBox, 'warning',
                        lambda _p, title, text: told.append((title, text)))
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        lambda *a, **k: asked.append(a) or ('', ''))
    window.issue_report()
    assert told and told[0][0] == 'Issue Report'
    assert 'Report number' in told[0][1] and 'not entered' in told[0][1]
    assert asked == [] and ri.record(report) == []


def test_the_window_refuses_a_changed_issued_report(editing, monkeypatch,
                                                    tmp_path):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    window, _editor, report, name = editing
    window.project.report_features(name, enable=[ri.FEATURE])
    window.project.report_identity(name, **COMPLETE)
    window.project._issue(report, window.unit_system, lambda _page: PDF,
                          'x.pdf')
    report.title = 'Changed After Issue'
    told = []
    monkeypatch.setattr(QMessageBox, 'warning',
                        lambda _p, title, text: told.append((title, text)))
    target = tmp_path / 'changed.html'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        lambda *a, **k: (str(target), ''))
    window.export_report()
    assert told and 'has changed since' in told[0][1]
    assert not target.exists()


# ---- a real PDF ------------------------------------------------------------------


#: reads a PDF's pages as text, in a process of its own
_READ_PAGES = """
import json, sys
from PySide6.QtGui import QGuiApplication
from PySide6.QtPdf import QPdfDocument
app = QGuiApplication([])
document = QPdfDocument()
document.load(sys.argv[1])
print(json.dumps([' '.join(document.getAllText(at).text().split())
                  for at in range(document.pageCount())]))
"""


def _pdf_pages(path):
    """The text of each page of the PDF at `path`, read with QtPdf in a
    separate process. QtPdf is never loaded into a process running Qt
    WebEngine: both carry the browser engine's own libraries, and a
    test worker that loaded QtPdf beside a live web engine crashed in
    its next event loop on Linux."""
    import os
    import subprocess
    import sys

    run = subprocess.run(
        [sys.executable, '-c', _READ_PAGES, str(path)], capture_output=True,
        text=True, encoding='utf-8', timeout=120, check=True,
        env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.fixture(scope='module')
def issued(tmp_path_factory, qt_app):
    """A report of several pages issued for real, through the web engine,
    and read back."""
    report = _report(**COMPLETE)
    for at in range(40):
        report.add({'kind': 'text', 'text': f'## Part {at}\n\n' + ' '.join(
            ['A sentence long enough to fill the line it sits on.'] * 8)})
    report.marking = 'CUI'
    project, name = _project(report)
    path = tmp_path_factory.mktemp('pdf') / 'R-0037.pdf'
    project.issue_report(name, path)
    return project, name, path, _pdf_pages(path)


def test_page_n_of_n_is_what_the_pdf_holds(issued):
    pages = issued[3]
    total = len(pages)
    assert total >= 3, 'forty parts run to several pages'
    for at, text in enumerate(pages, start=1):
        assert f'Page {at} of {total}' in text, at
        assert 'Report R-0037, Revision 0' in text, at
        assert 'Test Lab' in text and text.count('CUI') == 2, at
        assert 'Draft' not in text, 'issued: the margins say no draft'
    assert 'Date of receipt of the item' in pages[0]
    assert ri.FULL_ONLY in pages[0]
    assert 'End of report.' in pages[-1]
    assert not any('End of report.' in text for text in pages[:-1])


def test_the_issued_pdf_is_recorded_by_its_checksum(issued):
    project, name, path, _pages = issued
    [entry] = ri.record(project[name])
    assert entry['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry['file'] == 'R-0037.pdf'
    assert path.read_bytes().startswith(b'%PDF')


def test_the_window_issues_with_the_print_handed_to_its_thread(
        editing, monkeypatch, tmp_path):
    """Issue Report renders on the window's worker, and the print, which
    the web engine allows only on the application's thread, is handed
    there while the worker waits."""
    import sys
    import threading

    from PySide6.QtWidgets import QFileDialog

    from visualdynamics.report import pdf

    assert 'PySide6.QtPdf' not in sys.modules,         'QtPdf stays out of a process running the web engine (_pdf_pages)'
    window, editor, report, name = editing
    window.project.report_features(name, enable=[ri.FEATURE])
    window.project.report_identity(name, **COMPLETE)
    printed_on = []
    real = pdf.print_pdf

    def watched(*args, **kwargs):
        printed_on.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(pdf, 'print_pdf', watched)
    target = tmp_path / 'R-0037.pdf'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        lambda *a, **k: (str(target), ''))
    window.issue_report()
    assert target.read_bytes().startswith(b'%PDF')
    assert ri.value(report, 'status') == 'issued'
    assert ri.record(report)[0]['sha256'] == hashlib.sha256(
        target.read_bytes()).hexdigest()
    assert printed_on[0] is not threading.main_thread(), 'asked off it'
    assert printed_on[-1] is threading.main_thread(), 'printed on it'
    assert any(line.startswith(f'project.issue_report({name!r}')
               for line in window.project.journal)
    assert editor.amend_button is not None, 'the pane offers Amend now'


def test_one_owed_section_is_named_in_the_singular(tmp_path):
    report = _report(features=(sk.FEATURE, ri.FEATURE), **COMPLETE)
    for found in sk.SECTIONS:
        if found.key not in ('identification', 'record'):
            sk.write(report, found.key, 'Written.',
                     **({'by': 'J. Doe'} if found.key == sk.ATTRIBUTED
                        else {}))
    project, name = _project(report)
    with pytest.raises(ValueError, match=r'owes 1 section: 12\. Record'):
        project.issue_report(name, tmp_path / 'never.pdf')


def test_only_the_revision_the_report_carries_counts_as_issued():
    """A record of revision 0 does not make revision 1 issued, though
    the number is the same: the guard reads the entry that matches both."""
    report = _report(**COMPLETE)
    report.identity = {**report.identity, 'revision': '1', 'status': 'issued',
                       'record': [{'number': 'R-0037', 'revision': '0',
                                   'content': 'another page'}]}
    ri.guard(report, 'this page')                # nothing issued as rev 1
    assert ri.amendment(report) == 0, 'not an amendment'


def test_an_amendment_with_the_skeleton_ends_as_an_amendment():
    report = _report(features=(sk.FEATURE, ri.FEATURE), **COMPLETE)
    report.identity = {**report.identity, 'amends': {
        'number': 'R-0037', 'revision': '0', 'issued': '2026-10-05'}}
    blocks = _payload(render_html(report, {'P': _psd()}))['blocks']
    assert blocks[-1]['html'] == '<p class="endmark">End of amendment.</p>'
    assert 'Amendment 1 to Report R-0037' in blocks[1]['html']


def test_a_page_that_does_not_load_writes_nothing(qt_app, tmp_path,
                                                  monkeypatch):
    from PySide6.QtWebEngineWidgets import QWebEngineView

    from visualdynamics.report.pdf import print_pdf

    monkeypatch.setattr(QWebEngineView, 'load',
                        lambda view, _url: view.page().loadFinished.emit(False))
    target = tmp_path / 'never.pdf'
    with pytest.raises(OSError, match='did not load'):
        print_pdf('<html></html>', target, timeout=5)
    assert not target.exists()


def test_a_print_that_fails_writes_nothing(qt_app, tmp_path, monkeypatch):
    """The engine reports the print failed: nothing reaches the path,
    whatever was there is kept, and the report stays a draft."""
    from PySide6.QtWebEngineCore import QWebEnginePage

    from visualdynamics.report.pdf import print_pdf

    monkeypatch.setattr(
        QWebEnginePage, 'printToPdf',
        lambda page, path, _layout: page.pdfPrintingFinished.emit(path,
                                                                  False))
    target = tmp_path / 'kept.pdf'
    target.write_bytes(b'what was there')
    with pytest.raises(OSError, match='could not print'):
        print_pdf('<html><body>x</body></html>', target, timeout=60)
    assert target.read_bytes() == b'what was there'
    project, name = _project(_report(**COMPLETE))
    with pytest.raises(OSError, match='could not print'):
        project.issue_report(name, tmp_path / 'issued.pdf')
    assert ri.value(project[name], 'status') == 'draft'
    assert not (tmp_path / 'issued.pdf').exists()
