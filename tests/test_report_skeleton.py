"""Skeleton sections (report feature `skeleton_sections`).

The sections and their order, where generated blocks fall, the owed
card and the one-step None, the discussion's attribution and the
conclusion's sentence, the check before export and its agreement with
the page, pipe tables in the Markdown subset, the project verbs, the
editor's pane and Export's question, and the promise the feature rests
on: off, the page is the page it was.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.data import Psd
from visualdynamics.core.report import Report, prune_unbound
from visualdynamics.report import features as rf
from visualdynamics.report import page, render_html
from visualdynamics.report import skeleton as sk
from visualdynamics.report.markdown import to_html


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


@pytest.fixture(scope='module')
def run():
    """A random run worked up, with its report."""
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'))
    name = project.generate_report('random')
    return project, name


def _render(project, name, **kwargs):
    return render_html(project[name], project,
                       unit_system=visualdynamics.SI,
                       object_groups=project.object_groups, **kwargs)


@pytest.fixture
def skeletal(run):
    """The run's report with the skeleton on, put back afterwards."""
    project, name = run
    report = project[name]
    blocks = [dict(block) for block in report.blocks]
    project.report_features(name, enable=['skeleton_sections'])
    yield project, name, report
    report.blocks = blocks
    report.features = []


def _heads(blocks):
    """(heading, html) for each skeleton head on the page, in order."""
    out = []
    for block in blocks:
        html = block.get('html', '')
        if html.startswith('<h2 class="skeleton">'):
            out.append((html.split('>', 1)[1].split('</h2>')[0], html))
    return out


def _sections(blocks):
    """The page's blocks grouped under the skeleton head they follow."""
    out, current = {}, None
    for block in blocks:
        html = block.get('html', '')
        if html.startswith('<p class="endmark">'):
            break
        if html.startswith('<h2 class="skeleton">'):
            current = html.split('>', 1)[1].split('.')[0]
            out[current] = []
        elif current is not None:
            out[current].append(block)
    return out


# ---- the sections -----------------------------------------------------------


def test_the_sections_are_the_skeleton_in_order():
    assert [s.number for s in sk.SECTIONS] == list(range(1, 13))
    assert len(sk.BY_KEY) == 12
    for found in sk.SECTIONS:
        assert found.title == found.title.title() or ' and ' in found.title
        assert found.description.endswith('.')
        assert found.heading == f'{found.number}. {found.title}'
    filled = [s.key for s in sk.SECTIONS if not s.required]
    assert filled == ['instrumentation', 'results'], \
        'only what the project generates can fill a section'
    can_be_none = {s.key: s.none for s in sk.SECTIONS if s.none}
    assert can_be_none == {
        'deviations': 'No deviations. No anomalies. No nonconformances.',
        'discussion': 'No interpretation is offered beyond the results.'}


def test_the_feature_is_built_and_off_by_default():
    feature = rf.FEATURES[sk.FEATURE]
    assert feature.available and not feature.default
    assert feature.requires == ()
    assert sk.FEATURE in rf.PRESETS['test_report'].features
    assert sk.FEATURE in rf.configure(Report(), preset='test_report').enabled


def test_an_unknown_section_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown skeleton section 'notes'"):
        sk.section('notes')
    with pytest.raises(TypeError, match='key, a str'):
        sk.section(4)


# ---- off: the page it was -----------------------------------------------------


def test_off_the_page_is_the_page_it_was_even_with_sections_written(run):
    """The section blocks the feature writes are drawn only while it is
    on: a report whose skeleton was filled in and then turned off renders
    byte for byte as it did before, read and edit pages alike."""
    project, name = run
    report = project[name]
    blocks = [dict(block) for block in report.blocks]
    plain = _render(project, name)
    plain_edit = _render(project, name, edit=True, channel_js='')
    try:
        project.report_features(name, enable=['skeleton_sections'])
        project.report_section(name, 'article', 'A bracket, S/N 7.')
        project.report_section(name, 'deviations', none=True)
        project.report_section(name, 'discussion', 'Read as ok.', by='K.')
        on = _render(project, name)
        assert on != plain and 'NOT YET WRITTEN' in on
        assert 'Interpretation, by K.' in on
        project.report_features(name, disable=['skeleton_sections'])
        assert len(report.blocks) == len(blocks) + 3, 'the words are kept'
        assert _render(project, name) == plain
        assert _render(project, name, edit=True, channel_js='') == plain_edit
        assert f'<style>{page._CSS}</style>' in plain
        assert 'skeleton' not in plain and 'NOT YET WRITTEN' not in plain
    finally:
        report.blocks = blocks
        report.features = []


def test_the_stylesheet_is_added_only_with_the_feature():
    css, js = page.page_parts([sk.FEATURE])
    assert js is page._JS, 'the skeleton needs no script'
    assert css.startswith(page._CSS) and 'div.owed' in css
    assert page.page_parts(())[0] is page._CSS


# ---- the page -----------------------------------------------------------------


def test_every_section_heads_the_page_in_order_and_the_end_closes_it(skeletal):
    project, name, _report = skeletal
    blocks = _payload(_render(project, name))['blocks']
    assert [heading for heading, _html in _heads(blocks)] == [
        s.heading for s in sk.SECTIONS]
    assert blocks[-1]['html'] == '<p class="endmark">End of report.</p>'
    owed = {heading.split('.')[0] for heading, html in _heads(blocks)
            if 'NOT YET WRITTEN' in html}
    assert owed == {'1', '2', '3', '4', '5', '7', '9', '10', '11', '12'}, \
        'every section only the author can write, and none that is filled'


def test_generated_blocks_fall_under_the_sections_they_answer(skeletal):
    project, name, _report = skeletal
    under = _sections(_payload(_render(project, name))['blocks'])
    summary = [block['kind'] for block in under['2']]
    assert summary[0] == 'verdict', 'the pass/fail box heads the summary'
    assert '<h3>Test Summary</h3>' in under['2'][1]['html']
    assert [block['kind'] for block in under['6']] == ['text', 'table'], \
        'the channel table under Instrumentation'
    results = [block['kind'] for block in under['8']]
    assert results.count('grid') == 2 and 'map' in results
    assert '<h3>Conclusions</h3>' in under['11'][0]['html'], \
        'the reading guide under the Conclusion, after its head'
    for key in ('1', '3', '4', '5', '7', '9', '10', '12'):
        assert under[key] == [], key


def test_figures_and_tables_are_numbered_in_page_order(skeletal):
    project, name, _report = skeletal
    labels = []
    blocks = _payload(_render(project, name, labels=labels))['blocks']
    figures = [block['label'] for block in blocks
               if block.get('label', '').startswith('Figure')]
    assert figures == [f'Figure {n}' for n in range(1, len(figures) + 1)]
    assert [label for label, _caption in labels][:1] == ['Table 1']
    under = _sections(blocks)
    assert 'Table 1' in under['6'][0]['html'], \
        'a reference follows the number the page shows'


def test_a_figure_moved_up_by_the_skeleton_takes_the_first_number():
    """A photograph stored after a plot goes under Instrumentation,
    ahead of Results, and the numbers and references follow the page."""
    from visualdynamics.core.photos import Photos

    psd = Psd(abscissa=np.linspace(10.0, 100.0, 10), ordinate=np.ones((1, 10)),
              response_dof=['1X+'], ordinate_dim=['acceleration**2/frequency'])
    photos = Photos(['setup'], ['png'], [b'\x89PNG\r\n\x1a\n' + b'x' * 20])
    report = Report('R', [
        {'kind': 'text', 'text': '## Data\n\nSee {{figure:the spectrum}} '
                                 'and {{figure:the setup}}.'},
        {'kind': 'plot', 'source': 'P', 'caption': 'the spectrum'},
        {'kind': 'text', 'text': '## Setup'},
        {'kind': 'photo', 'source': 'Album', 'photo': 'setup',
         'caption': 'the setup'}])
    objects = {'P': psd, 'Album': photos}
    plain = _payload(render_html(report, objects))['blocks']
    assert [b.get('label') for b in plain if b.get('label')] == [
        'Figure 1', 'Figure 2']
    report.features = [sk.FEATURE]
    under = _sections(_payload(render_html(report, objects))['blocks'])
    assert [(b['kind'], b.get('label')) for b in under['6'][1:]] == [
        ('image', 'Figure 1')]
    assert [(b['kind'], b.get('label')) for b in under['8'][1:]] == [
        ('plot', 'Figure 2')]
    assert 'See Figure 2 and Figure 1.' in under['8'][0]['html']


def test_the_owed_card_says_what_belongs_there():
    report = Report('R', features=[sk.FEATURE])
    blocks = _payload(render_html(report, {}))['blocks']
    article = dict(_heads(blocks))['4. Test Article']
    assert article == (
        '<h2 class="skeleton">4. Test Article</h2><div class="owed">'
        f'<strong>NOT YET WRITTEN</strong> {sk.BY_KEY["article"].description}'
        '</div>')


def test_written_words_replace_the_card_as_markdown_with_live_references():
    report = Report('R', features=[sk.FEATURE])
    sk.write(report, 'article', 'A **bracket**, {{Spec.name}}.\n\n## Mounts')
    psd = Psd(abscissa=np.linspace(10.0, 100.0, 10), ordinate=np.ones((1, 10)),
              response_dof=['1X+'], ordinate_dim=['acceleration**2/frequency'])
    psd.name = 'the profile'
    blocks = _payload(render_html(report, {'Spec': psd}))['blocks']
    article = dict(_heads(blocks))['4. Test Article']
    assert 'NOT YET WRITTEN' not in article
    assert '<strong>bracket</strong>' in article
    assert '<h3>Mounts</h3>' in article, "the author's own subheadings sit under it"


def test_the_conclusion_carries_the_item_tested_sentence_written_or_not():
    report = Report('R', features=[sk.FEATURE])
    sentence = '<p class="tested">The results relate only to the item tested.</p>'
    owed = dict(_heads(_payload(render_html(report, {}))['blocks']))
    assert owed['11. Conclusion'].endswith(sentence)
    sk.write(report, 'conclusion', 'Met.')
    written = dict(_heads(_payload(render_html(report, {}))['blocks']))
    assert written['11. Conclusion'] == (
        f'<h2 class="skeleton">11. Conclusion</h2><p>Met.</p>{sentence}')
    for heading, html in written.items():
        if heading != '11. Conclusion':
            assert 'class="tested"' not in html, heading


def test_the_discussion_says_by_whom_and_an_unattributed_one_is_owed():
    report = Report('R', features=[sk.FEATURE])
    sk.write(report, 'discussion', 'The notch is the fixture.')

    def discussion():
        return dict(_heads(_payload(render_html(report, {}))['blocks']))[
            '10. Discussion and Interpretation']

    assert ('<p class="interpretation">Interpretation, by <span '
            'class="owed">not stated</span>.</p>') in discussion()
    assert 'discussion' in [o.key for o in sk.check(report, {})]
    sk.write(report, 'discussion', by='  <b>R. Ortiz</b> ')
    assert report.blocks[0]['by'] == '<b>R. Ortiz</b>'
    assert ('Interpretation, by &lt;b&gt;R. Ortiz&lt;/b&gt;.</p>'
            '<p>The notch is the fixture.</p>') in discussion(), 'escaped'
    assert 'discussion' not in [o.key for o in sk.check(report, {})]
    sk.write(report, 'discussion', by='')
    assert 'by' not in report.blocks[0]
    sk.write(report, 'discussion', none=True)
    assert 'class="interpretation"' not in discussion(), \
        'no interpretation offered, none to attribute'
    assert 'discussion' not in [o.key for o in sk.check(report, {})]


def test_a_written_head_names_its_first_block_for_the_editor():
    """A click on a written head selects the section's block; a section
    with two blocks (one pasted in, say) shows both texts and opens on
    the first."""
    report = Report('R', [{'kind': 'text', 'text': 'x'}],
                    features=[sk.FEATURE])
    sk.write(report, 'article', 'First.')
    report.add({'kind': 'text', 'text': 'Second.', 'section': 'article'})
    blocks = _payload(render_html(report, {}, edit=True,
                                  channel_js=''))['blocks']
    heads = [block for block in blocks
             if block.get('html', '').startswith('<h2 class="skeleton">')]
    assert heads[3]['index'] == 1
    assert '<p>First.</p>\n<p>Second.</p>' in heads[3]['html']
    assert heads[0]['index'] == 'section:identification'
    assert blocks[0]['index'] == 'section:identification'
    assert [block.get('index') for block in blocks[1:3]] == [
        'section:summary', 0]
    assert 'index' not in blocks[-1], 'the end mark is not a block'


# ---- write: the one place a section's words change ------------------------


def test_write_makes_the_block_once_and_replaces_its_words():
    report = Report('R', [{'kind': 'text', 'text': 'x'}])
    assert sk.write(report, 'chronology') == 1
    assert report.blocks[1] == {'kind': 'text', 'text': '',
                                'section': 'chronology'}
    assert sk.write(report, 'chronology', 'Run 1 at full level.') == 1
    assert sk.write(report, 'chronology', 'Run 1, then run 2.') == 1
    assert report.num_blocks == 2
    assert sk.words(report, 'chronology') == 'Run 1, then run 2.'
    assert sk.write(report, 'deviations', none=True) == 2
    assert report.blocks[2]['text'] == sk.BY_KEY['deviations'].none


def test_write_refuses_before_changing_anything():
    report = Report('R')
    sk.write(report, 'article', 'kept')
    before = [dict(block) for block in report.blocks]
    with pytest.raises(ValueError, match="unknown skeleton section 'notes'"):
        sk.write(report, 'notes', 'x')
    with pytest.raises(ValueError, match='text or none, not both'):
        sk.write(report, 'deviations', 'x', none=True)
    with pytest.raises(ValueError,
                       match='4. Test Article cannot be set to None'):
        sk.write(report, 'article', none=True)
    with pytest.raises(ValueError, match='2. Summary has no interpretation'):
        sk.write(report, 'summary', by='K.')
    with pytest.raises(TypeError, match='the text is a str'):
        sk.write(report, 'summary', 3)
    with pytest.raises(TypeError, match='by is a str'):
        sk.write(report, 'discussion', by=3)
    assert report.blocks == before


# ---- check: what is still owed ----------------------------------------------


def test_the_check_refuses_a_report_without_the_skeleton(run):
    """Without the skeleton there is nothing to check against; an empty
    list would read as a complete report."""
    project, name = run
    with pytest.raises(ValueError, match='does not have Skeleton sections'):
        sk.check(Report('R'), {})
    with pytest.raises(ValueError, match='does not have Skeleton sections'):
        project.report_check(name)
    with pytest.raises(TypeError, match='not a report'):
        project.report_check('Specification')


def test_the_check_lists_what_is_owed_and_empties_as_it_is_written(skeletal):
    project, name, _report = skeletal
    assert project.report_check(name) == [
        f'{s.heading}: not yet written' for s in sk.SECTIONS if s.required]
    before = list(project.journal)
    for found in sk.SECTIONS:
        if found.none:
            project.report_section(name, found.key, none=True)
        elif found.required:
            project.report_section(name, found.key, f'{found.title} text.')
    assert project.journal[len(before)] == (
        f"project.report_section({name!r}, 'identification', "
        "'Identification text.')")
    assert project.report_check(name) == []
    blocks = _payload(_render(project, name))['blocks']
    assert not any('NOT YET WRITTEN' in block.get('html', '')
                   for block in blocks)


def test_an_empty_report_owes_every_section():
    """Nothing generated: Instrumentation and Results have nothing in
    them, and say so."""
    owed = sk.check(Report('R', features=[sk.FEATURE]), {})
    assert [o.key for o in owed] == list(sk.BY_KEY)
    assert str(owed[5]) == '6. Instrumentation: nothing in it'
    assert str(owed[7]) == '8. Results: nothing in it'
    blocks = _payload(render_html(Report('R', features=[sk.FEATURE]),
                                  {}))['blocks']
    assert len(blocks) == 13
    assert all('NOT YET WRITTEN' in block['html'] for block in blocks[:12])


def test_dead_nan_and_unbound_inputs_and_blank_words():
    """An all-zero channel and a record with a NaN still fill Results:
    the figure is drawn, and judging it is another feature's work. A
    figure whose source is missing fills nothing, on the page and in
    the check alike. Words that are only white space are owed."""
    frequencies = np.linspace(10.0, 2000.0, 200)
    nan = np.full((1, 200), 0.01)
    nan[0, 50] = np.nan
    psds = {
        'Dead': Psd(abscissa=frequencies, ordinate=np.zeros((1, 200)),
                    response_dof=['1X+'],
                    ordinate_dim=['acceleration**2/frequency']),
        'Gap': Psd(abscissa=frequencies, ordinate=nan, response_dof=['1Y+'],
                   ordinate_dim=['acceleration**2/frequency'])}
    for name in psds:
        report = Report('R', [{'kind': 'plot', 'source': name}],
                        features=[sk.FEATURE])
        assert 'results' not in [o.key for o in sk.check(report, psds)], name
        under = _sections(_payload(render_html(report, psds))['blocks'])
        assert [block['kind'] for block in under['8']] == ['plot'], name
    missing = Report('R', [{'kind': 'plot', 'source': 'Elsewhere'}],
                     features=[sk.FEATURE])
    assert 'results' in [o.key for o in sk.check(missing, psds)]
    for edit in (False, True):
        heads = dict(_heads(_payload(render_html(
            missing, psds, edit=edit, channel_js=''))['blocks']))
        assert 'NOT YET WRITTEN' in heads['8. Results'], edit
    blank = Report('R', features=[sk.FEATURE])
    sk.write(blank, 'article', '  \n\t ')
    assert 'article' in [o.key for o in sk.check(blank, {})]


def test_the_check_and_the_page_agree():
    """Every section the check lists shows owed on the page, and no
    other: one rule (`state`) behind both."""
    frequencies = np.linspace(10.0, 2000.0, 50)
    psd = Psd(abscissa=frequencies, ordinate=np.ones((1, 50)),
              response_dof=['1X+'], ordinate_dim=['acceleration**2/frequency'])
    report = Report('R', [{'kind': 'text', 'text': '## Spectra'},
                          {'kind': 'plot', 'source': 'P'}],
                    features=[sk.FEATURE])
    sk.write(report, 'summary', 'Met.')
    sk.write(report, 'discussion', 'An opinion.')
    listed = {o.number for o in sk.check(report, {'P': psd})}
    heads = _heads(_payload(render_html(report, {'P': psd}))['blocks'])
    shown = {int(heading.split('.')[0]) for heading, html in heads
             if 'NOT YET WRITTEN' in html or 'not stated' in html}
    assert listed == shown == {1, 3, 4, 5, 6, 7, 9, 10, 11, 12}


# ---- where generated blocks fall ------------------------------------------


def _text(heading=None, body='Words.'):
    return {'kind': 'text',
            'text': (f'## {heading}\n\n' if heading else '') + body}


def test_place_reads_the_report_in_its_sections():
    plot = {'kind': 'plot', 'source': 'P'}
    entries = [
        ({'kind': 'verdict'}, False, False),
        (_text('Test Summary'), False, False),
        (_text('Setup'), False, False), ({'kind': 'photo'}, True, True),
        (_text('Data'), False, False), (plot, True, False),
        (_text('Aside'), False, False),
        (_text('More data'), False, False), (plot, True, False),
        (_text('Conclusions'), False, False),
        (_text('Afterword'), False, False)]
    assert sk.place(entries) == [
        'summary', 'summary', 'instrumentation', 'instrumentation',
        'results', 'results', 'results', 'results', 'results',
        'conclusion', 'conclusion']


def test_place_edges():
    plot = {'kind': 'plot', 'source': 'P'}
    assert sk.place([]) == []
    # no headings at all: one section, which holds a figure
    assert sk.place([(_text(), False, False), (plot, True, False)]) == [
        'results', 'results']
    # text only, no figure anywhere: it opens the report
    assert sk.place([(_text('Only'), False, False)]) == ['summary']
    # an undrawn figure in a text section does not make it a figure's
    assert sk.place([(_text('A'), False, False),
                     ({'kind': 'plot'}, False, False),
                     (_text('B'), False, False), (plot, True, False)]) == [
        'summary', 'summary', 'results', 'results']
    # an instrumentation block not drawn does not move its section
    assert sk.place([(_text('A'), False, False),
                     ({'kind': 'photo'}, False, True),
                     (_text('B'), False, False), (plot, True, False)]) == [
        'summary', 'summary', 'results', 'results']
    # only a level-two heading opens a section: text under a heading
    # stays in its section, whatever comes after
    assert sk.place([(_text('Data'), False, False), (plot, True, False),
                     (_text(body='More words.'), False, False)]) == [
        'results', 'results', 'results']
    assert sk.place([(_text(body='# Big\n\nWords.'), False, False),
                     (plot, True, False),
                     ({'kind': 'plot', 'text': '## odd'}, True, False)]) == [
        'results', 'results', 'results']
    # a figure that shows the instrumentation moves its whole section,
    # text and all, wherever it sits
    assert sk.place([(_text('A'), False, False), (plot, True, False),
                     (_text('B'), False, False), (plot, True, True),
                     (plot, True, False)]) == [
        'results', 'results', 'instrumentation', 'instrumentation',
        'instrumentation']
    # the pass/fail box is not a figure: text after it still opens
    assert sk.place([({'kind': 'verdict'}, False, False),
                     (_text('S'), False, False), (_text('D'), False, False),
                     (plot, True, False), (_text('C'), False, False)]) == [
        'summary', 'summary', 'results', 'results', 'conclusion']


def test_what_counts_as_instrumentation():
    table = ChannelTable.__new__(ChannelTable)
    objects = {'Channels': table, 'Psd': object()}
    assert sk.instrumentation({'kind': 'photo'}, objects)
    assert sk.instrumentation({'kind': 'scene', 'shapes': ''}, objects)
    assert not sk.instrumentation({'kind': 'scene', 'shapes': 'Modes'},
                                  objects), 'mode shapes are results'
    assert sk.instrumentation({'kind': 'table', 'source': 'Channels'},
                              objects)
    assert not sk.instrumentation({'kind': 'table', 'source': 'Psd'},
                                  objects), 'a shape table is results'
    assert not sk.instrumentation({'kind': 'table', 'source': 'Gone'},
                                  objects)
    assert not sk.instrumentation({'kind': 'plot', 'source': 'Psd'},
                                  objects)


def test_prune_unbound_never_takes_a_sections_words():
    report = Report('R', [_text('Geometry'),
                          {'kind': 'text', 'text': 'Mine.',
                           'section': 'article'},
                          {'kind': 'scene', 'geometry': 'Gone'},
                          _text('Notes')])
    prune_unbound(report, {'Other': object()})
    assert report.blocks == [
        {'kind': 'text', 'text': 'Mine.', 'section': 'article'},
        _text('Notes')]


# ---- pipe tables --------------------------------------------------------------


def test_a_pipe_table_with_the_feature():
    text = ('Before\n'
            '| Run | Level | Note |\n'
            '|:----|----:|:---:|\n'
            '| 1 | -6 dB | **ok** |\n'
            '| 2 | 0 dB | a \\| b, <i> |\n'
            '| 3 |\n'
            '| 4 | 1 | 2 | 3 |\n'
            'After')
    assert to_html(text, tables=True) == (
        '<p>Before</p>\n<div class="tablewrap"><table><thead><tr>'
        '<th style="text-align: left">Run</th>'
        '<th style="text-align: right">Level</th>'
        '<th style="text-align: center">Note</th></tr></thead><tbody>'
        '<tr><td style="text-align: left">1</td>'
        '<td style="text-align: right">-6 dB</td>'
        '<td style="text-align: center"><strong>ok</strong></td></tr>'
        '<tr><td style="text-align: left">2</td>'
        '<td style="text-align: right">0 dB</td>'
        '<td style="text-align: center">a | b, &lt;i&gt;</td></tr>'
        '<tr><td style="text-align: left">3</td>'
        '<td style="text-align: right"></td>'
        '<td style="text-align: center"></td></tr>'
        '<tr><td style="text-align: left">4</td>'
        '<td style="text-align: right">1</td>'
        '<td style="text-align: center">2</td></tr>'
        '</tbody></table></div>\n<p>After</p>')


def test_pipe_table_edges():
    plain = '| a | b |\n|---|---|\n| 1 | 2 |'
    assert to_html(plain) == ('<p>| a | b |<br>|---|---|<br>| 1 | 2 |</p>'), \
        'without the feature a pipe is text, as it always was'
    assert to_html('| a | b |\n| 1 | 2 |', tables=True) == \
        '<p>| a | b |<br>| 1 | 2 |</p>', 'no delimiter row, no table'
    assert to_html('| a |\n| -- |', tables=True) == (
        '<div class="tablewrap"><table><thead><tr><th>a</th></tr></thead>'
        '<tbody></tbody></table></div>'), 'a header alone'
    assert to_html('| a |\n|  |', tables=True) == '<p>| a |<br>|  |</p>'
    assert '<td>c\\d</td>' in to_html('| x |\n|---|\n| c\\d |', tables=True)
    assert '<td>e\\</td>' in to_html('| x |\n|---|\n| e\\ |', tables=True)
    # a list stops at the table, and the table at a blank line
    out = to_html('- item\n| x |\n|---|\n| 1 |\n\nAfter', tables=True)
    assert out.startswith('<ul>\n<li>item</li>\n</ul>\n<div class="tablewrap">')
    assert out.endswith('</table></div>\n<p>After</p>')


def test_a_section_reads_its_tables_and_other_text_does_not():
    table = '| Run | Level |\n|---|---|\n| 1 | full |'
    report = Report('R', [{'kind': 'text', 'text': table}],
                    features=[sk.FEATURE])
    sk.write(report, 'chronology', table)
    blocks = _payload(render_html(report, {}))['blocks']
    chronology = dict(_heads(blocks))['7. Chronology']
    assert '<table><thead><tr><th>Run</th>' in chronology
    report.features = []
    off = _payload(render_html(report, {}))['blocks']
    assert off == [{'kind': 'text', 'html': to_html(table)}]


# ---- persistence --------------------------------------------------------------


def test_the_sections_words_ride_in_the_project_file_and_the_template(tmp_path):
    from visualdynamics import io
    from visualdynamics.io import report_template

    report = Report('R', features=[sk.FEATURE])
    sk.write(report, 'discussion', 'An opinion.', by='K.')
    io.save(report, str(tmp_path / 'r.vdyn'))
    assert io.load(str(tmp_path / 'r.vdyn')).blocks == report.blocks
    report_template.save(report, tmp_path / 'house.vdreport')
    back = report_template.load(tmp_path / 'house.vdreport')
    assert back.blocks == report.blocks and back.features == [sk.FEATURE]


# ---- the editor and Export ---------------------------------------------------


@pytest.fixture
def editing(window, pump, project):
    for name, obj in project.items():
        window.add_object(name, obj)
    name = window.generate_report('modal')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    report = window.objects[name]
    window.report_editor._operate({'op': 'features',
                                   'enable': ['skeleton_sections']})
    return window, window.report_editor, report, name


def _page(editor):
    with open(editor._page_path, encoding='utf-8') as page_file:
        return _payload(page_file.read())


def test_an_owed_head_on_the_edit_page_opens_its_section(editing):
    window, editor, report, name = editing
    data = _page(editor)
    heads = [block for block in data['blocks']
             if block.get('html', '').startswith('<h2 class="skeleton">')]
    assert [block['index'] for block in heads] == [
        f'section:{key}' for key in sk.BY_KEY]
    before = report.num_blocks
    editor._operate({'op': 'select', 'at': 'section:deviations'})
    assert report.num_blocks == before + 1
    assert editor.selected == before
    assert report.blocks[before] == {'kind': 'text', 'text': '',
                                     'section': 'deviations'}
    assert editor.text_editor is not None and editor.none_button is not None
    assert editor.by_edit is None
    heads = [block for block in _page(editor)['blocks']
             if block.get('html', '').startswith('<h2 class="skeleton">9.')]
    assert heads[0]['index'] == before, 'the head now names its block'
    editor.none_button.click()
    assert report.blocks[before]['text'] == sk.BY_KEY['deviations'].none
    assert editor.text_editor.toPlainText() == sk.BY_KEY['deviations'].none
    journaled = [line for line in window.project.journal
                 if line.startswith(f'project[{name!r}].blocks = ')]
    assert len(journaled) == 1, 'journaled as the blocks that stand'
    assert sk.BY_KEY['deviations'].none in journaled[0]
    editor._operate({'op': 'select', 'at': 'section:nonsense'})
    assert report.num_blocks == before + 1, 'an unknown section makes nothing'


def test_the_pane_of_a_section_block(editing):
    _window, editor, report, _name = editing
    editor._operate({'op': 'select', 'at': 'section:article'})
    assert editor.none_button is None, 'every report describes its article'
    editor.text_editor.setPlainText('A bracket.')
    editor.flush_text()
    assert sk.words(report, 'article') == 'A bracket.'
    editor._operate({'op': 'select', 'at': 'section:discussion'})
    assert editor.by_edit is not None and editor.none_button is not None
    editor.by_edit.setText('R. Ortiz')
    editor.by_edit.editingFinished.emit()
    index = sk.blocks_of(report, 'discussion')[0]
    assert report.blocks[index]['by'] == 'R. Ortiz'
    # an op naming a block that is not a section changes nothing
    editor._operate({'op': 'section', 'at': 0, 'none': True})
    assert 'section' not in report.blocks[0]


def test_export_lists_what_is_owed_and_asks(editing, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    _window, editor, _report, name = editing
    asked, saved = [], []
    answer = [QMessageBox.StandardButton.No]
    monkeypatch.setattr(QMessageBox, 'question', staticmethod(
        lambda _parent, _title, text, *_rest: asked.append(text)
        or answer[0]))
    target = tmp_path / 'out.html'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', staticmethod(
        lambda *_args: saved.append(1) or (str(target), '')))
    editor._operate({'op': 'export'})
    assert len(asked) == 1 and saved == [], 'No stops before the file dialog'
    assert asked[0].startswith(f'{name} still owes 10 sections:\n\n'
                               '1. Identification: not yet written\n')
    assert asked[0].endswith('12. Record and Data Package: not yet '
                             'written\n\nExport anyway?')
    answer[0] = QMessageBox.StandardButton.Yes
    editor._operate({'op': 'export'})
    assert saved == [1] and 'NOT YET WRITTEN' in target.read_text(
        encoding='utf-8')
    # with the feature off nothing is asked
    editor._operate({'op': 'features', 'disable': ['skeleton_sections']})
    asked.clear()
    editor._operate({'op': 'export'})
    assert asked == [] and saved == [1, 1]
