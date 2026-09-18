"""The report editor's acts on the bar, its settings in the pane.

Brandon, 2026-09-08: the report's toolbar should be the same bar every
other object has. The page stays the preview — the very HTML the export
writes — and does two things: frame the selected block and say which
block was clicked. Insert, Reference, Move Up, Move Down, Delete and
Export are on the bar; the selected block's sources, caption or Markdown
are in the pane beside; every edit is one operation on the Report model.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton


@pytest.fixture
def editing(window, pump, project):
    for name, obj in project.items():
        window.add_object(name, obj)
    name = window.generate_report('modal')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    return window, window.report_editor, window.objects[name]


def _labels(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def test_the_bar_carries_the_reports_acts(editing):
    window, editor, _report = editing
    assert editor.toolbar.isVisibleTo(window), 'a bar above the page'
    buttons = [editor.toolbar.widgetForAction(a) for a in editor.toolbar.actions()
               if not a.isSeparator()]
    assert [b.text() for b in buttons] == [
        'Insert', 'Reference', 'Move Up', 'Move Down', 'Delete', 'Export']
    assert all(isinstance(b, QToolButton) and b.toolButtonStyle()
               == Qt.ToolButtonStyle.ToolButtonIconOnly for b in buttons), \
        'an icon each; the words are in the tooltip (Brandon, 2026-09-12)'
    assert all(b.toolTip().startswith(b.text() + ' — ') for b in buttons), \
        'the tooltip leads with the verb the icon stands for'
    # nothing selected: the block acts wait, the report's are live
    assert not editor.up_action.isEnabled() and not editor.delete_action.isEnabled()
    assert editor.insert_button.isEnabled() and editor.export_button.isEnabled()
    assert editor.title_edit is not None, "the pane shows the report's own settings"
    assert _labels(editor.export_menu) == ['HTML…', 'Report template…']


def test_insert_offers_markdown_and_every_result_and_inserts_after_the_selection(
        editing):
    from visualdynamics.core.report import insert_options

    window, editor, report = editing
    labels = _labels(editor.insert_menu)
    assert labels[0] == 'Markdown'
    assert labels[1:] == [label for _k, label in insert_options(window.objects)]
    before = report.num_blocks
    editor._operate({'op': 'select', 'at': 1})
    next(a for a in editor.insert_menu.actions() if a.text() == 'CMIF').trigger()
    assert report.num_blocks == before + 1
    assert report.blocks[2]['mode'] == 'cmif', 'after the selected block'
    assert editor.selected == 2, 'and the new block is what the bar acts on'
    assert 'source' in editor.field_boxes and editor.caption_edit is not None
    editor._operate({'op': 'select', 'at': None})
    next(a for a in editor.insert_menu.actions() if a.text() == 'Markdown').trigger()
    assert report.blocks[-1] == {'kind': 'text', 'text': ''}, 'nothing selected: at the end'
    assert editor.text_editor is not None


def test_a_clicked_block_selects_and_the_pane_follows(editing):
    _window, editor, report = editing
    text_at = next(i for i, b in enumerate(report.blocks) if b['kind'] == 'text')
    figure_at = next(i for i, b in enumerate(report.blocks) if b['kind'] == 'plot')
    # the page's one message: which block was clicked
    editor.bridge.apply(f'{{"op": "select", "at": {figure_at}, "scroll": 120}}')
    assert editor.selected == figure_at
    assert editor.text_editor is None and 'source' in editor.field_boxes
    assert editor.caption_edit.text() == report.blocks[figure_at].get('caption', '')
    assert editor.up_action.isEnabled() and editor.delete_action.isEnabled()
    editor._operate({'op': 'select', 'at': text_at})
    assert editor.text_editor is not None
    assert editor.text_editor.toPlainText() == report.blocks[text_at]['text']
    assert editor.field_boxes == {} and editor.caption_edit is None
    editor._operate({'op': 'select', 'at': None})
    assert editor.title_edit is not None and editor.text_editor is None


def test_markdown_typed_in_the_pane_lands_after_the_debounce(editing, pump):
    from PySide6.QtTest import QTest

    from visualdynamics.gui.report_editor import TEXT_DEBOUNCE_MS

    _window, editor, report = editing
    text_at = next(i for i, b in enumerate(report.blocks) if b['kind'] == 'text')
    editor._operate({'op': 'select', 'at': text_at})
    editor.text_editor.setPlainText('# Retitled\n\nA new paragraph.')
    assert report.blocks[text_at]['text'] != '# Retitled\n\nA new paragraph.', \
        'not per keystroke'
    editor.flush_text()
    assert report.blocks[text_at]['text'] == '# Retitled\n\nA new paragraph.'
    assert editor.selected == text_at and editor.text_editor is not None, \
        'the pane and its cursor stay where they were'
    with open(editor._page_path, encoding='utf-8') as page:
        assert 'Retitled' in page.read(), 'the page re-rendered'
    editor.text_editor.setPlainText('# Again')
    QTest.qWait(TEXT_DEBOUNCE_MS + 300)
    pump()
    assert report.blocks[text_at]['text'] == '# Again', 'and the timer lands it too'


def test_reference_inserts_a_token_at_the_cursor(editing):
    _window, editor, report = editing
    text_at = next(i for i, b in enumerate(report.blocks) if b['kind'] == 'text')
    editor._operate({'op': 'select', 'at': text_at})
    assert editor.reference_button.isEnabled(), 'a text block is being edited'
    labels = _labels(editor.reference_menu)
    assert labels and labels[0].startswith('Figure 1')
    captioned = next(a for a in editor.reference_menu.actions()
                     if a.isEnabled() and not a.isSeparator())
    label, caption = captioned.text().split(' — ', 1)
    editor.text_editor.setPlainText('See ')
    editor.text_editor.moveCursor(editor.text_editor.textCursor().MoveOperation.End)
    captioned.trigger()
    kind = 'table' if label.startswith('Table') else 'figure'
    assert editor.text_editor.toPlainText() == f'See {{{{{kind}:{caption}}}}}'
    editor.flush_text()          # rebuilds the page — and the menu's actions
    with open(editor._page_path, encoding='utf-8') as page:
        assert f'See {label}' in page.read(), \
            'resolved to the number the page shows'
    # a figure without a caption cannot be referred to
    editor._operate({'op': 'insert', 'at': 0, 'kind': 'cmif'})
    editor._operate({'op': 'select', 'at': text_at + 1})
    blank = [a for a in editor.reference_menu.actions() if 'no caption' in a.text()]
    assert blank and not any(a.isEnabled() for a in blank)
    editor._operate({'op': 'select', 'at': 0})
    assert not editor.reference_button.isEnabled(), 'a figure block: nowhere to type'


def test_move_and_delete_act_on_the_selection(editing):
    _window, editor, report = editing
    second = dict(report.blocks[1])
    editor._operate({'op': 'select', 'at': 1})
    editor.up_action.trigger()
    assert report.blocks[0] == second and editor.selected == 0
    assert not editor.up_action.isEnabled(), 'at the top'
    editor.down_action.trigger()
    assert report.blocks[1] == second and editor.selected == 1
    before = report.num_blocks
    editor.delete_action.trigger()
    assert report.num_blocks == before - 1 and second not in report.blocks
    assert editor.selected is None and editor.title_edit is not None


def test_caption_source_title_and_marking_edit_from_the_pane(editing):
    _window, editor, report = editing
    figure_at = next(i for i, b in enumerate(report.blocks) if b['kind'] == 'plot')
    editor._operate({'op': 'select', 'at': figure_at})
    editor.caption_edit.setText('A better caption')
    editor.caption_edit.editingFinished.emit()
    assert report.blocks[figure_at]['caption'] == 'A better caption'
    box = editor.field_boxes['source']
    other = next(i for i in range(box.count())
                 if box.itemData(i) and box.itemData(i) != box.currentData())
    box.setCurrentIndex(other)
    assert report.blocks[figure_at]['source'] == box.itemData(other)
    editor._operate({'op': 'select', 'at': None})
    editor.title_edit.setText('Renamed')
    editor.title_edit.editingFinished.emit()
    assert report.title == 'Renamed'
    editor.marking_edit.setText('  SECRET  ')
    editor.marking_edit.editingFinished.emit()
    assert report.marking == 'SECRET'
    editor.color_box.setCurrentIndex(1)
    assert report.marking_color == 'red'


def test_a_build_that_fails_says_so_on_the_page(window, pump, monkeypatch):
    """A render that raised left the view white and the pane reading
    "No report" (Brandon, 2026-09-18). The page carries the traceback,
    the pane and the status line name the error, and the object stays."""
    from visualdynamics.gui import report_editor as editor_module

    def broken(*_args, **_kwargs):
        raise RuntimeError('the figure that would not draw')

    monkeypatch.setattr(editor_module, 'render_html', broken, raising=False)
    import visualdynamics.report as report_module
    monkeypatch.setattr(report_module, 'render_html', broken)
    window.project.generate_report('empty')
    window.show_object('Report')
    pump()
    editor = window.report_editor
    assert editor.report is not None, 'the report is still the object'
    assert 'the figure that would not draw' in (editor.failure or '')
    assert 'could not be built' in window.statusBar().currentMessage()
    assert 'the figure that would not draw' in window.statusBar().currentMessage()
    with open(editor._page_path, encoding='utf-8') as page:
        assert 'This report could not be built' in page.read()
    from PySide6.QtWidgets import QLabel
    assert any('could not be built' in label.text()
               for label in editor.findChildren(QLabel))
