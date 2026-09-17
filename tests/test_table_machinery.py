"""The table conventions, exercised on the machinery itself.

Copy, paste, batch edit and clearing are what make declaring forty-five
channels one gesture; they have to behave identically in every table, so
they are tested here once against a plain model rather than through any
particular pane.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication

from visualdynamics.gui.tables import Column, CopyPasteTableView, TableModel

EDIT = Qt.ItemDataRole.EditRole


class Sheet:
    """A plain grid of text with one refusing column."""

    def __init__(self, rows=4, columns=3):
        self.cells = [[f'r{r}c{c}' for c in range(columns)]
                      for r in range(rows)]


def sheet_model(sheet, parent=None):
    def setter(column):
        def set_cell(sheet, row, text):
            if text == 'refuse':
                raise ValueError('this cell refuses that')
            sheet.cells[row][column] = text
        return set_cell

    columns = [Column(f'C{c}', lambda s, r, c=c: s.cells[r][c],
                      set=setter(c)) for c in range(3)]
    return TableModel(sheet, columns, lambda s: len(s.cells), parent)


@pytest.fixture
def table(qt_app):
    sheet = Sheet()
    view = CopyPasteTableView()
    model = sheet_model(sheet)
    view.setModel(model)
    view.resize(500, 300)
    view.show()
    qt_app.processEvents()
    yield view, model, sheet
    view.close()
    view.deleteLater()
    qt_app.processEvents()


def select_block(view, model, top, left, bottom, right):
    view.clearSelection()
    selection = view.selectionModel()
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            selection.select(model.index(r, c),
                             QItemSelectionModel.SelectionFlag.Select)


def test_copy_is_tsv_in_row_major_order(table):
    view, model, _sheet = table
    select_block(view, model, 0, 0, 1, 1)
    view.copy()
    assert QApplication.clipboard().text() == 'r0c0\tr0c1\nr1c0\tr1c1'


def test_paste_writes_a_block_from_the_anchor(table):
    view, model, sheet = table
    QApplication.clipboard().setText('a\tb\nc\td')
    view.setCurrentIndex(model.index(1, 1))
    view.clearSelection()
    view.selectionModel().select(
        model.index(1, 1), QItemSelectionModel.SelectionFlag.Select)
    view.paste()
    assert sheet.cells[1][1:] == ['a', 'b']
    assert sheet.cells[2][1:] == ['c', 'd']
    assert sheet.cells[0][0] == 'r0c0', 'above the anchor is untouched'


def test_a_single_value_pastes_at_the_anchor_alone(table):
    """Paste anchors at the selection's top-left and writes the block it
    was given — one value, one cell. Filling a selection with one value
    is batch edit's job, not an Excel-style paste surprise."""
    view, model, sheet = table
    QApplication.clipboard().setText('same')
    select_block(view, model, 0, 0, 2, 0)
    view.paste()
    assert sheet.cells[0][0] == 'same'
    assert [sheet.cells[r][0] for r in (1, 2)] == ['r1c0', 'r2c0']


def test_batch_edit_reports_what_refused(table):
    view, model, sheet = table
    select_block(view, model, 0, 0, 1, 2)
    applied, rejected = view.batch_edit('everywhere')
    assert (applied, rejected) == (6, 0)
    assert sheet.cells[0] == ['everywhere'] * 3
    applied, rejected = view.batch_edit('refuse')
    assert applied == 0 and rejected == 6, 'refusals are counted, not lost'
    assert sheet.cells[0] == ['everywhere'] * 3, 'and nothing changed'


def test_rejected_edits_say_why(table):
    _view, model, _sheet = table
    reasons = []
    model.edit_rejected.connect(reasons.append)
    assert not model.setData(model.index(0, 0), 'refuse', EDIT)
    assert reasons and 'refuses' in reasons[0]


def choice_model(parent=None):
    """A table whose two columns share one list of choices, and a third
    that has none — the shape `shared_choices` has to tell apart."""
    sheet = Sheet(rows=3, columns=3)
    shades = ['red', 'green', 'blue']

    def setter(column):
        def set_cell(sheet, row, text):
            if text not in shades:
                raise ValueError(f'{text!r} is not one of {shades}')
            sheet.cells[row][column] = text
        return set_cell

    def plain(sheet, row, text):
        sheet.cells[row][2] = text

    columns = [
        Column('Left', lambda s, r: s.cells[r][0], set=setter(0),
               choices=shades),
        Column('Right', lambda s, r: s.cells[r][1], set=setter(1),
               choices=shades),
        Column('Free', lambda s, r: s.cells[r][2], set=plain),
    ]
    return sheet, TableModel(sheet, columns, lambda s: len(s.cells), parent)


def test_a_choice_column_offers_its_list_to_the_batch_dialog(qt_app):
    """Batch editing a column of colors offers the same list as editing
    one cell of it, rather than a box to type a color name into."""
    from visualdynamics.gui.tables import ChoiceDialog

    sheet, model = choice_model()
    view = CopyPasteTableView()
    view.setModel(model)
    select_block(view, model, 0, 0, 2, 1)
    column = view.shared_choices(view.editable_selection())
    assert column is not None and column.choices == ['red', 'green', 'blue']

    dialog = ChoiceDialog(view, 'Set 6 selected cells to:', column)
    assert dialog.windowTitle() == 'Batch Edit'
    assert [dialog.combo.itemText(i) for i in range(dialog.combo.count())] \
        == ['red', 'green', 'blue']
    dialog.combo.setCurrentIndex(2)
    assert dialog.value() == 'blue'
    applied, rejected = view.batch_edit(dialog.value())
    assert (applied, rejected) == (6, 0)
    assert sheet.cells[0][:2] == ['blue', 'blue']
    dialog.deleteLater()
    view.deleteLater()
    qt_app.processEvents()


def test_a_selection_across_unlike_columns_has_no_shared_list(qt_app):
    _sheet, model = choice_model()
    view = CopyPasteTableView()
    view.setModel(model)
    select_block(view, model, 0, 0, 0, 2)     # two choice columns and a free one
    assert view.shared_choices(view.editable_selection()) is None, (
        'no common list, so batch edit falls back to typing')
    view.deleteLater()
    qt_app.processEvents()


def test_the_choice_delegate_puts_the_list_in_the_cell(qt_app):
    """Selecting a cell in a choice column shows its arrow; opening it
    opens the list already showing, rather than a closed box."""
    from PySide6.QtWidgets import QStyleOptionViewItem

    from visualdynamics.gui.tables import ChoiceDelegate

    _sheet, model = choice_model()
    view = CopyPasteTableView()
    view.setModel(model)
    delegate = ChoiceDelegate()
    option = QStyleOptionViewItem()
    editor = delegate.createEditor(view, option, model.index(0, 0))
    assert [editor.itemText(i) for i in range(editor.count())] \
        == ['red', 'green', 'blue']
    plain = delegate.createEditor(view, option, model.index(0, 2))
    assert not hasattr(plain, 'itemText'), 'a free column gets a plain editor'
    editor.deleteLater()
    plain.deleteLater()
    view.deleteLater()
    qt_app.processEvents()


def test_set_cells_applies_what_it_can(table):
    _view, model, sheet = table
    applied, rejected, _reason = model.set_cells(
        [(0, 0, 'good'), (1, 1, 'refuse'), (2, 2, 'also good')])
    assert (applied, rejected) == (2, 1)
    assert sheet.cells[0][0] == 'good'
    assert sheet.cells[2][2] == 'also good'
    assert sheet.cells[1][1] == 'r1c1'
