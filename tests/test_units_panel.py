"""The imported-units pane, and the table conventions it shares.

Every table in visualdynamics behaves the same way — see "How tables behave" in
PLAN.md — so most of what is checked here is checked through this pane but
is really a statement about all of them.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import fixture_path
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox

import visualdynamics

EDIT = Qt.ItemDataRole.EditRole


@pytest.fixture
def opened(window, pump):
    """Open the pane on an object, and hand back its model."""
    def open_on(obj, name='thing', records=None):
        window.add_object(name, obj)
        item = window.test_item.child(window.test_item.childCount() - 1)
        window.tree.clearSelection()
        if records is None:
            item.setSelected(True)
            window.tree.setCurrentItem(item)
        else:
            # selecting records inside the object, the way a user does now:
            # every expansion is a grid, and its cells are the records
            item.setExpanded(True)
            item.setSelected(True)
            window.record_grids[name].select_records(records)
        pump()
        window.define_units()
        pump()
        return window.units_table.model(), item
    return open_on


@pytest.fixture
def time_data():
    return visualdynamics.import_file(fixture_path('plate', 'time.npz'))


@pytest.fixture
def hinted_frf(tmp_path):
    """An FRF whose file names the quantity but has no 164 to size it."""
    source = fixture_path('plate', 'frfs.unv')
    with open(source) as f:
        text = f.read()
    path = tmp_path / 'no_164.unv'
    path.write_text(re.sub(r'    -1\n   164\n.*?    -1\n', '', text,
                           flags=re.DOTALL))
    return visualdynamics.import_file(str(path))


def choose(table, model, row, column, text, pump):
    """Pick from a cell's drop-down the way clicking one does.

    Returns whether the drop-down was actually showing, which is the part
    that broke once: committing rebuilt the plot under the open popup.
    """
    before = set(table.findChildren(QComboBox))
    index = model.index(row, column)
    table.setCurrentIndex(index)
    table.edit(index)
    pump(5)
    fresh = [c for c in table.findChildren(QComboBox) if c not in before]
    assert fresh, f'no editor opened on ({row}, {column})'
    editor = fresh[-1]
    showing = editor.view().isVisible()
    editor.setCurrentText(text)
    editor.activated.emit(max(0, editor.findText(text)))
    pump()
    return showing


def test_opening_the_list_is_not_a_choice(window, opened, time_data,
                                         pump, monkeypatch):
    """The drop-down that closed the instant it opened.

    On macOS `QComboBox.showPopup()` emits `activated` from inside
    itself, naming whatever entry the list opens on — no click, no
    key, nothing the person did (caught by stack trace 2026-09-01,
    with not one mouse event in the process). The delegate took that
    for a pick, committed and closed, and the list vanished in the
    same breath it appeared; the cell kept the value it already had,
    so nothing on screen said what had happened.

    Here showPopup is made to behave that way on every platform, so
    the defense is testable off a Mac too.
    """
    model, _item = opened(time_data)
    table = window.units_table
    column = next(i for i, c in enumerate(model.columns)
                  if c.choices_for(model.obj, 0))
    index = model.index(0, column)
    before = model.data(index)

    real_show = QComboBox.showPopup

    def macos_show(self):
        real_show(self)
        self.activated.emit(self.currentIndex())   # the platform's doing

    monkeypatch.setattr(QComboBox, 'showPopup', macos_show)
    table.setCurrentIndex(index)
    table.edit(index)
    for _ in range(6):
        QTest.qWait(20)
    editors = table.findChildren(QComboBox)
    assert editors, 'the editor was torn down by the list opening'
    assert editors[-1].view().isVisible(), 'the list closed as it opened'
    assert model.data(index) == before, (
        'opening the list must not change the cell')

    # and a real pick still commits and closes, as it always did
    editor = editors[-1]
    choice = next(text for text in
                  (editor.itemText(i) for i in range(editor.count()))
                  if text != before)
    editor.setCurrentText(choice)
    editor.activated.emit(editor.findText(choice))
    pump(5)
    assert model.data(index) == choice, 'a chosen unit still lands'


def select_cell(table, model, row, column):
    table.clearSelection()
    table.selectionModel().select(
        model.index(row, column),
        QItemSelectionModel.SelectionFlag.ClearAndSelect)


# ---- the pane itself --------------------------------------------------------

def test_the_pane_opens_beside_the_plot_rather_than_over_it(
        window, opened, time_data):
    """The curve is the evidence for what its unit is — in whichever
    reading is up. The waterfall is the default now, and it shows the
    same channels with the same labels, so it serves as the evidence
    exactly as the flat plot does; what matters is that *a* plot
    surface stays beside the pane rather than a dialog covering it."""
    opened(time_data)
    assert window.units_panel.isVisible()
    pane = window.data_pane
    surface_up = (pane.graphics.isVisible()
                  or (pane._waterfall_page is not None
                      and pane._waterfall_page.isVisible()))
    assert surface_up, 'the plot has to stay up'


def test_the_title_says_these_are_the_units_the_data_came_in(
        window, opened, time_data):
    """Plain 'Units' would read as the units things are shown and written
    in, which is the display selector in the status bar instead."""
    opened(time_data, name='shaker run')
    assert window.units_title.text() == 'Imported Units — shaker run'


def test_the_pane_closes_when_the_selection_moves_on(
        window, opened, pump, time_data):
    other = visualdynamics.import_file(fixture_path('plate', 'geometry.unv'))
    _model, _item = opened(time_data, name='data')
    window.add_object('geometry', other)
    item = window.test_item.child(window.test_item.childCount() - 1)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    assert not window.units_panel.isVisible()
    assert window.units_target is None


def test_the_pane_covers_only_the_selected_channels(opened, time_data):
    model, _item = opened(time_data, records=[1, 3])
    assert model.rowCount() == 2
    assert model.data(model.index(0, 0)) == time_data.record_label(1)


# ---- what a channel is, and what it is measured in --------------------------

def test_a_quantity_the_file_named_narrows_the_units_offered(
        window, opened, hinted_frf):
    # an FRF's channels come in two kinds, so it gets two tables — response
    # channels beside reference channels — each narrowed by its own type
    model, _item = opened(hinted_frf)
    references = window.units_reference_table.model()
    assert [c.title for c in model.columns] == ['Channel', 'Type', 'Unit']
    assert model.data(model.index(0, 1)) == 'acceleration'
    assert references.data(references.index(0, 1)) == 'force'
    assert model.columns[2].choices_for(hinted_frf, 0) == [
        'g', 'in/s**2', 'm/s**2', 'mm/s**2']
    assert references.columns[2].choices_for(hinted_frf, 0) == [
        'lbf', 'N', 'kN']


def test_a_channel_nothing_is_known_about_is_offered_everything(
        opened, time_data):
    from visualdynamics.core.unit_choices import ALL_ORDINATE_UNITS

    model, _item = opened(time_data)
    assert model.data(model.index(0, 1)) == '', 'nothing said what it is'
    assert model.columns[2].choices_for(time_data, 0) == ALL_ORDINATE_UNITS


def test_the_shortlist_is_a_shortlist_and_not_a_limit(
        window, opened, hinted_frf):
    """Anything visualdynamics can parse is still accepted, typed or pasted."""
    model, _item = opened(hinted_frf)
    references = window.units_reference_table.model()
    assert model.columns[2].choices_editable
    assert 'ft/s**2' not in model.columns[2].choices_for(hinted_frf, 0)
    assert model.setData(model.index(0, 2), 'ft/s**2', EDIT)
    assert references.setData(references.index(0, 2), 'N', EDIT)
    assert hinted_frf.ordinate_unit[0] == 'ft/s**2'


def test_nonsense_is_refused_and_changes_nothing(opened, time_data):
    model, _item = opened(time_data)
    assert not model.setData(model.index(0, 2), 'zorkmids', EDIT)
    assert time_data.ordinate_dim[0] == 'unknown'


def test_an_frf_waits_for_both_halves_rather_than_refusing_one(
        window, opened, hinted_frf):
    """Neither unit converts without the other, so a half-named record is
    held rather than rejected — otherwise there is no order that works."""
    model, _item = opened(hinted_frf)
    references = window.units_reference_table.model()
    assert model.setData(model.index(0, 2), 'g', EDIT)
    assert hinted_frf.ordinate_dim[0] == 'unknown', 'nothing to convert by'
    assert model.data(model.index(0, 2)) == 'g', 'but the cell holds it'
    for row in range(references.rowCount()):
        assert references.setData(references.index(row, 2), 'lbf', EDIT)
    assert hinted_frf.ordinate_dim[0] == 'acceleration/force'


def test_an_frf_declares_channels_not_records(window, opened, hinted_frf):
    """A 45-record FRF against one reference is 45 response channels and
    one reference channel — 46 declarations, not 90."""
    model, _item = opened(hinted_frf)
    references = window.units_reference_table.model()
    assert model.rowCount() == len(set(hinted_frf.response_dof))
    assert references.rowCount() == len(set(hinted_frf.reference_dof))
    for row in range(references.rowCount()):
        assert references.setData(references.index(row, 2), 'lbf', EDIT)
    for row in range(model.rowCount()):
        assert model.setData(model.index(row, 2), 'g', EDIT)
    assert all(dim == 'acceleration/force' for dim in hinted_frf.ordinate_dim)
    assert all(unit == 'lbf' for unit in hinted_frf.reference_unit)


def test_the_sides_stay_separate_at_a_drive_point(window, opened):
    """A drive point's accelerometer and its force gauge share a DOF label
    and are two different channels; merging them would force one unit on
    both."""
    from visualdynamics.core.data import Frf

    freq = np.linspace(0.0, 100.0, 11)
    frf = Frf(freq, np.ones((1, 11), dtype=complex),
              response_dof=['101X+'], reference_dof=['101X+'])
    model, _item = opened(frf)
    references = window.units_reference_table.model()
    assert model.setData(model.index(0, 2), 'g', EDIT)
    for row in range(references.rowCount()):
        assert references.setData(references.index(row, 2), 'lbf', EDIT)
    assert frf.ordinate_dim[0] == 'acceleration/force'


def test_declaring_a_unit_converts_once_to_si(opened, time_data):
    raw = time_data.ordinate[0].copy()
    model, _item = opened(time_data)
    model.setData(model.index(0, 2), 'g', EDIT)
    assert np.allclose(time_data.ordinate[0], raw * 9.80665)
    model.setData(model.index(0, 2), 'm/s**2', EDIT)
    assert np.allclose(time_data.ordinate[0], raw), 'reinterpreted, not rescaled'


def test_the_type_can_be_changed_and_drives_the_units_offered(
        opened, time_data):
    model, _item = opened(time_data)
    assert model.setData(model.index(0, 1), 'velocity', EDIT)
    assert model.columns[2].choices_for(time_data, 0) == ['in/s', 'm/s', 'mm/s']


def test_changing_the_type_withdraws_a_unit_of_the_wrong_kind(
        opened, time_data):
    """Leaving 'g' beside a type of 'force' would be two answers to one
    question, and the values would be scaled by the wrong one."""
    model, _item = opened(time_data)
    model.setData(model.index(0, 2), 'g', EDIT)
    raw = time_data._raw_record(0).copy()
    model.setData(model.index(0, 1), 'force', EDIT)
    assert time_data.ordinate_unit[0] is None
    assert time_data.ordinate_dim[0] == 'unknown'
    assert np.allclose(time_data.ordinate[0], raw), 'values are raw again'


# ---- reaching the pane from a grid ------------------------------------------

def test_a_right_clicked_grid_cell_offers_the_units_pane(
        window, pump, time_data):
    """A grid is a widget, so the tree's own context menu never fires
    inside one — records need their own route to Define Imported Units."""
    window.add_object('data', time_data)
    item = window.test_item.child(window.test_item.childCount() - 1)
    item.setExpanded(True)
    pump()
    grid = window.record_grids['data']
    cell = grid.item(1, 0)
    menu = window._grid_menu(grid, grid.visualItemRect(cell).center())
    assert menu is not None
    labels = [a.text() for a in menu.actions() if a.text()]
    assert any('Imported Units' in text for text in labels)
    assert cell.isSelected(), 'right-click picks the cell under it'
    window.define_units()
    pump()
    model = window.units_table.model()
    assert model.rowCount() == 1, 'the pane covers the clicked record only'
    assert model.data(model.index(0, 0)) == time_data.record_label(1)


def test_the_dimensionless_unit_does_not_crash_the_plot(
        window, opened, pump, time_data):
    """pint spells 'no dimension' as the empty expression; asking it about
    the word 'dimensionless' raises KeyError('') from inside rendering."""
    model, _item = opened(time_data)
    assert model.setData(model.index(0, 2), 'dimensionless', EDIT)
    pump()
    assert time_data.ordinate_dim[0] == 'dimensionless'


# ---- one unit for the whole object ------------------------------------------

def test_a_geometry_declares_its_one_unit_in_the_same_table(window, opened):
    geometry = visualdynamics.import_file(fixture_path('plate', 'geometry.unv'))
    model, _item = opened(geometry)
    assert window.units_panel.isVisible(), 'no separate window for this either'
    assert model.rowCount() == 1
    assert [model.data(model.index(0, c)) for c in range(3)] == [
        'Coordinates', 'length', '']
    assert model.columns[2].choices_for(geometry, 0) == [
        'in', 'ft', 'm', 'mm', 'cm']
    model.setData(model.index(0, 2), 'in', EDIT)
    assert geometry.units_defined and geometry.length_unit == 'in'


def test_a_shape_set_declares_its_mass_unit_the_same_way(opened):
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    model, _item = opened(shapes)
    # the quantity is the *modal* mass — what normalizes the shapes —
    # though any mass unit answers it
    assert [model.data(model.index(0, c)) for c in range(3)] == [
        'Mode shapes', 'modal mass', '']
    model.setData(model.index(0, 2), 'kg', EDIT)
    assert shapes.units_defined and shapes.mass_unit == 'kg'


# ---- table conventions, exercised through this pane -------------------------

def test_a_cell_with_choices_opens_its_drop_down_every_time(
        window, opened, pump, time_data):
    """It stopped reopening once: committing rebuilt the plot synchronously
    underneath the popup, which closed it."""
    model, _item = opened(time_data)
    table = window.units_table
    assert choose(table, model, 0, 2, 'g', pump), 'first open'
    assert time_data.ordinate_unit[0] == 'g'
    assert choose(table, model, 0, 2, 'm/s**2', pump), 'same cell again'
    assert time_data.ordinate_unit[0] == 'm/s**2'
    assert choose(table, model, 1, 2, 'N', pump), 'a different cell'


def test_delete_clears_a_cell_and_puts_the_raw_values_back(
        window, opened, time_data):
    model, _item = opened(time_data)
    raw = time_data.ordinate[0].copy()
    model.setData(model.index(0, 2), 'g', EDIT)
    select_cell(window.units_table, model, 0, 2)
    assert window.units_table.clear_cells() == (1, 0)
    assert time_data.ordinate_unit[0] is None
    assert time_data.ordinate_dim[0] == 'unknown'
    assert np.allclose(time_data.ordinate[0], raw)


def test_units_copy_and_paste_between_channels(window, opened, time_data):
    model, _item = opened(time_data)
    model.setData(model.index(0, 2), 'g', EDIT)
    select_cell(window.units_table, model, 0, 2)
    assert window.units_table.copy() == 'g'
    select_cell(window.units_table, model, 1, 2)
    assert window.units_table.paste() == (1, 0)
    assert time_data.ordinate_unit[1] == 'g'


def test_a_column_of_units_is_set_in_one_go(window, opened, time_data):
    opened(time_data)
    window.units_table.selectColumn(2)
    applied, rejected = window.units_table.batch_edit('g')
    assert (applied, rejected) == (time_data.num_records, 0)
    assert time_data.units_defined


def test_batch_edit_crosses_columns_as_well(window, opened, hinted_frf):
    """Selecting a whole table and setting one value writes every unit
    cell, which is what declaring 45 channels at once looks like."""
    opened(hinted_frf)
    window.units_table.selectAll()
    applied, _rejected = window.units_table.batch_edit('g')
    assert applied == len(set(hinted_frf.response_dof))
    references = window.units_reference_table
    references.selectAll()
    applied, _rejected = references.batch_edit('lbf')
    assert applied == len(set(hinted_frf.reference_dof))
    assert hinted_frf.ordinate_dim[0] == 'acceleration/force'


# ---- a PSD says what its unit means -----------------------------------------

@pytest.fixture
def psd():
    return visualdynamics.import_file(fixture_path('plate', 'psd.npz'))


def test_a_psd_offers_the_unit_it_actually_holds(opened, psd):
    """A cell reading 'g' hides two thirds of the truth: a PSD stores the
    square of that unit, per Hz."""
    model, _item = opened(psd)
    assert model.columns[2].choices_for(psd, 0)[:4] == [
        'g²/Hz', '(in/s²)²/Hz', '(m/s²)²/Hz', '(mm/s²)²/Hz']


def test_picking_a_psd_label_declares_the_engineering_unit(opened, psd):
    model, _item = opened(psd)
    raw = psd.ordinate[0].copy()
    assert model.setData(model.index(0, 2), 'g²/Hz', EDIT)
    assert psd.ordinate_unit[0] == 'g', 'the base unit is what is stored'
    assert psd.ordinate_dim[0] == 'acceleration**2/frequency'
    assert np.allclose(psd.ordinate[0], raw * 9.80665 ** 2), 'squared, once'
    assert model.data(model.index(0, 2)) == 'g²/Hz', 'and shown whole'


def test_a_psd_still_accepts_the_bare_unit_typed_in(opened, psd):
    """The suffix is what the interface adds for clarity, not a hoop."""
    model, _item = opened(psd)
    assert model.setData(model.index(0, 2), 'g', EDIT)
    assert psd.ordinate_unit[0] == 'g'


def test_clearing_a_psd_unit_still_works(window, opened, psd):
    model, _item = opened(psd)
    raw = psd.ordinate[0].copy()
    model.setData(model.index(0, 2), 'g²/Hz', EDIT)
    select_cell(window.units_table, model, 0, 2)
    assert window.units_table.clear_cells() == (1, 0)
    assert psd.ordinate_unit[0] is None
    assert np.allclose(psd.ordinate[0], raw)


# ---- the pane has no OK button; nothing is held back ------------------------

def test_escape_puts_the_pane_away(window, opened, pump, time_data):
    """Every cell applies as it is set, so there is nothing to confirm."""
    from PySide6.QtGui import QKeySequence

    model, _item = opened(time_data)
    model.setData(model.index(0, 2), 'g', EDIT)
    pump()
    assert time_data.ordinate_unit[0] == 'g', 'applied without confirming'
    dismiss = window.units_panel.actions()[0]
    assert dismiss.shortcut() == QKeySequence(Qt.Key.Key_Escape)
    dismiss.trigger()
    assert not window.units_panel.isVisible()
    assert time_data.ordinate_unit[0] == 'g', 'and it stayed applied'


# ---- a CPSD matrix has channel units, not record units -----------------------

@pytest.fixture
def cpsd():
    """A 2×2 CPSD matrix: two channels, four records."""
    from visualdynamics.core.data import Psd

    freq = np.linspace(0.0, 100.0, 11)
    ordinate = np.outer(np.arange(1.0, 5.0), np.ones(11)).astype(complex)
    return Psd(freq, ordinate,
               response_dof=['101X+', '101X+', '102X+', '102X+'],
               reference_dof=['101X+', '102X+', '101X+', '102X+'])


def test_a_cpsd_pane_has_one_row_per_channel(opened, cpsd):
    """A cross term's unit is the product of its two channels' units per
    Hz — declared per channel, or two declarations could disagree."""
    model, _item = opened(cpsd)
    assert model.rowCount() == 2, 'four records, two channels'
    assert model.data(model.index(0, 0)) == '101X+'
    assert model.data(model.index(1, 0)) == '102X+'


def test_declaring_both_channels_converts_the_cross_terms(opened, cpsd):
    model, _item = opened(cpsd)
    raw = cpsd.ordinate.copy()
    assert model.setData(model.index(0, 2), 'g', EDIT)
    # the diagonal needs only its own channel; the cross terms wait
    assert cpsd.ordinate_dim[0] == 'acceleration**2/frequency'
    assert cpsd.ordinate_dim[1] == 'unknown'
    assert model.setData(model.index(1, 2), 'lbf', EDIT)
    assert cpsd.ordinate_dim[1] == 'acceleration*force/frequency'
    assert cpsd.ordinate_dim[3] == 'force**2/frequency'
    g, lbf = 9.80665, 4.4482216152605
    assert np.allclose(cpsd.ordinate[1], raw[1] * g * lbf)
    assert np.allclose(cpsd.ordinate[0], raw[0] * g * g)


def test_a_declaration_from_one_cross_cell_reaches_the_diagonal(
        opened, cpsd):
    """A channel's unit is a property of the channel, not of the record it
    was declared from — every record it touches converts."""
    model, _item = opened(cpsd, records=[1])     # the 101X+ × 102X+ cell
    assert model.rowCount() == 2, 'a cross cell names two channels'
    assert model.setData(model.index(0, 2), 'g', EDIT)
    assert model.setData(model.index(1, 2), 'g', EDIT)
    assert all(dim == 'acceleration**2/frequency'
               for dim in cpsd.ordinate_dim), 'all four records converted'


def test_clearing_a_channel_takes_back_every_record_it_touches(
        opened, cpsd):
    model, _item = opened(cpsd)
    raw = cpsd.ordinate.copy()
    model.setData(model.index(0, 2), 'g', EDIT)
    model.setData(model.index(1, 2), 'lbf', EDIT)
    assert model.setData(model.index(0, 2), '', EDIT)
    assert cpsd.ordinate_dim[:3] == ['unknown'] * 3, 'diagonal and crosses'
    assert cpsd.ordinate_dim[3] == 'force**2/frequency', 'the other channel keeps its own'
    assert np.allclose(cpsd.ordinate[1], raw[1]), 'raw values are back'


def test_a_cpsd_channel_offers_the_squared_label_too(opened, cpsd):
    """Same convention as a plain PSD: the cell reads g²/Hz, the channel's
    engineering unit is what is stored."""
    model, _item = opened(cpsd)
    assert model.columns[2].choices_for(cpsd, 0)[0] == 'g²/Hz'
    assert model.setData(model.index(0, 2), 'g²/Hz', EDIT)
    assert cpsd.ordinate_unit[0] == 'g'


# ---- what a mass-normalized shape is *in* ------------------------------------

def test_modal_mass_units_are_offered_as_one_over_root_mass(opened):
    """A cell reading 'kg' does not say what the shape values are in; the
    radical does — the same reasoning that shows a PSD's unit as g²/Hz."""
    from conftest import fixture_path

    import visualdynamics

    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    model, _item = opened(shapes)
    unit_column = model.columns[2]
    assert unit_column.choices == ['1/√kg', '1/√slinch', '1/√slug',
                                   '1/√lbm', '1/√tonne']
    assert model.setData(model.index(0, 2), '1/√slinch')
    assert shapes.mass_unit == 'slinch', 'the mass unit is what is stored'
    assert model.data(model.index(0, 2)) == '1/√slinch'


@pytest.mark.parametrize('typed,stored', [
    ('kg', 'kg'),                    # the bare unit still works typed
    ('1/sqrt(tonne)', 'tonne'),      # and the ascii spelling of the radical
    ('1/√lbm', 'lbm'),               # lbm was a dead shortlist entry before
])
def test_any_spelling_of_the_radical_commits_the_mass_unit(
        opened, typed, stored):
    from conftest import fixture_path

    import visualdynamics

    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    model, _item = opened(shapes)
    assert model.setData(model.index(0, 2), typed)
    assert shapes.mass_unit == stored


def test_lbm_is_a_pound_mass():
    """pint refuses 'lbm' outright, which silently killed the shortlist
    entry; normalized, it is the pound."""
    from visualdynamics.units import si_transform

    scale, offset = si_transform('lbm', 'mass')
    assert scale == pytest.approx(0.45359237)
    assert offset == 0.0


def test_sub_item_right_click_offers_units(window, pump):
    """Right-clicking a record — one or several — offers Define
    Imported Units without a trip to the parent (Brandon, 2026-08-30);
    a channel table's rows, which have no imported units, offer the
    edit table instead of a dead click."""
    import numpy as np

    import visualdynamics
    from visualdynamics.core.data import Frf

    f = np.linspace(0, 100, 64)
    window.add_object('FRF', Frf(f, np.ones((3, 64), dtype=complex),
                                 response_dof=['1Z+', '2Z+', '3Z+'],
                                 reference_dof=['9Z+'] * 3))
    window.add_object('Channels', visualdynamics.load(
        fixture_path('plate', 'channel_table.vdyn')))
    window.resize(1100, 800)
    window.show()
    pump()
    for name in ('FRF', 'Channels'):
        window._item_for_object(name).setExpanded(True)
    pump()

    grid = window.record_grids['FRF']
    at = grid.visualRect(grid.model().index(0, 0)).center()
    menu = window._grid_menu(grid, at)
    assert any('Imported Units' in a.text() for a in menu.actions())

    grid.item(0, 0).setSelected(True)
    grid.item(1, 0).setSelected(True)
    pump()
    menu = window._grid_menu(grid, at)
    assert any('for 2 channels' in a.text() for a in menu.actions()), \
        'a multi-selection defines units for exactly those records'

    channels = window.record_grids['Channels']
    at = channels.visualRect(channels.model().index(0, 0)).center()
    menu = window._grid_menu(channels, at)
    assert menu is not None, 'never a dead right-click'
    assert any('Edit Channel Table' in a.text() for a in menu.actions())
