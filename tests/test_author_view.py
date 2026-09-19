"""The specification sheet in the window: one reading, three doors.

On a shape set the toggle on the table bar makes a specification at
the modal coordinates and opens the sheet on it; on a channel table,
the same at the control channels; on a specification, Edit on the
plot bar opens the sheet on the object. There is no button: every
edit lands on the specification as it is made (Brandon, 2026-09-06).
The sheet sits beside the plot, which previews the autospectra.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path
from conftest import select_objects as _select
from PySide6.QtCore import Qt

import visualdynamics
from visualdynamics.core.author import SpecificationDraft
from visualdynamics.core.rigid import MassProperties, rigid_body_shapes


def _populate(window, pump):
    rng = np.random.default_rng(1)
    geometry = visualdynamics.Geometry(np.arange(101, 106),
                                       rng.uniform(-1.0, 1.0, (5, 3)),
                                       length_unit='m')
    rigid = rigid_body_shapes(geometry, MassProperties((0, 0, 0)))
    window.add_object('Plate', geometry)
    window.add_object('Modes', rigid)
    window.project.link('Plate', 'Modes')
    pump()
    return rigid


def _rows(view):
    return view.model().rowCount()


def _columns(view):
    return view.model().columnCount()


def _header(view, column):
    return view.model().headerData(column, Qt.Orientation.Horizontal,
                                   Qt.ItemDataRole.DisplayRole)


def _text(view, row, column):
    return view.model().data(view.model().index(row, column),
                             Qt.ItemDataRole.DisplayRole)


def _type(view, row, column, text):
    """Type into a cell, the way the delegate commits an edit."""
    model = view.model()
    return model.setData(model.index(row, column), text,
                         Qt.ItemDataRole.EditRole)


def _open(window, pump, on_plot=False):
    """Open the sheet. On a shape set or a channel table the object is
    made and selected first, one event-loop tick later."""
    action = window.author_data_action if on_plot else window.author_action
    action.setChecked(True)
    window._author_toggled(True)
    pump()
    pump()
    return window.data_pane.author_panel


# ---- the shape set door -----------------------------------------------------


def test_the_reading_is_offered_to_a_lone_shape_set(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Modes')
    assert window.author_action.isVisible()
    assert not window.author_action.isChecked()
    assert not window.data_pane.isVisibleTo(window), 'no plot until asked'
    _select(window, pump, 'Plate')
    assert not window.author_action.isVisible()
    _select(window, pump, 'Modes', 'Plate')
    assert window.author_action.isVisible(), 'the geometry may come along'
    window.add_object('Other', window.project['Modes'])
    _select(window, pump, 'Modes', 'Other')
    assert not window.author_action.isVisible(), \
        'two sets are a comparison, not a sheet'


def test_the_shapes_picked_in_the_tree_are_the_channels_written(window, pump):
    """Brandon, 2026-09-06: a virtual point's target is its three
    translations; the rotations are left out, not written as zero — a
    shape with no channel contributes nothing when the specification is
    expanded through the set, and a sheet holds positive levels."""
    _populate(window, pump)
    window.tree.setCurrentItem(window._item_for_object('Modes'))
    window.tree.clearSelection()
    grid = window.record_grids['Modes']
    for row in (0, 1, 2):
        grid.item(row, 0).setSelected(True)
    pump()
    assert window.author_action.isVisible(), 'a pick of shapes is a door'
    panel = _open(window, pump)
    assert window._author_door[1] == 'Modes Specification', \
        'the sheet carries on at the object it made'
    made = window.project['Modes Specification']
    assert made.num_records == 3
    assert list(made.response_dof) == ['M1', 'M2', 'M3']
    assert 'specification sheet on 3 channels (M1, M2, M3)' in \
        window.statusBar().currentMessage()
    assert _columns(panel.points) == 4 and _rows(panel.pairs) == 3
    assert '3 pairs unstated' in panel.problem.text()
    # the set selected again carries on at the object it made, whole or
    # picked — the same object, not a second one
    _select(window, pump, 'Modes')
    pump()
    assert window._author_door[:2] == ('spec', 'Modes Specification')
    assert 'Modes Specification (2)' not in window.project


def test_opening_on_a_set_makes_the_object_and_edits_it(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Modes')
    panel = _open(window, pump)
    assert 'Modes Specification' in window.project, 'made on opening'
    made = window.project['Modes Specification']
    assert made.num_records == 6, 'autospectra alone; nothing assumed'
    assert 'Modes Specification' in window.project.links[0]['members']
    assert window.current_object() is made, 'and the sheet carries on there'
    assert panel.isVisible()
    assert window.author_data_action.isChecked(), 'Edit, on the plot bar'
    assert window.data_pane.isVisibleTo(window), 'the preview needs the plot'
    assert panel.origin.text().startswith('opened from Modes Specification')
    assert _rows(panel.points) == 2 and _columns(panel.points) == 7
    assert _header(panel.points, 1) == 'M1\ng²/Hz'
    assert _header(panel.points, 4) == 'M4\n(rad/s²)²/Hz', \
        'a rotation per radian, in the display system'
    assert _rows(panel.pairs) == 15
    assert _text(panel.pairs, 0, 1) == '', 'unstated, not zero'
    assert '15 pairs unstated' in panel.problem.text()
    assert '15 pairs unstated' in window.statusBar().currentMessage()
    assert not hasattr(panel, 'apply_button')


def test_stating_every_pair_lands_on_the_object_at_once(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Modes')
    panel = _open(window, pump)
    panel.all_coherence.setValue(1.0)
    panel.all_phase.setValue(0.0)
    panel.all_button.click()
    pump()
    assert panel.problem.text() == ''
    key = window._author_door[1]
    draft = window._drafts[key]
    assert draft.unset_pairs() == []
    assert draft.pairs[(0, 5)] == (1.0, 0.0)
    made = window.project['Modes Specification']
    assert made.num_records == 36, 'every cross term, the moment it was stated'
    assert window.project.journal[-1].startswith(
        "project.author_specification('Modes Specification', SpecificationDraft(")
    assert window.project.journal[-1].endswith(', replace=True)')
    lines = len(window.project.journal)
    panel.scale_box.setValue(3.0)
    panel.scale_button.click()
    pump()
    assert len(window.project.journal) == lines, \
        'a run of edits on one object collapses to the line that stands'


def test_a_typed_level_lands_in_si_and_an_untouched_one_stays_exact(window,
                                                                    pump):
    _populate(window, pump)
    _select(window, pump, 'Modes')
    panel = _open(window, pump)
    key = window._author_door[1]
    before = window._drafts[key].levels[1][0]
    assert _type(panel.points, 0, 1, '0.04')
    pump()
    draft = window._drafts[key]
    assert draft.levels[0][0] == pytest.approx(
        float(window.unit_system.to_si(0.04, 'acceleration**2/frequency')))
    assert draft.levels[1][0] == before, \
        'a cell nobody touched reads back exactly, not through six digits'
    made = window.project['Modes Specification']
    assert np.real(made.ordinate[0][0]) == pytest.approx(draft.levels[0][0]), \
        'the typed level is in the object already'
    _select(window, pump, 'Plate')
    assert not panel.isVisible()
    _select(window, pump, 'Modes Specification')
    assert window.author_data_action.isChecked() and panel.isVisible()
    assert _text(panel.points, 0, 1) == '0.04'


def test_breakpoints_are_added_removed_and_scaled(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Modes')
    panel = _open(window, pump)
    panel.points.setCurrentIndex(panel.points.model().index(0, 0))
    panel.add_button.click()
    pump()
    assert _rows(panel.points) == 3
    key = window._author_door[1]
    assert window._drafts[key].frequencies == pytest.approx(
        [20.0, 200.0, 2000.0]), 'halfway in log frequency'
    assert list(window.project['Modes Specification'].abscissa) == \
        pytest.approx([20.0, 200.0, 2000.0]), 'and so is the object'
    panel.points.setCurrentIndex(panel.points.model().index(1, 0))
    panel.remove_button.click()
    pump()
    assert _rows(panel.points) == 2
    panel.points.setCurrentIndex(panel.points.model().index(0, 0))
    panel.remove_button.click()
    assert _rows(panel.points) == 2, 'a specification keeps two'
    assert 'at least two' in panel.problem.text()


def test_a_bad_cell_is_named_and_lands_nowhere(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Modes')
    panel = _open(window, pump)
    panel.all_coherence.setValue(0.0)
    panel.all_button.click()
    pump()
    assert panel.problem.text() == ''
    before = window.project['Modes Specification']
    assert not _type(panel.points, 1, 2, 'lots')
    pump()
    assert 'lots' in panel.problem.text()
    assert window.project['Modes Specification'] is before, \
        'a cell that is not a number changes nothing'
    assert not _type(panel.pairs, 3, 1, '1.5')
    pump()
    assert 'coherence' in panel.problem.text()
    assert window.project['Modes Specification'] is before


def test_a_set_with_no_mass_unit_says_what_to_declare(window, pump):
    window.import_paths([fixture_path('plate', 'shapes.npy')])
    pump()
    name = next(n for n, obj in window.objects.items()
                if isinstance(obj, visualdynamics.ShapeSet))
    _select(window, pump, name)
    _open(window, pump)
    assert 'no mass unit' in window.statusBar().currentMessage()
    assert not window.data_pane.author_panel.isVisible()


# ---- the specification door -------------------------------------------------


def test_a_specification_opens_from_the_plots_bar_and_is_edited_live(window,
                                                                     pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    _select(window, pump, 'Specification')
    assert window.author_data_action.isVisible()
    assert not window.author_action.isVisible()
    panel = _open(window, pump, on_plot=True)
    assert panel.isVisible()
    assert panel.origin.text().startswith('opened from Specification — ')
    assert 'left out' in panel.origin.text(), \
        'the 0 Hz line and the rolled-off edges are said, not hidden'
    assert _columns(panel.points) == 9
    assert _rows(panel.pairs) == 28
    before = window.project['Specification']
    assert '28 pairs unstated' in panel.problem.text()
    panel.all_coherence.setValue(0.0)
    panel.all_button.click()
    pump()
    panel.points.selectAll()               # scale acts on the selection
    panel.scale_box.setValue(3.0)
    panel.scale_button.click()
    pump()
    after = window.project['Specification']
    assert after is not before
    assert after.num_records == 64, 'every cross term, stated as zero'
    autos = [i for i in range(after.num_records)
             if after.response_dof[i] == after.reference_dof[i]]
    kept = (before.abscissa > 0) & np.all(np.real(before.ordinate) > 0, axis=0)
    assert np.allclose(after.abscissa, before.abscissa[kept]), \
        'the whole sheet: the object takes the sheet\'s lines, the band'
    assert np.allclose(np.real(after.ordinate[autos]),
                       np.real(before.ordinate)[:, kept] * 10 ** 0.3)
    assert window.project.journal[-1].endswith(', replace=True)')
    assert window.author_data_action.isChecked(), 'the sheet stays open'


def test_picked_records_open_the_sheet_on_their_channels(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    spec = window.project['Specification']
    item = window._item_for_object('Specification')
    window.tree.setCurrentItem(item)
    window.tree.clearSelection()
    grid = window.record_grids['Specification']
    grid.item(0, 0).setSelected(True)
    grid.item(2, 0).setSelected(True)
    pump()
    assert window.author_data_action.isVisible(), \
        'a pick of records is an edit of those channels'
    panel = _open(window, pump, on_plot=True)
    assert _columns(panel.points) == 3
    assert _header(panel.points, 1).startswith(spec.response_dof[0])
    assert _header(panel.points, 2).startswith(spec.response_dof[2])
    assert _rows(panel.pairs) == 1
    assert (f'sheet on 2 channels ({spec.response_dof[0]}, '
            f'{spec.response_dof[2]})') in window.statusBar().currentMessage(), \
        'the witness: which channels an edit will land on'
    assert '2 of Specification channels picked' in panel.origin.text()
    before = spec
    panel.all_coherence.setValue(0.0)
    panel.all_button.click()
    panel.points.selectAll()
    panel.scale_box.setValue(-6.0)
    panel.scale_button.click()
    pump()
    after = window.project['Specification']
    assert after.num_records == 10, 'the picked pair, both ways'
    kept = (before.abscissa > 0) & np.all(np.real(before.ordinate) > 0, axis=0)
    assert np.allclose(np.real(after.ordinate[0])[kept],
                       np.real(before.ordinate[0])[kept] * 10 ** -0.6)
    assert np.allclose(np.real(after.ordinate[1]), np.real(before.ordinate[1])), \
        'a channel not picked is untouched'
    assert np.allclose(np.real(after.ordinate[2])[kept],
                       np.real(before.ordinate[2])[kept] * 10 ** -0.6)


def test_two_specifications_open_as_one_sheet(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    window.add_object('Other', window.project['Specification'])
    pump()
    _select(window, pump, 'Specification', 'Other')
    assert window.author_data_action.isVisible()
    panel = _open(window, pump, on_plot=True)
    assert _columns(panel.points) == 17
    assert _header(panel.points, 1).startswith('Specification\n')
    assert _header(panel.points, 9).startswith('Other\n')
    assert _rows(panel.pairs) == 56, 'pairs within each, none across'
    assert _text(panel.pairs, 0, 0).startswith('Specification: ')
    before = window.project['Specification']
    panel.all_coherence.setValue(0.0)
    panel.all_button.click()
    panel.points.selectAll()
    panel.scale_box.setValue(3.0)
    panel.scale_button.click()
    pump()
    for name in ('Specification', 'Other'):
        after = window.project[name]
        assert after.num_records == 64
        kept = (before.abscissa > 0) & np.all(np.real(before.ordinate) > 0,
                                              axis=0)
        autos = [i for i in range(after.num_records)
                 if after.response_dof[i] == after.reference_dof[i]]
        assert np.allclose(np.real(after.ordinate[autos]),
                           np.real(before.ordinate)[:, kept] * 10 ** 0.3), name
    assert window.project.journal[-1].startswith(
        "project.author_specification(['Specification', 'Other'], ")


# ---- the channel table door ---------------------------------------------------


def test_a_channel_table_opens_at_its_control_channels(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    _select(window, pump, 'Channel Table')
    assert window.author_action.isVisible()
    assert window.table_bar.isVisibleTo(window), 'the toggle has to be reachable'
    panel = _open(window, pump)
    assert panel.origin.text().startswith(
        'opened from Channel Table Specification')
    table = window.project['Channel Table']
    flagged = [dof for dof, on in zip(table.dof_strings(), table.controls())
               if on]
    assert _columns(panel.points) == 1 + len(flagged)
    assert 'Channel Table Specification' in window.project, 'made on opening'
    panel.all_coherence.setValue(1.0)
    panel.all_button.click()
    pump()
    made = window.project['Channel Table Specification']
    assert made.response_dof[0] == flagged[0]
    assert made.num_records == len(flagged) ** 2
    assert isinstance(SpecificationDraft.from_specification(made),
                      SpecificationDraft)


# ---- the sheet's width is the user's ----------------------------------------


def test_the_sheet_can_be_dragged_wider(window, pump):
    """The sheet sits in a splitter beside the plot (Brandon,
    2026-09-06): its width opens at the panel's own and follows the
    divider from there, on the flat plot and the stage alike."""
    from visualdynamics.gui.author_panel import PANEL_WIDTH

    _populate(window, pump)
    _select(window, pump, 'Modes')
    panel = _open(window, pump)
    split = window.data_pane.author_split
    assert split.indexOf(panel) == split.count() - 1, 'the sheet is last'
    assert split.indexOf(window.data_pane.graphics) == 0
    assert panel.width() >= PANEL_WIDTH
    assert panel.minimumWidth() == PANEL_WIDTH and panel.maximumWidth() > 5000
    # the 3-D page, once made, is in the splitter too, beside the
    # graphics; the sheet keeps its place beside whichever view is up
    # (the sheet itself holds the drawing flat, so the page is made
    # here rather than by the render)
    window.data_pane.create_waterfall_plotter()
    assert split.indexOf(window.data_pane._waterfall_page) == 1
    assert split.indexOf(panel) == 2
    sizes = split.sizes()
    total = sum(sizes)
    view = next(i for i in range(2) if sizes[i] > 0)   # the visible view
    wider = PANEL_WIDTH + 200
    sizes[view], sizes[2] = total - wider, wider
    split.setSizes(sizes)
    pump()
    assert panel.width() == wider, 'the divider sets the width'
    assert split.sizes()[view] == total - wider, 'the view gives it up'


# ---- the two forms on the sheet ------------------------------------------------


def test_the_sheet_opens_a_target_as_lines_and_folds_to_breakpoints(window,
                                                                    pump):
    """Brandon, 2026-09-06: editing is best done on the few points the
    lines were written from. Two buttons at the top of the sheet."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    _select(window, pump, 'Specification')
    panel = _open(window, pump, on_plot=True)
    assert panel.interpolated_button.isChecked()
    assert not panel.breakpoints_button.isChecked()
    assert panel.spacing_box.value() == 2.0, 'the target\'s own 2 Hz lines'
    lines = _rows(panel.points)
    assert lines > 100
    panel.breakpoints_button.click()
    pump()
    assert panel.breakpoints_button.isChecked()
    assert _rows(panel.points) < lines
    assert 'read as' in window.statusBar().currentMessage(), \
        'the conversion is said: n lines read as m breakpoints'
    draft = window._drafts[window._author_door[1]]
    assert draft.form == 'breakpoints' and draft.spacing == 2.0
    panel.interpolated_button.click()
    pump()
    assert panel.interpolated_button.isChecked()
    assert _rows(panel.points) == lines
    assert window._drafts[window._author_door[1]].form == 'lines'


def test_a_new_sheet_takes_its_spacing_from_a_time_history(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    _select(window, pump, 'Channel Table')
    panel = _open(window, pump)
    assert panel.breakpoints_button.isChecked()
    history = window.objects['Time History']
    assert panel.spacing_box.value() == pytest.approx(
        history.sample_rate / history.averaging.frame_length)
    panel.interpolated_button.click()
    pump()
    assert panel.interpolated_button.isChecked()
    assert _rows(panel.points) > 2
    assert window._drafts[window._author_door[1]].spacing == pytest.approx(
        history.sample_rate / history.averaging.frame_length)


def test_without_a_history_the_sheet_asks_for_the_spacing(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Modes')
    panel = _open(window, pump)
    assert panel.spacing_box.value() == 0.0
    assert panel.spacing_box.text() == 'spacing?'
    panel.interpolated_button.click()
    pump()
    assert panel.breakpoints_button.isChecked(), 'nothing was guessed'
    assert 'give the spacing' in panel.problem.text()
    panel.spacing_box.setValue(2.0)
    panel.interpolated_button.click()
    pump()
    assert panel.interpolated_button.isChecked()
    assert _rows(panel.points) == 991, '20 to 2000 Hz at 2 Hz'


# ---- the ordinary table, on the sheet -------------------------------------------


def test_the_sheets_grids_are_the_applications_tables(window, pump):
    """PLAN.md "How tables behave" (Brandon, 2026-09-06: every editable
    table should be the same table). Column and row headers select,
    Cmd-click adds, the fill handle fills, Delete clears a pair."""
    from PySide6.QtCore import QItemSelectionModel

    from visualdynamics.gui.tables import CopyPasteTableView

    _populate(window, pump)
    _select(window, pump, 'Modes')
    panel = _open(window, pump)
    assert isinstance(panel.points, CopyPasteTableView)
    assert isinstance(panel.pairs, CopyPasteTableView)
    assert panel.points.selectionMode() == \
        CopyPasteTableView.SelectionMode.ExtendedSelection
    # a column header selects the column
    panel.points.selectColumn(1)
    assert sorted(i.row() for i in panel.points.selectedIndexes()) == [0, 1]
    # a row header selects the row, across every channel
    panel.points.clearSelection()
    panel.points.selectRow(0)
    assert sorted(i.column() for i in panel.points.selectedIndexes()) == \
        list(range(7))
    # Cmd-click: two cells that share nothing
    panel.points.clearSelection()
    model = panel.points.model()
    selection = panel.points.selectionModel()
    for row, column in ((0, 1), (1, 3)):
        selection.select(model.index(row, column),
                         QItemSelectionModel.SelectionFlag.Select)
    assert len(panel.points.selectedIndexes()) == 2
    # the fill handle: the first row's level filled down the column
    panel.points.clearSelection()
    key = window._author_door[1]
    assert _type(panel.points, 0, 1, '0.5')
    pump()
    model = panel.points.model()
    selection = panel.points.selectionModel()
    selection.select(model.index(0, 1), QItemSelectionModel.SelectionFlag.Select)
    applied, rejected = panel.points.fill_from_selection(1)
    pump()
    assert (applied, rejected) == (1, 0)
    draft = window._drafts[key]
    assert draft.levels[0][1] == pytest.approx(draft.levels[0][0])
    assert np.real(window.project['Modes Specification'].ordinate[0][1]) == \
        pytest.approx(draft.levels[0][0]), 'and it landed on the object'
    # Delete on a stated pair leaves it unstated
    assert _type(panel.pairs, 0, 1, '1')
    pump()
    assert window._drafts[key].pairs[(0, 1)] == (1.0, 0.0)
    panel.pairs.selectionModel().select(
        panel.pairs.model().index(0, 1), QItemSelectionModel.SelectionFlag.Select)
    panel.pairs.clear_cells()
    pump()
    assert window._drafts[key].pairs[(0, 1)] is None


def test_scale_selected_scales_what_is_selected(window, pump):
    from PySide6.QtCore import QItemSelectionModel

    _populate(window, pump)
    _select(window, pump, 'Modes')
    panel = _open(window, pump)
    key = window._author_door[1]
    before = [list(row) for row in window._drafts[key].levels]
    panel.scale_box.setValue(6.0)
    panel.scale_button.click()
    pump()
    assert 'select the levels' in panel.problem.text()
    assert window._drafts[key].levels == before, 'nothing selected, nothing scaled'
    panel.points.selectColumn(2)                   # channel M2
    panel.scale_button.click()
    pump()
    after = window._drafts[key].levels
    assert after[1] == pytest.approx([v * 10 ** 0.6 for v in before[1]])
    assert after[0] == before[0], 'the channel beside it untouched'
    panel.points.clearSelection()
    model = panel.points.model()
    panel.points.selectionModel().select(
        model.index(1, 1), QItemSelectionModel.SelectionFlag.Select)
    panel.scale_box.setValue(-3.0)
    panel.scale_button.click()
    pump()
    assert window._drafts[key].levels[0][1] == pytest.approx(
        before[0][1] * 10 ** -0.3)
    assert window._drafts[key].levels[0][0] == before[0][0]
    made = window.project['Modes Specification']
    assert np.real(made.ordinate[0][1]) == pytest.approx(before[0][1] * 10 ** -0.3)


def test_a_frequency_typed_out_of_order_is_sorted_into_place(window, pump):
    """Brandon, 2026-09-06: a breakpoint added and given a frequency
    between two others stalled the sheet — the rows were never re-sorted."""
    _populate(window, pump)
    _select(window, pump, 'Modes')
    panel = _open(window, pump)
    key = window._author_door[1]
    panel.points.setCurrentIndex(panel.points.model().index(1, 0))
    panel.add_button.click()
    pump()
    assert window._drafts[key].frequencies == [20.0, 2000.0, 4000.0]
    assert _type(panel.points, 2, 0, '100')
    pump()
    assert window._drafts[key].frequencies == [20.0, 100.0, 2000.0], \
        're-sorted, with its levels along'
    assert [_text(panel.points, r, 0) for r in range(3)] == ['20', '100', '2000']
    assert list(window.project['Modes Specification'].abscissa) == \
        pytest.approx([20.0, 100.0, 2000.0])
    assert panel.problem.text().startswith('15 pairs unstated')
    assert not _type(panel.points, 1, 0, '2000')
    assert 'already a breakpoint' in panel.problem.text()


def _handles(window):
    from visualdynamics.plot.bands import BandHandle

    plot = next(item for item in window.data_pane.graphics.ci.items
                if hasattr(item, 'listDataItems'))
    return [item for item in plot.items if isinstance(item, BandHandle)]


def test_the_bands_are_dragged_on_the_plot_for_every_selected_channel(window,
                                                                      pump):
    """Brandon, 2026-09-06: the bands are shown on the plot and nowhere
    else; a drag lands when it stops, on every selected channel in that
    frequency range."""
    _populate(window, pump)
    _select(window, pump, 'Modes')
    window.data_pane.waterfall_action.setChecked(False)   # the flat plot edits
    panel = _open(window, pump)
    assert not hasattr(panel, 'band_boxes'), 'no band boxes on the sheet'
    handles = _handles(window)
    assert {(h.kind, h.side) for h in handles} == {
        ('warning', 'lower'), ('warning', 'upper'),
        ('abort', 'lower'), ('abort', 'upper')}
    assert all(h.segments == [0] for h in handles), 'one segment, one run'
    upper = next(h for h in handles if (h.kind, h.side) == ('abort', 'upper'))
    key = window._author_door[1]
    before = window.project['Modes Specification']
    upper.release(0.3)               # three tenths of a decade: 3 dB
    pump()
    draft = window._drafts[key]
    assert all(entry['abort'] == [(-9.0, 9.0)] for entry in draft.bands), \
        'every channel of the sheet, symmetric'
    made = window.project['Modes Specification']
    assert made is not before
    autos = [i for i in range(made.num_records)
             if made.response_dof[i] == made.reference_dof[i]]
    for i in autos:
        ratio = np.real(made.limits['abort_upper'][i]) / np.real(made.ordinate[i])
        assert np.allclose(10 * np.log10(ratio), 9.0)
    assert 'abort band above set to +9 dB on 6 channels' in \
        window.statusBar().currentMessage()
    handles = _handles(window)
    assert handles, 'the handles are back on the new drawing'
    upper = next(h for h in handles if (h.kind, h.side) == ('abort', 'upper'))
    assert upper.value_db == 9.0 and upper.label.textItem.toPlainText() == '+9 dB', \
        'the level is said on the plot'
    lower = next(h for h in handles if (h.kind, h.side) == ('abort', 'lower'))
    assert lower.label.textItem.toPlainText() == '−9 dB'
    upper.release(0.0)
    pump()
    assert window.project['Modes Specification'] is made, 'no drag, no edit'
    # the drag snaps to whole decibels: seven tenths of a decibel (seven
    # hundredths of a decade) is one whole decibel
    upper.release(0.07)
    pump()
    assert all(entry['abort'] == [(-10.0, 10.0)]
               for entry in window._drafts[key].bands), 'snapped to 10, not 9.7'
    assert upper.snapped(0.04) == 9.0, 'and four tenths is none at all'
    upper.release(0.04)
    pump()
    assert window.project['Modes Specification'] is not made
    assert all(entry['abort'] == [(-10.0, 10.0)]
               for entry in window._drafts[key].bands), 'nothing moved'


def test_a_specification_without_limits_offers_the_defaults_to_drag(window, pump):
    """Brandon, 2026-09-06: where there are no warning or abort limits,
    the sheet is where they are added — the default bands are on the
    plot to drag, the origin line says they are not on the object yet,
    and the first drag writes them."""
    axis = np.geomspace(10.0, 2000.0, 16)
    level = np.atleast_2d(np.full(16, 1e-3))
    bare = visualdynamics.Specification(
        abscissa=axis, ordinate=level, response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'])
    window.add_object('Bare', bare)
    window.data_pane.waterfall_action.setChecked(False)
    _select(window, pump, 'Bare')
    panel = _open(window, pump, on_plot=True)
    assert 'no warning band on the specification' in panel.origin.text()
    assert 'no abort band on the specification' in panel.origin.text()
    handles = _handles(window)
    assert {(h.kind, h.side) for h in handles} == {
        ('warning', 'lower'), ('warning', 'upper'),
        ('abort', 'lower'), ('abort', 'upper')}, 'the defaults, there to drag'
    upper = next(h for h in handles if (h.kind, h.side) == ('abort', 'upper'))
    upper.release(0.3)               # 6 to 9 dB
    pump()
    made = window.project['Bare']
    assert sorted(made.limits) == ['abort_lower', 'abort_upper',
                                   'warning_lower', 'warning_upper']
    ratio = np.real(made.limits['abort_upper'][0]) / np.real(made.ordinate[0])
    assert np.allclose(10 * np.log10(ratio), 9.0)
    assert 'no abort band' not in panel.origin.text(), 'on the object now'


def test_a_drag_lands_every_channel_at_the_level_reached(window, pump):
    """Brandon, 2026-09-06: one channel at 3 dB and another at 4 dB,
    the handle dragged from 3 to 4 — both must come out at 4 dB, not 4
    and 5. The handle reports where it was let go, and every channel
    the sheet holds is put there."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    window.data_pane.waterfall_action.setChecked(False)
    _select(window, pump, 'Specification')
    _open(window, pump, on_plot=True)
    key = window._author_door[1]
    draft = window._drafts[key]
    other = draft.channels[1]
    window._author_edited(draft.with_band('warning', -4.0, 4.0, [other]))
    pump()
    draft = window._drafts[key]
    assert draft.bands[0]['warning'][0] == (-3.0, 3.0)
    assert draft.bands[1]['warning'][0] == (-4.0, 4.0)
    upper = next(h for h in _handles(window)
                 if (h.kind, h.side) == ('warning', 'upper'))
    assert upper.value_db == 3.0, 'the drawn channel is the first'
    upper.release(0.1)               # a tenth of a decade: to 4 dB
    pump()
    draft = window._drafts[key]
    assert all(pair == (-4.0, 4.0) for entry in draft.bands
               for pair in entry['warning']), 'both at 4, not 4 and 5'
    assert 'warning band above set to +4 dB on' in \
        window.statusBar().currentMessage()
    made = window.project['Specification']
    for i in range(made.num_records):
        if made.response_dof[i] != made.reference_dof[i]:
            continue
        y = np.real(made.ordinate[i])
        lit = y > 0
        ratio = 10 * np.log10(np.real(made.limits['warning_upper'][i])[lit] / y[lit])
        assert np.allclose(ratio, 4.0), made.response_dof[i]


def test_a_drag_with_channels_picked_moves_those_channels_only(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    spec = window.project['Specification']
    window.tree.setCurrentItem(window._item_for_object('Specification'))
    window.tree.clearSelection()
    grid = window.record_grids['Specification']
    grid.item(0, 0).setSelected(True)
    grid.item(2, 0).setSelected(True)
    pump()
    window.data_pane.waterfall_action.setChecked(False)
    _open(window, pump, on_plot=True)
    lower = next(h for h in _handles(window)
                 if (h.kind, h.side) == ('warning', 'lower'))
    lower.release(-0.2)              # 2 dB further down
    pump()
    made = window.project['Specification']
    for i in range(made.num_records):
        if made.response_dof[i] != made.reference_dof[i]:
            continue
        y = np.real(made.ordinate[i])
        lit = y > 0
        ratio = 10 * np.log10(np.real(made.limits['warning_lower'][i])[lit]
                              / y[lit])
        expected = -5.0 if made.response_dof[i] in (
            spec.response_dof[0], spec.response_dof[2]) else -3.0
        assert np.allclose(ratio, expected), made.response_dof[i]


def test_a_write_through_one_door_reaches_the_drafts_cached_under_others(
        window, pump):
    """Brandon, 2026-09-06: with Uniform off he dragged the plateau's
    warning band to ±1 dB on the whole specification, looked at the
    sub-items (each opened its own sheet, cached under its own door),
    went back to the whole specification and dragged it to ±3 dB — and
    the sub-items still showed ±1 dB. The sheet caches one draft per
    door; a write through one door has to drop the drafts cached under
    the object's other doors, or they keep showing the object as it
    was. And the helper lines the first drag wrote must not come back
    as breakpoints — two rows at 40 Hz — on the reopened sheets."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    window.data_pane.waterfall_action.setChecked(False)
    _select(window, pump, 'Specification')
    _open(window, pump, on_plot=True)
    window._author_form_asked('breakpoints', None)
    pump()
    window._author_constraint_asked('uniform', False)
    pump()
    whole = window._author_door[1]
    breakpoints = list(window._drafts[whole].frequencies)
    assert 2 < len(breakpoints) < 20, 'a few breakpoints, several segments'
    plateau = max(_handles(window), key=lambda h: h.value_db == 3.0
                  and h.kind == 'warning' and h.side == 'upper' and len(h.segments))
    plateau.release(-0.2)
    pump()
    assert window._drafts[whole].bands[0]['warning'][plateau.segments[0]] == (-1.0, 1.0)
    # a sub-item: its own door, its own cached draft, reading ±1 dB
    grid = window.record_grids['Specification']
    window.tree.clearSelection()
    grid.item(2, 0).setSelected(True)
    pump()
    pump()
    sub = window._author_door[1]
    assert sub != whole and sub in window._drafts
    assert window._drafts[sub].frequencies == breakpoints, \
        'the helper lines the step was written with are not breakpoints here'
    assert window._drafts[sub].bands[0]['warning'][plateau.segments[0]] == (-1.0, 1.0)
    # back to the whole specification, and back to ±3 dB
    grid.clearSelection()
    _select(window, pump, 'Specification')
    pump()
    assert window._author_door[1] == whole
    plateau = next(h for h in _handles(window)
                   if (h.kind, h.side) == ('warning', 'upper') and h.value_db == 1.0)
    plateau.release(0.2)
    pump()
    assert window._drafts[whole].bands[0]['warning'][plateau.segments[0]] == (-3.0, 3.0)
    assert sub not in window._drafts, 'the stale sheet is gone'
    # and the sub-item, reopened, reads the object as it is now (the
    # write rebuilt the tree's children, so the grid is a new widget)
    grid = window.record_grids['Specification']
    window.tree.clearSelection()
    grid.item(2, 0).setSelected(True)
    pump()
    pump()
    assert window._author_door[1] == sub
    assert window._drafts[sub].frequencies == breakpoints
    assert all(pair == (-3.0, 3.0) for pair in window._drafts[sub].bands[0]['warning'])
    handles = _handles(window)
    assert handles and all(h.value_db == 3.0 for h in handles
                           if (h.kind, h.side) == ('warning', 'upper'))


def test_the_constraints_are_the_sheets_two_boxes_for_the_whole_object(
        window, pump):
    """Brandon, 2026-09-06: Symmetric and Uniform apply across every
    channel of the specification, as Breakpoints and Interpolated do."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    window.tree.setCurrentItem(window._item_for_object('Specification'))
    window.tree.clearSelection()
    grid = window.record_grids['Specification']
    grid.item(0, 0).setSelected(True)
    grid.item(2, 0).setSelected(True)
    pump()
    window.data_pane.waterfall_action.setChecked(False)
    panel = _open(window, pump, on_plot=True)
    assert panel.symmetric_box.isChecked() and panel.uniform_box.isChecked()
    panel.symmetric_box.setChecked(False)
    pump()
    from visualdynamics.core.author import SpecificationDraft

    whole = SpecificationDraft.from_specification(
        window.project['Specification'])
    assert not any(entry['symmetric'] for entry in whole.bands), \
        'every channel of the object, not the two the sheet holds'
    assert all(entry['uniform'] for entry in whole.bands)
    assert 'symmetric off for every channel' in window.statusBar().currentMessage()
    assert _columns(panel.points) == 3, 'the sheet is still on its two'
    key = window._author_door[1]
    upper = next(h for h in _handles(window)
                 if (h.kind, h.side) == ('warning', 'upper'))
    upper.release(0.1)
    pump()
    for entry in window._drafts[key].bands:
        assert all(pair == pytest.approx((-3.0, 4.0))
                   for pair in entry['warning']), \
            'one edge alone, on the picked channels'
    panel.uniform_box.setChecked(False)
    pump()
    whole = SpecificationDraft.from_specification(
        window.project['Specification'])
    assert not any(entry['uniform'] for entry in whole.bands)
    panel.symmetric_box.setChecked(True)
    pump()
    whole = SpecificationDraft.from_specification(
        window.project['Specification'])
    assert all(entry['symmetric'] for entry in whole.bands)
    assert all(pair == pytest.approx((-4.0, 4.0))
               for pair in whole.bands[0]['warning']), 'mirrored to the band above'


def test_the_form_switch_converts_the_whole_specification(window, pump):
    """Brandon, 2026-09-06: Breakpoints or Interpolated is the object's,
    whichever channels the sheet is open on."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    window.tree.setCurrentItem(window._item_for_object('Specification'))
    window.tree.clearSelection()
    grid = window.record_grids['Specification']
    grid.item(0, 0).setSelected(True)
    grid.item(2, 0).setSelected(True)
    pump()
    panel = _open(window, pump, on_plot=True)
    target = window.project['Specification']
    lines = len(target.abscissa)
    in_band = int(((target.abscissa > 0)
                   & np.all(np.real(target.ordinate) > 0, axis=0)).sum())
    assert _columns(panel.points) == 3, 'the sheet is on two channels'
    panel.breakpoints_button.click()
    pump()
    made = window.project['Specification']
    assert len(made.abscissa) < lines, 'every channel folded, not two'
    assert made.num_records == 8
    assert _columns(panel.points) == 3, 'and the sheet is still on its two'
    assert panel.breakpoints_button.isChecked()
    panel.interpolated_button.click()
    pump()
    assert len(window.project['Specification'].abscissa) == in_band, \
        'the lines of the band, every channel'


def test_uniform_off_shows_a_handle_per_section_moved_on_its_own(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    _select(window, pump, 'Specification')
    window.data_pane.waterfall_action.setChecked(False)
    panel = _open(window, pump, on_plot=True)
    upper = [h for h in _handles(window) if (h.kind, h.side) == ('abort', 'upper')]
    assert len(upper) == 1, 'uniform: one handle over the whole range'
    panel.uniform_box.setChecked(False)
    pump()
    key = window._author_door[1]
    draft = window._drafts[key]
    channel = draft.channels.index(window._plotted_pair[0])
    sections = draft.sections(channel)
    assert len(sections) > 1
    upper = [h for h in _handles(window) if (h.kind, h.side) == ('abort', 'upper')]
    assert len(upper) == len(sections), 'one handle per linear section'
    assert [h.segments for h in upper] == sections
    upper[1].release(0.3)
    pump()
    draft = window._drafts[key]
    for entry in draft.bands:
        for s in range(len(draft.frequencies) - 1):
            expected = (-9.0, 9.0) if s in sections[1] else (-6.0, 6.0)
            assert entry['abort'][s] == pytest.approx(expected), s
    labels = sorted(h.label.textItem.toPlainText()
                    for h in _handles(window) if (h.kind, h.side) == ('abort', 'upper'))
    assert labels.count('+9 dB') == 1 and labels.count('+6 dB') == len(sections) - 1


def test_the_sheet_edits_on_the_flat_plot_only(window, pump):
    """Brandon, 2026-09-06: no 3-D view in edit mode — every editing
    gesture lives on the flat plot."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    _select(window, pump, 'Specification')
    window.data_pane.waterfall_action.setChecked(True)   # the stage, as usual
    window.render_current()
    pump()
    assert window.data_pane.showing_waterfall, 'the stage before the sheet'
    _open(window, pump, on_plot=True)
    assert not window.data_pane.showing_waterfall, 'the flat plot while editing'
    assert not window.data_pane.waterfall_action.isVisible(), \
        'and no toggle to bring the stage back under the handles'
    assert _handles(window), 'the handles are there to drag'
    window.author_data_action.setChecked(False)
    window._author_toggled(False)
    pump()
    assert window.data_pane.showing_waterfall, 'the stage comes back after'


def test_the_dashed_lines_step_where_the_shading_steps(window, pump):
    """The handle of a section draws that section's band over its own
    span, so two sections meet at one frequency at two levels — the
    shading always did, the lines slanted (Brandon, 2026-09-06)."""
    from visualdynamics.core.author import SpecificationDraft
    from visualdynamics.plot.bands import band_edges

    draft = SpecificationDraft(
        ['1Z+'], ['acceleration'], [10.0, 40.0, 200.0, 2000.0],
        [[0.0002, 0.002, 0.002, 0.0002]]).constrained(uniform=False)
    draft = draft.with_band('abort', -9.0, 9.0, segments=[1])
    pieces = band_edges(draft, 0, visualdynamics.SI)[('abort', 'upper')]
    assert [run for run, _x, _y in pieces] == [[0], [1], [2]]
    _run, x0, y0 = pieces[0]
    _run, x1, y1 = pieces[1]
    assert x0[-1] == x1[0] == 40.0, 'the two pieces meet at the corner'
    assert y0[-1] == pytest.approx(0.002 * 10 ** 0.6), "the skirt's own +6 dB"
    assert y1[0] == pytest.approx(0.002 * 10 ** 0.9), "the plateau's own +9 dB"


def test_a_failure_in_a_sheet_gesture_is_said_and_logged(window, pump, tmp_path,
                                                          monkeypatch):
    """Brandon, 2026-09-06: Breakpoints spun, the button flipped, and
    the old rows stood. A refusal had been shown and then overwritten
    by the render that followed it; and a Qt slot that raises is
    swallowed. The failure is said after the render and its traceback
    kept in a log."""
    from visualdynamics.core.author import SpecificationDraft
    from visualdynamics.gui.main_window import MainWindow

    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    _select(window, pump, 'Specification')
    panel = _open(window, pump, on_plot=True)
    log = tmp_path / 'sheet.log'
    monkeypatch.setattr(MainWindow, 'SHEET_LOG', log)

    def refuse(self, *args, **kwargs):
        raise ValueError('boom: the fold refused')

    monkeypatch.setattr(SpecificationDraft, 'to_breakpoints', refuse)
    before = window.project['Specification']
    panel.breakpoints_button.click()
    pump()
    assert window.project['Specification'] is before, 'nothing was written'
    status = window.statusBar().currentMessage()
    assert status.startswith('breakpoints failed: boom: the fold refused'), \
        'the refusal stands after the render, not under it'
    assert str(log) in status
    text = log.read_text()
    assert '--- breakpoints' in text and 'ValueError: boom' in text
    assert 'Traceback' in text
    assert _rows(panel.points) > 4, 'the sheet still shows the lines it has'



def test_a_virtual_points_specification_opens_from_the_plots_bar(window, pump):
    """Rows numbered '1', '2' with no direction — a controller's
    virtual response — refused the sheet with "channel '1' names no
    direction" (Brandon, 2026-09-18). It opens now, on both the
    requirement on lines and the one on bands."""
    freq = np.array([20.0, 80.0, 800.0, 2000.0])
    level = np.array([[1e-3, 4e-3, 4e-3, 1e-3], [2e-3, 8e-3, 8e-3, 2e-3]])
    spec = visualdynamics.Specification(
        freq, level, response_dof=['1', '2'],
        ordinate_dim=['acceleration**2/frequency'] * 2,
        ordinate_unit=['m/s**2'] * 2)
    spec.interpolation = 'log_log'
    window.add_object('Virtual', spec)
    window.add_object('Virtual Octave', spec.to_octave(6))
    for name in ('Virtual', 'Virtual Octave'):
        _select(window, pump, name)
        assert window.author_data_action.isVisible()
        panel = _open(window, pump, on_plot=True)
        assert panel.isVisible(), name
        assert panel.origin.text().startswith(f'opened from {name}'), \
            window.statusBar().currentMessage()
        window.author_data_action.trigger()
        pump()
