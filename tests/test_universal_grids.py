"""Every expansion is a grid — even one column wide.

One format for every object that expands into sub-items means one set of
habits: the same selection, the same deletion, the same icons. Rows are DOFs
(modes for a shape set), columns are references or captures, and a single
unlabeled column when the row alone is the identity. Geometry keeps its
category list, on request.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import Frf
from visualdynamics.gui.record_grid import grid_axes, grid_plan


@pytest.fixture(scope='module')
def spectra():
    return visualdynamics.import_file(fixture_path('plate', 'modal_spectra.nc4'))


def test_a_channel_table_grids_by_dof(spectra):
    plan = grid_plan(spectra['channel_table'])
    assert plan.kind == 'channel'
    assert plan.columns == ['']
    assert len(plan.rows) == spectra['channel_table'].num_channels
    # channel one is the drive point, whose DOF two channels share —
    # the accelerometer and the force gauge — but their units name
    # different quantities, so their icons tell them apart and neither
    # wears a '#' marker
    dofs = spectra['channel_table'].dof_strings()
    assert plan.row_labels[0] == dofs[0]
    lone = dofs.index('107Z+')
    assert plan.row_labels[lone] == '107Z+'


def test_a_shape_set_grids_by_mode():
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    plan = grid_plan(shapes)
    assert plan.kind == 'mode'
    assert plan.columns == ['']
    assert len(plan.rows) == shapes.num_shapes
    assert plan.row_labels[0] == shapes.mode_label(0)


def test_a_ragged_set_is_a_grid_with_holes():
    """A record deleted out of a full matrix leaves a hole, not a 6859-row
    flat list. The hole says precisely what is absent."""
    frf = Frf(abscissa=np.linspace(0, 128, 16),
              ordinate=np.ones((6, 16), dtype=complex),
              response_dof=['1Z+', '1Z+', '2Z+', '2Z+', '3Z+', '3Z+'],
              reference_dof=['1Z+', '2Z+'] * 3,
              ordinate_dim='acceleration/force',
              ordinate_unit='m/s**2', reference_unit='N')
    frf.delete_records([3])                  # 2Z+/2Z+ gone
    plan = grid_plan(frf)
    assert len(plan.rows) == 3 and len(plan.columns) == 2
    assert len(plan.cells) == 5, 'five records, one hole'
    assert (1, 1) not in plan.cells


def test_a_hole_is_disabled_on_screen(qt_app):
    from PySide6.QtCore import Qt

    from visualdynamics.gui.record_grid import RecordGrid

    frf = Frf(abscissa=np.linspace(0, 128, 16),
              ordinate=np.ones((6, 16), dtype=complex),
              response_dof=['1Z+', '1Z+', '2Z+', '2Z+', '3Z+', '3Z+'],
              reference_dof=['1Z+', '2Z+'] * 3,
              ordinate_dim='acceleration/force',
              ordinate_unit='m/s**2', reference_unit='N')
    frf.delete_records([3])
    grid = RecordGrid(frf)
    hole = grid.item(1, 1)
    assert hole.flags() == Qt.ItemFlag.NoItemFlags
    assert hole.icon().isNull()
    grid.selectAll()
    assert len(grid.selected_records()) == 5, 'holes are not selectable'


def test_a_single_column_grid_hides_its_header(qt_app, spectra):
    from visualdynamics.gui.record_grid import RecordGrid

    grid = RecordGrid(spectra['Modal_coherence'])
    assert not grid.horizontalHeader().isVisible()
    frf_grid = RecordGrid(spectra['Modal_frf'])
    assert not frf_grid.horizontalHeader().isHidden()


def test_a_channel_cell_wears_its_quantity(qt_app, spectra):
    """The channel table names each channel's unit; m/s^2 is an
    acceleration, so its cell carries the acceleration icon."""
    from visualdynamics.gui.icons import quantity_icon
    from visualdynamics.gui.record_grid import RecordGrid

    table = spectra['channel_table']
    grid = RecordGrid(table)
    accel = bytes(quantity_icon('acceleration').pixmap(16, 16)
                  .toImage().constBits())
    force = bytes(quantity_icon('force').pixmap(16, 16).toImage().constBits())
    cells = [bytes(grid.item(i, 0).icon().pixmap(16, 16).toImage().constBits())
             for i in range(grid.rowCount())]
    assert cells.count(accel) == 11 and cells.count(force) == 2


def test_channel_grid_selection_speaks_the_tree_vocabulary(window, pump):
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    window.tree.clearSelection()
    item = window._item_for_object('Channel Table')
    window.tree.setCurrentItem(item)
    item.setExpanded(True)
    pump()
    grid = window.record_grids['Channel Table']
    assert grid.kind == 'channel'
    grid.item(2, 0).setSelected(True)
    pump()
    kinds = {(kind, detail) for kind, _n, _o, detail
             in window.selected_references()}
    assert ('channel', 2) in kinds


def test_deleting_grid_picked_channels_deletes_channels(window, pump):
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    window.tree.clearSelection()
    item = window._item_for_object('Channel Table')
    window.tree.setCurrentItem(item)
    item.setExpanded(True)
    pump()
    table = window.objects['Channel Table']
    before = table.num_channels
    window.record_grids['Channel Table'].select_records([0, 1])
    pump()
    window.delete_selected()
    assert table.num_channels == before - 2
    assert 'Channel Table' in window.objects


def test_clicking_the_owner_row_releases_its_grid_picks(window, pump):
    """A cell selected minutes ago must not turn a later delete-the-object
    into delete-that-record. Making the owner current means the whole
    object."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    window.tree.clearSelection()
    item = window._item_for_object('FRF')
    window.tree.setCurrentItem(item)
    pump()
    grid = window.record_grids['FRF']
    grid.item(0, 0).setSelected(True)
    pump()
    assert grid.selected_records() == [0]
    window.tree.setCurrentItem(window._item_for_object('Channel Table'))
    pump()
    window.tree.setCurrentItem(item)          # the user clicks FRF's row
    pump()
    assert grid.selected_records() == [], 'the pick was released'


def test_grid_axes_speaks_for_every_kind(spectra):
    for name in ('channel_table', 'Modal_frf', 'Modal_coherence'):
        axes = grid_axes(spectra[name])
        assert axes is not None, name


def test_a_mixed_cpsd_grid_is_the_full_matrix():
    """Channel identity is DOF plus quantity — for references too. A
    drive point's accelerometer and load cell share a DOF, and keyed on
    the DOF alone a 13-channel CPSD came up 11 columns with 26 records
    shadowed behind their cells, while the occurrence tiebreaker read
    the repeated reference DOF as one column measured twice and grew a
    phantom row per response: 26 rows for 13 channels."""
    import numpy as np

    from visualdynamics.core.data import Psd
    from visualdynamics.gui.record_grid import grid_plan

    channels = [('1Z+', 'acceleration'), ('1Z+', 'force'),
                ('2Z+', 'acceleration')]
    records, rdofs, fdofs, dims = [], [], [], []
    for rd, rq in channels:
        for fd, fq in channels:
            records.append(np.full(4, 1.0 + 0.5j))
            rdofs.append(rd)
            fdofs.append(fd)
            dims.append(f'{rq}**2/frequency' if rq == fq
                        else f'{rq}*{fq}/frequency')
    cpsd = Psd(np.arange(1.0, 5.0), np.array(records),
               response_dof=rdofs, reference_dof=fdofs,
               ordinate_dim=dims)
    plan = grid_plan(cpsd)
    assert len(plan.rows) == 3, 'one row per response channel'
    assert plan.columns == ['1Z+', '1Z+', '2Z+'], (
        'one column per reference channel, the header saying the DOF '
        'and nothing else')
    assert plan.column_marks == ['acceleration', 'force', None], (
        'the quantity marks the column — drawn as its icon — only '
        'where the DOF is ambiguous')
    assert len(plan.cells) == 9, 'every record reachable, none shadowed'


def test_channel_rows_marked_only_when_the_icons_cannot_help():
    """A drive point's load cell and accelerometer share a DOF and are
    told apart by their icons, so no '#2' — but two channels at one DOF
    whose units name the same quantity (or none) still need the
    marker, because their icons are identical."""
    from visualdynamics.core.channel_table import ChannelTable
    from visualdynamics.gui.record_grid import grid_plan

    told_apart = ChannelTable({'channel': [1, 2], 'node': ['101', '101'],
                               'direction': ['Z+', 'Z+'],
                               'unit': ['m/s**2', 'N']})
    assert grid_plan(told_apart).row_labels == ['101Z+', '101Z+']

    identical = ChannelTable({'channel': [1, 2], 'node': ['101', '101'],
                              'direction': ['Z+', 'Z+'],
                              'unit': ['m/s**2', 'm/s**2']})
    assert grid_plan(identical).row_labels == ['101Z+ #1', '101Z+ #2']


def _two_frfs(window, pump):
    f = np.linspace(0, 100, 64)

    def frf(dofs):
        return Frf(f, np.ones((len(dofs), 64), dtype=complex),
                   response_dof=dofs, reference_dof=['9Z+'] * len(dofs))

    window.add_object('A', frf(['1Z+', '2Z+', '3Z+']))
    window.add_object('B', frf(['4Z+', '5Z+', '6Z+']))
    # shown, sized and expanded, or neither focus nor synthetic clicks
    # flow and a deliberate break of what these tests pin passes
    # silently — the falsification demanded it
    window.resize(1200, 900)
    window.show()
    pump()
    for name in ('A', 'B'):
        window._item_for_object(name).setExpanded(True)
    pump()
    return window.record_grids['A'], window.record_grids['B']


def test_focus_entering_a_grid_keeps_the_selection(window, pump):
    """QAbstractItemView answers an index widget's FocusIn by moving
    the current index to the widget's own row — the grid's holder,
    unselectable, so the move cleared the whole tree selection. Any
    focus a cell pick did not follow (a click on a grid's header or
    margins, a plain setFocus) silently dropped every other object's
    picks: a record chosen in one FRF's grid vanished when a click
    aimed at another grid's first row landed a few pixels high
    (Brandon, 2026-08-30)."""
    grid_a, grid_b = _two_frfs(window, pump)
    grid_a.item(1, 0).setSelected(True)
    pump()
    assert grid_a.selected_records() == [1]
    assert window._item_for_object('A').isSelected()

    grid_b.setFocus()
    pump()
    assert grid_a.selected_records() == [1], \
        'focus into another grid is not a selection statement'
    assert window._item_for_object('A').isSelected(), \
        'the tree kept the selection; the grid kept the focus'


def test_a_plain_pick_reads_alone_and_the_modifier_adds(window, pump):
    """Brandon's rule (2026-08-30): a single left click on a sub-item
    deselects every other sub-item, whatever object holds it —
    multi-selection is the modifier's job. The old give-way skipped
    everything when the clicked grid's own item was already selected,
    so two objects browsed together kept both their picks through
    plain clicks."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    grid_a, grid_b = _two_frfs(window, pump)

    def click(grid, row, modifier=Qt.KeyboardModifier.NoModifier):
        QTest.mouseClick(
            grid.viewport(), Qt.MouseButton.LeftButton, modifier,
            grid.visualRect(grid.model().index(row, 0)).center())
        if modifier != Qt.KeyboardModifier.NoModifier:
            # a modified QTest click latches the process-wide modifier
            # state; without the release every later test in this
            # worker runs with Control held
            QTest.keyRelease(grid.viewport(), Qt.Key.Key_Control,
                             Qt.KeyboardModifier.NoModifier)
        pump()

    # his exact path: both objects selected first, then plain picks
    for name in ('A', 'B'):
        window._item_for_object(name).setSelected(True)
    pump()
    click(grid_a, 1)
    click(grid_b, 0)
    assert grid_a.selected_records() == [], 'the plain pick reads alone'
    assert grid_b.selected_records() == [0]
    assert not window._item_for_object('A').isSelected()

    click(grid_a, 2, Qt.KeyboardModifier.ControlModifier)
    assert grid_a.selected_records() == [2], 'the modifier adds'
    assert grid_b.selected_records() == [0], 'and the first pick stands'


def test_a_cross_object_selection_reads_on_the_stage(window, pump):
    """An FRF from each of two surveys had no 3-D reading at all
    (Brandon, 2026-08-30). Same type, same abscissa: the picks stack
    onto one stage, display only — the merge rules protect stored
    objects from telling false stories, and a drawing tells none."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    grid_a, grid_b = _two_frfs(window, pump)
    QTest.mouseClick(grid_a.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier,
                     grid_a.visualRect(grid_a.model().index(1, 0)).center())
    pump()
    QTest.mouseClick(grid_b.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ControlModifier,
                     grid_b.visualRect(grid_b.model().index(0, 0)).center())
    QTest.keyRelease(grid_b.viewport(), Qt.Key.Key_Control,
                     Qt.KeyboardModifier.NoModifier)
    pump()
    pane = window.data_pane
    assert pane.waterfall_action.isVisible(), \
        'the 3-D toggle is offered for the cross-object pick'
    if not pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
        pump()
    window.render_current()
    pump()
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible(), 'the stage is up'
    assert len(window._waterfall_drawn) == 2, 'one record from each'
    assert 'A + B' in window._waterfall_object
