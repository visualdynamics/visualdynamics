"""Every table in visualdynamics behaves the same way.

The conventions are written up under "How tables behave" in PLAN.md. This
checks them against every table model there is, so a new one cannot quietly
grow its own manners — the point is that the user learns a table once.
"""

from __future__ import annotations

import pytest
from conftest import fixture_path
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest

import visualdynamics
from visualdynamics.gui.object_tables import (
    channel_table_model,
    coordinate_system_table_model,
    element_group_table_model,
    element_table_model,
    node_table_model,
    shape_table_model,
    units_table_model,
)
from visualdynamics.gui.tables import ChoiceDelegate, CopyPasteTableView
from visualdynamics.units import DEFAULT_SYSTEM

EDIT = Qt.ItemDataRole.EditRole


def geometry():
    return visualdynamics.import_file(fixture_path('plate', 'geometry.npz'))


def meshed_geometry():
    """One with elements in it."""
    return visualdynamics.import_file(fixture_path('plate',
                                                   'geometry.npz'))


def every_model():
    """(name, model) for every table the interface can put on screen."""
    geo = geometry()
    meshed = meshed_geometry()
    data = visualdynamics.import_file(fixture_path('plate', 'time.npz'))
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    channels = visualdynamics.load(fixture_path('plate', 'channel_table.vdyn'))
    return [
        ('nodes', node_table_model(geo, DEFAULT_SYSTEM)),
        ('coordinate systems', coordinate_system_table_model(geo,
                                                             DEFAULT_SYSTEM)),
        ('elements', element_table_model(meshed, DEFAULT_SYSTEM)),
        ('blocks', element_group_table_model(meshed, DEFAULT_SYSTEM)),
        ('shapes', shape_table_model(shapes)),
        ('channels', channel_table_model(channels)),
        ('imported units', units_table_model(data)),
        ('imported units (geometry)', units_table_model(geometry())),
        *sheet_tables(),
    ]


def sheet_tables():
    """The specification sheet's two grids (2026-09-06): the same
    tables as every other, held to the same contract."""
    from visualdynamics.core.author import SpecificationDraft
    from visualdynamics.gui.author_panel import Sheet, sheet_models

    draft = SpecificationDraft.at_dofs(['101Z+', '104Z+']).with_all_pairs(0.5)
    points, pairs = sheet_models(Sheet(draft, DEFAULT_SYSTEM))
    return [('specification breakpoints', points),
            ('specification pairs', pairs)]


@pytest.fixture
def models(qt_app):
    return every_model()


def test_every_table_has_at_least_one_column_worth_editing(models):
    for name, model in models:
        assert any(column.editable for column in model.columns), name


def test_every_table_reads_every_cell_without_raising(models):
    """A getter that throws would blank the table or crash on scroll."""
    for name, model in models:
        for row in range(min(model.rowCount(), 5)):
            for column in range(model.columnCount()):
                model.data(model.index(row, column))
                model.data(model.index(row, column), EDIT)
                model.flags(model.index(row, column))
        assert model.headerData(0, Qt.Orientation.Horizontal), name


def test_a_bad_value_is_refused_rather_than_written(models, qt_app):
    """Invalid state is refused at entry, with a reason — never written and
    complained about afterwards."""
    for name, model in models:
        if not model.rowCount():
            continue
        reasons = []
        model.edit_rejected.connect(reasons.append)
        for column, spec in enumerate(model.columns):
            if not spec.editable:
                continue
            index = model.index(0, column)
            before = model.data(index)
            if model.setData(index, 'definitely not a valid value', EDIT):
                continue          # a free-text column: anything goes
            assert model.data(index) == before, f'{name}/{spec.title} changed'
            assert reasons, f'{name}/{spec.title} refused without saying why'


def test_every_choice_column_offers_choices_the_delegate_can_use(models):
    for name, model in models:
        for spec in model.columns:
            choices = spec.choices_for(model.obj, 0)
            if spec.choices or spec.row_choices:
                assert choices, f'{name}/{spec.title} offers an empty list'
                assert all(isinstance(c, str) for c in choices)


def test_copy_paste_and_clear_work_on_every_table(models, qt_app):
    """One table view serves them all, so these come from the same place —
    this is the check that they are all actually wired to it."""
    for name, model in models:
        if not model.rowCount():
            continue
        view = CopyPasteTableView()
        view.setModel(model)
        view.selectAll()
        assert view.to_tsv(), f'{name} copied nothing'
        assert view.editable_selection(), f'{name} has no editable cell'
        # Clear either empties a cell or refuses it, but never explodes
        applied, rejected = view.clear_cells()
        assert applied + rejected > 0, name
        view.setModel(None)


def test_the_shared_view_provides_the_conventions_once(qt_app):
    """These are properties of the view, not of any one table, which is why
    every table gets them. (`qt_app` is load-bearing: a widget built
    before any QApplication exists takes the whole worker down.)"""
    view = CopyPasteTableView()
    assert isinstance(view.itemDelegate(), ChoiceDelegate)
    triggers = view.editTriggers()
    assert triggers & CopyPasteTableView.EditTrigger.DoubleClicked
    assert view.selectionMode() == CopyPasteTableView.SelectionMode.\
        ExtendedSelection
    assert view.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    for name in ('copy', 'paste', 'clear_cells', 'batch_edit'):
        assert callable(getattr(view, name)), name


# ---- the fill handle --------------------------------------------------------

def _units_view(qt_app):
    data = visualdynamics.import_file(fixture_path('plate', 'time.npz'))
    model = units_table_model(data)
    view = CopyPasteTableView()
    view.setModel(model)
    view.resize(400, 600)
    view.show()
    qt_app.processEvents()
    return data, model, view


def test_a_selection_offers_a_fill_handle_at_its_corner(qt_app):
    _data, model, view = _units_view(qt_app)
    assert view.fill_handle_rect() is None, 'nothing selected, nothing to drag'
    view.setCurrentIndex(model.index(0, 2))
    view.selectionModel().select(model.index(0, 2),
                                 QItemSelectionModel.SelectionFlag.Select)
    handle = view.fill_handle_rect()
    assert handle is not None
    corner = view.visualRect(model.index(0, 2))
    assert corner.contains(handle), 'inside the cell, at its corner'
    assert handle.bottomRight() == corner.bottomRight()


def test_dragging_the_handle_fills_the_value_down(qt_app):
    data, model, view = _units_view(qt_app)
    model.setData(model.index(0, 2), 'g', EDIT)
    view.selectionModel().select(model.index(0, 2),
                                 QItemSelectionModel.SelectionFlag.Select)
    applied, rejected = view.fill_from_selection(4)
    # eight cells over four rows: the unit and the Type it carries
    # (a unit without its quantity is half a statement)
    assert (applied, rejected) == (8, 0)
    assert data.ordinate_unit[:5] == ['g'] * 5
    assert data.ordinate_unit[5] is None, 'it stopped where the drag stopped'


def test_a_fill_repeats_the_whole_block(qt_app):
    """Two alternating rows carry the alternation, as a spreadsheet does."""
    data, model, view = _units_view(qt_app)
    model.setData(model.index(0, 2), 'g', EDIT)
    model.setData(model.index(1, 2), 'N', EDIT)
    for row in (0, 1):
        view.selectionModel().select(model.index(row, 2),
                                     QItemSelectionModel.SelectionFlag.Select)
    view.fill_from_selection(5)
    assert data.ordinate_unit[:6] == ['g', 'N', 'g', 'N', 'g', 'N']


def test_a_fill_can_go_upward_too(qt_app):
    data, model, view = _units_view(qt_app)
    model.setData(model.index(3, 2), 'g', EDIT)
    view.selectionModel().select(model.index(3, 2),
                                 QItemSelectionModel.SelectionFlag.Select)
    assert view.fill_from_selection(1) == (4, 0), \
        'two rows, each a unit and its carried Type'
    assert data.ordinate_unit[:4] == [None, 'g', 'g', 'g']


def test_the_mouse_drives_the_fill_through_the_handle(qt_app):
    """The gesture itself: press on the handle, drag two rows down,
    release — the fill the semantic tests pin must be reachable by the
    mouse, or the handle is a drawing."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    data, model, view = _units_view(qt_app)
    model.setData(model.index(0, 2), 'g', EDIT)
    view.selectionModel().select(model.index(0, 2),
                                 QItemSelectionModel.SelectionFlag.Select)

    def mouse(kind, at, button=Qt.MouseButton.LeftButton):
        return QMouseEvent(kind, QPointF(at), button, button,
                           Qt.KeyboardModifier.NoModifier)

    grab = view.fill_handle_rect().center()
    view.mousePressEvent(mouse(QMouseEvent.Type.MouseButtonPress, grab))
    assert view._filling is not None, 'the press took the handle'
    below = view.visualRect(model.index(2, 2)).center()
    view.mouseMoveEvent(mouse(QMouseEvent.Type.MouseMove, below))
    assert view._fill_to == 2, 'the drag follows the row under it'
    view.mouseReleaseEvent(mouse(QMouseEvent.Type.MouseButtonRelease,
                                 below))
    assert view._filling is None
    assert data.ordinate_unit[:4] == ['g', 'g', 'g', None], \
        'released on row 2, filled through row 2 and no further'
    # a press anywhere else is an ordinary selection, not a fill
    elsewhere = view.visualRect(model.index(4, 2)).center()
    view.mousePressEvent(mouse(QMouseEvent.Type.MouseButtonPress,
                               elsewhere))
    assert view._filling is None


def test_double_clicking_the_handle_fills_to_the_bottom(qt_app):
    """The common case is 'this unit, for every channel'; dragging 45
    rows to say it is a chore — double-click does it, as Excel does."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QMouseEvent

    data, model, view = _units_view(qt_app)
    model.setData(model.index(0, 2), 'g', EDIT)
    view.selectionModel().select(model.index(0, 2),
                                 QItemSelectionModel.SelectionFlag.Select)
    grab = view.fill_handle_rect().center()
    event = QMouseEvent(QMouseEvent.Type.MouseButtonDblClick,
                        QPointF(grab), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    view.mouseDoubleClickEvent(event)
    assert all(unit == 'g' for unit in data.ordinate_unit), \
        'every channel took the unit'


def test_a_fill_stops_at_the_last_row(qt_app):
    data, model, view = _units_view(qt_app)
    model.setData(model.index(0, 2), 'g', EDIT)
    view.selectionModel().select(model.index(0, 2),
                                 QItemSelectionModel.SelectionFlag.Select)
    applied, _rejected = view.fill_from_selection(10_000)
    assert applied == 2 * (data.num_records - 1), \
        'filled to the end, not past it — a unit and its Type per row'


def test_double_clicking_the_handle_fills_all_the_way_down(qt_app):
    """The common case is 'this value, for every row'."""
    data, model, view = _units_view(qt_app)
    model.setData(model.index(0, 2), 'g', EDIT)
    view.selectionModel().select(model.index(0, 2),
                                 QItemSelectionModel.SelectionFlag.Select)
    handle = view.fill_handle_rect()
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton,
                      Qt.KeyboardModifier.NoModifier, handle.center())
    qt_app.processEvents()
    assert data.units_defined, 'every channel took the unit'
    assert data.ordinate_unit == ['g'] * data.num_records


def test_the_fill_handle_stands_out_from_the_selection(qt_app):
    """It sat in the selection color, which made it invisible against the
    selected cell it sits on the corner of."""
    _data, model, view = _units_view(qt_app)
    view.setCurrentIndex(model.index(1, 2))
    view.selectionModel().select(model.index(1, 2),
                                 QItemSelectionModel.SelectionFlag.Select)
    qt_app.processEvents()
    handle = view.fill_handle_rect()
    painted = view.grab().toImage().pixelColor(handle.center())
    selection = view.palette().highlight().color()
    contrast = view.palette().highlightedText().color()

    def apart(a, b):
        return (abs(a.red() - b.red()) + abs(a.green() - b.green())
                + abs(a.blue() - b.blue()))

    # exact equality would be brittle: grabbing on a scaled display blends
    # the outline into the fill by a few counts
    assert apart(painted, selection) > 200, 'it blends into the selection'
    assert apart(painted, contrast) < 30, 'it is the contrasting palette role'


def test_the_fill_handle_stays_inside_its_cell(qt_app):
    """An overhang lies outside the rect Qt invalidates when the selection
    moves, so it stayed painted on the cell you clicked away from. Keeping
    the square inside the cell is what fixes that — Qt already repaints the
    cell — and it needs no invalidation bookkeeping of its own."""
    _data, model, view = _units_view(qt_app)
    for row in (0, 1, 6):
        index = model.index(row, 2)
        view.setCurrentIndex(index)
        view.selectionModel().select(
            index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        qt_app.processEvents()
        cell = view.visualRect(index)
        handle = view.fill_handle_rect()
        assert cell.contains(handle), f'{handle} spills outside {cell}'
        assert handle.right() == cell.right(), 'tucked into the corner'
        assert handle.bottom() == cell.bottom()


def test_a_checkbox_cell_toggles_and_still_speaks_text(qt_app):
    """The control column draws as a check box: clicking it is the same
    statement as typing the word, and copy still says 'True'/'False' so
    a spreadsheet round trip keeps the value."""
    from visualdynamics.core.channel_table import ChannelTable
    from visualdynamics.gui.object_tables import channel_table_model

    table = ChannelTable({'channel': [1], 'node': ['101'],
                          'direction': ['Z+'], 'unit': ['g'],
                          'role': ['response']})
    model = channel_table_model(table)
    column = table.column_names.index('control')
    index = model.index(0, column)
    check = Qt.ItemDataRole.CheckStateRole
    assert model.data(index, check) == Qt.CheckState.Unchecked
    assert model.data(index, Qt.ItemDataRole.DisplayRole) is None, (
        'the box is the display; text beside it would say it twice')
    assert model.flags(index) & Qt.ItemFlag.ItemIsUserCheckable
    assert model.setData(index, Qt.CheckState.Checked, check)
    assert model.data(index, check) == Qt.CheckState.Checked
    assert bool(table.controls()[0])
    view = CopyPasteTableView()
    view.setModel(model)
    view.selectionModel().select(
        index, QItemSelectionModel.SelectionFlag.Select)
    assert view.to_tsv().strip() == 'True'
    # and the box sets Control whatever the role, as the keyboard does:
    # control and role are independent (Brandon, 2026-09-26 — a test
    # can be controlled to a force or a voltage)
    table.set_cell('control', 0, 'False')
    table.set_cell('role', 0, 'monitor')
    assert model.setData(index, Qt.CheckState.Checked, check)
    assert bool(table.controls()[0]) and table.roles()[0] == 'monitor'


def test_a_filled_unit_carries_its_type_down(qt_app):
    """Brandon's session (2026-08-30): reference channels typed as
    force in lbf, the first channel changed to voltage in Volts, the
    unit dragged down — and the rest came out force-in-Volts, a unit
    without its quantity. The unit column now carries the Type column
    with it when it fills, so the pair travels as one statement."""
    data, model, view = _units_view(qt_app)
    for row in range(5):
        model.setData(model.index(row, 1), 'force', EDIT)
        model.setData(model.index(row, 2), 'lbf', EDIT)
    model.setData(model.index(0, 1), 'voltage', EDIT)
    model.setData(model.index(0, 2), 'V', EDIT)
    view.selectionModel().select(model.index(0, 2),
                                 QItemSelectionModel.SelectionFlag.Select)
    view.fill_from_selection(4)
    assert data.ordinate_unit[:5] == ['V'] * 5
    assert [model.data(model.index(r, 1)) for r in range(5)] == \
        ['voltage'] * 5, 'the type traveled with its unit'


def test_a_chosen_unit_stamps_its_type_on_the_row(qt_app):
    """A unit may be set with no type picked — but once it is, there
    IS a type, and the row must say so rather than reading empty
    (Brandon, 2026-08-30). And a unit of another kind moves the type
    with it, the fill-down rule applied to a single cell."""
    import numpy as np

    t = np.linspace(0, 1, 8)
    history = visualdynamics.TimeHistory(
        t, np.ones((2, 8)), response_dof=['1X+', '9001X+'])
    model = units_table_model(history)
    assert model.data(model.index(0, 1)) == '', 'no type declared yet'
    model.setData(model.index(0, 2), 'lbf', EDIT)
    assert model.data(model.index(0, 1)) == 'force', \
        'the unit said what the channel is'
    model.setData(model.index(0, 2), 'V', EDIT)
    assert model.data(model.index(0, 1)) == 'voltage', \
        'a unit of another kind moves the type with it'


def test_every_editable_column_journals(models):
    """Brandon's rule (2026-08-30), held structurally: an edit the
    interface can make is an act the console must speak. A column
    journaled by another surface (the units pane) declares itself so;
    a new editable column cannot quietly skip the journal."""
    for name, model in models:
        for spec in model.columns:
            if spec.editable:
                assert spec.journal is not None, (
                    f'{name} / {spec.title} edits without journaling')


def test_every_editable_cell_actually_journals(models):
    """The stronger half of the rule (Brandon, 2026-08-30, doubting
    the sweep — rightly): declaring a journal hook is not producing a
    line. Every editable cell of every model is edited with its own
    current value, and either a replay suffix comes out or the column
    is explicitly declared journaled elsewhere. A hook that returns
    nothing is a silent edit, and this refuses it."""
    from visualdynamics.gui.object_tables import JOURNALED_ELSEWHERE

    for name, model in models:
        if not model.rowCount():
            continue
        heard = []
        model.edit_journaled.connect(heard.append)
        for column, spec in enumerate(model.columns):
            if not spec.editable:
                continue
            if getattr(spec.journal, 'tag', None) == JOURNALED_ELSEWHERE:
                continue
            index = model.index(0, column)
            current = model.data(index, EDIT)
            heard.clear()
            if not model.setData(index, str(current), EDIT):
                continue        # refused its own value: spoken, not silent
            assert heard and heard[0], (
                f'{name} / {spec.title} edited without a journal line')


# ---- a property shows only where it applies (principle 3) -------------------

#: the Element Groups table's rows whose meaning depends on the group's
#: elements, and the groups below each one applies to
APPLIES = {
    'Thickness [m]': {'plate'},
    'Section': {'beam'}, 'Shape': {'beam'}, 'Dimensions [m]': {'beam'},
    'A [m²]': {'beam'}, 'Iy [m⁴]': {'beam'}, 'Iz [m⁴]': {'beam'},
    'J [m⁴]': {'beam'}, 'Orientation': {'beam'},
    'Mass [kg]': {'point'}, 'Ground': {'point'},
    'Stiffness [N/m, N·m/rad]': {'point', 'beam'},
}
#: the groups' columns in the table, in the order `_four_groups` makes them
GROUPS = ('beam', 'plate', 'solid', 'point')


def _four_groups():
    """A group of beams, one of quads, one of bricks and one of points,
    in that order, with nothing set."""
    from test_springs import _whole_beam

    geometry = _whole_beam().geometry()
    geometry.group_properties = {}
    for name, nodes, code in (('plate', [1, 2, 3, 4], 44),
                              ('solid', list(range(1, 9)), 115),
                              ('point', [5], 161)):
        group = geometry.add_group(name)
        geometry.add_elements([nodes], [code], [group])
    return geometry, element_group_table_model(geometry)


def _row(model, title):
    titles = [model.headerData(r, Qt.Orientation.Vertical)
              for r in range(model.rowCount())]
    return titles.index(title)


def _editable(model, index):
    return bool(model.flags(index) & Qt.ItemFlag.ItemIsEditable)


def test_a_group_property_shows_only_for_the_groups_it_applies_to(qt_app):
    """Each row is live for the groups whose elements use it and blank,
    read-only and explained on hover for the rest: no thickness on
    beams, no section on plates, no mass on bricks."""
    geometry, model = _four_groups()
    assert model.columnCount() == len(GROUPS)
    for title, takers in APPLIES.items():
        for column, group in enumerate(GROUPS):
            index = model.index(_row(model, title), column)
            if group in takers:
                assert _editable(model, index), (title, group)
                assert not model.data(index, Qt.ItemDataRole.ToolTipRole), (title, group)
            else:
                assert not _editable(model, index), (title, group)
                assert model.data(index) == '', (title, group)
                assert model.data(index, Qt.ItemDataRole.ToolTipRole), (title, group)
    # the material is every group's
    for column in range(len(GROUPS)):
        assert _editable(model, model.index(_row(model, 'Material'), column))
    # and a cell that does not apply refuses a value typed in, saying why
    said = []
    model.edit_rejected.connect(said.append)
    thickness = model.index(_row(model, 'Thickness [m]'), GROUPS.index('beam'))
    assert not model.setData(thickness, '0.003', EDIT)
    assert said[-1] == 'a group of beams takes no thickness; triangles and quads do'
    section = model.index(_row(model, 'Shape'), GROUPS.index('solid'))
    assert not model.setData(section, 'rod', EDIT)
    assert 'takes a material alone' in said[-1]
    assert geometry.group_properties == {}


def test_moving_elements_into_a_group_changes_what_applies(qt_app):
    """The rule follows the elements: an empty group takes anything, and
    once the quad is moved into it, a section no longer applies."""
    geometry, model = _four_groups()
    model.beginResetModel()
    spare = geometry.add_group('spare')
    model.endResetModel()
    column = len(GROUPS)
    section = model.index(_row(model, 'Section'), column)
    assert _editable(model, section), 'an empty group takes anything'
    quad = int(geometry.elem_id[geometry.elem_group == int(geometry.group_id[1])][0])
    assert model.setData(model.index(_row(model, 'Elements'), column), str(quad), EDIT)
    assert int(geometry.elem_group[geometry.elem_id == quad][0]) == spare
    assert not _editable(model, section)
    assert _editable(model, model.index(_row(model, 'Thickness [m]'), column))


def test_elements_added_to_a_group_change_what_applies(qt_app):
    """Elements added by script replace the element arrays; the table's
    record of each group's families follows them."""
    geometry, model = _four_groups()
    model.beginResetModel()
    spare = geometry.add_group('spare')
    model.endResetModel()
    thickness = model.index(_row(model, 'Thickness [m]'), len(GROUPS))
    assert _editable(model, thickness), 'an empty group takes anything'
    geometry.add_elements([[1, 2]], [21], [spare])
    assert not _editable(model, thickness), 'a group of beams takes no thickness'


def _view(model):
    view = CopyPasteTableView()
    view.setModel(model)
    heard = []
    view.edits_applied.connect(lambda *said: heard.append(said))
    return view, heard


def _select(view, rows, columns):
    selection = view.selectionModel()
    model = view.model()
    selection.clearSelection()
    for row in rows:
        for column in columns:
            selection.select(model.index(row, column),
                             QItemSelectionModel.SelectionFlag.Select)


def test_a_paste_across_groups_lands_only_where_it_applies(qt_app):
    from PySide6.QtWidgets import QApplication

    geometry, model = _four_groups()
    view, heard = _view(model)
    row = _row(model, 'Thickness [m]')
    _select(view, [row], [0])
    QApplication.clipboard().setText('0.004\t0.004\t0.004\t0.004')
    assert view.paste() == (1, 3)
    assert heard[-1] == (1, 3, ('a group of beams takes no thickness; '
                                'triangles and quads do'))
    held = {name: geometry.group_properties.get(int(geometry.group_id[k]))
            for k, name in enumerate(GROUPS)}
    assert held['plate'].thickness == 0.004
    assert held['beam'] is None and held['solid'] is None and held['point'] is None


def test_a_batch_edit_across_groups_skips_and_says_so(qt_app):
    geometry, model = _four_groups()
    view, heard = _view(model)
    _select(view, [_row(model, 'Thickness [m]')], range(len(GROUPS)))
    assert view.batch_edit('0.002') == (1, 3)
    applied, rejected, reason = heard[-1]
    assert (applied, rejected) == (1, 3) and 'takes no thickness' in reason
    plate = int(geometry.group_id[GROUPS.index('plate')])
    assert set(geometry.group_properties) == {plate}
    assert geometry.group_properties[plate].thickness == 0.002


def test_a_fill_across_groups_carries_only_what_applies(qt_app):
    """Filled from the beam across: the beam's section goes nowhere else,
    and its blank thickness, blank because a beam has none, does not
    clear the plate's."""
    from test_fem import ALUMINUM

    from visualdynamics.core.fem import GroupProperties

    geometry, model = _four_groups()
    beam, plate = (int(geometry.group_id[GROUPS.index(name)]) for name in ('beam', 'plate'))
    geometry.group_properties[plate] = GroupProperties(ALUMINUM, thickness=0.003)
    model.setData(model.index(_row(model, 'Shape'), 0), 'rod', EDIT)
    model.setData(model.index(_row(model, 'Dimensions [m]'), 0), '0.01', EDIT)
    beam_props = geometry.group_properties[beam]
    view, heard = _view(model)
    _select(view, range(_row(model, 'Thickness [m]'), _row(model, 'Orientation') + 1), [0])
    applied, rejected = view.fill_from_selection(len(GROUPS) - 1)
    assert applied == 0 and rejected > 0
    assert heard[-1][2] == 'a group of quads takes no section; beams do'
    assert geometry.group_properties[plate] == GroupProperties(ALUMINUM, thickness=0.003)
    assert geometry.group_properties[beam] == beam_props
    assert set(geometry.group_properties) == {beam, plate}
