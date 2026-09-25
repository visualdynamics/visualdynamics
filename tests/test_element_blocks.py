"""Element blocks: which region of a mesh each element belongs to.

Exodus has always had them — a mesh is written as named blocks, and the
block is where the part identity lives — and until this they were read
and thrown away, so a file that went `wing`/`tail` in came back one
undifferentiated soup. That is a lossy round trip in a package whose rule
is that anything it reads, it writes.

They earn their place beyond the round trip: `fem.Model.from_geometry`
reads the section for a member off the block of the element it came from,
so a geometry is a complete description of a structure and a saved file
rebuilds it with nothing passed alongside. `tests/test_demo_drone.py`
pins that end of it.
"""

from __future__ import annotations

import pytest

from visualdynamics.core.geometry import Geometry
from visualdynamics.io import export_file, import_file, load, save


@pytest.fixture
def two_blocks():
    """Four quads over six nodes, split into a named pair of blocks."""
    return Geometry(
        node_id=[1, 2, 3, 4, 5, 6],
        node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0],
                  [0, 1, 0], [1, 1, 0], [2, 1, 0]],
        elem_id=[10, 11],
        elem_conn=[[1, 2, 5, 4], [2, 3, 6, 5]],
        elem_type=[44, 44],
        elem_block=[7, 9],
        block_id=[7, 9],
        block_name=['wing', 'tail'],
        length_unit='m')


def test_an_element_says_which_block_it_is_in(two_blocks):
    assert two_blocks.block_of(10) == 'wing'
    assert two_blocks.block_of(11) == 'tail'
    assert two_blocks.elements_in('wing') == [10]
    assert two_blocks.elements_in(9) == [11], 'by id as well as by name'


def test_a_mesh_with_no_blocks_is_one_block(two_blocks):
    """Most formats do not record the question, and nothing may invent an
    answer: everything lands in one block with no name."""
    plain = Geometry(node_id=[1, 2, 3],
                     node_xyz=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                     elem_conn=[[1, 2, 3]], elem_type=[41])
    assert list(plain.block_id) == [1]
    assert plain.block_name == ['']
    assert plain.elements_in(1) == list(plain.elem_id)


def test_a_block_no_element_belongs_to_is_refused(two_blocks):
    """An element in block 3 with no block 3 declared is a file saying
    two things that cannot both be true, and it is refused where it is
    written rather than found later by whatever reads it."""
    with pytest.raises(ValueError):
        Geometry(node_id=[1, 2, 3],
                 node_xyz=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                 elem_conn=[[1, 2, 3]], elem_type=[41],
                 elem_block=[3], block_id=[1], block_name=[''])


def test_a_new_element_joins_a_block(two_blocks):
    two_blocks.add_element([1, 2, 4], block=9)
    assert two_blocks.block_of(two_blocks.elem_id[-1]) == 'tail'
    two_blocks.add_element([2, 3, 5], block=12)
    assert 12 in two_blocks.block_id.tolist(), 'a new block is declared'
    assert two_blocks.block_of(two_blocks.elem_id[-1]) == ''


def test_deleting_an_element_takes_its_block_entry_with_it(two_blocks):
    two_blocks.delete_elements([10])
    assert len(two_blocks.elem_block) == len(two_blocks.elem_id) == 1
    assert two_blocks.block_of(11) == 'tail'


def test_blocks_ride_the_native_file(two_blocks, tmp_path):
    path = tmp_path / 'blocks.vdyn'
    save(two_blocks, path)
    back = load(path)
    assert list(back.elem_block) == [7, 9]
    assert list(back.block_id) == [7, 9]
    assert back.block_name == ['wing', 'tail']
    assert back == two_blocks


def test_blocks_survive_an_exodus_round_trip(two_blocks, tmp_path):
    """The format they come from, and the reason for the whole feature."""
    path = tmp_path / 'blocks.exo'
    export_file(two_blocks, path)
    back = import_file(path)
    assert back.block_name == ['wing', 'tail']
    assert list(back.block_id) == [7, 9]
    assert back.block_of(back.elem_id[0]) == 'wing'
    assert back.block_of(back.elem_id[1]) == 'tail'


# ---- blocks as a group you can read and edit --------------------------------

def test_blocks_are_a_view_like_the_other_four(two_blocks):
    """Same shape as `nodes` and friends: plural on the view, singular on
    a row, and a row written through writes the geometry."""
    blocks = two_blocks.blocks
    assert len(blocks) == 2
    assert list(blocks.ids) == [7, 9]
    assert blocks.names == ['wing', 'tail']
    assert blocks[1].id == 9 and blocks[1].name == 'tail'
    blocks[1].name = 'empennage'
    assert two_blocks.block_name == ['wing', 'empennage'], 'not a copy'


def test_a_block_can_be_added_empty(two_blocks):
    """It has to exist before an element can be put in it, and exodus
    files carry empty ones anyway."""
    block_id = two_blocks.blocks.add('fin')
    assert block_id == 10 and two_blocks.elements_in(10) == []
    two_blocks.validate()
    with pytest.raises(ValueError, match='already exists'):
        two_blocks.add_block('again', block_id=10)


def test_renumbering_a_block_carries_its_elements(two_blocks):
    """An element names its block by id, so leaving them behind would put
    them in a block that is not there — which `validate` refuses."""
    two_blocks.renumber_block(0, 4)
    assert list(two_blocks.elem_block) == [4, 9]
    two_blocks.validate()
    with pytest.raises(ValueError, match='already exists'):
        two_blocks.renumber_block(0, 9)


def test_deleting_a_block_moves_its_elements_rather_than_losing_them(
        two_blocks):
    """The block is a label on the elements; deleting the label must not
    delete the mesh under it."""
    report = two_blocks.delete_blocks([7])
    assert report == {'blocks': 1, 'elements_reassigned': 1}
    assert list(two_blocks.block_id) == [9]
    assert two_blocks.elements_in(9) == [10, 11], 'both, in the one left'
    two_blocks.validate()


def test_the_last_block_cannot_go_while_elements_name_one(two_blocks):
    with pytest.raises(ValueError, match='needs a block'):
        two_blocks.delete_blocks([7, 9])
    two_blocks.delete_elements([10, 11])
    two_blocks.delete_blocks([7, 9])
    assert list(two_blocks.block_id) == [], 'nothing to hold, so none needed'


@pytest.fixture
def mixed_block():
    """One named block holding a quad and a triangle — what the drone's
    'canopy' is, and what exodus cannot write as one block."""
    return Geometry(
        node_id=[1, 2, 3, 4, 5],
        node_xyz=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0.5, 0]],
        elem_id=[10, 11],
        elem_conn=[[1, 2, 3, 4], [2, 5, 3]],
        elem_type=[44, 41],
        elem_block=[7, 7],
        block_id=[7],
        block_name=['canopy'],
        length_unit='m')


def test_a_block_of_two_element_types_splits_into_distinct_blocks(
        mixed_block, tmp_path):
    """Exodus holds one element type per block, so a block with a quad and
    a tri in it goes out as two — and they may share neither the id nor
    the name it declared. The drone went out as 34 blocks under 22 ids and
    read back nowhere: this package refuses the duplicate ids outright,
    and ParaView's IOSS reader fails the file at REQUEST_INFORMATION
    without saying why."""
    import netCDF4

    path = tmp_path / 'mixed.exo'
    export_file(mixed_block, path)
    with netCDF4.Dataset(path) as ds:
        ids = [int(i) for i in ds.variables['eb_prop1'][:]]
    assert len(set(ids)) == len(ids) == 2, 'two blocks, two ids'
    assert 7 in ids, 'the declared id is kept where it can be'

    back = import_file(path)
    assert back.block_name == ['canopy', 'canopy TRI3'], (
        'the second piece says which type it is, since names must differ')
    assert len(back.elem_id) == 2
    assert back.block_of(back.elem_id[0]) == 'canopy'


def test_paraviews_reader_takes_a_split_block(mixed_block, tmp_path):
    """The reader ParaView actually uses. vtkExodusIIReader is more
    forgiving than IOSS and read the duplicate-id file happily, so only
    this one would have caught it."""
    from vtkmodules.vtkIOIOSS import vtkIOSSReader

    path = tmp_path / 'mixed.exo'
    export_file(mixed_block, path)
    reader = vtkIOSSReader()
    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutputDataObject(0)
    walk = output.NewIterator()
    walk.InitTraversal()
    cells = 0
    while not walk.IsDoneWithTraversal():
        cells += walk.GetCurrentDataObject().GetNumberOfCells()
        walk.GoToNextItem()
    assert cells == 2, 'the quad and the tri, both read'


# ---- blocks in the app ------------------------------------------------------

def _show_geometry(window, pump, geometry, name='Geometry'):
    window.add_object(name, geometry)
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    item.setExpanded(True)
    pump()
    return item


def _category(item, label):
    return next(item.child(i) for i in range(item.childCount())
                if item.child(i).text(0).startswith(label))


def test_the_tree_lists_blocks_as_a_geometry_category(two_blocks, window, pump):
    """A geometry says what is in it, and since the mesh is divided into
    parts, that is one of the things in it."""
    item = _show_geometry(window, pump, two_blocks)
    labels = [item.child(i).text(0) for i in range(item.childCount())]
    assert labels[-1] == 'Blocks (2)', 'after the elements it groups'
    blocks = _category(item, 'Blocks')
    assert not blocks.icon(1).isNull(), 'the edit pencil, like every category'
    blocks.setExpanded(True)
    window._populate_entities(blocks)
    pump()
    assert [blocks.child(i).text(0) for i in range(blocks.childCount())] == [
        'Block 7 — wing (1 elements)', 'Block 9 — tail (1 elements)']


def test_editing_blocks_opens_a_table_of_them(two_blocks, window, pump):
    from PySide6.QtCore import Qt

    item = _show_geometry(window, pump, two_blocks)
    window.tree.setCurrentItem(_category(item, 'Blocks'))
    window.edit_entities()
    pump()
    assert window.editing == ('Geometry', 'blocks')
    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert headers[:3] == ['Block', 'Name', 'Elements']
    assert headers[3] == 'Material', (
        'then what the block is made of — tests/test_solve_modes.py')
    assert model.rowCount() == 2
    assert model.data(model.index(0, 1)) == 'wing'
    # the elements it holds, by id and as runs — not a count, which is
    # not a thing anyone can edit. 'wing' holds element 10.
    assert model.data(model.index(0, 2)) == '10'
    assert model.data(model.index(1, 2)) == '11', 'and tail holds 11'
    assert model.flags(model.index(0, 2)) & Qt.ItemFlag.ItemIsEditable, (
        'and the list is editable: naming an element claims it for '
        'this block, the way a node is typed into a traceline')
    model.setData(model.index(0, 1), 'port wing')
    assert two_blocks.block_name[0] == 'port wing'


def test_the_plus_adds_an_empty_block_outright(two_blocks, window, pump):
    """There is nothing to click in the view for a block, so the button
    cannot arm a mode — it adds one and comes straight back up."""
    item = _show_geometry(window, pump, two_blocks)
    window.tree.setCurrentItem(_category(item, 'Blocks'))
    window.edit_entities()
    pump()
    assert window.add_action.isVisible()
    window.add_action.trigger()
    pump()
    assert not window.add_action.isChecked(), 'no add mode to be in'
    assert not window.add_mode
    assert list(two_blocks.block_id) == [7, 9, 10]
    assert window.table.model().rowCount() == 3
    assert 'Added block 10' in window.statusBar().currentMessage()


def test_deleting_a_block_row_keeps_its_elements(two_blocks, window, pump):
    item = _show_geometry(window, pump, two_blocks)
    window.tree.setCurrentItem(_category(item, 'Blocks'))
    window.edit_entities()
    pump()
    window.table.selectRow(0)
    window.delete_entity_rows()
    pump()
    assert list(two_blocks.block_id) == [9]
    assert len(two_blocks.elem_id) == 2, 'the mesh is not the label on it'
    assert 'elements reassigned' in window.statusBar().currentMessage()


def test_picking_a_block_highlights_the_elements_it_holds(two_blocks, window,
                                                          pump):
    """A block has no geometry of its own, so what it draws is its
    elements — an empty one correctly lights nothing up."""
    from visualdynamics.gui.main_window import _drawable

    components, entities = _drawable(two_blocks, None, {'blocks': [9]})
    assert entities == {'elements': [1]}, 'the tail element, by row'
    assert components is None
    components, _entities = _drawable(two_blocks, {'blocks'}, {})
    assert components == {'elements'}, 'the category is every element'
    two_blocks.add_block('fin')
    _components, entities = _drawable(two_blocks, None, {'blocks': [10]})
    assert entities == {'elements': []}, 'an empty block is not the whole model'


def test_an_element_moves_between_blocks_from_its_own_table(two_blocks, window,
                                                            pump):
    """A block holds nothing itself, so this is where the grouping is
    actually edited — and a block that does not exist is refused rather
    than quietly declared."""
    from PySide6.QtCore import Qt

    item = _show_geometry(window, pump, two_blocks)
    window.tree.setCurrentItem(_category(item, 'Elements'))
    window.edit_entities()
    pump()
    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert 'Block' in headers
    column = headers.index('Block')
    assert model.data(model.index(0, column)) == 'wing'
    model.setData(model.index(0, column), 'tail')
    assert two_blocks.elements_in('tail') == [10, 11]
    assert model.setData(model.index(0, column), 'nose') is False
    assert 'no block' in window.statusBar().currentMessage()


# ---- the blocks table's Elements column is editable ----------------------


def _blocks_model(geometry):
    from visualdynamics.gui.object_tables import block_table_model

    return block_table_model(geometry)


def _cell(model, row, header):
    from PySide6.QtCore import Qt

    columns = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    index = model.index(row, columns.index(header))
    return model.data(index, Qt.ItemDataRole.DisplayRole), index


def _refusal(model):
    """A refused edit says why on `edit_rejected` — the cell reverts,
    and the window puts the reason in the status bar."""
    said = []
    model.edit_rejected.connect(said.append)
    return said


def test_runs_read_and_parse_back(qt_app):
    from visualdynamics.gui.object_tables import id_runs_text, parse_id_runs

    assert id_runs_text([1, 2, 3, 7, 9, 10]) == '1-3 7 9-10'
    assert id_runs_text([]) == ''
    assert parse_id_runs('1-3, 7 9-10') == [1, 2, 3, 7, 9, 10]
    # a range typed backwards means the range
    assert parse_id_runs('10-7') == [7, 8, 9, 10]


def test_the_blocks_table_lists_its_elements_as_runs(qt_app, two_blocks):
    """It used to show a count, which is not a thing anyone can edit."""
    model = _blocks_model(two_blocks)
    text, _index = _cell(model, 0, 'Elements')
    first = int(two_blocks.block_id[0])
    expected = sorted(int(i) for i in
                      two_blocks.elem_id[two_blocks.elem_block == first])
    from visualdynamics.gui.object_tables import parse_id_runs
    assert parse_id_runs(text) == expected


def test_naming_an_element_claims_it_for_this_block(qt_app, two_blocks):
    """The same gesture as typing a node into a traceline: the element
    moves in, and leaves whatever block it was in by itself."""
    from PySide6.QtCore import Qt

    from visualdynamics.gui.object_tables import parse_id_runs

    model = _blocks_model(two_blocks)
    first, second = (int(b) for b in two_blocks.block_id[:2])
    moving = int(two_blocks.elem_id[two_blocks.elem_block == second][0])
    text, index = _cell(model, 0, 'Elements')
    assert model.setData(index, f'{text} {moving}',
                         Qt.ItemDataRole.EditRole)
    assert int(two_blocks.elem_block[
        two_blocks.elem_id == moving][0]) == first
    # and it is gone from the block it came from
    after, _ = _cell(model, 1, 'Elements')
    assert moving not in parse_id_runs(after)


def test_dropping_an_element_is_refused_with_the_way_to_do_it(qt_app,
                                                              two_blocks):
    """A node can be in no traceline; an element is always in exactly
    one block, so a removal with no destination is not a state the
    geometry can hold. The refusal says how to move it instead."""
    from PySide6.QtCore import Qt

    model = _blocks_model(two_blocks)
    said = _refusal(model)
    _text, index = _cell(model, 0, 'Elements')
    assert not model.setData(index, '', Qt.ItemDataRole.EditRole)
    assert said and 'no block' in said[0]
    assert 'Add it to the block it belongs in' in said[0]
    assert list(two_blocks.elem_block) == [7, 9], 'and nothing moved'


def test_an_unknown_element_is_refused(qt_app, two_blocks):
    from PySide6.QtCore import Qt

    model = _blocks_model(two_blocks)
    said = _refusal(model)
    text, index = _cell(model, 0, 'Elements')
    assert not model.setData(index, f'{text} 9999',
                             Qt.ItemDataRole.EditRole)
    assert said and 'unknown elements [9999]' in said[0]
