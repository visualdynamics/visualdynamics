"""Element groups: which region of a mesh each element belongs to.

Exodus has always had them — a mesh is written as named element groups, and the
element group is where the part identity lives — and until this they were read
and thrown away, so a file that went `wing`/`tail` in came back one
undifferentiated soup. That is a lossy round trip in a package whose rule
is that anything it reads, it writes.

They earn their place beyond the round trip: `fem.Model.from_geometry`
reads the section for a member off the element group of the element it came from,
so a geometry is a complete description of a structure and a saved file
rebuilds it with nothing passed alongside. `tests/test_demo_drone.py`
pins that end of it.
"""

from __future__ import annotations

import pytest

from visualdynamics.core.geometry import Geometry
from visualdynamics.io import export_file, import_file, load, save

try:
    from conftest import group_row
except ImportError:  # pragma: no cover - collected outside tests/
    group_row = None


@pytest.fixture
def two_blocks():
    """Four quads over six nodes, split into a named pair of element groups."""
    return Geometry(
        node_id=[1, 2, 3, 4, 5, 6],
        node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0],
                  [0, 1, 0], [1, 1, 0], [2, 1, 0]],
        elem_id=[10, 11],
        elem_conn=[[1, 2, 5, 4], [2, 3, 6, 5]],
        elem_type=[44, 44],
        elem_group=[7, 9],
        group_id=[7, 9],
        group_name=['wing', 'tail'],
        length_unit='m')


def test_an_element_says_which_block_it_is_in(two_blocks):
    assert two_blocks.group_of(10) == 'wing'
    assert two_blocks.group_of(11) == 'tail'
    assert two_blocks.elements_in('wing') == [10]
    assert two_blocks.elements_in(9) == [11], 'by id as well as by name'


def test_a_mesh_with_no_blocks_is_one_block(two_blocks):
    """Most formats do not record the question, and nothing may invent an
    answer: everything lands in one element group with no name."""
    plain = Geometry(node_id=[1, 2, 3],
                     node_xyz=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                     elem_conn=[[1, 2, 3]], elem_type=[41])
    assert list(plain.group_id) == [1]
    assert plain.group_name == ['']
    assert plain.elements_in(1) == list(plain.elem_id)


def test_a_block_no_element_belongs_to_is_refused(two_blocks):
    """An element in element group 3 with no element group 3 declared is a file saying
    two things that cannot both be true, and it is refused where it is
    written rather than found later by whatever reads it."""
    with pytest.raises(ValueError, match='element groups the geometry does not have'):
        Geometry(node_id=[1, 2, 3],
                 node_xyz=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                 elem_conn=[[1, 2, 3]], elem_type=[41],
                 elem_group=[3], group_id=[1], group_name=[''])


def test_a_new_element_joins_a_block(two_blocks):
    two_blocks.add_element([1, 2, 5, 4], group=9)
    assert two_blocks.group_of(two_blocks.elem_id[-1]) == 'tail'
    two_blocks.add_element([2, 3, 5], group=12)
    assert 12 in two_blocks.group_id.tolist(), 'a new element group is declared'
    assert two_blocks.group_of(two_blocks.elem_id[-1]) == ''
    # one element group, one family (2026-09-30): a triangle is refused an element group
    # of quads by name, and with no element group named finds its own family
    import pytest

    with pytest.raises(ValueError, match='element group 9 holds quads'):
        two_blocks.add_element([1, 2, 4], group=9)
    two_blocks.add_element([1, 2, 4])
    assert int(two_blocks.elem_group[-1]) == 12, 'the triangles already there'
    two_blocks.add_element([1, 2])
    assert int(two_blocks.elem_group[-1]) not in (7, 9, 12), 'a beam element group of its own'


def test_deleting_an_element_takes_its_block_entry_with_it(two_blocks):
    two_blocks.delete_elements([10])
    assert len(two_blocks.elem_group) == len(two_blocks.elem_id) == 1
    assert two_blocks.group_of(11) == 'tail'


def test_blocks_ride_the_native_file(two_blocks, tmp_path):
    path = tmp_path / 'element groups.vdyn'
    save(two_blocks, path)
    back = load(path)
    assert list(back.elem_group) == [7, 9]
    assert list(back.group_id) == [7, 9]
    assert back.group_name == ['wing', 'tail']
    assert back == two_blocks


def test_blocks_survive_an_exodus_round_trip(two_blocks, tmp_path):
    """The format they come from, and the reason for the whole feature."""
    path = tmp_path / 'element groups.exo'
    export_file(two_blocks, path)
    back = import_file(path)
    assert back.group_name == ['wing', 'tail']
    assert list(back.group_id) == [7, 9]
    assert back.group_of(back.elem_id[0]) == 'wing'
    assert back.group_of(back.elem_id[1]) == 'tail'


# ---- element groups as a group you can read and edit --------------------------------

def test_blocks_are_a_view_like_the_other_four(two_blocks):
    """Same shape as `nodes` and friends: plural on the view, singular on
    a row, and a row written through writes the geometry."""
    groups = two_blocks.groups
    assert len(groups) == 2
    assert list(groups.ids) == [7, 9]
    assert groups.names == ['wing', 'tail']
    assert groups[1].id == 9 and groups[1].name == 'tail'
    groups[1].name = 'empennage'
    assert two_blocks.group_name == ['wing', 'empennage'], 'not a copy'


def test_a_block_can_be_added_empty(two_blocks):
    """It has to exist before an element can be put in it, and exodus
    files carry empty ones anyway."""
    group_id = two_blocks.groups.add('fin')
    assert group_id == 10 and two_blocks.elements_in(10) == []
    two_blocks.validate()
    with pytest.raises(ValueError, match='already exists'):
        two_blocks.add_group('again', group_id=10)


def test_renumbering_a_block_carries_its_elements(two_blocks):
    """An element names its element group by id, so leaving them behind would put
    them in an element group that is not there — which `validate` refuses."""
    two_blocks.renumber_group(0, 4)
    assert list(two_blocks.elem_group) == [4, 9]
    two_blocks.validate()
    with pytest.raises(ValueError, match='already exists'):
        two_blocks.renumber_group(0, 9)


def test_deleting_a_block_deletes_its_elements_and_its_own_nodes(two_blocks):
    """An element group deleted is the part deleted (Brandon, 2026-09-27): its
    elements, and the nodes no other element group's element uses. The nodes it
    shares with the tail stay, so the tail is not cut into."""
    report = two_blocks.delete_groups([7])
    assert report == {'groups': 1, 'elements': 1, 'nodes': 2}
    assert list(two_blocks.group_id) == [9]
    assert two_blocks.elements_in(9) == [11], 'the tail, whole'
    assert list(two_blocks.node_id) == [2, 3, 5, 6], 'the shared edge kept'
    two_blocks.validate()


def test_deleting_every_block_leaves_no_mesh(two_blocks):
    two_blocks.delete_groups([7, 9])
    assert list(two_blocks.group_id) == [] and two_blocks.elem_conn == []
    assert two_blocks.num_nodes == 0


def test_blocks_of_one_kind_merge_into_the_first(two_blocks):
    """Merge Element Groups: the first keeps its id and name, the rest's elements
    move into it."""
    report = two_blocks.merge_groups([9, 7])
    assert report == {'into': 9, 'groups': 1, 'elements': 1}
    assert list(two_blocks.group_id) == [9]
    assert two_blocks.group_name == ['tail']
    assert two_blocks.elements_in(9) == [10, 11]
    two_blocks.validate()


def test_blocks_that_differ_are_not_merged(two_blocks, mixed_block):
    """Different element types, or different properties, are different
    parts: one material and thickness for both would change the model."""
    from visualdynamics import fem

    aluminum = fem.material('6061-T6')
    two_blocks.group_properties = {7: fem.GroupProperties(aluminum, 0.01),
                                   9: fem.GroupProperties(aluminum, 0.02)}
    assert 'thickness' in two_blocks.merge_refusal([7, 9])
    with pytest.raises(ValueError, match='differ in material'):
        two_blocks.merge_groups([7, 9])
    two_blocks.group_properties[9] = fem.GroupProperties(aluminum, 0.01)
    assert two_blocks.merge_refusal([7, 9]) is None
    tri = two_blocks.add_group('fin')
    two_blocks.add_element([3, 6, 5], group=tri)
    assert two_blocks.merge_refusal([7, tri]) == \
        'the element groups hold different element types'
    assert two_blocks.merge_refusal([7]) == 'select two element groups or more to merge'


@pytest.fixture
def mixed_block():
    """One named element group holding a quad and a triangle — what the drone's
    'canopy' is, and what exodus cannot write as one element group."""
    return Geometry(
        node_id=[1, 2, 3, 4, 5],
        node_xyz=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0.5, 0]],
        elem_id=[10, 11],
        elem_conn=[[1, 2, 3, 4], [2, 5, 3]],
        elem_type=[44, 41],
        elem_group=[7, 7],
        group_id=[7],
        group_name=['canopy'],
        length_unit='m')


def test_a_block_of_two_element_types_splits_into_distinct_blocks(
        mixed_block, tmp_path):
    """Exodus holds one element type per element group, so an element group with a quad and
    a tri in it goes out as two — and they may share neither the id nor
    the name it declared. The drone went out as 34 element groups under 22 ids and
    read back nowhere: this package refuses the duplicate ids outright,
    and ParaView's IOSS reader fails the file at REQUEST_INFORMATION
    without saying why."""
    import netCDF4

    path = tmp_path / 'mixed.exo'
    export_file(mixed_block, path)
    with netCDF4.Dataset(path) as ds:
        ids = [int(i) for i in ds.variables['eb_prop1'][:]]
    assert len(set(ids)) == len(ids) == 2, 'two element groups, two ids'
    assert 7 in ids, 'the declared id is kept where it can be'

    back = import_file(path)
    assert back.group_name == ['canopy', 'canopy triangles'], (
        'the second piece says which type it is, since names must differ')
    assert len(back.elem_id) == 2
    assert back.group_of(back.elem_id[0]) == 'canopy'


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


# ---- element groups in the app ------------------------------------------------------

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


def test_the_tree_lists_blocks_under_their_family(two_blocks, window, pump):
    """A geometry lists its nodes, its coordinate systems and then one
    row per element family, the six always; under a family, one row
    per element group holding elements of it, and under an element group its elements
    (Brandon, 2026-09-30). The pencil sits on a family and on an element group.
    The Solids row comes last, there before the first solid (2026-10-11)."""
    from conftest import group_row

    item = _show_geometry(window, pump, two_blocks)
    labels = [item.child(i).text(0) for i in range(item.childCount())]
    assert labels == ['Nodes (6)', 'Coordinate systems (1)', 'Beams (0)',
                      'Triangles (0)', 'Quads (2)', 'Tetras (0)',
                      'Wedges (0)', 'Hexes (0)', 'Points (0)', 'Solids (0)']
    quads = _category(item, 'Quads')
    assert not quads.icon(1).isNull(), 'the edit pencil, like every category'
    assert [quads.child(i).text(0) for i in range(quads.childCount())] == [
        'wing (1)', 'tail (1)']
    wing = group_row(window, 7)
    assert not wing.icon(1).isNull(), 'the pencil on an element group: its properties'
    window._populate_entities(wing)
    pump()
    assert [wing.child(i).text(0) for i in range(wing.childCount())] == [
        'Element 10 (quad4, 4 nodes)']
    assert _category(item, 'Beams').childCount() == 0, 'no element group of beams'


def test_editing_blocks_opens_a_table_of_them(two_blocks, window, pump):
    from PySide6.QtCore import Qt

    _show_geometry(window, pump, two_blocks)
    window.tree.setCurrentItem(group_row(window, 7))
    window.edit_entities()
    pump()
    assert window.editing == ('Geometry', 'groups')
    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Vertical)
               for c in range(model.rowCount())]
    assert headers[:3] == ['Element group', 'Name', 'Elements']
    assert headers[3] == 'Material', (
        'then what the element group is made of — tests/test_solve_modes.py')
    assert model.item_count() == 2
    assert model.data(model.cell(0, 1)) == 'wing'
    # the elements it holds, by id and as runs — not a count, which is
    # not a thing anyone can edit. 'wing' holds element 10.
    assert model.data(model.cell(0, 2)) == '10'
    assert model.data(model.cell(1, 2)) == '11', 'and tail holds 11'
    assert model.flags(model.cell(0, 2)) & Qt.ItemFlag.ItemIsEditable, (
        'and the list is editable: naming an element claims it for '
        'this element group, the way a node is typed into a traceline')
    model.setData(model.cell(0, 1), 'port wing')
    assert two_blocks.group_name[0] == 'port wing'


def test_the_plus_adds_an_empty_block_outright(two_blocks, window, pump):
    """There is nothing to click in the view for an element group, so the button
    cannot arm a mode — it adds one and comes straight back up."""
    _show_geometry(window, pump, two_blocks)
    window.tree.setCurrentItem(group_row(window, 7))
    window.edit_entities()
    pump()
    assert window.add_action.isVisible()
    window.add_action.trigger()
    pump()
    assert not window.add_action.isChecked(), 'no add mode to be in'
    assert not window.add_mode
    assert list(two_blocks.group_id) == [7, 9, 10]
    assert window.table.model().item_count() == 3
    assert 'Added element group 10' in window.statusBar().currentMessage()


def test_deleting_a_block_row_deletes_the_part(two_blocks, window, pump):
    """From the Element Groups table as from a script: the element group, its elements
    and its own nodes go (Brandon, 2026-09-27)."""
    _show_geometry(window, pump, two_blocks)
    window.tree.setCurrentItem(group_row(window, 7))
    window.edit_entities()
    pump()
    window.table.select_items([0])
    window.delete_entity_rows()
    pump()
    assert list(two_blocks.group_id) == [9]
    assert list(two_blocks.elem_id) == [11], 'the wing went with its element group'
    assert 'Removed 1 element groups, 1 elements, 2 nodes' in \
        window.statusBar().currentMessage()


def test_picking_a_block_highlights_the_elements_it_holds(two_blocks, window,
                                                          pump):
    """An element group has no geometry of its own, so what it draws is its
    elements — an empty one correctly lights nothing up."""
    from visualdynamics.gui.main_window import _drawable

    components, entities = _drawable(two_blocks, None, {'groups': [9]})
    assert entities == {'elements': [1]}, 'the tail element, by row'
    assert components is None
    components, _entities = _drawable(two_blocks, {'groups'}, {})
    assert components == {'elements'}, 'the category is every element'
    two_blocks.add_group('fin')
    _components, entities = _drawable(two_blocks, None, {'groups': [10]})
    assert entities == {'elements': []}, 'an empty element group is not the whole model'


def test_an_element_moves_between_blocks_from_its_own_table(two_blocks, window,
                                                            pump):
    """An element group holds nothing itself, so this is where the grouping is
    actually edited — and an element group that does not exist is refused rather
    than quietly declared."""
    from PySide6.QtCore import Qt

    item = _show_geometry(window, pump, two_blocks)
    window.tree.setCurrentItem(_category(item, 'Quads'))
    window.edit_entities()
    pump()
    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert 'Element group' in headers
    column = headers.index('Element group')
    assert model.data(model.index(0, column)) == 'wing'
    model.setData(model.index(0, column), 'tail')
    assert two_blocks.elements_in('tail') == [10, 11]
    assert model.setData(model.index(0, column), 'nose') is False
    assert 'no element group' in window.statusBar().currentMessage()


# ---- the element groups table's Elements column is editable ----------------------


def _blocks_model(geometry):
    from visualdynamics.gui.object_tables import element_group_table_model

    return element_group_table_model(geometry)


def _cell(model, row, header):
    from PySide6.QtCore import Qt

    columns = [model.headerData(c, Qt.Orientation.Vertical)
               for c in range(model.rowCount())]
    index = model.cell(row, columns.index(header))
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
    first = int(two_blocks.group_id[0])
    expected = sorted(int(i) for i in
                      two_blocks.elem_id[two_blocks.elem_group == first])
    from visualdynamics.gui.object_tables import parse_id_runs
    assert parse_id_runs(text) == expected


def test_naming_an_element_claims_it_for_this_block(qt_app, two_blocks):
    """The same gesture as typing a node into a traceline: the element
    moves in, and leaves whatever element group it was in by itself."""
    from PySide6.QtCore import Qt

    from visualdynamics.gui.object_tables import parse_id_runs

    model = _blocks_model(two_blocks)
    first, second = (int(b) for b in two_blocks.group_id[:2])
    moving = int(two_blocks.elem_id[two_blocks.elem_group == second][0])
    text, index = _cell(model, 0, 'Elements')
    assert model.setData(index, f'{text} {moving}',
                         Qt.ItemDataRole.EditRole)
    assert int(two_blocks.elem_group[
        two_blocks.elem_id == moving][0]) == first
    # and it is gone from the element group it came from
    after, _ = _cell(model, 1, 'Elements')
    assert moving not in parse_id_runs(after)


def test_dropping_an_element_is_refused_with_the_way_to_do_it(qt_app,
                                                              two_blocks):
    """A node can be in no traceline; an element is always in exactly
    one element group, so a removal with no destination is not a state the
    geometry can hold. The refusal says how to move it instead."""
    from PySide6.QtCore import Qt

    model = _blocks_model(two_blocks)
    said = _refusal(model)
    _text, index = _cell(model, 0, 'Elements')
    assert not model.setData(index, '', Qt.ItemDataRole.EditRole)
    assert said and 'no element group' in said[0]
    assert 'Add it to the element group it belongs in' in said[0]
    assert list(two_blocks.elem_group) == [7, 9], 'and nothing moved'


def test_an_unknown_element_is_refused(qt_app, two_blocks):
    from PySide6.QtCore import Qt

    model = _blocks_model(two_blocks)
    said = _refusal(model)
    text, index = _cell(model, 0, 'Elements')
    assert not model.setData(index, f'{text} 9999',
                             Qt.ItemDataRole.EditRole)
    assert said and 'unknown elements [9999]' in said[0]


def _pick_blocks(window, pump, item, rows):
    """Select these element group rows under the geometry's Quads family."""
    groups = _category(item, 'Quads')
    groups.setExpanded(True)
    pump()
    window.tree.clearSelection()
    window.tree.setCurrentItem(groups.child(rows[0]))
    for row in rows:
        groups.child(row).setSelected(True)
    pump()


def test_two_blocks_of_one_kind_offer_merge_from_the_tree(two_blocks, window,
                                                         pump):
    """Picking two element groups that may be one puts Merge Element Groups on the bar
    (Brandon, 2026-09-27); pressing it makes them one, journaled."""
    item = _show_geometry(window, pump, two_blocks)
    _pick_blocks(window, pump, item, [0, 1])
    acts = window.acts_for()
    # and Merge Coincident Nodes beside it, which element groups carry
    # rather than the geometry (2026-10-11), and Mate Surfaces, two
    # groups of plates being surfaces to mate (2026-10-20)
    assert [(verb, icon) for verb, _label, icon, _h, _t in acts] == [
        ('merge_groups', 'merge_groups'), ('merge_coincident_nodes', 'merge_nodes'),
        ('align_meshes', 'align_meshes'), ('mate_surfaces', 'mate_surfaces')]
    acts[0][3]()
    pump()
    assert list(two_blocks.group_id) == [7]
    assert two_blocks.elements_in(7) == [10, 11]
    assert window.project.journal[-1] == (
        "project.merge_groups('Geometry', [7, 9])")
    assert 'merged 1 element group into wing' in window.statusBar().currentMessage()


def test_blocks_that_differ_offer_no_merge(two_blocks, window, pump):
    from visualdynamics import fem

    aluminum = fem.material('6061-T6')
    two_blocks.group_properties = {7: fem.GroupProperties(aluminum, 0.01),
                                   9: fem.GroupProperties(aluminum, 0.02)}
    item = _show_geometry(window, pump, two_blocks)
    _pick_blocks(window, pump, item, [0, 1])
    assert [act[0] for act in window.acts_for()] == ['merge_coincident_nodes',
                                                    'align_meshes', 'mate_surfaces']
    _pick_blocks(window, pump, item, [0])
    assert [act[0] for act in window.acts_for()] == ['merge_coincident_nodes',
                                                    'align_meshes'], (
        'one element group has nothing to merge or mate with')


def test_the_blocks_table_offers_the_same_merge(two_blocks, window, pump):

    _show_geometry(window, pump, two_blocks)
    # the table holds the element groups picked (2026-10-10), so the two
    # to merge are picked together in the tree and edited together
    window.tree.setCurrentItem(group_row(window, 7))
    group_row(window, 9).setSelected(True)
    window.edit_entities()
    pump()
    window.table.select_items([0])
    pump()
    assert not window.merge_groups_action.isVisible(), 'one row'
    window.table.select_items([1], clear=False)
    pump()
    assert window.merge_groups_action.isVisible()
    window.merge_groups_action.trigger()
    pump()
    assert list(two_blocks.group_id) == [7] and len(two_blocks.elem_id) == 2


def test_picking_a_family_names_its_element_groups(two_blocks, window, pump,
                                                   monkeypatch):
    """A family row picked in the tree captions the element groups that
    hold its elements, by name — not its elements by id, which were past
    the caption limit on the BARC in bricks (9 440 hexes) and say little
    when they fit (Brandon, 2026-10-10)."""
    from visualdynamics.viz import geometry as drawing

    captioned = []
    real = drawing.label_spots

    def spy(geometry, points, kind, chosen=None):
        spots, texts = real(geometry, points, kind, chosen)
        captioned.append((kind, list(texts)))
        return spots, texts

    monkeypatch.setattr(drawing, 'label_spots', spy)
    item = _show_geometry(window, pump, two_blocks)
    captioned.clear()
    window.tree.setCurrentItem(_category(item, 'Quads'))
    pump()
    # every draw of the pick captions the same way (a tree pick draws
    # the scene once per signal it raises)
    assert captioned and set(map(str, captioned)) == {
        str(('groups', ['wing', 'tail']))}
    assert window.statusBar().currentMessage() == 'Highlighting 2 quads'


def test_the_pencil_on_element_groups_tables_those_and_no_others(
        two_blocks, window, pump):
    """The Element Groups table opened from element group rows holds the
    ones picked — one, or several picked together — and the others are
    out of it: not shown, not selectable, not copied (Brandon,
    2026-10-10: the pencil on one opened every one)."""
    _show_geometry(window, pump, two_blocks)
    window.tree.setCurrentItem(group_row(window, 9))
    window.edit_entities()
    pump()
    table = window.table
    shown = [r for r in range(table.model().item_count()) if not table.isColumnHidden(r)]
    assert shown == [1], 'tail alone'
    assert window._editing_words() == 'element group 9'
    table.selectAll()
    pump()
    assert table.selected_items() == [1], 'Select All reaches only what is shown'
    # the groups run across: one column copied, a line per property
    lines = table.copy(whole_table=True).splitlines()
    assert lines[1] == 'tail' and all(len(line.split('\t')) == 1 for line in lines)
    # an element group added rebuilds the table: wing stays out of it
    window.add_action.trigger()
    pump()
    assert table.isColumnHidden(0) and not table.isColumnHidden(1)
    assert not table.isColumnHidden(table.model().item_count() - 1), 'the new one in'
    window.stop_editing()
    pump()
    # both picked together, then Edit from the context menu (a right-click
    # keeps a multi-selection, where a click on a pencil re-picks its row)
    window.tree.setCurrentItem(group_row(window, 7))
    group_row(window, 9).setSelected(True)
    window.edit_action.trigger()
    pump()
    assert window.editing == ('Geometry', 'groups')
    assert window.editing_scope == ('group', (7, 9))
    assert [table.isColumnHidden(r) for r in range(table.model().item_count())] == [
        False, False, True], 'wing and tail; the one added before is not picked'
    # an element group added there joins the table, and stays after the rebuild
    window.add_action.trigger()
    pump()
    added = int(two_blocks.group_id[-1])
    assert window.editing_scope[1][:2] == (7, 9) and added in window.editing_scope[1]
    assert not table.isColumnHidden(table.model().item_count() - 1)


def test_the_pencil_on_a_family_tables_its_elements_and_no_others(window, pump):
    """The elements table opened from a family row holds that family's
    elements — the pencil on Beams listed every element of the geometry
    (Brandon, 2026-10-10) — and still does after the rebuild an edit
    makes."""
    mixed = Geometry(
        node_id=[1, 2, 3, 4, 5, 6],
        node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0],
                  [0, 1, 0], [1, 1, 0], [2, 1, 0]],
        elem_id=[10, 11, 12],
        elem_conn=[[1, 2, 5, 4], [2, 3, 6, 5], [1, 4]],
        elem_type=[44, 44, 21],
        elem_group=[7, 7, 9],
        group_id=[7, 9],
        group_name=['plate', 'rail'])
    item = _show_geometry(window, pump, mixed)
    table = window.table

    def shown():
        return [r for r in range(table.model().rowCount())
                if not table.isRowHidden(r)]

    window.tree.setCurrentItem(_category(item, 'Beams'))
    window.edit_entities()
    pump()
    assert window.editing == ('Geometry', 'elements')
    assert shown() == [2], 'the beam alone'
    window._reload_edit_table()
    pump()
    assert shown() == [2], 'and after a rebuild'
    window.stop_editing()
    pump()
    window.tree.setCurrentItem(_category(item, 'Quads'))
    window.edit_entities()
    pump()
    assert shown() == [0, 1]


def test_the_open_groups_table_follows_the_groups_picked_in_the_tree(
        two_blocks, window, pump):
    """With the Element Groups table open on one group, Cmd-clicking
    another group's row in the tree adds its column, and Cmd-clicking a
    picked one off takes its column away; the table stays open
    throughout (Brandon, 2026-10-20: multi-selecting groups showed only
    the one the table was opened on)."""
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    _show_geometry(window, pump, two_blocks)
    window.tree.setCurrentItem(group_row(window, 7))
    window.edit_entities()
    pump()
    table = window.table

    def shown():
        return [c for c in range(table.model().item_count()) if not table.isColumnHidden(c)]

    assert shown() == [0]
    group_row(window, 9).setSelected(True)
    pump()
    assert window.editing == ('Geometry', 'groups')
    assert window.editing_scope == ('group', (7, 9))
    assert shown() == [0, 1], 'tail joins wing'
    assert table.selected_items() == [0, 1]
    assert window._editing_words() == 'element groups 7, 9'
    assert window._pencil_on(group_row(window, 9).data(0, ROLE_REFERENCE))
    group_row(window, 7).setSelected(False)
    pump()
    assert window.editing_scope == ('group', (9,))
    assert shown() == [1], 'wing leaves'


def test_the_groups_table_runs_across_and_edits_several_groups_at_once(two_blocks, window, pump):
    """The Element Groups table is the other way up (Brandon,
    2026-10-20): a group's properties read down a column, the groups
    are the columns, and a value is set on several groups at once by
    selecting across them and batch editing, or by dragging the fill
    handle across."""
    from PySide6.QtCore import QItemSelectionModel, Qt

    _show_geometry(window, pump, two_blocks)
    # both groups picked together, then edited: both are columns
    first, second = group_row(window, 7), group_row(window, 9)
    window.tree.clearSelection()
    window.tree.setCurrentItem(first)
    first.setSelected(True)
    second.setSelected(True)
    window.edit_entities()
    pump()
    table = window.table
    model = table.model()
    assert model.transposed and model.item_count() == 2
    assert not table.isColumnHidden(0) and not table.isColumnHidden(1)
    assert [model.headerData(c, Qt.Orientation.Horizontal) for c in range(2)] == ['wing', 'tail']
    assert model.headerData(0, Qt.Orientation.Vertical) == 'Element group'
    material = [c.title for c in model.columns].index('Material')
    assert model.headerData(material, Qt.Orientation.Vertical) == 'Material'
    # both groups' Material cells selected across, one batch edit
    selection = table.selectionModel()
    selection.clearSelection()
    for group in (0, 1):
        selection.select(model.cell(group, material), QItemSelectionModel.SelectionFlag.Select)
    applied, rejected = table.batch_edit('6061-T6')
    assert (applied, rejected) == (2, 0)
    assert all(two_blocks.group_properties[int(g)].material.name == '6061-T6'
               for g in two_blocks.group_id)
    # the fill handle carries a thickness across from the first group
    thickness = next(i for i, c in enumerate(model.columns) if c.title.startswith('Thickness'))
    assert model.setData(model.cell(0, thickness), '0.004', Qt.ItemDataRole.EditRole)
    selection.clearSelection()
    selection.select(model.cell(0, thickness), QItemSelectionModel.SelectionFlag.Select)
    assert table.fill_from_selection(1) == (1, 0)
    held = [two_blocks.group_properties[int(g)].thickness for g in two_blocks.group_id]
    assert held[0] > 0 and held[1] == held[0]
    # the selection in the table is the groups in the view: whole columns
    table.select_items([1])
    pump()
    assert table.selected_items() == [1] and table.whole_rows() == [1]


def test_a_long_elements_list_does_not_push_the_values_out_of_sight(window, pump):
    """A group whose elements are a long list of ids (here every other
    element of a strip) opened its column in the Element Groups table to
    that list's width, and the right-aligned values below it were out of
    sight and read as blank (2026-10-21). No column opens wider than a
    reading; the list is cut short and shows whole on hover."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QToolTip

    count = 60
    strip = Geometry(
        node_id=list(range(1, 2 * (count + 1) + 1)),
        node_xyz=[[x, y, 0] for y in (0, 1) for x in range(count + 1)],
        elem_id=list(range(10, 10 + count)),
        elem_conn=[[k + 1, k + 2, count + k + 3, count + k + 2] for k in range(count)],
        elem_type=[44] * count,
        elem_group=[7, 9] * (count // 2),
        group_id=[7, 9],
        group_name=['wing', 'tail'],
        length_unit='m')
    _show_geometry(window, pump, strip)
    row = group_row(window, 7)
    window.tree.clearSelection()
    window.tree.setCurrentItem(row)
    row.setSelected(True)
    window.edit_entities()
    pump()
    table, model = window.table, window.table.model()
    assert model.transposed and window.editing == ('Geometry', 'groups')
    titles = [c.title for c in model.columns]
    elements = model.cell(0, titles.index('Elements'))
    listed = model.data(elements, Qt.ItemDataRole.DisplayRole)
    assert len(listed) > 2 * table.WIDEST_COLUMN_CHARACTERS, 'every other element: a long list'
    limit = table.fontMetrics().averageCharWidth() * table.WIDEST_COLUMN_CHARACTERS
    assert table.columnWidth(0) <= limit
    assert table.fontMetrics().horizontalAdvance(listed) > table.columnWidth(0)
    # a number fits its cell; the list is cut short and shows whole on hover
    name = model.cell(0, titles.index('Name'))
    assert table.cut_short(elements) and not table.cut_short(name)
    table.resize(600, 600)
    pump()
    table.scrollTo(elements)
    where = table.visualRect(elements).center()
    event = _tooltip_request(table, where)
    table.viewportEvent(event)
    assert QToolTip.text() == listed
    QToolTip.hideText()


def _tooltip_request(table, where):
    """A tooltip request at `where` in the table's viewport."""
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QHelpEvent

    return QHelpEvent(QEvent.Type.ToolTip, where, table.viewport().mapToGlobal(where))
