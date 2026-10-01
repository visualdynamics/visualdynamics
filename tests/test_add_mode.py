"""Creating geometry by clicking in the 3D view.

The one gesture in visualdynamics that *makes* something out of nothing: the
`+` on the toolbar arms it, and after that a click in the view is a
node, a coordinate system, or the next corner of an element. It is also
the part with the most state — what is being edited, what the cursor is
over, which nodes are half-picked — and it was covered only by the
legacy end-to-end script, twenty seconds in a subprocess whose failures
name a line number in a 1900-line file.

The clicks go in through `_add_at` and `hover_at`, which is how a click
arrives from VTK anyway; `conftest`'s offscreen plotter gives a real
projector, so the screen positions are the ones a mouse would land on.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import edit_category as edit
from conftest import fixture_path


@pytest.fixture
def plate(window, pump):
    """The beam plate as exodus — 45 nodes and 32 quads.

    The `.npz` of the same plate is tracelines only, and half of what is
    checked here is about elements that are already there.
    """
    window.import_paths([fixture_path('plate', 'geometry.exo')])
    pump()
    return window.objects['Geometry']


def screen_of(window):
    return window._projector.screen()[0]


def test_the_plus_appears_with_editing_and_says_what_it_will_do(plate, window,
                                                               pump):
    """Show only what applies: there is nothing to add until a category
    says what a new thing would be."""
    assert not window.add_action.isVisible(), 'nothing is being edited'
    edit(window, pump, 'Nodes')
    assert window.add_action.isVisible()
    assert not window.add_mode, 'shown, not armed'
    window.set_add_mode(True)
    assert 'click in the view to place a node' in \
        window.statusBar().currentMessage()
    edit(window, pump, 'Coordinate systems')
    window.set_add_mode(True)
    assert 'place a coordinate system' in window.statusBar().currentMessage()


def test_a_node_lands_where_the_click_did(plate, window, pump):
    """And on the model's own surface: the plate is flat, so a node
    placed on it has to arrive at z = 0 rather than somewhere along the
    ray the cursor cast."""
    edit(window, pump, 'Nodes')
    window.set_add_mode(True)
    screen = screen_of(window)
    before = plate.num_nodes
    window._add_at(*(screen[0] + screen[8]) / 2)
    assert plate.num_nodes == before + 1
    assert abs(float(plate.node_xyz[-1][2])) < 1e-9, 'on the plate, not above'
    assert 'Added node' in window.statusBar().currentMessage()
    assert window.table.model().rowCount() == plate.num_nodes, (
        'the table beside the model is the model'
    )


def test_a_coordinate_system_lands_where_the_click_did_too(plate, window,
                                                           pump):
    edit(window, pump, 'Coordinate systems')
    window.set_add_mode(True)
    screen = screen_of(window)
    before = len(plate.cs_id)
    window._add_at(*(screen[0] + screen[8]) / 2)
    assert len(plate.cs_id) == before + 1
    assert 'Added coordinate system' in window.statusBar().currentMessage()


def test_the_family_being_edited_says_what_a_click_builds(plate, window,
                                                          pump):
    """The beam / triangle / quad chooser is gone (2026-09-30): the
    family row the editing began from says what is added, so there is
    nothing to choose and nothing to get wrong."""
    assert not hasattr(window, 'element_type_actions')
    edit(window, pump, 'Quads')
    assert window.editing == ('Geometry', 'elements')
    assert window.editing_scope == ('family', 'quads')
    window.set_add_mode(True)
    assert window.element_type == (44, 4)
    assert window._picking_component() == 'nodes', (
        'an element is built out of nodes, so that is what the cursor picks')
    edit(window, pump, 'Beams')
    window.set_add_mode(True)
    assert window.element_type == (21, 2)
    edit(window, pump, 'Hexes')
    window.set_add_mode(True)
    assert window.element_type == (115, 8)


def test_each_element_type_commits_on_its_own_node_count(plate, window, pump):
    def pick(rows):
        screen = screen_of(window)
        for n, row in enumerate(rows):
            window.hover_at(*screen[row])
            window._add_at(*screen[row], extend=n > 0)

    for family, rows, code in (('Triangles', (0, 1, 6), 41),
                               ('Quads', (0, 1, 6, 5), 44)):
        edit(window, pump, family)
        window.set_add_mode(True)
        assert not window._picked_nodes
        before = len(plate.elem_conn)
        pick(rows)
        assert len(plate.elem_conn) == before + 1, f'{family} did not commit'
        assert int(plate.elem_type[-1]) == code
        assert not window._picked_nodes, 'the pick list resets after committing'
    # a beam has no count to commit on: it is a line, said done with Enter
    edit(window, pump, 'Beams')
    window.set_add_mode(True)
    before = len(plate.elem_conn)
    pick((2, 3))
    assert len(plate.elem_conn) == before, 'a line waits for Enter'
    window.commit_action.trigger()
    pump()
    assert len(plate.elem_conn) == before + 1
    assert int(plate.elem_type[-1]) == 21


def test_a_line_of_beams_is_as_long_as_you_say_and_commits_on_enter(
        plate, window, pump):
    """A beam takes the line's grammar (2026-09-30): it is however many
    nodes were picked, chained, so it is the one that needs saying when
    it is done — and the chain lands in one block."""
    edit(window, pump, 'Beams')
    window.set_add_mode(True)
    screen = screen_of(window)
    before = len(plate.elem_conn)
    for n, row in enumerate((2, 3, 4)):
        window.hover_at(*screen[row])
        window._add_at(*screen[row], extend=n > 0)
    assert len(plate.elem_conn) == before, 'still being drawn'
    window.commit_action.trigger()
    pump()
    assert len(plate.elem_conn) == before + 2
    assert [int(t) for t in plate.elem_type[-2:]] == [21, 21]
    assert len({int(b) for b in plate.elem_block[-2:]}) == 1, 'one block'
    assert [list(map(int, c)) for c in plate.elem_conn[-2:]] == \
        [[int(plate.node_id[2]), int(plate.node_id[3])],
         [int(plate.node_id[3]), int(plate.node_id[4])]]
    assert any(line.startswith("project['Geometry'].add_beams(")
               for line in window.project.journal)


def test_the_cursor_lights_the_node_it_is_over_not_an_element_of_that_id(
        plate, window, pump):
    """While building an element the picker returns nodes, and the hover
    mesh has to light one — ids are shared between the groups, so drawing
    'entity 6' without asking which group would light a face."""
    edit(window, pump, 'Quads')
    window.set_add_mode(True)
    screen = screen_of(window)
    window.hover_at(*screen[6])
    assert window._hovered == int(plate.node_id[6])
    assert list(window._hover_mesh.verts) == [1, 6]


def test_leaving_add_mode_and_editing_puts_the_toolbar_back(plate, window,
                                                            pump):
    edit(window, pump, 'Nodes')
    window.set_add_mode(True)
    window.escape_action.trigger()
    pump()
    assert window.editing is None, 'Escape leaves editing'
    assert not window.add_mode
    assert not window.add_action.isVisible(), '+ hides when not editing'


def test_the_pencil_on_the_tree_opens_the_table_and_closes_it(plate, window,
                                                              pump):
    """Editing lives on the tree's own pencil, not on a toolbar button —
    and the toolbar's state carrier follows it either way."""
    item = window._item_for_object('Geometry')
    item.setExpanded(True)
    pump()
    nodes = item.child(0)
    assert not nodes.icon(1).isNull(), 'every category wears the pencil'
    window._tree_item_clicked(nodes, 1)
    pump()
    assert window.editing == ('Geometry', 'nodes')
    assert window.edit_toggle_action.isChecked()
    window._tree_item_clicked(nodes, 1)
    pump()
    assert window.editing is None, 'the second click closes it'
    assert not window.edit_toggle_action.isChecked()
    assert not window.add_action.isVisible()


def test_an_edit_that_would_break_the_model_is_refused_at_entry(plate, window,
                                                               pump):
    """Never written and complained about afterwards."""
    edit(window, pump, 'Quads')
    model = window.table.model()
    column = [c.title for c in model.columns].index('Nodes')
    kept = list(plate.elem_conn[0])
    assert model.setData(model.index(0, column), '101 99999') is False
    assert list(plate.elem_conn[0]) == kept, 'refused, not written'
    assert 'unknown nodes' in window.statusBar().currentMessage()
    assert model.setData(model.index(0, column), '101 102 103 104') is True
    assert list(plate.elem_conn[0]) == [101, 102, 103, 104]


def test_the_geometry_is_still_valid_after_all_of_that(plate, window, pump):
    """The point of the end-to-end script was that a session of edits
    leaves something loadable, which is worth keeping as a claim."""
    edit(window, pump, 'Nodes')
    window.set_add_mode(True)
    screen = screen_of(window)
    window._add_at(*(screen[0] + screen[8]) / 2)
    edit(window, pump, 'Triangles')
    window.set_add_mode(True)
    for n, row in enumerate((0, 1, 6)):
        window.hover_at(*screen[row])
        window._add_at(*screen[row], extend=n > 0)
    window.stop_editing()
    pump()
    plate.validate()
    assert np.isfinite(plate.node_xyz).all()


# ---- which block new elements go into (2026-09-26) ------------------------

def _block_box(window):
    box = window.element_block_box
    return [box.itemText(i) for i in range(box.count())], box.currentText()


def test_the_block_drop_down_is_offered_only_while_adding_elements(plate, window,
                                                                  pump):
    edit(window, pump, 'Quads')
    assert not window._element_block_handle.isVisible()
    window.set_add_mode(True)
    assert window._element_block_handle.isVisible()
    items, _current = _block_box(window)
    assert items == ['Block 1 — body', 'New block']
    edit(window, pump, 'Nodes')
    window.set_add_mode(True)
    assert not window._element_block_handle.isVisible()


def test_the_drop_down_offers_the_familys_blocks_and_a_new_one(plate,
                                                                window,
                                                                pump):
    """A quad added to a model of quads joins them — a file's four-node
    shells are the same kind as the quad add mode makes; a beam added to
    it has no block of beams to join, so only a new block is offered —
    a block holds one family (2026-09-30), where a plate block would
    refuse it. Editing from a block row starts on that block."""
    edit(window, pump, 'Quads')
    window.set_add_mode(True)
    assert _block_box(window) == (['Block 1 — body', 'New block'],
                                  'Block 1 — body')
    edit(window, pump, 'Beams')
    window.set_add_mode(True)
    assert _block_box(window) == (['New block'], 'New block')
    from conftest import edit_block

    edit_block(window, pump, 1)
    assert window.editing == ('Geometry', 'blocks')
    assert window.editing_scope == ('block', 1)


def test_links_picked_one_after_another_share_the_new_block(plate, window,
                                                            pump):
    """The first beam makes the block and the drop-down then shows it, so
    every link picked after joins it — one block to make rigid."""
    edit(window, pump, 'Beams')
    window.set_add_mode(True)
    screen = screen_of(window)
    blocks_before = list(plate.block_id)
    for a, b in ((0, 8), (2, 10)):
        window.hover_at(*screen[a])
        window._add_at(*screen[a])
        window.hover_at(*screen[b])
        window._add_at(*screen[b], extend=True)
        window.commit_action.trigger()      # a line is said done with Enter
    pump()
    new = [int(b) for b in plate.block_id if int(b) not in blocks_before]
    assert len(new) == 1, 'one new block for both links'
    assert [int(b) for b in plate.elem_block[-2:]] == new * 2
    assert _block_box(window)[1] == f'Block {new[0]}'
    journal = window.project.journal
    assert any(line.endswith(".add_block()") for line in journal)
    assert journal[-1].endswith(f'block={new[0]}, elem_type=21)'), journal[-1]


def test_a_block_picked_in_the_drop_down_is_honored(plate, window, pump):
    """Two blocks of beams: the one picked in the drop-down is where
    the next line lands."""
    first = plate.add_beams([int(plate.node_id[0]), int(plate.node_id[1])])
    second = plate.add_beams([int(plate.node_id[2]), int(plate.node_id[3])])
    window._refresh_item(window._item_for_object('Geometry'), plate)
    edit(window, pump, 'Beams')
    window.set_add_mode(True)
    box = window.element_block_box
    assert {box.itemData(i) for i in range(box.count())} == {first, second, None}
    box.setCurrentIndex(box.findData(second))
    screen = screen_of(window)
    window.hover_at(*screen[4])
    window._add_at(*screen[4])
    window.hover_at(*screen[5])
    window._add_at(*screen[5], extend=True)
    window.commit_action.trigger()
    pump()
    assert int(plate.elem_block[-1]) == second, 'where it was told'
