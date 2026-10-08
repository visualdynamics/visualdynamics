"""What a geometry's entities read as in the tree, and in the status bar.

Expanding a category lists its entities, and each row has to say which
one it is: an id for a node, an id and a type for a coordinate system, a
node count for a line, a type and a node count for an element. They are
built lazily — a 200 000-node model must not build 200 000 tree items on
import — and the list is what the selection and the delete act on, so
the labels are load-bearing rather than decoration.

The status bar is the other half of the same reading: one picked node
says where it is, in the display units, and several say how many.
"""

from __future__ import annotations

import pytest
from conftest import fixture_path


@pytest.fixture
def geometry_item(window, pump):
    window.import_paths([fixture_path('plate', 'geometry.exo')])
    pump()
    geometry = window.objects['Geometry']
    geometry.add_beams([int(n) for n in geometry.node_id[:3]])
    item = window._item_for_object('Geometry')
    window._refresh_item(item, geometry)
    item.setExpanded(True)
    pump()
    return item


def category(item, label):
    return next(item.child(i) for i in range(item.childCount())
                if item.child(i).text(0).startswith(label))


def labels(window, item, name):
    child = category(item, name)
    window._populate_entities(child)
    return [child.child(i).text(0) for i in range(child.childCount())]


def test_each_kind_of_entity_says_which_one_it_is(geometry_item, window,
                                                  pump):
    geometry = window.objects['Geometry']
    nodes = labels(window, geometry_item, 'Nodes')
    assert nodes[0] == f'Node {int(geometry.node_id[0])}'

    systems = labels(window, geometry_item, 'Coordinate systems')
    assert systems[0].startswith(f'CS {int(geometry.cs_id[0])}')
    assert '(cartesian)' in systems[0], 'the kind of frame it is'

    # elements are listed under their element group, which sits under its family
    from conftest import group_row

    quads = category(geometry_item, 'Quads')
    assert quads.childCount() == 1, 'one element group of quads'
    group = group_row(window, int(geometry.group_id[0]))
    window._populate_entities(group)
    elements = [group.child(i).text(0) for i in range(group.childCount())]
    assert elements[0].startswith(f'Element {int(geometry.elem_id[0])} ')
    assert '(quadshell4, 4 nodes)' in elements[0], (
        'the type it is, and its node count')


def test_the_entities_are_listed_only_when_the_branch_is_opened(geometry_item,
                                                                window):
    """A big model must not build a tree item per node on import."""
    nodes = category(geometry_item, 'Nodes')
    assert nodes.childCount() == 0, 'nothing built yet'
    window._populate_entities(nodes)
    assert nodes.childCount() == window.objects['Geometry'].num_nodes
    window._populate_entities(nodes)
    assert nodes.childCount() == window.objects['Geometry'].num_nodes, (
        'and opening it again does not list them twice')


def test_one_picked_node_says_where_it_is(geometry_item, window, pump):
    """In the display units, which is the whole point of saying it."""
    geometry = window.objects['Geometry']
    nodes = category(geometry_item, 'Nodes')
    window._populate_entities(nodes)
    window.tree.clearSelection()
    window.tree.setCurrentItem(nodes.child(0))
    nodes.child(0).setSelected(True)
    pump()
    message = window.statusBar().currentMessage()
    assert message.startswith(f'Node {int(geometry.node_id[0])} at ')
    assert window.unit_system.label_text('length') in message


def test_several_picked_entities_are_counted_not_listed(geometry_item, window,
                                                        pump):
    nodes = category(geometry_item, 'Nodes')
    window._populate_entities(nodes)
    window.tree.clearSelection()
    window.tree.setCurrentItem(nodes.child(0))
    for i in range(3):
        nodes.child(i).setSelected(True)
    pump()
    assert 'Highlighting 3 nodes' in window.statusBar().currentMessage()
