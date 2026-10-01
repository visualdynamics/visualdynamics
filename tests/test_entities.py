"""A geometry's four groups, reached the way the tree lists them.

The tree has always shown a geometry as Nodes, Coordinate Systems,
Tracelines and Elements. These tests hold the scripted side to the same
four names, and to the thing that makes them worth having: a view is a
window onto the geometry, not a copy of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.geometry import Geometry


@pytest.fixture
def geometry():
    return Geometry(
        node_id=[1, 2, 3, 4],
        node_xyz=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        elem_conn=[[1, 2, 3]],
        elem_type=[41],
        block_name=['edge'],
        length_unit='m')


def test_the_groups_are_named_as_the_tree_names_them(geometry):
    assert len(geometry.nodes) == 4
    assert len(geometry.coordinate_systems) == 1
    assert len(geometry.elements) == 1
    assert len(geometry.blocks) == 1


def test_a_column_is_the_geometrys_own_array(geometry):
    """Not a copy — the whole point. Writing into it is an edit."""
    assert geometry.nodes.xyz is geometry.node_xyz
    geometry.nodes.xyz[:, 2] += 0.5
    assert geometry.node_xyz[0, 2] == 0.5


def test_a_row_writes_through_to_the_geometry(geometry):
    geometry.nodes[0].xyz = [9.0, 9.0, 9.0]
    assert list(geometry.node_xyz[0]) == [9.0, 9.0, 9.0]
    geometry.blocks[0].name = 'renamed'
    assert geometry.block_name[0] == 'renamed'


def test_a_row_reads_the_geometry_as_it_stands_now(geometry):
    """It holds a row number, not the values."""
    node = geometry.nodes[1]
    geometry.node_xyz[1] = [5.0, 5.0, 5.0]
    assert list(node.xyz) == [5.0, 5.0, 5.0]


def test_rows_iterate_and_slice_and_count_backwards(geometry):
    assert [int(node.id) for node in geometry.nodes] == [1, 2, 3, 4]
    assert [int(node.id) for node in geometry.nodes[1:3]] == [2, 3]
    assert int(geometry.nodes[-1].id) == 4
    assert geometry.nodes[2].row == 2


def test_asking_for_something_that_is_not_there_says_what_is(geometry):
    with pytest.raises(AttributeError, match="has no column 'positions'"):
        _ = geometry.nodes.positions
    with pytest.raises(AttributeError, match="has no 'position'"):
        _ = geometry.nodes[0].position
    with pytest.raises(IndexError, match='there are 4'):
        _ = geometry.nodes[9]


def test_adding_goes_through_the_geometrys_own_rules(geometry):
    """Which is what keeps ids unique and connectivity honest."""
    new_id = geometry.nodes.add([2.0, 0.0, 0.0])
    assert new_id == 5 and len(geometry.nodes) == 5
    with pytest.raises(ValueError, match='already exists'):
        geometry.nodes.add([0.0, 0.0, 0.0], node_id=1)
    with pytest.raises(ValueError, match='unknown nodes'):
        geometry.elements.add([1, 999])


def test_everything_is_deleted_by_id(geometry):
    """One rule for all four groups."""
    geometry.nodes.add([2.0, 2.0, 2.0])
    geometry.nodes.delete([5])
    assert [int(i) for i in geometry.nodes.ids] == [1, 2, 3, 4]

    element = int(geometry.elements[0].id)
    geometry.elements.delete([element])
    assert len(geometry.elements) == 0


def test_deleting_an_id_that_is_not_there_is_harmless(geometry):
    geometry.elements.delete([9999])
    assert len(geometry.elements) == 1


def test_the_columns_are_the_table_columns(geometry):
    assert geometry.nodes.columns == (
        'ids', 'xyz', 'colors', 'placement_systems', 'displacement_systems')
    assert geometry.elements.columns == (
        'ids', 'types', 'colors', 'blocks', 'nodes')


def test_a_view_says_what_it_holds(geometry):
    assert repr(geometry.nodes) == '<4 nodes>'
    assert repr(geometry.blocks) == '<1 block>'
    assert repr(geometry.elements) == '<1 element>'
    assert 'id=1' in repr(geometry.nodes[0])
    assert 'matrix' not in repr(geometry.coordinate_systems[0])


def test_the_view_follows_the_geometry_as_it_changes(geometry):
    """Built fresh on each call, so nothing is left holding a stale
    length after an add or a delete."""
    nodes = geometry.nodes
    geometry.add_node([7.0, 7.0, 7.0])
    assert len(nodes) == 5, 'the view reads the geometry, it does not cache'


def test_connectivity_comes_back_as_the_node_ids(geometry):
    assert [int(n) for n in geometry.elements[0].nodes] == [1, 2, 3]
    assert isinstance(geometry.elements[0].nodes, np.ndarray)
