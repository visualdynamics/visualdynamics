"""Plate meshes built from planes, tied where they meet (Brandon,
2026-09-26, for the BARC example).

Plates connect only through shared nodes, so planes meshed apart are
tied by merging the nodes they share along the lines where their
mid-surfaces meet. The proofs: an L of two planes shares its corner
line's nodes and solves as one structure; a box built from four planes
equals the same box meshed by hand; the merge refuses to fold an element
onto itself, keeps the lowest id, and refuses while linked data names a
node it would remove.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects
from test_fem import ALUMINUM

from visualdynamics.core import mesh
from visualdynamics.core.fem import GroupProperties, Model
from visualdynamics.core.geometry import Geometry
from visualdynamics.core.merge import merge
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.project import Project


def test_a_plane_is_a_block_of_rectangles():
    floor = mesh.plane((0, 0, 0), (0.4, 0, 0), (0, 0.2, 0), 0.1, 'floor')
    assert floor.num_nodes == 15 and len(floor.elem_conn) == 8
    assert list(floor.group_name) == ['floor'] and set(floor.elem_type) == {44}
    assert floor.length_unit == 'm'
    with pytest.raises(ValueError, match='not perpendicular'):
        mesh.plane((0, 0, 0), (1, 0, 0), (1, 1, 0), 0.5, 'skew')


def test_a_plane_takes_its_lengths_in_a_unit_and_holds_si():
    inch = mesh.plane((0, 0, 0), (6, 0, 0), (0, 3, 0), 0.5, 'wall', unit='in')
    assert inch.length_unit == 'in'
    assert inch.node_xyz[:, 0].max() == pytest.approx(6 * 0.0254)
    assert inch.num_nodes == 13 * 7


def test_elements_come_out_close_to_square_and_evenly_spaced():
    """Each edge divided evenly into the whole number of elements nearest
    the size (Brandon, 2026-09-26): a 0.13 by 1 strip at 0.125 is one
    element across, not two of 0.065 beside 0.125 lengthwise."""
    strip = mesh.plane((0, 0, 0), (1, 0, 0), (0, 0.13, 0), 0.125, 'strip')
    xs = np.unique(np.round(strip.node_xyz[:, 0], 9))
    ys = np.unique(np.round(strip.node_xyz[:, 1], 9))
    assert len(xs) == 9 and len(ys) == 2
    assert np.allclose(np.diff(xs), 0.125), 'even along its length'
    sides = (np.diff(xs)[0], np.diff(ys)[0])
    assert max(sides) / min(sides) < 1.1, 'close to square'
    # a size larger than an edge still gives that edge one element
    assert mesh.plane((0, 0, 0), (0.05, 0, 0), (0, 1, 0), 0.2, 'x').num_nodes == 12


def test_two_planes_meeting_share_their_corner_line():
    floor = mesh.plane((0, 0, 0), (0.4, 0, 0), (0, 0.2, 0), 0.1, 'floor')
    wall = mesh.plane((0, 0, 0), (0.4, 0, 0), (0, 0, 0.2), 0.1, 'wall')
    both = mesh.assemble(floor, wall)
    assert both.num_nodes == 15 + 15 - 5, 'the corner line shared'
    assert list(both.group_name) == ['floor', 'wall']
    both.group_properties = {1: GroupProperties(ALUMINUM, 0.003),
                             2: GroupProperties(ALUMINUM, 0.003)}
    model = Model.from_geometry(both)
    assert len(model.pieces()) == 1
    assert int(np.sum(model.eigensolution(num_modes=8).frequency == 0)) == 6


def test_a_box_from_planes_is_the_box_meshed_whole():
    """Four walls of a square tube as four planes, against the same tube
    meshed as one strip wrapped around it: the same model, to rounding."""
    s, h, size = 0.1, 0.05, 0.025
    walls = [mesh.plane((0, 0, 0), (s, 0, 0), (0, 0, h), size, 'bottom'),
             mesh.plane((s, 0, 0), (0, s, 0), (0, 0, h), size, 'right'),
             mesh.plane((s, s, 0), (-s, 0, 0), (0, 0, h), size, 'top'),
             mesh.plane((0, s, 0), (0, -s, 0), (0, 0, h), size, 'left')]
    box = mesh.assemble(*walls)
    box.group_properties = {b: GroupProperties(ALUMINUM, 0.002)
                            for b in range(1, 5)}
    built = Model.from_geometry(box)
    # the same tube by hand: nodes around the perimeter, up the height
    whole = Model()
    ring = [(x, 0) for x in np.arange(0, s, size)] + \
        [(s, y) for y in np.arange(0, s, size)] + \
        [(x, s) for x in np.arange(s, 0, -size)] + \
        [(0, y) for y in np.arange(s, 0, -size)]
    levels = np.arange(0, h + 1e-12, size)
    ids = {}
    for k, level in enumerate(levels):
        for i, (x, y) in enumerate(ring):
            ids[(i, k)] = whole.add_node(1000 * k + i + 1, x, y, level)
    for k in range(len(levels) - 1):
        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            whole.add_plate([ids[(i, k)], ids[(j, k)], ids[(j, k + 1)],
                             ids[(i, k + 1)]], ALUMINUM, 0.002)
    a = built.eigensolution(num_modes=16).frequency
    b = whole.eigensolution(num_modes=16).frequency
    assert np.allclose(a, b, rtol=1e-9, atol=1e-6)


def test_the_merge_keeps_the_lowest_id_and_refuses_to_fold_an_element():
    geometry = Geometry(node_id=[7, 3, 5, 9],
                        node_xyz=[[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1e-9, 0]],
                        elem_conn=[[7, 5], [3, 9]], elem_type=[21, 21],
                        length_unit='m')
    found = geometry.merge_coincident_nodes(1e-6)
    assert found == {'merged': 2, 'into': 2, 'duplicates': 0}, (
        'two beams on one line can be two members: kept')
    assert sorted(geometry.node_id) == [3, 5]
    assert [list(c) for c in geometry.elem_conn] == [[3, 5], [3, 5]]
    square = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5, 'p')
    with pytest.raises(ValueError, match='folds element 1 onto itself'):
        square.merge_coincident_nodes(0.6)


def test_the_verb_refuses_while_linked_data_names_a_node_it_would_remove():
    project = Project()
    floor = mesh.plane((0, 0, 0), (0.4, 0, 0), (0, 0.2, 0), 0.1, 'floor')
    wall = mesh.plane((0, 0, 0), (0.4, 0, 0), (0, 0, 0.2), 0.1, 'wall')
    joined = merge(
        [floor, _renumbered(wall, 100)])
    project.add('Geometry', joined)
    shapes = ShapeSet([10.0], [0.01], ['101Z+'], [[1.0]])
    project.add('Modes', shapes)
    project.link('Geometry', 'Modes')
    with pytest.raises(ValueError, match='Modes names node 101, which the '
                                          'merge would remove'):
        project.merge_coincident_nodes('Geometry')
    project.unlink('Modes')
    assert project.merge_coincident_nodes('Geometry') == {'merged': 5,
                                                          'into': 5,
                                                          'duplicates': 0}
    assert "project.merge_coincident_nodes('Geometry')" in project.journal[-1]


def _renumbered(geometry, offset):
    geometry.node_id = geometry.node_id + offset
    geometry.elem_conn = [c + offset for c in geometry.elem_conn]
    return geometry


def test_the_act_asks_a_tolerance_in_display_units(window, pump, monkeypatch):
    from visualdynamics.gui import main_window as window_module

    floor = mesh.plane((0, 0, 0), (4, 0, 0), (0, 2, 0), 1, 'floor', unit='in')
    wall = _renumbered(mesh.plane((0, 0, 0), (4, 0, 0), (0, 0, 2), 1, 'wall',
                                  unit='in'), 100)
    window.add_object('Geometry', merge([floor, wall]))
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    select_objects(window, pump, 'Geometry')
    from test_acts import _bar

    assert 'Merge Coincident Nodes' in _bar(window.data_pane) + _bar(window.scene)
    asked = []

    def get_double(parent, title, label, value, *rest):
        asked.append((label, value))
        # half an inch: under the one-inch grid, so only the shared line
        # merges — and read as meters it would fold every element
        return 0.5, True

    monkeypatch.setattr(window_module.QInputDialog, 'getDouble', get_double)
    window.merge_nodes_act()
    pump()
    assert asked[0][0].endswith('[in]:')
    assert window.objects['Geometry'].num_nodes == 15 + 15 - 5
    assert 'merged 5 nodes into 5 — 25 nodes remain' in \
        window.statusBar().currentMessage()


def test_planes_of_one_name_are_one_block():
    """Five walls named 'box' are the box — one block, given its material
    once; an unnamed plane stays a block of its own."""
    walls = [mesh.plane((0, 0, 0), (0.1, 0, 0), (0, 0, 0.05), 0.025, 'box'),
             mesh.plane((0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.05), 0.025, 'box'),
             mesh.plane((0, 0, 0.05), (0.1, 0, 0), (0, 0.1, 0), 0.025, 'lid'),
             mesh.plane((0, 0, 0), (0, 0.1, 0), (0, 0, 0.05), 0.025, ''),
             mesh.plane((0, 0.1, 0), (0.1, 0, 0), (0, 0, 0.05), 0.025, '')]
    whole = mesh.assemble(*walls)
    assert list(whole.group_name) == ['box', 'lid', '', '']
    assert len(whole.elements_in('box')) == 2 * len(walls[0].elem_conn)
