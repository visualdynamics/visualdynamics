"""Align Meshes: the step before Merge Coincident Nodes, where parts
meshed apart do not match (Brandon, 2026-10-11, for the Frame-Wing,
whose wing is meshed at an eighth of an inch on a frame meshed at a
twelfth).

The coarser piece's nodes move onto the finer piece's nearest, so the
two are one point where they meet and the merge can join them. The
geometry finds the moves (`align_moves`), refuses one that turns an
element inside out (`align_folds`) and makes them (`align_meshes`);
the project's verb is journaled and refused while linked data names a
node that would move; the app's pane is Merge's, opened for aligning.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import group_row

from visualdynamics import mesh
from visualdynamics.core.shapes import ShapeSet

INCH = 0.0254
GAP = 0.05


def _wing_on_frame(unit='in'):
    """A fine block (a quarter inch) and, kept apart, a coarse one (a
    third of an inch) resting a twentieth of an inch over it and off
    its grid, a third of an inch thick: 16 nodes of the coarse block's
    underside, none on a node of the fine block's top."""
    fine = mesh.block((0, 0, -0.5), (1, 0, 0), (0, 1, 0), (0, 0, 0.5), 0.25,
                      'frame', unit=unit)
    coarse = mesh.block((0.02, 0.01, GAP), (1, 0, 0), (0, 1, 0), (0, 0, 1 / 3),
                        1 / 3, 'wing', unit=unit)
    return mesh.assemble(fine, coarse, apart=True)


def _ids(geometry):
    return dict(zip(geometry.group_name, geometry.group_id.tolist()))


def test_the_coarser_mesh_moves_onto_the_finer_and_then_merges():
    geometry = _wing_on_frame()
    ids = _ids(geometry)
    reach = geometry.align_reach()
    assert reach == pytest.approx(0.5 / 3 * INCH), 'half the coarse spacing'
    moves = geometry.align_moves(reach)
    wing, frame = set(geometry.nodes_in([ids['wing']])), set(
        geometry.nodes_in([ids['frame']]))
    assert len(moves) == 16
    assert all(a in wing and b in frame for a, b in moves), 'coarse onto fine'
    assert len({b for _a, b in moves}) == 16, 'each onto a node of its own'
    before = geometry.node_xyz.copy()
    found = geometry.align_meshes(reach)
    assert found['moved'] == 16 and found['farthest'] <= reach
    moved = geometry.node_index([a for a, _b in moves])
    assert np.allclose(geometry.node_xyz[moved],
                       geometry.node_xyz[geometry.node_index([b for _a, b in moves])])
    untouched = np.setdiff1d(np.arange(geometry.num_nodes), moved)
    assert np.array_equal(geometry.node_xyz[untouched], before[untouched])
    assert geometry.merge_coincident_nodes(1e-9)['merged'] == 16
    assert len(geometry._parts(None)) == 1, 'one structure now'


def test_only_the_skins_move_and_a_lone_piece_has_nothing_to_align():
    """The coarse block's top is a third of an inch further from the fine
    one and out of reach; a fine node under the top face is inside its
    solid and never a target. One piece alone aligns with nothing."""
    geometry = _wing_on_frame()
    ids = _ids(geometry)
    inch = geometry.node_xyz / INCH
    for _a, b in geometry.align_moves(geometry.align_reach()):
        assert inch[geometry.node_index([b])[0], 2] == pytest.approx(0.0)
    assert geometry.align_moves(1.0, groups=[ids['frame']]) == []
    # the fine block's 5 x 5 x 3 nodes less the 3 x 3 x 1 inside it
    (frame_rows,) = geometry._parts([ids['frame']])
    assert len(geometry._skin(frame_rows)) == 75 - 9


def test_a_hole_does_not_make_a_coarse_mesh_read_fine():
    """A hole cut through the coarse block bends its rim bricks, some
    edges a fraction of the fine mesh's: the shortest edge anywhere
    (`node_spacing`) would call the coarse block the finer, and move the
    fine one. A piece is ranked by its typical element instead — the
    Frame-Wing's wing read finer than its frame that way (2026-10-11)."""
    fine = mesh.block((0, 0, -0.5), (1, 0, 0), (0, 1, 0), (0, 0, 0.5), 0.25,
                      'frame', unit='in')
    coarse = mesh.block((0.02, 0.01, GAP), (1, 0, 0), (0, 1, 0), (0, 0, 1 / 3),
                        1 / 3, 'wing', unit='in',
                        holes=[((0.52, 0.51, GAP), 0.12, 2, None, None)])
    geometry = mesh.assemble(fine, coarse, apart=True)
    ids = _ids(geometry)
    assert geometry.node_spacing([ids['wing']]) < geometry.node_spacing([ids['frame']])
    wing = set(geometry.nodes_in([ids['wing']]))
    moves = geometry.align_moves(geometry.align_reach())
    assert moves and all(a in wing for a, _b in moves)


def test_nodes_already_one_point_are_merged_not_moved():
    """Two planes sharing an edge exactly, kept apart: places to merge,
    nothing to align."""
    whole = mesh.plane((0, 0, 0), (2, 0, 0), (0, 1, 0), 0.5, 'floor')
    mesh.join(whole, mesh.plane((0, 0, 0), (2, 0, 0), (0, 0, 1), 0.5, 'wall'),
              -1.0)
    assert len(whole.coincident_spots(1e-6)) == 5
    assert whole.align_moves(0.25) == []


def test_a_move_left_out_moves_nothing_and_its_target_stays_free():
    geometry = _wing_on_frame()
    reach = geometry.align_reach()
    moves = geometry.align_moves(reach)
    left = moves[0]
    fewer = geometry.align_moves(reach, exclude=left)
    assert len(fewer) == 15
    assert not set(left) & {n for move in fewer for n in move}
    assert geometry.align_meshes(reach, exclude=left)['moved'] == 15


def test_a_move_that_turns_an_element_inside_out_is_refused():
    """A node carried past the far side of its own element is no
    alignment: said before Align (`align_folds`), refused by it."""
    geometry = _wing_on_frame()
    inch = geometry.node_xyz / INCH
    ids = _ids(geometry)
    wing = geometry.nodes_in([ids['wing']])
    frame = geometry.nodes_in([ids['frame']])
    corner = min(wing, key=lambda n: inch[geometry.node_index([n])[0]].sum())
    far = max(frame, key=lambda n: inch[geometry.node_index([n])[0], :2].sum())
    refusal = geometry.align_folds([[corner, far]])
    assert refusal is not None and 'inside out' in refusal
    assert 'element' in refusal and str(corner) in refusal
    assert geometry.align_folds(geometry.align_moves(geometry.align_reach())) is None


def test_a_link_drawn_to_nothing_is_not_refused_and_the_merge_removes_it():
    """A rigid link across the gap the move closes — the Frame-Wing's
    screw ties, found on the real model (2026-10-11) — has no inside to
    turn: the move goes through, and the merge after it removes the
    link that now joins one node to itself."""
    from visualdynamics.core.fem import RIGID, GroupProperties

    geometry = _wing_on_frame()
    parts = list(_ids(geometry).values())       # the blocks, not the tie
    reach = geometry.align_reach(parts)
    mover, target = geometry.align_moves(reach, parts)[0]
    group = geometry.add_group('ties')
    geometry.add_elements([[target, mover]], [21], [group])
    geometry.group_properties[group] = GroupProperties(RIGID)
    assert geometry.align_folds(geometry.align_moves(reach, parts)) is None
    assert geometry.align_meshes(reach, parts)['moved'] == 16
    assert geometry.merge_coincident_nodes(1e-9)['links'] == 1


def test_the_project_aligns_journals_and_refuses_under_linked_data():
    import visualdynamics

    project = visualdynamics.Project('p')
    geometry = _wing_on_frame()
    project.add('Geometry', geometry)
    ids = _ids(geometry)
    groups = [ids['frame'], ids['wing']]
    moves = geometry.align_moves(geometry.align_reach(groups), groups)
    project.add('Modes', ShapeSet([10.0], [0.01], [f'{moves[0][0]}Z+'], [[1.0]]))
    project.link('Geometry', 'Modes')
    with pytest.raises(ValueError, match=(
            f'Modes names node {moves[0][0]}, which aligning would move — '
            'align before measuring or solving, or unlink it first')):
        project.align_meshes('Geometry', groups=groups)
    project.unlink('Modes')
    assert project.align_meshes('Geometry', groups=groups)['moved'] == 16
    assert project.journal[-1] == (
        f"project.align_meshes('Geometry', groups={groups})")


# ---- the pane ------------------------------------------------------------

def _align_pane(window, pump, geometry=None):
    """The blocks, both groups picked in the tree, and Align Meshes
    opened from the bar."""
    geometry = geometry or _wing_on_frame()
    window.add_object('Geometry', geometry)
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.tree.clearSelection()
    window.tree.setCurrentItem(group_row(window, 1))
    group_row(window, 2).setSelected(True)
    pump()
    acts = {act[0]: act for act in window.acts_for()}
    acts['align_meshes'][3]()
    pump()
    return window.scene.merge_panel


def test_the_pane_marks_the_moves_and_aligns_the_marked_ones(window, pump):
    """Picked groups offer Align Meshes beside the merge; its pane is the
    merge's, titled for aligning, every move marked with a line to
    where it goes. A click leaves one out; Align moves the rest, the
    pane stays open with them drawn as squares, and the merge then finds
    them as places."""
    panel = _align_pane(window, pump)
    assert panel.title.text() == 'Align Meshes'
    assert panel.tolerance_label.text() == 'Reach [in]'
    assert panel.merge_button.text() == 'Align'
    assert panel.value() == pytest.approx(1 / 6, abs=1e-6), 'six decimals'
    assert panel.reading_label.text().startswith(
        '16 nodes of the coarser mesh can move onto the finer; 16 marked,')
    assert 'align-moves' in window.scene.plotter.actors
    window.toggle_merge_place(0)
    pump()
    assert '; 15 marked,' in panel.reading_label.text()
    panel.merge_button.click()
    pump()
    assert panel.isVisibleTo(window), 'the pane stays open'
    assert panel.reading_label.text().startswith(
        'Moved 15 nodes onto the finer mesh')
    # found again on the nodes as they now are: the one left out remains
    assert ('them. 1 node of the coarser mesh can move onto the finer; 1 marked,'
            in panel.reading_label.text())
    assert 'moved 15 nodes onto the finer mesh' in \
        window.statusBar().currentMessage()
    assert 'merge-done' in window.scene.plotter.actors
    assert window.project.journal[-1].startswith(
        "project.align_meshes('Geometry', ")
    geometry = window.objects['Geometry']
    assert len(geometry.coincident_spots(1e-9)) == 15
    panel.close_button.click()
    pump()
    window.tree.clearSelection()
    window.tree.setCurrentItem(group_row(window, 1))
    group_row(window, 2).setSelected(True)
    pump()
    {act[0]: act for act in window.acts_for()}['merge_coincident_nodes'][3]()
    pump()
    assert panel.title.text() == 'Merge Coincident Nodes'
    assert panel.merge_button.text() == 'Merge'
    # the move left out sits 0.055 in off its node, inside the merge's
    # quarter-spacing tolerance: marked there too, to leave out again
    assert panel.reading_label.text().startswith('16 places where nodes coincide')


def test_a_refused_align_says_why_in_the_pane(window, pump):
    geometry = _wing_on_frame()
    mover = geometry.align_moves(geometry.align_reach())[0][0]
    window.add_object('Modes', ShapeSet([10.0], [0.01], [f'{mover}Z+'], [[1.0]]))
    panel = _align_pane(window, pump, geometry)
    window.project.link('Geometry', 'Modes')
    pump()
    panel.merge_button.click()
    pump()
    assert panel.reading_label.text().startswith(
        f'Not aligned: modes names node {mover}, which aligning would move')


def test_the_frame_goes_onto_a_washer_and_nothing_else_moves():
    """The Frame-Wing's joint (2026-10-11): a wing on a 0.028 in washer,
    meshed together at an eighth of an inch, kept apart from a frame at
    a twelfth. The washer is ring-meshed about its circle, so its bricks
    are the finer (first the washer moved, when its rim was the grid
    pulled round), and the frame is the piece that moves: only its top,
    onto the washer's underside — the frame's next layer down has the
    top's nodes nearer it, taken first. Then every place merges."""
    import visualdynamics

    project = visualdynamics.Project('p')
    name = project.new_geometry('Joint', unit='in')
    frame = project.add_solid_block(name, (0, 0, -0.25), (1, 1, 0.5), name='frame',
                                    unit='in')
    project.mesh_solids(name, [frame], 1 / 12, unit='in')
    wing = project.add_solid_block(name, (0.03, 0.02, 0.028 + 0.125), (1, 1, 0.25),
                                   name='wing', unit='in')
    washer = project.add_solid_cylinder(name, (0.03, 0.02, 0.014), 0.196, 0.028,
                                        name='washer', unit='in')
    project.mesh_solids(name, [wing, washer], 1 / 8, unit='in', apart=True)
    geometry = project[name]
    ids = _ids(geometry)
    picked = [ids['frame'], ids['washer']]
    moves = geometry.align_moves(geometry.align_reach(picked), picked)
    inch = geometry.node_xyz / INCH
    washer_nodes = set(geometry.nodes_in([ids['washer']]))
    assert moves and all(a not in washer_nodes and b in washer_nodes for a, b in moves)
    assert np.allclose([inch[geometry.node_index([a])[0], 2] for a, _b in moves], 0.0), \
        "only the frame's top"
    # a reach long enough for the frame's next layer to reach the
    # washer: it still stays, its nearest washer node the top's, taken
    # first
    wide = geometry.align_moves(0.12 * INCH, picked)
    assert np.allclose([inch[geometry.node_index([a])[0], 2] for a, _b in wide], 0.0)
    found = project.align_meshes(name, groups=picked)
    assert found['moved'] == len(moves)
    merged = project.merge_coincident_nodes(name, 1e-9, groups=picked)
    assert merged['merged'] == len(moves)
