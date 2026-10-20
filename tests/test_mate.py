"""Mate Surfaces (`core.mates`): where element groups' surfaces mate,
found from the meshes' outer faces a patch at a time, kept on the
geometry as sets of mated surfaces, and tied when a model is built:
each node of the first side to the point of the second side's surface
it lies on, along the surface's normal there and in its plane, each
held rigidly, by a stiffness, or not at all (`fem.Model.add_tie`).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from visualdynamics.core import mates, mesh
from visualdynamics.core.fem import MATERIALS, GroupProperties, Model

INCH = 0.0254


def _project(*blocks, size=0.5, sizes=None):
    """A project whose geometry holds these blocks, each (center,
    widths, name), each meshed apart at its own size."""
    import visualdynamics

    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    for k, (center, widths, label) in enumerate(blocks):
        solid = project.add_solid_block(name, center, widths, name=label, unit='in')
        project.mesh_solids(name, [solid], (sizes or {}).get(label, size), unit='in',
                            apart=k > 0)
    return project, name, project[name]


def _two_blocks(size_top=0.5):
    """A 4 x 2 x 1 in base and a 1 x 2 x 1 in post standing on it (blocks
    are placed by center), meshed apart, the post at `size_top`."""
    return _project(((0, 0, 0), (4, 2, 1), 'base'), ((1.5, 0, 1), (1, 2, 1), 'post'),
                    sizes={'post': size_top})


def _group(geometry, name):
    return int(geometry.group_id[list(geometry.group_name).index(name)])


def _steel(geometry):
    for group in geometry.group_id:
        geometry.group_properties[int(group)] = GroupProperties(MATERIALS['1018 steel'])
    return geometry


def _frequencies(geometry, count=10):
    return np.asarray(Model.from_geometry(geometry).eigensolution(num_modes=count).frequency)


def _mated(geometry, normal_mode='rigid', shear_mode='rigid', k=0.0, tolerance=None):
    """A copy of `geometry` with every patch found mated as asked."""
    held = copy.deepcopy(geometry)
    for patch in mates.detect(held, list(held.group_id), tolerance):
        mates.add(held, patch['groups'], patch['nodes_a'], patch['nodes_b'],
                  normal_mode, shear_mode, k, k)
    return held


def test_the_skin_of_a_block_of_bricks_is_its_six_faces():
    _built, _name, geometry = _two_blocks()
    faces = mesh.group_skin(geometry, 'base')
    assert len(faces) == 2 * (8 * 4 + 8 * 2 + 4 * 2)
    assert all(len(face) == 4 for face in faces)


def test_the_post_on_the_base_is_one_patch_the_finer_side_first():
    _built, _name, geometry = _two_blocks(size_top=0.25)
    patch, = mates.detect(geometry, [_group(geometry, 'base'), _group(geometry, 'post')])
    assert patch['groups'] == (_group(geometry, 'post'), _group(geometry, 'base'))
    # the post's 5 x 9 bottom nodes, on the base's 3 x 5 nodes under it
    assert len(patch['nodes_a']) == 5 * 9 and len(patch['nodes_b']) == 3 * 5
    z = geometry.node_xyz[geometry.node_index(list(patch['nodes_a'])), 2]
    assert np.allclose(z, 0.5 * INCH), "the base's top"


def test_a_leg_flush_with_the_edges_takes_no_node_off_the_surface():
    """A leg under the base, flush with its side and both its ends: the
    leg's edge nodes lie on the base's bottom and on its side faces at
    once, and the side faces' corners up the wall were taken into the
    second side (Brandon, 2026-10-20: nodes that shouldn't be). The face
    a node mates with faces it."""
    _built, _name, geometry = _project(((0, 0, 0), (4, 2, 1), 'base'),
                                          ((1.5, 0, -1), (1, 2, 1), 'leg'),
                                          sizes={'leg': 0.25})
    patch, = mates.detect(geometry, list(geometry.group_id))
    for side in ('nodes_a', 'nodes_b'):
        z = geometry.node_xyz[geometry.node_index(list(patch[side])), 2]
        assert np.allclose(z, -0.5 * INCH), side
    assert len(patch['nodes_a']) == 5 * 9 and len(patch['nodes_b']) == 3 * 5
    # and every tie is along the bottom's normal, the edges' too: one on
    # the side's would let a free shear slide the leg up the wall
    mated = mates.add(geometry, patch['groups'], patch['nodes_a'], patch['nodes_b'])
    normals = np.array([n for _b, _w, _node, n, _p in mates.links(geometry, mated)])
    assert len(normals) == 5 * 9 and np.allclose(np.abs(normals[:, 2]), 1.0)


def test_a_point_is_carried_by_the_corners_around_it_alone():
    """A point on a face's corner is that corner's, on an edge the two
    ends', inside a triangle its three: a corner of no weight is not on
    the second side, or a node on an edge between two faces would name
    corners of the face beyond it (`_support`)."""
    from visualdynamics.core.geometry import Geometry

    square = Geometry(node_id=[1, 2, 3, 4],
                      node_xyz=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    face = (1, 2, 3, 4)
    for point, nodes in (((0.0, 0.0, 0.0), (1,)), ((0.5, 0.0, 0.0), (1, 2)),
                         ((0.75, 0.25, 0.0), (1, 2, 3))):
        found, weights, on = mates._support(square, np.array(point), face)
        assert sorted(found) == list(nodes), point
        assert np.isclose(sum(weights), 1.0) and np.allclose(on, point)


def test_four_legs_on_a_table_are_four_patches():
    """One group of four legs standing on a top: four patches, one per
    leg (Brandon, 2026-10-20: one entry per contiguous patch)."""
    import visualdynamics

    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    top = project.add_solid_block(name, (0, 0, 2), (4, 4, 0.5), name='top', unit='in')
    legs = [project.add_solid_block(name, (x, y, 1.0), (0.5, 0.5, 1.5), name='legs',
                                    unit='in') for x in (-1.5, 1.5) for y in (-1.5, 1.5)]
    project.mesh_solids(name, [top], 0.5, unit='in')
    project.mesh_solids(name, legs, 0.25, unit='in', apart=True)
    geometry = project[name]
    found = mates.detect(geometry, [_group(geometry, 'top'), _group(geometry, 'legs')])
    assert len(found) == 4 and all(len(p['nodes_a']) == 3 * 3 for p in found)
    centers = sorted(tuple(np.round(geometry.node_xyz[geometry.node_index(
        list(p['nodes_a']))].mean(axis=0)[:2] / INCH, 3)) for p in found)
    assert centers == [(-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5)]


def test_groups_apart_have_nothing_to_mate():
    project, name, geometry = _project(((0, 0, 0), (1, 1, 1), 'a'), ((3, 0, 0), (1, 1, 1), 'b'))
    assert mates.detect(geometry) == []
    with pytest.raises(ValueError, match='no surface'):
        project.mate_surfaces(name)


def test_rigid_in_both_is_the_seams_nodes_merged():
    """Tied rigidly along the normal and in the plane, two meshes whose
    seam nodes coincide are the same structure as with those nodes
    merged, mode for mode; held by a far stiffer spring, nearly so; and
    no elements are added to the geometry."""
    _built, _name, geometry = _two_blocks()
    _steel(geometry)
    merged = copy.deepcopy(geometry)
    merged.merge_coincident_nodes(1e-6)
    expected = _frequencies(merged)
    tied = _mated(geometry)
    assert len(tied.elem_conn) == len(geometry.elem_conn)
    assert np.allclose(_frequencies(tied), expected, rtol=1e-9, atol=1e-6)
    stiff = _frequencies(_mated(geometry, 'stiffness', 'stiffness', 1e13))
    assert np.allclose(stiff[6:], expected[6:], rtol=1e-4)
    assert expected[6] > 1000.0


def test_a_free_direction_frees_what_it_should():
    """Free in the plane, the post slides on the base and turns about
    the normal: three more modes at zero. Free along the normal with a
    shear stiffness, it lifts off and rocks: three more again."""
    _built, _name, geometry = _two_blocks()
    _steel(geometry)
    for held in (('rigid', 'free', 0.0), ('free', 'stiffness', 1e8)):
        found = _frequencies(_mated(geometry, *held))
        assert int(np.sum(found < 1.0)) == 9, held


def test_the_frame_is_the_surfaces_own_on_a_slant():
    """The same model turned 20, 35 and 10 degrees: every mode's
    frequency is the same, each combination of the normal and the
    shear, the seam's normal along no axis."""
    from scipy.spatial.transform import Rotation

    _built, _name, geometry = _two_blocks()
    _steel(geometry)
    turned = copy.deepcopy(geometry)
    turned.node_xyz = turned.node_xyz @ Rotation.from_euler(
        'xyz', [20, 35, 10], degrees=True).as_matrix().T
    for held in (('rigid', 'rigid', 0.0), ('rigid', 'free', 0.0),
                 ('stiffness', 'rigid', 1e8), ('free', 'stiffness', 1e8)):
        level = _frequencies(_mated(geometry, *held, tolerance=1e-9))
        slant = _frequencies(_mated(turned, *held, tolerance=1e-9))
        assert np.allclose(level, slant, rtol=1e-7, atol=1e-4), held


def _rod_in_sleeve():
    """A 1 in rod in a 2 in sleeve, meshed apart at 0.25 in: 16 nodes
    round the rod, 32 round the sleeve's bore, so half the bore's nodes
    sit a chord's sag off the rod's facets."""
    import visualdynamics

    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    rod = project.add_solid_cylinder(name, (0, 0, 0), 0.5, 2.0, name='rod', unit='in')
    outer = project.add_solid_cylinder(name, (0, 0, 0), 1.0, 2.0, unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0), 0.5, 2.0, unit='in')
    sleeve = project.combine_solids(name, 'subtract', [outer, hole], 'sleeve')
    project.mesh_solids(name, [sleeve], 0.25, unit='in')
    project.mesh_solids(name, [rod], 0.25, unit='in', apart=True)
    return _steel(project[name])


def test_a_shaft_in_its_bore_is_tied_along_each_nodes_own_normal():
    """On a curved surface each node is tied along the surface's normal
    where it lies, here radial: rigid in both, the shaft is one with the
    sleeve, six free modes; free in the plane, it also turns and slides
    in the bore, eight. A tie to the nearest node, or across the chord's
    sag, held the whole model against turning (2026-10-20)."""
    geometry = _rod_in_sleeve()
    tolerance = 0.02 * INCH
    patch, = mates.detect(geometry, list(geometry.group_id), tolerance)
    assert len(patch['nodes_a']) == 32 * 9
    rigid = _mated(geometry, tolerance=tolerance)
    ties = mates.links(rigid, rigid.mates[0])
    xyz = rigid.node_xyz[rigid.node_index([node for _b, _w, node, _n, _p in ties])]
    radial = xyz[:, :2] / np.linalg.norm(xyz[:, :2], axis=1)[:, None]
    normals = np.array([n for _b, _w, _node, n, _p in ties])
    assert np.allclose(np.abs(np.einsum('ij,ij->i', normals[:, :2], radial)), 1.0)
    assert np.allclose(normals[:, 2], 0.0)
    # half the bore's nodes lie between two of the rod's: a point of a
    # facet, two nodes weighted
    assert sum(len(nodes) > 1 for nodes, *_rest in ties) == 16 * 9
    assert int(np.sum(_frequencies(rigid, 12) < 1.0)) == 6
    sliding = _frequencies(_mated(geometry, 'rigid', 'free', tolerance=tolerance), 12)
    assert int(np.sum(sliding < 1.0)) == 8


def test_a_set_is_refused_when_it_would_hold_nothing():
    project, name, geometry = _two_blocks(size_top=0.25)
    patch, = project.detect_mated_surfaces(name, ['post', 'base'])
    with pytest.raises(ValueError, match='holds nothing'):
        project.add_mated_surfaces(name, patch['groups'], patch['nodes_a'], patch['nodes_b'],
                                   normal_mode='free', shear_mode='free')
    with pytest.raises(ValueError, match='normal stiffness a value above zero'):
        project.add_mated_surfaces(name, patch['groups'], patch['nodes_a'], patch['nodes_b'],
                                   normal_mode='stiffness')
    assert not geometry.mates


def test_a_spring_sets_stiffness_is_the_ties_and_edits_in_place():
    project, name, geometry = _two_blocks(size_top=0.25)
    patch, = project.detect_mated_surfaces(name, ['post', 'base'])
    mate_id = project.add_mated_surfaces(name, patch['groups'], patch['nodes_a'],
                                         patch['nodes_b'], normal_mode='stiffness',
                                         shear_mode='stiffness', normal_stiffness=2e6,
                                         shear_stiffness=5e5, name='bolts')
    assert mates.find(geometry, mate_id).describe() == 'stiffness'
    model = Model.from_geometry(_steel(geometry))
    assert len(model.ties) == 45 and not model.rigid_links and not model.springs
    assert {tie.stiffness for tie in model.ties} == {(2e6, 5e5, 5e5)}
    project.edit_mated_surfaces(name, mate_id, normal_mode='rigid', shear_mode='free')
    assert mates.find(geometry, mate_id).describe() == 'normal rigid, shear free'
    assert {tie.modes for tie in Model.from_geometry(geometry).ties} == {
        ('rigid', 'free', 'free')}
    assert project.delete_mated_surfaces(name, [mate_id]) == 1 and not geometry.mates


def test_a_node_following_two_ties_is_refused_by_name():
    model = Model('m')
    for k, xyz in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0)], start=1):
        model.add_node(k, *xyz)
    model.add_tie(1, 2, (0, 0, 1), group='first')
    model.add_tie(1, 3, (0, 0, 1), group='second')
    with pytest.raises(ValueError, match='node 1 follows two ties .first and second.'):
        model.constraint_transform()


def test_a_set_follows_its_nodes_and_groups():
    project, name, geometry = _two_blocks(size_top=0.25)
    mate_id, = project.mate_surfaces(name, ['post', 'base'])
    first = mates.find(geometry, mate_id).nodes_a[0]
    geometry.renumber_node(int(geometry.node_index([first])[0]), 99999)
    assert 99999 in mates.find(geometry, mate_id).nodes_a
    post = _group(geometry, 'post')
    geometry.renumber_group(list(geometry.group_id).index(post), 77)
    assert mates.find(geometry, mate_id).groups[0] == 77
    # merged: the post's seam nodes onto the base's where they coincide,
    # which are one node now and make no tie; the rest still do
    geometry.merge_coincident_nodes(1e-6, groups=[77, _group(geometry, 'base')])
    mated = mates.find(geometry, mate_id)
    assert len(set(mated.nodes_a) - set(mated.nodes_b)) == 45 - 15
    assert len(mates.links(geometry, mated)) == 45 - 15
    geometry.delete_groups([77])
    assert not geometry.mates


def test_a_set_naming_a_node_gone_is_refused_by_name_at_the_build():
    _built, _name, geometry = _two_blocks()
    _steel(geometry)
    geometry.mates.append(mates.MatedSet(1, 'stale', (1, 2), (999999,),
                                         (int(geometry.node_id[0]),)))
    with pytest.raises(ValueError, match="mated surfaces 'stale' name node 999999"):
        Model.from_geometry(geometry)


def test_a_set_is_saved_and_read_back_and_an_older_one_reads(tmp_path):
    from visualdynamics.core.geometry import Geometry

    project, name, geometry = _two_blocks(size_top=0.25)
    patch, = project.detect_mated_surfaces(name, ['post', 'base'])
    project.add_mated_surfaces(name, patch['groups'], patch['nodes_a'], patch['nodes_b'],
                               normal_mode='stiffness', shear_mode='free',
                               normal_stiffness=3e6)
    path = tmp_path / 'g.vdyn'
    geometry.save(path)
    assert Geometry.load(path).mates == geometry.mates
    # written the night before the normal and the shear were set apart
    older = {'id': 1, 'name': 'old', 'groups': [1, 2], 'nodes_a': [3], 'nodes_b': [4],
             'how': 'spring', 'normal_stiffness': 2.0, 'shear_stiffness': 1.0,
             'normal': [0.0, 0.0, 1.0]}
    read = mates.MatedSet.from_record(older)
    assert (read.normal_mode, read.shear_mode, read.normal_stiffness) == (
        'stiffness', 'stiffness', 2.0)
    assert mates.MatedSet.from_record({**older, 'how': 'rigid'}).describe() == 'rigid'


def _row(window, name, reference):
    """The tree row whose reference is `reference`, under geometry `name`."""
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    item = window._item_for_object(name)
    item.setExpanded(True)
    stack = [item.child(i) for i in range(item.childCount())]
    while stack:
        row = stack.pop()
        if row.data(0, ROLE_REFERENCE) == reference:
            row.parent().setExpanded(True)
            return row
        stack.extend(row.child(i) for i in range(row.childCount()))
    return None


def test_the_pane_on_two_groups_picks_nodes_keeps_edits_and_deletes(window, pump):
    """Mate Surfaces on two element groups picked in the tree: the patch
    listed; only the side being picked drawn, and only the nodes within
    the pick distance pickable, the one under the cursor lit; a click
    takes a node off and puts it back; Keep adds the set under the
    geometry's Mated Surfaces; its pencil opens the pane on it, where
    the normal is given a stiffness in lbf/in and the shear set free,
    and Save; Delete on the Mated Surfaces row removes every set
    (Brandon, 2026-10-20)."""
    from conftest import select_objects

    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    project = window.project
    name = project.new_geometry('Part', unit='in')
    base = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='base', unit='in')
    post = project.add_solid_block(name, (1.5, 0, 1), (1, 2, 1), name='post', unit='in')
    project.mesh_solids(name, [base], 0.5, unit='in')
    project.mesh_solids(name, [post], 0.25, unit='in', apart=True)
    geometry = project[name]
    window.show_object(name)
    pump()
    post_id, base_id = _group(geometry, 'post'), _group(geometry, 'base')
    first = _row(window, name, ('group', name, post_id))
    window.tree.setCurrentItem(first)
    pump()
    assert 'mate_surfaces' not in [act[0] for act in window.acts_for()]
    _row(window, name, ('group', name, base_id)).setSelected(True)
    pump()
    acts = {act[0]: act for act in window.acts_for()}
    acts['mate_surfaces'][3]()
    pump()
    panel = window.scene.mate_panel
    assert panel.isVisible() and panel.patches.count() == 1
    assert panel.patches.item(0).text() == 'post on base: 45 nodes on 15'
    assert panel.side_a.text() == 'post' and panel.side_b.text() == 'base'
    assert panel.normal_held == panel.shear_held == 'rigid' and not panel.normal.isVisible()
    assert panel.search_label.text() == 'Pick within [in]'
    actors = window.scene.plotter.renderer.actors
    # the post's side alone: no mark of the base's coincident nodes
    assert 'mate-a' in actors and 'mate-b' not in actors
    assert actors['mate-a'].GetMapper().GetInput().GetNumberOfPoints() == 45
    # pickable: the post's nodes on the base's top, its bottom face's 45
    # and no others; a wider band takes in the post's outer nodes a layer
    # up, its 24 round the sides (inner nodes are on no surface)
    assert sorted(window._mate_candidates(0, 'a')) == sorted(window._mating['patches'][0]['nodes_a'])
    panel.search.setValue(0.3)
    pump()
    assert len(window._mate_candidates(0, 'a')) == 45 + 24
    panel.search.setValue(0.0625)
    pump()
    patch = window._mating['patches'][0]
    node = patch['nodes_a'][0]
    window._mate_hover(node)
    assert 'mate-hover' in window.scene.plotter.renderer.actors
    window._mate_hover(None)
    assert 'mate-hover' not in window.scene.plotter.renderer.actors
    window.toggle_mate_node(node)
    assert panel.patches.item(0).text() == 'post on base: 44 nodes on 15'
    window.toggle_mate_node(node)
    # the base's side: its marks alone, and a node of its top beside the seam
    panel.side_b.click()
    pump()
    actors = window.scene.plotter.renderer.actors
    assert 'mate-b' in actors and 'mate-a' not in actors
    # within 0.0625 in of the post, the base's 15 under it alone; at
    # 0.6 in, its nodes a grid line beyond the post's sides too
    assert sorted(window._mate_candidates(0, 'b')) == sorted(patch['nodes_b'])
    panel.search.setValue(0.6)
    pump()
    # 31 of the base's outer nodes in the box the patch and 0.6 in
    # make; two of them, at its ends' mid-height, are 0.71 in from the
    # post and cannot be picked
    assert len(window._mate_candidates(0, 'b')) == 29
    extra = next(n for n in window._mate_candidates(0, 'b') if n not in patch['nodes_b'])
    window.toggle_mate_node(extra)
    assert panel.patches.item(0).text() == 'post on base: 45 nodes on 16'
    window.toggle_mate_node(extra)
    panel.keep_button.click()
    pump()
    mated, = geometry.mates
    assert mated.name == 'post on base' and mated.describe() == 'rigid'
    assert panel.isVisible() and panel.patches.count() == 0
    assert 'Kept post on base (rigid)' in panel.reading_label.text()
    holder = _row(window, name, ('mates', name, None))
    assert holder.text(0) == 'Mated Surfaces (1)'
    row = _row(window, name, ('mate', name, mated.id))
    assert row.text(0) == 'post on base (rigid)'
    window._tree_item_clicked(row, 1)
    pump()
    assert panel.isVisible() and panel.title.text() == 'Edit Mated Surfaces'
    assert not panel.tolerance.isVisible() and panel.search.isVisible()
    assert panel.keep_button.text() == 'Save'
    panel.normal_mode.setCurrentIndex(panel.normal_mode.findData('stiffness'))
    panel.shear_mode.setCurrentIndex(panel.shear_mode.findData('free'))
    pump()
    assert panel.normal.isVisible() and not panel.shear.isVisible()
    assert panel.normal_label.text() == 'Normal stiffness [lbf/in]'
    # 5710.1 lbf/in is 1e6 N/m
    panel.normal.setValue(5710.147154732645)
    panel.keep_button.click()
    pump()
    mated, = geometry.mates
    assert (mated.normal_mode, mated.shear_mode) == ('stiffness', 'free')
    assert mated.normal_stiffness == pytest.approx(1e6) and mated.shear_stiffness == 0.0
    assert _row(window, name, ('mate', name, mated.id)).text(0) == (
        'post on base (normal stiffness, shear free)')
    panel.close_button.click()
    pump()
    assert not panel.isVisible() and 'mate-hover' not in window.scene.plotter.renderer.actors
    select_objects(window, pump, name)
    holder = _row(window, name, ('mates', name, None))
    window.tree.setCurrentItem(holder)
    window.tree.clearSelection()
    holder.setSelected(True)
    pump()
    assert 'mate-a' in window.scene.plotter.renderer.actors, 'every set under it marked'
    window.delete_selected()
    pump()
    assert not geometry.mates and _row(window, name, ('mates', name, None)) is None


def _post_sharing_its_seam():
    """The post meshed onto the base, not apart: every other node of its
    5 x 9 seam lands on the base's 3 x 5 and is that node."""
    import visualdynamics

    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    base = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='base', unit='in')
    post = project.add_solid_block(name, (1.5, 0, 1), (1, 2, 1), name='post', unit='in')
    project.mesh_solids(name, [base], 0.5, unit='in')
    project.mesh_solids(name, [post], 0.25, unit='in')
    return project, name, project[name]


def test_a_node_both_groups_share_is_never_tied():
    """Where the meshes are one node already, the set ties the nodes
    between and not that one; added to the first side by hand, it still
    makes no tie (a tie naming it on both ends was refused at the build,
    2026-10-20)."""
    _built, _name, geometry = _post_sharing_its_seam()
    post, base = _group(geometry, 'post'), _group(geometry, 'base')
    shared = mates.shared(geometry, (post, base))
    assert len(shared) == 3 * 5
    patch, = mates.detect(geometry, [post, base])
    assert len(patch['nodes_a']) == 5 * 9 - 15 and not shared & set(patch['nodes_a'])
    # one shared node off the second side and onto the first by hand
    moved = min(shared)
    mated = mates.add(geometry, patch['groups'], (*patch['nodes_a'], moved),
                      [n for n in patch['nodes_b'] if n != moved])
    assert len(mates.links(geometry, mated)) == 5 * 9 - 15
    model = Model.from_geometry(_steel(geometry))
    assert len(model.ties) == 30 and len(model.pieces()) == 1


def test_the_pane_marks_the_shared_nodes_and_will_not_pick_them(window, pump):
    """The nodes the two groups share are drawn as squares on either
    side, said in the reading, and cannot be picked (Brandon, 2026-10-20:
    they looked left out of the set, being one node already)."""
    project = window.project
    name = project.new_geometry('Part', unit='in')
    base = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='base', unit='in')
    post = project.add_solid_block(name, (1.5, 0, 1), (1, 2, 1), name='post', unit='in')
    project.mesh_solids(name, [base], 0.5, unit='in')
    project.mesh_solids(name, [post], 0.25, unit='in')
    geometry = project[name]
    window.show_object(name)
    pump()
    window.mate_surfaces_act(name, [_group(geometry, 'post'), _group(geometry, 'base')])
    pump()
    panel = window.scene.mate_panel
    assert panel.patches.item(0).text() == 'post on base: 30 nodes on 15'
    shared = mates.shared(geometry, (_group(geometry, 'post'), _group(geometry, 'base')))
    actors = window.scene.plotter.renderer.actors
    assert actors['mate-shared'].GetMapper().GetInput().GetNumberOfPoints() == 15
    assert actors['mate-a'].GetMapper().GetInput().GetNumberOfPoints() == 30
    assert "15 nodes here are post's and base's both" in panel.reading_label.text()
    panel.search.setValue(0.6)
    pump()
    assert not shared & set(window._mate_candidates(0, 'a'))
    panel.side_b.click()
    pump()
    assert not shared & set(window._mate_candidates(0, 'b'))
    actors = window.scene.plotter.renderer.actors
    assert 'mate-shared' in actors and 'mate-b' not in actors, 'the base side is all shared'
