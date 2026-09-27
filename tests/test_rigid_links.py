"""Rigid, massless links: two nodes held rigidly together, adding no mass
(Brandon, 2026-09-26 — a bolt joining two plates whose mid-surfaces do
not meet, in a model built from planes).

Not a very stiff beam — that makes the stiffness matrix ill-conditioned
— but an exact constraint: each group of linked nodes moves as its first
node does, and the eigensolution eliminates the rest. The proofs: a
zero-length link is a shared node, exactly; a link is the limit a beam
approaches as it stiffens; every mode obeys the rigid-body kinematics at
every link; and a link adds no mass.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_fem import ALUMINUM

import visualdynamics
from visualdynamics.core.fem import RIGID, BlockProperties, Model, Section, material


def _strip(model, first, nx=6, ny=2, x0=0.0, z=0.0, length=0.3,
           width=0.1, thickness=0.004, group=''):
    """A rectangular strip of plates, nodes numbered from `first`."""
    ids = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            node = first + j * (nx + 1) + i
            model.add_node(node, x0 + length * i / nx, width * j / ny, z,
                           group=group)
            ids[(i, j)] = node
    for j in range(ny):
        for i in range(nx):
            model.add_plate([ids[(i, j)], ids[(i + 1, j)], ids[(i + 1, j + 1)],
                             ids[(i, j + 1)]], ALUMINUM, thickness,
                            group=group)
    return ids


def test_a_zero_length_link_is_a_shared_node():
    """Two strips meeting end to end: one mesh sharing the seam's nodes,
    and one with each strip its own nodes on the seam, linked there. The
    same structure, to rounding."""
    shared = Model('shared')
    _strip(shared, 1, nx=12, length=0.6)
    linked = Model('linked')
    left = _strip(linked, 1, nx=6, length=0.3)
    right = _strip(linked, 101, nx=6, x0=0.3, length=0.3)
    for j in range(3):
        linked.add_rigid_link(left[(6, j)], right[(0, j)])
    a = shared.eigensolution(num_modes=14).frequency
    b = linked.eigensolution(num_modes=14).frequency
    assert np.allclose(a, b, rtol=1e-9, atol=1e-6)
    assert len(linked.pieces()) == 1


def test_a_link_is_what_a_stiffening_beam_approaches():
    """Two plates stacked 20 mm apart, joined at four points: by rigid
    links, and by steel posts made a thousand times stiffer and a
    million times lighter. The posts approach the links."""
    def stacked(join):
        model = Model()
        lower = _strip(model, 1, nx=6, ny=2)
        upper = _strip(model, 101, nx=6, ny=2, z=0.02)
        for i, j in ((0, 0), (6, 0), (0, 2), (6, 2)):
            join(model, lower[(i, j)], upper[(i, j)])
        return model

    links = stacked(lambda m, a, b: m.add_rigid_link(a, b))
    stiff = material('1018 steel')
    stiff = type(stiff)('post', stiff.youngs_modulus * 1e3, 1e-3, 0.3)
    posts = stacked(lambda m, a, b: m.add_beam(
        a, b, stiff, Section.rod('post', 0.006), (1.0, 0.0, 0.0)))
    f_links = links.eigensolution(num_modes=16).frequency[6:]
    f_posts = posts.eigensolution(num_modes=16).frequency[6:]
    assert np.allclose(f_posts, f_links, rtol=5e-3)
    assert np.all(f_posts <= f_links * (1 + 1e-9)), 'a beam is never stiffer'


def test_every_mode_obeys_the_links_kinematics():
    """At each link, the follower's rotation is its lead's, and its
    translation is the lead's plus the lead's rotation crossed with the
    arm between them — in every mode, exactly."""
    model = Model()
    lower = _strip(model, 1)
    upper = _strip(model, 101, z=0.03)
    pairs = [(lower[(1, 1)], upper[(2, 0)]), (lower[(5, 1)], upper[(4, 2)])]
    for a, b in pairs:
        model.add_rigid_link(a, b)
    shapes = model.eigensolution(num_modes=12).shape_matrix
    index = {node: 6 * i for i, node in enumerate(model.node_ids)}
    for a, b in pairs:
        arm = model.position(b) - model.position(a)
        for phi in shapes:
            ua, ta = phi[index[a]:index[a] + 3], phi[index[a] + 3:index[a] + 6]
            ub, tb = phi[index[b]:index[b] + 3], phi[index[b] + 3:index[b] + 6]
            assert np.allclose(tb, ta, atol=1e-10)
            assert np.allclose(ub, ua + np.cross(ta, arm), atol=1e-10)


def test_a_link_adds_no_mass_and_joins_the_pieces():
    model = Model()
    lower = _strip(model, 1)
    upper = _strip(model, 101, z=0.03)
    before = model.structural_mass
    assert len(model.pieces()) == 2
    model.add_rigid_link(lower[(0, 0)], upper[(0, 0)])
    model.add_rigid_link(lower[(6, 2)], upper[(6, 2)])
    assert model.structural_mass == before
    assert len(model.pieces()) == 1
    shapes = model.eigensolution(num_modes=8)
    assert int(np.sum(shapes.frequency == 0.0)) == 6, 'one body, six'


def test_chained_links_are_one_rigid_body_led_by_the_first_node():
    model = Model()
    for node in (5, 3, 9, 7):
        model.add_node(node, float(node), 0.0, 0.0)
    model.add_rigid_link(9, 3)
    model.add_rigid_link(7, 9)
    assert model.rigid_bodies() == [[3, 9, 7]], 'in the model order'


def test_a_follower_cannot_be_grounded_alone():
    model = Model()
    lower = _strip(model, 1)
    upper = _strip(model, 101, z=0.03)
    model.add_rigid_link(lower[(0, 0)], upper[(0, 0)])
    with pytest.raises(ValueError, match='node 101 follows a rigid link; '
                                          'ground the node it follows'):
        model.eigensolution(num_modes=4, fixed=['101'])
    # grounding the lead grounds the body
    model.eigensolution(num_modes=4, fixed=[str(lower[(0, 0)])])


def test_links_are_refused_where_they_mean_nothing():
    model = Model()
    model.add_node(1, 0, 0, 0)
    with pytest.raises(ValueError, match='names node 1 twice'):
        model.add_rigid_link(1, 1)
    with pytest.raises(ValueError, match='names node 2, which is not'):
        model.add_rigid_link(1, 2)


def test_a_rigid_block_builds_links_and_the_file_keeps_it(tmp_path):
    """Picked as a block's material, a rigid link needs no section; its
    two-node lines become links, and a block of faces is refused by name.
    The geometry a model exports puts its links in a block of their own."""
    model = Model('stack')
    lower = _strip(model, 1, group='lower')
    upper = _strip(model, 101, z=0.03, group='upper')
    model.add_rigid_link(lower[(0, 0)], upper[(0, 0)], group='bolts')
    model.add_rigid_link(lower[(6, 2)], upper[(6, 2)], group='bolts')
    geometry = model.geometry()
    names = list(geometry.block_name)
    assert 'bolts' in names
    ids = {name: int(geometry.block_id[i]) for i, name in enumerate(names)}
    geometry.block_properties = {
        ids['lower']: BlockProperties(ALUMINUM, 0.004),
        ids['upper']: BlockProperties(ALUMINUM, 0.004),
        ids['bolts']: BlockProperties(RIGID, section=Section.rod('left', 0.01))}
    assert geometry.block_properties[ids['bolts']].kind == 'rigid', (
        'a section left from before the pick is ignored')
    rebuilt = Model.from_geometry(geometry)
    assert len(rebuilt.rigid_links) == 2 and not rebuilt.beams
    assert np.allclose(rebuilt.eigensolution(num_modes=10).frequency,
                       model.eigensolution(num_modes=10).frequency,
                       rtol=1e-9, atol=1e-6)
    visualdynamics.save(geometry, tmp_path / 'stack.vdyn')
    back = visualdynamics.load(tmp_path / 'stack.vdyn')
    assert back.block_properties[ids['bolts']].material.is_rigid
    geometry.block_properties[ids['lower']] = BlockProperties(RIGID)
    with pytest.raises(ValueError, match=r'a rigid \(massless\) block holds '
                                          'two-node lines'):
        Model.from_geometry(geometry)
