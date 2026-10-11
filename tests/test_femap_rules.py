"""Femap's topology table and output vector ids, stated from the
neutral file's own numbering (2026-10-11).

`tools/mutate.py` over femap.py caught 143 of 209 against
test_femap.py; 29 of the survivors were these two tables, which a
fixture exercises only for the shapes it happens to hold.
"""

from __future__ import annotations

from visualdynamics.core.geometry import ELEMENT_TYPES, element_family
from visualdynamics.io import femap

#: Femap topology -> (nodes, family): Line2, Line3, Tri3, Tri6, Quad4,
#: Quad8, Tetra4, Wedge6, Brick8, Point, Tetra10, Wedge15, Brick20
TOPOLOGIES = {0: (2, 'beams'), 1: (3, 'beams'), 2: (3, 'triangles'),
              3: (6, 'triangles'), 4: (4, 'quads'), 5: (8, 'quads'),
              6: (4, 'tetras'), 7: (6, 'wedges'), 8: (8, 'hexes'),
              9: (1, 'points'), 10: (10, 'tetras'), 11: (15, 'wedges'),
              12: (20, 'hexes')}


def test_each_topology_becomes_an_element_of_its_shape_and_node_count():
    assert set(femap._TOPOLOGY) == set(TOPOLOGIES)
    for topology, code in femap._TOPOLOGY.items():
        count, family = TOPOLOGIES[topology]
        assert (ELEMENT_TYPES[code][1], element_family(code)) == (count, family), topology


def test_the_topologies_that_break_the_stride_are_named():
    assert femap._STRIDE_BREAKERS == {13: 'Rigid', 15: 'MultiList', 16: 'Contact',
                                      17: 'Weld'}


def test_the_displacement_vectors_are_t1_t2_t3_then_r1_r2_r3():
    """Vector 1 is the total translation and 5 the total rotation,
    both derived; 2-4 and 6-8 are the components, in order."""
    assert {vector: femap._COMPONENTS[k]
            for vector, k in femap._VECTOR_COMPONENT.items()} == {
        2: 'X+', 3: 'Y+', 4: 'Z+', 6: 'RX+', 7: 'RY+', 8: 'RZ+'}
