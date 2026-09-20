"""The element descriptors a universal file may carry.

UFF 2412 names the same triangle six ways — plane stress, plane
strain, plate, membrane, axisymmetric solid, thin shell — and the
table knew three of those families, so a mesh of plane-strain
quadrilaterals (descriptor 54) was refused outright (Brandon,
2026-09-20). What a face is *drawn* as does not depend on the family:
a triangle is three corners and a quadrilateral four.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.geometry import ELEMENT_TYPES, face_corners

#: (descriptor, name, nodes) the standard's plane families, linear
#: through cubic, written from the 2412 descriptor list
FAMILIES = {
    'plane stress': (41, 42, 43, 44, 45, 46),
    'plane strain': (51, 52, 53, 54, 55, 56),
    'plate': (61, 62, 63, 64, 65, 66),
    'membrane': (71, 72, 73, 74, 75, 76),
    'thin shell': (91, 92, 93, 94, 95, 96),
}


@pytest.mark.parametrize('family,codes', FAMILIES.items())
def test_every_plane_family_is_known(family, codes):
    """Three triangles then three quadrilaterals, linear, parabolic,
    cubic — the shape of every plane family in the standard."""
    assert all(code in ELEMENT_TYPES for code in codes), family
    counts = [ELEMENT_TYPES[code][1] for code in codes]
    assert counts == [3, 6, 9, 4, 8, 12], family
    assert [face_corners(code) for code in codes] == [3, 3, 3, 4, 4, 4], family
    assert all(ELEMENT_TYPES[code][2] == 'face' for code in codes), family


def test_the_axisymmetric_solids_are_known():
    """Four of them: the standard skips a cubic axisymmetric triangle
    and numbers the quadrilaterals 84 and 85."""
    assert [ELEMENT_TYPES[c][1] for c in (81, 82, 84, 85)] == [3, 6, 4, 8]
    assert [face_corners(c) for c in (81, 82, 84, 85)] == [3, 3, 4, 4]


def test_the_thick_shells_are_solid_cells():
    for code, nodes in ((101, 6), (102, 15), (104, 8), (105, 20)):
        name, count, render = ELEMENT_TYPES[code]
        assert (count, render) == (nodes, 'volume'), name


def test_a_plane_strain_quadrilateral_imports_and_draws():
    """Brandon's mesh: descriptor 54, refused before this."""
    geometry = visualdynamics.Geometry(
        node_id=[1, 2, 3, 4],
        node_xyz=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    geometry.add_element([1, 2, 3, 4], elem_type=54)
    assert int(geometry.elem_type[0]) == 54
    assert ELEMENT_TYPES[54][0] == 'quad4_strain'
    assert face_corners(54) == 4


def test_a_face_is_drawn_by_its_name_not_its_node_count():
    """A nine-node cubic triangle and a nine-node quadrilateral are
    not the same figure, and the node count alone cannot tell them
    apart — which is why the corners are read off the name."""
    assert ELEMENT_TYPES[43][1] == ELEMENT_TYPES[53][1] == 9
    assert face_corners(43) == 3, 'nine nodes, three corners'
    assert face_corners(46) == 4, 'twelve nodes, four corners'
    for code, (name, nodes, render) in ELEMENT_TYPES.items():
        if render != 'face':
            continue
        assert face_corners(code) <= nodes, name
        assert name.startswith(('tri', 'quad')), (
            f'{name} ({code}): a face element is named for its shape, '
            'which is how the viewer draws it')


def test_every_name_is_its_own():
    """The element type editor sets a type by name, so two codes
    sharing one name would make a cell that cannot be set."""
    names = [name for name, _, _ in ELEMENT_TYPES.values()]
    assert len(set(names)) == len(names)


def test_a_universal_file_of_plane_strain_quads_round_trips(tmp_path):
    from conftest import fixture_path

    geometry = visualdynamics.import_file(fixture_path('plate', 'geometry.unv'))
    geometry.elem_type[:] = 54
    path = tmp_path / 'strain.unv'
    visualdynamics.export_file(geometry, path, 'unv')
    back = visualdynamics.import_file(path)
    assert set(np.unique(back.elem_type)) == {54}
    assert len(back.elem_conn) == len(geometry.elem_conn)
    assert [len(c) for c in back.elem_conn] == [len(c) for c in geometry.elem_conn]


def test_the_line_families_are_known():
    """Pipes, rigid links and the axisymmetric shells: every one of
    them is drawn as the line between the two nodes it names first."""
    for code, name, nodes in ((31, 'pipe2', 2), (32, 'pipe3', 3),
                              (121, 'rigid_bar', 2), (122, 'rigid_element', 2),
                              (171, 'shell_axisym2', 2),
                              (172, 'shell_axisym3', 3)):
        assert ELEMENT_TYPES[code] == (name, nodes, 'line')


def test_the_solid_and_thick_shell_cubics_are_known():
    """A cubic cell carries two nodes on every edge: a wedge has nine
    edges and a brick twelve, so 6 + 18 and 8 + 24."""
    assert ELEMENT_TYPES[103] == ('wedge24_thick', 24, 'volume')
    assert ELEMENT_TYPES[106] == ('hex32_thick', 32, 'volume')
    assert ELEMENT_TYPES[114] == ('wedge24', 24, 'volume')


def test_springs_dampers_and_gaps_come_in_two_kinds():
    """Between two nodes it is a line; from a node to ground it is one
    node and a point. 136 is the exception on purpose — the Nastran
    reader writes a CELAS2 as a 136 at a single grid."""
    for code in (137, 141, 151):
        _name, nodes, render = ELEMENT_TYPES[code]
        assert (nodes, render) == (2, 'line'), code
    for code in (138, 139, 142, 152, 161):
        _name, nodes, render = ELEMENT_TYPES[code]
        assert (nodes, render) == (1, 'point'), code
    assert ELEMENT_TYPES[136] == ('spring', 1, 'point'), 'a CELAS2 at one grid'


def test_a_line_element_names_at_least_the_two_it_is_drawn_between():
    """The viewer draws a line from `conn[:2]`, so an entry claiming
    fewer than two nodes would make a cell of one point."""
    for code, (name, nodes, render) in ELEMENT_TYPES.items():
        if render == 'line':
            assert nodes >= 2, f'{name} ({code})'


def test_the_nastran_and_exodus_codes_still_mean_what_they_write():
    """Both readers emit descriptors by number; adding families must
    not move one out from under them."""
    from visualdynamics.io.exodus import _ELEM_TYPES

    assert ELEMENT_TYPES[161][2] == 'point', 'CONM2, a mass at a grid'
    assert ELEMENT_TYPES[201][0] == 'pyramid5'
    assert set(_ELEM_TYPES.values()) <= set(ELEMENT_TYPES)
