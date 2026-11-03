"""Exodus's element-type tables against the element vocabulary
(2026-10-11).

`tools/mutate.py` over exodus.py caught 184 of 427 against its two
result and element-type test files; 79 survivors were these tables,
where a code one off is a different element of the same kind or of
none. Each exodus name says its shape and, where it has digits, its
node count; each written type must name the elements it carries.
"""

from __future__ import annotations

import re

from visualdynamics.core.geometry import ELEMENT_TYPES, element_family
from visualdynamics.io import exodus

FAMILY_OF_NAME = (('TRISHELL', 'triangles'), ('TRIANGLE', 'triangles'), ('TRI', 'triangles'),
                  ('QUAD', 'quads'), ('SHELL', 'quads'), ('TETRA', 'tetras'),
                  ('TET', 'tetras'), ('WEDGE', 'wedges'), ('HEX', 'hexes'),
                  ('PYRAMID', 'pyramids'), ('BAR', 'beams'), ('BEAM', 'beams'),
                  ('TRUSS', 'beams'), ('SPRING', 'beams'), ('SPHERE', 'points'),
                  ('CIRCLE', 'points'))


def _family(name):
    return next(family for prefix, family in FAMILY_OF_NAME if name.startswith(prefix))


def _count(name):
    digits = re.search(r'\d+$', name)
    return int(digits.group()) if digits else None


def test_each_exodus_name_reads_as_an_element_of_its_shape_and_count():
    """A nine-node quad is read as the eight-node one, its center node
    dropped; every other count is the element's own."""
    for name, code in exodus._ELEM_TYPES.items():
        assert element_family(code) == _family(name), name
        count = _count(name)
        if count is not None:
            assert ELEMENT_TYPES[code][1] == (8 if count == 9 else count), name


def test_a_circle_block_reads_as_points(tmp_path):
    """A CIRCLE block, the one-node element some models carry their
    lumped masses in, reads as a point on its node."""
    import netCDF4

    path = tmp_path / 'circles.exo'
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('num_dim', 3)
        ds.createDimension('num_nodes', 2)
        ds.createDimension('num_el_blk', 1)
        ds.createDimension('num_el_in_blk1', 2)
        ds.createDimension('num_nod_per_el1', 1)
        for axis, values in zip('xyz', ([0, 1], [0, 0], [0, 0])):
            ds.createVariable(f'coord{axis}', 'f8', ('num_nodes',))[:] = values
        ds.createVariable('eb_prop1', 'i4', ('num_el_blk',))[:] = [38]
        block = ds.createVariable('connect1', 'i4', ('num_el_in_blk1', 'num_nod_per_el1'))
        block.elem_type = 'CIRCLE'
        block[:] = [[1], [2]]
    geometry = exodus.load(path)
    assert [int(t) for t in geometry.elem_type] == [161, 161]
    assert [list(c) for c in geometry.elem_conn] == [[1], [2]]


def test_each_element_is_written_as_an_exodus_type_of_its_shape_and_count():
    for code, (name, count) in exodus._EXODUS_TYPES.items():
        assert element_family(code) == _family(name), code
        assert ELEMENT_TYPES[code][1] == count == _count(name), code
        assert exodus._ELEM_TYPES[name] in (code, {11: 21, 22: 21, 24: 23, 41: 91,
                                                   42: 92, 44: 94, 45: 95}.get(code)), (
            f'{name} reads back as what it was written from, or its plate twin')


def test_frames_are_tagged_rectangular_cylindrical_spherical():
    assert exodus._FRAME_TAGS == {0: 'R', 1: 'C', 2: 'S'}
