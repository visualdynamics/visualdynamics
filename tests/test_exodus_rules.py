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
                  ('TRUSS', 'beams'), ('SPRING', 'beams'), ('SPHERE', 'points'))


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


def test_each_element_is_written_as_an_exodus_type_of_its_shape_and_count():
    for code, (name, count) in exodus._EXODUS_TYPES.items():
        assert element_family(code) == _family(name), code
        assert ELEMENT_TYPES[code][1] == count == _count(name), code
        assert exodus._ELEM_TYPES[name] in (code, {11: 21, 22: 21, 24: 23, 41: 91,
                                                   42: 92, 44: 94, 45: 95}.get(code)), (
            f'{name} reads back as what it was written from, or its plate twin')


def test_frames_are_tagged_rectangular_cylindrical_spherical():
    assert exodus._FRAME_TAGS == {0: 'R', 1: 'C', 2: 'S'}
