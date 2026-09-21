"""A node is written in the frame it is placed in.

A source states a node's position in its *definition* frame, and the
frame is the node's `def_cs`. Read as though those numbers were
global, a node in a turned or cylindrical frame lands somewhere else
entirely — sdynpy drew the same geometry correctly and this package
did not (Brandon, 2026-09-20). `Geometry` holds global cartesian;
the readers resolve, the writers write it back.

The numbers below are sdynpy's own answer for the same geometry,
recorded by running it once: a program's output is data, and this
package computes the same thing from the standard's own definitions.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.geometry import placed, to_global, to_local, written_in

TURNED = np.radians(30.0)
#: three frames: global, a cartesian turned 30° about z and lifted 5,
#: and a cylindrical one at the origin
FRAMES = (
    [1, 2, 3],
    [0, 0, 1],
    np.array([np.eye(4, 3),
              [[np.cos(TURNED), np.sin(TURNED), 0],
               [-np.sin(TURNED), np.cos(TURNED), 0],
               [0, 0, 1], [0, 0, 5]],
              np.eye(4, 3)], dtype=float),
)
#: a node in each, as its own frame writes it
WRITTEN = np.array([[1, 0, 0], [1, 0, 0], [2, 90, 3]], dtype=float)
#: and where sdynpy puts them
SDYNPY = np.array([[1.0, 0.0, 0.0], [0.866025, 0.5, 5.0], [0.0, 2.0, 3.0]])


def test_the_frames_place_their_nodes_where_sdynpy_does():
    assert to_global(WRITTEN, [1, 2, 3], *FRAMES) == pytest.approx(SDYNPY, abs=1e-6)


def test_one_point_at_a_time():
    """A cylindrical frame writes (r, θ°, z) and a spherical one
    (r, θ° from +z, φ°) — degrees, as the universal file, a Nastran
    CORD2C/S and sdynpy all mean them."""
    about_x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [10, 0, 0]], dtype=float)
    assert placed([1, 90, 0], 1, about_x) == pytest.approx([10, 1, 0])
    identity = np.eye(4, 3)
    assert placed([1, 90, 0], 2, identity) == pytest.approx([1, 0, 0], abs=1e-9)
    assert placed([0, 0, 1], 2, identity) == pytest.approx([0, 0, 0], abs=1e-9)
    assert placed([2, 3, 4], 0, identity) == pytest.approx([2, 3, 4])


def test_a_global_frame_leaves_a_node_where_it_was():
    plain = to_global(WRITTEN, [1, 1, 1], *FRAMES)
    assert plain == pytest.approx(WRITTEN), 'the identity frame changes nothing'
    unknown = to_global(WRITTEN, [9, 9, 9], *FRAMES)
    assert unknown == pytest.approx(WRITTEN), 'a frame nobody defined: as written'


def test_the_inverse_puts_it_back():
    """A writer states each node the way the geometry says it is
    placed, or the file would carry global coordinates under a local
    frame and the next reader would resolve them a second time."""
    back = to_local(to_global(WRITTEN, [1, 2, 3], *FRAMES), [1, 2, 3], *FRAMES)
    assert back == pytest.approx(WRITTEN, abs=1e-9)
    about_x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [10, 0, 0]], dtype=float)
    for kind in (0, 1, 2):
        point = placed([2, 35, 1.5], kind, about_x)
        assert written_in(point, kind, about_x) == pytest.approx([2, 35, 1.5], abs=1e-9)


def _archive(path):
    """An sdynpy geometry archive holding those three nodes."""
    from visualdynamics.io.sdynpy_npz import CS_DTYPE, NODE_DTYPE

    node = np.zeros(3, NODE_DTYPE)
    node['id'] = [1, 2, 3]
    node['coordinate'] = WRITTEN
    node['def_cs'] = node['disp_cs'] = [1, 2, 3]
    cs = np.zeros(3, CS_DTYPE)
    cs['id'] = FRAMES[0]
    cs['cs_type'] = FRAMES[1]
    cs['matrix'] = FRAMES[2]
    np.savez(path, node=node, coordinate_system=cs)
    return path


def test_an_sdynpy_archive_places_its_nodes(tmp_path):
    """The case itself, and the reason a resolve must come before the
    unit scale: an angle is not a length."""
    geometry = visualdynamics.import_file(_archive(tmp_path / 'framed.npz'))
    assert geometry.node_xyz == pytest.approx(SDYNPY, abs=1e-6)
    assert [int(v) for v in geometry.node_def_cs] == [1, 2, 3], 'the frames are kept'
    scaled = visualdynamics.import_file(_archive(tmp_path / 'again.npz'),
                                        length_unit='mm')
    assert scaled.node_xyz == pytest.approx(SDYNPY / 1000.0, abs=1e-9), (
        'the whole placement scales, and the angle never did'
    )


def test_a_universal_file_places_its_nodes_and_writes_them_back(tmp_path):
    """Read and written through the format that states both."""
    geometry = visualdynamics.import_file(_archive(tmp_path / 'source.npz'))
    path = tmp_path / 'framed.unv'
    visualdynamics.export_file(geometry, path, 'unv')
    back = visualdynamics.import_file(path)
    assert back.node_xyz == pytest.approx(geometry.node_xyz, abs=1e-6), (
        'a round trip does not move a node'
    )
    assert [int(v) for v in back.node_def_cs] == [1, 2, 3]
    written = [line for line in path.read_text().splitlines()]
    assert any('2.0000000000000000E+00' in line and '9.0000000000000000E+01' in line
               for line in written), 'stated in its own cylindrical frame'


def test_a_bulk_deck_places_its_grids_and_writes_them_back(tmp_path):
    """The deck states a grid in the frame its CP field names. Writing
    basic-frame values under a local CP would move every framed grid on
    the next read, which is the same defect in the other direction."""
    geometry = visualdynamics.import_file(_archive(tmp_path / 'source.npz'))
    path = tmp_path / 'framed.bdf'
    visualdynamics.export_file(geometry, path, 'nastran')
    back = visualdynamics.import_file(path)
    assert back.node_xyz == pytest.approx(geometry.node_xyz, abs=1e-6)
    grids = [line for line in path.read_text().splitlines()
             if line.startswith('GRID*')]
    assert len(grids) == 3
    assert '2.000000000E+00' in grids[2] and '9.000000000E+01' in grids[2], (
        'the cylindrical grid is stated as radius and angle'
    )


def test_a_deck_that_places_a_grid_nowhere_says_so(tmp_path):
    """The one thing a shared primitive must not swallow: an unknown
    frame is silence in a universal file and an error in a deck, because
    a deck is supposed to define every frame it names."""
    path = tmp_path / 'orphan.bdf'
    path.write_text('BEGIN BULK\n'
                    'GRID,7,4,1.0,0.0,0.0\n'
                    'ENDDATA\n')
    with pytest.raises(ValueError, match='grid 7.*coordinate system 4'):
        visualdynamics.import_file(path)
