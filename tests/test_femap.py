"""Femap Neutral files: geometry and normal modes in, nothing out.

The fixtures are authored by hand in the layout pinned against real
Femap files (neutral version 11.1 for geometry, and both the 451 and
1051 output encodings); tests hold the field-by-field judgments the
reader makes — what is read, what is skipped knowingly, and what
refuses by name.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.io import femap


def _block(group_id, *lines):
    return '\n'.join(['   -1', f'   {group_id}', *lines, '   -1']) + '\n'


def _header():
    return _block(100, '<NULL>', '11.1,')


def _node(node_id, x, y, z, defcs=0, node_type=0):
    return (f'{node_id},{defcs},0,1,46,0,0,0,0,0,0,'
            f'{x},{y},{z},{node_type},0,')


def _element(elem_id, topology, nodes):
    padded = list(nodes) + [0] * (20 - len(nodes))
    return [f'{elem_id},124,1,21,{topology},1,0,1,1,0,0,0,0,0,',
            ','.join(str(n) for n in padded[:10]) + ',',
            ','.join(str(n) for n in padded[10:]) + ',',
            '0.,0.,0.,0,0,0,0,0,0,',
            '0.,0.,0.,', '0.,0.,0.,',
            '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,']


def _modal_set(set_id, frequency, title):
    return _block(450, f'{set_id},', title, '36,2,', f'{frequency},',
                  '1,', 'a note line', '0,0,')


def _451_vector(set_id, vec_id, pairs):
    lines = [f'{set_id},{vec_id},1,', f'vector {vec_id}',
             '0.,0.,0.,', '2,3,4,0,0,0,0,0,0,0,',
             '0,0,0,0,0,0,0,0,0,0,', '1,2,1,7,', '1,1,1,']
    lines += [f'{n},{v},' for n, v in pairs]
    lines += ['-1,0.,']
    return lines


def test_geometry_reads_nodes_and_mixed_elements(tmp_path):
    path = tmp_path / 'model.neu'
    path.write_text(
        _header()
        + _block(403, *(_node(n, float(n), 0.0, 0.0)
                        for n in (1, 2, 3, 4, 5)))
        + _block(404,
                 *_element(1, 4, [1, 2, 3, 4]),          # Quad4
                 *_element(2, 10, [1, 2, 3, 4, 5, 5, 5, 5, 5, 5]),
                 *_element(3, 9, [5])))                  # Point mass
    geometry = visualdynamics.import_file(str(path), length_unit='m')
    assert list(geometry.node_id) == [1, 2, 3, 4, 5]
    assert np.allclose(geometry.node_xyz[:, 0], [1, 2, 3, 4, 5])
    assert list(geometry.elem_type) == [94, 118, 161], \
        'Quad4 -> quadshell4, Tetra10 -> tet10, Point -> mass'
    assert len(geometry.elem_conn[1]) == 10


def test_modal_sets_become_a_shape_set_from_451_vectors(tmp_path):
    path = tmp_path / 'modes.neu'
    path.write_text(
        _header()
        + _block(403, _node(1, 0., 0., 0.), _node(2, 1., 0., 0.))
        + _modal_set(1, 12.5, 'Mode 1, 12.5 Hz')
        + _modal_set(2, 40.0, 'Mode 2, 40 Hz')
        + _block(451,
                 *_451_vector(1, 2, [(1, 0.1), (2, -0.1)]),
                 *_451_vector(1, 3, [(1, 0.0), (2, 0.0)]),
                 *_451_vector(1, 4, [(1, 0.2), (2, 0.2)]),
                 *_451_vector(2, 2, [(1, 0.3), (2, 0.3)]),
                 *_451_vector(2, 3, [(1, 0.0), (2, 0.0)]),
                 *_451_vector(2, 4, [(1, -0.4), (2, 0.4)])))
    out = visualdynamics.import_file(str(path))
    shapes = out['shapes']
    assert np.allclose(shapes.frequency, [12.5, 40.0]), \
        'the 450 value is the frequency for an analysis-type-2 set'
    assert list(shapes.coordinate) == ['1X+', '1Y+', '1Z+',
                                       '2X+', '2Y+', '2Z+']
    assert shapes.shape_matrix[0, 0] == pytest.approx(0.1)
    assert shapes.shape_matrix[1, 5] == pytest.approx(0.4)
    assert shapes.unscaled


def test_1051_packed_runs_reassemble_across_lines(tmp_path):
    """The modern encoding: `first, last, values...` runs, values
    flowing over line breaks, with a gap between runs."""
    path = tmp_path / 'modes.neu'
    vector = ['1,2,1,', 'T1', '0.,0.,0.,', '2,0,0,0,0,0,0,0,0,0,',
              '0,0,0,0,0,0,0,0,0,0,', '0,', '0,0,1,7,', '1,1,1,',
              '1,4,0.1,0.2,',       # run 1..4, split over two lines
              '0.3,0.4,',
              '10,11,1.0,1.1,',     # a second run after a gap
              '-1,0.,']
    path.write_text(_header()
                    + _modal_set(1, 5.0, 'Mode 1')
                    + _block(1051, *vector))
    # T2/T3 absent: a 2-D translator may write T1 alone
    out = visualdynamics.import_file(str(path))
    assert list(out.coordinate) == ['1X+', '2X+', '3X+', '4X+',
                                    '10X+', '11X+']
    assert np.allclose(out.shape_matrix[0],
                       [0.1, 0.2, 0.3, 0.4, 1.0, 1.1])


def test_static_sets_and_elemental_vectors_are_skipped(tmp_path):
    path = tmp_path / 'static.neu'
    static_set = _block(450, '1,', 'Static case', '36,1,', '0.,',
                        '1,', 'a note', '0,0,')
    path.write_text(_header()
                    + _block(403, _node(1, 0., 0., 0.))
                    + static_set
                    + _block(451, *_451_vector(1, 2, [(1, 0.5)])))
    out = visualdynamics.import_file(str(path))
    assert type(out).__name__ == 'Geometry', \
        'a static displacement is not a mode; only geometry imports'


def test_rigid_elements_refuse_by_name(tmp_path):
    path = tmp_path / 'rigid.neu'
    path.write_text(_header()
                    + _block(403, _node(1, 0., 0., 0.))
                    + _block(404, *_element(1, 13, [1])))
    with pytest.raises(ValueError, match='Rigid'):
        femap.load(str(path))


def test_an_unknown_vintage_refuses_naming_the_version(tmp_path):
    """An element header with the wrong field count means the record
    shape changed; misreading silently is the one forbidden outcome."""
    path = tmp_path / 'odd.neu'
    lines = _element(1, 4, [1, 2, 3, 4])
    lines[0] = '1,124,1,21,4,1,0,'          # a short, foreign header
    path.write_text(_header()
                    + _block(403, *(_node(n, 0., 0., 0.)
                                    for n in (1, 2, 3, 4)))
                    + _block(404, *lines))
    with pytest.raises(ValueError, match='11.1'):
        femap.load(str(path))


def test_a_node_in_a_local_system_refuses(tmp_path):
    path = tmp_path / 'local.neu'
    path.write_text(_header()
                    + _block(403, _node(1, 1., 90., 0., defcs=5)))
    with pytest.raises(ValueError, match='405'):
        femap.load(str(path))


def test_scalar_nodes_are_skipped_knowingly(tmp_path):
    path = tmp_path / 'scalar.neu'
    path.write_text(_header()
                    + _block(403, _node(1, 0., 0., 0.),
                             _node(901, 0., 0., 0., node_type=1)))
    geometry = femap.load(str(path))
    assert list(geometry.node_id) == [1]


def test_gambit_neutral_files_are_not_sniffed(tmp_path):
    """Gambit also writes .neu; it opens with words, not the -1
    bracket, and belongs to no reader here."""
    path = tmp_path / 'mesh.neu'
    path.write_text('        CONTROL INFO 2.4.6\n'
                    '** GAMBIT NEUTRAL FILE\n')
    assert not femap.sniff(str(path))
