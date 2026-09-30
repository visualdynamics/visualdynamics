"""Nastran bulk data: geometry in, geometry out.

Written from the public card layouts; the fixtures here are authored
by hand from those layouts, and the round-trips assert the writer and
reader agree with each other — the only oracle a text format this
well documented needs.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.io import nastran


def _mesh():
    rng = np.random.default_rng(11)
    return visualdynamics.Geometry(
        node_id=list(range(1, 16)),
        node_xyz=rng.normal(size=(15, 3)) * 0.5,
        elem_id=[1, 2, 3, 4, 5],
        elem_type=[94, 118, 115, 201, 161],
        elem_color=[1] * 5,
        elem_conn=[np.array([1, 2, 3, 4]), np.arange(1, 11),
                   np.arange(1, 9), np.arange(1, 6), np.array([15])],
        traceline_id=[1], traceline_color=[1],
        traceline_conn=[np.array([1, 5, 9])])


def test_a_geometry_round_trips_the_bulk_deck(tmp_path):
    geometry = _mesh()
    geometry.define_units('m')
    nastran.save(geometry, str(tmp_path / 'mesh.bdf'))
    back = visualdynamics.import_file(str(tmp_path / 'mesh.bdf'),
                                      length_unit='m')
    assert np.allclose(back.node_xyz, geometry.node_xyz)
    assert list(back.elem_type) == [94, 118, 115, 201, 161]
    assert np.array_equal(back.elem_conn[1], np.arange(1, 11)), 'the tet10'
    # the traceline came back as PLOTELs, re-joined pair by pair
    assert [list(t) for t in back.traceline_conn] == [[1, 5], [5, 9]]


def test_the_three_field_formats_read_the_same_grid(tmp_path):
    """Small field, large field and free field are one card three
    ways, including the bare-exponent float only Nastran writes."""
    deck = tmp_path / 'fields.bdf'
    deck.write_text(
        'BEGIN BULK\n'
        'GRID           1       0     1.0     2.0     3.0\n'
        'GRID*                  2               0             1.0'
        '             2.0*\n'
        '*                    3.0\n'
        'GRID,3,0,1.,2.,3.\n'
        'GRID           4       0   1.5-3     0.0     0.0\n'
        'ENDDATA\n')
    geometry = nastran.load(str(deck))
    assert list(geometry.node_id) == [1, 2, 3, 4]
    assert np.allclose(geometry.node_xyz[:3],
                       np.tile([1.0, 2.0, 3.0], (3, 1)))
    assert geometry.node_xyz[3, 0] == pytest.approx(1.5e-3), \
        "'1.5-3' is 1.5e-3 — the E dropped to fit eight characters"


def test_grids_in_a_chained_cylindrical_system_land_in_basic(tmp_path):
    """CP resolution through a reference chain: a cylindrical system
    defined in a shifted cartesian one. R=2 at theta=90 in a frame
    whose origin sits at x=10 is (10, 2, 5) in basic."""
    deck = tmp_path / 'systems.bdf'
    deck.write_text(
        'BEGIN BULK\n'
        'CORD2R,10,0, 10.,0.,0., 10.,0.,1., 11.,0.,0.\n'
        'CORD2C,20,10, 0.,0.,0., 0.,0.,1., 1.,0.,0.\n'
        'GRID,1,20, 2.,90.,5.\n'
        'ENDDATA\n')
    geometry = nastran.load(str(deck))
    assert np.allclose(geometry.node_xyz[0], [10.0, 2.0, 5.0])


def test_constraints_and_analysis_cards_are_skipped_knowingly(tmp_path):
    deck = tmp_path / 'deck.bdf'
    deck.write_text(
        'SOL 103\nCEND\nBEGIN BULK\n'
        'PSHELL,1,1,0.002\nMAT1,1,7.1+10,,0.33,2700.\n'
        'GRID,1,0,0.,0.,0.\nGRID,2,0,1.,0.,0.\n'
        'GRID,3,0,0.,1.,0.\n'
        'CTRIA3,1,1,1,2,3\n'
        'RBE2,99,1,123456,2,3\n'
        'ENDDATA\n')
    geometry = nastran.load(str(deck))
    assert list(geometry.elem_type) == [91], \
        'the mesh, without the constraint or the analysis cards'


def test_an_unknown_element_card_refuses_by_name(tmp_path):
    deck = tmp_path / 'deck.bdf'
    deck.write_text('BEGIN BULK\nGRID,1,0,0.,0.,0.\n'
                    'CGAP,7,1,1,1\nENDDATA\n')
    with pytest.raises(ValueError, match='CGAP'):
        nastran.load(str(deck))


def test_a_grid_point_coordinate_system_refuses_with_the_fix(tmp_path):
    deck = tmp_path / 'deck.bdf'
    deck.write_text('BEGIN BULK\nCORD1R,5,1,2,3\nENDDATA\n')
    with pytest.raises(ValueError, match='CORD2'):
        nastran.load(str(deck))


def test_includes_are_inlined_relative_to_the_deck(tmp_path):
    (tmp_path / 'nodes.blk').write_text('GRID,1,0,1.,2.,3.\n')
    deck = tmp_path / 'main.bdf'
    deck.write_text("BEGIN BULK\nINCLUDE 'nodes.blk'\nENDDATA\n")
    geometry = nastran.load(str(deck))
    assert list(geometry.node_id) == [1]


def test_the_written_deck_says_who_wrote_it_and_is_not_runnable(tmp_path):
    geometry = _mesh()
    nastran.save(geometry, str(tmp_path / 'mesh.bdf'))
    text = (tmp_path / 'mesh.bdf').read_text(encoding='utf-8')
    assert text.startswith('$ written by visualdynamics')
    assert 'PSHELL' not in text and 'MAT1' not in text, \
        'properties are the analyst\'s, not invented here'


def test_property_cards_become_block_properties(tmp_path):
    """A deck's PSOLID and PSHELL cards, with their MAT1, say what each
    block is made of, and each property id is a block (2026-09-30, so
    a shared solid model opens ready for Solve Modes). A property whose
    material is not in the deck carries nothing, and the Blocks table
    asks."""
    from visualdynamics.io import nastran

    deck = tmp_path / 'props.bdf'
    deck.write_text(
        'GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\nGRID,4,,0.,1.,0.\n'
        'GRID,5,,0.,0.,1.\nGRID,6,,1.,0.,1.\nGRID,7,,1.,1.,1.\nGRID,8,,0.,1.,1.\n'
        'CHEXA,1,11,1,2,3,4,5,6,\n,7,8\n'
        'CQUAD4,2,21,5,6,7,8\n'
        'CQUAD4,3,31,1,2,3,4\n'
        'MAT1,1,6.9E10,,0.33,2700.\n'
        'MAT1,2,2.1E11,8.0E10,,7800.\n'
        'PSOLID,11,1\nPSHELL,21,2,0.002\nPSHELL,31,9,0.001\n')
    geometry = nastran.load(deck)
    assert list(geometry.block_id) == [11, 21, 31]
    assert list(geometry.block_name) == ['property 11', 'property 21',
                                         'property 31']
    assert list(geometry.elem_block) == [11, 21, 31]
    solid = geometry.block_properties[11]
    assert solid.kind == 'solid'
    assert solid.material.youngs_modulus == 6.9e10
    assert solid.material.density == 2700.0
    assert solid.material.poissons_ratio == 0.33
    shell = geometry.block_properties[21]
    assert shell.kind == 'plate' and shell.thickness == 0.002
    # E and G given, nu left blank: nu follows from them
    assert shell.material.poissons_ratio == pytest.approx(2.1e11 / 1.6e11 - 1)
    assert shell.material.shear_modulus == 8.0e10
    assert 31 not in geometry.block_properties, 'MAT1 9 is not in the deck'
