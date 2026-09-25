"""A finite element model built from a geometry's blocks.

The way every finite element format states a structure: blocks, each of
one element type, each given what it is made of. A geometry whose blocks
carry `BlockProperties` builds every element as the element it is —
quads to plates, triangles to triangles, two-node lines to beams — and
the proof is the demonstration plate rebuilt from its own exported
geometry: the same matrices to rounding (Brandon, 2026-09-25).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from test_fem import ALUMINUM, square_plate, triangle_plate

import visualdynamics
from visualdynamics.core.fem import BlockProperties, Material, Model, Section


def _aligned(built: Model, original: Model):
    """(K, M) of `built` in `original`'s DOF order, with the originals."""
    mass_o, stiff_o = original.matrices()
    mass_b, stiff_b = built.matrices()
    order = [built.dof_strings().index(dof) for dof in original.dof_strings()]
    grid = np.ix_(order, order)
    return stiff_b[grid], mass_b[grid], stiff_o, mass_o


def _with_properties(model: Model, props: BlockProperties):
    geometry = model.geometry()
    assert len(geometry.block_id) == 1, 'one block, one property set'
    geometry.block_properties = {int(geometry.block_id[0]): props}
    return geometry


def test_the_demo_plate_rebuilds_from_its_own_geometry():
    from visualdynamics.demo import plate

    original = plate.build()
    geometry = _with_properties(original, BlockProperties(plate.ALUMINUM,
                                                          plate.THICKNESS))
    built = Model.from_geometry(geometry)
    assert len(built.plates) == len(original.plates) and not built.beams
    k_b, m_b, k_o, m_o = _aligned(built, original)
    assert np.abs(k_b - k_o).max() <= 1e-12 * np.abs(k_o).max()
    assert np.abs(m_b - m_o).max() <= 1e-12 * np.abs(m_o).max()
    assert np.allclose(built.eigensolution(num_modes=10).frequency,
                       original.eigensolution(num_modes=10).frequency,
                       rtol=1e-9, atol=1e-6)


def test_a_triangle_mesh_rebuilds_the_same_way():
    original = triangle_plate(6)
    built = Model.from_geometry(_with_properties(
        original, BlockProperties(ALUMINUM, 0.01)))
    assert len(built.triangles) == len(original.triangles)
    k_b, m_b, k_o, m_o = _aligned(built, original)
    assert np.abs(k_b - k_o).max() <= 1e-12 * np.abs(k_o).max()
    assert np.abs(m_b - m_o).max() <= 1e-12 * np.abs(m_o).max()


def _portal(section_props: BlockProperties | None = None):
    """A portal frame, built member by member — and, when asked, the
    geometry that would rebuild it."""
    steel = Material('steel', 200e9, 7850.0, 0.29)
    tube = Section.round_tube('tube', 0.05, 0.003)
    model = Model('portal')
    for k, (x, y, z) in enumerate(((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0),
                                   (0.5, 0, 1)), 1):
        model.add_node(k, x, y, z)
    orientation = (0.0, 1.0, 0.0)
    for a, b in ((1, 2), (2, 5), (5, 3), (3, 4)):
        model.add_beam(a, b, steel, tube, orientation, group='frame')
    return model, steel, tube, orientation


def test_a_beam_block_rebuilds_a_frame():
    original, steel, tube, orientation = _portal()
    geometry = original.geometry(beams=True)
    assert list(geometry.elem_type) == [21] * 4, 'beams export as beam2'
    geometry.block_properties = {
        int(geometry.block_id[0]): BlockProperties(steel, section=tube,
                                                   orientation=orientation)}
    built = Model.from_geometry(geometry)
    assert len(built.beams) == 4 and not built.plates
    k_b, m_b, k_o, m_o = _aligned(built, original)
    assert np.abs(k_b - k_o).max() <= 1e-12 * np.abs(k_o).max()
    assert np.abs(m_b - m_o).max() <= 1e-12 * np.abs(m_o).max()


def test_a_stiffened_panel_mixes_blocks():
    """A plate block and a beam block on shared nodes: one structure,
    six rigid modes, the stiffener's stiffness in the answer."""
    plate = square_plate(4)
    bare = plate.eigensolution(num_modes=8).frequency
    geometry = plate.geometry()
    block = int(geometry.block_id[0])
    # a stiffener along the middle row of nodes, as a second block
    middle = [1 + 2 * 5 + i for i in range(5)]
    conn = list(geometry.elem_conn) + [np.array([a, b]) for a, b
                                       in pairwise(middle)]
    types = list(geometry.elem_type) + [21] * 4
    blocks = list(geometry.elem_block) + [block + 1] * 4
    stiffened = visualdynamics.Geometry(
        node_id=geometry.node_id, node_xyz=geometry.node_xyz,
        elem_id=list(range(1, len(conn) + 1)), elem_type=types,
        elem_conn=conn, elem_block=blocks,
        block_id=[block, block + 1], block_name=['skin', 'stiffener'],
        length_unit='m',
        block_properties={
            block: BlockProperties(ALUMINUM, 0.01),
            block + 1: BlockProperties(
                ALUMINUM, section=Section.rectangle('rib', 0.01, 0.03),
                orientation=(0.0, 0.0, 1.0))})
    model = Model.from_geometry(stiffened)
    assert len(model.plates) == 16 and len(model.beams) == 4
    assert len(model.pieces()) == 1
    shapes = model.eigensolution(num_modes=8)
    assert int(np.sum(shapes.frequency == 0.0)) == 6
    assert shapes.frequency[6] > bare[6], 'the rib stiffens the panel'
    assert {b.group for b in model.beams} == {'stiffener'}
    assert {p.group for p in model.plates} == {'skin'}


def test_a_block_without_properties_is_refused_by_name():
    plate = square_plate(2)
    geometry = plate.geometry()
    geometry.block_name = ['skin']
    geometry.block_properties = {99: BlockProperties(ALUMINUM, 0.01)}
    with pytest.raises(ValueError, match=r'block 1 \(skin\) has no properties'):
        Model.from_geometry(geometry)


def test_the_wrong_kind_of_properties_is_refused_by_name():
    plate = square_plate(2)
    geometry = plate.geometry()
    block = int(geometry.block_id[0])
    geometry.block_properties = {block: BlockProperties(
        ALUMINUM, section=Section.rod('rod', 0.01))}
    with pytest.raises(ValueError, match='quad4 elements, which take a thickness'):
        Model.from_geometry(geometry)
    geometry.block_properties = {block: BlockProperties(
        ALUMINUM, thickness=0.01, section=Section.rod('rod', 0.01))}
    with pytest.raises(ValueError, match='both a thickness and a section'):
        Model.from_geometry(geometry)
    geometry.block_properties = {block: BlockProperties(ALUMINUM)}
    with pytest.raises(ValueError, match='neither a thickness nor a section'):
        Model.from_geometry(geometry)


def test_an_element_the_solver_has_no_element_for_is_refused_by_name():
    geometry = visualdynamics.Geometry(
        node_id=[1, 2, 3, 4, 5, 6],
        node_xyz=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [.5, 0, 0], [.5, .5, 0],
                  [0, .5, 0]],
        elem_id=[1], elem_type=[42], elem_conn=[np.arange(1, 7)],
        elem_block=[1], block_id=[1], block_name=['curved'], length_unit='m',
        block_properties={1: BlockProperties(ALUMINUM, 0.01)})
    with pytest.raises(ValueError, match='tri6 elements, and the solver has '
                                          'no element for them'):
        Model.from_geometry(geometry)


def test_without_block_properties_the_grillage_needs_its_inputs():
    geometry = square_plate(2).geometry()
    with pytest.raises(ValueError, match='no block properties'):
        Model.from_geometry(geometry)
    # and with them, the grillage as it always was
    model = Model.from_geometry(geometry, ALUMINUM, Section.rod('rod', 0.01))
    assert model.beams and not model.plates


def test_block_properties_survive_the_native_file(tmp_path):
    from visualdynamics.demo import plate

    original = plate.build()
    geometry = _with_properties(original, BlockProperties(plate.ALUMINUM,
                                                          plate.THICKNESS))
    visualdynamics.save(geometry, tmp_path / 'plate.vdyn')
    back = visualdynamics.load(tmp_path / 'plate.vdyn')
    props = back.block_properties[int(back.block_id[0])]
    assert props.thickness == plate.THICKNESS
    assert props.material.youngs_modulus == plate.ALUMINUM.youngs_modulus
    assert props.material.name == '6061-T6' and props.section is None
    rebuilt = Model.from_geometry(back)
    assert np.allclose(rebuilt.eigensolution(num_modes=10).frequency,
                       original.eigensolution(num_modes=10).frequency,
                       rtol=1e-9, atol=1e-6)
    # a beam block's section and orientation round-trip too
    frame, steel, tube, orientation = _portal()
    geometry = frame.geometry(beams=True)
    geometry.block_properties = {int(geometry.block_id[0]): BlockProperties(
        steel, section=tube, orientation=orientation)}
    visualdynamics.save(geometry, tmp_path / 'frame.vdyn')
    back = visualdynamics.load(tmp_path / 'frame.vdyn')
    props = back.block_properties[int(back.block_id[0])]
    assert props.section == tube and props.orientation == orientation
    assert props.thickness is None
