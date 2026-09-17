"""Complex modes: stored, saved, compared, exchanged, and animated.

The modal fit deliberately solves real normal modes only
(`test_shapes_are_real` holds it to that), but a ShapeSet must carry a
complex mode faithfully everywhere else: imported ADF and UNV shapes
can be complex, and a heavily non-proportionally damped structure's
modes genuinely are. The physics that distinguishes the two on screen:
a real mode's deflection collapses to zero at quarter phase, while a
complex mode travels — some node is always moving.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.shapes import ShapeSet


def traveling():
    """Four DOFs a quarter turn apart: the textbook traveling wave."""
    phases = np.exp(1j * np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]))
    return ShapeSet([10.0], [0.02], ['1X+', '2X+', '3X+', '4X+'],
                    [phases])


def line_geometry():
    return visualdynamics.Geometry(
        node_id=[1, 2, 3, 4],
        node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        length_unit='m')


def test_storage_and_vdyn_round_trip(tmp_path):
    shapes = traveling()
    assert np.iscomplexobj(shapes.shape_matrix)
    visualdynamics.save(shapes, tmp_path / 'complex.vdyn')
    back = visualdynamics.load(tmp_path / 'complex.vdyn')
    assert np.allclose(back.shape_matrix, shapes.shape_matrix), (
        'the native file keeps the phases exactly')


def test_a_complex_mode_travels_and_a_real_one_collapses():
    from visualdynamics.deform import ShapeDeflection

    geometry = line_geometry()
    complex_mode = ShapeDeflection(geometry, traveling().coordinate,
                                   traveling().shape_matrix[0])
    at_zero = complex_mode.offsets(0.0).copy()
    at_quarter = complex_mode.offsets(np.pi / 2).copy()
    assert np.abs(at_quarter).max() > 0.5, 'still moving at quarter phase'
    assert not np.allclose(at_zero, at_quarter), 'and differently: it travels'

    real_mode = ShapeDeflection(geometry, ['1X+', '2X+', '3X+', '4X+'],
                                np.array([1.0, 0.5, -0.5, -1.0]))
    assert np.abs(real_mode.offsets(np.pi / 2)).max() < 1e-12, (
        'a real mode is everywhere zero at quarter phase')


def test_the_peak_magnitude_is_the_true_orbit_peak():
    """The closed-form peak must bound the swept deflection tightly —
    it scales the animation, and a complex mode's peak is not simply
    the magnitude of the real part."""
    from visualdynamics.deform import ShapeDeflection

    rng = np.random.default_rng(11)
    shape = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    deflection = ShapeDeflection(line_geometry(),
                                 ['1X+', '2X+', '3X+', '4X+'], shape)
    swept = max(np.linalg.norm(deflection.offsets(t), axis=1).max()
                for t in np.linspace(0, 2 * np.pi, 720))
    assert deflection.peak_magnitude == pytest.approx(swept, rel=1e-3)


def test_the_mac_reads_complex_shapes():
    shapes = traveling()
    assert shapes.auto_mac()[0, 0] == pytest.approx(1.0)
    # a phase-rotated copy is the same mode; the MAC must say so
    rotated = ShapeSet([10.0], [0.02], shapes.coordinate,
                       shapes.shape_matrix * np.exp(1j * 0.7))
    from visualdynamics.core.shapes import cross_mac
    assert cross_mac(shapes, rotated)[0, 0] == pytest.approx(1.0)


def test_unv_round_trip_keeps_the_phases(tmp_path):
    """Dataset 55 densifies to whole nodal vectors (documented), so the
    comparison is at the original DOFs — where the complex values must
    survive exactly."""
    shapes = traveling()
    visualdynamics.export_file(shapes, str(tmp_path / 'complex.unv'),
                               format='unv')
    back = visualdynamics.import_file(str(tmp_path / 'complex.unv'))
    assert np.iscomplexobj(back.shape_matrix)
    where = [list(back.coordinate).index(dof)
             for dof in shapes.coordinate]
    assert np.allclose(back.shape_matrix[:, where],
                       shapes.shape_matrix, rtol=1e-4), (
        'phases exact at the DOFs the set actually carried')


def test_animation_renders_a_complex_mode(tmp_path):
    from visualdynamics.viz.animate import animate_shape

    animate_shape(line_geometry(), traveling(), 0,
                  screenshot=str(tmp_path / 'complex_mode.png'),
                  show=False)
    assert (tmp_path / 'complex_mode.png').stat().st_size > 1000
