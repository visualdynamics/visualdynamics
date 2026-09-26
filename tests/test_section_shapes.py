"""Beam sections from their shapes (Brandon, 2026-09-26: *couldn't J,
Iy and Iz be defined from the geometry?* — from the cross-section's
shape and dimensions, which is how a member is thought of).

Every shape's area and second moments are held against an exact
numerical integration of the cross-section itself — a fine grid of
cells, each inside or outside the shape — rather than against rolled
shape tables, whose fillets and tapered flanges the formulas leave out
on purpose. The torsion constants are held against their closed forms
and, where it tells anything, a published table within its fillet
allowance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from visualdynamics.core.fem import SHAPES, Section, angle_major_axis

INCH = 0.0254


def _integrated(inside, extent, cell=1e-4):
    """(area, I about y, I about z, product) of the region `inside(y, z)`
    over the square [-extent, extent]^2, by the midpoint rule, about the
    region's own centroid. y across, z up — the convention the shapes
    use. The cells are a tenth of a millimeter and every dimension here a
    whole number of them, so each wall's edge falls between cells and the
    area is exact; the moments carry the rule's own O(h^2)."""
    n = round(2.0 * extent / cell)
    edges = np.linspace(-extent, extent, n + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    y, z = np.meshgrid(centers, centers, indexing='ij')
    cell = (edges[1] - edges[0]) ** 2
    mask = inside(y, z)
    area = mask.sum() * cell
    cy, cz = y[mask].mean(), z[mask].mean()
    iy = ((z[mask] - cz) ** 2).sum() * cell
    iz = ((y[mask] - cy) ** 2).sum() * cell
    iyz = ((y[mask] - cy) * (z[mask] - cz)).sum() * cell
    return area, iy, iz, iyz


def _box(y, z, y0, y1, z0, z1):
    return (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)


def test_every_shape_has_a_constructor_and_its_dimensions():
    assert list(SHAPES) == ['round tube', 'rod', 'rectangle',
                            'rectangular tube', 'I-beam', 'channel', 'angle']
    for shape, (method, names) in SHAPES.items():
        assert callable(getattr(Section, method)), shape
        section = Section.of_shape('s', shape, [0.1, 0.05, 0.008, 0.006][:len(names)])
        assert section.shape == shape and len(section.dimensions) == len(names)
    with pytest.raises(ValueError, match='a rectangle takes 2 dimensions — '
                                          'width, height'):
        Section.of_shape('s', 'rectangle', [1.0])
    with pytest.raises(ValueError, match='an angle takes 3 dimensions'):
        Section.of_shape('s', 'angle', [1.0])
    with pytest.raises(ValueError, match='not a section shape'):
        Section.of_shape('s', 'hexagon', [1.0])


@pytest.mark.parametrize('section, inside, extent', [
    (Section.rectangular_tube('rt', 0.10, 0.06, 0.005),
     lambda y, z: _box(y, z, -0.05, 0.05, -0.03, 0.03)
     & ~_box(y, z, -0.045, 0.045, -0.025, 0.025), 0.051),
    (Section.i_beam('i', 0.20, 0.10, 0.01, 0.006),
     lambda y, z: _box(y, z, -0.05, 0.05, 0.09, 0.10)
     | _box(y, z, -0.05, 0.05, -0.10, -0.09)
     | _box(y, z, -0.003, 0.003, -0.09, 0.09), 0.101),
    (Section.channel('c', 0.15, 0.05, 0.009, 0.006),
     lambda y, z: _box(y, z, 0.0, 0.05, 0.066, 0.075)
     | _box(y, z, 0.0, 0.05, -0.075, -0.066)
     | _box(y, z, 0.0, 0.006, -0.066, 0.066), 0.076),
])
def test_the_closed_forms_match_the_integrated_section(section, inside, extent):
    area, iy, iz, iyz = _integrated(inside, extent)
    assert section.area == pytest.approx(area, rel=5e-3)
    assert section.iy == pytest.approx(iy, rel=5e-3)
    assert section.iz == pytest.approx(iz, rel=5e-3)
    assert abs(iyz) < 1e-3 * max(iy, iz), 'symmetric about y: no product'


@pytest.mark.parametrize('a, b, t', [(0.10, 0.10, 0.012), (0.15, 0.09, 0.01)])
def test_an_angles_moments_are_its_principal_ones(a, b, t):
    """The leg axes carry a product of inertia the element has no term
    for, so the section holds the principal moments: the eigenvalues of
    the integrated tensor, and the axis `angle_major_axis` names."""
    section = Section.angle('l', a, b, t)
    area, iy, iz, iyz = _integrated(
        lambda y, z: _box(y, z, 0.0, a, 0.0, t) | _box(y, z, 0.0, t, 0.0, b),
        max(a, b) + 0.001)
    values, vectors = np.linalg.eigh(np.array([[iy, -iyz], [-iyz, iz]]))
    assert section.area == pytest.approx(area, rel=5e-3)
    assert section.iy == pytest.approx(values[1], rel=5e-3), 'the major'
    assert section.iz == pytest.approx(values[0], rel=5e-3), 'the minor'
    major = vectors[:, 1]
    theta = math.degrees(math.atan2(major[1], major[0])) % 180.0
    assert angle_major_axis(a, b, t) == pytest.approx(theta, abs=0.2)
    if a == b:
        assert angle_major_axis(a, b, t) == pytest.approx(45.0, abs=1e-6)


def test_the_torsion_constants():
    """Round: the polar moment, exactly. A closed rectangular tube:
    Bredt's thin-wall form, equal to the square tube it generalizes. Open
    sections: the sum of b t^3 / 3 — within the fillet allowance of the
    rolled shapes' tables (W8x31 0.536 in⁴, L4x4x1/2 0.322 in⁴)."""
    tube = Section.round_tube('t', 0.05, 0.004)
    assert tube.j == pytest.approx(tube.iy + tube.iz)
    assert Section.square_tube('sq', 0.04, 0.003) == Section.rectangular_tube(
        'sq', 0.04, 0.04, 0.003)
    rect = Section.rectangular_tube('rt', 0.10, 0.06, 0.005)
    mean_w, mean_h = 0.095, 0.055
    assert rect.j == pytest.approx(4 * (mean_w * mean_h) ** 2 * 0.005
                                   / (2 * (mean_w + mean_h)))
    w8 = Section.i_beam('W8x31', 8.0 * INCH, 7.995 * INCH, 0.435 * INCH,
                        0.285 * INCH)
    assert w8.j / INCH ** 4 == pytest.approx(0.536, rel=0.1)
    assert w8.iy / INCH ** 4 == pytest.approx(110.0, rel=0.03)
    assert w8.iz / INCH ** 4 == pytest.approx(37.1, rel=0.03)
    angle = Section.angle('L4x4x1/2', 4 * INCH, 4 * INCH, 0.5 * INCH)
    assert angle.j / INCH ** 4 == pytest.approx(0.322, rel=0.05)


def test_a_shape_that_does_not_close_is_refused_by_name():
    with pytest.raises(ValueError, match='rib: wall 0.03 closes the section'):
        Section.rectangular_tube('rib', 0.05, 0.05, 0.03)
    with pytest.raises(ValueError, match='do not make an I'):
        Section.i_beam('w', 0.02, 0.05, 0.011, 0.005)
    with pytest.raises(ValueError, match='short leg no longer'):
        Section.angle('l', 0.05, 0.08, 0.005)
