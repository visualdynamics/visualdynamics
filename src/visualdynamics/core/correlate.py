"""Test-analysis correlation: FEM shapes sampled at the test's DOFs.

A finite element model carries a dense mesh with three or more DOFs per
node; the test measured a sparse set of directions at its own nodes.
Comparing the two starts by projecting the model onto the test: match
each test node to its nearest FEM node, take the FEM displacement there
(already resolved to global by the deform machinery), and dot it with
the test DOF's global direction — which handles uniaxial measurements
and local coordinate systems in one step. The result is an ordinary
ShapeSet at the test's DOFs, so everything downstream — cross-MAC,
overlay animation, reports — works on it unchanged.

Both geometries must overlay in one frame already; aligning mismatched
frames is a separate, later problem.

References
----------
1. Allemang, R. J., & Brown, D. L. (1982). "A Correlation Coefficient
   for Modal Vector Analysis." *Proceedings of the 1st International
   Modal Analysis Conference (IMAC)*, 110-116. The Modal Assurance
   Criterion that the cross-MAC downstream of this projection
   computes.
2. Allemang, R. J. (2003). "The Modal Assurance Criterion - Twenty
   Years of Use and Abuse." *Sound and Vibration*, 37(8), 14-21. What
   the number does and does not say, including why a high MAC between
   a model and a test is not by itself agreement.
3. Ewins, D. J. (2000). *Modal Testing: Theory, Practice and
   Application*, 2nd ed. Research Studies Press. Chapter 7 on
   comparing a model with a test, and on reducing one to the other's
   degrees of freedom.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from .geometry import Geometry
    from .shapes import ShapeSet


def project_shapes(fem_shapes: ShapeSet, fem_geometry: Geometry,
                   test_shapes: ShapeSet, test_geometry: Geometry,
                   tolerance: float = 0.02
                   ) -> tuple[ShapeSet, dict[str, Any]]:
    """`fem_shapes` reduced to `test_shapes`' DOFs.

    `tolerance` is a fraction of the test geometry's extent: a test
    node whose nearest FEM node lies further away is not matched, and
    its DOFs are dropped from the result rather than guessed.

    Returns (shape_set, report): the projected ShapeSet and a dict with
    'matched' / 'total' test node counts, 'worst' matched distance and
    'limit' (both meters, in the units-defined case), and 'dropped' —
    the test DOFs the projection could not carry.
    """
    from ..deform import dof_directions, node_displacements
    from ..units import SYSTEMS
    from ..viz.geometry import display_points
    from .shapes import ShapeSet

    if fem_geometry.units_defined != test_geometry.units_defined:
        raise ValueError('both geometries need units defined (or neither)'
                         ' to overlay them in one frame')
    si = SYSTEMS['m-kg-N-s']
    test_points = np.asarray(display_points(test_geometry, si)[0],
                             dtype=float)
    fem_points = np.asarray(display_points(fem_geometry, si)[0],
                            dtype=float)
    if not len(test_points) or not len(fem_points):
        raise ValueError('both geometries need nodes')

    dofs = list(test_shapes.coordinate)
    rows, directions, used = dof_directions(test_geometry, dofs)
    if not len(rows):
        raise ValueError("none of the test shapes' DOFs are "
                         'translations on the test geometry')

    center = test_points.mean(axis=0)
    extent = float(np.sqrt(((test_points - center) ** 2)
                           .sum(axis=1).max())) or 1.0
    limit = float(tolerance) * extent

    # nearest FEM node per unique test node, one row at a time — the
    # full (test x fem) distance matrix would not fit for a real mesh
    unique_rows = np.unique(rows)
    nearest, distance = {}, {}
    for row in unique_rows:
        deltas = fem_points - test_points[row]
        squared = (deltas ** 2).sum(axis=1)
        best = int(np.argmin(squared))
        nearest[int(row)] = best
        distance[int(row)] = float(np.sqrt(squared[best]))
    matched_rows = {row for row in nearest if distance[row] <= limit}
    if not matched_rows:
        raise ValueError(
            'no test node has a FEM node within the match tolerance — '
            'check that the geometries overlay, or loosen the tolerance')

    keep = [i for i, row in enumerate(rows) if int(row) in matched_rows]
    used_indices = np.flatnonzero(used)
    coordinate = [dofs[used_indices[i]] for i in keep]
    fem_rows = np.asarray([nearest[int(rows[i])] for i in keep])
    kept_directions = np.asarray([directions[i] for i in keep])

    complex_shapes = np.iscomplexobj(fem_shapes.shape_matrix)
    values = np.empty((fem_shapes.num_shapes, len(keep)),
                      dtype=np.complex128 if complex_shapes else np.float64)
    for m in range(fem_shapes.num_shapes):
        displacement = node_displacements(
            fem_geometry, fem_shapes.coordinate,
            fem_shapes.shape_matrix[m])
        sampled = displacement[fem_rows]
        if not complex_shapes:
            sampled = sampled.real
        values[m] = (kept_directions * sampled).sum(axis=1)

    dropped = ([dofs[i] for i in range(len(dofs)) if not used[i]]
               + [dofs[used_indices[i]] for i in range(len(rows))
                  if i not in keep])
    projected = ShapeSet(
        frequency=fem_shapes.frequency, damping=fem_shapes.damping,
        coordinate=coordinate, shape_matrix=values,
        modal_mass=fem_shapes.modal_mass,
        description=list(fem_shapes.description),
        comment=['projected to test DOFs'] * fem_shapes.num_shapes,
        mass_unit=fem_shapes.mass_unit,
        unscaled=getattr(fem_shapes, 'unscaled', False))
    report = {'matched': len(matched_rows), 'total': len(unique_rows),
              'worst': max(distance[row] for row in matched_rows),
              'limit': limit, 'dropped': dropped}
    return projected, report
