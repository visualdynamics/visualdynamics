"""Read and write sdynpy mode shape files (.npy).

The file is a structured NumPy array with one record per mode and the
fields 'frequency' (Hz), 'damping' (fraction of critical), 'coordinate'
((node, direction) pairs, one per DOF), 'shape_matrix' (a coefficient
per DOF), 'modal_mass' and 'comment1' to 'comment5'. Every mode must
cover the same DOFs.

The file does not record which mass unit the shapes were normalized
against, so they import unit-less unless `mass_unit` is declared.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.shapes import ShapeSet
    from ..units import UnitSystem

import numpy as np

from ..core.data import direction_code, dof_string, parse_dof
from ..core.shapes import ShapeSet

EXTENSIONS = ('.npy',)

_FIELDS = {'frequency', 'damping', 'coordinate', 'shape_matrix'}


def sniff(path: str | os.PathLike) -> bool:
    if not str(path).endswith(EXTENSIONS):
        return False
    try:
        array = np.load(path, allow_pickle=True)
        return array.dtype.names is not None and _FIELDS <= set(array.dtype.names)
    except Exception:  # noqa: BLE001 - sniffers must not raise on foreign files
        return False


def load(path: str | os.PathLike, mass_unit: str | None = None) -> ShapeSet:
    array = np.load(path, allow_pickle=True).reshape(-1)
    if array.size == 0:
        raise ValueError(f'{path} contains no modes')

    coordinate = array['coordinate'][0]
    dofs = [dof_string(entry['node'], entry['direction']) for entry in coordinate]
    for record in array['coordinate'][1:]:
        if len(record) != len(coordinate) or not np.array_equal(
                record.view(np.ndarray), coordinate.view(np.ndarray)):
            raise ValueError(
                f'{path}: modes are defined over differing DOF sets; '
                'a visualdynamics ShapeSet shares one set of coordinates')

    comments = ['; '.join(s for i in range(1, 6)
                          if (s := str(record[f'comment{i}']).strip()))
                for record in array]

    shapes = ShapeSet(
        frequency=array['frequency'],
        damping=array['damping'],
        coordinate=dofs,
        shape_matrix=np.stack(list(array['shape_matrix'])),
        modal_mass=array['modal_mass'] if 'modal_mass' in array.dtype.names
        else None,
        comment=comments,
    )
    if mass_unit is not None:
        shapes.define_units(mass_unit)
    return shapes


def handles(obj: Any) -> bool:
    return isinstance(obj, ShapeSet)


def save(shapes: ShapeSet, path: str | os.PathLike, unit_system: UnitSystem | None = None) -> None:
    """Write mode shapes in the sdynpy .npy layout.

    The shape coefficients are written against the mass unit of
    `unit_system` when the units are defined, and as stored otherwise;
    the modal masses are written as stored, since against coefficients in
    1/sqrt(mass) a modal mass carries no unit. The format records neither
    the unit nor the normalization. Complex shapes are written in the
    layout's complex form, coefficients and modal masses both complex;
    real shapes with a complex modal mass are refused with ValueError.
    The first comment field holds the mode's description, or
    its comment when it has none.
    """
    from .exporters import shape_values
    dofs = shapes.num_dofs
    # the layout has a complex form, the same fields with the shape
    # matrix and modal mass complex, and `load` reads either. Writing
    # only the real part threw away the phase of every complex mode
    # without a word (2026-10-13, found by the docstring pass). The form
    # follows the shapes: a complex modal mass on real shapes cannot be
    # written in the real form, and the complex one would read back as
    # complex shapes, which synthesize by the other formula.
    number = '<c16' if shapes.is_complex else '<f8'
    if number == '<f8' and np.any(np.imag(shapes.modal_mass)):
        raise ValueError('real shapes with a complex modal mass: the '
                         'sdynpy layout has no form for them')
    dtype = [('frequency', '<f8'), ('damping', '<f8'),
             ('coordinate', [('node', '<u8'), ('direction', 'i1')], (dofs,)),
             ('shape_matrix', number, (dofs,)), ('modal_mass', number),
             *[(f'comment{i}', '<U80') for i in range(1, 6)]]
    records = np.zeros(shapes.num_shapes, dtype=dtype)
    records['frequency'] = shapes.frequency
    records['damping'] = shapes.damping
    records['modal_mass'] = shapes.modal_mass
    records['shape_matrix'] = shape_values(shapes, unit_system)

    pairs = np.zeros(dofs, dtype=[('node', '<u8'), ('direction', 'i1')])
    for i, dof in enumerate(shapes.coordinate):
        node, direction = parse_dof(dof)
        pairs[i] = (node or 0, direction_code(direction) if direction else 0)
    records['coordinate'] = np.tile(pairs, (shapes.num_shapes, 1))
    # the format has one comment field where visualdynamics has two: the comment the
    # file arrived with, and the description typed in visualdynamics. What the user
    # wrote wins, since the other came from a file in the first place.
    for i in range(shapes.num_shapes):
        described = str(shapes.description[i]).strip()
        records['comment1'][i] = (described or str(shapes.comment[i]))[:80]
    np.save(path, records, allow_pickle=True)
