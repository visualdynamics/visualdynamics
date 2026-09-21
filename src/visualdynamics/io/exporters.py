"""The exporter registry.

Nearly anything visualdynamics can read, it can write — the exceptions are
named in docs/export.md — because a format that only goes one way makes
the tool a dead end for whoever needs the data next. Exporters register
themselves the same way importers do, and say which objects they can take.

Values are written in the unit system asked for, which the GUI takes from
the display selector: what you are looking at is what you get. Given no
system they go out as stored, which is SI once units are declared.

Display-only overrides do not travel. Reading accelerations in g is a
convenience of the screen; no file format here can record "g" as a unit —
UNV's units block carries length, force and temperature factors and
nothing else — so a file written in g would either be undeclarable or
misread by anyone who did the arithmetic. Exports use the coherent system
the display one is built on, and the status line names it.

An object whose units are unknown is written exactly as it stands. There
is no factor to convert it by, and multiplying by one would be the same as
claiming it was SI all along.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.data import DataArray
    from ..core.geometry import Geometry
    from ..core.shapes import ShapeSet
    from ..units import UnitSystem

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Exporter:
    """One format visualdynamics can write.

    `handles(obj)` says whether this exporter has anything to do with
    that object — which is what fills the Export menu with the formats
    that apply and no others — and `save(obj, path)` writes it.
    """

    name: str
    description: str
    suffix: str
    handles: Callable        # (obj) -> bool
    save: Callable           # (obj, path) -> None


_EXPORTERS: list[Exporter] = []

#: the formats that write a record's DOF as a node number and a
#: direction code. A modal coordinate (`M3`, `validate.modal_coordinate`)
#: has neither, and inventing a node for it would put a lie in a file
#: another tool trusts — so these refuse a modal object by name.
NODE_NUMBERED = frozenset({'sdynpy_data', 'exodus', 'unv', 'unv_binary',
                           'adf_functions', 'adf_time'})


def _refuse_modal(exporter: Exporter, obj: Any) -> None:
    from ..core.validate import modal_coordinate

    if exporter.name not in NODE_NUMBERED:
        return
    dofs = list(getattr(obj, 'response_dof', None) or [])
    modal = [dof for dof in dofs if modal_coordinate(dof) is not None]
    if modal:
        raise ValueError(
            f'{exporter.name} writes DOFs as node numbers, and '
            f'{len(modal)} of {len(dofs)} records are modal coordinates '
            f'({modal[0]} …) with no node to write — expand them to '
            'physical responses first, or save the project as .vdyn')


def register_exporter(name: str, description: str, suffix: str,
                      handles: Callable, save: Callable) -> None:
    """Teach visualdynamics to write a format."""
    _EXPORTERS.append(Exporter(name, description, suffix, handles, save))


def exporters(obj: Any = None) -> list[Exporter]:
    """Every exporter, or only those that can write this object."""
    if obj is None:
        return list(_EXPORTERS)
    return [e for e in _EXPORTERS if e.handles(obj)]


def geometry_values(geometry: Geometry, unit_system: UnitSystem | None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(coordinates, coordinate system matrices) in the target system.

    Only the origin row of a coordinate system carries length; its axes are
    directions and stay as they are.
    """
    points = np.asarray(geometry.node_xyz, dtype=np.float64)
    matrices = np.asarray(geometry.cs_matrix, dtype=np.float64).copy()
    if unit_system is None or not geometry.units_defined:
        return points, matrices
    unit_system = unit_system.coherent
    matrices[:, 3, :] = unit_system.from_si(matrices[:, 3, :], 'length')
    return unit_system.from_si(points, 'length'), matrices


def data_values(data: DataArray, unit_system: UnitSystem | None
                ) -> tuple[np.ndarray, np.ndarray]:
    """(abscissa, ordinate) in the target system."""
    if unit_system is None or not data.units_defined:
        return data.abscissa, data.ordinate
    unit_system = unit_system.coherent
    return (data.display_abscissa(unit_system),
            data.display_ordinate(unit_system))


def shape_values(shapes: ShapeSet,
                 unit_system: UnitSystem | None) -> np.ndarray:
    """Mode shape coefficients in the target system's mass unit.

    They are mass normalized, so they scale by 1/sqrt(mass) — the inverse
    of the factor that brought them to 1/sqrt(kg).
    """
    values = np.asarray(shapes.shape_matrix)
    if unit_system is None or not shapes.units_defined:
        return values
    from ..units import si_factor

    mass = unit_system.coherent.unit('mass')
    return values * np.sqrt(si_factor(mass, 'mass'))


def export_file(obj: Any, path: str | os.PathLike, format: str | None = None,
                unit_system: UnitSystem | None = None,
                **kwargs: Any) -> None:
    """Write `obj` to a foreign format, chosen by name or by suffix.

    `unit_system` is the system to write in; without one the stored values
    go out as they are. Raises ValueError naming what the object *can* be
    written as, since "cannot export" is nearly always a question of which
    format.
    """
    # `~` is the writer's to expand too, or a file lands in a folder
    # named for a tilde (the import's own courtesy, 2026-09-20)
    path = os.path.expanduser(str(path))
    available = exporters(obj)
    if format is not None:
        for exporter in _EXPORTERS:
            if exporter.name == format:
                if not exporter.handles(obj):
                    raise ValueError(
                        f'{exporter.name} cannot write {type(obj).__name__}; '
                        f'it takes {[e.name for e in available]}')
                _refuse_modal(exporter, obj)
                return exporter.save(obj, path, unit_system=unit_system,
                                     **kwargs)
        raise ValueError(f'No exporter named {format!r}; '
                         f'available: {[e.name for e in _EXPORTERS]}')
    for exporter in available:
        if path.endswith(exporter.suffix):
            _refuse_modal(exporter, obj)
            return exporter.save(obj, path, unit_system=unit_system,
                                 **kwargs)
    raise ValueError(
        f'Nothing writes {type(obj).__name__} to {path}; it can be written '
        f'as {[(e.name, e.suffix) for e in available]}')
