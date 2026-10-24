"""The exporter registry.

Most formats Visual Dynamics can read, it can also write (the
exceptions are listed in docs/export.md), so that data read here can be
passed on to the next tool in the same format. Exporters are registered
the same way importers are, and each says which objects it can write.

Values are written in the unit system asked for, which the app takes
from the display selector, so a file holds the numbers shown on screen.
With no system given, they are written as stored: SI once units are
declared.

A display system that shows acceleration in g writes the coherent system
it is built on (in/s² for in-slinch-lbf-s (g)), because most formats
cannot label a value as g: UNV's units block, for one, carries length,
force and temperature factors and nothing else. A few formats write
their own way, and `units_written` says how each one did, for the line
the app shows after a save.

An object whose units are unknown is written exactly as it stands. There
is no factor to convert it by.
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
    """One format Visual Dynamics can write.

    `handles(obj)` says whether this exporter can write that object,
    which is how the Export menu lists only the formats that apply,
    and `save(obj, path, unit_system=..., **kwargs)` writes it.
    `suffix` is the file suffix that selects this exporter when no
    format is named.
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
    """Add a format to the exporters.

    When `export_file` chooses by suffix, the first registered exporter
    that can write the object and matches the suffix is used, so of two
    exporters with one suffix the one registered first is the default.
    """
    _EXPORTERS.append(Exporter(name, description, suffix, handles, save))


def exporters(obj: Any = None) -> list[Exporter]:
    """Every exporter in registration order, or only those that can
    write `obj` when it is given."""
    if obj is None:
        return list(_EXPORTERS)
    return [e for e in _EXPORTERS if e.handles(obj)]


def units_written(format: str, unit_system: UnitSystem) -> str:
    """The units a file of `format` holds its values in when saved with
    `unit_system` chosen, in words.

    The status line shows this after a save, because most formats do
    not record it themselves. A `.mat` and a delimited-text `.csv` are
    in the display system itself, g included; Nastran and ADF files are SI; 3MF is in meters; ESCDF
    labels each value with the unit it was declared in; every other
    format is in the coherent system the display system is built on.

    Parameters
    ----------
    format : str
        An exporter's name (`exporters`).
    unit_system : UnitSystem
        The display system the save was asked for.

    Returns
    -------
    str
        For example 'in-slinch-lbf-s', 'SI', or 'the units each object
        was declared in'.
    """
    # MATLAB writes the display system itself, g included, and so does
    # a CSV, whose header names each column's unit; ADF's
    # definition and Nastran's writer are SI; 3MF declares meters; ESCDF
    # labels each value with the unit it was declared in. The rest write
    # the coherent system (2026-10-13: the line named the coherent system
    # whatever the format did)
    if format in ('matlab', 'delimited'):
        return unit_system.name
    if format == 'nastran' or format.startswith('adf'):
        return 'SI'
    if format == 'threemf':
        return 'meters'
    if format == 'escdf':
        return 'the units each object was declared in'
    return unit_system.coherent.name


def geometry_values(geometry: Geometry, unit_system: UnitSystem | None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(coordinates, coordinate system matrices) in the coherent form of
    `unit_system`, or as stored when it is None or the geometry has no
    units defined.

    Only the origin row of a coordinate system matrix has length; its
    axes are directions and are not changed.
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
    """(abscissa, ordinate) in the coherent form of `unit_system`, or as
    stored when it is None or the data has no units defined."""
    if unit_system is None or not data.units_defined:
        return data.abscissa, data.ordinate
    unit_system = unit_system.coherent
    return (data.display_abscissa(unit_system),
            data.display_ordinate(unit_system))


def shape_values(shapes: ShapeSet,
                 unit_system: UnitSystem | None) -> np.ndarray:
    """Mode shape coefficients in the mass unit of the coherent form of
    `unit_system`, or as stored when it is None or the shapes have no
    units defined.

    The shapes are mass normalized, so they scale by 1/sqrt(mass), the
    inverse of the factor that converted them to 1/sqrt(kg).
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
    """Write `obj` to a file in a format chosen by name or by suffix.

    `unit_system` is the system to write in; without one the stored
    values are written as they are. When no exporter can write the
    object to that format, the ValueError lists the formats it can be
    written as, since the fix is nearly always a different format. A
    ValueError is also raised when a format that writes DOFs as node
    numbers (UNV, ADF functions and time histories, Exodus, the sdynpy
    data array) is asked to write modal coordinates. A leading `~` in
    the path is expanded.

    Parameters
    ----------
    obj : object
        What to write: a geometry, data, shapes, a project, and so on.
    path : str or os.PathLike
        Where; the suffix picks the format when `format` is not given.
    format : str, optional
        An exporter by name (`visualdynamics.io.exporters`), whatever the
        suffix says.
    unit_system : UnitSystem, optional
        The units to write in; the stored values as they are when omitted.
    **kwargs
        Passed to the format's writer, e.g. an ESCDF file's `created_by`.

    Returns
    -------
    None
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
