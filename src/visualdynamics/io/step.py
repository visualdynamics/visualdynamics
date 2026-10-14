"""Read STEP and IGES files (B-rep CAD), tessellated by OpenCASCADE.

Unlike STL and 3MF, these files hold surfaces rather than triangles, so
reading them needs a CAD kernel: OCP (Apache-2.0 bindings) over
OpenCASCADE (LGPL-2.1 with an exception, a replaceable library as Qt
is; see NOTICE.md). The kernel is included in every packaged build,
because a frozen application cannot add it later. A pip install has it
as an option (``visualdynamics[step]``), and without it this module
refuses with the install command.

The result is the same as the 3MF reader's: triangle face elements in
meters, with the units defined, one named element group per top-level
(free) shape in the file. Part models downloaded from vendors in STEP or
IGES read directly.
"""

from __future__ import annotations

import os

import numpy as np

from ..core.geometry import Geometry
from ..core.solids import shape_triangles  # one rule, shared with solids

EXTENSIONS = ('.step', '.stp', '.iges', '.igs')

def sniff(path: str | os.PathLike) -> bool:
    return str(path).lower().endswith(EXTENSIONS)


def _require_kernel():
    from ..core.solids import require_kernel

    return require_kernel()


def _document(path: str):
    """The file read into an XCAF document — the form that keeps the
    assembly's names and placements, which a plain reader drops."""
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESCAFControl import IGESCAFControl_Reader
    from OCP.Interface import Interface_Static
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document

    document = TDocStd_Document(TCollection_ExtendedString('vd'))
    iges = str(path).lower().endswith(('.iges', '.igs'))
    reader = (IGESCAFControl_Reader() if iges
              else STEPCAFControl_Reader())
    reader.SetNameMode(True)
    # Meters at the kernel boundary, whatever the file counted in —
    # the same one-unit-inside rule the rest of the package lives by.
    # Set AFTER the reader exists and CHECKED: in a fresh process the
    # parameter is unregistered until a reader or writer constructs
    # the controller, and SetCVal before that returns False and does
    # nothing — millimeter magnitudes then arrive labeled meters. The
    # first real import did exactly that (a McMaster motor read 8470
    # inches wide); the test missed it because its own fixture writer
    # had already registered the parameter (Brandon, 2026-09-01).
    if not Interface_Static.SetCVal_s('xstep.cascade.unit', 'M'):
        raise ValueError('the CAD kernel refused the unit setting; '
                         'refusing to import at an unknown scale')
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f'{path} did not read as '
                         f'{"IGES" if iges else "STEP"}')
    if not reader.Transfer(document):
        raise ValueError(f'{path} read, but no shapes transferred')
    return document


def load(path: str | os.PathLike) -> Geometry:
    """Read a STEP or IGES file as a geometry, with each top-level shape
    as a named element group.

    Parameters
    ----------
    path : str or path-like
        The ``.step``/``.stp`` or ``.iges``/``.igs`` file.

    Returns
    -------
    Geometry
        Triangle face elements in meters, tessellated by the kernel, with
        one element group per free shape the file holds, named as the
        file names it or for the file when it has no name.

    Raises
    ------
    ValueError
        When the kernel is not installed, the kernel refuses the unit
        setting, the file does not read, nothing transfers, or no shape
        tessellates.
    """
    _require_kernel()
    from OCP.TDataStd import TDataStd_Name
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    try:
        from OCP.TDF import TDF_LabelSequence
    except ImportError:
        # the 8.0 binding names the sequence for what it is; 7.9 kept
        # the kernel's typedef (2026-09-13: Linux resolved 8.0.1 while
        # the Mac still had 7.9.3, and the reader has to take both)
        from OCP.collections import Sequence_TDF_Label as TDF_LabelSequence

    from .stl import mesh_geometry

    document = _document(str(path))
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    labels = TDF_LabelSequence()
    tool.GetFreeShapes(labels)
    parts: list[tuple[str, np.ndarray]] = []
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    for index in range(1, labels.Length() + 1):
        label = labels.Value(index)
        name_holder = TDataStd_Name()
        name = (name_holder.Get().ToExtString()
                if label.FindAttribute(TDataStd_Name.GetID_s(),
                                       name_holder) else '')
        corners = shape_triangles(tool.GetShape_s(label))
        if len(corners):
            parts.append((name or stem, corners))
    if not parts:
        raise ValueError(f'{path} holds no shapes that tessellate')
    return mesh_geometry(parts, length_unit='m')
