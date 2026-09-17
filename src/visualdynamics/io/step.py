"""STEP and IGES — B-rep CAD, tessellated here by OpenCASCADE.

Unlike STL and 3MF these files hold mathematics rather than
triangles, so reading them takes a geometry kernel: OCP (Apache-2.0
bindings) over OpenCASCADE (LGPL-2.1 with exception — the same
replaceable-library posture as Qt; see NOTICE.md). The kernel ships
in every packaged build, because a frozen application cannot grow it
later; the pip install keeps it optional (``visualdynamics[step]``)
and this module refuses with the install command when it is absent.

What comes out matches the 3MF importer's contract: the assembly's
named parts become named blocks, each instance placed by its own
transform, coordinates in meters. This is the importer that takes a
McMaster-Carr download directly — their STEP and IGES both land here.
"""

from __future__ import annotations

import os

import numpy as np

from ..core.geometry import Geometry

EXTENSIONS = ('.step', '.stp', '.iges', '.igs')

#: how fine the kernel tessellates, as a fraction of the shape's size.
#: Context geometry behind sensor points, not a render farm — coarse
#: on purpose, matching the guidance on CAD export settings.
DEFLECTION = 0.005


def sniff(path: str | os.PathLike) -> bool:
    return str(path).lower().endswith(EXTENSIONS)


def _require_kernel():
    try:
        import OCP
    except ImportError:
        raise ValueError(
            'STEP/IGES need the OpenCASCADE kernel, which is not '
            "installed — pip install 'visualdynamics[step]' (the "
            'packaged application already carries it)') from None
    return OCP


def _shape_triangles(shape) -> np.ndarray:
    """(n, 3, 3) corners from a tessellated shape's faces."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRep import BRep_Tool
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    # the 7.9 binding suffixes a static that shares its name with a
    # method; 8.0 drops the suffix for this one and keeps it for the
    # rest (2026-09-13, Linux on 8.0.1 beside the Mac on 7.9.3)
    _as_face = getattr(TopoDS, 'Face_s', None) or TopoDS.Face

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    size = max(box.CornerMax().XYZ().Subtracted(
        box.CornerMin().XYZ()).Modulus(), 1e-9)
    BRepMesh_IncrementalMesh(shape, size * DEFLECTION, False,
                             0.5, True)
    triangles: list[np.ndarray] = []
    walker = TopExp_Explorer(shape, TopAbs_FACE)
    while walker.More():
        face = _as_face(walker.Current())
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is not None:
            matrix = location.Transformation()
            points = np.asarray(
                [[mesh.Node(i + 1).X(), mesh.Node(i + 1).Y(),
                  mesh.Node(i + 1).Z()] for i in range(mesh.NbNodes())])
            rows = np.asarray(
                [[matrix.Value(r, c) for c in (1, 2, 3, 4)]
                 for r in (1, 2, 3)])
            points = points @ rows[:, :3].T + rows[:, 3]
            faces = np.asarray(
                [mesh.Triangle(i + 1).Get() for i in
                 range(mesh.NbTriangles())], dtype=np.int64) - 1
            if face.Orientation() == TopAbs_REVERSED:
                faces = faces[:, ::-1]
            triangles.append(points[faces])
        walker.Next()
    return (np.concatenate(triangles) if triangles
            else np.empty((0, 3, 3)))


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
    """Read a STEP or IGES file as a geometry — parts as named blocks.

    Parameters
    ----------
    path : str or path-like
        The ``.step``/``.stp`` or ``.iges``/``.igs`` file.

    Returns
    -------
    Geometry
        Triangle face elements in meters, tessellated by the kernel —
        one named block per free shape the file holds.
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
        corners = _shape_triangles(tool.GetShape_s(label))
        if len(corners):
            parts.append((name or stem, corners))
    if not parts:
        raise ValueError(f'{path} holds no shapes that tessellate')
    return mesh_geometry(parts, length_unit='m')
