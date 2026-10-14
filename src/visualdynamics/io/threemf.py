"""Read and write 3MF meshes, a CAD export format that keeps parts and
units.

A ``.3mf`` is a zip holding one XML model: triangles that are already
tessellated (read as `tri3` face elements as they are), named objects
for an assembly's parts, each of which becomes a named element group,
and a declared unit, so there is nothing to ask at import. SolidWorks
writes it from Save As, so a SolidWorks model can be imported by
exporting it to 3MF; the native format is closed, and STEP (`io.step`)
needs a CAD kernel, which a pip install has only as an option.

Assemblies are component references with affine transforms. They are
resolved recursively, so each placed instance of a part has its
triangles where the assembly put them. Only the standard library
(``zipfile`` and ``xml.etree``) and numpy are used.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

import numpy as np

from ..core.geometry import Geometry
from .stl import faces_as_triangles, mesh_geometry

CORE = 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'

#: meters per one of each unit the 3MF core spec allows
UNITS = {'micron': 1e-6, 'millimeter': 1e-3, 'centimeter': 1e-2,
         'inch': 0.0254, 'foot': 0.3048, 'meter': 1.0}


def sniff(path: str | os.PathLike) -> bool:
    return str(path).lower().endswith('.3mf') and zipfile.is_zipfile(path)


def _transform(text: str | None) -> np.ndarray:
    """A 3MF transform — twelve numbers, a 3x3 and a translation row —
    as a 4x3 matrix applied as ``v' = v @ m[:3] + m[3]``."""
    if not text:
        return np.vstack([np.eye(3), np.zeros(3)])
    numbers = np.asarray([float(w) for w in text.split()])
    return numbers.reshape(4, 3)


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """The transform that applies `inner` first, then `outer`."""
    return np.vstack([inner[:3] @ outer[:3],
                      inner[3] @ outer[:3] + outer[3]])


def _mesh_corners(node: ET.Element) -> np.ndarray:
    mesh = node.find(f'{{{CORE}}}mesh')
    vertices = np.asarray(
        [[float(v.get('x')), float(v.get('y')), float(v.get('z'))]
         for v in mesh.find(f'{{{CORE}}}vertices')], dtype=np.float64)
    indices = np.asarray(
        [[int(t.get('v1')), int(t.get('v2')), int(t.get('v3'))]
         for t in mesh.find(f'{{{CORE}}}triangles')], dtype=np.int64)
    return vertices[indices]


def load(path: str | os.PathLike) -> Geometry:
    """Read a 3MF file as a geometry, one named element group per placed
    part.

    Parameters
    ----------
    path : str or path-like
        The ``.3mf`` file.

    Returns
    -------
    Geometry
        Triangle face elements, converted from the file's declared unit
        (millimeters when it declares none) with the units defined. There
        is one element group per part instance the build places, named
        with the part's own name, or 'object <id>' when it has none.

    Raises
    ------
    ValueError
        When the file has no model document, declares a unit 3MF does
        not define, or places no meshes in its build.
    """
    with zipfile.ZipFile(path) as archive:
        model_name = next(
            (n for n in archive.namelist() if n.endswith('.model')), None)
        if model_name is None:
            raise ValueError(f'{path} holds no 3D model document')
        root = ET.fromstring(archive.read(model_name))
    scale = UNITS.get(root.get('unit', 'millimeter'))
    if scale is None:
        raise ValueError(f"unknown 3MF unit {root.get('unit')!r}")
    resources = root.find(f'{{{CORE}}}resources')
    objects = {node.get('id'): node for node in
               ([] if resources is None else
                resources.findall(f'{{{CORE}}}object'))}
    build = root.find(f'{{{CORE}}}build')
    parts: list[tuple[str, np.ndarray]] = []
    for item in ([] if build is None else
                 build.findall(f'{{{CORE}}}item')):
        parts += _placed_corners(objects, item.get('objectid'),
                                 _transform(item.get('transform')))
    if not parts:
        raise ValueError(f'{path} places no meshes in its build')
    parts = [(name, corners * scale) for name, corners in parts]
    return mesh_geometry(parts, length_unit='m')


def _placed_corners(objects: dict, oid: str, transform: np.ndarray
                    ) -> list[tuple[str, np.ndarray]]:
    """(name, transformed (n, 3, 3) corners) per mesh under `oid`."""
    node = objects[oid]
    if node.find(f'{{{CORE}}}mesh') is not None:
        corners = _mesh_corners(node)
        placed = corners.reshape(-1, 3) @ transform[:3] + transform[3]
        return [(node.get('name') or f'object {oid}',
                 placed.reshape(-1, 3, 3))]
    out = []
    components = node.find(f'{{{CORE}}}components')
    for component in ([] if components is None else
                      components.findall(f'{{{CORE}}}component')):
        out += _placed_corners(
            objects, component.get('objectid'),
            _compose(transform, _transform(component.get('transform'))))
    return out


def handles(obj: Any) -> bool:
    return isinstance(obj, Geometry)


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0"
  Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def save(obj: Geometry, path: str | os.PathLike,
         unit_system: Any = None) -> None:
    """Write the geometry's face elements as 3MF, one named object per
    element group.

    The coordinates are written as stored. The file declares them in
    meters when the geometry's units are defined, which is what they
    are, and declares no unit when they are not, which 3MF reads as
    millimeters.
    `unit_system` is accepted because every exporter takes it, and is not
    used, since the file declares its own unit. A geometry with no face
    elements is refused with ValueError, as for STL.
    """
    del unit_system
    parts = faces_as_triangles(obj)
    if not parts:
        raise ValueError('this geometry has no face elements; '
                         '3MF holds triangles and nothing else')
    names = {int(i): (name or f'block {int(i)}') for i, name in
             zip(obj.group_id, obj.group_name, strict=True)}
    # 3MF has no "unit-less": its unit attribute is optional and an
    # absent one means millimeters, so any undefined geometry comes back
    # with some unit. Declaring meters asserted a unit nobody chose
    # (2026-10-13, found by the docstring pass). Leaving the attribute
    # out says only what is known, and the spec's default is the
    # convention unit-less CAD meshes (STL above all) are made in.
    attributes = {'xmlns': CORE}
    if obj.units_defined:
        attributes['unit'] = 'meter'
    model = ET.Element('model', attributes)
    resources = ET.SubElement(model, 'resources')
    build = ET.SubElement(model, 'build')
    for index, (block, corners) in enumerate(parts, start=1):
        node = ET.SubElement(resources, 'object',
                             {'id': str(index), 'type': 'model',
                              'name': names.get(block, f'block {block}')})
        mesh = ET.SubElement(node, 'mesh')
        flat = corners.reshape(-1, 3)
        vertices, indices = np.unique(flat, axis=0, return_inverse=True)
        holder = ET.SubElement(mesh, 'vertices')
        for x, y, z in vertices:
            ET.SubElement(holder, 'vertex',
                          {'x': repr(float(x)), 'y': repr(float(y)),
                           'z': repr(float(z))})
        triangles = ET.SubElement(mesh, 'triangles')
        for v1, v2, v3 in indices.reshape(-1, 3):
            ET.SubElement(triangles, 'triangle',
                          {'v1': str(v1), 'v2': str(v2), 'v3': str(v3)})
        ET.SubElement(build, 'item', {'objectid': str(index)})
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', CONTENT_TYPES)
        archive.writestr('_rels/.rels', RELS)
        archive.writestr(
            '3D/3dmodel.model',
            ET.tostring(model, encoding='unicode',
                        xml_declaration=True))
