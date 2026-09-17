"""CAD meshes in and out: 3MF with parts as blocks, STL as triangles.

The CAD program tessellates at export, so these importers read
triangles — no geometry kernel, no new dependency. 3MF is the format
that keeps an assembly's structure: named objects become named
blocks, component transforms place each instance, and the declared
unit converts to meters on the way in. STL is the format everything
exports: unitless (the import asks, or a `length_unit` declares) and
one block per ASCII solid or per file.
"""

from __future__ import annotations

import os
import zipfile

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.geometry import Geometry
from visualdynamics.io import export_file, import_file


def _two_block_geometry():
    """A triangle 'wing' and a quad 'body', meters."""
    return Geometry(
        node_id=[1, 2, 3, 4, 5, 6, 7],
        node_xyz=[[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]],
        elem_id=[1, 2], elem_type=[41, 44],
        elem_conn=[np.array([1, 2, 3]), np.array([4, 5, 6, 7])],
        elem_block=[1, 2], block_id=[1, 2],
        block_name=['wing', 'body'], length_unit='m')


def test_3mf_round_trips_blocks_as_named_parts(tmp_path):
    path = tmp_path / 'model.3mf'
    export_file(_two_block_geometry(), path)
    back = import_file(path)
    assert list(back.block_name) == ['wing', 'body']
    assert back.length_unit == 'm', '3MF declares its unit; nothing asks'
    # the quad went out as two triangles, so three faces come back —
    # covering the same area at the same coordinates
    assert len(back.elem_id) == 3
    assert (back.elem_type == 41).all()
    wing = back.node_xyz[[int(n) - 1 for n in back.elem_conn[0]]]
    assert np.allclose(sorted(map(tuple, wing)),
                       [(0, 0, 0), (0, 1, 0), (1, 0, 0)])


def test_3mf_units_convert_to_meters(tmp_path):
    """A millimeter file arrives in meters with the unit declared —
    the whole reason 3MF beats STL for CAD hand-off."""
    model = """<?xml version="1.0"?>
<model unit="millimeter"
       xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
 <resources>
  <object id="1" type="model" name="plate">
   <mesh>
    <vertices>
     <vertex x="0" y="0" z="0"/><vertex x="1000" y="0" z="0"/>
     <vertex x="0" y="1000" z="0"/>
    </vertices>
    <triangles><triangle v1="0" v2="1" v3="2"/></triangles>
   </mesh>
  </object>
 </resources>
 <build><item objectid="1"/></build>
</model>"""
    path = tmp_path / 'mm.3mf'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('3D/3dmodel.model', model)
    back = import_file(path)
    assert back.length_unit == 'm'
    assert np.isclose(back.node_xyz.max(), 1.0), '1000 mm is one meter'
    assert list(back.block_name) == ['plate']


def test_3mf_components_place_each_instance(tmp_path):
    """An assembly: one part meshed once, placed twice through
    component transforms — two blocks, each where the assembly put
    it. This is how SolidWorks assemblies export."""
    model = """<?xml version="1.0"?>
<model unit="meter"
       xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
 <resources>
  <object id="1" type="model" name="bracket">
   <mesh>
    <vertices>
     <vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/>
     <vertex x="0" y="1" z="0"/>
    </vertices>
    <triangles><triangle v1="0" v2="1" v3="2"/></triangles>
   </mesh>
  </object>
  <object id="2" type="model">
   <components>
    <component objectid="1"/>
    <component objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/>
   </components>
  </object>
 </resources>
 <build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 0 0 5"/></build>
</model>"""
    path = tmp_path / 'assembly.3mf'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('3D/3dmodel.model', model)
    back = import_file(path)
    assert list(back.block_name) == ['bracket', 'bracket']
    assert len(back.elem_id) == 2
    # the item lifts everything to z=5; the second component also
    # shifts x by 10 — composed, not either alone
    first = back.node_xyz[[int(n) - 1 for n in back.elem_conn[0]]]
    second = back.node_xyz[[int(n) - 1 for n in back.elem_conn[1]]]
    assert np.allclose(first[:, 2], 5.0)
    assert np.allclose(second[:, 0] - first[:, 0], 10.0)
    assert np.allclose(second[:, 2], 5.0)


def test_stl_round_trips_triangles_and_merges_vertices(tmp_path):
    path = tmp_path / 'model.stl'
    export_file(_two_block_geometry(), path)
    back = import_file(path, length_unit='m')
    assert len(back.elem_id) == 3, 'the quad went out as two triangles'
    # the quad's split shares its diagonal: seven distinct corners,
    # not nine — the merge is what makes the mesh a mesh
    assert len(back.node_id) == 7
    assert list(back.block_name) == ['model'], (
        'binary STL has no names; the file names the one block')
    assert back.length_unit == 'm', 'declared at import, not by the file'


def test_stl_without_a_declared_unit_imports_unitless(tmp_path):
    path = tmp_path / 'model.stl'
    export_file(_two_block_geometry(), path)
    assert import_file(path).length_unit is None, (
        'STL cannot say; pretending it said millimeters would be a guess')


def test_ascii_stl_solids_become_named_blocks(tmp_path):
    plate = """solid left plate
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet
endsolid left plate
solid right plate
facet normal 0 0 1
 outer loop
  vertex 5 0 0
  vertex 6 0 0
  vertex 5 1 0
 endloop
endfacet
endsolid right plate
"""
    path = tmp_path / 'plates.stl'
    path.write_text(plate, encoding='ascii')
    back = import_file(path)
    assert list(back.block_name) == ['left plate', 'right plate']
    assert list(back.elem_block) == [1, 2]


def test_a_faceless_geometry_is_refused_by_both(tmp_path):
    lines_only = Geometry(node_id=[1, 2], node_xyz=[[0, 0, 0], [1, 0, 0]],
                          traceline_id=[1],
                          traceline_conn=[np.array([1, 2])])
    for suffix in ('.stl', '.3mf'):
        with pytest.raises(ValueError, match='no face elements'):
            export_file(lines_only, tmp_path / f'bad{suffix}')


def test_the_project_import_verb_reaches_them(tmp_path, window, pump):
    """The whole chain a user walks: export, import through the
    project verb (journaled), the parts standing as blocks."""
    path = tmp_path / 'craft.3mf'
    export_file(_two_block_geometry(), path)
    [name] = window.project.import_file(str(path))
    geometry = window.project[name]
    assert isinstance(geometry, Geometry)
    assert list(geometry.block_name) == ['wing', 'body']
    assert any('import_file' in line for line in window.project.journal)


def test_visualdynamics_export_speaks_them_too(tmp_path):
    """Anything the interface can do the API can do — and the other
    way: the top-level export path writes both formats by suffix."""
    project = visualdynamics.Project('t')
    project.add('Geometry', _two_block_geometry())
    for suffix in ('.stl', '.3mf'):
        out = tmp_path / f'out{suffix}'
        project.export('Geometry', out)
        assert os.path.getsize(out) > 0
