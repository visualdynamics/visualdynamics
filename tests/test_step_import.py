"""STEP and IGES import — B-rep tessellated by the bundled kernel.

The fixtures are written by the same kernel that reads them (OCP over
OpenCASCADE), because hand-writing a STEP file is an exercise in ISO
10303 and proves nothing more. The tests gate on the kernel being
installed; environments without the `step` extra skip, and the one
refusal test runs everywhere — the message that names the install
command must never rot.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from visualdynamics.io import import_file


def _write_box(path, dx=100.0, dy=50.0, dz=25.0):
    """A named box, written in millimeters."""
    pytest.importorskip('OCP')
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    # both statics, AFTER a writer exists to register them, and both
    # checked: the writer converts from the process-global cascade
    # unit to the write unit, so a leftover meters cascade (any prior
    # load() in this worker) turned this fixture's 100 into a
    # 100000 mm box — the intermittent gate failure was this fixture,
    # not the importer (2026-09-01, values: SI_UNIT MILLI, 100000.0)
    assert Interface_Static.SetCVal_s('xstep.cascade.unit', 'MM')
    assert Interface_Static.SetCVal_s('write.step.unit', 'MM')
    writer.Transfer(BRepPrimAPI_MakeBox(dx, dy, dz).Shape(),
                    STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone


def test_a_step_box_arrives_as_meter_triangles(tmp_path):
    path = tmp_path / 'bracket.step'
    _write_box(path)
    geometry = import_file(path)
    assert geometry.length_unit == 'm', 'the kernel converts at the door'
    extent = geometry.node_xyz.max(axis=0) - geometry.node_xyz.min(axis=0)
    assert np.allclose(sorted(extent), [0.025, 0.05, 0.1]), (
        'a 100 x 50 x 25 mm box is 0.1 x 0.05 x 0.025 m')
    assert (geometry.elem_type == 41).all(), 'tri3 face elements'
    assert len(geometry.elem_id) >= 12, 'a box tessellates to >= 12 tris'
    assert len(geometry.block_id) == 1


def test_iges_reads_through_the_same_door(tmp_path):
    pytest.importorskip('OCP')
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IGESControl import IGESControl_Controller, IGESControl_Writer
    from OCP.Interface import Interface_Static

    IGESControl_Controller.Init_s()
    path = tmp_path / 'bracket.igs'
    # same worker-state trap as the STEP fixture: the writer converts
    # from the cascade unit, so pin it
    assert Interface_Static.SetCVal_s('xstep.cascade.unit', 'MM')
    writer = IGESControl_Writer('MM', 0)
    writer.AddShape(BRepPrimAPI_MakeBox(100.0, 50.0, 25.0).Shape())
    assert writer.Write(str(path))
    geometry = import_file(path)
    assert geometry.length_unit == 'm'
    extent = geometry.node_xyz.max(axis=0) - geometry.node_xyz.min(axis=0)
    assert np.allclose(sorted(extent), [0.025, 0.05, 0.1], atol=1e-6)


def test_a_fresh_process_still_converts_to_meters(tmp_path):
    """The bug this pins: the kernel's unit parameter is unregistered
    until a reader or writer exists, and setting it before that
    silently no-ops — millimeter magnitudes then arrive labeled
    meters, and the first real import read a McMaster motor as 8470
    inches wide. Every other test here is contaminated by its own
    fixture *writer* registering the parameter, so this one imports
    in a subprocess where nothing has."""
    import json
    import pathlib
    import subprocess
    import sys

    path = tmp_path / 'box.step'
    _write_box(path)                       # contaminates THIS process only
    probe = (
        'import json, sys\n'
        'import visualdynamics\n'
        'from visualdynamics.io import import_file\n'
        f'g = import_file({str(path)!r})\n'
        'extent = (g.node_xyz.max(axis=0) - g.node_xyz.min(axis=0))\n'
        'print(json.dumps([visualdynamics.__file__,'
        ' sorted(extent.tolist())]))\n')
    # -P keeps the subprocess off sys.path[0] = cwd: the repo carries a
    # stale frozen copy of the package under dist/*/_internal, and a
    # bare -c one directory away would import that instead of the code
    # under test — the gate saw exactly one such ghost (2026-09-01)
    out = subprocess.run([sys.executable, '-P', '-c', probe],
                         capture_output=True, text=True, check=True,
                         cwd=tmp_path)
    where, extent = json.loads(out.stdout.strip().splitlines()[-1])
    assert 'dist' not in pathlib.Path(where).parts, (
        f'the probe imported a frozen snapshot: {where}')
    assert abs(extent[2] - 0.1) < 1e-6, (
        f'a 100 mm box read {extent[2]} "meters" in a fresh process '
        f'(imported from {where})')


def test_without_the_kernel_the_refusal_names_the_extra(tmp_path,
                                                        monkeypatch):
    """`pip install visualdynamics` has no kernel; the error must say
    what to install rather than being a bare ImportError."""
    path = tmp_path / 'part.step'
    path.write_text('ISO-10303-21;\n', encoding='ascii')
    real_import = builtins.__import__

    def no_ocp(name, *args, **kwargs):
        if name == 'OCP' or name.startswith('OCP.'):
            raise ImportError('No module named OCP')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_ocp)
    with pytest.raises(ValueError, match=r"visualdynamics\[step\]"):
        import_file(path)
