"""The shared four-unit frame mesh, solved here, against MSC Nastran.

The wiki's finite element models of the frame (Linderholt, Linnaeus
University, 2024) come as Nastran decks with their eigenfrequencies in
MATLAB files. Read with `visualdynamics.io.nastran` — PSOLID and MAT1
into block properties — and solved sparse, the frame's first ten
elastic modes land within 0.2 % of Nastran's on the same mesh: the
agreement oracle for the solid elements (2026-09-30, measured 0.09 %
at worst). The files are not in this repository and do not travel with
the package; point `VD_FRAME_WIKI` at an unpacked copy of
"Linderholt_Files_v2.zip"'s To_Wiki folder to run this. It takes two
minutes and eight gigabytes, so it is `slow` as well.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.slow

WIKI = os.environ.get('VD_FRAME_WIKI')


@pytest.mark.skipif(not WIKI, reason='VD_FRAME_WIKI names the wiki files')
def test_the_shared_frame_mesh_solves_to_nastrans_frequencies():
    from scipy.io import loadmat

    from visualdynamics.core.fem import Model
    from visualdynamics.io import nastran

    root = pathlib.Path(WIKI)
    deck = root / 'Fuselage' / 'Uncoupled' / 'MSC_Nastran' / 'fuselage_uncoupled.dat'
    reference = loadmat(root / 'Fuselage' / 'Uncoupled' / 'Matlab'
                        / 'fuselage_uncoupled.mat')
    key = next(k for k in reference if not k.startswith('__'))
    nastran_hz = np.asarray(reference[key]['f'][0, 0]).ravel()

    geometry = nastran.load(deck, length_unit='m')
    assert sorted(geometry.block_properties) == [11, 12]
    assert geometry.block_properties[11].kind == 'solid'
    model = Model.from_geometry(geometry)
    assert model.structural_mass == pytest.approx(0.6287, abs=5e-4)
    shapes = model.eigensolution(num_modes=16, solver='sparse')
    assert int(np.sum(shapes.frequency == 0.0)) == 6
    elastic = shapes.frequency[6:16]
    assert elastic == pytest.approx(nastran_hz[6:16], rel=2e-3), (
        elastic, nastran_hz[6:16])
