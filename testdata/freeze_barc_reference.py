"""Freeze the BARC's shared reference modes into `testdata/barc/`.

    ./.venv/bin/python testdata/freeze_barc_reference.py BARC_field_and_lab_data.mat

The source is the data file shared on the SEM Dynamic Substructuring
Focus Group wiki (https://wiki.sem.org/wiki/BARC, "Random Vibration
Data"): a MATLAB structure whose `field_modes` and `field_geometry`
hold a finite element model's free-free modes of the BARC — thirty
modes, six of them rigid, at 118 translational degrees of freedom on 59
points. Only those numbers are kept, as data the tests and the
`visualdynamics.demo.barc` comparison read; the 267 MB file stays
outside the repository.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import scipy.io as sio

OUT = pathlib.Path(__file__).resolve().parent / 'barc' / 'reference_modes.npz'


def main(source: str) -> None:
    data = sio.loadmat(source, squeeze_me=True, struct_as_record=False,
                       variable_names=['BARCdata'])['BARCdata']
    modes, geometry = data.field_modes, data.field_geometry
    dof = [str(d) for d in modes.DOF]
    if dof != [str(d) for d in geometry.DOF]:
        raise ValueError('the modes and the geometry list different DOFs')
    np.savez_compressed(
        OUT,
        frequency=np.asarray(modes.mode_frequency, dtype=float),
        damping=np.asarray(modes.mode_damping_ratio, dtype=float),
        dof=np.array(dof),
        shape=np.asarray(modes.mode_shape, dtype=float),
        coordinates=np.asarray(geometry.coordinates, dtype=float),
        units=np.array(str(geometry.units)))
    print(f'wrote {OUT}: {len(dof)} DOFs, {len(modes.mode_frequency)} modes')


if __name__ == '__main__':
    main(sys.argv[1])
