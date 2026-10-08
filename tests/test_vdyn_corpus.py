"""Frozen .vdyn files from each schema era open forever.

The files under `testdata/vdyn_corpus/` are written by
`testdata/generate_vdyn_corpus.py` when the schema changes and pinned
here; whatever the format grows into, these must keep opening with
their values intact. Until the first public release the freeze is soft
— the format may drop its past deliberately, deleting and regenerating
the orphaned era's file (Brandon, 2026-08-23: no released file exists
to orphan). After release a frozen file is written once and never
refreshed: the moment one fails, a change has orphaned every project
saved before it, which is the one regression a public format cannot
ship.
"""

import glob
import os
import shutil

import h5py
import numpy as np
import pytest
from conftest import fixture_path

from visualdynamics.io import native

CORPUS = fixture_path('vdyn_corpus')


@pytest.mark.parametrize('name', sorted(
    os.path.basename(p) for p in glob.glob(str(CORPUS) + '/*.vdyn')))
def test_every_frozen_era_still_opens(name):
    project = native.load(os.path.join(str(CORPUS), name))
    assert isinstance(project, native.Project)
    assert len(project) > 0


def test_schema1_values_are_intact():
    """Spot values from the frozen file, pinned the day it was written
    — every object kind the format holds, and every optional field
    (averaging, shocks, scale_db, SRS parameters, sine tones and
    levels, marking, provenance) whose read path is what future schema
    changes are most likely to disturb."""
    p = native.load(os.path.join(str(CORPUS), 'schema1.vdyn'))
    assert p.project_type == 'Modal Test'
    assert p.object_groups == [{'members': ['Geometry', 'Time History',
                                    'Time History PSDs',
                                    'Time History SRS', 'FRF', 'Shapes'],
                        'role': 'Basis'}]
    assert p.provenance['Time History PSDs'] == {
        'verb': 'compute_psds', 'source': 'Time History', 'params': {},
        'state': ['averaging', [64, 0.5, 'hann', 1, 0.0]]}
    assert p.provenance['Time History SRS']['state'] == \
        ['shocks', [[0.125, 0.25]]]

    geometry = p['Geometry']
    assert geometry.length_unit == 'm'
    assert np.allclose(geometry.node_xyz[3], [1.0, 0.5, 0.0])

    history = p['Time History']
    assert history.ordinate_unit == ['m/s**2'] * 2
    assert history.averaging.frame_length == 64
    assert history.averaging.overlap == 0.5
    assert [(s.start, s.duration) for s in history.shocks] == \
        [(0.125, 0.25)]

    psds = p['Time History PSDs']
    assert psds.scale_db == 3
    assert psds.ordinate[0][3] == pytest.approx(1.0 / 12.0)

    srs = p['Time History SRS']
    assert srs.q == pytest.approx(10.0)
    assert srs.kind == 'maximax'

    frf = p['FRF']
    assert frf.reference_dof == ['1X+'] * 2
    assert complex(frf.ordinate[1][10]) == pytest.approx(1.0 - 1.0j)

    shapes = p['Shapes']
    assert shapes.frequency.tolist() == [12.5, 47.25]
    assert shapes.modal_mass[1] == pytest.approx(1.5 + 0.25j)
    assert shapes.modal_damping[1] == pytest.approx(0.05 - 0.01j)
    assert shapes.shape_matrix[1, 2] == pytest.approx(-0.4j)
    assert shapes.mass_unit == 'kg'

    table = p['Channel Table']
    # written as 'serial number'; the schema spells it with an
    # underscore and the loader's aliasing maps it there
    assert list(table['serial_number']) == ['SN-1', 'SN-2']

    spec = p['Sine Specification']
    tone = spec.tones[0]
    assert tone.name == 'fundamental'
    assert tone.frequency.tolist() == [20.0, 200.0]
    assert tone.amplitude.tolist() == [[1.0], [2.0]]
    assert sorted(tone.limits) == ['abort_upper', 'warning_upper']

    level = p['Sine Levels'].levels[0]
    assert level.tone == 'fundamental'
    assert level.onset == pytest.approx(1.25)
    assert level.seconds.tolist() == [0.0, 30.0, 60.0]

    report = p['Report']
    assert report.title == 'Frozen Schema Report'
    assert report.marking == 'CUI'
    assert report.marking_color == 'red'
    assert report.blocks == [{'kind': 'text',
                              'text': 'One block, so the JSON '
                                      'round-trips.'}]

    matches = p['Matches']
    assert matches.pairs == [[0, 0], [1, 1]]
    assert matches.macs == [1.0, 0.978]
    assert matches.first_geometry == 'Geometry'

    photos = p['Photos']
    assert photos.names == ['setup']
    assert photos.formats == ['png']
    assert photos.images[0][:8] == bytes.fromhex('89504e470d0a1a0a'), \
        'the PNG bytes ride verbatim'


def test_a_newer_schema_is_refused_not_half_loaded(tmp_path):
    newer = tmp_path / 'from_the_future.vdyn'
    shutil.copy(os.path.join(str(CORPUS), 'schema1.vdyn'), newer)
    with h5py.File(newer, 'r+') as f:
        f.attrs['visualdynamics_schema'] = native.SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match='newer Visual Dynamics'):
        native.load(newer)


def test_a_missing_stamp_is_not_our_file(tmp_path):
    """Every writer stamps, so an unstamped HDF5 file is foreign —
    naming that beats a KeyError from the first attribute read."""
    foreign = tmp_path / 'unstamped.vdyn'
    shutil.copy(os.path.join(str(CORPUS), 'schema1.vdyn'), foreign)
    with h5py.File(foreign, 'r+') as f:
        del f.attrs['visualdynamics_schema']
    with pytest.raises(ValueError, match='not a Visual Dynamics file'):
        native.load(foreign)
