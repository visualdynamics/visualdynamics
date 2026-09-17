"""Rattlesnake importer test against a real controller run (Octave Band
project). Skipped when that file isn't present on this machine."""

import os

import numpy as np
import pytest

import visualdynamics

# A real controller run, wherever this machine keeps it. Read from the
# environment rather than written in: a path under someone's home
# directory is that machine's business and not the repository's, and
# it traveled into the public tree when it was hard-coded here
# (found by the public-tree sweep, 2026-08-25).
REAL_NC4 = os.environ.get(
    'VISUALDYNAMICS_RATTLESNAKE_NC4',
    os.path.expanduser('~/Library/Mobile Documents/com~apple~CloudDocs/'
                       'Claude/Octave Band Control Law/data/'
                       'plate_band_average_k10.nc4'))

pytestmark = pytest.mark.skipif(not os.path.exists(REAL_NC4),
                                reason='real rattlesnake output not available')


@pytest.fixture(scope='module')
def imported():
    return visualdynamics.import_file(REAL_NC4)


def test_channel_table(imported):
    table = imported['channel_table']
    assert isinstance(table, visualdynamics.ChannelTable)
    assert table.num_channels == 36
    assert set(table['unit']) == {'m/s^2', 'N'}
    assert table.dof_strings()[0] == '25X+'


def test_time_data_units_automatic(imported):
    time = imported['time_data']
    assert isinstance(time, visualdynamics.TimeHistory)
    assert time.num_records == 36
    # m/s^2 and N are already SI: ordinate must equal the file contents
    dims = set(time.ordinate_dim)
    assert dims == {'acceleration', 'force'}
    # sample rate 8192 -> dt
    dt = time.abscissa[1] - time.abscissa[0]
    assert dt == pytest.approx(1 / 8192)


def test_specification(imported):
    """The controller's target is a full matrix, read whole when its
    cross terms are real numbers (Brandon, 2026-09-04): this run's
    twenty-four control channels come in groups of eight that are
    fully coherent and in phase, and a virtual point transform needs
    exactly that. A file whose off-diagonal is all zero or NaN wrote
    placeholders and imports as autospectra alone."""
    spec = imported['Random_specification']
    assert isinstance(spec, visualdynamics.Psd)
    assert spec.num_records == 24 * 24
    autos = [i for i in range(spec.num_records)
             if spec.response_dof[i] == spec.reference_dof[i]]
    assert len(autos) == 24
    assert all(d == 'acceleration**2/frequency' for d in spec.ordinate_dim)
    assert not np.iscomplexobj(spec.ordinate), (
        'this requirement is written in phase — it is stored as it reads')
    first = spec.response_dof[autos[0]]
    partners = [i for i in range(spec.num_records)
                if spec.response_dof[i] == first
                and spec.reference_dof[i] != first
                and np.abs(spec.ordinate[i]).max() > 0]
    assert len(partners) == 7, 'coherent with the seven others of its group'
    j = partners[0]
    k = next(i for i in autos if spec.response_dof[i] == spec.reference_dof[j])
    lit = np.real(spec.ordinate[autos[0]]) > 0
    coherence = (np.abs(spec.ordinate[j][lit]) ** 2
                 / (np.real(spec.ordinate[autos[0]][lit])
                    * np.real(spec.ordinate[k][lit])))
    assert np.allclose(coherence, 1.0)


def test_mixed_dimension_display(imported):
    time = imported['time_data']
    disp = time.display_ordinate(visualdynamics.IN_LBF_S)
    accel_row = time.ordinate_dim.index('acceleration')
    force_row = time.ordinate_dim.index('force')
    assert np.allclose(disp[accel_row], time.ordinate[accel_row] / 0.0254)
    assert np.allclose(disp[force_row],
                       time.ordinate[force_row] / 4.4482216152605)
