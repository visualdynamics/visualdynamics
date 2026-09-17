"""Every small fixture, generated from the demonstration plate by the
package itself.

This replaces what `generate_airplane.py`, `generate_data.py`,
`generate_geometry.py`, `generate_test_geometry.py` and
`generate_shock.py` made from sdynpy's demo models — and unlike them
it runs in the project's own venv, because everything it writes goes
out through visualdynamics's own exporters: the sdynpy-format files
included. The formats stay foreign; the model and the physics are
this package's own.

    ./.venv/bin/python testdata/generate_plate.py

Products (testdata/plate/, SI; only the UNV declares units):

- geometry.npz / .unv / .exo   the full 169-node meshed plate
- test_geometry.npz            the modal run's nodes, with tracelines
- shapes.npy                   mass-normalized modes to 5 kHz
- frfs.npz / frfs.unv          25 Z responses x 4 references, accelerance
- time.npz                     the impulse response from the corner drive
- psd.npz / spectrum.npz       autospectra / linear spectra of those
- channel_table.vdyn           the survey's channels: 25 accels, 4 forces
- shock.nc4                    a Rattlesnake-format shock recording:
                               synthesized half-sine + ringdown at the
                               plate's own frequencies (see the note at
                               `write_shock` — format exercise, not a run)

The controller runs (modal/random .nc4) come from
`generate_plate_runs.py` (visualdynamics-generators), which drives
real Rattlesnake the way the
drone's generator does.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plate_layout
from common import recompress

import visualdynamics
from visualdynamics import io
from visualdynamics.core.data import Frf, Psd, Spectrum, TimeHistory
from visualdynamics.demo import plate

HERE = pathlib.Path(__file__).parent
OUT = HERE / 'plate'

#: Laid out like a modal test: responses everywhere on the survey grid,
#: a few fixed drive points. All on grid nodes, so every drive point is
#: measured and a fit from this data comes out scaled. One corner and
#: three edge points at the quarter stations — the generic positions.
#: Not the interior: the grid's interior stations all lie on a center
#: line or a diagonal of the square, and a reference there is blind to
#: whichever mode family nulls that line (see plate_layout.DRIVES).
REFERENCES = ['101Z+', '110Z+', '1304Z+', '1310Z+']

#: 2 Hz lines to 2.5 kHz: six elastic modes in band, and the lowest
#: (439 Hz at 2% damping, 17 Hz wide at half power) is nine lines wide.
FREQS = np.arange(2.0, 2501.0, 2.0)
DAMPING = 0.02


def survey(model):
    """The grid the test measures: Z at every survey node.

    Z only, not X and Y as well: the plate lies in the XY plane and its
    modes move normal to it — measured in-plane response down here is
    numerical noise, and carrying it would triple the files to say
    nothing. (The drone, being properly three-dimensional, keeps all
    three.)
    """
    return [f'{node}Z+' for node in plate.instrumented(model)['grid']]


def make_frfs(shapes, responses):
    rows = [dof for dof in responses for _ in REFERENCES]
    columns = [ref for _ in responses for ref in REFERENCES]
    ordinate = shapes.synthesize_frf(FREQS, rows, columns, power=2)
    frfs = Frf(abscissa=FREQS, ordinate=ordinate,
               response_dof=rows, reference_dof=columns,
               ordinate_dim=['acceleration/force'] * len(rows))
    frfs.define_units('m/s**2', reference_units='N')
    return frfs


def corner_column(frfs, responses):
    """The records driven at the first reference, response-ordered."""
    picks = [i for i, ref in enumerate(frfs.reference_dof)
             if ref == REFERENCES[0]]
    assert len(picks) == len(responses)
    return np.asarray(frfs.ordinate)[picks]


def make_time(column, responses):
    """One hammer hit at the corner: the impulse response everywhere,
    derived from the FRFs so the time data, the FRFs and the shapes all
    describe the same plate."""
    ordinate = np.fft.irfft(column, axis=-1)
    dt = 1.0 / (2 * FREQS[-1])
    t = np.arange(ordinate.shape[-1]) * dt
    decay = np.exp(-t / (0.35 * t[-1]))          # a light exponential window
    return TimeHistory(abscissa=t, ordinate=ordinate * decay,
                       response_dof=responses,
                       ordinate_dim=['acceleration'] * len(responses))


def make_channel_table(responses):
    nodes, directions, units, types = [], [], [], []
    for dof in responses:
        nodes.append(dof[:-2])
        directions.append(dof[-2:])
        units.append('m/s^2')
        types.append('Acceleration')
    for dof in REFERENCES:
        nodes.append(dof[:-2])
        directions.append(dof[-2:])
        units.append('N')
        types.append('Force')
    return visualdynamics.ChannelTable({
        'channel': list(range(1, len(nodes) + 1)),
        'node': nodes,
        'direction': directions,
        'unit': units,
        'channel_type': types,
    })


def make_test_geometry(model):
    """The display model of the *controller* survey: exactly the nodes
    the modal run measures (tests/test_correlate.py holds that
    identity), tracelines threaded along the rows and columns that
    have more than one sensor, and the measured patch meshed as
    display panels — every element corner is a measured node, so the
    shading claims nothing the survey did not see. Four quads over the
    grid and one triangle closing the notch the off-grid drive point
    cuts into the last panel."""
    nodes = plate_layout.modal_nodes()
    rows = sorted({n // 100 for n in nodes})
    columns = sorted({n % 100 for n in nodes})
    lines = []
    for r in rows:
        line = [n for n in nodes if n // 100 == r]
        if len(line) > 1:
            lines.append((f'row {r}', line))
    for c in columns:
        line = [n for n in nodes if n % 100 == c]
        if len(line) > 1:
            lines.append((f'column {c}', line))
    elements = [[101, 107, 707, 701], [107, 113, 713, 707],
                [701, 707, 1307, 1301], [707, 713, 1313, 1310],
                [707, 1310, 1307]]
    assert {n for conn in elements for n in conn} == set(nodes), (
        'every measured node carries a panel corner, and only those')
    return visualdynamics.Geometry(
        node_id=nodes,
        node_xyz=np.array([model.position(n) for n in nodes]),
        traceline_id=list(range(1, len(lines) + 1)),
        traceline_desc=[name for name, _line in lines],
        traceline_conn=[np.asarray(line) for _name, line in lines],
        elem_conn=[np.asarray(conn) for conn in elements],
        elem_type=[44, 44, 44, 44, 41],
        length_unit='m')


# ---- the shock recording ---------------------------------------------------
#
# Rattlesnake's *format* without running the controller (the transient
# environment needs an xlsx, a control law and a signal definition to
# stand up — generate_plate_runs.py, in visualdynamics-generators,
# is where real runs live). The
# responses are synthesized but shaped like measured ones: a half-sine
# base pulse, the ringdown of the plate's own first three modes, noise
# under everything. Four shocks in one stream with quiet between them,
# because a recording holding several events has to come back as
# several, not as one long record or an average.

RATE = 8192.0
SHOCKS = 4
QUIET = 0.35              #: seconds of nothing before and after each
PULSE = 0.004             #: half-sine duration, seconds
PEAK = 400.0              #: base acceleration at the crest, m/s^2
NOISE = 0.004             #: noise floor, fraction of the peak

#: the plate's own first three elastic modes (see demo.plate), with
#: plausible test damping — what the article rings at after the pulse
MODES = ((439.2, 0.02), (647.2, 0.015), (823.5, 0.03))

#: measured channels: corners answer hardest on a free plate, the
#: center sits closer to the node lines of the low modes
SHOCK_CHANNELS = (
    (101, 'Z+', 1.80),
    (107, 'Z+', 1.10),
    (113, 'Z+', 1.80),
    (707, 'Z+', 0.60),
    (1307, 'Z+', 1.10),
    (107, 'X+', 0.45),
)
DRIVE = (1, 'Z+', 'shaker')


def one_shock(samples, gain):
    t = np.arange(samples) / RATE
    pulse = np.zeros(samples)
    n = round(PULSE * RATE)
    pulse[:n] = PEAK * np.sin(np.pi * np.arange(n) / n)
    ring = np.zeros(samples)
    live = t >= PULSE
    since = t[live] - PULSE
    for frequency, damping in MODES:
        ring[live] += (np.exp(-damping * 2 * np.pi * frequency * since)
                       * np.sin(2 * np.pi * frequency * since))
    return gain * (pulse + 0.45 * PEAK * ring)


def write_shock(path, rng=None):
    import netCDF4

    rng = np.random.default_rng(4) if rng is None else rng
    per_shock = round((PULSE + 2 * QUIET) * RATE)
    samples = per_shock * SHOCKS
    rows = []
    for _node, _direction, gain in SHOCK_CHANNELS:
        row = rng.normal(0.0, NOISE * PEAK * gain, samples)
        for k in range(SHOCKS):
            at = k * per_shock + round(QUIET * RATE)
            level = gain * (0.7 + 0.1 * k)   # walked up to level
            piece = one_shock(per_shock - round(QUIET * RATE), level)
            row[at:at + piece.size] += piece
        rows.append(row)
    command = rng.normal(0.0, 1e-3, samples)
    for k in range(SHOCKS):
        at = k * per_shock + round(QUIET * RATE)
        command[at:at + round(PULSE * RATE)] += 1.0 + 0.15 * k
    rows.append(command)
    values = np.asarray(rows)

    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.sample_rate = RATE
        ds.createDimension('response_channels', values.shape[0])
        ds.createDimension('time_samples', samples)
        ds.createVariable('time_data', 'f8',
                          ('response_channels', 'time_samples'))[...] = values
        # EnvironmentType.TRANSIENT is 2: what makes this a shock run
        ds.createDimension('environments', 1)
        ds.createVariable('environment_types', 'i4', ('environments',))[:] = [2]
        ds.createVariable('environment_names', str, ('environments',))[:] = \
            np.array(['Shock'], dtype=object)
        group = ds.createGroup('channels')
        nodes = [str(n) for n, _d, _g in SHOCK_CHANNELS] + [str(DRIVE[0])]
        directions = [d for _n, d, _g in SHOCK_CHANNELS] + [DRIVE[1]]
        units = ['m/s^2'] * len(SHOCK_CHANNELS) + ['V']
        # only the drive has a feedback device, which is how a drive is
        # told from a response
        feedback = [''] * len(SHOCK_CHANNELS) + [DRIVE[2]]
        for name, strings in (('node_number', nodes),
                              ('node_direction', directions),
                              ('unit', units),
                              ('feedback_device', feedback),
                              ('channel_type', ['acceleration']
                               * len(SHOCK_CHANNELS) + ['voltage'])):
            group.createVariable(name, str, ('response_channels',))[:] = \
                np.array(strings, dtype=object)
        env = ds.createGroup('Shock')
        env.pulse_duration = PULSE
        env.shocks = SHOCKS
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = plate.build()
    responses = survey(model)
    assert set(REFERENCES) <= set(responses), 'a drive point is not measured'

    shapes = model.eigensolution(maximum_frequency=5000.0, damping=DAMPING)
    elastic = [f for f in shapes.frequency if f > 0]
    print(f'{model.num_nodes} nodes, {len(elastic)} elastic modes to 5 kHz, '
          f'{sum(1 for f in elastic if f <= FREQS[-1])} in band')

    geometry = model.geometry()
    # the geometry files deliberately declare nothing about units —
    # declaring them is the importer's test, and the UNV exporter
    # writes a 164 for a geometry that knows its unit
    geometry.undefine_units()
    io.export_file(geometry, str(OUT / 'geometry.npz'),
                   format='sdynpy_geometry')
    io.export_file(geometry, str(OUT / 'geometry.unv'), format='unv')
    io.export_file(geometry, str(OUT / 'geometry.exo'), format='exodus')
    io.export_file(make_test_geometry(model),
                   str(OUT / 'test_geometry.npz'), format='sdynpy_geometry')
    io.export_file(shapes, str(OUT / 'shapes.npy'), format='sdynpy_shapes')

    frfs = make_frfs(shapes, responses)
    io.export_file(frfs, str(OUT / 'frfs.npz'), format='sdynpy_data')
    io.export_file(frfs, str(OUT / 'frfs.unv'), format='unv')
    column = corner_column(frfs, responses)

    history = make_time(column, responses)
    io.export_file(history, str(OUT / 'time.npz'), format='sdynpy_data')

    psd = Psd(abscissa=FREQS, ordinate=np.abs(column) ** 2,
              response_dof=responses, reference_dof=responses,
              ordinate_dim=['acceleration**2/frequency'] * len(responses))
    io.export_file(psd, str(OUT / 'psd.npz'), format='sdynpy_data')

    spectrum = Spectrum(abscissa=FREQS, ordinate=column,
                        response_dof=responses,
                        ordinate_dim=['acceleration'] * len(responses))
    io.export_file(spectrum, str(OUT / 'spectrum.npz'), format='sdynpy_data')

    make_channel_table(responses).save(str(OUT / 'channel_table.vdyn'))
    write_shock(str(OUT / 'shock.nc4'))

    recompress(*[str(OUT / name) for name in
                 ('frfs.npz', 'time.npz', 'psd.npz', 'spectrum.npz')])
    total = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f'wrote {len(list(OUT.iterdir()))} files, {total:.1f} MB: '
          f'{len(responses)} responses x {len(REFERENCES)} references, '
          f'{len(FREQS)} lines to {FREQS[-1]:.0f} Hz')


if __name__ == '__main__':
    main()
