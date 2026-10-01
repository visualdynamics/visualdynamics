"""Freeze one .vdyn per schema era, so no release can orphan a file.

Run when SCHEMA_VERSION bumps to write that era's file.
`tests/test_vdyn_corpus.py` opens every file here and pins values that
would drift if a loader regressed. Until the first public release the
freeze is soft: the format may drop its past deliberately (Brandon,
2026-08-23 — no released file exists to orphan), and when it does, the
orphaned era's file is deleted and regenerated rather than kept
half-readable. After release the files are what the docstring says:
written once, never refreshed. The project is authored small on
purpose (a few KB): it is a schema exercise, not a dataset, but it
touches every object kind and the optional fields (averaging,
scale_db, complex modal mass and damping, links, roles, project type)
whose absent-with-a-default handling is what future readers must keep.
"""

import numpy as np

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.data import Frf, TimeHistory
from visualdynamics.core.geometry import Geometry
from visualdynamics.core.matches import MatchedModes
from visualdynamics.core.photos import Photos
from visualdynamics.core.report import Report
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.core.shocks import Shock
from visualdynamics.core.sine import (
    SineLevel,
    SineLevelSet,
    SineSweepSpecification,
    SineTone,
)
from visualdynamics.io.native import SCHEMA_VERSION

#: the smallest legal PNG: 1x1, opaque. The corpus needs image *bytes*
#: that survive verbatim, not a picture.
ONE_PIXEL_PNG = bytes.fromhex(
    '89504e470d0a1a0a0000000d494844520000000100000001080200000090'
    '7753de0000000c4944415408d763f8cfc00000030101'
    '00c9fe92ef0000000049454e44ae426082')


def build_project() -> visualdynamics.Project:
    geometry = Geometry(
        node_id=[1, 2, 3, 4],
        node_xyz=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                  [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
    geometry.add_beams([1, 2, 3, 4])
    geometry.length_unit = 'm'

    t = np.arange(256) / 256.0
    history = TimeHistory(
        t, [np.sin(2 * np.pi * 12 * t), 0.5 * np.cos(2 * np.pi * 20 * t)],
        response_dof=['1X+', '2Y+'],
        ordinate_dim='acceleration', ordinate_unit='m/s**2')
    history.averaging = Averaging(frame_length=64, overlap=0.5)
    history.shocks = (Shock(0.125, 0.25),)

    frequency = np.arange(33) * 4.0
    frf = Frf(frequency,
              np.outer([1.0, 2.0], 1.0 / (1.0 + 1j * frequency / 40.0)),
              response_dof=['1X+', '2Y+'], reference_dof='1X+',
              ordinate_dim='acceleration/force',
              ordinate_unit='m/s**2', reference_unit='N')

    coordinate = ['1X+', '2Y+', '3X+', '4Z+']
    shapes = ShapeSet(
        frequency=[12.5, 47.25], damping=[0.01, 0.032],
        coordinate=coordinate,
        shape_matrix=np.array([[1.0, -0.5, 0.25, 0.0],
                               [0.1 + 0.2j, 0.3 - 0.1j, -0.4j, 1.0]]),
        modal_mass=[1.0, 1.5 + 0.25j],
        modal_damping=[0.02, 0.05 - 0.01j],
        mass_unit='kg',
        comment=['first bending', 'complex import'])

    table = ChannelTable({'channel': [1, 2], 'node': ['1', '2'],
                          'direction': ['X+', 'Y+'],
                          'unit': ['m/s**2', 'm/s**2'],
                          'serial number': ['SN-1', 'SN-2']})

    tone = SineTone(
        name='fundamental', start_time=0.0,
        frequency=[20.0, 200.0], amplitude=[[1.0], [2.0]],
        segment_type=[1], segment_rate=[2.0],
        warning_upper=[[1.5], [3.0]], abort_upper=[[2.0], [4.0]])
    sine_spec = SineSweepSpecification(
        tones=[tone], response_dof=['1X+'],
        ordinate_dim='acceleration', ordinate_unit='m/s**2',
        comment='schema exercise sweep')
    lines = np.array([20.0, 63.0, 200.0])
    levels = SineLevelSet([SineLevel(
        lines, [1.02 * lines / lines], tone='fundamental', onset=1.25,
        seconds=[0.0, 30.0, 60.0], response_dof=['1X+'],
        ordinate_dim='acceleration', ordinate_unit='m/s**2')])

    report = Report(
        title='Frozen Schema Report',
        blocks=[{'kind': 'text',
                 'text': 'One block, so the JSON round-trips.'}],
        marking='CUI', marking_color='red')

    matches = MatchedModes('Shapes', 'Shapes', pairs=[[0, 0], [1, 1]],
                           macs=[1.0, 0.978], first_geometry='Geometry',
                           second_geometry='Geometry')

    photos = Photos(names=['setup'], formats=['png'],
                    images=[ONE_PIXEL_PNG])

    project = visualdynamics.Project('Frozen Schema Exercise')
    project.project_type = 'Modal Test'
    project.add('Geometry', geometry)
    project.add('Time History', history)
    project.add('FRF', frf)
    project.add('Shapes', shapes)
    project.add('Channel Table', table)
    project.add('Sine Specification', sine_spec)
    project.add('Sine Levels', levels)
    project.add('Report', report)
    project.add('Matches', matches)
    project.add('Photos', photos)
    # through the project's own verbs, so the file carries provenance
    # — the staleness fingerprints have to survive a save too
    psds = project.compute_psds('Time History')
    project[psds].scale_db = 3
    srs = project.compute_srs('Time History')
    project.link('Geometry', 'Time History', psds, srs, 'FRF', 'Shapes')
    project.set_basis('Geometry', 'Time History', psds, srs, 'FRF',
                      'Shapes')
    return project


if __name__ == '__main__':
    import os
    root = os.path.join(os.path.dirname(__file__), 'vdyn_corpus')
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f'schema{SCHEMA_VERSION}.vdyn')
    if os.path.exists(path):
        raise SystemExit(f'{path} is frozen; a schema bump writes a '
                         'new file, never this one again')
    build_project().save(path)
    print(f'froze {path} ({os.path.getsize(path)} bytes)')
