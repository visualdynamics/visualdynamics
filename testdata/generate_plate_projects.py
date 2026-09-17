"""Four ready-to-open .vdyn projects, one per workflow, skeleton full.

Each is the plate's own data assembled the way the workflow guides
assemble it — modal, random vibration, transient, shock — with every
gray slot of its project type filled: geometry, photos, channel
table, the data, the derived spectra, the report. They exist to be
dropped onto the window for workflow testing, and to sit behind the
documentation's screenshots.

    ./.venv/bin/python testdata/generate_plate_projects.py

Writes stressdata/plate_projects/{modal,random,transient,shock}.vdyn
— outside the repository, because each embeds megabytes of run data
that testdata already carries once. Regenerating takes seconds; the
transient one needs stressdata/plate/transient.nc4 first
(generate_plate_runs.py --transient, in visualdynamics-generators)
and is skipped with a note when
it is absent.

The photos slot gets rendered scenes of the article rather than
photographs — this article has never been photographed, having never
existed — captioned as what they are.

Every project is checked against `missing_expectations` before it is
saved: a generator whose output opens with gray slots would be
writing the very thing it exists to prevent.
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import visualdynamics
from visualdynamics.core.photos import Photos
from visualdynamics.core.report import missing_expectations

HERE = pathlib.Path(__file__).parent
PLATE = HERE / 'plate'
STRESS = HERE.parent / 'stressdata' / 'plate'
OUT = HERE.parent / 'stressdata' / 'plate_projects'


def photos(*scenes) -> Photos:
    """Rendered stand-ins for setup photographs: (name, geometry) pairs
    drawn offscreen and kept as PNG bytes, exactly as imported photos
    would be.

    Rendered on a neutral gray — a photograph belongs to neither app
    theme, and the first cut used the light plot palette, which read
    as one broken light-mode figure in a dark-mode report rather than
    as a picture of hardware."""
    import tempfile

    from visualdynamics.theme import theme as resolve_theme

    studio = dict(resolve_theme('dark'))
    studio['scene_background'] = '#6b6e74'
    studio['scene_background_top'] = '#43464c'

    names, formats, images = [], [], []
    for name, geometry in scenes:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        geometry.plot(screenshot=path, show=False, theme=studio)
        names.append(name)
        formats.append('png')
        images.append(pathlib.Path(path).read_bytes())
        os.remove(path)
    return Photos(names=names, formats=formats, images=images)


def survey_geometry():
    geometry = visualdynamics.import_file(str(PLATE / 'test_geometry.npz'))
    geometry.define_units('m')
    return geometry


def fem_pair():
    geometry = visualdynamics.import_file(str(PLATE / 'geometry.npz'))
    geometry.define_units('m')
    shapes = visualdynamics.import_file(str(PLATE / 'shapes.npy'),
                                        mass_unit='kg')
    return geometry, shapes


def finish(project, template, summary):
    """Report with the summary written in, then the completeness check."""
    project.generate_report(template)
    report = project.report
    for block in report.blocks:
        if block.get('kind') == 'text' and not block.get('text'):
            block['text'] = summary
            break
    missing = [(name, optional) for name, _cls, _icon, _n, optional, _role
               in missing_expectations(project.project_type, project,
                                       placed=project.placed())]
    required = [name for name, optional in missing if not optional]
    assert not required, f'{project.name}: unfilled slots {required}'
    return project


def modal_project():
    project = visualdynamics.Project('Plate Modal Survey')
    project.project_type = 'Modal Test'
    # the spectra save alone: its accepted captures are the time data,
    # exactly as examples/modal_workflow.py reads a survey
    project.import_file(str(PLATE / 'modal_spectra.nc4'))
    geometry = survey_geometry()
    project.add('Geometry', geometry)
    project.add('Photos', photos(('Test article, survey grid', geometry)))
    measured = [name for name in project.names]
    project.link(*measured)
    project.set_basis(*measured)

    project.compute_psds(project.basis.time_history)
    # limit=5: the article has five modes in the band — four distinct
    # peaks and the repeated pair's second tooth at 1142 Hz, which the
    # shape test reaches through the first tooth's exclusion. Left to
    # run uncapped, the loop pulled twenty-three 'modes' out of this
    # run's noise floor — eleven channels cannot distinguish that many
    project.fit_modes(project.basis.frf, bounds=(300.0, 1300.0),
                      limit=5, name='Identified Modes')

    fem_geometry, fem_shapes = fem_pair()
    project.add('FEM Geometry', fem_geometry)
    project.add('FEM Shape Set', fem_shapes)
    project.link('FEM Geometry', 'FEM Shape Set')
    project.match_modes(project.basis.shapes, fem_shapes, threshold=0.7)
    return finish(project, 'modal', (
        'Demonstration modal survey of the 12 in x 12 in x 0.5 in '
        'aluminum plate: a burst-random Rattlesnake run on two '
        'shakers, modes identified from the measured FRFs and '
        'correlated against the finite element model.'))


def random_project():
    project = visualdynamics.random_vibration_run(str(PLATE / 'random.nc4'))
    geometry = visualdynamics.import_file(str(PLATE / 'geometry.npz'))
    geometry.define_units('m')
    project.add('Geometry', geometry)
    project.add('Photos', photos(('Test article', geometry)))
    project.link(*[name for name in project.names])
    project.set_basis(*[name for name in project.names])
    # no compute calls: random_vibration_run already worked the run up
    # — PSDs, the octave banding, the coherence — and doing it again
    # here shipped a project with '(2)' beside all three
    return finish(project, 'random', (
        'Demonstration random vibration test of the aluminum plate: '
        'eight control channels against four shakers, controlled to a '
        'shaped acceleration specification.'))


def transient_project():
    run = STRESS / 'transient.nc4'
    if not run.exists():
        print('transient.nc4 absent — run '
              'generate_plate_runs.py --transient first (it lives in '
              'visualdynamics-generators); skipping')
        return None
    project = visualdynamics.Project('Plate Transient Replication')
    project.import_file(str(run))
    assert project.project_type == 'Transient', project.project_type
    geometry = visualdynamics.import_file(str(PLATE / 'geometry.npz'))
    geometry.define_units('m')
    project.add('Geometry', geometry)
    project.add('Photos', photos(('Test article', geometry)))
    project.link(*[name for name in project.names])
    project.set_basis(*[name for name in project.names])
    # the spectral pair: the target's own PSD arrives as a
    # specification, the measurement's as data — see the workflow
    # guide. No SRS: that judgment belongs to the shock type
    project.compute_psds(project.transient_specification)
    project.compute_psds(project.time_history)
    return finish(project, 'transient', (
        'Demonstration transient replication on the aluminum plate: '
        'the target waveform is the plate\'s own response to a '
        'burst at the corner drive, replicated by two shakers.'))


def shock_project():
    project = visualdynamics.Project('Plate Shock Series')
    project.import_file(str(PLATE / 'shock.nc4'))
    # a shock recording is indistinguishable from a transient at the
    # file level — the run arrives as Transient and the engineer says
    # what the test was, exactly as the workflow guide has it
    project.project_type = 'Shock'
    geometry = visualdynamics.import_file(str(PLATE / 'geometry.npz'))
    geometry.define_units('m')
    # the shock machine's drive channel measures at the fixture
    # interface under the plate's corner, not on the article — the
    # node exists so the channel table has somewhere to point
    geometry.add_node([0.0, 0.0, -0.05], node_id=1)
    project.add('Geometry', geometry)
    project.add('Photos', photos(('Test article', geometry)))
    project.link(*[name for name in project.names])
    project.set_basis(*[name for name in project.names])

    # The recommended shock workflow, in order (Brandon, 2026-08-25):
    # the recording is filtered first, and everything judged is
    # computed from the *filtered* record — the SRS, the density, and
    # the motion chain alike. Filtering after computing would leave
    # three answers that disagree about which record they describe.
    #
    # The corner is the record's own suggestion, a tenth of the
    # sample rate: the frequency where the integrate/differentiate
    # round trip was measured at about 2% (core.filters), and above
    # any content a shock specification is written to.
    raw = project.time_history
    # the events are marked on the *recording*, before anything is
    # derived from it: `core.filters` carries the marks onto whatever
    # it makes, so the filtered record, its SRS and its motion all
    # number the same events. Detecting after the filter instead left
    # the recording itself unmarked, and the report's own summary —
    # which counts the events on the recording — had nothing to count
    project.detect_shocks(raw)
    filtered = project.filter_data(raw)
    srs_name = project.compute_srs(filtered)
    velocity = project.integrate(filtered)
    project.integrate(velocity)
    srs = project[srs_name]

    # the requirement the series was walked up against: a tolerance
    # band around the final event's spectrum, authored the way a shock
    # specification is — breakpoints, not a copied curve
    from visualdynamics.core.data import ShockSpecification

    frequencies = np.asarray(srs.abscissa, dtype=float)
    top = np.asarray(srs.ordinate, dtype=float).max(axis=0)
    level = 10.0 ** np.interp(np.log10(frequencies),
                              np.log10([frequencies[0], 400.0, 2000.0,
                                        frequencies[-1]]),
                              np.log10([top.max() * 0.05, top.max() * 0.9,
                                        top.max() * 1.1, top.max() * 1.1]))
    project.add('Shock Specification', ShockSpecification(
        abscissa=frequencies, ordinate=level[None, :],
        response_dof=[srs.response_dof[0]],
        ordinate_dim='acceleration', ordinate_unit='m/s**2',
        q=srs.q, kind=srs.kind,
        abort_upper=level[None, :] * 10.0 ** 0.3,
        abort_lower=level[None, :] * 10.0 ** -0.3))
    # the authored specification joins the measured group like the rest
    project.link(*[name for name in project.names])
    project.set_basis(*[name for name in project.names])
    return finish(project, 'shock', (
        'Demonstration shock series on the aluminum plate: four '
        'half-sine events walked up to level, judged by their shock '
        'response spectra against the specification band.'))


def sine_project():
    run = STRESS / 'sine.nc4'
    if not run.exists():
        print('sine.nc4 absent — run generate_plate_sine.py first (it '
              'lives in visualdynamics-generators); skipping')
        return None
    project = visualdynamics.Project('Plate Sine Sweep')
    project.import_file(str(run))
    assert project.project_type == 'Sine Sweep', project.project_type
    geometry = visualdynamics.import_file(str(PLATE / 'geometry.npz'))
    geometry.define_units('m')
    project.add('Geometry', geometry)
    project.add('Photos', photos(('Test article', geometry)))
    project.link(*[name for name in project.names])
    project.set_basis(*[name for name in project.names])
    # the levels each tone actually held, read out of the recording by
    # the tracking demodulator — the sweep counterpart of the random
    # workflow's PSDs
    project.extract_sine(project.time_history)
    return finish(project, 'sine', (
        'Demonstration sine sweep on the aluminum plate: four '
        'simultaneous tones on two shakers, each swept on its own '
        'schedule, with the achieved levels extracted from the '
        'recording and judged against the specification bands.'))


def sysid_project():
    stream = STRESS / 'sysid_stream.nc4'
    if not stream.exists():
        print('sysid_stream.nc4 absent — run generate_plate_sysid.py '
              'first (it lives in visualdynamics-generators); skipping')
        return None
    project = visualdynamics.Project('Plate System ID')
    # everything computes from the *streamed* save (Brandon,
    # 2026-08-25): it carries the channel table, so the readings come
    # out with real node labels and engineering units, and the
    # geometry links. The spectral package cannot give any of that —
    # it records hardware channel numbers and no units, which is what
    # its import tests document — so the package stays the importer's
    # concern and the demonstration works the data up itself.
    project.import_file(str(stream))
    project.project_type = 'System ID'
    # the two phases named for what they are, quiet first — the same
    # reading the app's import question applies
    histories = sorted(
        (name for name in project.names
         if type(project[name]).__name__ == 'TimeHistory'),
        key=lambda name: float(np.asarray(project[name].ordinate).std()))
    project.rename(histories[0], 'Noise Time History')
    project.rename(histories[1], 'Excitation Time History')
    geometry = visualdynamics.import_file(str(PLATE / 'geometry.npz'))
    geometry.define_units('m')
    project.add('Geometry', geometry)
    project.add('Photos', photos(('Test article', geometry)))
    project.link(*[name for name in project.names])
    project.set_basis(*[name for name in project.names])
    # H1, the controller's own sysid estimator: the plant is measured
    # driving through known excitation, so the noise is on the response
    project.compute_frfs('Excitation Time History', 'H1')
    project.compute_multiple_coherence('Excitation Time History')
    # the two densities the ratio reading divides — identically framed
    # on purpose, the ambient borrowing the excitation's frames, since
    # a ratio of densities only exists on shared lines (the signal-to-
    # noise is a *reading* of this pair, never a third object)
    project.compute_psds('Excitation Time History')
    project['Noise Time History'].averaging = \
        project['Excitation Time History'].averaging
    project.compute_psds('Noise Time History')
    return finish(project, 'sysid', (
        'Demonstration system identification of the aluminum plate: '
        'the ambient noise measured first, then the driven '
        'excitation, with the plant FRFs, the multiple coherence and '
        'the signal-to-noise of the measurement all computed from '
        'the recording.'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (modal_project, random_project, transient_project,
                  shock_project, sine_project, sysid_project):
        project = build()
        if project is None:
            continue
        name = build.__name__.replace('_project', '')
        path = OUT / f'{name}.vdyn'
        project.save(str(path))
        print(f'{path.name}: {len(project)} objects, '
              f'{path.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
