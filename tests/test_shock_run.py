"""A Rattlesnake shock run, from the file to the project it opens as.

The file is `testdata/plate/shock.nc4` — four shocks in one
recording, six accelerometers and a drive. It is written in
Rattlesnake's format by `testdata/generate_plate.py` rather than
produced by the controller; see that module for why.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import Srs, TimeHistory
from visualdynamics.core.report import PROJECT_TYPES, project_expectations
from visualdynamics.io.rattlesnake import environment_kinds, project_type, run_kind


@pytest.fixture
def shock_file():
    return fixture_path('plate', 'shock.nc4')


# ---- what the file says it is -------------------------------------------


def test_the_run_says_it_is_a_transient(shock_file):
    assert environment_kinds(shock_file) == {'Shock': 'transient'}
    assert run_kind(shock_file) == 'transient'


def test_a_transient_run_is_a_transient_project(shock_file):
    """The controller's transient environment replicates a waveform. A
    shock test meets an SRS and it cannot run one, so a transient run is
    not a shock test however much the fixture's pulses look like one."""
    assert project_type(shock_file) == 'Transient'


def test_shock_is_a_project_type_of_its_own():
    assert 'Shock' in PROJECT_TYPES


# ---- what comes out of it -----------------------------------------------


def test_the_channels_come_back_with_their_units(shock_file):
    loaded = visualdynamics.import_file(shock_file)
    history = loaded['time_data']
    assert isinstance(history, TimeHistory)
    assert history.response_dof[:2] == ['101Z+', '107Z+']
    assert history.ordinate_dim[0] == 'acceleration'
    assert history.ordinate_unit[0] == 'm/s**2'


def test_the_drive_is_told_from_the_responses(shock_file):
    """A shock machine's command is a channel like any other in the
    file, and only its feedback device says otherwise."""
    loaded = visualdynamics.import_file(shock_file)
    history = loaded['time_data']
    assert history.ordinate_dim[-1] == 'voltage'
    assert history.response_dof[-1] == '1Z+'
    assert history.ordinate_dim.count('acceleration') == 6


def test_the_recording_holds_every_shock(shock_file):
    """Four events in one stream, not one run of noise.

    Counted off an envelope rather than off the trace: the article
    rings down after each pulse and a raw threshold counts every zero
    crossing of that ringing, which reads 69 events where there are
    four.
    """
    history = visualdynamics.import_file(shock_file)['time_data']
    trace = np.abs(history.ordinate[0])
    step = int(0.05 * history.sample_rate)
    blocks = trace[:len(trace) // step * step].reshape(-1, step).max(axis=1)
    loud = blocks > 0.25 * blocks.max()
    events = int(np.sum(np.diff(loud.astype(int)) == 1)) + int(loud[0])
    assert events == 4


# ---- what it is for -----------------------------------------------------


def test_a_shock_spectrum_comes_off_the_record(shock_file):
    """The reason the file exists: an SRS of the measured transient,
    and one that behaves — flat at the top, at the peak the channel
    actually saw.

    Taken to 4 kHz, not to 1. The asymptote is what an oscillator too
    stiff to respond reads, and "too stiff" means stiff against
    everything in the record: the article rings at 824 Hz here, and a
    2 kHz oscillator still answers that ringing well above the peak.
    """
    history = visualdynamics.import_file(shock_file)['time_data']
    accelerations = [i for i, dim in enumerate(history.ordinate_dim)
                     if dim == 'acceleration']
    only = TimeHistory(
        history.abscissa, history.ordinate[accelerations],
        response_dof=[history.response_dof[i] for i in accelerations],
        ordinate_dim=['acceleration'] * len(accelerations))
    spectra = only.compute_srs(low=20.0, high=4000.0)
    assert isinstance(spectra, Srs)
    assert spectra.num_records == len(accelerations)
    for row, record in zip(spectra.ordinate, only.ordinate):
        assert row[-1] == pytest.approx(np.abs(record).max(), rel=0.05)


def test_the_corner_answers_harder_than_the_center(shock_file):
    """A sanity check on the record rather than on the math: a free
    plate's corners amplify and its center sits nearer the low modes'
    node lines, so the corner's spectrum sits above the center's
    everywhere."""
    history = visualdynamics.import_file(shock_file)['time_data']
    rows = {dof: i for i, dof in enumerate(history.response_dof)}
    pair = [rows['707Z+'], rows['101Z+']]
    only = TimeHistory(history.abscissa, history.ordinate[pair],
                       response_dof=['707Z+', '101Z+'],
                       ordinate_dim=['acceleration'] * 2)
    near, far = only.compute_srs(low=20.0, high=1000.0).ordinate
    assert (far > near).all()


# ---- the project it opens as --------------------------------------------


def test_the_shock_project_expects_a_target_and_a_spectrum():
    wanted = {name for name, *_rest in project_expectations('Shock')}
    assert 'Shock Specification' in wanted
    assert 'SRS' in wanted
    assert 'Multiple Coherence' not in wanted, (
        'coherence is a statement about a stationary average, and a '
        'shock is one transient event')


def test_importing_a_transient_run_types_the_project(window, pump,
                                                     shock_file):
    window.import_paths([shock_file])
    pump()
    assert window.project_type == 'Transient'


def test_the_shock_template_builds_a_report(window, pump, shock_file):
    window.import_paths([shock_file])
    pump()
    name = window.project.generate_report('shock')
    report = window.objects[name]
    assert report.title == 'Shock Test Report'
    kinds = [block.get('kind') for block in report.blocks]
    assert kinds[0] == 'text' and kinds[-1] == 'text'
    assert 'plot' in kinds


def test_the_motion_chain_joins_the_report_when_present():
    """Filtered acceleration, velocity and displacement each get a
    figure between the measured events and the requirement, with a
    Motion section explaining the derivation — and a project without
    the chain gets none of it, not empty slots (Brandon, 2026-08-24).
    The filtered record is found by provenance, the integrals by what
    they measure, so nobody's naming is load-bearing."""
    import numpy as np

    import visualdynamics
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.report import shock_template
    from visualdynamics.core.shocks import Shock

    rng = np.random.default_rng(9)
    t = np.arange(16384) / 4096.0
    project = visualdynamics.Project('Shock')
    history = TimeHistory(t, rng.standard_normal((2, len(t))),
                          response_dof=['101Z+', '102Z+'],
                          ordinate_dim='acceleration')
    history.shocks = [Shock(1.0, 0.5)]
    project.add('Time Data', history)

    bare = shock_template(project, links=project.links)
    texts = ' '.join(block.get('text', '') for block in bare.blocks)
    assert '## The Motion' not in texts

    velocity = project.integrate(project.filter_data('Time Data'))
    project.integrate(velocity)
    report = shock_template(project, links=project.links)
    captions = [block.get('caption', '') for block in report.blocks
                if block.get('kind') == 'plot']
    order = [captions.index(next(c for c in captions if start in c))
             for start in ('Measured', 'Filtered',
                           'Integrated velocity',
                           'Integrated displacement',
                           'Shock specification')]
    assert order == sorted(order), \
        'measured, then the chain, then the requirement'
    texts = ' '.join(block.get('text', '') for block in report.blocks)
    assert '## The Motion' in texts
    assert 'zero-phase {{Time Data.filter_description}}' in texts, \
        'the prose names the filter live, off the record that carries it'


def test_the_shock_report_reads_frequency_through_the_scalogram():
    """The density left the shock report (Brandon, 2026-08-29): a PSD
    of a transient recording carries a level set by how much quiet
    air was captured, and nothing in the shock judgment reads it.
    The scalogram stands where it stood — where in frequency each
    event's energy sat, with *when* still attached."""
    import numpy as np

    import visualdynamics
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.report import shock_template
    from visualdynamics.core.shocks import Shock

    rng = np.random.default_rng(4)
    t = np.arange(16384) / 4096.0
    project = visualdynamics.Project('Shock')
    history = TimeHistory(t, rng.standard_normal((2, len(t))),
                          response_dof=['101Z+', '102Z+'],
                          ordinate_dim='acceleration')
    history.shocks = [Shock(1.0, 0.5)]
    project.add('Time Data', history)
    report = shock_template(project, links=project.links)
    modes = [block.get('mode') for block in report.blocks
             if block.get('kind') == 'plot']
    assert 'scalogram' in modes, 'the frequency figure is the scalogram'
    texts = ' '.join(block.get('text', '') for block in report.blocks)
    assert 'Power spectral density' not in texts
    assert not any('spectral density' in block.get('caption', '')
                   for block in report.blocks)
    assert 'names the source of an unexpected SRS peak' in texts, \
        'the prose says what the figure is for'


def test_the_shock_skeleton_offers_no_psd_slot():
    from visualdynamics.core.report import project_expectations

    names = [slot[0] for slot in project_expectations('Shock')]
    assert 'PSD' not in names, \
        'the density left the shock workflow entirely'
    assert 'SRS' in names
    assert names.count('Time History') == 4, (
        'the recording, then filtered, velocity and displacement — each '
        'slot named for its type, so the same label four times')
