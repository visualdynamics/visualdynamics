"""The four demonstration projects build complete, one per workflow.

Marked `slow` like the demonstration drone: each build runs a real
fit, real spectra and offscreen renders, and what they guard — that
`generate_plate_projects.py` still fills every slot of every project
type — is checked when the demos or the workflows change, not on
every gate. Run with `pytest -m slow`.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'testdata'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def builders():
    return importlib.import_module('generate_plate_projects')


@pytest.mark.parametrize('name,project_type', [
    ('modal_project', 'Modal Test'),
    ('random_project', 'Random Vibration'),
    ('transient_project', 'Transient'),
    ('shock_project', 'Shock'),
    ('sine_project', 'Sine Sweep'),
    ('sysid_project', 'System ID'),
])
def test_the_project_builds_with_no_gray_slots(builders, name,
                                               project_type):
    """The builders assert completeness themselves before returning —
    this holds that they still do, for every type, whenever the
    skeleton or the workflows move."""
    needs = {'transient_project': ['transient.nc4'],
             'sine_project': ['sine.nc4'],
             'sysid_project': ['sysid_stream.nc4', 'sysid_data.nc4']}
    for stem in needs.get(name, ()):
        if not (builders.STRESS / stem).exists():
            pytest.skip(f'{stem} not generated on this machine')
    project = getattr(builders, name)()
    assert project is not None
    assert project.project_type == project_type
    assert 'Report' in project
    assert project.basis is not None
    if name == 'modal_project':
        # five and no more: four distinct CMIF peaks and the repeated
        # pair's second tooth at 1142 Hz, which the FEM shows too.
        # Left uncapped the fit once dredged twenty-three 'modes' from
        # this run's noise floor
        assert project.basis.shapes.num_shapes == 5
        assert [round(float(f), 1) for f in
                project.basis.shapes.frequency] == [439.2, 647.2, 823.5,
                                                    1142.3, 1142.4]


def test_a_saved_project_reloads_complete(builders, tmp_path):
    """What the app will actually open: the .vdyn round trip keeps the
    skeleton full, links and roles included."""
    from visualdynamics import load
    from visualdynamics.core.report import missing_expectations

    project = builders.shock_project()
    path = tmp_path / 'shock.vdyn'
    project.save(str(path))
    back = load(str(path))
    assert back.project_type == 'Shock'
    missing = [slot for slot, _c, _i, _n, _optional, _r
               in missing_expectations(back.project_type, back,
                                       placed=back.placed())]
    # every slot, optional ones included: the shock demo exists to
    # show the recommended workflow, and a workflow with gray slots in
    # it is not a demonstration of anything (Brandon, 2026-08-25)
    assert missing == [], f'unfilled: {missing}'


def test_the_shock_demo_follows_the_recommended_workflow(builders):
    """Load, filter, then compute everything from the *filtered*
    record — the SRS, the densities and the motion chain alike
    (Brandon, 2026-08-25). Held by provenance rather than by name, so
    renaming an object cannot make this pass wrongly."""
    from visualdynamics.core.data import Psd, Specification, Srs, TimeHistory

    project = builders.shock_project()
    provenance = project.provenance
    # the recording is the record nothing was derived to make —
    # `project.time_history` is ambiguous once the chain exists, which
    # is itself the workflow working
    raw = next(name for name, obj in project.items()
               if isinstance(obj, TimeHistory) and name not in provenance)

    filtered = next(name for name, record in provenance.items()
                    if record['verb'] == 'filter_data')
    assert provenance[filtered]['source'] == raw, \
        'the filter reads the recording'
    assert project[raw].filtering is not None, 'and its settings are kept'

    for verb, kind in (('compute_srs', Srs),
                       ('integrate', TimeHistory)):
        made = [name for name, record in provenance.items()
                if record['verb'] == verb]
        assert made, f'the workflow runs {verb}'
        assert any(provenance[name]['source'] == filtered for name in made), \
            f'{verb} reads the filtered record, not the raw one'
        assert all(isinstance(project[name], kind) for name in made)
    # no density in a shock project (Brandon, 2026-08-29): its level
    # would be set by how much quiet air the capture holds, and the
    # report's scalogram answers the frequency question instead
    assert not any(isinstance(project[name], Psd)
                   and not isinstance(project[name], Specification)
                   for name in project.names), \
        'the demo carries no PSD any more'

    # displacement comes from velocity, not from the filtered record
    velocity = next(name for name, record in provenance.items()
                    if record['verb'] == 'integrate'
                    and record['source'] == filtered)
    displacement = next(name for name, record in provenance.items()
                        if record['verb'] == 'integrate'
                        and record['source'] == velocity)
    assert set(project[velocity].ordinate_dim) == {'velocity'}
    assert set(project[displacement].ordinate_dim) == {'length'}

    # the events are marked on the recording and carried forward
    assert project[raw].shocks, 'the recording carries its own events'
    assert tuple(project[filtered].shocks) == tuple(project[raw].shocks)
