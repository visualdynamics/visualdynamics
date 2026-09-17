"""Scaling a measurement onto its 0 dB specification for comparison.

Standard practice: a run captured at -6 dB is compared by scaling it up
to the requirement. The scale is detected in whole decibels (or held by
hand on the object), the data itself is never touched, and every reader
of the comparison — the drawn curves, the error metrics, the report —
resolves the same number through `comparison_scale_db`.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.compliance import (
    compare_all,
    comparison_scale_db,
    detect_scale_db,
    exceedances,
)
from visualdynamics.core.data import Psd, Specification

pytestmark = pytest.mark.usefixtures('flat_reading')

def _spec(channels=('101Z+',)):
    """A flat-ish power-law specification, one record per channel."""
    f = np.array([20.0, 100.0, 600.0, 2000.0])
    return Specification(
        f, np.tile([0.01, 0.04, 0.04, 0.01], (len(channels), 1)),
        response_dof=list(channels),
        ordinate_dim='acceleration**2/frequency',
        ordinate_unit='m/s**2',
        abort_upper=np.tile([0.02, 0.08, 0.08, 0.02], (len(channels), 1)),
        abort_lower=np.tile([0.005, 0.02, 0.02, 0.005],
                            (len(channels), 1)))


def _measured(spec, offsets_db, wiggle=0.0, seed=0):
    """PSDs on a fine grid: each channel its spec curve offset in dB,
    optionally with a repeatable ripple so no two lines agree."""
    from visualdynamics.core.compliance import log_interpolate

    f = np.linspace(20.0, 2000.0, 400)
    rng = np.random.default_rng(seed)
    rows = []
    for offset in offsets_db:
        base = log_interpolate(f, spec.abscissa, spec.ordinate[0])
        ripple = 10.0 ** (wiggle * rng.standard_normal(f.size) / 10.0)
        rows.append(base * 10.0 ** (-offset / 10.0) * ripple)
    return Psd(f, np.array(rows),
               response_dof=list(spec.response_dof[:len(offsets_db)]),
               ordinate_dim='acceleration**2/frequency',
               ordinate_unit='m/s**2')


def test_a_quiet_run_detects_its_commanded_level():
    spec = _spec()
    psds = _measured(spec, [6.0])
    assert detect_scale_db(spec, psds) == 6


def test_detection_rounds_to_whole_decibels():
    """Runs are commanded in whole dB; a genuine half-dB level error is
    the comparison's business, not the scaling's."""
    spec = _spec()
    assert detect_scale_db(spec, _measured(spec, [6.4])) == 6
    assert detect_scale_db(spec, _measured(spec, [5.6])) == 6


def test_monitors_cannot_outvote_the_controls():
    """A specification bounds monitors as well as controls, each monitor
    its own distance under its envelope. The commanded level is the
    smallest offset two channels agree on — the controls, sitting on
    their specification — not the median of wherever the monitors pile
    up. This is the shape of a real 36-channel run that detected +8
    for a section commanded at -6."""
    channels = tuple(f'{n}Z+' for n in range(101, 111))
    spec = _spec(channels)
    # two controls dead on the commanded -6; eight monitors scattered
    # from 7 to 25 dB under their envelopes, some agreeing with each
    # other well above the controls
    offsets = [6.0, 6.0, 8.0, 8.0, 8.0, 12.0, 16.0, 20.0, 22.0, 25.0]
    psds = _measured(spec, offsets, wiggle=1.0)
    assert detect_scale_db(spec, psds) == 6


def test_one_broken_channel_cannot_answer_alone():
    """Smaller than the controls but alone: a single channel over its
    limit does not drag the detected level down."""
    channels = tuple(f'{n}Z+' for n in range(101, 106))
    spec = _spec(channels)
    offsets = [-4.0, 6.0, 6.0, 9.0, 14.0]      # one channel 4 dB over
    psds = _measured(spec, offsets, wiggle=1.0)
    assert detect_scale_db(spec, psds) == 6


def test_the_held_scale_outranks_detection():
    spec = _spec()
    psds = _measured(spec, [6.0])
    assert comparison_scale_db(spec, psds) == 6
    psds.scale_db = 0
    assert comparison_scale_db(spec, psds) == 0, '0 is "do not scale"'
    psds.scale_db = 4
    assert comparison_scale_db(spec, psds) == 4
    psds.scale_db = None
    assert comparison_scale_db(spec, psds) == 6, 'blank returns to auto'


def test_the_error_metrics_judge_the_scaled_data():
    spec = _spec()
    psds = _measured(spec, [6.0])
    rows = compare_all(spec, psds)
    assert rows and rows[0][1]['lines'] > 0
    assert rows[0][1]['scale_db'] == 6.0, 'the table can say what it did'
    assert abs(rows[0][1]['difference_db']) < 0.1, (
        'scaled up, the -6 dB run sits on its specification')
    psds.scale_db = 0
    unscaled = compare_all(spec, psds)
    assert unscaled[0][1]['difference_db'] == pytest.approx(-6.0, abs=0.1)


def test_exceedances_judge_the_scaled_data():
    """A -6 dB run sits below abort_lower unscaled and inside scaled —
    the marks must agree with the metrics about which data was read."""
    spec = _spec()
    psds = _measured(spec, [6.0])
    _over, under = exceedances(spec, psds)
    assert not under.any(), 'scaled onto the spec, nothing is under'
    psds.scale_db = 0
    _over, under = exceedances(spec, psds)
    assert under.any(), 'unscaled, the quiet run is below abort_lower'


def test_the_drawn_comparison_scales_a_copy_and_says_so():
    from visualdynamics.plot import bounded_by_specification

    spec = _spec()
    psds = _measured(spec, [6.0])
    before = psds.ordinate.copy()
    series, _dropped = bounded_by_specification(
        [('PSDs', psds, None), ('Spec', spec, None)])
    names = {name for name, _d, _r in series}
    assert 'PSDs (+6 dB)' in names, 'a scaled curve says so in its legend'
    drawn = next(d for name, d, _r in series if name == 'PSDs (+6 dB)')
    assert np.allclose(drawn.ordinate, before * 10.0 ** 0.6)
    assert np.array_equal(psds.ordinate, before), 'the object is untouched'
    assert psds.scale_db is None, 'and still detects for itself'


def test_a_matching_run_is_drawn_as_it_is():
    from visualdynamics.plot import bounded_by_specification

    spec = _spec()
    psds = _measured(spec, [0.0])
    series, _dropped = bounded_by_specification(
        [('PSDs', psds, None), ('Spec', spec, None)])
    assert {name for name, _d, _r in series} == {'PSDs', 'Spec'}, (
        'nothing to scale, nothing renamed')


def test_a_held_scale_rides_the_project_file(tmp_path):
    import visualdynamics

    spec = _spec()
    psds = _measured(spec, [6.0])
    psds.scale_db = 3
    project = visualdynamics.Project('Scaled')
    project.add('Spec', spec)
    project.add('PSDs', psds)
    project.save(tmp_path / 'scaled.vdyn')
    back = visualdynamics.Project.open(tmp_path / 'scaled.vdyn')
    assert back['PSDs'].scale_db == 3, 'a held scale is a judgment'
    assert back['Spec'].scale_db is None
    psds.scale_db = None
    project.save(tmp_path / 'auto.vdyn')
    again = visualdynamics.Project.open(tmp_path / 'auto.vdyn')
    assert again['PSDs'].scale_db is None, 'auto stays auto, re-detected'


# ---- the Scaling control ----------------------------------------------------

@pytest.fixture
def comparing(window, pump):
    spec = _spec()
    psds = _measured(spec, [6.0])
    window.add_object('Spec', spec)
    window.add_object('PSDs', psds)
    window.tree.setCurrentItem(window._item_for_object('Spec'))
    window.tree.clearSelection()
    for name in ('Spec', 'PSDs'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    return window


def test_the_bar_shows_the_detected_scaling(comparing):
    window = comparing
    assert window.data_pane.scaling_action.isVisible()
    assert not window.data_pane.toolbar.isHidden(), 'the bar it lives in'
    assert window.data_pane.scaling_edit.text() == '+6 dB'
    plot = next(item for item in window.data_pane.graphics.ci.items
                if hasattr(item, 'listDataItems'))
    legend = [item.name() for item in plot.listDataItems() if item.name()]
    assert any('(+6 dB)' in name for name in legend), (
        'the drawn curve says it was scaled')


def test_editing_the_scaling_holds_it(comparing, pump):
    window = comparing
    psds = window.objects['PSDs']
    window._comparison_scale_edited('0')
    pump()
    assert psds.scale_db == 0, 'zero is "do not scale", held'
    assert window.data_pane.scaling_edit.text() == '0 dB'
    window._comparison_scale_edited('+4 dB')
    pump()
    assert psds.scale_db == 4, 'units may be typed along with the number'
    window._comparison_scale_edited('')
    pump()
    assert psds.scale_db is None, 'blank returns to automatic'
    assert 'detected +6 dB' in window.statusBar().currentMessage()
    assert window.data_pane.scaling_edit.text() == '+6 dB'


def test_a_refused_scaling_changes_nothing(comparing, pump):
    window = comparing
    psds = window.objects['PSDs']
    window._comparison_scale_edited('loud')
    pump()
    assert psds.scale_db is None
    assert 'whole decibels' in window.statusBar().currentMessage()
    assert window.data_pane.scaling_edit.text() == '+6 dB', (
        'the redraw restates what still holds')


def test_a_lone_psd_offers_no_scaling(window, pump):
    """No specification, no comparison, no control — not grayed, absent."""
    spec = _spec()
    window.add_object('PSDs', _measured(spec, [6.0]))
    window.tree.setCurrentItem(window._item_for_object('PSDs'))
    window.render_current()
    pump()
    assert not window.data_pane.scaling_action.isVisible()


# ---- the report -------------------------------------------------------------

def test_the_report_notes_the_scaling_prominently():
    """A reader who misses the scaling reads the figure as a claim
    about the raw run — so the caption says it, and so does the
    measured curve's own label."""
    import visualdynamics
    from visualdynamics.report import _bars_block, _comparison_block

    spec = _spec()
    psds = _measured(spec, [6.0])
    figure = _comparison_block({}, psds, spec, visualdynamics.SI, 'Control')
    assert 'measured data scaled +6 dB to the specification' in (
        figure['caption'])
    assert any(curve['label'] == 'measured (+6 dB)'
               for curve in figure['curves'])

    bars = _bars_block({'mode': 'error', 'caption': 'Levels'},
                       spec, psds, visualdynamics.SI)
    assert 'measured data scaled +6 dB' in bars['caption']
    assert max(abs(v) for v in bars['values']) < 0.2, (
        'the bars judge the scaled data')


def test_an_unscaled_comparison_claims_no_scaling():
    import visualdynamics
    from visualdynamics.report import _bars_block, _comparison_block

    spec = _spec()
    psds = _measured(spec, [0.0])
    figure = _comparison_block({}, psds, spec, visualdynamics.SI, 'Control')
    assert 'scaled' not in figure['caption']
    assert any(curve['label'] == 'measured' for curve in figure['curves'])
    bars = _bars_block({'mode': 'error', 'caption': 'Levels'},
                       spec, psds, visualdynamics.SI)
    assert 'scaled' not in bars['caption']


def test_banding_carries_the_held_scale():
    """The report bands the narrowband PSD itself, so a judgment held
    on it has to ride into the octave comparison."""
    spec = _spec()
    psds = _measured(spec, [6.0])
    psds.scale_db = 4
    assert psds.to_octave(6).scale_db == 4
    psds.scale_db = None
    assert psds.to_octave(6).scale_db is None


# ---- one measurement, one scale ---------------------------------------------

def _tilted(spec):
    """A channel at +5 dB below 300 Hz and +6 above: the narrowband
    median (linear grid, most lines high) rounds to 6, the octave
    median (log bands, most bands low) rounds to 5 — the rounding
    disagreement between gridings that one-resolution exists to kill."""
    from visualdynamics.core.compliance import log_interpolate

    f = np.linspace(20.0, 2000.0, 400)
    base = log_interpolate(f, spec.abscissa, spec.ordinate[0])
    offset = np.where(f < 300.0, 5.0, 6.0)
    return Psd(f, np.array([base * 10.0 ** (-offset / 10.0)]),
               response_dof=['101Z+'],
               ordinate_dim='acceleration**2/frequency',
               ordinate_unit='m/s**2')


def test_the_report_resolves_one_scale_for_every_grid():
    """The report bands the narrowband PSD itself for its octave
    figures. Detected again on the banded copy the offset rounded to a
    different decibel, and one report claimed two scalings for one
    run — seen live as +3 on the PSD figure and +2 on the octave bars."""
    import visualdynamics
    from visualdynamics.core.compliance import detect_scale_db
    from visualdynamics.report import _bars_block

    spec = _spec()
    psds = _tilted(spec)
    assert detect_scale_db(spec, psds) == 6
    assert detect_scale_db(spec, psds.to_octave(6)) == 5, (
        'the fixture must make the two gridings disagree')
    bars = _bars_block({'mode': 'error', 'octave': 6, 'caption': 'Bands'},
                       spec, psds, visualdynamics.SI)
    assert 'scaled +6 dB' in bars['caption'], (
        'the octave blocks use the scale resolved on the narrowband')


def _family(window, pump):
    spec = _spec()
    psds = _measured(spec, [6.0])
    window.add_object('Spec', spec)
    window.add_object('PSDs', psds)
    octave = window.project.compute_octave('PSDs', 6)
    window.show_object(octave)
    pump()
    return octave


def _select(window, pump, *names):
    window.tree.setCurrentItem(window._item_for_object(names[0]))
    window.tree.clearSelection()
    for name in names:
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()


def test_the_octave_view_reads_the_familys_scale(window, pump):
    """A held value anywhere in the family governs everywhere: the
    octave PSD is the same measurement on another grid."""
    octave = _family(window, pump)
    window.objects['PSDs'].scale_db = 4
    _select(window, pump, 'Spec', octave)
    assert window.data_pane.scaling_edit.text() == '+4 dB'


def test_editing_either_member_holds_the_family(window, pump):
    """The user's report: octave scaling edited to match the PSDs, and
    the report still showed the old number — because the report reads
    the narrowband, and the edit never reached it. Editing either
    object now sets both."""
    octave = _family(window, pump)
    _select(window, pump, 'Spec', octave)
    window._comparison_scale_edited('2')
    pump()
    assert window.objects[octave].scale_db == 2
    assert window.objects['PSDs'].scale_db == 2, (
        'the narrowband — the object the report reads — holds it too')
    window._comparison_scale_edited('')
    pump()
    assert window.objects[octave].scale_db is None
    assert window.objects['PSDs'].scale_db is None


def test_a_channel_on_spec_vetoes_a_coincidental_agreement():
    """The plate run's shape: the drive point dead on the
    specification, the rest scattered low because two shakers cannot
    hold eight channels — two of which happen to agree. A commanded
    level is a floor, and the on-spec channel sits under the
    candidate, so this is a badly controlled full-level run and not a
    well-controlled -4 dB one."""
    spec = _spec(('101Z+', '113Z+', '1301Z+', '1313Z+',
                  '404Z+', '410Z+'))
    psds = _measured(spec, [0.0, 3.0, 4.0, 4.0, 9.0, 14.0])
    assert detect_scale_db(spec, psds) == 0


def test_the_plate_demonstration_run_detects_full_level():
    """The real file that exposed the coincidence, kept as the pin."""
    from conftest import fixture_path

    import visualdynamics

    run = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spec = run['Random_specification']
    psds = run['time_data'].compute_psds()
    assert detect_scale_db(spec, psds) == 0


def test_the_floor_veto_spares_a_real_run_up():
    """A -6 run with monitors above the commanded level everywhere:
    nothing sits below the floor, so the detection stands."""
    spec = _spec(('101Z+', '113Z+', '1301Z+', '1313Z+',
                  '404Z+', '410Z+'))
    psds = _measured(spec, [6.0, 6.0, 6.0, 8.0, 11.0, 15.0])
    assert detect_scale_db(spec, psds) == 6


def test_a_force_record_stored_first_cannot_steal_the_pairing():
    """matched_records paired by DOF name alone once, and only came out
    right because acceleration records happened to be stored first. A
    drive point's force PSD ahead of its acceleration in the file must
    not be the record the specification is compared against."""
    from visualdynamics.core.compliance import matched_records

    spec = _spec(('101Z+',))
    f = np.linspace(20.0, 2000.0, 40)
    both = Psd(f, np.ones((2, f.size)),
               response_dof=['101Z+', '101Z+'],
               ordinate_dim=['force**2/frequency',
                             'acceleration**2/frequency'],
               ordinate_unit=['N', 'm/s**2'])
    rows = matched_records(spec, both)
    assert rows == [('101Z+', 0, 1)], 'the acceleration record, stored second'


# ---- what the detector is allowed to be fooled by -------------------------


def test_lines_the_specification_barely_occupies_cannot_set_the_level():
    """Brandon found the drone transient detecting -26 dB where it
    should read 0 (2026-08-25). Its specification is the PSD of a
    target waveform that stops around 2 kHz, so above that the
    requirement sits at 5e-15 against a measurement at its noise
    floor — a 40 dB difference saying nothing about level, over 51%
    of the frequency lines. The median went with the majority.

    Built here at the same shape, so the mechanism is pinned rather
    than the one file: content to 2 kHz, near-nothing above it, and a
    measurement on level throughout.
    """
    import numpy as np

    from visualdynamics.core.compliance import log_interpolate

    channels = tuple(f'{n}Z+' for n in range(101, 105))
    spec_f = np.array([20.0, 100.0, 1000.0, 2000.0, 2100.0, 8000.0])
    # flat to 2 kHz, then twelve decades down: a waveform's spectrum
    # ending, not a requirement of quiet
    level = np.array([0.04, 0.04, 0.04, 0.04, 4e-14, 4e-15])
    spec = Specification(
        spec_f, np.tile(level, (len(channels), 1)),
        response_dof=list(channels),
        ordinate_dim='acceleration**2/frequency', ordinate_unit='m/s**2')
    # the measurement runs to 8 kHz — most of its lines above the band
    # the specification occupies — and is dead on level inside it
    f = np.linspace(20.0, 8000.0, 4000)
    rows = []
    for _ in channels:
        row = log_interpolate(f, spec.abscissa, spec.ordinate[0])
        rows.append(np.where(f <= 2000.0, row, 5e-11))   # noise floor
    psds = Psd(f, np.array(rows), response_dof=list(channels),
               ordinate_dim='acceleration**2/frequency',
               ordinate_unit='m/s**2')
    assert (f > 2000.0).mean() > 0.5, 'the fixture has the same majority'
    assert detect_scale_db(spec, psds) == 0


def test_a_run_that_merely_held_imperfectly_is_not_a_scaled_run():
    """A level nobody commanded is not a level. Runs step in 3 dB and
    6; a channel spread of a decibel or two around the specification
    is a test that held imperfectly, and dividing it out would erase
    the control error the comparison exists to show (Brandon,
    2026-08-25 — the drone transient read -1 dB once its band was
    right, on two channels happening to share that value)."""
    channels = tuple(f'{n}Z+' for n in range(101, 109))
    spec = _spec(channels)
    psds = _measured(spec, [-1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                     wiggle=0.5)
    assert detect_scale_db(spec, psds) == 0

    # but a commanded step is still read, and is not swallowed
    assert detect_scale_db(
        spec, _measured(spec, [3.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 14.0],
                        wiggle=0.5)) == 3
    assert detect_scale_db(
        spec, _measured(spec, [6.0] * 2 + [9.0, 12.0, 14.0, 18.0, 20.0,
                                           25.0], wiggle=1.0)) == 6
