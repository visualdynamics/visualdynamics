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


def test_detection_snaps_to_the_three_decibel_ladder():
    """Runs are commanded at +3, 0, -3, -6 … (Brandon, 2026-09-19,
    from whole decibels): the detected offset lands on that ladder. A
    level whose bands center within `DETECTION_RUNG_DB` of a rung reads
    as that rung; one between rungs is a controller that held the wrong
    level, not a commanded step, and reads zero (2026-09-26 — it used
    to snap to the nearest rung, dividing out part of the control
    error)."""
    from visualdynamics.core.compliance import DETECTION_RUNG_DB, SCALE_STEP_DB

    assert (SCALE_STEP_DB, DETECTION_RUNG_DB) == (3, 0.75)
    spec = _spec()
    for offset, snapped in ((6.4, 6), (5.6, 6), (8.5, 9), (-3.5, -3),
                            (7.4, 0), (7.6, 0), (4.2, 0), (1.6, 0),
                            (-4.6, 0)):
        assert detect_scale_db(spec, _measured(spec, [offset])) == snapped, offset


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
    # read from the band levels (2026-09-26): monitors under their
    # envelopes spread the bands, and spread bands are not a level, so
    # unnamed the answer is zero — shown unscaled — never the monitors'
    # rung; with the controls named, only they are counted
    assert detect_scale_db(spec, psds) == 0
    assert detect_scale_db(spec, psds, controls=['101Z+', '102Z+']) == 6
    # even with the controls the majority, a few monitors far under
    # their envelopes spread the bands past the limit
    majority = _measured(spec, [6.0] * 7 + [12.0, 16.0, 22.0], wiggle=1.0)
    assert detect_scale_db(spec, majority) == 0
    assert detect_scale_db(spec, majority, controls=channels[:7]) == 6


def test_one_broken_channel_cannot_answer_alone():
    """Smaller than the controls but alone: a single channel over its
    limit does not drag the detected level to a hotter rung. Pooled
    with two controls and two monitors it is no one controlled test,
    so zero; with the controls named, their level."""
    channels = tuple(f'{n}Z+' for n in range(101, 106))
    spec = _spec(channels)
    offsets = [-4.0, 6.0, 6.0, 9.0, 14.0]      # one channel 4 dB over
    psds = _measured(spec, offsets, wiggle=1.0)
    assert detect_scale_db(spec, psds) == 0
    assert detect_scale_db(spec, psds, controls=['102Z+', '103Z+']) == 6


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
    """A channel at +6.5 dB below 300 Hz and +6.2 above. It was built to
    make the narrowband and octave gridings detect different levels (5
    and 6 on the whole-decibel rule, until 2026-09-19; 9 and 6 on the
    nearest rung, until 2026-09-26) — the disagreement one resolution
    exists to kill. Detection now reads sixth-octave bands whatever it
    is given, so the two cannot disagree; the fixture sits on the 6
    rung and the test pins that the report's bands use the one level."""
    from visualdynamics.core.compliance import log_interpolate

    f = np.linspace(20.0, 2000.0, 400)
    base = log_interpolate(f, spec.abscissa, spec.ordinate[0])
    offset = np.where(f < 300.0, 6.5, 6.2)
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
    # bands compare only with the same bands (2026-09-19): the banded
    # pair is the specification banded too, and a curve against bands
    # is not compared at all
    assert detect_scale_db(spec, psds.to_octave(6)) == 0, 'refused, not detected'
    # detection reads sixth-octave bands whatever it is given
    # (2026-09-26), so the narrowband pair and the banded pair are read
    # from the same bands and cannot disagree
    on_bands = detect_scale_db(spec.to_octave(6), psds.to_octave(6))
    assert on_bands == 6, 'the banded pair reads the same bands'
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


def test_a_real_run_up_with_its_monitors_reads_its_level():
    """A -6 run, three controls on the commanded level and three
    monitors under their envelopes: half the pool on the rung is not a
    majority, so unnamed it reads zero; named, the controls' -6. (This
    was the floor veto's test until detection pooled, 2026-09-26.)"""
    dofs = ('101Z+', '113Z+', '1301Z+', '1313Z+', '404Z+', '410Z+')
    spec = _spec(dofs)
    psds = _measured(spec, [6.0, 6.0, 6.0, 8.0, 11.0, 15.0])
    assert detect_scale_db(spec, psds) == 0
    assert detect_scale_db(spec, psds, controls=dofs[:3]) == 6


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

    # but a commanded step is still read, and is not swallowed — here
    # two controls among six monitors, so named
    stepped = _measured(spec, [3.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 14.0],
                        wiggle=0.5)
    assert detect_scale_db(spec, stepped, controls=channels[:2]) == 3
    assert detect_scale_db(
        spec, _measured(spec, [6.0] * 2 + [9.0, 12.0, 14.0, 18.0, 20.0,
                                           25.0], wiggle=1.0),
        controls=channels[:2]) == 6


# ---- the plausible-level and explains-the-data gates (2026-09-26) -------
#
# Scoring a 22-run control campaign, three runs were rescaled as if run
# at reduced level: two controllers that blew up (~50 dB hot on every
# channel) charted at -54 and -51 dB — an overtest read as an undertest
# — and an under-controlled run charted at +9. Each gate can only turn
# a detected level into zero.

EIGHT = tuple(f'{n}Z+' for n in range(101, 109))


def test_a_hot_failure_is_shown_not_scaled():
    """Every channel ~52 dB over its specification and agreeing within
    two: no channel is plausible, so the floor cannot veto, and two
    agreeing is easy. No controller commands +52 dB."""
    spec = _spec(EIGHT)
    hot = _measured(spec, [-52.0, -51.0, -52.0, -53.0, -51.0, -52.0,
                           -52.0, -51.0], wiggle=0.3)
    assert detect_scale_db(spec, hot) == 0


def _peaks_held(spec, channels):
    """Over specification at five resonant peaks, 9 dB low between
    them: the line median says +9, on its rung, and the RMS barely
    moves — a control failure, which moves the two differently, where
    a level moves both. On the rung on purpose, so only the energy
    disagreement can refuse it."""
    from visualdynamics.core.compliance import log_interpolate

    f = np.linspace(20.0, 2000.0, 400)
    base = log_interpolate(f, spec.abscissa, spec.ordinate[0])
    peaks = np.zeros(f.size, dtype=bool)
    for center in (150.0, 300.0, 450.0, 900.0, 1500.0):
        peaks |= np.abs(f - center) < 0.05 * center
    rng = np.random.default_rng(0)
    rows = [base * np.where(peaks, 10.0 ** 0.6, 10.0 ** -0.9)
            * 10.0 ** (0.02 * rng.standard_normal(f.size))
            for _ in channels]
    return Psd(f, np.array(rows), response_dof=list(channels),
               ordinate_dim='acceleration**2/frequency',
               ordinate_unit='m/s**2')


def test_broad_under_drive_with_the_peaks_held_is_not_a_level():
    channels = EIGHT[:4]
    spec = _spec(channels)
    measured = _peaks_held(spec, channels)
    rows = compare_all(spec, measured, scale_db=0)
    assert all(abs(r['difference_db']) < 3.0 for _label, r in rows), (
        'as measured, every channel is within the RMS tolerance')
    from visualdynamics.core.compliance import judge

    level = judge(spec, measured, 0, 0, None, True, scale_db=0)
    median = float(np.median(10.0 * np.log10(level['asked'] / level['held'])))
    assert abs(median - 9.0) < 0.75, 'the median alone would read +9'
    assert detect_scale_db(spec, measured) == 0


def test_a_genuine_reduced_level_run_is_still_detected():
    """The rule must not be too strict: runs commanded at -3 through
    -24 dB read their level — every channel a control, or the controls
    named among as many monitors — and -27 is past the window."""
    spec = _spec(EIGHT)
    for level in (3.0, 6.0, 12.0, 24.0):
        run = _measured(spec, [level] * 8, wiggle=0.5)
        assert detect_scale_db(spec, run) == level, level
        half = _measured(spec, [level] * 4 + [level + 3.0, level + 5.0,
                                              level + 8.0, level + 11.0],
                         wiggle=0.5)
        assert detect_scale_db(spec, half, controls=EIGHT[:4]) == level, level
    assert detect_scale_db(spec, _measured(spec, [27.0] * 8, wiggle=0.5)) == 0


@pytest.mark.parametrize(('offset', 'expected'), [
    (-9.0, 0),      # 9 dB hot: past the -6 edge, a fault
    (-6.0, -6),     # a run at +6 dB, the hottest believed
    (24.0, 24),     # a run-up from -24 dB
    (27.0, 0),      # past the +24 edge
])
def test_the_window_of_commanded_levels(offset, expected):
    spec = _spec(EIGHT)
    assert detect_scale_db(spec, _measured(spec, [offset] * 8,
                                           wiggle=0.3)) == expected


def test_a_detected_scale_is_said_aloud_to_a_script():
    """`channel_errors` drops the scale, so a script charting with it
    could not see one: `compare_all` warns when it detected a level
    itself, and not when the scale was given or held."""
    import warnings

    from visualdynamics.core.compliance import ScaleWarning

    spec = _spec(EIGHT)
    run = _measured(spec, [6.0] * 8, wiggle=0.3)
    with pytest.warns(ScaleWarning, match=r'scaled by \+6 dB, detected '
                                          r'from the data as a run at -6 dB'):
        compare_all(spec, run)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ScaleWarning)
        compare_all(spec, run, scale_db=0)
        compare_all(spec, run, scale_db=6)
        run.scale_db = 6
        compare_all(spec, run)
        run.scale_db = None
        compare_all(spec, _measured(spec, [0.0] * 8, wiggle=0.3))


# ---- the beat-zero gate (2026-09-26, the note's second pass) -----------

TWENTY_FOUR = tuple(f'{n}Z+' for n in range(101, 125))


def test_a_hot_minority_does_not_outvote_an_on_level_majority():
    """21 controls on specification and 3 running 1.6 dB hot: their
    medians snap to 0 and -3, the smallest agreed is -3, and every
    on-spec channel "explains" a one-step shift within the 3 dB RMS
    tolerance. A real level moves the majority to the candidate; this
    candidate holds 3 channels within half a step against zero's 21."""
    spec = _spec(TWENTY_FOUR)
    run = _measured(spec, [0.0] * 21 + [-1.6] * 3, wiggle=0.3)
    assert detect_scale_db(spec, run) == 0


def test_a_mixed_under_drive_is_not_a_level():
    """Half the channels on specification, half 3 dB low: no level
    could put both halves there. A uniform 3 dB under-drive, on the
    other hand, is indistinguishable from a run at -3 dB and reads as
    one — the limit no gate can remove, and why a detected scale is
    always said."""
    spec = _spec(TWENTY_FOUR)
    assert detect_scale_db(spec, _measured(spec, [0.0] * 12 + [3.0] * 12,
                                           wiggle=0.3)) == 0
    assert detect_scale_db(spec, _measured(spec, [3.0] * 24,
                                           wiggle=0.3)) == 3


# ---- one level for the whole test (2026-09-26, the note's third pass) ---


def test_a_hot_minority_run_at_minus_six_reads_minus_six():
    """The per-channel vote read this +3 — a wrong level applied. Pooled,
    the 21 controls on their level set the median, and the three hot
    ones cannot move it."""
    spec = _spec(TWENTY_FOUR)
    run = _measured(spec, [6.0] * 21 + [4.4] * 3, wiggle=0.3)
    assert detect_scale_db(spec, run) == 6


def test_monitors_in_the_majority_read_zero_never_their_rung():
    """A specification bounding more monitors than controls, nothing
    naming the controls: the pooled median is the monitors'. Pooling
    everything, as first proposed, read a -12 dB run as +15 and a -6
    as +9; the majority test reads zero, and the Scaling field says the
    rest."""
    spec = _spec(TWENTY_FOUR)
    for run, controls in (
            ([6.0] * 8 + [9.0, 11.0, 14.0, 8.5, 12.0, 10.0] * 2
             + [7.5, 13.0, 9.5, 15.0], 6),
            ([12.0] * 8 + [15.0] * 10 + [18.0] * 6, 12)):
        measured = _measured(spec, run, wiggle=0.3)
        assert detect_scale_db(spec, measured) == 0
        assert detect_scale_db(spec, measured,
                               controls=TWENTY_FOUR[:8]) == controls


# ---- the mode of the band levels (2026-09-26, Brandon's rule) -----------


def test_the_level_is_read_in_sixth_octave_bands():
    """Narrowband lines are noisy, weighted to the top decade and the
    anti-resonances; detection bands both to sixth octaves first, so a
    narrowband pair and its banded copy read the same level."""
    spec = _spec(EIGHT)
    run = _measured(spec, [6.0] * 8, wiggle=1.0)
    assert detect_scale_db(spec, run) == 6
    assert detect_scale_db(spec.to_octave(6), run.to_octave(6)) == 6


def test_tight_bands_between_rungs_are_not_a_level():
    """A full-level test the controller held 1.6 dB low: every band
    within half a step of +3 and a tiny spread — only the bands'
    median, 1.4 dB off the rung, says it is not a commanded -3."""
    spec = _spec(EIGHT)
    assert detect_scale_db(spec, _measured(spec, [1.6] * 8, wiggle=0.3)) == 0


def test_bands_that_split_between_two_rungs_are_not_a_level():
    """Five channels 3 dB low and three on level at full level: the mode
    is +3, the spread small, the median on the rung — but only 5 in 8
    bands agree with it."""
    spec = _spec(EIGHT)
    run = _measured(spec, [0.0] * 3 + [3.0] * 5, wiggle=0.3)
    assert detect_scale_db(spec, run) == 0


def test_the_top_decade_cannot_outvote_the_band():
    """A full-level run whose controller lost 3 dB above 400 Hz — where
    authority runs out first. Lines are evenly spaced in hertz, so four
    in five lie above 400 Hz and a narrowband reading has them agree,
    tightly, on a run at -3 dB; sixth-octave bands are evenly spaced as
    the specification is written, a third of them lie up there, and the
    level they agree on is full level."""
    from visualdynamics.core.compliance import log_interpolate

    spec = _spec(EIGHT)
    f = np.linspace(20.0, 2000.0, 400)
    base = log_interpolate(f, spec.abscissa, spec.ordinate[0])
    rng = np.random.default_rng(1)
    rows = [base * np.where(f > 400.0, 10.0 ** -0.3, 1.0)
            * 10.0 ** (0.02 * rng.standard_normal(f.size)) for _ in EIGHT]
    run = Psd(f, np.array(rows), response_dof=list(EIGHT),
              ordinate_dim='acceleration**2/frequency',
              ordinate_unit='m/s**2')
    high = float(np.mean(f > 400.0))
    assert high > 0.75, 'most narrowband lines are in the top decade'
    assert detect_scale_db(spec, run) == 0


def test_a_few_deep_bands_do_not_hide_a_level():
    """Real control has tails (the other session's 22 runs, 2026-09-26):
    runs held on level with 89% of their bands within 1.5 dB had σ of
    4.3 dB from a few deep bands of under-driven out-of-plane channels,
    and a σ test showed them unscaled. The share of agreeing bands reads
    the bulk, and the level stands."""
    from visualdynamics.core.compliance import judge, log_interpolate

    spec = _spec(EIGHT)
    f = np.linspace(20.0, 2000.0, 400)
    base = log_interpolate(f, spec.abscissa, spec.ordinate[0])
    rng = np.random.default_rng(2)
    rows = []
    for k, _dof in enumerate(EIGHT):
        # a run at -6 dB; two out-of-plane channels 15 dB under above
        # 1.2 kHz, where the shakers could not reach them
        deep = (k >= 6) & (f > 1200.0)
        rows.append(base * 10.0 ** (-(6.0 + 15.0 * deep) / 10.0)
                    * 10.0 ** (0.03 * rng.standard_normal(f.size)))
    run = Psd(f, np.array(rows), response_dof=list(EIGHT),
              ordinate_dim='acceleration**2/frequency',
              ordinate_unit='m/s**2')
    s, m = spec.to_octave(6), run.to_octave(6)
    d = np.concatenate([10.0 * np.log10(judge(s, m, i, i, None, True,
                                              scale_db=0)['asked']
                                        / judge(s, m, i, i, None, True,
                                                scale_db=0)['held'])
                        for i in range(8)])
    assert np.std(d) > 2.0, 'a σ test at 2 dB would refuse it'
    assert detect_scale_db(spec, run) == 6


def test_two_groups_straddling_a_rung_are_not_one_level():
    """Half the channels 1.4 dB over the -6 rung and half 1.4 under:
    every band is within half a step of +6, but the bulk is spread
    across a whole step — two levels, not one — and the median lands on
    one of the two groups, off the rung. (The robust spread tried here,
    2026-09-26, could only have caught this, and the rung test does.)"""
    spec = _spec(EIGHT)
    run = _measured(spec, [4.6] * 4 + [7.4] * 4, wiggle=0.05)
    assert detect_scale_db(spec, run) == 0
