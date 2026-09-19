"""Multiple coherence computed from time data.

Ordinary coherence asks what one reference explains. Multiple coherence
asks what a whole set of them explains at once, which is the only useful
question in a MIMO test: two shakers driving one article are correlated
with each other, so a response can look poorly coherent with either
alone while being fully accounted for by the pair.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import MultipleCoherence, TimeHistory

RATE = 1024.0


def driven(samples=8192, noise=0.0, seed=0, drives=2):
    """Responses built from known drives, so the answer is known.

    Every response is a fixed mixture of the drives plus whatever noise
    is asked for. With no noise the drives account for all of it and the
    coherence is one; the noise is what pulls it down.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / RATE
    forces = rng.normal(0.0, 1.0, (drives, samples))
    rows, dofs, dims = [], [], []
    for k in range(drives):
        rows.append(forces[k])
        dofs.append(f'2{k}0Z+')
        dims.append('force')
    for r in range(3):
        mix = sum((1.0 + 0.3 * r * (k + 1)) * forces[k] for k in range(drives))
        rows.append(mix + rng.normal(0.0, noise, samples))
        dofs.append(f'10{r}Z+')
        dims.append('acceleration')
    history = TimeHistory(t, np.array(rows), response_dof=dofs,
                          ordinate_dim=dims)
    history.averaging = Averaging(frame_length=1024, overlap=0.5,
                                  window='hann', frames=15)
    return history


# ---- what it computes ---------------------------------------------------


def test_a_response_made_only_of_the_drives_is_fully_coherent():
    coherence = driven(noise=0.0).compute_multiple_coherence()
    assert isinstance(coherence, MultipleCoherence)
    assert coherence.num_records == 3, 'the responses, not the drives'
    band = coherence.ordinate[:, 5:-5]
    assert band.min() > 0.99


def test_noise_pulls_it_down():
    quiet = driven(noise=0.05).compute_multiple_coherence()
    loud = driven(noise=2.0).compute_multiple_coherence()
    assert quiet.ordinate[:, 5:-5].mean() > loud.ordinate[:, 5:-5].mean()
    assert loud.ordinate[:, 5:-5].mean() < 0.9


def test_it_never_leaves_nought_to_one():
    """A ratio of powers cannot exceed one, and the object's whole
    contract is that it is a bounded ratio."""
    coherence = driven(noise=1.0).compute_multiple_coherence()
    assert coherence.ordinate.min() >= 0.0
    assert coherence.ordinate.max() <= 1.0


def test_one_reference_is_the_ordinary_coherence():
    """The cheapest check that the algebra is right: with a single
    reference the matrix expression collapses to |Gxy|^2/(Gxx Gyy)."""
    history = driven(noise=0.5, drives=1)
    coherence = history.compute_multiple_coherence()
    cpsds = history.compute_cpsds()

    def cross(a, b):
        for i, (r, f) in enumerate(zip(cpsds.response_dof,
                                       cpsds.reference_dof)):
            if r == a and f == b:
                return cpsds.ordinate[i]
        raise AssertionError(f'no {a}/{b}')

    for row, dof in enumerate(coherence.response_dof):
        ordinary = (np.abs(cross('200Z+', dof)) ** 2
                    / (np.real(cross('200Z+', '200Z+'))
                       * np.real(cross(dof, dof))))
        assert np.allclose(coherence.ordinate[row], np.clip(ordinary, 0, 1),
                           atol=1e-9)


def test_it_is_computed_over_the_frames_a_psd_would_use():
    """The averaging view sets both, so the coherence and the PSD beside
    it describe one measurement rather than two."""
    history = driven(noise=0.5)
    history.averaging = Averaging(frame_length=512, overlap=0.5,
                                  window='hann', frames=8)
    coherence = history.compute_multiple_coherence()
    psds = history.compute_psds()
    assert len(coherence.abscissa) == len(psds.abscissa)
    assert np.allclose(coherence.abscissa, psds.abscissa)
    assert '8 averages' in coherence.comment[0]


def test_a_shorter_window_gives_a_different_answer():
    """Which is the point of it following the averaging: a coherence
    worked out over the whole file would describe a different
    measurement from the PSD beside it."""
    history = driven(noise=0.5)
    whole = history.compute_multiple_coherence()
    history.averaging = Averaging(frame_length=256, overlap=0.5,
                                  window='hann', frames=4)
    part = history.compute_multiple_coherence()
    assert len(part.abscissa) != len(whole.abscissa)


# ---- which channels are the references ----------------------------------


def test_the_drives_are_guessed_from_the_quantities():
    history = driven()
    assert history.drive_dofs() == ['200Z+', '210Z+']


def test_references_can_be_named_outright():
    history = driven()
    coherence = history.compute_multiple_coherence(references=['200Z+'])
    assert coherence.num_records == 4, 'the other drive is a response now'
    assert '210Z+' in coherence.response_dof


def test_a_history_with_no_drives_says_so():
    t = np.arange(2048) / RATE
    only = TimeHistory(t, np.ones((2, 2048)),
                       response_dof=['101Z+', '102Z+'],
                       ordinate_dim=['acceleration'] * 2)
    with pytest.raises(ValueError, match='needs reference channels'):
        only.compute_multiple_coherence()


def test_a_history_that_is_all_drives_says_so():
    t = np.arange(2048) / RATE
    only = TimeHistory(t, np.ones((2, 2048)),
                       response_dof=['200Z+', '210Z+'],
                       ordinate_dim=['force'] * 2)
    with pytest.raises(ValueError, match='nothing left to explain'):
        only.compute_multiple_coherence()


# ---- against the controller's own answer --------------------------------


def test_it_agrees_with_the_controller():
    """The strongest check available: Rattlesnake computed its own
    multiple coherence for the same run, in its own code, and this has
    to land on it.

    Over the controller's own frame count. A coherence estimate is
    biased upward at few averages, and the controller's number came
    from its ten-frame CPSD buffer: ten frames here land on it
    (0.687 against 0.684), the full run's twenty-eight settle lower
    (0.57), and the import's default is the full run since
    2026-09-19. Like against like, so the count is read off the file.
    """
    from dataclasses import replace

    import netCDF4

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    spectra = visualdynamics.import_file(fixture_path('plate',
                                            'random_spectra.nc4'))
    theirs = next(v for k, v in spectra.items() if 'coherence' in k)
    with netCDF4.Dataset(fixture_path('plate', 'random.nc4')) as ds:
        frames = int(ds.groups['Random'].frames_in_cpsd)
    history = loaded['time_data']
    mine = history.compute_multiple_coherence(
        averaging=replace(history.averaging, frames=frames))
    for dof in mine.response_dof:
        assert dof in theirs.response_dof
        a = mine.ordinate[list(mine.response_dof).index(dof)]
        b = theirs.ordinate[list(theirs.response_dof).index(dof)]
        band = slice(len(a) // 20, len(a) // 2)
        assert a[band].mean() == pytest.approx(b[band].mean(), abs=0.02)


def test_the_calculator_adds_it_beside_the_history(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = next(n for n, o in window.objects.items()
                if isinstance(o, TimeHistory))
    window.tree.setCurrentItem(window._item_for_object(name))
    window.compute_multiple_coherence()
    added = next(o for o in window.objects.values()
                 if isinstance(o, MultipleCoherence))
    assert added.num_records == 8
    assert 'references' in window.statusBar().currentMessage()


# ---- too few averages is not an estimate --------------------------------


@pytest.mark.parametrize('frames', [1, 2])
def test_no_more_averages_than_references_is_refused(frames):
    """Not a poor estimate — no estimate. Fitting a response to two
    references from two or fewer averages is an exact fit at every
    line, so the explained power equals the measured power and every
    value reads 1.0. A wall of ones looks like a perfectly coherent
    test and is a statement about the arithmetic instead.
    """
    history = driven(noise=1.0)
    history.averaging = Averaging(frame_length=1024, overlap=0.5,
                                  frames=frames)
    with pytest.raises(ValueError, match='more averages than references'):
        history.compute_multiple_coherence()


def test_one_more_average_than_references_is_allowed_through():
    """The floor is where the fit stops being exact, not somewhere
    chosen to feel safe."""
    history = driven(noise=1.0)
    history.averaging = Averaging(frame_length=1024, overlap=0.5, frames=3)
    coherence = history.compute_multiple_coherence()
    assert coherence.ordinate[:, 5:-5].max() <= 1.0


def test_the_estimate_settles_as_averages_are_added():
    """Which is the whole reason the floor exists: the bias runs as the
    reference count over the average count, so few averages read high
    whatever the article is doing."""
    history = driven(noise=2.0, samples=32768)
    means = []
    for frames in (4, 8, 30):
        history.averaging = Averaging(frame_length=1024, overlap=0.5,
                                      frames=frames)
        means.append(history.compute_multiple_coherence(
        ).ordinate[:, 5:-5].mean())
    assert means[0] > means[1] > means[2], 'biased high on few averages'


# ---- the calculator sets the averaging rather than defaulting to one ----


def test_a_history_with_no_averaging_gets_one_worked_out(window, pump):
    """A single frame fits every reference exactly and reads 1.0 at
    every line, which is what a coherence over one long window looked
    like. Nothing defaults to that."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = next(n for n, o in window.objects.items()
                if isinstance(o, TimeHistory))
    history = window.objects[name]
    history.averaging = None
    window.tree.setCurrentItem(window._item_for_object(name))
    window.compute_multiple_coherence()
    assert history.averaging is not None, 'and stored, not used and dropped'
    assert history.averaging.frames > 2
    coherence = next(o for o in window.objects.values()
                     if isinstance(o, MultipleCoherence))
    # the failure mode guarded against reads ~1.0 at *every* line; a
    # genuine two-drive measurement legitimately touches 0.9999 at a
    # resonance, so the middle of the distribution is what discriminates
    assert np.median(coherence.ordinate[:, 5:-5]) < 0.99, (
        'not a wall of ones')
    assert 'detected' in window.statusBar().currentMessage()


def test_the_averaging_it_sets_is_the_one_a_psd_then_uses(window, pump):
    """Storing it is the point. A coherence that quietly averaged
    differently from the PSD beside it would describe a different
    measurement, and following the averaging is exactly so that it
    does not."""
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = next(n for n, o in window.objects.items()
                if isinstance(o, TimeHistory))
    history = window.objects[name]
    history.averaging = None
    window.tree.setCurrentItem(window._item_for_object(name))
    window.compute_multiple_coherence()
    coherence = next(o for o in window.objects.values()
                     if isinstance(o, MultipleCoherence))
    assert np.allclose(history.compute_psds().abscissa, coherence.abscissa)


# ---- a DOF is not a channel ---------------------------------------------


def drive_point(samples=8192, noise=0.3, seed=1):
    """The same thing, but with an accelerometer *on* the drive point.

    Which is how a modal test is actually instrumented: the shaker's
    force cell and a response accelerometer sit at the same DOF, and the
    file carries both under the one name. `channel_key` has always said
    a DOF is not a channel; the reference selection did not listen.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / RATE
    force = rng.normal(0.0, 1.0, samples)
    # accelerations first and the forces last, which is the order a
    # controller writes its channels in — and the order that made the
    # bug bite, since the first channel at '210Z+' is then the
    # accelerometer
    rows = [2.0 * force + rng.normal(0.0, noise, samples),   # 210Z+ accel
            0.7 * force + rng.normal(0.0, noise, samples),   # 101Z+ accel
            force]                                           # 210Z+ force
    history = TimeHistory(
        t, np.array(rows), response_dof=['210Z+', '101Z+', '210Z+'],
        ordinate_dim=['acceleration', 'acceleration', 'force'])
    history.averaging = Averaging(frame_length=1024, overlap=0.5,
                                  window='hann', frames=15)
    return history


def test_the_reference_at_a_drive_point_is_the_force():
    """Not the accelerometer that shares its DOF.

    Resolved by DOF alone the first channel at '210Z+' wins, and on a
    real file that is the accelerometer — so every reference was a
    response, the drive-point accelerations were struck from the
    responses, and the whole coherence was of the wrong quantity. On a
    343-channel modal survey it read 0.85 where the controller's own
    coherence read 0.94.
    """
    history = drive_point()
    coherence = history.compute_multiple_coherence()
    left = list(coherence.response_dof)
    assert left == ['210Z+', '101Z+'], (
        'the force is the reference; the accelerometer at the same DOF '
        f'is still a response, but got {left}')


def test_naming_the_dof_still_names_the_force():
    """An explicit `references=['210Z+']` means the drive there, which
    is the same channel the guess would have picked."""
    history = drive_point()
    guessed = history.compute_multiple_coherence()
    named = history.compute_multiple_coherence(references=['210Z+'])
    assert list(named.response_dof) == list(guessed.response_dof)
    assert np.allclose(named.ordinate, guessed.ordinate)


def test_a_dof_with_no_drive_on_it_is_still_a_reference():
    """A caller who names a response as the reference gets that
    response. It is an odd thing to ask for and a perfectly clear one."""
    history = drive_point()
    coherence = history.compute_multiple_coherence(references=['101Z+'])
    assert list(coherence.response_dof) == ['210Z+', '210Z+'], (
        'the named channel is out and everything else stays')
    assert coherence.num_records == 2


def test_it_matches_the_controllers_own_coherence(tmp_path):
    """The whole point, on the file the mistake was found in.

    A Rattlesnake modal run carries the multiple coherence the
    controller computed. Ours is worked out from the time data it saved
    beside it, so the two are independent routes to one number — and
    they agree to the last bit, which also says the frame length, the
    count, the window and the frame alignment all came in right.
    """
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'modal_spectra.nc4'))
    theirs = project.coherence
    ours = project.time_history.compute_multiple_coherence()
    assert list(ours.response_dof) == list(theirs.response_dof)
    assert np.allclose(ours.abscissa, theirs.abscissa)
    assert np.abs(ours.ordinate - theirs.ordinate).max() < 1e-9
