"""Frequency response functions computed from time data.

Three estimators, differing in one assumption — where the noise is —
and agreeing wherever there is little of it. H1 puts it on the response,
H2 on the reference, Hv on both. They part company exactly where a
measurement is worst, which is why the choice matters.

Two kinds of check here. The algebraic ones hold at every line whatever
the data is (`H1 = gamma^2 H2`, and `|H1| <= |Hv| <= |H2|`), so a wrong
formula cannot slip past them. And each estimator is given data built to
satisfy its own assumption and asked to be the best of the three, which
is the claim each one actually makes.

Then the one that matters most: a Rattlesnake modal run carries the FRFs
the controller computed *and* the time data it computed them from, so
the two are independent routes to one number.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import Frf, TimeHistory

RATE = 1024.0


def shaken(samples=16384, noise=0.02, seed=0, drives=2, gains=None):
    """Responses built from the drives through a known gain, so the
    answer is a number this file wrote down.

    Flat gains rather than a resonance: what is under test here is the
    estimator, and a constant is the one FRF whose right answer needs no
    second opinion.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / RATE
    forces = rng.normal(0.0, 1.0, (drives, samples))
    gains = gains if gains is not None else [
        [1.0 + 0.5 * r + 2.0 * k for k in range(drives)] for r in range(3)]
    rows, dofs, dims = [], [], []
    for r, row in enumerate(gains):
        rows.append(sum(row[k] * forces[k] for k in range(drives))
                    + rng.normal(0.0, noise, samples))
        dofs.append(f'10{r}Z+')
        dims.append('acceleration')
    for k in range(drives):
        rows.append(forces[k])
        dofs.append(f'2{k}0Z+')
        dims.append('force')
    history = TimeHistory(t, np.array(rows), response_dof=dofs,
                          ordinate_dim=dims)
    history.averaging = Averaging(frame_length=1024, overlap=0.5,
                                  window='hann', frames=30)
    return history, np.asarray(gains)


def _pairs(frfs):
    return {(str(r), str(f)): i for i, (r, f)
            in enumerate(zip(frfs.response_dof, frfs.reference_dof))}


# ---- what it computes ---------------------------------------------------


def test_it_recovers_the_gains_it_was_built_from():
    """Three responses, two drives, six FRFs, and every one of them the
    constant it was made with."""
    history, gains = shaken(noise=0.0)
    frfs = history.compute_frfs()
    assert isinstance(frfs, Frf)
    assert frfs.num_records == 6
    where = _pairs(frfs)
    band = slice(5, -5)
    for r in range(3):
        for k in range(2):
            got = frfs.ordinate[where[(f'10{r}Z+', f'2{k}0Z+')]][band]
            assert np.abs(got - gains[r, k]).max() < 1e-9, (r, k)


def test_the_drives_are_not_among_the_responses():
    history, _gains = shaken()
    frfs = history.compute_frfs()
    assert set(map(str, frfs.reference_dof)) == {'200Z+', '210Z+'}
    assert set(map(str, frfs.response_dof)) == {'100Z+', '101Z+', '102Z+'}


def test_it_reads_response_over_reference():
    history, _gains = shaken()
    frfs = history.compute_frfs()
    assert set(frfs.ordinate_dim) == {'acceleration/force'}


def test_correlated_drives_are_told_apart():
    """The reason the matrix inverse is there.

    Two shakers driving one article are correlated, and dividing each
    response by each drive on its own credits both with the same motion.
    Here the second drive is the first plus a little: read separately
    the answers are wrong by a factor, and the MIMO estimate gets them
    both right.
    """
    rng = np.random.default_rng(3)
    samples = 16384
    t = np.arange(samples) / RATE
    first = rng.normal(0.0, 1.0, samples)
    second = first + 0.3 * rng.normal(0.0, 1.0, samples)
    response = 2.0 * first + 5.0 * second
    history = TimeHistory(
        t, np.array([response, first, second]),
        response_dof=['101Z+', '200Z+', '210Z+'],
        ordinate_dim=['acceleration', 'force', 'force'])
    history.averaging = Averaging(frame_length=1024, overlap=0.5,
                                  window='hann', frames=30)
    frfs = history.compute_frfs()
    where = _pairs(frfs)
    band = slice(5, -5)
    got = [frfs.ordinate[where[('101Z+', dof)]][band]
           for dof in ('200Z+', '210Z+')]
    assert np.abs(got[0] - 2.0).max() < 1e-8
    assert np.abs(got[1] - 5.0).max() < 1e-8
    # what dividing by one drive alone would have said, for contrast
    naive = np.mean(np.abs(response) ** 2) / np.mean(np.abs(first) ** 2)
    assert abs(naive - 2.0) > 1.0, 'the single-drive reading really is wrong'


def test_one_reference_is_the_textbook_ratio():
    """With a single drive the matrix expression collapses to Gfx/Gff,
    which is the cheapest check that the algebra is right."""
    history, _gains = shaken(noise=0.3, drives=1)
    frfs = history.compute_frfs(method='H1')
    _f, scale, groups = history._spectral_frame(None)
    keys = list(groups)
    spectra = {k: np.fft.rfft(groups[k], axis=1) for k in keys}
    drive = next(k for k in keys if k[1] == 'force')

    def cross(a, b):
        return np.mean(np.conj(spectra[a]) * spectra[b], axis=0) * scale

    where = _pairs(frfs)
    for key in keys:
        if key[1] != 'acceleration':
            continue
        wanted = cross(drive, key) / cross(drive, drive)
        got = frfs.ordinate[where[(key[0], drive[0])]]
        assert np.allclose(got, wanted)


def test_noise_on_the_response_costs_the_coherence_not_the_frf():
    """H1's whole assumption, and the reason a shaker test uses it.

    Asked for by name: the default is Hv, which makes no such claim.

    Noise on the response is uncorrelated with the drives, so it
    averages out of the cross spectrum. It does not average out of the
    response's *own* power, which is where the coherence reads it. So
    the same noise that takes seven points off the coherence moves the
    estimate by under two percent.

    Not *no* percent: with a finite number of averages H1 carries a
    small bias of order (1-gamma^2)/(gamma^2 n), and at 0.93 over 30
    averages that is about the one and a half percent measured here. It
    is asymptotically unbiased, which is a different claim, and this
    test does not pretend otherwise — at noise 3.0 the coherence falls
    to 0.63 and the bias grows to 5%, exactly as that expression says.
    """
    _clean, gains = shaken(noise=0.0)
    noisy, _ = shaken(noise=1.0)
    band = slice(5, -5)
    frfs = noisy.compute_frfs(method='H1')
    where = _pairs(frfs)
    for r in range(3):
        for k in range(2):
            got = frfs.ordinate[where[(f'10{r}Z+', f'2{k}0Z+')]][band]
            error = abs(got.mean() - gains[r, k]) / gains[r, k]
            assert error < 0.02, f'off by {error:.1%} at {r},{k}'
    coherence = noisy.compute_multiple_coherence().ordinate[:, band]
    assert 0.9 < coherence.mean() < 0.95, (
        'the noise shows up here, where it is meant to')


# ---- what it refuses ----------------------------------------------------


def test_it_needs_more_averages_than_references():
    """Fewer averages than drives makes the reference matrix singular:
    the fit is exact and the FRF is whichever of infinitely many answers
    the pseudo-inverse picks."""
    history, _gains = shaken(samples=2048)
    history.averaging = Averaging(frame_length=1024, overlap=0.0,
                                  window='hann', frames=2)
    with pytest.raises(ValueError, match='more averages than references'):
        history.compute_frfs()


def test_a_history_with_no_drive_says_so():
    history, _gains = shaken()
    responses = TimeHistory(
        history.abscissa, history.ordinate[:3],
        response_dof=list(history.response_dof[:3]),
        ordinate_dim=['acceleration'] * 3)
    responses.averaging = history.averaging
    with pytest.raises(ValueError, match='needs reference channels'):
        responses.compute_frfs()


# ---- and the controller's own answer ------------------------------------


def test_it_matches_the_controllers_own_frfs():
    """The check that means anything.

    A Rattlesnake modal run carries the FRFs the controller estimated
    and the time data it estimated them from. Computing ours from the
    second and holding it against the first tests the estimator, the
    reference selection, the framing, the window and the scaling all at
    once — and there is nothing in that chain the file does not pin.

    H1 by name, because the file says `frf_technique = 'H1'` and the
    comparison is only worth anything against the estimator it used.
    """
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'modal_spectra.nc4'))
    theirs = project.frf
    ours = project.time_history.compute_frfs(method='H1')
    assert ours.num_records == theirs.num_records
    assert np.allclose(ours.abscissa, theirs.abscissa)
    where = _pairs(ours)
    for k, (r, f) in enumerate(zip(map(str, theirs.response_dof),
                                   map(str, theirs.reference_dof))):
        mine = ours.ordinate[where[(r, f)]]
        yours = theirs.ordinate[k]
        relative = np.abs(mine - yours).max() / np.abs(yours).max()
        assert relative < 1e-10, f'{r}/{f} differs by {relative:.2e}'


def test_the_coherence_beside_it_is_that_frfs_coherence():
    """Both readings come off one `_cross_spectral_frame`, so they
    cannot disagree about which channel is a reference or how the record
    was framed."""
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'modal_spectra.nc4'))
    history = project.time_history
    frfs = history.compute_frfs(method='H1')
    coherence = history.compute_multiple_coherence()
    assert np.allclose(frfs.abscissa, coherence.abscissa)
    assert set(map(str, coherence.response_dof)) == set(
        map(str, frfs.response_dof))


# ---- the verb and the button --------------------------------------------


def test_the_project_verb_links_them():
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'modal_spectra.nc4'))
    name = project.compute_frfs(project.time_history)
    assert isinstance(project[name], Frf)
    assert name in (project.group_of('Time History') or [])
    assert name.endswith('Hv FRFs'), 'the estimator is part of the name'


def test_two_estimators_are_two_objects():
    """They differ in nothing a reader can see but the numbers, so the
    name has to carry which is which."""
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'modal_spectra.nc4'))
    first = project.compute_frfs(project.time_history, 'Hv')
    second = project.compute_frfs(project.time_history, 'H1')
    assert first != second
    assert not np.allclose(project[first].ordinate, project[second].ordinate)


def _answer(monkeypatch, chosen):
    """Stand in for the estimator dialog, and check what it offered."""
    from PySide6.QtWidgets import QInputDialog

    seen = {}

    def getItem(_parent, title, label, items, current, editable):
        seen.update(title=title, label=label, items=list(items),
                    current=current, editable=editable)
        if chosen is None:
            return '', False
        return next(i for i in items if i.startswith(chosen)), True

    monkeypatch.setattr(QInputDialog, 'getItem', staticmethod(getItem))
    return seen


def test_the_calculator_offers_it(window, pump, monkeypatch):
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    pump()
    name = next(n for n, obj in window.objects.items()
                if isinstance(obj, TimeHistory))
    labels = [label for _verb, label, *_rest in window.acts_for([name])]
    assert not any('FRF' in label for label in labels), \
        'the verb lives on the averaging panel now (Brandon, 2026-08-28)'
    seen = _answer(monkeypatch, 'Hv')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.compute_frfs()
    assert seen['items'][0].startswith('Hv'), 'Hv leads, and is the default'
    assert not seen['editable'], 'three estimators, not a free-text box'
    assert seen['current'] == 0
    made = [obj for key, obj in window.objects.items()
            if isinstance(obj, Frf) and key != 'FRF']
    assert len(made) == 1
    assert 'Hv FRFs' in window.statusBar().currentMessage()


def test_the_estimator_it_was_told_is_the_one_it_used(window, pump,
                                                      monkeypatch):
    """And the name goes on the object: two FRF sets from one history
    differ in nothing else a reader can see."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    pump()
    name = next(n for n, obj in window.objects.items()
                if isinstance(obj, TimeHistory))
    _answer(monkeypatch, 'H1')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.compute_frfs()
    made = next(key for key in window.objects if key.endswith('H1 FRFs'))
    assert 'H1' in window.objects[made].comment[0]
    assert 'H1 FRFs' in window.statusBar().currentMessage()


def test_canceling_the_dialog_computes_nothing(window, pump, monkeypatch):
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    pump()
    name = next(n for n, obj in window.objects.items()
                if isinstance(obj, TimeHistory))
    before = set(window.objects)
    _answer(monkeypatch, None)
    window.tree.setCurrentItem(window._item_for_object(name))
    window.compute_frfs()
    assert set(window.objects) == before


def test_the_button_needs_a_time_history(window, pump, monkeypatch):
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    pump()
    _answer(monkeypatch, 'Hv')
    window.tree.setCurrentItem(window._item_for_object('FRF'))
    window.compute_frfs()
    assert 'Select a time history' in window.statusBar().currentMessage()


# ---- the three estimators -----------------------------------------------


def noisy_siso(gain=3.0, on_reference=0.0, on_response=0.0, seed=1,
               samples=131072):
    """One drive, one response, a flat gain, and noise placed exactly
    where the caller says — which is the only way to ask an estimator
    whether its own assumption is the one it acts on."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / RATE
    force = rng.normal(0.0, 1.0, samples)
    history = TimeHistory(
        t,
        np.array([gain * force + rng.normal(0.0, on_response, samples),
                  force + rng.normal(0.0, on_reference, samples)]),
        response_dof=['101Z+', '210Z+'],
        ordinate_dim=['acceleration', 'force'])
    history.averaging = Averaging(frame_length=1024, overlap=0.5,
                                  window='hann', frames=200)
    return history


def _levels(history, band=slice(20, -20)):
    return {method: np.abs(
        history.compute_frfs(method=method).ordinate[0][band]).mean()
        for method in TimeHistory.FRF_METHODS}


def test_hv_is_the_default():
    """Because the claim H1 and H2 each make — that one instrument is
    the trustworthy one — is the claim a test least often gets to make
    honestly."""
    history = noisy_siso(on_reference=0.3, on_response=0.9)
    assert np.allclose(history.compute_frfs().ordinate,
                       history.compute_frfs(method='Hv').ordinate)
    assert TimeHistory.FRF_METHODS[0] == 'Hv'


def test_h1_is_the_coherence_times_h2():
    """The identity that ties the two one-sided estimators together, and
    it holds line by line whatever the data is: H1 = gamma^2 H2."""
    history = noisy_siso(on_reference=0.4, on_response=1.0)
    band = slice(20, -20)
    h1 = history.compute_frfs(method='H1').ordinate[0][band]
    h2 = history.compute_frfs(method='H2').ordinate[0][band]
    gamma = history.compute_multiple_coherence().ordinate[0][band]
    assert np.allclose(h1, gamma * h2, rtol=1e-9)


def test_hv_falls_between_the_other_two():
    """At every line, not on average. Hv is the total-least-squares fit
    and the two one-sided fits bracket it — a value outside that bracket
    would mean the eigenvector picked was not the smallest one."""
    history = noisy_siso(on_reference=0.4, on_response=1.0)
    band = slice(20, -20)
    got = {m: np.abs(history.compute_frfs(method=m).ordinate[0][band])
           for m in ('H1', 'H2', 'Hv')}
    assert (got['H1'] <= got['Hv'] + 1e-12).all()
    assert (got['Hv'] <= got['H2'] + 1e-12).all()
    assert (got['H1'] < got['H2']).mean() > 0.99, 'and strictly, with noise'


@pytest.mark.parametrize('placed,best', [
    ({'on_response': 1.2}, 'H1'),
    ({'on_reference': 0.4}, 'H2'),
    ({'on_reference': 0.4, 'on_response': 0.4}, 'Hv'),
])
def test_each_estimator_wins_where_its_assumption_holds(placed, best):
    """The whole reason there are three.

    Hv's turn is the case where the noise is comparable *in absolute
    terms* on both channels, which is what a total-least-squares fit
    assumes — it weighs a unit of error on the force against a unit of
    error on the acceleration, and those are not the same thing. So the
    gain here is 1.0: the honest demonstration of Hv is the one where
    its assumption is actually met, not one where it is flattered.
    """
    gain = 1.0 if best == 'Hv' else 3.0
    levels = _levels(noisy_siso(gain=gain, **placed))
    errors = {m: abs(v - gain) / gain for m, v in levels.items()}
    assert min(errors, key=errors.get) == best, errors
    assert errors[best] < 0.01, errors


def test_h2_refuses_a_non_square_multi_reference_set():
    """Three responses against two references: the coupled square form
    (which this package computes since 2026-08-28, matching sdynpy)
    needs as many responses as references, and one response against R
    references alone is one equation in R unknowns. Hv answers the
    same question for any shape."""
    history, _gains = shaken()
    with pytest.raises(ValueError, match='needs exactly 2 responses'):
        history.compute_frfs(method='H2')
    assert history.compute_frfs(method='Hv').num_records == 6


def test_h2_computes_the_coupled_form_for_a_square_set():
    """As many responses as references: the classical coupled
    Gxx * Gfx^-1 (Rocklin, Crowley and Vold, 1985), adopted to match
    sdynpy (Brandon, 2026-08-28) — test_sdynpy_oracle pins it to
    sdynpy's own numbers at 3e-15. Here the ground truth: on nearly
    noise-free data with known flat gains, the coupled estimate lands
    on the gains this file wrote down."""
    gains = [[1.0, 3.0], [1.5, 4.0]]
    history, _ = shaken(noise=1e-8, gains=gains)
    rows = [i for i, d in enumerate(history.response_dof)
            if str(d) in ('100Z+', '101Z+', '200Z+', '210Z+')]
    square = TimeHistory(
        history.abscissa, np.asarray(history.ordinate)[rows],
        response_dof=[str(history.response_dof[i]) for i in rows],
        ordinate_dim=[history.ordinate_dim[i] for i in rows])
    square.averaging = history.averaging

    frfs = square.compute_frfs(method='H2')
    assert frfs.num_records == 4
    where = _pairs(frfs)
    band = slice(20, 400)          # away from DC, where nothing drives
    for r, response in enumerate(('100Z+', '101Z+')):
        for k, reference in enumerate(('200Z+', '210Z+')):
            estimate = frfs.ordinate[where[(response, reference)]][band]
            assert np.allclose(estimate, gains[r][k], rtol=1e-3), (
                response, reference)


def test_a_name_that_is_not_an_estimator_says_which_are():
    history, _gains = shaken()
    with pytest.raises(ValueError, match='Hv, H1, H2'):
        history.compute_frfs(method='H3')


def test_they_all_agree_when_there_is_no_noise():
    """The estimators differ only in what they do about noise, so with
    none they are one estimator with three names."""
    history = noisy_siso()
    levels = _levels(history)
    assert max(levels.values()) - min(levels.values()) < 1e-9
    assert abs(levels['Hv'] - 3.0) < 1e-9
