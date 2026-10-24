"""Partial, reference and virtual coherence of a multiple-input record.

Pinned against a two-input case worked by hand at one line, against the
identities that tie the readings together (the multiple coherence as a
product of an ordinary and a partial coherence, the virtual coherences
summing to the multiple), against the package's own multiple coherence
from the same frames, and against records whose answer is known: two
independent drives, two drives that are nearly the same, and one that
is exactly a multiple of the other.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.partial_coherence import (
    conditioned,
    multiple_coherence,
    read_partial_coherence,
)

RATE = 1024.0

# ---- the hand case: one line, two references, one response -----------------

G11, G22, G12 = 2.0, 1.0, 0.6 + 0.2j
G1Y, G2Y, GYY = 1.0 - 0.5j, 0.3 + 0.4j, 3.0


def _hand_matrix():
    return np.array([[G11, G12, G1Y],
                     [np.conj(G12), G22, G2Y],
                     [np.conj(G1Y), np.conj(G2Y), GYY]])[None]


def test_the_two_input_case_by_hand():
    """Worked out on paper, with G_ab = E[A* B]:

    γ²_12 = |0.6+0.2j|²/(2·1) = 0.4/2 = 0.2
    γ²_1y = |1−0.5j|²/(2·3) = 1.25/6
    G_2y·1 = G_2y − G_21 G_1y / G_11
           = (0.3+0.4j) − (0.6−0.2j)(1−0.5j)/2
           = (0.3+0.4j) − (0.5−0.5j)/2 = 0.05+0.65j
    G_22·1 = G_22 (1 − γ²_12) = 0.8
    G_yy·1 = G_yy − |G_1y|²/G_11 = 3 − 0.625 = 2.375
    γ²_2y·1 = |0.05+0.65j|² / (0.8 · 2.375) = 0.425 / 1.9
    """
    reading = read_partial_coherence(np.array([10.0]), _hand_matrix(),
                                     ['1Z+', '2Z+'], ['3Z+'])
    assert reading.averages == 0, 'none stated, none claimed'
    assert reading.reference[0, 1, 0] == pytest.approx(0.2)
    assert reading.ordinary[0, 0, 0] == pytest.approx(1.25 / 6.0)
    assert reading.partial[0, 1, 0] == pytest.approx(0.425 / 1.9)
    # and the identity 1 − γ²_y:x = (1 − γ²_1y)(1 − γ²_2y·1)
    assert 1.0 - reading.multiple[0, 0] == pytest.approx(
        (1.0 - 1.25 / 6.0) * (1.0 - 0.425 / 1.9))


def test_the_conditioned_matrix_is_the_schur_complement_by_hand():
    block = conditioned(_hand_matrix(), [1, 2], [0])[0]
    assert block[0, 0] == pytest.approx(0.8)
    assert block[1, 1] == pytest.approx(2.375)
    assert block[0, 1] == pytest.approx(0.05 + 0.65j)
    assert np.allclose(conditioned(_hand_matrix(), [0, 2], [])[0],
                       _hand_matrix()[0][np.ix_([0, 2], [0, 2])])


def test_the_virtual_coherences_sum_to_the_multiple():
    reading = read_partial_coherence(np.array([10.0]), _hand_matrix(),
                                     ['1Z+', '2Z+'], ['3Z+'])
    assert np.sum(reading.virtual[0, :, 0]) == pytest.approx(
        reading.multiple[0, 0])
    eigen = np.linalg.eigvalsh(_hand_matrix()[0, :2, :2])[::-1]
    assert np.allclose(reading.principal[:, 0], eigen)
    point = reading.at(10.0)
    assert point.principal[0] == 1.0
    assert point.principal[1] == pytest.approx(eigen[1] / eigen[0])


def test_the_shared_multiple_coherence_is_bounded_and_undefined_when_dead():
    grr = _hand_matrix()[:, :2, :2]
    gry = _hand_matrix()[:, :2, 2:]
    assert multiple_coherence(grr, gry, np.array([[0.0]]))[0, 0] != \
        multiple_coherence(grr, gry, np.array([[0.0]]))[0, 0], \
        'a response that measured nothing has no coherence'
    tiny = multiple_coherence(grr, gry, np.array([[1e-9]]))
    assert tiny[0, 0] == 1.0, 'held at one, not read past it'
    bad = np.array(gry)
    bad[0, 0, 0] = np.nan
    assert np.isnan(multiple_coherence(grr, bad, np.array([[GYY]]))[0, 0])


# ---- records whose answer is known ------------------------------------------


def _history(x2=None, noise=0.2, samples=1 << 15, seed=11):
    """Drives 1Z+ and 2Z+ (forces, so references by role); the response
    101Z+ is 1Z+ through a short filter, plus a third of 2Z+, plus
    noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / RATE
    x1 = rng.standard_normal(samples)
    x2 = rng.standard_normal(samples) if x2 is None else x2(x1, rng)
    y = (np.convolve(x1, [1.0, 0.5], 'same') + 0.3 * x2
         + noise * rng.standard_normal(samples))
    history = visualdynamics.TimeHistory(
        t, np.vstack([x1, x2, y]), response_dof=['1Z+', '2Z+', '101Z+'],
        ordinate_dim=['force', 'force', 'acceleration'])
    history.averaging = Averaging(frame_length=512, overlap=0.5,
                                  window='hann', frames=120)
    return history


def test_independent_drives_have_partial_near_ordinary():
    reading = _history().read_partial_coherence()
    assert reading.references == ['1Z+', '2Z+']
    assert reading.responses == ['101Z+']
    point = reading.at(150.0)
    assert point.frequency == pytest.approx(150.0, abs=1.0), \
        'the line nearest the frequency asked for'
    band = slice(5, -5)
    # independent drives: removing one leaves the other's share of what
    # is left, so the partial is a little over the ordinary, never under
    assert np.all(reading.partial[0, :, band] + 0.02
                  >= reading.ordinary[0, :, band])
    assert np.nanmedian(reading.reference[0, 1, band]) < 0.05
    assert reading.coupled() == []
    assert 'most explained by 1Z+' in reading.describe('101Z+')
    assert np.median(reading.effective_references()) == 2


def test_each_response_is_read_against_its_own_spectra():
    """A second response driven by 2Z+ alone: its partial coherence
    with 2Z+ is high and with 1Z+ nil, the reverse of 101Z+'s, and its
    virtual coherences sum to its own multiple coherence."""
    history = _history()
    x2 = np.asarray(history.ordinate[1])
    rng = np.random.default_rng(12)
    second = x2 + 0.2 * rng.standard_normal(len(x2))
    both = visualdynamics.TimeHistory(
        history.abscissa, np.vstack([history.ordinate, second]),
        response_dof=[*history.response_dof, '102Z+'],
        ordinate_dim=[*history.ordinate_dim, 'acceleration'])
    both.averaging = history.averaging
    reading = both.read_partial_coherence()
    assert reading.responses == ['101Z+', '102Z+']
    band = slice(5, -5)
    assert np.median(reading.partial[1, 1, band]) > 0.9
    assert np.median(reading.partial[1, 0, band]) < 0.05
    assert np.median(reading.partial[0, 0, band]) > 0.9
    assert np.allclose(np.nansum(reading.virtual, axis=1), reading.multiple,
                       atol=1e-9)
    # and each response's own 1 − γ²_y:x = (1 − γ²_1y)(1 − γ²_2y·1)
    assert np.allclose(1.0 - reading.multiple,
                       (1.0 - reading.ordinary[:, 0])
                       * (1.0 - reading.partial[:, 1]), atol=1e-9)
    assert 'most explained by 2Z+' in reading.describe('102Z+')


def test_the_pairs_are_ordered_by_how_much_of_the_band_they_couple():
    """Three drives at two lines, worked so that 1Z+ and 2Z+ pass 0.9
    at one line of two (largest 0.95) and 1Z+ and 3Z+ at both (largest
    0.91): ordered by the share of the lines, not by the largest. A
    third line where nothing was measured has no drive in effect."""
    def correlation(r12, r13, r23):
        return np.array([[1.0, r12, r13], [r12, 1.0, r23],
                         [r13, r23, 1.0]])

    lines = [correlation(np.sqrt(0.95), np.sqrt(0.91), 0.93),
             correlation(0.5, np.sqrt(0.91), 0.5), np.zeros((3, 3))]
    matrix = np.zeros((3, 4, 4), dtype=complex)
    for k, block in enumerate(lines):
        matrix[k, :3, :3] = block
        matrix[k, 3, 3] = 1.0 if k < 2 else 0.0
    reading = read_partial_coherence(np.array([10.0, 20.0, 30.0]), matrix,
                                     ['1Z+', '2Z+', '3Z+'], ['101Z+'])
    coupled = reading.coupled()
    assert [(one, other) for one, other, _s, _t in coupled] == \
        [('1Z+', '3Z+'), ('1Z+', '2Z+')]
    assert coupled[1][2] == pytest.approx(0.5)
    assert coupled[1][3] == pytest.approx(0.95)
    assert reading.effective_references()[2] == 0, 'nothing measured'


def test_the_multiple_coherence_is_the_packages_own():
    history = _history()
    reading = history.read_partial_coherence()
    theirs = history.compute_multiple_coherence()
    assert np.allclose(reading.multiple, theirs.ordinate, atol=1e-12)
    assert np.allclose(reading.frequency, theirs.abscissa)
    assert reading.averages == 120


def test_correlated_drives_show_in_the_reference_coherence():
    """2Z+ is mostly 1Z+: alone it looks coherent with the response,
    but with 1Z+ removed it explains little, and the pair is named."""
    history = _history(lambda x1, rng: x1 + 0.2 * rng.standard_normal(
        len(x1)))
    reading = history.read_partial_coherence()
    band = slice(5, -5)
    assert np.median(reading.ordinary[0, 1, band]) > 0.8
    assert np.median(reading.partial[0, 1, band]) < \
        np.median(reading.partial[0, 0, band])
    one, other, share, top = reading.coupled()[0]
    assert (one, other) == ('1Z+', '2Z+') and share > 0.9 and top > 0.9
    assert 'coherent over 0.9' in reading.describe()
    assert np.median(reading.effective_references(floor=0.1)) == 1, \
        'two drives, one of them in effect'


def test_a_drive_that_is_exactly_another_has_no_partial_coherence():
    # scaled up, so what the conditioning leaves is round-off of a large
    # number, and it is the floor relative to the autospectrum that
    # calls it nothing, not its smallness
    history = _history(lambda x1, rng: 2.0 * x1)
    big = visualdynamics.TimeHistory(
        history.abscissa, np.asarray(history.ordinate) * 1e4,
        response_dof=list(history.response_dof),
        ordinate_dim=list(history.ordinate_dim))
    big.averaging = history.averaging
    reading = big.read_partial_coherence()
    assert np.all(np.isnan(reading.partial[0])), \
        'neither drive has anything left once the other is removed'
    assert any('1Z+, 2Z+ are each explained completely' in note
               for note in reading.notes)
    # and the virtual coherence says the same: one principal reference
    assert np.all(np.isnan(reading.virtual[0, 1]))


def test_one_reference_reads_as_the_ordinary_coherence():
    reading = _history().read_partial_coherence(references=['1Z+'])
    assert reading.references == ['1Z+']
    assert np.allclose(reading.partial, reading.ordinary, equal_nan=True)
    assert np.allclose(reading.multiple, reading.ordinary[:, 0],
                       equal_nan=True)
    assert any('one reference' in note for note in reading.notes)
    assert reading.describe('101Z+').startswith('101Z+ against 1 reference,')


def test_a_dead_response_is_undefined_and_said():
    history = _history()
    values = np.array(history.ordinate)
    values[2] = 0.0
    dead = visualdynamics.TimeHistory(
        history.abscissa, values, response_dof=list(history.response_dof),
        ordinate_dim=list(history.ordinate_dim))
    dead.averaging = history.averaging
    reading = dead.read_partial_coherence()
    assert np.all(np.isnan(reading.partial[0]))
    assert np.all(np.isnan(reading.multiple[0]))
    assert any('101Z+ recorded nothing, so its coherence is undefined'
               in note for note in reading.notes)
    # the multiple coherence object keeps reading zero for it, as it did
    assert np.all(dead.compute_multiple_coherence().ordinate[0] == 0.0)


def test_a_line_that_is_not_a_number_and_two_dead_channels_are_said():
    matrix = np.repeat(_hand_matrix(), 3, axis=0)
    matrix[1, 0, 2] = np.nan
    reading = read_partial_coherence(np.array([1.0, 2.0, 3.0]), matrix,
                                     ['1Z+', '2Z+'], ['3Z+'])
    assert np.isnan(reading.partial[0, :, 1]).all()
    assert np.isnan(reading.multiple[0, 1])
    assert np.isfinite(reading.partial[0, :, 0]).all()
    assert any(note.startswith('1 line holding') for note in reading.notes)
    dead = np.array(_hand_matrix())
    dead[0, 1, :] = dead[0, :, 1] = 0.0
    dead[0, 2, :] = dead[0, :, 2] = 0.0
    reading = read_partial_coherence(np.array([1.0]), dead,
                                     ['1Z+', '2Z+'], ['3Z+'])
    assert any(note.startswith('2Z+, 3Z+ recorded nothing, so their '
                               'coherences are undefined')
               for note in reading.notes)


def test_a_sample_that_is_not_a_number_is_refused_by_name():
    history = _history()
    values = np.array(history.ordinate)
    values[1, 100] = np.nan
    broken = visualdynamics.TimeHistory(
        history.abscissa, values, response_dof=list(history.response_dof),
        ordinate_dim=list(history.ordinate_dim))
    broken.averaging = history.averaging
    with pytest.raises(ValueError, match=r'reference channel 2Z\+ holds '
                                         'samples that are not finite'):
        broken.read_partial_coherence()


def test_too_few_averages_and_no_reference_are_refused():
    history = _history()
    history.averaging = Averaging(frame_length=512, frames=2)
    with pytest.raises(ValueError, match='more averages than references'):
        history.read_partial_coherence()
    with pytest.raises(ValueError, match='at least one reference'):
        read_partial_coherence(np.array([1.0]), np.ones((1, 1, 1)), [],
                               ['101Z+'])
    with pytest.raises(ValueError, match=r'is \(1, 2, 2\), not \(1, 3, 3\)'):
        read_partial_coherence(np.array([1.0]), np.ones((1, 2, 2)),
                               ['1Z+', '2Z+'], ['101Z+'])


def test_a_response_is_found_by_name_or_position():
    reading = _history().read_partial_coherence()
    assert reading.index('101Z+') == 0 == reading.index(0)
    with pytest.raises(ValueError, match='not a response'):
        reading.index('7Z+')
    with pytest.raises(ValueError, match='no response 1'):
        reading.index(1)


# ---- through the project ----------------------------------------------------


def test_the_project_reads_behind_a_multiple_coherence_or_a_history():
    project = visualdynamics.Project()
    project.add('Run', _history())
    coherence = project.compute_multiple_coherence('Run')
    behind = project.read_partial_coherence(coherence)
    assert np.allclose(behind.multiple, project[coherence].ordinate,
                       atol=1e-12)
    direct = project.read_partial_coherence('Run')
    assert np.allclose(direct.partial, behind.partial, equal_nan=True)
    psds = project.compute_psds('Run')
    with pytest.raises(ValueError, match='neither a time history nor a '
                                         'multiple coherence'):
        project.read_partial_coherence(psds)


# ---- the edges and the default window, and what chance gives -------------------


def _drives_and_responses(seed, n=1 << 15, start=0.0, extra=None):
    """Two independent drives and two responses, every channel with its
    own noise as measured (the drives' as well as the responses'), under
    the default Hann window, from a record that starts at `start`."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, n))
    if extra is not None:
        x = x + extra
    y = np.vstack([x[0], 0.8 * x[1]])
    measured = np.vstack([x + 0.05 * rng.standard_normal((2, n)),
                          y + 0.1 * rng.standard_normal((2, n))])
    history = visualdynamics.TimeHistory(
        start + np.arange(n) / RATE, measured,
        response_dof=['1Z+', '2Z+', '101Z+', '102Z+'],
        ordinate_dim=['force', 'force', 'acceleration', 'acceleration'])
    history.averaging = Averaging(frame_length=512, overlap=0.5).filled(
        n, RATE)
    assert history.averaging.window == 'hann'
    return history


def test_the_edges_a_late_start_and_crossing_tones_are_read():
    """A tone on the first line in one drive and one at the edge of the
    span in the other, two tones that cross, noise on every channel, and
    a record that starts 9.5 s in: every line is defined, the multiple
    coherence of a response made of the first drive is near one at every
    line and the edges, and the start time changes nothing."""
    n = 1 << 15
    t = np.arange(n) / RATE
    df = RATE / 512
    tones = np.vstack([
        np.sin(2 * np.pi * df * t)
        + np.sin(2 * np.pi * (60.0 * t + 2.0 * t ** 2)),
        np.sin(2 * np.pi * (RATE / 2 - df) * t)
        + np.sin(2 * np.pi * (180.0 * t - 2.0 * t ** 2))])
    early = _drives_and_responses(41, n, extra=tones).read_partial_coherence()
    late = _drives_and_responses(41, n, start=9.5,
                                 extra=tones).read_partial_coherence()
    lines = early.frequency > 0.0
    assert np.isfinite(early.partial[:, :, lines]).all()
    assert early.multiple[0, lines].min() > 0.95
    first, last = np.flatnonzero(lines)[[0, -2]]
    assert early.multiple[0, first] > 0.95 and early.multiple[0, last] > 0.95
    np.testing.assert_allclose(late.partial, early.partial)
    assert early.describe(0).startswith('101Z+ against 2 references, most '
                                        'explained by 1Z+')


def test_independent_drives_are_never_said_to_be_coupled_by_chance():
    """The coupling rule, checked against what chance gives: twenty
    records of two independent white-noise drives under the default Hann
    window, 127 frames. The coherence estimate of independent drives
    sits near the inverse of the averages and passes 0.9 with a
    likelihood of 0.1 raised to the averages less one: no pair is named
    at any line, and the response made of the first drive alone is
    explained by it every time."""
    named = 0
    for seed in range(20):
        reading = _drives_and_responses(500 + seed).read_partial_coherence()
        named += len(reading.coupled())
        assert reading.describe(0).startswith(
            '101Z+ against 2 references, most explained by 1Z+')
        lines = reading.frequency > 0.0
        assert np.nanmedian(reading.reference[0, 1, lines]) < 0.03
    assert named == 0
