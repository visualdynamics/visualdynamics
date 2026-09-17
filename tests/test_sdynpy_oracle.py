"""Visual Dynamics' spectral answers, checked against sdynpy's — as data.

sdynpy is the tool this package's audience already trusts, and this
package will be scrutinized precisely because it was built with AI
assistance, so agreement is worth demonstrating (Brandon, 2026-08-28).
What must never happen is an import: sdynpy is GPL-3.0, and one import
from anything distributed with Visual Dynamics would settle its
licensing permanently. So the comparison arrives the way the ADF
corpus's truth files do — sdynpy runs privately in the generators
repository, and its *numbers* are frozen into
`testdata/sdynpy_oracle/oracle.npz` beside the exact input signals.
Running a GPL program privately carries no conditions, and a program's
numerical output is not a covered work; nothing here imports sdynpy,
and `test_license_boundary` now checks the tests as well as the
package.

**An agreement oracle, not a truth oracle.** sdynpy has around 13% test
coverage, so when the two disagree the discrepancy is investigated from
first principles — Parseval, closed forms, the scipy cross-checks
already in this suite — and the verdict is recorded, whichever way it
falls. Nothing here may be "fixed" toward sdynpy on disagreement alone.

What building the oracle actually found, for the record:

* **Agreement at machine precision** everywhere the conventions
  coincide: PSD 4e-16, FRF H1 2e-15, Hv 1e-14, H2 5e-16, coherence
  1e-15, SRS 9e-15, RMS 6e-16 (worst relative errors, 2026-08-28). The
  framings provably match — same frames, same periodic windows, same
  Welch scaling — so these are two implementations of the same
  definitions, not merely similar answers.
* **CPSD is conjugated between the tools.** Visual Dynamics computes
  Bendat & Piersol's `Gxy = conj(X)·Y`; sdynpy computes `X·conj(Y)`.
  Both are self-consistent conventions with opposite phase sign; the
  comparison conjugates and says so, and neither side is wrong.
* **H2 was the one deliberate divergence, and Brandon closed it.**
  sdynpy computes the classical coupled square form, Gxx * Gfx^-1
  (Rocklin, Crowley and Vold, 1985); this package refused it for a
  day, because the coupling makes a channel's H2 depend on which
  sibling channels are in the set — measured at 0.2% on these very
  signals, where H1 is bit-identical under the same swap. Brandon
  chose compatibility with the tool his audience trusts (2026-08-28),
  so the square form is computed, matched below at 4e-15, and the
  coupling is documented in compute_frfs instead of avoided.
  Single-reference H2 keeps the uncoupled per-response reading both
  tools share, and non-square multi-reference stays refused by both.
* **Multiple coherence is checked against sdynpy's own computation**,
  which lives inline in its SignalProcessingGUI rather than in a
  callable function; the generator drives that GUI off-screen rather
  than copying the block out. Agreement at 1e-15, both references at
  once.
* And the first "discrepancy" was the harness: sdynpy's `srs` returns
  a `(values, frequencies)` tuple, the first oracle draft stored the
  tuple whole, and the comparison read the frequency grid as an SRS
  100% wrong. Worth remembering before blaming either tool.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import TimeHistory

#: observed worst-case relative error is 1e-14; a hundred times that
#: still fails on any real convention difference, whose scale is
#: percent, not parts per trillion
TOLERANCE = 1e-12


@pytest.fixture(scope='module')
def oracle():
    return np.load(fixture_path('sdynpy_oracle', 'oracle.npz'))


@pytest.fixture(scope='module')
def history(oracle):
    """The oracle's own input signals, as a Visual Dynamics record."""
    fs = float(oracle['sample_rate'])
    samples = oracle['signals'].shape[1]
    names = [str(name) for name in oracle['signal_names']]
    kinds = [str(kind) for kind in oracle['signal_kinds']]
    record = TimeHistory(np.arange(samples) / fs, oracle['signals'],
                         response_dof=names, ordinate_dim=kinds)
    record.averaging = Averaging(
        frame_length=int(oracle['frame_length']),
        overlap=float(oracle['overlap']),
        window=str(oracle['window']), frames=int(oracle['frames']))
    return record


def names_of(oracle):
    return [str(name) for name in oracle['signal_names']]


def test_the_framings_provably_coincide(oracle, history):
    """The precondition for machine-precision agreement, checked rather
    than assumed: sdynpy cuts frames at every hop from sample zero for
    as long as a whole frame fits, and this record's length holds
    exactly the frame count the Averaging names."""
    averaging = history.averaging
    samples = len(history.abscissa)
    assert averaging.span == samples, 'the frames tile the record exactly'
    hop = int(averaging.frame_length * (1 - averaging.overlap))
    sdynpy_frames = len(range(0, samples - averaging.frame_length + 1, hop))
    assert sdynpy_frames == averaging.frames


def test_psds_agree_with_sdynpy(oracle, history):
    psd = history.compute_psds()
    ours = {str(dof): row.real for dof, row in zip(psd.response_dof,
                                                   psd.ordinate)}
    for column, name in enumerate(names_of(oracle)):
        theirs = oracle['psd'][:, column].real
        assert np.max(np.abs(ours[name] - theirs)) <= (
            TOLERANCE * theirs.max()), name


def test_the_cpsd_matrix_agrees_up_to_the_documented_conjugation(
        oracle, history):
    """conj(X)·Y here, X·conj(Y) there: identical magnitudes, opposite
    phase sign, both self-consistent. Compared against the conjugate,
    with the unconjugated comparison asserted to *fail* — so if either
    side ever changes convention, this notices in both directions."""
    cpsds = history.compute_cpsds()
    ours = {(str(r), str(f)): row for r, f, row in
            zip(cpsds.response_dof, cpsds.reference_dof, cpsds.ordinate)}
    matrix = oracle['cpsd']
    names = names_of(oracle)
    unconjugated_disagrees = False
    for i, row_name in enumerate(names):
        for j, column_name in enumerate(names):
            mine = ours[(row_name, column_name)]
            theirs = matrix[:, i, j]
            scale = np.abs(theirs).max()
            assert np.max(np.abs(mine - np.conj(theirs))) <= (
                TOLERANCE * scale), (row_name, column_name)
            if np.max(np.abs(mine - theirs)) > 1e-3 * scale:
                unconjugated_disagrees = True
    assert unconjugated_disagrees, (
        'the off-diagonal phases really do differ, so the conjugation '
        'above is doing something')


@pytest.mark.parametrize('method', ['H1', 'Hv'])
def test_frfs_agree_with_sdynpy(oracle, history, method):
    frfs = history.compute_frfs(method=method)
    ours = {(str(r), str(f)): row for r, f, row in
            zip(frfs.response_dof, frfs.reference_dof, frfs.ordinate)}
    H = oracle[f'frf_{method}']
    for i, response in enumerate(names_of(oracle)[:3]):
        for j, reference in enumerate(names_of(oracle)[3:]):
            theirs = H[:, i, j]
            assert np.max(np.abs(ours[(response, reference)] - theirs)) <= (
                TOLERANCE * np.abs(theirs).max()), (method, response,
                                                    reference)


def test_square_h2_matches_sdynpy(oracle, history):
    """The coupled Gxx * Gfx^-1, matched by decision (Brandon,
    2026-08-28) and pinned by number — including the conjugation that
    translates this package's Bendat & Piersol convention into the
    classical form's."""
    fs = float(oracle['sample_rate'])
    names = names_of(oracle)
    kinds = [str(k) for k in oracle['signal_kinds']]
    responses = [str(r) for r in oracle['frf_H2_square_responses']]
    references = names[3:]
    keep = [names.index(n) for n in responses + references]
    square = TimeHistory(np.arange(oracle['signals'].shape[1]) / fs,
                         oracle['signals'][keep],
                         response_dof=[names[k] for k in keep],
                         ordinate_dim=[kinds[k] for k in keep])
    square.averaging = history.averaging
    frfs = square.compute_frfs(method='H2')
    ours = {(str(r), str(f)): row for r, f, row in
            zip(frfs.response_dof, frfs.reference_dof, frfs.ordinate)}
    H = oracle['frf_H2_square']
    for i, response in enumerate(responses):
        for j, reference in enumerate(references):
            theirs = H[:, i, j]
            assert np.max(np.abs(ours[(response, reference)] - theirs)) <= (
                TOLERANCE * np.abs(theirs).max()), (response, reference)


def test_h2_agrees_where_the_definitions_coincide(oracle, history):
    """One response against one reference, where the coupled form and
    the per-response form are the same formula — checked separately
    from the square case because it exercises the other code path."""
    fs = float(oracle['sample_rate'])
    names = names_of(oracle)
    response = str(oracle['frf_H2_response'])
    reference = str(oracle['frf_H2_reference'])
    keep = [names.index(response), names.index(reference)]
    kinds = [str(k) for k in oracle['signal_kinds']]
    pair = TimeHistory(np.arange(oracle['signals'].shape[1]) / fs,
                       oracle['signals'][keep],
                       response_dof=[names[k] for k in keep],
                       ordinate_dim=[kinds[k] for k in keep])
    pair.averaging = history.averaging
    frf = pair.compute_frfs(method='H2')
    theirs = oracle['frf_H2_single'][:, 0, 0]
    assert np.max(np.abs(frf.ordinate[0] - theirs)) <= (
        TOLERANCE * np.abs(theirs).max())


def test_multiple_coherence_agrees_with_sdynpy(oracle, history):
    """Both references at once, against sdynpy's own computation.

    The first oracle settled for the single-reference identity on the
    belief that sdynpy had no multiple-coherence computation; Brandon
    pointed at the one it has, inline in the SignalProcessingGUI's
    compute(). The generator drives that GUI off-screen rather than
    copying the block out — a copied block would be copied GPL code —
    so this is sdynpy's own code on sdynpy's own path, agreeing at
    1e-15.
    """
    coherence = history.compute_multiple_coherence(
        references=[str(n) for n in names_of(oracle)[3:]])
    ours = {str(dof): row.real for dof, row in
            zip(coherence.response_dof, coherence.ordinate)}
    for i, response in enumerate(oracle['multiple_coherence_responses']):
        theirs = oracle['multiple_coherence'][i]
        assert np.max(np.abs(ours[str(response)] - theirs)) <= TOLERANCE, \
            str(response)


def test_single_reference_multiple_coherence_is_sdynpys_coherence(
        oracle, history):
    """With one reference, multiple coherence *is* ordinary coherence,
    and sdynpy computes that by a different road (`cpsd_coherence`) —
    so the identity is checked too, against a second implementation."""
    coherence = history.compute_multiple_coherence(
        references=[str(oracle['coherence_reference'])])
    ours = {str(dof): row.real for dof, row in
            zip(coherence.response_dof, coherence.ordinate)}
    mine = ours[str(oracle['coherence_response'])]
    assert np.max(np.abs(mine - oracle['coherence_pair'])) <= TOLERANCE


def test_the_srs_agrees_with_sdynpy(oracle):
    """Both are Smallwood's ramp-invariant recursion at Q of 10,
    maximax absolute acceleration; sdynpy runs it through lfilter where
    this package wrote the recursion out, and the two ways of applying
    the same coefficients agree to 9e-15."""
    from visualdynamics.core.srs import maximax

    ours = maximax(oracle['transient'], oracle['srs_frequencies'],
                   float(oracle['sample_rate']))
    theirs = np.asarray(oracle['srs'])
    assert theirs.ndim == 1, 'the oracle stored values, not the tuple'
    assert np.max(np.abs(ours - theirs) / theirs) <= TOLERANCE


def test_band_rms_agrees_with_sdynpy(oracle, history):
    """compliance.rms sums line times bin width; sdynpy's rms_csd sums
    the diagonal times df. On the even rfft axis those are the same
    quadrature, so the same PSD gives the same RMS."""
    from visualdynamics.core.compliance import rms

    psd = history.compute_psds()
    ours = {str(dof): row.real for dof, row in zip(psd.response_dof,
                                                   psd.ordinate)}
    axis = np.arange(oracle['psd'].shape[0]) * float(oracle['df'])
    for column, name in enumerate(names_of(oracle)):
        theirs = float(oracle['rms'][column])
        assert abs(rms(axis, ours[name]) - theirs) <= TOLERANCE * theirs, name


def test_the_oracle_says_where_it_came_from(oracle):
    """The provenance rides in the file, so the numbers can be traced
    to the sdynpy version and script that made them without leaving
    this repository."""
    import json

    record = json.loads(str(oracle['provenance']))
    assert record['sdynpy'], 'the version is named'
    assert 'never imported' in record['note']


# ---- the second round: what the first oracle left out -------------------
# (Brandon asked what checks were missing, 2026-08-28. Found and added:
# MAC, the CMIF, the other windows and the no-overlap hop, and the SRS
# family beyond one type at one Q. Deliberately still out: octave
# banding — sdynpy's lives in a documentation-support class whose
# conventions were not verified — and the linear spectra and
# integrate/differentiate, whose conventions differ by design and are
# documented where they are implemented.)


def test_mac_agrees_with_sdynpy(oracle):
    """One formula, but both sides conjugate somewhere, and agreeing on
    random *complex* shapes proves the conjugates sit in the right
    places — real shapes would pass with them wrong. sdynpy lays
    shapes (dof, mode); this package (mode, dof); the transpose is the
    whole translation."""
    from visualdynamics.core.shapes import mac_matrix

    ours = mac_matrix(oracle['mac_shapes'].T)
    assert np.max(np.abs(ours - oracle['mac'])) < TOLERANCE


def test_the_cmif_agrees_with_sdynpy(oracle):
    """The singular values of the measured FRF matrix, per line — the
    curve Find Mode hunts — against sdynpy's compute_cmif on the same
    matrix.

    Compared in the coherent SI system, deliberately: cmif_curves
    serves the display and converts to the display units, and the
    first comparison came back off by exactly 0.45359237 — a
    kilogram's worth of pounds — because the default display is
    in-slinch-lbf-s (g) and the oracle's matrix is unitless SI. The
    conversion is correct behavior, not error, and naming the system
    is what makes the comparison about the SVD."""
    from visualdynamics.plot import cmif_curves
    from visualdynamics.units import SI

    H = oracle['frf_H1']
    rows, response, reference = [], [], []
    for i, r in enumerate(names_of(oracle)[:3]):
        for j, f in enumerate(names_of(oracle)[3:]):
            rows.append(H[:, i, j])
            response.append(r)
            reference.append(f)
    from visualdynamics.core.data import Frf

    built = Frf(oracle['frf_frequencies'], np.asarray(rows),
                response_dof=response, reference_dof=reference,
                ordinate_dim=['acceleration/force'] * len(rows))
    values, _x = cmif_curves(built, unit_system=SI)
    theirs = oracle['cmif']
    assert values.shape == theirs.shape
    assert np.max(np.abs(values - theirs)) <= TOLERANCE * theirs.max()


@pytest.mark.parametrize('window', ['rectangle', 'flattop'])
def test_the_other_windows_agree_too(oracle, history, window):
    """The first oracle held one window, and a hand-rolled flattop with
    a wrong fifth coefficient would have sailed through it. sdynpy
    reaches these windows through scipy, so this holds this package's
    own window tables to scipy's through the whole spectral pipeline —
    spelling included: scipy says boxcar where this package says
    rectangle."""
    from dataclasses import replace

    record = history
    record.averaging = replace(record.averaging, window=window)
    psd = record.compute_psds()
    ours = {str(dof): row.real for dof, row in zip(psd.response_dof,
                                                   psd.ordinate)}
    for column, name in enumerate(names_of(oracle)):
        theirs = oracle[f'psd_{window}'][:, column].real
        assert np.max(np.abs(ours[name] - theirs)) <= (
            TOLERANCE * theirs.max()), name
    record.averaging = replace(record.averaging, window='hann')


def test_the_no_overlap_hop_agrees(oracle):
    """The hop is its own convention — half-overlap holds it at half a
    frame, and zero holds it at a whole one — and the first oracle
    only ever exercised one of the two."""
    fs = float(oracle['sample_rate'])
    frame = int(oracle['frame_length'])
    frames = int(oracle['no_overlap_frames'])
    names = names_of(oracle)
    record = TimeHistory(
        np.arange(frame * frames) / fs,
        oracle['signals'][:, :frame * frames],
        response_dof=names,
        ordinate_dim=[str(k) for k in oracle['signal_kinds']])
    record.averaging = Averaging(frame_length=frame, overlap=0.0,
                                 window=str(oracle['window']),
                                 frames=frames)
    psd = record.compute_psds()
    ours = {str(dof): row.real for dof, row in zip(psd.response_dof,
                                                   psd.ordinate)}
    for column, name in enumerate(names):
        theirs = oracle['psd_no_overlap'][:, column].real
        assert np.max(np.abs(ours[name] - theirs)) <= (
            TOLERANCE * theirs.max()), name


def test_the_srs_family_agrees(oracle):
    """Beyond one type at one Q: the positive and negative peaks
    against sdynpy's spectrum types 7 and 8 (the signed maxima), and
    the maximax again at Q of 50, where the wavelets are ten times
    narrower and a coefficient error would scale differently."""
    from visualdynamics.core.srs import maximax, peaks

    fs = float(oracle['sample_rate'])
    frequencies = oracle['srs_frequencies']
    highest, lowest = peaks(oracle['transient'], frequencies, fs)
    type7 = oracle['srs_types'][6]
    type8 = np.abs(oracle['srs_types'][7])
    assert np.max(np.abs(highest - type7) / type7) <= TOLERANCE
    assert np.max(np.abs(-lowest - type8) / type8) <= TOLERANCE

    at_q50 = maximax(oracle['transient'], frequencies, fs, q=50.0)
    assert np.max(np.abs(at_q50 - oracle['srs_q50'])
                  / oracle['srs_q50']) <= TOLERANCE
