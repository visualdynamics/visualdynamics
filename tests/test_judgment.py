"""The comparison as an area against an area, over every pairing.

A requirement can be a curve between breakpoints, a controller's
density per line, or either banded; a measurement can be on lines or
on bands. One rule judges all six pairings (`compliance.cells`,
`compliance.judge`; PLAN.md "A comparison is an area against an
area"), and these are the cases the guide page draws — a measurement
on its target is out nowhere, a departure wide enough to move a band
is out everywhere, a requirement's hole is judged nowhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import compliance
from visualdynamics.core.compliance import log_interpolate
from visualdynamics.core.data import Psd, Specification

BREAKS = np.array([20.0, 80.0, 800.0, 2000.0])
TARGET = np.array([[0.01, 0.04, 0.04, 0.007]])


def breakpoints():
    limits = {name: TARGET * 10 ** (db / 10)
              for name, db in (('warning_lower', -3), ('warning_upper', 3),
                               ('abort_lower', -6), ('abort_upper', 6))}
    return Specification(BREAKS, TARGET, response_dof=['101Z+'],
                         ordinate_dim=['acceleration**2/frequency'], **limits)


def lines(df=2.0, high=2000.0, hole=None):
    """The same requirement as a controller writes it: NaN beyond
    `high`, and zero across `hole` when one is asked for."""
    written = breakpoints()
    f = np.arange(0.0, 2560.0 + df / 2, df)
    rows = {'target': log_interpolate(f, BREAKS, TARGET[0])}
    for name, values in written.limits.items():
        rows[name] = log_interpolate(f, BREAKS, values[0])
    for values in rows.values():
        values[f > high] = np.nan
        if hole is not None:
            values[(f >= hole[0]) & (f <= hole[1])] = 0.0
    out = Specification(f, np.atleast_2d(rows.pop('target')),
                        response_dof=['101Z+'],
                        ordinate_dim=['acceleration**2/frequency'],
                        **{name: np.atleast_2d(v) for name, v in rows.items()})
    out.interpolation = Specification.reading_of(f)
    return out


def response(shape=None, df=1.0, high=2560.0, seed=None):
    """A measurement on the target, `shape` times it where given, to
    Nyquist — past the requirement's end — and 2e-3 where the
    requirement says nothing."""
    f = np.arange(0.0, high + df / 2, df)
    target = log_interpolate(f, BREAKS, TARGET[0])
    values = np.where(np.isfinite(target), target, 2e-3)
    if seed is not None:
        # about three decibels line to line, what a real PSD does
        values = values * np.exp(0.35 * np.random.default_rng(seed).standard_normal(f.size))
    if shape is not None:
        values = values * shape(f)
    out = Psd(f, np.atleast_2d(values), response_dof=['101Z+'],
              ordinate_dim=['acceleration**2/frequency'])
    out.scale_db = 0
    return out


def pairings(measured_shape=None, seed=None):
    """The three comparisons there are (Brandon, 2026-09-19): a
    narrowband response against a curve or a requirement on lines,
    and a banded response against a requirement on the same bands."""
    narrow = response(measured_shape, seed=seed)
    octave = narrow.to_octave(3)
    written, controller = breakpoints(), lines()
    banded = controller.to_octave(3)
    return {
        'lines-vs-breakpoints': (narrow, written),
        'lines-vs-lines': (narrow, controller),
        'octave-vs-octave': (octave, banded),
    }


@pytest.mark.parametrize('name', list(pairings()))
def test_a_measurement_on_its_target_is_out_nowhere(name):
    """Whatever form the two take, a response that is the target is
    inside every limit, its RMS is the requirement's, and nothing of
    the band is outside abort. A band is whole: a banded object's end
    bands hold what fell in them over their whole width, so where one
    side is banded and the other is not the RMS reads a few
    hundredths of a decibel off at the ends; on lines it is exact."""
    measured, specification = pairings()[name]
    over, under = compliance.exceedances(specification, measured)
    assert not over.any() and not under.any(), name
    got = compliance.compare(specification, measured, scale_db=0)
    assert got['lines'] > 0
    assert got['abort_percent'] == 0.0 and got['warning_percent'] == 0.0
    # and a requirement on lines is the law sampled at its lines, a
    # thousandth of a decibel from the law's exact area
    slack = (0.1 if 'octave' in name
             else 1e-6 if name.endswith('breakpoints') else 1e-3)
    assert got['difference_db'] == pytest.approx(0.0, abs=slack), name


@pytest.mark.parametrize('name', list(pairings()))
def test_a_departure_wide_enough_to_move_a_band_is_out_everywhere(name):
    """A resonance six times the target across 250–350 Hz and a notch
    at a fifth across 1100–1450 Hz — each wide enough to fill a
    third-octave band and far enough past the 6 dB abort limits to
    stay past them once averaged over one: over the upper limit and
    under the lower one on lines and on bands alike, and each marked
    on the cells that hold them."""
    def shape(f):
        out = np.ones(f.size)
        out[(f >= 250.0) & (f <= 350.0)] = 6.0
        out[(f >= 1100.0) & (f <= 1450.0)] = 0.2
        return out

    measured, specification = pairings(shape)[name]
    over, under = compliance.exceedances(specification, measured)
    x = np.asarray(measured.abscissa)
    assert over.any() and under.any(), name
    assert x[over].min() >= 200.0 and x[over].max() <= 400.0, name
    assert x[under].min() >= 1000.0 and x[under].max() <= 1500.0, name
    got = compliance.compare(specification, measured, scale_db=0)
    assert got['abort_percent'] > 0.0
    assert got['abort_lines'] == compliance.judge(
        specification, measured, limit='abort_upper', over=True,
        scale_db=0)['out'].sum() + compliance.judge(
        specification, measured, limit='abort_lower', over=False,
        scale_db=0)['out'].sum()


def test_the_share_outside_is_a_share_of_width_not_a_count():
    """The same departure read on lines and on bands is the same share
    of the band, near enough — a count of cells would differ by a
    factor of fifty."""
    def shape(f):
        out = np.ones(f.size)
        out[(f >= 250.0) & (f <= 350.0)] = 6.0
        return out

    cases = pairings(shape)
    on_lines = compliance.compare(*reversed(cases['lines-vs-lines']), scale_db=0)
    on_bands = compliance.compare(*reversed(cases['octave-vs-octave']), scale_db=0)
    assert on_lines['abort_lines'] > 10 * on_bands['abort_lines']
    assert on_bands['abort_percent'] == pytest.approx(on_lines['abort_percent'], abs=2.5)


def test_bands_compare_only_with_the_same_bands():
    """A requirement on octave bands says nothing about the lines
    under them, and a band's power cannot be attributed to part of
    its width — so a narrowband response is not judged against a
    banded requirement, a banded response is not judged against a
    curve or lines, and sixths are not judged against thirds
    (Brandon, 2026-09-19). Each refusal says why, and nothing is
    marked."""
    narrow = response(seed=3)
    octave = narrow.to_octave(3)
    written, controller = breakpoints(), lines()
    banded = controller.to_octave(3)
    refused = [(narrow, banded), (octave, written), (octave, controller),
               (narrow.to_octave(6), banded)]
    for measured, specification in refused:
        why = compliance.comparable(specification, measured)
        assert why and 'octave bands' in why, (why, specification, measured)
        assert compliance.cells(specification, measured) == []
        over, under = compliance.exceedances(specification, measured, scale_db=0)
        assert not over.any() and not under.any()
        got = compliance.compare(specification, measured, scale_db=0)
        assert got['lines'] == 0 and got['refused'] == why
    for measured, specification in pairings().values():
        assert compliance.comparable(specification, measured) is None
    # the array form refuses widths on one side only
    marks = compliance.outside(np.asarray(octave.abscissa), np.real(octave.ordinate[0]),
                               BREAKS, TARGET[0] * 4, over=True,
                               widths=octave.bandwidth)
    assert not marks.any()


def test_a_hole_in_a_requirement_is_judged_nowhere():
    """A controller's notch, written as zero across 400–500 Hz: zero
    is not a requirement of silence. No cell lies in the notch, a
    response ten times over the target there is not out, and the RMS
    on both sides counts the written lines only."""
    def shape(f):
        out = np.ones(f.size)
        out[(f >= 420.0) & (f <= 480.0)] = 10.0
        return out

    notched = lines(hole=(400.0, 500.0))
    measured = response(shape)
    found = compliance.cells(notched, measured)
    assert not any(400.0 < c['low'] < 500.0 or 400.0 < c['high'] < 500.0
                   for c in found)
    stretches = compliance.coverage(notched)
    assert len(stretches) == 2 and stretches[0][1] == pytest.approx(399.0) \
        and stretches[1][0] == pytest.approx(501.0)
    over, _under = compliance.exceedances(notched, measured, scale_db=0)
    assert not over.any()
    got = compliance.compare(notched, measured, scale_db=0)
    assert got['difference_db'] == pytest.approx(0.0, abs=0.05)


def test_a_requirement_ending_inside_a_band_is_judged_over_the_whole_band():
    """The lines end at 1450 Hz inside the third-octave band that
    reaches 1778. Banded, the requirement's last band is that whole
    band — an octave band is a defined band (Brandon, 2026-09-19) —
    holding the lines' content over its whole width, and the
    comparison judges the whole band: the measurement's power from
    1413 to 1778 against that. A measurement that runs on past
    1450 Hz at the target's level therefore holds several times what
    the banded requirement asks in that band, and is over there — the
    consequence of banding a requirement onto a band it does not
    fill, which the narrowband comparison does not have. Past the
    last band nothing is judged."""
    short = lines(high=1450.0).to_octave(3)
    measured = response().to_octave(3)
    left, right = short.bin_bounds()
    assert right[-1] == pytest.approx(1778.279, rel=1e-4), 'the standard band, whole'
    found = compliance.cells(short, measured)
    assert found[-1]['high'] == pytest.approx(right[-1])
    assert found[-1]['low'] == pytest.approx(left[-1])
    verdict = compliance.judge(short, measured, limit='abort_upper', over=True,
                               scale_db=0)
    assert verdict['out'][-1], 'the last band reads over'
    assert not verdict['out'][:-1].any(), 'and only it'
    x = np.asarray(measured.abscissa)
    assert verdict['lines'][x > right[-1]].sum() == 0
    # the case that happens (Brandon, 2026-09-20): a controller holds
    # the response to the requirement and drives nothing past it, so
    # the response falls away past 1450 Hz too, both last bands sit
    # low together, and the band is in
    controlled = response(lambda f: np.where(f > 1450.0, 0.02, 1.0)).to_octave(3)
    over, under = compliance.exceedances(short, controlled, scale_db=0)
    assert not over.any() and not under.any()
    # the same requirement ending on a band edge is filled and clean
    whole = lines(high=1777.0).to_octave(3)   # the lines end on the band's edge
    over, under = compliance.exceedances(whole, measured, scale_db=0)
    assert not over.any() and not under.any()




def test_the_window_says_why_a_pair_is_not_compared(window, pump):
    """A narrowband PSD selected with a requirement on octave bands is
    drawn, and the bars stand empty — and the status line says why,
    and what to do: band the response the same way."""
    banded = lines().to_octave(3)
    narrow = response()
    window.add_object('Spec', banded)
    window.add_object('PSDs', narrow)
    window.tree.setCurrentItem(window._item_for_object('Spec'))
    window.tree.clearSelection()
    for name in ('Spec', 'PSDs'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    message = window.statusBar().currentMessage()
    assert 'Not compared: ' in message
    assert 'octave bands' in message and 'band the response' in message
