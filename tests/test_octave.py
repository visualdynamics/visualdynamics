"""Proportional-band spectra: the grid, the integration, and the reading.

The band grid is the base-ten system of ANSI S1.11 / IEC 61260, which
is what sdynpy uses — so the numbers below are sdynpy's own, recorded
by running `nth_octave_freqs` once and written down here. Numbers are
facts and not expression: the implementation was written from the
standard, and these pin it to the same grid without the package ever
importing a GPL library. See `tests/test_license_boundary.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core import octave
from visualdynamics.core.data import Psd

#: (per_octave: (first five lower edges, first five centers, band count,
#: last upper edge)) over 20-2000 Hz, from sdynpy.
SDYNPY = {
    1: ([11.220184543, 22.387211386, 44.668359215, 89.125093813,
         177.827941004],
        [15.848931925, 31.622776602, 63.095734448, 125.892541179,
         251.188643151], 8, 2818.382931264),
    3: ([17.7827941, 22.387211386, 28.183829313, 35.481338923,
         44.668359215],
        [19.95262315, 25.118864315, 31.622776602, 39.810717055,
         50.118723363], 21, 2238.721138568),
    6: ([19.95262315, 22.387211386, 25.118864315, 28.183829313,
         31.622776602],
        [21.134890398, 23.713737057, 26.607250598, 29.853826189,
         33.496543916], 41, 2238.721138568),
}


# ---- the grid -----------------------------------------------------------


@pytest.mark.parametrize('per_octave', sorted(SDYNPY))
def test_the_bands_are_the_ones_sdynpy_uses(per_octave):
    lower, centers, count, last = SDYNPY[per_octave]
    got_centers, _widths, bounds = octave.bands(20.0, 2000.0, per_octave)
    assert len(got_centers) == count
    assert bounds[:5] == pytest.approx(lower, rel=1e-9)
    assert got_centers[:5] == pytest.approx(centers, rel=1e-9)
    assert bounds[-1] == pytest.approx(last, rel=1e-9)


def test_an_octave_is_ten_to_the_three_tenths_not_two():
    """The base-ten system. A base-two reading differs in the fourth
    digit, which is enough to put every edge in a different place."""
    assert octave.ratio(1) == pytest.approx(10.0 ** 0.3, rel=1e-12)
    assert octave.ratio(6) == pytest.approx(10.0 ** 0.05, rel=1e-12)
    assert octave.ratio(6) != pytest.approx(2.0 ** (1 / 6), rel=1e-5)


@pytest.mark.parametrize('per_octave', [1, 2, 3, 6, 12, 24])
def test_a_thousand_hertz_anchors_every_fraction(per_octave):
    """What makes it a standard grid rather than an arbitrary one."""
    centers, _widths, bounds = octave.bands(20.0, 5000.0, per_octave)
    assert (np.any(np.isclose(bounds, 1000.0))
            or np.any(np.isclose(centers, 1000.0)))


def test_odd_fractions_put_the_centers_on_the_grid_and_even_the_edges():
    """Getting this backwards puts every band half a step out, which is
    a real disagreement and not a rounding one."""
    for per_octave in (1, 3):
        centers, _w, _b = octave.bands(20.0, 2000.0, per_octave)
        steps = np.log10(centers) * 10 * per_octave / 3
        assert np.allclose(steps, np.round(steps)), per_octave
    for per_octave in (2, 6, 12):
        _c, _w, bounds = octave.bands(20.0, 2000.0, per_octave)
        steps = np.log10(bounds) * 10 * per_octave / 3
        assert np.allclose(steps, np.round(steps)), per_octave


def test_the_grid_does_not_move_with_the_range_asked_for():
    """Brandon's question, and the reason a specification is not
    consulted: the range only chooses which bands of one fixed grid
    come back, so two runs banded the same way land on the same bands.
    """
    narrow, _w, _b = octave.bands(20.0, 2000.0, 6)
    wide, _w2, _b2 = octave.bands(5.0, 8000.0, 6)
    shared = {round(float(f), 9) for f in narrow} & {
        round(float(f), 9) for f in wide}
    assert len(shared) == len(narrow)


def test_bands_tile_and_center_on_their_own_edges():
    centers, widths, bounds = octave.bands(20.0, 2000.0, 6)
    assert bounds[1:] == pytest.approx(bounds[:-1] + widths)
    assert centers == pytest.approx(np.sqrt(bounds[:-1] * bounds[1:]))


def test_a_range_that_is_not_a_range_is_refused():
    with pytest.raises(ValueError, match='not a range'):
        octave.bands(2000.0, 20.0, 6)


# ---- the integration ----------------------------------------------------


def flat(level=1e-3, df=0.5, top=2000.0):
    lines = np.arange(df, top + df / 2, df)
    return Psd(abscissa=lines,
               ordinate=np.atleast_2d(np.full(lines.size, level)),
               response_dof=['101Z+'],
               ordinate_dim=['acceleration**2/frequency'])


def test_banding_keeps_the_area_under_the_spectrum():
    """It is an integration, not a resampling: reading the narrowband
    curve at each band center would throw away everything between the
    centers and conserve nothing."""
    narrow = flat()
    banded = narrow.to_octave(6)
    before = float(np.sum(np.real(narrow.ordinate[0]) * narrow.bin_widths()))
    after = float(np.nansum(np.real(banded.ordinate[0])
                            * banded.bin_widths()))
    assert after == pytest.approx(before, rel=1e-6)


def test_a_flat_spectrum_stays_flat():
    """Every band of a constant density holds that density, whatever
    its width."""
    banded = flat(level=4e-3).to_octave(6)
    inside = banded.ordinate[0][np.isfinite(banded.ordinate[0])]
    assert np.allclose(np.real(inside[2:-2]), 4e-3, rtol=1e-6)


def test_a_band_carries_its_own_width():
    """Its bins are geometric and come from a standard, so reading them
    off the centers would be a hair out at every band and plainly wrong
    at the two ends."""
    banded = flat().to_octave(6)
    assert banded.bandwidth is not None
    assert np.allclose(banded.bin_widths(), banded.bandwidth)
    assert not np.allclose(banded.bin_widths(),
                           np.gradient(banded.abscissa)), (
        'the midpoint reading is not the same as the standard one')


def test_a_narrowband_spectrum_infers_its_bins_as_before():
    narrow = flat(df=0.5)
    assert narrow.bandwidth is None
    assert np.allclose(narrow.bin_widths(), 0.5)


def test_the_bands_come_back_as_the_standard_edges():
    """The plot reconstructs the edges from center and width, and has
    to land back on the grid or a step plot draws gaps."""
    from visualdynamics.plot import bin_edges

    banded = flat().to_octave(6)
    _c, _w, bounds = octave.bands(banded.abscissa[0], banded.abscissa[-1], 6)
    assert bin_edges(banded.abscissa, banded.bin_widths()) == pytest.approx(
        bounds, rel=1e-9)


def test_a_cross_spectrum_keeps_its_phase_through_the_bands():
    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    cpsds = loaded['time_data'].compute_cpsds()
    banded = cpsds.to_octave(6)
    assert np.iscomplexobj(banded.ordinate)
    assert banded.num_records == cpsds.num_records
    assert np.abs(np.imag(banded.ordinate)).max() > 0


def test_a_spectrum_with_no_positive_frequency_is_refused():
    bare = Psd(abscissa=np.array([0.0]), ordinate=np.ones((1, 1)),
               response_dof=['101Z+'])
    with pytest.raises(ValueError, match='above'):
        bare.to_octave(6)


def test_it_round_trips_through_visualdynamics(tmp_path):
    banded = flat().to_octave(6)
    visualdynamics.save(banded, tmp_path / 'banded')
    back = visualdynamics.load(tmp_path / 'banded.vdyn')
    assert back.bandwidth is not None
    assert np.allclose(back.bandwidth, banded.bandwidth)


# ---- and the comparison it is for ---------------------------------------


def compared():
    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    psds = loaded['time_data'].compute_psds()
    # held unscaled — and before the banding, so both gridings carry
    # it: the subject is that banding conserves the error, and the two
    # grids detecting different whole decibels would hide that behind
    # a scaling disagreement (which one-resolution handles elsewhere)
    psds.scale_db = 0
    return loaded['Random_specification'], psds, psds.to_octave(6)


def test_the_rms_error_is_the_same_on_bands_as_on_lines():
    """The whole argument for not banding the specification. Banding
    conserves the area under the measurement, the specification's own
    level is integrated from its breakpoints either way, so the level
    each channel came out at cannot depend on how the measurement was
    arranged."""
    from visualdynamics.core.compliance import channel_errors, compare_all

    spec, narrow, banded = compared()
    on_lines = {label: db for label, db, _p in
                channel_errors(compare_all(spec, narrow))}
    on_bands = {label: db for label, db, _p in
                channel_errors(compare_all(spec, banded))}
    assert set(on_lines) == set(on_bands)
    for label, db in on_lines.items():
        assert on_bands[label] == pytest.approx(db, abs=0.05), label


def test_the_share_outside_abort_is_counted_over_bands():
    """And this one *should* differ: it counts how much of the band is
    outside, and a sixth-octave spectrum has tens of bands where the
    narrowband had hundreds of lines. Same question, coarser ruler."""
    from visualdynamics.core.compliance import compare_all

    spec, narrow, banded = compared()
    on_lines = dict(compare_all(spec, narrow))
    on_bands = dict(compare_all(spec, banded))
    assert on_bands['101Z+']['lines'] < on_lines['101Z+']['lines']
    assert on_bands['101Z+']['abort_percent'] != pytest.approx(
        on_lines['101Z+']['abort_percent'], abs=1e-6)


def test_the_specification_is_not_offered_a_banding():
    """It is a continuous curve whose area over a band is the same
    however the measurement beside it was arranged."""
    from visualdynamics.core.data import Specification

    spec = Specification(
        abscissa=np.array([20.0, 2000.0]),
        ordinate=np.atleast_2d([1e-3, 1e-3]), response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'])
    assert isinstance(spec, Psd), 'it is a Psd underneath, which is the trap'
    # the applicability table excludes a specification, and the
    # calculator menu is a presentation of the same table
    import visualdynamics

    offered = dict(visualdynamics.Project('t').verbs(spec))
    assert 'compute_octave' not in offered


# ---- banding what is already banded -------------------------------------


def test_banding_the_same_way_twice_changes_nothing():
    """It has to be idempotent, and it was not.

    Two faults, both worth keeping. `resample` inferred the source's
    bins from the midpoints between its centers, which is exact for
    evenly spaced FFT lines and wrong for a spectrum already on
    geometric bands — the guess ran 0.17% wide through the middle and
    5.9% wide at the first band, smearing content into its neighbors.
    And the grid grew a band each time: a band edge comes back from
    ten-to-the-power and a logarithm a float's breadth adrift, and
    flooring that reached one step further down.
    """
    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    psds = loaded['time_data'].compute_psds()
    once = psds.to_octave(6)
    twice = once.to_octave(6)
    assert len(twice.abscissa) == len(once.abscissa)
    assert twice.abscissa == pytest.approx(once.abscissa, rel=1e-12)
    assert twice.bandwidth == pytest.approx(once.bandwidth, rel=1e-12)
    good = np.isfinite(once.ordinate) & np.isfinite(twice.ordinate)
    assert np.real(twice.ordinate)[good] == pytest.approx(
        np.real(once.ordinate)[good], rel=1e-9)


def test_it_stays_put_however_many_times():
    """Not merely stable to one more pass."""
    banded = flat().to_octave(6)
    again = banded
    for _ in range(4):
        again = again.to_octave(6)
    assert len(again.abscissa) == len(banded.abscissa)
    good = np.isfinite(banded.ordinate) & np.isfinite(again.ordinate)
    assert np.real(again.ordinate)[good] == pytest.approx(
        np.real(banded.ordinate)[good], rel=1e-9)


def test_a_coarser_banding_of_a_banded_spectrum_still_conserves():
    """Sixths to thirds is a real conversion rather than a no-op, and
    the area still has to survive it — which needs the source's own
    bins, not a guess at them."""
    banded = flat().to_octave(6)
    coarser = banded.to_octave(3)
    before = float(np.nansum(np.real(banded.ordinate[0])
                             * banded.bin_widths()))
    after = float(np.nansum(np.real(coarser.ordinate[0])
                            * coarser.bin_widths()))
    assert len(coarser.abscissa) < len(banded.abscissa)
    assert after == pytest.approx(before, rel=1e-6)


def test_the_bins_it_reads_are_the_bands_it_carries():
    """The mechanism under all three: a banded spectrum hands over its
    own bin edges rather than letting them be inferred."""
    banded = flat().to_octave(6)
    left, right = banded.bin_bounds()
    assert right - left == pytest.approx(banded.bandwidth, rel=1e-12)
    assert left[1:] == pytest.approx(right[:-1], rel=1e-12), 'they tile'
    guessed = np.gradient(banded.abscissa)
    assert not np.allclose(guessed, banded.bandwidth, rtol=1e-4), (
        'and the guess really is different, or this proves nothing')
