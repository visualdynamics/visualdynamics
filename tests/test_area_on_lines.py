"""A breakpoint specification read onto lines keeps its area, at any
spacing, and the comparison reads those lines as what they are."""

import dataclasses

import numpy as np
import pytest

from visualdynamics.core import compliance
from visualdynamics.core.author import SpecificationDraft
from visualdynamics.core.compliance import area_on_lines, log_log_area, record_on_lines
from visualdynamics.core.data import Psd, Specification

DIM = 'acceleration**2/frequency'
FLAT = ([21.9, 2000.0], [0.01, 0.01])
SLOPED = ([20.0, 80.0, 350.0, 2000.0], [0.01, 0.04, 0.04, 0.007])


def _sheet(frequencies, levels):
    draft = SpecificationDraft.at_dofs(['1Z+'], 'acceleration')
    n = len(frequencies) - 1
    return dataclasses.replace(
        draft, frequencies=list(frequencies), levels=[list(levels)],
        bands=[{**draft.bands[0], 'warning': draft.bands[0]['warning'][:1] * n,
                'abort': draft.bands[0]['abort'][:1] * n}])


def _spec(frequencies=FLAT[0], levels=FLAT[1], **limits):
    return Specification(np.array(frequencies), np.array([levels], dtype=float),
                         response_dof=['1Z+'], ordinate_dim=[DIM], **limits)


def _psd(f, values):
    return Psd(f, np.array([values], dtype=float), response_dof=['1Z+'],
               ordinate_dim=[DIM])


@pytest.mark.parametrize('spacing', [0.25, 0.5, 1.0, 2.0, 4.0, 3.0])
@pytest.mark.parametrize('written', [FLAT, SLOPED])
def test_the_lines_hold_the_area_of_the_curve_at_any_spacing(written, spacing):
    lines = _sheet(*written).to_lines(spacing)
    area = np.sum(lines.levels[0]) * spacing
    assert area == pytest.approx(log_log_area(*written), rel=1e-12)


def test_an_end_line_holds_the_part_of_its_bin_the_curve_covers():
    """21.9 Hz is 0.6 of the way through the 22 Hz line's bin, and 2000 Hz
    is half way through its own."""
    lines = _sheet(*FLAT).to_lines(1.0)
    assert lines.frequencies[0] == 22.0 and lines.frequencies[-1] == 2000.0
    assert lines.levels[0][0] == pytest.approx(0.006)
    assert lines.levels[0][-1] == pytest.approx(0.005)
    assert np.allclose(lines.levels[0][1:-1], 0.01)
    assert any('an end line holds only' in note for note in lines.notes)


def test_a_curve_reaching_barely_into_a_bin_makes_that_line():
    lines = _sheet([22.4, 2000.0], [0.01, 0.01]).to_lines(1.0)
    assert lines.frequencies[0] == 22.0, 'its center is below the curve'
    assert lines.levels[0][0] == pytest.approx(0.01 * 0.1)
    lines = _sheet([22.5, 2000.0], [0.01, 0.01]).to_lines(1.0)
    assert lines.frequencies[0] == 23.0, 'the next bin starts on the curve'
    assert lines.levels[0][0] == pytest.approx(0.01)
    lines = _sheet([21.9, 2000.0], [0.01, 0.01]).to_lines(4.0)
    assert lines.frequencies[0] == 20.0
    assert lines.levels[0][0] == pytest.approx(0.01 * 0.1 / 4.0)


def test_a_curve_written_to_the_edges_of_its_bins_is_flat_on_its_lines():
    """The writer's way to a flat target at 1 Hz: every line whole."""
    lines = _sheet([19.5, 2000.5], [0.01, 0.01]).to_lines(1.0)
    assert lines.frequencies[0] == 20.0 and lines.frequencies[-1] == 2000.0
    assert np.allclose(lines.levels[0], 0.01)
    assert not any('an end line holds only' in note for note in lines.notes)


@pytest.mark.parametrize('spacing', [0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize('written', [FLAT, SLOPED,
                                     ([10.0, 40.0, 200.0, 2000.0],
                                      [0.0002, 0.002, 0.002, 0.0002])])
def test_the_lines_fold_back_to_the_breakpoints(written, spacing):
    lines = _sheet(*written).to_lines(spacing)
    folded = lines.to_breakpoints()
    assert folded.frequencies == pytest.approx(written[0], abs=1e-6)
    assert folded.levels[0] == pytest.approx(written[1], rel=1e-6)
    assert np.allclose(folded.to_lines(spacing).levels, lines.levels)


def test_a_sheet_on_lines_read_again_is_not_read_twice():
    lines = _sheet(*FLAT).to_lines(1.0)
    assert lines.to_lines(1.0) is lines
    assert np.sum(lines.to_lines(2.0).levels[0]) * 2.0 == pytest.approx(
        log_log_area(*FLAT), rel=1e-9)


def test_a_cross_term_keeps_its_coherence_and_phase_on_a_part_bin():
    draft = SpecificationDraft.at_dofs(['1X+', '2X+'], 'acceleration')
    spec = dataclasses.replace(draft, levels=[[1.0, 0.01], [4.0, 0.4]],
                               pairs={(0, 1): (0.8, -40.0)}).make()
    lines = np.arange(0.0, 2100.0, 5.0)
    i = next(k for k in range(spec.num_records)
             if spec.record_pair(k) == ('1X+', '1X+'))
    j = next(k for k in range(spec.num_records)
             if spec.record_pair(k) == ('2X+', '2X+'))
    cross = next(k for k in range(spec.num_records)
                 if spec.response_dof[k] != spec.reference_dof[k])
    term = record_on_lines(spec, cross, lines)
    auto_a = record_on_lines(spec, i, lines)
    auto_b = record_on_lines(spec, j, lines)
    held = np.isfinite(auto_a) & np.isfinite(auto_b)
    assert held.sum() > 300 and np.isnan(term[~held]).all()
    np.testing.assert_allclose(
        np.abs(term[held]) ** 2 / (auto_a[held] * auto_b[held]), 0.8, rtol=1e-9)
    np.testing.assert_allclose(np.abs(np.angle(term[held], deg=True)), 40.0,
                               rtol=1e-9)


def test_the_area_read_is_nan_where_the_bin_holds_none_of_the_curve():
    lines = np.arange(0.0, 2010.0, 1.0)
    read = area_on_lines(*FLAT, lines)
    assert np.isnan(read[:22]).all() and np.isnan(read[2001:]).all()
    assert read[22] == pytest.approx(0.006)


def _limits(spec_f=FLAT[0], level=0.01):
    return {'abort_lower': np.array([[level / 4.0] * 2]),
            'abort_upper': np.array([[level * 4.0] * 2])}


@pytest.mark.parametrize('spacing', [1.0, 4.0])
def test_a_response_on_the_lines_read_from_the_curve_is_out_nowhere(spacing):
    """An end line holding a sliver of its bin is not low against a curve
    that only reached a sliver into it."""
    spec = _spec(**_limits())
    lines = _sheet(*FLAT).to_lines(spacing)
    f = np.arange(0.0, 4096.0, spacing)
    values = np.zeros_like(f)
    values[np.searchsorted(f, lines.frequencies)] = lines.levels[0]
    over, under = compliance.exceedances(spec, _psd(f, values), scale_db=0)
    assert not over.any() and not under.any()


def _warned(low=20.0, high=2000.0):
    level = np.array([[0.01, 0.01]])
    return Specification(np.array([low, high]), level, response_dof=['1Z+'],
                         ordinate_dim=[DIM], warning_lower=level / 10 ** 0.3,
                         warning_upper=level * 10 ** 0.3)


@pytest.mark.parametrize('df', [0.5, 1.0, 2.0, 4.0])
def test_a_line_is_judged_by_the_energy_in_its_whole_bin(df):
    """The curve's energy does not depend on the spacing (Brandon,
    2026-11-07). A response on the area-equivalent lines holds exactly
    that at any spacing and is out nowhere. One running flat through the
    end lines holds more, by an amount that grows with the bin, and those
    two lines are over."""
    spec = _warned()
    f = np.arange(0.0, 4096.0 + df, df)
    on = area_on_lines(spec.abscissa, spec.ordinate[0], f)
    held = _psd(f, np.nan_to_num(on))
    over, under = compliance.exceedances(spec, held, 0, 0, 'warning', 0.0)
    assert not over.any() and not under.any()
    got = compliance.compare(spec, held, scale_db=0)
    assert got['measured_rms'] ** 2 == pytest.approx(19.8)
    assert got['specification_rms'] ** 2 == pytest.approx(19.8)
    assert 'partly_written' not in got

    flat = _psd(f, np.where(np.isfinite(on), 0.01, 0.0))
    over, under = compliance.exceedances(spec, flat, 0, 0, 'warning', 0.0)
    assert not under.any()
    assert f[over].tolist() == [20.0, 2000.0]
    got = compliance.compare(spec, flat, scale_db=0)
    assert got['measured_rms'] ** 2 == pytest.approx(19.8 + 0.01 * df)
    assert [line for line, _share in got['partly_written']] == pytest.approx([20.0, 2000.0])
    assert [share for _line, share in got['partly_written']] == pytest.approx([0.5, 0.5])
    assert compliance.partly_written_text(got['partly_written']).endswith(
        'whole bin: 20 Hz (50 %), 2000 Hz (50 %)')


def test_a_spec_written_to_the_bins_has_one_reading():
    """Written from 19.5 to 2000.5 Hz, every 1 Hz line is whole, so a flat
    response is on it, and half a line is under the limits."""
    spec = _warned(19.5, 2000.5)
    f = np.arange(0.0, 4096.0)
    flat = np.where((f >= 20.0) & (f <= 2000.0), 0.01, 0.0)
    over, under = compliance.exceedances(spec, _psd(f, flat), 0, 0, 'warning', 0.0)
    assert not over.any() and not under.any()
    flat[2000] /= 2.0
    over, under = compliance.exceedances(spec, _psd(f, flat), 0, 0, 'warning', 0.0)
    assert not over.any() and f[under].tolist() == [2000.0]


def test_a_range_cut_from_a_longer_curve_is_not_an_end():
    """A flat response is flat across a step in the tolerance: the line
    on the step is read in each range over its own part, and only the
    curve's real ends read whole."""
    from visualdynamics.core.requirement import within

    spec = _warned()
    f = np.arange(0.0, 4096.0)
    held = _psd(f, np.full(f.size, 0.01))
    below = compliance.cells(within(spec, 0, None, 500.0), held)
    above = compliance.cells(within(spec, 0, 500.0, None), held)
    assert below[0]['pieces'] == [(19.5, 20.5)]
    assert below[-1]['pieces'] == [(499.5, 500.0)]
    assert above[0]['pieces'] == [(500.0, 500.5)]
    assert above[-1]['pieces'] == [(1999.5, 2000.5)]
    assert below[0]['low'] == 20.0 and above[-1]['high'] == 2000.0


def test_a_dead_end_line_and_a_hot_one_are_still_found():
    spec = _spec(**_limits())
    f = np.arange(0.0, 4096.0, 1.0)
    values = np.full(f.size, 0.01)
    values[22] = 0.0
    values[2000] = 0.05
    over, under = compliance.exceedances(spec, _psd(f, values), scale_db=0)
    assert under[22] and over[2000]
    assert under.sum() == 1 and over.sum() == 1


def test_a_sheet_on_lines_is_made_a_density_per_line():
    """The Interpolated button lands here. Marked a curve, the end lines
    would be read as points on it and the area would come out wrong."""
    sheet = _sheet(*SLOPED).to_lines(1.0)
    made = sheet.make('x')
    assert made.interpolation == 'bin'
    assert made.area(0) == pytest.approx(log_log_area(*SLOPED), rel=1e-12)
    as_a_curve = sheet._copy(form='breakpoints').make('x')
    assert as_a_curve.interpolation == 'log_log'
    assert abs(as_a_curve.area(0) / made.area(0) - 1.0) > 1e-5, \
        'the reading has to matter for this to pin anything'


def test_a_few_lines_read_from_a_curve_are_lines_too():
    """Fewer than `Specification.LINES_AT_LEAST`, which alone would read
    as breakpoints."""
    made = _sheet([100.0, 120.0], [0.01, 0.01]).to_lines(10.0).make('x')
    assert len(made.abscissa) < Specification.LINES_AT_LEAST
    assert made.interpolation == 'bin'
    assert made.area(0) == pytest.approx(0.01 * 20.0)


def test_a_few_even_breakpoints_stay_a_curve_through_the_sheet():
    spec = _spec([100.0, 200.0, 300.0], [0.01, 0.04, 0.02])
    spec.interpolation = 'log_log'
    opened = SpecificationDraft.from_specification(spec)
    assert opened.form == 'breakpoints'
    assert opened.make('x').interpolation == 'log_log'


def test_a_gap_in_a_curve_holds_no_area_in_a_bin_that_reaches_it():
    """A zero written between two points is a gap, as `coverage` reads it."""
    f = [10.0, 20.0, 30.0, 40.0, 50.0]
    w = [1.0, 1.0, 0.0, 1.0, 1.0]
    assert log_log_area(f, w) == pytest.approx(20.0)
    lines = np.arange(10.0, 51.0, 2.0)
    on = area_on_lines(f, w, lines)
    assert np.isnan(on[lines == 30.0])
    assert on[lines == 20.0] == pytest.approx(0.5), 'half its bin is written'
    assert np.nansum(on) * 2.0 == pytest.approx(20.0)


@pytest.mark.parametrize('off', [2e-12, 2e-11, 2e-10, 2e-9])
@pytest.mark.parametrize('band', [(500.0, 501.0), (500.0, 500.01)])
def test_the_area_keeps_its_digits_a_hair_off_a_slope_of_minus_one(off, band):
    n = -1.0 + off
    f, w = [100.0, 1000.0], [1.0, 10.0 ** n]
    low, high = band
    exact = (100.0 ** -n * low ** (n + 1.0)
             * np.expm1((n + 1.0) * np.log(high / low)) / (n + 1.0))
    got = compliance.log_log_areas(f, w, [low], [high])[0]
    assert got == pytest.approx(exact, rel=1e-6)


def test_a_flagged_end_line_is_named_beside_the_numbers():
    spec = _spec([20.0, 2000.0], [0.01, 0.01],
                 warning_upper=np.array([[0.01, 0.01]]) * 10 ** 0.3,
                 warning_lower=np.array([[0.01, 0.01]]) / 10 ** 0.3)
    f = np.arange(20.0, 2000.5, 1.0)
    got = compliance.compare(spec, _psd(f, np.full(f.size, 0.01)), scale_db=0)
    assert got['warning_lines'] == 2, 'both end lines read +3.01 dB'
    assert got['partly_written'] == [(20.0, pytest.approx(0.5)),
                                     (2000.0, pytest.approx(0.5))]
    said = compliance.comparison_note([('1Z+', got)])
    assert said.endswith('20 Hz (50 %), 2000 Hz (50 %)')
    assert compliance.comparison_note([('1Z+', {'lines': 3})]) is None


def test_a_cross_term_with_no_autospectra_holds_the_share_of_its_bin():
    """With nothing to follow, it takes the part of the bin it covers."""
    spec = Specification(np.array(FLAT[0]), np.array([[0.004 + 0j, 0.004 + 0j]]),
                         response_dof=['1Z+'], reference_dof=['2Z+'],
                         ordinate_dim=[DIM])
    read = record_on_lines(spec, 0, np.array([21.0, 22.0, 23.0]))
    assert np.isnan(read[0]), 'its bin ends where the curve starts'
    assert read[1] == pytest.approx(0.004 * 0.6)
    assert read[2] == pytest.approx(0.004)


def test_an_edited_line_is_not_lost_when_the_sheet_folds():
    """A line changed by hand is not on the laws around it, and folding to
    them would write it away; the fold reads its answer back onto every
    line and gives up."""
    sheet = _sheet(*SLOPED).to_lines(1.0)
    at = sheet.frequencies.index(80.0)
    assert sheet.to_breakpoints().frequencies == SLOPED[0], 'unedited, it folds'
    levels = [list(sheet.levels[0])]
    levels[0][at] *= 1.5
    folded = dataclasses.replace(sheet, levels=levels).to_breakpoints()
    assert folded.frequencies != SLOPED[0]
    assert 80.0 in folded.frequencies
    assert folded.levels[0][folded.frequencies.index(80.0)] == pytest.approx(levels[0][at])


def test_the_preview_of_a_sheet_on_lines_reads_as_the_object_will():
    assert _sheet(*FLAT).to_lines(1.0).preview().interpolation == 'bin'
    assert _sheet(*FLAT).preview().interpolation == 'log_log'


def test_a_generated_signal_is_written_from_the_area_on_each_line():
    from visualdynamics.core.signals import _level_at

    lines = np.arange(0.0, 2100.0, 1.0)
    level = _level_at(_spec(), 0, lines)
    assert level[22] == pytest.approx(0.006)
    assert level[2000] == pytest.approx(0.005)
    assert level[21] == 0.0 and level[2001] == 0.0 and level[0] == 0.0
    assert level.sum() == pytest.approx(log_log_area(*FLAT), rel=1e-12)


def test_two_lines_hold_their_bins_areas_too():
    """Two lines at the curve's start: each holds its bin's area, which is
    not the curve at its center."""
    lines = np.array([20.0, 22.0])
    held = area_on_lines(*SLOPED, lines)
    expected = compliance.log_log_areas(*SLOPED, lines - 1.0, lines + 1.0) / 2.0
    assert held == pytest.approx(expected, rel=1e-12)
    assert not np.allclose(held, compliance.log_interpolate(lines, *SLOPED))


def test_a_bin_wholly_outside_the_curve_is_not_a_number():
    held = area_on_lines(*FLAT, np.array([10.0, 11.0, 12.0]))
    assert np.isnan(held).all()


def _cross(frequencies, values, **kw):
    return Specification(np.array(frequencies), np.array([values]),
                         response_dof=['1Z+'], reference_dof=['2Z+'],
                         ordinate_dim=[DIM], **kw)


def test_a_cross_term_alone_is_read_on_its_own_curve_at_the_ends():
    """Rising and turning, so an end line clipped to the wrong point or
    scaled by the wrong width reads off."""
    f = [21.9, 100.0, 2000.0]
    v = [0.001 + 0j, 0.004 * np.exp(0.5j), 0.002 * np.exp(1.0j)]
    lines = np.arange(20.0, 2003.0, 2.0)
    read = record_on_lines(_cross(f, v), 0, lines)
    at = compliance.cross_term_at(f, v, np.clip(lines, f[0], f[-1]))
    left, right = lines - 1.0, lines + 1.0
    share = np.clip(np.minimum(right, f[-1]) - np.maximum(left, f[0]), 0.0, None) / 2.0
    reached = (right > f[0]) & (left < f[-1])
    assert np.isnan(read[~reached]).all()
    assert read[reached] == pytest.approx((at * share)[reached], rel=1e-12)
    assert read[lines == 22.0][0] == pytest.approx(at[lines == 22.0][0] * 0.55)
    assert read[lines == 2000.0][0] == pytest.approx(at[lines == 2000.0][0] * 0.5)
    two = record_on_lines(_cross(f, v), 0, np.array([22.0, 24.0]))
    assert two[0] == pytest.approx(read[lines == 22.0][0], rel=1e-12)


def test_a_curve_starting_on_a_bin_edge_reaches_no_bin_below_it():
    read = record_on_lines(_cross([21.5, 2000.0], [0.004 + 0j, 0.004 + 0j]), 0,
                           np.array([20.0, 21.0, 22.0]))
    assert np.isnan(read[:2]).all()
    assert read[2] == pytest.approx(0.004)
    flat = _spec([21.5, 2000.0], [0.01, 0.01])
    assert np.isnan(record_on_lines(flat, 0, np.array([20.0, 21.0, 22.0]))[:2]).all()
    ending = record_on_lines(_cross([21.5, 2000.5], [0.004 + 0j, 0.004 + 0j]), 0,
                             np.array([1999.0, 2000.0, 2001.0]))
    assert np.isnan(ending[2]), 'its bin starts where the curve ends'


def test_a_band_step_off_the_knees_is_kept_when_the_lines_fold():
    """A limit stepping at 1000 Hz, on the plateau between laws, needs a
    point there to hold the step; a step at a knee needs none."""
    sheet = _sheet(*SLOPED)
    bands = {**sheet.bands[0], 'uniform': False}
    bands['abort'] = [(-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)]
    stepped = dataclasses.replace(sheet, frequencies=[20.0, 80.0, 350.0, 1000.0, 2000.0],
                                  levels=[[0.01, 0.04, 0.04,
                                           0.04 * (1000 / 350) ** (np.log(0.007 / 0.04) / np.log(2000 / 350)),
                                           0.007]],
                                  bands=[{**bands, 'warning': [(-3.0, 3.0)] * 4,
                                          'abort': [(-6.0, 6.0)] * 3 + [(-9.0, 9.0)]}])
    lines = stepped.to_lines(1.0)
    folded = lines.to_breakpoints()
    assert folded.frequencies == [20.0, 80.0, 350.0, 1000.0, 2000.0]
    assert folded.bands[0]['abort'][-1] == pytest.approx((-9.0, 9.0))
    assert folded.bands[0]['abort'][-2] == pytest.approx((-6.0, 6.0))
    at_knee = dataclasses.replace(stepped, frequencies=SLOPED[0], levels=[list(SLOPED[1])],
                                  bands=[{**bands, 'warning': [(-3.0, 3.0)] * 3,
                                          'abort': [(-6.0, 6.0)] * 2 + [(-9.0, 9.0)]}])
    assert at_knee.to_lines(1.0).to_breakpoints().frequencies == SLOPED[0]


def _pair():
    draft = SpecificationDraft(['1Z+', '2Z+'], ['acceleration'] * 2, list(FLAT[0]),
                               [list(FLAT[1]), [0.02, 0.02]], {(0, 1): None})
    return draft


def test_a_sheet_on_lines_is_not_merged_into_a_curve():
    """Two channels of a curve, one opened and moved to lines, written back
    by itself: the object would hold both grids, read as a curve."""
    import visualdynamics as vd

    project = vd.Project()
    project['s'] = _pair().make('x')
    one = SpecificationDraft.from_specification(project['s'], channels=['1Z+'])
    with pytest.raises(ValueError, match='the sheet is on lines and the '
                       'specification is a curve at breakpoints'):
        project.author_specification('s', one.to_lines(1.0), replace=True)
    assert len(project['s'].abscissa) == 2, 'left as it was'
    # the whole of it moves form, and a part on its lines merges at them
    whole = SpecificationDraft.from_specification(project['s']).to_lines(1.0)
    project.author_specification('s', whole, replace=True)
    assert project['s'].interpolation == 'bin'
    part = SpecificationDraft.from_specification(project['s'], channels=['1Z+'])
    project.author_specification('s', part.scaled_db(1.0), replace=True)
    assert project['s'].ordinate[0][5].real == pytest.approx(0.01 * 10 ** 0.1)


def test_an_older_curve_on_lines_still_takes_a_part_of_its_sheet():
    """Written before the sheet made densities per line: its own lines,
    marked a curve."""
    import visualdynamics as vd

    old = _pair().to_lines(1.0)._copy(form='breakpoints').make('x')
    assert old.interpolation == 'log_log' and len(old.abscissa) > 8
    project = vd.Project()
    project['s'] = old
    part = SpecificationDraft.from_specification(project['s'], channels=['1Z+'])
    assert part.form == 'lines'
    project.author_specification('s', part.scaled_db(1.0), replace=True)
    assert project['s'].ordinate[0][5].real == pytest.approx(0.01 * 10 ** 0.1)
