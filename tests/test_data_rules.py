"""Rules in `core.data` pinned by value where a mutation pass found them
reached only through the window, or not at all (2026-10-10).

`tools/mutate.py` over data.py against its nineteen core test files
left these unguarded: which records a selector deletes, the principal
shape of a cross-spectral matrix, the band an SRS can support, the
default filter, how captures are numbered, and the numbers a refused
abscissa quotes back. Each is stated here as the rule it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import Frf, Psd, Specification, TimeHistory
from visualdynamics.core.filters import Filtering
from visualdynamics.core.shocks import Shock

RATE = 1000.0


def _frf():
    """Rows 1X+, 1X+, 2X+ against references 9X+, 8X+, 9X+; the second
    record a force per force, the others accelerations per force."""
    f = np.linspace(0.0, 100.0, 11)
    return Frf(f, np.ones((3, f.size), dtype=complex),
               response_dof=['1X+', '1X+', '2X+'],
               reference_dof=['9X+', '8X+', '9X+'],
               ordinate_dim=['acceleration/force', 'force/force',
                             'acceleration/force'])


def _left(data):
    return list(zip(data.response_dof, data.reference_dof))


@pytest.mark.parametrize('selector, kept', [
    ({'dof': '1X+'}, [('2X+', '9X+')]),
    ({'dof': ['2X+']}, [('1X+', '9X+'), ('1X+', '8X+')]),
    ({'reference': '8X+'}, [('1X+', '9X+'), ('2X+', '9X+')]),
    ({'dof': '1X+', 'dim': 'force/force'}, [('1X+', '9X+'), ('2X+', '9X+')]),
    ({'dof': '1X+', 'reference': '9X+'}, [('1X+', '8X+'), ('2X+', '9X+')]),
    ({'indices': [2], 'dof': '1X+', 'dim': 'force/force'}, [('1X+', '9X+')]),
])
def test_selectors_intersect_and_indices_add(selector, kept):
    """A DOF, a quantity and a reference narrow one another; explicit
    indices are added to whatever they chose."""
    frf = _frf()
    frf.delete_records(**selector)
    assert _left(frf) == kept


def test_a_selection_that_matches_nothing_or_everything_is_refused():
    frf = _frf()
    with pytest.raises(ValueError, match='nothing matches'):
        frf.delete_records(dof='7X+')
    with pytest.raises(IndexError, match=r'no such records: \[3\]'):
        frf.delete_records(indices=[3])
    with pytest.raises(ValueError, match='delete the object instead'):
        frf.delete_records(dim=['acceleration/force', 'force/force'])
    assert len(frf.response_dof) == 3, 'a refusal changes nothing'
    psd = Psd(np.linspace(1.0, 10.0, 5), np.ones((2, 5)),
              response_dof=['1X+', '2X+'])
    with pytest.raises(ValueError, match="references are a matrix's"):
        psd.delete_records(reference='1X+')
    with pytest.raises(ValueError, match="captures are a time history's"):
        psd.delete_records(capture=1)


def test_a_capture_is_each_channels_nth_playing():
    t = np.arange(8) / RATE
    history = TimeHistory(t, np.ones((4, 8)),
                          response_dof=['1X+', '1X+', '2X+', '1X+'],
                          ordinate_dim='acceleration')
    assert history.capture_indices() == [0, 1, 0, 2]
    history.delete_records(capture=1)
    assert history.response_dof == ['1X+', '2X+', '1X+']


def test_a_specifications_limits_go_with_its_records():
    f = np.linspace(1.0, 10.0, 5)
    spec = Specification(f, np.ones((2, 5)), response_dof=['1X+', '2X+'],
                         abort_upper=np.array([np.full(5, 2.0), np.full(5, 3.0)]))
    spec.delete_records(dof='1X+')
    assert spec.limits['abort_upper'].tolist() == [[3.0] * 5]


def _cpsd(vector, power):
    """The cross-spectral matrix of one coherent source: G_ij = v_i
    conj(v_j) p at every line, written record by record."""
    f = np.linspace(1.0, 10.0, 4)
    dofs = ['1X+', '2X+']
    rows, response, reference = [], [], []
    for i, a in enumerate(dofs):
        for j, b in enumerate(dofs):
            rows.append(vector[i] * np.conj(vector[j]) * power)
            response.append(a)
            reference.append(b)
    return Psd(f, np.array(rows), response_dof=response, reference_dof=reference,
               ordinate_dim='acceleration**2/frequency')


def test_the_principal_shape_of_one_coherent_source_is_that_source():
    """Rank one, so the largest eigenvector is the source itself, its
    length the square root of the power — up to the phase an
    eigenvector is free to take, so the ratio between channels is what
    is pinned."""
    vector, power = np.array([1.0, 2.0j]), np.array([4.0, 9.0, 16.0, 25.0])
    dofs, shapes, quantity = _cpsd(vector, power).principal_shapes()
    assert dofs == ['1X+', '2X+'] and quantity == 'acceleration'
    assert np.allclose(np.abs(shapes[0]), np.sqrt(power))
    assert np.allclose(np.abs(shapes[1]), 2.0 * np.sqrt(power))
    assert np.allclose(shapes[1] / shapes[0], 2.0j)


def test_a_principal_shape_needs_a_whole_cross_block_of_two_or_more():
    f = np.linspace(1.0, 10.0, 4)
    autos = Psd(f, np.ones((2, 4)), response_dof=['1X+', '2X+'],
                ordinate_dim='acceleration**2/frequency')
    with pytest.raises(ValueError, match='no cross records'):
        autos.principal_shapes()
    holed = _cpsd(np.array([1.0, 1.0]), np.ones(4))
    holed.delete_records(indices=[1])            # 1X+ against 2X+
    with pytest.raises(ValueError, match='block is incomplete'):
        holed.principal_shapes()
    one = Psd(f, np.ones((1, 4), dtype=complex), response_dof=['1X+'],
              reference_dof=['1X+'], ordinate_dim='acceleration**2/frequency')
    with pytest.raises(ValueError, match='fewer than two acceleration channels'):
        one.principal_shapes()


def test_an_srs_band_runs_from_the_shortest_window_to_a_fifth_of_the_rate():
    """Low enough that the shortest window holds a cycle — never below
    1 Hz — and up to a fifth of the rate, where the filter still
    resolves."""
    t = np.arange(int(4 * RATE)) / RATE
    history = TimeHistory(t, np.zeros((1, t.size)), response_dof=['1X+'],
                          ordinate_dim='acceleration')
    low, high = history.srs_band([Shock(0.5, 0.25), Shock(2.0, 0.5)])
    assert (low, high) == (pytest.approx(4.0), pytest.approx(RATE / 5.0))
    low, _high = history.srs_band([Shock(0.0, 3.0)])
    assert low == pytest.approx(1.0), 'a three-second window still starts at 1 Hz'


def test_the_starting_filter_is_a_low_pass_at_a_tenth_of_the_rate():
    t = np.arange(100) / RATE
    history = TimeHistory(t, np.zeros((1, 100)), response_dof=['1X+'])
    assert history.suggest_filtering() == Filtering(high=RATE / 10.0)


def test_a_refused_abscissa_quotes_the_samples_that_broke_it():
    out_of_order = TimeHistory(np.array([0.0, 0.1, 0.3, 0.2, 0.4]),
                               np.zeros((1, 5)), response_dof=['1X+'])
    with pytest.raises(ValueError, match=r'sample 3 is at 0\.2 s, after 0\.3'):
        _ = out_of_order.sample_rate
    uneven = TimeHistory(np.array([0.0, 0.1, 0.2, 0.35, 0.4]),
                         np.zeros((1, 5)), response_dof=['1X+'])
    with pytest.raises(ValueError, match=r'the step varies by 0\.05 s'):
        _ = uneven.sample_rate
