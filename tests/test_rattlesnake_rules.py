"""A Rattlesnake file's channels in units other than SI, by value
(2026-10-11).

`tools/mutate.py` over rattlesnake.py caught 141 of 377 against its
three test files. Every fixture's channels are in m/s^2 and N, whose
scale to SI is one, so a scale multiplied where it should divide, or
applied to one side of a cross term and not the other, read the same.
Here the plate's random file is relabeled — the first control channel
in g, the second in in/s^2 — and every number that came through it is
held to the SI file's times the factors it should carry.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from conftest import fixture_path

from visualdynamics.io import rattlesnake

G, INCH = 9.80665, 0.0254


@pytest.fixture(scope='module')
def both(tmp_path_factory):
    import netCDF4

    original = fixture_path('plate', 'random.nc4')
    relabeled = tmp_path_factory.mktemp('rattlesnake') / 'relabeled.nc4'
    shutil.copy(original, relabeled)
    with netCDF4.Dataset(relabeled, 'a') as ds:
        unit = ds.groups['channels'].variables['unit']
        unit[0], unit[1] = 'g', 'in/s^2'
    return (rattlesnake.load(original, full_cpsd=True),
            rattlesnake.load(relabeled, full_cpsd=True))


def _record(spec, response, reference):
    pairs = list(zip(spec.response_dof, spec.reference_dof))
    return pairs.index((response, reference))


def test_a_specifications_records_carry_both_channels_scales(both):
    si, scaled = (loaded['Random_specification'] for loaded in both)
    for response, reference, factor in (('101Z+', '101Z+', G * G),
                                        ('104Z+', '104Z+', INCH * INCH),
                                        ('101Z+', '104Z+', G * INCH),
                                        ('104Z+', '110Z+', INCH),
                                        ('110Z+', '110Z+', 1.0)):
        k = _record(si, response, reference)
        assert np.allclose(np.asarray(scaled.ordinate[k]),
                           np.asarray(si.ordinate[k]) * factor), (response, reference)


def test_a_specifications_limits_carry_the_square_of_their_channels_scale(both):
    si, scaled = (loaded['Random_specification'] for loaded in both)
    k = _record(si, '101Z+', '101Z+')
    for name in ('warning_upper', 'abort_lower'):
        written = np.isfinite(si.limits[name][k])
        assert written.any(), name
        assert np.allclose(scaled.limits[name][k][written],
                           si.limits[name][k][written] * G * G), name


def test_a_time_history_carries_its_channels_scale(both):
    si, scaled = (loaded['time_data'] for loaded in both)
    assert np.allclose(np.asarray(scaled.ordinate[0]), np.asarray(si.ordinate[0]) * G)
    assert np.allclose(np.asarray(scaled.ordinate[1]), np.asarray(si.ordinate[1]) * INCH)
    assert np.array_equal(np.asarray(scaled.ordinate[2]), np.asarray(si.ordinate[2]))
