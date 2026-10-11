"""Matched modes' row refusals and renames, pinned where a mutation
pass found them unguarded (2026-10-11)."""

from __future__ import annotations

import pytest

from visualdynamics.core.matches import MatchedModes


def _matches():
    return MatchedModes('Test', 'FEM', [[0, 0], [1, 2]], [0.99, 0.95],
                        first_geometry='Test Geometry', second_geometry='FEM Geometry')


def test_a_delete_names_the_first_row_that_is_not_there():
    matches = _matches()
    with pytest.raises(ValueError, match='no match at row 2'):
        matches.delete_matches([0, 2])
    with pytest.raises(ValueError, match='no match at row -1'):
        matches.delete_matches([-1])
    matches.delete_matches([1])
    assert matches.pairs == [[0, 0]]


def test_a_renamed_geometry_is_followed_and_no_other_is():
    matches = _matches()
    matches.rename_source('FEM Geometry', 'Model Geometry')
    assert (matches.first_geometry, matches.second_geometry) == (
        'Test Geometry', 'Model Geometry')
    matches.rename_source('Test Geometry', 'Rig')
    assert (matches.first_geometry, matches.second_geometry) == ('Rig', 'Model Geometry')
