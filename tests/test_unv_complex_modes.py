"""Complex modes through a universal file: dataset 55, analysis type 3.

A curve fitter exports complex modes as dataset 55 of analysis type 3,
the complex eigenvalue — the eigenvalue itself, modal A and modal B —
where a normal mode (type 2) carries frequency, modal mass and damping.
Until 2026-09-23 only type 2 was read, so such a file imported nothing,
and a complex set was *written* as a type 2 with complex values, which
no other reader looks for.

Checked three ways: against a file sdynpy wrote and sdynpy's own reading
of it (frozen by `generate_unv_complex_oracle.py` in the private
generators repository — data, never an import, `AGENTS.md` hard rule 1);
by round trip, complex modal A and modal B included; and end to end
against physics, a non-proportionally damped system's modes written,
read and synthesized back to the exact inverse of its dynamic stiffness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import fixture_path
from test_frf_synthesis_exact import _complex_modes, _exact, _non_proportional

import visualdynamics
from visualdynamics.core.shapes import ShapeSet

WRITTEN_BY_SDYNPY = fixture_path('sdynpy_oracle', 'complex_modes.unv')
SDYNPY_READING = fixture_path('sdynpy_oracle', 'complex_modes.npz')


@pytest.fixture(scope='module')
def oracle():
    with np.load(SDYNPY_READING) as frozen:
        return {key: frozen[key] for key in frozen.files}


def _definition_records(path):
    """Each dataset 55's record 6, as integers."""
    lines = Path(path).read_text().splitlines()
    return [[int(v) for v in lines[i + 7].split()]
            for i, line in enumerate(lines[:-1])
            if line.strip() == '-1' and lines[i + 1].strip() == '55']


def test_a_file_sdynpy_wrote_reads_as_sdynpy_reads_it(oracle):
    shapes = visualdynamics.import_file(WRITTEN_BY_SDYNPY)
    assert shapes.is_complex
    assert list(shapes.coordinate) == [str(d) for d in oracle['dofs']]
    assert np.abs(shapes.shape_matrix - oracle['shape_matrix']).max() < 1e-12
    assert np.allclose(shapes.frequency, oracle['frequency'], rtol=1e-12)
    assert np.allclose(shapes.damping, oracle['damping'], rtol=1e-12)
    assert np.allclose(shapes.modal_mass, oracle['modal_mass'], rtol=1e-12)


def test_and_synthesizes_the_frfs_sdynpy_does_from_it(oracle):
    """The chain a user runs: a third party's file, imported, then
    resynthesized."""
    shapes = visualdynamics.import_file(WRITTEN_BY_SDYNPY)
    pairs = oracle['coordinate_pairs'].reshape(-1, 2)
    got = shapes.synthesize_frf(oracle['lines'], [str(p[0]) for p in pairs],
                                [str(p[1]) for p in pairs], power=2)
    want = oracle['frf_acceleration'].reshape(-1, len(oracle['lines']))
    worst = np.abs(got - want).max() / np.abs(want).max()
    assert worst < 1e-12, f'departs from sdynpy by {worst:.2e} of peak'


def test_the_characteristic_decides_how_many_values_a_node_has():
    """sdynpy writes 1 in the values-per-node field of a 3-DOF vector.
    Trusting the field read X alone at every node and dropped Y and Z
    without a word; the characteristic (2, a 3-DOF translation) is what
    says a node has three."""
    definitions = _definition_records(WRITTEN_BY_SDYNPY)
    assert definitions and all(d[1] == 3 for d in definitions), 'type 3'
    assert all(d[2] == 2 and d[5] == 1 for d in definitions), (
        'the fixture still carries the miscount this test is about')
    assert visualdynamics.import_file(WRITTEN_BY_SDYNPY).num_dofs == 9


def _complex_set():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(2, 6)) + 1j * rng.normal(size=(2, 6))
    dofs = [f'{n}{d}' for n in (1, 2) for d in ('X+', 'Y+', 'Z+')]
    return ShapeSet([22.0, 61.0], [0.02, 0.07], dofs, matrix,
                    modal_mass=[0.4 + 1.3j, -2.1 + 0.2j],
                    modal_damping=[5.0 - 3.0j, 1.5 + 8.0j])


def test_a_complex_set_is_written_as_type_3_and_comes_back_whole(tmp_path):
    source = _complex_set()
    path = str(tmp_path / 'complex.unv')
    visualdynamics.export_file(source, path, format='unv')
    assert all(d[1] == 3 for d in _definition_records(path)), (
        'a complex set goes out as the complex eigenvalue, analysis type 3')
    back = visualdynamics.import_file(path)
    assert back.is_complex
    assert np.allclose(back.frequency, source.frequency, rtol=1e-5)
    assert np.allclose(back.damping, source.damping, rtol=1e-4)
    assert np.allclose(back.modal_mass, source.modal_mass, rtol=1e-5), (
        'complex modal A, imaginary part and all')
    assert np.allclose(back.modal_damping, source.modal_damping, rtol=1e-5)
    assert np.allclose(back.shape_matrix, source.shape_matrix, rtol=1e-4)


def test_modal_b_is_written_from_the_pole_when_a_set_has_none(tmp_path):
    """For the first-order problem the eigenvalue is -B / A, so a set
    with no modal B of its own is written B = -l A: a zero there would
    say the pole sits at the origin."""
    source = _complex_set()
    source.modal_damping = None
    path = str(tmp_path / 'no_b.unv')
    visualdynamics.export_file(source, path, format='unv')
    back = visualdynamics.import_file(path)
    omega = 2.0 * np.pi * source.frequency
    pole = -source.damping * omega + 1j * omega * np.sqrt(1 - source.damping ** 2)
    assert np.allclose(-back.modal_damping / back.modal_mass, pole, rtol=1e-5)


def test_modes_written_and_read_back_are_still_the_structure(tmp_path):
    """End to end against physics: the complex modes of a system damped
    at one mass only, written to a universal file, read back, and
    synthesized. The form itself is exact
    (`tests/test_frf_synthesis_exact.py`); what the file costs is its six
    significant figures. A pole moved by 5e-6 of itself shifts a
    resonance by that over twice the damping ratio, of the peak — 6e-4
    at this system's lightest mode, 0.39% — hence 1e-3 here. The
    conventions this guards (modal A for modal mass, the sign of the
    eigenvalue's real part) are each worth about 100% when wrong."""
    m, k, c = _non_proportional()
    poles, psi, modal_a = _complex_modes(m, k, c)
    dofs = ['1X+', '2X+', '3X+']
    source = ShapeSet(np.abs(poles) / (2 * np.pi), -poles.real / np.abs(poles),
                      dofs, psi.T, modal_mass=modal_a)
    path = str(tmp_path / 'structure.unv')
    visualdynamics.export_file(source, path, format='unv')
    back = visualdynamics.import_file(path)
    lines = np.linspace(0.5, 1.3 * np.abs(poles).max() / (2 * np.pi), 1500)
    exact = _exact(k, m, c, 2 * np.pi * lines)
    pairs = [(j, r) for j in dofs for r in dofs]
    got = back.synthesize_frf(lines, [p[0] for p in pairs],
                              [p[1] for p in pairs])
    want = np.array([exact[:, dofs.index(j), dofs.index(r)] for j, r in pairs])
    worst = np.abs(got - want).max() / np.abs(want).max()
    assert worst < 1e-3, f'departs from the exact inverse by {worst:.1e}'


def test_a_file_mixing_normal_and_complex_modes_is_refused(tmp_path):
    """The two carry different scalings, a normal mode's modal mass and
    a complex mode's modal A, and one set holds one of them."""
    real = ShapeSet([10.0], [0.01], ['1X+', '1Y+', '1Z+'], [[1.0, 0.5, 0.2]])
    real_path, complex_path = tmp_path / 'r.unv', tmp_path / 'c.unv'
    visualdynamics.export_file(real, str(real_path), format='unv')
    visualdynamics.export_file(_complex_set(), str(complex_path), format='unv')
    mixed = tmp_path / 'mixed.unv'
    mixed.write_text(real_path.read_text() + complex_path.read_text())
    with pytest.raises(ValueError, match='mixes normal modes'):
        visualdynamics.import_file(str(mixed))
