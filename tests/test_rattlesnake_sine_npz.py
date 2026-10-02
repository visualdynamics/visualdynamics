"""Rattlesnake's sine specification files (.npz): one tone per file,
written by the controller's own `save_specification`, merged into the
one specification the sine report reads (Brandon, 2026-10-02: nine
files refused as 'an npz archive holding 5 arrays').
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics import io
from visualdynamics.core.merge import merge, mergeable
from visualdynamics.core.sine import SineSweepSpecification


def _write(path, *, name='Sweep 3', with_extras=True, transposed=True,
           channels=3):
    """What the controller writes: `amplitude` as channels by
    breakpoints (the table's column transposed), the segment arrays
    one short of the breakpoints, and the optional arrays when the
    writing version had them."""
    frequency = np.array([20.0, 200.0, 2000.0])
    amplitude = np.array([[1.0, 2.0, 4.0]] * channels)        # (channels, breakpoints)
    data = {'frequency': frequency,
            'amplitude': amplitude if transposed else amplitude.T,
            'sweep_type': np.array([1, 0]), 'sweep_rate': np.array([2.0, 100.0])}
    if with_extras:
        warning = np.full((3, 2, 2, channels), np.nan)
        warning[:, 1, :, :] = (amplitude.T * 1.5)[:, None, :]
        warning[:, 0, :, :] = (amplitude.T * 0.5)[:, None, :]
        data.update({'phase': np.zeros((channels, 3)),
                     'warning': np.transpose(warning, (1, 2, 3, 0)),
                     'abort': np.transpose(warning * 1.2, (1, 2, 3, 0)),
                     'start_time': np.float64(12.5), 'name': np.array(name)})
    np.savez(path, **data)
    return str(path)


def test_the_five_array_file_imports_as_one_tone(tmp_path):
    path = _write(tmp_path / 'sweep3.npz', with_extras=False)
    assert io.rattlesnake_sine.sniff(path)
    spec = visualdynamics.import_file(path)
    assert isinstance(spec, SineSweepSpecification)
    assert spec.response_dof == ['1', '2', '3'], 'the controller\'s channel order'
    assert spec.ordinate_unit is None, 'the file carries no unit'
    (tone,) = spec.tones
    assert tone.name == 'sweep3', 'named by the file when the file has no name'
    assert tone.start_time == 0.0
    assert tone.frequency.tolist() == [20.0, 200.0, 2000.0]
    assert tone.amplitude.shape == (3, 3) and tone.amplitude[:, 0].tolist() == [1.0, 2.0, 4.0]
    assert tone.segment_type.tolist() == [1, 0] and tone.segment_rate.tolist() == [2.0, 100.0]
    assert tone.limits == {}


def test_the_full_file_brings_its_name_start_phase_and_bands(tmp_path):
    path = _write(tmp_path / 'full.npz')
    spec = visualdynamics.import_file(path, response_dof=['101Z+', '102Z+', '103Z+'],
                                      ordinate_unit='g')
    (tone,) = spec.tones
    assert tone.name == 'Sweep 3' and tone.start_time == 12.5
    assert spec.response_dof == ['101Z+', '102Z+', '103Z+'] and spec.ordinate_unit == 'g'
    assert tone.phase.shape == (3, 3) and np.all(tone.phase == 0.0)
    assert tone.limits['warning_upper'][:, 0].tolist() == [1.5, 3.0, 6.0]
    assert tone.limits['abort_lower'][:, 2].tolist() == pytest.approx([0.6, 1.2, 2.4])


def test_an_untransposed_amplitude_is_read_by_its_shape(tmp_path):
    spec = visualdynamics.import_file(_write(tmp_path / 't.npz', transposed=False, channels=2))
    assert spec.tones[0].amplitude.shape == (3, 2) and spec.response_dof == ['1', '2']


def test_the_random_specification_file_is_not_claimed(tmp_path):
    path = tmp_path / 'random.npz'
    np.savez(path, f=np.linspace(10, 2000, 5), cpsd=np.ones((5, 2, 2)),
             frequency=np.zeros(1), amplitude=np.zeros(1), sweep_type=np.zeros(1),
             sweep_rate=np.zeros(1))
    assert not io.rattlesnake_sine.sniff(path)


def test_nine_files_merge_into_one_specification(tmp_path):
    specs = [visualdynamics.import_file(_write(tmp_path / f's{k}.npz', name=f'Sweep {k}'))
             for k in range(9)]
    assert mergeable(specs) is None
    merged = merge(specs)
    assert isinstance(merged, SineSweepSpecification)
    assert [t.name for t in merged.tones] == [f'Sweep {k}' for k in range(9)]
    assert merged.response_dof == ['1', '2', '3']
    project = visualdynamics.Project()
    names = [project.add(f'Sweep {k}', spec) for k, spec in enumerate(specs[:3])]
    one = project.merge(*names, name='Sine Specification')
    assert [t.name for t in project[one].tones] == ['Sweep 0', 'Sweep 1', 'Sweep 2']
    assert not any(n in project for n in names)


def test_merging_refuses_different_channels_and_repeated_names(tmp_path):
    a = visualdynamics.import_file(_write(tmp_path / 'a.npz', name='A'))
    b = visualdynamics.import_file(_write(tmp_path / 'b.npz', name='A'))
    assert 'repeat' in mergeable([a, b])
    c = visualdynamics.import_file(_write(tmp_path / 'c.npz', name='C', channels=2))
    assert 'different control channels' in mergeable([a, c])
