"""Write a modal state-space model as Rattlesnake's state-space hardware file.

The controller's state-space virtual hardware loads a NumPy `.npz` (or
MATLAB `.mat`) holding `A`, `B`, `C` and `D`, integrates the model at its
sample rate, and returns `C x + D u` as each channel's value. It checks
that `C` and `D` have a row per channel of its channel table and that `B`
and `D` have a column per channel with a feedback device, the drives, so
the model is written for one table and matches that table row for row
(`core.state_space`).

The file also holds `channel_dof`, `input_dof`, `channel_unit` and
`mode_frequency`, which the controller ignores. They say what the rows,
the columns and the states are, since the file has no other place for
units.
"""

from __future__ import annotations

import os

import numpy as np

from ..core.state_space import StateSpaceModel


def save(model: StateSpaceModel, path: str | os.PathLike) -> None:
    """Write `model` to `path` as an uncompressed `.npz`.

    Parameters
    ----------
    model : StateSpaceModel
        The model to write.
    path : str or os.PathLike
        Where to write it.

    Returns
    -------
    None
    """
    np.savez(path, A=model.A, B=model.B, C=model.C, D=model.D,
             channel_dof=np.array(model.channel_dof, dtype=str),
             input_dof=np.array(model.input_dof, dtype=str),
             channel_unit=np.array(model.channel_unit, dtype=str),
             mode_frequency=np.asarray(model.mode_frequency, dtype=float))
