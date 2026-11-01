"""A shape set at a channel table's channels, as a modal state-space model.

The model answers forces at the table's reference rows with every row of
the table, in table order:

    ẋ = A x + B u
    y = C x + D u

Each mode r of the set gives two states, its modal displacement and
velocity, `[q_r, q̇_r]`, in a 2×2 block of A:

    A_r = [[0, 1], [-ω_r², -2 ζ_r ω_r]]      B_r = [0; φ_in,r / m_r]

with ω_r the mode's frequency in rad/s, ζ_r its damping and m_r its
modal mass. A response row reads the modes at its DOF, φ_out, as its
channel type asks:

    displacement   C = [φ_out, 0]                 D = 0
    velocity       C = [0, φ_out]                 D = 0
    acceleration   C = [-ω_r² φ_out, -2 ζ_r ω_r φ_out]
                   D = Σ_r φ_out φ_in / m_r

so C (iωI − A)⁻¹ B + D is the set's own synthesized FRF
(`ShapeSet.synthesize_frf`) at the same DOFs. A reference row reads its
own input: its C row is zero and its D row is that input's unit vector,
as a force gauge reads the force applied.

Two changes are made to the set as it stands, and the model says how
many modes each touched:

- Every mode at or above `maximum_frequency` is dropped, its states and
  its share of D both. A controller that integrates the model at its
  sample rate and keeps every n-th sample, as Rattlesnake's state-space
  hardware does, has no filter before that decimation, so a mode above
  the Nyquist frequency aliases into the band.
- Every mode below `rigid_frequency` is moved up to it, keeping its
  damping. A rigid-body mode at 0 Hz is a double integrator, and its
  displacement state drifts without bound through a long random run. At
  1 Hz the accelerance above a few hertz is unchanged.

Each output row is written in its channel's unit and each input column
in its reference row's force unit, so the model reads the controller's
channel table directly: no sensitivity is applied to a state-space
channel. A blank unit cell takes the unit system's unit for the
channel's type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..units import SI, UnitSystem, si_factor
from .channel_table import ChannelTable
from .shapes import ShapeSet, _split_sign

#: the channel types a mode can be read as, and the power of iω each is
#: the displacement multiplied by
OUTPUT_POWER = {'length': 0, 'velocity': 1, 'acceleration': 2}


@dataclass
class StateSpaceModel:
    """The matrices of a modal state-space model and what they hold.

    Attributes:
        A: `(2n, 2n)` for n modes, a 2×2 block per mode.
        B: `(2n, inputs)`, a column per reference row.
        C: `(channels, 2n)`, a row per channel in table order.
        D: `(channels, inputs)`.
        channel_dof: Each row's DOF, as the table names it.
        input_dof: Each input's DOF, the reference rows in table order.
        channel_unit: The unit each row is in, which for a reference row
            is also its input's unit.
        mode_frequency: Each kept mode's frequency in Hz, after lifting.
        lifted: How many modes were moved up to the rigid frequency.
        dropped: How many modes were at or above the maximum frequency.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    channel_dof: list[str]
    input_dof: list[str]
    channel_unit: list[str]
    mode_frequency: np.ndarray
    lifted: int
    dropped: int

    def summary(self) -> str:
        """One line on what the model holds, for a status line."""
        said = (f'{len(self.mode_frequency)} modes, '
                f'{len(self.input_dof)} inputs, '
                f'{len(self.channel_dof)} channels')
        if self.lifted:
            said += (f'; {self.lifted} mode{"s" * (self.lifted != 1)} '
                     f'moved up to {self.mode_frequency.min():.4g} Hz')
        if self.dropped:
            said += (f'; {self.dropped} mode{"s" * (self.dropped != 1)} '
                     'above the maximum frequency left out')
        units = ', '.join(dict.fromkeys(self.channel_unit))
        return f'{said}; written in {units}'

    def frf(self, frequencies) -> np.ndarray:
        """C (iωI − A)⁻¹ B + D at each frequency, in Hz: `(lines,
        channels, inputs)`, in the rows' and inputs' units.

        The model's own FRF, evaluated from the matrices rather than the
        set, so a preview shows what the file holds."""
        eye = np.eye(self.A.shape[0])
        return np.array([self.C @ np.linalg.solve(2j * np.pi * f * eye - self.A, self.B)
                         + self.D for f in np.asarray(frequencies, dtype=float)])


def state_space_model(shapes: ShapeSet, table: ChannelTable, *,
                      maximum_frequency: float | None = None,
                      rigid_frequency: float = 1.0,
                      unit_system: UnitSystem | None = None
                      ) -> StateSpaceModel:
    """The modal state-space model of `shapes` at `table`'s channels.

    Parameters
    ----------
    shapes : ShapeSet
        Real, mass-normalized shapes with a declared mass unit, covering
        every channel's DOF.
    table : ChannelTable
        The channels. Its reference rows are the inputs, each a force,
        and every row is an output, in table order.
    maximum_frequency : float, optional
        Modes at or above this frequency, in Hz, are left out. Every
        mode is kept when omitted. For a controller, its Nyquist
        frequency.
    rigid_frequency : float, default 1.0
        Modes below this frequency, in Hz, are moved up to it.
    unit_system : UnitSystem, optional
        The units for a channel whose unit cell is blank. SI when
        omitted.

    Returns
    -------
    StateSpaceModel
        The matrices, with the rows' DOFs and units and the modes kept.

    Raises
    ------
    ValueError
        If the set is complex, has no mass unit or no drive-point scale;
        if the table has no reference row, a reference row that is not
        a force, a channel with no node or direction, or a response
        whose type is not a displacement, velocity or acceleration; if
        the set has no coefficient at a channel's DOF; or if the maximum
        frequency leaves out every mode.
    """
    if shapes.is_complex or np.iscomplexobj(shapes.modal_mass):
        raise ValueError('the shape set is complex; a state-space model '
                         'here is built from real modes')
    if not shapes.units_defined:
        raise ValueError('the shape set has no mass unit, so the model '
                         'would have no physical scale; declare it first')
    if shapes.unscaled:
        raise ValueError('the shape set was fitted with no drive point, so '
                         'its modal masses are a convention, not a scale')
    unit_system = unit_system or SI
    dofs, kinds, roles = table.dof_strings(), table.types(), table.roles()
    numbers = [int(n) for n in table.frame['channel']]
    inputs = [row for row, role in enumerate(roles) if role == 'reference']
    if not inputs:
        raise ValueError('the table has no reference rows, so the model '
                         'has no inputs; mark the drives as references')
    units = []
    for row, (dof, kind, role) in enumerate(zip(dofs, kinds, roles)):
        node, direction = table.frame.loc[row, ['node', 'direction']]
        if str(node).strip() == '' or str(direction).strip() == '':
            raise ValueError(f'channel {numbers[row]} has no node or '
                             'direction, so no DOF to read the modes at')
        if role == 'reference' and kind != 'force':
            raise ValueError(
                f'channel {numbers[row]} is a reference but measures '
                f'{kind or "nothing declared"}; an input here is a force')
        if role != 'reference' and kind not in OUTPUT_POWER:
            raise ValueError(
                f'channel {numbers[row]} is a {kind or "channel of no type"}'
                ', which the modes do not give; a response here is a '
                'displacement, velocity or acceleration')
        unit = str(table.frame.loc[row, 'unit']).strip()
        units.append(unit or unit_system.unit(kind))

    keep = np.ones(shapes.num_shapes, dtype=bool)
    if maximum_frequency is not None:
        keep = shapes.frequency < float(maximum_frequency)
        if not keep.any():
            raise ValueError(
                f'a maximum of {maximum_frequency:g} Hz drops every mode; '
                f'the lowest is at {shapes.frequency.min():.4g} Hz')
    frequency = shapes.frequency[keep]
    lifted = int(np.count_nonzero(frequency < rigid_frequency))
    frequency = np.maximum(frequency, float(rigid_frequency))
    omega = 2.0 * np.pi * frequency
    zeta = shapes.damping[keep]
    mass = shapes.modal_mass[keep].astype(np.float64)
    lookup = shapes._dof_lookup()

    def phi(dof):
        base, sign = _split_sign(dof)
        if base not in lookup:
            raise ValueError(f'the shape set has no coefficient at {base}, '
                             'a DOF the table names')
        column, stored = lookup[base]
        return shapes.shape_matrix[keep, column].real * (sign * stored)

    n = len(frequency)
    a = np.zeros((2 * n, 2 * n))
    b = np.zeros((2 * n, len(inputs)))
    c = np.zeros((len(dofs), 2 * n))
    d = np.zeros((len(dofs), len(inputs)))
    position, velocity = np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)
    a[position, velocity] = 1.0
    a[velocity, position] = -omega ** 2
    a[velocity, velocity] = -2.0 * zeta * omega
    # the stored shapes are SI, so the model is first built in N in and
    # m, m/s or m/s² out; each column then takes its force unit and each
    # row its own unit
    into = np.array([si_factor(units[row], 'force') for row in inputs])
    for column, row in enumerate(inputs):
        b[velocity, column] = phi(dofs[row]) / mass * into[column]
    for row, (dof, kind, role) in enumerate(zip(dofs, kinds, roles)):
        if role == 'reference':
            d[row, inputs.index(row)] = 1.0
            continue
        out = phi(dof) / si_factor(units[row], kind)
        power = OUTPUT_POWER[kind]
        if power == 0:
            c[row, position] = out
        elif power == 1:
            c[row, velocity] = out
        else:
            c[row, position] = -omega ** 2 * out
            c[row, velocity] = -2.0 * zeta * omega * out
            d[row] = [out @ (phi(dofs[i]) / mass) * into[k]
                      for k, i in enumerate(inputs)]
    return StateSpaceModel(a, b, c, d, channel_dof=list(dofs),
                           input_dof=[dofs[row] for row in inputs],
                           channel_unit=units, mode_frequency=frequency,
                           lifted=lifted, dropped=int(np.count_nonzero(~keep)))
