"""The drives a specification asks for, predicted from an FRF before
the test, and what they ask of the amplifiers and the shakers.

A random test is controlled on spectra. The control channels'
spectral density matrix is the drives' passed through the plant,

    S_yy = H S_vv H^H,

at every line, where `H` is the FRF matrix from the drives to the
control channels (one row per control channel, one column per drive,
in g/V or (m/s^2)/V), `S_vv` the drives' spectral density matrix and
`H^H` the conjugate transpose of `H`. Turned round, the drives that
would produce a target `S_ref` are

    S_vv = Z S_ref Z^H,

with `Z` an inverse of `H`. On one drive and one control channel the
matrices are single numbers and this is the drive autospectrum a
controller starts from, `G_vv = G_ref / |H|^2`. With several, `H` is
inverted line by line, and where it is ill-conditioned (two drives that
reach the control channels in nearly the same proportions, or a line
where the plant barely responds) the plain inverse is made of noise, so
it is regularized, in one of two stated ways:

- **Truncated** (the default): the singular values of `H` at a line
  below `tolerance` times that line's largest are dropped, and the
  pseudo-inverse is built from the rest. With the default tolerance of
  one hundredth, a line whose condition number is over 100 loses its
  weakest direction, the target there is met only in the directions
  kept, and the reading says so. On one drive and one channel nothing
  is ever dropped, and the inverse is exactly `1 / H`.
- **Tikhonov**: `Z = (H^H H + lambda I)^(-1) H^H`, with `lambda` the
  stated `weight` times the largest squared singular value of `H`
  anywhere in the band, so the weight has no units and means the same
  for any plant. Where `|H|^2` is large against `lambda` this is the
  inverse; where the plant barely responds it stays small instead of
  growing without limit. On one channel it is `H* / (|H|^2 + lambda)`.

Either way the prediction carries the control channels' response to
the drives it predicts, `H S_vv H^H`, and the lines where that misses
the target by more than `UNMET_DB` are the lines the regularization
gave up, named as bands. Any other channel the FRF holds against the
same drives (a force gauge, a current monitor, an accelerometer on the
article) is predicted the same way, its autospectrum the diagonal of
`H_u S_vv H_u^H`.

The prediction is open loop, made once, before anything is played: it
is what a controller's first drive would be with this FRF as its plant
estimate, and it does not iterate.

What it asks of the rig
-----------------------
Every quantity is a Gaussian random signal with a predicted
autospectrum, so its rms is the root of the area under it
(`compliance.rms`) and its expected largest peak over a run of
`duration` seconds is the rms times the Gaussian peak factor
sqrt(2 ln(2 nu0 T)), with nu0 its own crossing rate by Rice's formula
(`time_statistics.rice_rate` and `expected_largest_peak`), counted both
ways: a rating is reached by a peak of either sign, the drive's limit
and the travel each way alike, so the largest of either sign is the
one judged. A stated
`peak_factor` replaces that, for a controller that clips its drive at
a set multiple of the rms. Each quantity is judged on its peak:

- the drive at each drive, against `drive_limit`, in the drive's unit
  (the volts a source or amplifier input accepts, or the newtons of a
  force-controlled drive);
- for a drive given a `ShakerRating`, its shaker, read at its table
  channel (the acceleration measured at the drive's own DOF, or the one
  control channel of a single-input test): the force, the moving mass
  times the peak acceleration; the velocity, from the table's
  acceleration autospectrum over (2 pi f)^2; and the travel each way
  from the middle of the stroke, from the autospectrum over
  (2 pi f)^4, with the body's motion counted. All three go through
  `core.shaker`'s one rule;
- any other channel given a peak `limits` entry, in that channel's
  unit: a force gauge in newtons, a current monitor in the volts its
  scale gives at the amplifier's current rating.

Each judged quantity names the line that contributes most to its mean
square and the band of lines holding half of it, so the reading can say
which drive or shaker runs out first and at what frequency.

References
----------
The spectral relations between the inputs and outputs of a linear
system with several inputs, and the Gaussian peak statistics, are
Bendat and Piersol's; the singular value decomposition, the
pseudo-inverse and its truncation are Golub and Van Loan's; the
regularized inverse is Tikhonov's, in the form Hansen sets out.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The output spectral density
   matrix of a multiple-input linear system, `S_yy = H S_xx H^H`, and
   the expected largest peak of a stationary Gaussian process.
2. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*, 4th
   ed. Johns Hopkins University Press. The singular value
   decomposition, the pseudo-inverse and its truncation at a relative
   tolerance.
3. Hansen, P. C. (1998). *Rank-Deficient and Discrete Ill-Posed
   Problems: Numerical Aspects of Linear Inversion*. SIAM. Tikhonov
   regularization as a filter on the singular values,
   `sigma / (sigma^2 + lambda)`, beside truncation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .progress import Ticker
from .shaker import ShakerRating

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem
    from .data import Frf, Psd

#: the two inverses
METHODS = ('truncated', 'tikhonov')

#: the share of a line's largest singular value under which the
#: truncated inverse drops a direction: a condition number of 100, past
#: which a one percent error in the plant can become a hundred percent
#: in its inverse. A choice, set per prediction
TRUNCATION = 0.01

#: the Tikhonov weight, as a share of the largest squared singular value
#: in the band: an inverse held to about 30 dB below the plant's peak
#: gain. A choice, set per prediction
TIKHONOV = 1e-3

#: how long the run lasts, in seconds, for the expected largest peak:
#: a minute, a common duration for a random test at full level. The
#: peak grows slowly with it (about 4.9 standard deviations over a
#: minute at a thousand crossings a second either way, 5.7 over an hour)
DURATION = 60.0

#: how far the control channels' predicted autospectrum may sit from
#: the target, in dB, before a line counts as one the inverse did not
#: meet
UNMET_DB = 1.0

#: the share of a quantity's mean square its dominant band holds
DOMINANT_SHARE = 0.5

# The matrix assembly below (`_labels`, `_plant`, `runs`) mirrors the
# FRF matrix builder, the channel labels and the flagged runs proposed
# beside the conditioning check (`matrix_checks.frf_matrix`, `_labels`,
# `flagged_runs`). They stay here until those land: `_plant` keys a
# channel by its declared dimension and refuses one that is not, and
# `runs` gives line indices where `flagged_runs` gives frequencies.


# ---- the inverse ------------------------------------------------------------

def regularized_inverse(plant: Any, *, method: str = 'truncated',
                        tolerance: float = TRUNCATION,
                        weight: float = TIKHONOV,
                        scale: float | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """An inverse of the plant at every line, truncated or Tikhonov.

    `plant` is (lines, rows, columns), one row per control channel and
    one column per drive. The inverse is (lines, columns, rows), built
    from the singular value decomposition at each line: the truncated
    one keeps the singular values of at least `tolerance` times the
    line's largest, and the Tikhonov one filters every singular value
    sigma to sigma / (sigma^2 + lambda), with lambda `weight` times
    `scale`, the largest squared singular value over every line when
    `scale` is not given. A line holding a value that is not a number
    has an inverse of not-a-number; a line that is zero has an inverse
    of zero. The module docstring says how the two differ.

    Parameters
    ----------
    plant : array_like
        The FRF matrix, (lines, rows, columns), complex.
    method : {'truncated', 'tikhonov'}, default 'truncated'
        Which inverse.
    tolerance : float, default `TRUNCATION`
        The truncated inverse's relative threshold, between 0 and 1.
    weight : float, default `TIKHONOV`
        The Tikhonov weight, above zero.
    scale : float, optional
        The squared singular value the weight is a share of.

    Returns
    -------
    tuple of numpy.ndarray
        The inverse, (lines, columns, rows), and the number of
        directions kept at each line: the singular values kept by the
        truncation, or those whose square is at least lambda.

    Raises
    ------
    ValueError
        If `method` is not one of `METHODS`, or the tolerance or the
        weight is out of its range.
    """
    method = str(method)
    if method not in METHODS:
        raise ValueError(f'the inverse is {" or ".join(map(repr, METHODS))}, '
                         f'not {method!r}')
    tolerance = float(tolerance)
    weight = float(weight)
    if method == 'truncated' and not 0.0 < tolerance < 1.0:
        raise ValueError('the truncation tolerance is a share of the '
                         'largest singular value, above 0 and under 1, '
                         f'not {tolerance:g}')
    if method == 'tikhonov' and not (np.isfinite(weight) and weight > 0.0):
        raise ValueError('the Tikhonov weight has to be above zero and '
                         f'finite, not {weight:g}')
    plant = np.asarray(plant, dtype=np.complex128)
    lines, rows, columns = plant.shape
    inverse = np.full((lines, columns, rows), np.nan, dtype=np.complex128)
    kept = np.zeros(lines, dtype=int)
    finite = np.isfinite(plant).all(axis=(1, 2))
    if not finite.any():
        return inverse, kept
    u, sigma, vh = np.linalg.svd(plant[finite], full_matrices=False)
    largest = sigma[:, :1]
    if method == 'truncated':
        keep = (sigma > 0.0) & (sigma >= tolerance * largest)
        with np.errstate(divide='ignore'):
            gain = np.where(keep, 1.0 / np.where(keep, sigma, 1.0), 0.0)
    else:
        if scale is None:
            scale = float(np.max(largest) ** 2)
        lam = weight * float(scale)
        keep = sigma ** 2 >= lam if lam > 0.0 else sigma > 0.0
        denominator = sigma ** 2 + lam
        with np.errstate(divide='ignore', invalid='ignore'):
            gain = np.where(denominator > 0.0, sigma / denominator, 0.0)
    # V diag(gain) U^H, line by line
    inverse[finite] = np.einsum('lji,lj,lkj->lik', vh.conj(), gain, u.conj())
    kept[finite] = keep.sum(axis=1)
    return inverse, kept


def through(plant: Any, density: Any) -> np.ndarray:
    """`plant` times `density` times the plant's conjugate transpose,
    at every line: the spectral density matrix of the outputs of a
    linear system whose inputs have the spectral density matrix
    `density`. With an inverse for `plant`, the inputs that produce a
    target. Both (lines, ...) arrays; the result is (lines, rows,
    rows)."""
    plant = np.asarray(plant, dtype=np.complex128)
    density = np.asarray(density, dtype=np.complex128)
    return plant @ density @ np.conj(np.swapaxes(plant, 1, 2))


# ---- channels and the two matrices --------------------------------------------

def _channel(dof: Any, dimension: str, side: int) -> tuple[str, str]:
    """(DOF, quantity) for one side of a record: 0 the response, 1 the
    reference, read from the record's dimension (`channel_quantities`),
    so a drive point's accelerometer and its force gauge are two
    channels."""
    from .data import channel_quantities

    return str(dof), channel_quantities(dimension)[side]


def _labels(keys: list[tuple[str, str]]) -> list[str]:
    """A label per (DOF, quantity) channel: the DOF, with the quantity
    added only where one DOF carries two."""
    dofs = [dof for dof, _quantity in keys]
    return [dof if dofs.count(dof) == 1 else f'{dof} {quantity}'
            for dof, quantity in keys]


def _plant(frf: Frf) -> tuple[list[tuple[str, str]], list[tuple[str, str]],
                              np.ndarray, np.ndarray]:
    """The FRF's records as (rows, columns, matrix, filled): a row per
    response channel and a column per drive, (lines, rows, columns),
    with `filled` saying which entries a record supplied. Refuses a
    record whose units are not declared and drives of two quantities."""
    from ..units import UNKNOWN

    rows: list[tuple[str, str]] = []
    columns: list[tuple[str, str]] = []
    keys = []
    for i in range(frf.num_records):
        dimension = str(frf.ordinate_dim[i])
        if dimension == UNKNOWN:
            raise ValueError(
                f'the units of the FRF {frf.response_dof[i]} over '
                f'{frf.reference_dof[i]} are not declared, so the drives '
                'it predicts would be in a scale nobody chose; declare '
                'them first')
        row = _channel(frf.response_dof[i], dimension, 0)
        column = _channel(frf.reference_dof[i], dimension, 1)
        if row not in rows:
            rows.append(row)
        if column not in columns:
            columns.append(column)
        keys.append((rows.index(row), columns.index(column)))
    drives = sorted({quantity for _dof, quantity in columns})
    if len(drives) > 1:
        raise ValueError(
            f'the FRF is against drives of {" and ".join(drives)}; the '
            'drives predicted together have to be one quantity')
    matrix = np.zeros((len(frf.abscissa), len(rows), len(columns)),
                      dtype=np.complex128)
    filled = np.zeros((len(rows), len(columns)), dtype=bool)
    ordinate = np.asarray(frf.ordinate)
    for i, (r, c) in enumerate(keys):
        matrix[:, r, c] = ordinate[i]
        filled[r, c] = True
    return rows, columns, matrix, filled


def _per_bin(specification: Psd, values: np.ndarray,
             lines: np.ndarray) -> np.ndarray:
    """A density on lines read at other lines: the value of the bin
    each one falls in, not-a-number outside every bin."""
    left, right = specification.bin_bounds()
    order = np.argsort(left)
    left, right, values = left[order], right[order], values[order]
    k = np.searchsorted(right, lines, side='right')
    inside = k < len(right)
    k = np.minimum(k, len(right) - 1)
    inside &= left[k] <= lines
    out = np.full(len(lines), np.nan, dtype=values.dtype)
    out[inside] = values[k[inside]]
    return out


def _read_target(specification: Psd, channels: list[tuple[str, str]],
                 lines: np.ndarray) -> np.ndarray:
    """The specification's matrix on the control channels, at `lines`.

    A curve written at breakpoints is read the way it is drawn: the
    autospectra log-log (`compliance.log_interpolate`, not-a-number
    outside what is written) and the cross terms magnitude log-log and
    phase linear (`author.read_at_lines`), as the authoring sheet reads
    them. A density on lines is read per bin. A cross term held only by
    its mirror is its mirror's conjugate. Lines where any autospectrum
    is not written are not-a-number throughout."""
    from ..units import UNKNOWN
    from .author import read_at_lines
    from .compliance import log_interpolate

    references = (specification.reference_dof
                  if specification.reference_dof is not None
                  else specification.response_dof)
    held: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    for i in range(specification.num_records):
        dimension = str(specification.ordinate_dim[i])
        if dimension == UNKNOWN:
            raise ValueError(
                f'the units of the specification at '
                f'{specification.response_dof[i]} are not declared, so '
                'its levels cannot be read against the FRF; declare '
                'them first')
        row = _channel(specification.response_dof[i], dimension, 0)
        column = _channel(references[i], dimension, 1)
        held[(row, column)] = i
    labels = _labels(channels)
    n = len(channels)
    breakpoints = getattr(specification, 'interpolation', 'bin') == 'log_log'
    ordinate = np.atleast_2d(np.asarray(specification.ordinate))
    target = np.zeros((len(lines), n, n), dtype=np.complex128)
    for a in range(n):
        for b in range(a, n):
            pair = (channels[a], channels[b])
            mirror = (channels[b], channels[a])
            record = held.get(pair, held.get(mirror))
            if record is None:
                if a == b:
                    raise ValueError(
                        f'the specification has no autospectrum at '
                        f'{labels[a]}')
                raise ValueError(
                    f'the specification has no cross term between '
                    f'{labels[a]} and {labels[b]}: with several control '
                    'channels the drives depend on how each pair moves '
                    'together, so every pair has to be written')
            values = ordinate[record]
            if a == b:
                values = np.real(values)
                read = (log_interpolate(lines, specification.abscissa, values)
                        if breakpoints else
                        _per_bin(specification, values.astype(float), lines))
                with np.errstate(invalid='ignore'):
                    read = np.where(read > 0.0, read, np.nan)
            else:
                values = values.astype(np.complex128)
                read = (read_at_lines(np.asarray(specification.abscissa,
                                                 dtype=float), values, lines)
                        if breakpoints else
                        _per_bin(specification, values, lines))
                if pair not in held:
                    read = np.conj(read)
            target[:, a, b] = read
            target[:, b, a] = np.conj(read)
    unwritten = ~np.isfinite(np.real(np.einsum('lii->li', target))).all(axis=1)
    target[unwritten] = np.nan
    return target


def _control_channels(specification: Psd, rows: list[tuple[str, str]]
                      ) -> list[tuple[str, str]]:
    """The FRF rows the specification asks something of, in the
    specification's order, refusing a channel the FRF does not hold or
    holds as another quantity."""
    from ..units import UNKNOWN
    from .compliance import units_check

    wanted: list[tuple[str, str]] = []
    for i in range(specification.num_records):
        dimension = str(specification.ordinate_dim[i])
        if dimension == UNKNOWN:
            raise ValueError(
                f'the units of the specification at '
                f'{specification.response_dof[i]} are not declared, so '
                'its levels cannot be read against the FRF; declare '
                'them first')
        key = _channel(specification.response_dof[i], dimension, 0)
        if key not in wanted:
            wanted.append(key)
    for dof, quantity in wanted:
        if (dof, quantity) in rows:
            continue
        there = [q for d, q in rows if d == dof]
        if there:
            refused, _note = units_check(quantity, there[0],
                                         wanted_name='the specification',
                                         held_name='the FRF')
            raise ValueError(f'at {dof}, {refused}')
        raise ValueError(f'the specification asks for {dof}, and the FRF '
                         'has no response there')
    return wanted


# ---- what the drives ask of the rig ---------------------------------------------

def _widths(frequency: np.ndarray) -> np.ndarray:
    """Each line's bin, from the whole axis, as `compliance.rms` takes
    it, so a line left out does not widen its neighbors'."""
    return np.gradient(np.asarray(frequency, dtype=float))


def _gaussian(frequency: np.ndarray, density: np.ndarray, duration: float,
              peak_factor: float | None) -> tuple[float, float, float]:
    """(rms, peak, peak factor) of a Gaussian signal with this one-sided
    autospectrum: the root of the area (`compliance.rms`) and the
    expected largest peak over `duration` from its own crossing rate,
    or `peak_factor` when one is stated."""
    from .compliance import rms
    from .time_statistics import expected_largest_peak, rice_rate

    level = rms(frequency, density)
    if level == 0.0:
        # a channel the drives do not reach: nothing crosses, and the
        # peak of nothing is nothing, not the not-a-number of Rice's
        # formula on an empty spectrum
        return 0.0, 0.0, 0.0
    if peak_factor is None:
        usable = np.where(np.isfinite(density), density, 0.0)
        # both ways: the largest peak of either sign, which is what a
        # limit or a stroke each way is reached by. Counted one way, the
        # peak read about 3 percent under what simulated noise reaches
        factor = expected_largest_peak(2.0 * rice_rate(frequency, usable),
                                       duration)
    else:
        factor = float(peak_factor)
    return level, level * factor, factor


def _dominant(frequency: np.ndarray, density: np.ndarray
              ) -> tuple[float, tuple[float, float]]:
    """The line contributing most to a mean square, and the span of the
    fewest lines holding `DOMINANT_SHARE` of it."""
    contribution = np.where(np.isfinite(density), density, 0.0) \
        * _widths(frequency)
    if not contribution.max() > 0.0:
        return float('nan'), (float('nan'), float('nan'))
    # largest first, and the lowest line first among equals
    order = np.argsort(-contribution, kind='stable')
    running = np.cumsum(contribution[order])
    count = int(np.searchsorted(running, DOMINANT_SHARE * running[-1])) + 1
    chosen = frequency[order[:count]]
    return (float(frequency[order[0]]),
            (float(chosen.min()), float(chosen.max())))


def runs(flags: Any) -> list[tuple[int, int]]:
    """(first, last) index of each run of consecutive true flags."""
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return []
    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    return [(int(a), int(b) - 1) for a, b in
            zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _amount(value: float, dimension: str,
            unit_system: UnitSystem | None) -> str:
    """`value`, in SI, as a number and a unit in `unit_system`, SI's own
    when none is given, written as `core.stroke` writes its readings."""
    from ..units import SI
    from .stroke import amount as written

    return written(value, dimension, SI if unit_system is None
                   else unit_system)


#: what each kind of demand is called in a sentence
KINDS = {'drive': 'drive', 'force': 'force', 'velocity': 'velocity',
         'stroke': 'travel each way', 'response': 'response'}


@dataclass(frozen=True)
class Demand:
    """One quantity the predicted drives ask of the rig, against its
    rating.

    `kind` is 'drive', 'force', 'velocity', 'stroke' (the travel each
    way from the middle of the stroke) or 'response' (another channel
    against its own limit). `channel` names the drive or the channel it
    is read at, `dimension` the quantity of `rms`, `peak` and `rating`,
    all in SI. For the force, the velocity and the travel, `peak` is
    what `core.shaker` judges: the force at the peak acceleration, the
    velocity and the travel against the body. `share` is the peak over
    the rating, by `core.shaker`'s rule for a shaker, None where no
    rating was given. `frequency` is the line contributing most to the
    mean square and `band` the span of the lines holding half of it.
    """

    kind: str
    channel: str
    dimension: str
    rms: float
    peak: float
    rating: float | None
    share: float | None
    frequency: float
    band: tuple[float, float]

    @property
    def exceeds(self) -> bool | None:
        """Whether the peak is over the rating; None without one."""
        return None if self.share is None else bool(self.share > 1.0)

    @property
    def name(self) -> str:
        """The quantity and where, in words: 'travel each way at 1Z+'."""
        return f'{KINDS[self.kind]} at {self.channel}'

    def describe(self, unit_system: UnitSystem | None = None) -> str:
        """The demand in a sentence, in `unit_system`'s units."""
        if self.kind == 'stroke':
            words = (f'{self.name}: {_amount(self.peak, self.dimension, unit_system)}'
                     f' expected')
        else:
            words = (f'{self.name}: {_amount(self.rms, self.dimension, unit_system)}'
                     f' rms, {_amount(self.peak, self.dimension, unit_system)}'
                     ' expected peak')
        if self.rating is not None and self.share is not None:
            words += (f' against {_amount(self.rating, self.dimension, unit_system)}'
                      f' ({100.0 * self.share:.0f} %'
                      f'{", over" if self.share > 1.0 else ""})')
        else:
            words += ', no rating given'
        if np.isfinite(self.frequency):
            low, high = self.band
            words += (f'; most at {self.frequency:.4g} Hz, half from '
                      f'{low:.4g} to {high:.4g} Hz')
        return words


@dataclass(frozen=True)
class DrivePrediction:
    """The drives a specification asks for through an FRF, the
    responses they make, and what they ask of the rig.

    Everything is in SI, on the FRF's lines. `drive_csd` is the drives'
    spectral density matrix (lines, drives, drives) and `asd` the
    predicted autospectrum of every FRF row (lines, rows); `target` is
    the specification's matrix on the control channels and `control`
    the control channels' predicted matrix. `judged` marks the lines
    predicted: above zero hertz, the plant finite and every control
    autospectrum written; the arrays are not-a-number elsewhere.
    `kept` is how many directions the inverse kept at each line and
    `condition` the plant's condition number there. `demands` are the
    quantities judged, `notes` what the prediction should be read with.
    """

    frequency: np.ndarray
    drives: list[str]
    drive_quantity: str
    rows: list[str]
    row_quantities: list[str]
    controls: list[int]
    drive_csd: np.ndarray
    asd: np.ndarray
    target: np.ndarray
    control: np.ndarray
    judged: np.ndarray
    kept: np.ndarray
    condition: np.ndarray
    method: str
    tolerance: float
    weight: float
    duration: float
    peak_factor: float | None
    demands: list[Demand] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def single(self) -> bool:
        """One drive and one control channel."""
        return len(self.drives) == 1 and len(self.controls) == 1

    def drive_asd(self, drive: int | str = 0) -> np.ndarray:
        """One drive's predicted autospectrum, by index or label."""
        j = self.drives.index(drive) if isinstance(drive, str) else int(drive)
        return np.real(self.drive_csd[:, j, j])

    def drive_coherence(self, first: int | str, second: int | str
                        ) -> np.ndarray:
        """The coherence between two predicted drives at every line:
        one where the specification makes them copies of one another,
        zero where it makes them independent."""
        i = self.drives.index(first) if isinstance(first, str) else int(first)
        j = (self.drives.index(second) if isinstance(second, str)
             else int(second))
        s = self.drive_csd
        with np.errstate(divide='ignore', invalid='ignore'):
            return (np.abs(s[:, i, j]) ** 2
                    / np.real(s[:, i, i] * s[:, j, j])).real

    @property
    def shortfall_db(self) -> np.ndarray:
        """The control channels' predicted autospectra over the target,
        in dB, (lines, controls): zero where the inverse met it."""
        ours = np.real(np.einsum('lii->li', self.control))
        asked = np.real(np.einsum('lii->li', self.target))
        with np.errstate(divide='ignore', invalid='ignore'):
            return 10.0 * np.log10(ours / asked)

    def missed(self, control: int | None = None) -> np.ndarray:
        """The lines where a control channel's prediction misses the
        target by more than `UNMET_DB`, by its place among the control
        channels; with none, where any of them does. A line where the
        prediction is not a number, or zero against a target, misses."""
        gap = np.abs(self.shortfall_db)
        off = ~(np.where(np.isfinite(gap), gap, np.inf) <= UNMET_DB)
        off = off.any(axis=1) if control is None else off[:, int(control)]
        return self.judged & off

    @property
    def unmet(self) -> np.ndarray:
        """The lines where some control channel's prediction misses the
        target by more than `UNMET_DB` (`missed`)."""
        return self.missed()

    @property
    def unmet_bands(self) -> list[tuple[float, float]]:
        """(first, last) frequency of each run of unmet lines."""
        return [(float(self.frequency[first]), float(self.frequency[last]))
                for first, last in runs(self.unmet)]

    @property
    def judged_demands(self) -> list[Demand]:
        """The demands with a rating, most of their rating first."""
        rated = [d for d in self.demands if d.share is not None]
        return sorted(rated, key=lambda d: d.share, reverse=True)

    @property
    def first_out(self) -> Demand | None:
        """The demand nearest its rating, or furthest past it: what runs
        out first as the level comes up. None when nothing is rated."""
        rated = self.judged_demands
        return rated[0] if rated else None

    @property
    def fits(self) -> bool | None:
        """Whether every rated demand is within its rating; None when
        nothing is rated."""
        rated = self.judged_demands
        if not rated:
            return None
        return all(not d.exceeds for d in rated)

    @property
    def headroom_db(self) -> float | None:
        """How far the level could rise before the first demand reaches
        its rating, in dB on the test level (twenty times the log of
        one over its share: every peak scales with the level); negative
        when it is already past. None when nothing is rated."""
        first = self.first_out
        if first is None or not first.share > 0.0:
            return None
        return float(-20.0 * np.log10(first.share))

    @property
    def title(self) -> str:
        """What was predicted, in a few words."""
        drives = len(self.drives)
        controls = len(self.controls)
        return (f'{drives} drive{"s" * (drives != 1)} for '
                f'{controls} control channel{"s" * (controls != 1)}')

    def describe(self, unit_system: UnitSystem | None = None) -> list[str]:
        """The prediction in sentences: the verdict, each demand, the
        unmet bands and the notes."""
        lines = []
        first = self.first_out
        if first is None:
            lines.append(f'{self.title}: no ratings given')
        elif first.exceeds:
            lines.append(f'{self.title}: the {first.name} runs out first, '
                         f'at {100.0 * first.share:.0f} % of its rating, '
                         f'most at {first.frequency:.4g} Hz')
        else:
            lines.append(f'{self.title}: within every rating; the '
                         f'{first.name} is nearest, at '
                         f'{100.0 * first.share:.0f} % '
                         f'({self.headroom_db:.1f} dB of headroom), most at '
                         f'{first.frequency:.4g} Hz')
        lines.extend(d.describe(unit_system) for d in self.demands)
        bands = self.unmet_bands
        if bands:
            spans = ', '.join(f'{lo:.4g} Hz' if lo == hi else
                              f'{lo:.4g} to {hi:.4g} Hz'
                              for lo, hi in bands[:3])
            more = (f' and {len(bands) - 3} more band'
                    f'{"s" * (len(bands) - 3 != 1)}' if len(bands) > 3
                    else '')
            lines.append(f'target not met by the inverse within '
                         f'{UNMET_DB:g} dB at {spans}{more}')
        lines.extend(self.notes)
        return lines


# ---- the prediction -------------------------------------------------------------

def _positive(value: Any, what: str) -> float:
    """A rating or a limit as a float, refused unless finite and above
    zero (the same words `ShakerRating` refuses with)."""
    value = float(value)
    if not (np.isfinite(value) and value > 0.0):
        bound = 'above zero' if np.isfinite(value) else 'above zero and finite'
        raise ValueError(f'the {what} has to be {bound}, not {value:g}')
    return value


def _for_each(given: Any, labels: list[str], what: str) -> dict[str, Any]:
    """A setting given once for every drive, or as a mapping by drive
    label (or DOF), as {label: value}; refuses a key that is no drive."""
    if given is None:
        return {}
    if not isinstance(given, Mapping):
        return dict.fromkeys(labels, given)
    out = {}
    for key, value in given.items():
        label = _find(str(key), labels)
        if label is None:
            raise ValueError(f'{what} names {key}, which is not a drive of '
                             f'the FRF ({", ".join(labels)})')
        out[label] = value
    return out


def _find(key: str, labels: list[str]) -> str | None:
    """The label `key` names, or None. A label is its channel's DOF, with
    the quantity added only where one DOF holds two (`_labels`), so a
    DOF names a channel wherever it can."""
    return key if key in labels else None


def predict_drives(frf: Frf, specification: Psd, *,
                   method: str = 'truncated', tolerance: float = TRUNCATION,
                   weight: float = TIKHONOV,
                   drive_limit: float | Mapping[str, float] | None = None,
                   shakers: ShakerRating | Mapping[str, ShakerRating]
                   | None = None,
                   tables: Mapping[str, str] | None = None,
                   limits: Mapping[str, float] | None = None,
                   duration: float = DURATION,
                   peak_factor: float | None = None,
                   progress: Callable[[int, int], None] | None = None
                   ) -> DrivePrediction:
    """The drives that would make the specification at its control
    channels through this FRF, the responses they would make at every
    channel the FRF holds, and what they ask of the amplifiers and the
    shakers.

        prediction = predict_drives(
            frf, specification, drive_limit=10.0,
            shakers=ShakerRating(force=26700.0, stroke=0.0508,
                                 velocity=1.8, moving_mass=60.0))

    The FRF is from the drives (volts at the amplifier inputs, or force)
    to the control channels and to any other channel measured against
    the same drives; the specification names the control channels, a
    full cross-spectral matrix when there are several and its
    autospectrum when there is one. The drives are `Z S_ref Z^H` at
    every line, with `Z` the regularized inverse `method` names, and the
    module docstring says how each is regularized, how the peaks are
    read and what each rating is judged against. The prediction is open
    loop: what a controller's first drive would be with this FRF as its
    plant, made once.

    Parameters
    ----------
    frf : Frf
        Drives to responses, units declared, every control channel
        measured against every drive.
    specification : Psd
        The target at the control channels (a `Specification`, or any
        spectral density matrix), units declared.
    method : {'truncated', 'tikhonov'}, default 'truncated'
        The inverse (`regularized_inverse`).
    tolerance : float, default `TRUNCATION`
        The truncated inverse's share of each line's largest singular
        value.
    weight : float, default `TIKHONOV`
        The Tikhonov weight, a share of the largest squared singular
        value in the band.
    drive_limit : float or mapping, optional
        The peak drive each drive may reach, in the drive's SI unit
        (V, or N for a force drive); one number for every drive, or by
        drive.
    shakers : ShakerRating or mapping, optional
        The shaker behind each drive (`core.shaker.ShakerRating`); one
        for every drive, or by drive.
    tables : mapping, optional
        The channel each shaker's table is measured at, by drive. By
        default the acceleration at the drive's own DOF, or the control
        channel of a single-input test.
    limits : mapping, optional
        A peak limit for any channel of the FRF, by channel, in its SI
        unit: a force gauge in N, a current monitor in its volts.
    duration : float, default `DURATION`
        The run's length in seconds, for the expected largest peaks.
    peak_factor : float, optional
        A stated peak over rms for every quantity, in place of the
        expected largest peak.
    progress : callable, optional
        Told ``(done, total)`` as the stages finish (the plant read, its
        inverse, the drives and responses, the demands); what it raises
        stops the prediction. The app's progress bar uses it.

    Returns
    -------
    DrivePrediction
        The drives' matrix, every channel's predicted autospectrum, the
        control channels' prediction beside the target, the demands
        against their ratings, and the notes.

    Raises
    ------
    TypeError
        If `frf` is not an FRF, `specification` not a spectral density,
        or a shaker's rating not a `ShakerRating`.
    ValueError
        If either object holds no records or undeclared units, the
        drives are of two quantities, a control channel is missing from
        the FRF or held there as another quantity, a control channel is
        not measured against every drive, the specification lacks an
        autospectrum or (with several control channels) a cross term,
        the two share no line, the plant is zero or not a number at
        every line shared, a setting is out of range, or a rating names
        a drive or channel the FRF does not have or a shaker has no
        acceleration measured at its table.
    """
    from .data import Frf, Psd

    if not isinstance(frf, Frf):
        raise TypeError(f'a prediction reads an FRF, not a '
                         f'{type(frf).__name__}')
    if not isinstance(specification, Psd):
        raise TypeError(f'a prediction reads a specification, not a '
                         f'{type(specification).__name__}')
    if frf.num_records == 0 or specification.num_records == 0:
        raise ValueError('nothing to predict: '
                         + ('the FRF' if frf.num_records == 0
                            else 'the specification') + ' holds no records')
    duration = _positive(duration, 'duration')
    if peak_factor is not None:
        peak_factor = _positive(peak_factor, 'peak factor')
    # the settings are checked before any arithmetic, so a refusal never
    # waits on the inverse
    regularized_inverse(np.zeros((0, 1, 1)), method=method,
                        tolerance=tolerance, weight=weight)

    ticker = Ticker(progress)
    ticker.add(4)
    rows, columns, matrix, filled = _plant(frf)
    row_labels = _labels(rows)
    drive_labels = _labels(columns)
    controls = _control_channels(specification, rows)
    control_index = [rows.index(key) for key in controls]
    for r in control_index:
        missing = [drive_labels[c] for c in range(len(columns))
                   if not filled[r, c]]
        if missing:
            raise ValueError(
                f'the control channel {row_labels[r]} is not measured '
                f'against {", ".join(missing)}; every control channel '
                'needs its FRF from every drive')
    notes: list[str] = []
    complete = filled.all(axis=1)
    if not complete.all():
        dropped = [row_labels[r] for r in np.flatnonzero(~complete)]
        notes.append(f'{", ".join(dropped)} not predicted: not measured '
                     'against every drive')

    f = np.asarray(frf.abscissa, dtype=float)
    target = _read_target(specification, controls, f)
    plant = matrix[:, control_index, :]
    finite = np.isfinite(plant).all(axis=(1, 2))
    written = np.isfinite(np.real(np.einsum('lii->li', target))).all(axis=1)
    judged = (f > 0.0) & written & finite
    if not (written & (f > 0.0)).any():
        raise ValueError('the specification and the FRF share no line '
                         'where both are written')
    nan_lines = int((written & (f > 0.0) & ~finite).sum())
    where = _not_a_number(frf)
    if not judged.any():
        raise ValueError('the FRF is not a number at every line the '
                         f'specification is written over, first at {where}')
    if nan_lines:
        notes.append(f'{nan_lines} line{"s" * (nan_lines != 1)} where the '
                     f'FRF is not a number not predicted, first at {where}')
    ticker.tick()
    ticker.busy()
    gains = np.linalg.svd(plant[judged], compute_uv=False)
    if not gains.max() > 0.0:
        raise ValueError('the FRF is zero at every line the specification '
                         'is written over: the drives reach nothing')
    # a drive that reaches no control channel, or a control channel no
    # drive reaches, is a dead channel: named, since the inverse sets
    # its direction aside without a word otherwise
    reached = np.abs(plant[judged]).max(axis=0)
    for c in range(len(columns)):
        if not reached[:, c].any():
            notes.append(f'the drive {drive_labels[c]} reaches no control '
                         'channel at any line: it is predicted to do '
                         'nothing')
    for k, r in enumerate(control_index):
        if not reached[k].any():
            notes.append(f'no drive reaches the control channel '
                         f'{row_labels[r]} at any line: its target cannot '
                         'be met')
    with np.errstate(divide='ignore', invalid='ignore'):
        condition = np.full(len(f), np.nan)
        condition[judged] = gains[:, 0] / gains[:, -1]
    dead = int((gains[:, 0] == 0.0).sum())
    if dead:
        notes.append(f'the plant is zero at {dead} line{"s" * (dead != 1)}: '
                     'nothing can be driven there')

    inverse = np.full((len(f), len(columns), len(controls)), np.nan,
                      dtype=np.complex128)
    kept = np.zeros(len(f), dtype=int)
    inverse[judged], kept[judged] = regularized_inverse(
        plant[judged], method=method, tolerance=tolerance, weight=weight)
    ticker.tick()
    drive_csd = np.full((len(f), len(columns), len(columns)), np.nan,
                        dtype=np.complex128)
    drive_csd[judged] = through(inverse[judged], target[judged])
    responses = np.full((len(f), len(rows), len(rows)), np.nan,
                        dtype=np.complex128)
    responses[judged] = through(matrix[judged], drive_csd[judged])
    asd = np.real(np.einsum('lii->li', responses))
    asd[:, ~complete] = np.nan
    control = responses[:, control_index][:, :, control_index]
    target = np.where(judged[:, None, None], target, np.nan)
    ticker.tick()

    if len(columns) > len(controls):
        notes.append('more drives than control channels: of the drives '
                     'that meet the target, the inverse takes the one of '
                     'least power')
    elif len(columns) < len(controls):
        notes.append('fewer drives than control channels: the target is '
                     'met in the least-squares sense, and the cross terms '
                     'the drives cannot make are not')
    short = judged & (kept < min(len(columns), len(controls)))
    if short.any():
        count = int(short.sum())
        notes.append(f'the inverse set aside a direction at {count} '
                     f'line{"s" * (count != 1)} where the plant is '
                     'ill-conditioned')

    prediction = DrivePrediction(
        frequency=f, drives=drive_labels, drive_quantity=columns[0][1],
        rows=row_labels, row_quantities=[q for _d, q in rows],
        controls=control_index, drive_csd=drive_csd, asd=asd,
        target=target, control=control, judged=judged, kept=kept,
        condition=condition, method=str(method), tolerance=float(tolerance),
        weight=float(weight), duration=duration, peak_factor=peak_factor,
        notes=notes)
    demands = _demands(prediction, rows, columns, drive_limit, shakers,
                       tables, limits)
    prediction.demands.extend(demands)
    ticker.tick()
    return prediction


def _not_a_number(data: Any) -> str | None:
    """Where a spectrum first holds a value that is not a number, as the
    record's label, or None when it holds none."""
    bad = ~np.isfinite(np.asarray(data.ordinate)).all(axis=1)
    if not bad.any():
        return None
    return str(data.record_label(int(np.flatnonzero(bad)[0])))


def _demands(prediction: DrivePrediction, rows: list[tuple[str, str]],
             columns: list[tuple[str, str]], drive_limit: Any, shakers: Any,
             tables: Any, limits: Any) -> list[Demand]:
    """Each drive against its limit, each rated shaker's force, velocity
    and travel, and each limited channel, in that order."""
    f = prediction.frequency
    drives = prediction.drives
    duration, factor = prediction.duration, prediction.peak_factor
    out: list[Demand] = []

    def read(kind, channel, dimension, density, rating, judge):
        """A demand from a predicted autospectrum: its rms and expected
        peak, as `judge` restates them, with the share of `rating`."""
        level, peak, _factor = _gaussian(f, density, duration, factor)
        top, band = _dominant(f, density)
        level, peak, share = judge(level, peak)
        return Demand(kind=kind, channel=channel, dimension=dimension,
                      rms=level, peak=peak, rating=rating, share=share,
                      frequency=top, band=band)

    def against(ceiling):
        """The judgment of a plain peak against a plain limit."""
        return lambda level, peak: (level, peak, None if ceiling is None
                                    else peak / ceiling)

    ceilings = {label: _positive(value, f'drive limit at {label}')
                for label, value in
                _for_each(drive_limit, drives, 'the drive limit').items()}
    for j, label in enumerate(drives):
        ceiling = ceilings.get(label)
        out.append(read('drive', label, prediction.drive_quantity,
                        prediction.drive_asd(j), ceiling, against(ceiling)))

    ratings = _for_each(shakers, drives, 'the shaker ratings')
    chosen = _for_each(tables, drives, 'the tables')
    for label, rating in ratings.items():
        if not isinstance(rating, ShakerRating):
            raise TypeError(f'the shaker at {label} is a ShakerRating, '
                             f'not a {type(rating).__name__}')
        table = _table(prediction, rows, columns, label, chosen.get(label))
        acceleration = prediction.asd[:, table]
        omega2 = (2.0 * np.pi * f) ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            velocity = acceleration / omega2
            displacement = velocity / omega2
        where = prediction.rows[table]
        if rating.moving_mass is not None:
            # the moving mass times the acceleration, rms and peak alike;
            # the share is `core.shaker`'s, None without a force rating
            out.append(read(
                'force', label, 'force', acceleration, rating.force,
                lambda level, peak, r=rating: (
                    r.force_needed(level), r.force_needed(peak),
                    r.force_share(peak))))
        elif rating.force is not None:
            prediction.notes.append(f'the force at {label} is not judged: '
                                    'the force rating needs the moving mass')
        if rating.velocity is not None:
            # against the body, as the rating limits it
            out.append(read(
                'velocity', label, 'velocity', velocity, rating.velocity,
                lambda level, peak, r=rating: (
                    r.relative(level), r.relative(peak),
                    r.velocity_share(peak))))
        if rating.stroke is not None:
            # a Gaussian travel reaches its peak either way from the
            # middle alike, so both sides of the stroke share are it
            out.append(read(
                'stroke', label, 'length', displacement, rating.room,
                lambda level, peak, r=rating: (
                    r.relative(level), r.relative(peak),
                    r.stroke_share(r.relative(peak), r.relative(peak)))))
        if where != label.split(' ')[0]:
            prediction.notes.append(f'the shaker at {label} is read at '
                                    f'{where}')

    for key, value in (limits or {}).items():
        found = _find(str(key), prediction.rows)
        if found is None:
            raise ValueError(f'a limit is given at {key}, which is not a '
                             'channel of the FRF')
        row = prediction.rows.index(found)
        ceiling = _positive(value, f'limit at {found}')
        if not np.isfinite(prediction.asd[:, row]).any():
            raise ValueError(f'a limit is given at {found}, which is not '
                             'measured against every drive')
        out.append(read('response', found, rows[row][1],
                        prediction.asd[:, row], ceiling, against(ceiling)))
    return out


def _table(prediction: DrivePrediction, rows: list[tuple[str, str]],
           columns: list[tuple[str, str]], label: str,
           asked: str | None) -> int:
    """The row a shaker's table is read at: the channel asked for, or
    the acceleration at the drive's own DOF, or the one control channel
    of a single-input test. Refused unless it is a predicted
    acceleration."""
    if asked is not None:
        found = _find(str(asked), prediction.rows)
        if found is None:
            raise ValueError(f'the table of the shaker at {label} is '
                             f'{asked}, which is not a channel of the FRF')
        row = prediction.rows.index(found)
    else:
        dof = columns[prediction.drives.index(label)][0]
        here = [r for r, key in enumerate(rows)
                if key == (dof, 'acceleration')]
        if here:
            row = here[0]
        elif prediction.single:
            row = prediction.controls[0]
        else:
            raise ValueError(
                f'the shaker at {label} has no acceleration measured at '
                f'{dof}; name the channel its table is read at in tables')
    if rows[row][1] != 'acceleration':
        raise ValueError(f'the table of the shaker at {label} is read in '
                         f'acceleration, and {prediction.rows[row]} holds '
                         f'{rows[row][1]}')
    if not np.isfinite(prediction.asd[:, row]).any():
        raise ValueError(f'the table of the shaker at {label}, '
                         f'{prediction.rows[row]}, is not measured against '
                         'every drive')
    return row
