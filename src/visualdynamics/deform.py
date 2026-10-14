"""Turning values at DOFs into node motion.

Mode shape animation, operating deflection shapes, PSD envelopes and time
history playback all answer the same question: given values at DOFs,
where does each node go? A DOF is a node plus a signed direction, so a
value displaces its node along that direction, rotated out of the node's
displacement coordinate system into global coordinates.

The expensive work is done once, when a deflection is built, so drawing
a frame is a multiply and a write:

- direction signs and coordinate system rotations are combined into one
  unit vector per DOF
- only the nodes that move are written each frame
- for time histories, DOFs are grouped into passes with no repeated
  node, so a frame is a few vectorized adds instead of a scatter-add

Rotational DOFs are ignored, since a rotation does not displace the node
it belongs to. Nodes with no data do not move.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .core.data import parse_dof

if TYPE_CHECKING:                                    # pragma: no cover
    from .core.data import DataArray
    from .core.geometry import Geometry

AXES = {'X': 0, 'Y': 1, 'Z': 2}


def animation_records(data: DataArray, records: Sequence[int] | None
                      ) -> tuple[list[int], str]:
    """Which records a deflection animation shows, as (indices, note).

    An animation gives every node one trajectory, so it can use at most
    one record per DOF. Two rules follow:

    - When `records` is None and the object holds repeated captures (its
      `block` labels), only the first capture is animated, since twenty
      averages summed into one waveform is not a measurement anyone
      took. This is the same convention as a shape set: the first, with a
      note saying how to pick another.
    - Of what remains, the first record for each response DOF is kept and
      the rest are dropped. A drive point has a load cell beside its
      accelerometer, and meters per second squared plus newtons is not a
      deflection. An FRF repeats every response DOF once per reference
      in the same way, so a whole FRF is animated against its first
      reference.

    The note is empty when nothing was dropped, and otherwise is text to
    append to a status line, saying what was left out. Dropping records
    is reported, never refused.
    """
    notes = ''
    if records is None and data.block is not None:
        first = data.block[0]
        indices = [i for i in range(data.num_records)
                   if data.block[i] == first]
        distinct = len(dict.fromkeys(data.block))
        notes += (f' — animating {first} of {distinct} averages; pick a '
                  'grid column for another')
    else:
        indices = list(range(data.num_records) if records is None
                       else records)
    seen, kept, collided = set(), [], 0
    for i in indices:
        if data.response_dof[i] in seen:
            collided += 1
        else:
            seen.add(data.response_dof[i])
            kept.append(i)
    if collided:
        notes += (f' — {collided} records repeat an animated DOF and are '
                  'not shown')
    return kept, notes


def envelope_records(data: DataArray, records: Sequence[int] | None = None,
                     quantity: str | None = None
                     ) -> tuple[list[int], str | None, int, int]:
    """Which records a PSD envelope shows, as (indices, quantity, cross
    records dropped, other-quantity records dropped).

    An envelope is drawn from the autospectra: a cross record's phase
    belongs to an operating deflection shape, so cross records are
    dropped first. Then only one quantity is kept, since newtons and
    meters per second squared cannot share a normalization; when
    `quantity` is None it is the most common one among the remaining
    records. Both filters run before the one-record-per-DOF rule in
    `animation_records`, or a cross record or a drive point's force PSD
    listed first would take the place of that DOF's accelerometer
    autospectrum. The app and the headless call both use this function.
    """
    # The filter order broke twice before the app and the headless call
    # shared this one function.
    from .core.report import base_quantity

    initial = (list(records) if records is not None
               else list(range(data.num_records)))
    # a PSD computed here has no reference list at all — every record
    # is an auto
    kept = (initial if data.reference_dof is None
            else [i for i in initial
                  if not data.reference_dof[i]
                  or data.reference_dof[i] == data.response_dof[i]])
    kinds = [base_quantity(data.known_dim(i)) for i in kept]
    if quantity is None and kinds:
        quantity = max(set(kinds), key=kinds.count)
    of_kind = [i for i, kind in zip(kept, kinds) if kind == quantity]
    return (of_kind, quantity,
            len(initial) - len(kept), len(kept) - len(of_kind))


def dof_directions(geometry: Geometry, dofs: Sequence[str]
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, directions, used) for the DOFs this geometry can move.

    `rows` indexes into the geometry's nodes, `directions` is the global
    unit vector each DOF displaces along, and `used` is a mask over
    `dofs` of the ones kept. Rotational DOFs, DOFs that do not parse,
    and DOFs on nodes the geometry does not have are dropped.
    """
    rows, directions, used = [], [], np.zeros(len(dofs), dtype=bool)
    known = {int(node): row for row, node in enumerate(geometry.node_id)}
    cs_rotation = _cs_rotations(geometry)

    for i, dof in enumerate(dofs):
        node, direction = parse_dof(dof)
        if node is None or not direction:
            continue
        axis = AXES.get(direction[0].upper())
        if axis is None:          # rotational (RX/RY/RZ) or unrecognized
            continue
        row = known.get(int(node))
        if row is None:           # the geometry does not define this node
            continue
        vector = np.zeros(3)
        vector[axis] = -1.0 if direction.endswith('-') else 1.0
        rotation = cs_rotation.get(int(geometry.node_disp_cs[row]))
        if rotation is not None:
            vector = vector @ rotation
        rows.append(row)
        directions.append(vector)
        used[i] = True

    if not rows:
        return (np.empty(0, dtype=np.int64), np.empty((0, 3)), used)
    return np.asarray(rows, dtype=np.int64), np.asarray(directions), used


def _cs_rotations(geometry: Geometry) -> dict[int, np.ndarray]:
    """Non-identity displacement rotations, by coordinate system id.

    Rows 0-2 of a coordinate system's matrix are its axes in global
    coordinates, so a local vector maps to global as `v @ rotation`.
    Identity systems are left out so the common case costs nothing.
    """
    rotations = {}
    for index, cs_id in enumerate(geometry.cs_id):
        rotation = np.asarray(geometry.cs_matrix[index][:3], dtype=np.float64)
        if not np.allclose(rotation, np.eye(3)):
            rotations[int(cs_id)] = rotation
    return rotations


def node_displacements(geometry: Geometry, dofs: Sequence[str],
                       values: ArrayLike) -> np.ndarray:
    """The (num_nodes, 3) displacement of every node from one value per
    DOF.

    Values may be complex, and the result is then complex too. DOFs on
    the same node add, and nodes with no usable DOF are zero.
    """
    rows, directions, used = dof_directions(geometry, dofs)
    values = np.asarray(values)[used]
    result = np.zeros((geometry.num_nodes, 3),
                      dtype=np.result_type(values.dtype, np.float64))
    if len(rows):
        np.add.at(result, rows, directions * values[:, np.newaxis])
    return result


def _unique_rows(rows):
    """(unique rows, index of each row within them)."""
    if not len(rows):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    unique = np.unique(rows)
    return unique, np.searchsorted(unique, rows)


def _passes(targets):
    """Split indices into groups with no repeated target.

    A node carries at most three translational DOFs, so this yields at most
    three groups, letting a frame be a few vectorized adds rather than a
    scatter-add.
    """
    groups, seen = [], {}
    for index, target in enumerate(targets):
        slot = seen.get(int(target), 0)
        seen[int(target)] = slot + 1
        while len(groups) <= slot:
            groups.append([])
        groups[slot].append(index)
    return [np.asarray(group, dtype=np.int64) for group in groups if group]


def model_size(geometry: Geometry) -> float:
    """The length of the geometry's bounding-box diagonal, which
    deflections are scaled against; 1.0 for a geometry with no extent."""
    low, high = geometry.extent
    diagonal = float(np.linalg.norm(np.asarray(high) - np.asarray(low)))
    return diagonal or 1.0


def auto_scale(geometry: Geometry, peak: float,
               fraction: float = 0.1) -> float:
    """The scale that draws a deflection of `peak` at `fraction` of the
    model size (`model_size`); 1.0 when `peak` is zero or not finite."""
    peak = float(abs(peak))
    if not peak or not np.isfinite(peak):
        return 1.0
    return fraction * model_size(geometry) / peak


class Deflection:
    """Node offsets as a function of one parameter (a phase, or a sample
    index).

    Subclasses have `rows`, the geometry rows of the nodes that move,
    and `offsets(parameter)`, those rows' (len(rows), 3) displacement.
    No other node moves, so the cost of a frame depends on the nodes
    with data and not on the size of the model.

    `offsets` returns a buffer it reuses on every call, so a frame
    allocates nothing. Use the result before calling again, or copy it.
    """

    rows: np.ndarray

    def offsets(self, parameter: float) -> np.ndarray:
        """The (len(rows), 3) displacement of the moving nodes at
        `parameter`."""
        raise NotImplementedError

    @property
    def peak(self) -> float:
        """The largest single displacement component over the whole
        animation, which the animator scales against."""
        raise NotImplementedError

    @property
    def peak_magnitude(self) -> float:
        """The largest distance any node travels, over the whole animation.

        This differs from `peak`, the largest value of a single DOF: a
        node moving in both X and Z travels sqrt(x^2 + z^2), further than
        either. It is the top of the color scale, so the top color
        appears only at the instant the model is at its furthest.
        """
        raise NotImplementedError


class ShapeDeflection(Deflection):
    """One mode shape, animated through phase: the offsets at phase
    theta are `Re(phi * exp(i*theta))`."""

    def __init__(self, geometry: Geometry, dofs: Sequence[str],
                 shape: ArrayLike) -> None:
        rows, directions, used = dof_directions(geometry, dofs)
        values = np.asarray(shape)[used]
        contributions = directions * values[:, np.newaxis]
        # a node with X, Y and Z DOFs appears three times; sum them once here
        self.rows, targets = _unique_rows(rows)
        total = np.zeros((len(self.rows), 3), dtype=contributions.dtype)
        if len(rows):
            np.add.at(total, targets, contributions)
        self._real = np.ascontiguousarray(total.real)
        self._imag = np.ascontiguousarray(
            total.imag if np.iscomplexobj(total) else np.zeros_like(total.real))
        self._complex = bool(np.any(self._imag))
        self._buffer = np.empty_like(self._real)

    def offsets(self, parameter: float) -> np.ndarray:
        """The offsets at phase `parameter`, in radians."""
        np.multiply(self._real, np.cos(parameter), out=self._buffer)
        if self._complex:
            self._buffer -= self._imag * np.sin(parameter)
        return self._buffer

    @property
    def peak(self) -> float:
        """The largest magnitude of any node's shape coefficient along a
        global axis."""
        return float(np.abs(self._real + 1j * self._imag).max(initial=0.0))

    @property
    def peak_magnitude(self) -> float:
        """The largest distance any node travels over a cycle, in closed
        form, so no phase sweep is needed.

        With offsets r*cos(t) - m*sin(t), the squared distance a node
        travels is (A+B)/2 + (A-B)/2*cos(2t) - (r.m)*sin(2t), where A=r.r
        and B=m.m. Its largest value over t is (A+B)/2 + hypot((A-B)/2, r.m).
        """
        if not len(self.rows):
            return 0.0
        a = np.einsum('ij,ij->i', self._real, self._real)
        b = np.einsum('ij,ij->i', self._imag, self._imag)
        cross = np.einsum('ij,ij->i', self._real, self._imag)
        largest = (a + b) / 2 + np.hypot((a - b) / 2, cross)
        return float(np.sqrt(largest.max(initial=0.0)))


class OdsDeflection(Deflection):
    """An operating deflection shape read from complex spectra.

    At one abscissa line the records' complex values are a deflection
    pattern, a magnitude and phase per DOF, and the animation sweeps it
    the same way as a complex mode: `Re(H(w_line) * exp(i*theta))`. The
    plot cursor sets `line`, the frequency shown, and moving it
    recomputes only the pattern at that line, not the scene.

    Every line is drawn at full scale: the pattern is normalized by its
    own largest record, so an anti-resonance shows its shape as clearly
    as a peak. How large the response is at that line is shown by the
    plot's curve. The pattern's overall phase is also rotated so that
    phase zero is its fullest deflection, since a steady state has no
    time origin.
    """

    def __init__(self, geometry: Geometry, dofs: Sequence[str],
                 ordinate: ArrayLike) -> None:
        # Keeping the true relative amplitude was tried first and looked
        # broken: away from resonance the model barely moved, and what a
        # frequency does is what the cursor is there to ask.
        rows, directions, used = dof_directions(geometry, dofs)
        self._ordinate = np.ascontiguousarray(
            np.asarray(ordinate, dtype=np.complex128)[used])
        self._directions = directions
        self.rows, self._targets = _unique_rows(rows)
        self._real = np.zeros((len(self.rows), 3))
        self._imag = np.zeros((len(self.rows), 3))
        self._buffer = np.empty_like(self._real)
        self.num_lines: int = self._ordinate.shape[1] if len(rows) else 0
        self._line = 0
        if self.num_lines:
            self._bake()

    @property
    def line(self) -> int:
        """The abscissa index whose pattern is shown. Setting it clips
        to the valid range."""
        return self._line

    @line.setter
    def line(self, index: int) -> None:
        if not self.num_lines:
            return
        self._line = int(np.clip(index, 0, self.num_lines - 1))
        self._bake()

    def _bake(self):
        column = self._ordinate[:, self._line]
        # each line at its own full scale: the shape is the answer here,
        # the amplitude is the curve on the plot
        strongest = float(np.abs(column).max(initial=0.0))
        if strongest:
            column = column / strongest
        # a steady state has no time origin, so the pattern's global
        # phase is free — align it so phase zero is the fullest
        # deflection (the fit's rotate-real, applied per line). Near a
        # resonance an FRF is almost purely imaginary, and without this
        # the first frame, and every still, is the pattern's null: a
        # flat model that looks like the feature not working.
        total = complex(np.sum(column * column))
        if total != 0:
            column = column * np.exp(-0.5j * np.angle(total))
        contributions = self._directions * column[:, np.newaxis]
        self._real[:] = 0.0
        self._imag[:] = 0.0
        np.add.at(self._real, self._targets, contributions.real)
        np.add.at(self._imag, self._targets, contributions.imag)

    @property
    def strongest_line(self) -> int:
        """The line where a record is largest, where the cursor
        starts, since line 0 of a spectrum is often nearly zero."""
        if not self.num_lines:
            return 0
        return int(np.argmax(np.abs(self._ordinate).max(axis=0)))

    def offsets(self, parameter: float) -> np.ndarray:
        """The offsets at phase `parameter`, in radians."""
        np.multiply(self._real, np.cos(parameter), out=self._buffer)
        self._buffer -= self._imag * np.sin(parameter)
        return self._buffer

    @property
    def peak(self) -> float:
        """1.0, the value the animator scales against, or 0.0 when there
        is no data. Every line is normalized to its own largest record, so
        the largest DOF value any line shows is 1."""
        return 1.0 if self.num_lines and np.any(self._ordinate) else 0.0

    @property
    def peak_magnitude(self) -> float:
        """The furthest any node gets, over every line and phase, with
        each line at its normalized scale.

        Each line uses `ShapeDeflection`'s closed form. The lines are
        processed in chunks to limit memory, as `TimeDeflection` does
        with its samples.
        """
        nodes = len(self.rows)
        if not nodes or not self.num_lines:
            return 0.0
        largest = 0.0
        chunk = max(1, int(4e6) // (3 * nodes) or 1)
        for start in range(0, self.num_lines, chunk):
            block = self._ordinate[:, start:start + chunk]
            norms = np.abs(block).max(axis=0)
            block = block / np.where(norms, norms, 1.0)
            spread = np.zeros((nodes, block.shape[1], 3), dtype=np.complex128)
            np.add.at(spread, self._targets,
                      block[:, :, np.newaxis]
                      * self._directions[:, np.newaxis, :])
            a = np.einsum('ijk,ijk->ij', spread.real, spread.real)
            b = np.einsum('ijk,ijk->ij', spread.imag, spread.imag)
            cross = np.einsum('ijk,ijk->ij', spread.real, spread.imag)
            largest = max(largest, float(np.sqrt(
                ((a + b) / 2 + np.hypot((a - b) / 2, cross))
                .max(initial=0.0))))
        return largest


class EnvelopeDeflection(Deflection):
    """The envelope of a PSD at one line, which can be moved.

    An autospectrum gives each DOF an amplitude, the square root of the
    PSD value, and no phase or sign. So what is drawn is the envelope:
    the extreme each DOF reaches, with no claim about when. One of these
    deflects a copy of the geometry by +pattern; the animator draws a
    second copy at -pattern by giving it a negative scale, and the pair
    is the envelope.

    `line` moves as `OdsDeflection`'s does, and each line is drawn at its
    own full scale for the same reason. The level is shown by color, on
    an absolute scale: `node_decibels` gives each node's level against
    the loudest node at any line, with `FLOOR_DB` as the bottom.

    A node measured along two axes is drawn at the in-phase diagonal of
    the two, because two copies of the geometry cannot show the four
    corners of the true rectangle. That is exact for single-axis surveys,
    and the plotting guide says so for the rest.
    """

    #: the bottom of the color scale, in dB below the loudest node at
    #: any line: two decades of amplitude (four of power), below which
    #: everything reads equally quiet
    FLOOR_DB = -40.0

    def __init__(self, geometry: Geometry, dofs: Sequence[str],
                 ordinate: ArrayLike) -> None:
        rows, directions, used = dof_directions(geometry, dofs)
        # amplitude, not power: a 6 dB drop should halve the picture.
        # The clip guards measurement noise — a PSD is non-negative in
        # principle and float error can dip a tail line under zero.
        self._amplitude = np.sqrt(np.clip(
            np.asarray(ordinate, dtype=np.float64).real[used], 0.0, None))
        self._directions = directions
        self.rows, self._targets = _unique_rows(rows)
        self._offsets = np.zeros((len(self.rows), 3))
        self._db = np.zeros(len(self.rows))
        self.num_lines: int = self._amplitude.shape[1] if len(rows) else 0
        self._line = 0
        self._loudest = self._loudest_norm()
        if self.num_lines:
            self._bake()

    def _node_norms(self, columns):
        """(nodes, lines) travel per node for amplitude `columns`."""
        spread = np.zeros((len(self.rows), columns.shape[1], 3))
        np.add.at(spread, self._targets,
                  columns[:, :, np.newaxis]
                  * self._directions[:, np.newaxis, :])
        return np.sqrt(np.einsum('ijk,ijk->ij', spread, spread))

    def _loudest_norm(self) -> float:
        """The largest travel any node has at any line — the 0 dB mark."""
        if not len(self.rows) or not self.num_lines:
            return 0.0
        largest = 0.0
        chunk = max(1, int(4e6) // (3 * max(len(self.rows), 1)) or 1)
        for start in range(0, self.num_lines, chunk):
            norms = self._node_norms(self._amplitude[:, start:start + chunk])
            largest = max(largest, float(norms.max(initial=0.0)))
        return largest

    @property
    def line(self) -> int:
        """The abscissa index whose envelope is shown. Setting it clips
        to the valid range."""
        return self._line

    @line.setter
    def line(self, index: int) -> None:
        if not self.num_lines:
            return
        self._line = int(np.clip(index, 0, self.num_lines - 1))
        self._bake()

    def _bake(self):
        column = self._amplitude[:, self._line]
        norms = self._node_norms(column[:, np.newaxis])[:, 0]
        with np.errstate(divide='ignore'):
            self._db = (np.clip(20.0 * np.log10(norms / self._loudest),
                                self.FLOOR_DB, 0.0)
                        if self._loudest else
                        np.full(len(self.rows), self.FLOOR_DB))
        # each line at its own full scale, the ODS's own convention: the
        # shape is the answer here, the level is the color
        strongest = float(column.max(initial=0.0))
        if strongest:
            column = column / strongest
        self._offsets[:] = 0.0
        np.add.at(self._offsets, self._targets,
                  self._directions * column[:, np.newaxis])

    @property
    def strongest_line(self) -> int:
        """The line where a record is largest, where the cursor starts."""
        if not self.num_lines:
            return 0
        return int(np.argmax(self._amplitude.max(axis=0)))

    def offsets(self, parameter: float) -> np.ndarray:
        """The +pattern at the current line. `parameter` is not used:
        the line sets the pattern, and the sign is the animator's
        scale."""
        return self._offsets

    def node_decibels(self) -> np.ndarray:
        """Each moving node's level at the current line, in dB relative
        to the loudest node at any line (0 dB), no lower than
        `FLOOR_DB`. The color scale shows these values."""
        return self._db

    @property
    def peak(self) -> float:
        """1.0, since every line is normalized to its own largest
        record, or 0.0 when there is no data."""
        return 1.0 if self.num_lines and self._amplitude.any() else 0.0

    @property
    def peak_magnitude(self) -> float:
        """The furthest any node gets, with each line at its normalized
        scale. The animator does not use it when it fixes the color
        range in dB, but every deflection provides it."""
        if not len(self.rows) or not self.num_lines:
            return 0.0
        largest = 0.0
        chunk = max(1, int(4e6) // (3 * len(self.rows)) or 1)
        for start in range(0, self.num_lines, chunk):
            block = self._amplitude[:, start:start + chunk]
            tops = block.max(axis=0)
            norms = self._node_norms(block / np.where(tops, tops, 1.0))
            largest = max(largest, float(norms.max(initial=0.0)))
        return largest


class TimeDeflection(Deflection):
    """Time histories played back: the offsets are the samples at one
    index."""

    def __init__(self, geometry: Geometry, dofs: Sequence[str],
                 ordinate: ArrayLike) -> None:
        rows, directions, used = dof_directions(geometry, dofs)
        self._ordinate = np.ascontiguousarray(np.asarray(ordinate)[used].real)
        self.rows, targets = _unique_rows(rows)
        # groups with no repeated target, so a frame is a few vector adds
        self._passes = []
        for group in _passes(targets):
            self._passes.append((targets[group], directions[group], group,
                                 np.empty(len(group)),
                                 np.empty((len(group), 3))))
        self._buffer = np.zeros((len(self.rows), 3))
        self.num_frames: int = self._ordinate.shape[1] if len(rows) else 0

    def offsets(self, parameter: float) -> np.ndarray:
        """The offsets at sample index `parameter`."""
        column = self._ordinate[:, int(parameter)]
        self._buffer[:] = 0.0
        for targets, directions, group, values, scratch in self._passes:
            np.take(column, group, out=values)
            np.multiply(directions, values[:, np.newaxis], out=scratch)
            self._buffer[targets] += scratch
        return self._buffer

    @property
    def peak(self) -> float:
        """The largest magnitude of any sample of any record used."""
        return float(np.abs(self._ordinate).max(initial=0.0))

    @property
    def peak_magnitude(self) -> float:
        """The furthest any node gets over all samples.

        The samples are processed in chunks, because the full (nodes,
        samples, 3) array would be far larger than the data it came from.
        """
        nodes, samples = len(self.rows), self.num_frames
        if not nodes or not samples:
            return 0.0
        chunk = max(1, min(samples, int(4e6) // (3 * nodes) or 1))
        largest = 0.0
        for start in range(0, samples, chunk):
            block = self._ordinate[:, start:start + chunk]
            spread = np.zeros((nodes, block.shape[1], 3))
            for targets, directions, group, _values, _scratch in self._passes:
                np.add.at(spread, targets,
                          block[group][:, :, np.newaxis]
                          * directions[:, np.newaxis, :])
            largest = max(largest, float(
                np.sqrt(np.einsum('ijk,ijk->ij', spread, spread)).max()))
        return largest
