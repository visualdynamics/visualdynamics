"""Fit real normal modes to measured FRFs, one mode at a time.

The session holds the measured FRF matrix and the modes confirmed so far.
Everything it shows is computed on the *residual* — measurement minus the
synthesis of the confirmed modes — so each confirmed mode collapses its
own CMIF peak and the largest remaining peak is the natural next
suggestion.

One mode is fitted from the residual alone:

- the peak line's SVD gives the shape direction (its first left singular
  vector, rotated to the nearest real vector — these are real normal
  modes);
- half-power points on the residual CMIF give the damping estimate;
- a least-squares fit of the SDOF term over a band around the peak gives
  one real residue per reference, which scales the unit shape into a
  mass-normalized one. A drive point ties the scale down exactly; without
  one the scale comes from the residue magnitudes, which still
  resynthesizes the measured records correctly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import Frf
    from .shapes import ShapeSet

from ..units import UNKNOWN

_POWERS = {'length': 0, 'velocity': 1, 'acceleration': 2}


class ModalFitSession:
    """An interactive fit of real normal modes to one FRF set."""

    def __init__(self, frf: Frf, records: Sequence[int] | None = None,
                 coherence: Any | None = None) -> None:
        self.frf: Frf = frf
        self.rows: list[int] = (list(range(frf.num_records)) if records is None
                     else [int(i) for i in records])
        self.responses: list[str] = list(dict.fromkeys(
            frf.response_dof[i] for i in self.rows))
        self.references: list[str] = list(dict.fromkeys(
            frf.reference_dof[i] for i in self.rows))
        self.frequencies: np.ndarray = np.asarray(frf.abscissa, dtype=np.float64)
        self.omega: np.ndarray = 2.0 * np.pi * self.frequencies
        # the measured quantity, from the first record that says;
        # accelerance is the modal-test norm when nothing does
        self.power: int = 2
        for i in self.rows:
            numerator = frf.known_dim(i).partition('/')[0]
            if numerator in _POWERS:
                self.power = _POWERS[numerator]
                break
        row = {dof: k for k, dof in enumerate(self.responses)}
        column = {dof: k for k, dof in enumerate(self.references)}
        self.matrix: np.ndarray = np.zeros(
            (len(self.frequencies), len(self.responses),
             len(self.references)), dtype=np.complex128)
        for i in self.rows:
            self.matrix[:, row[frf.response_dof[i]],
                        column[frf.reference_dof[i]]] = frf.ordinate[i]
        # shapes cover every DOF seen, response or reference
        self.coordinate: list[str] = self.responses + [
            dof for dof in self.references if dof not in row]
        #: one dict per confirmed mode: frequency, damping, shape,
        #: description — the published ShapeSet is built from these
        self.modes: list[dict[str, Any]] = []
        #: (frequency, damping) in the order confirmed — the residual
        #: is peeled sequentially, so order is part of the fit, and
        #: this is what `fit_modes(at=...)` replays and the session
        #: journal records. `refined` counts refine passes for the
        #: same reason.
        self.confirmed: list[tuple[float, float]] = []
        self.refined: int = 0
        #: per-(line, response) least-squares weight from a coherence
        #: object, or None. Coherence is the measurement's own statement
        #: of which channel to believe where: the variance of an FRF
        #: estimate goes as (1 - g2)/g2, so the weight is its inverse,
        #: capped — a line claiming perfect coherence must not carry
        #: infinite weight into a least squares.
        self.weights: np.ndarray | None = self._coherence_weights(coherence)
        #: the residual and its CMIF, held against the mode count. Both
        #: are independent of the cursor, and dragging wants them every
        #: tick.
        self._residual = None
        self._residual_stamp = None
        self._cmif = None
        self.pending: dict[str, Any] = {'frequency': 0.0, 'damping': 0.01,
                        'description': '', 'overridden': False}
        #: the pending mode once Fit Mode has run, before Confirm
        self.preview: dict[str, Any] | None = None
        #: the whole measured range; a suggestion never leaves it, and
        #: what narrows it is the plot's own x limits and nothing else
        self.span: tuple[float, float] = (float(self.frequencies[0]),
                                          float(self.frequencies[-1]))
        self.suggest()

    def _coherence_weights(self, coherence) -> np.ndarray | None:
        """(lines, responses) weights from a coherence object's curves.

        Matched to this fit's responses by DOF name and to its lines by
        interpolation; a response the coherence does not cover weighs
        1.0 — no statement is not a bad statement. Values are clipped
        into [0, 0.999] before the variance inversion so a perfect line
        cannot claim infinite weight, and normalized to a maximum of
        one so the weights say *relative* trust and nothing else.
        """
        if coherence is None:
            return None
        curves = {dof: i for i, dof in enumerate(coherence.response_dof)}
        out = np.ones((len(self.frequencies), len(self.responses)))
        found = False
        for j, dof in enumerate(self.responses):
            if dof not in curves:
                continue
            found = True
            gamma = np.interp(self.frequencies,
                              np.asarray(coherence.abscissa, dtype=float),
                              np.asarray(coherence.ordinate[curves[dof]],
                                         dtype=float))
            gamma = np.clip(gamma, 0.0, 0.999)
            out[:, j] = gamma / (1.0 - gamma)
        if not found:
            return None
        peak = float(out.max())
        return out / peak if peak > 0 else None

    def adopt(self, shapes: ShapeSet) -> None:
        """Seed the fit with an existing shape set's modes.

        Editing a published fit means starting from it, not from nothing —
        and a ShapeSet already carries the whole fit state: frequency,
        damping, shape and description per mode. Shapes map onto this
        session's DOFs by name; DOFs the set lacks contribute nothing, and
        DOFs beyond this FRF's records are left behind.
        """
        positions = {dof: k for k, dof in enumerate(shapes.coordinate)}
        for m in range(shapes.num_shapes):
            shape = np.zeros(len(self.coordinate),
                             dtype=shapes.shape_matrix.dtype)
            for k, dof in enumerate(self.coordinate):
                index = positions.get(dof)
                if index is not None:
                    shape[k] = shapes.shape_matrix[m, index]
            self.modes.append({'frequency': float(shapes.frequency[m]),
                               'damping': float(shapes.damping[m]),
                               'shape': shape,
                               'scaled': not getattr(shapes, 'unscaled',
                                                     False),
                               'description': shapes.description[m]})
            # a published set has lost its confirm order, so an
            # adopted session replays in frequency order — the one
            # canonical reading of a set whose sequence is gone
            self.confirmed.append((float(shapes.frequency[m]),
                                   float(shapes.damping[m])))
        self.modes.sort(key=lambda mode: mode['frequency'])
        self.confirmed.sort()
        self.suggest()

    # ---- the residual -------------------------------------------------------

    def _mode_term(self, mode):
        """This mode's contribution to the matrix, (freqs, resp, ref).

        An adopted rigid-body mode meets the 0 Hz line as 0/0 — the same
        cancellation ShapeSet.synthesize_frf handles: its denominator is
        exactly -w^2, so its accelerance there is the finite residue. For
        other powers that line is unbounded and contributes nothing here
        rather than a NaN that stops the whole residual's SVD converging.
        """
        q = self._mode_scalars(mode)
        shape = mode['shape']
        residue = np.outer(shape[:len(self.responses)],
                           [shape[self.coordinate.index(dof)]
                            for dof in self.references])
        return q[:, None, None] * residue[None, :, :]

    def _mode_scalars(self, mode):
        """This mode's frequency response, one complex value per line —
        the scalar half of `_mode_term`, shared with the factored CMIF
        so the two paths cannot disagree about what a mode sounds
        like."""
        omega_r = 2.0 * np.pi * mode['frequency']
        with np.errstate(divide='ignore', invalid='ignore'):
            q = ((1j * self.omega) ** self.power
                 / (omega_r ** 2 - self.omega ** 2
                    + 2j * mode['damping'] * omega_r * self.omega))
        if omega_r == 0.0:
            q[self.omega == 0.0] = 1.0 if self.power == 2 else 0.0
        return q

    def synthesis_matrix(self) -> np.ndarray:
        out = np.zeros_like(self.matrix)
        for mode in self.modes:
            out += self._mode_term(mode)
        return out

    def residual_matrix(self) -> np.ndarray:
        """The matrix with every confirmed mode taken out of it.

        Cached against the modes, because it does not depend on where
        the cursor is and dragging asks for it constantly. The SVD over
        it is 4.7 ms on the airplane's FRFs — a third of a frame at 60
        Hz — which is what made dragging have to stay free of it.
        """
        stamp = len(self.modes)
        if self._residual is None or self._residual_stamp != stamp:
            self._residual = self.matrix - self.synthesis_matrix()
            self._residual_stamp = stamp
            self._cmif = None
        residual = self._residual
        assert residual is not None
        return residual

    def synthesis_singular_values(self, include_preview: bool = True
                                  ) -> np.ndarray | None:
        """The synthesis CMIF's singular values, (k, freqs), exactly —
        without ever materializing the synthesis.

        The modal synthesis is rank-`m` by construction: every mode
        contributes `q_k(w) * outer(shape, participation)`, so the
        whole matrix is `U diag(q(w)) V^T` with *constant* U and V.
        Singular values are invariant under matrices with orthonormal
        columns on the left and orthonormal rows on the right, so QR
        both factors once and every line's SVD collapses from
        (responses x references) to (m x references) on the tiny R
        factors. On the hard drone survey that is the difference
        between 6.5 seconds — building a two-gigabyte record array and
        decomposing 1197x12 at every line — and a few milliseconds,
        for bit-identical curves.

        Padded with zero rows up to min(responses, references), which
        is what the materialized SVD reports: the extra singular
        values of a rank-m matrix are zero, and the plot's floor test
        counts rows.

        None with nothing to synthesize.
        """
        modes = list(self.modes)
        if include_preview and self.preview is not None:
            modes.append(self.preview)
        if not modes:
            return None
        left = np.stack([mode['shape'][:len(self.responses)]
                         for mode in modes], axis=1)
        right = np.stack(
            [[mode['shape'][self.coordinate.index(dof)]
              for dof in self.references] for mode in modes], axis=1)
        scalars = np.stack([self._mode_scalars(mode) for mode in modes],
                           axis=1)                       # (freqs, m)
        r_left = np.linalg.qr(left, mode='r')            # (min(nr,m), m)
        r_right = np.linalg.qr(right, mode='r')          # (min(nc,m), m)
        core = np.einsum('im,fm,jm->fij', r_left, scalars, r_right)
        singular = np.linalg.svd(core, compute_uv=False).T   # (k, freqs)
        full = min(len(self.responses), len(self.references))
        if singular.shape[0] < full:
            singular = np.vstack([
                singular, np.zeros((full - singular.shape[0],
                                    singular.shape[1]))])
        return singular

    def synthesis_records(self, include_preview: bool = True) -> np.ndarray:
        """The confirmed (and previewed) modes' synthesis, in the
        source's record layout — what the dashed CMIF is drawn from."""
        modes = list(self.modes)
        if include_preview and self.preview is not None:
            modes.append(self.preview)
        total = np.zeros_like(self.matrix)
        for mode in modes:
            total += self._mode_term(mode)
        row = {dof: k for k, dof in enumerate(self.responses)}
        column = {dof: k for k, dof in enumerate(self.references)}
        return np.stack([
            total[:, row[self.frf.response_dof[i]],
                  column[self.frf.reference_dof[i]]]
            for i in self.rows])

    def _residual_cmif(self):
        residual = self.residual_matrix()
        if self._cmif is None:
            self._cmif = np.linalg.svd(residual, compute_uv=False)[:, 0]
        return self._cmif

    # ---- suggesting where the next mode is ----------------------------------

    def searched(self,
                 within: tuple[float, float] | None = None
                 ) -> tuple[float, float]:
        """(low, high) a suggestion may land in — the view, or all of it.

        Zooming to a region is the plainest way to say "the next mode is
        in here", and it is now the *only* way. There used to be a pair
        of draggable bounds as well, shading the rest of the plot out,
        and `searched` intersected the two. They earned nothing: the
        answer was almost always the view, and saying the same thing
        twice meant zooming and then dragging two bars to match.

        Clamped to the measured range, so a view scrolled past the end
        of the data still searches data.
        """
        if within is None:
            return self.span
        near, far = sorted((float(within[0]), float(within[1])))
        low, high = max(self.span[0], near), min(self.span[1], far)
        # a view entirely off the end of the data leaves nothing between
        # them; there is no mode out there to find, so the whole range
        # answers rather than an empty one
        return (low, high) if low < high else self.span

    def suggest(self, within: tuple[float, float] | None = None
                ) -> tuple[float, float]:
        """Move the pending mode to the most *prominent* residual CMIF
        peak inside `within` — the frequencies on screen — or anywhere.

        Prominence, not height, and the difference is the difference
        between the eye and a ruler. The CMIF is accelerance in most
        tests, so its raw height grows as frequency squared, and the
        tallest residual line is nearly always a shoulder of the
        highest dense cluster — ranked by height, ten Confirms on a
        real survey spent every one of them inside a single 1340 Hz
        cluster and never visited the 64 Hz fundamental any engineer
        would fit first. Ranked by how far a peak stands above its own
        local floor, in log height — which is what a peak *looks like*
        on the plot — the same ten confirms landed on the very set of
        modes the survey's engineer had picked by hand.

        A confirmed mode's half-power width is also spoken for — unless
        the shape standing there is somebody else's. The subtraction of
        a fitted mode is exact only when the mode is alone; in a cluster
        it leaves ridge enough that the same line stays the tallest, and
        the loop confirmed fourteen modes at one frequency without ever
        moving on. But frequency alone cannot tell that ridge from a
        genuine neighbor — a repeated pair's second mode stands at the
        confirmed frequency and was shadowed for exactly that reason —
        and shape can: ridge keeps the shape of the mode that left it,
        a real neighbor has its own (`_another_mode_at`). The cursor
        can still be dragged anywhere by hand.
        """
        s1 = self._residual_cmif()
        low, high = self.searched(within)
        inside = (self.frequencies >= low) & (self.frequencies <= high)
        if not inside.any():
            # a view narrower than the line spacing, or one falling
            # between two lines: nothing to choose from, so widen to
            # everything rather than refuse to suggest
            inside = np.ones(len(s1), dtype=bool)
        free = inside.copy()
        for mode in self.modes:
            width = mode['damping'] * mode['frequency']
            free &= np.abs(self.frequencies
                           - mode['frequency']) > width
        if not free.any():
            free = inside
        interior = [i for i in range(1, len(s1) - 1)
                    if inside[i]
                    and s1[i] >= s1[i - 1] and s1[i] >= s1[i + 1]
                    and (free[i] or self._another_mode_at(i))]
        if interior:
            window = max(25, int(np.count_nonzero(inside)) // 12)
            logs = np.log(np.maximum(s1, 1e-300))

            def prominence(i: int) -> float:
                lo, hi = max(0, i - window), min(len(s1), i + window + 1)
                return float(logs[i] - np.median(logs[lo:hi]))

            index = max(interior, key=prominence)
        else:
            candidates = np.where(free)[0]
            index = int(candidates[np.argmax(s1[candidates])])
        frequency, damping = self.search(index)
        self.pending = {'frequency': frequency,
                        'damping': damping,
                        'description': self.pending['description'],
                        'overridden': False}
        return self.pending['frequency'], self.pending['damping']

    #: how much of a residual peak's direction may lie in the confirmed
    #: modes' span before the peak is theirs. The threshold has to sit
    #: below the *blends*, not merely below pure ridge: measured on the
    #: plate's repeated pair, leftover ridge projects 0.99 onto the mode
    #: that left it and the pair's second tooth 0.002 — but in a dense
    #: worked-over cluster, an unfit neighbor's skirt mixed with fit
    #: error projects 0.25-0.30, and at 0.5 the loop fit one of those
    #: 0.26 Hz from a confirmed mode. 0.1 passes only what the confirmed
    #: modes explain almost none of, which is what a genuinely new mode
    #: looks like.
    RIDGE_FRACTION = 0.1

    def _another_mode_at(self, index: int) -> bool:
        """Whether the residual at this line belongs to a mode not yet
        confirmed, judged by shape.

        Asked only for a peak inside a confirmed mode's half-power
        width. The test is against the *span* of every confirmed mode
        covering the line, not each shape one at a time: the ridge left
        in a worked-over cluster is a blend of the modes taken from it,
        and a blend can sit below any per-mode MAC threshold while
        lying entirely in their span. One SVD of one line, on a Find
        Mode click — never on the drag path.
        """
        covering = [mode['shape'][:len(self.responses)]
                    for mode in self.modes
                    if abs(self.frequencies[index] - mode['frequency'])
                    <= mode['damping'] * mode['frequency']]
        basis = np.array([shape for shape in covering
                          if np.vdot(shape, shape).real > 0.0]).T
        if basis.size == 0:
            # adopted modes with no shape on these DOFs: nothing to
            # compare, so the frequency exclusion stands
            return False
        u = np.linalg.svd(self.residual_matrix()[index])[0][:, 0]
        q, _ = np.linalg.qr(basis)
        return float(np.linalg.norm(q.conj().T @ u) ** 2) \
            < self.RIDGE_FRACTION

    def move_to(self, frequency: float, damping: bool = True) -> None:
        """The cursor moved. With `damping`, say what damping that
        frequency implies.

        This used to be free by necessity: the damping estimate meant a
        full SVD of the residual every mouse tick. The residual does
        not depend on the cursor, so it is cached against the modes
        instead, and what a tick actually costs is a one-dimensional
        search over a couple of dozen lines.

        A caller that cannot afford even that — a drag already running
        behind — passes False and gets the old behavior.
        """
        self.pending['frequency'] = float(frequency)
        if damping and not self.pending['overridden']:
            # seeded from the half-power width *at this line*, never
            # from the previous tick's answer. The residual cost over
            # damping often has two minima — a sharp fit to the narrow
            # peak and a fat one that also swallows the shoulders — and
            # a search seeded from the last answer stayed in whichever
            # basin the drag entered first: the same line gave different
            # dampings depending on where the cursor had come from, and
            # the synthesis jumped between solutions mid-drag. Seeded
            # locally, the answer is a pure function of the line.
            self.pending['damping'] = self.search_damping(frequency)
        self.preview = None       # fitted somewhere else, if at all

    def override_damping(self, damping: float) -> None:
        """Hold the damping by hand: the pending mode keeps this value
        until Find Mode suggests afresh or a confirm resets it.

        This is the second axis of the search put in the user's hand.
        Where two solutions sit at nearly the same frequency they are
        told apart by their damping, and the automatic estimate can
        only pick one; holding the damping says which neighborhood the
        fit is meant to land in.
        """
        self.pending['damping'] = float(np.clip(damping, 1e-4, 0.5))
        self.pending['overridden'] = True
        self.preview = None

    def pending_curve(self, unit_system: Any | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
        """The SDOF magnitude the pending (frequency, damping) claims,
        scaled to the residual CMIF at the cursor — the parabola under
        the cursor that shows how sharp a mode is about to be fitted.

        Closed form: no SVD, no least squares, safe to redraw on every
        tick of a drag. The crown touches the residual CMIF at the
        cursor's line, so the curve reads as "a mode this tall and this
        wide, here"; the width is the damping made visible, which is
        what lets a vertical drag be *aimed*.

        `unit_system` converts the heights the way the plotted CMIF is
        converted. The session works in SI; the plot draws display
        units, and on in-lbf-s data the two are a couple of orders of
        magnitude apart — the SI-valued parabola was drawn that far
        below the curves it was meant to hug, which on screen is
        indistinguishable from not being drawn at all. It passed every
        test, because the test fixture's units were undefined and
        undefined values display as they stand.
        """
        frequency = self.pending['frequency']
        damping = self.pending['damping']
        index = int(np.argmin(np.abs(self.frequencies - frequency)))
        band = self._search_band(index, max(2.0 * damping, 0.005))
        omega = self.omega[band]
        omega_r = 2.0 * np.pi * max(frequency, 1e-9)
        q = np.abs((1j * omega) ** self.power
                   / (omega_r ** 2 - omega ** 2
                      + 2j * damping * omega_r * omega))
        peak = float(q.max()) or 1.0
        height = float(self._residual_cmif()[index])
        values = np.asarray(height * q / peak, dtype=np.float64)
        if unit_system is not None:
            # the dimension the plotted CMIF was converted by: with one
            # dimension across the fit rows the CMIF scales by exactly
            # this factor, and mixed or undefined rows draw as they
            # stand — the same rule as display_ordinate
            dim = next((self.frf.ordinate_dim[i] for i in self.rows
                        if self.frf.ordinate_dim[i] != UNKNOWN), None)
            if dim is not None:
                values = unit_system.from_si(values, dim)
        return self.frequencies[band], values

    def estimate_damping(self, frequency: float) -> float:
        s1 = self._residual_cmif()
        index = int(np.argmin(np.abs(self.frequencies - frequency)))
        return self._half_power(s1, index)

    def _half_power(self, s1, index):
        """Damping from the half-power points around a CMIF peak.

        A side that never crosses (a shoulder, or the edge of the band)
        borrows the other side's width; neither crossing falls back to a
        nominal 1 %.
        """
        target = s1[index] / np.sqrt(2.0)
        f = self.frequencies

        def crossing(step: int) -> float | None:
            i = index
            while 0 < i + step < len(s1) - (step > 0):
                if s1[i + step] < target:
                    span = s1[i] - s1[i + step]
                    fraction = (s1[i] - target) / span if span > 0 else 0.0
                    return f[i] + fraction * (f[i + step] - f[i])
                i += step
            return None

        low, high = crossing(-1), crossing(+1)
        peak = f[index]
        if low is None and high is None:
            return 0.01
        width = ((high - low) if low is not None and high is not None
                 else 2.0 * abs((high if high is not None else low) - peak))
        return float(np.clip(width / (2.0 * peak), 1e-4, 0.5)) if peak else 0.01

    # ---- fitting one mode ---------------------------------------------------

    def fit_pending(self) -> dict[str, Any]:
        """Fit the pending mode where the cursor is, without adding it.

        Estimates the damping (unless the user typed one), computes the
        shape, and holds the result as a preview — the synthesis and MAC
        show it, and Confirm adopts it.
        """
        if not self.pending['overridden']:
            self.pending['damping'] = self.estimate_damping(
                self.pending['frequency'])
        self.preview = self._fit_mode(self.pending['frequency'],
                                      self.pending['damping'],
                                      self.pending['description'],
                                      pin=self.pending['overridden'])
        return self.preview

    def confirm(self, frequency: float | None = None,
                damping: float | None = None,
                description: str | None = None) -> dict[str, Any]:
        """Add the pending mode, fitting it first if Fit Mode has not."""
        frequency = (self.pending['frequency'] if frequency is None
                     else float(frequency))
        damping = (self.pending['damping'] if damping is None
                   else float(damping))
        description = (self.pending['description'] if description is None
                       else description)
        mode = self.preview
        if (mode is None or mode['frequency'] != frequency
                or mode['damping'] != damping):
            mode = self._fit_mode(frequency, damping, description,
                                  pin=self.pending['overridden'])
        mode['description'] = description
        self.modes.append(mode)
        self.confirmed.append((float(mode['frequency']),
                               float(mode['damping'])))
        self.modes.sort(key=lambda entry: entry['frequency'])
        self.pending['description'] = ''
        self.pending['overridden'] = False
        self.preview = None
        return mode

    def _fit_mode(self, frequency, damping, description, pin=False):
        """One real normal mode from the residual.

        The shape direction is the residual's first left singular vector at
        the peak line, rotated to the nearest real vector; the residues
        come from a least-squares fit of the SDOF term over the half-power
        band, one per reference; a drive point ties down the
        mass-normalized scale.
        """
        index = int(np.argmin(np.abs(self.frequencies - frequency)))
        residual = self.residual_matrix()

        u, _s, _vh = np.linalg.svd(residual[index])
        direction = u[:, 0]
        # the nearest real vector: rotate by half the phase of sum(u^2)
        direction = np.real(
            direction * np.exp(-0.5j * np.angle(np.sum(direction ** 2))))
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm else direction

        band = self._band(index)
        omega = self.omega[band]
        omega_r = 2.0 * np.pi * frequency
        q = ((1j * omega) ** self.power
             / (omega_r ** 2 - omega ** 2 + 2j * damping * omega_r * omega))
        projected = np.einsum('j,ljk->lk', direction, residual[band])
        weight = float(np.sum(np.abs(q) ** 2)) or 1.0
        residues = np.real(np.conj(q) @ projected) / weight   # one per ref

        # The parabola's crown is a promise the fit keeps — but only
        # for a fit the user *aimed*. With the damping held by hand,
        # the amplitude is scaled so this mode's own CMIF at the fitted
        # line meets the residual CMIF there, the very value the
        # cursor's parabola showed: the plain band LSQ traded that peak
        # away whenever the held damping differed from the data's, up
        # to 25 % low at twice the true width. For an *automatic* fit
        # the pin is exactly wrong, and it shipped on for one day: in a
        # dense cluster the residual's peak is several modes deep, and
        # pinning each rank-one fit to the whole of it let an auto loop
        # stack fourteen modes at one frequency, each inflated toward a
        # total forty times the measurement. The least squares knows
        # how much of the residual its one direction explains; the pin
        # only knows what the user meant — so it applies only when the
        # user has said what they mean.
        if pin:
            omega_line = self.omega[index]
            with np.errstate(divide='ignore', invalid='ignore'):
                q_line = ((1j * omega_line) ** self.power
                          / (omega_r ** 2 - omega_line ** 2
                             + 2j * damping * omega_r * omega_line))
            base = float(np.abs(q_line) * np.linalg.norm(residues))
            target = float(self._residual_cmif()[index])
            if base > 0 and np.isfinite(base) and target > 0:
                residues = residues * float(
                    np.clip(target / base, 0.25, 4.0))

        shape, scaled = self._shape_from(direction, residues)
        return {'frequency': frequency, 'damping': damping,
                'shape': shape, 'description': description,
                'scaled': scaled}

    def _shape_from(self, direction, residues):
        """(shape over the coordinate, scaled?) from a unit direction
        and one residue per reference.

        Mass-normalized scale: residue_jk = phi_j phi_k, so a reference
        that was also measured as a response — a drive point — says
        what phi_k is: ratio_k = A_k / u_k is phi_k-squared up to the
        common scale. Each drive point votes, weighted by u_k squared —
        how much of the mode actually lives at that drive — because a
        drive point near a node has no standing on scale: the plate's
        647 Hz mode sat on one shaker's node, that drive's noise-signed
        ratio averaged the scale down by half, and the synthesis
        visibly missed the CMIF. A ratio still negative after the sign
        convention below is that same noise self-declared — a true
        drive-point residue is phi_k squared, never negative — and is
        discarded rather than voted.

        A shape is only defined up to sign, so when the *weighted
        consensus* is negative the SVD's sign was against the residues:
        both flip, and an all-negative vote scales normally. Without
        any usable vote — no drive point measured, or every one nodal —
        the residue magnitudes stand in and the returned flag says so.
        """
        row = {dof: k for k, dof in enumerate(self.responses)}
        votes = [(residues[k] / direction[row[dof]],
                  float(direction[row[dof]]) ** 2)
                 for k, dof in enumerate(self.references)
                 if dof in row and abs(direction[row[dof]]) > 1e-6]
        if sum(ratio * weight for ratio, weight in votes) < 0:
            direction, residues = -direction, -residues
            votes = [(-ratio, weight) for ratio, weight in votes]
        kept = [(ratio, weight) for ratio, weight in votes if ratio > 0]
        if kept:
            squared = (sum(ratio * weight for ratio, weight in kept)
                       / sum(weight for _ratio, weight in kept))
        else:
            squared = float(np.linalg.norm(residues))
        scale = np.sqrt(squared) if squared > 0 else 1.0

        shape = np.empty(len(self.coordinate))
        shape[:len(self.responses)] = scale * direction
        for position, dof in enumerate(self.coordinate[len(self.responses):],
                                       start=len(self.responses)):
            k = self.references.index(dof)
            shape[position] = residues[k] / scale if scale else 0.0
        return shape, bool(kept)

    def refine_residues(self) -> int:
        """Re-fit every confirmed mode's residues together, poles held —
        and let the CMIF choose how far to trust the joint solve.

        Fitting is sequential peeling: each mode was fit on the residual
        as it stood, so the first of a close pair was fit on data that
        still contained the second's tail, and its residues absorbed a
        piece of it. The sum of the pair tracks the measurement — the
        error is in the *decomposition* — and nothing in the loop ever
        went back. This goes back: the poles stay exactly where the user
        put them, and the residues are re-estimated with all the SDOF
        terms present at once, over the union of the modes' own bands —
        lines far from every fitted mode hold only what was never
        fitted, and a full-band solve would smear that into the shapes.

        The pure joint solve has a failure mode that shipped and was
        caught by eye: when close poles come with *similar shapes* the
        basis is ill-conditioned, and plain least squares amplifies
        measurement noise into amplitude swaps between the pair — the
        shapes' directions stay right while the synthesized CMIF walks
        visibly away from the measured one. Regularizing toward the
        sequential answer damps the swap, but no fixed weight suits
        both a parallel pair and a well-separated one.

        So the weight is not fixed. A short ladder of candidates is
        solved — the pure joint answer, three ridge strengths pulled
        toward the sequential residues, and the sequential answer
        itself — and each is scored by how far its synthesized CMIF
        sits from the *measured* CMIF over the band. The winner is
        adopted. The residues are optimized on the FRFs; the model
        among them is selected on the CMIF, which is the curve the fit
        is judged by — and with the sequential answer in the ladder,
        refining can never worsen that curve.

        Returns the number of modes refined; the shapes are unchanged
        when the sequential answer won.
        """
        if not self.modes:
            return 0
        self.refined += 1
        df = (self.frequencies[1] - self.frequencies[0]
              if len(self.frequencies) > 1 else 1.0)
        lines: set[int] = set()
        for mode in self.modes:
            index = int(np.argmin(np.abs(self.frequencies
                                         - mode['frequency'])))
            half_width = 3.0 * mode['damping'] * mode['frequency']
            reach = max(round(half_width / df), 2)
            lines.update(range(max(index - reach, 0),
                               min(index + reach + 1,
                                   len(self.frequencies))))
        chosen = np.array(sorted(lines))
        omega = self.omega[chosen]
        terms = []
        for mode in self.modes:
            omega_r = 2.0 * np.pi * mode['frequency']
            terms.append((1j * omega) ** self.power
                         / (omega_r ** 2 - omega ** 2
                            + 2j * mode['damping'] * omega_r * omega))
        basis = np.stack(terms, axis=1)               # (lines, modes)
        count = len(self.modes)
        responses, references = len(self.responses), len(self.references)
        # the sequential answer, as residue matrices: the ridge's prior
        # and the ladder's safety net
        prior = []
        for mode in self.modes:
            phi = mode['shape'][:responses]
            psi = np.array([mode['shape'][self.coordinate.index(dof)]
                            for dof in self.references])
            prior.append(np.outer(phi, psi))
        prior = np.stack(prior).reshape(count, -1)
        # real residues: the real and imaginary parts are two statements
        # of the same unknowns, stacked into one real least squares.
        # With coherence weights the solve runs per response, each
        # channel's equations scaled by the root of its weight — the
        # measurement's own statement of which channel to believe where,
        # which is what keeps one bad channel from voting noise into
        # every mode's residues.

        # Modes couple through the solve only when their bands overlap.
        # An accelerance SDOF term tends to a constant at frequencies
        # far above its pole — the mass line — so a low mode's basis
        # column has presence across every higher cluster's lines, and
        # the global joint solve inflated a 64 Hz fundamental fivefold
        # to shim milli-errors across a 1400 Hz cluster's hundreds of
        # lines. Clustered by band overlap, distant modes decouple
        # exactly — which is also what the sequential fit assumed of
        # them, correctly.
        bands = []
        for mode in self.modes:
            index = int(np.argmin(np.abs(self.frequencies
                                         - mode['frequency'])))
            half_width = 3.0 * mode['damping'] * mode['frequency']
            reach = max(round(half_width / df), 2)
            bands.append((max(index - reach, 0),
                          min(index + reach + 1, len(self.frequencies))))
        clusters, current = [], [0]
        for r in range(1, count):
            if bands[r][0] <= max(bands[i][1] for i in current):
                current.append(r)
            else:
                clusters.append(current)
                current = [r]
        clusters.append(current)

        def solve_joint(fraction):
            answer = prior.copy()
            for members in clusters:
                lo = min(bands[r][0] for r in members)
                hi = max(bands[r][1] for r in members)
                lines = chosen[(chosen >= lo) & (chosen < hi)]
                omega_c = self.omega[lines]
                columns_c = []
                for r in members:
                    omega_r = 2.0 * np.pi * self.modes[r]['frequency']
                    columns_c.append(
                        (1j * omega_c) ** self.power
                        / (omega_r ** 2 - omega_c ** 2
                           + 2j * self.modes[r]['damping'] * omega_r
                           * omega_c))
                basis_c = np.stack(columns_c, axis=1)
                n = len(members)
                sub_prior = prior[members]
                if self.weights is None:
                    stacked_c = np.vstack([basis_c.real, basis_c.imag])
                    block = self.matrix[lines].reshape(len(lines), -1)
                    rhs = np.vstack([block.real, block.imag])
                    g = stacked_c.T @ stacked_c
                    u = float(np.trace(g)) / n
                    if fraction == 0.0:
                        solved, *_ = np.linalg.lstsq(stacked_c, rhs,
                                                     rcond=None)
                    else:
                        solved = np.linalg.solve(
                            g + fraction * u * np.eye(n),
                            stacked_c.T @ rhs + fraction * u * sub_prior)
                    answer[members] = solved
                    continue
                for j in range(responses):
                    root = np.sqrt(self.weights[lines, j])
                    weighted = np.vstack([basis_c.real * root[:, None],
                                          basis_c.imag * root[:, None]])
                    block = self.matrix[lines][:, j, :]
                    rhs = np.vstack([block.real * root[:, None],
                                     block.imag * root[:, None]])
                    g = weighted.T @ weighted
                    u = float(np.trace(g)) / n
                    cols = [j * references + k for k in range(references)]
                    if fraction == 0.0:
                        solved, *_ = np.linalg.lstsq(weighted, rhs,
                                                     rcond=None)
                    else:
                        solved = np.linalg.solve(
                            g + fraction * u * np.eye(n),
                            weighted.T @ rhs
                            + fraction * u * sub_prior[:, cols])
                    answer[np.ix_(members, cols)] = solved
            return answer
        # Each cluster judges its own ladder, on its own band. A
        # global score was tried twice and failed twice: linearly, a
        # quiet fundamental's fivefold inflation cost nothing against a
        # loud cluster's lines; and in log, the noise floor's vote came
        # back. Locally, the mode being judged dominates the lines
        # doing the judging, so a weak mode's sequential rung wins
        # whenever the refit would only be chasing noise — and the
        # never-worse guarantee holds cluster by cluster, in the score
        # each cluster is chosen by. The other clusters stand as a
        # fixed background at their prior residues while one is judged.
        candidates = {fraction: solve_joint(fraction)
                      for fraction in (0.0, 0.01, 0.05, 0.2)}
        best_shapes: list[tuple[np.ndarray, bool] | None] = [None] * count

        def adopted_forms(residues_flat, members):
            """(shapes, rank-one adopted matrices) for these modes."""
            shapes, adopted = [], []
            for r in members:
                matrix = residues_flat[r].reshape(responses, references)
                u, _s, _vh = np.linalg.svd(matrix)
                direction = u[:, 0]
                shape, scaled = self._shape_from(direction,
                                                 direction @ matrix)
                shapes.append((shape, scaled))
                adopted.append(np.outer(
                    shape[:responses],
                    [shape[self.coordinate.index(dof)]
                     for dof in self.references]).ravel())
            return shapes, adopted

        for members in clusters:
            if len(members) == 1:
                # A cluster of one has nobody to borrow from, and
                # refining it can only chase what is not the mode:
                # noise, and the smooth tails of louder neighbors —
                # a quiet fundamental under a loud cluster came back
                # fourfold inflated from its own locally-judged refit,
                # because absorbing the cluster's stiffness-line tail
                # genuinely fits the measurement better. The sequential
                # estimator's projection is the robust answer for a
                # lone mode, and it stands.
                mode = self.modes[members[0]]
                best_shapes[members[0]] = (mode['shape'], mode['scaled'])
                continue
            lo = min(bands[r][0] for r in members)
            hi = max(bands[r][1] for r in members)
            lines = chosen[(chosen >= lo) & (chosen < hi)]
            local_basis = np.stack(
                [basis[np.isin(chosen, lines)][:, r]
                 for r in range(count)], axis=1)
            others = [r for r in range(count) if r not in members]
            background = (local_basis[:, others]
                          @ prior[others] if others else 0.0)
            if self.weights is None:
                trust = None
                judged = self.matrix[lines]
            else:
                trust = np.sqrt(self.weights[lines])[:, :, None]
                judged = self.matrix[lines] * trust
            s_measured = np.linalg.svd(judged, compute_uv=False)[:, 0]
            chosen_shapes, chosen_score = None, None
            for fraction in (0.0, 0.01, 0.05, 0.2, None):
                flat = prior if fraction is None else candidates[fraction]
                shapes, adopted = adopted_forms(flat, members)
                cluster_part = local_basis[:, members] @ np.stack(adopted)
                synthesis = (background + cluster_part).reshape(
                    len(lines), responses, references)
                if trust is not None:
                    synthesis = synthesis * trust
                s_synthesis = np.linalg.svd(synthesis,
                                            compute_uv=False)[:, 0]
                score = float(np.linalg.norm(s_synthesis - s_measured))
                if chosen_score is None or score < chosen_score:
                    chosen_shapes, chosen_score = shapes, score
            for r, pair in zip(members, chosen_shapes):
                best_shapes[r] = pair

        for mode, (shape, scaled) in zip(self.modes, best_shapes):
            mode['shape'] = shape
            mode['scaled'] = scaled
        # the caches stamp on the mode *count*, which did not change —
        # the shapes did, so the residual is stale and must be dropped
        # by hand or the plot goes on drawing the unrefined world
        self._residual = None
        self._cmif = None
        self.preview = None
        return count

    def residual_records(self) -> np.ndarray:
        """The residual back in the source's record layout,
        (records, freqs) — what a plot of the residual draws."""
        row = {dof: k for k, dof in enumerate(self.responses)}
        column = {dof: k for k, dof in enumerate(self.references)}
        residual = self.residual_matrix()
        return np.stack([
            residual[:, row[self.frf.response_dof[i]],
                     column[self.frf.reference_dof[i]]]
            for i in self.rows])

    def _search_band(self, index, widest):
        """Lines to search over: three half-power widths of the widest
        damping considered, at least five lines, clipped to the data.

        Wider than the band a fit uses, because the search has to be
        able to move the damping out to `widest` and still be looking
        at the shoulders that would tell it to.
        """
        half_width = 3.0 * widest * self.frequencies[index]
        df = (self.frequencies[1] - self.frequencies[0]
              if len(self.frequencies) > 1 else 1.0)
        lines = max(round(half_width / df), 2)
        return slice(max(index - lines, 0),
                     min(index + lines + 1, len(self.frequencies)))

    def _residual_cost(self, index, widest):
        """A function of (frequency, damping) saying what a mode there
        would leave behind, and the cheapest possible one.

        Everything that does not depend on the two parameters is worked
        out once: the shape direction from the residual's first left
        singular vector at the peak, and the residual projected onto it
        over the band. What is left per candidate is one SDOF term over
        a couple of dozen lines, a dot product for the residues, and a
        norm — microseconds, where a full `_fit_mode` is six
        milliseconds on the airplane's FRFs.

        That is the whole reason a grid search is affordable here: only
        the frequency and the damping are nonlinear. Given them, the
        residues are a linear least squares, so the two-parameter
        surface can be walked directly rather than handed to an
        optimizer that would have to rediscover the same structure.
        Returns a cost normalized by the projected residual's own size,
        so 0 is a perfect mode and 1 is having explained nothing.
        """
        residual = self.residual_matrix()
        u, _s, _vh = np.linalg.svd(residual[index])
        direction = np.real(u[:, 0] * np.exp(
            -0.5j * np.angle(np.sum(u[:, 0] ** 2))))
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm else direction

        band = self._search_band(index, widest)
        omega = self.omega[band]
        projected = np.einsum('j,ljk->lk', direction, residual[band])
        total = float(np.sum(np.abs(projected) ** 2)) or 1.0

        def cost(frequency: float, damping: float) -> float:
            omega_r = 2.0 * np.pi * frequency
            with np.errstate(divide='ignore', invalid='ignore'):
                q = ((1j * omega) ** self.power
                     / (omega_r ** 2 - omega ** 2
                        + 2j * damping * omega_r * omega))
            if not np.all(np.isfinite(q)):
                return 1.0
            weight = float(np.sum(np.abs(q) ** 2))
            if weight <= 0.0:
                return 1.0
            residues = np.real(np.conj(q) @ projected) / weight
            left = projected - q[:, None] * residues[None, :]
            return float(np.sum(np.abs(left) ** 2)) / total

        return cost

    @staticmethod
    def _vertex(low, middle, high):
        """Where three costs put the minimum, as a fraction of a step.

        Zero when they do not bracket one — a flat or rising triple has
        no vertex to find, and pretending otherwise is how interpolation
        invents an answer.
        """
        curvature = low - 2.0 * middle + high
        if curvature <= 0.0:
            return 0.0
        return float(np.clip(0.5 * (low - high) / curvature, -0.5, 0.5))

    def search_damping(self, frequency: float, seed: float | None = None,
                       refine: int = 2, steps: int = 5) -> float:
        """The damping that leaves the least behind *at this frequency*.

        The one-dimensional half of `search`, for a cursor being
        dragged: the user is choosing the frequency by hand, so there
        is nothing to search along it, and what is worth showing live
        is what damping that frequency implies and how well it fits.

        Cheap enough to run on a drag — the residual and its SVD are
        cached against the modes rather than the cursor, so what is
        left is a couple of dozen SDOF terms.
        """
        index = int(np.argmin(np.abs(self.frequencies - frequency)))
        seed = float(seed) if seed else self._half_power(
            self._residual_cmif(), index)
        seed = float(np.clip(seed, 1e-4, 0.5))
        cost = self._residual_cost(index, min(seed * 4.0, 0.5))
        best, span = seed, seed
        for _pass in range(max(int(refine), 1)):
            candidates = np.clip(best + np.linspace(-span, span, steps),
                                 1e-4, 0.5)
            costs = [cost(frequency, z) for z in candidates]
            j = int(np.argmin(costs))
            offset = (self._vertex(costs[j - 1], costs[j], costs[j + 1])
                      if 0 < j < steps - 1 else 0.0)
            step = candidates[1] - candidates[0]
            best = float(np.clip(candidates[j] + offset * step, 1e-4, 0.5))
            span /= 5.0
        return best

    def search(self, index: int, damping: float | None = None,
               refine: int = 2, steps: int = 5) -> tuple[float, float]:
        """The (frequency, damping) that leave the least behind.

        Coarse to fine: a small grid over half a line either side and a
        wide span of damping, then the same grid again around the
        winner at a fifth the span, then a parabola through the best
        and its neighbors in each direction. Fifty cheap evaluations
        reach what a twenty-five-by-twenty-five grid would, and the
        parabola lands between grid points, where the surface really is
        quadratic and interpolating it is unbiased.

        This replaces reading the damping off two half-power crossings.
        Those are two points on a curve, they assume the peak is one
        isolated mode, and where two modes sit close the width they
        measure is the pair's — so the damping comes back too high and
        nothing says so. A residual minimum assumes none of that.

        `damping` seeds the span when the user has typed one; otherwise
        the half-power estimate seeds it, which is a fine starting
        guess even where it is a poor answer.
        """
        line = float(self.frequencies[index])
        df = float(self.frequencies[1] - self.frequencies[0]
                   if len(self.frequencies) > 1 else 1.0)
        seed = (float(damping) if damping
                else self._half_power(self._residual_cmif(), index))
        seed = float(np.clip(seed, 1e-4, 0.5))
        widest = min(seed * 4.0, 0.5)
        cost = self._residual_cost(index, widest)

        best = (line, seed)
        span_f, span_z = 0.5 * df, seed
        # The frequency may walk; the damping may not, and the leash on
        # each is a measured decision (2026-08-28). The fixed schedule
        # used to stop *on* its frequency edge when the minimum lay
        # just outside and answer the edge as though converged — on the
        # plate's own recording the 823 Hz mode came back 0.29 Hz low
        # from an unpadded FRF and right from a padded one, which is a
        # grid artifact in an estimator that should not have one. So a
        # pass whose winner sits on the frequency edge re-centers and
        # looks again rather than shrinking around a point that only
        # won by being nearest the door. Bounded twice: never further
        # than `farthest` from the picked line (a search that can
        # wander past the next line can slide onto a neighboring
        # mode), and never into a confirmed mode's own half-power span
        # — the rule `suggest` already speaks about its motion, spoken
        # here about this one, because a cluster's imperfect
        # subtraction leaves ridge enough to walk home on and stack a
        # second pole over the first.
        #
        # The damping axis is different: in a cluster a fatter mode
        # always explains more of the band, so a walking damping
        # inflated fits into blends that swallowed their neighbors —
        # measured, not feared: the four-mode cluster in
        # test_a_confirmed_peak_is_not_offered_again came back as one
        # z=0.027 blend the moment damping could roam.
        farthest = 1.5 * df
        for mode in self.modes:
            width = float(mode['damping'] * mode['frequency'])
            edge_low = mode['frequency'] - width
            edge_high = mode['frequency'] + width
            if line < edge_low:
                farthest = min(farthest, edge_low - line)
            elif line > edge_high:
                farthest = min(farthest, line - edge_high)
            else:
                # starting inside a claim: a line `suggest` offers only
                # when the ridge's shape belongs to a mode not yet
                # confirmed. A delicate cluster case — no roaming.
                farthest = 0.0
        passes = max(int(refine), 1)
        completed = 0
        walked = 0
        while completed < passes:
            frequencies = best[0] + np.linspace(-span_f, span_f, steps)
            dampings = np.clip(best[1] + np.linspace(-span_z, span_z, steps),
                               1e-4, 0.5)
            grid = [[cost(f, z) for z in dampings] for f in frequencies]
            i, j = np.unravel_index(int(np.argmin(grid)), (steps, steps))
            # a parabola through the winner and its neighbors, in each
            # direction independently — the surface is separable enough
            # near its minimum for that, and it costs nothing
            step_f = frequencies[1] - frequencies[0]
            step_z = dampings[1] - dampings[0]
            offset_f = (self._vertex(grid[i - 1][j], grid[i][j],
                                     grid[i + 1][j])
                        if 0 < i < steps - 1 else 0.0)
            offset_z = (self._vertex(grid[i][j - 1], grid[i][j],
                                     grid[i][j + 1])
                        if 0 < j < steps - 1 else 0.0)
            best = (float(frequencies[i] + offset_f * step_f),
                    float(np.clip(dampings[j] + offset_z * step_z,
                                  1e-4, 0.5)))
            if (i in (0, steps - 1) and walked < 8
                    and abs(best[0] - line) < farthest):
                walked += 1
                continue
            span_f, span_z = span_f / 5.0, span_z / 5.0
            completed += 1
        return best

    def _band(self, index):
        """Frequency lines around a peak worth fitting over: three
        half-power widths, at least five lines, clipped to the data."""
        s1 = self._residual_cmif()
        zeta = self._half_power(s1, index)
        half_width = 3.0 * zeta * self.frequencies[index]
        df = (self.frequencies[1] - self.frequencies[0]
              if len(self.frequencies) > 1 else 1.0)
        lines = max(round(half_width / df), 2)
        return slice(max(index - lines, 0),
                     min(index + lines + 1, len(self.frequencies)))

    # ---- what the fit is ----------------------------------------------------

    def shape_set(self) -> ShapeSet:
        """The confirmed modes as a ShapeSet, ready for the project."""
        from .shapes import ShapeSet

        # defined, not merely hinted: known_dim falls back to the file's
        # claim, and a hint scales nothing — shapes fitted to raw values
        # are raw, whatever the file said they measured
        known = all(self.frf.ordinate_dim[i] != UNKNOWN for i in self.rows)
        return ShapeSet(
            frequency=[mode['frequency'] for mode in self.modes],
            damping=[mode['damping'] for mode in self.modes],
            coordinate=self.coordinate,
            shape_matrix=np.array([mode['shape'] for mode in self.modes]),
            description=[mode['description'] for mode in self.modes],
            mass_unit='kg' if known else None,
            unscaled=any(not mode.get('scaled', True)
                         for mode in self.modes))
