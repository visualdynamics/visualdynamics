"""Filtering, integration and differentiation of records.

The shock workflow's chain (Brandon, 2026-08-24): an acceleration
record is low-pass filtered, integrated to velocity, integrated again
to displacement — and each step is a derived object with provenance,
so a changed filter corner cascades staleness down the chain the same
way a changed averaging does.

**Whole record, never the shock windows.** Three reasons, decided
before the first line: a zero-phase filter's edge transients land at
the record's ends, far from the events, where per-window filtering
would put them exactly on the data that matters; per-window
integration hands each window its own arbitrary velocity baseline; and
a derived record keeping the source's abscissa keeps its marks — the
shock windows and averaging frames carry over untouched, so the report
numbers the same events on the displacement it numbered on the
acceleration.

**Time domain, on purpose.** The frequency-domain route (divide by iw)
is the more familiar one, but it wraps the record onto a circle: the
FFT's periodicity turns the mismatch between first and last sample
into content. The trapezoid and the central difference read the record
as the finite thing it is. Their cost is a known amplitude roll-off
near Nyquist — sin(wT)/wT for the central difference — which is
exactly why the chain filters *first*: below a corner of a tenth of
the sample rate the round trip acceleration - velocity - displacement
- velocity - acceleration comes back within ~2% RMS (measured, not
assumed, 2026-08-24; ~4% at a corner of a fifth).

**Differentiation is `np.gradient`, not the trapezoid's inverse.**
The exact algebraic inverse (x[i] = 2*(y[i]-y[i-1])/dt - x[i-1]) looks
like it should round-trip perfectly and was tried: its recursion holds
a pole at Nyquist, and on real noise it rang to 280% error where the
central difference sat under 4%. Exactness that diverges is not
accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

#: where each quantity goes under one integration. Only motion
#: integrates: the integral of a force is an impulse and the integral
#: of a voltage is nothing a test report reads, so records of any
#: other quantity are left out of the result rather than carried
#: through with a unit no one asked for.
INTEGRATED = {'acceleration': 'velocity', 'velocity': 'length'}

#: and the same chain read upward
DIFFERENTIATED = {value: key for key, value in INTEGRATED.items()}

#: the default drift high-pass corner, in Hz. Integration turns any
#: sensor bias into a ramp — a 0.05 m/s**2 offset walked a 2 s record
#: to over a meter of displacement in the design measurement — and
#: nothing in the data can tell drift from true motion below the
#: corner, so the corner is a declared judgment, not a detection.
#: 2 Hz sits below any shock content worth reporting while killing
#: the ramp dead (the same measurement ended at micrometers).
DRIFT_CORNER = 2.0


@dataclass(frozen=True, kw_only=True)
class Filtering:
    """The pass band a record is read through: its edges and order.

    `low` and `high` are the edges of the *pass band*, in Hz: set
    `high` alone for a low-pass (everything below it passes), `low`
    alone for a high-pass, both for a band-pass. Which filter this is
    follows from which edges exist (`kind`), so an invalid pairing —
    a band-pass with one corner, a kind that contradicts its numbers
    — cannot be stated at all. Keyword-only on purpose: the fields
    were `(corner, order)` until 2026-08-28, and a bare
    `Filtering(200.0)` quietly flipping its meaning from low-pass to
    high-pass is exactly the kind of silent change a call site must
    not survive.

    Frozen like `Averaging`, and for the same reason — the settings
    ride the history, the staleness fingerprint is their fields, and
    a mutable setting would be a fingerprint that lies.

    The filter itself is a Butterworth applied forward and backward
    (`sosfiltfilt`): zero phase, so a shock's peak stays at its
    measured instant — a causal filter would shift and smear the
    front, and the report's time axis would be quietly wrong by the
    group delay. The doubled pass also doubles the effective order;
    `order` here is the design order of the underlying filter. A
    band-pass of design order N carries N poles per edge (scipy's
    convention), so each skirt rolls off like the matching low- or
    high-pass of the same order.
    """

    low: float | None = None
    high: float | None = None
    order: int = 4

    def __post_init__(self) -> None:
        if self.low is None and self.high is None:
            raise ValueError('a filter needs at least one pass-band '
                             'edge: high for a low-pass, low for a '
                             'high-pass, both for a band-pass')
        for name in ('low', 'high'):
            edge = getattr(self, name)
            if edge is not None:
                if not float(edge) > 0.0:
                    raise ValueError(f'the {name} edge must be above zero')
                object.__setattr__(self, name, float(edge))
        if self.low is not None and self.high is not None \
                and not self.low < self.high:
            raise ValueError(
                f'a band-pass runs upward: low ({self.low:g} Hz) must '
                f'sit below high ({self.high:g} Hz)')
        if int(self.order) < 1:
            raise ValueError('the filter needs at least first order')
        object.__setattr__(self, 'order', int(self.order))

    @property
    def kind(self) -> str:
        """'low-pass', 'high-pass' or 'band-pass' — derived from
        which edges exist, never stored beside them."""
        if self.low is None:
            return 'low-pass'
        if self.high is None:
            return 'high-pass'
        return 'band-pass'

    def describe(self) -> str:
        """The filter in words — 'low-pass at 500 Hz'. One
        implementation: the status line, the staleness story and the
        report prose all say it this way."""
        if self.low is None:
            return f'low-pass at {self.high:g} Hz'
        if self.high is None:
            return f'high-pass at {self.low:g} Hz'
        return f'band-pass from {self.low:g} to {self.high:g} Hz'


def design(filtering: Filtering, sample_rate: float) -> Any:
    """The filter itself, as second-order sections.

    The one place a `Filtering` becomes a scipy design — the
    application, the response plot and the GUI previews all take
    their sections from here, so they cannot disagree about what the
    settings mean. Refuses an edge at or above Nyquist for the same
    reason: every caller refuses the same way.
    """
    from scipy.signal import butter

    rate = float(sample_rate)
    for edge in (filtering.low, filtering.high):
        if edge is not None and not edge < rate / 2.0:
            raise ValueError(
                f'the corner ({edge:g} Hz) must sit below the '
                f'Nyquist frequency ({rate / 2.0:g} Hz)')
    if filtering.low is None:
        return butter(filtering.order, filtering.high, 'lowpass',
                      fs=rate, output='sos')
    if filtering.high is None:
        return butter(filtering.order, filtering.low, 'highpass',
                      fs=rate, output='sos')
    return butter(filtering.order, [filtering.low, filtering.high],
                  'bandpass', fs=rate, output='sos')


def carrying_marks(source: Any, result: Any) -> Any:
    """The source's readings, carried onto a derived record.

    Same abscissa, so the marks still say what they said: the
    averaging frames and shock windows land on the same samples, and a
    report numbers the same events on the displacement it numbered on
    the acceleration — or on the modal coordinate it numbered on the
    channels, since a transform through a shape set combines samples
    and moves none (Brandon, 2026-09-04). The filtering is *not*
    carried — the result of a filter has been filtered; offering to
    filter it again with the same settings would be a button that
    does nothing.
    """
    result.averaging = source.averaging
    # a tuple, matching `core.shocks.find` and the loader: the windows
    # are a fixed set for a record, and copying rather than sharing
    # keeps an edit to one record's marks off another's
    result.shocks = tuple(source.shocks) if source.shocks else source.shocks
    return result


def filtered(data: TimeHistory, filtering: Filtering) -> TimeHistory:
    """The record through its filter, every channel, zero phase.

    Every channel whatever it measures: a filter reads shape, not
    quantity, and the drive force belongs at the same bandwidth as the
    responses it drove.
    """
    from scipy.signal import sosfiltfilt

    from .data import TimeHistory

    rate = data.sample_rate       # raises on uneven sampling, correctly
    sos = design(filtering, rate)
    result = TimeHistory(
        data.abscissa, sosfiltfilt(sos, np.real(data.ordinate), axis=-1),
        response_dof=list(data.response_dof),
        ordinate_dim=list(data.ordinate_dim),
        ordinate_unit=list(data.ordinate_unit),
        dimension_hint=list(data.dimension_hint),
        comment=list(data.comment),
        block=None if data.block is None else list(data.block))
    return carrying_marks(data, result)


def _mapped(data: Any, mapping: dict[str, str]) -> list[int]:
    """Which records the chain has somewhere to take."""
    return [i for i in range(data.num_records)
            if data.known_dim(i) in mapping]


def _derived(data: Any, rows: np.ndarray, kept: list[int],
             mapping: dict[str, str]) -> Any:
    """A new history of the mapped channels, in the mapped quantity.

    Units are left for `_fill_si_units` to name: the values are stored
    in SI, so an integrated acceleration *is* in m/s and asking the
    constructor to work that out keeps one implementation of the rule.
    """
    from .data import TimeHistory

    result = TimeHistory(
        data.abscissa, rows,
        response_dof=[data.response_dof[i] for i in kept],
        ordinate_dim=[mapping[data.known_dim(i)] for i in kept],
        comment=[data.comment[i] for i in kept],
        block=(None if data.block is None
               else [data.block[i] for i in kept]))
    return carrying_marks(data, result)


def integrate(data: TimeHistory,
              drift_corner: float | None = DRIFT_CORNER) -> TimeHistory:
    """One integration: acceleration to velocity, velocity to
    displacement, channel by channel.

    Cumulative trapezoid, then the drift control: the channel's mean
    removed before integrating (a bias integrates to a ramp, and
    `sosfiltfilt`'s edge handling scales with how far the ramp runs)
    and a gentle zero-phase high-pass at `drift_corner` after. Pass
    ``None`` to integrate raw — the caller is saying the record's own
    low end is trusted, drift and all.

    Records of quantities the chain has nowhere to take — forces,
    voltages, anything undefined — are left out of the result; a
    record with none of its channels mappable is refused, because an
    empty answer delivered politely is still nothing.
    """
    from scipy.integrate import cumulative_trapezoid
    from scipy.signal import butter, sosfiltfilt

    rate = data.sample_rate
    kept = _mapped(data, INTEGRATED)
    if not kept:
        raise ValueError(
            'nothing here integrates: the record holds no acceleration '
            'or velocity channels')
    if drift_corner is not None and not 0.0 < drift_corner < rate / 2.0:
        raise ValueError(
            f'the drift corner ({drift_corner:g} Hz) must sit between '
            f'zero and the Nyquist frequency ({rate / 2.0:g} Hz)')
    rows = np.real(data.ordinate[kept])
    if drift_corner is not None:
        rows = rows - rows.mean(axis=-1, keepdims=True)
    rows = cumulative_trapezoid(rows, dx=1.0 / rate, initial=0.0, axis=-1)
    if drift_corner is not None:
        sos = butter(2, drift_corner, 'high', fs=rate, output='sos')
        rows = sosfiltfilt(sos, rows, axis=-1)
    return _derived(data, rows, kept, INTEGRATED)


def differentiate(data: TimeHistory) -> TimeHistory:
    """One differentiation: displacement to velocity, velocity to
    acceleration, channel by channel.

    Second-order central differences (`np.gradient`) — see the module
    docstring for why the trapezoid's exact inverse was tried and
    thrown out. No drift control: differentiation kills offsets
    instead of growing them.
    """
    rate = data.sample_rate
    kept = _mapped(data, DIFFERENTIATED)
    if not kept:
        raise ValueError(
            'nothing here differentiates: the record holds no '
            'displacement or velocity channels')
    rows = np.gradient(np.real(data.ordinate[kept]), 1.0 / rate, axis=-1)
    return _derived(data, rows, kept, DIFFERENTIATED)


#: how many decades below Nyquist the response plot reaches. Fixed
#: rather than following the corner, so the curve moves against a
#: still axis while the corner is dragged — an axis that rescales with
#: the thing being set makes every setting look the same. The one
#: stretch: a typed edge below the floor (a drift high-pass on a fast
#: record) extends the axis down to show it — the drag is bounded to
#: the axis, so a drag can never cause the stretch itself.
RESPONSE_DECADES = 3.0


def response(filtering: Filtering, sample_rate: float,
             lines: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, magnitude in dB) the record actually sees.

    **The squared magnitude, not the filter's own** (Brandon,
    2026-08-25). `filtered` runs the filter forward and backward for
    zero phase, so the data is multiplied by |H| twice and the
    effective response is |H|². The consequences are worth stating
    because they contradict what the settings say: a Butterworth is
    −3 dB at its nominal corner by definition, so the zero-phase pair
    is **−6.02 dB** there, and its true −3 dB point sits at about
    0.90 of the corner. Plotting |H| instead would draw a filter
    nothing here applies.

    Log-spaced over `RESPONSE_DECADES` below Nyquist, which is where
    a filter has anything to say.
    """
    from scipy.signal import sosfreqz

    nyquist = float(sample_rate) / 2.0
    edges = [e for e in (filtering.low, filtering.high)
             if e is not None]
    start = min(nyquist / 10.0 ** RESPONSE_DECADES, min(edges) * 0.5)
    frequencies = np.logspace(np.log10(start), np.log10(nyquist),
                              int(lines))
    # edges clamped rather than refused: mid-edit the boxes pass
    # through states the filter itself would refuse, and the panel
    # wants a curve to draw against, not an exception
    safe = replace(
        filtering,
        low=(None if filtering.low is None
             else min(filtering.low, nyquist * 0.998)),
        high=(None if filtering.high is None
              else min(filtering.high, nyquist * 0.999)))
    _w, h = sosfreqz(design(safe, sample_rate), worN=frequencies,
                     fs=sample_rate)
    one_pass = np.abs(h)
    # the floor keeps log10 finite where the response underflows; at
    # −240 dB the curve is off any axis a person would draw anyway
    squared = np.maximum(one_pass ** 2, 1e-12)
    return frequencies, 20.0 * np.log10(squared)
