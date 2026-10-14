"""Filtering, integration and differentiation of records.

These make up the shock workflow's chain: an acceleration record is
low-pass filtered, integrated to velocity, and integrated again to
displacement. Each step is a derived object with provenance, so a
changed filter corner marks everything after it in the chain out of
date, the same way a changed averaging does.

The whole record is processed, never the shock windows one at a time,
for three reasons. A zero-phase filter's edge transients land at the
record's ends, far from the events, where filtering each window would
put them on the data that matters. Integrating each window would give
each one its own arbitrary velocity baseline. And a derived record that
keeps the source's abscissa keeps its marks: the shock windows and
averaging frames carry over unchanged, so a report numbers the same
events on the displacement as on the acceleration.

Integration and differentiation are done in the time domain. The
frequency-domain route (dividing by iω) treats the record as periodic,
so the FFT turns the mismatch between the first and last samples into
content. The trapezoid rule and the central difference treat the record
as the finite length it is. Their cost is a known amplitude roll-off
near Nyquist, sin(ωT)/(ωT) for the central difference, which is why the
chain filters first: with a corner at a tenth of the sample rate, the
round trip from acceleration to displacement and back to acceleration
comes back within about 2 % RMS (about 4 % with a corner at a fifth).

Differentiation is `np.gradient` (second-order central differences),
not the exact inverse of the trapezoid rule. That inverse,
x[i] = 2*(y[i]-y[i-1])/dt - x[i-1], has a pole at Nyquist in its
recursion, and on real noise its error reached 280 % where the central
difference stayed under 4 %.

References
----------
The filter forms are textbook; these are where they and the drift a
double integration invents are set out.

1. Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal
   Processing*, 3rd ed. Prentice Hall. The bilinear transform, the
   filter forms and zero-phase forward-backward filtering.
2. ISO 18431-1:2005, *Mechanical vibration and shock - Signal
   processing - Part 1: General introduction*, and ISO 18431-2 on
   time-domain integration.
3. Boore, D. M., & Bommer, J. J. (2005). "Processing of strong-motion
   accelerograms: needs, options and consequences." *Soil Dynamics
   and Earthquake Engineering*. The drift a double integration
   invents, and what filtering does and does not cure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

# The chain is Brandon's (2026-08-24), and whole-record processing was
# decided before the first line was written. The round-trip figures in
# the module docstring were measured on 2026-08-24, and the
# trapezoid's exact inverse was tried and removed after the 280 %
# result: it looks like it should round-trip perfectly, but exactness
# that diverges is not accuracy.

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


def threaded_sosfiltfilt(sos: Any, rows: Any, axis: int = -1) -> np.ndarray:
    """`scipy.signal.sosfiltfilt` over a record's channels, one channel
    per thread.

    The filter releases the interpreter lock, so a thread pool over the
    rows is about six times faster on a 32-channel record. The results
    are the serial ones to the bit, since each row is the same call. A
    single row, an array under a million values, or filtering along
    another axis than the last goes to one serial call.
    """
    # Threaded on 2026-10-01 (Brandon: use the cores).
    from scipy.signal import sosfiltfilt

    rows = np.asarray(rows)
    if rows.ndim < 2 or axis not in (-1, rows.ndim - 1) or rows.shape[0] < 2 \
            or rows.size < 1_000_000:
        return sosfiltfilt(sos, rows, axis=axis)
    return np.stack(list(_filter_pool().map(
        lambda row: sosfiltfilt(sos, row), rows)))


@cache
def _filter_pool() -> Any:
    """The threads `threaded_sosfiltfilt` shares, made once and kept.

    A pool per call started up to eight threads every time the filter
    preview redrew. One gate run stalled for an hour inside exactly
    that: the main thread in `Thread.start`, waiting on a new worker
    that never said it had begun, from a render of the filter view
    (2026-10-09; CPython 3.13.2, not reproduced in 1 200 calls). A kept
    pool starts its threads once a session, so no redraw waits on a
    thread being born — and skips the start-up besides.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(min(8, os.cpu_count() or 1),
                              thread_name_prefix='visualdynamics-filter')


@dataclass(frozen=True, kw_only=True)
class Filtering:
    """The pass band a record is filtered to: its edges and order.

    `low` and `high` are the edges of the pass band, in Hz. Set `high`
    alone for a low-pass (everything below it passes), `low` alone for
    a high-pass, and both for a band-pass. Which filter it is follows
    from which edges are set (`kind`), so an invalid pairing, such as a
    band-pass with one corner, cannot be written. The fields are
    keyword-only, so a bare `Filtering(200.0)` is an error. At least one
    edge is required, edges must be above zero, `low` must be below
    `high`, and `order` must be at least 1.

    The settings are frozen, like `Averaging`, because they are stored
    with the history and their fields are its staleness fingerprint; a
    setting that could change in place would make that fingerprint
    wrong.

    The filter is a Butterworth applied forward and backward
    (`sosfiltfilt`). It has zero phase, so a shock's peak stays at its
    measured time; a causal filter would shift and smear the front by
    its group delay. The double pass doubles the effective order;
    `order` is the design order of the underlying filter. A band-pass of
    design order N has N poles per edge (scipy's convention), so each
    skirt rolls off like the low- or high-pass of the same order.
    """

    # Keyword-only because the fields were `(corner, order)` until
    # 2026-08-28, and a bare `Filtering(200.0)` changing its meaning
    # from low-pass to high-pass without an error is the kind of change
    # a call site must not survive.

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
        """'low-pass', 'high-pass' or 'band-pass', from which edges are
        set; never stored beside them."""
        if self.low is None:
            return 'low-pass'
        if self.high is None:
            return 'high-pass'
        return 'band-pass'

    def describe(self) -> str:
        """The filter in words, such as 'low-pass at 500 Hz'. The status
        line, the staleness message and the report text all use this
        wording."""
        if self.low is None:
            return f'low-pass at {self.high:g} Hz'
        if self.high is None:
            return f'high-pass at {self.low:g} Hz'
        return f'band-pass from {self.low:g} to {self.high:g} Hz'


def design(filtering: Filtering, sample_rate: float) -> Any:
    """The filter at `sample_rate`, as second-order sections from
    `scipy.signal.butter`.

    This is the one place a `Filtering` becomes a scipy design:
    filtering a record, the response plot and the GUI previews all take
    their sections from here, so they agree on what the settings mean.
    An edge at or above the Nyquist frequency raises ValueError here,
    so every caller refuses it the same way.
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
    """Copy the source's averaging and shock windows onto a derived
    record, and return the record.

    The derived record has the same abscissa, so the marks still mean
    what they meant: the averaging frames and shock windows land on the
    same samples, and a report numbers the same events on the
    displacement as on the acceleration, or on the modal coordinates as
    on the channels, since a transform through a shape set combines
    samples and moves none. The filtering is not carried: the result of
    a filter is already filtered, and offering to filter it again with
    the same settings would be a button that does nothing.
    """
    # Modal coordinates carry the marks too (Brandon, 2026-09-04).
    result.averaging = source.averaging
    # a tuple, matching `core.shocks.find` and the loader: the windows
    # are a fixed set for a record, and copying rather than sharing
    # keeps an edit to one record's marks off another's
    result.shocks = tuple(source.shocks) if source.shocks else source.shocks
    return result


def filtered(data: TimeHistory, filtering: Filtering) -> TimeHistory:
    """The record through its filter, every channel, zero phase.

    Every channel is filtered whatever it measures: a filter acts on
    the waveform, not the quantity, and the drive force belongs at the
    same bandwidth as the responses it drove. The averaging and shock
    windows are carried over (`carrying_marks`). A record with uneven
    sampling, or an edge at or above Nyquist, is refused.
    """

    from .data import TimeHistory

    rate = data.sample_rate       # raises on uneven sampling, correctly
    sos = design(filtering, rate)
    result = TimeHistory(
        data.abscissa, threaded_sosfiltfilt(sos, np.real(data.ordinate)),
        **data.record_fields())
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
    """One integration, acceleration to velocity or velocity to
    displacement, channel by channel.

    The cumulative trapezoid rule, with drift control: each channel's
    mean is removed before integrating, and a second-order Butterworth
    high-pass at `drift_corner` in Hz (`DRIFT_CORNER`, 2 Hz) is applied
    forward and backward afterward. The mean is removed first because a
    bias integrates to a ramp, and the high-pass's edge transients grow
    with how far the ramp runs. Below the corner nothing in the data
    tells drift from true motion. Pass ``None`` to integrate raw, which
    says the record's own low end is trusted, drift and all.

    Records of quantities the chain cannot take (forces, voltages,
    anything undefined) are left out of the result. A record with no
    acceleration or velocity channels is refused, as is a drift corner
    that is not between zero and Nyquist.
    """
    from scipy.integrate import cumulative_trapezoid
    from scipy.signal import butter

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
        rows = threaded_sosfiltfilt(sos, rows)
    return _derived(data, rows, kept, INTEGRATED)


def differentiate(data: TimeHistory) -> TimeHistory:
    """One differentiation, displacement to velocity or velocity to
    acceleration, channel by channel.

    Second-order central differences (`np.gradient`); the module
    docstring says why the exact inverse of the trapezoid rule is not
    used. There is no drift control, since differentiation removes
    offsets instead of growing them. Records of other quantities are
    left out, and a record with no displacement or velocity channels
    is refused.
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
    """(frequencies, magnitude in dB) of the filter as the record sees
    it.

    This is the squared magnitude |H|², not the single-pass response.
    `filtered` runs the filter forward and backward for zero phase, so
    the data is multiplied by |H| twice. A Butterworth is -3 dB at its
    nominal corner by definition, so the zero-phase pair is -6.02 dB
    there, which is not what the settings suggest; for a fourth-order
    low-pass (the default order) the -3 dB point is at about 0.90 of
    the corner. Plotting |H| would draw a filter that is not the one
    applied.

    The `lines` frequencies are log-spaced from `RESPONSE_DECADES` (3)
    decades below Nyquist up to Nyquist, extended down to half the
    lowest edge when an edge is lower than that. Edges at or above
    Nyquist are clamped just under it instead of refused, so a curve
    can be drawn while the settings are being edited. Values are
    floored at -240 dB.
    """
    # Squared magnitude, not the filter's own (Brandon, 2026-08-25).
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
