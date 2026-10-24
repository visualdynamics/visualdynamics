"""Where the limit channels would notch a random test, predicted before
the test from a low-level transmissibility.

A limit channel is a response channel with a ceiling, its limit
profile: an autospectrum the response there may not exceed. A
controller holds its control channel to the reference everywhere in the
band, except where holding it would put a limit channel over its
profile. There it lowers the drive until the limit channel sits on its
profile, and the control channel falls below the reference over those
lines. That dip is the notch, and it is a change to the test: the
article receives less than the requirement asked for in that band.

With one drive, every channel at a line moves in proportion to the
control channel at that line, by the transmissibility `T` from the
control channel to it. So the response a limit channel would have at
full level is the reference passed through its transmissibility,

    R = |T|^2 G_ref,

at every line, with `G_ref` the reference autospectrum. Where `R` is
over the limit profile `L`, the control has to come down to `L / |T|^2`
for that channel to sit on its limit, and the control the controller
can hold at a line is the lowest of what every limit channel allows:

    G_ctrl = min(G_ref, L_1 / |T_1|^2, L_2 / |T_2|^2, ...).

The notch is where `G_ctrl` is under `G_ref`. Its depth at a line is
`10 log10(G_ref / G_ctrl)` decibels on the ASD, and its ratio, the
control left over the reference, `G_ctrl / G_ref`: a limit channel
twice over its profile cuts a notch 3 dB deep, a ratio of a half. A run
of adjacent notched lines is one notch, named by its band, its deepest
line, its depth there, the control left there and the limit channels
that cut it. Each notch is then read against the specification's
tolerance: where `G_ctrl` falls under the lower edge of the
tolerance (the abort pair by default), the notch takes the control
outside its tolerance, and the test as specified cannot be run on this
setup without a written departure from it.

Every curve is read over each line's bin the way a comparison reads it
(`compliance.coverage` and the curve's own area rule, `Psd.areas` and
`Bounded.limit_areas`): its area over the part of the line's bin it is
written over, divided by that width. For a curve written at
breakpoints that is the cell `compliance.judge` judges the line by.
Outside the reference nothing is specified and nothing is predicted,
and outside a limit profile nothing is limited.

What it is and what it is not
-----------------------------
This is a prediction, made once, open loop: the notch a controller
would settle on if the article stayed linear between the low level and
full level. It does not iterate a control loop. Three things make the
as-run notch differ from it, and the reading says so where it applies:

- a controller cannot cut a notch only a line or two wide; it cuts
  wider than predicted (`NARROW_LINES`);
- joints and isolators usually damp more and soften as the level
  rises, so a sharp low-level peak is often lower and a little lower in
  frequency at full level, and the predicted notch deeper than the one
  cut;
- a limit channel that would notch most of the band is, in effect,
  controlling the test, and whether the test as written exists on this
  setup is a question for the requirement's owner before the run
  (`MOST_OF_BAND`).

The prediction is single-input: every transmissibility is taken
against the one control channel the specification is written at. A
multi-point control strategy is the controller's to report and is not
predicted here.

References
----------
The output autospectrum of a single-input linear system, the input's
autospectrum times the squared magnitude of its frequency response, is
Bendat and Piersol's.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The single-input output
   spectral density, `G_yy = |H|^2 G_xx`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..units import UNKNOWN

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem
    from .data import Frf, Psd, Specification

#: the tolerance pairs a notch can be read against
TOLERANCES = ('abort', 'warning')

#: how few adjacent lines make a notch narrower than a controller can
#: cut: it cuts a wider band than that, so the notch it cuts is wider
#: than the one predicted
NARROW_LINES = 3

#: the share of the predicted lines past which a notch is said to cover
#: most of the band, where the limit channel is in effect the control
MOST_OF_BAND = 0.5

#: how far under the reference (or under the tolerance's lower edge), as
#: a share of it, the control has to be before a line counts as notched
#: (or outside); anything closer is the rounding of two numbers that are
#: equal
_NOTCHED = 1e-9


# ---- the one-input relation and the runs ------------------------------------

def response_through(transmissibility: Any, density: Any) -> np.ndarray:
    """The autospectrum of a response, from the autospectrum of the
    channel it is measured against and the transmissibility between
    them: `|T|^2 G` at every line.

    `transmissibility` is complex (or real), `density` real, one value
    per line each. A line where either is not a number is not a number.
    It is the one-input case of the output spectral density matrix of a
    linear system, `H S H^H`, with `H` the transmissibility and `S` the
    autospectrum.
    """
    # The drive prediction proposed beside this (open as its own pull
    # request) builds `H S H^H` for matrices as `through`; on one channel
    # it is this line. When that lands, this becomes a call to it.
    t = np.asarray(transmissibility)
    g = np.real(np.asarray(density, dtype=complex)).astype(float)
    return (np.abs(t) ** 2) * g


def _first_max(values: np.ndarray) -> int:
    """The index of the largest finite value, the first of several that
    are equal to nine decimals: a flat stretch names its first line,
    not whichever line the arithmetic's last digit favored."""
    values = np.asarray(values, dtype=float)
    good = np.flatnonzero(np.isfinite(values))
    return int(good[np.argmax(np.round(values[good], 9))])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) indices of every run of adjacent True values."""
    # The drive prediction proposed beside this has the same helper as
    # `runs`; when it lands, one of the two goes.
    flags = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(flags.astype(int)))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def _density(curve: Psd, record: int, left: np.ndarray, right: np.ndarray,
             limit: str | None = None) -> np.ndarray:
    """One record of `curve` (or its limit curve `limit`) read over each
    bin from `left` to `right`: its area over the part of the bin it is
    written over, divided by that width, NaN where it is written over
    none of the bin or holds no number there."""
    from .compliance import coverage

    area = np.zeros(left.shape)
    width = np.zeros(left.shape)
    for a, b in coverage(curve, record):
        low, high = np.maximum(left, a), np.minimum(right, b)
        inside = high > low
        if not inside.any():
            continue
        got = (curve.areas(record, low[inside], high[inside]) if limit is None
               else curve.limit_areas(limit, record, low[inside],
                                      high[inside]))
        area[inside] += got
        width[inside] += high[inside] - low[inside]
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(width > 0.0, area / width, np.nan)
    return np.where(np.isfinite(out) & (out > 0.0), out, np.nan)


# ---- the reading ------------------------------------------------------------

def _level(value: float, dimension: str, us: UnitSystem | None) -> str:
    """An autospectral level in the display units, with its unit."""
    from ..units import DEFAULT_SYSTEM

    if not np.isfinite(value):
        return 'not a number'
    if dimension == UNKNOWN:
        return f'{value:.4g}'
    us = DEFAULT_SYSTEM if us is None else us
    return (f'{float(us.from_si(value, dimension)):.4g} '
            f'{us.label_text(dimension)}')


def _band(low: float, high: float) -> str:
    return (f'at {low:.4g} Hz' if low == high
            else f'from {low:.4g} to {high:.4g} Hz')


@dataclass(frozen=True)
class Exceedance:
    """One band where a limit channel's predicted response is over its
    limit profile at full level.

    `low` and `high` are the first and last lines over, in Hz, and
    `lines` how many lines that is. `frequency` is the line where it is
    furthest over, and `over_db` how far, in dB on the ASD, above zero.
    """

    channel: str
    low: float
    high: float
    lines: int
    frequency: float
    over_db: float

    def describe(self) -> str:
        """The band in words."""
        return (f'{self.over_db:.1f} dB over its limit profile at '
                f'{self.frequency:.4g} Hz, {_band(self.low, self.high)}')


@dataclass(frozen=True, eq=False)
class LimitChannel:
    """One response channel's predicted response at full level, against
    its limit profile where it has one.

    `label` is the channel's DOF and `dimension` the autospectral
    dimension both curves are in. `response` is the predicted
    autospectrum at every line of the prediction, SI, NaN where it is not
    predicted; `profile` the limit profile read at the same lines, NaN
    where the profile is not written, or None for a channel with no
    limit profile, which is predicted and does not notch. `judged` is
    False for a limited channel whose prediction says nothing (a
    transmissibility that is zero or not a number at every predicted
    line), which is left out of the notch. `exceedances` are the bands
    over the profile.
    """

    label: str
    dimension: str
    response: np.ndarray
    profile: np.ndarray | None
    judged: bool
    exceedances: tuple[Exceedance, ...] = ()

    @property
    def limited(self) -> bool:
        """Whether the channel has a limit profile."""
        return self.profile is not None

    @property
    def over_db(self) -> np.ndarray:
        """The predicted response over the profile at every line, in dB
        on the ASD: above zero where it is over, NaN where either is not
        a number (and everywhere for a channel with no profile)."""
        if self.profile is None:
            return np.full(self.response.shape, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            return 10.0 * np.log10(self.response / self.profile)

    @property
    def worst(self) -> Exceedance | None:
        """The exceedance furthest over, or None."""
        if not self.exceedances:
            return None
        return max(self.exceedances, key=lambda e: e.over_db)

    def closest(self, frequency: np.ndarray) -> tuple[float, float] | None:
        """(dB, Hz): where the predicted response comes nearest its
        profile, or furthest over it; None where nothing is compared."""
        over = self.over_db
        good = np.isfinite(over)
        if not good.any():
            return None
        k = _first_max(over)
        return float(over[k]), float(frequency[k])

    def describe(self, frequency: np.ndarray,
                 unit_system: UnitSystem | None = None) -> str:
        """The channel in a line: how far over its profile and where, or
        how close it comes; or its predicted peak with no profile."""
        if self.profile is None:
            good = np.isfinite(self.response)
            if not good.any():
                return f'{self.label}: no limit profile, and not predicted'
            k = _first_max(self.response)
            return (f'{self.label}: no limit profile, predicted only, '
                    f'highest {_level(float(self.response[k]), self.dimension, unit_system)} '
                    f'at {frequency[k]:.4g} Hz')
        if not self.judged:
            return (f'{self.label}: not judged, its transmissibility '
                    'predicts nothing')
        if self.exceedances:
            parts = [e.describe() for e in sorted(
                self.exceedances, key=lambda e: -e.over_db)]
            return f'{self.label}: ' + '; '.join(parts)
        near = self.closest(frequency)
        if near is None:
            return (f'{self.label}: its limit profile is written over none '
                    'of the predicted lines')
        return (f'{self.label}: under its limit profile everywhere, '
                f'{-near[0]:.1f} dB under at its closest, {near[1]:.4g} Hz')


@dataclass(frozen=True)
class Notch:
    """One notch the limit channels would cut in the control spectrum.

    `low` and `high` are its first and last notched lines, in Hz, and
    `lines` how many. At `frequency`, its deepest line, the control is
    `depth_db` decibels under the reference (`reference`, SI), at
    `level`: `ratio` of the reference on the ASD. `channels` are the
    limit channels that set the control anywhere in it, the one that
    sets the deepest line first. `outside` says whether the control
    falls under the tolerance's lower edge anywhere in it, at
    `outside_lines` lines and by `beyond_db` at most; None where the
    tolerance is not written over it, so nothing was judged.
    """

    low: float
    high: float
    lines: int
    frequency: float
    depth_db: float
    ratio: float
    level: float
    reference: float
    channels: tuple[str, ...]
    outside: bool | None
    outside_lines: int
    beyond_db: float

    @property
    def narrow(self) -> bool:
        """Whether it is fewer than `NARROW_LINES` lines wide, narrower
        than a controller cuts."""
        return self.lines < NARROW_LINES

    def tolerance_words(self, pair: str) -> str:
        """Whether it leaves the tolerance, in words."""
        if self.outside is None:
            return f'the {pair} tolerance is not written over it'
        if self.outside:
            return (f'outside the {pair} tolerance at {self.outside_lines} '
                    f'line{"s" * (self.outside_lines != 1)}, by up to '
                    f'{self.beyond_db:.1f} dB')
        return f'inside the {pair} tolerance'

    def describe(self, dimension: str, pair: str,
                 unit_system: UnitSystem | None = None) -> str:
        """The notch in a line: band, depth in dB and as a ratio, the
        control left, the channels and the tolerance."""
        lines = f'{self.lines} line{"s" * (self.lines != 1)}'
        return (f'notch {_band(self.low, self.high)} ({lines}): '
                f'{self.depth_db:.1f} dB deep at {self.frequency:.4g} Hz, '
                f'a ratio of {self.ratio:.3g}, the control left at '
                f'{_level(self.level, dimension, unit_system)}, for '
                f'{", ".join(self.channels)}; '
                f'{self.tolerance_words(pair)}')


@dataclass(frozen=True, eq=False)
class LimitPrediction:
    """The notch a set of limit channels would cut in a random test at
    full level, predicted from the transmissibilities from the control
    channel: what `predict_limits` returns.

    `frequency` holds the transmissibility's lines, in Hz. `reference`
    is the specification read at each, SI, NaN outside it; `control`
    the control autospectrum the controller could hold with every limit
    channel at or under its profile, NaN where a limit channel's
    transmissibility is not a number; `lower` the lower edge of the
    tolerance pair `tolerance` read at each, or None where the
    specification carries none. `control_label` names the control
    channel and `dimension` the autospectral dimension of the control's
    curves. `channels` are the response channels, limited or not;
    `notches` the notches, in frequency order; `notes` what the reading
    should be read with.
    """

    frequency: np.ndarray
    reference: np.ndarray
    control: np.ndarray
    lower: np.ndarray | None
    tolerance: str
    control_label: str
    dimension: str
    channels: tuple[LimitChannel, ...]
    notches: tuple[Notch, ...]
    notes: tuple[str, ...] = ()

    @property
    def depth_db(self) -> np.ndarray:
        """How far under the reference the control is at every line, in
        dB on the ASD: zero where there is no notch, NaN where nothing
        is predicted."""
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = 10.0 * np.log10(self.reference / self.control)
        return np.where(np.isfinite(depth), np.maximum(depth, 0.0), np.nan)

    @property
    def notched(self) -> np.ndarray:
        """True at every line inside a notch."""
        with np.errstate(invalid='ignore'):
            return np.isfinite(self.control) & (
                self.control < self.reference * (1.0 - _NOTCHED))

    @property
    def outside(self) -> np.ndarray:
        """True at every line where the control falls under the
        tolerance's lower edge."""
        if self.lower is None:
            return np.zeros(self.frequency.shape, dtype=bool)
        with np.errstate(invalid='ignore'):
            return (np.isfinite(self.control) & np.isfinite(self.lower)
                    & (self.control < self.lower * (1.0 - _NOTCHED)))

    @property
    def limited(self) -> tuple[LimitChannel, ...]:
        """The channels with a limit profile."""
        return tuple(c for c in self.channels if c.limited)

    @property
    def judged(self) -> tuple[LimitChannel, ...]:
        """The limited channels whose prediction says something."""
        return tuple(c for c in self.limited if c.judged)

    @property
    def deepest(self) -> Notch | None:
        """The deepest notch, or None."""
        if not self.notches:
            return None
        return max(self.notches, key=lambda n: n.depth_db)

    @property
    def within_tolerance(self) -> bool | None:
        """Whether every notch leaves the control inside its tolerance:
        True with no notch, None where no limit channel was judged or a
        notch's tolerance was not written."""
        if not self.judged:
            return None
        verdicts = [n.outside for n in self.notches]
        if any(v is True for v in verdicts):
            return False
        if any(v is None for v in verdicts):
            return None
        return True

    def _rms(self, values: np.ndarray) -> float:
        from .compliance import rms

        both = np.isfinite(self.reference) & np.isfinite(self.control)
        return rms(self.frequency, np.where(both, values, np.nan))

    @property
    def reference_rms(self) -> float:
        """The reference's rms over the lines the control is predicted
        on (`compliance.rms`), SI."""
        return self._rms(self.reference)

    @property
    def control_rms(self) -> float:
        """The predicted control's rms over the same lines, SI."""
        return self._rms(self.control)

    def describe(self, unit_system: UnitSystem | None = None) -> list[str]:
        """The verdict, then a line a notch and a line a channel, then
        the notes; the lines carry no full stop."""
        lines = [self.verdict()]
        lines += [n.describe(self.dimension, self.tolerance, unit_system)
                  for n in self.notches]
        lines += [c.describe(self.frequency, unit_system)
                  for c in self.channels]
        if self.notches and np.isfinite(self.reference_rms):
            ratio = 20.0 * np.log10(self.control_rms / self.reference_rms)
            lines.append(f'the control rms {ratio:+.2f} dB from the '
                         "reference's over the predicted lines")
        return lines + list(self.notes)

    def verdict(self) -> str:
        """The prediction in one line."""
        count = len(self.limited)
        which = ('the limit channel' if count == 1
                 else f'{count} limit channels')
        if not self.judged:
            return (f'nothing judged: {"its" if count == 1 else "every"} '
                    'transmissibility predicts nothing, so no notch can be '
                    'said to be needed or not')
        if not self.notches:
            near = [(c.closest(self.frequency), c.label) for c in self.judged]
            near = [(n, label) for n, label in near if n is not None]
            if not near:
                return ('no notch: no limit profile is written over the '
                        'predicted lines')
            (margin, at), label = max(near, key=lambda item: item[0][0])
            stays = ('stays under its limit profile' if count == 1
                     else 'stay under their limit profiles')
            return (f'no notch: {which} {stays} at full level, '
                    f'{label} closest, {-margin:.1f} dB under at '
                    f'{at:.4g} Hz')
        deepest = self.deepest
        assert deepest is not None
        count = len(self.notches)
        head = (f'{count} notch{"es" * (count != 1)} in the control at full '
                f'level, the deepest {deepest.depth_db:.1f} dB at '
                f'{deepest.frequency:.4g} Hz for {deepest.channels[0]}')
        if self.lower is None:
            return (f'{head}; the specification carries no lower '
                    f'{self.tolerance} limit, so the tolerance is not judged')
        verdict = self.within_tolerance
        if verdict is True:
            return f'{head}; the control stays inside the {self.tolerance} tolerance'
        if verdict is False:
            out = sum(n.outside is True for n in self.notches)
            which = ('it takes' if count == 1 else
                     f'{out} of them take{"s" * (out == 1)}')
            return (f'{head}; {which} the control outside the '
                    f'{self.tolerance} tolerance')
        return (f'{head}; the {self.tolerance} tolerance is not written '
                'over every notch')

    def table(self, unit_system: UnitSystem | None = None
              ) -> list[dict[str, str]]:
        """A row a notch, as text: 'band' (Hz), 'depth' (dB and ratio),
        'level' (the control left at the deepest line), 'channels' and
        'tolerance' ('inside', 'outside' or 'not judged'), with
        'describe', the whole notch in words."""
        rows = []
        for n in self.notches:
            rows.append({
                'band': (f'{n.low:.4g}' if n.low == n.high
                         else f'{n.low:.4g}–{n.high:.4g}'),
                'depth': f'{n.depth_db:.1f} ({n.ratio:.2g})',
                'level': _level(n.level, self.dimension, unit_system),
                'channels': ', '.join(n.channels),
                'tolerance': ('not judged' if n.outside is None
                              else 'outside' if n.outside else 'inside'),
                'describe': n.describe(self.dimension, self.tolerance,
                                       unit_system)})
        return rows


# ---- the prediction -----------------------------------------------------------

def _autospectra(curve: Psd) -> dict[str, list[int]]:
    """The records of `curve` that are autospectra, by response DOF."""
    out: dict[str, list[int]] = {}
    for i in range(curve.num_records):
        reference = (None if curve.reference_dof is None
                     else str(curve.reference_dof[i]))
        dof = str(curve.response_dof[i])
        if reference in (None, '', dof):
            out.setdefault(dof, []).append(i)
    return out


def _quantity(dimension: str) -> str:
    from .data import channel_quantities

    return channel_quantities(str(dimension))[0]


def _record_at(curve: Psd, dof: str, quantity: str) -> int | None:
    """The autospectrum of `curve` at `dof` that measures `quantity`,
    preferring an exact match, then one of undeclared quantity, then any
    (which the units check then refuses with the reason)."""
    found = _autospectra(curve).get(dof, [])
    if not found:
        return None
    exact = [i for i in found if _quantity(curve.ordinate_dim[i]) == quantity]
    loose = [i for i in found
             if UNKNOWN in (_quantity(curve.ordinate_dim[i]), quantity)]
    return (exact or loose or found)[0]


def predict_limits(transmissibility: Frf, specification: Specification,
                   profiles: Psd | None = None, *,
                   tolerance: str = 'abort') -> LimitPrediction:
    """Predict, before the test, where each limit channel would exceed
    its limit profile at full level and the notch the controller would
    cut in the control spectrum to hold it there.

    `transmissibility` holds one record per response channel, each the
    response over the control channel (H1 from a low-level run, or any
    FRF from the control channel's quantity), all against the one
    control channel. `specification` holds the reference at that
    control channel, with its tolerance. `profiles` holds the limit
    profiles, one autospectrum per limit channel at that channel's DOF
    and quantity; when omitted, they are the specification's own
    records at the response channels (a specification may carry them
    beside the reference, with no tolerance of their own). A response
    channel with no limit profile is predicted and does not notch.
    `tolerance` is the pair, 'abort' or 'warning', whose lower edge the
    notched control is read against. The module docstring sets out the
    method.

    Parameters
    ----------
    transmissibility : Frf
        The response channels over the control channel.
    specification : Specification
        The reference at the control channel, and its tolerance.
    profiles : Psd, optional
        The limit profiles; the specification's records when omitted.
    tolerance : {'abort', 'warning'}, default 'abort'
        Which tolerance the notch is read against.

    Returns
    -------
    LimitPrediction
        The predicted responses, the notched control, the notches and
        the notes.

    Raises
    ------
    ValueError
        If the transmissibility holds no record, fewer than two lines,
        records against more than one channel, or two records to one
        channel; if the specification holds no autospectrum at the
        control channel; if a pair's quantities cannot be compared
        (`compliance.units_check`) or a curve is on octave bands
        (`compliance.comparable`); if the specification is written over
        none of the lines; or if no response channel has a limit profile.
    """
    from .compliance import comparable, units_check
    from .data import Psd, channel_quantities
    from .octave import bin_bounds

    tolerance = str(tolerance)
    if tolerance not in TOLERANCES:
        raise ValueError(f'the tolerance read against is '
                         f'{" or ".join(map(repr, TOLERANCES))}, not '
                         f'{tolerance!r}')
    if transmissibility.num_records == 0:
        raise ValueError('the transmissibility holds no record to predict from')
    f = np.asarray(transmissibility.abscissa, dtype=float)
    if f.size < 2:
        raise ValueError('a prediction needs at least two lines')
    if transmissibility.reference_dof is None:
        raise ValueError('the transmissibility names no control channel it '
                         'is taken against')
    references = {(str(transmissibility.reference_dof[i]),
                   channel_quantities(str(transmissibility.ordinate_dim[i]))[1])
                  for i in range(transmissibility.num_records)}
    if len({dof for dof, _q in references}) > 1:
        names = ' and '.join(sorted({dof for dof, _q in references}))
        raise ValueError(f'the transmissibilities are taken against {names}; '
                         'a prediction from one control channel needs them '
                         'all against the same one')
    if len(references) > 1:
        raise ValueError('the transmissibilities are taken against the '
                         'control channel in more than one quantity')
    (control, control_quantity), = references

    # the reference, at the control channel
    record = _record_at(specification, control, control_quantity)
    if record is None:
        raise ValueError(f'the specification has no autospectrum at '
                         f'{control}, the channel the transmissibilities are '
                         'taken against')
    spec_quantity = _quantity(specification.ordinate_dim[record])
    refused, raw = units_check(spec_quantity, control_quantity,
                               wanted_name='the specification',
                               held_name='the control channel of the '
                                         'transmissibility')
    if refused is not None:
        raise ValueError(refused)
    carrier = Psd(f, np.ones((1, f.size)), response_dof=[control],
                  ordinate_dim=[UNKNOWN])
    refused = comparable(specification, carrier)
    if refused is not None:
        raise ValueError(refused)
    left, right = bin_bounds(f, None)
    reference = _density(specification, record, left, right)
    if not np.isfinite(reference).any():
        raise ValueError('the specification is written over none of the '
                         "transmissibility's lines")
    edge = f'{tolerance}_lower'
    lower = (_density(specification, record, left, right, edge)
             if edge in getattr(specification, 'limits', {}) else None)
    if lower is not None and not np.isfinite(lower[np.isfinite(reference)]).any():
        lower = None

    # the response channels, each with its limit profile where it has one
    source = specification if profiles is None else profiles
    if profiles is not None:
        refused = comparable(profiles, carrier)
        if refused is not None:
            raise ValueError(refused)
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, np.ndarray, int | None]] = []
    for i in range(transmissibility.num_records):
        dof = str(transmissibility.response_dof[i])
        quantity = _quantity(transmissibility.ordinate_dim[i])
        if dof == control and quantity == control_quantity:
            continue                     # the control over itself
        if (dof, quantity) in seen:
            raise ValueError(f'the transmissibility holds two records to '
                             f'{dof}; keep one')
        seen.add((dof, quantity))
        found = _record_at(source, dof, quantity)
        if found is not None:
            refused, _raw = units_check(
                _quantity(source.ordinate_dim[found]), quantity,
                wanted_name=f'the limit profile at {dof}',
                held_name='the transmissibility to it')
            if refused is not None:
                raise ValueError(refused)
        rows.append((dof, quantity,
                     np.asarray(transmissibility.ordinate[i]), found))
    if not any(found is not None for *_r, found in rows):
        names = ', '.join(dof for dof, *_r in rows) or 'none'
        where = ('the specification' if profiles is None
                 else 'the limit profiles')
        raise ValueError(f'no response channel the transmissibilities reach '
                         f'({names}) has a limit profile in {where}, so '
                         'nothing can notch; a limit profile is an '
                         "autospectrum at its channel's DOF")

    notes: list[str] = []
    if raw is not None:
        notes.append(raw)
    predicted = np.isfinite(reference)
    allowed_rows: list[np.ndarray] = []
    labels: list[str] = []
    channels: list[LimitChannel] = []
    for dof, quantity, t, found in rows:
        dimension = (UNKNOWN if quantity == UNKNOWN
                     else f'{quantity}**2/frequency')
        squared = np.abs(t) ** 2
        response = np.where(predicted, response_through(t, reference), np.nan)
        if found is None:
            channels.append(LimitChannel(dof, dimension, response, None, True))
            continue
        profile = _density(source, found, left, right)
        finite = predicted & np.isfinite(squared)
        judged = bool((finite & (squared > 0.0)).any())
        if not np.isfinite(squared[predicted]).all():
            bad = int((predicted & ~np.isfinite(squared)).sum())
            notes.append(
                f'the transmissibility to {dof} is not a number at {bad} of '
                f'the {int(predicted.sum())} predicted lines: the record '
                'holds a value that is not one, and the control is not '
                'predicted there')
        elif not judged:
            notes.append(f'the transmissibility to {dof} is zero at every '
                         'predicted line: a channel that recorded nothing '
                         'predicts no response, so its limit is not judged')
        unlimited = predicted & ~np.isfinite(profile)
        if judged and unlimited.any() and np.isfinite(profile).any():
            notes.append(f'the limit profile at {dof} is not written at '
                         f'{int(unlimited.sum())} of the predicted lines, and '
                         'nothing limits it there')
        with np.errstate(divide='ignore', invalid='ignore'):
            allowed = np.where(np.isfinite(profile) & (squared > 0.0),
                               profile / squared, np.inf)
        allowed = np.where(np.isfinite(squared), allowed, np.nan)
        # over by more than the rounding, the same margin the notch is
        # found by, so a channel exactly on its profile is not over it
        over = (predicted & np.isfinite(profile)
                & (response > profile * (1.0 + _NOTCHED)))
        exceedances = []
        with np.errstate(divide='ignore', invalid='ignore'):
            over_db = 10.0 * np.log10(response / profile)
        for first, last in _runs(over):
            k = first + _first_max(over_db[first:last + 1])
            exceedances.append(Exceedance(
                dof, float(f[first]), float(f[last]), last - first + 1,
                float(f[k]), float(over_db[k])))
        channels.append(LimitChannel(dof, dimension, response, profile,
                                     judged, tuple(exceedances)))
        if judged:
            allowed_rows.append(allowed)
            labels.append(dof)

    # the control the controller could hold: the lowest anything allows
    controlled = np.where(predicted, reference, np.nan)
    setter = np.full(f.shape, -1)
    if allowed_rows:
        stack = np.vstack(allowed_rows)
        with np.errstate(invalid='ignore'):
            lowest = np.min(stack, axis=0)       # NaN where any is NaN
        lowest_at = np.argmin(np.where(np.isnan(stack), np.inf, stack), axis=0)
        cut = predicted & np.isfinite(lowest) & (lowest < controlled)
        controlled = np.where(np.isnan(lowest) & predicted, np.nan,
                              np.where(cut, lowest, controlled))
        setter = np.where(cut, lowest_at, -1)

    dimension = str(specification.ordinate_dim[record])
    with np.errstate(invalid='ignore'):
        notched = np.isfinite(controlled) & (
            controlled < reference * (1.0 - _NOTCHED))
    depth = np.full(f.shape, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        depth[notched] = 10.0 * np.log10(reference[notched]
                                         / controlled[notched])
        below = (np.isfinite(lower) & np.isfinite(controlled)
                 & (controlled < lower * (1.0 - _NOTCHED))
                 ) if lower is not None else None
    notches = []
    for first, last in _runs(notched):
        span = slice(first, last + 1)
        k = first + _first_max(depth[span])
        acting = [labels[setter[k]]] + [
            labels[s] for s in setter[span] if s >= 0]
        names = tuple(dict.fromkeys(acting))
        outside: bool | None = None
        count, beyond = 0, float('nan')
        if lower is not None and np.isfinite(lower[span]).any():
            hit = below[span]
            count = int(hit.sum())
            outside = bool(count)
            if outside:
                with np.errstate(divide='ignore'):
                    beyond = float(np.max(10.0 * np.log10(
                        lower[span][hit] / controlled[span][hit])))
        notches.append(Notch(
            low=float(f[first]), high=float(f[last]),
            lines=last - first + 1, frequency=float(f[k]),
            depth_db=float(depth[k]),
            ratio=float(controlled[k] / reference[k]),
            level=float(controlled[k]), reference=float(reference[k]),
            channels=names, outside=outside, outside_lines=count,
            beyond_db=beyond))

    narrow = [n for n in notches if n.narrow]
    if narrow:
        where = ', '.join(f'{n.frequency:.4g} Hz' for n in narrow)
        notes.append(f'{"a notch" if len(narrow) == 1 else "notches"} '
                     f'fewer than {NARROW_LINES} lines wide ({where}): a '
                     'controller cuts a wider band than that, so the notch '
                     'cut will be wider than predicted')
    total = int(np.isfinite(controlled).sum())
    if total and notched.sum() > MOST_OF_BAND * total:
        notes.append(f'the notches cover {100.0 * notched.sum() / total:.0f} '
                     '% of the predicted lines: the limit channels are in '
                     'effect the control, and whether the test as written '
                     'exists on this setup is a question for the '
                     "requirement's owner before the run")
    if lower is None and notches:
        notes.append(f'the specification carries no lower {tolerance} limit '
                     f'at {control}, so no notch is read against a tolerance')
    return LimitPrediction(
        frequency=f, reference=np.where(predicted, reference, np.nan),
        control=controlled, lower=lower, tolerance=tolerance,
        control_label=control, dimension=dimension,
        channels=tuple(channels), notches=tuple(notches), notes=tuple(notes))
