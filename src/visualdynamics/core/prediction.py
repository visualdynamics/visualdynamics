"""Responses against prediction: each response channel read against
the response its analyst predicted, scaled to the level run.

A `Prediction` is one response channel's predicted response, as the
analyst gave it: a list of peaks (frequency, level and Q) or a predicted
curve (an ASD, an amplitude or a transmissibility) with the frequencies
of the modes it is to be compared at; the input level it was predicted
at; its source; and the model's stated uncertainty. A `Predictions`
object holds a test's predictions, one per channel, and is saved in the
project.

`compare` reads a measured object against them. The prediction is
scaled to the level run: a level of L dB relative to the level it was
predicted at multiplies an ASD by 10^(L/10) and an amplitude by
10^(L/20) (`scale_factor`). A transmissibility is a ratio of response
to input, the same at every level for a linear article, and is not
scaled. Near each predicted peak, inside its band, the measured peak is
read by the package's half-power reader (`core.half_power`), and each
row gives the frequency, level and Q measured and predicted and their
ratio, named as an ASD ratio or an amplitude ratio and given in
decibels, so the decibels cannot be read two ways.

What a difference means is not the program's to say. The table states
the numbers and the ratio beside the uncertainty the model stated;
whether the response is as predicted, different and explained, or
different and not explained is the analyst's answer.

What cannot be read one way is refused, with the cause named:

- a prediction of one quantity against a record of another (an ASD
  against a transmissibility), or in units of another dimension;
- an ASD or an amplitude whose input level, or whose run level, is not
  stated (a transmissibility takes neither: a level would do nothing);
- a peak with no band to look in, from neither its own band nor the
  model's stated frequency uncertainty;
- two peaks of one channel whose bands overlap, since one measured peak
  inside both could be read as either.

A predicted peak whose band holds no measured peak says so: a peak is
never matched to one outside its band. A band runs from the predicted
frequency less its stated uncertainty to the frequency plus it, unless
the analyst states a band for the peak (to compare a mode the model
placed far from where it was found, say).

The level the record was run at is taken in one place, `level_run`:
the level stated for the comparison, or the scaling held on a PSD
(`Psd.scale_db`, the decibels it is scaled up by to meet its
specification, so a run at -12 dB holds +12), refused when the two
disagree.

References
----------
1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The response of a linear
   system to a stationary random input: the output spectral density is
   the input spectral density times the squared gain, so it scales with
   the input's density, and the half-power bandwidth of a resonance.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .half_power import HalfPower, _figures, read_half_power

#: the quantities a prediction can be of
QUANTITIES = ('asd', 'amplitude', 'transmissibility')

#: what the ratio of a measured level to a predicted one is called, by
#: quantity: an ASD is a power, a transmissibility an amplitude ratio
RATIO = {'asd': 'ASD ratio', 'amplitude': 'amplitude ratio',
         'transmissibility': 'amplitude ratio'}

#: decibels per decade of ratio: 10 for a power, 20 for an amplitude
DECIBELS = {'asd': 10.0, 'amplitude': 20.0, 'transmissibility': 20.0}

#: what each quantity is called in a sentence
NOUN = {'asd': 'an ASD', 'amplitude': 'an amplitude',
        'transmissibility': 'a transmissibility'}


def scale_factor(quantity: str, level_db: float) -> float:
    """What a prediction is multiplied by to read at `level_db` decibels
    relative to the level it was predicted at.

    10^(L/10) on an ASD, a power; 10^(L/20) on an amplitude. A
    transmissibility is a ratio of response to input and is not scaled,
    so a level for one is refused.

    Raises
    ------
    ValueError
        For an unknown quantity, a level that is not a finite number,
        or any level for a transmissibility.
    """
    _quantity(quantity)
    if quantity == 'transmissibility':
        raise ValueError('a transmissibility is a ratio of response to '
                         'input and is not scaled by a level')
    level = _finite(level_db, 'the level')
    return 10.0 ** (level / DECIBELS[quantity])


def _quantity(quantity: Any) -> str:
    if quantity not in QUANTITIES:
        raise ValueError(f'a prediction is of {", ".join(QUANTITIES)}, '
                         f'not {quantity!r}')
    return quantity


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float,
                                                         np.number)):
        raise TypeError(f'{what} is a number, not {type(value).__name__}')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{what} is {value}, not a finite number')
    return value


def _positive(value: Any, what: str) -> float:
    value = _finite(value, what)
    if value <= 0.0:
        raise ValueError(f'{what} is {value:g}; it must be above zero')
    return value


@dataclass(frozen=True)
class PredictedPeak:
    """One predicted peak of a response channel.

    `frequency` is in Hz. `level` is the predicted level at the peak, in
    the prediction's unit (for an ASD, that unit squared per hertz) and
    at the prediction's input level, and `q` its predicted Q (None when
    the model does not state one). A peak of a predicted curve names
    its frequency only: its level and Q are read off the curve. `band`,
    (low, high) in Hz, is where its measured counterpart is looked for;
    None takes the band from the model's stated frequency uncertainty.
    """

    frequency: float
    level: float | None = None
    q: float | None = None
    band: tuple[float, float] | None = None


def _peak_of(value: Any) -> PredictedPeak:
    """A peak from a `PredictedPeak`, a frequency alone, a (frequency,
    level[, q[, band]]) tuple or a mapping of its fields."""
    if isinstance(value, PredictedPeak):
        return value
    if isinstance(value, Mapping):
        unknown = set(value) - {'frequency', 'level', 'q', 'band'}
        if unknown:
            raise ValueError(f'a predicted peak has no field '
                             f'{min(unknown)!r}; its fields are '
                             'frequency, level, q and band')
        return PredictedPeak(**value)
    if isinstance(value, (int, float)):
        return PredictedPeak(value)
    try:
        parts = tuple(value)
    except TypeError:
        raise TypeError(f'a predicted peak is (frequency, level, q, band), '
                        f'not {type(value).__name__}') from None
    if not 1 <= len(parts) <= 4:
        raise ValueError(f'a predicted peak is (frequency, level, q, band); '
                         f'{len(parts)} values were given')
    return PredictedPeak(*parts)


def _peak(frequencies: np.ndarray, magnitude: np.ndarray,
          band: tuple[float, float], what: str
          ) -> tuple[HalfPower | None, str]:
    """The peak of `magnitude` (an amplitude, never a power) inside
    `band`, read by the half-power reader, and ''. (None, why) when the
    band holds no peak: no line in it, a line with no value, or a curve
    still rising at an edge of the band, whose peak, if it has one, is
    outside it.
    """
    low, high = band
    inside = np.flatnonzero((frequencies >= low) & (frequencies <= high))
    span = f'{_figures(low, 4)} and {_figures(high, 4)} Hz'
    if not len(inside):
        return None, f'no line of {what} lies between {span}'
    empty = inside[~np.isfinite(magnitude[inside])]
    if len(empty):
        return None, (f'{what} has no value at '
                      f'{_figures(frequencies[empty[0]], 4)} Hz, inside the '
                      f'band {span}')
    top = int(inside[np.argmax(magnitude[inside])])
    if top in (inside[0], inside[-1]):
        return None, f'no peak of {what} between {span}'
    return read_half_power(frequencies, magnitude, frequencies[top],
                           within=band, warn=False), ''


@dataclass(frozen=True, eq=False)
class Prediction:
    """One response channel's predicted response.

    Parameters
    ----------
    channel : str
        The response DOF the prediction is for, as the records name it
        ('201Z+').
    quantity : str
        'asd', 'amplitude' or 'transmissibility'.
    peaks : sequence
        The predicted peaks: `PredictedPeak`, (frequency, level[, q])
        tuples or mappings. At least one.
    unit : str, optional
        The unit of the quantity the levels are in, the way a PSD
        declares it: 'g' for an ASD in g²/Hz, 'g' for an amplitude in
        g. Required for an ASD or an amplitude; a transmissibility has
        none.
    level_db : float, optional
        The input level the prediction is stated at, in dB relative to
        full level (0 for a full-level prediction). Required for an ASD
        or an amplitude; a transmissibility takes none.
    curve : (frequencies, values), optional
        The predicted curve, in the same unit; its peaks' levels and Q
        are read off it, so a peak of a curve names its frequency only.
    reference : str, optional
        The input DOF of a transmissibility ('1Z+'); required for one,
        refused for anything else.
    source : str, optional
        Where the prediction comes from: the analysis, its document and
        revision.
    customer : bool, default False
        Whether the customer supplied it.
    frequency_uncertainty : float, optional
        The model's stated frequency uncertainty, plus or minus, in
        percent. It sets each peak's band unless the peak states one.
    level_uncertainty : float, optional
        The model's stated level uncertainty, plus or minus, in dB.

    Raises
    ------
    ValueError
        For what cannot be read one way: see the module's notes.
    TypeError
        When a field is of the wrong type.
    """

    channel: str
    quantity: str
    peaks: tuple[PredictedPeak, ...] = ()
    unit: str | None = None
    level_db: float | None = None
    curve: tuple[np.ndarray, np.ndarray] | None = None
    reference: str | None = None
    source: str = ''
    customer: bool = False
    frequency_uncertainty: float | None = None
    level_uncertainty: float | None = None
    #: the bands each peak is looked for in, in order (set on creation)
    bands: tuple[tuple[float, float], ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        from .validate import dofs

        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError('a prediction names its channel, the response '
                             'DOF the records name')
        object.__setattr__(self, 'channel',
                           dofs(self.channel.strip(), 'the channel')[0])
        quantity = _quantity(self.quantity)
        what = f'the prediction for {self.channel}'
        if not isinstance(self.source, str):
            raise TypeError(f'{what}: the source is a str, not '
                            f'{type(self.source).__name__}')
        object.__setattr__(self, 'customer', bool(self.customer))
        if quantity == 'transmissibility':
            if self.unit is not None:
                raise ValueError(f'{what}: a transmissibility is a ratio '
                                 'and takes no unit')
            if self.level_db is not None:
                raise ValueError(f'{what}: a transmissibility is a ratio of '
                                 'response to input, the same at any level, '
                                 'and takes no input level')
            if not isinstance(self.reference, str) or \
                    not self.reference.strip():
                raise ValueError(f'{what}: a transmissibility names its '
                                 'input DOF (reference)')
            object.__setattr__(self, 'reference', dofs(
                self.reference.strip(), 'the reference')[0])
        else:
            if self.reference is not None:
                raise ValueError(f'{what}: only a transmissibility has a '
                                 'reference DOF')
            if not isinstance(self.unit, str) or self.dimension is None:
                raise ValueError(f'{what}: {NOUN[quantity]} states the unit '
                                 "of its quantity ('g' for g²/Hz or g); "
                                 f'{self.unit!r} cannot be read as one')
            if self.level_db is None:
                raise ValueError(f'{what}: the input level the prediction '
                                 'is stated at is not given (level_db, in '
                                 'dB relative to full level; 0 for full '
                                 'level)')
            object.__setattr__(self, 'level_db',
                               _finite(self.level_db, f'{what}: the level'))
        for name in ('frequency_uncertainty', 'level_uncertainty'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(
                    value, f'{what}: the {name.replace("_", " ")}'))
        if (self.frequency_uncertainty is not None
                and self.frequency_uncertainty >= 100.0):
            raise ValueError(f'{what}: a frequency uncertainty of '
                             f'{self.frequency_uncertainty:g} percent leaves '
                             'no band above zero')
        self._take_curve(what)
        if isinstance(self.peaks, (str, bytes)):
            raise TypeError(f'{what}: peaks is a list of peaks, not a str')
        peaks = tuple(_peak_of(peak) for peak in (self.peaks or ()))
        if not peaks:
            raise ValueError(f'{what} names no peak: give each predicted '
                             "peak, or for a curve each mode's frequency")
        checked = tuple(self._check(peak, what) for peak in peaks)
        object.__setattr__(self, 'peaks', checked)
        bands = tuple(self._band(peak, what) for peak in checked)
        order = sorted(range(len(bands)), key=lambda k: bands[k][0])
        for a, b in itertools.pairwise(order):
            if bands[b][0] <= bands[a][1]:
                low, high = sorted((checked[a].frequency,
                                    checked[b].frequency))
                raise ValueError(
                    f'{what}: the bands of the peaks at '
                    f'{_figures(low, 4)} and {_figures(high, 4)} Hz '
                    'overlap, so a '
                    'measured peak inside both could be read as either; '
                    'state narrower bands')
        object.__setattr__(self, 'bands', bands)
        if self.curve is not None:
            for peak, band in zip(checked, bands, strict=True):
                found, why = _peak(self.curve[0],
                                   self._magnitude(self.curve[1]), band,
                                   'the predicted curve')
                if found is None:
                    raise ValueError(f'{what}: at the mode named at '
                                     f'{_figures(peak.frequency, 4)} Hz, '
                                     f'{why}')

    def _take_curve(self, what: str) -> None:
        if self.curve is None:
            return
        try:
            frequencies, values = self.curve
        except (TypeError, ValueError):
            raise TypeError(f'{what}: the curve is (frequencies, values)') \
                from None
        f = np.asarray(frequencies, dtype=np.float64).ravel()
        v = np.asarray(values, dtype=np.float64).ravel()
        if len(f) != len(v):
            raise ValueError(f'{what}: the curve has {len(f)} frequencies '
                             f'for {len(v)} values')
        if len(f) < 3:
            raise ValueError(f'{what}: a curve needs at least three lines')
        if not (np.all(np.isfinite(f)) and np.all(np.isfinite(v))):
            raise ValueError(f'{what}: the curve holds a value that is not '
                             'a finite number')
        if not np.all(np.diff(f) > 0.0) or f[0] < 0.0:
            raise ValueError(f'{what}: the curve\'s frequencies must be '
                             'ascending and not below zero')
        if np.any(v < 0.0):
            raise ValueError(f'{what}: the curve holds a level below zero')
        f.flags.writeable = v.flags.writeable = False
        object.__setattr__(self, 'curve', (f, v))

    def _check(self, peak: PredictedPeak, what: str) -> PredictedPeak:
        frequency = _positive(peak.frequency, f'{what}: a peak frequency')
        at = f'{what}: the peak at {_figures(frequency, 4)} Hz'
        if self.curve is not None:
            if peak.level is not None or peak.q is not None:
                raise ValueError(f'{at} gives a level or Q beside a curve; '
                                 'they are read off the curve, so give '
                                 'one or the other')
            level = q = None
        else:
            if peak.level is None:
                raise ValueError(f'{at} gives no level')
            level = _positive(peak.level, f'{at}: the level')
            q = None if peak.q is None else _positive(peak.q, f'{at}: Q')
        band = None
        if peak.band is not None:
            try:
                low, high = peak.band
            except (TypeError, ValueError):
                raise TypeError(f'{at}: the band is (low, high) in Hz') \
                    from None
            low = _positive(low, f'{at}: the band\'s low edge')
            high = _finite(high, f'{at}: the band\'s high edge')
            if not low < frequency < high:
                raise ValueError(f'{at}: the band {_figures(low, 4)} to '
                                 f'{_figures(high, 4)} Hz does not hold the '
                                 'peak')
            band = (low, high)
        return PredictedPeak(frequency, level, q, band)

    def _band(self, peak: PredictedPeak, what: str) -> tuple[float, float]:
        if peak.band is not None:
            return peak.band
        if self.frequency_uncertainty is None:
            raise ValueError(
                f'{what}: no band to look for the peak at '
                f'{_figures(peak.frequency, 4)} Hz in; state the model\'s '
                'frequency uncertainty, or a band for the peak')
        share = self.frequency_uncertainty / 100.0
        return (peak.frequency * (1.0 - share),
                peak.frequency * (1.0 + share))

    @property
    def dimension(self) -> str | None:
        """The dimension of the quantity the levels are in
        ('acceleration'), or None for a transmissibility or a unit that
        cannot be read."""
        # a transmissibility has no unit (refused on creation)
        if not isinstance(self.unit, str):
            return None
        from ..units import dimension_of

        return dimension_of(self.unit)

    @property
    def record_dimension(self) -> str | None:
        """The `ordinate_dim` a record compared with this prediction
        has: 'acceleration**2/frequency' for an ASD of acceleration,
        'acceleration' for an amplitude, None for a transmissibility
        (any quantity over itself)."""
        if self.quantity == 'asd':
            return f'{self.dimension}**2/frequency'
        return self.dimension

    def si(self, values: Any) -> Any:
        """Levels in the prediction's unit, in SI (the unit squared per
        hertz for an ASD; a transmissibility unchanged)."""
        if self.quantity == 'transmissibility':
            return values
        from ..units import si_factor

        factor = si_factor(str(self.unit), self.dimension)
        return values * factor ** (2 if self.quantity == 'asd' else 1)

    def own(self, values: Any) -> Any:
        """SI levels in the prediction's unit: the inverse of `si`."""
        return values / self.si(1.0)

    @property
    def unit_label(self) -> str:
        """The unit the levels are written in, as text: 'g²/Hz', 'g',
        or '' for a transmissibility."""
        if self.quantity == 'transmissibility':
            return ''
        from ..units import pretty_unit

        unit = pretty_unit(str(self.unit))
        return f'{unit}²/Hz' if self.quantity == 'asd' else unit

    def _magnitude(self, values: np.ndarray) -> np.ndarray:
        """`values` as the amplitude the half-power reader reads: the
        square root of an ASD, whose half power is half its height."""
        return _amplitude(np.asarray(values, dtype=np.float64),
                          self.quantity)

    def as_record(self) -> dict[str, Any]:
        """The prediction as plain values (the curve left out), what a
        file stores; `from_record` reads it back. A field that is not
        stated is None, and a peak is [frequency, level, Q, band]."""
        return {'channel': self.channel, 'quantity': self.quantity,
                'unit': self.unit, 'level_db': self.level_db,
                'reference': self.reference, 'source': self.source,
                'customer': self.customer,
                'frequency_uncertainty': self.frequency_uncertainty,
                'level_uncertainty': self.level_uncertainty,
                'peaks': [[peak.frequency, peak.level, peak.q,
                           None if peak.band is None else list(peak.band)]
                          for peak in self.peaks]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any],
                    curve: tuple[Any, Any] | None = None) -> Prediction:
        """A prediction from `as_record`'s values and its curve."""
        fields = dict(record)
        peaks = [PredictedPeak(frequency, level, q,
                               None if band is None else tuple(band))
                 for frequency, level, q, band in fields.pop('peaks')]
        return cls(peaks=peaks, curve=curve, **fields)

    @property
    def key(self) -> tuple[str, str, str | None]:
        """What makes two predictions the same one: the channel, the
        quantity and, for a transmissibility, the input DOF."""
        return (self.channel, self.quantity, self.reference)

    def describe(self) -> str:
        """The prediction in one line, in its own unit: '201Z+, ASD in
        g²/Hz at 0 dB: 385 Hz 0.5 (Q 12); from the stress analysis
        (customer-supplied)'."""
        unit = f' in {self.unit_label}' if self.unit_label else ''
        level = (f' at {self.level_db:+g} dB' if self.level_db is not None
                 else f' from {self.reference}')
        noun = 'ASD' if self.quantity == 'asd' else self.quantity
        peaks = []
        for peak in self.peaks:
            text = f'{_figures(peak.frequency, 4)} Hz'
            if peak.level is not None:
                text += f' {_figures(peak.level, 3)}'
            if peak.q is not None:
                text += f' (Q {_figures(peak.q, 3)})'
            peaks.append(text)
        tail = ''
        if self.source:
            tail = f'; from {self.source}'
        if self.customer:
            tail += ' (customer-supplied)'
        return (f'{self.channel}, {noun}{unit}{level}: '
                f'{", ".join(peaks)}{tail}')


def _amplitude(values: np.ndarray, quantity: str) -> np.ndarray:
    magnitude = np.abs(values) if np.iscomplexobj(values) else values
    if quantity != 'asd':
        return np.abs(magnitude)
    with np.errstate(invalid='ignore'):
        # a density below zero is no measurement: no value, said so
        return np.where(magnitude >= 0.0, np.sqrt(np.abs(magnitude)),
                        np.nan)


class Predictions:
    """A test's predictions, one per response channel (and quantity, and
    for a transmissibility input DOF).

    Parameters
    ----------
    predictions : iterable of Prediction, optional
        The predictions.
    comment : str, optional
        What the set is: the axis or the run setup it predicts.

    Raises
    ------
    ValueError
        When two predictions are for the same channel and quantity.
    """

    def __init__(self, predictions: Iterable[Prediction] = (),
                 comment: str = '') -> None:
        self.predictions: list[Prediction] = []
        self.comment: str = str(comment)
        for prediction in predictions:
            if not isinstance(prediction, Prediction):
                raise TypeError(f'a Predictions object holds Prediction, '
                                f'not {type(prediction).__name__}')
            if any(held.key == prediction.key for held in self.predictions):
                raise ValueError(f'two predictions for {prediction.channel} '
                                 f'as {NOUN[prediction.quantity]}; one each')
            self.predictions.append(prediction)

    def __len__(self) -> int:
        return len(self.predictions)

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self.predictions)

    def __repr__(self) -> str:
        return f'Predictions({len(self)} channels)'

    @property
    def channels(self) -> list[str]:
        """The channels predicted, in order."""
        return [prediction.channel for prediction in self.predictions]

    def put(self, prediction: Prediction) -> None:
        """Add `prediction`, replacing the one for the same channel and
        quantity (and input DOF) in its place."""
        if not isinstance(prediction, Prediction):
            raise TypeError(f'a Predictions object holds Prediction, not '
                            f'{type(prediction).__name__}')
        for k, held in enumerate(self.predictions):
            if held.key == prediction.key:
                self.predictions[k] = prediction
                return
        self.predictions.append(prediction)

    def compare(self, measured: Any, level_db: float | None = None
                ) -> PredictionComparison:
        """`compare` against this set."""
        return compare(self, measured, level_db)


def level_run(measured: Any, stated: float | None = None
              ) -> tuple[float | None, str]:
    """The level `measured` was run at, in dB relative to full level, and
    where it was read: 'stated' or 'held on the record'.

    The level `stated` for the comparison when one is; otherwise the
    scaling a PSD holds for its comparison against its specification
    (`Psd.scale_db`: a run at -12 dB is scaled up by 12, so it holds
    +12); otherwise (None, ''). This is the one place a comparison takes
    its level from, so a later source of run levels is read here.

    Raises
    ------
    ValueError
        When the level stated and the scaling held on the record
        disagree, or the level is not a finite number.
    """
    from .data import Psd

    held = getattr(measured, 'scale_db', None) if isinstance(
        measured, Psd) else None
    if stated is not None:
        stated = _finite(stated, 'the level run')
        if held is not None and not math.isclose(-float(held), stated,
                                                 abs_tol=1e-9):
            raise ValueError(
                f'the level stated, {stated:+g} dB, disagrees with the '
                f'scaling held on the record for its comparison against '
                f'the specification, {float(held):+g} dB (a run at '
                f'{-float(held):+g} dB)')
        return stated, 'stated'
    if held is not None:
        return -float(held), 'held on the record'
    return None, ''


def _measured_quantity(measured: Any) -> str:
    """The quantity a measured object holds. TypeError for an object
    that is not a measured response, ValueError for an octave-band PSD,
    each saying why."""
    from .data import Bounded, Frf, Psd, Spectrum, TransientSpecification

    kind = type(measured).__name__
    if isinstance(measured, (Bounded, TransientSpecification)):
        raise TypeError(f'a {kind} is a requirement, not a measured '
                        'response')
    if isinstance(measured, Psd):
        if getattr(measured, 'bandwidth', None) is not None:
            raise ValueError('an octave-band PSD has no lines across a peak '
                             'to read it by; compare the narrowband PSD')
        return 'asd'
    if isinstance(measured, Frf):
        return 'transmissibility'
    if isinstance(measured, Spectrum):
        return 'amplitude'
    raise TypeError(f'a {kind} cannot be compared with a prediction; a '
                     'PSD, an amplitude spectrum or sine level, or a '
                     'transmissibility can')


@dataclass(frozen=True)
class PeakRow:
    """One predicted peak against its measured counterpart.

    Levels are in SI, the predicted one scaled to the level run. The
    measured fields are None when the band holds no measured peak, and
    `missing` says why. `notes` are the half-power reader's, said where
    the reading is read (too few lines across the peak, a crossing
    outside the record).
    """

    quantity: str
    predicted_frequency: float
    predicted_level: float
    predicted_q: float | None
    band: tuple[float, float]
    measured_frequency: float | None = None
    measured_level: float | None = None
    measured_q: float | None = None
    missing: str = ''
    notes: tuple[str, ...] = ()

    @property
    def ratio(self) -> float | None:
        """The measured level over the predicted one: an ASD ratio for
        an ASD, an amplitude ratio otherwise (`RATIO`)."""
        if self.measured_level is None:
            return None
        return self.measured_level / self.predicted_level

    @property
    def ratio_db(self) -> float | None:
        """`ratio` in decibels: 10 log10 of an ASD ratio, 20 log10 of an
        amplitude ratio, so either reads as the same decibels. None when
        the ratio is zero or missing."""
        ratio = self.ratio
        if ratio is None or ratio <= 0.0:
            return None
        return DECIBELS[self.quantity] * math.log10(ratio)

    @property
    def frequency_difference(self) -> float | None:
        """The measured frequency's difference from the predicted, in
        percent of the predicted."""
        if self.measured_frequency is None:
            return None
        return ((self.measured_frequency - self.predicted_frequency)
                / self.predicted_frequency * 100.0)


@dataclass(frozen=True)
class ChannelComparison:
    """One prediction read against its record.

    `record` is the record's index in the measured object, None when the
    object has no record of the channel. `factor` is what the
    prediction was multiplied by for the level run. `note` says why the
    channel was not compared (rows then empty).
    """

    prediction: Prediction
    record: int | None
    label: str
    factor: float
    rows: tuple[PeakRow, ...] = ()
    note: str = ''


@dataclass(frozen=True)
class PredictionComparison:
    """What `compare` found.

    `quantity` is what the measured object holds; `level_db` the level
    it was run at (None for a transmissibility) and `level_source` where
    that was read (`level_run`). `channels` are the predictions with a
    record in the object, in the set's order; `absent` the channels
    predicted that the object has no record of.
    """

    quantity: str
    level_db: float | None
    level_source: str
    channels: tuple[ChannelComparison, ...]
    absent: tuple[str, ...] = ()

    @property
    def ratio_name(self) -> str:
        """'ASD ratio' or 'amplitude ratio': what every row's ratio is."""
        return RATIO[self.quantity]

    @property
    def rows(self) -> list[tuple[ChannelComparison, PeakRow]]:
        """Every row with its channel, in order."""
        return [(channel, row) for channel in self.channels
                for row in channel.rows]

    @property
    def scaling(self) -> str:
        """How the predictions were scaled, in words: 'scaled by -12 dB
        on the ASD, 10^(L/10)', or that a transmissibility is not."""
        if self.quantity == 'transmissibility':
            return 'a transmissibility is a ratio and is not scaled'
        rule = '10^(L/10)' if self.quantity == 'asd' else '10^(L/20)'
        noun = 'the ASD' if self.quantity == 'asd' else 'the amplitude'
        return (f'scaled to the level run, {self.level_db:+g} dB '
                f'({self.level_source}), as {rule} on {noun}')

    def lines(self) -> list[str]:
        """The comparison as text, one line per row and per channel not
        compared, levels in each prediction's own unit. Nothing judged
        says so."""
        out = []
        for channel in self.channels:
            prediction = channel.prediction
            if channel.note:
                out.append(f'{channel.label}: not compared, {channel.note}')
                continue
            unit = (f' {prediction.unit_label}' if prediction.unit_label
                    else '')
            for row in channel.rows:
                head = (f'{channel.label}: predicted '
                        f'{_figures(row.predicted_frequency, 4)} Hz, '
                        f'{_figures(prediction.own(row.predicted_level), 3)}'
                        f'{unit}')
                if row.measured_frequency is None:
                    out.append(f'{head}; {row.missing}')
                    continue
                assert row.measured_level is not None
                ratio, decibels = row.ratio, row.ratio_db
                assert ratio is not None
                said = (f'{decibels:+.1f} dB' if decibels is not None
                        else 'no decibels')
                out.append(
                    f'{head}; measured '
                    f'{_figures(row.measured_frequency, 4)} Hz, '
                    f'{_figures(prediction.own(row.measured_level), 3)}'
                    f'{unit}; {self.ratio_name} {_figures(ratio, 3)}, '
                    f'{said}')
        if self.absent:
            out.append(f'No record of {", ".join(self.absent)}.')
        if not self.rows:
            out.append('Nothing was compared.')
        return out


def compare(predictions: Predictions | Sequence[Prediction], measured: Any,
            level_db: float | None = None) -> PredictionComparison:
    """Read `measured` against `predictions`, each scaled to the level
    run.

    Each prediction of the quantity `measured` holds is paired with the
    record of its channel (its response DOF, and for a transmissibility
    its input DOF); near each predicted peak, inside the peak's band,
    the measured peak is read by the half-power reader. A channel that
    cannot be read one way is not compared and says why: a prediction
    of another quantity, a record in units of another dimension or with
    its units undefined, several records of the channel, or a record
    that is zero, constant or empty.

    Parameters
    ----------
    predictions : Predictions or sequence of Prediction
        What was predicted.
    measured : Psd, Spectrum or Frf
        The response: a narrowband PSD for ASD predictions, an amplitude
        spectrum or a sine level for amplitude ones, an FRF for a
        transmissibility.
    level_db : float, optional
        The level the record was run at, dB relative to full level. When
        None, the scaling a PSD holds is used (`level_run`). Required
        for an ASD or an amplitude; refused for a transmissibility.

    Returns
    -------
    PredictionComparison

    Raises
    ------
    ValueError
        For an empty set or record, an octave-band PSD, a run level not
        stated or one given for a transmissibility, or a stated level
        that disagrees with the record's held scaling.
    TypeError
        For a measured object that is not a measured response (a
        specification, a time history), or a prediction that is not a
        `Prediction`.
    """
    held = list(predictions)
    if not held:
        raise ValueError('there are no predictions to compare against')
    for prediction in held:
        if not isinstance(prediction, Prediction):
            raise TypeError(f'a prediction is a Prediction, not '
                            f'{type(prediction).__name__}')
    quantity = _measured_quantity(measured)
    if not measured.num_records:
        raise ValueError('the measured object holds no records')
    level, source = level_run(measured, level_db)
    if quantity == 'transmissibility':
        if level is not None:
            raise ValueError('a transmissibility is a ratio of response to '
                             'input and is not scaled; leave the level '
                             'unstated')
    elif level is None:
        raise ValueError(f'the level the record was run at is not stated; '
                         f'{NOUN[quantity]} prediction is scaled to it (dB '
                         'relative to full level, 0 for a full-level run)')
    frequencies = np.asarray(measured.abscissa, dtype=np.float64)
    channels: list[ChannelComparison] = []
    absent: list[str] = []
    for prediction in held:
        at_dof = [i for i in range(measured.num_records)
                  if measured.response_dof[i] == prediction.channel
                  and _same_input(measured, i, prediction, quantity)]
        # the package's channel identity is (DOF, quantity): a drive
        # point's accelerometer and force gauge share a DOF and are two
        # records, and the prediction's quantity picks one. Matched by
        # DOF alone, every drive point was refused as two records
        # (2026-10-28). With none of the quantity, the refusal below
        # still names the dimension that was there.
        records = [i for i in at_dof
                   if _holds_quantity(measured, i, prediction, quantity)
                   ] or at_dof
        if not records:
            absent.append(prediction.channel if prediction.reference is None
                          else f'{prediction.channel}/{prediction.reference}')
            continue
        label = measured.record_label(records[0])
        note = _refusal(prediction, quantity, measured, records)
        if note:
            # not compared, so not scaled
            channels.append(ChannelComparison(prediction, records[0], label,
                                              1.0, note=note))
            continue
        record = records[0]
        factor = 1.0
        if quantity != 'transmissibility':
            assert level is not None and prediction.level_db is not None
            factor = scale_factor(quantity, level - prediction.level_db)
        magnitude = _amplitude(np.asarray(measured.ordinate[record]),
                               quantity)
        below = getattr(measured, 'below_floor', None)
        if below is not None:
            # a sine level reported at its noise floor is not a reading
            magnitude = np.where(np.asarray(below)[record], np.nan,
                                 magnitude)
        rows = tuple(_row(prediction, peak, band, factor, frequencies,
                          magnitude)
                     for peak, band in zip(prediction.peaks,
                                           prediction.bands, strict=True))
        channels.append(ChannelComparison(prediction, record, label, factor,
                                          rows))
    return PredictionComparison(quantity, level, source, tuple(channels),
                                tuple(absent))


def _same_input(measured: Any, i: int, prediction: Prediction,
                quantity: str) -> bool:
    """Whether record `i` has the prediction's input: the input DOF of
    a transmissibility, an autospectrum (no reference, or itself) for
    anything else."""
    references = measured.reference_dof
    if quantity == 'transmissibility':
        return (references is not None and prediction.reference is not None
                and references[i] == prediction.reference)
    return references is None or references[i] == measured.response_dof[i]


def _holds_quantity(measured: Any, i: int, prediction: Prediction,
                    quantity: str) -> bool:
    """Whether record `i` holds the prediction's quantity: its dimension
    is the prediction's record dimension, or undefined (declared units
    are refused by name later, not passed over here). A
    transmissibility is judged by its input alone."""
    from ..units import UNKNOWN

    if quantity == 'transmissibility' or prediction.record_dimension is None:
        return True
    dimension = measured.ordinate_dim[i]
    return dimension in (prediction.record_dimension, UNKNOWN)


def _refusal(prediction: Prediction, quantity: str, measured: Any,
             records: list[int]) -> str:
    """Why this channel cannot be compared, or ''."""
    from ..units import UNKNOWN

    if prediction.quantity != quantity:
        return (f'the prediction is {NOUN[prediction.quantity]} and the '
                f'record is {NOUN[quantity]}')
    if len(records) > 1:
        return (f'{len(records)} records of {prediction.channel}; compare '
                'one')
    record = records[0]
    dimension = measured.ordinate_dim[record]
    if dimension == UNKNOWN:
        return 'the record\'s units are not defined'
    if quantity == 'transmissibility':
        response, slash, reference = dimension.partition('/')
        if not slash or response != reference:
            return (f'the record is {dimension}, not a transmissibility '
                    '(a quantity over itself)')
    elif dimension != prediction.record_dimension:
        return (f'the record is {dimension} and the prediction '
                f'{prediction.record_dimension}')
    values = np.asarray(measured.ordinate[record])
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 'no line of the record has a value'
    if np.all(finite == 0):
        return 'the record is zero at every line'
    if np.all(finite == finite[0]):
        return 'the record is constant at every line'
    return ''


def _row(prediction: Prediction, peak: PredictedPeak,
         band: tuple[float, float], factor: float,
         frequencies: np.ndarray, magnitude: np.ndarray) -> PeakRow:
    """One predicted peak, scaled, against the measured peak in its
    band."""
    quantity = prediction.quantity
    power = 2.0 if quantity == 'asd' else 1.0
    if prediction.curve is not None:
        read, _why = _peak(prediction.curve[0],
                           prediction._magnitude(prediction.curve[1]), band,
                           'the predicted curve')
        # checked on creation: every named mode has its peak
        assert read is not None
        frequency, level, q = read.frequency, read.height ** power, read.q
    else:
        frequency = peak.frequency
        assert peak.level is not None
        level, q = peak.level, peak.q
    predicted = float(prediction.si(level)) * factor
    found, why = _peak(frequencies, magnitude, band, 'the record')
    if found is None:
        return PeakRow(quantity, frequency, predicted, q, band, missing=why)
    return PeakRow(quantity, frequency, predicted, q, band, found.frequency,
                   found.height ** power, found.q, notes=found.notes)
