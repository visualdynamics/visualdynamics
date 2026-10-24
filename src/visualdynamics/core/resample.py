"""A record taken to a lower sample rate, its anti-alias filter first.

A record sampled at one rate can represent frequencies up to half that
rate, its Nyquist frequency, and no higher. Content above it does not
disappear when the record is sampled: it lands in the record at a lower
frequency, the alias frequency |n fs - F| for whichever whole number n
brings it into the band from zero to half the sample rate, and once it
is in the samples nothing in the record can tell it from real content.
Keeping every q-th sample of a record is sampling it again, at a q-th
of the rate, and the same rule applies: everything the record holds
between the new Nyquist frequency and the old one folds into the new
band.

So `resample` filters before it decimates. The filter is a zero-phase
Butterworth low-pass (`core.filters.design`, run forward and backward
as `filters.filtered` runs it) that keeps the alias-free span flat:
the span is the new rate divided by 2.56, the band of lines an analyzer
keeps and shows. Above the span, up to the new Nyquist frequency, is the
filter's roll-off, and those lines are computed and dropped. A frame of
1,024 samples has 512 lines up to the Nyquist frequency, of which
1,024 / 2.56 = 400 lie inside the span and are kept, and 112 are
dropped; the line counts 400, 800 and 1,600 on an analyzer's menu come
from that division. The lowest frequency that can fold into the span is
the new rate less the span, 1.56 spans. The filter's corner is at 1.2
spans (`CORNER`), among the dropped lines, so a zero-phase pair, which
is 6.02 dB down at its nominal corner, is about 0.1 dB down at the span
and 0.01 dB at 0.9 of it. The lowest fold is then 1.3 corners up, and
the default twelfth-order design, doubled by the forward and backward
pass, is 55 to 56 dB down there at the factors of 10 to 16 in common
use and never less than 54.7 dB at any factor, past 50 dB of
attenuation where content would fold.

The order matters in one direction only. A filter run after the record
is decimated cannot help: the alias is in the band by then, at a
frequency where real content could be, and the same low-pass run on
the decimated record leaves it exactly where it was. `read_resample` reads that out on a
record: the record's autospectrum as recorded, and at the new rate
three ways (filtered first, decimated bare, and low-passed after
decimating), with the level each puts at the frequency where each
given frequency above the new Nyquist frequency folds.

`resample` with ``anti_alias=False`` keeps every q-th sample with no
filter, for the rare record already band-limited below the new span,
and warns (`AliasWarning`) every time, because whether it was is not
something the record can show.

References
----------
Decimation and the folding rule are textbook; these are where the
folding rule and the anti-alias filter a sampled record needs are set
out, and decimation is the same rule applied to a record already
sampled.

1. Department of Defense (2019). *MIL-STD-810H, Environmental
   Engineering Considerations and Laboratory Tests*. Method 516.8,
   Annex A, paragraph 1.1: the alias frequency |n Fs - F|, and an
   anti-alias filter ahead of the sampling that attenuates by 50 dB or
   more anything that would fold into the passband.
2. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. Sampling, aliasing, and the
   folding of content above the Nyquist frequency into the band below
   it.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .filters import Filtering, _filter_pool, design, response

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

# Asked for on 2026-10-15 from a worked example: a 4,600 Hz tone on a
# channel sampled at 5,120 Hz lands at 520 Hz at its full height, and a
# 2,000 Hz low-pass run on the sampled record leaves it there. The
# package had the filter and the spectrum and no resampling step, so
# the every-n-th-sample step was done by hand on a record's arrays.
# `decimate` (the peak-keeping thinning the plots and reports draw
# with) was looked at and left alone: it keeps each slice's extremes
# for drawing, which a filter ahead of it would defeat, and it never
# makes a record anything is computed from.

#: the sample rate over the alias-free span. Analyzers and controllers
#: commonly use 2.56: a power-of-two frame divided by 2.56 is a round
#: line count
SPAN_RATIO = 2.56

#: the anti-alias filter's corner over the alias-free span. At 1.2 the
#: corner is among the dropped lines (0.94 of the new Nyquist
#: frequency), so the span is flat to 0.1 dB, and the lowest frequency
#: that folds into the span (1.56 spans) is 1.3 corners up. With the
#: corner on the span the span's top line was 6.02 dB down and 0.9 of
#: it 1.46 dB down (Brandon, 2026-10-23)
CORNER = 1.2

#: the anti-alias filter's design order. Doubled by the forward and
#: backward pass, twelfth order with the corner at 1.2 spans is 55.8 dB
#: down at the lowest frequency that folds into the span at a factor of
#: 10, 55.1 dB at 16, and 54.7 dB in the limit of a large factor, past
#: 50 dB; eleventh order is 50.2 dB in that limit, and tenth falls short
#: (45.6 dB)
ORDER = 12

#: the frame the reading's spectra are taken with at the new rate, in
#: samples, when none is given: 400 lines kept, as an analyzer's first
#: menu entry. A shorter record takes the longest power of two it holds
FRAME = 1024

#: how much of a frame consecutive frames share in the reading's
#: spectra: the usual half
OVERLAP = 0.5

#: the shortest decimated record the reading frames, in samples; a
#: high filter order asks for more (`read_resample`)
SHORTEST = 32

#: how close two rates must be, as a share, to count as one: a sample
#: rate is read off the abscissa's step, so 51,200 Hz can arrive as
#: 51,199.9999
RATE_TOLERANCE = 1e-6


class AliasWarning(UserWarning):
    """A record was resampled with no anti-alias filter, so whatever it
    held between the new Nyquist frequency and the old one has folded
    into the new band, where nothing can take it out again."""


def alias_free_span(sample_rate: float, ratio: float = SPAN_RATIO
                    ) -> float:
    """The band of lines kept at `sample_rate`, in Hz: the rate over
    `ratio` (2.56). The lines above it, up to the Nyquist frequency,
    sit in the anti-alias filter's roll-off and are dropped."""
    if not float(sample_rate) > 0.0:
        raise ValueError('a sample rate must be above zero')
    if not float(ratio) > 2.0:
        raise ValueError('the span ratio must be above 2: at 2 the span '
                         'is the Nyquist frequency and the filter has no '
                         'room to roll off')
    return float(sample_rate) / float(ratio)


def alias_frequency(frequency: Any, sample_rate: float) -> Any:
    """Where content at `frequency` lands in a record sampled at
    `sample_rate`: |n fs - F| for the whole number n nearest F / fs,
    which brings it into the band from zero to the Nyquist frequency.

        alias_frequency(4600.0, 5120.0)    # 520.0

    A frequency at or below the Nyquist frequency is its own alias.
    Takes a number or an array, and returns the same.
    """
    rate = float(sample_rate)
    if not rate > 0.0:
        raise ValueError('a sample rate must be above zero')
    f = np.abs(np.asarray(frequency, dtype=float))
    alias = np.abs(np.round(f / rate) * rate - f)
    return float(alias) if alias.ndim == 0 else alias


def line_counts(frame_length: int, ratio: float = SPAN_RATIO
                ) -> tuple[int, int, int]:
    """(lines, kept, dropped) for a frame of `frame_length` samples:
    the lines from zero up to the Nyquist frequency (half the frame),
    the ones inside the alias-free span, and the ones between the span
    and the Nyquist frequency.

        line_counts(1024)    # (512, 400, 112)
    """
    n = int(frame_length)
    if n < 2:
        raise ValueError('a frame needs at least two samples')
    lines = n // 2
    # line k is at k fs / n, inside the span while k <= n / ratio; the
    # nudge keeps 1024 / 2.56 = 400.0000000001 at 400
    kept = min(int(np.floor(n / float(ratio) + 1e-9)), lines)
    return lines, kept, lines - kept


def decimation_factor(source_rate: float, rate: float) -> int:
    """How many samples of a record at `source_rate` make one at
    `rate`: a whole number of at least 2.

    A rate that does not divide the source rate into a whole factor is
    refused with the nearest two that do, since keeping every q-th
    sample is the only resampling done here, and a rate at or above the
    source rate is refused, since that is not a decimation.
    """
    source, target = float(source_rate), float(rate)
    if not target > 0.0 or not source > 0.0:
        raise ValueError('sample rates must be above zero')
    ratio = source / target
    if ratio < 1.0 + RATE_TOLERANCE * 10:
        raise ValueError(
            f'{target:g} Hz is not below the record\'s {source:g} Hz: '
            'resampling here lowers a rate, it never raises one')
    factor = round(ratio)
    if abs(ratio - factor) > RATE_TOLERANCE * ratio or factor < 2:
        low, high = int(np.floor(ratio)), int(np.ceil(ratio))
        near = [f'{source / q:g} Hz (every {ordinal(q)} sample)'
                for q in (low, high) if q >= 2]
        raise ValueError(
            f'{target:g} Hz does not divide the record\'s {source:g} Hz '
            'into a whole factor; the nearest that do are '
            + ' and '.join(near))
    return int(factor)


def anti_alias_filter(rate: float, order: int = ORDER,
                      ratio: float = SPAN_RATIO) -> Filtering:
    """The low-pass a record is filtered through before it is decimated
    to `rate`: its corner at `CORNER` (1.2) times the alias-free span,
    `rate` over `ratio` (2.56), of design order `order` (`ORDER`, 12).

    The corner sits among the dropped lines, above the span and below
    the new Nyquist frequency, so the span is flat; a `ratio` that puts
    it at or above the Nyquist frequency is refused.
    """
    corner = CORNER * alias_free_span(rate, ratio)
    if not corner < float(rate) / 2.0:
        raise ValueError(
            f'a span ratio of {ratio:g} puts the corner ({corner:g} Hz) at '
            f'or above the Nyquist frequency ({float(rate) / 2.0:g} Hz)')
    return Filtering(high=corner, order=order)


def _filtered_every(sos: Any, rows: np.ndarray, factor: int) -> np.ndarray:
    """Each row of `rows` through `sos` at zero phase, every
    `factor`-th sample of it kept, a row at a time.

    Filtering the whole record and then decimating holds the record
    filtered at the full rate beside it, twice the record at the peak
    (2.06 times, Brandon, 2026-10-23). Here only the rows being filtered
    are held at the full rate: `scipy.signal.sosfiltfilt` takes about
    three rows of working space for each, and the rows run on the
    filter threads (`filters.threaded_sosfiltfilt`) a few at a time, at
    most a quarter as many as the record has rows, so for a record of
    four rows or more the working space stays under the record's own
    size; a single row still takes the filter's own working space of
    about three rows. The values are the
    filter-then-decimate ones to the bit, since each row is the same
    call.
    """
    from scipy.signal import sosfiltfilt

    count, samples = rows.shape
    kept = np.empty((count, len(range(0, samples, factor))),
                    dtype=np.result_type(rows.dtype, np.float64))

    def one(i: int) -> None:
        kept[i] = sosfiltfilt(sos, rows[i])[::factor]

    at_once = min(8, max(1, count // 4))
    if at_once == 1 or rows.size < 1_000_000:
        for i in range(count):
            one(i)
        return kept
    pool = _filter_pool()
    for start in range(0, count, at_once):
        list(pool.map(one, range(start, min(start + at_once, count))))
    return kept


def resample(data: TimeHistory, rate: float, *, anti_alias: bool = True,
             order: int = ORDER) -> TimeHistory:
    """`data` at the lower sample rate `rate`, every channel: filtered
    through `anti_alias_filter` at zero phase, then every q-th sample
    kept.

        slow = resample(history, 5120.0)

    `rate` must divide the record's sample rate into a whole factor of
    at least 2 (`decimation_factor`). The filter runs at the record's
    own rate over the whole record, so its edge transients land at the
    record's ends; the first sample is kept, and the result's clock is
    the source's own at every q-th sample. The records' descriptions
    and the shock windows are carried over, the windows being in
    seconds from the record's start, which is kept. The averaging is
    not: it counts its frame in samples, which spans another time at
    the new rate, so the result is framed afresh. The record is
    filtered and decimated a row at a time, so the record filtered at
    the full rate is never held whole beside it.

    With ``anti_alias=False`` no filter is applied and an
    `AliasWarning` is issued: whatever the record holds between the new
    Nyquist frequency and the old one is folded into the new band, and
    no filter applied afterward can find it. Use it only for a record
    known to be band-limited below the new span.

    Parameters
    ----------
    data : TimeHistory
        The record, evenly sampled.
    rate : float
        The new sample rate, in Hz.
    anti_alias : bool, default True
        Filter before decimating.
    order : int, default 12
        The anti-alias filter's design order (`ORDER`).

    Returns
    -------
    TimeHistory
        The record at `rate`.

    Raises
    ------
    ValueError
        If the record is not evenly sampled, or `rate` is not a whole
        division of its sample rate.
    """
    from .data import TimeHistory

    source = data.sample_rate          # raises on uneven sampling
    factor = decimation_factor(source, rate)
    new_rate = source / factor
    rows = np.real(np.asarray(data.ordinate))
    if anti_alias:
        sos = design(anti_alias_filter(new_rate, order), source)
        kept = _filtered_every(sos, rows, factor)
    else:
        kept = np.ascontiguousarray(rows[:, ::factor])
        warnings.warn(AliasWarning(
            f'resampled from {source:g} to {new_rate:g} Hz with no '
            f'anti-alias filter: anything the record holds between '
            f'{new_rate / 2.0:g} and {source / 2.0:g} Hz has folded into '
            f'0 to {new_rate / 2.0:g} Hz, and no filter applied '
            'afterward can take it out'), stacklevel=2)
    result = TimeHistory(np.asarray(data.abscissa)[::factor], kept,
                         **data.record_fields())
    # the windows are in seconds from the record's start, and the start
    # is kept, so they mark the same events (Brandon, 2026-10-23); a
    # tuple, as `filters.carrying_marks` copies them
    result.shocks = tuple(data.shocks) if data.shocks else data.shocks
    return result


@dataclass(frozen=True)
class Fold:
    """Where one frequency above the new Nyquist frequency lands, and
    the level it lands at in each of the reading's spectra.

    `frequency` is the content's own frequency and `alias` where it
    lands, |n fs - F| with n `multiple`. `in_span` says whether it lands
    inside the alias-free span, where it would be read as content, or
    among the dropped lines above it. The levels are densities in SI,
    as `TimeHistory.compute_psds` gives them, each the tallest line
    within one line of the frequency: `source` at `frequency` in the
    record as recorded, and at `alias` the record decimated bare
    (`bare`), decimated then low-passed through the same filter
    (`after`), and filtered first (`filtered`).
    """

    frequency: float
    alias: float
    multiple: int
    in_span: bool
    source: float = float('nan')
    bare: float = float('nan')
    after: float = float('nan')
    filtered: float = float('nan')

    @property
    def after_db(self) -> float:
        """What low-passing after the decimation did at the alias, in dB
        against the bare record's level there: about zero inside the
        span, where the alias stays."""
        return _db(self.after, self.bare)

    @property
    def filtered_db(self) -> float:
        """What filtering first did at the alias, in dB against the bare
        record's level there."""
        return _db(self.filtered, self.bare)

    def describe(self) -> str:
        """The fold in words: where it lands, and what each order of
        filtering left there."""
        where = ('inside the span' if self.in_span
                 else 'among the dropped lines')
        text = f'{self.frequency:g} Hz folds to {self.alias:g} Hz, {where}'
        if np.isfinite(self.filtered_db):
            text += (f': {level_change(self.filtered_db)} filtered first, '
                     f'{level_change(self.after_db)} low-passed after '
                     'decimating')
        return text


def _db(value: float, reference: float) -> float:
    """10 log10 of a ratio of densities, NaN when either is not a
    positive number."""
    if not (np.isfinite(value) and np.isfinite(reference)
            and value > 0.0 and reference > 0.0):
        return float('nan')
    return float(10.0 * np.log10(value / reference))


def level_change(db: float) -> str:
    """A level change in words: '61.8 dB down', 'unchanged' within a
    tenth of a dB, '3.1 dB up'."""
    if abs(db) < 0.05:
        return 'unchanged'
    return f'{abs(db):.1f} dB {"down" if db < 0 else "up"}'


@dataclass(frozen=True)
class ResampleReading:
    """A record's resampling read out: the rates, the span and its
    lines, the filter and what it does at the edges that matter, where
    each given frequency folds, and the spectra that show it.

    Rates and frequencies are in Hz and gains in dB. `corner` is the
    anti-alias filter's corner, `CORNER` (1.2) spans; `gain_at_span` is
    the zero-phase gain at the span and `gain_at_fold` at
    `lowest_fold`, the lowest frequency that folds into the span (the
    new rate less the span). `lines`, `kept` and `dropped` are `line_counts` for the
    spectra's frame at the new rate, and `resolution` their line
    spacing. `spectra` holds (frequencies, densities) for 'source', the
    record as recorded on the same line spacing, and at the new rate
    'filtered' (filtered first), 'bare' (decimated with no filter) and
    'after' (decimated, then low-passed through the same filter); the
    densities are in SI with the dimension `density_dim`. `unfolded`
    holds the frequencies asked for that are at or below the new Nyquist
    frequency, which do not fold, and `notes` names them in words.
    """

    record: int
    label: str
    source_rate: float
    rate: float
    factor: int
    order: int
    frame_length: int
    lines: int
    kept: int
    dropped: int
    folds: tuple[Fold, ...]
    density_dim: str
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        compare=False, repr=False)
    unfolded: tuple[float, ...] = ()

    @property
    def nyquist(self) -> float:
        """Half the new rate: the highest frequency the result holds."""
        return self.rate / 2.0

    @property
    def span(self) -> float:
        """The alias-free span at the new rate."""
        return alias_free_span(self.rate)

    @property
    def corner(self) -> float:
        """The anti-alias filter's corner, `CORNER` (1.2) spans."""
        return self.filtering.high

    @property
    def resolution(self) -> float:
        """The spectra's line spacing, Hz."""
        return self.rate / self.frame_length

    @property
    def lowest_fold(self) -> float:
        """The lowest frequency that folds into the span: the new rate
        less the span."""
        return self.rate - self.span

    @property
    def filtering(self) -> Filtering:
        """The anti-alias filter the record was read through."""
        return anti_alias_filter(self.rate, self.order)

    @property
    def gain_at_span(self) -> float:
        """The filter's zero-phase gain at the span's top line, dB."""
        return _gain(self.filtering, self.source_rate, self.span)

    @property
    def gain_at_fold(self) -> float:
        """The filter's zero-phase gain at the lowest frequency that
        folds into the span, dB."""
        return _gain(self.filtering, self.source_rate, self.lowest_fold)

    @property
    def notes(self) -> tuple[str, ...]:
        """Each frequency asked for that does not fold, in words: a
        frequency typed at or below the new Nyquist frequency is said to
        be there rather than dropped without a word."""
        def where(f: float) -> str:
            at = abs(f - self.nyquist) <= RATE_TOLERANCE * self.nyquist
            return 'at' if at else 'below'

        return tuple(f'{f:g} Hz is {where(f)} the new Nyquist frequency '
                     'and does not fold' for f in self.unfolded)

    def describe(self) -> str:
        """The reading in words, as the plot's title and the status line
        give it."""
        text = (f'{self.source_rate:g} Hz to {self.rate:g} Hz (every '
                f'{ordinal(self.factor)} sample) behind a low-pass at '
                f'{self.corner:g} Hz: Nyquist {self.nyquist:g} Hz, span '
                f'{self.span:g} Hz, {self.kept} of {self.lines} lines '
                'kept')
        parts = [f.describe() for f in self.folds] + list(self.notes)
        if not parts:
            return text + '.'
        return text + '. ' + '; '.join(parts) + '.'


def _gain(filtering: Filtering, sample_rate: float, frequency: float
          ) -> float:
    """`filters.response` at one frequency: the zero-phase gain, dB."""
    return float(response(filtering, sample_rate,
                          frequencies=[frequency])[1][0])


def ordinal(n: int) -> str:
    """2nd, 3rd, 10th, 21st: the sample kept, in words."""
    suffix = ('th' if 10 <= n % 100 <= 20
              else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'))
    return f'{n}{suffix}'


def _one_record(data: TimeHistory, record: int) -> TimeHistory:
    """Channel `record` of `data` on its own, so a reading of one
    channel filters and transforms one channel."""
    from .data import TimeHistory

    fields = {key: (value if value is None else [value[record]])
              for key, value in data.record_fields().items()}
    return TimeHistory(np.asarray(data.abscissa),
                       np.real(np.asarray(data.ordinate)[record:record + 1]),
                       **fields)


def _spectrum(history: TimeHistory, frame: int
              ) -> tuple[np.ndarray, np.ndarray, str]:
    """(frequencies, densities, dimension) of `history`'s one channel:
    Hann frames of `frame` samples, half overlapped, as many as fit."""
    from .averaging import Averaging

    rate = history.sample_rate
    averaging = Averaging(frame_length=frame, overlap=OVERLAP,
                          window='hann').filled(len(history.abscissa), rate)
    psd = history.compute_psds(averaging)
    return (np.asarray(psd.abscissa, dtype=float),
            np.real(np.asarray(psd.ordinate)[0]), str(psd.ordinate_dim[0]))


def _level(spectrum: tuple[np.ndarray, np.ndarray], frequency: float
           ) -> float:
    """The tallest line within one line of `frequency`: a tone between
    two lines puts its spike on either, and the window spreads it."""
    f, g = spectrum
    if len(f) < 2 or not f[0] <= frequency <= f[-1]:
        return float('nan')
    k = round((frequency - f[0]) / (f[1] - f[0]))
    return float(np.max(g[max(k - 1, 0):k + 2]))


def read_resample(data: TimeHistory, rate: float, record: int = 0, *,
                  frequencies: float | Sequence[float] | None = None,
                  order: int = ORDER,
                  frame_length: int | None = None) -> ResampleReading:
    """Read what taking channel `record` of `data` to `rate` does.

        reading = read_resample(history, 5120.0, frequencies=[4600.0])
        print(reading.describe())

    The channel is resampled filtered first (`resample`), decimated
    bare, and decimated then low-passed through the same filter at the
    new rate, and each of the
    three and the channel as recorded is turned into an autospectrum on
    one line spacing: Hann frames of `frame_length` samples at the new
    rate (`FRAME`, 1,024, or the longest power of two the decimated
    record holds) and as many samples again times the factor at the
    record's own rate, half overlapped. `frequencies` above the new
    Nyquist frequency are folded (`alias_frequency`) and the level each
    spectrum has where each lands is read; frequencies at or below the
    new Nyquist frequency do not fold: they are left out of the folds
    and named in the reading's `notes`. With none given, the tallest line the record has above the new Nyquist
    frequency is the one folded.

    Parameters
    ----------
    data : TimeHistory
        The record, evenly sampled.
    rate : float
        The new sample rate, in Hz, a whole division of the record's.
    record : int, default 0
        Which channel.
    frequencies : float or sequence of float, optional
        Frequencies above the new Nyquist frequency to fold, in Hz.
    order : int, default 12
        The anti-alias filter's design order.
    frame_length : int, optional
        The spectra's frame at the new rate, in samples.

    Returns
    -------
    ResampleReading
        The rates, the span and its lines, the folds and the spectra.

    Raises
    ------
    ValueError
        As `resample` raises; if `record` names no channel, the frame
        is shorter than two samples, or the decimated record is
        shorter than the frame, than `SHORTEST` samples, or than the
        filter's padding needs.
    """
    if not 0 <= int(record) < data.num_records:
        raise ValueError(f'there is no channel {record}: the record has '
                         f'{data.num_records}')
    record = int(record)
    one = _one_record(data, record)
    source_rate = one.sample_rate
    factor = decimation_factor(source_rate, rate)
    new_rate = source_rate / factor
    # the low-pass after the decimation runs forward and backward on
    # the decimated record, which needs more samples than the filter's
    # padding (three per pole and one more per section, as
    # `scipy.signal.sosfiltfilt` pads)
    samples = len(one.abscissa[::factor])
    needed = max(SHORTEST, 3 * (2 * ((int(order) + 1) // 2) + 1) + 1)
    if samples < needed:
        raise ValueError(
            f'at {new_rate:g} Hz the record keeps {samples} samples, too '
            f'few for a spectrum (at least {needed})')
    filtered_first = resample(one, new_rate, order=order)
    with warnings.catch_warnings():
        # the bare decimation is the comparison the reading exists to
        # show, not a resampling anyone asked for
        warnings.simplefilter('ignore', AliasWarning)
        bare = resample(one, new_rate, anti_alias=False)
    from .filters import filtered

    span = alias_free_span(new_rate)
    after = filtered(bare, anti_alias_filter(new_rate, order))
    if frame_length is None:
        frame = min(FRAME, 2 ** int(np.floor(np.log2(len(one.abscissa)
                                                      // factor))))
    else:
        frame = int(frame_length)
        if frame < 2:
            raise ValueError('a frame needs at least two samples')
        if frame * factor > len(one.abscissa):
            raise ValueError(
                f'a frame of {frame} samples at {new_rate:g} Hz is longer '
                f'than the record ({len(one.abscissa) // factor} samples '
                'there)')
    spectra = {}
    f, g, density_dim = _spectrum(one, frame * factor)
    spectra['source'] = (f, g)
    for name, history in (('filtered', filtered_first), ('bare', bare),
                          ('after', after)):
        f, g, _dim = _spectrum(history, frame)
        spectra[name] = (f, g)

    nyquist = new_rate / 2.0
    if frequencies is None:
        f, g = spectra['source']
        above = f > nyquist
        wanted = ([float(f[above][np.argmax(g[above])])]
                  if above.any() and np.max(g[above]) > 0.0 else [])
    else:
        wanted = [float(x) for x in np.atleast_1d(
            np.asarray(frequencies, dtype=float))]
    folds, unfolded = [], []
    for frequency in wanted:
        frequency = abs(frequency)
        if not frequency > nyquist:
            unfolded.append(frequency)
            continue
        alias = alias_frequency(frequency, new_rate)
        folds.append(Fold(
            frequency=frequency, alias=alias,
            multiple=int(np.round(frequency / new_rate)),
            in_span=alias <= span,
            source=_level(spectra['source'], frequency),
            bare=_level(spectra['bare'], alias),
            after=_level(spectra['after'], alias),
            filtered=_level(spectra['filtered'], alias)))
    lines, kept, dropped = line_counts(frame)
    return ResampleReading(
        record=record, label=data.record_label(record),
        source_rate=source_rate, rate=new_rate, factor=factor,
        order=int(order), frame_length=frame, lines=lines, kept=kept,
        dropped=dropped, folds=tuple(folds), density_dim=density_dim,
        spectra=spectra, unfolded=tuple(unfolded))
