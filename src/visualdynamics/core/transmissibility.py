"""The two transmissibilities of one random record, side by side, and
what the cross-spectrum estimate is worth.

Transmissibility is the motion at one point over the motion at another,
a response over an input, at each frequency. From random data it is
computed two ways, and both are commonly labeled transmissibility.

The ASD ratio divides the response's autospectrum by the input's and
takes the root, because an autospectrum is a squared quantity:

    T_ASD = sqrt(Gyy / Gxx)

It needs nothing but two autospectra, so any two plots can be divided.
It has no phase, and it cannot tell how much of `Gyy` the input caused:
whatever is on the response channel that the input did not cause (a
sensor's noise floor, a rattle, a cable beating, another source) is in
`Gyy`, and the ratio counts it as response.

H1 divides the cross-spectrum by the input's autospectrum:

    H1 = Gxy / Gxx

The cross-spectrum keeps the phase between the two channels and
averages toward zero whatever in the response is unrelated to the
input, so H1 is not lifted by uncorrelated noise on the response. The
coherence,

    gamma^2 = |Gxy|^2 / (Gxx Gyy),

says what share of the response's mean square at each line the input
accounts for linearly. The three are tied exactly: squaring the ratio
and dividing by |H1| squared leaves one over the coherence, so

    T_ASD = |H1| / sqrt(gamma^2)

at every line. The ratio and H1 agree where the coherence is near one
and part company where it falls, the ratio always above: at a coherence
of 0.68 it reads 1.7 dB over H1, at 0.29 it reads 5.4 dB over. The gap
is the coherence restated, minus ten times its log, and it says the
coherence is under one, not why, nor which of the two is nearer the
truth. That depends on where the content the input did not cause is:

- on the response (a noise floor, a rattle, another source), H1 is
  unbiased and the ratio reads high, by one over the root of the
  coherence;
- on the input channel (its sensor's noise), H1 reads low, by the
  coherence, and the ratio low by its root, so the ratio is the nearer
  of the two. With 6 g rms of noise on the input of a survey whose
  truth is 0.100 at 1 kHz, H1 reads 0.073 and the ratio 0.084;
- leakage at a sharp resonance, a nonlinearity or a second input lower
  the coherence too, and bias both.

A transmissibility drawn without saying which of the two it is has not
said what it is, and the gap between them is a reason to look at the
measurement, not a correction to apply to either.

What H1 is worth at a line follows from its coherence and the number of
averages behind it. The normalized random error of its magnitude is

    eps = sqrt(1 - gamma^2) / (sqrt(gamma^2) * sqrt(2 n))

with `n` the number of independent averages: 3.0 percent at a coherence
of 0.9 and 60 averages, 7.5 percent at 0.9 and 10. It is one standard
deviation of the estimate about its expected value, and it is a
first-order result, good where it is small; past about 20 percent it
says the estimate is poor rather than how poor. The band drawn about H1
is |H1| (1 - k eps) to |H1| (1 + k eps), `k` the spread asked for (1 by
default); where k eps reaches one the band has no lower edge.

The count `n` is the averages the estimate is worth, not the frames cut.
Overlapped frames share samples and steady an average less than the same
number of independent ones would; a history's averaging gives the
equivalent count, discounted for overlap (`Averaging.statistical_dof`,
half its equivalent degrees of freedom), and that is the count the error
is read at, the same discount the confidence band on an averaged PSD
takes. With no overlap the two counts are equal.

`read_transmissibility` reads all of this from three averaged spectra
and returns a `TransmissibilityReading`; `TimeHistory.read_transmissibility`
computes the spectra from a record, from the same frames its PSDs and
FRFs use, and `Project.read_transmissibility` reads the one behind an
FRF in a project. `plot_transmissibility` draws a reading.

References
----------
The single-input frequency response estimate, the coherence function
and the random error of an estimated gain factor are Bendat and
Piersol's; the discount for overlapped frames is Welch's.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The H1 estimate, the
   ordinary coherence function, and the normalized random error of the
   gain factor estimate, sqrt(1 - gamma^2) / (|gamma| sqrt(2 n)), with
   its limit to small errors.
2. Welch, P. D. (1967). "The Use of Fast Fourier Transform for the
   Estimation of Power Spectra: A Method Based on Time Averaging Over
   Short, Modified Periodograms." *IEEE Transactions on Audio and
   Electroacoustics*, 15(2), 70-73.
   [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
   The variance of an average of overlapped frames, which is why such
   frames are worth fewer independent averages than their count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..units import UNKNOWN
from .data import TimeHistory, channel_quantities, ordinary_coherence

#: the random error past which the first-order expression says only
#: that the estimate is poor. The expression is a small-error result;
#: its source gives it for errors under about 0.2
SMALL_ERROR = 0.20


def frf_random_error(coherence: Any, averages: float) -> np.ndarray:
    """The normalized random error of an H1 magnitude, from its
    coherence and the number of averages behind it.

        eps = sqrt(1 - gamma^2) / (sqrt(gamma^2) * sqrt(2 n))

    `coherence` is the ordinary coherence at each line, 0 to 1, any
    shape; `averages` is `n`, the independent averages (an equivalent
    count, discounted for overlap, need not be whole). The result is one
    standard deviation of the magnitude as a fraction of it: 0.030 at a
    coherence of 0.9 and 60 averages, 0.075 at 0.9 and 10, 0.0092 at
    0.99 and 60. A coherence of one gives zero and a coherence of zero
    gives infinity, the two ends of the expression; a line whose
    coherence is not a number gives not a number. It is a first-order
    result, good where it is small (`SMALL_ERROR`).
    """
    n = float(averages)
    if not n > 0.0:
        raise ValueError('a random error needs a positive number of averages')
    gamma2 = np.asarray(coherence, dtype=np.float64)
    finite = gamma2[np.isfinite(gamma2)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise ValueError('a coherence lies between 0 and 1')
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(1.0 - gamma2) / (np.sqrt(gamma2) * np.sqrt(2.0 * n))


@dataclass(frozen=True)
class TransmissibilityPoint:
    """The two transmissibilities and the error of H1 at one line:
    what `TransmissibilityReading.at` returns.

    `frequency` is the line, `h1` the complex H1 there and `asd_ratio`
    the root of the response's autospectrum over the input's, both in
    the reading's units (SI, as a record is stored). `coherence` is the
    ordinary coherence, `error` the normalized random error of |H1| and
    `spread` how many of those errors the band reaches either side.
    `averages` is the count the error was read at.
    """

    frequency: float
    h1: complex
    asd_ratio: float
    coherence: float
    error: float
    spread: float
    averages: float

    @property
    def magnitude(self) -> float:
        """|H1|."""
        return float(abs(self.h1))

    @property
    def phase(self) -> float:
        """The phase of H1, in degrees from -180 to 180: the response's
        lag behind the input reads negative, minus 90 at a mode."""
        return float(np.degrees(np.angle(self.h1)))

    @property
    def lower(self) -> float | None:
        """The band's lower edge, |H1| (1 - spread * error), or None
        where the band reaches zero and has no lower edge."""
        factor = 1.0 - self.spread * self.error
        if not np.isfinite(factor) or factor <= 0.0:
            return None
        return self.magnitude * factor

    @property
    def upper(self) -> float:
        """The band's upper edge, |H1| (1 + spread * error)."""
        return self.magnitude * (1.0 + self.spread * self.error)

    @property
    def ratio_to_h1_db(self) -> float:
        """The ASD ratio to |H1|, in decibels of a magnitude: minus ten
        times the log of the coherence, since the ratio is |H1| over the
        root of the coherence. It says the coherence is under one, not
        which of the two estimates is nearer the truth (the module's
        introduction sets out why)."""
        if not self.magnitude > 0.0:
            return float('nan')
        return 20.0 * float(np.log10(self.asd_ratio / self.magnitude))

    @property
    def small(self) -> bool:
        """Whether the error is small enough for its expression to say
        how large it is (`SMALL_ERROR`)."""
        return bool(np.isfinite(self.error) and self.error <= SMALL_ERROR)

    def describe(self, unit: str = '', scale: float = 1.0) -> str:
        """One line for a status bar: |H1| with its error and coherence,
        then the ASD ratio and how far it reads over. `scale` converts
        the reading's units into `unit`."""
        unit = f' {unit}' if unit else ''
        error = (f'±{100.0 * self.error:.1f} %' if np.isfinite(self.error)
                 else 'error unbounded')
        return (f'{self.frequency:.4g} Hz: H1 {self.magnitude * scale:.4g}'
                f'{unit} ({error}, coherence {self.coherence:.3f}); '
                f'ASD ratio {self.asd_ratio * scale:.4g}{unit}, '
                f'{self.ratio_to_h1_db:+.1f} dB from H1')


@dataclass(frozen=True, eq=False)
class TransmissibilityReading:
    """The ASD-ratio transmissibility beside H1, with H1's coherence and
    random error, at every line: what `read_transmissibility` returns.

    `frequency` holds the lines in Hz. `h1` is the complex H1 at each,
    `asd_ratio` the root of the response's autospectrum over the
    input's, in the same units (SI, as a record is stored; `dimension`
    names them, such as 'acceleration/acceleration'), and `coherence`
    the ordinary coherence. A line where the input has no power holds
    not a number in all three. `frames` is how many frames were averaged
    and `averages` the independent count they are worth, the one the
    error is read at. `response` and `reference` name the two channels.
    `notes` says, as text, what the reading should be read with.
    """

    frequency: np.ndarray
    h1: np.ndarray
    asd_ratio: np.ndarray
    coherence: np.ndarray
    frames: int
    averages: float
    response: str = ''
    reference: str = ''
    dimension: str = UNKNOWN
    notes: tuple[str, ...] = ()

    @property
    def is_transmissibility(self) -> bool:
        """Whether the two channels measure one quantity, so the reading
        is a transmissibility, motion over motion. A force reference
        makes H1 an FRF (acceleration over force, say), and undeclared
        quantities leave it unsaid."""
        if self.dimension == UNKNOWN or '/' not in self.dimension:
            return False
        out, into = channel_quantities(self.dimension)
        return out == into

    @property
    def title(self) -> str:
        """What to call the reading: 'Transmissibility' where it is one
        (`is_transmissibility`), else 'ASD ratio and H1'."""
        return ('Transmissibility' if self.is_transmissibility
                else 'ASD ratio and H1')

    @property
    def magnitude(self) -> np.ndarray:
        """|H1| at every line."""
        return np.abs(self.h1)

    @property
    def phase(self) -> np.ndarray:
        """The phase of H1 at every line, in degrees."""
        return np.degrees(np.angle(self.h1))

    @property
    def error(self) -> np.ndarray:
        """The normalized random error of |H1| at every line
        (`frf_random_error`), at the reading's `averages`."""
        return frf_random_error(self.coherence, self.averages)

    @property
    def ratio_to_h1_db(self) -> np.ndarray:
        """The ASD ratio to |H1| at every line, in decibels: minus ten
        times the log of the coherence. Like the coherence, it does not
        say which of the two is nearer the truth."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return 20.0 * np.log10(self.asd_ratio / self.magnitude)

    def band(self, spread: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper): the band about |H1| that reaches `spread`
        random errors either side, |H1| (1 -+ spread * error).

        The lower edge is not a number at a line where the band reaches
        zero, since a magnitude has no edge below nothing; a plot on a
        log axis leaves the gap. One error either side is one standard
        deviation of the estimate; two is about the 95 percent band
        where the error is small.
        """
        spread = float(spread)
        if not spread > 0.0:
            raise ValueError('a band reaches more than zero errors')
        magnitude, error = self.magnitude, self.error
        with np.errstate(invalid='ignore'):
            factor = 1.0 - spread * error
            lower = np.where(factor > 0.0, magnitude * factor, np.nan)
            upper = magnitude * (1.0 + spread * error)
        return lower, upper

    def line(self, frequency: float) -> int:
        """The index of the line nearest `frequency`."""
        f = np.asarray(self.frequency, dtype=np.float64)
        return int(np.argmin(np.abs(f - float(frequency))))

    def at(self, frequency: float, spread: float = 1.0
           ) -> TransmissibilityPoint:
        """The reading at the line nearest `frequency`, with its band
        reaching `spread` random errors either side."""
        k = self.line(frequency)
        return TransmissibilityPoint(
            frequency=float(self.frequency[k]), h1=complex(self.h1[k]),
            asd_ratio=float(self.asd_ratio[k]),
            coherence=float(self.coherence[k]),
            error=float(self.error[k]), spread=float(spread),
            averages=float(self.averages))

    def describe(self) -> str:
        """What was read, in a few words: the two channels and the
        averages."""
        pair = (f'{self.response} over {self.reference}'
                if self.response or self.reference else self.title)
        return (f'{pair}: H1 and the ASD ratio from {self.frames} frames, '
                f'{self.averages:.4g} independent averages')


def read_transmissibility(frequency: Any, gxx: Any, gyy: Any, gxy: Any,
                          frames: int, averages: float | None = None, *,
                          response: str = '', reference: str = '',
                          dimension: str = UNKNOWN
                          ) -> TransmissibilityReading:
    """The ASD-ratio transmissibility and H1 from three averaged spectra,
    with the coherence and the random error of H1.

    `gxx` is the input's averaged autospectrum and `gyy` the
    response's, one value per line of `frequency`; `gxy` is their
    averaged cross-spectrum, conj(X) Y, from the same frames and the
    same window, so the three are on one scaling (any common factor
    cancels in every quantity here). `frames` is how many frames were
    averaged and `averages` the independent count they are worth, the
    frames themselves when omitted (right when they do not overlap).
    `response`, `reference` and `dimension` are carried onto the reading
    for its labels.

    A line where the input has no power has no H1, no ratio and no
    coherence: all three are not a number there. A line where the
    response has no power has an H1 and a ratio of zero and no
    coherence. A line where any of the three spectra is not a finite
    number (a sample in the record that is not one) has no reading
    either, and is noted as such rather than as missing power. The
    coherence is `ordinary_coherence`'s, the one a CPSD's coherence is,
    and H1 is `TimeHistory.compute_frfs`'s single-reference H1.
    """
    f = np.asarray(frequency, dtype=np.float64)
    gxx = np.real(np.asarray(gxx)).astype(np.float64)
    gyy = np.real(np.asarray(gyy)).astype(np.float64)
    gxy = np.asarray(gxy, dtype=np.complex128)
    if not (f.shape == gxx.shape == gyy.shape == gxy.shape) or f.ndim != 1:
        raise ValueError('the three spectra and the frequencies are one '
                         'value per line, all the same length')
    frames = int(frames)
    if frames < 2:
        # one frame's cross-spectrum is the product of its two transforms,
        # so its coherence is exactly one at every line whatever the two
        # channels are: not a poor estimate, no estimate
        raise ValueError('a coherence needs at least two frames; from one '
                         'it is exactly 1 at every line')
    averages = float(frames if averages is None else averages)
    if not averages > 0.0:
        raise ValueError('the averages an estimate is worth are more than '
                         'zero')
    finite = np.isfinite(gxx) & np.isfinite(gyy) & np.isfinite(gxy)
    powered = finite & (gxx > 0.0)
    # H1 as the FRF computation makes it with one reference, a 1 x 1
    # input matrix per line, handed only the lines it can divide by
    h1 = TimeHistory._frf_h1(np.where(powered, gxx, 1.0)[:, None, None],
                             np.where(powered, gxy, 0.0)[:, None], None)
    h1 = np.where(powered, h1[:, 0], np.nan + 0j)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(powered, np.sqrt(np.maximum(gyy, 0.0) / gxx),
                         np.nan)
    coherence = ordinary_coherence(gxx, gyy, gxy)

    notes = []
    if frames != round(averages):
        notes.append(f'the {frames} frames overlap and are worth '
                     f'{averages:.4g} independent averages; the error is '
                     f'read at that count')
    error = frf_random_error(coherence, averages)
    defined = np.isfinite(coherence)
    large = defined & ~(error <= SMALL_ERROR)
    if large.any():
        span = f[large]
        notes.append(f'{int(large.sum())} of {int(defined.sum())} '
                     f'lines, from {span.min():.4g} to {span.max():.4g} Hz, '
                     f'have a random error over '
                     f'{100.0 * SMALL_ERROR:.0f} %, where its expression '
                     f'says the estimate is poor rather than how poor')
    if not finite.all():
        which = [name for name, values in (('input', gxx), ('response', gyy))
                 if not np.isfinite(values).all()] or ['cross']
        notes.append(f'{int((~finite).sum())} lines are not finite numbers '
                     f'in the {" and ".join(which)} spectrum: the record '
                     f'holds a sample that is not a number, and there is '
                     f'no reading there')
    unpowered = finite & ~powered
    if unpowered.any():
        notes.append(f'{int(unpowered.sum())} lines carry no input power '
                     f'and have no reading')
    dead = powered & ~(gyy > 0.0)
    if dead.any():
        notes.append(
            'the response carries no power at any line the input drives: '
            'a channel that recorded nothing, so H1 and the ratio read '
            'zero and the coherence is undefined'
            if dead.sum() == powered.sum() else
            f'{int(dead.sum())} lines carry no response power: H1 and the '
            f'ratio read zero there and the coherence is undefined')
    if dimension != UNKNOWN and '/' in str(dimension):
        out, into = channel_quantities(str(dimension))
        if out != into:
            notes.append(f'the response is {out} and the reference {into}, '
                         f'so H1 is an FRF in {out} over {into}, not a '
                         f'transmissibility, which is motion over motion')
    return TransmissibilityReading(
        frequency=f, h1=h1, asd_ratio=ratio, coherence=coherence,
        frames=frames, averages=averages, response=str(response),
        reference=str(reference), dimension=str(dimension),
        notes=tuple(notes))
