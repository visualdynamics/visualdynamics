"""A tone read off an averaged autospectrum, in amplitude, and why its
spike is not that amplitude.

An autospectrum (an ASD, in g²/Hz for an acceleration) is a density:
mean square per hertz. A broadband floor has mean square in every slice
of its band, so its density reads the same however finely the band is
cut into lines. A sine has all of its mean square, A²/2 for an
amplitude A, at one frequency, and the analysis spreads that over the
lines the window leaks it into. The tallest of them stands about

    (A²/2) / (B_w · Δf)

over the floor, where Δf is the line spacing and B_w the window's
equivalent noise bandwidth in lines (1.5 for the Hann window, 1 for a
rectangle). The spike's height goes as one over the spacing: the same
tone analyzed at a quarter of the lines stands a quarter as tall, with
the tone and the floor unchanged. It also reads lower when the tone
falls between two lines than when it falls on one, by as much as
1.4 dB under Hann and 3.9 dB under a rectangle. So the height of a
spike is a property of the analysis as much as of the tone, and it
cannot be compared with a tone table written in amplitude.

The tone itself can be read off the same spectrum by summing instead
of by height. The lines of an autospectrum times their spacing add up
to the mean square (Parseval's theorem, for any window), so the lines
the tone leaks into, less the floor beneath them, times the spacing,
are the tone's mean square whatever the spacing and the window. Its
root is the tone's rms, and root two times that is its amplitude.
`read_tone` does that and returns a `ToneReading`, which carries the
amplitude beside the spike the spacing and the window gave, so the two
can be seen together.

The reading in steps, at a frequency near the tone:

1. The window's leakage kernel, the share of a tone's mean square each
   line receives as a function of where the tone sits between lines,
   is worked out from the window's own shape (`leakage`). It gives the
   noise bandwidth, the main lobe's half-width in lines (2 for Hann:
   the lines the tone is summed over lie within it of the tone), and
   the share of the tone those lines hold.
2. The tone's line is the tallest within the main lobe of the frequency
   asked, and where the tone sits between lines is found by fitting
   the kernel to the lines about it, floor taken away.
3. The floor is the median of the lines either side, outside the main
   lobe and a guard of the same width, a straight floor through the two
   sides under the tone.
4. The lines within the main lobe of the tone, less the floor, times
   the spacing, are the tone's mean square, divided by the share of the
   tone those lines hold (under 1 % of it falls outside them for Hann,
   about 19 % for a rectangle with the tone midway between lines).

Four things are said rather than hidden, as notes on the reading and
as a `ToneWarning`: a tone that sits between lines (its spike reads
lower than it would on a line; the amplitude does not move), a floor
that differs either side of the tone (a tone at the edge of the floor's
band, where there is no floor under it to take away), a tone too near
the end of the spectrum for its lines or its floor to be read, and a
frequency where nothing stands above the floor.

References
----------
The noise bandwidth of a window and the loss of a tone's peak between
lines are Harris's; that the lines of an averaged autospectrum times
their spacing sum to the mean square is how the estimate is scaled.

1. Harris, F. J. (1978). "On the Use of Windows for Harmonic Analysis
   with the Discrete Fourier Transform." *Proceedings of the IEEE*,
   66(1), 51-83.
   [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)
   Equivalent noise bandwidth, scalloping loss and the leakage of a
   tone between lines, window by window.
2. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The one-sided autospectrum,
   its scaling, and its integral as the mean square.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

import numpy as np

#: the frame the leakage kernel is worked out on when the analysis's
#: own is longer. A window's kernel, read in lines, converges with the
#: frame length: worked out on 1024 and on 65536 samples the Hann
#: kernel differs by 1e-12 out to three lines, and the cost of one is
#: the frame length times the placements asked for, on every move of a
#: cursor.
KERNEL_LENGTH = 4096

#: the lines either side, beyond the tone's main lobe and a guard as
#: wide again, that the floor is read from
FLOOR_LINES = 16

#: the fewest lines of floor a side is read from
FLOOR_LEAST = 3

#: how far over the lower quartile of the floor's lines another tone's
#: spike stands before it is left out of the floor (10 dB)
SPIKE_RATIO = 10.0

#: the share of its on-line height a tone's spike may lose, by sitting
#: between lines, before the reading says so. Under Hann a tone loses
#: a tenth of its spike at 0.29 of a line off, and the most it can lose,
#: midway between lines, is 28 % (1.4 dB)
BETWEEN_LINES = 0.10

#: how far apart the floor's two sides may be, as a ratio, before the
#: floor under the tone is called uncertain
FLOOR_RATIO = 2.0

#: what is said when no floor can be read beside the tone
NO_FLOOR = ('no floor could be read either side of the tone; none is '
            'taken away, and its reading is uncertain')

#: the placements the tone's position between lines is searched over,
#: in lines from the tallest line
_PLACEMENTS = np.linspace(-0.5, 0.5, 201)


def density_label(unit: str = '') -> str:
    """The unit of a density whose amplitude is in `unit`: 'g²/Hz' for
    'g', '(m/s²)²/Hz' for 'm/s²', and '/Hz' for no unit."""
    if not unit:
        return '/Hz'
    # a compound unit is squared whole: m/s²²/Hz would square the
    # seconds only
    plain = unit.isalpha()
    return f'{unit}²/Hz' if plain else f'({unit})²/Hz'


def _density_text(value: float, unit: str = '') -> str:
    return f'{value:.3g} {density_label(unit)}'


class ToneWarning(UserWarning):
    """Warned when a tone reading rests on something the reader should
    know: a tone between lines, a floor that differs either side, a
    spectrum that ends too near the tone, or no tone at all. The reading
    is returned either way; `ToneReading.notes` carries the same words
    for a caller that shows them rather than warns."""


@dataclass(frozen=True, eq=False)
class Leakage:
    """A window's leakage, read in lines: what `leakage` returns.

    `shape` holds the window's samples over the frame the kernel was
    worked out on. `noise_bandwidth` is the equivalent noise bandwidth
    in lines, N·Σw²/(Σw)². `half_width` is the main lobe's half-width
    in whole lines, the reach of the lines a tone is summed over.
    `parameter` is the Tukey or Kaiser window's number, None for the
    others.
    """

    window: str
    parameter: float | None
    shape: np.ndarray
    noise_bandwidth: float
    half_width: int

    def share(self, offset: Any) -> np.ndarray:
        """The share of a tone's mean square a line receives, at
        `offset` lines from the tone (any real number, any shape).

        It is |W(δ)|²/(N·Σw²), with W the window's transform at δ
        lines; it is 1/`noise_bandwidth` on the line under the tone, and
        summed over every line it is 1, which is why the lines times
        their spacing are the tone's mean square.
        """
        delta = np.asarray(offset, dtype=np.float64)
        w = self.shape
        n = len(w)
        phase = np.exp(-2j * np.pi * np.multiply.outer(delta, np.arange(n))
                       / n)
        transform = phase @ w
        return np.abs(transform) ** 2 / (n * float(np.sum(w ** 2)))

    def peak_share(self, offset: float) -> float:
        """The share the tallest line receives with the tone `offset`
        lines from its nearest line (-0.5 to 0.5): 1/`noise_bandwidth`
        on a line, less between lines."""
        return float(np.max(self.share(np.array([offset - 1.0, offset,
                                                 offset + 1.0]))))

    def scallop(self, offset: float) -> float:
        """The spike's height with the tone `offset` lines off a line,
        as a share of its height on a line (1 on a line; 0.72 midway
        between lines under Hann)."""
        return self.peak_share(offset) * self.noise_bandwidth


@lru_cache(maxsize=32)
def _leakage(window: str, parameter: float | None, length: int) -> Leakage:
    from .averaging import window_shape

    shape = np.asarray(window_shape(window, length, parameter),
                       dtype=np.float64)
    bandwidth = len(shape) * float(np.sum(shape ** 2)) / float(
        np.sum(shape)) ** 2
    made = Leakage(window, parameter, shape, bandwidth, 1)
    # the main lobe's half-width: the first minimum of the kernel off
    # the tone, read on a grid of a fiftieth of a line out to sixteen
    # lines (a Kaiser at its highest beta, 40, has its first null at
    # 12.8: sqrt(1 + (beta/pi)^2)); rounded up, so a null on a whole
    # line (Hann's, at 2) counts as 2
    grid = np.arange(0.0, 16.0, 0.02)
    kernel = made.share(grid)
    # a minimum deep enough to be a null, not the ripple across a flat
    # top's crown (scipy's flat top climbs and falls by 0.1 % across its
    # first line, and was read as one line wide)
    rising = np.flatnonzero((np.diff(kernel) > 0.0)
                            & (kernel[:-1] < 0.01 * kernel[0]))
    first = grid[rising[0]] if len(rising) else grid[-1]
    half = max(1, int(np.ceil(first - 1e-6)))
    return Leakage(window, parameter, shape, bandwidth, half)


def leakage(window: Any = 'hann', length: int | None = None,
            parameter: float | None = None) -> Leakage:
    """A window's leakage in lines: its shape, noise bandwidth and main
    lobe.

    Parameters
    ----------
    window : str or Averaging, default 'hann'
        A window by any name `core.averaging.normalize_window` takes, or
        an `Averaging`, whose window, parameter and frame length are
        used.
    length : int, optional
        The frame length in samples. The kernel is worked out on at most
        `KERNEL_LENGTH` samples, over which it has converged; omitted, it
        is worked out on that many.
    parameter : float, optional
        The Tukey window's alpha or the Kaiser window's beta.

    Returns
    -------
    Leakage
        Cached: the same window asked for again is the same object.
    """
    from .averaging import Averaging, normalize_window

    if isinstance(window, Averaging):
        parameter = window.window_parameter
        length = window.frame_length if length is None else length
        window = window.window
    name = normalize_window(window)
    n = KERNEL_LENGTH if length is None else min(int(length), KERNEL_LENGTH)
    # an even frame, as an FFT analysis cuts one
    n = max(16, n - n % 2)
    return _leakage(name, None if parameter is None else float(parameter), n)


def noise_bandwidth(window: Any = 'hann', length: int | None = None,
                    parameter: float | None = None) -> float:
    """A window's equivalent noise bandwidth, in lines: 1 for a
    rectangle, 1.5 for Hann, 3.77 for scipy's flat top.

    The width of the ideal filter that passes the same noise as one line
    of the analysis does, and so the factor between a tone's mean square
    and the height its spike stands on an autospectrum
    (`spike_height`). `window`, `length` and `parameter` are as for
    `leakage`.
    """
    return leakage(window, length, parameter).noise_bandwidth


def spike_height(amplitude: float, spacing: float, window: Any = 'hann', *,
                 offset: float = 0.0, length: int | None = None,
                 parameter: float | None = None) -> float:
    """The height a tone's spike stands above the floor on an averaged
    autospectrum: (A²/2)/(B_w·Δf) on a line, less between lines.

    This is what a controller that draws its tone references as spikes
    on the ASD scales them to, and what the tone markers of
    `plot.tone_readout` are drawn at.

    Parameters
    ----------
    amplitude : float
        The tone's amplitude (peak), in the units whose square per hertz
        the autospectrum is in.
    spacing : float
        The line spacing in Hz.
    window : str or Averaging, default 'hann'
        As for `leakage`.
    offset : float, default 0
        Where the tone sits, in lines from its nearest line (-0.5 to
        0.5).
    length, parameter
        As for `leakage`.

    Returns
    -------
    float
        The spike's height in the autospectrum's units, floor excluded.
    """
    if not spacing > 0.0:
        raise ValueError('the line spacing must be positive')
    kernel = leakage(window, length, parameter)
    return float(amplitude) ** 2 / 2.0 / float(spacing) \
        * kernel.peak_share(float(offset))


@dataclass(frozen=True)
class ToneReading:
    """One tone read off an averaged autospectrum: what `read_tone`
    returns.

    `amplitude` is the tone's peak, in the units whose square per hertz
    the spectrum is in (g for an ASD in g²/Hz, as the spectrum is
    stored); `mean_square` its mean square. `spike` is the tallest line
    under the tone as measured, floor included, and `height` the spike
    this tone stands above the floor at this spacing, window and
    placement: what a tone reference drawn on the spectrum would stand
    at. `floor` is the floor estimated under the tone. `frequency` is
    the tone's frequency as read between lines, `spacing` the line
    spacing, `offset` where the tone sits from its nearest line, in
    lines. `lines` is the (first, last) index of the lines summed, and
    `captured` the share of the tone they hold. `notes` holds the
    warnings the reading raised, as text, with any density in them in
    the units the spectrum is stored in (`notes_in` gives them in
    others), and `uncertain` says whether any of them bears on the
    amplitude (a tone between lines does not; a floor that differs
    either side, another spike among the tone's lines, or no tone at all
    does).
    """

    frequency: float
    asked: float
    amplitude: float
    mean_square: float
    spike: float
    height: float
    floor: float
    spacing: float
    offset: float
    window: str
    noise_bandwidth: float
    lines: tuple[int, int]
    captured: float
    notes: tuple[str, ...] = ()
    uncertain: bool = False
    #: each note as a template and the densities that fill it, kept so
    #: `notes_in` can write them in display units
    said: tuple[tuple[str, tuple[float, ...]], ...] = field(
        default=(), repr=False)

    def notes_in(self, unit: str = '', scale: float = 1.0
                 ) -> tuple[str, ...]:
        """The notes with their densities in display units: `scale`
        converts the amplitude into `unit`, and the densities take its
        square, as `describe` does."""
        if not self.said:
            return self.notes
        return tuple(
            text.format(*(_density_text(value * scale ** 2, unit)
                          for value in values))
            for text, values in self.said)

    @property
    def rms(self) -> float:
        """The tone's rms: its amplitude over root two."""
        return float(np.sqrt(self.mean_square))

    @property
    def on_line(self) -> float:
        """The spike this tone would stand above the floor with the
        tone on a line: (A²/2)/(B_w·Δf)."""
        return self.mean_square / (self.noise_bandwidth * self.spacing)

    @property
    def between_lines(self) -> bool:
        """Whether the tone sits far enough between lines that its spike
        has lost more than `BETWEEN_LINES` of its on-line height."""
        return bool(self.on_line > 0.0
                    and self.height < (1.0 - BETWEEN_LINES) * self.on_line)

    def describe(self, unit: str = '', scale: float = 1.0) -> str:
        """The reading in one line, as the application shows it:
        '19.44 Hz: 1.94 g peak; its spike 0.978 g²/Hz (1.26 on a line)
        is not the tone'. `scale` converts the amplitude into `unit`;
        the densities take its square."""
        density = density_label(unit)
        peak = f'{unit} peak' if unit else 'peak'
        if not self.amplitude:
            return (f'{self.frequency:.4g} Hz: no tone above the floor '
                    f'({self.floor * scale ** 2:.3g} {density})')
        return (f'{self.frequency:.4g} Hz: {self.amplitude * scale:.3g} '
                f'{peak}; its spike {self.spike * scale ** 2:.3g} {density} '
                f'({(self.on_line + self.floor) * scale ** 2:.3g} on a line) '
                'is not the tone')


def line_spacing(frequencies: Any) -> float:
    """The spacing of evenly spaced lines, in Hz. Lines that are not
    ascending and evenly spaced (to a millionth of the spacing) raise
    ValueError: that test decides what a tone can be read off, here and
    in `Psd.tone_refusal`."""
    f = np.asarray(frequencies, dtype=np.float64).ravel()
    steps = np.diff(f)
    if not len(steps) or not np.all(steps > 0.0):
        raise ValueError('the frequencies must be ascending')
    spacing = float(np.median(steps))
    if np.max(np.abs(steps - spacing)) > 1e-6 * spacing:
        raise ValueError('a tone is read off evenly spaced lines; these are '
                         'not (an octave-band spectrum has no line spacing '
                         'to undo)')
    return spacing


def _placement(kernel: Leakage, values: np.ndarray, top: int,
               floor: float) -> float:
    """Where the tone sits from line `top`, in lines (-0.5 to 0.5): the
    placement whose kernel, scaled, best fits the lines about it with
    the floor taken away."""
    reach = kernel.half_width
    lo, hi = max(top - reach, 0), min(top + reach, len(values) - 1)
    seen = values[lo:hi + 1] - floor
    js = np.arange(lo, hi + 1) - top
    table = _placement_table(kernel)
    # rows: placements; columns: lines from -reach to reach
    pattern = table[:, js + reach]
    weight = np.einsum('ij,ij->i', pattern, pattern)
    scale = pattern @ seen / weight
    misfit = np.sum((pattern * scale[:, None] - seen) ** 2, axis=1)
    return float(_PLACEMENTS[int(np.argmin(misfit))])


@lru_cache(maxsize=32)
def _table(window: str, parameter: float | None, length: int) -> np.ndarray:
    kernel = _leakage(window, parameter, length)
    reach = kernel.half_width
    js = np.arange(-reach, reach + 1)
    # the share line j receives with the tone at δ from line 0 is the
    # kernel at j - δ
    return kernel.share(js[None, :] - _PLACEMENTS[:, None])


def _placement_table(kernel: Leakage) -> np.ndarray:
    """The kernel at every searched placement and every line of the
    main lobe, cached with the kernel's own key."""
    return _table(kernel.window, kernel.parameter, len(kernel.shape))


def _floor(g: np.ndarray, center: float, reach: int, count: int
           ) -> tuple[float, list[float],
                      list[tuple[str, bool, tuple[float, ...]]]]:
    """(floor, each side's floor, notes) under a tone at line position
    `center`; each note is (template, whether it bears on the
    amplitude, the densities that fill the template's places).

    Each side's floor is the median of its `count` lines beyond the
    tone's main lobe and a guard as wide again, leaving out every other
    spike among them with its own main lobe and one line more: a local
    maximum standing more than `SPIKE_RATIO` times over the lower
    quartile of the lines examined. A side with fewer than
    `FLOOR_LEAST` lines left is not read. The floor under the tone is
    the mean of the sides, a straight floor through them.
    """
    # Other tones are left out because a median of eight neighbors
    # was not enough: four tones 19.44 Hz apart, analyzed at 4 Hz lines,
    # put half of each tone's neighboring lines inside another tone's
    # lobe, and the floor read as the blade-passage tone's own spike
    # (the 1.944 g tone came back at 1.04 g; 2026-10-16).
    index = np.arange(len(g))
    away = index - center
    examined = (np.abs(away) > 2 * reach) & (np.abs(away) <= 2 * reach
                                             + count)
    notes: list[tuple[str, bool, tuple[float, ...]]] = []
    if not examined.any():
        return 0.0, [], [(NO_FLOOR, True, ())]
    quartile = float(np.percentile(g[examined], 25.0))
    keep = examined.copy()
    # a spike just outside the lines examined still reaches into them
    reached = np.flatnonzero((np.abs(away) > reach)
                             & (np.abs(away) <= 3 * reach + count + 1))
    for i in reached:
        left = g[i - 1] if i > 0 else -np.inf
        right = g[i + 1] if i < len(g) - 1 else -np.inf
        if g[i] > SPIKE_RATIO * quartile and g[i] >= left and g[i] >= right:
            keep[max(i - reach - 1, 0):i + reach + 2] = False
    sides = []
    for sign, word, other in ((-1, 'below', 'above'), (1, 'above', 'below')):
        lines = keep & (sign * away > 0)
        if lines.sum() < FLOOR_LEAST:
            text = (f'fewer than {FLOOR_LEAST} lines of floor {word} the '
                    f'tone: the floor is read from {other} it')
            notes.append((text, False, ()))
            continue
        sides.append(float(np.median(g[lines])))
    if not sides:
        return 0.0, [], [*notes, (NO_FLOOR, True, ())]
    if len(sides) == 2:
        low, high = sorted(sides)
        if high > FLOOR_RATIO * max(low, 0.0):
            text = ('the floor differs either side of the tone '
                    '({} below, {} above): the floor under it is a guess '
                    'between them, and its reading is uncertain')
            notes.append((text, True, (sides[0], sides[1])))
    return float(np.mean(sides)), sides, notes


def read_tone(frequencies: Any, density: Any, frequency: float, *,
              window: Any = 'hann', length: int | None = None,
              parameter: float | None = None,
              floor_lines: int = FLOOR_LINES, snap: bool = True,
              unit: str = '', warn: bool = True) -> ToneReading:
    """Read the tone near `frequency` off an averaged autospectrum, as
    an amplitude, beside the spike it stands at.

    The tone's line is the tallest within the window's main lobe of
    `frequency`, and where the tone sits between lines is fitted from
    the lines about it; with `snap=False` the tone is taken to be at
    `frequency` exactly, as a tone table states it. The lines within the
    main lobe of the tone, less the floor estimated from the lines
    either side, times the spacing, are its mean square (divided by the
    share of the tone those lines hold, which the window's shape gives);
    the amplitude is root two times its root. See the module's notes for the method and for
    what is warned.

    Parameters
    ----------
    frequencies : array-like
        The spectrum's lines in Hz, ascending and evenly spaced.
    density : array-like
        The autospectrum, one value per line, in the square of the
        tone's units per hertz (g²/Hz for a tone in g).
    frequency : float
        A frequency at or near the tone, in Hz.
    window : str or Averaging, default 'hann'
        The window the spectrum was averaged with, or the `Averaging`
        itself (its window, parameter and frame length).
    length : int, optional
        The frame length, when `window` is a name.
    parameter : float, optional
        The Tukey or Kaiser window's number, when `window` is a name.
    floor_lines : int, default `FLOOR_LINES`
        Lines either side, beyond the tone's main lobe and a guard, that
        the floor is read from (other tones among them left out).
    snap : bool, default True
        Find the tone's own spike near `frequency` and where it sits
        between lines, as a cursor wants; False reads the lines about
        `frequency` itself, as a tone table at known frequencies does.
    unit : str, optional
        The tone's unit as `density` holds it ('g' for g²/Hz), which the
        notes name their densities in. `ToneReading.notes_in` writes
        them in another.
    warn : bool, default True
        Warn the notes as `ToneWarning` as well as keeping them. A
        display that reads on every mouse move passes False.

    Returns
    -------
    ToneReading
        The tone's amplitude, mean square and frequency, the spike it
        stands at, the floor, and the notes.

    Raises
    ------
    ValueError
        When the lines and the values differ in length, hold too few
        lines, are not evenly spaced, hold no finite value, or
        `frequency` is outside them.
    """
    f = np.asarray(frequencies, dtype=np.float64).ravel()
    g = np.asarray(density)
    if np.iscomplexobj(g):
        # a cross spectrum's off-diagonal is not a density of one
        # signal; its diagonal is real, and a real part is what is read
        g = g.real
    g = g.astype(np.float64).ravel()
    if len(f) != len(g):
        raise ValueError(f'{len(f)} frequencies for {len(g)} values')
    kernel = leakage(window, length, parameter)
    reach = kernel.half_width
    if len(f) < 2 * reach + 3:
        raise ValueError(f'{len(f)} lines are too few to read a tone '
                         f'under a {kernel.window} window')
    spacing = line_spacing(f)
    asked = float(frequency)
    if not f[0] - spacing / 2 <= asked <= f[-1] + spacing / 2:
        raise ValueError(f'{asked:g} Hz is outside the spectrum, '
                         f'{f[0]:g} to {f[-1]:g} Hz')
    finite = np.isfinite(g)
    if not finite.any():
        raise ValueError('no line of this spectrum holds a number')
    g = np.where(finite, g, 0.0)
    said: list[tuple[str, tuple[float, ...]]] = []
    doubts: list[str] = []

    def say(text: str, doubtful: bool = True,
            densities: tuple[float, ...] = ()) -> None:
        # `text` is a template: each {} is filled by a density, in
        # whatever unit the reading is shown in (`notes_in`)
        said.append((text, tuple(float(d) for d in densities)))
        if doubtful:
            doubts.append(text)

    # where the frequency asked falls, in lines from the first
    position = (asked - f[0]) / spacing
    nearest = int(np.clip(round(position), 0, len(f) - 1))
    lo, hi = max(nearest - reach, 0), min(nearest + reach, len(f) - 1)
    top = lo + int(np.argmax(g[lo:hi + 1]))
    # a top on the edge of the lobe with a taller line beyond it is the
    # flank of another spike, not a spike of this frequency's own
    own = ((top == 0 or g[top] >= g[top - 1])
           and (top == len(f) - 1 or g[top] >= g[top + 1]))
    if snap and own:
        floor, _sides, _said = _floor(g, float(top), reach, floor_lines)
        center = top + _placement(kernel, g, top, floor)
    else:
        center = float(position)
        if snap:
            say(f'no spike stands at {asked:.4g} Hz: the lines rise '
                'toward a larger neighbor, and the lines at the '
                'frequency asked are read as they are')
    floor, _sides, floor_notes = _floor(g, center, reach, floor_lines)
    for text, doubtful, densities in floor_notes:
        say(text, doubtful, densities)
    under = np.flatnonzero(np.abs(np.arange(len(f)) - center)
                           <= reach + 1e-9)
    if not len(under):
        under = np.array([nearest])
    if under[0] == 0 or under[-1] == len(f) - 1:
        say('the tone\'s lines reach the end of the spectrum: some of '
            'its mean square is outside it')
    # another spike inside this tone's own lines: the two are summed
    # together, whatever is done with the floor (a flat top's main
    # lobe, five lines either side, swallows a tone 4.9 lines away)
    peak_line = under[int(np.argmax(g[under]))]
    for edge, beyond in ((under[0], under[0] - 1), (under[-1], under[-1] + 1)):
        if (peak_line == edge and 0 <= beyond < len(f)
                and g[beyond] > g[edge]):
            # the tallest of the lines summed is the edge one, and the
            # lines climb on past it: they are a larger neighbor's
            say(f'the lines of a larger spike, at {f[beyond]:.4g} Hz and '
                'beyond, reach into the lines read at '
                f'{f[0] + center * spacing:.4g} Hz: its mean square is '
                'summed with this one\'s')
    for i in under:
        if abs(i - peak_line) < 2 or not 0 < i < len(f) - 1:
            continue
        if (g[i] >= g[i - 1] and g[i] >= g[i + 1]
                and g[i] > SPIKE_RATIO * max(floor, 0.0)):
            say(f'another spike, at {f[i]:.4g} Hz, stands inside this '
                'tone\'s lines: the two are summed together; finer lines '
                'or a narrower window would part them')
            break
    # a line among the tone's own, or in the guard beside them, reading
    # far under the floor: the floor does not run under the tone (a
    # tone at the edge of the floor's band), and taking it away takes
    # away what was never there. The guard too, because the floor's
    # edge leaks a line or two into the lines summed and hides there.
    # A tenth, not a half: a floor line of a lightly averaged spectrum
    # falls under half its mean often, under a tenth seldom
    guard = np.flatnonzero(np.abs(np.arange(len(f)) - center)
                           <= 2 * reach + 1e-9)
    hollow = guard[g[guard] < floor / SPIKE_RATIO]
    if floor > 0.0 and len(hollow):
        say(f'the line at {f[hollow[0]]:.4g} Hz, beside the tone, reads '
            '{}, far under the floor of {}: the floor does not run under '
            'this tone (a tone at the edge of the floor\'s band), and its '
            'reading is uncertain', densities=(g[hollow[0]], floor))
    # a line that holds no number was read as zero; among the tone's
    # lines or its floor's that is a reading of nothing
    read = np.abs(np.arange(len(f)) - center) <= 3 * reach + floor_lines + 1
    missing = int(np.count_nonzero(read & ~finite))
    if missing:
        say(f'{missing} of the lines read about '
            f'{f[0] + center * spacing:.4g} Hz hold no number and are read '
            'as zero: its reading is uncertain')
    captured = float(np.sum(kernel.share(under - center)))
    total = float(np.sum(g[under] - floor)) * spacing
    mean_square = max(total, 0.0) / captured
    amplitude = float(np.sqrt(2.0 * mean_square))
    tone_frequency = float(f[0] + center * spacing)
    offset = float(center - round(center))
    height = mean_square / spacing * kernel.peak_share(offset)
    spike = float(np.max(g[under]))
    if not mean_square or height < floor:
        say(f'nothing at {tone_frequency:.4g} Hz stands clear of the '
            'floor ({}): this is no tone, or one too small to read at this '
            'spacing', densities=(floor,))
    reading = ToneReading(
        frequency=tone_frequency, asked=asked, amplitude=amplitude,
        mean_square=mean_square, spike=spike, height=height, floor=floor,
        spacing=spacing, offset=offset, window=kernel.window,
        noise_bandwidth=kernel.noise_bandwidth,
        lines=(int(under[0]), int(under[-1])), captured=captured)
    if mean_square and reading.between_lines:
        lost = 1.0 - kernel.scallop(offset)
        # the spike moves; the sum does not, so the amplitude stands
        say(f'the tone at {tone_frequency:.4g} Hz sits {abs(offset):.2f} '
            f'of a line from the nearest line ({spacing:.3g} Hz apart): '
            f'its spike reads {100 * lost:.0f} % under its height on a '
            'line; the amplitude read from the lines does not move', False)
    reading = replace(reading, said=tuple(said), uncertain=bool(doubts))
    notes = reading.notes_in(unit)
    if warn:
        for note in notes:
            warnings.warn(note, ToneWarning, stacklevel=2)
    return replace(reading, notes=notes)
