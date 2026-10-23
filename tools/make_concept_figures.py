"""The concept figures in the guide, drawn by the package itself.

Each figure is a small constructed object rendered through the same
plot calls the application uses, so the pictures cannot drift from
what the code actually draws. Regenerate after anything touching the
PSD reading rules:

    QT_QPA_PLATFORM=offscreen ./.venv/bin/python tools/make_concept_figures.py
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'docs' / 'guide' / 'images'


def psd_reading():
    """Three readings of 'the area is the picture': steps, bars, law."""
    import numpy as np

    from visualdynamics.core.data import Psd, Specification
    from visualdynamics.units import SI

    # ten coarse lines whose values sum to 10: the bins are the
    # midpoints, the edges land at 5 and 105 Hz, and the area is
    # exactly 100 (m/s^2)^2 -> 10 m/s^2 RMS, a number worth stating
    values = np.array([[0.5, 1.5, 0.8, 1.7, 1.2,
                        1.0, 0.9, 1.3, 0.6, 0.5]])
    narrowband = Psd(np.arange(10.0, 101.0, 10.0),
                     values, response_dof=['101Z+'],
                     ordinate_dim='acceleration**2/frequency',
                     ordinate_unit='m/s**2')
    narrowband.plot(path=str(OUT / 'psd-reading-steps.png'),
                    unit_system=SI, show=False)
    print(f'narrowband RMS {np.sqrt(narrowband.area()):.3f}')

    banded = narrowband.to_octave(3)
    banded.plot(path=str(OUT / 'psd-reading-octave.png'),
                unit_system=SI, show=False)
    print(f'octave RMS     {np.sqrt(banded.area()):.3f}')

    specification = Specification(
        np.array([20.0, 80.0, 350.0, 2000.0]),
        np.array([[0.01, 0.04, 0.04, 0.007]]),
        response_dof=['101Z+'],
        ordinate_dim='acceleration**2/frequency',
        ordinate_unit='m/s**2')
    specification.plot(path=str(OUT / 'psd-reading-law.png'),
                       unit_system=SI, show=False)
    print(f'specification RMS {np.sqrt(specification.area()):.3f}')


def _breakpoints():
    """The four-breakpoint specification every compliance figure judges
    against: a decade of flat top with power-law skirts, warning at
    ±3 dB and abort at ±6 dB."""
    import numpy as np

    from visualdynamics.core.data import Specification

    f = np.array([20.0, 80.0, 800.0, 2000.0])
    target = np.array([[0.01, 0.04, 0.04, 0.007]])
    limits = {name: target * 10 ** (db / 10)
              for name, db in (('warning_lower', -3), ('warning_upper', 3),
                               ('abort_lower', -6), ('abort_upper', 6))}
    return Specification(f, target, response_dof=['101Z+'],
                         ordinate_dim='acceleration**2/frequency',
                         ordinate_unit='m/s**2', **limits)


def _lines(specification, df=2.0, high=2000.0):
    """The same requirement as a controller writes it: a density per
    line, on lines `df` apart, NaN beyond `high` — stored to Nyquist
    and written only where it was controlled."""
    import numpy as np

    from visualdynamics.core.compliance import log_interpolate
    from visualdynamics.core.data import Specification

    f = np.arange(0.0, 2560.0 + df / 2, df)
    rows = {}
    for name, values in (('target', specification.ordinate[0]),
                         *specification.limits.items()):
        written = log_interpolate(f, specification.abscissa, np.real(values[0])
                                  if name != 'target' else np.real(values))
        written[f > high] = np.nan
        rows[name] = np.atleast_2d(written)
    out = Specification(f, rows.pop('target'), response_dof=['101Z+'],
                        ordinate_dim='acceleration**2/frequency',
                        ordinate_unit='m/s**2', **rows)
    out.interpolation = Specification.reading_of(f)
    return out


def _response(specification, df=1.0, high=2560.0, seed=7):
    """A measurement of the target with two departures: a resonance
    over the abort limit around 300 Hz and a notch under it around
    1200 Hz, on `df` lines to `high` — past the requirement's end."""
    import numpy as np

    from visualdynamics.core.compliance import log_interpolate
    from visualdynamics.core.data import Psd

    f = np.arange(0.0, high + df / 2, df)
    target = log_interpolate(f, specification.abscissa,
                             np.real(specification.ordinate[0]))
    rng = np.random.default_rng(seed)
    shape = np.exp(0.15 * rng.standard_normal(f.size))
    # wide enough to be out on a third-octave band too, not only on
    # the lines under it: a requirement on bands is on their power
    shape *= 1.0 + 6.0 * np.exp(-((f - 300.0) / 60.0) ** 2)
    shape *= 1.0 - 0.85 * np.exp(-((f - 1200.0) / 120.0) ** 2)
    values = np.where(np.isfinite(target), target, 0.002) * shape
    out = Psd(f, np.atleast_2d(values), response_dof=['101Z+'],
              ordinate_dim='acceleration**2/frequency',
              ordinate_unit='m/s**2')
    out.scale_db = 0
    return out


def compliance():
    """One figure per comparison there is, plus the two edge cases —
    each judged by the one cell rule (`compliance.judge`) and drawn
    by `plot_comparison`."""
    from visualdynamics.core.compliance import compare
    from visualdynamics.plot import plot_comparison
    from visualdynamics.units import SI

    breakpoints = _breakpoints()
    lines = _lines(breakpoints)
    octave_spec = lines.to_octave(3)
    narrow = _response(breakpoints)
    octave = narrow.to_octave(3)
    # the three comparisons there are (Brandon, 2026-09-19): bands
    # compare only with the same bands
    cases = {
        'lines-vs-breakpoints': (narrow, breakpoints),
        'lines-vs-lines': (narrow, lines),
        'octave-vs-octave': (octave, octave_spec),
    }
    for name, (measured, specification) in cases.items():
        plot_comparison(measured, specification, unit_system=SI, show=False,
                        path=str(OUT / f'compliance-{name}.png'))
        got = compare(specification, measured, scale_db=0)
        print(f'{name:>24}: {got["lines"]} cells, {got["abort_percent"]:.1f}% '
              f'of the band outside abort, RMS {got["difference_db"]:+.2f} dB')
    # the end of a requirement inside a band: the lines end at 1450 Hz
    # and the third-octave band holding 1450 reaches to 1778, whole.
    # The controller held the narrowband response to the requirement
    # and drove nothing past it, so the response falls away there
    # too — the case that happens (Brandon, 2026-09-20)
    import numpy as np

    short = _lines(breakpoints, high=1450.0).to_octave(3)
    controlled = _response(breakpoints, seed=7)
    beyond = np.asarray(controlled.abscissa) > 1450.0
    controlled.ordinate[:, beyond] *= 0.02
    plot_comparison(controlled.to_octave(3), short, unit_system=SI, show=False,
                    path=str(OUT / 'compliance-edge.png'))
    got = compare(short, controlled.to_octave(3), scale_db=0)
    print(f'{"edge":>24}: {got["lines"]} cells, judged to {got["band"][1]:.0f} Hz, '
          f'the whole last band; {got["abort_lines"]} band out')
    # a hole: a controller's notch, written as zero across 400-500 Hz

    notched = _lines(breakpoints)
    hole = (notched.abscissa >= 400.0) & (notched.abscissa <= 500.0)
    notched.ordinate[:, hole] = 0.0
    for values in notched.limits.values():
        values[:, hole] = 0.0
    plot_comparison(narrow, notched, unit_system=SI, show=False,
                    path=str(OUT / 'compliance-hole.png'))
    got = compare(notched, narrow, scale_db=0)
    print(f'{"hole":>24}: {got["lines"]} cells, nothing judged in the notch')



def pulse_check():
    """Two classical pulses judged against their envelopes: a 40 g,
    11 ms sawtooth compensated as a shaker makes it, its fall taking
    nearly the whole tenth allowed, and a 50 g half-sine followed by a
    fixture ringing at 700 Hz past the band after the pulse."""
    import numpy as np

    from visualdynamics.core.classical_pulse import ClassicalPulse, check_pulse
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.plot import plot_pulse_check
    from visualdynamics.units import IN_LBF_S

    g, rate, duration = 9.80665, 51200.0, 0.011
    shown = IN_LBF_S.with_units(acceleration='g')
    t = np.arange(int(0.25 * rate)) / rate
    opens = 0.11
    noise = 0.15 * g * np.random.default_rng(5).standard_normal(t.size)

    def record(values):
        return TimeHistory(t, values[None], response_dof=['101Z+'],
                           ordinate_dim=['acceleration'])

    # the sawtooth, with a 40 ms half-sine of the other sign either side
    # carrying half its area each, so the table starts and ends at rest
    made = ClassicalPulse('sawtooth', 40.0 * g, duration, 0.001)
    area = 0.5 * made.peak * (duration + made.edge)
    length = 0.04
    compensation = ClassicalPulse('half-sine', -np.pi * area / (4 * length),
                                  length)
    sawtooth = (made.at(t - opens) + compensation.at(t - opens + length)
                + compensation.at(t - opens - made.extent) + noise)
    check = check_pulse(record(sawtooth),
                        ClassicalPulse('sawtooth', 40.0 * g, duration))
    print(f'sawtooth: change {check.velocity_change / 0.0254:.1f} in/s, '
          f'{100 * (check.velocity_ratio - 1):+.1f} % of nominal, '
          f'{"passes" if check.passed else "flagged"}')
    plot_pulse_check(record(sawtooth), 'sawtooth', 40.0 * g, duration,
                     unit_system=shown, show=False,
                     path=str(OUT / 'pulse-check-sawtooth.png'))

    # the half-sine, and a fixture mode ringing after it
    nominal = ClassicalPulse('half-sine', 50.0 * g, duration)
    after = np.clip(t - opens - duration, 0.0, None)
    ringing = np.where(t > opens + duration, 15.0 * g
                       * np.exp(-after / 0.008)
                       * np.sin(2 * np.pi * 700.0 * after), 0.0)
    ringing_record = record(nominal.at(t - opens) + ringing + noise)
    check = check_pulse(ringing_record, nominal)
    print(f'ringing: {check.outside_count} of {check.points} outside, '
          f'{"passes" if check.passed else "flagged"}')
    plot_pulse_check(ringing_record, 'half-sine', 50.0 * g, duration,
                     unit_system=shown, show=False,
                     path=str(OUT / 'pulse-check-ringing.png'))

def sine_tracking():
    """One swept tone read the ways a controller reads it: a clipped
    drive's third harmonic through each detector and band, and a step
    in level through bands of different widths."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.sine import LOG, SineSweepSpecification, SineTone
    from visualdynamics.core.sine_tracking import SineTracking, track_sine
    from visualdynamics.plot import plot_sine_tracking
    from visualdynamics.units import SI

    rate = 16384.0
    dof = '101Z+'

    def record(tone, envelope, harmonic=0.0, onset=0.0, tail=0.0):
        t = np.arange(int((onset + tone.duration() + tail) * rate)) / rate
        local = np.clip(t - onset, 0.0, None)
        inside = (t >= onset) & (local <= tone.duration())
        phase = tone.phase_at(local)
        wave = np.cos(phase) + harmonic * np.cos(3.0 * phase)
        x = np.where(inside, envelope(local) * wave, 0.0)
        return TimeHistory(t, x[None], response_dof=[dof],
                           ordinate_dim=['acceleration'])

    # a log sweep, 20 to 320 Hz in four seconds, at 10 m/s**2 with a
    # third harmonic of 30 % in phase: the clipped drive's signature
    tone = SineTone('Sine Tone 1', 0.5, [20.0, 320.0], [[10.0], [10.0]],
                    [LOG], [60.0])
    spec = SineSweepSpecification([tone], [dof], ordinate_unit='m/s**2')
    clipped = record(tone, lambda _t: 10.0, harmonic=0.3, onset=0.5,
                     tail=0.25)
    settings = [SineTracking(detector='peak'), SineTracking(detector='rms'),
                SineTracking(proportional=0.5),
                SineTracking(proportional=0.1)]
    plot_sine_tracking(clipped, spec, settings, unit_system=SI, show=False,
                       size=(1000, 600),
                       path=str(OUT / 'sine-tracking-harmonic.png'))
    for setting, level in zip(settings, track_sine(clipped, spec, settings)):
        print(f'{setting.describe():>34}: median '
              f'{np.median(np.abs(level.ordinate[0])):.3f} m/s^2')

    # a linear sweep at 100 Hz/s whose level doubles at 250 Hz: the
    # requirement steps over one hertz, the record at the same instant
    stepped = SineTone('Sine Tone 1', 0.0, [50.0, 250.0, 251.0, 450.0],
                       [[10.0], [10.0], [20.0], [20.0]], [0, 0, 0],
                       [100.0, 100.0, 100.0])
    spec = SineSweepSpecification([stepped], [dof], ordinate_unit='m/s**2')
    doubled = record(stepped, lambda t: np.where(t < 2.0, 10.0, 20.0))
    settings = [SineTracking(detector='peak'),
                SineTracking(proportional=0.5),
                SineTracking(proportional=0.1), SineTracking(fixed=5.0)]
    plot_sine_tracking(doubled, spec, settings, onset=0.0, lines=2000,
                       unit_system=SI, show=False, size=(1000, 600),
                       path=str(OUT / 'sine-tracking-step.png'))
    for setting, level in zip(settings, track_sine(doubled, spec, settings,
                                                   onset=0.0, lines=2000)):
        after = (level.seconds > 2.0) & (np.abs(level.ordinate[0]) >= 15.0)
        lag = level.seconds[np.flatnonzero(after)[0]] - 2.0
        print(f'{setting.describe():>34}: half way {1000 * lag:.0f} ms '
              'after the step')


def sine_tracking_band():
    """The tracking band drawn on the record it reads: the same clipped
    sweep, now under broadband noise too, through a 50 % proportional
    band and a 5 Hz fixed one, the cursor where the tone passes 100 Hz."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.sine import LOG, SineSweepSpecification, SineTone
    from visualdynamics.core.sine_tracking import SineTracking
    from visualdynamics.plot import plot_tracking_filter
    from visualdynamics.units import SI

    rate = 16384.0
    dof = '101Z+'
    tone = SineTone('Sine Tone 1', 0.5, [20.0, 320.0], [[10.0], [10.0]],
                    [LOG], [60.0])
    spec = SineSweepSpecification([tone], [dof], ordinate_unit='m/s**2')
    t = np.arange(int((0.5 + tone.duration() + 0.25) * rate)) / rate
    local = np.clip(t - 0.5, 0.0, None)
    inside = (t >= 0.5) & (local <= tone.duration())
    phase = tone.phase_at(local)
    # the harmonic figure's record, with noise at a fifth of the tone
    # over the whole recording, lead-in and tail included
    x = np.where(inside, 10.0 * (np.cos(phase) + 0.3 * np.cos(3.0 * phase)),
                 0.0)
    x = x + 2.0 * np.random.default_rng(7).standard_normal(len(t))
    history = TimeHistory(t, x[None], response_dof=[dof],
                          ordinate_dim=['acceleration'])
    for setting, name in ((SineTracking(proportional=0.5), 'proportional'),
                          (SineTracking(fixed=5.0), 'fixed')):
        plot_tracking_filter(history, spec, setting, onset=0.5,
                             cursor_hz=100.0, unit_system=SI, show=False,
                             path=str(OUT / f'sine-tracking-band-{name}.png'))
        print(f'{setting.describe():>34}: drawn')


def srs_tolerance():
    """A shock spectrum judged point by point: a requirement written at
    three breakpoints, a maximax spectrum on it but for a narrow, deep
    dip, and a negative spectrum that falls away at the top. The dip
    leaves the RMS reading well inside 3 dB and fails the points rule."""
    import numpy as np

    from visualdynamics.core.compliance import srs_errors, srs_tolerance
    from visualdynamics.core.data import ShockSpecification, Srs
    from visualdynamics.plot import plot_srs_tolerance

    spec = ShockSpecification.from_breakpoints(
        [10.0, 100.0, 2000.0], [10.0, 100.0, 100.0], '101Z+',
        ordinate_unit='g')
    f = spec.abscissa
    octaves = np.log2(f / 10.0)
    # a little ripple, as a synthesized pulse leaves, and a dip a third
    # of an octave wide and 5 dB deep near 400 Hz
    db = 0.6 * np.sin(2.0 * np.pi * octaves / 1.5)
    db -= 5.0 * np.exp(-0.5 * (np.log2(f / 400.0) / 0.12) ** 2)
    maximax = Srs(f, spec.ordinate * 10.0 ** (db / 20.0),
                  response_dof=['101Z+'], ordinate_dim=['acceleration'],
                  kind='maximax')
    # the negative spectrum on the requirement to the knee, then falling
    # 6 dB an octave from 400 Hz: a pulse whose other side is small
    fall = -0.4 - 6.0 * np.clip(np.log2(f / 400.0), 0.0, None)
    negative = Srs(f, spec.ordinate * 10.0 ** (fall / 20.0),
                   response_dof=['101Z+'], ordinate_dim=['acceleration'],
                   kind='negative')
    plot_srs_tolerance([maximax, negative], spec, outer=(-3.0, 6.0),
                       show=False, size=(1000, 640),
                       path=str(OUT / 'srs-tolerance.png'))
    for data in (maximax, negative):
        result = srs_tolerance(spec, data, outer=(-3.0, 6.0))
        (row,) = srs_errors(spec, data)
        rms_db = row.db
        print(f'{data.kind:>9}: RMS {rms_db:+.2f} dB, '
              f'{result["inside_count"]}/{result["points"]} inside, '
              f'worst {result["worst_db"]:+.1f} dB at '
              f'{result["worst_frequency"]:.0f} Hz, passed {result["passed"]}')
def shock_check():
    """A shock record checked before its spectrum: a ringing shock with
    a 4 g zero shift from the moment it arrives, and a compensated
    half-sine whose two spectra part above the pulse."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.shock_check import check_shock
    from visualdynamics.plot import plot_shock_check
    from visualdynamics.units import IN_LBF_S

    g = 9.80665
    rate = 102400.0
    shown = IN_LBF_S.with_units(acceleration='g')
    t = np.arange(int(0.25 * rate)) / rate

    def record(values):
        return TimeHistory(t, values[None], response_dof=['101Z+'],
                           ordinate_dim=['acceleration'])

    # four decaying tones from 600 Hz to 6 kHz arriving at 20 ms, the
    # ringing a structure answers a pyrotechnic event with. Each is the
    # rate of change of a decaying sine velocity, so the motion it
    # describes starts and ends at rest, as a structure's does. The
    # velocity rises over half a millisecond, as sin² of the time since
    # arrival, so the acceleration starts from zero: tones that started
    # on a step (590 g, 2026-10-16) left the trapezoid rule half a
    # sample of it, 1.11 in/s, and the good record ended at 6 % of its
    # peak for that reason alone
    after = np.clip(t - 0.02, 0.0, None)
    rise = 0.0005
    onset = np.where(after < rise, np.sin(np.pi * after / (2 * rise)) ** 2,
                     1.0)
    onset_rate = np.where(after < rise, np.pi / (2 * rise)
                          * np.sin(np.pi * after / rise), 0.0)

    def tone(f, level, decay):
        w = 2 * np.pi * f
        fading = level * np.exp(-after / decay)
        return (onset * fading * (np.cos(w * after)
                                  - np.sin(w * after) / (w * decay))
                + onset_rate * fading * np.sin(w * after) / w)

    ringing = sum(tone(f, level, decay)
                  for f, level, decay in ((600.0, 120.0, 0.012),
                                          (1500.0, 200.0, 0.008),
                                          (3000.0, 160.0, 0.005),
                                          (6000.0, 110.0, 0.003))) * g
    noise = 0.05 * g * np.random.default_rng(11).standard_normal(t.size)
    shift = np.where(t >= 0.02, 4.0 * g, 0.0)
    for name, values in (('good', ringing + noise),
                         ('shifted', ringing + noise + shift)):
        history = record(values)
        check, = check_shock(history)
        print(f'{name:>8}: ends at {check.end_velocity / 0.0254:7.3f} in/s, '
              f'{100 * check.end_velocity_share:6.2f} % of its peak, '
              f'peak {np.abs(values).max() / g:.0f} g, '
              f'{"flagged" if check.flagged else "passes"}')
        if name == 'shifted':
            plot_shock_check(history, low=10.0, high=10000.0,
                             unit_system=shown, show=False,
                             size=(1000, 900),
                             path=str(OUT / 'shock-check-shifted.png'))

    # a 50 g, 11 ms half-sine with a 2.5 g half-sine either side, each
    # carrying half its area: the shaker's compensated pulse
    duration, peak, depth = 0.011, 50.0 * g, 2.5 * g
    compensation = peak * duration / (2.0 * depth)
    opens = 0.125 - duration / 2.0

    def half_sine(start, height, length):
        inside = (t >= start) & (t <= start + length)
        return np.where(inside,
                        height * np.sin(np.pi * (t - start) / length), 0.0)

    pulse = (half_sine(opens, peak, duration)
             + half_sine(opens - compensation, -depth, compensation)
             + half_sine(opens + duration, -depth, compensation))
    history = record(pulse)
    check, = check_shock(history)
    print(f'compensated: change {check.velocity_change / 0.0254:.1f} in/s, '
          f'ends at {100 * check.end_velocity_share:.4f} % of its peak')
    plot_shock_check(history, low=10.0, high=10000.0, unit_system=shown,
                     show=False, size=(1000, 900),
                     path=str(OUT / 'shock-check-compensated.png'))
def averaging_confidence():
    """What an average is worth: a resonance under random excitation,
    averaged over sixteen Hann frames at three-quarter overlap, drawn
    with the chi-square band its equivalent degrees of freedom give."""
    import numpy as np
    from scipy.signal import iirpeak, lfilter

    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.units import SI

    rate = 2048.0
    averaging = Averaging(frame_length=512, overlap=0.75, window='hann',
                          frames=16)
    noise = np.random.default_rng(11).standard_normal(averaging.span)
    # one resonance at 200 Hz over a flat floor, so the band is seen
    # riding a peak as well as a plateau
    b, a = iirpeak(200.0, 8.0, fs=rate)
    x = 0.2 * noise + lfilter(b, a, noise)
    history = TimeHistory(np.arange(averaging.span) / rate, x[None],
                          response_dof=['101Z+'],
                          ordinate_dim=['acceleration'])
    history.averaging = averaging
    psd = history.compute_psds()
    psd.plot(marks='confidence', unit_system=SI, show=False,
             path=str(OUT / 'averaging-confidence.png'))
    worth = averaging.statistical_dof()
    print(f'DOF {worth.nominal} nominal, {worth.equivalent:.1f} equivalent')


def tone_readout():
    """One composite read at 1 Hz and at 4 Hz lines: a 1 grms floor from
    5 to 500 Hz with tones of 0.12, 1.944, 2.5 and 1.5 g peak added. The
    spikes move with the spacing; the amplitudes read off the lines do
    not, and the small tone at the floor's edge is called uncertain."""
    import warnings

    import numpy as np

    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.plot import plot_tone_readout

    fs, seconds, g = 2048.0, 60.0, 9.80665
    tones = ((4.86, 0.12), (19.44, 1.944), (38.88, 2.50), (58.32, 1.50))
    rng = np.random.default_rng(3304)
    n = int(fs * seconds)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum[(f < 5.0) | (f > 500.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x = x / np.sqrt(np.mean(x ** 2))
    t = np.arange(n) / fs
    for frequency, amplitude in tones:
        x = x + amplitude * np.sin(2.0 * np.pi * frequency * t + frequency)
    history = TimeHistory(t, x[None] * g, response_dof=['1Z+'],
                          ordinate_dim=['acceleration'], ordinate_unit=['g'])
    for block, name in ((2048, 'tone-readout-1hz.png'),
                         (512, 'tone-readout-4hz.png')):
        averaging = Averaging(frame_length=block, overlap=0.5,
                              window='hann').filled(n, fs)
        asd = history.compute_psds(averaging)
        asd.log_abscissa = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot_tone_readout(asd, [4.86, 19.44, 58.32], window=averaging,
                              snap=False,
                              band=(2.0, 200.0), show=False,
                              size=(1000, 600), path=str(OUT / name))
            for frequency, amplitude in tones:
                reading = asd.read_tone(frequency, window=averaging,
                                        snap=False)
                state = ('uncertain' if reading.uncertain else
                         'between lines' if reading.between_lines else '')
                print(f'{asd.abscissa[1]:g} Hz lines, {frequency:5.2f} Hz '
                      f'({amplitude:.3f} g): {reading.amplitude / g:.3f} g '
                      f'peak, spike {reading.spike / g ** 2:.4f} g2/Hz, '
                      f'{reading.on_line / g ** 2:.4f} on a line, floor '
                      f'{reading.floor / g ** 2:.5f} {state}')


def time_statistics():
    """A record's time statistics: twelve seconds of a flat 20 to
    2,000 Hz profile at 0.04 g^2/Hz, through a 25 g range that its
    largest peaks pass, read with the default bands."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.plot import plot_time_statistics
    from visualdynamics.units import SI

    g = 9.80665
    rate = 5120.0
    n = int(12.0 * rate)
    # white noise shaped line by line to the profile: unit variance is
    # an ASD of 2/rate, so each line in the band is scaled to 0.04
    spectrum = np.fft.rfft(np.random.default_rng(1403).standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / rate)
    gain = np.where((f >= 20.0) & (f <= 2000.0),
                    np.sqrt(0.04 / (2.0 / rate)), 0.0)
    clean = np.fft.irfft(spectrum * gain, n)
    history = TimeHistory(np.arange(n) / rate,
                          np.clip(clean, -25.0, 25.0)[None] * g,
                          response_dof=['1Z+'],
                          ordinate_dim=['acceleration'],
                          ordinate_unit=['g'])
    stats = history.time_statistics(0, full_scale=25.0 * g)
    print(f'through 25 g: {stats.rms / g:.2f} grms, crest '
          f'{stats.crest:.2f}, kurtosis {stats.kurtosis:.2f}, '
          f'{stats.rail} on the rail; {stats.describe()}')
    plot_time_statistics(history, 0, full_scale=25.0 * g,
                         unit_system=SI.with_units(acceleration='g'),
                         show=False, size=(1000, 980),
                         path=str(OUT / 'time-statistics-clipped.png'))


def replication_segments():
    """A minute of rough road replicated, judged segment by segment: a
    stationary stretch, a pothole and a braking transient. The run's
    error elsewhere is under five percent of the record's rms, and it
    misses most of the pothole's 250 Hz ring, which the whole-record
    figure hides and the pothole's own spectrum and peaks do not.
    The record is the one `tests/test_replication_segments.py` judges."""
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from visualdynamics.core.data import TimeHistory, TransientSpecification
    from visualdynamics.plot import plot_replication_segments
    from visualdynamics.units import SI

    rate, seconds = 2048.0, 60.0
    n = int(seconds * rate)
    t = np.arange(n) / rate
    rng = np.random.default_rng(27)

    def ring(at, peak, frequency, zeta=0.06):
        w = 2.0 * np.pi * frequency
        s = np.clip(t - at, 0.0, None)
        return np.where(t >= at, peak * np.exp(-zeta * w * s)
                        * (np.cos(w * s) - zeta * np.sin(w * s)), 0.0)

    def filtered(x, low, high=None):
        y = sosfiltfilt(butter(2, low, 'highpass', fs=rate, output='sos'), x)
        if high is not None:
            y = sosfiltfilt(butter(2, high, 'lowpass', fs=rate,
                                   output='sos'), y)
        return y

    # ride motion and a broadband floor to 400 Hz, onto and off the
    # course over 2 s, swelling as the vehicle brakes at 46 s
    f = np.fft.rfftfreq(n, 1.0 / rate)
    asd = np.zeros_like(f)
    inside = (f >= 0.7) & (f <= 400.0)
    asd[inside] = np.exp(np.interp(
        np.log(f[inside]), np.log([0.7, 3.5, 6.0, 150.0, 400.0]),
        np.log([0.02, 0.02, 0.015, 0.015, 0.0015])))
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal(n))
                     * np.sqrt(asd * rate / 2.0), n)
    x *= np.clip(np.minimum(t, seconds - t) / 2.0, 0.0, 1.0)
    x *= 1.0 + 0.8 * np.exp(-((t - 46.0) / 2.5) ** 2)
    # a dozen impacts ringing a bracket at 60 and 250 Hz, the largest a
    # 14 g pothole at 32 s
    for at, peak in ((5.2, 9.0), (9.7, 7.5), (13.1, 10.0), (17.8, 8.0),
                     (21.4, 11.0), (26.0, 7.0), (32.0, 14.0), (36.5, 9.5),
                     (39.9, 8.5), (52.3, 10.5), (55.6, 7.5), (57.1, 9.0)):
        x += ring(at, 0.4 * peak, 60.0) + ring(at, 0.6 * peak, 250.0)
    reference = filtered(x - x.mean(), 3.0)
    error = filtered(rng.standard_normal(n), 3.0, 400.0)
    control = (reference + 0.044 * reference.std() / error.std() * error
               - 0.7 * ring(32.0, 0.6 * 14.0, 250.0))

    def record(values, kind):
        return kind(t, values[None] * 9.80665, response_dof=['1Z+'],
                    ordinate_dim=['acceleration'], ordinate_unit=['g'])

    g = SI.with_units('SI (g)', acceleration='g')
    marks = [(6.0, 30.0, 'stationary', 'rough road'),
             (31.9, 32.7, 'shock', 'pothole'),
             (41.0, 51.0, 'transient', 'braking')]
    run = record(control, TimeHistory)
    target = record(reference, TransientSpecification)
    plot_replication_segments(run, target, marks, band=(5.0, 400.0),
                              window=0.5, unit_system=g, show=False,
                              size=(1100, 820),
                              path=str(OUT / 'replication-segments.png'))
    from visualdynamics.core.replication_segments import replication_segments

    result = replication_segments(run, target, marks, band=(5.0, 400.0),
                                  window=0.5)
    print(result.describe())
    for segment in result.segments:
        print(f'  {segment.describe()}; {segment.share:.2f} % of the mean '
              f'square, own rms error {segment.rms_error:.1f} %')
        for check in segment.checks:
            print(f'    {check.describe()}')


def sweep_rate():
    """A sweep from 5 to 100 Hz at 4 oct/min checked against three
    modes: a 22 Hz mode with a Q of 40 that it passes too fast to build,
    and two it lets build."""
    from visualdynamics.core.sine import LOG, SineSweepSpecification, SineTone
    from visualdynamics.core.sweep_rate import check_sweep_rate
    from visualdynamics.plot import plot_sweep_rate

    g = 9.80665
    tone = SineTone('5 to 100 Hz at 4 oct/min', 0.0, [5.0, 100.0],
                    [[g], [g]], [LOG], [4.0])
    spec = SineSweepSpecification([tone], ['1Z+'], ordinate_unit='m/s**2')
    modes = [(9.0, 10.0), (22.0, 40.0), (60.0, 15.0)]
    for check in check_sweep_rate(spec, modes):
        print(f'{check.describe()}')
    plot_sweep_rate(spec, modes, theme='light', show=False, size=(1000, 620),
                    path=str(OUT / 'sweep-rate.png'))


def signature():
    """Two signatures of a two-mode article, before and after a test,
    compared mode by mode: a global first mode that drops 3.3 percent
    in frequency and 28 percent in height, and a local mode near
    400 Hz that moves further and does not count."""
    import numpy as np

    from visualdynamics.core.data import Frf
    from visualdynamics.core.signature import SignatureMode, compare_signatures
    from visualdynamics.plot import plot_signature

    def transmissibility(f, modes):
        # each mode a resonator on the moving base, seen at the point
        # by its share of the motion
        out = np.zeros(f.shape, dtype=complex)
        for fn, q, share in modes:
            r, zeta = f / fn, 0.5 / q
            out += share * (1 + 2j * zeta * r) / (1 - r ** 2 + 2j * zeta * r)
        return out

    f = np.arange(1, 2401) * 0.25
    signatures = [
        Frf(f, transmissibility(f, modes)[None], response_dof=['101X+'],
            reference_dof=['1X+'],
            ordinate_dim=['acceleration/acceleration'])
        for modes in ([(84.07, 21.1, 0.85), (412.0, 30.0, 0.15)],
                      [(81.32, 15.25, 0.85), (377.0, 20.0, 0.15)])]
    modes = [84.0, SignatureMode(412.0, counts=False)]
    plot_signature(*signatures, modes, show=False, size=(1300, 640),
                   path=str(OUT / 'signature.png'))
    result = compare_signatures(*signatures, modes)
    for shift in result.shifts:
        print(f'{shift.mode.label():>7}: {shift.describe()}')
    print(f'{"":>7}  {result.describe()}')


def replication_fit():
    """A road record checked against a shaker before its first pass:
    sixty seconds at a bracket, 2 grms with a 14 g pothole, logged
    0.001 g high, against a 2 in stroke carrying 150 lb on a body
    twenty times as heavy."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.replication_fit import replication_fit
    from visualdynamics.core.shaker import ShakerRating
    from visualdynamics.plot import plot_replication_fit
    from visualdynamics.units import IN_LBF_S

    g, rate, seconds = 9.80665, 2048.0, 60.0
    samples = round(seconds * rate)
    t = np.arange(samples) / rate
    # ride motion from 0.7 to 3.5 Hz on a floor to 400 Hz: white noise
    # shaped line by line to the breakpoints, straight on log-log axes
    frequency = np.fft.rfftfreq(samples, 1.0 / rate)
    density = np.zeros_like(frequency)
    breakpoints = np.array([0.7, 3.5, 6.0, 150.0, 400.0])
    levels = np.array([0.02, 0.02, 0.015, 0.015, 0.0015])
    inside = (frequency >= breakpoints[0]) & (frequency <= breakpoints[-1])
    density[inside] = np.exp(np.interp(np.log(frequency[inside]),
                                       np.log(breakpoints), np.log(levels)))
    noise = np.random.default_rng(3404).standard_normal(samples)
    x = np.fft.irfft(np.fft.rfft(noise) * np.sqrt(density * rate / 2.0),
                     samples)
    # onto and off the course over 2 s, a braking swell at 46 s, and a
    # dozen impacts ringing the bracket's modes at 60 and 250 Hz
    x *= np.clip(np.minimum(t, seconds - t) / 2.0, 0.0, 1.0)
    x *= 1.0 + 0.8 * np.exp(-((t - 46.0) / 2.5) ** 2)
    for hit, peak in ((5.2, 9.0), (9.7, 7.5), (13.1, 10.0), (17.8, 8.0),
                      (21.4, 11.0), (26.0, 7.0), (32.0, 14.0), (36.5, 9.5),
                      (39.9, 8.5), (52.3, 10.5), (55.6, 7.5), (57.1, 9.0)):
        since = np.clip(t - hit, 0.0, None)
        for mode, share in ((60.0, 0.4), (250.0, 0.6)):
            w = 2.0 * np.pi * mode
            x += (t >= hit) * share * peak * np.exp(-0.06 * w * since) * (
                np.cos(w * since) - 0.06 * np.sin(w * since))
    x = x - x.mean() + 0.001
    record = TimeHistory(t, x[None] * g, response_dof=['1Z+'],
                         ordinate_dim=['acceleration'])
    rating = ShakerRating(force=17793.0, stroke=0.0508, velocity=1.778,
                          moving_mass=68.04, body_mass=1361.0)
    fit = replication_fit(record, rating)
    print(f'replication fit: {fit.describe()}')
    plot_replication_fit(record, rating,
                         unit_system=IN_LBF_S.with_units(acceleration='g'),
                         show=False, size=(1100, 800),
                         path=str(OUT / 'replication-fit.png'))


def stroke_readout():
    """Whether a pulse fits on the shaker: the 40 g, 13 ms half-sine
    compensated at a fifth of its peak, against a 2.5 in stroke and a
    70 in/s shock velocity, with a moving mass of 100 lb on a body of
    3,000 lb and a 6,000 lbf force rating."""
    from visualdynamics.core.stroke import compensated_half_sine, read_stroke
    from visualdynamics.plot import plot_stroke
    from visualdynamics.units import DEFAULT_SYSTEM, to_si

    pulse = compensated_half_sine(40.0, 0.013, 102400.0, unit='g')
    ratings = {'stroke': float(to_si(2.5, 'in')),
               'velocity': float(to_si(70.0, 'in/s')),
               'force': float(to_si(6000.0, 'lbf')),
               'moving_mass': float(to_si(100.0, 'lb')),
               'body_mass': float(to_si(3000.0, 'lb'))}
    reading = read_stroke(pulse, **ratings)
    for line in reading.describe(DEFAULT_SYSTEM):
        print(f'  {line}')
    plot_stroke(pulse, **ratings, unit_system=DEFAULT_SYSTEM, show=False,
                size=(1000, 900), path=str(OUT / 'stroke-readout.png'))


def dwell():
    """A tone held at a resonance and then stopped: a 1 g tone at 20 Hz
    on a one-mass resonator of Q 30, ninety cycles on and sixty off,
    read against time with the peak over each cycle. The level builds to
    95 % in about Q cycles and rings down to 5 % in about as many."""
    import numpy as np
    from scipy.signal import lfilter

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.dwell import read_dwell
    from visualdynamics.core.sine import SineSweepSpecification, SineTone
    from visualdynamics.core.srs import ramp_invariant
    from visualdynamics.plot import plot_dwell

    g, rate, fn, q = 9.80665, 4000.0, 20.0, 30.0
    n = np.arange(150 * 200)                     # 200 samples a cycle
    x = np.sin(2.0 * np.pi * fn * n / rate) * (n < 90 * 200)
    b, a = ramp_invariant([fn], rate, 1.0 / (2.0 * q))
    y = lfilter(b[0], a[0], x)
    history = TimeHistory(n / rate, np.vstack([x, y]) * g,
                          response_dof=['1Z+', '101Z+'],
                          ordinate_dim=['acceleration'] * 2,
                          ordinate_unit=['m/s**2'] * 2)
    spec = SineSweepSpecification([SineTone.dwell('Dwell', fn, [g], 4.5)],
                                  ['1Z+'], ordinate_unit='m/s**2')
    plot_dwell(history, spec, onset=0.0, show=False, size=(1000, 600),
               path=str(OUT / 'dwell.png'))
    reading = read_dwell(history, spec, onset=0.0)
    print(f'dwell: {reading.level[29] / g:.2f} g at {reading.seconds[29]:.2f} s;'
          f' {100 * reading.share_at(q):.1f} % after Q cycles, '
          f'{100 * reading.share_at(2 * q):.1f} % after 2Q, '
          f'{100 * reading.remaining_at(q):.1f} % Q cycles after the stop; '
          f'{reading.describe()}')


def mixed_mode():
    """Two mixed tests read as one specification each. Sine on random:
    one minute of a 1 grms floor from 5 to 500 Hz with tones of 0.12,
    1.944, 2.5 and 1.5 g peak, the guide composite of the tone readout,
    its sheet reading 1.00 for the floor, 2.48 for the tones and 2.67
    for the total. Random on random: half a minute of the same floor
    with a band of 0.02 g^2/Hz plus or minus 5 percent about 300 Hz and
    a 16 Hz band of 0.05 g^2/Hz sweeping 150 to 250 Hz at 10 Hz/s."""
    import numpy as np

    from visualdynamics.core.data import Specification, TimeHistory
    from visualdynamics.core.mixed_mode import (
        MixedSpecification,
        Narrowband,
        Tone,
        judge_mixed,
    )
    from visualdynamics.plot import plot_mixed_mode
    from visualdynamics.units import IN_LBF_S

    fs, g = 2048.0, 9.80665
    us = IN_LBF_S.with_units(f'{IN_LBF_S.name} (g)', acceleration='g')
    floor = Specification(np.array([5.0, 500.0]),
                          np.array([[1.0 / 495.0] * 2]) * g ** 2,
                          response_dof=['1Z+'],
                          ordinate_dim=['acceleration**2/frequency'])

    def noise(seconds, low, high, grms, rng):
        n = int(fs * seconds)
        spectrum = np.fft.rfft(rng.standard_normal(n))
        f = np.fft.rfftfreq(n, 1.0 / fs)
        spectrum[(f < low) | (f > high)] = 0.0
        x = np.fft.irfft(spectrum, n)
        return x * grms / np.sqrt(np.mean(x ** 2))

    def record(x):
        t = np.arange(len(x)) / fs
        return TimeHistory(t, x[None] * g, response_dof=['1Z+'],
                           ordinate_dim=['acceleration'], ordinate_unit=['g'])

    # sine on random, the tone readout's composite
    tones = ((4.86, 0.12), (19.44, 1.944), (38.88, 2.50), (58.32, 1.50))
    x = noise(60.0, 5.0, 500.0, 1.0, np.random.default_rng(3304))
    t = np.arange(len(x)) / fs
    for frequency, amplitude in tones:
        x = x + amplitude * np.sin(2.0 * np.pi * frequency * t + frequency)
    spec = MixedSpecification(floor, [Tone(f, a * g) for f, a in tones],
                              seconds=60.0)
    history = record(x)
    plot_mixed_mode(history, spec, unit_system=us, show=False,
                    size=(1400, 620), path=str(OUT / 'mixed-mode-sheet.png'))
    reading = judge_mixed(history, spec)
    print(f'mixed: residual every line {reading.floor.residual / g:.3f} '
          f'grms, tones {reading.tones_measured / g:.3f} g rms, '
          f'{reading.describe("g", 1 / g)}')

    # random on random: a fixed band and a swept one over the floor
    rng = np.random.default_rng(3305)
    seconds = 30.0
    x = noise(seconds, 3.0, 520.0, np.sqrt(517.0 / 495.0), rng)
    x = x + noise(seconds, 285.0, 315.0, np.sqrt(0.02 * 30.0), rng)
    swept = Narrowband(150.0, 0.05 * g ** 2, width=16.0, sweep_to=250.0,
                       rate=10.0, name='track')
    n = len(x)
    base = np.fft.fft(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    base[np.abs(np.fft.fftfreq(n, 1.0 / fs)) > 8.0] = 0.0
    phase = 2.0 * np.pi * np.cumsum(swept.center_at(np.arange(n) / fs)) / fs
    y = np.real(np.fft.ifft(base) * np.exp(1j * phase))
    x = x + y * np.sqrt(0.05 * 16.0 / np.mean(y ** 2))
    bands = MixedSpecification(
        floor, None, [Narrowband.plus_minus(300.0, 5.0, 0.02 * g ** 2,
                                            name='engine'), swept])
    plot_mixed_mode(record(x), bands, unit_system=us, show=False,
                    size=(1400, 620), path=str(OUT / 'mixed-mode-bands.png'))
    for row in judge_mixed(record(x), bands).narrowbands:
        print(f'band {row.name}: {row.measured / g:.3f} grms against '
              f'{row.reference / g:.3f}, {row.error_db:+.2f} dB in '
              f'{row.low_db:+.2f} to {row.high_db:+.2f} at {row.dof:.0f} '
              f'dof ({row.slices} slices of {row.slice_dof:.1f})')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    psd_reading()
    compliance()
    pulse_check()
    sine_tracking()
    sine_tracking_band()
    srs_tolerance()
    shock_check()
    averaging_confidence()
    tone_readout()
    time_statistics()
    replication_segments()
    sweep_rate()
    signature()
    replication_fit()
    stroke_readout()
    dwell()
    mixed_mode()
    print('done')


if __name__ == '__main__':
    main()
