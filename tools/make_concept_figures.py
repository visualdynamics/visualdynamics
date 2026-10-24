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


def transmissibility():
    """The two transmissibilities of one survey: 24 s of random from 20
    to 2000 Hz at 0.04 g²/Hz, through a 300 Hz resonance at Q 20, with
    0.6 g rms of noise the input did not cause on the response channel,
    analyzed at 2.5 Hz lines with 119 Hann frames at half overlap. H1
    and the ASD ratio are one curve where the coherence is high; above
    about 1 kHz the ratio parts from H1, and H1's band widens."""
    import numpy as np

    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.plot import plot_transmissibility

    g, fs, seconds = 9.80665, 5120.0, 24.0
    n = int(fs * seconds)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = np.fft.rfft(np.random.default_rng(1505).standard_normal(n))
    spectrum[(f < 20.0) | (f > 2000.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x *= np.sqrt(0.04 * 1980.0) / np.std(x)
    r, z = f / 300.0, 1.0 / 40.0
    resonance = (1.0 + 2j * z * r) / (1.0 - r ** 2 + 2j * z * r)
    y = np.fft.irfft(np.fft.rfft(x) * resonance, n)
    y += np.random.default_rng(1506).standard_normal(n) * 0.6
    history = TimeHistory(np.arange(n) / fs, np.vstack([x, y]) * g,
                          response_dof=['1Z+', '101Z+'],
                          ordinate_dim=['acceleration'] * 2,
                          ordinate_unit=['g'] * 2)
    history.averaging = Averaging(frame_length=2048, overlap=0.5,
                                  window='hann').filled(n, fs)
    reading = history.read_transmissibility('101Z+', '1Z+')
    plot_transmissibility(reading, band=(20.0, 1990.0), show=False,
                          size=(1000, 760),
                          path=str(OUT / 'transmissibility.png'))
    for frequency in (100.0, 300.0, 1000.0, 1500.0):
        point = reading.at(frequency)
        r = point.frequency / 300.0
        truth = abs((1.0 + 2j * z * r) / (1.0 - r ** 2 + 2j * z * r))
        print(f'{point.describe("g/g")}; truth {truth:.4g}')


def sine_distortion():
    """A 1 g sweep from 100 to 500 Hz at 100 Hz/s with a third harmonic
    at 10 % of the tone all the way and at 50 % from 250 to 350 Hz, as a
    rattle would add, sampled at 4096 Hz: the distortion against
    frequency, over the limit only through the rattle."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.sine import (
        SineSweepSpecification,
        SineTone,
        extract_sine,
    )
    from visualdynamics.core.sine_distortion import check_distortion
    from visualdynamics.plot import plot_sine_distortion

    g, rate = 9.80665, 4096.0
    tone = SineTone('Up', 0.5, [100.0, 500.0], [[g]] * 2, [0], [100.0])
    spec = SineSweepSpecification([tone], ['1Z+'], ordinate_unit='m/s**2')
    n = int(6.0 * rate)
    x = np.zeros(n)
    argument = tone.argument(1.0 / rate)
    drive = tone.frequency_at(np.arange(len(argument)) / rate)
    third = np.where((drive >= 250.0) & (drive <= 350.0), 0.5, 0.1)
    start = int(0.5 * rate)
    x[start:start + len(argument)] = g * (np.cos(argument)
                                          + third * np.cos(3.0 * argument))
    history = TimeHistory(np.arange(n) / rate, x[None, :],
                          response_dof=['1Z+'],
                          ordinate_dim=['acceleration'],
                          ordinate_unit=['m/s**2'])
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    plot_sine_distortion(history, spec, levels, show=False,
                         size=(1000, 600),
                         path=str(OUT / 'sine-distortion.png'))
    check = check_distortion(history, spec, levels)
    quiet = (check.frequency > 150.0) & (check.frequency < 230.0)
    over = check.over[0]
    print(f'sine distortion: ratio {np.median(check.ratio[0][quiet]):.4f} '
          f'and {100 * np.median(check.distortion[0][quiet]):.1f} % at '
          f'150-230 Hz; {int(over.sum())} of {over.size} lines over, '
          f'{check.frequency[over].min():.0f} to '
          f'{check.frequency[over].max():.0f} Hz; {check.describe()}')


def breakpoint_slopes():
    """A random specification entered by slope, read by segment: a
    level at 20 Hz, plus 3 dB per octave to 80 Hz, flat to 500 Hz,
    minus 3 dB per octave to 2000 Hz. The falling segment, short on
    log axes, is 60 percent of the area, and the running rms climbs
    most steeply there."""
    from visualdynamics.core.author import SpecificationDraft
    from visualdynamics.core.breakpoints import OCTAVE_DB
    from visualdynamics.plot import plot_breakpoints

    g2 = 9.80665 ** 2
    draft = SpecificationDraft(['1Z+'], ['acceleration'], [20.0, 80.0],
                               [[0.01 * g2, 0.04 * g2]], entry='slopes')
    draft = draft.with_slope_breakpoint(500.0, 0.0)
    draft = draft.with_slope_breakpoint(2000.0, -OCTAVE_DB)
    plot_breakpoints(draft, show=False, size=(1000, 700),
                     path=str(OUT / 'breakpoint-slopes.png'))
    reading = draft.reading(0)
    print('areas, g^2:', ', '.join(f'{a / g2:.2f}' for a in reading.areas),
          '| running rms, g:',
          ', '.join(f'{r / 9.80665:.2f}' for r in reading.running))


def amplitude_spectrum():
    """Two tones on a 0.89 grms floor from 20 to 2,000 Hz, read at 5 Hz
    lines under Hann and under the flat top: 1 g peak at 160 Hz, on a
    line, and 2 g peak at 502.5 Hz, midway between two. Both windows
    read the tone on a line at 1 g; Hann reads the one between lines
    1.42 dB low, the flat top within 0.01 dB."""
    import numpy as np

    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.plot import plot_amplitude_spectrum

    fs, seconds, g = 5120.0, 12.0, 9.80665
    rng = np.random.default_rng(1502)
    n = int(fs * seconds)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    # white noise of unit variance has a one-sided ASD of 2/fs
    gain = np.where((f >= 20.0) & (f <= 2000.0),
                    np.sqrt(0.0004 / (2.0 / fs)), 0.0)
    x = np.fft.irfft(spectrum * gain, n)
    t = np.arange(n) / fs
    x = (x + 1.0 * np.sin(2.0 * np.pi * 160.0 * t)
         + 2.0 * np.sin(2.0 * np.pi * 502.5 * t + 0.3))
    history = TimeHistory(t, x[None] * g, response_dof=['1Z+'],
                          ordinate_dim=['acceleration'], ordinate_unit=['g'])
    history.averaging = Averaging(frame_length=1024, overlap=0.5,
                                  window='hann').filled(n, fs)
    for window in ('hann', 'flattop'):
        plot_amplitude_spectrum(history, 0, window=window, show=False,
                                size=(1000, 600),
                                path=str(OUT / f'amplitude-spectrum-{window}.png'))
        read = history.amplitude_spectrum(window=window)
        on_line = read.tallest(0, band=(100.0, 200.0))
        between = read.tallest(0, band=(400.0, 600.0))
        print(f'{window}: {read.correction_text()}; {on_line[1] / g:.4f} g '
              f'at {on_line[0]:g} Hz, {between[1] / g:.4f} g at '
              f'{between[0]:g} Hz, floor about '
              f'{np.median(read.amplitude[0, 200:380]) / g:.4f} g')


def comparison_chance():
    """Lines outside a band by chance: white noise averaged over sixty
    unwindowed frames (120 degrees of freedom) against a flat
    specification at its true level, warning at 1.5 dB and abort at
    3 dB, with sixteen adjacent lines raised 2.5 dB as a real excursion
    would be. Judged eight lines at a time, the scatter goes and the
    excursion stays."""
    import numpy as np

    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.chance import compare_chance, describe_chance
    from visualdynamics.core.data import Specification, TimeHistory
    from visualdynamics.plot import plot_chance

    rate = 5120.0
    averaging = Averaging(frame_length=1024, overlap=0.0, window='boxcar',
                          frames=60)
    noise = np.random.default_rng(3).standard_normal(averaging.span)
    history = TimeHistory(np.arange(averaging.span) / rate, noise[None],
                          response_dof=['1Z+'], ordinate_dim=['acceleration'])
    history.averaging = averaging
    psd = history.compute_psds()
    # the excursion: sixteen lines from 1,200 Hz, 2.5 dB over
    raised = (psd.abscissa >= 1200.0) & (psd.abscissa < 1280.0)
    psd.ordinate[0, raised] *= 10.0 ** 0.25
    level = np.full((1, 2), 2.0 / rate)
    spec = Specification(np.array([30.0, 1970.0]), level, response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency',
                         **{f'{pair}_{side}': level * 10.0 ** (sign * db / 10.0)
                            for pair, db in (('warning', 1.5), ('abort', 3.0))
                            for side, sign in (('lower', -1), ('upper', 1))})
    for combine in (1, 8):
        result = compare_chance(spec, psd, scale_db=0, combine=combine)
        print(f'combine {combine}: {describe_chance(result, "warning")}')
    plot_chance(psd, spec, combine=8, scale_db=0, show=False,
                path=str(OUT / 'comparison-chance.png'))
def _two_drive_plant():
    """Two shakers driving two control accelerometers through two
    modes at 2 % damping: at 100 Hz both drives move together, at
    400 Hz against each other. Accelerance, 5 to 1000 Hz at 1.25 Hz."""
    import numpy as np

    from visualdynamics.core.data import Frf

    f = np.arange(5.0, 1000.0 + 0.625, 1.25)
    w = 2 * np.pi * f
    h = np.zeros((len(f), 2, 2), dtype=complex)
    for fn, phi in ((100.0, np.array([1.0, 1.0])),
                    (400.0, np.array([1.0, -1.0]))):
        wn = 2 * np.pi * fn
        h += np.outer(phi, phi)[None] * -w[:, None, None] ** 2 / (
            wn ** 2 - w[:, None, None] ** 2
            + 2j * 0.02 * wn * w[:, None, None])
    rows, resp, refs = [], [], []
    for i, r in enumerate(('101Z+', '102Z+')):
        for j, d in enumerate(('1Z+', '2Z+')):
            rows.append(h[:, i, j])
            resp.append(r)
            refs.append(d)
    return Frf(f, np.asarray(rows), response_dof=resp, reference_dof=refs,
               ordinate_dim=['acceleration/force'] * 4)


def matrix_checks():
    """The two matrix checks of a multi-input test. Conditioning: the
    two-drive plant of `_two_drive_plant`, whose columns are one at the
    first mode, where the condition number passes 100. Realizability:
    three control channels at 0.01 g²/Hz with coherence 0.36 between
    every pair, 1-2 and 2-3 in step, and the 1-3 phase rising from 0 at
    20 Hz to 180 degrees at 2000 Hz (linear in log frequency); past the
    phase the other two can carry, no signals have the matrix."""
    import numpy as np

    from visualdynamics.core.data import Specification
    from visualdynamics.plot import plot_conditioning, plot_realizability

    reading = _two_drive_plant().conditioning()
    plot_conditioning(reading, band=(5.0, 1000.0), show=False,
                      size=(1000, 700),
                      path=str(OUT / 'mimo-conditioning.png'))
    print(reading.describe())
    for frequency in (5.0, 100.0, 250.0, 400.0):
        print(reading.at(frequency).describe('m/s²/N'))

    f = np.geomspace(20.0, 2000.0, 201)
    phase = 180.0 * np.log(f / 20.0) / np.log(100.0)
    g2 = 9.80665 ** 2
    a = 0.6
    terms = {(0, 1): np.full(len(f), a + 0j), (1, 2): np.full(len(f), a + 0j),
             (0, 2): a * np.exp(1j * np.radians(phase))}
    channels = ['1Z+', '2Z+', '3Z+']
    rows, resp, refs = [], [], []
    for i in range(3):
        for j in range(3):
            if i == j:
                value = np.ones(len(f), dtype=complex)
            elif i < j:
                value = terms[(i, j)]
            else:
                value = np.conj(terms[(j, i)])
            rows.append(value * 0.01 * g2)
            resp.append(channels[i])
            refs.append(channels[j])
    spec = Specification(f, np.asarray(rows), response_dof=resp,
                         reference_dof=refs,
                         ordinate_dim=['acceleration**2/frequency'] * 9)
    reading = spec.realizability()
    plot_realizability(reading, band=(20.0, 2000.0), pair=(0, 2), show=False,
                       size=(1000, 860),
                       path=str(OUT / 'mimo-realizability.png'))
    print(reading.describe())
    print('largest changes', reading.changes())
    first = reading.bands()[0][0]
    k = reading.line(first)
    print(f'first failing line {first:.4g} Hz, 1-3 phase {phase[k]:.4g} deg')
    for frequency in (100.0, 1000.0, 2000.0):
        print(reading.at(frequency).describe())
        before, after, pb, pa = reading.pair(0, 2)
        k = reading.line(frequency)
        print(f'  1Z+-3Z+ coherence {before[k]:.3f} -> {after[k]:.3f}, '
              f'phase {pb[k]:.1f} -> {pa[k]:.1f}')


def plot_composition():
    """A figure no single view draws: a record's two quantities on the
    left, two runs' ASDs on the right on one pinned scale, with the
    second run dashed and a mode marked and labeled in words."""
    import numpy as np
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.core.data import Psd, TimeHistory
    from visualdynamics.plot import (
        as_series,
        build_plots,
        compose,
        data_curves,
        figure_plots,
        pin_range,
        to_view,
    )

    g = 9.80665
    rate = 2048.0
    t = np.arange(int(0.5 * rate)) / rate
    acceleration = g * np.sin(2 * np.pi * 12 * t) * np.exp(-3 * t)
    stroke = 0.002 * np.cos(2 * np.pi * 12 * t) * np.exp(-3 * t)
    record = TimeHistory(t, np.vstack([acceleration, stroke]),
                         response_dof=['1Z+', '1Z+'],
                         ordinate_dim=['acceleration', 'length'],
                         ordinate_unit=['m/s**2', 'm'])
    f = np.geomspace(20.0, 2000.0, 400)

    def run(peak, q):
        r = f / peak
        lift = np.sqrt((1 + (r / q) ** 2) / ((1 - r ** 2) ** 2 + (r / q) ** 2))
        return Psd(f, (0.01 * g ** 2 * lift ** 2)[None], ['1Z+'],
                   ordinate_dim=['acceleration**2/frequency'])

    before, after = run(180.0, 8.0), run(165.0, 6.0)

    def build(layout):
        left = layout.addLayout(row=0, col=0)
        right = layout.addLayout(row=0, col=1)
        build_plots(left, as_series([record]), theme='light')
        build_plots(right, as_series([('before', before), ('after', after)]),
                    theme='light')
        asd = figure_plots(right, 'acceleration**2/frequency')[0]
        pin_range(asd, x=(20.0, 2000.0), y=(1e-4, 1.0))
        _, dashed = data_curves(asd)
        pen = pg.mkPen(dashed.opts['pen'])
        pen.setStyle(Qt.PenStyle.DashLine)
        dashed.setPen(pen)
        x, y = to_view(asd, x=180.0, y=0.5)
        asd.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(
            '#555555', width=1, style=Qt.PenStyle.DotLine)))
        label = pg.TextItem('first mode before, 180 Hz', color='#222222',
                            anchor=(0, 1))
        label.setPos(x, y)
        asd.addItem(label)

    compose(build, path=str(OUT / 'plot-composition.png'), theme='light',
            size=(1000, 460), frequency_axis='log')
    print(f'plot composition: before peaks at '
          f'{f[np.argmax(before.ordinate[0])]:.0f} Hz, after at '
          f'{f[np.argmax(after.ordinate[0])]:.0f} Hz')


def rms_check():
    """A random record checked against its own ASD twice: as the package
    computes it, Hann at half overlap, which agrees within hundredths of
    a decibel, and with every line made 1.5 times too high, the Hann
    amplitude correction applied to a density, which is flagged and
    named. Twelve seconds at 5120 samples a second, flat at 0.04 g2/Hz
    from 20 to 2000 Hz, 1024-sample frames."""
    import copy

    import numpy as np

    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.tone_readout import noise_bandwidth
    from visualdynamics.plot import plot_rms_check

    fs, seconds, g = 5120.0, 12.0, 9.80665
    rng = np.random.default_rng(1502)
    n = int(fs * seconds)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum[(f < 20.0) | (f > 2000.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x = x * np.sqrt(0.04 * 1980.0) / np.sqrt(np.mean(x ** 2))
    history = TimeHistory(np.arange(n) / fs, x[None] * g,
                          response_dof=['1Z+'],
                          ordinate_dim=['acceleration'], ordinate_unit=['g'])
    averaging = Averaging(frame_length=1024, overlap=0.5,
                          window='hann').filled(n, fs)
    asd = history.compute_psds(averaging)
    wrong = copy.deepcopy(asd)
    wrong.ordinate = asd.ordinate * noise_bandwidth(averaging)
    for spectrum_, name in ((asd, 'rms-check-agree.png'),
                            (wrong, 'rms-check-amplitude.png')):
        plot_rms_check(spectrum_, history, averaging=averaging, show=False,
                       size=(1000, 600), path=str(OUT / name))
        check = spectrum_.rms_check(history, averaging=averaging)
        print(f'rms check: {check.describe("g", 1.0 / g)}')


def sine_profile():
    """A minimum-integrity profile of the kind test documents give,
    written in four segments (0.20 in double amplitude, 2 g, 0.036 in
    double amplitude, 5 g, 5 to 500 Hz) and drawn three ways with its
    crossovers checked."""
    from visualdynamics.core.sine_profile import (
        SineProfile,
        acceleration,
        displacement,
    )
    from visualdynamics.plot import plot_sine_profile

    profile = SineProfile([
        displacement(0.20, 'in', 'peak to peak', start=5.0, end=14.0),
        acceleration(2.0, 'g', start=14.0, end=33.0),
        displacement(0.036, 'in', 'peak to peak', start=33.0, end=52.1),
        acceleration(5.0, 'g', start=52.1, end=500.0)])
    plot_sine_profile(profile, show=False, size=(1000, 900),
                      path=str(OUT / 'sine-profile.png'))
    frequency, peak = profile.maxima()['velocity']
    print(f'sine profile: {"; ".join(profile.describe())}; most velocity '
          f'{peak / 0.0254:.2f} in/s at {frequency:.4g} Hz')


def sine_frequency():
    """A 1 g sweep from 20 to 500 Hz at 100 Hz/s whose tone drifts
    high along it, on frequency at the start and 2.5 % high at the end,
    sampled at 4096 Hz, against a band of 2 % or 1 Hz."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.sine import (
        SineSweepSpecification,
        SineTone,
        extract_sine,
    )
    from visualdynamics.core.sine_frequency import check_frequency
    from visualdynamics.plot import plot_sine_frequency

    g, rate = 9.80665, 4096.0
    tone = SineTone('Up', 0.5, [20.0, 500.0], [[g]] * 2, [0], [100.0])
    spec = SineSweepSpecification([tone], ['1Z+'], ordinate_unit='m/s**2')
    t = np.arange(int(tone.duration() * rate)) / rate
    drive = tone.frequency_at(t) * (1.0 + 0.025 * t / tone.duration())
    x = np.zeros(int(6.0 * rate))
    start = int(0.5 * rate)
    x[start:start + len(t)] = g * np.cos(2.0 * np.pi * np.cumsum(drive)
                                         / rate)
    history = TimeHistory(np.arange(len(x)) / rate, x[None, :],
                          response_dof=['1Z+'],
                          ordinate_dim=['acceleration'],
                          ordinate_unit=['m/s**2'])
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    plot_sine_frequency(history, spec, levels, tolerance=0.02, floor=1.0,
                        show=False, size=(1000, 600),
                        path=str(OUT / 'sine-frequency.png'))
    check = check_frequency(history, spec, levels, tolerance=0.02,
                            floor=1.0)
    over = check.over[0]
    print(f'sine frequency: outside from {check.specified[over].min():.0f} '
          f'Hz; {check.describe()}')


def resample():
    """A record taken from 81,920 to 5,120 samples a second: a 0.1 g
    tone at 300 Hz on a light floor, and a 1 g sensor resonance at
    4,600 Hz above the new Nyquist frequency, which folds to 520 Hz."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.resample import read_resample
    from visualdynamics.plot import plot_resample
    from visualdynamics.units import DEFAULT_SYSTEM

    rate, samples, g = 81920.0, 16 * 8192, 9.80665
    t = np.arange(samples) / rate
    floor = 0.005 * np.random.default_rng(14).standard_normal(samples)
    y = (0.1 * np.sin(2 * np.pi * 300.0 * t)
         + np.sin(2 * np.pi * 4600.0 * t) + floor)
    history = TimeHistory(t, y[None] * g, response_dof=['1Z+'],
                          ordinate_dim=['acceleration'], ordinate_unit=['g'])
    reading = read_resample(history, 5120.0, frequencies=[4600.0])
    print(f'  {reading.describe()}')
    fold = reading.folds[0]
    print(f'  at 520 Hz: bare {fold.bare / g ** 2:.4g}, after '
          f'{fold.after / g ** 2:.4g}, filtered {fold.filtered / g ** 2:.3g}'
          f' g^2/Hz; gain at the span {reading.gain_at_span:.2f} dB, at '
          f'{reading.lowest_fold:g} Hz {reading.gain_at_fold:.1f} dB')
    plot_resample(history, 5120.0, frequencies=[4600.0],
                  unit_system=DEFAULT_SYSTEM, show=False, size=(1100, 720),
                  path=str(OUT / 'resample.png'))


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
    transmissibility()
    matrix_checks()
    sine_distortion()
    breakpoint_slopes()
    amplitude_spectrum()
    comparison_chance()
    plot_composition()
    rms_check()
    sine_profile()
    sine_frequency()
    resample()
    print('done')


if __name__ == '__main__':
    main()
