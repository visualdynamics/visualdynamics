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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    psd_reading()
    compliance()
    print('done')


if __name__ == '__main__':
    main()
