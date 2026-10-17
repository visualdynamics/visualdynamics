# Checking a classical pulse

A classical shock test asks for a pulse of a given shape, peak and
duration: a half-sine, a terminal-peak sawtooth or a trapezoid. It is
accepted on the measured pulse, point by point against a tolerance
envelope drawn around the nominal shape, and as a whole on its
velocity change, the area under it. `visualdynamics.core.classical_pulse`
makes that judgment, and `plot_pulse_check` and the **Pulse Check**
reading on a time history's bar draw it.

## The pulse

`ClassicalPulse(shape, peak, duration, edge)` is the nominal pulse,
the peak signed and in SI (m/s²), the duration and edge in seconds.

| Shape | Duration | Edge | Nominal velocity change |
|---|---|---|---|
| `'half-sine'` | the half cycle | none | 2AT/π |
| `'sawtooth'` | the straight rise | the fall after it; instant if omitted | AT/2 |
| `'trapezoid'` | rise, top and fall together | the rise and the fall, each; required | A(T − e) |

The sawtooth's nominal is AT/2 with the fall taken as instant, as the
formula is usually stated. A fall that takes time adds half its length
times the peak to the measured change, and the check reports that as
it is rather than moving the nominal: a 40 g, 11 ms sawtooth whose fall
takes 1.1 ms changes velocity by 93.4 in/s against 84.9, a tenth over
before anything else varies.

## What it is judged against

Each tolerance is an argument, because the document the test procedure
cites decides it. The defaults are this package's starting point, not
any one document's figures:

| Argument | Default | What it holds |
|---|---|---|
| `band` | 0.2 | the band either side of the ideal shape on the pulse, and of zero before and after it, as a share of the peak; it holds the measured peak to 0.8 to 1.2 of nominal as well |
| `duration_tolerance` | 0.1 | the measured duration within a tenth of nominal |
| `edge_limit` | 0.1 | the sawtooth's fall, or the trapezoid's rise and fall, no longer than a tenth of the duration |
| `velocity_tolerance` | 0.2 | the velocity change within a fifth of nominal; documents and their revisions differ on this one, so look it up |
| `quiet` | 2.5 | how many durations before and after the pulse are held to the band |
| `window` | (0.4, 0.1) | the velocity window, in durations before the pulse starts and after it ends |

The envelope is the highest and the lowest of every admissible pulse
at each instant, widened by the band: admissible is the shape at any
duration inside the duration tolerance and with any edge from instant
to the limit. Built that way it agrees with the duration and edge
tolerances rather than contradicting them, so a pulse a twentieth long
is inside the envelope as well as inside its duration tolerance. At a
half-sine's crest the envelope's lower edge, the lowest of the
admissible durations less a fifth, sits at 0.788 of the peak, and the
peak itself is held to 0.8 apart.

## How the measured pulse is read

The pulse is the record's largest excursion in the nominal's
direction. Its start, end and edges come from where it crosses a tenth
and nine tenths of that peak, walking out from the peak so that noise
before and after the pulse is not taken for it: each straight edge
extended through its two crossings to zero, and for the half-sine the
span between its tenth-of-peak crossings over the share of the
duration that span covers. The envelope is placed on the measured
start, and every sample from 2.5 durations before it to 2.5 after the
latest admissible end is judged.

The velocity change is integrated raw, by the trapezoid rule with
nothing removed (`filters.integrate` with `drift_corner=None`), over
the window from 0.4 nominal durations before the measured start to 0.1
after the measured end. The window is part of the rule: a shaker's
compensated pulse, with a shallow pulse of the other sign either side,
changes velocity by nothing over the whole record by design, and a mean
taken out first would take out exactly the area being measured. The
samples have to be evenly spaced, and a record holding a sample that
is not a number is refused.

This velocity change is the area under the record across the window:
the velocity where the window closes, counted from rest where it
opens. The [shock check](shock-check.md) reports a different number
under the same name, the span from the lowest velocity to the highest
across its window. For a lone pulse of one sign the two agree. An
overshoot of the other sign inside the window takes from the area and
leaves the span alone, and the part of a compensation pulse that runs
into the window counts against the area and not against the span.

`check_pulse(history, pulse)` returns a `PulseCheck` with the record
and the envelope over the judged span, the samples outside, the
velocity across the window, and these readings:

| Reading | What it is |
|---|---|
| `measured` | the start, end, duration, peak and edges read from the record |
| `outside_count`, `points` | samples outside the envelope, of those judged |
| `peak_ratio`, `duration_ratio` | measured over nominal |
| `velocity_change`, `nominal_velocity_change`, `velocity_ratio` | the change over the window, the formula's, and the one over the other |
| `envelope_ok`, `peak_ok`, `duration_ok`, `rise_ok`, `fall_ok`, `velocity_ok` | each part of the judgment |
| `covered` | the record reaches across the quiet either side and the velocity window |
| `passed`, `failures()`, `describe()` | the verdict, what was not met, and both in words |

## An example

A 40 g, 11 ms sawtooth made the way a shaker makes it: a 40 ms
half-sine of the other sign either side, each carrying half its area,
so the table starts and ends at rest; the fall takes 1 ms of the 1.1
allowed, and the record carries a noise floor.

```python
from visualdynamics.core.classical_pulse import ClassicalPulse, check_pulse
from visualdynamics.plot import plot_pulse_check
from visualdynamics.units import IN_LBF_S

g = 9.80665
check = check_pulse(history, ClassicalPulse('sawtooth', 40 * g, 0.011))
check.passed                      # True
check.velocity_change / 0.0254    # 91.2 in/s against 84.9
print(check.describe())

plot_pulse_check(history, 'sawtooth', 40 * g, 0.011,
                 unit_system=IN_LBF_S.with_units(acceleration='g'),
                 path='pulse.png')
```

Every point is inside the envelope, the compensation inside the band
either side, and the velocity change is 7.4 % over nominal: the fall's
own area, less the tail of the pre-pulse the window takes in.

![A compensated 40 g sawtooth on its envelope, the velocity window
shaded across it, and its velocity climbing across the window to
91.2 in/s inside the band about the nominal
84.9](images/pulse-check-sawtooth.png)

A 50 g, 11 ms half-sine followed by a fixture ringing at 700 Hz, its
first swings at 15 g, is flagged: the pulse is in, and 44 samples after
it are outside the band of 10 g, each marked. The velocity change is
within a percent of nominal, because only the ringing's first swing
falls inside the window. The band after the pulse is where the ringing
shows, and it should show somewhere: the test article felt it.

![A 50 g half-sine with a fixture ringing after it, the samples past the
band after the pulse marked](images/pulse-check-ringing.png)

## In the app

**Pulse Check** on a time history's bar replaces the trace with the
check, for the first selected record (or the object's first
acceleration channel), and puts the pulse and its tolerances in the
panel beside it: the shape, the peak in the window's units, the
duration and edge in milliseconds, and the band, duration, edge-limit
and velocity tolerances in percent with the quiet in durations. The
edge rows show only for a sawtooth or a trapezoid. Every edit redraws
the check. Under the settings the panel lists the measured peak,
duration and edges, the velocity change with its distance from
nominal, and the samples outside, any value out of its tolerance
marked "out", then the verdict. The reading is journaled as its
`plot_pulse_check` call.

A record opens on the half-sine it most resembles
(`suggest_pulse`), rounded, as a place to start. Type the test's own
pulse over it: a nominal read off the record being judged would pass
whatever the record did.

## Where judgment lives

- The tolerances. The defaults are an example; the envelope, the edge
  limit, the velocity tolerance and the record length are copied from
  the document the procedure cites, with its revision.
- The window around the pulse. The check reads the velocity change
  over its window only, so compensation outside it is not counted; a
  pre-pulse that runs into the window is, as the rule intends.
- A sawtooth at the edge of its fall. A fall at its full allowance
  spends half of a fifth's velocity tolerance on its own. Under a
  tenth, the same pulse sits on the edge of its band.
- A flagged record that may be a sensor problem. Integrate the record
  to velocity over its whole length first: a zero shift or a saturated
  channel makes the pulse and its velocity change fiction.
