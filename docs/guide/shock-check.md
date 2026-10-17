# Checking a shock record before its spectrum

A shock response spectrum is computed from whatever the record holds,
including what the table never did. The commonest fault is a zero
shift: after a severe shock a piezoelectric accelerometer can be left
reading a few g that are not there, and on an acceleration plot that
peaks in the hundreds of g, a few g are hard to see. A DC offset
anywhere in the signal chain does the same. Both turn the spectrum's
low end into a flat shelf, because an oscillator far below the shock's
content responds to the step and to nothing else.

Integrated to velocity, the fault is plain. A shock on a shaker starts
and ends with the table at rest, so its velocity leaves zero, swings,
and comes back. A zero shift or an offset integrates to a ramp, and the
velocity never comes back. `visualdynamics.core.shock_check` makes that
check, and `plot_shock_check` draws it beside the spectra it protects.

## Why the integration is raw

`project.integrate` and `TimeHistory.integrate` remove the record's
mean and apply a 2 Hz drift high-pass by default, which is the right
default for turning a good record into a displacement estimate. It is
the wrong one for this check, because the drift it removes is the fault
the check looks for. Integrated with the default, a zero-shifted
record's velocity ends at zero as though nothing were wrong, and every
classical pulse loses its velocity change with the mean.

The check integrates by the same trapezoid rule with nothing removed
(`drift_corner=None`), the record's acceleration channels only, twice:
once to velocity and once to displacement. Each window is read from its
own start, so every event begins at rest whatever the record did before
it. The windows are the ones the SRS is computed over: the record's
shocks, or its averaging frames, or the whole record.

## What it reports

`check_shock(history)` returns one `ShockCheck` per acceleration channel
per window, with the raw velocity and displacement through the window
and these readings:

| Reading | What it is |
|---|---|
| `velocity_change` | the span from the lowest velocity to the highest; 2AT/π for a half-sine of peak A and duration T, and still the main pulse's change when compensation pulses sit either side of it |
| `end_velocity`, `end_velocity_share` | where the velocity ends, and that as a share of its largest excursion |
| `end_displacement`, `end_displacement_share` | the same for the displacement |
| `flagged` | the velocity ends further from rest than `tolerance` of its peak |
| `drifting` | the velocity is still changing when the window closes |

The tolerance is a tenth of the peak by default (`RETURN_SHARE`), and an
argument: a procedure that names its own figure passes it. A tenth is a
judgment; no standard gives the figure. A good record's velocity does
not end at exactly zero, because the sensor's noise is integrated too
and walks: by about the noise's standard deviation times the square
root of the number of samples times the sample interval. A zero shift
worth catching ramps the velocity well past its pulse. In the example
below, the ringing shock with its noise floor ends at 0.04 % of its
peak velocity, and with its zero shift at 100 %.

A flagged check says which of two things the velocity did.
Still changing when the record stops is a zero shift or an offset, and
the spectrum is false at its low end. Held at a net velocity change is
a pulse with nothing to bring the velocity back, which a drop machine
makes and a shaker cannot. A half-sine on its own is the second case:
it ends at its full velocity change, 2AT/π, and holds there.

## The positive and negative spectra

`TimeHistory.compute_srs_polarities()` returns the positive, negative
and maximax spectra from one pass of the filter, at the Q, spacing and
band `compute_srs` takes (Q = 10 and a twelfth of an octave unless
told otherwise). The negative spectrum is reported as a magnitude so the
two are drawn on one log axis. They are worth reading together: the
maximax spectrum is the larger of the two at every frequency, so a
response that went nearly all one way looks no different in it from
one that went both ways.

## An example

A ringing shock, four decaying tones from 600 Hz to 6 kHz peaking near
400 g, sampled at 102.4 kHz for a quarter of a second, with a 4 g zero
shift from the moment it arrives:

```python
from visualdynamics.core.shock_check import check_shock, summarize
from visualdynamics.plot import plot_shock_check
from visualdynamics.units import IN_LBF_S

check, = check_shock(history)
check.flagged                  # True
check.end_velocity / 0.0254    # 355.2 in/s: no table did that
print(summarize([check]))

plot_shock_check(history, low=10.0, high=10000.0,
                 unit_system=IN_LBF_S.with_units(acceleration='g'),
                 path='shock-check.png')
```

On the acceleration the shift is hard to see. On the velocity it is a
ramp to 355 in/s, far outside the band the velocity has to end in,
and the verdict says so. The positive spectrum stands on a shelf near
7.4 g below about 100 Hz that the shock does not have, and the negative
spectrum, which the step does not reach, falls away there:

![A ringing shock with a 4 g zero shift: the record, its raw velocity
ramping away to 355 in/s with the verdict above it, and its positive
spectrum standing on a false shelf below 100 Hz where the negative
falls away](images/shock-check-shifted.png)

The same record without the shift passes, its velocity ending at
0.007 in/s. A 50 g, 11 ms half-sine compensated by a 2.5 g half-sine
either side, the shaker's way of bringing the table back to rest,
passes too: the velocity runs down to −67.6 in/s through the
pre-pulse, up through the main pulse to +67.6 in/s, a velocity change
of 135.2 in/s, and back to rest. Its spectra part above the pulse: the
positive settles at the pulse's 50 g and the negative at the
compensation's 2.5 g.

![A compensated half-sine: the velocity returning to rest inside its
band, and the negative spectrum falling to 2.5 g while the positive
settles at 50 g](images/shock-check-compensated.png)

## In the app

The Shocks view states the check beside the windows, under the SRS
settings and before **Compute SRS**: one line when every channel
returns to rest in every window, and otherwise how many do not and the
verdict of the worst, by channel and shock. It is the same
`check_shock` over the same windows, read again whenever a window is
detected, added, dragged or typed in. A record with no shocks marked
(and no averaging) is not checked yet, because Compute SRS detects
its windows before it computes and the check would otherwise speak for
windows it was not read over; the view says so. A window holding a
sample that is not a finite number is refused, and the view says
which channel.

## Where judgment lives

- What the record is from. The flag says the velocity did not come
  back to rest. For a shaker shock that is a fault; for a drop machine
  a held velocity change is the pulse doing what it should, and the
  words say which of the two the record did.
- The window's length. The check reads the velocity at the window's
  end, so a window that closes before the ringing or the compensation
  is over reads a velocity that had not finished. The shock windows on
  the trace are where to check.
- What to do with a flagged record. A zero shift is a sensor problem to
  be found, not a spectrum to be reported. Filtering the spectrum's
  low end away, or integrating with the drift high-pass on, hides the
  symptom and keeps the fault.
