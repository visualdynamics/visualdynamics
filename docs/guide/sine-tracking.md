# How a tracking filter reads a sweep

A swept-sine controller does not measure the response the way
**Extract Sine Levels** does. It reads the response at the drive
frequency through a *tracking filter*, a band-pass centered on the
drive frequency that follows it along the sweep, and it turns what
comes out into a level with a *detector*. Both are settings, and the
same response record gives a different level curve for each. When a
controller's level and an extracted one disagree, the reason is
usually in those two settings, and
`visualdynamics.core.sine_tracking` reads a record each way so the
difference can be seen.

## The band and the detector

The band has a **fixed** width in Hz, or a width **proportional** to
the drive frequency, as a fraction of it. A common setting on the
test floor is proportional, around half the drive frequency; the
ranges offered run from a few percent to the whole drive
frequency proportional, or from about one hertz to several hundred
fixed. With no band, the detector reads the record as it stands.

The detector is one of four:

| Detector | `detector=` | Reads | Reported as |
|---|---|---|---|
| Filter output | `'filtered'` | the band's own output amplitude, at the instant | itself |
| Peak | `'peak'` | the largest absolute sample over the last cycle | itself |
| RMS | `'rms'` | the root mean square over the last cycle | times √2 |
| Mean | `'mean'` | the mean absolute value over the last cycle | times π/2 |

The last two are reported as the peak of the sine they would be, so
for a clean tone all four read its amplitude. For anything else they
differ, and that is the point.

## What a harmonic does

A clipped amplifier drive puts a third harmonic on the response. Read
without a filter, a peak detector counts the harmonic's peaks: a
harmonic of 30 % in phase with its tone reads 30 % high. An RMS
detector counts its power instead: the same harmonic reads
√(1 + 0.3²), about 4.4 % high. Through the band, neither is seen: the
harmonic sits at three times the drive frequency, outside any band
narrower than four times the drive. (A mean detector on the same
record reads about 10 % *low*: the harmonic in phase sharpens the
peaks and hollows the shoulders, and the shoulders hold more of the
area.) Broadband noise behaves the same way: an
unfiltered RMS reading counts all of its power, and a band counts its
own share.

![A 10 m/s² log sweep with a 30 % third harmonic, read four ways: the
unfiltered peak at 13, the unfiltered RMS at about 10.4, and both
filtered readings on the specification at
10](images/sine-tracking-harmonic.png)

## What the band costs

A band takes time to follow a change in level, roughly one over its
width. A level that doubles part way through a sweep is followed in a
millisecond by an unfiltered peak detector, in 9 ms through a 50 %
band at 250 Hz, in 37 ms through a 10 % band, and in 180 ms through a
5 Hz fixed band (the time to reach half way, in each case). On a
sweep at 100 Hz/s, 180 ms is 18 Hz of the frequency axis, read low.
The fourth-order filter also overshoots a step by about a tenth of
it before it settles.

![A linear sweep whose level doubles at 250 Hz, read through bands of
different widths: the wide band follows the step at once, the 5 Hz
fixed band lags it by tens of hertz and
overshoots](images/sine-tracking-step.png)

So the trade is settling time against rejection: a narrow band holds
out harmonics and noise and lags a change in level, such as a
resonance passed through quickly; a wide band follows the change and
counts more of what is not the tone.

## In a script

```python
from visualdynamics.core.sine_tracking import SineTracking, track_sine
from visualdynamics.plot import plot_sine_tracking

settings = [SineTracking(detector='peak'),          # no band
            SineTracking(proportional=0.5),          # 50 % of the drive
            SineTracking(proportional=0.1),          # 10 %
            SineTracking(fixed=10.0, detector='rms')]

levels = track_sine(history, specification, settings)   # one SineLevel each
plot_sine_tracking(history, specification, settings, path='tracking.png')
```

`track_sine` reads one tone (the specification's first, or `tone=`) on
the control channels, or on any channels named with `channels=`. The
tone's sweep is found in the record the way the extraction finds it,
or placed at `onset=`. Every setting is read at the same lines along
the sweep, so the levels overlay line for line. Each is a `SineLevel`
in the record's own units, stamped with the second each line was
read, its comment naming the reading (`SineTracking.describe()`).
`SineTracking.settling(frequency)` is the one-over-the-bandwidth rule
at a given drive frequency.

The plot draws one channel, each reading in its own color and with
its own marker shape, and the specification's target behind them in
gray.

## Where judgment lives

- **This is a reading for understanding a controller, not a better
  level.** The extraction solves every tone jointly, debiases the
  amplitude and chooses its smoothing from the data; a report should
  use it. A tracking filter reads one tone at a time and carries
  whatever its band lets through.
- **The filter runs forward only.** A controller's filter cannot see
  ahead, so neither does this one: the lag is the measurement. It
  starts settled on the tone's first cycle, as a controller's has
  settled during its ramp-up.
- **A wide fixed band at the bottom of a sweep is not about the drive
  frequency.** Where half the band reaches past the drive frequency
  toward zero, the band includes everything below the drive as well,
  and the reading shows a ripple at twice the drive.
- **A peak is the peak of the samples.** It reads low by at most
  cos(π × drive / sample rate): 0.3 % at a fortieth of the sample
  rate, more on a harmonic near the top of the band.
- **No view in the application yet.** The readings and the plot are
  scripting calls; the window does not offer them.
