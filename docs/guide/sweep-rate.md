# Checking a sweep rate against the modes it passes

A mode needs about Q cycles at its own frequency to build to its full
response. A sweep never holds it there. It gives the mode a short time
inside its half-power band, the band \(f_n/Q\) wide around \(f_n\)
where the response is within 3 dB of its peak, and moves on. When that
time holds fewer cycles than the mode needs, the swept response reads
the peak low, places it late (above the mode on a sweep up, below it on
a sweep down) and spreads it, so a Q read from its half-power width
comes out low too. All three are properties of the rate, not of the
article, and all three can be worked out before the test.

## The arithmetic

A logarithmic sweep at \(R\) octaves per minute crosses the band, which
is \(1/(Q \ln 2)\) octaves wide, in

\[
t_{hp} = \frac{60}{\ln 2}\,\frac{1}{Q R} \approx \frac{86.6}{Q R}
\ \text{seconds},
\]

whatever the mode's frequency. The mode completes \(f_n\) times as many
cycles in that time. Setting those cycles equal to the Q it needs gives
the rate at which a sweep only just lets it build:

\[
R = \frac{60}{\ln 2}\,\frac{f_n}{Q^2} \approx \frac{87\, f_n}{Q^2}
\ \text{oct/min}.
\]

A linear sweep at \(r\) Hz/s crosses the band, \(f_n/Q\) Hz wide, in
\(f_n/(Q r)\) seconds, and the same reasoning gives
\(r = f_n^2/Q^2\) Hz/s. A log sweep spends the same time in every band
of a given Q, so it gives a high mode more cycles than a low one; a
linear sweep spends the same time in every hertz, so it rushes the low
modes and lingers on the high ones.

This rule is Visual Dynamics' own criterion, worked out above from the
cycles a mode needs to build. It is not taken from a published
standard; the references in `core.sweep_rate` are for the effect it
guards against, not for the rule.

The rule is a ceiling to stay well under, not a target. At the rule's
rate a mode still reads only about 94 % of its steady height, and the Q
read from its width comes out about 20 % low. Simulated
for one mode at 22 Hz with a Q of 40, whose rule rate is 1.19 oct/min:

| Sweep up | Time in band | Cycles | Rate over the rule's | Peak read | At | Q read |
|---|---|---|---|---|---|---|
| 0.25 oct/min | 8.66 s | 190 | 0.21 | 99 % | 22.07 Hz | 39.2 |
| 1 oct/min | 2.16 s | 47.6 | 0.84 | 95 % | 22.22 Hz | 33.1 |
| 4 oct/min | 0.54 s | 11.9 | 3.4 | 81 % | 22.61 Hz | 21.6 |
| 8 oct/min | 0.27 s | 5.95 | 6.7 | 71 % | 22.97 Hz | 16.6 |

The Q suffers first. At 1 oct/min the sweep is under the rule's rate
and the peak is within 5 % of its height, but its width has already
spread the Q 17 % low. A sweep meant to measure a Q, rather than to
reach a level, has to run several times slower than the rule.

Whether a fast sweep is a fault depends on what the sweep is for. A
search for resonances is a measurement, and slow is right for it. A
sweep that stands for a short transient is written fast on purpose,
because the transient never brings a sharp mode to its full response
either; slowing it would build every low mode further than the
environment does. The check says what the rate does to each mode; what
the rate should be is the test plan's decision.

## From a script

`check_sweep_rate` takes a `SineSweepSpecification` (or one `SineTone`)
and the modes, each a natural frequency in Hz and a Q, or a `ShapeSet`,
whose damping gives the Q:

```python
from visualdynamics.core.sweep_rate import check_sweep_rate

for check in check_sweep_rate(spec, [(9.0, 10.0), (22.0, 40.0),
                                     (60.0, 15.0)]):
    print(check.describe())
# 9 Hz, Q 10, up at 4 oct/min: 2.17 s and 19.5 cycles in the band
#   (10 to build), 0.51 times the rule's 7.79 oct/min; reads 96 % of
#   the peak and a Q of 8.77; builds
# 22 Hz, Q 40, up at 4 oct/min: 0.541 s and 11.9 cycles in the band
#   (40 to build), 3.4 times the rule's 1.19 oct/min; reads 81 % of
#   the peak and a Q of 21.6; too fast
# 60 Hz, Q 15, up at 4 oct/min: 1.44 s and 86.6 cycles in the band
#   (15 to build), 0.17 times the rule's 23.1 oct/min; reads 100 % of
#   the peak and a Q of 14.8; builds

visualdynamics.plot.plot_sweep_rate(spec, [(9.0, 10.0), (22.0, 40.0),
                                           (60.0, 15.0)],
                                    path='sweep_rate.png')
```

Each `SweepRateCheck` is one pass of one tone through one mode's band:
`start`, `seconds` and `cycles` in the band, the `rate` written on the
segment holding the mode, the rule's rate as `limit` in the same unit,
`ratio` between them, the simulated `peak_share`, `q_read` and
`reading.peak_frequency`, and `too_fast`, which is the ratio over
`margin` (1 by default). A tone swept up and back passes each mode
twice, and each pass is a row; a band that straddles a breakpoint is one
pass, judged at the faster of the two rates (the larger multiple of the
rule's); a sweep that starts,
ends or turns inside a band is marked `crossed=False`. A mode the sweep
never reaches has no row.

What a mode reads at a rate (`simulate_sweep(frequency, q, rate)`) is
found by simulating it: the one-mass resonator, run through the shock
spectrum's ramp-invariant filter, is driven from rest through its band
at the segment's rate and law, and its response is read once per cycle
of the drive as that cycle's rms reported as the peak of a sine, the
rms detector a controller uses. The swept curve's peak and half-power
width are then read as they would be off a plot. The result depends
only on the rate over the mode's frequency (over its square for a
linear sweep), so the simulation is made once at 1 Hz and scaled.

![A sweep from 5 to 100 Hz at 4 oct/min through three modes: the 22 Hz,
Q 40 band is crossed in 0.54 s, a dashed crossing with a cross, too
fast; the 9 Hz and 60 Hz bands are solid crossings with circles](images/sweep-rate.png)

The plot is the sweep as frequency against time, on a log frequency
axis, with each mode's half-power band as a strip across it. Where the
sweep crosses a strip its path is drawn heavy: solid with a circle where
the mode builds, dashed with a cross where the sweep is too fast for it,
and the label over the strip gives the time, the cycles against the Q
needed, the rate over the rule's and the verdict in words.

## A Q read from a swept level

Most Qs in a sine test are read off a swept level, so most of them carry
the bias above. `read_swept_q(level, spec, near)` reads the peak nearest
`near` on a tracked or extracted level, its top placed between readings
and its half-power crossings found along the pass of the sweep that
holds it, and judges the Q against the rate the level was swept at. The
level's clock puts its readings back in sweep order, so a tone swept up
and back is read one pass at a time, and readings under the noise floor
are left out. `swept_q(frequency, q_read,
rate)` does the same for a Q read some other way.

```python
from visualdynamics.core.sweep_rate import read_swept_q

judged = read_swept_q(level, spec, 22.0)
print(judged.describe())
# Q 22.1 read at 22.61 Hz from a sweep at 4 oct/min: reads low; a mode
#   reads this low at a Q of about 41.5, for which the rate is 3.5
#   times the rule's
```

The judgment inverts the simulation: it finds the Q a mode would need
to read this low at this rate. The swept Q rises with the true one to a
ceiling the rate sets, so a reading near the ceiling implies a wide
range of true Qs, and a reading above it implies none (`q_true` is None
and the reading is flagged as too fast to read any Q). A reading is
flagged when its implied Q is more than 5 % above the Q read, or when
the rate is over the rule's for the implied Q.

## In the app

Select a sine specification and press **Sweep Rate** on the bar. The
panel beside the plot takes the modes as pairs of frequency and Q
(`22 40, 180 20`) and the multiple of the rule's rate a pass may reach
before it is flagged; text that does not read as pairs is refused when
it is entered, and the last good modes are put back. The plot is the
tone chosen on the bar's drop-down, drawn as above, and the table under
it lists every pass: the tone, the mode, its Q, the direction, the rate
and the rule's, the multiple, the seconds and cycles in the band, the
peak and Q a swept reading gives, and the verdict.

Select a tracked or extracted level of the tone with the specification,
and each mode's Q is also read off the level and judged against its
sweep: a row per level and mode, named for the level's DOF, with the Q
read and, where it reads low, the Q it implies. The panel repeats each
judgment in words, and the status line counts the passes too fast and
the readings flagged.

The reading makes nothing, so there is no button to press. The console
records it as the `plot_sweep_rate` call that draws it, with the modes
and the margin set, and the Q read off each level as the `read_swept_q`
calls that read it.
