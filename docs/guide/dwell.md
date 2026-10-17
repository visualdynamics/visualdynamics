# Reading a dwell against time

A *dwell* holds one tone at one frequency for a time: a tone the
environment really has (a rotor rate, an engine order), or a resonance
a search found, held for fatigue. A sweep's level is read against
frequency because the frequency is what changes. A dwell's frequency
does not change, so against frequency its level is one point, and what
it has to say is in time.

At a resonance the response does not jump to its steady height when the
tone arrives. It builds, cycle by cycle, and when the tone stops it
rings down at the same pace. For a mode of quality factor Q driven at
its own frequency from rest, the share of the steady height reached
after n cycles is very nearly

\[
1 - e^{-\pi n / Q}
\]

so after Q cycles it stands at 1 − e^−π, 95.7 %, and after 2Q cycles at
99.8 %, whatever the frequency and whatever the Q; n cycles after the
stop, e^−πn/Q of it remains. About Q cycles to arrive and about Q to die
away: a mode at 20 Hz with a Q of 30 takes a second and a half to reach
its full response.

## A dwell in the specification

A controller cannot hold a segment with no span, and neither does a
sine specification here, so a dwell is written the way a controller's
file writes it: a linear segment across a hair of a hertz, crossed in
the time the tone holds. `SineTone.dwell` builds one, sweeping a
millionth of its frequency (`DWELL_SWEEP`), and `SineTone.is_dwell`
reads any segment that spans no more than one part in ten thousand of
its frequency (`DWELL_SPAN`) as a dwell, so a controller's file reads
the same and nothing new is written to any file:

```python
from visualdynamics.core.sine import (DWELL_SWEEP, LINEAR, LOG,
                                      SineSweepSpecification, SineTone)

hold = SineTone.dwell('Dwell', 20.0, [9.80665], 4.5)    # 1 g at 20 Hz, 4.5 s
spec = SineSweepSpecification([hold], ['1Z+'], ordinate_unit='m/s**2')
hold.dwell_segments()                                    # [0]

# or as one segment of a tone that sweeps too: 10 to 20 Hz at 5 Hz/s,
# 20 Hz held for 30 s, then on up at 1 octave a minute
hair = 20.0 * DWELL_SWEEP
mixed = SineTone('Search and hold', 0.0, [10.0, 20.0, 20.0 + hair, 40.0],
                 [[9.8]] * 4, [LINEAR, LINEAR, LOG], [5.0, hair / 30.0, 1.0])
mixed.dwell_segments()                                   # [1]
```

## The reading

`read_dwell` places the tone on the recording (by matched filter on the
control channels, as the tracking readings do, or at `onset=`) and reads
one channel with a sine-tracking detector at the end of every whole
cycle of the dwell, then on at the same frequency after the stop, to the
end of the record. By default it reads a response (the first channel
the specification does not control, since the controller holds the
control at its level and the control does not build) with the peak over
each cycle, unfiltered; `setting=` takes any band and detector
[the tracking readings](sine-tracking.md) take. It reports:

- the steady level, the median of the readings over the last quarter
  of the dwell;
- the build-up, the cycles from the dwell's start until the level first
  reaches a share of the steady level, 95 % by default (`share=`);
- the ring-down, the cycles after the stop until it first falls to the
  rest of that share, 5 %;
- whether the level had settled (`settled`), and notes on anything the
  numbers cannot be trusted for.

Take a 1 g tone at 20 Hz on a one-mass resonator of
Q 30, held ninety cycles and then stopped:

```python
from visualdynamics.core.dwell import read_dwell
from visualdynamics.plot import plot_dwell
from visualdynamics.units import DEFAULT_SYSTEM

reading = read_dwell(history, spec, onset=0.0)          # 101Z+, the response
reading.level[29] / 9.80665, reading.seconds[29]        # 28.72 g at 1.50 s
reading.share_at(30), reading.share_at(60)              # 0.957, 0.998
reading.remaining_at(30)                                # 0.043, e to the -pi
print(reading.describe())
# steady 294.3 m/s², 95 % in 28.6 cycles (1.43 s), down to 5 % 28.6
# cycles after the stop
print(reading.describe(DEFAULT_SYSTEM))
# steady 30.01 g, 95 % in 28.6 cycles (1.43 s), ...

plot_dwell(history, spec, onset=0.0, path='dwell.png')
```

![The 20 Hz, Q 30 dwell read against time: the level builds from the
start, passes 95 % of its steady 30.01 g at 28.6 cycles (a filled
circle on the dashed line), holds, and after the stop at 4.5 s falls to
5 % (a filled triangle on the dotted line) 28.6 cycles
later](images/dwell.png)

`plot_dwell` takes the same arguments as `read_dwell`. The steady level
is the wide gray line, the share is dashed and its complement dotted,
the two crossings are a filled circle and a filled triangle, and the
legend names each with its count, so the figure reads in gray as well
as in color.

In the window, select the record and turn on the sine view with the
specification in the project. When the specification holds a dwell, a
**Dwell** readout sits under the extraction's settings: the channel and
the share, then the frequency, the steady level, the build-up and the
ring-down in cycles and seconds, whether it settled, and the same
picture, small. It is a reading, so there is no button; each edit reads
again and is journaled as the `plot_dwell` call that draws it.

## Where judgment lives

- A reading is reported after the instant it stands for. The peak over
  a cycle is reported at the cycle's end, but in a decay the largest
  value is at the cycle's start, a whole cycle earlier; RMS and mean
  stand for the middle of what they average. Counted on the reported
  clock, the ring-down would read a cycle longer than the build-up. So
  every number the reading gives (the counts, `share_at`,
  `remaining_at` and the plot) places each reading where it stands
  (`reading.stands`): the build-up and the ring-down of one Q read
  alike, and each number is the response's, whatever the detector.
- The steady level is the record's, not the mode's. A dwell too short
  to settle, or a resonance that moved during it as the article warmed
  or cracked, reaches a level that is not the mode's steady response.
  The reading says so (`settled`, and a note) rather than borrowing a
  number from the formula above.
- A dwell between sweeps does not start from rest or stop. A sweep
  that runs into the dwell leaves a response the build-up starts from,
  and a sweep that leaves it keeps driving, so nothing rings down from a
  stop and no ring-down count is read.
- A record that starts after the dwell has built holds no build-up. Its
  first reading is already over the share, so no count is given and a
  note says how many cycles into the dwell the record starts. A channel
  that reads no level at all over the dwell's last quarter, as a dead
  one does, is refused; read anyway, its zero would pass for a build-up
  done in one cycle.
- The matched filter places a held tone's start to within a quarter of
  a cycle. A pure tone looks the same one cycle along, so its edges
  place it, and the specification's phase sets where in the cycle it
  lands; that is enough to move each count by about a quarter cycle. Give
  `onset=` when the start is known to the sample.
