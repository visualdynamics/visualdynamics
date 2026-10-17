# Will a pulse fit on the shaker

A shaker can deliver a shock pulse only if it can deliver three things
at once: the force for the peak, the velocity the pulse leaves the
table with, and the travel the whole event needs, compensation
included. The three are spent by different parts of the pulse. The
peak acceleration costs force, the moving mass times the peak. The
duration costs stroke, because displacement grows with the peak times
the duration squared. The velocity sits between them. So the check is
done before the first pulse, against the shaker's shock ratings, and
`visualdynamics.core.stroke` makes it: from a record, or from a
nominal pulse with the compensation chosen.

## The nominal pulse

`compensated_half_sine(peak, duration, sample_rate, share=0.2)` builds
the pulse a stroke is usually first estimated on. The half-sine sits
between a half-sine pre-pulse and post-pulse of the opposite sign, each
at `share` times the peak (a fifth by default, the share test documents
commonly allow) and each lasting `duration / (2 * share)`, so each
carries back half the main pulse's velocity change. The table starts
and ends at rest: the pre-pulse takes it to minus half the velocity
change, the main pulse to plus half, the post-pulse back to zero. So
the peak velocity is half the velocity change, 2AT/π, and at a fifth
the whole event lasts six durations; at a tenth it lasts eleven, and
the table travels 1.80 times as far before it is brought back.

## Why the integration is raw

`read_stroke` integrates the acceleration twice by the trapezoid rule
with nothing removed, from rest at the record's first sample: the same
`filters.integrate` the Integrate act uses, called with
`drift_corner=None`. The default 2 Hz drift high-pass is no longer
transparent here, because the displacement of a compensated pulse is
content at a few hertz. With it, the 50 g, 11 ms pulse below reads
1.212 in peak to peak instead of 1.166 in, 3.9 % high, and its
displacement no longer starts and ends at rest. A stroke estimate is
integrated raw.

## What it reports

`read_stroke(history, channel, stroke=, velocity=, force=,
moving_mass=, body_mass=)` returns a `StrokeReading`. Every rating is
optional and in SI (`visualdynamics.units.to_si` converts), and a
rating not given is reported without a verdict. The channel must be
acceleration in a declared unit, and a channel with a sample that is
not a number is refused. The ratings are judged by the rules in
`visualdynamics.core.shaker`, the same ones the
[replication fit](replication-fit.md) uses.

| Reading | What it is |
|---|---|
| `peak_acceleration`, `peak_velocity`, `velocity_change` | the largest acceleration and table velocity either way, and the span of the velocity |
| `relative_velocity` | the peak velocity with the shaker body counted, the coil's velocity against the body |
| `peak_to_peak`, `lowest`, `highest` | the table's displacement from rest |
| `travel`, `travel_down`, `travel_up` | the travel with the shaker body counted, peak to peak and each way from center: the table's displacement times one plus the moving mass over the body's mass |
| `rule` | the half-sine rule for the record's own peak and velocity change: A T² / 2, about 193 A T² inches with A in g |
| `force` | the moving mass times the peak acceleration |
| `shares()`, `fits` | each demand as a share of its rating, and whether every rating given holds |
| `fits_stroke` | whether the travel stays inside half the stroke each way from center, where the stops are |

The body is counted because the ratings are the armature's motion
relative to the body, and a body on isolation mounts moves the other
way by the ratio of the moving mass to its own. Given both masses, the
travel and the velocity count it. The compensation is low in
frequency, where the body moves most.

The stroke share is the larger travel each way from center over half
the stroke, because the armature rests at center, held there by its
suspension, so the stops are half the stroke away on each side. The
symmetric arrangement travels one way only, to its full peak to peak
and back, so it can be inside the stroke peak to peak and still reach a
stop. The peak to peak is reported as a number and not judged. When it
fits the stroke and the travel one way does not, the reading says that
centering the travel would fit; centering is the controller's
compensation, an asymmetric pre- and post-pulse.

The half-sine rule is reported beside the integration as the first
check it is. On the nominal pulse it reads 0.16 % high on every pulse
(a 100 g, 6 ms pulse needs 0.694 in, the rule says 0.695 in; a 20 g,
30 ms one needs 3.47 in). On a record of any other shape the difference
between the two is what the shape and the compensation are worth. A
controller that shapes its compensation for least displacement does
better than the symmetric arrangement, so a pulse the reading passes
with room fits, and one it puts near a rating sends the operator to the
controller's own prediction and to a measurement.

## An example

The 50 g, 11 ms half-sine compensated at a fifth, on a shaker with a
2.5 in shock stroke, a 70 in/s shock velocity and a 6,000 lbf shock
force, with 100 lb moving on a body of 400 lb:

```python
from visualdynamics.core.stroke import compensated_half_sine, read_stroke
from visualdynamics.plot import plot_stroke
from visualdynamics.units import to_si

pulse = compensated_half_sine(50.0, 0.011, 102400.0, unit='g')
ratings = {'stroke': float(to_si(2.5, 'in')),
           'velocity': float(to_si(70.0, 'in/s')),
           'force': float(to_si(6000.0, 'lbf')),
           'moving_mass': float(to_si(100.0, 'lb')),
           'body_mass': float(to_si(400.0, 'lb'))}
reading = read_stroke(pulse, **ratings)
print('\n'.join(reading.describe()))
plot_stroke(pulse, **ratings, path='stroke.png')
```

The table moves 1.17 in peak to peak and the body adds a quarter, so
the travel against the body is 1.46 in. It runs 1.46 in down from
center and none up, against 1.25 in to the stop each way: 117 % of the
room, so the pulse as compensated here reaches a stop. The 1.46 in peak
to peak is inside the 2.5 in stroke, so compensation that centers the
travel would put about 0.73 in each way, inside the stops. The table's
velocity peaks at 67.6 in/s, and with the body's quarter the coil moves
84.5 in/s against the body, 121 % of the 70 in/s rating; the force is
5,000 lbf, 83 % of its own. So the pulse does not fit as compensated
here, and on its velocity it would not fit centered either.

![The compensated 50 g, 11 ms half-sine in three rows: the pulse, its
velocity with the velocity against the body dashed past the 70 in/s
rating lines, and the table's displacement with the travel relative to
the body dashed beneath it, past the lower stop half the stroke below
center](images/stroke-readout.png)

To check the force at the top of a tolerance band, build the nominal
pulse at the band's upper edge (1.2 times the peak for a band of plus
20 %); the stroke grows with it, as it would on the shaker.

## In the app

Select a time history that holds acceleration in a declared unit and
choose **Stroke** on the bar. The plot shows
the first selected acceleration channel in three rows, each rating on
the row it limits: the shock velocity either side of zero, and the
stops half the stroke either side of center. The panel beside it takes
the ratings in the window's units (zero is no rating) and reads out
the peaks, the peak to peak beside the half-sine rule, the travel with
the body, peak to peak and each way from center, the force, and each
demand against its rating in words. Every edit redraws at once. The ratings are kept when
another record is selected, since a shaker's ratings do not change with
the record, and the reading is journaled as its `plot_stroke` call.

## Where judgment lives

- The record's length. The integration runs over the whole record from
  its first sample, so a record should open at rest before the event
  and close after the compensation. A record whose velocity does not
  come back to rest carries a zero shift or an offset, and its stroke
  is not to be believed until that is found.
- The ratings. Many shaker sheets give a shock velocity and a shock
  stroke above the continuous figures; where only continuous figures
  are given, those are the ratings.
- The body's mass, and the compensation the controller will actually
  build, which the symmetric arrangement only stands in for. Where the reading puts a pulse close
  to the stroke, measure the displacement on a calibration load before
  the article goes on.
