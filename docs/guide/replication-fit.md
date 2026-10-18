# Checking a replication reference before its first pass

A sine profile states its peak, velocity and displacement at every
frequency, and a random profile states its rms and, through the
Gaussian assumption, its peaks. A record to be replicated states
nothing until it is worked. Its largest peak may come once, in one
second of a minute, and its largest excursion at another second
entirely. So whether a shaker can play it is checked on the record
itself, sample by sample, before the first pass:
`visualdynamics.core.replication_fit` makes the check, and
`plot_replication_fit` draws it.

## What it checks

The force is the moving mass times the record's largest absolute
acceleration. The moving mass is everything the shaker accelerates:
the armature, the head expander or slip table, the fixture and the
article. The peak is read from the record, not estimated from its rms,
because a record with impacts has a crest factor well past the three
sigma a Gaussian test would suggest, and the force at that one peak is
what goes against the rating. A cutoff changes the largest peak a
little, so each candidate judges the force at its own.

The travel and the velocity come from integrating the record
twice, and they live at the bottom of its band: a component of
acceleration *a* at frequency *f* moves the table *a*/(2π*f*)², so the
lowest content decides the stroke almost alone. That is also where a
record is least trustworthy. A logger's offset of a thousandth of a g,
held for a minute, integrates to hundreds of inches. So the low end is
edited before the integration, and the check prices the edit at a few
candidate cutoffs.

Each candidate is worked the same way. The record, its offset taken
out, has its ends tapered by a half-cosine over two periods of the
cutoff. It is then high-passed (a second-order Butterworth run forward
and backward, so nothing moves in time), integrated to velocity, the
velocity high-passed again at the same cutoff, and integrated to
displacement. The taper is there because a reference cut from a longer
record does not start at rest: a tone at full amplitude from its first
sample makes the high-pass's start-up transient into a displacement
offset several times the tone's own motion at a 1 Hz cutoff. Tapered,
the same tone moves the table *a*/(2π*f*)² to a fifth of a percent. The velocity is filtered
too because an integration from rest cannot know the record's velocity
at its first sample, and that unknown constant walks the displacement
off as a ramp of its own. The integration itself is `integrate` with
`drift_corner=None`, the trapezoid rule with nothing removed, the raw
path the shock check also takes: the cutoff is the edit being priced,
so the integrator adds no high-pass of its own. At 2 Hz the candidate's
filter is the one `integrate` applies by default. Away from the
record's first and last seconds, where the candidate tapers and the
default does not, the two agree to a thousandth of an inch.

The record is also integrated as logged, with no cutoff at all, so
a logger's offset shows as the drift it is, and the offset alone is
integrated beside it.

The body moves opposite the armature by the ratio of the moving
mass to the body's mass, so the travel that reaches the stops, and the
velocity the coil moves at against the body, are the table's times one
plus that ratio. The body moves most at low frequency, which is where a
replication record's displacement lives. Give the body's mass and both
count it; leave it out, as for a body bolted to a seismic mass, and
they are the table's alone. The travel is
checked each way from the middle of the stroke against half the
stroke: a record that dips into a hole and climbs out is not symmetric
about zero, and a span inside the stroke can still run one way past a
stop. These rules live in `visualdynamics.core.shaker` and are the same
ones the [stroke readout](stroke.md) judges a shock pulse by.

## What it reports

`replication_fit(history, rating, cutoffs=(2, 3, 5, 8))` returns a
`ReplicationFit` for one acceleration channel whose unit is declared.
`rating` is a `core.shaker.ShakerRating` in SI: `force` (N), `stroke`
(m peak to peak), `velocity` (m/s) and `moving_mass` (kg), all needed,
and optionally `body_mass` (kg). A channel with a sample that is not a
number is refused, and so is one too short to filter.

| Reading | What it is |
|---|---|
| `force`, `force_share`, `fits_force` | the moving mass times the largest peak, and that against the rating |
| `peak_acceleration`, `peak_time`, `crest_factor` | the largest peak, when, and over the rms |
| `offset`, `offset_velocity`, `offset_displacement` | the record's mean, and what it alone integrates to by the end |
| `as_logged` | the record integrated with no cutoff |
| `cases` | one `FitCase` per cutoff, lowest first; `case(cutoff)` finds one |
| `lowest_fitting` | the lowest cutoff that fits, the one that takes least out of the record |

Each `FitCase` carries the table's `velocity` and `displacement`
against `time`, and:

| Reading | What it is |
|---|---|
| `travel`, `travel_up`, `travel_down`, `peak_to_peak` | the displacement with the body's share counted, its largest excursion each way from the middle and its span |
| `excursion_time`, `first_past_stroke()` | when the largest excursion happens, and when the travel first leaves the room |
| `peak_velocity`, `relative_velocity` | the table's largest velocity, and the armature's against the body, which the rating is judged by |
| `peak_acceleration`, `force`, `force_share` | the largest peak at this cutoff, and the force it asks for against the rating |
| `mean_square_removed`, `peak_change` | the price of the cutoff: the share of the record's mean square it removed and the change in its largest peak, both against the same tapered record with its offset out |
| `fits_stroke`, `fits_velocity`, `fits_force`, `fits` | the verdict, and `describe()` says it in words as shares of the ratings |

## An example

Sixty seconds of acceleration at a bracket while a vehicle crosses
rough ground: ride motion from 0.7 to 3.5 Hz, below anything a shaker
can carry, on a broadband floor to 400 Hz, a braking swell about 46 s,
and a dozen impacts, the largest a 14 g pothole at 32 s. It is 1.98
grms with a crest factor of 7.05, and the logger added 0.001 g.

```python
from visualdynamics.core.replication_fit import replication_fit
from visualdynamics.core.shaker import ShakerRating
from visualdynamics.plot import plot_replication_fit

rating = ShakerRating(force=17793.0, stroke=0.0508, velocity=1.778,
                      moving_mass=68.04, body_mass=1361.0)
fit = replication_fit(record, rating)
print(fit.describe())
plot_replication_fit(record, rating, path='fit.png', show=False)
```

The force at the pothole is 2,098 lbf on 150 lb, where three sigma of
the rms would have said 890. As logged, the offset alone integrates to
23.2 in/s and 695 in by the end of the minute, and the travel leaves a
2 in stroke 0.65 s in. Taking the mean out is not enough: the ride
motion still spans 48.6 in. High-passed, the record spans:

| Cutoff | Span (table) | Mean square removed | Largest peak |
|---|---|---|---|
| 2 Hz | 1.40 in | 1.25 % | −0.44 % |
| 3 Hz | 0.71 in | 2.02 % | −1.34 % |
| 5 Hz | 0.43 in | 3.33 % | −2.69 % |
| 8 Hz | 0.20 in | 5.10 % | −4.27 % |

The span falls by a factor of seven while the price climbs from about
one percent to five. With a body of twenty times the moving mass the
travel and the velocity are five percent more than the table's, and
every candidate fits the stroke. A velocity rating of 22 in/s would
rule out 2 Hz, where the table peaks at 23.9 in/s and the coil moves
25.0 in/s against the body, and the lowest cutoff that fits would move
to 3 Hz, at 20.9 in/s.

![The road record's travel at 2 Hz inside a 2 in stroke, the verdict
and the drift as logged above it; under it the travel each way, the
peak velocity and the price of each candidate
cutoff](images/replication-fit.png)

## In the app

Select a record that holds acceleration in a declared unit and choose Replication
Fit among the readings on the plot bar. The panel beside the plot
holds the ratings in the display units, the candidate cutoffs, the
record's largest peak and the force it asks for, and a table with a
column per candidate after the record as logged: the travel down and
up, the peak velocity against the body, the mean square removed, the
change in the largest peak, and whether the stroke, the velocity and
the force all fit. Click a column to draw that cutoff,
or the record as logged. The ratings stay as typed from record to
record. The reading makes nothing: the edited record itself is made
with the Filter reading, a high-pass at the chosen cutoff of order 2.

## Where judgment lives

- The cutoff against the article. Content below the cutoff loads
  anything in the article that resonates there, and the high-pass
  takes that load out of the test. A common first check is to keep
  the cutoff at least an octave below the article's first mode, and
  where the article sits on its own isolators with a mount frequency
  in the band the filter removes, the cutoff is a question for the
  article's owner, not a setting.
- The integration's own limits. Integration of a measured record
  can be wrong in ways nothing in the record shows. Where the margin
  is small, or the record is doubtful, the stroke is proved on a dummy
  mass with the displacement measured before the article goes on.
- The controller's estimate. A replication controller estimates
  the drive's peak velocity and displacement from the drive it built.
  This check is made on the reference, before the drive exists, and
  the two are read side by side: where they disagree, or either sits
  close to the stroke, the measurement decides.
