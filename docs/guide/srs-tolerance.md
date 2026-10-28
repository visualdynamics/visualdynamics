# Judging an SRS against its tolerance

A shock test is accepted on a band and a count. The requirement is a
shock response spectrum, the band is written in decibels below and
above it, and the test document says how many of the measured
spectrum's points must lie inside the band. This page describes that
judgement in Visual Dynamics, how it differs from the RMS deviation
the shock workflow's bars show, and how to run it from a script.

## Two readings of one spectrum

`srs_errors` in `visualdynamics.core.compliance` gives one number per
channel and event: the RMS of the point-by-point deviation in dB,
signed by the side its mean falls on. It is a good summary of how far
a spectrum strays, and the **SRS Error** bars of the
[shock workflow](workflows/shock-workflow.md) are drawn from it. It is
not the tolerance check. It does not count points, and a spectrum can
sit well inside 3 dB in RMS while several of its points are outside
the band: ten neighboring points 5 dB under, among 92 on the
requirement, read −1.6 dB in RMS and leave 89 percent of the points
inside.

`srs_tolerance` is the tolerance check. It reads every point of the
measured spectrum against the requirement at that natural frequency,
counts the points inside the band, and applies a rule to the count.
Both readings are kept, because they answer different questions: the
bars say which channels and events strayed and in which direction,
and the count says whether a spectrum met its tolerance.

## The rule

1. The points. The measured spectrum's own points are judged, at the
   natural frequencies its spacing defines, because a tolerance on an
   SRS is applied at those points. The requirement is read at each one
   log-log between its breakpoints, the way it is drawn. Points
   outside the band the requirement is written over are not judged.
2. The distance. Each point's distance from the requirement is
   twenty log ten of the ratio, since an SRS is an amplitude. A point
   on a band's edge is inside it.
3. The bands. A specification that carries limits is judged against
   them: its warning limits are the inner band and its abort limits
   the outer one, read at each point the way the comparison shades
   them. A band given in the call replaces the limits, and
   `limits=False` sets them aside. Without either, the inner band is
   −1.5 dB below and +3 dB above the requirement, and there is no
   outer band. When there is an outer band, every point must lie
   inside it whatever the rule says.
4. The count. Two rules are offered, as examples of the kinds of rule
   test documents ask for. The test plan states which one applies and
   with which numbers.
    - `rule='share'` (the default) passes when at least `percent` of
      the points (90 by default) lie inside the inner band.
    - `rule='above_lower'` passes when every point is at or above the
      inner band's lower edge. The band has no upper edge under this
      rule, so every point at or above the lower edge counts as
      inside.

   A rule that asks for a share of the points above the requirement is
   the share rule with a band from 0 dB up:
   `inner=(0.0, numpy.inf), percent=50.0`.
5. The worst point. The result names the point farthest outside the
   band the rule applies, with its distance from the requirement. That
   is not always the point farthest from the requirement: under
   −1.5/+3 dB, a point 2.5 dB over is inside and one 2 dB under is
   not.

The Q and the quantity must match. A spectrum computed at another Q
is a different curve, and a spectrum of velocity says nothing about a
requirement in acceleration, so such a pair is refused with the reason
instead of judged. Units follow the rule every comparison uses
([Units in a comparison](compliance.md#units-in-a-comparison)): a pair
with only one side's units declared is refused, and a pair with
neither declared is judged as raw numbers and says so. `srs_errors`
refuses by the same rules. A refused pair keeps its row, with `db` NaN
and `refused` giving the reason, and the SRS Error bars, the table
under them and the report write that reason in the row where the bar
would be.

## Each polarity

A spectrum is judged in the polarity it holds. To judge a pulse's
positive and negative spectra against one maximax requirement, compute
each (`compute_srs(kind='positive')`, `kind='negative'`) and judge each.
A pulse that is not symmetric often meets the requirement in one
direction and not the other, and the maximax spectrum, which takes the
larger of the two at every point, hides that.

## In the app

Select the shock specification and a measured SRS together, and choose
**Points Rule** on the bar, beside **Spectra** and **SRS Error**. To
judge several polarities at once, select the positive, negative or
maximax spectra together with the specification. **Points Rule** is
then the only reading on the bar, because the spectra and the bars
read one measurement at a time; the table under the plot gives each
polarity's RMS deviation on its own row. Spectra of one polarity are
one measurement each and are compared one at a time. The plot is the
picture below, at the channel and event chosen in the box on the bar.
The panel beside it holds the rule (a share inside, or every point at
or above the lower edge) and the share asked for. When the
specification carries limits, **Spec limits** is ticked and the
judgement is against them; untick it to set the bands by hand,
starting from the specification's own numbers. Otherwise the panel
holds the inner band and, when ticked, the outer band. Under them,
for each polarity, are the count and share of its points inside, the
verdict, and its worst point with its distance from the requirement
in dB.
Every edit redraws the judgement at once. Nothing is made, so there is
no Apply button; the console records the `plot_srs_tolerance` call
that draws what is on screen.

## From a script

```python
from visualdynamics.core.compliance import srs_tolerance, srs_tolerance_all
from visualdynamics.core.data import ShockSpecification
from visualdynamics.plot import plot_srs_tolerance

# the requirement as written: breakpoints, Q, spacing, type and bands
spec = ShockSpecification.from_breakpoints(
    [10.0, 100.0, 2000.0], [10.0, 100.0, 100.0], '101Z+',
    per_octave=12, q=10.0, kind='maximax', ordinate_unit='g',
    warning=(-1.5, 3.0), abort=(-3.0, 6.0))

# the spectra, at the same Q and spacing, in each polarity
positive = history.compute_srs(low=10.0, per_octave=12, q=10.0,
                               kind='positive')
negative = history.compute_srs(low=10.0, per_octave=12, q=10.0,
                               kind='negative')

# the bands are the specification's limits, -1.5/+3 dB inside -3/+6 dB
result = srs_tolerance(spec, positive)
print(result['inside_count'], 'of', result['points'], 'inside;',
      'passed' if result['passed'] else 'failed',
      f"worst {result['worst_db']:+.1f} dB at {result['worst_frequency']:.0f} Hz")

# every channel and event, paired as srs_errors pairs them
for dof, event, row in srs_tolerance_all(spec, negative):
    print(dof, event, row['inside_percent'], row['passed'])

plot_srs_tolerance([positive, negative], spec, path='tolerance.png')
```

`ShockSpecification.from_breakpoints` reads the breakpoints onto the
points of the spacing, from the first breakpoint up, so a spectrum
computed at the same spacing from the same lowest frequency lands on
the requirement's points. The spectrum's last point can fall past the
last breakpoint, where nothing is required, and is not judged. Each
band given becomes a pair of limit curves, so the specification
carries its tolerance, the comparison the application draws when the
spectrum and the specification are selected together shades it, and
the judgement reads it.
`visualdynamics.core.compliance.srs_points` is the same judgement on
plain arrays.

## The picture

`plot_srs_tolerance` draws the requirement in gray, the zones of its
band the way a random specification's are shaded (yellow outside the
inner band, red above and blue below the outer one), and each spectrum
in its own color and marker. Every point the rule counts against a
spectrum is its marker filled. Each spectrum's legend entry gives its
polarity, the count and share of its points inside, and whether it
passed.

![A requirement at three breakpoints with its −1.5/+3 dB inner and
−3/+6 dB outer bands; a maximax spectrum on it but for a narrow dip
near 400 Hz, five points filled, three of them past −3 dB; and a
negative spectrum falling away above 400 Hz, its last 25 points
filled](images/srs-tolerance.png)

In the figure the maximax spectrum reads −0.97 dB in RMS, inside the
3 dB the bars shade at. It has 87 of its 92 points inside the inner
band, which meets a 90 percent share, and it still fails, because
three of the points in the dip are past the outer band. The negative
spectrum fails on both counts.
