# A mixed-mode test as one specification

A sine-on-random test drives one signal made of parts: a broadband
random floor written as breakpoints, and tones written in amplitude
(peak), each at a fixed frequency or swept. A random-on-random test
lays narrow bands of random over the floor instead, each a few percent
of its center wide, fixed or swept. The parts are judged by different
rules, and a report that quotes one rms figure for the test has said
nothing, because there are three: the floor's rms with the tone lines
left out, each tone in amplitude, and the total rms of the whole drive.

Take four tones of 0.12, 1.944, 2.5 and 1.5 g peak over a 1.0 grms
floor. The tones' mean squares, \(A^2/2\), are 0.0072, 1.89, 3.13 and
1.13 g², 2.48 g rms together. The parts are independent, so their mean
squares add, and the total is

\[
g_{\text{rms,total}} = \sqrt{g_{\text{rms,floor}}^2
  + \sum_i \frac{A_i^2}{2} + \sum_j W_j B_j}
  = \sqrt{1.0 + 6.15} = 2.67
\]

with \(W_j\) a narrowband's level and \(B_j\) its width. All three are
right, and each misleads when quoted alone as "the test level": 2.67
read as the floor is nearly three times too high, and a tone's g peak
set beside the floor's grms compares two different quantities.

## The specification

`MixedSpecification` holds the parts together: the floor (a random
`Specification`), the tones (a `SineSweepSpecification`), the
narrowbands (`Narrowband`) and the width left out of the floor either
side of each tone.

```python
from visualdynamics.core.mixed_mode import (MixedSpecification,
                                            Narrowband, Tone)

g = 9.80665
spec = MixedSpecification(
    floor,                                    # the random specification
    [Tone(4.86, 0.12 * g), Tone(19.44, 1.944 * g),
     Tone(38.88, 2.50 * g), Tone(58.32, 1.50 * g)],
    seconds=60.0,                             # how long a fixed tone holds
    exclusion=2.0)                            # Hz left out either side
spec.reference_rms()                          # floor, tones, bands, total
```

The tone table is entered as `Tone` rows. A fixed tone is a row with a
frequency and an amplitude; a swept tone adds `sweep_to`, a `rate` and
a `mode` (`LINEAR` in Hz/s, `LOG` in octaves a minute). A fixed tone
is written into the sine specification as the dwell a controller's
file holds (`SineTone.dwell`).
`spec.tone_table()` reads any sine specification back as rows, fixed
where every segment holds one frequency.

A narrowband is written the way a standard writes it, or by its edges:

```python
Narrowband.plus_minus(400.0, 5.0, 0.01 * g**2)    # 380 to 420 Hz
Narrowband.between(380.0, 420.0, 0.01 * g**2)     # the same band
Narrowband(56.0, 0.05 * g**2, width=4.0,          # swept 56 to 93 Hz
           sweep_to=93.0, rate=2.0)               # and back, at 2 Hz/s
```

Its `level` is the density it adds over the floor between its edges, so
its mean square is the level times its width. A band given as a share
of its center (`plus_minus`) widens with the center as it sweeps, the
way a harmonic's band does; one given in hertz keeps its width. A
"5" typed into a field that wants the full width makes a band half as
wide, with half the mean square, which is why a band is checked by its
edges in hertz: `band.edges()`, and on the plot below. `tolerance_db`
is the tolerance the plan states for a band, since a swept band cannot
have the floor's statistics.

### Nothing new in the file

A mixed test adds no object of its own to a project file. The floor is
saved as the random specification it is and the tones as the sine
specification they are, the two objects a random and sine project
already holds; `MixedSpecification(floor, sweep)` puts them back
together. A fixed tone is a dwell, so no segment type is added. The
narrowbands and the width left out are attributes of the floor,
`Specification.narrowbands` and `Specification.tone_exclusion`, saved
with it the way its band constraints are, so a file written with them
opens in an older version as the plain random specification. Given to
`MixedSpecification`, they are written onto a copy of the floor, which
`spec.floor` then holds; add that to the project to keep them.

## The data sheet

`judge_mixed(history, spec)` reads one channel of a record against it,
part by part:

1. Where the tones begin in the record is found by matched filter, as
   the extraction finds it, unless `onset` is given. Each tone is read
   through a tracking band locked to its frequency (0.5 Hz wide by
   default), the way a controller reads it
   ([How a tracking filter reads a sweep](sine-tracking.md)). The band
   settles in about one over its width, so nothing before that is read;
   the tone's level is the median of the readings after it, compared
   with the table at plus or minus 10 percent, an example of what test
   documents ask for. A small tone near the floor scatters through its
   band from reading to reading, and the median reads its level through
   that scatter.
2. What each band passed is taken out of the record. What is left, the
   residual, is the floor and the narrowbands.
3. The floor is judged on the residual's averaged spectrum with the
   lines that held a tone, and every narrowband's lines, left out. A
   line held a tone when the tone spent a tenth of its run or more
   within the left-out width of it: every line beside a fixed tone, and
   few or none of the lines a long sweep passes, since it is near each
   only briefly and what the band took with it is a small share of that
   line's power. The floor is judged as a comparison is
   (`compliance.judge`): its rms against the reference's over the same
   lines (plus or minus 10 percent), and its lines against the
   reference moved 3 dB either way. An averaged line scatters, so a
   correctly controlled floor has a few lines outside 3 dB on a short
   record: on a 10 s record read from 2 s, about 28 degrees of freedom
   a line, 1.4 percent of lines, five of 384. The lines fail only when
   more are outside than chance would put there one time in a thousand;
   the sheet gives the count beside the number chance expects and the
   number allowed. A floor with no line left to judge is not judged, and
   the reading says so.
4. Each narrowband is judged at the degrees of freedom of a narrow
   band. A fixed band's level is the mean of its lines, worth the
   per-line degrees of freedom times its lines over the window's noise
   bandwidth. A swept band stays over any one frequency only as long
   as it takes to move its own width, so it is read in slices half that
   long, each worth only the few frames it holds, and its level is the
   slices pooled. Its tolerance is the stated one, or the floor's,
   widened to the chi-square scatter at those degrees of freedom, so
   it is never failed for scatter its statistics cannot avoid.
5. The total is every line of the record's spectrum. It has no
   tolerance of its own: it is the figure the amplifier, the coil and
   the fixture see, and the first one another lab computes.

```python
from visualdynamics.core.mixed_mode import judge_mixed

reading = judge_mixed(history, spec, onset=0.0)  # tones from the start
reading.floor.residual / g        # 1.003, the residual over every line
reading.tones_measured / g        # 2.477, the tracked tones' rms
reading.total.measured / g        # 2.671, the whole drive
print(reading.describe('g', 1 / g))

visualdynamics.plot.plot_mixed_mode(history, spec, onset=0.0,
                                    path='sheet.png')
```

The figures are for one minute of the composite above, the guide
composite of [Reading a tone off an ASD](tone-readout.md). The tracked
tones read 0.122, 1.945, 2.496 and 1.498 g peak, 2.477 g rms together
against 2.479 written; the residual 1.003 grms over every line, and
0.986 against 0.986 written with the tone lines left out (2.8 percent
of the floor's lines); and the whole drive 2.671. Every value is
stored in SI, as every record is; `describe` and the plot take the display unit.

`plot_mixed_mode` draws the residual's spectrum over the reference, the
record with its tones dotted, a tone marker on each fixed tone (the
height its spike stands at for the spacing and window, with its
amplitude read off the spectrum in words), the lines left out hatched
and labeled "left out", and the data sheet beside it: the floor, the
tones together and each tone, each narrowband and the total, each with
its written and measured figure, its tolerance and its verdict in
words, and the settings that decide what the numbers mean under them.

![Sine on random: the residual and the record over the floor, the lines
left out hatched, and the sheet: 0.986 grms for the floor, 2.48 g rms
for the tones, 2.67 for the total](images/mixed-mode-sheet.png)

![Random on random: a band plus or minus 5 percent about 300 Hz and a
16 Hz band sweeping 150 to 250 Hz, each judged at its own degrees of
freedom](images/mixed-mode-bands.png)

In the second figure the swept band is read in 37 slices of about 10
degrees of freedom each, where a floor line has 112; one slice's
scatter alone is wider than 3 dB either way, and the band's level
pooled over the slices reads 0.2 dB over its written level. Its edges
are drawn dash-dotted and its travel cross-hatched, so the plot shows
where the band went as well as how strong it was. The figures are made
by `tools/make_concept_figures.py`.

## In the app

In a random and sine project, select the record and choose the sine
view on the bar. When the project also holds the floor's random
specification at a channel the record has, a **Mixed Mode** readout
sits under the extraction's settings: the channel, the width left out
either side of a tone, the tracking band and the tones' tolerance.
Reading a long record takes seconds, so the first reading waits for
**Read Data Sheet**; once the sheet is up, an edit reads again at once.
The sheet shows the same rows as the plot, the settings under them,
and a small spectrum of the residual over the reference with the
left-out lines hatched. The narrowbands are the floor's own
(`Specification.narrowbands`). The reading is journaled as its
`plot_mixed_mode` call.

## What to watch

- A tone's spike on the spectrum is not its level. The tone is judged
  in amplitude, read through its band; the marker on the plot shows
  where the spike stands for this spacing, and is a picture.
- The width left out is part of the result. A floor feature inside it,
  a fixture resonance at a tone, is not judged against the floor's
  tolerance at all. The sheet gives the width and the share of the
  floor's lines it took, and says so in a note when more than half the
  floor was left out.
- A steady tone has nothing in time for the matched filter to lock onto
  but where it starts, so with no `onset` given it is placed somewhere
  it plays, never before it, and the floor is read from there. Give
  `onset` when the tones' start is known.
- A record of another quantity than the floor or the tones (velocity
  against an acceleration floor) is refused, by the rule every
  comparison uses.
- A tone at the floor's edge has no floor under it on one side. The
  reading says so in a note.
- A swept band moving faster than a frame fine enough to resolve it
  reads low; the note says so and gives the line spacing used.
