# How a comparison is judged

A random vibration test is judged against a band: the specification
says what the control PSD must be, and the warning and abort lines say
how far it may depart from that. Whether a measurement met the
specification is one question, asked cell by cell, and it is answered
the same way whatever form the two objects take: a controller's target
on its lines, a requirement written at a few breakpoints, or either one
banded onto octave bands, against a narrowband PSD or a banded one.
This page gives the rule and shows every comparison there is. Each
figure is drawn by the same call the application uses
(`tools/make_concept_figures.py` regenerates them).

## The rule

A PSD is a density, and a density means its area. [How a PSD means
its area](psd-reading.md) explains how the plotted curve and the
integral come from one reading. A comparison is built on that:

1. The cells. The comparison is made on the coarser of the two grids,
   because a requirement written per octave band is a requirement on
   the band's power, not on every line under it. When both are on
   lines, the wider lines are the cells. When the requirement is a
   curve between breakpoints, the measurement's own bins are the
   cells. A measurement on octave bands is compared band against band
   with a requirement on the same bands, and with nothing else.
2. The stretch. Each cell is cut to the part both objects cover: the
   lines or bands the requirement is written on, and the lines the
   measurement holds. A controller stores its target on every FFT line
   up to Nyquist and writes zero or NaN where it did not control.
   Those lines are holes in the requirement; they do not ask for zero.
   A cell with nothing written in it is not judged. A cell written over
   part of its width is judged over that part only.
3. The judgment. Over each cell's stretch, the measurement's power
   (its area, read as it is drawn) is compared with the power the limit
   asks for over the same stretch. The limit is read the way its
   specification is read: the exact area under the power law between
   breakpoints, or a density per line or per band otherwise. The cell
   is out when it holds more than the upper abort limit asks for, or
   less than the lower one.
4. The level. The RMS error compares the two areas summed over every
   cell, the measurement's against the requirement's, both over the
   same stretches. The *band outside abort* counts the cells that are
   out as a share of the cells judged, each weighted by how much of it
   was judged, so a line cut in half by the end of a written stretch
   counts for half. Six bands out of forty-seven read as thirteen
   percent whether they sit at the bottom of the range or at the top.
   A share of the width in hertz would not give that: proportional
   bands are equal in log frequency, so in a sixth-octave set from
   10 Hz to 2 kHz the highest band is about two hundred times as wide
   as the lowest.

One function holds the rule, `visualdynamics.core.compliance.judge`,
and the compliance table, the application's comparison plot and the
report's page all use it. A line boxed in red on the plot is a cell
counted in the table.

## The three comparisons

A measurement is either on lines or on octave bands. A requirement is
either a curve at breakpoints, a controller's density per line, or on
octave bands. Three of these pairings are comparisons. A narrowband
PSD is judged against a breakpoint curve or against a requirement on
lines; a banded PSD is judged against a requirement on the same bands,
and only that. A requirement on octave bands says nothing about the
lines under its bands, and a band's power cannot be assigned to part
of its width. So a narrowband PSD against a banded requirement, a
banded PSD against a curve or against lines, and sixth-octave bands
against third-octave bands are not judged: nothing is marked, the
comparison returns the reason, and the window's status line shows it.
To compare them, band the other object the same way
(`compute_octave` on the specification bands its limits with it) and
compare band against band. A report block that bands a measurement
itself bands the specification with it.

Every figure below is one measurement of the same requirement: a
decade of flat top with power-law skirts, warning at ±3 dB and abort
at ±6 dB, written at four breakpoints. The measurement follows the
target with two departures, a resonance over the upper abort limit
near 300 Hz and a notch under the lower one near 1200 Hz, and it
continues up to Nyquist, past the end of the requirement at 2000 Hz.

### A narrowband PSD against a breakpoint specification

The requirement is a curve, so the cells are the measurement's own
1 Hz bins. The power in each bin is judged against the exact power-law
area of the limit under that bin. On a slope, that area is a little
more than the limit's value at the bin's center times its width, and a
bin that contains a breakpoint is judged on both sides of the corner
at once. The bins at the two ends are cut at the first and last
breakpoint and judged over the part inside. A line's density is
constant across its 1 Hz bin, so the cut is exact.

![A 1 Hz PSD against a four-breakpoint specification: the resonance
boxed red over the upper abort limit, the notch boxed blue under the
lower one, nothing marked past the requirement's
end](images/compliance-lines-vs-breakpoints.png)

### A narrowband PSD against a specification on lines

A controller writes its target as a density per line. Here the
requirement's lines are 2 Hz apart and the PSD's are 1 Hz apart, so
the cells are the requirement's lines, and each one takes the two
measured bins under it, integrated. Lines the controller wrote as NaN
(everything past 2000 Hz) are not judged.

![The same PSD against the requirement on 2 Hz lines: the marks land
on the requirement's lines](images/compliance-lines-vs-lines.png)

### An octave-band PSD against an octave-band specification

Both are on the same bands with the same fraction, so the comparison
is band against band: each band's power against that band's limit. The
resonance's band is boxed red as a whole. The notch, which the
line-by-line figure boxed blue, averages back inside its band's lower
limit, because a requirement on bands asks about band power and that
is what is judged. Banding conserves area, so the RMS error matches
the narrowband pair's, apart from the end bands.

![Third-octave against third-octave: band against
band](images/compliance-octave-vs-octave.png)

## The edges

### Where a requirement ends inside a band

Here the requirement's lines end at 1450 Hz, and the third-octave band
that contains 1450 Hz extends to 1778 Hz. An octave band is a defined
frequency band, so banding keeps it whole: its center, width and edges
are the standard's. The band holds what the lines put into it, spread
over its whole width, so the banded requirement's last band reads lower
than its neighbors. That is the correct value for a band in which the
requirement ends, and the octave-band specification shows it instead
of assuming the requirement reached 1778 Hz.

The banded measurement is judged against that band over its whole
width, 1413 to 1778 Hz, and in the usual case that is fair. A
controller holds the narrowband response to the requirement and drives
nothing past it, so the response also falls away above 1450 Hz. Its
band then holds about what the requirement's band holds, and the band
is in. The figure shows that case: the response is on the target up to
1450 Hz and a fiftieth of it above, the last band on each side is low
by about the same amount, and nothing is judged past the last band.

A response that did not fall away, one that stayed at the target's
level up to 1778 Hz, would hold several times what the banded
requirement asks for in that band, and the band would be out. That is
a true statement about the band: the banded requirement asks for that
much power in that band and no more. The narrowband comparison would
not say it, because its cells stop where the lines stop, so read the
narrowband figures for the end of a requirement.

![The banded requirement's last band whole and lower, the banded
measurement's last band low beside it and in, nothing judged
beyond](images/compliance-edge.png)

### A hole in a requirement

A controller's notch: lines written as zero from 400 to 500 Hz. Zero
here does not ask for silence. Those lines say nothing, and cells that
fall in the notch are not judged. The RMS is taken over the written
lines only, on both sides.

![The requirement with a notch of zeros: nothing marked inside it,
the comparison resuming on either
side](images/compliance-hole.png)

## A specification that arrives as a PSD

A comparison needs to know which object is the requirement. A
Rattlesnake file says so, and so does a project saved by Visual
Dynamics, but some formats have no data type for a specification.
ESCDF's types include power spectral density and nothing for a target,
so a specification written by another program arrives as a PSD that
only its name identifies. Visual Dynamics does not guess from the
name. Select the PSD and use *Convert to Specification* on the bar
(`project.convert_to_specification(name)` from a script). It keeps its
name, its group and everything that refers to it, and when it is
selected together with a measurement the two are compared. *Convert to
PSD* (`project.convert_to_psd(name)`) goes the other way. The button
reads *Convert to CPSD* when the specification has cross terms. A PSD
has no warning or abort limits, so a specification's limits are left
out, and the status line says so. This is how a target written on the
Specification sheet, which always has its limit bands, becomes a CPSD
to generate a signal or predict a response from.

A specification on frequency lines converts as it is. One written at
breakpoints is read onto evenly spaced lines first, since a density to
drive or predict with is read line by line. *Convert to PSD* then opens
a pane beside the plot with the **Spacing** of the lines. It opens at
the spacing the specification's sheet remembers, and blank when there
is none. The plot previews the result, the pane counts the breakpoints
and the lines, and **Generate** converts it. The reading is the same as
the sheet's *Interpolated* form: each channel on its power law between
breakpoints, the lines at multiples of the spacing over the range the
breakpoints cover. A cross term that changes with frequency is refused,
because the reading keeps one coherence and phase per pair. From a
script, `project.convert_to_psd(name, spacing=5.0)`; without
`spacing`, the remembered one is used, and with none remembered the
conversion is refused rather than guessed.

## The test level

A run is often recorded below the level the specification states, at
-6 dB or -3 dB on the way up to full level. It is judged after the
measurement is scaled up by that amount. The data itself is never
changed; the scale is applied only where the comparison is drawn and
scored.

When no level has been typed, it is detected. Both spectra are banded
to sixth octaves, each band gives the decibels it would need added,
and each of those is rounded to the nearest 3 dB step, since a
controller's level is set in 3 dB steps. The most common step is used
when at least three quarters of the bands lie within half a step of
it, their median is within 0.75 dB of it, and the total power agrees
with it within 3 dB. A step outside -6 to +24 dB added is never used.
Otherwise the comparison is shown as measured, at 0 dB. The *Scaling*
field on the plot bar shows the decibels added to the measurement (+6
for a run at -6 dB). It takes a typed value in whole decibels (0 holds
the comparison unscaled) and goes back to detection when cleared.

The level belongs to the measurement and to the octave PSDs made from
it, all of them at once, so the narrowband and banded comparisons and
the report use the same number. From a script:

```python
project.comparison_scale('Specification', 'Time History PSDs')
project.set_comparison_scale('Time History PSDs', 6)     # hold +6 dB
project.set_comparison_scale('Time History PSDs', None)  # detect again
```

## Units in a comparison

A specification and a measurement are compared channel by channel, and
the values are held in SI once their units are declared and as they
came otherwise. So every comparison (the PSD levels and lines here, the
SRS readings, the sine levels against their tones and a transient
against its target) follows one rule, `compliance.units_check`:

- Both sides declared: they must be the same quantity. A velocity PSD
  says nothing about an acceleration specification, and the pair is
  refused.
- One side declared: refused. One side is in SI and the other is not,
  so the ratio would be off by the unit's scale. Set the other side's
  units first. A sine specification whose unit is not set counts as
  undeclared even though it names its quantity.
- Neither side declared: compared, on the reading that two sets of raw
  numbers were brought in the same unit. The result carries a note,
  and the bars' count and the report's caption say the values were
  compared as raw numbers.

A refused channel keeps its row. The bars write the reason along it,
the table shows it in place of the numbers, the status line names it,
and the channel is left out of the count and the pass/fail verdict.

## Checking it

```python
from visualdynamics.core.compliance import cells, compare, judge

found = cells(specification, measured)        # the cells, cut to what both hold
verdict = judge(specification, measured, limit='abort_upper', over=True)
verdict['out']                                # a bool per cell
verdict['lines']                              # the same, on the measurement's lines
result = compare(specification, measured)     # the RMS levels and their difference,
result['difference_db']                       # in dB,
result['abort_percent']                       # and the band outside abort
```

The array form, `compliance.outside`, judges a limit and a measurement
held as plain arrays by the same rule, and `compliance.comparable`
returns, in words, why a pair is not compared, or None when it is.
`core.chance.compare_chance` makes the same comparison and sets beside
each count the number of lines chance alone puts outside at the PSD's
degrees of freedom ([Lines outside a band by chance](comparison-chance.md)).
