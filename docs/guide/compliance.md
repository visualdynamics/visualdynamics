# How a comparison is judged

A random vibration test is judged against a band: the specification
says what the control PSD must be, and warning and abort lines say
how far it may stray. Whether a measurement met that is one question
asked cell by cell, and the answer is the same whatever form the
two objects take — a controller's target on its lines, a requirement
written at a few breakpoints, or either banded onto octave bands,
against a narrowband PSD or a banded one. This page states the rule
and shows every comparison there is, each figure drawn by the same
call the application uses (`tools/make_concept_figures.py` regenerates them).

## The rule

A PSD is a density, and a density means its area — [how a PSD means
its area](psd-reading.md) says why the picture and the integral come
from one reading. A comparison is built on that:

1. **The cells.** The comparison happens on the coarser of the two
   grids. A requirement written per octave band is a requirement on
   the band's power, not on every line under it. When both are on
   lines, the wider lines are the cells; when the requirement is a
   curve between breakpoints, the measurement's own bins are. A
   measurement on octave bands is compared band against band with a
   requirement on the same bands, and with nothing else.
2. **The stretch.** Each cell is cut to what both objects speak for:
   the lines or bands the requirement is written on, and the lines
   the measurement holds. A controller stores its target on every
   FFT line to Nyquist and writes zero or NaN where it did not
   control — those lines are holes, not a requirement of nothing.
   A cell nothing is written in is not judged; one written over part
   of its width is judged over that part, and only that part.
3. **The judgment.** Over each cell's stretch, the measurement's
   power — its area, read as it is drawn — against the power the
   limit asks for over the very same stretch, the limit read the way
   its specification is: the exact area under the power law between
   breakpoints, a density per line or per band otherwise. The cell is
   out when it holds more than the upper abort limit asks, or less
   than the lower one.
4. **The level.** The RMS error compares the two areas summed over
   every cell: the measurement's against the requirement's, both over
   the same stretches. The *band outside abort* is the width of the
   cells that are out as a share of the width judged, so it reads
   the same whether the cells are lines or bands.

One function holds it — `visualdynamics.core.compliance.judge` — and
the compliance table, the application's comparison plot and the
report's page all ask it. A line boxed red on the plot is a cell
counted in the table.

## The three comparisons

A measurement is on lines or on octave bands; a requirement is a
curve at breakpoints, a controller's density per line, or on octave
bands. Three of the pairings are comparisons. A narrowband PSD is
judged against a breakpoint curve or against a requirement on lines;
a banded PSD is judged against a requirement on the same bands, and
only that. A requirement on octave bands says nothing about the
lines under its bands, and a band's power cannot be attributed to
part of its width, so a narrowband PSD against a banded requirement,
a banded PSD against a curve or lines, and sixth-octave bands
against third-octave bands are not judged: nothing is marked, the
comparison answers with the reason, and the window's status line
shows it. Band the other object the same way — `compute_octave` on
the specification bands its limits with it — and compare band
against band. A report block that bands a measurement for itself
bands the specification with it.

Every figure below is one measurement of the same requirement: a
decade of flat top with power-law skirts, warning at ±3 dB and abort
at ±6 dB, written at four breakpoints. The measurement sits on the
target with two departures — a resonance over the upper abort limit
near 300 Hz and a notch under the lower one near 1200 Hz — and runs
on to Nyquist, past the requirement's end at 2000 Hz.

### A narrowband PSD against a breakpoint specification

The requirement is a curve, so the cells are the measurement's own
1 Hz bins. Under each bin the exact power-law area of the limit is
what the bin's power is judged against; on a slope that is a little
more than the limit's value at the bin's center, and on a breakpoint
the bin is judged on both sides of the corner at once. The bins at
the two ends are cut at the first and last breakpoint and judged
over the part inside — a line's density is flat across a hertz, so
the cut is exact.

![A 1 Hz PSD against a four-breakpoint specification: the resonance
boxed red over the upper abort limit, the notch boxed blue under the
lower one, nothing marked past the requirement's
end](images/compliance-lines-vs-breakpoints.png)

### A narrowband PSD against a specification on lines

A controller writes its target as a density per line. Here the lines
are 2 Hz apart and the PSD's are 1 Hz apart, so the cells are the
requirement's lines and each takes the two measured bins under it,
integrated. Lines the controller wrote as NaN — everything past
2000 Hz — are not judged.

![The same PSD against the requirement on 2 Hz lines: the marks land
on the requirement's lines](images/compliance-lines-vs-lines.png)

### An octave-band PSD against an octave-band specification

Both on the same bands, same fraction: band against band, each
band's power against its band's limit. The resonance's band is
boxed red as a whole; the notch, which the line-by-line figure
boxed blue, averages back inside its band's lower limit — a
requirement on bands asks about band power, and that is what is
judged. Banding conserves area, so the RMS error is the narrowband
pair's to within the end bands.

![Third-octave against third-octave: band against
band](images/compliance-octave-vs-octave.png)

## The edges

### Where a requirement ends inside a band

Here the requirement's lines end at 1450 Hz, and the third-octave
band that holds 1450 Hz reaches to 1778. An octave band is a defined
frequency band: banding keeps it whole — its center, width and edges
are the standard's — and the band holds what the lines put in it,
spread over its whole width, so it reads lower than its neighbors.
That is what a band of a requirement ending inside it is. The
comparison then judges the whole band: the measurement's power from
1413 to 1778 Hz against that. A measurement running on past 1450 Hz
at the target's level holds several times what the banded
requirement asks there, and the band is out — the consequence of
banding a requirement onto a band it does not fill, which the
narrowband comparison (the lines figures above) does not have.
Past the last band nothing is judged.

![The banded requirement's last band whole and lower, the banded
measurement over it there and unjudged beyond
it](images/compliance-edge.png)

### A hole in a requirement

A controller's notch — lines written as zero across 400–500 Hz. Zero
is not a requirement of silence: those lines say nothing, and cells
falling in the notch are not judged. The RMS is taken over the
written lines only, on both sides.

![The requirement with a notch of zeros: nothing marked inside it,
the comparison resuming on either
side](images/compliance-hole.png)

## Checking it

```python
from visualdynamics.core.compliance import cells, compare, judge

found = cells(specification, measured)        # the cells, cut to what both hold
verdict = judge(specification, measured, limit='abort_upper', over=True)
verdict['out']                                # a bool per cell
verdict['lines']                              # the same, on the measurement's lines
result = compare(specification, measured)     # RMS error, band outside abort
```

The array form, `compliance.outside`, judges a limit and a
measurement held as plain arrays through the same rule, and
`compliance.comparable` says in words why a pair is not compared,
or None when it is.
