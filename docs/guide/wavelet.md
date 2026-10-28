# The wavelet reading

A PSD says which frequencies a record contains and nothing about when
they occur. For a stationary run that is all there is to know. For
anything else, such as a rattle that comes and goes, a shock's
ring-down, or a resonance a sweep passes through, the timing is what
matters, and the spectrum has averaged it away. The wavelet reading
shows it: time across, frequency up a logarithmic axis, and amplitude
as color. In 3-D, amplitude is the height of the surface above the
floor.

![The wavelet reading of a four-tone sweep: two sweeps crossing, a
near-dwell holding its frequency, a log sweep walking its
octaves](images/sine-wavelet.png)

## Where it lives

*Wavelet* is on a time history's plot bar with the record's other
readings, and at most one of them is on at a time. It opens in
3-D, because a scalogram is a function of two variables and a surface
shows that directly; the *3D* button switches it to the flat picture
shown above. In a report, the transient and shock templates include
the same figure, drawn from the same computation. For a shock it is
used in place of a PSD, because a density's level depends on how much
quiet time before and after the event the recording holds.

From a script, the same reading is one call:

```python
from visualdynamics.plot import plot_scalogram
plot_scalogram(history, path='scalogram.png')
```

## The panel

The scalogram shows one record at a time, because it is dense and two
side by side are hard to read. The record is chosen the same way as
everywhere else, in the project tree. Pick a record in the object's
grid and the scalogram is of that record; select the whole object and
it opens on the first record, with the tree's selection moved to that
record so it always says what the picture shows. The panel holds the
settings the tree cannot:

- *From* and *To* set the frequency range. The default covers
  everything the record can carry: from the frequency where the longest
  wavelets still fit inside the record up to just under Nyquist
  (0.98 of it). The amplitude stays accurate to a fraction of a percent
  right up to the top.
- *Per octave* sets how many lines are computed per octave. The axis is
  logarithmic because a wavelet's bandwidth is a constant fraction of
  its frequency. The default of twelve draws a ridge that looks
  continuous without computing rows that cannot be told apart.
- *Cycles* sets the trade between time and frequency resolution. The
  transform slides a wave packet along the record, and *Cycles* is how
  many cycles fit under its window. Few cycles resolve when something
  happens and blur its frequency; many cycles resolve the frequency and
  blur the timing. The default of six is the usual choice and close to
  the lowest value for which the transform is valid. The box takes 3
  to 30.

Below the settings, the derived rows (*Cone at bottom*, *Resolves* and
*Lines*) show what the settings imply before the transform runs. The
most important of these is the cone, described below.

## What the picture can and cannot show

Four limits apply, and each one can be seen in the picture itself.

The cone of influence. A low-frequency wavelet is long, and near the
ends of the record it runs off the edge, so what it reads there is
where the record was cut, not the test article. The affected span is
shaded in the flat picture and walled off in 3-D, and it is widest at
the bottom of the frequency axis. It is marked because it looks exactly
like data. A range whose bottom rows are entirely inside the cone shows
only the edges of the recording; the panel warns when the cone covers
a third of the record or more at the lowest frequency, and says so
plainly when it covers all of it.

Amplitude is in the record's own units. A 2 g tone reads 2 wherever it
sits on the frequency axis, so the color bar carries the channel's
units and a value can be read off the picture. (The other convention
in the literature, unit energy, draws the same tone four times taller
at 50 Hz than at 800 Hz; Visual Dynamics does not use it.)

A ridge that ripples usually means the record itself ripples. Two
tones within one wavelet's bandwidth of each other add up to an
amplitude-modulated signal, exactly as they would through any filter
that wide, and the ripple is at their difference frequency. Raising
*Cycles* narrows the band and separates tones that are far enough
apart. Tones a hertz apart at hundreds of hertz cannot be separated by
any window short of one that removes the time axis, and their beat is
part of the measurement. A single steady tone draws perfectly smooth,
which shows that the transform adds no ripple of its own.

A long record is drawn at a picture's width. The transform is computed
for every sample, but the picture holds time to 4096 columns, each
showing the largest magnitude in its slice of the record. Because it is
a peak-hold, a burst only a few samples wide still shows up as a
column; taking every nth sample instead could skip the samples it is
in. A record shorter than that is drawn sample for sample, with
nothing held. This is what lets a five-minute record at 16 kHz open in
a few seconds. The scripting `plot_scalogram` draws the same way, and a
report figure holds time the same way to the width of a document and
says so in its note.

## When to use it

- A sweep: each tone's path is a ridge, and crossings, dwells and the
  resonances it passes through are easy to see (the
  [sine workflow](workflows/sine-workflow.md) starts with it).
- A transient or a shock: what rang, and for how long after the event.
  The transient and shock reports include the figure for this reason.
- A record that fails kurtosis: the kurtosis bars say a channel has
  peaks its spectrum did not predict, and the scalogram shows when they
  happened and at what frequency, which is usually enough to find the
  source of the rattle.
- Not a stationary random run: everything about it is in the PSD, and
  the scalogram will draw a texture with nothing to find in it.
