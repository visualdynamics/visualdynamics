# How a PSD means its area

For a power spectral density, the area under the curve is the
measurement. The RMS of the signal is the square root of the area
under its PSD, a specification's overall level is the area under its
breakpoints, and octave banding is the same area divided up
differently. So Visual Dynamics stores how a value is read between the
points as one property of each object, `interpolation`, and both the
integral and the plotted curve come from it. A spectrum is never drawn
one way and integrated another.

There are two readings, `'bin'` and `'log_log'`. Every figure below is
drawn by the same call the application uses
(`tools/make_concept_figures.py` regenerates them).

## 'bin': constant across its own bin

A computed PSD is a density per analysis bin: each line's value is
constant across its bin, the area is the sum of value times width, and
the spectrum is drawn as steps. For a narrowband spectrum the bin edges
are the midpoints between neighboring lines, which is exact. In this
example the values of ten coarse lines sum to 10 over 10 Hz bins, so
the area is exactly 100 (m/s²)² and the RMS is exactly 10 m/s². The
steps show that sum directly.

![Ten narrowband lines drawn as steps: each value flat across its
midpoint-to-midpoint bin, the edges landing half a bin beyond the
first and last lines](images/psd-reading-steps.png)

## The same power on octave bands

Banding onto proportional bands (the *Octave Bands* reading on the
plot bar and its *Apply Octave Bands* button, or
`project.compute_octave` in a script) conserves the area: the same
power, arranged by band. A specification is banded the same way, with
its warning and abort limits. A curve written at breakpoints
contributes the exact power-law area over each band, a controller's
target on lines is integrated as a density, and the result is a
specification on the bands, to be compared band for band against a
measurement banded the same way. An octave-band spectrum sets
`bandwidth`, because its bins are geometric and its edges come from
the standard, not from its neighbors. Edges inferred from the
band centers would be slightly off at every band and wrong at the two
ends. The steps land on the standard's own edges, and the RMS is still
exactly 10 m/s².

![The same spectrum on third-octave bands: geometric bins on the
standard's edges, wider with frequency, holding the same
area](images/psd-reading-octave.png)

## 'log_log': a power law between breakpoints

A specification written at a few breakpoints is a continuous
requirement. Between two breakpoints it is the straight line the two
points make on log-log axes, which is a power law, and its area has a
closed form. It is drawn as that curve (on the linear frequency axis
below, the power-law segments show their true curvature), and its
`area()` integrates the same power law: 6.06 m/s² RMS for this one.

![A four-breakpoint specification drawn as the power law it is: flat
top, rising and falling power-law skirts](images/psd-reading-law.png)

The reading belongs to each object, not to its class. A specification
written at a dozen breakpoints is a curve, but a specification
computed from a record, such as a transient target's PSD, is a density
like any other and carries `'bin'`.

## Checking it

```python
psd.area()                  # the one integral, read as drawn
psd.area(low=20.0, high=2000.0)      # over a band
rms = psd.area() ** 0.5

banded = psd.to_octave(6)   # sixth octaves, conserving the area
assert abs(banded.area() - psd.area()) < 1e-9 * psd.area()
```

`area()` is the only integral, and it reads a spectrum the same way it
is drawn; nothing outside the object makes that choice. It integrates
one record, the first unless `record=` says otherwise.

In the window, a specification selected on its own opens as its
spectra. *RMS* on the plot bar is the other reading: one bar per
channel showing the level the specification asks for (the square root
of that same area), with a table of the numbers beneath. Nothing is
colored, since a specification on its own has nothing to be compared
against. Records picked in the tree restrict both readings. A
specification selected together with a measurement offers the
comparison's three readings instead: *Spectra*, *RMS Error* and
*Lines Out* ([How a comparison is judged](compliance.md)).
