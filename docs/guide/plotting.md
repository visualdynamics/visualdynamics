# Plotting

Every plot the app draws can also be drawn from a script, and every
plotting call takes `path=` to write the figure to a file instead of
opening a window. The script and the window use the same renderer, so
a figure made in a script is the figure the app shows.

## From an object

```python
frf.plot()                              # curves, in a window
frf.plot(records=[0, 1], path='frf.png')  # or straight to a file
frf.plot(component='imag')              # magnitude / real / imag / phase
frf.plot_cmif(shapes)                   # CMIF, synthesis dashed over it

shapes.plot_mac()                       # auto-MAC
shapes.plot_mac(fem_shapes)             # cross-MAC, on shared DOF names
project.plot_mac('Test', 'FEM')         # cross-MAC across two geometries,
                                        # projected the way the app shows it
shapes.animate(geometry, mode=2)        # the deflection animation
shapes.animate(geometry, 2, screenshot='mode3.png')
shapes.animate(geometry, 2, movie='mode3.mp4')   # whole cycles, as a video
shapes.animate_pair(geometry, 2, fem_shapes, 3, fem_geometry)
                                        # two modes overlaid, as a MAC
                                        # cell animates them

history.animate(geometry)               # time records on the geometry
history.animate(geometry, quantity='force')   # one quantity of a mixed record
history.animate(geometry, movie='run.mp4', seconds=10.0)

frf.animate(geometry)                   # the operating deflection shape,
                                        # at the strongest line
frf.animate(geometry, frequency=647.0)  # or at a chosen frequency

psd.animate(geometry)                   # the envelope: two copies at
                                        # +/- sqrt(PSD), no phase claimed
psd.animate(geometry, frequency=1000.0, quantity='acceleration')
psd.animate(geometry, movie='envelope.mp4')   # every line, once through
```

In the app, an FRF (or a complex spectrum) selected together with a
geometry animates the same way: the records' magnitude and phase at
one frequency line, swept like a complex mode. The cursor on the plot
picks the line, starting on the strongest one, and Play sweeps the
phase. Each line is drawn at full scale, so the shape at an
antiresonance is as easy to see as the shape at a peak; how strongly
the structure responds at that line is read from the plot.

A PSD has no phase, so it is drawn as an *envelope* instead: two
translucent copies of the geometry, deflected by `+sqrt(PSD)` and
`-sqrt(PSD)` at the cursor line. Together they show the extreme each
channel reaches, with no claim about when it reaches it. Size shows
each line at full scale. Color is absolute: each node is colored in dB
below the loudest node at any line, down to a floor of 40 dB, so a
quiet line still shows its shape but is colored as quiet. Play moves
the cursor up the spectrum. A node measured along two axes deflects to
the in-phase diagonal of the two, because two copies of the geometry
cannot show anything else.

A CPSD carries more information than a set of autospectra: its cross
records give each channel's phase relative to the others. A whole CPSD
selected with a geometry therefore animates the *principal operating
deflection shape*, which is the dominant eigenvector of the
cross-spectral matrix at each line (the direction of the output
spectra's own CMIF). No reference has to be chosen. At a well-excited
resonance it is the same shape the FRF's operating deflection shape
shows, obtained from operating data alone. Where a mode is weakly
excited, the dominant eigenvector belongs to whatever carries more
power at that frequency. Picking a reference column in the grid reads
the shape relative to that one channel instead, and picking only
autospectra falls back to the envelope, since phase relative to
nothing is not defined.

Every animation shows one quantity at a time, because a node moved by
an acceleration and the next one moved by a force would show neither.
A record of accelerations and forces animates its accelerations, or
whichever quantity it holds most of, and a box on the toolbar offers
the others. The plot under the animation shows only the quantity being
animated, and the cursor is on that plot. In a script, `quantity=`
picks it, as in the examples above.

```python
coherence.plot_map()                    # frequency across, channel down
geometry.plot()                         # the 3D scene, with its bar
geometry.plot_dofs(frf, 'force')        # labeled DOF arrows
geometry.plot_channels(table, 'reference')  # a channel table's, by role
project.plot_transformation('Rigid', dofs='reference', row=5)  # a row of Φᵀ as arrows
```

The readings on a time history's bar have their own calls in
`visualdynamics.plot`: `plot_scalogram(history)` (the wavelet reading,
drawn at a picture's width whatever the record's length; the
[wavelet guide](wavelet.md) explains how), `plot_kurtosis(history)`,
`plot_time_statistics(history, record)` (one channel's trace, histogram
against the Gaussian and kurtosis frame by frame; see
[time statistics](time-statistics.md)),
`plot_replication_fit(history, rating)` (whether a shaker can play the
record; see [checking a replication reference](replication-fit.md)),
`plot_stroke(history)` (a pulse's acceleration, velocity and
displacement integrated raw, against the shaker's shock ratings; see
[will a pulse fit on the shaker](stroke.md)),
`plot_resample(history, rate)` (the record's spectrum taken to a lower
rate, filtered first, bare and low-passed after, with the span, the
dropped lines and the folds marked; see
[taking a record to a lower sample rate](resample.md)),
and, for a specification comparison, `plot_bars(measured,
specification)`, `plot_comparison(...)`, `plot_ratio(...)`,
`plot_snr(...)` and `plot_replication(...)`. Each draws the figure the
app shows, and each takes `path=`. `plot_srs_tolerance(...)` draws a
shock response spectrum against its tolerance band, point by point
([Judging an SRS against its tolerance](srs-tolerance.md)).
`plot_breakpoints(specification)` draws one channel of a random
specification written at breakpoints (a density per bin is refused)
with each segment's slope and area written on it and
the running rms beneath, as the specification sheet's *Areas and
running rms* does ([Writing a random specification by
slope](breakpoint-slopes.md)).
`plot_chance(psd, spec, combine=8)` draws one channel of a random
comparison as its level over the target in dB, the lines outside each
band counted beside what chance puts there
([Lines outside a band by chance](comparison-chance.md)).
`plot_shock_check(history)` draws a shock record's raw velocity beside
its positive and negative spectra, a figure the app does not show: the
Shocks view states the same check in words
([checking a shock record](shock-check.md)).
`plot_half_power(frf, 180.0)` draws one record with the half-power
reading of a peak on it, the reading the fitting screen's cursor shows
([Reading Q from a peak's half-power width](half-power.md)).
`plot_tone_readout(asd, [19.44, 58.32], window=...)` draws an
autospectrum with each tone read off it, the reading the Tones cursor
shows ([Reading a tone off an ASD](tone-readout.md)).
`plot_amplitude_spectrum(history, 0, window='flattop')` draws one
channel's amplitude spectrum in peak units with its window and
correction in the title, the Amplitude reading's plot
([Reading a tone's amplitude from a record](amplitude-spectrum.md)).
`plot_rms_check(asd, history)` draws an autospectrum with its rms
check written in the corner, the reading the RMS Check toggle shows
([Checking a spectrum against its record](rms-check.md)).
`plot_replication_segments(...)` draws a replicated run judged segment
by segment ([Judging a replication by
segment](replication-segments.md)). A sine specification's sweep
against the modes it passes is `plot_sweep_rate(specification, [(22.0,
40.0)])`, the Sweep Rate reading ([checking a sweep
rate](sweep-rate.md)). `plot_signature(before, after, modes)` draws
two signatures with a window on each mode for its limits and the table
of shifts beside ([comparing signatures](signature-comparison.md)).
`plot_dwell(history, specification)` draws a held tone's level against
time, the reading the sine view's Dwell readout shows ([Reading a
dwell against time](dwell.md)). `plot_transmissibility(reading)` draws
H1 solid and the ASD ratio dashed with H1's band and the coherence
beneath, the reading the FRF view's Transmissibility toggle shows ([The
two transmissibilities](transmissibility.md)).
`plot_conditioning(reading)` draws a matrix's singular values with its
condition number beneath, and `plot_realizability(reading)` the
eigenvalues of a cross-spectral matrix's coherence matrix with what the
repair does to one pair, the readings of the Conditioning and
Realizability toggles ([Checking the matrices of a multi-input
test](mimo-matrix-checks.md)).
`plot_sine_distortion(history, specification, levels)` draws a tone's
distortion against frequency with the limit across it, the reading the
sine view's Distortion readout shows ([Checking a sine record for
distortion](sine-distortion.md)).
`plot_sine_frequency(history, specification, levels)` draws a tone's
frequency error against frequency with the band either side of zero,
the reading the sine view's Frequency readout shows ([Checking a sine
tone's frequency](sine-frequency.md)). `plot_sine_profile(profile)`
draws a sine profile, or a specification's tone read as one, three
ways with its crossovers marked, the reading the Sine Profile toggle on
a sine specification's bar shows ([Writing a sine profile in
segments](sine-profile.md)).
The 3-D forms are in
`visualdynamics.viz`: `scalogram.plot_scalogram_stage(history)` is the
surface the wavelet reading opens in, and
`sinespec.plot_sine_stage(specification, levels)` puts measured sine
levels on the specification's stage. Both take `screenshot=`.

The Pulse Check reading has its call too:
`plot_pulse_check(history, shape, peak, duration, edge)` draws a
classical pulse on its tolerance envelope with its velocity change
beneath; see [checking a classical pulse](pulse-check.md).

The toggles that draw a reading over the data are an argument:

```python
history.plot(marks='filter')        # also 'averaging', 'shocks', 'truncation'
psd.plot(marks='octave', per_octave=3)   # the bands it would integrate to
psd.plot(marks='confidence')            # its chi-square band, 90 and 95 %
psd.plot_waterfall(marks='octave', screenshot='bands.png')   # on the stage
```

Each reading is taken from the object the same way the app takes it:
from the settings the object holds, or from the ones its detector
would suggest. In a script the marks are locked and cannot be dragged.
The confidence band is the exception: it is not a setting but what the
PSD's own average is worth, read from the degrees of freedom it carries
or from `dof=` when it carries none
([What an average is worth](averaging-confidence.md)), and it is drawn
flat only.

## Copying what is shown

Whenever something is drawn, the plot bar or the 3-D view's bar ends
with a copy button. It puts what the pane shows on the clipboard as an
image (the flat plot or the stage, whichever is up), in the current
theme and at the screen's own pixel density, for pasting into a
document, a chat or an email without taking a screenshot. Every table
in the application has the same button in its top-right corner while
the pointer is over it. A table is copied whole, headers included, as
cells for a spreadsheet and as a table for a document or an email. The
report's figures, tables and photographs have the same button
([Reports](reports.md#copying-a-figure-a-table-or-a-photo)).

## Saving an animation

While the 3-D view is animating something (a mode shape, an operating
deflection shape, a pair of modes picked on the MAC, a time history,
or a PSD envelope on its geometry), its bar has a second button after
Copy, *Save Animation*, with a strip of film on it. It saves the
animation as an MP4 video, exactly as the view is playing it: the same
camera, deflection scale and speed. A mode shape is saved as whole
cycles, at least six seconds of them, and the last frame stops one
step short of the first so that a player set to loop shows no jump. A
time history or an envelope is saved once through, at the speed it is
playing.

The video is H.264, which plays with nothing extra installed on macOS,
Windows, Linux, phones and in a browser, and gives small files for
this kind of picture: six seconds of a plate mode at 1024 × 768 was
1.4 MB. It is encoded through Qt by the operating system's own H.264
encoder. Every Mac and Windows computer has one. A Linux computer has
one only when its graphics driver provides it (VAAPI or NVENC), and
where there is none the button is not shown. From a script, the same
check is `visualdynamics.gui.movie.unavailable_reason()`, which returns
the reason or None. An installed copy can be tested with
`visualdynamics --check-movie out.mp4`, which writes one second of
video and reads it back.

## A plot from a script is the app's own pane

When a plot is shown rather than written to a file, it opens in the
same widget the application uses, with the same bar over it. A control
you would use in the window is therefore there in a script too, and
does not have to be passed in advance as an argument.

`geometry.plot()` returns a `ScenePane` opened on the geometry's
default view (`geometry.view`). *Reset View* returns to that view, and
the labeled axes and the orientation triad are toggles on its bar.
`frf.plot()` returns a `DataPane`. Complex data gets the component
box, so `frf.plot(component='imag')` sets the starting component
without fixing it, and you can switch to the real part without calling
again. Both panes expose what they wrap: `pane.plotter` is the PyVista
plotter and `pane.graphics` is the pyqtgraph layout.

The rest of the app's bar is left out because those controls only make
sense inside the window: in a script, filtering an FRF to its drive
points is done by choosing `records=`, and *Edit Fit* opens a fitting
session.

`show=False` returns the pane without starting an event loop, which is
what tests and notebooks need. Passing `path=` never creates a pane or
a toolbar: a bare plot widget is laid out off screen just long enough
to export the image.

## A figure of your own

When no one view draws the figure you need (two runs on one axis with
one dashed, a mode marked and named, panels side by side on one
scale), `visualdynamics.plot.compose` draws it in the same frame the
standalone plots use, with `as_series`, `figure_plots`, `to_view`
and `pin_range` to put the readings, marks and ranges where they
belong. See
[Composing a figure from a script](plot-composition.md).

## From a project

```python
project.plot('FRF')                     # dispatches on what it is
project.animate('Experimental Modes', mode=0)   # finds its geometry
project.animate_pair('Test Modes', 0, 'FEM Modes', 0, movie='pair.mp4')
                                        # each set on its own geometry, the
                                        # Basis solid and the other drawn
                                        # through it
```

## What is drawn, and why

The plots follow the same rules as the app. Records are grouped by
dimension, so mixed quantities are drawn on separate, comparable axes.
Axis labels carry units in the display system. Magnitudes in the
frequency domain are drawn on a log axis. Curves are peak-downsampled,
so a history of a million samples stays interactive. A complex FRF is
drawn as its magnitude unless you ask for another component; the
signed components (real, imaginary and phase) are drawn on a linear axis,
because a log axis cannot show a sign.

You choose how the frequency axis is drawn. By convention a shock
response spectrum opens in decades and everything else in hertz.
*Log f* on the plot bar, offered whenever what is drawn is over
frequency, switches every frequency plot, the 3-D stage and the
report's figures to decades, or back, for the rest of the session.
From a script it is `visualdynamics.frequency_axis('log')`, `'linear'`
or `'default'`. The choice is a viewing preference and not a property
of the data, so it is not saved with a project. A
controller's target starts at 0 Hz, which a log axis cannot draw, and
the status line says that the 0 Hz line is off the axis. On the 3-D
stage the decades are drawn the same way the flat plot draws them,
with a grid line and a label at every power of ten (`1`, `10¹`, `10²`
…) in place of the axis's even divisions.

Legends sit below the plot, in one horizontal row centered on the axes
and wrapping to their width, so they never cover the curves they name.

## The 3-D reading: a waterfall

With many channels on one axis, the curves hide each other exactly
where it matters: resonances line up, and the tenth curve lands on top
of the first nine. The 3-D reading is therefore the default. Selecting
data spreads its records along a depth axis, one curve per record
(a single record included), each labeled with its channel and colored
by level on one scale for the whole scene, so a channel four decades
quieter is drawn four decades darker. Data read on a log axis is
plotted as log10, and the vertical axis title says so. The *3D* button
on the plot bar switches to the flat plot instead, and the choice
stays either way. Views that mark the flat plot (the averaging and
shock frames, the animation cursor, the fitting screen) put it up
while they are in use and then go back to the reading you chose.

The camera is yours: redraws, unit changes and record picks happen
under the view you set, and the scene re-frames only when you look at
a different object. One vertical axis holds one quantity. A mixed
object draws its largest quantity group, and the status bar names the
quantities not shown; pick those records in the grid to see them.
Records are peak-decimated first, so a history of a million samples
arrives as the few thousand points that keep every peak. A scene holds
at most ten million vertices; when the records need more, they are
split into pages and the page box on the bar picks which page is on
the stage. Nothing is thinned further to make them fit.

From a script it is one call:

```python
psd.plot_waterfall()                          # a window of its own
psd.plot_waterfall(screenshot='stack.png')    # headless, to a file
```

## Two PSDs together: overlaid, or divided

Selecting two PSD objects offers three more readings on the bar.
*Overlaid* puts both on the 3-D axes, station by station where their
channels share a record, with the louder set colored by level and the
quieter one drawn behind it in gray. *Signal to noise (dB)* pairs them
where they match (same DOF, same quantity) and takes the louder as
signal plus noise and the quieter as the noise: the louder minus the
quieter, divided by the quieter, in decibels, line by line. *RMS signal
to noise* is the same ratio over the whole band, one bar per channel,
judged against a threshold that can be dragged. With the driven and
ambient densities of a system ID, these readings are the
signal-to-noise ratio. It is computed when you look at it and is not
stored as a third object. The
[system ID guide](workflows/sysid-workflow.md#how-the-signal-to-noise-is-computed)
gives the definition and its sources. *Overlaid* and the line-by-line
reading follow the *3D* toggle, and a pair with mixed quantities gets
a quantity box that names what to compare.

## Two FRFs together

Selecting two FRF objects (one survey measured twice, or two system ID
runs) pairs them on the same stage. They are matched channel by
channel, by response and reference; the one higher in the project tree
is colored by level and the other is drawn behind it in gray. Picking
the same record in each object's grid (click it in one and Cmd-click it
in the other; Ctrl-click on Windows and Linux) puts just those two
curves at one station. The *3D* toggle switches back to the flat
curves.

## A measurement and its resynthesis

Picking modes from a shape set together with measured FRFs asks the
modal model to predict those FRFs, and the pair is drawn on the same
kind of stage: each shared channel is a station, with the measurement
colored by level and the synthesis behind it in gray. This matches the
flat plot, which draws the synthesis dashed under the measurement.
Ninety-six FRFs and their ninety-six predictions on one axis are hard
to read; spread in depth, the fit of each channel can be read on its
own. The *3D* toggle switches back to the dashed flat overlay.

Plots are drawn in a display system, SI by default:

```python
from visualdynamics.units import SYSTEMS
frf.plot(unit_system=SYSTEMS['in-slinch-lbf-s'])
```

## Files

The extension picks the format: `.png` or `.svg`. 3-D scenes take
`screenshot=` in place of `path=`, because a scene renders through its
own plotter.

## Figures for print

A figure for print is laid out at its printed size, in logical pixels
(inches times 96), and written at a print resolution, so every font
prints at its set size:

```python
import visualdynamics.plot as vdplot

vdplot.EXPORT_DPI = 300                       # every standalone .png
psd.save_plot('psd.png', size=(328, 240))     # 3.42 in by 2.5 in

vdplot.save_image(layout, 'fig.png', dpi=300) # a layout of your own
```

`save_image` paints the laid-out plot scaled by dpi/96 (text, lines,
markers and layout together) and writes the resolution into the file.
The legend's text size is `vdplot.LEGEND_TEXT_SIZE`, 8 pt. On screen a
curve is one pixel wide (`vdplot.CURVE_WIDTH`), the width Qt draws
fastest. An image of a plot (a .png, a print figure, a copy to the
clipboard) draws every curve at least `vdplot.EXPORT_CURVE_WIDTH` (2)
wide, since it is drawn only once. On the light theme the curves use a
darker shade of each color, with a contrast of 4.5:1 or better against
white. A 3-D scene takes the same two numbers:

```python
from visualdynamics.viz.geometry import plot_dofs

plot_dofs(geometry, frf, 'acceleration', screenshot='dofs.png',
          size_in=(4.5, 3.25), dpi=300, bounds=False)
```

The window is set to the printed size, labels are sized in points, and
nodes, lines and the bounds box's labels scale with it. `bounds=False`
and `orientation=False` leave out the box and the corner triad.

## One figure as an interactive page

`visualdynamics.export_html` writes one figure, the same one a report
would draw, as a single HTML file that opens offline. A plot keeps its
zoom, pan and readout, and a scene can be turned and animated. The
page fills whatever frame holds it, such as an iframe or a box on a
slide:

```python
import visualdynamics

visualdynamics.export_html('spec.html', psd, specification=spec,
                           channel='101Z+', theme='light')
visualdynamics.export_html('modes.html', geometry=geometry, shapes=modes)
visualdynamics.export_html('dofs.html', geometry=geometry,
                           dofs=('acceleration', frf))
```

`theme` fixes the figure's colors; without it, the figure follows the
reader's theme. `fill=False` lays the figure out the way a report page
does.
