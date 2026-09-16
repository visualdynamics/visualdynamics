# Plotting

Every plot the app draws has a call, and every call takes `path=` to
render to a file instead of opening a window. One renderer serves
both, so a figure in a script is the figure on the screen.

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

frf.animate(geometry)                   # the operating deflection shape,
                                        # at the strongest line
frf.animate(geometry, frequency=647.0)  # or at a chosen frequency

psd.animate(geometry)                   # the envelope: two copies at
                                        # +/- sqrt(PSD), no phase claimed
psd.animate(geometry, frequency=1000.0, quantity='acceleration')
```

An FRF (or a complex spectrum) selected beside a geometry in the app
animates the same thing: the records' magnitude and phase at one
frequency line, swept like a complex mode. The cursor on the plot picks
the line — it starts on the strongest one — and Play sweeps the phase.

A PSD has no phase, so it gets the *envelope* instead: two translucent
copies of the geometry deflected `+sqrt(PSD)` and `-sqrt(PSD)` at the
cursor line — every extreme every channel reaches, with no claim about
when. Size shows each line at full scale; colour is absolute, dB below
the loudest node at any line, so a quiet line shows its shape but wears
its quietness. Play walks the cursor up the spectrum. One quantity
deflects at a time (the toolbar box picks), and a node measured along
two axes reaches their in-phase diagonal, which two copies cannot
avoid claiming.

A CPSD holds more: its cross records carry each channel's phase
relative to the others, so a whole CPSD beside a geometry animates the
**principal operating deflection shape** — the dominant eigenvector of
the cross-spectral matrix at each line, the direction of the output
spectra's own CMIF, no reference to choose. At a well-excited
resonance it is the same shape the FRF ODS shows, read from operating
data alone; where a mode is weakly excited the dominant eigenvector
honestly belongs to whatever carries more power there. Picking a
reference column in the grid reads the shape *relative to that one
channel* instead, and picking autos alone falls back to the envelope
— phase relative to nothing is no phase at all.

```python

coherence.plot_map()                    # frequency across, channel down
geometry.plot()                         # the 3D scene, with its bar
geometry.plot_dofs(frf, 'force')        # labelled DOF arrows
```

The readings on a time history's bar have calls of their own, in
`visualdynamics.plot`: `plot_scalogram(history)` (the wavelet reading,
drawn at a picture's width whatever the record's length — the
[wavelet guide](wavelet.md) says how), `plot_kurtosis(history)`, and
for a specification comparison `plot_bars(measured, specification)`,
`plot_comparison(...)`, `plot_ratio(...)` and `plot_replication(...)`
— each the figure the app shows, each taking `path=`.

## A plot from a script is the app's own pane

Shown rather than written to a file, a plot comes up in the same widget
the application uses, with the same bar over it — so a control you would
reach for in the window is there in a script too, and does not have to
be known about in advance as an argument.

`geometry.plot()` returns a `ScenePane`: the labelled axes and the
orientation triad are toggles on its bar. `frf.plot()` returns a
`DataPane`, and complex data gets the component box, so
`frf.plot(component='imag')` sets where it starts rather than fixing it
— you can switch to the real part without calling again. Both expose
what they wrap: `pane.plotter` is the PyVista plotter, `pane.graphics`
the pyqtgraph layout.

The rest of the app's bar is absent because it is not a live option
outside the window: filtering an FRF to its drive points is a record
selection, and Edit Fit opens a fitting session.

`show=False` returns the pane without starting an event loop, which is
what tests and notebooks want. Passing `path=` never comes near a pane
or a toolbar: a bare plot widget is laid out off screen just long
enough to export the image.

## From a project

```python
project.plot('FRF')                     # dispatches on what it is
project.animate('Experimental Modes', mode=0)   # finds its geometry
```

## What is drawn, and why

The reading rules are the app's own: records grouped by dimension so
mixed quantities land on comparable axes, unit-aware labels in the
display system, log magnitude in the frequency domain, curves
peak-downsampled so a million-sample history stays interactive. A
complex FRF reads as magnitude unless you ask for a component; the
signed ones go linear, because a signed quantity on a log axis is a
lie.

The frequency axis is the viewer's to choose. By convention a
shock response spectrum reads in decades and everything else in
hertz, and that is how each opens; **Log f** on the plot bar, offered
whenever what is drawn is over frequency, switches every frequency
plot, the 3-D stage and the report's figures to decades, or back,
for the session (`visualdynamics.frequency_axis('log')`,
`'linear'` or `'default'` from a script). It is a habit rather than
a property of the data, so it is not saved with a project. A
controller's target starts at 0 Hz, which no log axis can draw; the
status line says the line is off the axis rather than letting it
vanish. On the 3-D stage the decades are drawn the way the flat plot
draws them — a grid line and a label at every power of ten, `1`,
`10¹`, `10²` … — in place of the axis's even divisions.

Legends sit **below** the plot, one horizontal row centred on the
axes and wrapping to their width, rather than floating over the
curves — a legend inside the axes hides the data it names, and a long
one hid most of it.

## The 3-D reading: a waterfall

Many channels on one axis hide each other exactly where it matters —
resonances line up, and the tenth curve lands on the first nine. So
the 3-D reading is the **default**: selecting data spreads its records
along a depth axis, one curve per record — a single record included —
labelled with the channel it is, coloured by level on **one scale for
the whole scene**, so a channel four decades quieter draws four
decades darker. Log-read data plots as log10 and the vertical axis
title says so. The **3D** button on the plot bar is how the flat plot
is asked for instead; the choice sticks either way. Views that mark
the flat plot — the averaging and shock frames, the animation cursor,
the fitting screen — put it up while they are in use and hand back to
the reading you chose.

The scene follows the app's standing rules. The camera is yours —
redraws, unit switches and record picks happen under the view you
set, and it re-frames only when you look at a different object. One
vertical axis holds one quantity: a mixed object draws its largest
quantity group and the status bar names what waits; pick the other
records in the grid to see them. Records are peak-decimated first,
so a million-sample history arrives as the few thousand points that
keep every peak.

From a script it is one call:

```python
psd.plot_waterfall()                          # a window of its own
psd.plot_waterfall(screenshot='stack.png')    # headless, to a file
```

## Two PSDs together: overlaid, or divided

Selecting two PSD objects offers two more readings on the bar. The
**overlay** stages both on the 3-D axes, station by station where
their channels share a record, the louder set coloured by level and
the quieter drawn behind it in grey. The **ratio** divides them where
they match — same DOF, same quantity — and draws the quotient in
decibels; with the driven and ambient densities of a system ID that
reading *is* the signal-to-noise, computed where it is looked at
rather than stored as a third object. The louder set is the
numerator, so the healthy reading is positive. Both honour the 2D/3D
toggle, and a mixed pair offers a quantity box naming what to
compare.

## A measurement and its resynthesis

Picking modes from a shape set beside measured FRFs asks the modal
model to predict them, and the pair reads on that same stage: each
shared channel a station, the measurement coloured by level and the
synthesis behind it in grey — the emphasis the flat plot gives them
when it draws the synthesis dashed under the measurement. Ninety-six
FRFs and their ninety-six predictions on one axis is a band with the
answer buried in it; given depth, each channel's fit can be read on
its own. The 2D/3D toggle hands back to the dashed flat overlay.

Plots draw in a display system, SI by default:

```python
from visualdynamics.units import SYSTEMS
frf.plot(unit_system=SYSTEMS['in-slinch-lbf-s'])
```

## Files

`.png` or `.svg` — the extension picks the exporter. 3D scenes use
`screenshot=` rather than `path=`, because a scene renders through
its own plotter.
