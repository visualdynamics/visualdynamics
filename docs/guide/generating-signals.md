# Generating reference signals

A test is planned before it is run, and a response is predicted before
there is a record to read it from. Both start from the signals the
references will carry: random shaped to a specification, bursts sized
to a capture, an impulse or a shock pulse, a sine sweep. Generate
Signal makes them as time histories at the reference DOFs, in the
quantity they are (a force, a voltage, an acceleration), beside the
object they were made from.

## What a signal is made from

Select one object and press **Generate Signal** on the bar. The pane
beside the plot offers only the kinds that object makes.

| Selected | Kinds | At |
|---|---|---|
| A specification or a PSD | random, burst random, periodic random | its DOFs, drawn to its levels and cross terms |
| A sine specification | sine | its control DOFs, following its tone |
| A shock specification | SRS synthesis | its DOFs, a sum of wavelets meeting its spectrum, checked against the shaker's shock ratings ([Synthesizing a shock to its spectrum](shock-synthesis.md)) |
| A channel table | random, burst random, periodic random, impulse, classical pulse | its reference channels, the random kinds to a flat band and level typed in the pane |

A channel table's reference channels are the rows whose role is
reference, and they have to say what they measure (the channel type),
all the same quantity: a signal is one quantity.
The level and the peak typed for a channel table are in each
reference channel's own unit, as the table writes it: a level of 1.0
is 1 lbf²/Hz on a channel in lbf and 1 N²/Hz on a channel in N. A
channel with no unit is read in SI. The pane shows the unit beside
the box when the channels share one.

## The kinds

Random is stationary and Gaussian, with the target's PSD at every
DOF and, with Cross terms ticked, the coherence and phase its cross
terms state. It is built a frame at a time in the frequency domain,
each line at the target's level with a random phase, and the frames
overlap by half under a half-cycle sine window, so the level is the
same from the first sample to the last, there is no seam where frames
meet, and the record never repeats. The frame sets the line spacing
the target is drawn at; Auto picks lines no coarser than 1 Hz or a
quarter of the lowest frequency written.

Burst random is the same signal gated on for a share of each frame
(On) and quiet for the rest, with a cosine ramp at each end of the
burst (Taper). Each frame holds one burst from its start, so a
capture of the frame's length holds the burst and the response dying
away. The level while a burst is on is the target's; averaged over the
whole frame, the PSD is the target times the share on.

Periodic random repeats one random frame. Read with a rectangular
window at the frame's length, every line is exactly the target's and
nothing leaks between them. Realization draws a new frame every so
many periods; All keeps one frame throughout (pseudo-random).

An impulse is a half-sine of a given Peak and Width, starting at
Starts, the shape of the force a hammer tip makes. Its width sets the
band it excites. The spectrum of a half-sine of width T is flat at low
frequency, falls 3 dB by 0.59/T and 10 dB by 1.02/T, and reaches its
first zero at 1.5/T, so a 1 ms impulse excites up to about 1 kHz and a
0.25 ms impulse up to about 4 kHz. The pane gives the three frequencies
for the width typed, and the plot draws the impulse's spectrum under
its trace, as the record holds it: the magnitude of its Fourier
transform, in the quantity times seconds (N·s for a force), whose level
at low frequency is the impulse's area, 2·peak·width/π. The width has to
span at least two samples. A classical pulse is a half-sine, a
terminal-peak sawtooth or a trapezoid, at its Peak and Duration,
starting at Starts.

Sine follows a sine specification's tone: its frequency at the
tone's rates, dwells included, and its amplitude and phase between
breakpoints as the controller follows them.

An SRS synthesis is iterated whole, since a shorter one would meet a
different spectrum, and off the window's thread once the settings rest;
[Synthesizing a shock to its spectrum](shock-synthesis.md) says how it
is made and read.

## Read the way it was made

A generated signal carries the analysis settings it was made for, and
its averaging opens on them. Random is read in frames of the length it was drawn at, under a
Hann window, overlapped by half. Burst random is read in one rectangular
frame per burst, and periodic random in one per period, with no
overlap: 50 bursts are 50 averages, each leakage-free. An impulse or a
pulse is the whole record in one rectangular frame. A sine has no
frames of its own and opens with the usual defaults.

## Preview, then Generate

The plot previews the signal as its settings change: for the random
kinds, the preview's PSD over the target; for an impulse, its trace and
its spectrum; for the others, its trace.
The preview is a short stretch of the signal, up to 131,072 samples
per DOF, so that it answers while a number is being typed. The rows
under the settings give the samples the full signal holds, its line
spacing and the preview's rms.

A setting the signal cannot have is said in the pane, not raised: a
band at or above the Nyquist frequency names the rate it needs, and a
target no set of signals can have (coherences that cannot all hold
together) names the first line where it fails.

**Generate** makes the full signal as a time history in the source's
group. Its settings are kept with it, and the random seed with them:
Seed set to Drawn picks one on Generate and keeps it, so the
same record can be made again. When the source changes (a level, a
tone, a reference channel), the signal is marked as computed from
something that has since changed, and Recompute makes it again from the
same settings and seed.

## From a script

    import visualdynamics

    project = visualdynamics.Project.open('test.vdyn')
    drives = project.generate_signal('Random Specification', 'random',
                                     duration=60.0, sample_rate=5120.0,
                                     seed=1)
    tap = project.generate_signal('Channel Table', 'impulse',
                                  duration=2.0, sample_rate=10240.0,
                                  at=0.1, peak=500.0, width=0.0005)

The settings are the fields of `core.signals.SignalSettings`, and the
generators behind each kind (`random_history`, `burst_random_history`,
`periodic_random_history`, `impulse_history`, `sine_history`) can be
called on their own, with `flat_spectrum` for a target typed in.
`impulse_rolloff(width)` gives an impulse's −3 dB, −10 dB and first-zero
frequencies, and `impulse_spectrum(history)` its spectrum.
