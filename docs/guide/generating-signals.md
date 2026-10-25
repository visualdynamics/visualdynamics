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
| A channel table | random, burst random, periodic random, impulse, classical pulse | its reference channels, the random kinds to a flat band and level typed in the pane |

A channel table's reference channels are the rows whose role is
reference, and they have to say what they measure (the channel type),
all the same quantity: a signal is one quantity.

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

An impulse is one sample whose integral is the Area, and its
spectrum is flat to the Nyquist frequency. A classical pulse is a
half-sine, a terminal-peak sawtooth or a trapezoid, at its Peak and
Duration, starting at Starts.

Sine follows a sine specification's tone: its frequency at the
tone's rates, dwells included, and its amplitude and phase between
breakpoints as the controller follows them.

## Preview, then Generate

The plot previews the signal as its settings change: for the random
kinds, the preview's PSD over the target; for the others, its trace.
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
                                  at=0.1, area=1.0)

The settings are the fields of `core.signals.SignalSettings`, and the
generators behind each kind (`random_history`, `burst_random_history`,
`periodic_random_history`, `impulse_history`, `sine_history`) can be
called on their own, with `flat_spectrum` for a target typed in.
