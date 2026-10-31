# Modal responses from forces

A shape set is a model of the structure: its modes, with a frequency,
a damping ratio and a modal mass for each. Given the forces applied to
it, the response of each mode can be computed directly, and the
response at any DOF follows from those modal responses as u = Φq. For
a large finite element model this is much smaller than the physical
response, since there are far fewer modes in a bandwidth than DOFs, so
Visual Dynamics computes and keeps the modal responses.

## What to select

Select a time history that holds force channels together with a shape
set, and press **Compute Modal Responses** on the bar. The modes used
are the ones selected:

- the shape set itself in the tree for every mode in it;
- modes picked in the set's grid for only those modes.

The time history's force channels are the excitation. The shape set
has to hold real modes, and its DOFs have to include the DOFs the
forces are at.

## How it is computed

The physical forces are made modal first,

    pᵣ = Σₖ φₖᵣ fₖ,

summed over the force channels. Channels that already hold modal forces
(DOFs `M1`, `M2` and so on, as **Transform to Modal Coordinates**
makes them from a force record) are used as they are. Each mode then
answers its own equation,

    q̈ᵣ + 2ζᵣωᵣ q̇ᵣ + ωᵣ² qᵣ = pᵣ(t) / mᵣ,

with ωᵣ, ζᵣ and mᵣ from the set (mᵣ is one for a mass-normalized set).
The equation is solved exactly for a force that changes linearly
between samples, starting from rest at the first sample. There is no
other approximation: a transient, the start of the record and a
rigid-body mode (ωᵣ = 0) are all computed the same way. Under a net
force, a rigid-body mode's displacement grows as F·t²/2m, which is the
correct answer.

## The pane

The pane beside the plot has one setting, the **Response** kept for
each mode: displacement, velocity or acceleration (the default). Under
it are the modes used, the force channels, the number of samples and
the highest mode's frequency. A note says when the record or the set
cannot be used, for example a set with no mode at a force DOF, or a
record whose samples are not evenly spaced.

The plot previews the modal responses over the first 65,536 samples of
the record while the setting changes.

**Compute** makes a new time history, named for the forces' time
history followed by *Modal Responses* (`Run 1 Modal Responses`), and
links it into the same group. It holds one record per mode, named
`M1` … `Mn` by the mode's number in its set, on the same time axis and
with the same averaging frames and shock windows. Each record's comment
names the set, the quantity and the mode's frequency. The time history
of forces is not changed. Computing again makes another object, and
the first one is kept.

To keep the forces and the responses in one time history, select both
and press **Merge**. A merged history works everywhere the two separate
objects do: it plays on the geometry and expands to physical responses
the same way.

## Playing them on the geometry

Select the modal responses together with the geometry, and they play
on it: at each sample, every mode's shape is scaled by
its modal response at that sample, and the shapes are added. The
physical response at every DOF is never computed or stored, and a
frame's cost is set by the number of modes and of nodes that move.
For a model with a million DOFs and fifty modes, a frame takes about
7 ms, where expanding a ten-thousand-sample record first would take
80 GB.

The shape set is the one the responses were computed through. For modal
coordinates with no set recorded (an imported record, say), select the
set as well. Pick some of the modal records in the history's grid to
play only those modes' contribution. When the forces and the responses
have been merged into one history, the quantity box above the view
switches between the forces and the modal responses; it opens on
whichever has more records.

## The physical response

Select the modal responses and the shape set together and press
**Expand to Physical Responses**: the modal responses are expanded,
u = Φq, at every DOF the set covers, in a new time history beside the
set's geometry. In a merged history only the modal records are read;
the force channels are not part of the expansion. To see one mode's
contribution, pick its record in the grid instead of the whole object.
The forces' time history and the shape set together also offer
**Transform to Modal Coordinates**, which only projects the force
channels into modal forces and uses none of the set's frequencies,
damping or masses.

## From a force CPSD

A CPSD of forces works the same way. Select it together with a shape
set and press **Compute Modal Responses**. The forces are made modal as
a matrix, S_pp = Φᵀ S_ff Φ, and each mode answers through its own
frequency response,

    Hᵣ(ω) = (iω)ᵖ / [mᵣ (ωᵣ² − ω² + 2i ζᵣ ωᵣ ω)],

with p = 0, 1 or 2 for displacement, velocity or acceleration. The
modal response CPSD is then

    S_qq,rs = conj(Hᵣ) S_pp,rs Hₛ,

every mode against every other, on the CPSD's own frequency lines. The
(iω)ᵖ is inside the modal term, so a rigid-body mode's acceleration at
0 Hz is finite (its displacement and velocity there are left blank). A
modal force CPSD, as **Transform to Modal Coordinates** makes it, is
used as it is.

The CPSD needs every cross term between the drives, because the forces
are combined as a matrix. Cross terms stored as zero are fine: that is
the case of drives that are not correlated. A density that holds only
autospectra at several drives is refused, as Transform refuses it. A
density on octave bands is refused too, since a mode's response
changes across a band, and so is a density written at breakpoints,
since the modes between two breakpoints would be missed. *Convert to
CPSD* on a breakpoint specification reads it onto lines at a spacing
you set.

The pane shows the number of frequency lines instead of samples, and
the plot previews each mode's response autospectrum. **Compute** makes
a CPSD named for the forces followed by *Modal Responses*. **Expand to
Physical Responses** on it and the shape set gives the response CPSD
at every DOF the set covers, S_xx = Φ S_qq Φᵀ. This is the same as
driving the FRFs synthesized from the set with the force CPSD, cross
terms and phase included.

A modal force CPSD is not offered **Expand to Physical Responses**,
since many force distributions give the same modal force.

## Units

Through a mass-normalized set (one with a mass unit) the modal
coordinates carry a half power of mass, the same modal dimensions
**Transform to Modal Coordinates** gives: a modal displacement is in
m·kg^½, for example. Through a set with no mass unit, the responses
are left with an undeclared unit and a hint of what they are.

## From a script

    import visualdynamics

    project = visualdynamics.Project.open('test.vdyn')
    made = project.compute_modal_responses('Run 1', 'FE Modes',
                                           quantity='displacement',
                                           modes=[0, 1, 2, 3])

returns the new object's name, `'Run 1 Modal Responses'`. `modes`
counts from zero. The calculation is in `core.modal_response`:
`modal_forces` for the projection and `modal_responses` for the
responses as a time history; `modal_force_density` and
`modal_response_density` for a CPSD.

    project.expand(made, 'FE Modes')

expands every modal record, and `records=` picks some of them, by
their index in the object.

    project.animate(made, screenshot='modes.png')

plays the modal records on the set's geometry, the same way the window
does; `TimeHistory.animate(geometry, shapes=...)` does it for a record
and a set held outside a project.

    project.merge('Run 1', made)

puts the forces and the responses back in one time history, named
`Run 1`.
