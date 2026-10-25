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

**Compute** appends one record per mode to the time history, named
`M1` … `Mn` by the mode's number in its set. Each record's comment
names the set, the quantity and the mode's frequency. Computing again
through the same set replaces the records the earlier run added, so a
history never holds two answers for one mode; records computed through
a different set are kept. The rest of the time history (its channels,
averaging and shock windows) is not changed.

## Playing them on the geometry

Select the time history together with the geometry, and the modal
responses play on it: at each sample, every mode's shape is scaled by
its modal response at that sample, and the shapes are added. The
physical response at every DOF is never computed or stored, and a
frame's cost is set by the number of modes and of nodes that move.
For a model with a million DOFs and fifty modes, a frame takes about
7 ms, where expanding a ten-thousand-sample record first would take
80 GB.

The shape set is the one the responses were computed through. For modal
coordinates with no set recorded (an imported record, say), select the
set as well. Pick some of the modal records in the history's grid to
play only those modes' contribution. When the history holds force
channels too, the quantity box above the view switches between the
forces and the modal responses; it opens on whichever has more records.

## The physical response

Select the time history and the shape set together and press **Expand
to Physical Responses**: the modal responses are expanded, u = Φq, at
every DOF the set covers, in a new time history beside the set's
geometry. Only the modal records are read; the force channels are not
part of the expansion. To see one mode's contribution, pick its record
in the time history's grid instead of the whole history. The same
selection also offers **Transform to Modal Coordinates**, which only
projects the force channels into modal forces and uses none of the
set's frequencies, damping or masses.

## Units

Through a mass-normalized set (one with a mass unit) the modal
coordinates carry a half power of mass, the same modal dimensions
**Transform to Modal Coordinates** gives: a modal displacement is in
m·kg^½, for example. Through a set with no mass unit, the responses
are left with an undeclared unit and a hint of what they are.

## From a script

    import visualdynamics

    project = visualdynamics.Project.open('test.vdyn')
    project.compute_modal_responses('Run 1', 'FE Modes',
                                    quantity='displacement',
                                    modes=[0, 1, 2, 3])

`modes` counts from zero. The calculation is in
`core.modal_response`: `modal_forces` for the projection,
`modal_responses` for the responses as a separate time history, and
`appended` to add them to a record.

    project.expand('Run 1', 'FE Modes')

expands every modal record of the history, and `records=` picks
some of them, by their index in the history.

    project.animate('Run 1', screenshot='modes.png')

plays the modal records on the set's geometry, the same way the window
does; `TimeHistory.animate(geometry, shapes=...)` does it for a record
and a set held outside a project.
