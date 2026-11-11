# Modal FRFs from a shape set

A shape set holds each mode's frequency, damping ratio and modal mass,
and that is enough to write each mode's own frequency response: the
response of its modal coordinate to a unit modal force. These are the
modal FRFs. Through the shapes at two DOFs they give the physical FRF
between them, so a finite element model's modes and a test's channel
table give the FRFs between the test's references and responses before
the test is run.

## Synthesizing modal FRFs

Select a shape set, or pick some of its modes in its grid, and press
**Synthesize Modal FRFs** on the bar. Each mode's FRF is

    Hᵣ(ω) = (iω)ᵖ / [mᵣ (ωᵣ² − ω² + 2i ζᵣ ωᵣ ω)],

with p = 0, 1 or 2 for displacement, velocity or acceleration per modal
force. The (iω)ᵖ is inside the modal term, so a rigid-body mode's
acceleration at 0 Hz is 1/mᵣ; its displacement and velocity there are
unbounded and are left blank.

The pane beside the plot has three settings:

- **Sample rate** and **Frame**: the lines are those of a test frame of
  that many samples at that rate, every sample rate / frame Hz from
  0 Hz to the Nyquist frequency. They are the lines Compute FRFs gives
  for a record framed the same way, so the synthesized FRFs line up
  with measured ones. The pane opens at 8192 Hz and 8192 samples, 1 Hz
  lines, and keeps what you set for the next time.
- **Response**: displacement, velocity or acceleration (the default).

Under them are the modes used, the line spacing, the number of lines,
the Nyquist frequency and the highest mode. A note says how many modes
are above the Nyquist frequency. They are kept, because a mode above
the band still adds its stiffness below it. The plot previews the modal
FRFs.

**Generate** makes a new FRF named for the set followed by *Modal FRFs*
(`FE Modes Modal FRFs`), linked beside the set. It holds one record per
mode, `M1` against `M1`, `M2` against `M2` and so on, numbered by the
mode's place in its set. Each record's comment names the set, the
response and the mode's frequency. Only the modes' frequencies, damping
and modal masses are read, so a change to any of them marks the modal
FRFs for recomputing, and a change to the shapes alone does not.

## FRFs between a channel table's channels

Select the modal FRFs together with a channel table and press **Expand
to Physical Responses**. Each response and monitor channel is answered
against each reference channel,

    Hⱼₖ(ω) = Σᵣ φⱼᵣ Hᵣ(ω) φₖᵣ,

the modal sum, with the shapes read at the channels' DOFs. A channel
read negative (`X-`) gets the negative of its shape. To use only some
modes, pick their records in the modal FRFs' grid first and then
Cmd-click the table.

Each row answers in what its channel measures. Modal FRFs in
acceleration give a velocity channel's row by dividing by iω, and a
displacement channel's by dividing by (iω)². Turned down that way, the
0 Hz line divides by zero and is left blank. Turned up, from
displacement to acceleration, every line has a value. To have a value
at 0 Hz for every channel, synthesize the modal FRFs in the lowest
response the table measures.

Where the shapes come from depends on the geometries:

- When the shape set and the channel table are linked to the same
  geometry, or either has none, the set's own DOFs are read.
- When the set is linked to a finite element geometry and the table to
  a test geometry, the shapes are taken at each channel from the
  nearest model node, the way **Project onto Basis DOFs** takes them,
  including each node's coordinate system. The **Tolerance** says how
  far that node may be, as a share of the test geometry's size, and is
  only editable in this case. A channel with no node close enough is
  dropped.

The pane shows the modes used, the number of references, responses and
FRFs, how the shapes were taken (with the nodes matched and the worst
distance when they were projected) and the channels dropped. A note
names the channels left out because they measure something other than
a force (a reference) or a motion (a response), and the rows that have
no value at 0 Hz. The plot previews the drive points: each response on
a reference's DOF, or the first response against each reference when
none shares one.

**Expand** makes a new FRF named for the modal FRFs, `@` and the table
(`FE Modes Modal FRFs @ Channels`), linked into the table's group. Its
`transform_report` says which set the modal FRFs came through, what
was dropped or skipped and, when the shapes were projected, the
projection's report. Changing the modal FRFs or the table's channels
marks it for recomputing.

The modal FRFs and the shape set together also offer **Expand to
Physical Responses**, which gives the FRFs between every pair of the
set's DOFs. For a finite element model that is a great many records;
the channel table is the way to ask for the few that matter.

## Units

Through a mass-normalized set the modal displacement is in m·kg½ and
the modal force in N/kg½, so a modal FRF is in s² for displacement, s
for velocity, and has no unit for acceleration. In any coherent unit
system the numbers are the same as in SI, since the second is the time
unit in each. A mass-normalized mode's accelerance tends to one above
its frequency. A set with no mass unit gives modal FRFs with no unit,
and the FRFs at the channels follow.

## From a script

    import visualdynamics

    project = visualdynamics.Project.open('model.vdyn')
    modal = project.synthesize_modal_frfs('FE Modes', sample_rate=4096.0,
                                          frame_length=8192,
                                          quantity='acceleration')
    frfs = project.expand(modal, 'Test Channels')

`modes=` limits the synthesis to some modes, counting from zero, and
`records=` limits the expansion to some of the modal FRFs' records.
`tolerance=` sets how far a channel's nearest node may be when the
shapes are projected. `project.channel_frfs(modal, 'Test Channels')`
returns the FRFs without adding them to the project, and
`project.channel_shapes('FE Modes', 'Test Channels')` the shapes at the
channels. The calculation is in `core.modal_frf`:
`synthesize_modal_frfs` for the modal FRFs, `modal_matrix` for the
matrix a modal FRF's records hold, and `frfs_at_channels` for the
expansion.
