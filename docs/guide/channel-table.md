# The channel table

Every test carries one: the list of what was plugged in where, what it
measures, and what its calibration says. Visual Dynamics treats it as
what it is — a spreadsheet — so it imports from and exports to `.xlsx`,
and behaves like every other table in the window.

What it is *not* is a free-form sheet. Every column says what it holds,
which is what lets a column have rules about its own contents: a
direction can be a drop-down because there are exactly twelve of them,
a unit can be narrowed to what the channel measures, and an expiration
can offer a calendar.

![The channel table in the window: fifteen typed columns, roles read
from the run, the control check boxes, and the units the channels were
imported with](images/channel-table.png)

## The columns

| column | holds |
|---|---|
| **Channel** | the number the hardware reports. Positive, whole, and unique — it is how a row is identified, so two rows cannot claim one number |
| **Node** | the node it sits on. A whole number, or blank when nobody wrote it down |
| **Direction** | one of `X+ X- Y+ Y- Z+ Z-` and the six rotations |
| **Role** | `reference` (it drives), `response` (it answers), or `monitor` (recorded, judged as neither) |
| **Control** | whether the controller controlled to it. A check box |
| **Channel Type** | what it measures: acceleration, velocity, displacement, force, pressure, strain, voltage, temperature |
| **Unit** | the engineering unit, narrowed to ones the type could be in |
| **Sensitivity (mV/Unit)** | millivolts per whatever the Unit cell holds — the two are one calibration statement, which is why the header says so |
| **Range (V)** | the instrumentation's voltage limit, whole volts, defaulted to the ±5 V most hardware has |
| **Serial Number, Make, Model** | the sensor's identity, as text |
| **Triax DOF** | `X`, `Y` or `Z` — which leg of a triaxial accelerometer this is. Not a direction; the Direction column says where it points |
| **Expiration** | when its calibration runs out, with a calendar to pick from |
| **Comment** | anything else worth writing down |

Anything else a source carries is dropped. A Rattlesnake save brings
its coupling, excitation and feedback wiring along — how the controller
ran, rather than what was measured — and carrying columns nothing
interprets made this object a place data went to be ignored.

That is only safe because what *does* belong is recognized however it
was spelled. Case, spaces, hyphens, a trailing colon and a parenthetical
unit are all tidied away first, then the words that genuinely differ are
mapped: `Cal Due` is the expiration, `Point` is the node, `SN` and
`Serial Number` are the same column, and a calibration lab's
`Sensitivity (mV/g)` lands where you would expect.

## What a channel measures

**Type** is the quantity the channel measures, and **Unit** is what it
is measured in. A source states the type in its own words, and the
import is lenient about the spelling: a word the schema does not know
is dropped rather than refused. When that leaves the type blank, the
unit answers for it — `G` is an acceleration, `V` a voltage, `N` a
force — so a controller that writes `Accel`, or nothing at all, still
comes in typed. A type the file does state is never overruled: a
disagreement between the two is yours to resolve, and the table
refuses to create one when you edit a cell. A unitless unit says
nothing, because strain and a bare ratio share a dimensionality,
which is the whole reason they are two names.

## Direction columns from the geometry

Beside a geometry the table shows five columns it derives rather
than holds: each channel's measured direction as a unit vector in
the geometry's global system (**Unit X**, **Unit Y**, **Unit Z**),
the global axis it is nearest (**Nearest Axis**, signed: `X+`,
`Z-`), and the angle between the two (**Angle to Axis (deg)**). The
direction is the row's node and direction read through the frame
that node is measured in, so a channel mounted in a turned local
frame reads as the global vector it really is. Edit the node or the
direction and the five restate at once; a node the geometry lacks
leaves them blank. They are the geometry's to say, so they are not
editable and do not travel to Excel, but the report's instrumentation
table carries them beside the basis geometry and a script reads the
same rows from `visualdynamics.core.tables.table_of(table, geometry)`.

## Role and Control

Role is one value rather than three check boxes, so the one nonsense
state cannot be written: a channel is not both the input and the output
of the same estimation. Blank means undeclared, exactly as an
undeclared unit does.

**A control channel is a response**, and that is enforced from both
directions — checking Control on a monitor is refused, and so is
changing a control channel's role out from under its check. Both
refusals say why, in the status bar.

A controller run declares these for you. A channel with a feedback
device is a drive, which is Rattlesnake's own rule, so it imports as a
reference; every other enabled channel imports as a response. And a
random, transient or sine-sweep run states its control channels
through its specification — the requirement is written per control
channel — so
those arrive already checked.

## Two strictnesses

They are deliberately different, and it is worth knowing which you are
meeting:

- **A file is read leniently.** A value that will not parse is blanked
  rather than refusing the whole import. Refusing would leave you
  nothing to fix, and this table is where you would fix it.
- **An edit is refused.** Once you are typing, an invalid value is
  rejected at entry with the reason, never written and complained about
  later.

So a spreadsheet full of rubbish still imports, with the bad cells
empty and waiting; typing the same rubbish into one of them does not.

## Merging tables

A survey acquired in passes brings one channel table per pass, and
`project.merge` combines them (see
[Projects](projects.md#merging)). Two rows are the *same channel*
when everything but the channel number matches — the number is the
hardware's bookkeeping, not the channel's identity — and the merged
table renumbers 1..N. Rows that agree on point and role but disagree
elsewhere refuse, naming the point: keeping either description
silently would silently drop the other.

## Round-tripping through Excel

*Export…* writes an `.xlsx` a calibration lab can fill in, and the
constraints travel with it. Every column that limits itself carries
Excel's own data validation — Direction, Role, Control, Channel Type,
Triax DOF — so Excel refuses a bad direction there for the same reason
the window does here.

The unit list follows the channel's type. Excel cannot hold a different
list per row, but it can look one up: the units for each quantity sit
on a hidden sheet as named ranges, and the rule is `INDIRECT` of the
Channel Type cell beside it. Declare a channel a voltage and its unit
cell offers V and mV; leave the type blank and it offers all of them.

Channel and node are written as numbers, so Excel does not flag every
cell with "number stored as text", and a blank node stays blank rather
than becoming a zero.

```python
project.channel_table.dof_strings()      # ['101Z+', '107Z+', ...]
project.channel_table.roles()            # ['response', 'reference', ...]
project.channel_table.controls()         # a boolean per channel
project.channel_table.sensitivities()    # mV/EU, NaN where undeclared
project.channel_table.units_for(0)       # what the first row's channel could be in

visualdynamics.export_file(table, 'channels.xlsx', format='excel')
table = visualdynamics.import_file('channels.xlsx')
```
