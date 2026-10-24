# The channel table

Every test has one: the list of what was plugged in where, what each
channel measures, and what its calibration says. Visual Dynamics treats
it as a spreadsheet, so it imports from and exports to `.xlsx` and
works like every other table in the window.

It is not a free-form sheet, though. Every column has a fixed type,
which lets each column have rules about its contents: a direction can
be a drop-down because there are exactly twelve of them, a unit can be
limited to what the channel measures, and an expiration can offer a
calendar.

![The channel table in the window: fifteen typed columns, roles read
from the run, the control check boxes, and the units the channels were
imported with](images/channel-table.png)

## Starting one by hand

A table usually arrives with a run or from a spreadsheet, but it can
also be typed in. *New Channel Table*, on the project row's bar, adds
an empty one. *Add Channel*, on the table's own bar, adds a blank row
numbered after the highest channel in the table, and puts the cursor
on it to fill in. From a script, `project.new_channel_table()` and
`project.add_channels(name, count)` do the same.

Selected together with a geometry, the table is drawn on it as arrows,
one per channel, picked by role from a drop-down on the 3-D view's bar
([the project tree](project-tree.md#a-channel-table-on-a-geometry));
`geometry.plot_channels(table, role)` draws the same from a script.

## The columns

| column | holds |
|---|---|
| Channel | the number the hardware reports. A positive whole number, unique in the table, since it identifies the row |
| Node | the node the sensor is on. A whole number, or blank when nobody wrote it down |
| Direction | one of `X+ X- Y+ Y- Z+ Z-` or the six rotations |
| Role | `reference` (it drives), `response` (it answers), or `monitor` (recorded, but neither). When the table is linked with the time data it describes, this column shows the data's roles, and an edit here changes the data's role; see [the tree](project-tree.md) |
| Control | whether the controller controlled to it. A check box |
| Channel Type | what it measures: acceleration, velocity, displacement, force, pressure, strain, voltage or temperature |
| Unit | the engineering unit, limited to the units the type can be in |
| Sensitivity (mV/Unit) | millivolts per whatever the Unit cell holds, greater than zero. The two cells are one calibration statement, which is why the header names both |
| Range (V) | the instrumentation's voltage limit, in whole volts, 5 V unless stated. Over the sensitivity it is the channel's range in its unit, which the *Overload* reading of a linked time history reads every channel against ([checking each channel against its range](overload.md)) |
| Serial Number, Make, Model | the sensor's identity, as text |
| Triax DOF | `X`, `Y` or `Z`: which leg of a triaxial accelerometer this is. This is not a direction; the Direction column says where it points |
| Expiration | when its calibration runs out, with a calendar to pick from |
| Comment | anything else worth writing down |

Any other column a source carries is dropped. A Rattlesnake save
includes its coupling, excitation and feedback wiring, which describe
how the controller ran rather than what was measured, and nothing in
Visual Dynamics reads them.

Dropping other columns works because the columns that do belong are
recognized however they are spelled. Case, spaces, hyphens, a trailing
colon and a unit in parentheses are removed first, and then the words
that actually differ are mapped: `Cal Due` is the expiration, `Point`
is the node, `SN` and `Serial Number` are the same column, and a
calibration lab's `Sensitivity (mV/g)` goes into the sensitivity column.
Four columns are required: Channel, Node, Direction and Unit. A source
missing one of them is refused.

## What a channel measures

Channel Type is the quantity the channel measures, and Unit is what it
is measured in. A source states the type in its own words, and the
import is lenient about the spelling: a word the table does not know is
left blank instead of failing the import. When that leaves the type
blank, the unit fills it in (`G` is an acceleration, `V` a voltage, `N`
a force), so a controller that writes `Accel`, or nothing at all, still
comes in with its types. A type the file does state is never
overruled; a disagreement between the two is yours to resolve. When you
edit, the table does not allow one: a unit of a different kind than
the declared type is refused, and changing the type clears a unit that
no longer matches it. A unit without dimensions does not set a type,
because strain and a plain ratio have the same (empty) dimensions,
which is the reason they are two separate types.

## Direction columns from the geometry

Beside a geometry the table shows five more columns, computed from the
geometry rather than stored: each channel's measured direction as a
unit vector in the geometry's global system (Unit X, Unit Y and Unit
Z), the global axis it is nearest (Nearest Axis, signed: `X+`, `Z-`),
and the angle between the two (Angle to Axis (deg)). The direction is
the row's node and direction read through the coordinate system that
node is measured in, so a channel mounted in a rotated local system
shows its actual global vector. Edit the node or the direction and the
five columns update at once; a node the geometry does not have leaves
them blank. They come from the geometry, so they are not editable and
are not written to Excel. The report's instrumentation table includes
them beside the basis geometry, and a script reads the same rows from
`visualdynamics.core.tables.table_of(table, geometry)`.

## Role and Control

Role is one value instead of three check boxes, so the one invalid
combination cannot be entered: a channel cannot be both the input and
the output of the same estimate. Blank means undeclared, the same as an
undeclared unit.

Control and Role are separate. Control says what the controller drove
the test to match; Role says what the channel is to an FRF. A test can
be controlled to a force or a voltage as well as to a motion, so a
control channel can have any role: a controlled force can be the
reference of the FRFs, and a controlled accelerometer can be set aside
as a monitor.

A controller run sets these for you. A channel with a feedback device
is a drive, which is Rattlesnake's own rule, so it imports as a
reference; every other enabled channel imports as a response. A random,
transient or sine-sweep run also states its control channels through
its specification, since the requirement is written per control
channel, so a response channel whose DOF the specification names is
imported with Control checked.

## Two levels of strictness

A file and an edit are treated differently:

- A file is read leniently. A value that cannot be read is left blank
  instead of failing the whole import, since a refused import would
  leave you nothing to fix, and this table is where you would fix it.
- An edit is checked when you type it. An invalid value is refused with
  the reason, and is never written.

So a spreadsheet full of bad values still imports, with the bad cells
empty, but typing the same values into those cells is refused. Dates
are read in ISO form (`2027-03-31`) and in the common American and
European forms, and are stored in ISO form. The last channel of a
table cannot be deleted; delete the object instead.

## Merging tables

A survey acquired in passes has one channel table per pass, and
`project.merge` combines them (see [Projects](projects.md#merging)).
Two rows are the same channel when everything except the channel number
matches, since the number belongs to the hardware of one acquisition,
not to the point measured. The merged table renumbers its channels 1 to
N. Rows that agree on point and role but differ in anything else are
refused, with the point named, because keeping either description would
drop the other without notice.

## Round trip through Excel

*Save As…* on the table's context menu, with *Excel workbook (.xlsx)*
picked as the file type, writes an `.xlsx` a calibration lab can fill
in, and the column rules go with it. Every column that limits its
values carries Excel's own data validation (Direction, Role, Control,
Channel Type and Triax DOF), so Excel refuses a bad direction just as
the window does.

The unit list depends on the channel's type. Excel cannot hold a
different list per row, but it can look one up: the units for each
quantity are on a hidden sheet as named ranges, and the validation rule
is `INDIRECT` of the Channel Type cell in the same row. Declare a
channel a voltage and its unit cell offers V and mV; leave the type
blank and it offers every unit.

Channel and node are written as numbers, so Excel does not flag each
cell with "number stored as text", and a blank node stays blank instead
of becoming a zero.

```python
name = project.new_channel_table()       # an empty table
project.add_channels(name, 4)            # channels 1 to 4, blank
project[name].set_cell('node', 0, '101') # what typing in a cell does

project.channel_table.dof_strings()      # ['101Z+', '107Z+', ...]
project.channel_table.roles()            # ['response', 'reference', ...]
project.channel_table.controls()         # a boolean per channel
project.channel_table.sensitivities()    # mV/EU, NaN where undeclared
project.channel_table.units_for(0)       # the units the first channel could be in

visualdynamics.export_file(table, 'channels.xlsx', format='excel')
table = visualdynamics.import_file('channels.xlsx')
```
