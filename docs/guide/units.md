# Units

Values are stored in SI and converted only where they are shown or
written out. Nothing downstream has to ask what scale a number is on:
an FRF is acceleration per force in SI whether it came from a file in
g/lbf or in mm/s²/N.

## Declaring units

A file that does not say what its numbers mean imports with
**undefined** units: the raw values, dimension `unknown`, nothing
converted. Declare the units to fix that:

```python
geometry.define_units('m')                 # a geometry has one length unit
shapes.define_units('kg')                  # the mass the shapes are normalized to
frf.define_units('m/s**2', reference_units='N')
time.define_units(['g', 'lbf', 'g'])       # per record, when they differ
```

Declaring converts the stored values to SI once and remembers the
unit, so a wrong guess is corrected by declaring again rather than by
importing the file again.

A file can say what a quantity *is* without saying its scale (a UNV
file with no dataset 164, for example). That information is kept as a
`dimension_hint`. It is advisory only and nothing is scaled by it, but
it labels an axis and narrows the units offered.

## Display systems

```python
from visualdynamics.units import SYSTEMS
list(SYSTEMS)          # 'm-kg-N-s', 'mm-kg-N-s', 'in-slinch-lbf-s',
                       # 'ft-slug-lbf-s', and a '(g)' variant of each
frf.display_ordinate(SYSTEMS['in-slinch-lbf-s'], [0])
frf.plot(unit_system=SYSTEMS['in-slinch-lbf-s'])
```

The app's display-unit selector, in the status bar, sets one system
for everything on screen; a script passes a system where it needs one.
Reports are drawn in the system they are given and redrawn when it
changes. A `.mat` file exported in a display system is written in that
system's units, and every unit field in it names the unit its values
are in, so a record in g reads as g and a geometry in inches says
`'in'`; the file also names the system it was written in, and reading
it back converts each object to SI from its own labels
([Importing and exporting](../export.md)).

## Two unit names to watch

`g` is standard gravity, never grams, and `mil` is a thousandth of an
inch, never the angular mil. Unit strings are normalized before pint
reads them, because pint reads a bare `g` as 0.001 kg and `mil` as
1/6400 of a turn, and neither mistake would raise an error. `kg`, `mg`
and `gram` still mean mass, but grams are not offered as a mass unit
anywhere.

Letter case is corrected where that can be done safely. A
controller's channel table holds whatever was typed into it, and `G`,
`LBF` and `Volts` are common. Pint reads `G` as gauss and does not
recognize the other two. A unit string that means nothing as written
is read again against the spellings Visual Dynamics knows, ignoring
case, and the record keeps the corrected spelling (`g`, `lbf`,
`volts`). A string that already names a unit is never changed: `mV` is
millivolts and `MV` is megavolts, as written.
