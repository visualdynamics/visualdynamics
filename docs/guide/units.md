# Units

Values are stored in SI and converted at the view boundary. Nothing
downstream has to ask what scale a number is on: an FRF is
acceleration per force in SI whether it came from a file in g/lbf or
mm/s²/N.

## Declaring units

A file that never said what its numbers meant imports **undefined** —
raw values, dimension `unknown`, nothing converted:

```python
geometry.define_units('m')                 # a geometry is one length
shapes.define_units('kg')                  # mass normalization
frf.define_units('m/s**2', reference_units='N')
time.define_units(['g', 'lbf', 'g'])       # per record, when they differ
```

Declaring converts the stored values to SI once and remembers the
unit, so a wrong guess is corrected by reinterpreting rather than
reimporting.

A file can say what a quantity *is* without saying what scale it is
on (a UNV with no dataset 164). That claim is kept as a
`dimension_hint`: advisory only, nothing scales by it, but it labels
an axis and narrows the units offered.

## Display systems

```python
from visualdynamics.units import SYSTEMS
list(SYSTEMS)          # 'm-kg-N-s', 'mm-kg-N-s', 'in-slinch-lbf-s',
                       # 'ft-slug-lbf-s', and a '(g)' variant of each
frf.display_ordinate(SYSTEMS['in-slinch-lbf-s'], [0])
frf.plot(unit_system=SYSTEMS['in-slinch-lbf-s'])
```

The app's unit selector picks one for everything on screen; a script
passes one where it matters. Reports render in the system they are
given and re-render when it changes.

## Two traps worth knowing

`g` is standard gravity, never grams, and `mil` is a thousandth of an
inch, never the angular one. Unit strings are normalized before pint
sees them, because pint reads a bare `g` as 0.001 kg and `mil` as
1/6400 of a turn — both wrong, both silent. `kg`, `mg` and `gram`
still mean mass; grams are not offered as a mass unit anywhere.

Case is forgiven where it can be. A controller's channel table holds
whatever was typed into it, and `G`, `LBF` and `Volts` are typed all
the time; pint reads `G` as gauss and refuses the other two, so those
channels used to arrive unit-less. A unit string that means nothing as
written is re-read against the spellings the toolset knows, ignoring
case, and the record carries the corrected spelling (`g`, `lbf`,
`volts`). A string that already names a quantity is never respelled:
`mV` is millivolts and `MV` is megavolts, as written.
