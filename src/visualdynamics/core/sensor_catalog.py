"""The sensors Visual Dynamics can draw by model: simplified bodies built
from each outline drawing's dimensions.

A body is described in its own frame. For a triaxial accelerometer the
frame is the sensor's X, Y and Z, with the cable or connector on the −X
face and Z normal to the mounting base; the X, Y and Z faces carry
their letters, as the sensors do. For a uniaxial accelerometer or a
force transducer, z is the sensing axis (normal to the mounting base)
and the connector or cable is on the +x side. Sizes are the extents of
the body alone along x, y and z in millimeters: for a box its sides,
for a hex prism the width across the flats twice and then the height,
for a puck its diameter twice and then its height, for a teardrop its
length, the diameter of its round end and its height, and for a rounded
box its length with the round nose, its width and its height. The plug
is the connector out of the connector face, the cable what is drawn
beyond it, and a force transducer's studs stand out of both ends.

The bodies are drawn only, centered on their node. Nothing here enters
the finite element model, the mass properties or a mesh.
"""

# The catalog is Brandon's list of 2026-10-26. The shapes and sizes are
# read from PCB's outline drawings (2026-10-27; drawing numbers beside
# each entry): the masses from the specification tables. The drawings
# settled what the first cut had guessed: a rectangular triax's long
# side is X with the connector on -X and the X, Y and Z letters on the
# positive faces; a uniaxial senses along its height; the 352A24 is a
# teardrop and the 352A73 a box with a round nose.

from __future__ import annotations

from dataclasses import dataclass

#: the kinds of sensor a body is drawn for
KINDS = ('triax', 'uniaxial', 'force')

#: the channel type each kind measures
MEASURES = {'triax': 'acceleration', 'uniaxial': 'acceleration',
            'force': 'force'}

#: the shapes a body is built from
SHAPES = ('box', 'hex', 'puck', 'teardrop', 'rounded')

NOTHING = (0.0, 0.0)


@dataclass(frozen=True)
class SensorModel:
    """One sensor model's body: what it is, its shape and its size."""

    model: str                              # as the datasheet names it
    make: str
    kind: str                               # one of KINDS
    shape: str                              # one of SHAPES
    size: tuple[float, float, float]        # mm along x, y, z, the body alone
    mass: float                             # grams
    connector: str                          # the face it is on: '+x' or '-x'
    description: str = ''
    plug: tuple[float, float] = NOTHING     # mm: diameter, length out of the face
    cable: tuple[float, float] = NOTHING    # mm: diameter, length drawn
    stud: tuple[float, float] = NOTHING     # mm: diameter, length out of each end


def _triax(model, size, mass, connector_text, plug, cable=NOTHING):
    return SensorModel(model, 'PCB', 'triax', 'box', size, mass, '-x',
                       f'triaxial accelerometer, {connector_text}', plug,
                       cable)


def _uniaxial(model, shape, size, mass, connector_text, plug, cable=NOTHING):
    return SensorModel(model, 'PCB', 'uniaxial', shape, size, mass, '+x',
                       f'uniaxial accelerometer, {connector_text}', plug,
                       cable)


def _force(model, rating):
    # drawing 8561: a 15.88 mm hex, a 10-32 tapped hole in each end for
    # the supplied studs, the coaxial connector out of a flat
    return SensorModel(model, 'PCB', 'force', 'hex', (15.88, 15.88, 15.88),
                       22.7, '+x', f'force sensor, {rating}, 10-32 coaxial',
                       plug=(8.4, 5.8), stud=(4.8, 3.5))


CATALOG: tuple[SensorModel, ...] = (
    # drawing 21823: a 6.3 mm cube, the cable's boot 3.4 mm out of -X
    _triax('356A01', (6.3, 6.3, 6.3), 1.0, 'integral cable',
           plug=(3.2, 3.4), cable=(1.3, 12.0)),
    _triax('356A03', (6.3, 6.3, 6.3), 1.0, 'integral cable',
           plug=(3.2, 3.4), cable=(1.3, 12.0)),
    # drawing 10324: a 14 mm cube, 20.3 mm over the connector
    _triax('356A02', (14.0, 14.0, 14.0), 10.5, '1/4-28 4-pin connector',
           plug=(6.35, 6.3)),
    # drawing 19954: a 10.2 mm cube, 19.6 mm over the connector
    _triax('356A33', (10.2, 10.2, 10.2), 5.3, '1/4-28 4-pin connector',
           plug=(6.35, 9.4)),
    # drawing 31660: a 10.2 mm cube, 19.1 mm over the connector
    _triax('356A43', (10.2, 10.2, 10.2), 4.2, '1/4-28 4-pin connector',
           plug=(6.35, 8.9)),
    # drawing 12331: a teardrop 9.9 mm long with a 7.1 mm round end,
    # 4.8 mm high, the 3-56 connector out of the narrow end
    _uniaxial('352A24', 'teardrop', (9.9, 7.1, 4.8), 0.8,
              '3-56 coaxial connector', plug=(2.5, 2.3)),
    # drawing 24787: 6.75 mm long with a round nose, 3.96 mm wide, 2.8 mm
    # high, a twisted pair out of the flat end (Brandon, 2026-10-27: the
    # drawing's side view rounds the potting at the cable end, which a
    # first cut took for the plan)
    _uniaxial('352A73', 'rounded', (6.75, 3.96, 2.8), 0.3, 'integral cable',
              plug=NOTHING, cable=(0.9, 10.0)),
    _force('208C01', '10 lb'),
    _force('208C02', '100 lb'),
    _force('208C03', '500 lb'),
    _force('208C04', '1000 lb'),
    _force('208C05', '5000 lb'),
)

#: what a model not in the catalog is drawn as, by kind: a cube for an
#: accelerometer and a puck for a force transducer, at a typical size
GENERIC: dict[str, SensorModel] = {
    'triax': SensorModel('', '', 'triax', 'box', (10.0, 10.0, 10.0), 0.0,
                         '-x', 'triaxial accelerometer', plug=(5.0, 6.0)),
    'uniaxial': SensorModel('', '', 'uniaxial', 'box', (8.0, 8.0, 8.0), 0.0,
                            '+x', 'uniaxial accelerometer', plug=(2.5, 3.0)),
    'force': SensorModel('', '', 'force', 'puck', (16.0, 16.0, 12.0), 0.0,
                         '+x', 'force transducer', plug=(6.0, 5.0),
                         stud=(4.8, 3.5)),
}


def _key(model: str) -> str:
    return ''.join(str(model).split()).upper()


_BY_MODEL = {_key(entry.model): entry for entry in CATALOG}


def lookup(model: str) -> SensorModel | None:
    """The catalog entry for a model, ignoring case and spaces; None when
    the model is not in the catalog.

    Parameters
    ----------
    model : str
        The model as a channel table names it, such as '356A02'.

    Returns
    -------
    SensorModel or None
    """
    return _BY_MODEL.get(_key(model))


def models_for(channel_type: str) -> list[SensorModel]:
    """The catalog's models that measure a channel type: accelerometers
    for 'acceleration', force transducers for 'force', none otherwise.

    Parameters
    ----------
    channel_type : str
        A channel table's type, such as 'acceleration'.

    Returns
    -------
    list of SensorModel
    """
    return [entry for entry in CATALOG
            if MEASURES[entry.kind] == channel_type]
