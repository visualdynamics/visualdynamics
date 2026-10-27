"""The sensors Visual Dynamics can draw by model: simplified bodies built
from each datasheet's dimensions.

A body is described in its own frame. For a triaxial accelerometer the
frame is the sensor's X, Y and Z, with the cable or connector on the −X
face, X the long side and Z normal to the mounting base. For a
uniaxial accelerometer or a force transducer, z is the sensing axis
(normal to the mounting base) and the connector is on the +x face.
Sizes are the extents along x, y and z in millimeters: for a box its
sides, for a hex prism the width across the flats twice and then the
height, for a puck its diameter twice and then its height.

The bodies are drawn only, centered on their node. Nothing here enters
the finite element model, the mass properties or a mesh.
"""

# The catalog is Brandon's list of 2026-10-26, its dimensions and masses
# read from PCB's specification tables that day (PLAN.md, "Sensors drawn
# on the geometry"; H x L x W there is z x x x y here). The axes follow
# the decisions made then, not an outline drawing: pcb.com refuses an
# automated fetch, and its drawings come as downloads. That a
# rectangular triax's long side is X, with the connector on -X, and that
# a uniaxial senses along its height, are for Brandon to confirm against
# the drawings (2026-10-27).

from __future__ import annotations

from dataclasses import dataclass

#: the kinds of sensor a body is drawn for
KINDS = ('triax', 'uniaxial', 'force')

#: the channel type each kind measures
MEASURES = {'triax': 'acceleration', 'uniaxial': 'acceleration',
            'force': 'force'}


@dataclass(frozen=True)
class SensorModel:
    """One sensor model's body: what it is, its shape and its size."""

    model: str                              # as the datasheet names it
    make: str
    kind: str                               # one of KINDS
    shape: str                              # 'box', 'hex' or 'puck'
    size: tuple[float, float, float]        # mm along x, y, z
    mass: float                             # grams
    connector: str                          # the face it is on: '+x' or '-x'
    description: str = ''


def _triax(model, size, mass, connector_text):
    return SensorModel(model, 'PCB', 'triax', 'box', size, mass, '-x',
                       f'triaxial accelerometer, {connector_text}')


def _uniaxial(model, size, mass, connector_text):
    return SensorModel(model, 'PCB', 'uniaxial', 'box', size, mass, '+x',
                       f'uniaxial accelerometer, {connector_text}')


def _force(model, rating):
    return SensorModel(model, 'PCB', 'force', 'hex', (15.88, 15.88, 15.88),
                       22.7, '+x', f'force sensor, {rating}, 10-32 coaxial')


CATALOG: tuple[SensorModel, ...] = (
    _triax('356A01', (6.35, 6.35, 6.35), 1.0, 'integral cable'),
    _triax('356A03', (6.35, 6.35, 6.35), 1.0, 'integral cable'),
    _triax('356A02', (20.3, 14.0, 14.0), 10.5, '1/4-28 4-pin connector'),
    _triax('356A33', (19.6, 10.2, 10.2), 5.3, '1/4-28 4-pin connector'),
    _triax('356A43', (19.1, 10.2, 10.2), 4.2, '1/4-28 4-pin connector'),
    _uniaxial('352A24', (12.2, 7.1, 4.8), 0.8, '3-56 coaxial connector'),
    _uniaxial('352A73', (8.6, 4.1, 2.8), 0.3, 'integral cable'),
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
                         '-x', 'triaxial accelerometer'),
    'uniaxial': SensorModel('', '', 'uniaxial', 'box', (8.0, 8.0, 8.0), 0.0,
                            '+x', 'uniaxial accelerometer'),
    'force': SensorModel('', '', 'force', 'puck', (16.0, 16.0, 12.0), 0.0,
                         '+x', 'force transducer'),
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
