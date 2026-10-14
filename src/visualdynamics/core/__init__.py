"""The data model: what a test holds. Nothing here uses Qt.

The geometry, the data arrays, the shapes, the channel tables and the
computations on them. All of it works the same in a script, in a
notebook and in the app, because none of it depends on where it is
used; the app displays these objects and does not hold its own copy
of them. The guide's [The objects](../guide/objects.md) describes each
kind of object (what it stores, where it comes from, what can be done
with it) and links back here for the signatures.
"""

from . import fem, mesh
from .channel_table import ChannelTable
from .data import (
    DataArray,
    Frf,
    Psd,
    ShockSpecification,
    Specification,
    Spectrum,
    Srs,
    TimeHistory,
    TransientSpecification,
)
from .geometry import Geometry, View
from .shapes import ShapeSet

__all__ = [
    'ChannelTable',
    'DataArray',
    'Frf',
    'Geometry',
    'Psd',
    'ShapeSet',
    'ShockSpecification',
    'Specification',
    'Spectrum',
    'Srs',
    'TimeHistory',
    'TransientSpecification',
    'View',
    'fem',
    'mesh',
]
