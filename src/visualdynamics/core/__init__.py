"""The data model: what a test holds, with no Qt anywhere.

Geometry, the data arrays, shapes, channel tables, and the
computations over them. Everything here works in a script, in a
notebook and in the app equally, because none of it knows which
it is in — the app is a view onto these objects, not the place
they live. The guide's [The objects](../guide/objects.md) maps every
kind — what it stores, where it comes from, what can be done with it
— and points back here for the signatures.
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
from .geometry import Geometry
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
    'fem',
    'mesh',
]
