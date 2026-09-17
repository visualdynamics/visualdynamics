"""Every object type is tellable apart in the tree.

Icons are drawn rather than shipped, so the thing to guard is that each type
actually has one — a type with no entry falls back to a gray circle, which
is what `Specification` and `Coherence` did until someone noticed — and that
no two of them come out the same.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QSize

import visualdynamics
from visualdynamics.gui.icons import _DRAW, COLORS, object_icon, type_icon


# every object type the tree can show, taken from the object model rather
# than from the icon table — asking the icon table what needs an icon is
# what let two types go without one
def showable_types():
    from visualdynamics.core.channel_table import ChannelTable
    from visualdynamics.core.data import NAMED_CLASSES
    from visualdynamics.core.geometry import Geometry
    from visualdynamics.core.shapes import ShapeSet

    # 'Test' is the tree's root, not a core class yet, so it is named here
    return sorted({cls.__name__ for cls in NAMED_CLASSES.values()}
                  | {Geometry.__name__, ShapeSet.__name__,
                     ChannelTable.__name__, 'Test'})


TYPES = showable_types()


def test_every_object_type_has_an_icon_of_its_own():
    """A type missing from the table silently gets a gray circle."""
    assert not [name for name in TYPES if name not in _DRAW]


def test_every_drawn_type_has_a_color():
    assert not [name for name in _DRAW if name not in COLORS]


@pytest.mark.parametrize('name', TYPES)
def test_a_type_gets_its_own_icon_and_not_the_fallback(qt_app, name):
    """The fallback is a plain circle: no shape, so nothing to read."""
    icon = type_icon(name)
    assert not icon.isNull()
    assert icon.pixmap(QSize(64, 64)).toImage() != \
        type_icon('NoSuchType').pixmap(QSize(64, 64)).toImage()


def test_no_two_types_look_the_same(qt_app):
    """Shape carries the meaning, so two types sharing one would be a bug
    even if their colors differed."""
    seen = {}
    for name in TYPES:
        image = type_icon(name).pixmap(QSize(64, 64)).toImage()
        key = bytes(image.constBits())
        assert key not in seen, f'{name} looks exactly like {seen.get(key)}'
        seen[key] = name


def test_they_are_still_distinct_at_tree_size(qt_app):
    """16 px is where they are actually used, and where fine detail is lost."""
    seen = {}
    for name in TYPES:
        image = type_icon(name).pixmap(QSize(16, 16)).toImage()
        key = bytes(image.constBits())
        assert key not in seen, f'{name} and {seen.get(key)} merge at 16 px'
        seen[key] = name


def test_the_new_rattlesnake_types_resolve_from_a_real_object(qt_app):
    """object_icon goes by class name, so a type absent from the table gets
    the fallback silently — which is how this was missed."""
    from conftest import fixture_path

    out = visualdynamics.import_file(fixture_path('plate', 'modal_spectra.nc4'))
    coherence = out['Modal_coherence']
    spec = visualdynamics.import_file(
        fixture_path('plate', 'random.nc4'))['Random_specification']
    fallback = type_icon('NoSuchType').pixmap(QSize(64, 64)).toImage()
    for obj in (coherence, spec):
        image = object_icon(obj).pixmap(QSize(64, 64)).toImage()
        assert image != fallback, f'{type(obj).__name__} has no icon'


def test_a_specification_does_not_look_like_the_psd_it_subclasses(qt_app):
    """It is a PSD underneath; the tree still has to tell them apart."""
    assert type_icon('Specification').pixmap(QSize(64, 64)).toImage() != \
        type_icon('Psd').pixmap(QSize(64, 64)).toImage()
