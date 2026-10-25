"""The project-tree page names every reading the tree gives.

Brandon, 2026-09-11: users may not know the pairings exist — mode
shapes on a geometry, a time history animating on it, the operating
deflection shape from FRFs, a PSD on a geometry, a specification
beside a PSD — unless the documentation shows them. The page carries
every object kind, every pairing the window implements, and a picture
for each reading; this pins that it keeps doing so as kinds and
pairings are added.
"""

from __future__ import annotations

import pathlib
import re

GUIDE = pathlib.Path(__file__).resolve().parent.parent / 'docs' / 'guide'
PAGE = GUIDE / 'project-tree.md'


def _page():
    return PAGE.read_text(encoding='utf-8')


def test_every_object_kind_has_its_single_selection_reading():
    text = _page()
    for kind in ('Geometry', 'Photos', 'Channel table', 'Time history',
                 'Spectrum', 'PSD', 'Specification', 'FRF', 'Coherence',
                 'SRS', 'Shock specification', 'Sine sweep specification',
                 'sine levels', 'Shape set', 'Matched modes', 'Report'):
        assert f'| {kind}' in text, f'{kind}: no row in the one-object table'
    assert 'project row' in text and 'Generate Report' in text


def test_every_pairing_the_window_implements_is_on_the_page():
    """The pairings, as `_deflection_for`, `_render_series` and the
    comparison finders implement them — each named the way a user would
    look for it."""
    text = _page()
    for reading in ('a shape set', 'a time history', 'an FRF or a spectrum',
                    'a PSD of autospectra', 'a whole CPSD',
                    'a picked reference column', 'a second geometry',
                    'Two shape sets', 'MAC', 'Edit\nFit', 'Resynthesis',
                    'PSD + specification', 'SRS + shock specification',
                    'time history + transient specification',
                    'sine levels + sine sweep specification',
                    'operating deflection shape', 'envelope', 'principal shape',
                    'DOF arrows', 'Transform to Modal Coordinates',
                    'Merge into One'):
        assert reading.replace('\\n', '\n') in text, f'{reading}: not on the page'


def test_every_picture_the_page_shows_exists():
    images = re.findall(r'\]\(images/([^)]+)\)', _page())
    assert len(images) >= 10, 'a picture per reading'
    for name in images:
        assert (GUIDE / 'images' / name).is_file(), f'{name} is not in docs/guide/images'
    assert 'tree-geometry-time.png' in images and 'tree-geometry-ods.png' in images
    assert 'tree-project.png' in images
