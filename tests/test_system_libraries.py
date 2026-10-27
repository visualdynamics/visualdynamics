"""What a Linux bundle may not carry, and what it must.

The 0.0.1 AppImage started on the Debian release it was built on and died
on the next one, before drawing anything, because PyInstaller had
bundled `libstdc++` and mesa's driver resolved against that copy instead
of the host's. `packaging/system_libraries.py` carries the reasoning;
these tests hold the rule to the two things the day's measurements
actually established — that the C++ runtime goes, and that the X
libraries stay.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'packaging'))


@pytest.fixture(scope='module')
def strip():
    module = pytest.importorskip('system_libraries')
    return module.strip_system_libraries


# A plausible slice of what PyInstaller collects: the runtime that has to
# go, the X libraries that must not, and the application's own.
COLLECTED = [
    ('libstdc++.so.6', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6', 'BINARY'),
    ('libgcc_s.so.1', '/usr/lib/x86_64-linux-gnu/libgcc_s.so.1', 'BINARY'),
    ('libX11.so.6', '/usr/lib/x86_64-linux-gnu/libX11.so.6', 'BINARY'),
    ('libX11-xcb.so.1', '/usr/lib/x86_64-linux-gnu/libX11-xcb.so.1', 'BINARY'),
    ('libxcb-glx.so.0', '/usr/lib/x86_64-linux-gnu/libxcb-glx.so.0', 'BINARY'),
    ('PySide6/Qt/lib/libQt6Core.so.6', '/w/libQt6Core.so.6', 'BINARY'),
    ('vtkmodules/libvtkCommonCore.so', '/w/libvtkCommonCore.so', 'BINARY'),
    ('libpython3.14.so.1.0', '/usr/lib/libpython3.14.so.1.0', 'BINARY'),
]


def kept(strip, platform):
    return {entry[0] for entry in strip(COLLECTED, platform)}


def test_the_cxx_runtime_is_left_to_the_host_on_linux(strip):
    """The bug itself: ours shadows the system's, mesa's driver fails to
    load against it, and the application exits before a window."""
    names = kept(strip, 'linux')
    assert 'libstdc++.so.6' not in names
    assert 'libgcc_s.so.1' not in names, 'libgcc travels with libstdc++'


def test_the_x_libraries_stay(strip):
    """Removing these was tried on the same day and segfaulted, so the
    fix is the C++ runtime specifically and not 'system libraries' as a
    category."""
    names = kept(strip, 'linux')
    assert {'libX11.so.6', 'libX11-xcb.so.1', 'libxcb-glx.so.0'} <= names


def test_what_the_application_needs_survives(strip):
    names = kept(strip, 'linux')
    assert 'PySide6/Qt/lib/libQt6Core.so.6' in names
    assert 'vtkmodules/libvtkCommonCore.so' in names
    assert 'libpython3.14.so.1.0' in names


@pytest.mark.parametrize('platform', ['darwin', 'win32'])
def test_mac_and_windows_are_left_alone(strip, platform):
    """Neither loads OpenGL by `dlopen`ing a distribution driver, and a
    .app or an installer is expected to carry its own runtime."""
    assert kept(strip, platform) == {entry[0] for entry in COLLECTED}


def test_the_order_is_not_disturbed(strip):
    """PyInstaller's TOC order decides what a collision resolves to, so
    filtering must not reshuffle what it keeps."""
    keep = [entry[0] for entry in strip(COLLECTED, 'linux')]
    assert keep == [entry[0] for entry in COLLECTED
                    if entry[0] not in {'libstdc++.so.6', 'libgcc_s.so.1'}]


def test_the_spec_actually_applies_the_rule():
    """A rule nothing calls is a comment. The spec is not importable —
    PyInstaller injects its globals — so this reads it."""
    spec = (ROOT / 'packaging' / 'visualdynamics.spec').read_text(encoding='utf-8')
    assert 'strip_system_libraries(analysis.binaries' in spec
