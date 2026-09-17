"""Shared libraries a Linux bundle must leave to the system.

PyInstaller collects everything the application links against, which on
Linux includes the C++ runtime. That is right for most libraries and
wrong for this one, because of how OpenGL is loaded: VTK asks the
system's libGL for a rendering config, libGL `dlopen`s the distribution's
mesa driver (`swrast_dri.so` and friends), and that driver resolves
`libstdc++` against **whatever is already loaded** — which, inside a
frozen application, is the copy the build machine had.

So a bundle built on Debian 12 carries `GLIBCXX_3.4.30`, Debian 13's mesa
needs `GLIBCXX_3.4.33`, the driver fails to load, no GL config exists,
and the application exits during startup with

    vtkXOpenGLRenderWindow: Could not find a decent config

before it has drawn anything. Measured on 2026-08-27: the 0.0.1 AppImage
started on bookworm, died on trixie, and deleting these two files from
the bundle by hand made it start there in four seconds. The whole point
of an AppImage is that it runs on distributions newer than the one that
built it, so this is the difference between a portable artifact and one
that only works where it was made.

The list is deliberately these two and no more. AppImage's own
excludelist is far longer, but every extra name is a library the
application then requires the *host* to provide, and the evidence here
covers exactly this pair. `libX11` in particular must stay bundled:
removing it was tried on the same day and produced a segmentation fault.
"""

from __future__ import annotations

import os

# libgcc travels with libstdc++: they are one runtime, and a mismatched
# pair is its own failure mode.
SYSTEM_OWNED = frozenset({'libstdc++.so.6', 'libgcc_s.so.1'})


def strip_system_libraries(binaries, platform):
    """Drop the libraries the host must own, on the platforms that care.

    Parameters
    ----------
    binaries : list of tuple
        PyInstaller's binary TOC — ``(destination, source, typecode)``
        entries, as ``Analysis.binaries`` holds them.
    platform : str
        A `sys.platform` value. Only Linux is filtered; macOS and Windows
        load OpenGL without `dlopen`ing a driver that resolves against
        the bundle, and their frameworks expect the runtime to be there.

    Returns
    -------
    list of tuple
        The entries to keep, in their original order.
    """
    if not platform.startswith('linux'):
        return list(binaries)
    return [entry for entry in binaries
            if os.path.basename(entry[0]) not in SYSTEM_OWNED]
