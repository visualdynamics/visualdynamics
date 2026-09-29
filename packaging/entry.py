"""What the packaged application starts from.

A frozen build has no console script, so `visualdynamics-gui` — the entry
point in `pyproject.toml` — is not available to it. This is the same
call, reachable as a file for PyInstaller to analyze.

`freeze_support` first and unconditionally: anything here that starts a
process (VTK does, on some paths) re-executes this file in the child,
and without it the child runs the application again instead of the work
it was given. On a frozen macOS build that is an infinite fan of
windows, and it is the classic way a bundle that works from source dies
the moment it is packaged.
"""

from __future__ import annotations

import multiprocessing
import os
import sys


def matplotlib_cache() -> str:
    """Where matplotlib keeps its font cache for the packaged app: a
    per-user cache folder that outlives the process.

    The packaged app took 15.6 s to show a window where the same code
    from a checkout took 1.6 (measured 2026-09-28): pyvista imports
    matplotlib.pyplot for its colors, pyplot loads the font cache, and
    PyInstaller's matplotlib hook points MPLCONFIGDIR at a new temporary
    folder on every launch — so the cache was rebuilt every time, and
    on macOS building it runs `system_profiler SPFontsDataType`, ten
    seconds and more. The hook runs before this file; setting the
    variable here, before anything imports matplotlib, is what wins.
    The cache file names its own matplotlib version, so a newer build
    rebuilds it once rather than misreading it.
    """
    if sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Caches/org.visualdynamics.app')
    elif sys.platform == 'win32':
        base = os.path.join(os.environ.get('LOCALAPPDATA')
                            or os.path.expanduser('~'), 'Visual Dynamics')
    else:
        base = os.path.join(os.environ.get('XDG_CACHE_HOME')
                            or os.path.expanduser('~/.cache'), 'visualdynamics')
    return os.path.join(base, 'matplotlib')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    cache = matplotlib_cache()
    try:
        os.makedirs(cache, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = cache
    except OSError:
        pass            # unwritable: the hook's temporary folder stands
    from visualdynamics.gui import main

    sys.exit(main(sys.argv[1:]))
