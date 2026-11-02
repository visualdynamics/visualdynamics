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


def bundled_mkl() -> None:
    """Point pypardiso at the oneMKL the Windows and Linux builds carry,
    and thread it through TBB, the one threading layer they carry.

    pypardiso looks for `mkl_rt` with `find_library`, then under
    `sys.prefix`'s lib folders; the spec lays oneMKL in the bundle's
    library folder itself, beside the rest of the bundle's libraries,
    and `PYPARDISO_MKL_RT` is pypardiso's own way to be told where. Set
    rather than defaulted: the build's own copy is the one it was
    tested with.
    """
    import glob

    folder = getattr(sys, '_MEIPASS', None)
    if folder is None:
        return
    # the library's own names: a looser '*mkl_rt*' took a renamed
    # 'mkl_rt.3.dll.hidden', which Windows loads all the same
    found = sorted((path for pattern in ('mkl_rt*.dll', 'libmkl_rt.so*')
                    for path in glob.glob(os.path.join(folder, pattern))), key=len)
    if found:
        os.environ['PYPARDISO_MKL_RT'] = found[0]
        # the one threading layer the build carries (the spec leaves the
        # OpenMP one out); oneMKL told to load another ends the process
        os.environ['MKL_THREADING_LAYER'] = 'TBB'


if __name__ == '__main__':
    multiprocessing.freeze_support()
    bundled_mkl()
    cache = matplotlib_cache()
    try:
        os.makedirs(cache, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = cache
    except OSError:
        pass            # unwritable: the hook's temporary folder stands
    from visualdynamics.gui import main

    sys.exit(main(sys.argv[1:]))
