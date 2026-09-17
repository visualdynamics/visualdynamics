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
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    from visualdynamics.gui import main

    sys.exit(main(sys.argv[1:]))
