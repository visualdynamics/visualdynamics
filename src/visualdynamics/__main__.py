"""`python -m visualdynamics` — the same entry point as the `visualdynamics`
script, so the app starts without one being installed.

The guard is not ceremony. The extraction's worker processes are
spawned, and a spawned worker re-imports the parent's main module —
this one, as `__mp_main__` — before it takes its first task. Without
the guard every worker launched the whole app, and with a file on the
command line every one of them loaded it: eight copies of a 23 GB
record (Brandon, 2026-10-01, from a source checkout). The console
script's wrapper and the frozen app's entry are guarded already.
"""

import sys

from .gui import main

if __name__ == '__main__':
    sys.exit(main())
