"""`python -m visualdynamics` — the same entry point as the `visualdynamics`
script, so the app starts without one being installed.
"""

import sys

from .gui import main

sys.exit(main())
