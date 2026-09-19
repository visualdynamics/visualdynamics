"""The full random vibration workflow, scripted — no GUI anywhere.

A controller run in, a report out. The short way is one call — it
takes the geometry, the photographs and a `last=` window for a long
run as keywords; the rest of this is that same call written out,
because a real test wants the project afterwards — to write the test
summary the report has a slot for, and to save it.

Every step is the verb behind a button in the app, and every plot is
the one the app draws, rendered to a file instead of a window.

Run from the repository root:

    python examples/random_workflow.py [output-directory]
"""

from __future__ import annotations

import pathlib
import sys

import visualdynamics
from visualdynamics.plot import plot_bars, plot_coherence_map, plot_comparison

HERE = pathlib.Path(__file__).resolve().parent.parent
DATA = HERE / 'testdata' / 'plate'
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '.')

# ---- the whole thing, in one call -------------------------------------------
# import, PSDs, octave bands, multiple coherence, report, HTML
visualdynamics.random_vibration_report(DATA / 'random.nc4', OUT / 'quick_report.html')

# ---- ...and the same thing written out --------------------------------------
project = visualdynamics.random_vibration_run(DATA / 'random.nc4')
# the run says what kind of test it was, so the project comes back typed
assert project.project_type == 'Random Vibration'

# objects are reached by what they are, not by what they were named
history = project.time_history
spec = project.specification
psds, octave = project.psds      # the spec is not among them: asking
#                                  for PSDs means the measured kind

# ---- look at it -------------------------------------------------------------
# every plot the GUI draws has a call; `path=` renders headless
history.save_plot(OUT / 'time.png')
psds.save_plot(OUT / 'psds.png')
plot_comparison(psds, spec, path=OUT / 'control.png', show=False)
plot_comparison(octave, spec, path=OUT / 'control_octave.png', show=False)
plot_bars(psds, spec, 'error', path=OUT / 'error.png', show=False)
plot_bars(psds, spec, 'lines', path=OUT / 'lines.png', show=False)
plot_coherence_map(project.coherence, path=OUT / 'coherence.png', show=False)
# without `path` each of these opens the app's own window:
#     plot_comparison(psds, spec)

# ---- report and save --------------------------------------------------------
project.generate_report('random')            # bound symbolically
project.export_report(project.report, OUT / 'random_report.html')
project.save(OUT / 'random.vdyn')           # opens in the GUI as-is

print(project)
print(f'\nwrote {OUT / "random.vdyn"} and {OUT / "random_report.html"}')
