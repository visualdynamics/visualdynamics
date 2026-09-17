"""The full modal-test workflow, scripted — no GUI anywhere.

Every step here is the verb behind a button in the app: the GUI's
window holds the same `Project` and calls the same methods, so a
project built by clicking and one built by running this are the same
thing, and either opens in the other.

Run from the repository root:

    python examples/modal_workflow.py [output-directory]
"""

from __future__ import annotations

import pathlib
import sys

import visualdynamics

HERE = pathlib.Path(__file__).resolve().parent.parent
DATA = HERE / 'testdata' / 'plate'
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '.')

# ---- 1. the project ---------------------------------------------------------
project = visualdynamics.Project('Plate Modal Survey')
project.project_type = 'Modal Test'          # what report it is for

# ---- 2. import ---------------------------------------------------------
# anything visualdynamics reads: a Rattlesnake run, a geometry, a saved project
project.import_file(DATA / 'modal_spectra.nc4')
project.import_file(DATA / 'test_geometry.npz')
project.geometry.define_units('m')           # the file did not say

# objects are reached by what they are, not by what they were named:
# `project.geometry`, or `project.basis.frf` once the Basis is set

# the measured side is the Basis: comparisons happen in its DOFs, its
# modes are the MAC rows and the frequency-error baseline
project.set_basis(*project.names)

# the model side, linked to itself and left unroled
project.add('FEM Geometry', visualdynamics.import_file(DATA / 'geometry.npz'))
project.add('FEM Modes', visualdynamics.import_file(DATA / 'shapes.npy'))
project.link('FEM Geometry', 'FEM Modes')
project.other.geometry.define_units('m')     # the group that is not the Basis
project.other.shapes.define_units('kg')

# ---- 3. process and identify ------------------------------------------------
# a verb takes the object as readily as its name
project.compute_psds(project.basis.time_history)   # Compute PSDs
# limit is the judgment a person makes on the screen by stopping:
# the plate has five modes in this band — four distinct peaks and the
# repeated pair's second tooth — and left uncapped the loop would go
# on dredging the noise floor
project.fit_modes(project.basis.frf, bounds=(300.0, 1300.0),
                  limit=5, name='Experimental Modes')

# ---- 4. correlate -----------------------------------------------------------
# comparing across geometries projects onto the Basis DOFs first —
# match_modes does it the way the comparison screen does
project.match_modes(project.basis.shapes, project.other.shapes,
                    threshold=0.7)

# ---- 5. look at it ----------------------------------------------------------
# every plot the GUI draws has a call; `path=` renders headless
basis, fem = project.basis, project.other
basis.frf.plot(path=OUT / 'frfs.png')
basis.frf.plot_cmif(basis.shapes, path=OUT / 'cmif.png')
basis.shapes.plot_mac(path=OUT / 'automac.png')
project.plot_mac(basis.shapes, fem.shapes, path=OUT / 'crossmac.png')
basis.coherence.plot_map(path=OUT / 'coherence.png')
basis.geometry.plot_dofs(basis.frf, 'force',
                         screenshot=str(OUT / 'excitation.png'))
project.animate(basis.shapes, mode=0, screenshot=str(OUT / 'mode1.png'))
# without `path`/`screenshot` each of these opens the app's own window:
#     basis.frf.plot()
#     project.animate(basis.shapes, mode=2)

# ---- 6. report and save -----------------------------------------------------
project.generate_report('modal')             # bound symbolically
project.export_report(project.report, OUT / 'modal_report.html')
project.save(OUT / 'modal.vdyn')             # opens in the GUI as-is

print(project)
print(f'\nwrote {OUT / "modal.vdyn"} and {OUT / "modal_report.html"}')
