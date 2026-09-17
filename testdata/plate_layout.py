"""The plate survey's channel layout, shared by both generators.

`generate_plate.py` (project venv) and `generate_plate_runs.py`
(visualdynamics-generators)
(rattlesnake python) must agree on where the sensors and shakers are:
the test display geometry is cut to the nodes the modal run measures,
and `tests/test_correlate.py` holds that identity. One list, imported
by both, so they cannot drift.
"""

#: two shakers: the corner, and an edge point that sits on no node
#: line of the low modes. The grid's interior stations all lie on a
#: center line or a diagonal — the plate's own symmetry — and a
#: shaker there goes blind to whichever mode family nulls that line:
#: the first reference set had three of four references unable to
#: see the 647 Hz mode at all, and its fitted shape could not
#: resynthesize the measurement. Edge stations at the quarter points
#: are the generic positions.
DRIVES = ['101Z+', '1310Z+']

#: the modal survey: the 3x3 grid in Z, both drive points (their FRFs
#: pin the fitted scale), and one in-plane channel because a real
#: survey always has a sensor pointed somewhere nothing moves
MODAL = ['101Z+', '107Z+', '113Z+', '701Z+', '707Z+', '713Z+',
         '1301Z+', '1307Z+', '1313Z+', '1310Z+', '707X+']

#: the random test's shakers: the same four stations the FRF
#: fixture uses as references — a corner and the three generic edge
#: points. Two drives were tried against the eight control channels
#: and genuinely could not hold them: the drive point ended on spec
#: and the rest scattered 3-14 dB low, which made the demonstration
#: report read as a failed test and (before the detection learned
#: the floor veto) read to the scale detection as a -4 dB run.
#: Four against eight is the drone's ratio, and it controls.
RANDOM_DRIVES = ['101Z+', '110Z+', '1304Z+', '1310Z+']

#: the random test: eight control channels on the perimeter ring —
#: the four corners and the four quarter-edge stations, which are the
#: shakers' own points. The first control set put four channels on
#: the grid's interior stations, and every interior station sits on a
#: center line or a diagonal of the square: there the edge drives'
#: responses are nearly linearly dependent, the reachable response
#: space cannot hold the flat target, and no controller — at any
#: pseudoinverse truncation, both were tried — could level them. The
#: ring is what the drives span, which is what makes this a test the
#: loop can actually pass.
CONTROL = ['101Z+', '104Z+', '110Z+', '113Z+',
           '1301Z+', '1304Z+', '1310Z+', '1313Z+']


def modal_nodes() -> list[int]:
    """Every node the modal run touches, drives included, sorted."""
    return sorted({int(dof[:-2]) for dof in set(MODAL) | set(DRIVES)})
