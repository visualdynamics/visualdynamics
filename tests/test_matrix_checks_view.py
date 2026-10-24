"""The two matrix checks in the window: Conditioning on an FRF from
several references or a CSD, Realizability on a CSD or specification,
each drawn as a figure of its own with a cursor read in its pane, and
the specification sheet's refusal of entries that cannot hold.

Readings of objects that exist, so toggles on the bar with no act, one
up at a time; Realizability's pane carries the one apply, Make
Realizable, offered only while there is something to repair.
"""

from __future__ import annotations

import numpy as np

from visualdynamics.core.data import Frf, Specification

F = np.linspace(10.0, 1000.0, 100)


def _plant(references=('1Z+', '2Z+')):
    """Two modes, at 100 Hz moving both drives together and at 400 Hz
    against each other, read at two responses: the columns are one at
    the first mode, where the matrix is singular."""
    w = 2 * np.pi * F
    h = np.zeros((len(F), 2, 2), dtype=complex)
    for fn, phi in ((100.0, np.array([1.0, 1.0])),
                    (400.0, np.array([1.0, -1.0]))):
        wn = 2 * np.pi * fn
        h += np.outer(phi, phi)[None] / (
            wn ** 2 - w[:, None, None] ** 2
            + 2j * 0.02 * wn * w[:, None, None])
    rows, resp, refs = [], [], []
    for i, r in enumerate(('101Z+', '102Z+')):
        for j, d in enumerate(references):
            rows.append(h[:, i, j])
            resp.append(r)
            refs.append(d)
    return Frf(F, np.asarray(rows), response_dof=resp, reference_dof=refs,
               ordinate_dim=['acceleration/force'] * len(rows))


def _specification(phase_13=180.0, unstated=False):
    """Three channels at 0.01 g²/Hz, coherence one between every pair,
    in step between 1-2 and 2-3 and at `phase_13` between 1-3; with
    `unstated` the 1-3 pair is left out."""
    channels = ['1Z+', '2Z+', '3Z+']
    phases = {(0, 1): 0.0, (1, 2): 0.0, (0, 2): phase_13}
    rows, resp, refs = [], [], []
    for i in range(3):
        for j in range(3):
            if i == j:
                value = 1.0 + 0j
            else:
                key = (min(i, j), max(i, j))
                if unstated and key == (0, 2):
                    continue
                value = np.exp(1j * np.radians(phases[key])
                               * (1 if i < j else -1))
            rows.append(np.full(len(F), value * 0.01 * 9.80665 ** 2))
            resp.append(channels[i])
            refs.append(channels[j])
    return Specification(F, np.asarray(rows), response_dof=resp,
                         reference_dof=refs,
                         ordinate_dim=['acceleration**2/frequency']
                         * len(rows))


def _show(window, pump, name, obj):
    window.add_object(name, obj)
    window.show_object(name)
    pump()
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
    window.render_current()
    pump()
    return pane


def test_conditioning_is_offered_on_an_frf_of_two_references(window, pump):
    pane = _show(window, pump, 'Plant', _plant())
    assert pane.conditioning_action.isVisible()
    assert not pane.realizability_action.isVisible(), \
        'an FRF is not a cross-spectral matrix'
    single = _plant()
    one = [i for i in range(single.num_records)
           if single.reference_dof[i] == '1Z+']
    pane = _show(window, pump, 'One', Frf(
        F, single.ordinate[one], response_dof=['101Z+', '102Z+'],
        reference_dof=['1Z+', '1Z+'],
        ordinate_dim=['acceleration/force'] * 2))
    assert not pane.conditioning_action.isVisible(), \
        'one reference has nothing to compare'


def test_the_conditioning_figure_and_its_reading(window, pump):
    from PySide6.QtCore import Qt

    pane = _show(window, pump, 'Plant', _plant())
    pane.conditioning_action.trigger()
    pump()
    state = window._matrix_check
    assert state is not None and state['check'] == 'conditioning'
    assert pane.conditioning_panel.isVisible()
    reading = state['reading']
    # the cursor starts on the first flagged line, the first mode
    point = reading.at(10.0 ** state['cursor'].value())
    assert point.ill and abs(point.frequency - 100.0) < 15.0
    derived = pane.conditioning_panel.derived
    assert derived['condition'].text().endswith('(over 100)')
    assert derived['rank'].text() == '1 of 2'
    assert derived['matrix'].text() == 'FRF from 2 references to 2 responses'
    assert 'condition number over 100' in pane.conditioning_panel.verdict.text()
    # the singular values are told apart by line style
    top = state['plot']
    styles = [item.opts['pen'].style() for item in top.listDataItems()]
    assert styles[:2] == [Qt.PenStyle.SolidLine, Qt.PenStyle.DashLine]
    # moving the cursor off the mode reads a line within the limit
    state['cursor'].setValue(np.log10(400.0))
    pump()
    assert derived['condition'].text().endswith('(within 100)')
    assert derived['rank'].text() == '2 of 2'
    assert '400' in window.statusBar().currentMessage()
    assert window.project.journal[-1].startswith(
        "visualdynamics.plot.plot_conditioning(project['Plant'].conditioning()")


def test_realizability_reads_a_specification_and_makes_it_realizable(window,
                                                                     pump):
    pane = _show(window, pump, 'Spec', _specification())
    assert pane.realizability_action.isVisible()
    assert pane.conditioning_action.isVisible(), 'a CSD is conditioned too'
    pane.realizability_action.trigger()
    pump()
    panel = pane.realizability_panel
    assert panel.isVisible()
    assert panel.derived['line'].text() == 'not realizable'
    assert panel.derived['smallest'].text() == '-1'
    assert panel.derived['changes'].text() == 'coherence 0.75, phase 0°'
    assert 'not realizable at 100 of 100 lines' in panel.verdict.text()
    assert panel.apply_button.isEnabled()
    panel.apply_button.click()
    pump()
    assert 'Spec Realizable' in window.project
    assert window.current_object() is window.project['Spec Realizable']
    window.render_current()
    pump()
    assert 'realizable at all 100 lines' in panel.verdict.text()
    assert not panel.apply_button.isEnabled(), 'nothing left to repair'
    assert any(line.startswith("project.make_realizable('Spec')")
               for line in window.project.journal)


def test_the_two_checks_are_one_at_a_time(window, pump):
    pane = _show(window, pump, 'Spec', _specification(phase_13=0.0))
    pane.realizability_action.trigger()
    pump()
    assert window._matrix_check['check'] == 'realizability'
    pane.conditioning_action.trigger()
    pump()
    assert not pane.realizability_action.isChecked()
    assert not pane.realizability_panel.isVisible()
    assert window._matrix_check['check'] == 'conditioning'
    pane.conditioning_action.trigger()
    pump()
    assert window._matrix_check is None
    assert not pane.conditioning_panel.isVisible()


def test_a_refusal_is_said_in_the_pane(window, pump):
    pane = _show(window, pump, 'Partial', _specification(unstated=True))
    pane.realizability_action.trigger()
    pump()
    panel = pane.realizability_panel
    assert panel.isVisible() and window._matrix_check is None
    assert '1Z+ and 3Z+ have no cross term' in panel.verdict.text()
    assert panel.derived['line'].text() == '—'
    assert not panel.apply_button.isEnabled()


def test_the_sheet_refuses_entries_that_cannot_hold(window, pump):
    import visualdynamics

    table = visualdynamics.ChannelTable({
        'channel': [1, 2, 3], 'node': ['101', '102', '103'],
        'direction': ['Z+'] * 3, 'unit': ['g'] * 3,
        'control': ['True'] * 3})
    window.add_object('Channels', table)
    window.show_object('Channels')
    pump()
    window.author_action.setChecked(True)
    window._author_toggled(True)
    pump()
    pump()
    panel = window.data_pane.author_panel
    panel.all_coherence.setValue(1.0)
    panel.all_phase.setValue(90.0)
    panel.all_button.click()
    pump()
    assert 'cannot hold together' in panel.problem.text()
    assert 'nearest entries that hold' in panel.problem.text()
    made = window.project['Channels Specification']
    assert made.num_records == 3, 'the refused entries landed nowhere'
    panel.all_phase.setValue(0.0)
    panel.all_button.click()
    pump()
    assert panel.problem.text() == ''
    assert window.project['Channels Specification'].num_records == 9


def test_the_reading_and_the_repair_run_off_the_window_thread(window, pump,
                                                              monkeypatch):
    """A repair of a large target is seconds: the reading and Make
    Realizable run on the worker, and a cursor moved or a drawing made
    again reads the reading held rather than making it again."""
    import threading

    from visualdynamics.core import matrix_checks

    threads = []
    original = matrix_checks.check_spectral_matrix

    def recorded(*args, **kwargs):
        threads.append(threading.current_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(matrix_checks, 'check_spectral_matrix', recorded)
    pane = _show(window, pump, 'Spec', _specification())
    pane.realizability_action.trigger()
    pump()
    assert len(threads) == 1
    assert threads[0] is not threading.main_thread()
    state = window._matrix_check
    state['cursor'].setValue(np.log10(400.0))
    pump()
    window.render_current()
    pump()
    assert len(threads) == 1, 'the reading held is read again'
    pane.realizability_panel.apply_button.click()
    pump()
    assert 'Spec Realizable' in window.project
    assert all(thread is not threading.main_thread() for thread in threads)
    assert len(threads) == 3, 'the repair, then the repaired reading'


def test_the_reading_follows_its_object(window, pump):
    """A reading held is let go when the object it was made of changes or
    another is shown: the pane reads what is selected now."""
    pane = _show(window, pump, 'Spec', _specification())
    pane.realizability_action.trigger()
    pump()
    panel = pane.realizability_panel
    assert 'not realizable' in panel.verdict.text()
    spec = window.project['Spec']
    spec.ordinate = _specification(phase_13=0.0).ordinate
    window.render_current()
    pump()
    assert 'realizable at all 100 lines' in panel.verdict.text()
    _show(window, pump, 'Other', _specification())
    assert 'not realizable at 100 of 100 lines' in panel.verdict.text()
    assert window._matrix_check['name'] == 'Other'


def test_a_cancelled_reading_says_so(window, pump, monkeypatch):
    from visualdynamics.core.progress import Cancelled

    def cancelled(self, *args, **kwargs):
        raise Cancelled('Reading the realizability')

    pane = _show(window, pump, 'Spec', _specification())
    monkeypatch.setattr(Specification, 'realizability', cancelled)
    pane.realizability_action.trigger()
    pump()
    panel = pane.realizability_panel
    assert panel.verdict.text() == 'No realizability read: cancelled'
    assert window._matrix_check is None
    assert window.statusBar().currentMessage() == 'Spec: realizability cancelled'
