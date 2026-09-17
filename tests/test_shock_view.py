"""The shock view: the button, the windows on the plot, the table.

The same shape as the averaging view, and tested the same way — the
parameters live on the history, the view shows and edits them, and the
calculator uses whatever is there when it runs.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import Srs, TimeHistory
from visualdynamics.core.shocks import Shock

pytestmark = pytest.mark.usefixtures('flat_reading')

@pytest.fixture
def shown(window, pump):
    """A shock run, selected, with the shock view up."""
    window.import_paths([fixture_path('plate', 'shock.nc4')])
    pump()
    name = next(n for n, obj in window.objects.items()
                if isinstance(obj, TimeHistory))
    window.show_object(name)
    pump()
    # importing a shock run makes it a Shock project, which already
    # opens the record as shocks — so ensure rather than toggle, or the
    # trigger turns off the very view this fixture is for
    if not window.data_pane.shocks_action.isChecked():
        window.data_pane.shocks_action.trigger()
    pump()
    return window, window.objects[name]


def overlays(window):
    return window.shock_overlays


# ---- the button ---------------------------------------------------------


def test_a_time_history_is_offered_the_shock_view(window, pump):
    window.import_paths([fixture_path('plate', 'shock.nc4')])
    pump()
    name = next(n for n, obj in window.objects.items()
                if isinstance(obj, TimeHistory))
    window.show_object(name)
    pump()
    assert window.data_pane.shocks_action.isVisible()


def test_a_spectrum_is_not(window, pump, survey):
    """The view describes stretches of a record. There are none in a
    frequency-domain object."""
    _shapes, frfs = survey
    window.add_object('FRF', frfs)
    window.show_object('FRF')
    pump()
    assert not window.data_pane.shocks_action.isVisible()


def test_the_button_puts_the_windows_up(shown):
    window, _history = shown
    assert overlays(window), 'an overlay per plot'
    assert window.data_pane.shock_panel.isVisible()


def test_the_button_takes_them_away_again(shown):
    window, _history = shown
    window.data_pane.shocks_action.trigger()
    window.data_pane.graphics.repaint()
    assert not overlays(window)


def test_the_choice_survives_moving_to_another_object(window, pump):
    """Sticky, like the averaging button: a view asked for once stays
    asked for until it is turned off — asked for by the user, since no
    project type pre-selects it any more (Brandon, 2026-08-30)."""
    window.import_paths([fixture_path('plate', 'shock.nc4')])
    pump()
    window.set_project_type('Shock')
    pump()
    name = next(n for n, obj in window.objects.items()
                if isinstance(obj, TimeHistory))
    window.show_object(name)
    pump()
    assert not window.data_pane.shocks_wanted, 'plain until asked'
    window.data_pane.shocks_action.trigger()
    pump()
    other = next(n for n in window.objects if n != name)
    window.show_object(other)
    pump()
    window.show_object(name)
    pump()
    assert window.data_pane.shocks_wanted, 'asked for once, still on'
    assert not window.data_pane.averaging_wanted


# ---- what it shows ------------------------------------------------------


def test_detect_fills_the_table_from_the_record(shown):
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    assert len(history.shocks) == 4
    assert window.data_pane.shock_panel.table.rowCount() == 4


def test_detect_reaches_the_plot_as_well_as_the_table(shown):
    window, _history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    for overlay in overlays(window):
        assert len(overlay.regions) == 4
        assert len(overlay.labels) == 4


def test_each_window_is_numbered(shown):
    """`shock 3` in a table of spectra means nothing until you can see
    which of the events on the trace it was."""
    window, _history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    overlay = overlays(window)[0]
    assert [text.toPlainText() for text in overlay.labels] == \
        ['1', '2', '3', '4']


def test_the_table_reads_what_the_record_holds(shown):
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    table = window.data_pane.shock_panel.table
    for row, shock in enumerate(history.shocks):
        assert float(table.item(row, 0).text()) == pytest.approx(
            shock.start, abs=1e-4)
        assert float(table.item(row, 1).text()) == pytest.approx(
            shock.duration, abs=1e-4)


# ---- editing ------------------------------------------------------------


def test_an_edit_in_the_table_reaches_the_record_and_the_plot(shown):
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    moved = (Shock(0.5, 0.25), Shock(1.5, 0.25))
    window.data_pane.shock_panel.changed.emit(moved)
    assert history.shocks == moved
    assert len(overlays(window)[0].regions) == 2


def test_a_drag_on_the_plot_reaches_the_record_and_the_table(shown):
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    overlay = overlays(window)[0]
    # the plumbing, not the mode: shared lengths have their own tests,
    # and this window widened that far would be capped by the room the
    # last event has left before the end of the record
    overlay.common = False
    overlay.regions[0].setRegion((0.10, 0.60))
    assert history.shocks[0].start == pytest.approx(0.10, abs=1e-6)
    assert history.shocks[0].duration == pytest.approx(0.50, abs=1e-6)
    assert float(window.data_pane.shock_panel.table.item(0, 0).text()) == \
        pytest.approx(0.10, abs=1e-4)


def test_a_typo_in_the_table_puts_the_old_value_back(shown):
    """A control that raises at a typo is a trap."""
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    was = history.shocks
    window.data_pane.shock_panel.table.item(0, 0).setText('not a number')
    assert history.shocks == was
    assert float(window.data_pane.shock_panel.table.item(0, 0).text()) == \
        pytest.approx(was[0].start, abs=1e-4)


def test_a_length_of_nothing_is_refused(shown):
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    was = history.shocks
    window.data_pane.shock_panel.table.item(0, 1).setText('0')
    assert history.shocks == was


def test_an_event_that_was_never_one_can_be_removed(shown):
    """A switch transient or a dropped cable is dropped here, rather
    than by moving a threshold until the detector agrees."""
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    panel = window.data_pane.shock_panel
    panel.table.selectRow(1)
    panel.remove_button.click()
    assert len(history.shocks) == 3


# ---- and what it is all for ---------------------------------------------


def test_the_calculator_computes_one_spectrum_per_channel_per_shock(shown):
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    window.tree.setCurrentItem(window._item_for_object(
        next(n for n, obj in window.objects.items() if obj is history)))
    window.compute_srs()
    added = next(obj for obj in window.objects.values()
                 if isinstance(obj, Srs))
    assert added.num_records == history.num_records * 4
    assert '4 shocks' in window.statusBar().currentMessage()


def test_the_calculator_detects_for_itself_when_nothing_was_asked(shown):
    """The view is a convenience, not a precondition. Computing an SRS
    from a history nobody looked at still answers."""
    window, history = shown
    assert not history.shocks
    window.tree.setCurrentItem(window._item_for_object(
        next(n for n, obj in window.objects.items() if obj is history)))
    window.compute_srs()
    assert history.shocks, 'it went and found them'
    assert any(isinstance(obj, Srs) for obj in window.objects.values())


def test_the_windows_are_saved_with_the_test(shown, tmp_path):
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    was = history.shocks
    visualdynamics.save(history, tmp_path / 'run')
    assert visualdynamics.load(tmp_path / 'run.vdyn').shocks == was


def test_an_unevenly_sampled_record_puts_nothing_up(window, pump):
    """There are no windows in a record with no sample rate. The view
    declines rather than raising — and the `no_swallowed_errors` fixture
    is what says it declined instead of throwing into a Qt slot."""
    uneven = TimeHistory(np.array([0.0, 0.1, 0.3]), np.ones((1, 3)),
                         response_dof=['101Z+'], ordinate_dim=['acceleration'])
    window.add_object('Uneven', uneven)
    window.show_object('Uneven')
    pump()
    window.data_pane.shocks_action.trigger()
    pump()
    assert not window.shock_overlays
    assert not window.data_pane.shock_panel.isVisible()


# ---- one reading of a record at a time ----------------------------------


def test_the_two_views_are_exclusive(shown):
    """They mark the same record with different readings of it — frames
    to average against events to analyze. A record is being read one way
    or the other, and overlaid they are two sets of shading on one
    trace."""
    window, _history = shown
    pane = window.data_pane
    assert pane.shocks_wanted and not pane.averaging_wanted
    pane.averaging_action.trigger()
    assert pane.averaging_wanted and not pane.shocks_wanted
    assert not pane.shocks_action.isChecked()
    pane.shocks_action.trigger()
    assert pane.shocks_wanted and not pane.averaging_wanted
    assert not pane.averaging_action.isChecked()


def test_turning_one_off_does_not_turn_the_other_on(shown):
    """Exclusive, not a pair of radio buttons: neither view is a state
    the record has to be in."""
    window, _history = shown
    pane = window.data_pane
    pane.shocks_action.trigger()          # off again
    assert not pane.shocks_wanted and not pane.averaging_wanted


def test_a_random_project_opens_a_record_plain(window, pump):
    """The default is the record itself, no reading pre-selected
    (Brandon, 2026-08-30) — the averaging view used to come up first,
    marks before the data had been looked at."""
    window.set_project_type('Random Vibration')
    assert not window.data_pane.averaging_wanted
    assert not window.data_pane.shocks_wanted


def test_importing_a_transient_run_opens_it_plain(window, pump):
    """The import sets the type, and the record opens plain — never on
    detection: a transient run's events are not found, they are known,
    the controller played one waveform over and over. The averaging
    view (whose frames are exactly those repeats) is a toggle away.
    """
    window.import_paths([fixture_path('plate', 'shock.nc4')])
    pump()
    assert window.project_type == 'Transient'
    assert not window.data_pane.averaging_wanted
    assert not window.data_pane.shocks_wanted


def test_a_shock_project_opens_its_records_plain_too(window, pump):
    """The last pre-selected reading to go (Brandon, 2026-08-30):
    detection is a toggle away, not a greeting."""
    window.set_project_type('Shock')
    assert not window.data_pane.shocks_wanted
    assert not window.data_pane.averaging_wanted


def test_another_type_leaves_both_alone(window, pump):
    """A modal test is read as neither, so neither is forced on."""
    window.data_pane.averaging_wanted = True
    window.set_project_type('Modal Test')
    assert window.data_pane.averaging_wanted, 'not overridden'


# ---- the report carries the windows too ---------------------------------


def _history_with_shocks():
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.shocks import Shock

    t = np.linspace(0.0, 3.0, 3072)
    rng = np.random.default_rng(2)
    values = 0.01 * rng.standard_normal((2, 3072))
    history = TimeHistory(t, values, response_dof=['101Z+', '102Z+'],
                          ordinate_dim=['acceleration'] * 2,
                          ordinate_unit=['m/s**2'] * 2)
    history.shocks = (Shock(0.5, 0.1), Shock(1.5, 0.1), Shock(2.5, 0.1))
    return history


def test_the_report_carries_the_shock_windows():
    """Brandon: the shock report showed the trace and not the bands the
    spectra came from. The SRS figures beside it are one curve per
    event, so 'shock 2' has to point at something findable."""
    import visualdynamics
    from visualdynamics.report import _plot_block

    history = _history_with_shocks()
    built = _plot_block({'kind': 'plot', 'source': 'T', 'mode': 'curves'},
                        history, {}, visualdynamics.SI)
    marks = built['shocks']
    assert len(marks['windows']) == 3
    assert marks['windows'][0] == [0.5, 0.6], 'start and end, in seconds'
    assert '3 shocks' in marks['label']


def test_a_history_with_no_shocks_carries_no_windows():
    import visualdynamics
    from visualdynamics.report import _plot_block

    history = _history_with_shocks()
    history.shocks = None
    built = _plot_block({'kind': 'plot', 'source': 'T', 'mode': 'curves'},
                        history, {}, visualdynamics.SI)
    assert 'shocks' not in built


def test_the_page_draws_what_the_payload_carries():
    """The canvas is JavaScript and cannot import anything, so the two
    halves drift silently — which is exactly how the MAC figure kept
    its old stripes for a week. This holds the drawing to the field."""
    from visualdynamics.report.page import _JS

    assert 'block.shocks' in _JS, 'the page reads the field'
    assert 'block.shocks.windows' in _JS, 'and draws the windows'


# ---- one length for the series ------------------------------------------


def test_a_drag_widens_every_window_when_the_lengths_are_shared(shown):
    """Brandon's question. Pull any edge and they all follow."""
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    overlay = overlays(window)[0]
    assert overlay.common, 'detection gives one length, so drags share it'
    was = history.shocks
    assert len(was) > 2

    wider = was[0].duration + 0.05
    overlay.regions[0].setRegion((was[0].start, was[0].start + wider))
    assert len({round(s.duration, 6) for s in history.shocks}) == 1
    assert history.shocks[-1].duration == pytest.approx(wider, abs=1e-6), \
        'the window nobody touched took the new length too'


def test_they_widen_only_until_one_runs_into_another(shown):
    """'...until one runs into another' — the tightest gap is the
    ceiling for the whole series, not just for the pair that met."""
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    overlay = overlays(window)[0]
    was = history.shocks

    overlay.regions[0].setRegion((was[0].start, was[0].start + 10.0))
    now = history.shocks
    assert len({round(s.duration, 6) for s in now}) == 1, 'still one length'
    for one, next_one in pairwise(now):
        assert one.stop <= next_one.start + 1e-9, 'and none overlaps'
    assert now[0].duration < 10.0, 'the drag was capped'


def test_moving_a_window_moves_only_that_one(shown):
    """Brandon: dragging an event window along the record must not
    take the others with it. A start belongs to the event it was
    measured from; only the length is ever shared."""
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    overlay = overlays(window)[0]
    assert overlay.common, 'and this holds even in shared-length mode'
    was = history.shocks

    overlay.regions[1].setRegion((was[1].start + 0.03, was[1].stop + 0.03))
    now = history.shocks
    assert now[1].start == pytest.approx(was[1].start + 0.03, abs=1e-6)
    assert now[1].duration == pytest.approx(was[1].duration, abs=1e-6)
    assert now[0] == was[0] and now[2:] == was[2:], 'nothing else moved'

    # and where it bites: a move that closes a gap to less than the
    # shared length. Routed as a resize this would hand the tightened
    # gap to the whole series and shorten every window in the record,
    # so it is the state that tells the two apart.
    window.data_pane.shock_panel.detect_asked.emit()
    was = history.shocks
    close = was[2].start - was[1].duration / 2.0
    overlay.regions[1].setRegion((close, close + was[1].duration))
    now = history.shocks
    assert now[1].stop <= now[2].start + 1e-9, 'held off its neighbor'
    assert now[1].duration < was[1].duration, 'the one that moved gave way'
    assert now[0].duration == pytest.approx(was[0].duration, abs=1e-9), \
        'and no other window was shortened by it'
    assert now[3].duration == pytest.approx(was[3].duration, abs=1e-9)


def test_resizing_shares_the_length_without_moving_any_start(shown):
    """The other half of the split: a resize is shared, a move is not,
    and a resize still leaves every start where its event put it."""
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    overlay = overlays(window)[0]
    was = history.shocks

    overlay.regions[0].setRegion((was[0].start, was[0].stop + 0.04))
    now = history.shocks
    assert len({round(s.duration, 6) for s in now}) == 1, 'length shared'
    assert [s.start for s in now] == [s.start for s in was], 'starts kept'


def test_per_event_windows_are_dragged_one_at_a_time(shown):
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    overlay = overlays(window)[0]
    overlay.common = False
    was = history.shocks

    overlay.regions[0].setRegion((was[0].start, was[0].start + 0.05))
    assert history.shocks[0].duration == pytest.approx(0.05, abs=1e-6)
    assert history.shocks[1:] == was[1:], 'the others stayed where they were'


def test_the_box_reads_the_lengths_rather_than_a_stored_flag(shown):
    window, _history = shown
    panel = window.data_pane.shock_panel
    panel.detect_asked.emit()
    assert panel.same_length.isChecked(), 'detection gives one length'

    panel.changed.emit((Shock(0.2, 0.1), Shock(1.0, 0.3)))
    panel.set_shocks((Shock(0.2, 0.1), Shock(1.0, 0.3)))
    assert not panel.same_length.isChecked(), 'these two disagree'


def test_ticking_the_box_makes_the_windows_agree(shown):
    window, history = shown
    panel = window.data_pane.shock_panel
    panel.detect_asked.emit()
    mixed = (Shock(0.2, 0.1), Shock(1.0, 0.3), Shock(2.0, 0.2))
    panel.changed.emit(mixed)
    panel.set_shocks(mixed)
    assert not panel.same_length.isChecked()

    panel.same_length.setChecked(True)
    assert len({round(s.duration, 6) for s in history.shocks}) == 1
    assert history.shocks[0].duration == pytest.approx(0.3), 'the longest'


# ---- windows are held off each other ------------------------------------


def test_a_dragged_window_cannot_be_pulled_over_its_neighbor(shown):
    """`_windowed` promised no two windows share a sample, and until
    now nothing kept that promise once the plot could edit them: a
    window pulled back over the event before it would have had its
    spectrum computed from that event's ringdown."""
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    overlay = overlays(window)[0]
    overlay.common = False
    was = history.shocks

    # shock 2's left edge, dragged back well inside shock 1's window
    overlay.regions[1].setRegion((was[0].start + 0.01, was[1].stop))
    now = history.shocks
    assert now[1].start >= now[0].stop - 1e-9, 'stopped at the edge'
    assert now[0] == was[0], 'and shock 1, which nobody touched, did not move'


def test_a_typed_window_cannot_overlap_either(shown):
    """The same edit arriving by the other route."""
    window, history = shown
    panel = window.data_pane.shock_panel
    panel.detect_asked.emit()
    panel.same_length.setChecked(False)
    was = history.shocks

    panel.table.item(1, 0).setText(f'{was[0].start + 0.01:.4f}')
    now = history.shocks
    assert now[1].start >= now[0].stop - 1e-9


# ---- the report's Figures 2 and 3 ---------------------------------------


def _bounded_spec_and_events():
    """A one-channel shock specification with its band, and a measured
    SRS carrying two events of that channel."""
    from visualdynamics.core.data import ShockSpecification, Srs

    frequencies = np.geomspace(10.0, 1000.0, 40)
    level = np.full(40, 100.0)
    spec = ShockSpecification(
        abscissa=frequencies, ordinate=level[None, :],
        response_dof=['101Z+'], ordinate_dim='acceleration',
        ordinate_unit='m/s**2',
        abort_upper=level[None, :] * 2.0,
        abort_lower=level[None, :] * 0.5)
    measured = Srs(
        abscissa=frequencies,
        ordinate=np.vstack([level * 0.8, level * 1.1]),
        response_dof=['101Z+', '101Z+'],
        ordinate_dim=['acceleration'] * 2,
        ordinate_unit=['m/s**2'] * 2,
        block=['shock 1', 'shock 2'])
    return spec, measured


def test_figure_2_carries_the_tolerance_band():
    """Brandon: Figure 2 does not look like it shows the SRS limits.
    The report gated the band on `Specification` — the PSD kind — and
    a shock specification is `Bounded, Srs`: same four limit curves,
    different spectrum underneath. The gate reads the mixin now."""
    import visualdynamics
    from visualdynamics.report import _plot_block

    spec, _measured = _bounded_spec_and_events()
    built = _plot_block({'kind': 'plot', 'source': 'S', 'mode': 'curves'},
                        spec, {}, visualdynamics.SI)
    zones = built['channels'][0]['zones']
    assert zones, 'the band is in the payload'
    assert {z['severity'] for z in zones} == {'abort'}, \
        'this specification wrote abort limits and nothing else'


def test_figure_3_is_the_events_over_the_specification():
    """Brandon: Figure 3 should show the control channel SRS on top of
    the specification — the reading the app gives when the two are
    selected together. Every event of the channel, the target and its
    band behind, no comparison scaling and no outside marks: a
    spectrum sitting low is the finding, and with several events
    overlaid a mark cannot say whose line went out."""
    import visualdynamics
    from visualdynamics.report import _plot_block

    spec, measured = _bounded_spec_and_events()
    built = _plot_block(
        {'kind': 'plot', 'source': 'M', 'mode': 'curves',
         'specification': 'S'},
        measured, {'S': spec}, visualdynamics.SI)
    channel = built['channels'][0]
    assert [r['label'] for r in channel['responses']] == \
        ['shock 1', 'shock 2'], 'every event, named as the app names it'
    assert channel['zones'], 'the band rides behind them'
    assert 'over' not in channel and 'under' not in channel, \
        'no outside marks over stacked events'
    assert built['curves'][0].get('gray'), 'the target is the reference'


def test_the_shock_template_names_the_specification():
    import visualdynamics
    from visualdynamics.core.report import shock_template

    project = visualdynamics.Project()
    report = shock_template(project, links=[])
    srs_blocks = [b for b in report.blocks
                  if b.get('source') == '@basis:Srs'
                  and b.get('kind') == 'plot']
    assert srs_blocks and srs_blocks[0].get('specification') == \
        '@basis:ShockSpecification'


def test_the_page_draws_the_response_list():
    """The canvas cannot import the field name — the same seam the
    shock windows and the MAC checker drifted through."""
    from visualdynamics.report.page import _JS

    assert 'ch.responses' in _JS


# ---- adding a window the detector missed ---------------------------------


def test_with_added_places_a_window_in_the_largest_gap():
    from visualdynamics.core.shocks import with_added

    grown = with_added((Shock(1.0, 0.5), Shock(4.0, 0.5)), 10.0)
    assert len(grown) == 3
    new = next(s for s in grown if s.start not in (1.0, 4.0))
    assert new.duration == pytest.approx(0.5), "the series' own length"
    assert 4.5 < new.start and new.stop < 10.0, \
        'in the stretch after the last window — the largest gap'
    assert new.start == pytest.approx(4.5 + (5.5 - 0.5) / 2), 'centered'


def test_with_added_serves_the_empty_list():
    from visualdynamics.core.shocks import with_added

    (only,) = with_added((), 20.0)
    assert only.duration == pytest.approx(2.0), 'a tenth of the record'
    assert (only.start + only.stop) / 2 == pytest.approx(10.0), 'centered'


def test_with_added_shrinks_into_a_tight_gap():
    from visualdynamics.core.shocks import with_added

    grown = with_added((Shock(0.0, 4.0), Shock(5.0, 5.0)), 10.0)
    new = [s for s in grown if s.duration != 4.0 and s.duration != 5.0]
    assert len(new) == 1 and new[0].duration == pytest.approx(0.9), \
        'shrunk to nine tenths of the only gap'
    assert new[0].start >= 4.0 and new[0].stop <= 5.0


def test_with_added_refuses_a_wall_to_wall_record():
    from visualdynamics.core.shocks import with_added

    packed = (Shock(0.0, 5.0), Shock(5.0, 5.0))
    assert with_added(packed, 10.0) == packed


def test_the_add_button_grows_the_series(shown):
    window, history = shown
    window.data_pane.shock_panel.detect_asked.emit()
    was = len(history.shocks)
    panel = window.data_pane.shock_panel
    assert panel.add_button.isEnabled()
    panel.add_button.click()
    assert len(history.shocks) == was + 1, 'one more window, stored'
    assert panel.table.rowCount() == was + 1, 'and on the panel'
    starts = [s.start for s in history.shocks]
    assert starts == sorted(starts), 'the series stays in record order'


def test_add_is_offered_to_an_empty_list_but_not_a_locked_one(window,
                                                              pump):
    """Remove needs a row to act on; Add is most needed when there are
    none. A locked record offers neither."""
    panel = window.data_pane.shock_panel
    panel.set_shocks(())
    assert panel.add_button.isEnabled()
    assert not panel.remove_button.isEnabled()
    panel.set_shocks((Shock(1.0, 0.5),), locked=True)
    assert not panel.add_button.isEnabled()


def test_the_panel_computes_the_srs_over_these_windows(shown):
    """The act where its settings are set (Brandon, 2026-08-28): the
    same verb the calculator offers, from the panel that edits the
    windows it reads."""
    window, _history = shown
    window.data_pane.shock_panel.srs_button.click()
    name = next(n for n, obj in window.objects.items()
                if type(obj).__name__ == 'Srs')
    assert window.project.provenance[name]['verb'] == 'compute_srs'


def test_the_panel_carries_the_srs_analysis_choices(shown):
    """Q, which peak, and the line spacing — the whole parameter
    space (there is no mass setting in the standard SRS: the SDOF's
    mass cancels for base-input response). Parameters of the act,
    recorded in its recipe, so a refresh after the windows move
    recomputes at the same choices (Brandon, 2026-08-29)."""
    from visualdynamics.core.srs import DEFAULT_Q, PER_OCTAVE

    window, history = shown
    panel = window.data_pane.shock_panel
    assert panel.q_box.value() == DEFAULT_Q
    assert panel.per_octave_box.value() == PER_OCTAVE
    assert panel.kind_box.currentData() == 'maximax'
    low, high = history.srs_band()
    assert panel.band_label.text() == f'{low:.4g}–{high:.4g} Hz', \
        'the band the windows support, stated beside the settings'

    panel.q_box.setValue(25.0)
    panel.kind_box.setCurrentIndex(panel.kind_box.findData('positive'))
    panel.per_octave_box.setValue(6)
    panel.srs_button.click()
    name = next(n for n, r in window.project.provenance.items()
                if r['verb'] == 'compute_srs')
    srs = window.objects[name]
    assert srs.q == 25.0 and srs.kind == 'positive'
    assert window.project.provenance[name]['params'] == \
        {'per_octave': 6, 'q': 25.0, 'kind': 'positive'}

    # the recipe survives a refresh: move a window and recompute
    from visualdynamics.core.shocks import Shock

    history.shocks = tuple(Shock(s.start, s.duration * 0.9)
                           for s in history.shocks)
    assert name in window.project.stale()
    window.project.refresh(name)
    refreshed = window.project[name]
    assert refreshed.q == 25.0 and refreshed.kind == 'positive', \
        'the same choices, over the moved windows'


# ---- one event against the target ---------------------------------------


def _pick_event(window, pump, records):
    """Records picked in the grid, the measured row selected beside the
    target's — and the target's row *current*: making an owner's row
    current releases its grid picks (`_selection_changed`), which is
    what a click on the object itself has always meant."""
    window.tree.clearSelection()
    window.tree.setCurrentItem(window._item_for_object('Target'))
    item = window._item_for_object('Measured')
    item.setExpanded(True)
    pump()
    window.record_grids['Measured'].select_records(records)
    item.setSelected(True)
    window._item_for_object('Target').setSelected(True)
    window.render_current()
    pump()


def test_one_picked_event_is_drawn_against_the_target(window, pump):
    """Brandon, 2026-09-11: selecting one SRS event beside the shock
    specification could not make the plot in 2-D or 3-D — the render
    asked a shock spectrum for a time history's channel key and
    crashed. A pick of an event's records is that event against the
    target: the curves, the stage, the bars and the table all narrow
    to it; the whole object beside the target is still every event."""
    spec, measured = _bounded_spec_and_events()
    window.add_object('Target', spec)
    window.add_object('Measured', measured)
    pump()
    window.data_pane.waterfall_action.setChecked(False)
    _pick_event(window, pump, [1])                  # shock 2 alone
    plot = next(item for item in window.data_pane.graphics.ci.items
                if hasattr(item, 'listDataItems'))
    names = [c.name() for c in plot.listDataItems() if c.name()]
    assert names == ['Measured: 101Z+ shock 2', 'Specification: 101Z+']
    assert '101Z+: 1 event against the one required' in \
        window.statusBar().currentMessage()
    from PySide6.QtCore import Qt

    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Horizontal,
                                Qt.ItemDataRole.DisplayRole)
               for c in range(model.columnCount())]
    assert headers == ['Channel', 'Event 1 [dB]'], 'the table reads the pick: one event'
    assert model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole) == '0.8279', \
        "shock 2's own deviation, not shock 1's"
    # the stage draws the picked record only
    banded = []
    real = window._render_banded
    window._render_banded = lambda *a, **k: (banded.append((a, k)), real(*a, **k))[1]
    window.data_pane.waterfall_action.setChecked(True)
    _pick_event(window, pump, [1])
    (args, _kw), = banded[-1:]
    assert args[5] == [1] and args[2] == [0], \
        'measured record 1 against target record 0, nothing else'
    # the whole object: every event, as before
    window._render_banded = real
    window.data_pane.waterfall_action.setChecked(False)
    _pick_event(window, pump, [])
    plot = next(item for item in window.data_pane.graphics.ci.items
                if hasattr(item, 'listDataItems'))
    names = [c.name() for c in plot.listDataItems() if c.name()]
    assert names == ['Measured: 101Z+ shock 1', 'Measured: 101Z+ shock 2',
                     'Specification: 101Z+']
    assert 'all 2 events against the one required' in \
        window.statusBar().currentMessage()
