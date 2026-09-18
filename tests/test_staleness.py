"""Derived objects know when their source's settings have moved.

Brandon's design (2026-08-23): recomputing is an act, never
automatic — a report must not rewrite itself — so a settings change
marks the downstream objects stale, a refresh badge says why, and one
click recomputes in place. Fingerprints of the settings, never
timestamps: the badge appears exactly when the numbers would differ,
and survives a save and a load.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import fixture_path

import visualdynamics


@pytest.fixture
def worked_up():
    project = visualdynamics.Project('t')
    project.import_file(fixture_path('plate', 'modal.nc4'))
    psds = project.compute_psds('Time History')
    return project, psds


def test_fresh_is_fresh(worked_up):
    project, _psds = worked_up
    assert project.stale() == {}


def test_a_framing_change_marks_the_psds(worked_up):
    project, psds = worked_up
    history = project['Time History']
    history.averaging = replace(history.averaging, frames=3)
    stale = project.stale()
    assert list(stale) == [psds]
    assert 'now says 3 frames' in stale[psds]


def test_a_shock_change_marks_the_srs_and_only_the_srs(worked_up):
    from visualdynamics.core.shocks import Shock

    project, psds = worked_up
    history = project['Time History']
    history.shocks = (Shock(0.1, 0.2),)
    srs = project.compute_srs('Time History')
    assert project.stale() == {}, 'freshly computed'
    history.shocks = (Shock(0.1, 0.2), Shock(0.6, 0.2))
    stale = project.stale()
    assert srs in stale and psds not in stale, \
        'the fingerprint follows the dependency: shocks never mark PSDs'
    assert '2 windows' in stale[srs]


def test_refresh_recomputes_in_place_and_references_hold(worked_up):
    project, psds = worked_up
    project.link('Time History', psds)
    history = project['Time History']
    history.averaging = replace(history.averaging, frames=4)
    before = project[psds]
    project.refresh(psds)
    assert project.stale() == {}
    after = project[psds]
    assert after is not before, 'genuinely recomputed'
    assert psds in project, 'under its own name'
    assert any(psds in group['members'] for group in project.links), \
        'the link held through the refresh'
    assert len(after.abscissa) == history.averaging.frame_length // 2 + 1


def test_the_cascade_flows_by_content(worked_up):
    project, psds = worked_up
    octave = project.compute_octave(psds)
    history = project['Time History']
    history.averaging = replace(history.averaging, frames=6)
    assert list(project.stale()) == [psds], \
        'the octave banding is honest until its own source changes'
    project.refresh(psds)
    assert list(project.stale()) == [octave], \
        'refreshing the PSDs is what stales the banding of them'
    project.refresh(octave)
    assert project.stale() == {}


def test_refresh_stale_settles_the_whole_chain(worked_up):
    project, psds = worked_up
    octave = project.compute_octave(psds)
    history = project['Time History']
    history.averaging = replace(history.averaging, frames=8)
    done = project.refresh_stale()
    assert set(done) == {psds, octave}
    assert project.stale() == {}


def test_staleness_survives_the_file(worked_up, tmp_path):
    project, psds = worked_up
    history = project['Time History']
    history.averaging = replace(history.averaging, frames=3)
    path = str(tmp_path / 't.vdyn')
    project.save(path)
    back = visualdynamics.Project.open(path)
    stale = back.stale()
    assert list(stale) == [psds]
    back.refresh(psds)
    assert back.stale() == {}


def test_a_rename_follows_into_the_provenance(worked_up):
    project, psds = worked_up
    project.rename('Time History', 'Recording')
    project.rename(psds, 'Densities')
    history = project['Recording']
    history.averaging = replace(history.averaging, frames=3)
    assert list(project.stale()) == ['Densities']
    project.refresh('Densities')
    assert project.stale() == {}


def test_a_deleted_source_is_not_stale(worked_up):
    project, psds = worked_up
    del project['Time History']
    assert project.stale() == {}, \
        'absence of evidence is not staleness'
    with pytest.raises(ValueError, match='gone'):
        project.refresh(psds)


def test_the_frf_refresh_keeps_its_estimator(worked_up):
    project, _psds = worked_up
    frfs = project.compute_frfs('Time History', 'H1')
    history = project['Time History']
    history.averaging = replace(history.averaging, frames=3)
    assert frfs in project.stale()
    project.refresh(frfs)
    assert frfs not in project.stale(), \
        'refreshed (the PSDs beside it stay honestly stale)'
    assert 'H1' in frfs, 'the choice made once is the choice reused'


# ---- the badge ----------------------------------------------------------


def test_the_badge_appears_when_the_panel_moves(window, pump):
    window.import_paths([fixture_path('plate', 'modal.nc4')])
    pump()
    name = window.project.compute_psds('Time History')
    window.show_object(name)
    pump()
    item = window._item_for_object(name)
    # fresh, a PSD wears no icon at all now: its one act (octave
    # banding) lives on the bar (Brandon, 2026-08-29), so the column
    # is empty until staleness has something to say
    assert item.toolTip(1) == '', 'fresh: nothing to click'
    # open the averaging view and move the framing
    history_item = window._item_for_object('Time History')
    window.tree.clearSelection()
    history_item.setSelected(True)
    window.tree.setCurrentItem(history_item)
    pump()
    pane = window.data_pane
    pane.averaging_action.setChecked(True)
    pane.averaging_wanted = True
    window.render_current()
    pump()
    pane.averaging_panel.frames_box.setValue(4)
    pump()
    assert name in window._stale
    assert 'Settings changed' in item.toolTip(1), \
        'the badge says why, right on the row'
    # the badge's click-through: refresh in place
    window.refresh_object(name)
    pump()
    assert window._stale == {}
    assert item.toolTip(1) == '', 'fresh again: the badge stood down'


def test_a_loaded_projects_provenance_reaches_the_window(window, pump,
                                                         tmp_path):
    """Opening a .vdyn through the window absorbs its objects into the
    window's own project — and used to drop the provenance on the
    floor, so a freshly regenerated demo opened with no staleness
    bookkeeping at all (Brandon: 'still no refresh button')."""
    project = visualdynamics.Project('t')
    project.import_file(fixture_path('plate', 'modal.nc4'))
    psds = project.compute_psds('Time History')
    path = str(tmp_path / 't.vdyn')
    project.save(path)

    window.import_paths([path])
    pump()
    assert psds in window.project.provenance, \
        'the bookkeeping crossed the absorb, renamed like the links'
    item = window._item_for_object('Time History')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    pane = window.data_pane
    pane.averaging_action.setChecked(True)
    pane.averaging_wanted = True
    window.render_current()
    pump()
    pane.averaging_panel.frames_box.setValue(4)
    pump()
    assert psds in window._stale, 'the badge follows a loaded file too'
    badge = window._item_for_object(psds)
    assert 'Settings changed' in badge.toolTip(1)


def test_the_badge_click_recomputes_without_a_menu(window, pump,
                                                   monkeypatch):
    """The badge is one verb, so the click is that verb — a menu
    between them was a question with one answer (Brandon). The menu
    is intercepted at the window's own seam: QMenu.exec is a C++
    slot no monkeypatch reaches, and a popup under offscreen never
    returns — hanging, not failing."""
    from dataclasses import replace

    popped = []
    monkeypatch.setattr(window, '_pop_menu',
                        lambda menu: popped.append(menu))
    window.import_paths([fixture_path('plate', 'modal.nc4')])
    pump()
    name = window.project.compute_psds('Time History')
    window.show_object(name)
    pump()
    history = window.objects['Time History']
    history.averaging = replace(history.averaging, frames=4)
    window._refresh_stale_badges()
    assert name in window._stale
    item = window._item_for_object(name)
    window._tree_item_clicked(item, 1)
    pump()
    assert not popped, 'no menu stood between the badge and the verb'
    assert window._stale == {}, 'the click was the recompute'
    # fresh again, the same click is nothing at all: a PSD's one act
    # lives on the bar now, so there is no menu to come back
    # (Brandon, 2026-08-29)
    window._tree_item_clicked(item, 1)
    assert not popped, 'no calculator menu on a verbless object'


def test_the_open_report_follows_a_refresh(window, pump, monkeypatch):
    """Figure 5 kept the old averages through a refresh: the editor's
    guard re-renders only when the Report object or the unit system
    changes — right for selection churn, wrong after a recompute,
    where the same Report must draw the new numbers (Brandon)."""
    from dataclasses import replace

    window.import_paths([fixture_path('plate', 'modal.nc4')])
    pump()
    name = window.project.compute_psds('Time History')
    window.show_object(name)
    window.project.generate_report('modal')
    window.show_object('Report')
    pump()

    class FakeEditor:
        stale = False
        failure = None

        def isHidden(self):
            return False
        report = window.objects['Report']
        rebuilds = 0
        # the window's own system, or _render_report_builder's
        # units-changed branch calls rebuild and the test passes
        # for the wrong reason — it did
        unit_system = window.unit_system
        def rebuild(self):
            FakeEditor.rebuilds += 1
        def hide(self):
            pass
        def show(self):
            pass
    monkeypatch.setattr(window, 'report_editor', FakeEditor())

    history = window.objects['Time History']
    history.averaging = replace(history.averaging, frames=4)
    window._refresh_stale_badges()
    window.refresh_object(name)
    pump()
    assert FakeEditor.rebuilds >= 1, \
        'the recompute re-rendered the open report'


def test_a_fingerprint_from_before_a_rename_compares_clean(worked_up):
    """The other way a fingerprint drifts without the settings moving:
    a *rename*. 'rectangle' became 'boxcar' (the scipy spelling), the
    loader normalizes the object's averaging through the class, and
    every project saved before the rename opened with a refresh badge
    whose story named the same window twice (Brandon, 2026-08-30 —
    modal.vdyn did exactly this). Legacy fingerprints go through the
    class's own normalization, positional shape and old spelling both."""
    project, psds = worked_up
    averaging = project['Time History'].averaging
    # what an old file recorded: positional values, pre-rename spelling,
    # fields added since absent — the same computation in old clothes
    project.provenance[psds]['state'] = [
        'averaging', [averaging.frame_length, averaging.overlap,
                      'rectangle', averaging.frames, averaging.start]]
    assert averaging.window == 'boxcar', 'the live object is normalized'
    assert project.stale() == {}, \
        'same computation, older vocabulary: no badge'
