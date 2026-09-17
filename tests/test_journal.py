"""The session journal: every act, as the line that replays it.

Piece one of the console (Brandon, 2026-08-30): the project records
its own verbs — clicked in the GUI or called from a script, they are
the same calls — as runnable Python, in memory, for this sitting. The
anchor test replays `session_script` with exec and compares projects;
it is what keeps the journal honest forever.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics


def _worked_up():
    project = visualdynamics.Project('Journaled')
    project.import_file(fixture_path('plate', 'modal.nc4'))
    project.compute_psds('Time History')
    project.rename('Time History PSDs', 'PSDs')
    return project


def test_each_verb_records_one_runnable_line():
    project = _worked_up()
    assert project.journal[0] == "project = visualdynamics.Project('Journaled')"
    assert project.journal[1] == \
        f"project.import_file({fixture_path('plate', 'modal.nc4')!r})"
    assert "project.compute_psds('Time History')" in project.journal
    assert "project.rename('Time History PSDs', 'PSDs')" in project.journal


def test_a_verb_calling_verbs_records_once():
    """merge removes, adds and links; the journal says merge."""
    project = _worked_up()
    before = len(project.journal)
    said = [line for line in project.journal if 'add(' in line]
    assert not said, 'the adds inside import and compute stayed quiet'
    assert len(project.journal) == before


def test_an_object_argument_records_as_its_name():
    """`project.compute_psds(project.basis.time_history)` is how the
    console will often be driven; the journal writes the name the verb
    resolved, which is what a script would say."""
    project = visualdynamics.Project('N')
    project.import_file(fixture_path('plate', 'modal.nc4'))
    project.set_basis(*project.names)
    project.compute_psds(project.basis.time_history)
    assert "project.compute_psds('Time History')" in project.journal


def test_a_refusal_records_nothing():
    project = visualdynamics.Project('R')
    before = list(project.journal)
    with pytest.raises(KeyError):
        project.compute_psds('nothing here')
    assert project.journal == before, 'a verb that raised changed nothing'


def test_opening_a_file_is_the_whole_genesis(tmp_path):
    project = _worked_up()
    path = project.save(tmp_path / 'j.vdyn')
    back = visualdynamics.Project.open(path)
    assert back.journal == [
        f'project = visualdynamics.Project.open({path!r})'], \
        'whatever the loader did on the way, the story starts here'


def test_the_script_replays_the_session(tmp_path):
    """The anchor: exec the script, get the same project."""
    project = _worked_up()
    project.save(tmp_path / 'out.vdyn')
    script = project.session_script()
    assert script.startswith('import visualdynamics\n')

    room: dict = {}
    exec(script, room)                                  # noqa: S102
    replayed = room['project']
    assert replayed.names == project.names
    for name in project.names:
        ours, theirs = project[name], replayed[name]
        assert type(ours) is type(theirs), name
        if hasattr(ours, 'ordinate'):
            assert np.allclose(np.asarray(ours.ordinate),
                               np.asarray(theirs.ordinate)), name
    assert (tmp_path / 'out.vdyn').exists(), \
        'the save replayed too — the script writes what the session wrote'


# ---- piece two: settings writes through the funnel -----------------------


def test_a_setting_records_as_the_assignment_a_script_makes():
    from visualdynamics.core.averaging import Averaging

    project = visualdynamics.Project('S')
    project.import_file(fixture_path('plate', 'modal.nc4'))
    framing = Averaging(frame_length=4096, frames=6, window='hann')
    project['Time History'].averaging = framing
    project.record_setting(project['Time History'], 'averaging', framing)
    assert project.journal[-1] == \
        f"project['Time History'].averaging = {framing!r}"


def test_nudging_a_setting_settles_to_one_line():
    """A drag session is many writes to the same slot; the journal
    keeps the assignment that stands, not a line per pixel."""
    from visualdynamics.core.truncate import Truncation

    project = visualdynamics.Project('S')
    project.import_file(fixture_path('plate', 'modal.nc4'))
    history = project['Time History']
    before = len(project.journal)
    for stop in (0.5, 0.4, 0.3):
        project.record_setting(history, 'truncation', Truncation(0.0, stop))
    assert len(project.journal) == before + 1
    assert 'Truncation(start=0.0, stop=0.3)' in project.journal[-1]


def test_the_script_replays_the_settings_too(tmp_path):
    """The anchor, piece-two edition: a framing stored by assignment
    shapes the PSDs the replay computes."""
    from visualdynamics.core.averaging import Averaging

    project = visualdynamics.Project('S')
    project.import_file(fixture_path('plate', 'modal.nc4'))
    history = project['Time History']
    framing = Averaging(frame_length=4096, frames=6, window='hann')
    history.averaging = framing
    project.record_setting(history, 'averaging', framing)
    project.compute_psds('Time History')

    script = project.session_script()
    assert 'from visualdynamics.core.averaging import Averaging' in script
    room: dict = {}
    exec(script, room)                                  # noqa: S102
    replayed = room['project']
    assert replayed['Time History'].averaging == framing
    assert np.allclose(
        np.asarray(project['Time History PSDs'].ordinate),
        np.asarray(replayed['Time History PSDs'].ordinate)), \
        'the framing shaped the replayed PSDs exactly'


def test_the_gui_panels_write_through_the_funnel(window, pump):
    """The six GUI assignment sites — averaging and truncation, edit
    and drag, the filter panel, the shock windows — all echo. Exercised
    through the window's own handlers, the way the panels fire them."""
    import numpy as np

    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.filters import Filtering
    from visualdynamics.core.truncate import Truncation

    t = np.arange(8192) / 2048.0
    history = visualdynamics.TimeHistory(
        t, np.random.default_rng(3).standard_normal((2, len(t))),
        response_dof=['101Z+', '104Z+'], ordinate_dim='acceleration')
    window.add_object('Run', history)
    item = window._item_for_object('Run')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()

    window._averaging_edited(Averaging(frame_length=1024, frames=4))
    window._filtering_edited(Filtering(high=200.0))
    window._truncation_dragged(Truncation(0.5, 2.0))
    said = '\n'.join(window.project.journal)
    assert "project['Run'].averaging = Averaging(" in said
    assert "project['Run'].filtering = Filtering(" in said
    assert "project['Run'].truncation = Truncation(" in said


# ---- the GUI import journals the line a script would say ------------------


def test_a_gui_import_journals_as_import_file(window, pump):
    """The window imports a file by building the objects and adding
    them one by one; journaled verb by verb that was a pile of
    not-replayable comments (Brandon, 2026-08-30, asking why anything
    is not replayable) — when the honest record is the one call a
    script would make."""
    path = fixture_path('plate', 'modal_spectra.nc4')
    window.import_paths([path])
    pump()
    said = window.project.journal
    assert f'project.import_file({path!r})' in said
    assert not any(line.startswith('#') for line in said), \
        'no comment lines: everything this session did replays'

    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    assert set(room['project'].names) == set(window.project.names), \
        'the replayed import produced the same objects'


def test_opening_a_vdyn_into_a_fresh_window_is_the_genesis(window, pump,
                                                           tmp_path):
    project = visualdynamics.Project('Saved')
    project.import_file(fixture_path('plate', 'modal.nc4'))
    path = project.save(tmp_path / 'saved.vdyn')
    window.import_paths([path])
    pump()
    assert window.project.journal == [
        f'project = visualdynamics.Project.open({path!r})'], \
        'a project opened into a fresh window IS the session genesis'


def test_naming_units_journals_the_define_units_call(window, pump):
    """The pane writes unit by unit, each write converting its record;
    the journal records one define_units with everything currently
    declared, settling in place as the declarations grow (Brandon,
    2026-08-30: naming units said nothing in the console) — and the
    replay lands the same converted values."""
    path = fixture_path('plate', 'frfs.npz')
    window.import_paths([path])
    pump()
    name = next(n for n, o in window.objects.items()
                if hasattr(o, 'reference_dof'))
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    window.define_units()
    pump()
    model = window.units_table.model()
    unit_column = model.columnCount() - 1
    from PySide6.QtCore import Qt

    model.setData(model.index(0, unit_column), 'm/s**2',
                  Qt.ItemDataRole.EditRole)
    window._restate_after_units()
    model.setData(model.index(1, unit_column), 'm/s**2',
                  Qt.ItemDataRole.EditRole)
    window._restate_after_units()
    # an FRF declares per side, and nothing lands until a record has
    # both halves — the reference too, or the journal rightly says
    # nothing yet
    reference = window.units_reference_table.model()
    for row in range(reference.rowCount()):
        reference.setData(
            reference.index(row, reference.columnCount() - 1), 'N',
            Qt.ItemDataRole.EditRole)
        window._restate_after_units()
    pump()
    lines = [line for line in window.project.journal
             if 'define_units' in line]
    assert len(lines) == 1, 'the declarations settle into one line'
    assert lines[0].startswith(f'project[{name!r}].define_units(')

    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    replayed = room['project'][name]
    ours = window.objects[name]
    assert list(replayed.ordinate_unit)[:3] == \
        list(ours.ordinate_unit)[:3], 'the replay declared the same units'
    import numpy as np

    assert np.allclose(np.asarray(replayed.ordinate),
                       np.asarray(ours.ordinate)), \
        'and landed the same converted values'


def test_deleting_sub_items_journals_the_call(window, pump, tmp_path):
    """Two modes deleted from a shape set said nothing in the console
    (Brandon, 2026-08-30). The deletion journals as the call a script
    makes — row indices as they stood, which a sequential replay
    re-creates — and the replay comes out with the same modes gone."""
    import shutil

    source = tmp_path / 'shapes.npy'
    shutil.copy(fixture_path('plate', 'shapes.npy'), source)
    window.import_paths([str(source)])
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'ShapeSet')
    before = window.objects[name].num_shapes
    window._select_records(name, [1, 2])
    pump()
    window.delete_selected()
    pump()
    assert window.objects[name].num_shapes == before - 2
    assert f"project[{name!r}].delete_modes([1, 2])" in \
        window.project.journal

    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    replayed = room['project'][name]
    assert replayed.num_shapes == before - 2
    import numpy as np

    assert np.allclose(replayed.frequency,
                       window.objects[name].frequency), \
        'the same modes are gone'


# ---- the edit surface: geometry, channel table, descriptions -------------


def test_geometry_edits_journal_and_replay(window, pump):
    """Adding a traceline said nothing (Brandon, 2026-08-30) — the
    audit that followed put the whole edit surface on the record:
    structural adds as their core verbs, cell edits as post-state
    echoes, all replayable."""
    from visualdynamics.gui.object_tables import node_table_model
    from visualdynamics.units import DEFAULT_SYSTEM

    window.import_paths([fixture_path('plate', 'geometry.npz')])
    pump()
    geometry = window.objects['Geometry']
    nodes = [int(geometry.node_id[0]), int(geometry.node_id[1])]

    # the picking gesture's commit, exactly as the scene calls it
    window.editing = ('Geometry', 'tracelines')
    window._picked_nodes = list(nodes)
    window._commit_picked_nodes()
    pump()
    assert f"project['Geometry'].add_traceline({nodes!r})" in \
        window.project.journal

    # a cell edit in the node table: post-state echo, in SI
    from PySide6.QtCore import Qt

    model = node_table_model(geometry, DEFAULT_SYSTEM)
    window.editing = None
    window._set_table_model(model)
    model.setData(model.index(0, 1), '0.5', Qt.ItemDataRole.EditRole)
    assert any(line.startswith("project['Geometry'].node_xyz[0, 0] = ")
               for line in window.project.journal)

    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    replayed = room['project']['Geometry']
    assert len(replayed.traceline_conn) == len(geometry.traceline_conn)
    assert np.allclose(replayed.node_xyz, geometry.node_xyz), \
        'the echoed cell edit landed the same coordinates'


def test_channel_table_edits_journal_and_replay(window, pump):
    from PySide6.QtCore import Qt

    from visualdynamics.gui.object_tables import channel_table_model

    window.import_paths([fixture_path('plate', 'channel_table.vdyn')])
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'ChannelTable')
    table = window.objects[name]
    model = channel_table_model(table)
    window._set_table_model(model)
    column = next(i for i, c in enumerate(model.columns)
                  if c.title == 'Unit')
    model.setData(model.index(0, column), 'g', Qt.ItemDataRole.EditRole)
    assert f"project[{name!r}].set_cell('unit', 0, 'g')" in \
        window.project.journal

    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    assert room['project'][name]['unit'][0] == 'g'


def test_report_edits_journal_as_the_state_that_stands(window, pump):
    """Captions typed, blocks moved, the title changed — the editor's
    ops all land on the Report object, and the journal records its
    post-state as settling assignments: an editing session reads as
    what the report became, and the replay becomes it too."""
    from visualdynamics.gui.report_editor import ReportEditor
    from visualdynamics.units import DEFAULT_SYSTEM

    window.generate_report('modal')
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'Report')
    report = window.objects[name]
    editor = ReportEditor()
    window.report_editor = editor
    editor.edited.connect(window._journal_report_edit)
    editor.show_report(report, lambda: window.objects, DEFAULT_SYSTEM)
    editor._operate({'op': 'title', 'value': 'Shaker Test 12'})
    editor._operate({'op': 'field', 'at': 0, 'field': 'text',
                     'value': 'The summary, written by hand.'})
    editor._operate({'op': 'field', 'at': 0, 'field': 'text',
                     'value': 'The summary, rewritten.'})
    lines = [line for line in window.project.journal
             if line.startswith(f'project[{name!r}].blocks = ')]
    assert len(lines) == 1, 'the state settles, not a keystroke log'

    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    replayed = room['project'][name]
    assert replayed.title == 'Shaker Test 12'
    assert replayed.blocks[0]['text'] == 'The summary, rewritten.'


def test_moving_an_object_between_groups_journals_the_verb(window, pump):
    """A drag between link groups journals as the command a script
    would write — place, relink or unlink, one object by name — not
    as a wholesale project.links list (Brandon, 2026-08-30: 'the user
    needs to list all objects to do it')."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4'),
                         fixture_path('plate', 'test_geometry.npz')])
    pump()
    window._apply_move(['Geometry'], None)      # out of its group
    pump()
    said = [line for line in window.project.journal
            if line == "project.unlink('Geometry')"]
    assert len(said) == 1
    assert not [line for line in window.project.journal
                if line.startswith('project.links = ')], (
        'no wholesale list restatement')

    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    replayed = room['project']
    grouped = {n for g in replayed.links for n in g['members']}
    assert 'Geometry' not in grouped, 'the move replayed'


# ---- the views journal too (Brandon, 2026-08-30) --------------------------


def test_the_wavelet_view_journals_and_replays_its_figure(window, pump,
                                                          tmp_path,
                                                          monkeypatch):
    """Every parameterized view records the headless plot line that
    renders it — settling in place, so a session of nudging reads as
    the reading you ended on, and the replay writes the figures the
    session looked at."""
    window.import_paths([fixture_path('plate', 'time.npz')])
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'TimeHistory')
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    window.data_pane.wavelet_action.trigger()
    pump()
    window.data_pane.wavelet_panel.per_octave_box.setValue(6)
    pump()
    window.data_pane.wavelet_panel.per_octave_box.setValue(9)
    pump()
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_scalogram(')]
    assert len(lines) == 1, 'the view settles to the reading that stands'
    assert 'per_octave=9' in lines[0]

    monkeypatch.chdir(tmp_path)
    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    assert (tmp_path / 'scalogram.png').stat().st_size > 0, \
        'the replay wrote the figure the session looked at'


def test_the_octave_preview_journals_its_reading(window, pump):
    t = np.arange(16384) / 2048.0
    history = visualdynamics.TimeHistory(
        t, np.random.default_rng(6).standard_normal((2, len(t))),
        response_dof=['101Z+', '104Z+'], ordinate_dim='acceleration')
    window.add_object('Record', history)
    name = window.project.compute_psds('Record')
    window.show_object(name)
    pump()
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
    window.render_current()
    pump()
    pane.octave_action.trigger()
    pump()
    box = pane.octave_panel.per_box
    box.setCurrentIndex(box.findData(3))
    pump()
    lines = [line for line in window.project.journal
             if line.startswith(f'project[{name!r}].to_octave(')]
    wanted = (f"project[{name!r}].to_octave(3)"
              ".plot(path='octave.png', show=False)")
    assert lines == [wanted]


def test_the_kurtosis_view_journals_its_reading(window, pump):
    window.import_paths([fixture_path('plate', 'time.npz')])
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'TimeHistory')
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    window.data_pane.kurtosis_action.trigger()
    pump()
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_kurtosis(')]
    assert len(lines) == 1
    assert f"project[{name!r}]" in lines[0]
    assert "path='kurtosis.png'" in lines[0]


def test_deleting_a_whole_capture_or_channel_speaks_its_name(window,
                                                             pump):
    """Thirteen indices where one capture number carries the meaning
    read as the machine talking to itself (Brandon, 2026-08-30):
    picks that are exactly a capture journal as capture=, exactly a
    channel as dof=, anything else as the indices they are — and the
    core verb takes all three forms."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    pump()
    name = 'Time History'
    history = window.objects[name]
    ordinals = history.capture_indices()
    assert max(ordinals) > 3, 'the fixture really is multi-capture'

    # a whole capture, picked in the grid and deleted
    capture_rows = [i for i, o in enumerate(ordinals) if o == 3]
    window._select_records(name, capture_rows)
    pump()
    window.delete_selected()
    pump()
    assert f"project[{name!r}].delete_records(capture=3)" in \
        window.project.journal

    # a whole channel next
    dof = history.response_dof[0]
    channel_rows = [i for i in range(history.num_records)
                    if history.response_dof[i] == dof]
    window._select_records(name, channel_rows)
    pump()
    window.delete_selected()
    pump()
    assert f"project[{name!r}].delete_records(dof={dof!r})" in \
        window.project.journal

    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    replayed = room['project'][name]
    assert replayed.num_records == history.num_records
    assert dof not in list(replayed.response_dof)
    assert max(replayed.capture_indices()) == max(history.capture_indices())


def test_a_drive_point_deletion_speaks_dof_and_quantity(window, pump,
                                                        tmp_path):
    """Brandon's question (2026-08-30): dof alone at a drive point
    names two records. The journal only says dof= when the picks are
    exactly everything at that DOF; picking one quantity of the pair
    says dof= and dim= together, and the replay deletes exactly it."""
    import numpy as np

    t = np.arange(1024) / 1024.0
    history = visualdynamics.TimeHistory(
        t, np.random.default_rng(4).standard_normal((3, len(t))),
        response_dof=['2Z+', '2Z+', '5Z+'],
        ordinate_dim=['force', 'acceleration', 'acceleration'])
    window.project.add('Drive', history)
    del window.project.journal[:]           # the add is not the subject
    window.project.journal.append(
        "project = visualdynamics.Project('X')")

    line = window._deletion_line('Drive', history, 'delete_records', [1])
    assert line == \
        "project['Drive'].delete_records(dof='2Z+', dim='acceleration')"
    line = window._deletion_line('Drive', history, 'delete_records', [0, 1])
    assert line == "project['Drive'].delete_records(dof='2Z+')"

    history.delete_records(dof='2Z+', dim='acceleration')
    assert list(history.ordinate_dim) == ['force', 'acceleration']
    assert list(history.response_dof) == ['2Z+', '5Z+'], \
        'the force gauge at the drive point survived'


def test_deleting_a_capture_renumbers_the_survivors():
    """'avg 1, avg 3' after deleting the second capture is a gap the
    tree's columns faithfully showed (Brandon, 2026-08-30). Block
    labels renumber by the merge rule — bookkeeping, not identity —
    so the survivors read contiguous."""
    import numpy as np

    t = np.arange(256) / 256.0
    history = visualdynamics.TimeHistory(
        t, np.ones((6, 256)),
        response_dof=['1Z+', '2Z+'] * 3,
        block=['avg 1', 'avg 1', 'avg 2', 'avg 2', 'avg 3', 'avg 3'])
    history.delete_records(capture=1)
    assert history.block == ['avg 1', 'avg 1', 'avg 2', 'avg 2'], \
        'the third capture became the second'
    assert history.capture_indices() == [0, 0, 1, 1]


def test_deleting_an_frf_column_speaks_its_reference(window, pump):
    """A reference column deleted by twenty-two indices (Brandon,
    2026-08-30): picks that are exactly a matrix column journal as
    reference=, one cell as dof= and reference= together, and the
    verb takes both forms."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    pump()
    frf = window.objects['FRF']
    reference = frf.reference_dof[0]
    column = [i for i in range(frf.num_records)
              if frf.reference_dof[i] == reference]
    assert len(column) > 1
    line = window._deletion_line('FRF', frf, 'delete_records', column)
    assert line == \
        f"project['FRF'].delete_records(reference={reference!r})"
    cell = window._deletion_line('FRF', frf, 'delete_records', [column[0]])
    assert cell == (f"project['FRF'].delete_records("
                    f"dof={frf.response_dof[column[0]]!r}, "
                    f"reference={reference!r})")

    before = frf.num_records
    frf.delete_records(reference=reference)
    assert frf.num_records == before - len(column)
    assert reference not in list(frf.reference_dof)


def test_a_drag_trial_journals_nothing_and_the_landing_settles(window,
                                                               pump):
    """One drag wrote eighty relink lines (Brandon, 2026-08-30): the
    per-tick would-this-land trial really moves and rolls back, and
    every tick journaled. The trial is a question and stays quiet;
    the landing's record is the one verb line."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4'),
                         fixture_path('plate', 'test_geometry.npz')])
    pump()
    before = list(window.project.journal)
    for _tick in range(10):
        window._trial_move(['Geometry'], None)
    assert window.project.journal == before, 'questions are not acts'

    window._apply_move(['Geometry'], None)
    pump()
    lines = [line for line in window.project.journal
             if 'relink' in line or 'unlink' in line
             or line.startswith('project.links = ')]
    assert lines == ["project.unlink('Geometry')"], lines
