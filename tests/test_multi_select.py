"""A specification and the PSD it bounds, plotted together."""
from __future__ import annotations

import pytest
from conftest import prepared_comparison as prepare

pytestmark = pytest.mark.usefixtures('flat_reading')

def pick(window, name, dof='101Z+'):
    """Select the record for `dof` in that object's grid, the way a
    click on it would — through the grid's own signal, since the
    handler reads `sender()` and a direct call leaves it None."""
    obj = window.objects[name]
    row = next(i for i in range(obj.num_records)
               if obj.response_dof[i] == dof
               and (obj.reference_dof is None
                    or obj.reference_dof[i] == dof))
    window.record_grids[name].select_records([row])
    window.record_grids[name].selection_changed.emit()
    return row


def curves(window):
    import pyqtgraph as pg

    from visualdynamics.plot import data_curves

    return sum(len(data_curves(item))
               for item in window.data_pane.graphics.ci.items
               if isinstance(item, pg.PlotItem))


def test_a_grid_pick_replaces_another_on_its_own(window, pump):
    """The behavior that makes a single click predictable: one grid at
    a time unless asked otherwise."""
    spec, psd = prepare(window, pump)
    window.tree.clearSelection()
    pick(window, spec)
    pick(window, psd)
    pump()
    _kinds, series = window._current_series()
    assert [name for name, _d, _r in series] == [psd]


def test_the_modifier_puts_a_specification_over_its_psd(window, pump,
                                                        monkeypatch):
    """The comparison a random vibration test exists to make. Each grid
    is its own widget, so Qt never tells one that a pick in another was
    meant to add — the modifier has to be read here or the two can never
    be on the plot together."""
    spec, psd = prepare(window, pump)
    window.tree.clearSelection()
    pick(window, spec)
    monkeypatch.setattr(window, '_adding_to_selection', lambda: True)
    pick(window, psd)
    pump()
    _kinds, series = window._current_series()
    assert sorted(name for name, _d, _r in series) == sorted([psd, spec])
    assert curves(window) >= 2, 'both are drawn'


def selected_names(window):
    _kinds, series = window._current_series()
    return sorted(name for name, _data, _records in series)


def both(window, pump, monkeypatch):
    """Both records picked, which is where every deselection starts.

    The counts below read the flat plot's curves; the waterfall is the
    default reading now, so the flat plot is asked for."""
    window.data_pane.waterfall_action.setChecked(False)
    spec, psd = prepare(window, pump)
    window.tree.clearSelection()
    pick(window, spec)
    monkeypatch.setattr(window, '_adding_to_selection', lambda: True)
    pick(window, psd)
    pump()
    assert selected_names(window) == sorted([spec, psd])
    return spec, psd


def clear(window, name):
    window.record_grids[name].select_records([])
    window.record_grids[name].selection_changed.emit()


def test_deselecting_a_psd_leaves_the_specification(window, pump, monkeypatch):
    """It read as 'all of my records' instead, and a specification then
    cut those back to the pairs it bounds — the same curves as before,
    so the plot answered a deselection by not changing."""
    spec, psd = both(window, pump, monkeypatch)
    clear(window, psd)
    pump()
    assert selected_names(window) == [spec]
    assert curves(window) == 1, 'the target alone; its limits are shading'


def test_deselecting_a_specification_leaves_the_psd(window, pump, monkeypatch):
    spec, psd = both(window, pump, monkeypatch)
    clear(window, spec)
    pump()
    assert selected_names(window) == [psd]
    assert curves(window) == 1


def test_the_last_grid_emptied_still_shows_the_whole_object(window, pump):
    """A grid going empty on its own means 'all of it' — what clicking
    the object has always shown. Read as 'none of it' there would be
    nothing left on the plot at all.

    'All of it' is about the selection, not about the curve count. A
    specification is drawn one channel at a time however it was
    selected, so what says the whole object is here is that every
    channel is offered in the bar — the curve count says only that the
    plot is doing its job.
    """
    spec, _psd = prepare(window, pump)
    window.tree.clearSelection()
    pick(window, spec)
    pump()
    clear(window, spec)
    pump()
    _kinds, series = window._current_series()
    assert [(name, records) for name, _d, records in series] == [(spec, None)]
    assert window.data_pane.pair_box.count() == \
        window.objects[spec].num_records, 'every record of it, not one'
    assert curves(window) == 1, 'and one of them drawn'


def test_changing_which_record_keeps_both_up(window, pump, monkeypatch):
    """Only an *empty* grid means take mine off. Picking a different row
    in one grid while another object is on the plot is refining the
    selection, not abandoning it."""
    spec, psd = both(window, pump, monkeypatch)
    obj = window.objects[psd]
    other = next(i for i in range(obj.num_records)
                 if obj.response_dof[i] == '113Z+'
                 and (obj.reference_dof is None
                      or obj.reference_dof[i] == '113Z+'))
    window.record_grids[psd].select_records([other])
    window.record_grids[psd].selection_changed.emit()
    pump()
    assert selected_names(window) == sorted([spec, psd])
