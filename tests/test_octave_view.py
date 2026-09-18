"""The octave-band reading: principle 13's four parts for a spectrum.

The toggle on the bar, the spacing in the pane, the banded steps
previewed over the narrowband they integrate, and the Apply button
running the same `compute_octave` verb a script calls (Brandon,
2026-08-29 — moved here from the calculator, which no longer offers
it).
"""

from __future__ import annotations

import numpy as np

import visualdynamics


def _psds(window, pump):
    rng = np.random.default_rng(6)
    t = np.arange(16384) / 2048.0
    history = visualdynamics.TimeHistory(
        t, rng.standard_normal((2, len(t))),
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
    return window.objects[name], pane


def test_the_reading_is_offered_on_a_plain_density_alone(window, pump):
    _spectra, pane = _psds(window, pump)
    assert pane.octave_action.isVisible()
    # and not on the time history, which has nothing to band
    item = window._item_for_object('Record')
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert not pane.octave_action.isVisible()


def test_the_toggle_previews_the_steps_over_the_narrowband(window, pump):
    _spectra, pane = _psds(window, pump)
    pane.octave_action.trigger()
    pump()
    assert pane.octave_panel.isVisible()
    assert window.octave_previews, 'the steps are on the plot'
    _plot, curve = window.octave_previews[0]
    assert curve.xData is not None and len(curve.xData) > 4
    bands_shown = pane.octave_panel.derived['bands'].text()
    assert bands_shown.isdigit() and int(bands_shown) * 2 == \
        len(curve.xData), 'two points per band: the flat step outline'


def test_the_spacing_redraws_the_preview(window, pump):
    _spectra, pane = _psds(window, pump)
    pane.octave_action.trigger()
    pump()
    sixth = len(window.octave_previews[0][1].xData)
    box = pane.octave_panel.per_box
    box.setCurrentIndex(box.findData(3))
    pump()
    third = len(window.octave_previews[0][1].xData)
    assert third < sixth, 'coarser bands, fewer steps, live'


def test_apply_makes_exactly_the_previewed_banding(window, pump):
    _spectra, pane = _psds(window, pump)
    pane.octave_action.trigger()
    pump()
    box = pane.octave_panel.per_box
    box.setCurrentIndex(box.findData(3))
    pump()
    pane.octave_panel.apply_button.click()
    pump()
    name = next(n for n in window.project.provenance
                if window.project.provenance[n]['verb'] == 'compute_octave')
    assert window.project.provenance[name]['params']['per_octave'] == 3, \
        'the record carries the spacing the preview was drawn with'


def test_no_act_offers_it_beside_the_reading(window, pump):
    psds, _pane = _psds(window, pump)
    assert window.acts_for([window.project.name_of(psds)]) == [], \
        'the act lives on the reading\'s pane (Brandon, 2026-08-29)'


def test_the_preview_honors_a_record_selection(window, pump):
    """One sub-item picked, one curve on the plot — and one banded
    step over it. The preview used to band every channel across the
    lone selected curve (Brandon, 2026-08-30): the selection said one
    thing and the plot another."""
    spectra, pane = _psds(window, pump)
    name = window.project.name_of(spectra)
    pane.octave_action.trigger()
    pump()
    assert len(window.octave_previews) == 2, 'whole object: both channels'
    window._select_records(name, [1])
    window.render_current()
    pump()
    assert len(window.octave_previews) == 1, \
        'one record selected, one preview'
    banded = spectra.to_octave(pane.octave_panel.per_octave())
    from visualdynamics.plot import step_outline
    _x, rows = step_outline(banded.abscissa,
                            banded.display_ordinate(window.unit_system),
                            banded.bin_widths())
    assert np.allclose(window.octave_previews[0][1].yData,
                       np.abs(np.atleast_2d(rows)[1])), \
        'and it is the selected record, not the first'


def test_the_preview_stands_on_the_stage_too(window, pump):
    """The 3-D reading gets the same preview: steps at each record's
    own station, panel beside — it drew nothing at all there at first
    (Brandon, 2026-08-30)."""
    _spectra, pane = _psds(window, pump)
    pane.octave_action.trigger()
    pump()
    pane.waterfall_action.trigger()      # onto the stage
    pump()
    window.render_current()
    pump()
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible(), 'the stage is up'
    assert 'marks-octave' in set(pane.waterfall_plotter.actors), \
        'the banded steps are stage geometry'
    assert pane.octave_panel.isVisible(), 'the panel rides the stage'
    bands = pane.octave_panel.derived['bands'].text()
    assert bands.isdigit() and int(bands) > 0


def test_applying_stands_the_reading_down_and_shows_the_bands_plain(window,
                                                                    pump):
    """The banded object came up wearing the preview: the octave toggle
    stayed on after Apply, so the new object's own stage was drawn and
    then its re-banding stepped over it in the preview color — every
    ribbon pink (Brandon, 2026-09-18: "they all just turn pink").
    Apply is the end of the reading; the object it made shows plain."""
    _spectra, pane = _psds(window, pump)
    pane.octave_action.trigger()
    pump()
    pane.waterfall_action.trigger()      # onto the stage
    pump()
    window.render_current()
    pump()
    assert 'marks-octave' in set(pane.waterfall_plotter.actors), 'previewing'
    pane.octave_panel.apply_asked.emit()
    pump()
    banded = window.current_object()
    assert banded is not None and banded.bandwidth is not None, \
        'the banded object is what is shown'
    assert not pane.octave_action.isChecked(), 'the reading stood down'
    assert not pane.octave_panel.isVisible()
    assert 'marks-octave' not in set(pane.waterfall_plotter.actors), \
        'no preview over the object the reading made'
    assert 'waterfall' in set(pane.waterfall_plotter.actors), 'its own ribbons'


def test_a_specification_is_offered_the_reading_and_apply_bands_its_limits(
        window, pump):
    """The same toggle, pane and button a PSD has; what Apply makes is a
    specification on bands, limits and all (Brandon, 2026-09-18)."""
    from visualdynamics.core.data import Specification

    spec = visualdynamics.import_file(
        __import__('conftest').fixture_path('plate', 'random_spectra.nc4')
    )['Random_specification']
    window.add_object('Spec', spec)
    window.show_object('Spec')
    pump()
    pane = window.data_pane
    assert pane.octave_action.isVisible(), 'offered on a specification'
    pane.octave_action.trigger()
    pump()
    # the reading shows on the specification's own stage — the banded
    # stage, not the waterfall's — and on the flat plot (the button
    # "did nothing" on the stage, Brandon, 2026-09-18)
    assert pane.waterfall_action.isChecked(), 'a specification opens on its stage'
    assert 'marks-octave' in set(pane.waterfall_plotter.actors), \
        'the banded steps on the banded stage'
    assert pane.octave_panel.isVisible()
    pane.waterfall_action.trigger()      # and flat
    pump()
    assert window.octave_previews, 'the steps over the flat bands'
    pane.waterfall_action.trigger()      # back to the stage for Apply
    pump()
    pane.octave_panel.apply_asked.emit()
    pump()
    made = window.current_object()
    assert isinstance(made, Specification) and made.bandwidth is not None
    assert set(made.limits) == set(spec.limits)
    assert not pane.octave_action.isChecked()

