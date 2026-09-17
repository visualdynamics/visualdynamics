"""Coherence reads two ways, and the plot lets you pick.

A line plot answers "is this channel any good". It cannot answer that for 339
channels at once — they overlay into a band and the curve budget means most
are never drawn. The map answers it for all of them: a poor channel is a dark
row, a poor band a dark column.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import MultipleCoherence
from visualdynamics.plot import MAP_LABELS, build_coherence_map


def channels(count, samples=32):
    """`count` coherence channels, each a ramp from 0 to 1 along frequency."""
    ordinate = np.tile(np.linspace(0.0, 1.0, samples), (count, 1))
    return MultipleCoherence(abscissa=np.linspace(0, 128, samples),
                             ordinate=ordinate,
                             response_dof=[f'{100 + i}Z+' for i in range(count)])


def built(data, records=None):
    import pyqtgraph as pg

    layout = pg.GraphicsLayoutWidget()
    drawn, requested = build_coherence_map(layout, [('', data, records)])
    return layout, layout.getItem(0, 0), drawn, requested


def image_of(plot):
    return next(item for item in plot.items
                if type(item).__name__ == 'ImageItem')


def test_every_channel_is_drawn(qt_app):
    """Nothing is budgeted away — that is the point of the map."""
    data = channels(300)
    _layout, _plot, drawn, requested = built(data)
    assert drawn == requested == 300


def test_the_image_is_channels_by_frequency(qt_app):
    data = channels(7, samples=32)
    _layout, plot, _d, _r = built(data)
    assert image_of(plot).image.shape == (32, 7)


def test_the_color_scale_is_pinned_to_nought_and_one(qt_app):
    """Coherence is a bounded ratio. A scale fitted to the selection would
    make a good channel look bad beside a better one."""
    poor = channels(3)
    poor.ordinate[:] = 0.2                      # nothing near 1 anywhere
    _layout, plot, _d, _r = built(poor)
    assert tuple(image_of(plot).getLevels()) == (0.0, 1.0)


def test_it_spans_the_real_frequency_axis(qt_app):
    data = channels(4)
    _layout, plot, _d, _r = built(data)
    mapped = image_of(plot).mapRectToParent(image_of(plot).boundingRect())
    assert mapped.left() == pytest.approx(float(data.abscissa[0]))
    assert mapped.right() == pytest.approx(float(data.abscissa[-1]))


def test_the_channel_axis_reads_downward(qt_app):
    """The tree lists channels top to bottom; image coordinates run the other
    way, so left alone the first channel sits at the bottom."""
    _layout, plot, _d, _r = built(channels(5))
    assert plot.getViewBox().yInverted()


def test_the_channels_are_labeled_by_dof(qt_app):
    data = channels(6)
    _layout, plot, _d, _r = built(data)
    labels = [text for _position, text in plot.getAxis('left')._tickLevels[0]]
    assert labels == data.response_dof


def test_the_labels_thin_out_rather_than_collide(qt_app):
    """339 labels is a gray smear; one in every nth is a scale."""
    _layout, plot, _d, _r = built(channels(339))
    ticks = plot.getAxis('left')._tickLevels[0]
    assert 0 < len(ticks) <= MAP_LABELS
    positions = [position for position, _text in ticks]
    assert positions[0] == 0.5, 'sits at the middle of its row'


def test_a_color_bar_is_drawn(qt_app):
    """Without it the colors are not numbers. It is nested inside the plot
    rather than added to the layout, so look in the scene."""
    layout, _plot, _d, _r = built(channels(4))
    bars = [item for item in layout.scene().items()
            if type(item).__name__ == 'ColorBarItem']
    assert len(bars) == 1
    assert tuple(bars[0].levels()) == (0.0, 1.0)


def test_a_record_subset_maps_just_those(qt_app):
    data = channels(10)
    _layout, plot, drawn, _r = built(data, records=[2, 5])
    assert drawn == 2
    assert image_of(plot).image.shape[1] == 2


def test_it_works_on_the_real_file(qt_app):
    out = visualdynamics.import_file(fixture_path('plate', 'modal_spectra.nc4'))
    data = out['Modal_coherence']
    _layout, plot, drawn, _r = built(data)
    assert drawn == data.num_records
    values = image_of(plot).image
    assert values.min() >= 0.0 and values.max() <= 1.0 + 1e-9
