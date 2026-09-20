"""What makes a report large, and what keeps it small.

A 144-channel run at 16 kHz made a 200 MB report (Brandon,
2026-09-20): the time-history stage's two million points a page,
each written with a float's seventeen digits. The random report's
time histories are 2-D now — paged like the stage, one shared time
axis a page, an envelope per curve on a budget — and every number on
the page keeps seven significant digits.
"""

from __future__ import annotations

import json
import re

import numpy as np

import visualdynamics
from visualdynamics.core.data import TimeHistory
from visualdynamics.report import MAX_REPORT_CURVES, TIME_FIGURE_POINTS, _plot_block


def _history(channels, seconds=2.0, rate=16384.0, seed=1):
    n = int(seconds * rate)
    rng = np.random.default_rng(seed)
    return TimeHistory(np.arange(n) / rate, rng.standard_normal((channels, n)),
                       response_dof=[f'{100 + i // 3}{"XYZ"[i % 3]}+'
                                     for i in range(channels)],
                       ordinate_dim=['acceleration'] * channels,
                       ordinate_unit=['m/s**2'] * channels)


def test_the_random_report_draws_its_time_histories_flat():
    from visualdynamics.core.report import random_template

    project = visualdynamics.random_vibration_run(
        visualdynamics.__file__.replace('src/visualdynamics/__init__.py',
                                        'testdata/plate/random.nc4'))
    blocks = random_template(project, links=project.links).blocks
    time = [b for b in blocks if b.get('source') == '@basis:TimeHistory'
            and b.get('kind') == 'plot' and 'time histor' in b.get('caption', '')]
    assert time and all(b['mode'] == 'curves' for b in time)


def test_many_channels_page_and_share_one_axis():
    history = _history(60)
    built = _plot_block({'kind': 'plot', 'source': 'T', 'mode': 'curves',
                         'caption': 'Measured'}, history, {'T': history},
                        visualdynamics.SI)
    assert isinstance(built, list) and len(built) == 3
    assert [len(page['curves']) for page in built] == [24, 24, 12]
    assert built[0]['caption'].endswith('channels 1–24 of 60')
    assert built[2]['caption'].endswith('channels 49–60 of 60')
    assert '(first' not in built[0]['caption'], 'paged, not cut'
    for page in built:
        assert all(curve['x'] is None for curve in page['curves']), 'one axis'
        # the budget is per curve; the one shared axis rides beside it
        assert len(page['x']) * len(page['curves']) <= TIME_FIGURE_POINTS * 1.02
        assert 'Thinned' in page['note']
    labels = [c['label'] for page in built for c in page['curves']]
    assert labels == list(history.response_dof)


def test_a_page_holds_seven_significant_digits():
    history = _history(3, seconds=0.5)
    built = _plot_block({'kind': 'plot', 'source': 'T', 'mode': 'curves',
                         'caption': 'Measured'}, history, {'T': history},
                        visualdynamics.SI)
    text = json.dumps(built)
    numbers = [float(m) for m in re.findall(r'-?\d+\.\d+(?:e-?\d+)?', text)]
    assert numbers, 'the page holds numbers'
    assert all(float(f'{v:.7g}') == v for v in numbers), \
        'every number on the page is what seven significant digits say'


def test_the_envelope_keeps_every_extreme_on_common_bins():
    from visualdynamics.decimate import envelope_rows

    x = np.arange(1000.0)
    rows = np.zeros((2, 1000))
    rows[0, 137] = 5.0
    rows[0, 138] = -3.0
    rows[1, 900] = -7.0
    axis, out = envelope_rows(x, rows, 50)
    assert axis.shape == (100,) and out.shape == (2, 100)
    assert out[0].max() == 5.0 and out[0].min() == -3.0
    assert out[1].min() == -7.0
    k = 137 // 20
    assert list(out[0][2 * k:2 * k + 2]) == [5.0, -3.0], 'the earlier extreme first'
    assert axis[2 * k] == 120.0 and axis[2 * k + 1] == 139.0
    same_x, same = envelope_rows(x[:40], rows[:, :40], 50)
    assert same_x is not None and same.shape == (2, 40), 'under the budget, untouched'


def test_a_wide_run_is_a_fraction_of_what_it_was():
    """The measure behind the change: 144 channels of a 16 kHz run,
    two seconds of it, drawn as the random report draws them."""
    history = _history(144)
    pages = _plot_block({'kind': 'plot', 'source': 'T', 'mode': 'curves',
                         'caption': 'Measured'}, history, {'T': history},
                        visualdynamics.SI)
    size = sum(len(json.dumps(page)) for page in pages)
    assert len(pages) == -(-144 // MAX_REPORT_CURVES)
    assert size < 12 * 2 ** 20, f'{size / 2 ** 20:.1f} MB for six pages'
