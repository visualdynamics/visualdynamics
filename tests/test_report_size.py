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


def _modal_project():
    import visualdynamics as vd

    loaded = vd.import_file(_fixture('plate', 'modal.nc4'))
    project = vd.Project()
    for name, obj in loaded.items():
        project.add(name, obj)
    project.link(*project.names)
    return project


def _fixture(*parts):
    from conftest import fixture_path

    return fixture_path(*parts)


def test_no_template_puts_a_time_history_on_the_stage():
    """A stage figure carries two million points, and a report's time
    histories are the largest thing in it: the plate's modal report
    was 36 MB, 88% of it two of them (Brandon, 2026-09-20). Every
    built-in template draws its time histories flat. The stage stays
    for what it is good at — the spectra, the coherence, the FRFs."""
    from visualdynamics.core.report import (
        modal_template,
        random_template,
        shock_template,
        sine_template,
        sysid_template,
        transient_template,
    )

    builders = {'modal': modal_template, 'random': random_template,
                'shock': shock_template, 'sine': sine_template,
                'sysid': sysid_template, 'transient': transient_template}
    for name, build in builders.items():
        histories = [b for b in build({}).blocks
                     if b.get('kind') == 'plot'
                     and str(b.get('source', '')).endswith('TimeHistory')]
        assert histories, f'{name} draws its time histories'
        for block in histories:
            assert block.get('mode') != 'stage', f'{name}: {block.get("caption")}'


def test_the_modal_report_is_megabytes_not_tens_of_them():
    """The measure behind the change, on the fixture it was found on:
    36.8 MB before, and the two time-history stages were 31.8 and
    5.8 MB of it."""
    from visualdynamics.core.report import modal_template
    from visualdynamics.report import render_html

    project = _modal_project()
    page = render_html(modal_template(project, links=project.links), project,
                       links=project.links)
    assert len(page) < 6 * 2 ** 20, f'{len(page) / 2 ** 20:.1f} MB'
    payload = json.loads(re.search(
        r'<script id="data"[^>]*>(.*?)</script>', page, re.DOTALL).group(1))
    biggest = max(payload['blocks'], key=lambda b: len(json.dumps(b)))
    assert biggest['kind'] == 'plot', 'the time histories, drawn flat'


def test_a_filtered_figure_counts_its_own_channels():
    """A figure filtered to one quantity pages over that quantity's
    channels, and says so: it counted every record the object held
    when the paging was written."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory

    n, channels = 4096, 30
    rng = np.random.default_rng(2)
    history = TimeHistory(
        np.arange(n) / 4096.0, rng.standard_normal((channels, n)),
        response_dof=[f'{101 + i}Z+' for i in range(channels)],
        ordinate_dim=['force'] * 6 + ['acceleration'] * (channels - 6),
        ordinate_unit=['N'] * 6 + ['m/s**2'] * (channels - 6))
    built = _plot_block({'kind': 'plot', 'source': 'T', 'mode': 'curves',
                         'select': 'dim:acceleration', 'caption': 'Response'},
                        history, {'T': history}, visualdynamics.SI)
    # 24 acceleration channels of 30 records: one page, all of them,
    # and no claim that six were dropped
    assert not isinstance(built, list)
    assert built['caption'] == 'Response', built['caption']
    assert len(built['curves']) == 24
    wide = TimeHistory(
        np.arange(n) / 4096.0, rng.standard_normal((60, n)),
        response_dof=[f'{201 + i}Z+' for i in range(60)],
        ordinate_dim=['force'] * 6 + ['acceleration'] * 54,
        ordinate_unit=['N'] * 6 + ['m/s**2'] * 54)
    pages = _plot_block({'kind': 'plot', 'source': 'T', 'mode': 'curves',
                         'select': 'dim:acceleration', 'caption': 'Response'},
                        wide, {'T': wide}, visualdynamics.SI)
    assert [p['caption'].split('—')[-1].strip() for p in pages] == [
        'channels 1–24 of 54', 'channels 25–48 of 54', 'channels 49–54 of 54']
