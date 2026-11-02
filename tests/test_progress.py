"""A stage that cannot count itself: announced as BUSY, shown as a
moving bar, and counted again on the next ordinary report."""

from __future__ import annotations

from visualdynamics.core.progress import BUSY, Ticker


def test_busy_is_told_at_once_and_the_next_tick_counts_again():
    told = []
    ticker = Ticker(lambda done, total: told.append((done, total)),
                    interval=0.0)
    ticker.add(4)
    ticker.tick()
    ticker.busy()
    ticker.tick()
    assert told == [(0, 4), (1, 4), (1, BUSY), (2, 4)]
    Ticker(None).busy()     # no callback, nothing to tell


def test_the_window_shows_a_moving_bar_for_busy(window):
    bar = window._import_progress
    window._strip_tick(1, 4)
    assert (bar.minimum(), bar.maximum(), bar.value()) == (0, 4, 1)
    window._strip_tick(1, BUSY)
    assert (bar.minimum(), bar.maximum()) == (0, 0), \
        'a range of nothing is Qt\'s moving bar'
    window._strip_tick(2, 4)
    assert (bar.minimum(), bar.maximum(), bar.value()) == (0, 4, 2)
    bar.hide()


def test_a_share_spreads_its_own_count_over_fixed_units():
    """A stage's own count fills its share and no more, whatever the
    count: 190,000 elements' ticks against a few hundred solves' left
    the solves a sliver and the bar read full for an hour."""
    told = []
    ticker = Ticker(lambda done, total: told.append((done, total)),
                    interval=0.0)
    first, second = ticker.share(200), ticker.share(800)
    assert ticker.total == 1000, 'both shares are in the total at once'
    first.add(190_000)
    first.tick(95_000)
    assert ticker.done == 100
    first.tick(95_000)
    assert ticker.done == 200, 'a finished stage fills its share exactly'
    second.add(4)
    second.tick()
    assert ticker.done == 400
    second.extend_to(6)      # the estimate grew: 1 of 7 now
    assert ticker.done == 400, 'never ticked back'
    second.tick(6)
    assert ticker.done == ticker.total == 1000
    assert [d for d, _t in told] == sorted(d for d, _t in told)


def test_a_count_after_busy_brings_the_bar_back():
    told = []
    ticker = Ticker(lambda done, total: told.append((done, total)),
                    interval=0.0)
    share = ticker.share(10)
    share.add(1000)
    share.busy()
    share.tick()             # less than a unit: nothing owed, still told
    assert told[-2:] == [(0, BUSY), (0, 10)]
