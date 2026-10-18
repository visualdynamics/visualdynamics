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
