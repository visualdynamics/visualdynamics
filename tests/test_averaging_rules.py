"""The averaging's own arithmetic, pinned by value where a mutation pass
found it reached only through drags and drawings (2026-10-11).

`tools/mutate.py` over averaging.py against its five core test files
caught 47 of 92: what a dragged span means in frames, how overlapping
frames stack into rows, where the rail's rows sit, and the edges of
what an averaging will hold were each checked only by a picture or a
drag. Stated here as the rules they are.
"""

from __future__ import annotations

import pytest

from visualdynamics.core import averaging as av
from visualdynamics.core.averaging import Averaging, from_span

RATE, SAMPLES = 100.0, 1000


def _frames(overlap=0.5, frames=1, start=0.0):
    return Averaging(frame_length=100, overlap=overlap, window='hann',
                     frames=frames, start=start)


def test_a_dragged_span_is_whole_frames_from_where_it_starts():
    """Frames of 100 samples at half overlap step 50: a span of 300
    samples from 2 s holds 1 + (300 - 100) // 50 = 5 of them."""
    got = from_span(_frames(), 2.0, 5.0, RATE, SAMPLES)
    assert (got.start, got.frames) == (2.0, 5)
    # narrower than one frame is still one
    assert from_span(_frames(), 2.0, 2.5, RATE, SAMPLES).frames == 1
    # dragged before the record, it starts at the record
    assert from_span(_frames(), -1.0, 2.0, RATE, SAMPLES).start == 0.0
    # past the last place a whole frame fits, it stops there with one
    late = from_span(_frames(), 9.5, 9.9, RATE, SAMPLES)
    assert (late.start, late.frames) == (9.0, 1)
    # never more frames than the record holds from that start
    assert from_span(_frames(), 0.0, 50.0, RATE, SAMPLES).frames == 19
    # the drag is on the record's clock: an origin of 10 s moves nothing
    assert from_span(_frames(), 12.0, 15.0, RATE, SAMPLES, origin=10.0) == \
        from_span(_frames(), 2.0, 5.0, RATE, SAMPLES)


@pytest.mark.parametrize('overlap, rows, count', [
    (0.0, [0, 0, 0, 0], 1),
    (0.5, [0, 1, 0, 1], 2),
    (0.75, [0, 1, 2, 3], 4),
])
def test_overlapping_frames_stack_into_the_fewest_rows(overlap, rows, count):
    """Each frame takes the lowest row whose last frame has ended."""
    levels, got = _frames(overlap=overlap, frames=4).levels(RATE)
    assert (levels, got) == (rows, count)


def test_the_rail_shrinks_its_slots_to_fit_its_band():
    """One row takes a whole slot; four rows share the band; the trace
    keeps what the rail leaves it, never less than a quarter."""
    one = _frames(overlap=0.0, frames=3).rail(RATE)
    glyph = av.RAIL_SLOT * av.GLYPH_SHARE
    assert one['slot'] == pytest.approx(av.RAIL_SLOT)
    assert one['baselines'] == pytest.approx([av.RAIL_TOP - glyph] * 3)
    assert one['foot'] == pytest.approx(av.RAIL_TOP - glyph - av.RAIL_GAP)
    four = _frames(overlap=0.75, frames=4).rail(RATE)
    slot = av.RAIL_BAND / 4
    glyph = slot * av.GLYPH_SHARE
    assert four['slot'] == pytest.approx(slot)
    assert four['baselines'] == pytest.approx(
        [av.RAIL_TOP - glyph - level * slot for level in (0, 1, 2, 3)])
    assert four['foot'] == pytest.approx(
        max(av.RAIL_TOP - glyph - 3 * slot - av.RAIL_GAP, 0.25))


def test_an_averaging_holds_only_what_can_be_averaged():
    Averaging(frame_length=2, overlap=0.0, window='hann', frames=1)
    with pytest.raises(ValueError, match='at least two samples'):
        Averaging(frame_length=1, overlap=0.0, window='hann', frames=1)
    with pytest.raises(ValueError, match='at least one frame'):
        Averaging(frame_length=8, overlap=0.0, window='hann', frames=0)
    # a window parameter's range is closed at both ends
    Averaging(frame_length=8, window='tukey', window_parameter=0.0)
    Averaging(frame_length=8, window='tukey', window_parameter=1.0)
    with pytest.raises(ValueError, match='alpha runs 0 to 1'):
        Averaging(frame_length=8, window='tukey', window_parameter=1.01)
    assert _frames(overlap=0.999).hop == 1, 'a hop of at least one sample'
    with pytest.raises(ValueError, match='at least one sample'):
        av.window_shape('hann', 0)
    assert len(av.window_shape('hann', 1)) == 1


def test_an_unknown_detrend_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown detrend 'sideways'; visualdynamics has"):
        av.normalize_detrend('sideways')


def test_filling_a_record_shorter_than_a_frame_leaves_one():
    assert _frames(frames=7).filled(50, RATE).frames == 1
    assert _frames(frames=7).filled(SAMPLES, RATE).frames == 19
