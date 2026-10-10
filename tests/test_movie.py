"""Saving an animation as a video: MP4 holding H.264, or nothing.

The file is read back at the byte level — the sample table says how
many frames went in, the sample entry which codec they went in as —
because the failure this guards is a file that *exists* and is wrong:
on a machine with no H.264 encoder, Qt records MPEG-4 Part 2 into the
same .mp4 without a word (PLAN.md, "Saving an animation").
"""

import numpy as np
import pytest
from conftest import plate_geometry_and_shapes

from visualdynamics.gui import movie
from visualdynamics.gui.movie import mp4_reading
from visualdynamics.viz.animate import cycle_parameters, sweep_parameters


def _box(kind, body):
    import struct

    return struct.pack('>I4s', 8 + len(body), kind) + body


def _mp4(codec, count, frames=b''):
    """The boxes an MP4's reading comes from: the frames' data, then
    the index Qt 6.12 writes after them."""
    sizes = _box(b'stsz', bytes(8) + count.to_bytes(4, 'big'))
    described = _box(b'stsd', bytes(4) + (1).to_bytes(4, 'big')
                     + _box(codec, bytes(8)))
    index = _box(b'moov', _box(b'trak', _box(b'mdia', _box(
        b'minf', _box(b'stbl', described + sizes)))))
    return _box(b'mdat', frames) + index


def test_the_reading_walks_boxes_rather_than_searching_bytes(tmp_path):
    """Encoded frames hold any four bytes. With the index after them, a
    search for 'stsz' found one inside a frame and counted 10 frames of
    a 180-frame movie (Qt 6.12, 2026-10-09)."""
    path = tmp_path / 'boxes.mp4'
    path.write_bytes(_mp4(b'avc1', 180,
                          frames=b'..stsz' + bytes(8)
                          + (10).to_bytes(4, 'big') + b'mp4v..'))
    assert mp4_reading(path) == (b'avc1', 180)


def frames(count, height=120, width=160, channels=3):
    yy, xx = np.mgrid[0:height, 0:width]
    for i in range(count):
        image = np.zeros((height, width, channels), np.uint8)
        image[..., 1] = (127 + 127 * np.sin(xx / 9 + i / 5)).astype(np.uint8)
        image[..., 0] = (yy * 2 % 256).astype(np.uint8)
        yield image


def test_a_movie_is_h264_with_every_frame(h264, tmp_path):
    path = tmp_path / 'a.mp4'
    assert movie.write_movie(path, frames(45)) == 45
    assert mp4_reading(path) == (b'avc1', 45)


def test_odd_sizes_and_rgba_are_taken(h264, tmp_path):
    """H.264 needs even dimensions and a window can be any size; the
    recorder trims the odd edge itself."""
    path = tmp_path / 'odd.mp4'
    movie.write_movie(path, frames(10, 121, 161, channels=4))
    assert mp4_reading(path) == (b'avc1', 10)


def test_an_existing_file_is_replaced_not_renamed_around(h264, tmp_path):
    path = tmp_path / 'again.mp4'
    path.write_bytes(b'old')
    movie.write_movie(path, frames(5))
    assert mp4_reading(path) == (b'avc1', 5)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['again.mp4']


def test_no_h264_refuses_and_writes_nothing(qt_app, tmp_path, monkeypatch):
    """Without H.264 the recorder would quietly write MPEG-4 Part 2,
    a file browsers and Keynote will not play. Refused instead."""
    monkeypatch.setattr(movie, '_unloadable', lambda: None)
    monkeypatch.setattr(movie, '_mp4_codecs', list)
    assert 'H.264' in movie.unavailable_reason()
    path = tmp_path / 'never.mp4'
    with pytest.raises(movie.MovieUnavailable, match='H.264'):
        movie.write_movie(path, frames(5))
    assert not path.exists()


def test_no_frames_is_refused(h264, tmp_path):
    with pytest.raises(ValueError, match='at least one frame'):
        movie.write_movie(tmp_path / 'empty.mp4', iter(()))


def test_cycles_are_whole_and_loop_without_a_stutter():
    phases = cycle_parameters(2.0, fps=30)
    assert len(phases) == 180, 'three 2 s cycles fill the six seconds'
    step = phases[1] - phases[0]
    assert phases[-1] + step == pytest.approx(3 * 2 * np.pi)
    # a slow cycle is one whole cycle, however long that takes
    assert len(cycle_parameters(16.0, fps=30)) == 480
    # a fast one is still whole cycles past six seconds
    fast = cycle_parameters(0.25, fps=30)
    assert len(fast) % round(0.25 * 30) == 0 and len(fast) >= 180


def test_a_sweep_visits_the_record_once_in_order():
    indices = sweep_parameters(1000, seconds=10.0, fps=30)
    assert len(indices) == 300
    assert indices[0] == 0 and indices[-1] < 1000
    assert np.all(np.diff(indices) > 0)
    # fewer samples than frames repeats them rather than inventing any
    short = sweep_parameters(10, seconds=1.0, fps=30)
    assert len(short) == 30 and set(short) == set(range(10))


def test_a_mode_shape_saves_as_a_movie_headless(h264, tmp_path):
    geometry, shapes = plate_geometry_and_shapes()
    path = tmp_path / 'mode.mp4'
    assert shapes.animate(geometry, 8, movie=str(path)) == str(path)
    assert mp4_reading(path) == (b'avc1', 180)


def test_a_screenshot_and_a_movie_together_are_refused(tmp_path):
    geometry, shapes = plate_geometry_and_shapes()
    with pytest.raises(ValueError, match='not both'):
        shapes.animate(geometry, 8, screenshot=str(tmp_path / 'a.png'),
                       movie=str(tmp_path / 'a.mp4'))


@pytest.fixture(scope='module')
def plate_run():
    from conftest import fixture_path

    import visualdynamics

    run = visualdynamics.random_vibration_run(fixture_path('plate',
                                                           'random.nc4'))
    geometry = visualdynamics.import_file(fixture_path('plate',
                                                       'test_geometry.npz'))
    geometry.define_units('m')
    return geometry, run.time_history


@pytest.mark.parametrize('seconds', [1.0])
def test_a_time_history_saves_once_through(h264, plate_run, tmp_path,
                                           seconds):
    geometry, history = plate_run
    path = tmp_path / 'record.mp4'
    history.animate(geometry, movie=str(path), seconds=seconds)
    assert mp4_reading(path) == (b'avc1', round(seconds * 30))


def test_a_time_history_still_is_its_furthest_sample(plate_run, tmp_path,
                                                     monkeypatch):
    """Sample 0 of a record is usually rest; the still is the sample
    where some record is furthest from zero."""
    from visualdynamics.viz import animate

    geometry, history = plate_run
    placed = []
    original = animate.GeometryAnimator.set_parameter
    monkeypatch.setattr(animate.GeometryAnimator, 'set_parameter',
                        lambda self, p: (placed.append(p),
                                         original(self, p)))
    path = tmp_path / 'still.png'
    history.animate(geometry, screenshot=str(path))
    assert path.stat().st_size > 2000
    furthest = int(np.abs(np.asarray(history.ordinate)).max(axis=0).argmax())
    assert placed[-1] == furthest != 0



def test_an_envelope_sweeps_its_lines_with_the_frequency_on_each(
        h264, tmp_path, monkeypatch):
    """The window's Play walks an envelope through its lines with the
    frequency in the corner; a script's movie does the same."""
    from conftest import fixture_path

    import visualdynamics
    from visualdynamics.viz import animate

    geometry = visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz'), length_unit='m')
    psd = visualdynamics.import_file(fixture_path('plate', 'psd.npz'))
    # what the corner says as each frame is taken — the caption actor is
    # made once and rewritten, so read it, not the calls that made it
    shown = []
    real = animate.movie_frames

    def watched(plotter, step, parameters):
        for image in real(plotter, step, parameters):
            shown.append(plotter.actors['scene-caption'].GetText(2))
            yield image

    monkeypatch.setattr(animate, 'movie_frames', watched)
    path = tmp_path / 'envelope.mp4'
    assert psd.animate(geometry, movie=str(path), seconds=2.0) == str(path)
    assert mp4_reading(path) == (b'avc1', 60)
    assert len(shown) == 60 and len(set(shown)) > 30, \
        'a caption a frame, naming the line it is on'
    lines = np.asarray(psd.abscissa)
    assert shown[0] == f'Envelope — {lines[0]:.5g} Hz'


def test_the_self_check_answers_for_this_machine(qt_app, tmp_path):
    """What the packaged builds are tested with: 0 and an H.264 file
    where the encoder is, 2 and the reason where it is not."""
    path = tmp_path / 'check.mp4'
    code, message = movie.check(path)
    if movie.unavailable_reason() is None:
        assert code == 0, message
        assert mp4_reading(path) == (b'avc1', movie.CHECK_FRAMES)
    else:
        assert (code, message) == (2, movie.unavailable_reason())


def test_the_self_check_refuses_a_wrong_file(qt_app, tmp_path,
                                             monkeypatch):
    """A file that is not H.264 with every frame fails the check, so a
    build whose Qt quietly wrote Part 2 cannot pass it."""
    monkeypatch.setattr(movie, 'unavailable_reason', lambda: None)
    part2 = _mp4(b'mp4v', 30)
    monkeypatch.setattr(movie, 'write_movie', lambda path, frames, fps=30:
                        (list(frames), path.write_bytes(part2)))
    code, message = movie.check(tmp_path / 'part2.mp4')
    assert code == 1 and "b'mp4v'" in message


def test_the_flag_runs_the_check_without_a_window(qt_app, tmp_path,
                                                  monkeypatch, capsys):
    from visualdynamics import gui
    from visualdynamics.gui import main_window

    monkeypatch.setattr(main_window, 'MainWindow',
                        lambda: pytest.fail('no window for a check'))
    monkeypatch.setattr(movie, 'check', lambda path: (2, f'asked {path}'))
    assert gui.main(['--check-movie', str(tmp_path / 'x.mp4')]) == 2
    assert capsys.readouterr().out.strip() == f'asked {tmp_path / "x.mp4"}'


def test_a_second_movie_keeps_every_frame(h264, tmp_path):
    """Qt 6.12 takes every frame offered while its encoder is still
    starting and keeps ten; the second recording in a process starts
    slowly enough to lose the rest. Paced by what reached the file,
    a movie after a movie is whole (2026-10-09)."""
    geometry, shapes = plate_geometry_and_shapes()
    movie.write_movie(tmp_path / 'first.mp4', frames(45))
    path = tmp_path / 'second.mp4'
    shapes.animate(geometry, 8, movie=str(path))
    assert mp4_reading(path) == (b'avc1', 180)


def test_an_encoder_that_stops_answering_fails_the_movie(h264, tmp_path,
                                                        monkeypatch):
    """Nothing back from the encoder ends the movie with the reason,
    rather than a window waiting on it for ever."""
    monkeypatch.setattr(movie, 'AHEAD', 0)          # never sends a frame
    monkeypatch.setattr(movie, 'STALL_MS', 300)
    with pytest.raises(RuntimeError, match='stopped taking frames'):
        movie.write_movie(tmp_path / 'stuck.mp4', frames(5))


def test_a_qt_multimedia_that_will_not_load_refuses_the_movie(qt_app,
                                                             tmp_path,
                                                             monkeypatch):
    """PySide6 6.12's Linux Qt Multimedia links PulseAudio; without
    `libpulse.so.0` the import fails, and it failed every animated
    selection on the public CI (2026-10-09). It is a reason now, like
    a missing encoder."""
    monkeypatch.setattr(movie, '_unloadable', lambda: (
        'libpulse.so.0: cannot open shared object file'))
    reason = movie.unavailable_reason()
    assert 'Qt Multimedia does not load' in reason and 'libpulse' in reason
    with pytest.raises(movie.MovieUnavailable, match='libpulse'):
        movie.write_movie(tmp_path / 'never.mp4', frames(3))
    assert movie.check(tmp_path / 'never.mp4') == (2, reason)


def test_a_movie_that_came_out_short_is_an_error(h264, tmp_path,
                                                 monkeypatch):
    """The pacing leans on how one Qt buffers frames; a Qt that buffers
    otherwise would write a short movie without a word, so the file is
    read back and a short one refused."""
    monkeypatch.setattr(movie, 'mp4_reading', lambda path: (b'avc1', 3))
    with pytest.raises(RuntimeError, match='3 of 12 frames'):
        movie.write_movie(tmp_path / 'short.mp4', frames(12))
