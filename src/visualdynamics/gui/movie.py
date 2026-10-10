"""An animation, saved as a video anyone can play.

MP4 holding H.264: the one pairing that opens with nothing installed in
QuickTime, the Windows player, every browser, Keynote and PowerPoint,
and on a phone. It is also small for what an animation is — a still
background behind a smoothly moving mesh — where a GIF of the same five
seconds measured seven times larger and bands the color map to 256
levels (PLAN.md, "Saving an animation").

The encoder is Qt Multimedia's: the FFmpeg inside the PySide6 wheels is
an LGPL build with no libx264, and it hands H.264 to the platform's own
encoder — VideoToolbox on macOS, Media Foundation on Windows, a GPU's
VAAPI or NVENC on Linux. That last is the catch. A Linux machine
without a working GPU encoder has no H.264 at all, and Qt then records
MPEG-4 Part 2 without a word — five times the size, and refused by
browsers and Keynote. So the codec is asked for before anything is
written, and a machine that cannot give it is told so rather than
handed a file that will not play (`unavailable_reason`).
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from ..plot import _application
from ..viz.animate import MOVIE_FPS

#: frames sent ahead of what the file holds. The encoder answers after
#: two or three (measured, VideoToolbox, Qt 6.12), so four never waits
#: on a frame it has not been given, and Qt 6.12 keeps ten of a burst
#: sent before it has started.
AHEAD = 4

#: milliseconds without a word from the encoder before the movie fails
STALL_MS = 10_000


class MovieUnavailable(RuntimeError):
    """This machine cannot encode H.264, so no movie is written."""


@cache
def _mp4_codecs() -> list:
    """The video codecs this machine can encode into an MP4 — asked
    once, since the window asks on every selection whether to offer
    the save. Names, not enum members, so a machine whose Qt
    Multimedia cannot load answers too (`_unloadable`)."""
    from PySide6.QtMultimedia import QMediaFormat

    _application()
    return [codec.name for codec in
            QMediaFormat(QMediaFormat.FileFormat.MPEG4)
            .supportedVideoCodecs(QMediaFormat.ConversionMode.Encode)]


@cache
def _unloadable() -> str | None:
    """Why Qt Multimedia will not load here, or None. Its Linux build
    links PulseAudio, and a machine without `libpulse.so.0` — GitHub's
    Ubuntu runner, a minimal desktop — fails the import (PySide6 6.12,
    2026-10-09). Asked once: a failed import is not cached by Python,
    and the window asks on every selection."""
    try:
        import PySide6.QtMultimedia  # noqa: F401
    except ImportError as error:
        return str(error)
    return None


def unavailable_reason() -> str | None:
    """Why no movie can be saved here, or None when one can."""
    failure = _unloadable()
    if failure is not None:
        return (f'Qt Multimedia does not load on this computer '
                f'({failure}), so an animation cannot be saved as a '
                'video here')
    if 'H264' in _mp4_codecs():
        return None
    return ('This computer has no H.264 video encoder that Qt can use '
            '(on Linux it needs a working VAAPI or NVENC GPU encoder), '
            'so an animation cannot be saved as a video here')


def _image(frame: np.ndarray) -> Any:
    """A screenshot as an RGBA QImage that owns its pixels — the encoder
    reads a frame on its own thread, after the array may be gone. An
    RGB screenshot is converted by Qt rather than padded in numpy: 0.07
    against 3.5 ms a frame at 1024 × 768 (measured 2026-10-10)."""
    from PySide6.QtGui import QImage

    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    height, width, depth = frame.shape
    if depth == 3:
        return QImage(frame.data, width, height, 3 * width,
                      QImage.Format.Format_RGB888).convertToFormat(
                          QImage.Format.Format_RGBA8888)
    return QImage(frame.data, width, height, 4 * width,
                  QImage.Format.Format_RGBA8888).copy()


def write_movie(path: str | Path, frames: Iterable[np.ndarray],
                fps: int = MOVIE_FPS) -> int:
    """Encode `frames` — RGB or RGBA images, all one size — into an
    H.264 MP4 at `path`, overwriting it.

    Frames are pulled one at a time as the encoder asks for them, so a
    long animation is never held in memory. Input to the window is held
    off while it runs: the frames come from a scene a click could
    rebuild underneath them.

    Returns
    -------
    int
        How many frames were written.

    Raises
    ------
    MovieUnavailable
        When the machine has no H.264 encoder; nothing is written.
    """
    # asked before Qt Multimedia is imported: where it cannot load, the
    # reason is the answer, not an ImportError
    reason = unavailable_reason()
    if reason is not None:
        raise MovieUnavailable(reason)
    from PySide6.QtCore import QEventLoop, QSize, QTimer, QUrl
    from PySide6.QtMultimedia import (
        QMediaCaptureSession,
        QMediaFormat,
        QMediaRecorder,
        QVideoFrame,
        QVideoFrameFormat,
        QVideoFrameInput,
    )

    source_frames = iter(frames)
    first = next(source_frames, None)
    if first is None:
        raise ValueError('an animation needs at least one frame')
    first = _image(first)
    width, height = first.width(), first.height()

    path = Path(path).resolve()

    media = QMediaFormat(QMediaFormat.FileFormat.MPEG4)
    media.setVideoCodec(QMediaFormat.VideoCodec.H264)
    frame_format = QVideoFrameFormat(
        QSize(width, height), QVideoFrameFormat.PixelFormat.Format_RGBA8888)
    frame_format.setStreamFrameRate(fps)
    source = QVideoFrameInput(frame_format)
    session = QMediaCaptureSession()
    session.setVideoFrameInput(source)
    recorder = QMediaRecorder()
    session.setRecorder(recorder)
    recorder.setMediaFormat(media)
    recorder.setVideoResolution(width, height)
    recorder.setVideoFrameRate(fps)
    recorder.setQuality(QMediaRecorder.Quality.HighQuality)
    recorder.setOutputLocation(QUrl.fromLocalFile(str(path)))
    recorder.setAutoStop(True)          # an empty frame ends the file

    state = {'frames': source_frames, 'pending': first, 'sent': 0,
             'ended': False, 'error': None, 'finished': False}
    loop = QEventLoop()
    # nothing heard from the encoder for this long ends the movie with
    # an error, rather than a window waiting on it for ever
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.setInterval(STALL_MS)

    def muxed() -> int:
        """Frames the file holds so far, by its duration."""
        return round(recorder.duration() * fps / 1000)

    def send() -> None:
        watchdog.start()
        while not state['ended']:
            if state['sent'] - muxed() >= AHEAD:
                # paced by what reached the file, not by what the input
                # took: Qt 6.12 takes every frame while its encoder is
                # still starting and keeps ten (2026-10-09). The next
                # durationChanged sends on.
                return
            image = state['pending']
            if image is None:
                frame = next(state['frames'], None)
                if frame is None:
                    # the end marker queues like a frame: refused when
                    # the encoder is full, and then the recorder never
                    # stops — sent again on the next ready signal
                    state['ended'] = source.sendVideoFrame(QVideoFrame())
                    return
                image = _image(frame)
                if (image.width(), image.height()) != (width, height):
                    # raised here it would vanish into Qt's slot call
                    state['error'] = 'the frames changed size mid-movie'
                    state['frames'] = iter(())
                    continue
            video = QVideoFrame(image)
            video.setStartTime(state['sent'] * 1_000_000 // fps)
            video.setEndTime((state['sent'] + 1) * 1_000_000 // fps)
            if not source.sendVideoFrame(video):
                # the encoder is full: keep this one for when it asks
                state['pending'] = image
                return
            state['pending'] = None
            state['sent'] += 1

    def changed(recording) -> None:
        if recording == QMediaRecorder.RecorderState.RecordingState:
            send()
        elif recording == QMediaRecorder.RecorderState.StoppedState:
            state['finished'] = True
            loop.quit()

    def failed(_error, message) -> None:
        state['error'] = state['error'] or message
        state['finished'] = True
        loop.quit()

    def stalled() -> None:
        if state['ended']:
            return          # every frame is in; the file is being closed
        state['error'] = (f'the encoder stopped taking frames after '
                          f'{muxed()} of them')
        state['frames'], state['pending'] = iter(()), None
        recorder.stop()

    connected = ((watchdog.timeout, stalled),
                 (source.readyToSendVideoFrame, send),
                 (recorder.durationChanged, lambda _duration: send()),
                 (recorder.recorderStateChanged, changed),
                 (recorder.errorOccurred, failed))
    for signal, slot in connected:
        signal.connect(slot)
    recorder.record()
    if not state['finished']:
        loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    watchdog.stop()
    # the slots hold the recorder, the input and the timer they are
    # connected to — a cycle Python cannot see through Qt — so a save
    # kept all of it alive for the life of the process until let go
    for signal, slot in connected:
        signal.disconnect(slot)
    if state['error']:
        raise RuntimeError(f'The movie could not be written: '
                           f'{state["error"]}')
    # read back, not trusted: the pacing above leans on how one Qt
    # buffers frames, and a Qt that buffers otherwise would write a
    # short movie without a word (Principle 12 takes each Qt as it comes)
    codec, kept = mp4_reading(path)
    if codec != b'avc1' or kept != state['sent']:
        raise RuntimeError(
            f'The movie came out wrong: {kept} of {state["sent"]} frames '
            f'as {(codec or b"nothing").decode(errors="replace")}')
    return state['sent']


def _boxes(data: bytes, start: int, end: int):
    """(type, body start, body end) of each MP4 box in data[start:end]."""
    at = start
    while at + 8 <= end:
        size, kind = struct.unpack('>I4s', data[at:at + 8])
        head = 8
        if size == 1:                       # a 64-bit size follows
            size = struct.unpack('>Q', data[at + 8:at + 16])[0]
            head = 16
        elif size == 0:                     # to the end of its parent
            size = end - at
        if size < head:
            return
        yield kind, at + head, min(at + size, end)
        at += size


def _box(data: bytes, path: tuple[bytes, ...], start: int = 0,
         end: int | None = None) -> tuple[int, int] | None:
    """The body of the box at `path` (moov, trak, ...), or None."""
    end = len(data) if end is None else end
    for kind, body, stop in _boxes(data, start, end):
        if kind == path[0]:
            return ((body, stop) if len(path) == 1
                    else _box(data, path[1:], body, stop))
    return None


def mp4_reading(path: str | Path) -> tuple[bytes | None, int]:
    """(codec four-cc, frame count) read from an MP4's own boxes: the
    sample description says what the frames were encoded as, the
    sample-size table how many there are. `b'avc1'` is H.264; `b'mp4v'`
    is the Part 2 file Qt writes when it has no H.264 to give.

    Walked as boxes, never searched as bytes: the encoded frames can
    hold any four bytes, and once the index followed the frames (Qt
    6.12 writes it last) a search for 'stsz' found one inside a frame
    and read its count as 10 (2026-10-09).
    """
    data = Path(path).read_bytes()
    table = _box(data, (b'moov', b'trak', b'mdia', b'minf', b'stbl'))
    if table is None:
        return None, 0
    sizes = _box(data, (b'stsz',), *table)
    count = (struct.unpack('>I', data[sizes[0] + 8:sizes[0] + 12])[0]
             if sizes else 0)
    described = _box(data, (b'stsd',), *table)
    codec = None
    if described:
        # version/flags and an entry count, then the first entry's box
        entry = described[0] + 8
        codec = bytes(data[entry + 4:entry + 8])
    return codec, count


#: frames in the packaged build's self-check: a second of video
CHECK_FRAMES = MOVIE_FPS


def check(path: str | Path) -> tuple[int, str]:
    """Can this installation save an animation? Render a second of a
    turning mesh off screen, encode it, and read the file back.

    `visualdynamics --check-movie out.mp4` runs this and exits with the
    code: what a packaged build is tested with, since the files a
    bundle carries say nothing about whether the encoder inside them
    answers (the Windows release smoke test, the macOS build).

    Returns
    -------
    tuple of (int, str)
        0 and what was written when the file is H.264 with every frame;
        2 and the reason when the machine has no H.264 encoder; 1 and
        what went wrong otherwise.
    """
    import pyvista as pv

    reason = unavailable_reason()
    if reason is not None:
        return 2, reason
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    try:
        sphere = pv.Sphere()
        plotter.add_mesh(sphere, scalars=np.arange(sphere.n_points) % 7,
                         show_scalar_bar=False)

        def frames():
            for _ in range(CHECK_FRAMES):
                plotter.camera.azimuth = plotter.camera.azimuth + 6
                plotter.render()
                yield plotter.screenshot(return_img=True)

        write_movie(path, frames())
    except RuntimeError as error:  # the recorder's own failure
        return 1, str(error)
    finally:
        plotter.close()
    codec, count = mp4_reading(path)
    if codec != b'avc1' or count != CHECK_FRAMES:
        return 1, (f'wrote {count} frames as {codec!r}, expected '
                   f'{CHECK_FRAMES} as H.264')
    return 0, (f'H.264 MP4 with {count} frames, '
               f'{Path(path).stat().st_size} bytes: {path}')
