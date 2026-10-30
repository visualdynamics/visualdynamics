"""Save an animation as a video that plays almost anywhere.

The file is an MP4 holding H.264, which opens with nothing installed in
QuickTime, the Windows player, every browser, Keynote and PowerPoint,
and on a phone. It is also small for an animation, which is a still
background behind a smoothly moving mesh: a GIF of the same five
seconds measured seven times larger, and it limits the color map to 256
levels.

The encoder is Qt Multimedia's. The FFmpeg inside the PySide6 wheels is
an LGPL build without libx264, and it passes H.264 to the platform's own
encoder: VideoToolbox on macOS, Media Foundation on Windows, and a GPU's
VAAPI or NVENC on Linux. A Linux machine without a working GPU encoder
has no H.264 at all, and Qt then records MPEG-4 Part 2 with no warning,
a file five times the size that browsers and Keynote refuse. So the
codec is checked before anything is written, and a machine that cannot
provide it is told why (`unavailable_reason`) instead of being given a
file that will not play.
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

# The format comparison and the encoder survey: PLAN.md, "Saving an
# animation".

#: frames sent ahead of what the file holds. The encoder answers after
#: two or three (measured, VideoToolbox, Qt 6.12), so four never waits
#: on a frame it has not been given, and Qt 6.12 keeps ten of a burst
#: sent before it has started.
AHEAD = 4

#: milliseconds without a word from the encoder before the movie fails
STALL_MS = 10_000

#: the average bits a movie is encoded at, per pixel per frame. Qt's
#: quality setting asks for a constant quality, which the Mac's encoder
#: (VideoToolbox) does not offer, so it fell back on its own rate: 0.6
#: Mbit/s for a 1386 × 1552 view of a dense cone mesh, 0.01 bits a pixel,
#: and the element edges smeared (Brandon, 2026-10-25). Asked for, the
#: average is held loosely (measured on 2 s clips): the cone's frames
#: came out at 4.2 Mbit/s against 7.7 asked and 0.3 unasked, a plate
#: mode at 9.9 against 2.8 asked and 1.75 unasked, a synthetic mesh at
#: 21 against 3.3 asked and 6.1 unasked. In this mode the encoder spends
#: more on a simple scene at any figure (0.01 still gave the plate
#: 4.4), and the constant-quality modes ignore a rate. 0.04 brought most
#: of the cone's mesh back; Brandon chose 0.12, the larger files being
#: fine.
BITS_PER_PIXEL = 0.12


class MovieUnavailable(RuntimeError):
    """Raised when this machine cannot write an H.264 movie (it has no
    H.264 encoder, or Qt Multimedia does not load), so nothing is
    written."""


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
    """Why no movie can be saved on this machine, or None when one can.

    The reasons are that Qt Multimedia does not load, or that there is
    no H.264 encoder Qt can use.
    """
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


#: the most of the scene's height the plot under it takes in a movie
STRIP_SHARE = 0.4


def widget_frame(widget: Any, width: int, height: int) -> np.ndarray:
    """`widget` drawn as an RGB array exactly `width` × `height` pixels:
    scaled to fit, its shape kept, and the rest filled with the color
    of its own corner, so a plot narrower than the frame sits centered
    on its own background."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage

    image = widget.grab().toImage().convertToFormat(
        QImage.Format.Format_RGB888)
    image = image.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    rows = np.frombuffer(image.constBits(), np.uint8,
                         image.bytesPerLine() * image.height())
    drawn = rows.reshape(image.height(), image.bytesPerLine())[
        :, :3 * image.width()].reshape(image.height(), image.width(), 3)
    out = np.empty((height, width, 3), np.uint8)
    out[:] = drawn[0, 0]
    top = (height - drawn.shape[0]) // 2
    left = (width - drawn.shape[1]) // 2
    out[top:top + drawn.shape[0], left:left + drawn.shape[1]] = drawn
    return out


def over_plot(frames: Iterable[np.ndarray], widget: Any,
              share: float = STRIP_SHARE) -> Any:
    """Each of `frames` with `widget` drawn under it, as the window
    shows a record's plot under its animation: the plot, with its cursor
    where the frame is, says where in the record each frame is.

    The plot is grabbed fresh for every frame, so whatever moved it
    (the cursor, an envelope's caption) is in that frame. It is as wide
    as the scene and as tall as its shape allows, up to `share` of the
    scene's height, and the stacked frame is kept to an even height for
    the H.264 encoder.
    """
    for frame in frames:
        frame = np.asarray(frame)[..., :3]
        height, width = frame.shape[:2]
        aspect = widget.height() / max(widget.width(), 1)
        strip = min(int(share * height), round(width * aspect))
        strip -= (height + strip) % 2
        yield np.vstack([frame, widget_frame(widget, width, strip)])


def write_movie(path: str | Path, frames: Iterable[np.ndarray],
                fps: int = MOVIE_FPS) -> int:
    """Encode `frames` into an H.264 MP4 at `path`, overwriting it.

    Frames are taken one at a time as the encoder asks for them, so a
    long animation is never held in memory. User input to the window is
    held off while it runs, because a click could rebuild the scene the
    frames come from. The file is read back afterward and its codec and
    frame count are checked.

    Parameters
    ----------
    path : str or Path
        The file to write.
    frames : iterable of ndarray
        RGB or RGBA images, (height, width, 3 or 4) arrays read as
        uint8, all the same size.
    fps : int
        Frames per second, `viz.animate.MOVIE_FPS` by default.

    Returns
    -------
    int
        How many frames were written.

    Raises
    ------
    MovieUnavailable
        When the machine cannot encode H.264; nothing is written.
    ValueError
        When `frames` is empty.
    RuntimeError
        When the encoder reports an error or takes no frames for
        `STALL_MS` (10 s), the frames change size partway through, or the
        file read back is not H.264 holding every frame.

    Notes
    -----
    The encoder is asked for an average of `BITS_PER_PIXEL` bits per pixel
    per frame, 7.7 Mbit/s for a 1386 × 1552 view at 30 frames a second,
    so a fine mesh's edges are kept. It takes the figure as a target
    rather than a rule: a view it compresses well comes out under it
    (about half a megabyte a second of video for a view that size of a
    dense cone), and detail it cannot compress comes out over it.
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
    recorder.setEncodingMode(
        QMediaRecorder.EncodingMode.AverageBitRateEncoding)
    recorder.setVideoBitRate(round(BITS_PER_PIXEL * width * height * fps))
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
    """The codec's four-character code and the frame count, read from an
    MP4's own boxes.

    The sample description says what the frames were encoded as, and
    the sample-size table how many there are. `b'avc1'` is H.264, and
    `b'mp4v'` is the MPEG-4 Part 2 file Qt writes when it has no H.264
    encoder. Returns (None, 0) when the file has no sample table.

    The file is walked box by box and never searched for bytes, because
    the encoded frames can hold any four bytes.
    """
    # Qt 6.12 writes the index after the frames, and a byte search for
    # 'stsz' once found one inside a frame and read its count as 10
    # (2026-10-09).
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
    """Check that this installation can save an animation: render one
    second of a turning mesh off screen, encode it to `path`, and read
    the file back.

    `visualdynamics --check-movie out.mp4` runs this and exits with the
    returned code. The packaged builds are tested with it (the Windows
    release smoke test and the macOS build), because the files a bundle
    carries do not show whether the encoder inside them works.

    Parameters
    ----------
    path : str or Path
        The movie to write.

    Returns
    -------
    tuple of (int, str)
        0 and what was written when the file is H.264 with every frame;
        2 and the reason when the machine has no H.264 encoder, or no
        OpenGL to render with (`opengl.ready`); 1 and what went wrong
        otherwise.
    """
    from .opengl import refusal, three_d_view

    reason = unavailable_reason()
    if reason is not None:
        return 2, reason
    plotter = three_d_view(offscreen=True, window_size=(320, 240))
    if plotter is None:
        return 2, refusal()
    import pyvista as pv

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
