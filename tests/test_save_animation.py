"""Save Animation: the 3-D view's bar, beside Copy, while something moves.

The movie is the window's own playback written down — its camera, its
scale, its speed — so these pin what a viewer would check: the button
is there exactly when it can work, the file holds the frames the rule
says at the speed the window is playing, and the window is left
playing where it was (PLAN.md, "Saving an animation").
"""

from __future__ import annotations

import pytest

from visualdynamics.gui import movie
from visualdynamics.gui.main_window import SPEEDS
from visualdynamics.gui.movie import mp4_reading


def _bar(pane):
    """The act buttons showing on a pane's bar, by label."""
    actions = pane.__dict__.get('_acts', {}).get('actions', {}).values()
    return [a.text() for a in pane.toolbar.actions()
            if a in actions and a.isVisible()]


def test_a_mode_shape_offers_it_beside_copy(h264, shaking):
    assert _bar(shaking.scene)[-2:] == ['Copy', 'Save Animation']


def test_a_time_history_offers_it_on_the_scene_while_copy_is_the_plots(
        h264, playing):
    """Copy follows the plot when the plot is up; the movie is the
    scene's, so it stays on the scene's bar."""
    assert _bar(playing.data_pane)[-1] == 'Copy'
    assert _bar(playing.scene) == ['Save Animation']


def test_nothing_moving_offers_no_movie(h264, shaking, pump):
    shaking.tree.clearSelection()
    shaking._item_for_object('Geometry').setSelected(True)
    shaking.render_current()
    pump()
    assert 'Save Animation' not in _bar(shaking.scene)
    assert 'Copy' in _bar(shaking.scene)


def test_no_h264_offers_no_movie(shaking, monkeypatch):
    """Where every save would be refused, the button is not offered
    (Principle 1)."""
    monkeypatch.setattr(movie, '_mp4_codecs', list)
    shaking._offer_acts()
    assert 'Save Animation' not in _bar(shaking.scene)


@pytest.mark.parametrize('speed, frames', [(1.0, 180), (0.5, 240)])
def test_a_mode_saves_whole_cycles_at_the_playing_speed(
        h264, shaking, tmp_path, speed, frames):
    shaking._speed_index = SPEEDS.index(speed)
    path = tmp_path / 'mode.mp4'
    assert shaking.save_animation(str(path)) == str(path)
    assert mp4_reading(path) == (b'avc1', frames)
    assert shaking.statusBar().currentMessage().startswith('Saved mode.mp4')


def test_a_record_saves_once_through_and_playback_comes_back(
        h264, playing, tmp_path):
    playing._speed_index = SPEEDS.index(2.0)
    playing.set_playing(True)
    held = playing._cursor_abscissa[700]
    playing._cursor.setValue(float(held))
    path = tmp_path / 'record.mp4'
    playing.save_animation(str(path))
    assert mp4_reading(path) == (b'avc1', 150), '10 s at 2x is 5 s'
    assert playing._cursor.value() == pytest.approx(held)
    assert playing._frame_index == 700
    assert playing._playing, 'it was playing, so it plays on'


def test_a_mode_is_left_at_its_phase_and_paused_if_it_was(
        h264, shaking, tmp_path):
    shaking.set_playing(False)
    shaking.phase_slider.setValue(37)
    before = shaking.animator.view.copy()
    shaking.save_animation(str(tmp_path / 'mode.mp4'))
    assert shaking.phase_slider.value() == 37
    assert (shaking.animator.view == before).all(), \
        'the nodes are back where the slider says'
    assert not shaking._playing


def test_the_button_saves_through_the_dialog(h264, shaking, tmp_path,
                                             monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    target = tmp_path / 'chosen'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (str(target), '')))
    shaking.scene._acts['actions']['movie'].trigger()
    assert mp4_reading(tmp_path / 'chosen.mp4')[0] == b'avc1'


def test_a_cancelled_dialog_saves_nothing(h264, shaking, tmp_path,
                                          monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    assert shaking.save_animation() is None
    assert list(tmp_path.iterdir()) == []


def test_a_records_frames_go_through_the_cursor(h264, playing, tmp_path):
    """The cursor is how a sweep's frame is drawn — and where an
    envelope rewrites its caption per line — so the movie walks it
    rather than setting the animator behind its back."""
    moves = []
    playing._cursor.sigPositionChanged.connect(lambda *_: moves.append(1))
    playing.save_animation(str(tmp_path / 'record.mp4'))
    assert len(moves) >= 300, 'one cursor move a frame, 10 s at 30 fps'
