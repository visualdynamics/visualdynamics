"""Faster and Slower, for both kinds of animation.

A mode shape sweeps its phase and a time history walks its cursor, and
until now each did it at one fixed rate — two seconds a cycle, ten
seconds a record. Neither is right for every model: a mode that reads
well on a beam is a blur on a 400-node survey, and a long capture takes
a coffee break to watch.

The two buttons flank the transport, which is where a media player puts
them, and step a ladder of doublings rather than offering a number
nobody wants to type.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import TimeHistory
from visualdynamics.gui.main_window import (
    FRAMES_PER_SECOND,
    NORMAL_SPEED,
    PHASE_STEPS,
    SECONDS_PER_CYCLE,
    SECONDS_PER_RECORD,
    SPEEDS,
)

FRAMES = 31          # not a whole number of cycles at any speed


def _geometry():
    geometry = visualdynamics.import_file(fixture_path('plate', 'geometry.npz'))
    geometry.define_units('m')
    return geometry


@pytest.fixture
def shaking(window, pump):
    """A mode shape animating on its geometry."""
    geometry = _geometry()
    window.add_object('Geometry', geometry)
    window.add_object('Shapes',
                      visualdynamics.import_file(fixture_path('plate',
                                                     'shapes.npy')))
    window.tree.clearSelection()
    for name in ('Geometry', 'Shapes'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    return window


@pytest.fixture
def playing(window, pump):
    """A time history animating on its geometry."""
    geometry = _geometry()
    window.add_object('Geometry', geometry)
    dofs = [f'{int(node)}Z+' for node in geometry.node_id[:20]]
    t = np.arange(2048) / 512.0
    window.add_object('Time', TimeHistory(
        t, np.sin(2 * np.pi * 3 * t)[None, :].repeat(len(dofs), 0),
        response_dof=dofs, ordinate_dim='acceleration',
        ordinate_unit='m/s**2'))
    window.tree.clearSelection()
    for name in ('Geometry', 'Time'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    return window


def _traveled(window, frames, position, wrap):
    """How far the animation moved over `frames`, wraps counted in."""
    setattr(window, position, 0.0)
    moved = 0.0
    for _ in range(frames):
        before = getattr(window, position)
        window._advance()
        moved += (getattr(window, position) - before) % wrap
    return moved


# ---- the ladder ---------------------------------------------------------


def test_normal_is_one_and_the_ladder_doubles():
    assert SPEEDS[NORMAL_SPEED] == 1.0
    assert all(b == pytest.approx(2 * a) for a, b in itertools.pairwise(SPEEDS))


def test_the_buttons_step_the_ladder(shaking, pump):
    window = shaking
    assert window.speed == 1.0
    window._change_speed(1)
    assert window.speed == 2.0
    window._change_speed(-1)
    window._change_speed(-1)
    assert window.speed == 0.5
    assert '0.5x' in window.statusBar().currentMessage()


def test_the_ends_of_the_ladder_hold(shaking, pump):
    window = shaking
    for _ in range(len(SPEEDS) + 3):
        window._change_speed(1)
    assert window.speed == SPEEDS[-1]
    assert not window.faster_action.isEnabled(), 'and says so'
    assert window.slower_action.isEnabled()
    for _ in range(len(SPEEDS) + 3):
        window._change_speed(-1)
    assert window.speed == SPEEDS[0]
    assert not window.slower_action.isEnabled()


def test_the_tooltips_carry_the_speed(shaking):
    """Two arrow glyphs are not a readout, so the number lives here."""
    window = shaking
    window._change_speed(1)
    assert '2x' in window.slower_action.toolTip()
    assert '2x' in window.faster_action.toolTip()


def test_both_buttons_sit_with_the_transport(shaking):
    """Visible exactly when the play button is, and not otherwise."""
    window = shaking
    assert window.play_action.isVisible()
    assert window.slower_action.isVisible()
    assert window.faster_action.isVisible()
    window._show_animation_controls(False)
    assert not window.slower_action.isVisible()
    assert not window.faster_action.isVisible()


# ---- and what they actually do ------------------------------------------


@pytest.mark.parametrize('speed', SPEEDS)
def test_a_mode_shape_sweeps_at_the_speed_asked_for(shaking, speed):
    window = shaking
    window.set_playing(True)
    window._speed_index = SPEEDS.index(speed)
    moved = _traveled(window, FRAMES, '_phase_position', PHASE_STEPS)
    wanted = FRAMES * PHASE_STEPS / (SECONDS_PER_CYCLE * FRAMES_PER_SECOND)
    assert moved == pytest.approx(wanted * speed)


@pytest.mark.parametrize('speed', SPEEDS)
def test_a_time_history_walks_at_the_speed_asked_for(playing, speed):
    window = playing
    assert not window._shape_mode
    total = len(window._cursor_abscissa)
    window._speed_index = SPEEDS.index(speed)
    moved = _traveled(window, FRAMES, '_frame_position', total)
    wanted = FRAMES * total / (SECONDS_PER_RECORD * FRAMES_PER_SECOND)
    assert moved == pytest.approx(wanted * speed)


def test_a_slow_speed_is_slower_and_not_merely_the_same(shaking):
    """The reason the position is a float.

    Stepping an integer cannot go slower than one step a frame — the
    `max(1, ...)` that keeps it moving sees to that — so every speed under
    about a half used to come out identical, which is precisely the range
    anyone reaches for Slower to get to.
    """
    window = shaking
    window.set_playing(True)
    traveled = []
    for speed in (0.125, 0.25, 0.5, 1.0):
        window._speed_index = SPEEDS.index(speed)
        traveled.append(_traveled(window, FRAMES, '_phase_position',
                                    PHASE_STEPS))
    assert all(b > a for a, b in itertools.pairwise(traveled)), traveled
    assert traveled[-1] == pytest.approx(8 * traveled[0])


def test_dragging_the_cursor_moves_the_playback_position(playing):
    """An external move is a statement about where playback is; the
    accumulator has to follow it or the next frame jumps back."""
    window = playing
    window._speed_index = SPEEDS.index(1.0)
    for _ in range(5):
        window._advance()
    assert window._frame_position > 0
    window._cursor.setValue(float(window._cursor_abscissa[1000]))
    assert int(window._frame_position) == pytest.approx(1000, abs=2)


def test_dragging_the_phase_slider_moves_it_too(shaking):
    window = shaking
    window.set_playing(True)
    for _ in range(5):
        window._advance()
    window.phase_slider.setValue(90)
    assert int(window._phase_position) == 90


# ---- the mode and the scale, stepped the same way ------------------------


def test_the_mode_and_scale_boxes_step_their_value(shaking, pump):
    """The boxes wear their own up/down arrows, like the wavelet
    panel's (Brandon, 2026-08-28).

    They used to be flanked by two toolbar arrows with Qt's stacked
    spinner turned off, on the argument that the stacked pair is a
    pixel-hunt. Put beside the plain boxes, the plain ones read better:
    the arrows belong to the number, where two toolbar buttons either
    side of a box only look as though they do.
    """
    from PySide6.QtWidgets import QAbstractSpinBox

    window = shaking
    assert all(a.isVisible() for a in window._mode_actions)
    assert all(a.isVisible() for a in window._scale_actions)
    for box in (window.mode_box, window.scale_box):
        assert box.buttonSymbols() != QAbstractSpinBox.ButtonSymbols.NoButtons, \
            'the box has its own arrows'

    start = window.mode_box.value()
    window.mode_box.stepUp()
    window.mode_box.stepUp()
    pump()
    assert window.mode_box.value() == start + 2
    window.mode_box.stepDown()
    pump()
    assert window.mode_box.value() == start + 1

    scale_start = window.scale_box.value()
    window.scale_box.stepUp()
    pump()
    assert window.scale_box.value() > scale_start
    window.scale_box.stepDown()
    pump()
    assert window.scale_box.value() == scale_start

    for box in (window.mode_box, window.scale_box):
        assert not box.isReadOnly(), 'the number stays editable'


def test_stepping_the_mode_keeps_the_phase_and_says_which_one(shaking, pump):
    """Browsing modes mid-swing must not snap the model straight, and the
    scene has to say which mode it is now showing."""
    window = shaking
    window.set_playing(True)
    window.phase_slider.setValue(40)
    pump()
    window.set_playing(False)
    phase = window.animator.parameter
    shapes = window.objects['Shapes']

    window.mode_box.setValue(3)
    pump()
    assert window.animator.parameter == pytest.approx(phase), (
        'the phase carries over to the new mode')
    assert shapes.mode_label(2) in window.statusBar().currentMessage()


def test_a_scale_change_redraws_without_rebuilding(shaking, pump):
    window = shaking
    animator = window.animator
    window.scale_box.setValue(window.scale_box.value() * 2)
    pump()
    assert window.animator is animator, 'the same animator, rescaled'


def test_the_mode_and_scale_arrows_go_with_the_animation(shaking, window,
                                                          pump):
    window.tree.clearSelection()
    window._item_for_object('Geometry').setSelected(True)
    window.render_current()
    pump()
    assert not any(a.isVisible() for a
                   in window._mode_actions + window._scale_actions)
