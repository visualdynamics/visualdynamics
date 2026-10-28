"""The guide's figure tools draw where they are told and nothing on a
help request.

A review ran `tools/make_concept_figures.py --help`; the script took no
arguments, ran whole, and rewrote every concept figure in the guide
(2026-10-28). Both tools parse their arguments now.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGES = ROOT / 'docs' / 'guide' / 'images'


def _run(*args):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, *args], capture_output=True, check=False,
                          text=True, env=env, cwd=ROOT, timeout=600)


def _stamps():
    return {path.name: path.stat().st_mtime_ns for path in IMAGES.glob('*.png')}


@pytest.mark.parametrize('tool', ['make_concept_figures', 'make_guide_screenshots'])
def test_help_draws_nothing(tool):
    before = _stamps()
    done = _run(ROOT / 'tools' / f'{tool}.py', '--help')
    assert done.returncode == 0, done.stderr
    assert 'usage:' in done.stdout and '--out' in done.stdout
    assert _stamps() == before, 'the guide was not touched'


def test_one_figure_is_drawn_where_it_is_told(tmp_path):
    before = _stamps()
    done = _run(ROOT / 'tools' / 'make_concept_figures.py', '--only',
                'psd_reading', '--out', str(tmp_path))
    assert done.returncode == 0, done.stderr
    assert sorted(p.name for p in tmp_path.glob('*.png')) == [
        'psd-reading-law.png', 'psd-reading-octave.png',
        'psd-reading-steps.png']
    assert _stamps() == before
    refused = _run(ROOT / 'tools' / 'make_concept_figures.py', '--only',
                   'nothing_of_the_kind', '--out', str(tmp_path))
    assert refused.returncode != 0 and 'no figure named' in refused.stderr
