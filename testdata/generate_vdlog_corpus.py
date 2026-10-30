"""Freeze one run log per format version, so no release can orphan a log.

Run when `run_log.FORMAT_VERSION` moves to write that version's file
(`vdlog_corpus/v<N>.vdlog`); every earlier file stays as it is.
`tests/test_run_log.py` opens every file here with the current build,
verifies its chains and pins values a reader that regressed would
lose. The log is small on purpose: one entry of most kinds, two tests
and the entries between them, a correction, a clock check and a file,
written on a fixed clock and under a fixed login and machine name.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

from visualdynamics.core import run_log

ZONE = timezone(timedelta(hours=-6))
START = datetime(2026, 10, 30, 8, 0, tzinfo=ZONE)


def write(path: str) -> None:
    ticks = iter(range(10_000))

    def clock() -> datetime:
        return START + timedelta(seconds=30 * next(ticks))

    with run_log.RunLog.create(path, title='Corpus article', clock=clock,
                               who=('tester', 'bench')) as log:
        log.append('clock_check', author='AB', fields={'offset_s': 1.25,
                                                       'recorder': 'rack'})
        log.append('person_in', author='AB', fields={'person': 'CD',
                                                     'role': 'operator'})
        log.append('test_opened', author='AB', test=1,
                   fields={'title': 'Random, Z', 'test_kind': 'Random '
                           'Vibration'})
        log.append('level_row', author='AB', test=1, axis='Z', run=1,
                   fields={'level_db': -12.0, 'voltage': 20.0,
                           'current': 63.0, 'drive_fraction': 0.25,
                           'answer': 'as predicted', 'go': True})
        abort = log.append('abort', author='AB', test=1, axis='Z', run=2,
                           fields={'stop': 'amplifier trip'})
        log.correct(abort, author='AB', fields={'channel': '4'},
                    text='channel read off the panel')
        log.append('deviation', author='AB', test=1,
                   fields={'fact': 'a notch at 180 Hz not in the plan'})
        log.append('note', author='AB', test=1, text='Résumé: ünïcode kept',
                   files=[run_log.FileRef('run2.csv', '0' * 64, 12)])
        log.append('test_closed', author='AB', test=1)
        log.append('test_opened', author='AB', test=2,
                   fields={'title': 'Shock, Z'})
        log.append('shot', author='AB', test=2, axis='Z', run=1,
                   fields={'polarity': 'positive', 'n': 1, 'of': 3})


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'vdlog_corpus', f'v{run_log.FORMAT_VERSION}.vdlog')
    if os.path.exists(out):
        sys.exit(f'{out} is frozen; it is never written again')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write(out)
    print(out)
