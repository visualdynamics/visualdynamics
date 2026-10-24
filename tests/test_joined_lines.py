"""No line of Python holds two pieces of code run together across a gap.

A line continuation lost between two edits leaves the second line's code
pasted onto the first after its indentation, `x = a(b)             if c
else d`. It still parses and ruff passes it, so nothing else notices. It
came in with five contributed pull requests in two days (2026-10-23/24),
each found by eye in review; this finds it by the gap between two tokens
on one physical line, which ordinary code never has (comments and the
text inside strings are not tokens between code, so alignment there is
untouched).
"""

from __future__ import annotations

import io
import os
import subprocess
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: the widest gap between two tokens of code on one line; a joined
#: continuation leaves the next line's whole indentation, eight or more
GAP = 4

_SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
         tokenize.DEDENT, tokenize.ENDMARKER}


def joined_lines(source: str) -> list[int]:
    """The line numbers in `source` where two tokens of code sit `GAP` or
    more spaces apart on one line."""
    found, prev = [], None
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in _SKIP:
            if tok.type in (tokenize.NEWLINE, tokenize.NL):
                prev = None
            continue
        if (prev is not None and prev.end[0] == tok.start[0]
                and tok.start[1] - prev.end[1] >= GAP):
            found.append(tok.start[0])
        prev = tok
    return found


def test_a_joined_continuation_is_found():
    joined = 'x = max(a, b)             if c else d\n'
    assert joined_lines(joined) == [1]
    split = 'x = (max(a, b)\n     if c else d)\n'
    assert joined_lines(split) == []
    aligned = "x = 1      # a comment set out to the right\ny = '  a    b  '\n"
    assert joined_lines(aligned) == []


def test_no_tracked_python_file_has_a_joined_line():
    files = subprocess.run(['git', 'ls-files', '*.py'], cwd=ROOT,
                           capture_output=True, text=True,
                           check=True).stdout.split()
    assert files, 'no Python files listed'
    bad = []
    for name in files:
        with open(os.path.join(ROOT, name), encoding='utf-8') as f:
            source = f.read()
        bad += [f'{name}:{line}' for line in joined_lines(source)]
    assert not bad, ('two pieces of code run together on one line; '
                     'a line continuation was lost: ' + ', '.join(bad))
