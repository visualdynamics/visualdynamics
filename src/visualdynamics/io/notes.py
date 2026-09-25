"""What a reader says about a file it imported anyway.

A reader that read a value differently from how the file wrote it — a
negative frequency taken as a rigid-body mode's zero — says so with an
`ImportNote`. The window records these, and only these, from an import
and shows them in its dialog: a warning of any other category is a
library's own business (an unclosed socket, a deprecation) and went
into the dialog the first day, which is how this class came to exist
(2026-09-25). A script sees the note as the warning it is.
"""

from __future__ import annotations


class ImportNote(UserWarning):
    """A reader's note about a file it imported: what it read
    differently from how the file wrote it, and why."""
