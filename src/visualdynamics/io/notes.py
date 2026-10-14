"""The warning a reader gives about a file it imported anyway.

A reader that reads a value differently from how the file wrote it (a
negative frequency read as a rigid-body mode's zero, for example) says
so with an `ImportNote`. The window records these, and only these, from
an import and shows them in its dialog; a warning of any other category
(an unclosed socket, a deprecation) belongs to the library that raised
it. A script sees the note as an ordinary warning.
"""

from __future__ import annotations

# The class exists because on the first day every warning went into the
# import dialog, library warnings included (2026-09-25).


class ImportNote(UserWarning):
    """A reader's note about a file it imported: what it read differently
    from how the file wrote it, and why."""
