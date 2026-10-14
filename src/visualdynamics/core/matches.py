"""Matched mode pairs between two shape sets.

Squares picked on a MAC plot and committed become a project object of
their own, which can be saved, renamed and edited row by row. The two
shape sets are referenced by name and looked up each time the matches
are shown. The MAC values are stored with the pairs, because they came
from the comparison as it was displayed, projected onto the basis DOFs
when the geometries required it, and recomputing them from DOFs matched
by name would not give the same values.
"""

from __future__ import annotations

from collections.abc import Sequence


class MatchedModes:
    """Pairs of (first-set mode, second-set mode) with their MAC.

    `first` and `second` name shape sets in the project. The first set
    is the basis of the comparison, and the frequency error Δf% is
    measured from its frequencies.

    Committing the squares picked on a cross-MAC creates one of these,
    so the pairing is saved in the project instead of only in the
    window: a report refers to it by name, and reopening the project
    gives the same pairs. The pairs are kept sorted by the first set's
    mode, then by the second set's.

    Attributes:
        first: The name of the set whose modes are the rows, and whose
            frequencies Δf% is measured from.
        second: The name of the set whose modes are the columns.
        pairs: `[row mode, column mode]` for each committed pair,
            zero-based.
        macs: The MAC of each pair as computed when it was committed,
            possibly after projecting onto the basis DOFs, which is why
            it is stored and not recomputed.
        first_geometry: The geometry `first` was linked to when the
            pairs were committed, since a projection only has meaning
            for that geometry.
        second_geometry: The same for `second`.
    """

    def __init__(self, first: str, second: str,
                 pairs: Sequence[Sequence[int]] | None = None,
                 macs: Sequence[float] | None = None,
                 first_geometry: str | None = None,
                 second_geometry: str | None = None) -> None:
        self.first: str = str(first)
        self.second: str = str(second)
        self.pairs: list[list[int]] = [[int(row), int(column)]
                      for row, column in (pairs or [])]
        self.macs: list[float] = [float(mac) for mac in (macs or [])]
        # the geometries each set was linked to when the matches were
        # committed — what an overlaid animation of the pair deflects
        self.first_geometry: str | None = str(first_geometry) if first_geometry \
            else None
        self.second_geometry: str | None = str(second_geometry) if second_geometry \
            else None
        self._sort()

    def _sort(self) -> None:
        """Matches in the first set's own order, which is by frequency.

        A matched table is read down the page against the mode list
        beside it, and pairs land in it in the order squares were
        clicked on the MAC — which is the order somebody's eye moved,
        not an order. Sorting by the first set's mode index is sorting
        by its frequency, because a `ShapeSet` is always in frequency
        order; doing it that way needs no lookup of a set that may not
        be in the project any more.

        The second index breaks ties, so one mode matched to two of the
        other set reads low to high rather than by which was clicked
        first.
        """
        order = sorted(range(len(self.pairs)),
                       key=lambda i: (self.pairs[i][0], self.pairs[i][1]))
        self.pairs = [self.pairs[i] for i in order]
        self.macs = [self.macs[i] for i in order]

    @property
    def num_matches(self) -> int:
        return len(self.pairs)

    def add(self, pairs: Sequence[Sequence[int]],
            macs: Sequence[float]) -> None:
        """Add pairs with their MACs; a pair already present gets the new MAC."""
        for pair, mac in zip(pairs, macs):
            pair = [int(pair[0]), int(pair[1])]
            if pair in self.pairs:
                self.macs[self.pairs.index(pair)] = float(mac)
            else:
                self.pairs.append(pair)
                self.macs.append(float(mac))
        self._sort()

    def delete_matches(self, rows: Sequence[int]) -> None:
        """Remove the matches at these row indices; raises ValueError for a row out of range."""
        bad = [row for row in rows
               if not 0 <= int(row) < len(self.pairs)]
        if bad:
            raise ValueError(f'no match at row {bad[0]}')
        keep = [i for i in range(len(self.pairs))
                if i not in {int(row) for row in rows}]
        self.pairs = [self.pairs[i] for i in keep]
        self.macs = [self.macs[i] for i in keep]

    def rename_source(self, old: str, new: str) -> None:
        """Update the references after a shape set or geometry is renamed from `old` to `new`."""
        if self.first == old:
            self.first = new
        if self.second == old:
            self.second = new
        if self.first_geometry == old:
            self.first_geometry = new
        if self.second_geometry == old:
            self.second_geometry = new
