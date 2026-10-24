"""How many lines of a comparison fall outside its bands by chance.

An averaged autospectrum is an estimate. Every line of it scatters
about the true level, and the estimate over the truth is a chi-square
variable over its statistical degrees of freedom (DOF), the same law
the confidence band on an averaged PSD is drawn from
(`core.averaging.chi_square_factors`). So a response whose true level
sits exactly on its specification still puts some lines outside a band
about it, and how many depends on nothing but the band's width in
decibels and the DOF. A line reads more than `b` dB high when the
chi-square variable exceeds

    dof · 10^(b/10)

and more than `b` dB low when it falls under dof · 10^(-b/10). At 120
DOF about 0.85 % of lines fall outside plus or minus 1.5 dB, four in
five of them low, and about one in a million outside plus or minus
3 dB. On a 400-line comparison that is three or four lines outside the
narrower band on any one average, in different places on the next.

A comparison counts the lines outside each pair of limits
(`compliance.compare`). `compare_chance` sets beside each count the
number chance alone would put there at the comparison's DOF, with the
true level exactly on the target: the sum over the judged lines of
each line's chance of reading past each edge (`chance_outside`). A
count near its expectation is the scatter of the estimate. A count
well above it, from a group of adjacent lines on one side of the
target, is a real excursion.

**Combining lines.** Some test documents allow adjacent control lines
to be combined for judging the tolerance: their powers added into one
wider band, judged against the limits' power over the same band. The
sum of N independent lines' estimates is worth about N times a line's
DOF, so the chance scatter shrinks, while a real excursion spread
across neighboring lines survives the combination. `compare_chance`
with `combine` N judges the comparison in groups of N adjacent lines.
A group's DOF is Satterthwaite's for a sum of independent chi-square
estimates weighted by the power each line was asked for,

    (sum of a_i)² / (sum of a_i² / dof_i)

which is exactly N times a line's on a flat stretch and a little less
where the target slopes across the group, since the louder lines carry
more of the sum. That holds for independent lines, which a windowed
average's are not.

**Neighboring lines are not independent.** A window spreads each line's
power into its neighbors, so the estimates at adjacent lines vary
together, and a sum of them steadies less than independent lines would.
The PSD carries how much (`Psd.line_correlation`, worked out from its
window, hop and frame count by `core.averaging.line_correlation`): 0
for a rectangle without overlap, 0.44 at one line apart for a Hann
window. With it the variance of a sum of lines weighted a_i is

    sum over i, j of a_i a_j r(|i - j|) / n

and its degrees of freedom are (sum of a_i)² n over that sum, which is
Satterthwaite's above when r is 0 past the line itself. Eight lines of
a Hann window at half overlap are worth 4.3 times a line's, not eight;
white noise averaged that way measures the same. The same holds for a
cell of the specification coarser than the lines, which is a sum of
the lines under it. A PSD that does not carry its correlation (read
from a file without it, or built by hand) has its lines taken as
independent, and the result says so.

References
----------
That an averaged autospectrum is the true one times a chi-square
variable over its degrees of freedom is the standard result for the
averaged periodogram, and this is where it is set out. The degrees of
freedom of a weighted sum of such estimates (Satterthwaite's
approximation) and the correlation a window leaves between adjacent
lines are worked out above, from the variance of the sum and from the
window itself.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The sampling distribution
   of an autospectral density estimate and its random error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import DataArray, Specification

#: the pairs of limits a comparison may carry, in the order they are
#: read out
PAIRS = ('warning', 'abort')

#: how far apart two cells may be and still count as adjacent, as a
#: fraction of the frequency: the edges of neighboring bins are worked
#: out separately and may differ in the last digit
_TOUCHING = 1e-9


class ChanceShare(NamedTuple):
    """The share of lines that read outside a band by chance alone:
    `below` its lower edge and `above` its upper one, as fractions
    between 0 and 1."""

    below: Any
    above: Any

    @property
    def total(self) -> Any:
        """The share outside either edge."""
        return self.below + self.above


def chance_outside(dof: ArrayLike, lower_db: ArrayLike | None = None,
                   upper_db: ArrayLike | None = None) -> ChanceShare:
    """The share of an averaged autospectrum's lines that read outside a
    band about the true level by chance alone.

    A line averaged to `dof` statistical degrees of freedom reads the
    true level times a chi-square variable over `dof`. It reads below
    `lower_db` (a negative number of dB from the truth) when that
    variable falls under dof · 10^(lower_db/10), and above `upper_db`
    when it exceeds dof · 10^(upper_db/10). Either edge may be None
    (or infinite) for a band with no edge on that side. The arguments
    broadcast against each other, so a band whose edges vary along the
    frequency axis gives a share per line.

    It is the same law as `core.averaging.chi_square_factors`, read the
    other way: that gives where the truth lies about an estimate at a
    confidence level, this the chance of an estimate reading a given
    distance from the truth. At the band `chi_square_factors` gives for
    a level, turned about, each side holds half of one minus the level.

    Examples: at two degrees of freedom (one frame, an exponential
    variable) a line reads more than 3 dB low with probability
    1 - exp(-10^-0.3), 0.394, and more than 3 dB high with probability
    exp(-10^0.3), 0.136. At 120 the shares outside plus or minus 1.5 dB
    are 0.65 % below and 0.20 % above.

    Parameters
    ----------
    dof : float or array_like
        The statistical degrees of freedom, more than zero; need not be
        whole.
    lower_db, upper_db : float or array_like, optional
        The band's edges in dB from the true level. A lower edge at or
        above zero, or an upper edge at or below it, is a band the
        truth itself is outside, and gives a share of a half or more.

    Returns
    -------
    ChanceShare
        `below` and `above`, floats for scalar arguments and arrays
        otherwise; `total` is their sum.
    """
    # the regularized incomplete gamma functions are the chi-square's
    # distribution exactly, at any count: P(chi2_n < x) is
    # gammainc(n/2, x/2), and gammaincc its complement, computed
    # directly so a share of one in a million keeps its digits.
    # `chi_square_factors` holds the inverse (gammaincinv), the quantile
    # at a level; the share past a given edge is the forward function,
    # which inverting the band numerically would only approximate
    from scipy.special import gammainc, gammaincc

    dof = np.asarray(dof, dtype=float)
    if np.any(~(dof > 0.0)):
        raise ValueError('degrees of freedom are more than zero')
    scalar = dof.ndim == 0

    def edge(db: ArrayLike | None, sign: float) -> np.ndarray:
        if db is None:
            return np.full(dof.shape, sign * np.inf)
        return np.asarray(db, dtype=float)

    lower, upper = edge(lower_db, -1.0), edge(upper_db, 1.0)
    scalar = scalar and lower.ndim == 0 and upper.ndim == 0
    half = dof / 2.0
    with np.errstate(over='ignore'):
        below = np.where(np.isneginf(lower), 0.0,
                         gammainc(half, half * 10.0 ** (lower / 10.0)))
        above = np.where(np.isposinf(upper), 0.0,
                         gammaincc(half, half * 10.0 ** (upper / 10.0)))
    if scalar:
        return ChanceShare(float(below), float(above))
    return ChanceShare(np.asarray(below), np.asarray(above))


def _lines_in(cell: dict[str, Any], left: np.ndarray,
              right: np.ndarray) -> int:
    """How many of the measurement's lines a cell holds: those whose bins
    reach into it, found as `compliance.judge` finds them. A bin that
    ends where the cell begins shares an edge and no width."""
    first = int(np.searchsorted(right, cell['low'] * (1.0 + 1e-12),
                                side='right'))
    last = int(np.searchsorted(left, cell['high'] * (1.0 - 1e-12),
                               side='left'))
    return max(last - first, 1)


def _groups(cells: list[dict[str, Any]], counted: np.ndarray,
            combine: int) -> list[list[int]]:
    """The judged cells in groups of `combine` adjacent ones. A cell
    that is not judged, or a gap between two that are (a stretch the
    specification leaves unwritten), ends a group early, since lines
    either side of a gap are not neighbors; so does the top of the
    band, where the last group may hold fewer."""
    out: list[list[int]] = []
    for k in np.flatnonzero(counted):
        k = int(k)
        group = out[-1] if out else None
        joins = (group is not None and len(group) < combine
                 and group[-1] == k - 1
                 and cells[k]['low'] <= cells[k - 1]['high'] * (1.0 + _TOUCHING))
        if joins:
            group.append(k)
        else:
            out.append([k])
    return out


def _sum_dof(weights: np.ndarray, dof: float,
             correlation: np.ndarray) -> float:
    """The degrees of freedom of a sum of adjacent lines, each worth
    `dof` and weighted by `weights` (the power asked of each), whose
    estimates vary together by `correlation` at 0, 1, 2, ... lines
    apart: (sum of the weights)² `dof` over the sum over every pair of
    lines of their weights times their correlation."""
    spread = float(np.sum(weights ** 2))
    for apart in range(1, min(len(correlation), len(weights))):
        spread += 2.0 * float(correlation[apart]) * float(
            np.sum(weights[:-apart] * weights[apart:]))
    return float(np.sum(weights)) ** 2 * dof / spread


def _summed(values: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Each group's sum; NaN where any of its cells is not finite."""
    return np.array([float(np.sum(values[g])) for g in groups], dtype=float)


def compare_chance(specification: Specification, measured: DataArray,
                   spec_record: int = 0, measured_record: int = 0, *,
                   dof: float | None = None, combine: int = 1,
                   scale_db: float | None = None) -> dict[str, Any]:
    """The lines of one comparison outside each pair of limits, beside how
    many chance alone would put there at the comparison's degrees of
    freedom.

    The comparison is `compliance.compare`'s, cell for cell: the same
    cells, the same scale, the same areas. For each pair of limits the
    specification carries, the lines outside its lower and its upper
    edge are counted, and the number expected outside each by chance is
    the sum over the judged lines of `chance_outside` at that line's
    edges, in dB from the target over the same cell, with the true level
    taken to sit exactly on the target. With `combine` 1 the counts are
    `compare`'s own.

    With `combine` N the judged lines are taken N adjacent at a time,
    their powers added and judged against the limits' power over the
    same stretch, each group at its own degrees of freedom: those of a
    sum of correlated lines (Satterthwaite's, with the PSD's
    `line_correlation`), N times a line's only for independent lines on
    a flat stretch. A group does not
    reach across a gap the specification leaves unwritten, and the last
    one in a stretch may hold fewer lines. Some test documents allow this
    for judging a tolerance; it changes how the test is judged, not the
    test.

    Parameters
    ----------
    specification, measured : Specification, Psd
        The pair: an autospectrum on narrowband lines and the
        specification it was controlled to.
    spec_record, measured_record : int, default 0
        Which record of each.
    dof : float, optional
        The statistical degrees of freedom of one line of the
        measurement; read from the PSD (`Psd.statistical_dof`) when
        omitted.
    combine : int, default 1
        How many adjacent lines are judged together.
    scale_db : float, optional
        The comparison's scale in dB, added to the measurement; resolved
        through `compliance.comparison_scale_db` when omitted.

    Returns
    -------
    dict
        'lines' (how many lines, or groups, were judged), 'band' (low and
        high, Hz), 'scale_db', 'dof' (one line's), 'combine', 'judged_dof'
        (the median group's DOF: one line's times `combine` for
        independent lines on a flat stretch, fewer for windowed ones).
        For each pair the specification carries, as
        '{pair}_lines', '{pair}_below' and '{pair}_above': how many were
        outside the pair, under its lower edge and over its upper one; and
        '{pair}_expected', '{pair}_expected_below' and
        '{pair}_expected_above', the numbers chance alone puts there. And
        'groups', a dict of arrays, one value per line or group judged:
        'low' and 'high' (its reach, Hz), 'lines' (the measurement's
        lines in it), 'dof', 'miss_db' (the measurement over the target,
        in dB) and, per pair, '{pair}_lower_db' and '{pair}_upper_db'
        (its edges in dB from the target, NaN where it has none),
        '{pair}_below' and '{pair}_above' (outside, bool) and
        '{pair}_chance' (the chance of reading outside either edge).

        A pair that cannot be compared returns {'lines': 0, 'refused':
        reason}, as `compare` does, and so does a measurement on
        proportional bands, whose bands each hold a different number of
        lines, a cross term (whose error is not chi-square, as
        `Psd.confidence_band` refuses it), or one whose DOF is neither
        given nor carried. One with nothing to compare returns
        {'lines': 0}. Compared with neither side's units declared, or
        summing lines of a PSD that does not carry its
        `line_correlation`, the result carries 'note'.

    Raises
    ------
    ValueError
        When `combine` is not a whole number of at least one, or `dof`
        is not more than zero.
    """
    from .compliance import (
        beyond,
        comparable,
        comparison_scale_db,
        judge,
        units_check,
    )

    if isinstance(combine, bool) or int(combine) != combine or combine < 1:
        raise ValueError('lines are combined a whole number at a time, '
                         'one or more')
    combine = int(combine)
    if dof is not None and not float(dof) > 0.0:
        raise ValueError('degrees of freedom are more than zero')
    refused = comparable(specification, measured)
    if refused is not None:
        return {'lines': 0, 'refused': refused}
    refused, note = units_check(specification.ordinate_dim[spec_record],
                                measured.ordinate_dim[measured_record],
                                wanted_name='the specification')
    if refused is not None:
        return {'lines': 0, 'refused': refused}
    if getattr(measured, 'bandwidth', None) is not None:
        return {'lines': 0, 'refused': (
            'the chance of a line reading outside is worked out on '
            'narrowband lines; each proportional band holds a different '
            'number of them, so compare the narrowband spectrum')}
    # the chi-square law is an autospectrum's, as the confidence band's
    if not measured.is_autospectrum(measured_record):
        return {'lines': 0, 'refused': (
            'the chance of a line reading outside is the chi-square law of '
            'an autospectrum; this record is a cross term '
            f'({measured.record_label(measured_record)})')}
    if dof is None:
        counts = getattr(measured, 'statistical_dof', None)
        if counts is None:
            return {'lines': 0, 'refused': (
                'the PSD does not say how many degrees of freedom its '
                'average has; give them')}
        dof = float(np.asarray(counts, dtype=float).reshape(-1)[
            measured_record if np.size(counts) > 1 else 0])
        if not np.isfinite(dof) or dof <= 0.0:
            return {'lines': 0, 'refused': (
                'the PSD carries no usable count of degrees of freedom; '
                'give them')}
    dof = float(dof)
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured)

    level = judge(specification, measured, spec_record, measured_record,
                  None, True, scale_db)
    cells = level['cells']
    asked, held = level['asked'], level['held']
    counted = np.isfinite(asked) & np.isfinite(held) & level['judged']
    if not counted.any():
        return {'lines': 0}
    left, right = measured.bin_bounds()
    held_in = np.array([_lines_in(cell, left, right) for cell in cells])
    groups = _groups(cells, counted, combine)
    correlation = getattr(measured, 'line_correlation', None)
    independent = correlation is None
    correlation = (np.ones(1) if independent
                   else np.asarray(correlation, dtype=float).reshape(-1))

    target = _summed(asked, groups)
    got = _summed(held, groups)
    # Satterthwaite: the sum of estimates a_i X_i / n, each X_i a
    # chi-square on n, has the mean and variance of a chi-square on
    # (sum a_i)^2 n / sum(a_i a_j r_ij), weighted by the power asked of
    # each line because that is what each estimate averages to when
    # the truth sits on the target, and with r the correlation the
    # window leaves between lines. A cell holding several lines spreads
    # its power evenly over them; the lines of a group run on from one
    # cell to the next.
    group_dof = np.array([_sum_dof(np.concatenate(
        [np.full(held_in[k], asked[k] / held_in[k]) for k in g]),
        dof, correlation) for g in groups])
    with np.errstate(divide='ignore', invalid='ignore'):
        miss_db = 10.0 * np.log10(got / target)

    low = np.array([cells[g[0]]['low'] for g in groups])
    high = np.array([cells[g[-1]]['high'] for g in groups])
    out: dict[str, Any] = {
        'lines': len(groups),
        'band': (float(cells[groups[0][0]]['low']),
                 float(cells[groups[-1][-1]]['high'])),
        'scale_db': float(scale_db),
        'dof': dof,
        'combine': combine,
        'judged_dof': float(np.median(group_dof)),
    }
    table: dict[str, np.ndarray] = {
        'low': low, 'high': high,
        'lines': np.array([int(sum(_lines_in(cells[k], left, right)
                                   for k in g)) for g in groups]),
        'dof': group_dof, 'miss_db': miss_db,
    }
    for pair in PAIRS:
        edges = [f'{pair}_lower', f'{pair}_upper']
        if not any(specification.limits.get(name) is not None
                   for name in edges):
            continue
        sides = {}
        for name, over in zip(edges, (False, True)):
            if specification.limits.get(name) is None:
                sides[over] = (np.full(len(groups), np.nan),
                               np.zeros(len(groups), dtype=bool),
                               np.zeros(len(groups)))
                continue
            found = judge(specification, measured, spec_record,
                          measured_record, name, over, scale_db)
            limit = found['asked']
            usable = np.isfinite(limit) & (limit > 0.0)
            # a group is judged against an edge only where every one of
            # its cells has that edge: a limit missing from part of a
            # group says nothing about the whole of it
            whole = np.array([bool(usable[g].all()) for g in groups])
            bound = np.where(whole, _summed(np.where(usable, limit, 0.0),
                                            groups), np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                edge_db = np.where(whole, 10.0 * np.log10(bound / target),
                                   np.nan)
            # uncombined, the cells out are `judge`'s own; a group is
            # read by the same rule on its sums
            outside = (whole & found['out'][[g[0] for g in groups]]
                       if combine == 1 else whole & beyond(got, bound, over))
            share = np.zeros(len(groups))
            if whole.any():
                chance = chance_outside(
                    group_dof[whole],
                    upper_db=edge_db[whole] if over else None,
                    lower_db=None if over else edge_db[whole])
                share[whole] = chance.above if over else chance.below
            sides[over] = (edge_db, outside, share)
        lower_db, below, below_chance = sides[False]
        upper_db, above, above_chance = sides[True]
        out[f'{pair}_lines'] = int((below | above).sum())
        out[f'{pair}_below'] = int(below.sum())
        out[f'{pair}_above'] = int(above.sum())
        out[f'{pair}_expected'] = float(below_chance.sum()
                                        + above_chance.sum())
        out[f'{pair}_expected_below'] = float(below_chance.sum())
        out[f'{pair}_expected_above'] = float(above_chance.sum())
        table[f'{pair}_lower_db'] = lower_db
        table[f'{pair}_upper_db'] = upper_db
        table[f'{pair}_below'] = below
        table[f'{pair}_above'] = above
        table[f'{pair}_chance'] = below_chance + above_chance
    out['groups'] = table
    notes = [] if note is None else [note]
    if independent and (combine > 1 or bool(np.any(held_in[counted] > 1))):
        notes.append('the PSD does not say how its window ties neighboring '
                     'lines together, so the lines summed are taken as '
                     'independent, and read high if they were windowed')
    if notes:
        out['note'] = '; '.join(notes)
    return out


def describe_chance(result: dict[str, Any], pair: str) -> str:
    """One pair of a `compare_chance` result in words, as the plot's
    title and the reading's pane give it: '5 of 389 lines outside the
    warning band (5 below, 0 above), 3.3 expected by chance (2.5
    below, 0.8 above)'. Groups are named as groups when lines were
    combined. A pair the result does not carry is said to be absent."""
    if f'{pair}_lines' not in result:
        return f'no {pair} band'
    unit = 'line' if result.get('combine', 1) == 1 else 'group'
    judged = result['lines']
    count = result[f'{pair}_lines']
    return (f'{count} of {judged} {unit}{"s" * (judged != 1)} outside the '
            f'{pair} band ({result[f"{pair}_below"]} below, '
            f'{result[f"{pair}_above"]} above), '
            f'{expected_text(result[f"{pair}_expected"])} expected by chance '
            f'({expected_text(result[f"{pair}_expected_below"])} below, '
            f'{expected_text(result[f"{pair}_expected_above"])} above)')


def expected_text(value: float) -> str:
    """An expected count in words: one decimal from 0.1 up, and a share
    too small for that given to two significant figures ('0.00041'),
    or '0' when it is under one in a billion, so a band chance never
    reaches does not read as one it sometimes does."""
    if value >= 0.095:
        return f'{value:.1f}'
    if value < 1e-9:
        return '0'
    return f'{value:.2g}'
