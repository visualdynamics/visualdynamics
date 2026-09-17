# Acknowledgments

**Thanks, not authorship.** Nothing on this page confers a copyright
interest, an authorship claim, or any right in Visual Dynamics. It is
a record of help gratefully received — advice, review, test data,
measurements, bug reports, arguments that changed a decision.

Where the rights actually live is written down elsewhere, and the
records are kept apart on purpose:

| record | what it means | where |
| --- | --- | --- |
| this page | thanks, freely given, at the author's discretion | here |
| the terms of the work itself | GPL-3.0-or-later | [`LICENSE`](LICENSE) |
| what a contribution is given under | the contributor license agreement | [`CLA.md`](CLA.md) |
| third-party licenses | attribution the dependencies *require* | [`NOTICE.md`](NOTICE.md) |

Being thanked here is not a claim on anything — authorship, ownership or otherwise. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for how code is actually taken.

---

## Author

Visual Dynamics is written by Brandon Zwink, who holds the copyright
in it — see [`LICENSE`](LICENSE).


## With thanks to

<!-- Add entries here. A name and one line of what it was for; the
     specific thing is worth more than the adjective. Keep it in
     alphabetical order by surname so nobody has to read a ranking
     into the order.

     Examples of the shape, to delete:

     - **Name** — for the ADF corpus that the reader was tested
       against, and for finding the case where it read the wrong
       units.
     - **Name** — for the review that turned the averaging panel from
       five numbers into five numbers and their consequences.
     - **Name** — for asking why the SRS was drawn on a linear axis.
-->

*Nobody yet — this page was written before there was anyone on it, so
that the policy would be settled before the first case rather than
decided about a particular person.*


## Organizations

<!-- The same, for groups. Naming a group rather than its individual
     members is often the better choice: some organizations have
     policies about staff being named in outside products, and a
     collective acknowledgment carries the thanks without asking
     anyone to clear it.

     Worth a deliberate decision rather than a default, and one to
     make before the first name goes on rather than after. -->

*Nothing yet.*


## Standing on

Not people, but worth naming: Visual Dynamics is written against a
body of work it does not include and could not replace. The
dependencies it actually ships with — and what their licenses ask in
return — are in [`NOTICE.md`](NOTICE.md); this is the shorter list of
what shaped it.

- **Bendat & Piersol**, *Random Data* — the spectral conventions here
  follow it, and where this codebase had to choose between two common
  readings of a term it went with theirs.
- The **UFF dataset 58** convention, for a fifty-year-old format that
  still says what a channel is and what it measures.
- **sdynpy**, **Rattlesnake** and **forcefinder**, for showing what
  these tools should do. Nothing is taken from them: all three are
  GPL-3.0, and taking their code would have settled this project's
  licensing by accident — so their methods were read and then written
  from scratch here. `tests/test_license_boundary.py`
  enforces that on every run, and `AGENTS.md` rule 1 says why.
