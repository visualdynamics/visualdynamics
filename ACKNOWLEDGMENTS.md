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
| the terms of the work itself, and of every contribution to it | MIT | [`LICENSE`](LICENSE) |
| third-party licenses | attribution the dependencies *require* | [`NOTICE.md`](NOTICE.md) |

Being thanked here is not a claim on anything — authorship, ownership or otherwise. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for how code is actually taken.

---

## Author

Visual Dynamics is written by Brandon Zwink, who holds the copyright
in it apart from contributed code. A contributor keeps the copyright in
their own work and offers it under the same MIT License (see
[`LICENSE`](LICENSE) and [`CONTRIBUTING.md`](CONTRIBUTING.md)); who
wrote what is in the history.


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

- **Dr. Chris Beale**, for the inspiration behind the report plots.
- **Kevin Cross**, for the sine tracking and thirteen of the readings a
  test is judged by, from the SRS points rule to the dwell, which he
  contributed.
- **Matt McDowell**, for the inspiration behind the project tree.
- **Dr. Angela Montoya**, for suggesting the wavelet computation.
- **Dr. Dan Rohe**, for showing what open-source tools for structural
  dynamics can be.
- **Dr. Ryan Schultz**, whose implementations of bandwidth averaging,
  in his own tools and in sdynpy, inspired the bandwidth averaging
  here.


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
- **[sdynpy](https://github.com/sandialabs/sdynpy)**,
  **[Rattlesnake](https://github.com/sandialabs/rattlesnake-vibration-controller)**
  and **forcefinder**, for showing what these tools should do. Nothing is taken from them: all three are
  GPL-3.0, and their code could not have been taken into an
  MIT project — so their methods were read and then written from
  scratch here. `tests/test_license_boundary.py`
  enforces that on every run, and `AGENTS.md` rule 1 says why.
