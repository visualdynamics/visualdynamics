# Remaining tasks

Work that is known, wanted, and not done — kept here rather than in
someone's head. A task earns a line when it is specific enough to
start; vaguer wishes belong in `PLAN.md` where they can be argued
about.

This list is honest about what the software does not yet do. Anything
here that contradicts `PRINCIPLES.md` says so, and says why it is a
gap rather than a decision.

## Writers for formats that can only be read

Principle 5 says what can be read can be written, and two formats do
not yet hold up their end. Both are gaps rather than decisions —
unlike Rattlesnake `.nc4`, which is read-only permanently and for a
reason.

- **Femap Neutral (`.neu`) writer.** The reader takes geometry and
  modes; `io/femap.py` says outright that "writing waits until
  someone needs it". Sending a correlated test set back to Femap is
  the case that would want it.
- **Nastran punch (`.pch`) writer.** The reader takes eigenvectors as
  a `ShapeSet`. Writing one would let test modes go back to a solver
  deck.

Neither is hard; both are unstarted because nothing has needed them
yet. When one is written, the exception comes out of principle 5.

## The project file: `.escdf` in the next alpha

Decided 2026-09-13, releasing with `.vdyn` explicitly alpha-only and a
dated end (PLAN.md, "The project file is provisional"). The Engineering
Sciences Common Data Format is public since 2026-09-30 (BSD-3-Clause;
HDF5, a standard structure for the data types this project holds).
**Import and export come first** (PLAN.md "ESCDF: import and export
first", with the decisions), and whether `.escdf` replaces `.vdyn` is
decided after they are solid.

1. **Done 2026-09-30**: read with and write from the specification,
   on h5py (`io/escdf.py`, `io/escdf_objects.py`); the reference
   package a test-only dependency, the agreement oracle both ways.
2. **Done**: the extras — what the standard has no field for — live in
   the `attachments` every group carries, each object whole in its own
   layout; a report is attached rendered as well.
3. Then, if decided: the next alpha saves projects as `.escdf` and still opens `.vdyn`.
   The rename is broad and mechanical: `io/native.py`, the file
   dialogs, the app's document type in the macOS and Windows packaging,
   the website's format table and its test, the guide, the disclaimer
   and this note, and the rulebook's "three names" line (two names and
   a shared extension). Every `.escdf` file from any tool then opens
   Visual Dynamics, and every project it saves opens in every other
   tool that speaks the format.
4. The `.vdyn` reader is removed at the first release that is not an
   alpha. Until then the migration is: open the file in that alpha,
   save it again. The four notes that promise this (the disclaimer,
   the downloads page, the README, the guide) come out at the same
   time, with `tests/test_disclaimer.py` and `tests/test_website.py`
   pinning them until they do.

## The website

- **`web/launch/` is what visualdynamics.org serves** (since the
  first release, 2026-09-14; `web/public/` was the holding page): a
  downloads page with one tile per build (macOS Apple silicon, macOS
  Intel, Windows, Linux), the example projects, links to the source,
  the documentation and the principles. The release workflow's
  `site` job deploys it on every published release.
- **The download tiles point at the latest release.** They fall back
  to the releases page and upgrade themselves to the real assets by
  asking the GitHub API. Previewed locally they hand out the builds
  `packaging/stage_downloads.py` staged (gitignored; run `clear`
  before any wrangler deploy of `web/launch`, which uploads the
  directory as it stands).

## The documentation toolchain

Decided and done 2026-09-04: ProperDocs and MaterialX, the maintained
continuations of MkDocs 1.x and Material (PLAN.md, "The generator is
ProperDocs"). The trigger to look again, rather than a date: a build
fails on a Python that is needed, or Zensical's mkdocstrings bridge
gains cross-references — and Sphinx deserves a fresh look at that
point, since the NumPy docstrings are its native convention.

## The suite's memory

Found and fixed 2026-09-13, and written down so the shape of it is
not learned twice. An xdist worker of this suite reached 2.7 GB and
four of them were what killed both CI jobs on a 16 GB runner. The
cause was not the tests' data but the test fixture: `deleteLater`
only posts a deferred delete, and `processEvents` never delivers
one, so every `MainWindow` a test made outlived its test — 15 MB
empty, far more with a project loaded — and a lambda connected to the
application's style hints held the Python side of each for good.
`tests/conftest.py::destroy_window` sends the delete by hand, the
hints get a bound method, and `tests/test_window_lifetime.py` pins
both: eight windows made and destroyed leave nothing. After it, four
workers peak at 7.4 GB together (measured in an 8 GB container) and a
full run takes 12 minutes on four cores.

What is left is peaks, not leaks, and each is a real computation.
The largest was the wavelet scalogram of a long record, which
transformed every scale at once; it transforms in bands under
`core.wavelet.BAND_BYTES` now and draws from a peak-held reading
(2026-09-15), so the view tests' peak is a few hundred megabytes and
a five-minute record at 16 kHz is 6.5 GB end to end, most of it the
import. One local-only fixture CI never sees remains.

## Graphics without a GPU

The third release-workflow run on real Windows (2026-09-01) found
what a machine with no OpenGL driver gets: VTK asks for a pixel
format, is refused, looks for `osmesa.dll`, and the application dies
with 0xC0000005 before a window. A GitHub runner is that machine, so
the smoke test puts Mesa's llvmpipe beside the installed exe for
itself. A user over Remote Desktop or in a bare virtual machine is
that machine too, and gets the crash rather than a sentence. Two
honest fixes, neither done: detect the refusal before the first
render and say what is missing, or ship Mesa as a fallback the
launcher reaches for only when the driver refuses. The macOS and
Linux builds are unaffected (Metal-backed GL and the AppImage's own
Mesa respectively).

## Release

**Done 2026-09-14: Visual Dynamics 0.1.0a1 is public.** The order
below is kept as the record of what a release takes, for the next
one. **The day-of order** (written 2026-09-01, when the shared repository
was brought current and the ownership question closed). Nothing here
is public until step 6, steps 1 to 5 are all reversible, and step 7
happens on its own once 6 is done.

0. Level the environments: `tools/level_venv.sh` upgrades both venvs
   to the latest of everything, as CI and PyPI installs get, and runs
   the gate. Fix what it finds before anything else (2026-10-09: a41
   took three syncs to find what this finds first, and the macOS pair
   is built from these venvs).
1. Pick the version and set `__version__` in `src/visualdynamics/__init__.py`;
   `pyproject.toml` reads it from there.
2. Rebuild all four packages from that version
   (`packaging/refresh_builds.sh`) and open each one once. The two
   macOS images come out signed and notarized when the Developer ID
   certificate and the `vd-notary` profile are in the keychain
   (packaging/README.md, "Signing") — do those two one-time steps
   before this one, and check the build printed "notarized and
   stapled" twice. Unattended, the build reads the app-specific
   password from the login-keychain item `vd-notary-password`
   (packaging/README.md; the command that makes it ends in `-w`),
   so nobody needs to be at the Mac.
3. Swap the grant (the copyleft license with a CLA — decided
   2026-09-03, PLAN.md "Open source, revisited"; **done — merged
   into `main` 2026-09-13**; *superseded 2026-09-28: the tree is MIT
   and the CLA is retired, PLAN.md "MIT, and the CLA retired"*): `LICENSE` becomes
   the license text (`packaging/licenses/` holds a copy) with
   the "or any later version" statement at its head, `license` in
   `pyproject.toml` becomes the matching SPDX expression (it reads
   `"LicenseRef-Proprietary"` until then), the `Private :: Do Not
   Upload` classifier comes out of `pyproject.toml` (PyPI refuses an
   upload that carries it, and the `pypi` job uploads the real
   package), the README's license badge
   follows, and the CLA is published beside `CONTRIBUTING.md`, which
   already names it. Commit.
4. Sync the shared repository from this tree — `tools/sync_public.sh
   "<title>"`, which opens a pull request that merges itself once CI
   is green; the public `main` takes no direct push, the owner
   included (2026-09-14) — and review. (The first release also
   started the public history over from an orphan commit, so that it
   begins with the release and holds nothing older; that was once,
   and is not a step now.) Commits there carry **the GitHub noreply
   address**; the account blocks pushes that carry the private one.
5. Tag `v<version>` on the shared repository — an *annotated* tag,
   `git tag -a v<version> -m "Visual Dynamics <version>"` in
   `~/visualdynamics-shared` at `origin/main`, then push the tag: a
   bare `git tag` there stops at "no tag message?" and creates
   nothing (2026-09-16). The release workflow
   builds Linux and Windows there and opens a **draft** release with
   them and a `SHA256SUMS` attached; then `packaging/attach_macos.sh
   v<version>` uploads the two macOS images built here and adds their
   lines to that file, `packaging/release_updates.sh v<version>` adds
   the Sparkle updates (the two update zips, their deltas and the two
   appcasts; since 2026-09-28), and read the Windows smoke-test
   screenshots before going further.
6. Publish the draft release. This is the step that cannot be walked
   back. (The first release also made the repository public and put
   branch protection on `main` — a pull request with its checks
   green, administrators enforced, linear history — which is what
   `tools/sync_public.sh` leans on. Both are done and stay done. The
   `cla` check was one of those checks until the CLA was retired on
   2026-09-28.)
7. **The site deploys itself when the release is published** (Brandon,
   2026-09-01, "option 1"): the release workflow's `site` job builds
   the documentation into `web/launch/documentation/`, writes
   `latest.json` from the release, and deploys `web/launch` to
   Cloudflare Pages — the launch pages replace the holding page, the
   docs answer at `visualdynamics.org/documentation/`, and *File →
   Check for Updates* starts finding the release. The two repository
   secrets it needs, `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`,
   exist on the shared repository (verified 2026-09-11; web/README.md
   says how they were made). Watch the job, then open the site. The
   manual fallback is the three lines in web/README.md, after
   `packaging/stage_downloads.py clear`.
   **PyPI goes up in the same moment**: the `pypi` job builds the
   distribution from the public tree and publishes it by trusted
   publishing (packaging/README.md, "PyPI" — the pending publisher
   and the `pypi` environment have to exist, and the name is held by
   the placeholder from the `reserve` run).
8. Afterwards: Windows signing. SignPath Foundation was applied to
   on 2026-09-28, the day the tree went MIT with no commercial terms
   (their terms, read that day: an OSI license, no commercial
   dual-licensing, no proprietary components, already released, the
   download page naming the Foundation as signer, a code signing
   policy page, MFA for everyone on the project), and **declined on
   2026-09-29** — not on the terms but on public-trust signals (stars,
   forks, contributors, outside references, sustained activity) the
   project does not yet show at two weeks public; they invite a
   reapplication once it does. **Decided (Brandon, 2026-09-29):
   unsigned for now, reapply later**; the downloads page and
   `code-signing.html` say so and point at each release's
   `SHA256SUMS`. When the project has the visibility they named,
   apply again at signpath.org/apply and, on acceptance, wire their
   GitHub Action into the release workflow's Windows job. Azure
   Artifact Signing (about $10 a month, packaging/README.md) remains
   the paid alternative if waiting stops being acceptable. (The Apple
   side is done: Brandon joined the Developer Program on 2026-09-02.)

- **A verifying updater, after signing.** *File → Check for Updates…*
  exists (2026-09-01) and stops at telling: it reads
  `visualdynamics.org/latest.json` and opens the download page. The
  half that downloads and replaces the running build waits on a
  signing identity to verify against — Sparkle on macOS, WinSparkle
  on Windows, an AppImage swap on Linux — because an updater that
  runs what it fetched unverified is remote code execution with a
  friendly name (`update.py` says so).
