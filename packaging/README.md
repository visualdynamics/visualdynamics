# Packaging — what a person who does not have Python downloads

One PyInstaller spec, four thin platform wrappers, and a workflow that
runs **only on a tag, a button press, or a published release** (the
last is what deploys the site and publishes to PyPI).

| platform | script | artefact | verified |
| --- | --- | --- | --- |
| macOS, Apple silicon | `build_macos.sh` | `.dmg` around a `.app`, signed and notarised | **yes** — 486 MB, launched from the mounted image |
| macOS, Intel | `build_macos_intel.sh` | the same, built under Rosetta | **yes** — 539 MB, notarised |
| Linux | `build_linux.sh` | `.AppImage` | **yes** — 476 MB, x86_64, launched under Xvfb |
| Windows | `build_windows.ps1` | Inno Setup `.exe` installer | **yes** — installed and launched on a `windows-latest` runner by the release workflow's smoke test (2026-09-01); `build_windows_wine.sh` is the local loop |

Two of the three are built and *launched*, not merely produced. Linux
was done in a `python:3.13-slim-bookworm` container on an x86_64 Debian
box, which is also how to repeat it:

    docker run --rm --cpus 3 --memory 8g -v "$PWD":/work -w /work \
      -e APPIMAGE_EXTRACT_AND_RUN=1 python:3.13-slim-bookworm bash -c '
        set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        grep -vE "^[[:space:]]*(#|$)" packaging/linux/apt-packages.txt |
          tr '\n' '\0' | xargs -0 apt-get satisfy -y -qq --no-install-recommends
        wget -q https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
        chmod +x appimagetool-x86_64.AppImage
        mv appimagetool-x86_64.AppImage /usr/local/bin/appimagetool
        pip install -q . pyinstaller pillow
        PYTHON=python packaging/build_linux.sh'

PyInstaller cannot cross-compile, but a Windows *environment* on this
Mac turned out to be enough: `build_windows_wine.sh` builds the x64
installer under Wine (see below), and the app launches there too, which
is the local debug loop for Windows-only failures. Real Windows remains
unexercised — expect the first tagged run to find something anyway; the
Linux script failed twice before it worked.

### Windows under Wine (2026-08-31)

`build_windows_wine.sh setup` once, then `build`; `run` launches what
was built. Windows x64 Python + PyInstaller + Inno Setup all run under
Wine (via Rosetta), so the installer in `dist/` is a real Windows
artefact made without a Windows machine, licence, or Actions minutes.
The traps, each found the hard way and remembered in the script:

- **Wine devel, not stable**: Qt 6.7+ links Windows' native ICU DLLs
  (`icuuc`/`icuin`), which Wine grew only after 11.0-stable.
- **`winetricks vcrun2022`**: a bare prefix has no `msvcp140.dll`.
- **Windows Python from the NuGet zip**: the python.org installer is a
  Burn bundle and silently does nothing under Wine.
- **Never build where Qt cannot import.** PyInstaller's Qt hooks learn
  the plugin list by importing Qt; in a broken prefix the build exits 0
  and bundles *no Qt plugins* — an installer that installs, launches,
  and dies with "no Qt platform plugin". The script hard-stops before
  building and re-checks `qwindows.dll` after.
- **Mesa software GL to run**: Wine's Mac OpenGL stops at 2.1 and VTK
  crashes native (a null jump during window creation). llvmpipe's
  `opengl32.dll` beside the exe fixes running under Wine; it is never
  put in the installer.
- **x64 only**: VTK publishes no `win_arm64` wheels, so a
  Windows-on-ARM package is not buildable from any host today.

What running under Wine can and cannot tell you: import errors,
missing-DLL chains, packaging omissions and startup crashes are all
real Windows findings; rendering fidelity, file dialogs and installer
UX need actual Windows eyes — the release workflow's smoke test
installs and launches the build on a real runner and screenshots the
window (proven 2026-09-01), and a human is still the judge of how it
looks.

### Keeping the builds fresh (2026-08-31)

`packaging/refresh_builds.sh` rebuilds all four distributables from
the working tree and stages each for the launch site's local preview
(`stage_downloads.py`, cumulative): macOS natively, Windows under
Wine, and Linux on the media server via `build_linux_remote.sh` —
this Mac has no container runtime, and that x86_64 Debian box with
Docker is exactly what the container recipe above wants. The point is
that http://127.0.0.1:8710/downloads.html always hands out installers
built from the code as it stands, so testing a packaged build is a
click rather than an errand.

### What the Linux build taught, so it is not learned twice

Both failures were missing packages, and neither was obvious:

- **`binutils`** — PyInstaller shells out to `objdump` on Linux. Without
  it the build stops before it starts.
- **`libkrb5-3`** — PyInstaller's Qt hook imports `QtNetwork` to ask
  whether OpenSSL is enabled, and that pulls `libgssapi_krb5.so.2`. The
  *build* fails, not the application, which makes it read like a
  packaging bug rather than a missing library.

Both now live in `packaging/linux/apt-packages.txt`, which the build
script and the release workflow both read. That file is the single
list; a hand-copied second one is what caused this.

Also: `appimagetool` is itself an AppImage and needs FUSE, which
containers and hosted runners have not got — `APPIMAGE_EXTRACT_AND_RUN=1`
is the documented way round, and both the script and the workflow set
it. The container recipe above installs it too; the build script only
checks that it is on PATH, and without it you get the application
directory and no AppImage.

### The bundle must not carry the C++ runtime (2026-08-27)

The first build to be run on a distribution *newer* than the one that
made it did not start. Built in the bookworm container above, the 0.0.1
AppImage opened a project on bookworm and died on trixie during startup,
before any window, with

    vtkXOpenGLRenderWindow: Could not find a decent config

which reads like a graphics-driver problem and is not one. VTK asks the
system's libGL for a config; libGL `dlopen`s the distribution's mesa
driver; that driver resolves `libstdc++` against whatever is already
loaded, which inside a frozen application is the copy PyInstaller
collected. Bookworm's offers `GLIBCXX_3.4.30`, trixie's mesa wants
`GLIBCXX_3.4.33`, the driver does not load, and no GL config exists.

`packaging/system_libraries.py` now drops `libstdc++.so.6` and
`libgcc_s.so.1` from the Linux bundle, so the host's are used;
`tests/test_system_libraries.py` holds the rule. What the day's
measurements support and what they do not:

- Deleting **only** those two files from the broken bundle made it start
  on trixie in four seconds. That is the whole fix.
- It is **not** the missing multisample fbconfigs under Xvfb — a
  pip-installed pyvista renders there with `multi_samples` at 8 and 0.
- It is **not** the bundled `libX11`/`libxcb`. Removing those was tried
  the same day: same error, plus a segmentation fault. They stay.

The general lesson is that building on the oldest distribution you
support is necessary but not sufficient: an artefact also has to be
*launched* somewhere newer, because this failure appears nowhere else.

## Why --onedir, always

PySide6 is LGPL-3.0. A closed-source application may link to it only if
the recipient can **replace the Qt libraries** with their own build
(NOTICE.md has the detail). A directory bundle keeps them as separate
dylibs/DLLs/.so files and satisfies that; `--onefile` does not. This is
the one packaging decision that cannot be revisited cheaply, so it is
made once, in the spec, with a comment.

## The minutes arithmetic

GitHub bills runner minutes by a multiplier, and macOS is the whole
story:

| runner | multiplier | build | **quota cost** |
| --- | --- | --- | --- |
| Linux | 1x | ~15 min | 15 |
| Windows | 2x | ~20 min | 40 |
| macOS | **10x** | ~20 min | **200** |

So the release workflow builds Linux and Windows there, and **leaves
macOS off by default** — `build_macos.sh` produces the same artefact on
the maintainer's own machine in about two minutes for nothing. A release
therefore costs ~55 minutes rather than ~255, out of 2 000 a month.
`workflow_dispatch` has a `macos` checkbox for a release where that
machine is not to hand.

Builds are attached to a **GitHub Release**, never left as Actions
artifacts: artifacts count against a 500 MB storage quota that three
~1 GB builds would exhaust immediately, while release assets are a
separate and effectively unlimited store.

## No nightly builds

A nightly build of a project with one author spends the year's minutes
on artefacts nobody downloads, and produces a version number that means
nothing. Tag when there is something worth handing over:

    git tag v0.1.0 && git push origin v0.1.0

The release is created as a **draft**, so it can be looked at before the
world sees it.

## Signing, and what it costs to skip

Signed and notarised on macOS; unsigned on Windows and Linux. What
that means for whoever you hand a build to:

- **macOS** — **signed and notarised from this desk** (Brandon joined
  the Apple Developer Program 2026-09-02, both one-time steps below
  were done the same evening, and `build_macos.sh` does the rest
  unasked, on both the arm64 and the Intel image). The two steps, for
  a new machine:
  1. **The certificate.** Xcode → Settings → Accounts → the Apple ID →
     *Manage Certificates…* → **+** → *Developer ID Application*. It
     lands in the login keychain; `security find-identity -v -p
     codesigning` then lists it, and the build picks it up. (The
     *Apple Development* certificate already there is for running on
     your own Macs and does not satisfy Gatekeeper.)
  2. **The notarisation credentials**, stored once in the keychain so
     no password ever sits in a file or a shell history:
     ```bash
     xcrun notarytool store-credentials vd-notary --apple-id bzwink@gmail.com --team-id 2542NQ9D95
     ```
     It prompts for an *app-specific password*, made at
     account.apple.com → Sign-In and Security → App-Specific
     Passwords. The team ID is the *paid* team's, from
     developer.apple.com → Membership details (2542NQ9D95) — the one
     in the older Apple Development certificate's name is the free
     personal team's, and notarytool refuses it with a 403 (found
     2026-09-02).

  With both in place a build signs every dylib and the bundle with the
  hardened runtime and `entitlements.plist`, notarises the app and
  staples its ticket into the bundle, then signs, notarises and
  staples the image — and `spctl` assesses both before the build calls
  itself done. Without the certificate the bundle is ad-hoc signed
  (required on Apple silicon, where an unsigned binary is refused
  outright) and Gatekeeper asks once; without the profile it is signed
  but not notarised, and the build warns. **The profile can be
  unreachable for a stretch** — "No Keychain password item found for
  profile: vd-notary", from the foreground too, with nothing changed
  here — and settled on 2026-09-15: `notarytool store-credentials`
  keeps the profile in the **data-protection keychain**, under the
  `com.apple.gke.notary` access group (Keychain Access shows it as
  "cannot be edited"), and macOS opens that only to an unlocked user
  session. An agent's shell is not one; Brandon touching the profile
  from his own Terminal (`xcrun notarytool history --keychain-profile
  vd-notary`) opens it for a while. **The way round it, so a build
  needs nobody**: keep the app-specific password as an ordinary
  login-keychain item, which `build_macos.sh` prefers when it exists
  (`NOTARY_ITEM`, default `vd-notary-password`; the Apple ID is the
  item's account, the team is `NOTARY_TEAM`). Made once, by Brandon,
  with the password typed at the prompt rather than on the command
  line:

  ```
  security add-generic-password -a bzwink@gmail.com -s vd-notary-password \
      -T /usr/bin/security -T "$(xcrun --find notarytool)" -U
  ```

  (the `-T`s let the build read it without a prompt; the login
  keychain has no timeout and does not lock on sleep, so it answers
  whenever Brandon is logged in). Without the item the script falls
  back to the profile, and when neither answers it says so and leaves
  the images signed: notarise those by hand rather than rebuilding —
  `xcrun notarytool submit … --wait` then `xcrun stapler staple`. Notarisation takes a few
  minutes per submission, twice per image.
- **Windows** — SmartScreen warns until the signature earns
  reputation. **The plan is SignPath Foundation** (Brandon,
  2026-08-28, reaffirmed 2026-09-03 under the copyleft-with-a-CLA
  plan: nothing is sold under other terms and may never be, so in
  practice this is a plain copyleft project and their no-dual-licensing condition
  is met for as long as that stays true — the day it does not, the
  paid fallback at the end takes over). Their conditions:
  - The certificate is the Foundation's, so the publisher line in
    SmartScreen and the installer reads **"SignPath Foundation"** —
    not the author's name. Accepted as the price of free; the macOS
    side still carries the author's own Apple identity.
  - They require an OSI licence **without commercial dual-licensing**
    — a plain copyleft licence qualifies; a copyleft-plus-commercial
    plan would not, which is why no commercial terms are offered.
  - They sign projects that are *already released*, so the sequence
    is: public grant → first Windows release unsigned → apply at
    signpath.org → wire their GitHub Action into the release
    workflow's Windows job. Signing is release two, not day one.
  - The download page must say what the software does; `web/launch/`
    already does.

  The paid fallbacks, if the Foundation declines or the posture
  changes: **Azure Artifact Signing** (was "Trusted Signing"),
  $9.99/month, open to individual US developers, no hardware, signs
  from CI — the closest thing Windows has to the Apple account. A
  traditional OV certificate ($100–400/yr) now ships on a USB token
  by CA/Browser Forum rule (since 2023), which cannot sign on a
  hosted runner; EV needs a registered entity and the same token.

  Whichever way: sign the installer *and* the application inside it —
  an unsigned installer around a signed app still trips SmartScreen —
  and always timestamp, so signatures outlive the certificate.

The macOS side is paid for and wired; the Windows side is SignPath
after the first release, with the paid signer as the fallback.

## PyPI

`pip install visualdynamics` is what the downloads page promises, and
the release workflow keeps it by **trusted publishing**: PyPI accepts
the workflow's own identity, so no token exists anywhere. One-time
setup, in the PyPI account (2026-09-03):

1. An account at pypi.org with two-factor authentication on (PyPI
   requires it).
2. Account settings → Publishing → *Add a new pending publisher*,
   with exactly these values:
   - PyPI project name: `visualdynamics`
   - Owner: `visualdynamics`
   - Repository name: `visualdynamics`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. On the shared repository, Settings → Environments → New
   environment named `pypi` (no secrets, no rules — its existence is
   what the publisher form names).

Then **reserve the name**: Actions → release → *Run workflow* with
"reserve" ticked. Done 2026-09-03: that published an empty
placeholder, version 0.0.0, holding the name — nothing of the software
left the repository, and the pending publisher became the project's
publisher at that first upload. The real distribution goes up by
itself when the first release is published, the same moment the site
deploys.

## Regenerating the icons

`icon.icns`, `icon.ico` and `icon-512.png` are generated from
`visualdynamics.gui.app_icon.draw_app_icon` — the same mark the
application wears — so they cannot drift from it. `iconutil` (macOS
only) makes the `.icns` — `tools/make_app.py` has the `build_icns`
step; Pillow makes the `.ico` and the `.png` from the same drawing,
a few lines that live in this repository's history rather than in a
tool.

## Updating an installed copy

There is no auto-updater, on purpose. `src/visualdynamics/update.py`
asks `https://visualdynamics.org/latest.json` whether a newer version
exists and says so; it downloads and executes nothing. An updater that
runs what it fetched without verifying a signature is remote code
execution with a friendly name. The macOS builds now carry an Apple
identity to verify against; Windows waits on its signing route.
Sparkle and WinSparkle do this properly and can be adopted when both
halves are signed.

Publishing a new version therefore means: tag, let the draft release
build, and publish it. The release workflow's `site` job writes
`web/launch/latest.json` from the published release and redeploys the
site; nothing is edited by hand. (`web/public/latest.json` is the
pre-release holding page's copy.)

    {"version": "0.1.0",
     "url": "https://github.com/visualdynamics/visualdynamics/releases",
     "notes": "What changed."}
