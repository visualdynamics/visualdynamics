# Replacing the Qt libraries

Visual Dynamics is built on Qt through PySide6, which is licensed under
the LGPL-3.0. That license gives you, as the person who received a
packaged build, the right to replace the Qt libraries inside it with
your own build of Qt, and this page says how. The packaged application
is a folder of separate shared libraries, not one sealed executable,
so that this is possible.

The bundled Qt version is the one the libraries themselves report: the
framework's `Info.plist` on macOS, the DLL's file properties on
Windows, `strings` on the shared object on Linux. It is whichever
PySide6 release was current when the package was built, since the
version is not pinned. Replace like with like, with a Qt build of the
same major.minor series (for example, any 6.12.x for a 6.12 bundle).
The sources are [Qt's own source releases](https://download.qt.io/official_releases/qt/)
and [PySide6 on PyPI](https://pypi.org/project/PySide6/#files) (the
sdist is the corresponding source for the bindings). The full license
texts ship in the package beside this file.

## Where the libraries live

| Platform | Location inside the package |
| --- | --- |
| macOS | `Visual Dynamics.app/Contents/Frameworks/PySide6/Qt/lib/Qt*.framework` (the `Contents/Frameworks/Qt*` entries are links into it) |
| Windows | `<install dir>\_internal\PySide6\Qt6*.dll` and the `PySide6` plugin folders beside them |
| Linux | inside the AppImage; extract it first (below) |

## macOS

1. Right-click the app, choose **Show Package Contents**, and replace
   the `Qt*.framework` directories under
   `Contents/Frameworks/PySide6/Qt/lib/` with your own builds of the
   same modules.
2. Replacing any library invalidates the code signature, and Apple
   silicon will not launch a bundle whose signature is broken. Re-sign
   it ad hoc, which needs no Apple account:

   ```bash
   codesign --force --deep --sign - "/Applications/Visual Dynamics.app"
   ```

3. Launch it. The ad hoc signature replaces the release's own signature
   and notarization, so Gatekeeper may ask you to confirm the first
   open, as it does for any unsigned download.

## Windows

Replace the `Qt6*.dll` files under `_internal\PySide6\` in the
installation directory (by default `C:\Program Files\Visual Dynamics`,
or `%LOCALAPPDATA%\Programs\Visual Dynamics` for an install for you
only) with your own builds, keeping the file names. The Windows package is
not code-signed, so there is no signature to repair. Start the
application normally.

## Linux

An AppImage is a read-only image, so unpack it, swap the libraries, and
either run the unpacked tree or repack it:

```bash
./VisualDynamics-*.AppImage --appimage-extract
# replace the Qt libraries under squashfs-root/usr/bin/_internal/PySide6/
./squashfs-root/AppRun            # run it unpacked, or:
appimagetool squashfs-root        # repack into a new AppImage
```

## What "your own build" has to be

It has to be binary-compatible with what it replaces: the same Qt
major.minor, the same platform and architecture, built as shared
libraries. Qt's [building-from-source guide](https://doc.qt.io/qt-6/build-sources.html)
covers the rest. If the application fails to start afterwards, suspect
the replaced libraries first; restoring the originals from a fresh
download returns the application to its released state.
