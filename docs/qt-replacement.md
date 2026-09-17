# Replacing the Qt libraries

Visual Dynamics is built on Qt through PySide6, which is licensed
under the LGPL-3.0. That license grants you, the person who received
a packaged build, the right to replace the Qt libraries inside it
with your own build of Qt — and this page is how. The packaged
application is deliberately a *folder* of separate shared libraries
rather than one sealed executable, precisely so this is possible.

The bundled version is the one the Qt libraries themselves report
(the framework's `Info.plist` on macOS, the DLL's file properties on
Windows, `strings` on the shared object on Linux); it is whatever
PySide6 release was current when the package was built, since nothing
is pinned. Replace like with like: a Qt build of
the same major.minor series (for example, any 6.11.x for a 6.11
bundle). Sources: [Qt's own source releases](https://download.qt.io/official_releases/qt/)
and [PySide6 on PyPI](https://pypi.org/project/PySide6/#files)
(the sdist is the corresponding source for the bindings). The full
license texts ship in the package beside this file.

## Where the libraries live

| Platform | Location inside the package |
| --- | --- |
| macOS | `Visual Dynamics.app/Contents/Frameworks/PySide6/Qt/lib/Qt*.framework` (the `Contents/Frameworks/Qt*` entries are links into it) |
| Windows | `<install dir>\_internal\PySide6\Qt6*.dll` and the `PySide6` plugin folders beside them |
| Linux | inside the AppImage — extract first (below) |

## macOS

1. Right-click the app, **Show Package Contents**, and replace the
   `Qt*.framework` directories under
   `Contents/Frameworks/PySide6/Qt/lib/` with your own builds of the
   same modules.
2. Replacing any library invalidates the code signature, and Apple
   silicon refuses to launch a bundle with a broken one. Re-sign it
   ad hoc — no Apple account is needed:

   ```bash
   codesign --force --deep --sign - "/Applications/Visual Dynamics.app"
   ```

3. Launch. Gatekeeper may ask you to confirm the first open
   (right-click → Open), exactly as for any unsigned download.

## Windows

Replace the `Qt6*.dll` files under `_internal\PySide6\` in the
installation directory (default `C:\Program Files\Visual Dynamics`)
with your own builds, keeping the filenames. There is no signature to
repair. Start the application normally.

## Linux

An AppImage is a read-only image, so unpack it, swap, and either run
the unpacked tree or repack:

```bash
./VisualDynamics-*.AppImage --appimage-extract
# replace squashfs-root/usr/bin/_internal/PySide6/*.so.6 / Qt libraries
./squashfs-root/AppRun            # run it unpacked, or:
appimagetool squashfs-root        # repack into a new AppImage
```

## What "your own build" has to be

Binary-compatible with what it replaces: same Qt major.minor, same
platform and architecture, built as shared libraries. Qt's
[building-from-source guide](https://doc.qt.io/qt-6/build-sources.html)
covers the rest. If the application fails to start afterwards, the
replaced libraries are the first suspect — restore the originals from
a fresh download and the application is whole again.
