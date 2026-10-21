# Third-party notices

Visual Dynamics depends on these packages. The `visualdynamics` package
on PyPI does not include them: a normal install fetches each from PyPI
under its own license, and nothing here changes those terms. The
packaged applications carry them, and the macOS app carries Sparkle
besides.

| Package | License | Obligation if this is redistributed |
| --- | --- | --- |
| numpy, scipy, h5py, pandas, pint, VTK | BSD-3-Clause | keep the copyright notice |
| netCDF4, pyvista, pyvistaqt, pyqtgraph, openpyxl, truststore (the update check's use of the system's certificate store) | MIT | keep the copyright notice |
| pyobjc-core (the macOS app only) | MIT | keep the copyright notice |
| **Sparkle** 2, the macOS app's updater: a framework the build copies into the app from Sparkle's own release, not a PyPI package | MIT, with parts under BSD-2-Clause (bsdiff, the signature verifier), MIT (sais-lite) and zlib (Ed25519) | its text, with those parts' notices, ships as `licenses/third-party/Sparkle-LICENSE.txt`; source at github.com/sparkle-project/Sparkle |
| certifi (Mozilla's CA bundle, as netCDF4's dependency and the update check's trust store) | MPL-2.0 | keep the notice; its files ship unmodified, and its source is at github.com/certifi/python-certifi |
| properdocs, mkdocs-materialx, mkdocstrings and friends (docs only) | BSD-2 / MIT / ISC | docs build only; nothing ships |
| pytest, pytest-cov, pytest-xdist, ruff, sdypy-sep005 (dev only) | MIT | test suite only; nothing ships |
| escdf, the Engineering Sciences Common Data Format's reference implementation (dev only; not on PyPI, so CI installs it from its repository in a step of its own and it is not in the package metadata) | BSD-3-Clause | test suite only, as the format's agreement oracle; nothing of it ships. Its **specification files** do ship, vendored as `io/escdf_specifications/*.txt` with their `LICENSE` beside them, so the reader here knows the standard's types |
| **PySide6 (Qt for Python)** | **LGPL-3.0** | see below |
| **OpenCASCADE** 8.0.1 (via cadquery-ocp/OCP, Apache-2.0 bindings), and the libraries its wheel carries | **LGPL-2.1 with the Open CASCADE exception**; the others as listed below | see below |
| **libquadmath**, inside the numpy and scipy wheels | **LGPL-2.1-or-later** | see below |
| **FFmpeg** (libavcodec, libavformat, libavutil, libswscale, libswresample), inside the PySide6 wheels for Qt Multimedia | **LGPL-2.1-or-later** — built without its GPL parts (no libx264), as the libraries report of themselves | the same as Qt's: shared libraries, shipped unmodified and replaceable; source at ffmpeg.org (n7.1.5) |
| libgfortran / libgcc, inside those same wheels | GPL-3.0 **with the GCC Runtime Library Exception** | none — the exception exists to permit this |

Nine Qt modules ship: `QtCore`, `QtGui`, `QtWidgets`, `QtOpenGL` and
`QtSvg` from Qt Essentials, plus `QtWebEngineWidgets`,
`QtWebEngineCore`, `QtWebChannel` and `QtMultimedia` (saving an
animation as a video) — add-ons, and deliberate; `QtTest` is used by the test
suite only. **All of them are offered under LGPLv3**, checked against
Qt's own module licensing pages, so the obligations below cover the
whole of what is used. Essentials-vs-add-ons is not the boundary that
matters; some Qt modules are GPL-or-commercial, and any new one gets
its licensing page read before it is imported.


## What rides inside the numpy and scipy wheels

Those wheels are not pure Python: they carry compiled Fortran and the
runtime it needs, and two of those pieces are copyleft. Neither is a
problem, and both are written down here because a reader who greps this
tree for "GPL" will find them and should not have to work out why they
are allowed.

- **libgfortran and libgcc** are GPL-3.0 **with the GCC Runtime Library
  Exception** (its text ships as
  `licenses/GCC-runtime-library-exception-3.1.txt`). That exception exists for exactly this: it grants
  permission to combine the runtime with independent modules and
  convey the result under terms of your choosing, proprietary
  included, as long as the compilation was done with GCC. scipy's own
  wheels are, so the condition is met and nothing propagates.
- **libquadmath** is LGPL-2.1-or-later, and is a separate dynamically
  linked shared library inside the wheel. The obligation is the same
  one PySide6 already imposes and that `--onedir` already satisfies:
  the recipient must be able to replace it. A `--onefile` build would
  make this a live question; the packaging rule that keeps Qt legal
  keeps this legal too, which is one more reason it is not revisited.

None of this touches the rule about sdynpy, rattlesnake and
forcefinder. That one is about *source* copied into this tree, where
the copyleft would attach to Visual Dynamics' own code; depending on a
BSD library that links a GPL-with-exception runtime is a different
question with a different answer, and `tests/test_license_boundary.py`
still enforces the first one on every run.


## PySide6 is the one with conditions

Everything else above is permissive: ship it inside anything, closed or
open, as long as the notice travels with it. PySide6 is LGPL-3.0, which
is compatible with shipping a closed-source application **and** carries
obligations that have to be designed for rather than discovered.

Importing PySide6 from Python is dynamic linking, so the application's
own source stays the author's. What the LGPL asks in return is that the
person who receives the application can **replace the Qt libraries with
their own build**:

- Ship the Qt/PySide6 libraries as separate shared objects. Do not fold
  them into a single statically linked executable — that is the case
  the LGPL requires relinkable object files for, and the case worth
  avoiding entirely.
- Ship the LGPL text, and either the corresponding PySide6/Qt source or
  a written offer to supply it.
- Say which version was used, so a replacement can be built against it.

`packaging/visualdynamics.spec` is where this becomes real: --onedir
keeps the Qt frameworks as their own dylibs/DLLs, which satisfies the
replaceability requirement; a single frozen binary would not. The
packaged builds carry this file, the LGPL and GPL texts
(`licenses/`), and `qt-replacement.md` — the recipient-facing
instructions, also on the documentation site — so the right to
replace arrives with the copy it applies to.

**OpenCASCADE — the STEP/IGES kernel — rides the same rule.** Its
LGPL-2.1-with-exception asks what Qt's LGPL-3.0 asks and `--onedir`
already grants: the recipient can replace the kernel's shared
libraries in the package they received. The bindings (`OCP`, from the
cadquery-ocp wheels) are Apache-2.0 and impose nothing further. The
per-platform swap mechanics in `qt-replacement.md` apply to these
libraries the same way — they live beside the Qt ones in the bundle.
The LGPL-2.1 text ships as `licenses/LGPL-2.1.txt` (it covers
libquadmath, FFmpeg and LibRaw too), the exception as
`licenses/third-party/OpenCASCADE-LGPL-exception.txt`. The version is
8.0.1; its source is at github.com/Open-Cascade-SAS/OCCT, tag `V8_0_1`,
and the wheel's at github.com/CadQuery/OCP. The libraries are shipped
unmodified.


## OpenCASCADE, and what its wheel carries

**Visual Dynamics makes use of facilities provided by the Open CASCADE
Technology software** (OCCT, https://dev.opencascade.org). The bindings
are compiled against OCCT's header files, so this is the notice the
Open CASCADE exception asks for in return.

The cadquery-ocp wheel carries shared libraries of its own beside
OCCT's, and every package ships them. Each one's license text is in
`licenses/third-party/`, the file named in the right column. Where a
library offers a choice of licenses, the one chosen here is the one
listed. Read off the macOS wheel (cadquery-ocp 8.0.1.1.0) and the
Linux one, whose builds of some projects are older and which carries
a few more (marked *Linux*); `tests/test_app_bundle.py` holds the list
against whichever wheel is installed where it runs. The Windows wheel
has not been read yet.

| Library (as bundled) | Project | License | Text |
| --- | --- | --- | --- |
| libfreeimage | FreeImage 3.18 | FreeImage Public License 1.0 (chosen of FIPL, GPL-2.0, GPL-3.0) | `FreeImage-license-fi.txt` |
| libfreetype | FreeType 2.12.1 | FreeType License (chosen of FTL, GPL-2.0) | `FreeType-FTL.txt` |
| libraw | LibRaw | LGPL-2.1 (chosen of LGPL-2.1, CDDL-1.0); shipped as a replaceable shared library, source at github.com/LibRaw/LibRaw | `LibRaw-COPYRIGHT.txt`, `../LGPL-2.1.txt` |
| libOpenEXR, libOpenEXRCore, libIex, libIlmThread | OpenEXR | BSD-3-Clause | `OpenEXR-LICENSE.md` |
| libImath | Imath | BSD-3-Clause | `Imath-LICENSE.md` |
| libIlmImf, libHalf, libIexMath (and Iex, IlmThread, Imath 2.2), *Linux* | OpenEXR 2.2 | BSD-3-Clause, Industrial Light & Magic | `OpenEXR-2.2-LICENSE.txt` |
| libbz2, *Linux* | bzip2 | bzip2 License (BSD-like) | `bzip2-LICENSE.txt` |
| libfontconfig, *Linux* | Fontconfig | MIT-like | `fontconfig-COPYING.txt` |
| libjasper, *Linux* | JasPer | JasPer License 2.0 (MIT-like) | `JasPer-LICENSE.txt` |
| libuuid, *Linux* | util-linux libuuid | BSD-3-Clause | `libuuid-COPYING.txt`, `util-linux-COPYING.BSD-3-Clause.txt` |
| libgomp, *Linux* | GCC's OpenMP runtime | GPL-3.0 with the GCC Runtime Library Exception, as libgfortran above | `../GPL-3.0.txt`, `../GCC-runtime-library-exception-3.1.txt` |
| libjpeg | libjpeg-turbo | IJG License and BSD-3-Clause | `libjpeg-turbo-LICENSE.md`, `libjpeg-turbo-README.ijg.txt` |
| libjpegxr, libjxrglue | jxrlib | BSD-2-Clause | `jxrlib-LICENSE.txt` |
| libopenjp2 | OpenJPEG | BSD-2-Clause | `OpenJPEG-LICENSE.txt` |
| libopenjph | OpenJPH | BSD-2-Clause | `OpenJPH-LICENSE.txt` |
| libpng16 | libpng | libpng License | `libpng-LICENSE.txt` |
| libtiff | LibTIFF | LibTIFF License (BSD-like) | `libtiff-LICENSE.md` |
| libwebp, libwebpmux, libsharpyuv | libwebp | BSD-3-Clause | `libwebp-COPYING.txt` |
| libLerc | LERC | Apache-2.0 | `Lerc-LICENSE.txt` |
| liblcms2 | Little CMS | MIT | `LittleCMS-LICENSE.txt` |
| liblzma | XZ Utils | 0BSD | `xz-liblzma-COPYING.0BSD.txt` |
| libz | zlib | zlib License | `zlib-LICENSE.txt` |
| libzstd | Zstandard | BSD-3-Clause (chosen of BSD, GPL-2.0) | `zstd-LICENSE.txt` |
| libdeflate | libdeflate | MIT | `libdeflate-COPYING.txt` |
| libfmt | {fmt} | MIT | `fmt-LICENSE.txt` |
| libc++ | LLVM libc++ | Apache-2.0 with LLVM Exceptions | `libcxx-LICENSE.txt` |

The FreeType License asks for this credit in a product's
documentation: Portions of this software are copyright © 2022 The
FreeType Project (https://freetype.org). All rights reserved.

The alternative is a commercial Qt license from the Qt Company, which
removes the obligation and is priced accordingly. Worth pricing before
any plan depends on a single-file distributable.

None of this touches evaluation or in-house use: the obligations attach
to **distribution**, and until something is distributed there is nothing
to satisfy.

## If the download size ever matters: the QtWebEngine question

Measured in this environment: PySide6 installs at 1.1 GB, and **588 MB
of it is `QtWebEngineCore` — Chromium.** One widget in one file needs
it: the `QWebEngineView` in `gui/report_editor.py`, which is where a
report is read and edited inside the window.

It is worth knowing what the ways out cost, because the obvious one is
the worst one.

**Do not simply drop it.** The report page is JavaScript-driven — it
ships a JSON payload and draws client-side — so Qt's other HTML widget,
`QTextBrowser`, cannot show it at all. Removing `QWebEngineView` means
the report leaves the window for the system browser, which is a real
downgrade against "the panes live in the window", in exchange for
megabytes off a distributable that may not exist. The read-only page is
already self-contained (816 kB, no external references) and opens in any
browser today, so that fallback is always available — it is just not an
improvement.

**The middle path, when it is worth two days.** Check its license
first — swapping one Qt module for another is exactly the move the rule
above exists for, and a saving in megabytes is no bargain if it costs
the LGPL. `PySide6.QtWebView` is
164 kB and wraps the platform engine — WKWebView on macOS, no Chromium.
It needs QtQuick to host it (12 MB), so the trade is roughly **12 MB in
place of 588 MB, and the view stays in the window**. Two catches: it is
QML-only, so it needs a `QQuickWidget` host and a small QML wrapper; and
it has no `QWebChannel`, so the editor's JS-to-Python bridge
(`gui/report_editor.py`, and the `new QWebChannel(...)` block in
`report/page.py`) would have to be rebuilt on URL-scheme interception or
polling.

Nothing here is worth doing until a distributable is real. Written down
so the measurements do not have to be taken again.
