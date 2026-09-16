# Third-party notices

Visual Dynamics depends on these packages. It does not include them: a
normal install fetches each from PyPI under its own licence, and nothing
here changes those terms.

| Package | Licence | Obligation if this is redistributed |
| --- | --- | --- |
| numpy, scipy, h5py, pandas, pint, VTK | BSD-3-Clause | keep the copyright notice |
| netCDF4, pyvista, pyvistaqt, pyqtgraph, openpyxl | MIT | keep the copyright notice |
| certifi (Mozilla's CA bundle, as netCDF4's dependency and the update check's trust store) | MPL-2.0 | keep the notice; its files ship unmodified, and its source is at github.com/certifi/python-certifi |
| properdocs, mkdocs-materialx, mkdocstrings and friends (docs only) | BSD-2 / MIT / ISC | docs build only; nothing ships |
| pytest, pytest-cov, pytest-xdist, ruff, sdypy-sep005 (dev only) | MIT | test suite only; nothing ships |
| **PySide6 (Qt for Python)** | **LGPL-3.0** | see below |
| **OpenCASCADE** (via cadquery-ocp/OCP, Apache-2.0 bindings) | **LGPL-2.1 with exception** | see below |
| **libquadmath**, inside the numpy and scipy wheels | **LGPL-2.1-or-later** | see below |
| libgfortran / libgcc, inside those same wheels | GPL-3.0 **with the GCC Runtime Library Exception** | none — the exception exists to permit this |

Seven Qt modules ship: `QtCore`, `QtGui`, `QtWidgets`, `QtOpenGL` and
`QtSvg` from Qt Essentials, plus `QtWebEngineWidgets` and
`QtWebChannel` — add-ons, and deliberate; `QtTest` is used by the test
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
  Exception**. That exception exists for exactly this: it grants
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
question with a different answer, and `tests/test_licence_boundary.py`
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

The alternative is a commercial Qt licence from the Qt Company, which
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

**The middle path, when it is worth two days.** Check its licence
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
