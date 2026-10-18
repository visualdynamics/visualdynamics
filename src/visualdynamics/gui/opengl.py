"""Whether this machine can draw the 3-D view, asked before it is built.

The 3-D view is VTK, and VTK draws with OpenGL 3.2 or later. On Windows
it asks the system's `opengl32.dll` for a pixel format. A machine with no
graphics driver (a Remote Desktop session, a virtual machine without
one, a server) has only Windows' own OpenGL 1.1, and VTK, refused, looks
for Mesa's off-screen library and dies with an access violation before
a window appears. So the launcher asks the same `opengl32.dll` first, in
the same way, and a machine that cannot draw is told what is missing.

The question is put to Windows directly, not through Qt: Qt can fall
back to its own software renderer where VTK cannot, and would answer
yes for a machine that VTK then crashes on. The macOS and Linux builds
are not asked. macOS always has OpenGL, and the Linux AppImage carries
Mesa.
"""

from __future__ import annotations

import re
import sys

#: the least OpenGL VTK's renderer accepts
NEEDED = (3, 2)


def judge(version: str | None, renderer: str | None) -> str | None:
    """Why OpenGL `version` from `renderer` cannot draw the 3-D view, or
    None when it can.

    `version` and `renderer` are the strings OpenGL reports
    (`GL_VERSION`, `GL_RENDERER`), such as '4.6.0 NVIDIA 551.23' and
    '1.1.0' with 'GDI Generic'; None for both when Windows offered no
    OpenGL pixel format at all.
    """
    remedy = ('Install or update the graphics driver. Over Remote Desktop, '
              'or in a virtual machine without a graphics driver, Windows '
              'offers only its own OpenGL 1.1; placing Mesa\'s '
              'opengl32.dll (llvmpipe) beside visualdynamics.exe gives '
              'the 3-D view a software renderer instead.')
    need = f'OpenGL {NEEDED[0]}.{NEEDED[1]}'
    if version is None:
        return (f'Visual Dynamics draws its 3-D view with {need} or later, '
                f'and Windows offered no OpenGL here. {remedy}')
    found = re.match(r'\s*(\d+)\.(\d+)', version)
    if found and (int(found[1]), int(found[2])) >= NEEDED:
        return None
    what = version.split()[0] if version.split() else version
    by = f' ({renderer})' if renderer else ''
    return (f'Visual Dynamics draws its 3-D view with {need} or later, and '
            f'this machine offers OpenGL {what}{by}. {remedy}')


def windows_opengl() -> tuple[str | None, str | None]:
    """(GL_VERSION, GL_RENDERER) from the OpenGL Windows gives a plain
    double-buffered window, asked through `opengl32.dll` as VTK asks;
    (None, None) when no OpenGL pixel format is offered. Windows only."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL('user32')
    gdi32 = ctypes.WinDLL('gdi32')
    # the same search VTK's load goes through, so a Mesa opengl32.dll
    # beside the exe is the one asked
    opengl32 = ctypes.WinDLL('opengl32')

    class PixelFormat(ctypes.Structure):
        _fields_ = [('nSize', wintypes.WORD), ('nVersion', wintypes.WORD),
                    ('dwFlags', wintypes.DWORD)] + [
            (name, ctypes.c_ubyte) for name in (
                'iPixelType', 'cColorBits', 'cRedBits', 'cRedShift',
                'cGreenBits', 'cGreenShift', 'cBlueBits', 'cBlueShift',
                'cAlphaBits', 'cAlphaShift', 'cAccumBits', 'cAccumRedBits',
                'cAccumGreenBits', 'cAccumBlueBits', 'cAccumAlphaBits',
                'cDepthBits', 'cStencilBits', 'cAuxBuffers', 'iLayerType',
                'bReserved')] + [
            ('dwLayerMask', wintypes.DWORD), ('dwVisibleMask', wintypes.DWORD),
            ('dwDamageMask', wintypes.DWORD)]

    # handles are pointer-sized; left to ctypes' default int they are cut
    # to 32 bits on a 64-bit process
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
    user32.GetDC.restype = wintypes.HDC
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    gdi32.ChoosePixelFormat.argtypes = [wintypes.HDC,
                                        ctypes.POINTER(PixelFormat)]
    gdi32.SetPixelFormat.argtypes = [wintypes.HDC, ctypes.c_int,
                                     ctypes.POINTER(PixelFormat)]
    opengl32.wglCreateContext.restype = wintypes.HGLRC
    opengl32.wglCreateContext.argtypes = [wintypes.HDC]
    opengl32.wglMakeCurrent.argtypes = [wintypes.HDC, wintypes.HGLRC]
    opengl32.wglDeleteContext.argtypes = [wintypes.HGLRC]
    opengl32.glGetString.restype = ctypes.c_char_p
    opengl32.glGetString.argtypes = [ctypes.c_uint]

    window = user32.CreateWindowExW(0, 'STATIC', 'OpenGL check',
                                    0x80000000,     # WS_POPUP, never shown
                                    0, 0, 1, 1, None, None, None, None)
    if not window:
        return None, None
    dc = user32.GetDC(window)
    context = None
    try:
        wanted = PixelFormat()
        wanted.nSize = ctypes.sizeof(PixelFormat)
        wanted.nVersion = 1
        # PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
        wanted.dwFlags = 0x4 | 0x20 | 0x1
        wanted.cColorBits, wanted.cDepthBits, wanted.cStencilBits = 32, 24, 8
        index = gdi32.ChoosePixelFormat(dc, ctypes.byref(wanted))
        if not index or not gdi32.SetPixelFormat(dc, index,
                                                 ctypes.byref(wanted)):
            return None, None
        context = opengl32.wglCreateContext(dc)
        if not context or not opengl32.wglMakeCurrent(dc, context):
            return None, None
        version = opengl32.glGetString(0x1F02)          # GL_VERSION
        renderer = opengl32.glGetString(0x1F01)         # GL_RENDERER
        return ((version or b'').decode(errors='replace') or None,
                (renderer or b'').decode(errors='replace') or None)
    finally:
        if context:
            opengl32.wglMakeCurrent(None, None)
            opengl32.wglDeleteContext(context)
        user32.ReleaseDC(window, dc)
        user32.DestroyWindow(window)


def missing() -> str | None:
    """Why this machine cannot draw the 3-D view, or None when it can or
    the platform is not one that is asked (`judge` on Windows)."""
    # Asked since 2026-10-18: a release run on a driverless Windows
    # runner (2026-09-01) died with 0xC0000005 before a window, and a
    # user over Remote Desktop got the same crash rather than a sentence
    if sys.platform != 'win32':
        return None
    return judge(*windows_opengl())


def check() -> tuple[int, str]:
    """`visualdynamics --check-gl`: 0 and what OpenGL is here when the 3-D
    view can be drawn, 2 and the reason when it cannot. The Windows
    release smoke test runs it with and without Mesa beside the exe."""
    if sys.platform != 'win32':
        return 0, 'not asked on this platform'
    version, renderer = windows_opengl()
    reason = judge(version, renderer)
    if reason is not None:
        return 2, reason
    return 0, f'OpenGL {version} ({renderer})'
