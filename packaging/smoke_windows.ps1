# Smoke-test the freshly built installer on this very Windows.
#
#     powershell -ExecutionPolicy Bypass -File packaging\smoke_windows.ps1
#
# The release job builds the installer and uploads it, and for a long
# time nothing ever *ran* what it built — the first person to launch a
# build was whoever it was handed to. This walks the chain that person
# walks: install silently, open a project file in the installed
# application, wait for a real window, photograph the screen, close it.
# The screenshots land in dist\smoke\ for the workflow to upload, so a
# release manager can look at the application with their own eyes
# without owning a Windows machine.
#
# What a pass means: the installer lays the files down, the frozen
# build imports and starts (the classic packaging failures), a window
# titled for the application appears, and it survives opening a .vdyn.
# What it cannot mean: that the interface *behaves* — nothing here
# clicks. That still takes a human on Windows, but this is the part a
# human cannot check without one.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = if ($env:PYTHON) { $env:PYTHON }
          elseif (Test-Path '.\.venv\Scripts\python.exe') { '.\.venv\Scripts\python.exe' }
          else { 'python' }

$setup = Get-ChildItem dist\*setup.exe -ErrorAction SilentlyContinue |
         Select-Object -First 1
if (-not $setup) { throw 'no installer under dist\ to test' }

$work = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$target = Join-Path $work 'VisualDynamicsSmoke'
New-Item -ItemType Directory -Force -Path dist\smoke | Out-Null

# 1. Install it the way a user would, minus the clicking. Inno's
# silent mode still detaches, so wait on the process, not the call.
Start-Process -FilePath $setup.FullName -Wait -ArgumentList `
    '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', "/DIR=$target"
$exe = Join-Path $target 'VisualDynamics.exe'
if (-not (Test-Path $exe)) { throw "installed, but $exe is not there" }
Write-Host "installed to $target"

# A bounded run of the exe: the 0.1.0a56 release run hung in a check
# behind Windows' crash dialog until the job's forty-minute ceiling,
# with nothing in the logs. A check that has not exited in time is
# killed and fails with whatever it wrote.
function Invoke-Bounded($arguments, $outLog, $errLog, $seconds) {
    $p = Start-Process -FilePath $exe -PassThru -ArgumentList $arguments `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog
    # PowerShell 7 reads a process's exit code only when its handle
    # was taken before it exited and the untimed WaitForExit has run:
    # the 0.1.0a57 release run saw the launcher refuse correctly and
    # read its code as empty, and failed on that
    $null = $p.Handle
    if (-not $p.WaitForExit($seconds * 1000)) {
        $p.Kill()
        Get-Content $errLog | Select-Object -Last 40
        throw "the application did not exit within $seconds s running: $arguments"
    }
    $p.WaitForExit()
    if ($null -eq $p.ExitCode) { throw "no exit code read running: $arguments" }
    return $p.ExitCode
}

# 1a. Before Mesa, the runner is the machine a Remote Desktop user has:
# no OpenGL driver. The launcher must say so (gui.opengl.check, exit 2)
# instead of dying in VTK with 0xC0000005. Exit 0 means this runner has
# OpenGL of its own, so the path went unexercised: a warning, not a
# failure. Anything else, the access violation above all, fails.
$glLog = 'dist\smoke\gl-before-mesa.log'
$glErr = 'dist\smoke\gl-before-mesa-stderr.log'
$glCode = Invoke-Bounded @('--check-gl') $glLog $glErr 180
Get-Content $glLog
switch ($glCode) {
    2 { Write-Host 'without a driver, the launcher says what is missing' }
    0 { Write-Warning 'this runner has OpenGL of its own: the no-driver path was not exercised' }
    default {
        Get-Content $glErr | Select-Object -Last 40
        throw "the OpenGL check without a driver failed (exit $glCode)"
    }
}

# 1b. The runner has no GPU. Without one, Windows offers OpenGL 1.1
# and VTK cannot get a pixel format from it, looks for Mesa, and dies
# with 0xC0000005 before any window (the third run on real Windows,
# 2026-09-01, was the first to say so). A user's machine has a driver;
# this one gets Mesa's llvmpipe *beside the installed exe*, which is
# where Windows looks for opengl32.dll first — the same thing Wine
# needed to run the build at all. This is the test's fixture, not the
# product's: the installer ships no Mesa. One pinned version, because
# a fixture should be the same one every run.
$mesaVersion = '26.2.0'
$mesa = Join-Path $work "mesa3d-$mesaVersion-release-msvc.7z"
$mesaUrl = "https://github.com/pal1000/mesa-dist-win/releases/download/$mesaVersion/mesa3d-$mesaVersion-release-msvc.7z"
Invoke-WebRequest -Uri $mesaUrl -OutFile $mesa
$mesaDir = Join-Path $work 'mesa'
& 7z x -y "-o$mesaDir" $mesa 'x64\*' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'could not unpack Mesa' }
Copy-Item (Join-Path $mesaDir 'x64\*.dll') $target -Force
if (-not (Test-Path (Join-Path $target 'opengl32.dll'))) { throw 'Mesa did not land beside the exe' }
Write-Host "Mesa $mesaVersion (llvmpipe) placed beside the exe: the runner has no GPU"
# and with Mesa there, the same check must find OpenGL it can draw with
$glCode = Invoke-Bounded @('--check-gl') 'dist\smoke\gl-with-mesa.log' 'dist\smoke\gl-with-mesa-stderr.log' 180
Get-Content 'dist\smoke\gl-with-mesa.log'
if ($glCode -ne 0) {
    Get-Content 'dist\smoke\gl-with-mesa-stderr.log' | Select-Object -Last 40
    throw "with Mesa beside the exe, the OpenGL check failed (exit $glCode)"
}

# 1c. Can the installed build save an animation? Its files being there
# says nothing about whether the encoder inside them answers, so the
# frozen app writes and reads back a second of H.264 (gui.movie.check;
# no window). Exit 2 is "no H.264 on this machine": Windows Server can
# lack Media Foundation where a desktop never does, so that is a loud
# warning and the logs, not a failed build; anything else is a failure.
$movieLog = 'dist\smoke\movie.log'
$movieErr = 'dist\smoke\movie-stderr.log'
$movieCode = Invoke-Bounded @('--check-movie', "`"$(Join-Path $work 'check.mp4')`"") $movieLog $movieErr 600
Get-Content $movieLog
switch ($movieCode) {
    0 { Write-Host 'saves an animation as H.264' }
    2 { Write-Warning 'no H.264 encoder on this runner: Save Animation is not offered here' }
    default {
        Get-Content $movieErr | Select-Object -Last 40
        throw "the movie check failed (exit $movieCode)"
    }
}

# 2. A small real project for it to open — made by the *source* tree's
# python, read by the *installed* build, which is exactly the round
# trip a colleague's machine performs on a file made on this one.
$vdyn = Join-Path $work 'smoke.vdyn'
& $python -c @"
import numpy as np
import visualdynamics
project = visualdynamics.Project('Windows Smoke Test')
t = np.arange(2048) / 256.0
project.add('Time History', visualdynamics.TimeHistory(
    t, np.atleast_2d(np.sin(2 * np.pi * 12.5 * t)),
    response_dof=['101Z+']))
project.save(r'$vdyn')
"@
if ($LASTEXITCODE -ne 0) { throw 'could not write the smoke project' }

# 3. Launch the installed application on it and wait for a real
# window. A frozen Qt+VTK build cold-starts slowly on a busy runner,
# so patience up to two minutes before calling it dead.
# Its output goes to files: the first run on real Windows died with
# 0xC0000005 before any window and said nothing else (2026-09-01), so
# on an early exit the log gets whatever it printed plus the faulting
# module from the event log, which is the one fact an access violation
# leaves behind.
$out = 'dist\smoke\stdout.log'
$err = 'dist\smoke\stderr.log'
# --no-disclaimer: the alpha notice wants a box ticked before the
# window takes input, and nothing here can tick it — without the flag
# the test stopped at the dialog and never opened the project
# (2026-09-14, Brandon's call). The flag is what a script uses too.
$proc = Start-Process -FilePath $exe -ArgumentList '--no-disclaimer', "`"$vdyn`"" -PassThru `
    -RedirectStandardOutput $out -RedirectStandardError $err
$title = ''
foreach ($tick in 1..120) {
    Start-Sleep -Seconds 1
    if ($proc.HasExited) {
        Start-Sleep -Seconds 5      # let Windows Error Reporting write its event
        foreach ($f in $out, $err) {
            if (Test-Path $f) { Write-Host "--- $f"; Get-Content $f | Select-Object -Last 40 }
        }
        Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000; StartTime=(Get-Date).AddMinutes(-5)} `
            -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host '--- Application Error event:'; Write-Host $_.Message }
        throw "the application exited (code $($proc.ExitCode)) before showing a window"
    }
    $proc.Refresh()
    $title = $proc.MainWindowTitle
    if ($title) { break }
}
if (-not $title) { throw 'no main window after two minutes' }
if ($title -notlike '*Visual Dynamics*') {
    throw "a window appeared, but titled '$title'"
}
Write-Host "window up: '$title' after ${tick}s"

# 4. Let the project tree and the first drawing settle, then
# photograph the whole screen — the artifact a person actually reads.
Start-Sleep -Seconds 10
if ($proc.HasExited) { throw 'the application died after opening the project' }
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$screen = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bitmap = New-Object System.Drawing.Bitmap $screen.Width, $screen.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($screen.Left, $screen.Top, 0, 0, $bitmap.Size)
$bitmap.Save('dist\smoke\launched.png')
Write-Host 'screenshot: dist\smoke\launched.png'

# 4b. A window up and a traceback on stderr is a crash with a
# screenshot: the first Windows launch of 0.1.0a1 showed an unhandled-
# exception dialog behind a titled window, and this test called it a
# pass (2026-09-14). The frozen build prints its tracebacks to stderr
# before the dialog; any there fails the build, screenshot or not.
if ((Test-Path $err) -and (Select-String -Path $err -Pattern 'Traceback|Unhandled exception|Error calling Python override' -Quiet)) {
    Write-Host "--- $err"; Get-Content $err | Select-Object -Last 40
    throw 'the application raised after opening: see stderr.log and launched.png'
}

# 5. Ask it to close; insist only if it will not. A hang on close is
# worth knowing about but not worth failing a build the screenshot
# already vouches for — it goes to the log, loudly.
$proc.CloseMainWindow() | Out-Null
if (-not $proc.WaitForExit(15000)) {
    Write-Warning 'did not close within 15 s of being asked; killed'
    Stop-Process -Id $proc.Id -Force
} else {
    Write-Host "closed cleanly (exit $($proc.ExitCode))"
}
Write-Host 'smoke test passed'
