# aipotluck-local-client Windows bootstrapper
#
# This is the TRUE entry point for a Windows user with no prerequisites --
# it does not assume Python is already installed (that's the whole point:
# `python -m aipotluck.installer.install` can't run without Python, so
# something that doesn't need Python has to get Python first).
#
# STUB STATUS: written against documented winget/PowerShell behavior and not
# itself run. Logic mirrors aipotluck/installer/python_bootstrap.py's Windows
# path so both entry points agree; that module is the one covered by
# unit-style testing (see tests), this script just drives it. The winget
# branch that installs Python when it is missing has never been exercised.
#
# Usage (from an elevated OR regular PowerShell prompt):
#   powershell -ExecutionPolicy Bypass -File packaging\windows\install.ps1
#
# What it does:
#   1. Look for a working `python`/`py` on PATH (>=3.9).
#   2. If missing, install Python via winget (silent, accept agreements).
#   3. Re-resolve python.exe (winget doesn't refresh the current session's
#      PATH, so we probe well-known install locations directly).
#   4. Hand off to `python -m aipotluck.installer.install` with any args
#      this script was called with.

param(
    [switch]$System,
    [string]$Backend = "auto",
    [string]$ModelHf,
    [string]$Tag,
    [switch]$NoStart,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

function Find-Python {
    foreach ($cmd in @("py", "python", "python3")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) {
            try {
                $verOut = & $cmd -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    $parts = $verOut.Trim().Split(".")
                    if ([int]$parts[0] -ge 3 -and [int]$parts[1] -ge 9) {
                        return $found.Source
                    }
                }
            } catch { }
        }
    }
    return $null
}

function Install-PythonViaWinget {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Error "winget not found. Install Python 3.9+ manually from https://www.python.org/downloads/windows/ (check 'Add python.exe to PATH'), then re-run this script."
        exit 1
    }
    Write-Host "No suitable Python found. Installing via winget (Python.Python.3.12)..."
    & winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Error "winget install failed (exit $LASTEXITCODE). Install Python manually from https://www.python.org/downloads/windows/ and re-run."
        exit 1
    }
}

function Find-PythonAfterInstall {
    $found = Find-Python
    if ($found) { return $found }

    # winget doesn't broadcast a PATH update to this already-running
    # process; probe the well-known install directories directly.
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python"),
        $env:ProgramFiles
    )
    foreach ($base in $candidates) {
        if (Test-Path $base) {
            $pyDirs = Get-ChildItem -Path $base -Directory -Filter "Python3*" -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending
            foreach ($dir in $pyDirs) {
                $exe = Join-Path $dir.FullName "python.exe"
                if (Test-Path $exe) { return $exe }
            }
        }
    }
    return $null
}

# --- main ---

$pythonExe = Find-Python
if (-not $pythonExe) {
    Install-PythonViaWinget
    $pythonExe = Find-PythonAfterInstall
    if (-not $pythonExe) {
        Write-Error "Python was installed by winget but can't be located in this session. Close and re-open PowerShell, then re-run this script."
        exit 1
    }
}
Write-Host "Using Python: $pythonExe"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$installerArgs = @("-m", "aipotluck.installer.install", "--backend", $Backend)
if ($System) { $installerArgs += "--system" }
if ($ModelHf) { $installerArgs += @("--model-hf", $ModelHf) }
if ($Tag) { $installerArgs += @("--tag", $Tag) }
if ($NoStart) { $installerArgs += "--no-start" }
if ($Verbose) { $installerArgs += "-v" }

Push-Location $repoRoot
try {
    & $pythonExe @installerArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
