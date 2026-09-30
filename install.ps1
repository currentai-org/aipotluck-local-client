# aipotluck-local-client -- one-line bootstrap installer (Windows)
#
# This is the script meant to be posted publicly and run via:
#
#   irm https://raw.githubusercontent.com/<OWNER>/<REPO>/main/install.ps1 | iex
#
# or, to pass installer flags (irm|iex can't take arguments, so use this
# form instead when you need e.g. -Backend or -System):
#
#   $script = irm https://raw.githubusercontent.com/<OWNER>/<REPO>/main/install.ps1
#   Invoke-Expression "& { $script } -Backend cuda -System"
#
# It does the minimum possible on its own -- clone the repo (installing git
# via winget first if needed) -- then hands off to
# packaging\windows\install.ps1, which does the real work already built and
# documented there: find-or-winget-install Python, then invoke
# aipotluck/installer/install.py. No logic is duplicated here.
#
# Configurable via -RepoUrl / -Ref / -SrcDir parameters or matching
# AIPOTLUCK_REPO_URL / AIPOTLUCK_REF / AIPOTLUCK_SRC_DIR environment
# variables. All other parameters are forwarded to
# packaging\windows\install.ps1 (-System, -Backend, -ModelHf, -Tag,
# -NoStart, -Verbose).
#
# STATUS: STUB. Written against documented winget/git/PowerShell behavior,
# not exercised on real Windows hardware (no Windows host available in
# this environment). packaging\windows\install.ps1 (which this hands off
# to) carries the same caveat; see that file and
# aipotluck/installer/python_bootstrap.py for details.

param(
    [string]$RepoUrl,
    [string]$Ref,
    [string]$SrcDir,
    [switch]$System,
    [string]$Backend = "auto",
    [string]$ModelHf,
    [string]$Tag,
    [switch]$NoStart,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$DefaultRepoUrl = "https://github.com/currentai-org/aitpotluck-local-client.git"

$repoUrl = if ($RepoUrl) { $RepoUrl } elseif ($env:AIPOTLUCK_REPO_URL) { $env:AIPOTLUCK_REPO_URL } else { $DefaultRepoUrl }
$ref = if ($Ref) { $Ref } elseif ($env:AIPOTLUCK_REF) { $env:AIPOTLUCK_REF } else { "main" }
$srcDir = if ($SrcDir) { $SrcDir } elseif ($env:AIPOTLUCK_SRC_DIR) { $env:AIPOTLUCK_SRC_DIR } else { Join-Path $env:LOCALAPPDATA "aipotluck\src" }

function Find-Git {
    $found = Get-Command git -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    return $null
}

function Install-GitViaWinget {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Error "git is required but not found, and winget is unavailable to install it. Install git manually from https://git-scm.com/download/win, then re-run this script."
        exit 1
    }
    Write-Host "git not found. Installing via winget (Git.Git)..."
    & winget install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Error "winget install of git failed (exit $LASTEXITCODE). Install git manually from https://git-scm.com/download/win and re-run."
        exit 1
    }
}

# --- main ---

$gitExe = Find-Git
if (-not $gitExe) {
    Install-GitViaWinget
    # Refresh PATH for this process from the machine+user environment,
    # since winget doesn't broadcast the update to the already-running
    # PowerShell session.
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
    $gitExe = Find-Git
    if (-not $gitExe) {
        Write-Error "git was installed by winget but can't be located in this session. Close and re-open PowerShell, then re-run this script."
        exit 1
    }
}

# NOTE: deliberately NOT --recurse-submodules -- see install.sh's note at the same point. The only
# submodule is vendor/llama.cpp (~174MB) and it is needed only for a from-source build, which
# source_build._init_vendor_submodule fetches on demand.
if (Test-Path (Join-Path $srcDir ".git")) {
    Write-Host "Existing checkout found at $srcDir -- updating"
    & $gitExe -C $srcDir fetch --depth 1 origin $ref
    & $gitExe -C $srcDir checkout $ref
    & $gitExe -C $srcDir reset --hard "origin/$ref"
    & $gitExe -C $srcDir submodule update --recursive
} else {
    Write-Host "Cloning $repoUrl (ref: $ref) into $srcDir"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $srcDir) | Out-Null
    & $gitExe clone --branch $ref --depth 1 $repoUrl $srcDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "git clone failed (exit $LASTEXITCODE)."
        exit 1
    }
}

$bootstrapScript = Join-Path $srcDir "packaging\windows\install.ps1"
if (-not (Test-Path $bootstrapScript)) {
    Write-Error "Expected bootstrap script not found at $bootstrapScript -- checkout may be incomplete."
    exit 1
}

$forwardArgs = @{ Backend = $Backend }
if ($System) { $forwardArgs["System"] = $true }
if ($ModelHf) { $forwardArgs["ModelHf"] = $ModelHf }
if ($Tag) { $forwardArgs["Tag"] = $Tag }
if ($NoStart) { $forwardArgs["NoStart"] = $true }
if ($Verbose) { $forwardArgs["Verbose"] = $true }

Write-Host "Handing off to: $bootstrapScript"
& $bootstrapScript @forwardArgs
exit $LASTEXITCODE
