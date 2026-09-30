#!/usr/bin/env bash
# aipotluck-local-client -- one-line bootstrap installer (Linux & macOS)
#
# This is the script meant to be posted publicly and piped into bash, with
# NO arguments -- standard practice for this kind of installer (rustup,
# Homebrew, etc.): it assumes nothing is installed yet, does its best to
# install whatever's missing itself, and leaves the software in a "logged
# out" state. Pairing (tunnel id/secret/endpoint) is a separate, later step
# -- see `aipotluck-local-client login` at the end of this script's
# output, or README.md.
#
#   curl -fsSL https://raw.githubusercontent.com/<OWNER>/<REPO>/main/install.sh | bash
#
# It does the minimum possible on its own -- make sure git + python3 exist
# (installing them via the system package manager if not), fetch the repo
# -- then hands off to the already-implemented, already-tested installer
# (aipotluck/installer/install.py) for everything else (llama.cpp
# download+verify, service registration, PATH shim, etc). No installer
# logic is duplicated here.
#
# Configurable via environment variables (all optional):
#   AIPOTLUCK_REPO_URL   git remote to clone (default: see below)
#   AIPOTLUCK_REF        branch/tag to check out (default: main)
#   AIPOTLUCK_SRC_DIR     where to clone the source (default: ~/.aipotluck/src)
#
# IMPORTANT when setting one of these through a `curl | bash` pipe: `VAR=value curl ... | bash`
# does NOT work -- a leading assignment on a pipeline only applies to the first command in it
# (curl here), never to a later one (bash, which is what actually reads AIPOTLUCK_REF/etc). This
# is silent, not an error: bash just sees the variable as unset and falls back to its default, so
# the failure mode is "it quietly checked out main instead," not a visible one (confirmed live,
# CUR-1965). Put the assignment on bash's side of the pipe instead, or export it beforehand:
#
#   curl -fsSL .../install.sh | AIPOTLUCK_REF=my-branch bash
#   # or:
#   export AIPOTLUCK_REF=my-branch
#   curl -fsSL .../install.sh | bash
#
# Extra arguments (optional, advanced use only -- the published one-liner
# needs none of these) are passed straight through to
# `python3 -m aipotluck.installer.install`, e.g.:
#
#   curl -fsSL .../install.sh | bash -s -- --backend cuda --model-hf "org/repo:Q4_K_M"
#
# STATUS: this script is a thin, tested-by-inspection wrapper. The
# heavy-lifting installer it calls (aipotluck/installer/install.py) is
# tested end-to-end on Linux; this bootstrap layer just needs a shell, and makes a
# best-effort attempt to install git/python3 itself via whatever package
# manager it finds (apt-get/dnf/yum/pacman/apk/zypper/brew) before falling
# back to a clear, actionable error if none of that works.

set -euo pipefail

DEFAULT_REPO_URL="https://github.com/currentai-org/aitpotluck-local-client.git"

REPO_URL="${AIPOTLUCK_REPO_URL:-$DEFAULT_REPO_URL}"
REF="${AIPOTLUCK_REF:-main}"
SRC_DIR="${AIPOTLUCK_SRC_DIR:-$HOME/.aipotluck/src}"

log() { printf '\033[1;34m[aipotluck]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[aipotluck] WARNING:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[aipotluck] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# sudo prompts for a password via the controlling terminal (/dev/tty), not stdin -- so this still
# works interactively even when this whole script arrived through a `curl | bash` pipe. Skip the
# prefix entirely when already root (e.g. inside a minimal container).
SUDO=""
if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

# Best-effort: install $1 (a generic package name, e.g. "git") via whatever package manager is
# found on PATH. Returns non-zero (never fatal on its own) if nothing usable was found or the
# install failed -- the caller re-checks `command -v` and only then decides whether to give up.
install_pkg() {
    local pkg="$1" pacman_pkg="${2:-$1}"
    if command -v apt-get >/dev/null 2>&1; then
        log "Installing $pkg via apt-get..."
        $SUDO apt-get update -qq || true
        $SUDO apt-get install -y "$pkg"
    elif command -v dnf >/dev/null 2>&1; then
        log "Installing $pkg via dnf..."
        $SUDO dnf install -y "$pkg"
    elif command -v yum >/dev/null 2>&1; then
        log "Installing $pkg via yum..."
        $SUDO yum install -y "$pkg"
    elif command -v pacman >/dev/null 2>&1; then
        log "Installing $pacman_pkg via pacman..."
        $SUDO pacman -Sy --noconfirm "$pacman_pkg"
    elif command -v apk >/dev/null 2>&1; then
        log "Installing $pkg via apk..."
        $SUDO apk add --no-cache "$pkg"
    elif command -v zypper >/dev/null 2>&1; then
        log "Installing $pkg via zypper..."
        $SUDO zypper --non-interactive install "$pkg"
    elif command -v brew >/dev/null 2>&1; then
        log "Installing $pkg via Homebrew..."
        brew install "$pkg"
    else
        return 1
    fi
}

ensure_dependency() {
    # $1: human name, $2: command to check for, $3: apt/dnf/etc. package name, $4: pacman package
    # name (only python3's differs -- pacman's package is "python", everyone else's is "python3")
    local name="$1" cmd="$2" pkg="$3" pacman_pkg="${4:-$3}"
    if command -v "$cmd" >/dev/null 2>&1; then
        return 0
    fi
    log "$name not found; attempting to install it automatically"
    if install_pkg "$pkg" "$pacman_pkg" && command -v "$cmd" >/dev/null 2>&1; then
        log "$name installed successfully"
        return 0
    fi
    warn "Automatic $name install did not succeed (or no supported package manager was found)"
    return 1
}

if ! ensure_dependency "git" git git; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        die "git is required but not found and could not be installed automatically. Run 'xcode-select --install' (or 'brew install git' via https://brew.sh), then re-run this script."
    else
        die "git is required but not found and could not be installed automatically. Install it with your distro's package manager (e.g. 'sudo apt install git', 'sudo dnf install git'), then re-run this script."
    fi
fi

# NOTE: deliberately NOT --recurse-submodules. The only submodule is vendor/llama.cpp (~174MB),
# and it is needed only when this host has to COMPILE llama.cpp. Most hosts don't: they resolve to
# an upstream release asset or one of our own cached custom builds and never build anything. So the
# source is fetched lazily, at the one moment it is genuinely required, by
# source_build._init_vendor_submodule -- see that function for the full reasoning.
# On the update path, `submodule update` WITHOUT --init deliberately refreshes only submodules that
# are already populated, so a host that has previously source-built stays current while one that
# never has doesn't start downloading now.
if [[ -d "$SRC_DIR/.git" ]]; then
    log "Existing checkout found at $SRC_DIR -- updating"
    git -C "$SRC_DIR" fetch --depth 1 origin "$REF"
    git -C "$SRC_DIR" checkout "$REF"
    git -C "$SRC_DIR" reset --hard "origin/$REF"
    git -C "$SRC_DIR" submodule update --recursive
else
    log "Cloning $REPO_URL (ref: $REF) into $SRC_DIR"
    mkdir -p "$(dirname "$SRC_DIR")"
    git clone --branch "$REF" --depth 1 "$REPO_URL" "$SRC_DIR"
fi

PYTHON_BIN=""
find_python() {
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            return 0
        fi
    done
    return 1
}

if ! find_python; then
    ensure_dependency "python3" python3 python3 python || true
    find_python || true
fi

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        die "No Python 3.9+ found and it could not be installed automatically. Install it with 'brew install python3' (https://brew.sh) or from https://www.python.org/downloads/macos/, then re-run this script."
    else
        die "No Python 3.9+ found and it could not be installed automatically. Install it with your distro's package manager (e.g. 'sudo apt install python3', 'sudo dnf install python3'), then re-run this script."
    fi
fi

log "Using source checkout: $SRC_DIR"
log "Handing off to: $PYTHON_BIN -m aipotluck.installer.install $*"
cd "$SRC_DIR"
exec "$PYTHON_BIN" -m aipotluck.installer.install "$@"
