"""aipotluck.installer.newt_fetch against the REAL newt_version.json shipped in this repo.

`cli login` refuses a newt pin entry whose sha256 is still the UNVERIFIED placeholder, so a
platform left on the placeholder can install and run the service but can never be paired. These
tests read the real manifest (no fixture copy), so a release that ships a placeholder for a
supported platform goes red here rather than on a user's machine.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aipotluck.installer import newt_fetch
from aipotluck.installer.platform_detect import HostProfile

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = newt_fetch.load_newt_manifest(REPO_ROOT / "newt_version.json")


@pytest.mark.parametrize(
    "os_name,arch,file",
    [
        ("linux", "x64", "newt_linux_amd64"),
        ("linux", "arm64", "newt_linux_arm64"),
        ("macos", "arm64", "newt_darwin_arm64"),
        ("macos", "x64", "newt_darwin_amd64"),
    ],
)
def test_every_supported_platform_resolves_to_a_verified_newt_so_login_can_pair(os_name, arch, file):
    key, entry = newt_fetch.resolve_newt_asset(MANIFEST, HostProfile(os_name=os_name, arch=arch, backend="cpu"))
    assert key == f"{os_name}-{arch}"
    assert entry["file"] == file
    assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["sha256"]


def test_a_placeholder_checksum_is_refused_rather_than_trusted():
    """The fail-closed half of the contract: an unverified entry must stop login, never download
    an unchecked binary. Windows is still on the placeholder at this pin."""
    with pytest.raises(newt_fetch.NewtVersionManifestError, match="never been downloaded/verified"):
        newt_fetch.resolve_newt_asset(MANIFEST, HostProfile(os_name="windows", arch="x64", backend="cpu"))
