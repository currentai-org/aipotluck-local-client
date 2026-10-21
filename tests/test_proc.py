"""aipotluck.proc.NO_WINDOW: under pythonw.exe (the Windows logon task), the console binaries the
service starts must not each pop a visible console window.

Real processes, no mocks: a pythonw.exe parent starts a console child with and without NO_WINDOW,
and the child reports whether it has a console window (kernel32.GetConsoleWindow). Without the flag
the child gets a fresh window -- the failure this exists to prevent -- which the second test shows.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from aipotluck.proc import NO_WINDOW

REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows console windows")


def _child_has_a_window(tmp_path: Path, use_flag: bool) -> bool:
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        pytest.skip("no pythonw.exe beside this interpreter")
    out = tmp_path / "result.txt"
    parent = tmp_path / "parent.py"
    parent.write_text(textwrap.dedent(f"""
        import subprocess, sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from aipotluck.proc import NO_WINDOW
        child = [{str(Path(sys.executable))!r}, "-c",
                 "import ctypes; print(bool(ctypes.windll.kernel32.GetConsoleWindow()))"]
        kw = NO_WINDOW if {use_flag!r} else {{}}
        r = subprocess.run(child, capture_output=True, text=True, **kw)
        open({str(out)!r}, "w").write(r.stdout.strip())
    """))
    subprocess.run([str(pythonw), str(parent)], timeout=60, check=True)
    return out.read_text() == "True"


def test_a_console_binary_started_from_pythonw_gets_no_window(tmp_path):
    assert NO_WINDOW, "NO_WINDOW must carry CREATE_NO_WINDOW on Windows"
    assert not _child_has_a_window(tmp_path, use_flag=True)


def test_without_it_the_same_child_pops_a_console_window(tmp_path):
    """The failure shown real, so the test above can't pass by accident."""
    assert _child_has_a_window(tmp_path, use_flag=False)
