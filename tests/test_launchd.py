"""aipotluck.installer.service.launchd against a fake `launchctl` that keeps launchd's state in a
file and behaves the way the real one did on a Mac mini (macOS 27):

- a KeepAlive job that exits (or is `stop`ped) is relaunched with a new PID;
- `list <label>` exits 0 for any loaded job and prints a "PID" line only while it runs;
- `bootstrap` refuses a job that's already loaded; legacy `load` keeps the OLD definition;
- `bootout` takes the job out, so nothing relaunches it.

The first launchd backend passed for none of the contracts below; each test names one. Real
launchd is exercised on a real Mac, not here.
"""

from __future__ import annotations

import json
import os
import plistlib
import stat
import sys
from pathlib import Path

import pytest

from aipotluck.installer import layout as layout_mod, uninstall
from aipotluck.installer.platform_detect import HostProfile
from aipotluck.installer.service import launchd
from aipotluck.installer.service.base import ServiceState

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake launchctl is a POSIX script")

FAKE_LAUNCHCTL = r'''#!/usr/bin/env python3
import json, os, plistlib, sys
state_path = os.environ["FAKE_LAUNCHD_STATE"]
st = json.load(open(state_path)) if os.path.exists(state_path) else {"jobs": {}, "next_pid": 1000, "calls": [], "disabled": []}
a = sys.argv[1:]
st["calls"].append(a)
jobs = st["jobs"]
def save(): json.dump(st, open(state_path, "w"))
def new_pid():
    st["next_pid"] += 1
    return st["next_pid"]
def load(plist):
    d = plistlib.load(open(plist, "rb"))
    jobs[d["Label"]] = {"args": d["ProgramArguments"], "keepalive": d.get("KeepAlive", False),
                        "pid": new_pid() if d.get("RunAtLoad") else None}
cmd = a[0]
if cmd == "list":
    j = jobs.get(a[1])
    if j is None:
        print('Could not find service "%s" in domain for port' % a[1], file=sys.stderr); save(); sys.exit(113)
    print("{")
    print('\t"Label" = "%s";' % a[1])
    if j["pid"]:
        print('\t"PID" = %d;' % j["pid"])
    print('\t"LastExitStatus" = 0;')
    print("};")
elif cmd == "bootstrap":
    label = plistlib.load(open(a[2], "rb"))["Label"]
    if label in st.get("disabled", []):  # as real launchd: a disabled label won't bootstrap
        print("Bootstrap failed: 5: Input/output error", file=sys.stderr); save(); sys.exit(5)
    if label in jobs:
        print("Bootstrap failed: 5: Input/output error", file=sys.stderr); save(); sys.exit(5)
    load(a[2])
elif cmd == "bootout":
    label = a[1].split("/")[-1]
    if label not in jobs:
        print("Boot-out failed: 3: No such process", file=sys.stderr); save(); sys.exit(3)
    del jobs[label]
elif cmd == "kickstart":
    label = a[-1].split("/")[-1]
    if label not in jobs:
        save(); sys.exit(113)
    jobs[label]["pid"] = jobs[label]["pid"] or new_pid()
elif cmd == "stop":
    j = jobs.get(a[1])
    if j:  # the process ends; KeepAlive brings it straight back
        j["pid"] = new_pid() if j["keepalive"] else None
elif cmd == "start":
    j = jobs.get(a[1])
    if j:
        j["pid"] = j["pid"] or new_pid()
elif cmd == "load":
    label = plistlib.load(open(a[-1], "rb"))["Label"]
    if label in jobs:
        print("service already loaded", file=sys.stderr)  # and keeps the old definition
    else:
        load(a[-1])
elif cmd == "unload":
    label = plistlib.load(open(a[-1], "rb"))["Label"]
    jobs.pop(label, None)
    if "-w" in a:  # -w writes a persistent "disabled" override
        st.setdefault("disabled", []).append(label)
elif cmd == "enable":
    label = a[1].split("/")[-1]
    st["disabled"] = [l for l in st.get("disabled", []) if l != label]
else:
    print("unknown subcommand", cmd, file=sys.stderr); save(); sys.exit(64)
save()
'''

LABEL = "com.aipotluck.aipotluck"


@pytest.fixture
def launchd_env(tmp_path, monkeypatch):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    fake = bin_dir / "launchctl"
    fake.write_text(FAKE_LAUNCHCTL)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    state = tmp_path / "launchd.json"
    monkeypatch.setenv("FAKE_LAUNCHD_STATE", str(state))
    agents = tmp_path / "LaunchAgents"
    monkeypatch.setattr(launchd, "USER_AGENT_DIR", agents)

    def read():
        return json.loads(state.read_text()) if state.exists() else {"jobs": {}, "calls": []}

    return {"state": read, "agents": agents}


def install(mgr, script="/opt/aipotluck/v1/aipotluck_service.py"):
    mgr.install("aipotluck", Path("/usr/bin/python3"), [script], working_dir=Path("/opt/aipotluck"))


def test_install_loads_the_job_and_it_runs(launchd_env):
    mgr = launchd.LaunchdServiceManager()
    install(mgr)
    assert (launchd_env["agents"] / f"{LABEL}.plist").exists()
    assert mgr.status("aipotluck").state is ServiceState.RUNNING


def test_stop_leaves_no_process_running_despite_keepalive(launchd_env):
    """The bug found on the Mac mini: `launchctl stop` on a KeepAlive job is undone at once, so
    uninstall waited for a stop that never came ("still running after being asked to stop")."""
    mgr = launchd.LaunchdServiceManager()
    install(mgr)
    mgr.stop("aipotluck")
    assert launchd_env["state"]()["jobs"] == {}
    assert mgr.status("aipotluck").state is not ServiceState.RUNNING


def test_a_loaded_job_with_no_process_is_stopped_not_running(launchd_env, monkeypatch):
    """`launchctl list <label>` succeeds for any loaded job; only a PID means it's running."""
    mgr = launchd.LaunchdServiceManager()
    install(mgr)
    st = launchd_env["state"]()
    st["jobs"][LABEL]["pid"] = None
    Path(os.environ["FAKE_LAUNCHD_STATE"]).write_text(json.dumps(st))
    assert mgr.status("aipotluck").state is ServiceState.STOPPED


def test_status_is_not_installed_without_a_plist_or_job(launchd_env):
    assert launchd.LaunchdServiceManager().status("aipotluck").state is ServiceState.NOT_INSTALLED


def test_reinstalling_runs_the_new_definition_not_the_old_one(launchd_env):
    """An upgrade re-installs with a new interpreter/source path; the legacy `load` kept running
    the old one."""
    mgr = launchd.LaunchdServiceManager()
    install(mgr, "/opt/aipotluck/v1/aipotluck_service.py")
    install(mgr, "/opt/aipotluck/v2/aipotluck_service.py")
    job = launchd_env["state"]()["jobs"][LABEL]
    assert job["args"][1] == "/opt/aipotluck/v2/aipotluck_service.py"
    assert mgr.status("aipotluck").state is ServiceState.RUNNING


def test_install_works_where_the_old_backend_left_the_label_disabled(launchd_env):
    """Found on the Mac mini: the old uninstall's `unload -w` marks the label disabled for good, and
    launchd then refuses to bootstrap it ("Bootstrap failed: 5")."""
    Path(os.environ["FAKE_LAUNCHD_STATE"]).write_text(
        json.dumps({"jobs": {}, "next_pid": 1000, "calls": [], "disabled": [LABEL]}))
    mgr = launchd.LaunchdServiceManager()
    install(mgr)
    assert mgr.status("aipotluck").state is ServiceState.RUNNING


def test_start_after_stop_brings_it_back(launchd_env):
    mgr = launchd.LaunchdServiceManager()
    install(mgr)
    mgr.stop("aipotluck")
    mgr.start("aipotluck")
    assert mgr.status("aipotluck").state is ServiceState.RUNNING


def test_uninstall_removes_the_job_and_the_plist(launchd_env):
    mgr = launchd.LaunchdServiceManager()
    install(mgr)
    mgr.uninstall("aipotluck")
    assert launchd_env["state"]()["jobs"] == {}
    assert not (launchd_env["agents"] / f"{LABEL}.plist").exists()
    assert mgr.status("aipotluck").state is ServiceState.NOT_INSTALLED


def test_stop_and_uninstall_of_a_service_that_isnt_loaded_are_not_errors(launchd_env):
    mgr = launchd.LaunchdServiceManager()
    mgr.stop("aipotluck")
    mgr.uninstall("aipotluck")


def test_the_cli_uninstall_completes_on_macos(launchd_env, tmp_path, monkeypatch):
    """End to end through uninstall.uninstall with the real launchd backend: no "still running",
    and the job, its plist and the install are gone."""
    monkeypatch.setattr(uninstall.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(uninstall.cli_shim, "_bin_dir", lambda profile, system_scope: tmp_path / "bin")
    monkeypatch.setattr(uninstall, "_STOP_TIMEOUT_SECONDS", 2.0)
    lay = layout_mod.get_layout("macos", override_root=tmp_path / "aipotluck")
    layout_mod.ensure_layout_dirs(lay)
    (lay.config_dir / "runtime.json").write_text("{}")
    mgr = launchd.LaunchdServiceManager()
    install(mgr)

    report = uninstall.uninstall(HostProfile("macos", "arm64", "metal"), lay, mgr)

    assert report.service_removed and report.warnings == []
    assert launchd_env["state"]()["jobs"] == {}
    assert not (launchd_env["agents"] / f"{LABEL}.plist").exists()
    assert not lay.config_dir.exists()
