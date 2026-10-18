"""aipotluck.installer.pairing: one-click pairing's URL building and its poll loop.

The poll runs against a real local HTTP server (conftest.py's PairingServerStub) playing
aipotluck.org's poll route, so urllib, HTTP status handling and JSON parsing are all real. Only
time is faked: `sleep` and `clock` are injected, so a 10-minute timeout takes milliseconds.
"""

from __future__ import annotations

import re
import socket

import pytest

from aipotluck.installer import pairing

CREDENTIALS = {
    "id": "endpoint-1",
    "label": "nx16g",
    "installArgs": {"tunnelId": "tid", "tunnelSecret": "tsecret", "tunnelEndpoint": "https://tunnel.example"},
}


class FakeClock:
    """A monotonic clock that only moves when the poll loop sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def wait(server_url: str, clock: FakeClock, **kwargs) -> dict:
    return pairing.wait_for_install_args(
        server_url, "p" * 43, sleep=clock.sleep, clock=clock.clock, **kwargs
    )


class TestResolveServerUrl:
    def test_defaults_to_aipotluck_org(self, monkeypatch):
        monkeypatch.delenv(pairing.SERVER_URL_ENV, raising=False)
        assert pairing.resolve_server_url(None) == "https://www.aipotluck.org"

    def test_env_var_is_used_when_no_flag_is_given(self, monkeypatch):
        monkeypatch.setenv(pairing.SERVER_URL_ENV, "https://develop.example.org/")
        assert pairing.resolve_server_url(None) == "https://develop.example.org"

    def test_flag_wins_over_env_var(self, monkeypatch):
        monkeypatch.setenv(pairing.SERVER_URL_ENV, "https://develop.example.org")
        assert pairing.resolve_server_url("https://other.example.org") == "https://other.example.org"

    def test_plain_http_is_accepted_for_a_dev_server_on_this_machine(self):
        assert pairing.resolve_server_url("http://localhost:5199/") == "http://localhost:5199"
        assert pairing.resolve_server_url("http://127.0.0.1:5199") == "http://127.0.0.1:5199"

    def test_refuses_plain_http_to_any_other_host_since_the_secret_would_travel_in_the_clear(self):
        with pytest.raises(pairing.PairingError, match="unencrypted"):
            pairing.resolve_server_url("http://www.aipotluck.org")

    @pytest.mark.parametrize("value", ["www.aipotluck.org", "ftp://x.org", "https://", "not a url"])
    def test_refuses_anything_that_is_not_an_http_url(self, value):
        with pytest.raises(pairing.PairingError):
            pairing.resolve_server_url(value)


class TestPairingIdAndUrls:
    def test_pairing_id_satisfies_the_servers_32_to_128_url_safe_rule(self):
        for _ in range(50):
            assert re.fullmatch(r"[A-Za-z0-9_-]{32,128}", pairing.new_pairing_id())

    def test_pairing_ids_do_not_repeat(self):
        assert len({pairing.new_pairing_id() for _ in range(200)}) == 200

    def test_pairing_url_carries_the_id_and_the_device_name(self):
        url = pairing.pairing_url("https://www.aipotluck.org", "abc_DEF-123", "my desk")
        assert url == "https://www.aipotluck.org/chat/local-inference/pair?id=abc_DEF-123&name=my+desk"

    def test_pairing_url_leaves_out_an_empty_name_so_the_server_picks_one(self):
        assert "name=" not in pairing.pairing_url("https://x.org", "abc", "")

    def test_device_name_is_the_hostnames_first_label(self, monkeypatch):
        monkeypatch.setattr(socket, "gethostname", lambda: "nx16g.local")
        assert pairing.device_name() == "nx16g"

    def test_device_name_is_empty_when_the_hostname_cannot_be_read(self, monkeypatch):
        def _fail():
            raise OSError("no hostname")

        monkeypatch.setattr(socket, "gethostname", _fail)
        assert pairing.device_name() == ""


class TestOpenInBrowser:
    def test_skips_the_browser_on_a_headless_linux_box_rather_than_launching_a_console_browser(
        self, monkeypatch
    ):
        monkeypatch.setattr(pairing.sys, "platform", "linux")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        opened = []
        monkeypatch.setattr(pairing.webbrowser, "open", lambda *a, **kw: opened.append(a) or True)

        assert pairing.open_in_browser("https://x.org") is False
        assert opened == []

    def test_opens_the_browser_when_linux_has_a_display(self, monkeypatch):
        monkeypatch.setattr(pairing.sys, "platform", "linux")
        monkeypatch.setenv("DISPLAY", ":0")
        opened = []
        monkeypatch.setattr(pairing.webbrowser, "open", lambda url, new=0: opened.append(url) or True)

        assert pairing.open_in_browser("https://x.org/pair") is True
        assert opened == ["https://x.org/pair"]

    def test_a_browser_that_raises_reads_as_not_opened_never_an_exception(self, monkeypatch):
        monkeypatch.setattr(pairing.sys, "platform", "darwin")

        def _boom(*_a, **_kw):
            raise RuntimeError("bad $BROWSER")

        monkeypatch.setattr(pairing.webbrowser, "open", _boom)
        assert pairing.open_in_browser("https://x.org") is False


class TestWaitForInstallArgs:
    def test_polls_through_empty_answers_until_the_credentials_arrive(self, pairing_server):
        clock = FakeClock()
        with pairing_server([(200, {}), (200, {}), (200, CREDENTIALS)]) as server:
            install_args = wait(server.base_url, clock)

        assert install_args == CREDENTIALS["installArgs"]
        assert server.requests == ["/chat/api/v2/local-inference/pairings/" + "p" * 43] * 3
        assert clock.sleeps == [pairing.POLL_INTERVAL_SECONDS] * 2

    def test_keeps_polling_through_a_server_error(self, pairing_server):
        with pairing_server([(503, {"message": "busy"}), (200, CREDENTIALS)]) as server:
            assert wait(server.base_url, FakeClock()) == CREDENTIALS["installArgs"]
        assert len(server.requests) == 2

    def test_fails_fast_on_404_pointing_at_manual_pairing(self, pairing_server):
        with pairing_server([(404, {"message": "Not found"})]) as server:
            with pytest.raises(pairing.PairingError, match="credentials-file"):
                wait(server.base_url, FakeClock())
        assert len(server.requests) == 1

    def test_fails_fast_on_400_a_pairing_id_the_server_rejects(self, pairing_server):
        with pairing_server([(400, {"message": "Pairing id must be 32-128 URL-safe characters"})]) as server:
            with pytest.raises(pairing.PairingError, match="HTTP 400"):
                wait(server.base_url, FakeClock())
        assert len(server.requests) == 1

    def test_gives_up_with_an_error_not_credentials_when_nobody_approves_before_the_timeout(
        self, pairing_server
    ):
        clock = FakeClock()
        with pairing_server([(200, {})]) as server:
            with pytest.raises(pairing.PairingError, match="Timed out"):
                wait(server.base_url, clock, timeout_seconds=10, interval_seconds=2)
        assert clock.now >= 10
        assert len(server.requests) == 6  # t = 0, 2, 4, 6, 8, 10

    def test_an_unreachable_server_is_retried_until_the_timeout_then_fails(self):
        # Port 1 refuses immediately: a server that is down, not one that answered.
        clock = FakeClock()
        with pytest.raises(pairing.PairingError, match="Timed out"):
            wait("http://127.0.0.1:1", clock, timeout_seconds=6, interval_seconds=2)
        assert len(clock.sleeps) == 3

    def test_a_non_json_answer_is_an_error_not_a_retry(self, pairing_server):
        # e.g. a login wall or a deployment-protection page answering 200 with HTML.
        with pairing_server([(200, "<html>Sign in</html>")]) as server:
            with pytest.raises(pairing.PairingError, match="other than JSON"):
                wait(server.base_url, FakeClock())

    def test_a_non_empty_answer_without_install_args_is_an_error(self, pairing_server):
        with pairing_server([(200, {"id": "x", "label": "y"})]) as server:
            with pytest.raises(pairing.PairingError, match="installArgs"):
                wait(server.base_url, FakeClock())
