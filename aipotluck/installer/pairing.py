"""One-click pairing: `aipotluck-local-client login` with no credential flags.

The alternative to copying the pairing JSON out of Settings by hand. This CLI makes up a random
pairing id, opens aipotluck.org's pairing page for it in a browser, and polls aipotluck.org until
the person's signed-in visit to that page has provisioned a managed endpoint. The server side
(aipotluck.org) is two routes:

    GET <server>/chat/local-inference/pair?id=<pairing id>&name=<device name>   (the browser)
    GET <server>/chat/api/v2/local-inference/pairings/<pairing id>              (this module)

The poll answers `{}` until the visit has provisioned, then exactly once
`{"id", "label", "installArgs": {"tunnelId", "tunnelSecret", "tunnelEndpoint"}}`, the same
`installArgs` the Settings page's Copy button copies, then `{}` again. The server clears the
credentials as it hands them over, so a poll that loses its response loses the pairing: the person
removes the half-paired server in Settings and runs `login` again.

The pairing id is the poll's only credential (no session, no cookie), which is why it comes from
`secrets` and why the server refuses one shorter than 32 characters. It never leaves this process
except in the pairing URL and the poll URL.

The browser need not run on this machine: the URL is always printed, so on a headless box (an SSH
session, a Jetson with no display) the person opens it on any device where they're signed in.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Callable, Optional

log = logging.getLogger("aipotluck.pairing")

DEFAULT_SERVER_URL = "https://www.aipotluck.org"
SERVER_URL_ENV = "AIPOTLUCK_SERVER_URL"

# How long `login` waits for the person to approve in the browser. The server keeps a claimed
# pairing for 10 minutes, so waiting longer than that could only ever collect nothing.
DEFAULT_TIMEOUT_SECONDS = 10 * 60
POLL_INTERVAL_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 10.0

# Plain http is accepted only for a dev server on this machine: the poll's one real answer carries
# the tunnel secret, and anywhere else it would cross the network in the clear.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class PairingError(Exception):
    """Pairing cannot finish. The message is written to be shown to the person as-is."""


def resolve_server_url(cli_value: Optional[str]) -> str:
    """The server to pair with: `--server-url`, else $AIPOTLUCK_SERVER_URL, else aipotluck.org.
    Normalized to scheme://host[:port][/path] with no trailing slash. Raises PairingError for a
    value that isn't an http(s) URL, or that is plain http to anything but this machine."""
    raw = (cli_value or os.environ.get(SERVER_URL_ENV) or DEFAULT_SERVER_URL).strip()
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise PairingError(f"--server-url must be an http(s) URL such as {DEFAULT_SERVER_URL} (got {raw!r}).")
    if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK_HOSTS:
        raise PairingError(
            f"Refusing plain http for {parsed.hostname}: the tunnel secret would cross the network "
            "unencrypted. Use https, or http only for a dev server on this machine (localhost)."
        )
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def new_pairing_id() -> str:
    """43 URL-safe characters (256 bits) from the OS CSPRNG: inside the server's 32-128
    `[A-Za-z0-9_-]` rule, and far past guessable, since knowing it is all the poll asks for."""
    return secrets.token_urlsafe(32)


def device_name() -> str:
    """This machine's hostname, which the server stores as the device's name in Settings. Only the
    first label (`nx16g.local` -> `nx16g`): the domain part is network noise, not a name anyone
    picked. Empty when the hostname can't be read; the server then names the device itself."""
    try:
        name = socket.gethostname()
    except OSError:
        return ""
    return name.split(".")[0].strip()


def pairing_url(server_url: str, pairing_id: str, name: str) -> str:
    query = {"id": pairing_id}
    if name:
        query["name"] = name
    return f"{server_url}/chat/local-inference/pair?{urllib.parse.urlencode(query)}"


def poll_url(server_url: str, pairing_id: str) -> str:
    return f"{server_url}/chat/api/v2/local-inference/pairings/{urllib.parse.quote(pairing_id, safe='')}"


def can_open_browser() -> bool:
    """Whether webbrowser.open could show a GUI browser here. On Linux and the BSDs, with no
    $DISPLAY or $WAYLAND_DISPLAY (an SSH session, a headless board), Python's webbrowser falls back
    to a console browser such as lynx or w3m, which would take over this terminal mid-login, so
    the answer there is no and the printed URL is the way through. macOS and Windows always have a
    desktop browser to hand off to."""
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def open_in_browser(url: str) -> bool:
    """Best-effort: True when a browser was asked to open `url`. Never raises."""
    if not can_open_browser():
        return False
    try:
        return bool(webbrowser.open(url, new=2))
    except Exception as exc:  # webbrowser can raise from a misconfigured $BROWSER, among others
        log.debug("Could not open a browser: %s", exc)
        return False


def _fetch_json(url: str) -> object:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "aipotluck-local-client"}
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_install_args(
    server_url: str,
    pairing_id: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    interval_seconds: float = POLL_INTERVAL_SECONDS,
    fetch: Callable[[str], object] = _fetch_json,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Polls until the server hands over this pairing's `installArgs` dict, and returns it
    unvalidated (cli.py checks its keys the same way it checks a credentials file).

    Fails fast (PairingError) on an answer that will never improve: a 4xx (400 is a pairing id the
    server rejects; 404 is a server without one-click pairing, or with it switched off), or a 200
    that isn't the expected JSON. Keeps polling through what can: a 5xx, a timeout, a dropped
    connection. Raises PairingError when `timeout_seconds` passes with no credentials."""
    url = poll_url(server_url, pairing_id)
    deadline = clock() + timeout_seconds
    warned_unreachable = False
    while True:
        try:
            body = fetch(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise PairingError(
                    f"{server_url} doesn't offer one-click pairing (HTTP 404): it may predate it, or "
                    "have it switched off. Pair manually instead: copy the JSON from Settings -> "
                    "Local Inference and run `login --credentials-file <file>`."
                ) from exc
            if 400 <= exc.code < 500:
                raise PairingError(f"{server_url} refused the pairing poll (HTTP {exc.code}).") from exc
            log.debug("Pairing poll got HTTP %s; retrying.", exc.code)
        except (urllib.error.URLError, OSError) as exc:
            # URLError is an OSError; this also covers socket timeouts and dropped connections.
            if not warned_unreachable:
                log.warning("Can't reach %s right now (%s); still trying.", server_url, exc)
                warned_unreachable = True
        except ValueError as exc:  # json.JSONDecodeError and a bad UTF-8 body alike
            raise PairingError(f"{server_url} answered the pairing poll with something other than JSON.") from exc
        else:
            if not isinstance(body, dict):
                raise PairingError(f"{server_url} answered the pairing poll with unexpected JSON.")
            if body:
                install_args = body.get("installArgs")
                if not isinstance(install_args, dict):
                    raise PairingError(f"{server_url} answered the pairing poll without installArgs.")
                if body.get("label"):
                    log.info("Paired as %r.", body["label"])
                return install_args

        if clock() >= deadline:
            raise PairingError(
                f"Timed out after {int(timeout_seconds // 60)} minutes waiting for the pairing to be "
                "approved in the browser. Run `login` again to get a new link."
            )
        sleep(interval_seconds)
