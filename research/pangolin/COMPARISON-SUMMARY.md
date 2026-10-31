# Implementation Scenario Summary — chiSSL vs Pangolin vs wstunnel vs sish

## Scenario recap

Each tool deployed as a managed service on a cloud VPS (systemd/Docker),
reachable from the SvelteKit backend over some API/control-plane. The
client library runs inside aipotluck-local-client's monitor service. During
install, the user links their local client to SvelteKit by exchanging an
access token; SvelteKit then configures the reverse-proxy service to accept
that client's connection and starts routing requests to it. Each tool was
assessed from its source and docs; the Pangolin plan is in [PLAN.md](PLAN.md).

## At a glance

| | chiSSL | Pangolin | wstunnel | sish |
|---|---|---|---|---|
| **Install path** | Shell installer script → systemd; **no Docker image exists** | Official installer → Docker Compose (pangolin+gerbil+traefik) | Static binary download; Docker image also published | Docker image (official) or build-from-source; Compose deploy folder in repo |
| **Global config** | `--tls-domain` (auto Let's Encrypt), `--auth admin:pass` at startup = admin identity, SQLite auto-created | Interactive installer: domain, LE email, admin account via one-time setup token, then create first Org | TLS cert files (self-supplied — wstunnel does *not* auto-renew), one static `restrict-config` YAML | Wildcard DNS + `--domain`, SSH host-key auto-gen, choice of 4 pluggable auth methods |
| **Real REST/control-plane API?** | **Yes** — OpenAPI-documented (`/api/users`, `/api/listeners`, `/api/tunnels`, `/api/sessions`, per-user tokens) | **Yes** — "Integration API" (Bearer org-scoped keys), full site→resource→target CRUD, off by default on self-hosted | **No** — zero network API of any kind | **Partial** — read-only client list + force-disconnect only; **no write/authorize endpoint** |
| **Client link/pairing model** | Static username:password created via `POST /api/users`; no token exchange, no pairing flow | **ID+secret pair** minted server-side via API, exchanged over HTTP for a session token, then WS control channel | Nothing built-in — 100% custom (bearer token regex-matched at WS upgrade) | Nothing built-in — bare SSH pubkey auth; author's own doc suggests manually curling a GitHub `.keys` URL |
| **Request routing model** | **Port-based** — each tunnel claims a distinct server TCP port; no subdomain/Host dispatch found in source | **Explicit object graph** — site → public-resource (hostname) → target (ip:port); exactly "bind route to client" | **Port-based**, static `-R` mapping; no HTTP multiplexing at all | **Subdomain-based**, per-connection, Host-header dispatch built in — the most "free" routing of the four |
| **Status/health polling** | `GET /api/tunnels/{id}` (`status` field), `/api/tunnels/active`, `/api/stats`, `/health` | `GET /org/{orgId}/site/{niceId}` → `online` boolean; plus target-level HTTP/TCP health checks with failover | **None** — must build from scratch (`ss`/journalctl, or HTTP health-check through the tunnel) | `GET /_sish/api/clients` (poll-only, no push), matched by pubkey fingerprint |
| **Custom glue code required** | Moderate — DB/token model exists, but *routing* (port↔client mapping) is 100% ours | Low — the object model (site/resource/target) already matches our scenario almost exactly | **Very high** — device DB, token minting, config templating+distribution, routing proxy, health polling: all ours | High — pairing/token UX, "authorize a key" write-path, subdomain convention: all ours; routing/health-read is free |

## Key finding: only two of four ship a real, writable control-plane API

**Pangolin** and **chiSSL** both expose genuine REST APIs we can call from
SvelteKit to register/manage tunnels programmatically — the core of the
scenario's requirement. **wstunnel** and **sish** have *no* API surface for
the "write" side of the scenario (issuing tokens, authorizing new clients,
binding routes) at all; sish at least offers a read-only admin API and a
synchronous authorization callback hook, while wstunnel offers literally
nothing beyond CLI flags and two hot-reloaded config files.

On client leanness and tool maturity alone, wstunnel and sish look like the
strongest options. Once you actually design the concrete "SvelteKit registers a
tunnel, links a device, and polls status" workflow, the *lack of a control-
plane API* in wstunnel/sish surfaces as a large, unavoidable amount of
custom backend work — a device registry, token issuer, config-file
templating and safe distribution to the VPS, a reverse-proxy/routing layer,
and a polling system, all built from scratch. Both remain viable, but only
by building roughly what Pangolin or chiSSL already ship.

## Routing is the sharpest differentiator

- **Pangolin** is the only tool whose native data model — site, resource,
  target — already *is* "bind a public hostname to a specific client's
  local port," with no additional reverse-proxy layer needed on our side.
- **sish** is close behind: subdomain-per-connection routing is automatic
  and free once a tunnel is up; we just need a fixed-subdomain convention
  per device (easy to enforce via `--force-requested-subdomains`).
- **chiSSL** and **wstunnel** both only allocate raw TCP *ports*, not
  hostnames — for either of these, we would need to bolt on our own
  Host/SNI-based reverse proxy (e.g. Caddy/Traefik) in front of a
  device→port mapping table we maintain ourselves, adding an entire extra
  infrastructure layer neither tool provides.

## Client linking: none of the four match the scenario's "token exchange" model out of the box

- **chiSSL**: static username:password, no token/pairing flow at all —
  closest to a "shared secret," further from a one-time exchange.
  Provisioning still requires an admin (or our backend, acting as admin)
  calling `POST /api/users` per device.
- **Pangolin**: closest fit — server-generated ID+secret pair exchanged
  over HTTP for a session token is architecturally identical to what an
  access-token exchange looks like, and it even has a fleet-scale
  provisioning-key mechanism (`spk_...`) for baking into a golden image.
- **wstunnel / sish**: no linking flow exists in either tool; the entire
  token-exchange UX described in the scenario has to be designed and built
  by us from nothing, using each tool's narrow auth primitive (a regex-
  matched bearer header for wstunnel, a synchronous pubkey-authorization
  callback for sish) as the enforcement point underneath our own database.

## Recommendation for this specific scenario

**Pangolin is the strongest fit for the exact workflow described** — it is
the only tool where "SvelteKit calls an API to register a device, gets back
an ID+secret to hand to the local client, the client links up, and
SvelteKit binds a route + polls status" is close to a first-class supported
flow rather than something we construct on top of low-level primitives. The
tradeoffs are real: it's the heaviest deployment (Docker Compose with three
containers, WireGuard kernel capabilities `NET_ADMIN`/`SYS_MODULE` on the
Gerbil component — though the *client* newt itself is TUN-free), and some of its API surface (exact GET/list routes, the precise
mechanism flipping a site's `online` flag) wasn't fully traceable through
static reading alone and would need a hands-on spike to nail down before
committing.

**chiSSL is the leaner fallback** if Pangolin's operational footprint proves
too heavy — it has a real, simpler REST API and a much smaller
single-binary deployment, at the cost of having to build our own routing
layer (it has no hostname-based dispatch) and a somewhat unusual
static-password client auth model in place of a token exchange.

**wstunnel and sish would be a materially larger build** — both are excellent,
mature *transport* layers, but neither ships the control-plane, device
registry, or routing logic this scenario needs, so choosing either means
building most of what Pangolin/chiSSL already provide, on top of a
transport that is arguably not meaningfully better than what the API-having
options already use underneath (chiSSL is literally a chisel/SSH-over-WS
fork; Pangolin uses WireGuard). If client-side leanness or protocol maturity
turns out to matter more than we're currently weighting it, sish is the
better of the two remaining fallbacks (native subdomain routing, a
read-only status API, and a synchronous auth hook) versus wstunnel (zero
API surface of any kind).
