# Pangolin Implementation Plan — aipotluck-local-client

Sources: the `fosrl/pangolin` and `fosrl/newt` source trees, and https://docs.pangolin.net, as of
September 2026. File paths below are relative to those repos unless a URL is given.

---

## Installation & Global VPS Configuration

Pangolin explicitly supports a Docker Compose self-hosted deployment, driven by an official
installer script (docs: https://docs.pangolin.net/self-host/quick-install).

**Prerequisites** (from quick-install doc):
- Linux server, root access, public IP
- A domain name pointed at the server (for the dashboard)
- Email address (Let's Encrypt + admin login)
- Firewall ports open: **80/TCP, 443/TCP, 51820/UDP, 21820/UDP** (the last two are WireGuard
  data-plane / relay ports used by the "Gerbil" exit-node component and clients)

**Install steps (quoted from docs.pangolin.net/self-host/quick-install):**
```
curl -fsSL https://static.pangolin.net/get-installer.sh | bash
sudo ./installer
```
The interactive installer asks for: Edition (Community/Enterprise), Base Domain, Dashboard
Domain (default `pangolin.example.com`), Let's Encrypt email, whether to install "Gerbil" for
tunneling (default yes — without it Pangolin runs as a plain reverse proxy with no tunnel
capability), and optional SMTP config. It then pulls Docker images (`pangolin`, `gerbil`,
`traefik`) and starts the containers.

**Actual docker-compose config** the installer generates
(`install/config/docker-compose.yml`, template-rendered):
```yaml
name: pangolin
services:
  pangolin:
    image: docker.io/fosrl/pangolin:{{...}}{{.PangolinVersion}}
    container_name: pangolin
    restart: unless-stopped
    volumes:
      - ./config:/app/config
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3001/api/v1/"]

  gerbil:
    image: docker.io/fosrl/gerbil:{{.GerbilVersion}}
    container_name: gerbil
    depends_on:
      pangolin: {condition: service_healthy}
    command:
      - --reachableAt=http://gerbil:3004
      - --generateAndSaveKeyTo=/var/config/key
      - --remoteConfig=http://pangolin:3001/api/v1/
    cap_add: [NET_ADMIN, SYS_MODULE]
    ports:
      - 51820:51820/udp
      - 21820:21820/udp
      - 443:443
      - 443:443/udp   # HTTP/3 QUIC
      - 80:80

  traefik:
    image: docker.io/traefik:v3.7
    network_mode: service:gerbil   # ports inherited from gerbil
    command: [--configFile=/etc/traefik/traefik_config.yml]
    volumes:
      - ./config/traefik:/etc/traefik:ro
      - ./config/letsencrypt:/letsencrypt

  # optional: postgres, redis (Postgres/Redis are opt-in; SQLite is the default DB)
```
So the exposed ports are **80, 443 (TCP+UDP/QUIC), 51820/udp (WireGuard exit-node), 21820/udp
(client relay)**. Pangolin's own API/dashboard listens internally on **3001** (health check
`http://localhost:3001/api/v1/`); Traefik fronts it externally on 80/443.

**Required config file**: `config/config.yml`, example at
`config/config.example.yml`:
```yaml
gerbil:
    start_port: 51820
    base_endpoint: "{{.DashboardDomain}}"
app:
    dashboard_url: "https://{{.DashboardDomain}}"
    log_level: "info"
domains:
    domain1:
        base_domain: "{{.BaseDomain}}"
server:
    secret: "{{.Secret}}"
    cors:
        origins: ["https://{{.DashboardDomain}}"]
flags:
    require_email_verification: false
    disable_signup_without_invite: true
    disable_user_create_org: false
    allow_raw_resources: true
```
`server.secret` is a session/signing secret; `domains.*.base_domain` establishes the DNS domain(s)
resources can be published under.

**Global one-time setup after containers are up** (quoted from
https://docs.pangolin.net/self-host/quick-install "Post-Installation Setup"):
1. Visit `https://<your-dashboard-domain>/auth/initial-setup`.
2. Retrieve the one-time setup token with `sudo docker compose logs pangolin`.
3. Paste the token, set admin email + password → creates the first admin account.
4. Create the first **Organization** (name + description) — required before any site/resource
   can exist.

Only after an org exists can sites (newt/tunnel connectors), resources, and clients be created.

---

## Cloud-Side Control-Plane API (SvelteKit ↔ Pangolin server)

Pangolin does expose a documented REST API, called the **"Integration API"**:
- Doc: https://docs.pangolin.net/manage/integration-api and
  https://docs.pangolin.net/manage/common-api-routes
- Full interactive Swagger/OpenAPI UI, generated straight from the TypeScript route
  definitions (`server/openApi.ts` `registry.registerPath({...})` calls seen throughout
  `server/routers/**`, e.g. `server/routers/site/getSite.ts` lines 56‑112) is published at
  **https://api.pangolin.net/v1/docs** for Pangolin Cloud, and at
  `https://api.<your-domain>/v1/docs` for self-hosted.
- **On self-hosted instances the Integration API is disabled by default** and must be turned on
  (`docs/self-host/advanced/integration-api`):
  ```yaml
  # config.yml
  flags:
    enable_integration_api: true
  server:
    integration_port: 3003   # default port
  ```
  and Traefik must be given a router/service to expose it (example dynamic_config.yml router
  pointing at `http://pangolin:3003`, quoted verbatim in the doc) at `https://api.example.com/v1`.

**Auth**: Bearer token in `Authorization` header. Two key types
(docs.pangolin.net/manage/integration-api "API Key Types"):
- **Organization API keys** — created by an org admin, scoped to that org only. This is what our
  SvelteKit backend should use.
- **Root API keys** — self-hosted only, cross-org, higher privilege; not needed for our use case.

**Concrete endpoints our backend would call** (from
https://docs.pangolin.net/manage/common-api-routes, all under `/v1` prefix, `orgId` from
dashboard/API):
- `GET /org/{orgId}/pick-site-defaults` → generates a fresh `newtId` + `newtSecret` +
  `clientAddress` for a new site.
- `PUT /org/{orgId}/site` (body `{name, type:"newt", address, newtId, secret}`) → creates the
  site record; response includes `siteId`, `niceId`, `online:false`, `newtId`, `secret`.
- `GET /org/{orgId}/domains` → list verified domains (`domainId`) to publish resources under.
- `PUT /org/{orgId}/public-resource` (body `{name, http:true, domainId, protocol:"tcp",
  subdomain}`) → creates a public HTTP resource, returns `resourceId`, `fullDomain`.
- `PUT /public-resource/{resourceId}/target` (body `{siteId, ip, port, method:"http"}`) → binds
  a backend (local service behind a specific site/newt) to the public resource.
- `POST /public-resource/{resourceId}/roles/add` / `users/add` → access control.
- `GET /org/{orgId}/site/{niceId}` or `GET /site/{siteId}` → fetch a site's current state
  (see status polling section).

This is exactly the create-site → create-resource → create-target pattern our SvelteKit backend
needs to programmatically provision a tunnel + route per customer/device.

---

## Client Linking Flow (access token exchange)

Newt's own README (`newt/README.md` lines 19‑21) states plainly:

> **Registers with Pangolin** — Using the Newt ID and a secret, the client will make HTTP
> requests to Pangolin to receive a session token. Using that token, it will connect to a
> websocket and maintain that connection. Control messages will be sent over the websocket.

So the flow is: **ID+secret pair generated server-side ahead of time** (either via the dashboard
"create site" UI, or via the API calls above — `pick-site-defaults` + `PUT /org/{orgId}/site`
return `newtId`/`secret`, later called just `id`/`secret` in the CLI) → newt exchanges them over
HTTP for a session token → opens a persistent websocket to the Pangolin server → server pushes
WireGuard control messages (peer public key, endpoint) over that socket → newt brings up a
userspace WireGuard tunnel via netstack (no TUN device/root) and pings the Gerbil-side peer to
confirm it's live (`newt/README.md` "Receives WireGuard Control Messages").

**Actual invocation, quoted from https://docs.pangolin.net/manage/sites/install-newt:**
```bash
newt \
--id 31frd0uzbjvp721 \
--secret h51mmlknrvrwv8s4r1i210azhumt6isgbpyavxodibx1k2d6 \
--endpoint https://app.pangolin.net
```
Docker run form:
```bash
docker run -it fosrl/newt --id 31frd0uzbjvp721 \
  --secret h51mmlknrvrwv8s4r1i210azhumt6isgbpyavxodibx1k2d6 \
  --endpoint https://app.pangolin.net
```
Docker Compose (env-var form, recommended in docs):
```yaml
services:
  newt:
    image: fosrl/newt
    environment:
      - PANGOLIN_ENDPOINT=https://app.pangolin.net
      - NEWT_ID=2ix2t8xk22ubpfy
      - NEWT_SECRET=nnisrfsdfc7prqsp9ewo1dvtvci50j5uiqotez00dgap0ii2
```
Or a JSON config file (`newt-config.secret`, can be injected as a Docker Compose secret):
```json
{
  "id": "2ix2t8xk22ubpfy",
  "secret": "nnisrfsdfc7prqsp9ewo1dvtvci50j5uiqotez00dgap0ii2",
  "endpoint": "https://app.pangolin.net",
  "tlsClientCert": ""
}
```

**At scale**, Pangolin also supports **provisioning keys** (long-lived `spk_...` tokens embedded
in a golden image/script) that a fleet of newt clients exchange for their own unique
id/secret on first boot (docs.pangolin.net/manage/sites/site-provisioning):
```bash
pangolin up site --config-file /var/site.json --endpoint https://app.pangolin.net \
  --provisioning-key 'spk_...'
```
This may be directly relevant to aipotluck-local-client if we want to bake a provisioning key
into a distributed installer instead of minting an id/secret pair per install via the API — worth
evaluating instead of the plain per-device `PUT /org/{orgId}/site` flow above. Provisioned sites
can optionally land in a "pending" review queue for admin approval before going live.

Note: the docs currently describe newt as being **phased out in favor of "Pangolin CLI"**
(`fosrl/cli`, invoked as `pangolin site up`) — see `newt/README.md` line 6-7 banner and
docs.pangolin.net/manage/sites/understanding-sites ("In engineering contexts... sometimes
referred to as Newt. New installs should use the Pangolin CLI. Newt remains available as a
standalone connector"). Functionally identical id/secret/endpoint flow either way; newt is
still fully supported and is the smaller footprint of the two, which likely matters for us given
we're bundling this with an inference server wrapper.

---

## Request Routing (public traffic → local client)

Yes — the SvelteKit/API caller must explicitly wire hostname → site → local port. Per
`docs.pangolin.net/manage/common-api-routes`, provisioning is three explicit API objects:
1. **Site** (the newt instance) — `PUT /org/{orgId}/site`.
2. **Public Resource** — `PUT /org/{orgId}/public-resource` with `domainId` + `subdomain`,
   producing a `fullDomain` (e.g. `my-subdomain.pangolin.net`).
3. **Target** — `PUT /public-resource/{resourceId}/target`, body:
   ```json
   { "ip": "localhost", "port": 8080, "method": "http", "siteId": 5165 }
   ```
   This is literally "map this resource to this site's local ip:port" — i.e. exactly the mapping
   needed to route `https://<subdomain>.<domain>` → the specific newt client's local
   `llama.cpp` server port.

A resource can have **multiple targets** (across sites) for load-balancing/failover
(docs.pangolin.net/manage/resources/public/healthchecks-failover), so 1 newt client = 1 site =
potentially several local ports/services, each becoming its own target(s) on one or more public
resources.

Traffic path once wired up: public request hits Traefik (fronted by Gerbil for TLS/QUIC) →
Traefik/Pangolin resolve the resource by hostname → traffic is forwarded down the WireGuard
tunnel associated with the target's `siteId` → newt's userspace netstack proxy relays it to the
configured local `ip:port` (`newt/netstack2/proxy.go`, `newt/netstack2/http_handler.go` implement
this local TCP/HTTP proxying; confirmed present in the cloned repo file listing).

Private (non-HTTP, ZTNA-style) resources exist too (`PUT /org/{orgId}/private-resource`), but for
our use case (public SvelteKit backend reaching an inference server) the **public HTTP
resource + target** pattern above is the relevant one.

---

## Status Polling / Health Checks

Two mechanisms found, one clearly documented at the API level and one confirmed directly in
source:

1. **Site online/offline state.** The `sites` DB table (referenced in
   `server/routers/site/getSite.ts`) carries state that a `GET /site/{siteId}` or
   `GET /org/{orgId}/site/{niceId}` call returns, joined with the associated `newts` row
   (`newtVersion`, `agent`, `agentVersion`). The **create-site** API response documented at
   docs.pangolin.net/manage/common-api-routes already shows the shape:
   ```json
   { "data": { "siteId": 8723, "niceId": "...", "online": false, "address": "100.90.128.0",
       "newtId": "...", "secret": "..." }, "success": true }
   ```
   So polling `GET /org/{orgId}/site/{niceId}` (or `GET /site/{siteId}` by numeric id) and reading
   the `online` boolean is the mechanism for "is this specific newt client currently connected."
   There is also `server/routers/site/listSites.ts` for listing/polling all sites in an org at
   once, and `server/routers/newt/handleNewtPingMessage.ts` /
   `server/routers/newt/pingAccumulator.ts` in source show the server actively receives
   keepalive/ping messages from each newt over the websocket, which is presumably what backs the
   `online` flag (the exact handler that flips `online` was not traced, so treat that link as
   inferred rather than proven from source).
2. **Historical status / uptime.** `server/routers/site/getStatusHistory.ts` and
   `getBatchedStatusHistory.ts` expose a `GET` endpoint (`/site/{siteId}/status-history` per the
   route naming, params `siteId`, query `days`, `tzOffsetMinutes`) backed by
   `@server/lib/statusHistory.getCachedStatusHistory` — gives a time series rather than just
   current state, useful for an uptime dashboard on our side.
3. **Target-level health checks** (separate from site online/offline): public resource targets
   can have HTTP/TCP health checks configured
   (docs.pangolin.net/manage/resources/public/healthchecks-failover) with states
   `Unknown|Healthy|Unhealthy`, intervals/thresholds, and automatic removal from the routing pool.
   These are per-target (i.e. per local llama.cpp port), a level below "is the newt tunnel up" —
   both are useful signals for our monitor service: site `online` tells you the tunnel exists,
   target health tells you the local llama.cpp process is actually answering.

For aipotluck-local-client's monitor service, the practical polling design would be:
`GET /org/{orgId}/site/{niceId}` on an interval → read `online`, and optionally
`GET /public-resource/{resourceId}` (implied to exist alongside the create-resource endpoint,
per Swagger, though its route file was not checked) or the health-check state on the
associated target, to distinguish "tunnel down" from "tunnel up but llama.cpp not responding."

---

## Open Questions / Assumptions

Not confirmed from source or docs:

1. **Exact route/path naming for target and resource "get" endpoints** (e.g. the literal path for
   fetching one target's current health state). The *create* (`PUT`) routes are documented in the
   common-api-routes walkthrough and confirmed via `registerPath` calls in a handful of source
   files (`getSite.ts`, `getStatusHistory.ts`), but the full Swagger tree at
   `https://api.pangolin.net/v1/docs` was not enumerated, so some GET/list endpoint paths above
   (e.g. `GET /public-resource/{resourceId}`) are inferred by pattern rather than quoted.
2. **The exact mechanism that flips `online: true/false` on a site row.**
   `server/routers/newt/handleNewtPingMessage.ts` and `pingAccumulator.ts` exist and suggest a
   websocket keepalive/ping model, but the code path from "ping received" to "DB row updated" was
   not traced.
3. Whether the **Integration API and the dashboard's internal API are the same route table** (i.e.
   whether all the `common-api-routes` examples work identically whether integration API is
   enabled via `flags.enable_integration_api` or not) is not fully spelled out; the docs imply the
   Integration API is the same REST surface just exposed on a separate port with its own
   auth/Swagger, but this is an assumption rather than a directly quoted statement.
4. Newt is being deprecated in favor of the newer "Pangolin CLI" (`fosrl/cli`, `pangolin site up`)
   per the README banner — for a new integration it may be worth evaluating that CLI instead of
   raw `newt`, since Newt's long-term support window is unclear from what's documented. This
   doesn't block the plan (newt's id/secret/endpoint model is unchanged) but is a decision point.

The caveats above are about *endpoint completeness/exhaustiveness*, not about whether the core architecture claims
(Docker Compose install, org API key + REST API, id/secret newt registration, explicit
site+resource+target routing, `online` status field) are real — those are all directly quoted
from the actual docs or source.
