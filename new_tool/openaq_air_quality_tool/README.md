# Restricted OpenAQ live-air-quality tool

This standalone experimental prototype resolves a named place and returns the
**latest available, sufficiently recent** PM2.5 and NO2 mass-concentration
observations from nearby fixed OpenAQ stations. It is not registered with the
submitted four-tool agent and does not replace the project's 2024 archival
measurement tools.

The geographical scope is deliberately fixed to France (`FR`), Germany (`DE`),
and Italy (`IT`). The restriction is enforced in three places:

1. Nominatim receives `countrycodes=fr,it,de`.
2. OpenAQ receives the resolved ISO country code.
3. Every returned OpenAQ station is checked again before use.

PM2.5 and NO2 may come from different stations, providers, and timestamps.
The nearest fresh fixed station among the bounded OpenAQ candidates is selected
independently for each pollutant. The result is not a city mean, population
exposure estimate, annual mean, or legal-compliance determination.

## Status

The prototype uses the same `mcp==1.28.1` package and Streamable HTTP transport
as the main project, but it remains a separate experiment with its own tests and
configuration. A live contract check is required before relying on it.

Before any integration or deployment:

- keep the OpenAQ key only in the ignored `.env`;
- use an identifying Nominatim `User-Agent` with a real contact;
- run the opt-in live contract smoke check;
- add the fifth tool to the main L4 action matrix and end-to-end tests;
- preserve timestamps, units, station provenance, distance, and limitations in
  synthesis.

## Setup

Python 3.10 or newer is required.

macOS/Linux:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Create a free OpenAQ key at <https://explore.openaq.org/> and put it only in
the local `.env`:

```dotenv
OPENAQ_API_KEY=
NOMINATIM_USER_AGENT=air-quality-agent/1.0 (+https://github.com/zaizou1003/Air-Quality-Agent)
```

Never put a real key in `.env.example`, source code, tests, screenshots, or
commits. Install the pinned dependency:

```bash
python -m pip install -r requirements.txt
```

## Run as Streamable HTTP MCP

The standalone development server uses port `8001` to avoid the main agent's
default port:

```bash
python src/mcp_server.py
```

Its endpoint is:

```text
http://127.0.0.1:8001/mcp
```

The tool is `get_current_air_quality`:

```json
{"location": "Berlin, Germany"}
```

The MCP wrapper uses the same outer envelope as the main project:

```json
{
  "status": "ok",
  "data": {
    "status": "partial",
    "requested_location": "Berlin, Germany",
    "pollutants": {
      "pm25": {
        "value": 8.4,
        "unit": "µg/m³",
        "measured_at_utc": "2026-07-22T09:30:00Z",
        "station": {
          "name": "Example PM station",
          "distance_km": 1.2
        }
      },
      "no2": null
    }
  }
}
```

Valid data-level statuses are:

- `ok`: both pollutants have fresh observations;
- `partial`: exactly one pollutant has a fresh observation;
- `no_data`: neither pollutant has a fresh observation nearby;
- `ambiguous_location`: the user must choose one returned candidate;
- `rejected`: the location is invalid, unresolved, or outside FR/DE/IT.

Configuration and upstream failures use the outer MCP error envelope. Returned
payloads never include the API key or raw upstream response bodies.

## Definition of “current”

“Current” means the latest observation returned by OpenAQ that is no older
than the configured freshness window, 24 hours by default:

```dotenv
MAX_DATA_AGE_HOURS=12
```

OpenAQ's `latest` resource is the last value in each sensor time series; it is
not guaranteed to be the observation most recently ingested. Future-dated
values beyond a ten-minute clock-skew allowance, stale values, negative values,
unsupported units, non-finite distances, and stations beyond 25 km are
discarded.

Only PM2.5 and NO2 sensors reported in mass concentration are accepted and the
unit is normalized to `µg/m³`. A recent reading must never be compared directly
with an annual WHO guideline, annual EU limit, or the project's 2024 annual
country statistics.

## Bounded external work

A cold lookup is capped at:

- one Nominatim request;
- one OpenAQ parameter request;
- one OpenAQ location page per pollutant;
- eight OpenAQ station `latest` requests.

That is at most 12 logical upstream requests, reduced further when a station
contains only a pollutant already found. OpenAQ requests have bounded retry and
per-request timeouts, the whole lookup has a 90-second deadline, successful
results are cached for five minutes, and caches are size-bounded.

OpenAQ documents a general-use rate limit of 60 requests/minute and 2,000/hour.
Rate limits and terms can change, so consult:

- <https://docs.openaq.org/using-the-api/rate-limits>
- <https://docs.openaq.org/about/terms>
- <https://docs.openaq.org/api/operations/location_latest_get_v3_locations__locations_id__latest_get>

## Ambiguous locations

The tool never silently chooses between distinct settlements with the same
name. `ambiguous_location` includes candidates, suggested queries, and:

```json
{
  "next_action": "ask_user_to_choose_one_candidate_then_retry"
}
```

The agent must show the candidates, ask the user to choose, and call the tool
again with the clarified location.

## Privacy, attribution, and availability

The textual location is sent to the public Nominatim service. The resolved
coordinates and country are then sent to OpenAQ. Do not use this tool for
sensitive or person-specific locations.

The response attributes OpenAQ, the provider reported per measurement, and
OpenStreetMap contributors. OpenAQ coverage is incomplete and underlying data
may have provider-specific licensing or attribution obligations:

- <https://docs.openaq.org/about/terms>
- <https://docs.openaq.org/resources/licenses>
- <https://operations.osmfoundation.org/policies/nominatim/>

The public Nominatim policy requires an identifying `User-Agent`, caching,
attribution, and no more than one request per second. This implementation
serializes calls, does not internally retry Nominatim, and caches geocodes.

## Tests

Offline tests use fake transports, need no API key, and make no network calls:

```bash
python -B -m unittest discover -s tests -v
```

They cover country gating, ambiguity, independent stations, stale/future
observations, units, malformed values, bounded request selection, cache
behavior, envelopes, and tool documentation. A separate live smoke check is
still required to prove current OpenAQ/Nominatim compatibility. After placing
the key in the ignored `.env`, run:

```bash
python -B scripts/live_smoke_test.py "Berlin, Germany"
```

The script prints only the controlled tool payload; it never prints the API key
or raw upstream response bodies.

## Main-agent boundary

This prototype is intentionally outside the submitted agent. Integrating it
would require registration on the existing MCP server, a new L4 rule,
structured `M#` evidence handling, planner guidance, and a new five-tool
evaluation.
