"""Name/address -> coordinates lookup for the custom-route (ObLoS live) mode.

Lets dashboard users type a start/end location by name ("Osnabrück",
"Neumarkt 1, Köln") instead of clicking the map; the resulting coordinates
feed the same POST /api/oblos/simulate flow.

Uses the public Nominatim API (OpenStreetMap), Germany-limited -- matching
the self-hosted OSRM instance's whole-Germany coverage (results outside it
couldn't be routed anyway). Two policy/robustness measures, mirroring the
dwd_weather.py pattern:

  - every response is disk-cached in traces/geocode_cache/ keyed by the
    normalized query, so a repeated demo lookup is instant and works fully
    offline once primed;
  - live requests carry a descriptive User-Agent and are rate-limited to
    one per second module-wide (Nominatim usage policy).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time

import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(REPO_ROOT, "traces", "geocode_cache")
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = os.environ.get(
    "COSME_GEOCODE_USER_AGENT",
    "COSME-SIGCOMM26-demo/1.0 (research demo; contact: claude@brainfact.de)",
)
MIN_REQUEST_INTERVAL_S = 1.0

_rate_lock = threading.Lock()
_last_request_at = 0.0


class GeocodingUnavailable(RuntimeError):
    pass


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _cache_path(query: str) -> str:
    key = hashlib.sha1(_normalize(query).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{key}.json")


def _throttle() -> None:
    """Blocks until at least MIN_REQUEST_INTERVAL_S since the last live request."""
    global _last_request_at
    with _rate_lock:
        wait = _last_request_at + MIN_REQUEST_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def geocode(query: str, limit: int = 5) -> dict:
    """Returns {"results": [{display_name, lat, lon}], "source": "cache"|"nominatim"}.

    An empty results list is a valid answer (no match), not an error;
    GeocodingUnavailable is raised only when the network fails and there is
    no cached answer to fall back on.
    """
    if not _normalize(query):
        return {"results": [], "source": "empty-query"}

    path = _cache_path(query)
    if os.path.exists(path):
        with open(path) as f:
            return {"results": json.load(f), "source": "cache"}

    _throttle()
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "jsonv2", "limit": limit, "countrycodes": "de"},
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
    except (requests.RequestException, ValueError) as e:
        raise GeocodingUnavailable(f"Nominatim lookup failed for {query!r}: {e}") from e

    results = [
        {"display_name": r.get("display_name"), "lat": float(r["lat"]), "lon": float(r["lon"])}
        for r in raw
        if "lat" in r and "lon" in r
    ]
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f)
    return {"results": results, "source": "nominatim"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Geocode a place name (Germany).")
    parser.add_argument("query")
    args = parser.parse_args()
    print(json.dumps(geocode(args.query), indent=2))
