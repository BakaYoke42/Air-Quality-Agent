"""Restricted real-time air-quality tool for agentic systems.

The public entry point is :func:`get_air_quality`. It accepts a textual place
name and returns only recent PM2.5 and NO2 measurements from France, Italy, or
Germany. Country restrictions are enforced both during geocoding and after
OpenAQ responds.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ALLOWED_COUNTRIES: dict[str, str] = {
    "FR": "France",
    "IT": "Italy",
    "DE": "Germany",
}
TARGET_POLLUTANTS = ("pm25", "no2")
POLLUTANT_LABELS = {"pm25": "PM2.5", "no2": "NO2"}
CANONICAL_MASS_UNIT = "µg/m³"
MAX_LOCATION_LENGTH = 120
OPENAQ_MAX_RADIUS_METERS = 25_000
OPENAQ_LOCATION_PAGE_SIZE = 1_000
MAX_RESPONSE_BYTES = 10_000_000
MAX_FUTURE_CLOCK_SKEW = timedelta(minutes=10)

logger = logging.getLogger(__name__)


class AirQualityToolError(RuntimeError):
    """Base exception for expected tool failures."""


class ConfigurationError(AirQualityToolError):
    """Raised when required local configuration is missing."""


class LocationRejectedError(AirQualityToolError):
    """Raised when a location cannot be resolved inside the allowlist."""


class LocationAmbiguousError(AirQualityToolError):
    """Raised when several distinct allowlisted settlements match a query."""

    def __init__(self, query: str, candidates: list[dict[str, Any]]) -> None:
        self.query = query
        self.candidates = candidates
        super().__init__(
            f"Several cities match '{query}'. The user must choose one before "
            "air-quality data can be requested."
        )


class UpstreamServiceError(AirQualityToolError):
    """Raised when an external service fails or returns invalid data."""


class JsonTransport(Protocol):
    """Small protocol that makes HTTP behavior replaceable in tests."""

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        service: str,
    ) -> Any:
        """Perform an HTTP GET and decode the JSON response."""


class UrllibJsonTransport:
    """Standard-library JSON transport with bounded retries and timeouts."""

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        timeout_seconds: float = 10.0,
        max_attempts: int = 2,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._sleep = sleeper

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        service: str,
    ) -> Any:
        query = urlencode(params)
        request_url = f"{url}?{query}" if query else url
        request = Request(request_url, headers=dict(headers), method="GET")

        for attempt in range(1, self.max_attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw_body = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw_body) > MAX_RESPONSE_BYTES:
                        raise UpstreamServiceError(
                            f"{service} response exceeded the size limit."
                        )
                    body = raw_body.decode("utf-8")
                    return json.loads(body)
            except HTTPError as exc:
                if (
                    exc.code in self.RETRYABLE_STATUS_CODES
                    and attempt < self.max_attempts
                    and service != "Nominatim"
                ):
                    retry_after = exc.headers.get("Retry-After")
                    delay = _safe_retry_delay(retry_after, fallback=0.5 * attempt)
                    self._sleep(delay)
                    continue
                if exc.code == 401 and service == "OpenAQ":
                    raise UpstreamServiceError(
                        "OpenAQ rejected the API key (HTTP 401)."
                    ) from exc
                raise UpstreamServiceError(
                    f"{service} returned HTTP {exc.code}."
                ) from exc
            except URLError as exc:
                if attempt < self.max_attempts and service != "Nominatim":
                    self._sleep(0.5 * attempt)
                    continue
                raise UpstreamServiceError(
                    f"{service} is unreachable: {exc.reason}."
                ) from exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                if attempt < self.max_attempts and service != "Nominatim":
                    self._sleep(0.5 * attempt)
                    continue
                raise UpstreamServiceError(
                    f"{service} request failed due to a network error."
                ) from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise UpstreamServiceError(
                    f"{service} returned an invalid JSON response."
                ) from exc

        raise UpstreamServiceError(f"{service} request failed.")


def _safe_retry_delay(value: str | None, fallback: float) -> float:
    try:
        return min(max(float(value), 0.0), 10.0) if value else fallback
    except ValueError:
        return fallback


def _load_simple_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding existing variables."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    openaq_api_key: str
    openaq_base_url: str = "https://api.openaq.org/v3"
    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    nominatim_user_agent: str = (
        "air-quality-agent/1.0 "
        "(+https://github.com/zaizou1003/Air-Quality-Agent)"
    )
    max_data_age_hours: float = 24.0
    radius_meters: int = OPENAQ_MAX_RADIUS_METERS
    max_location_pages_per_pollutant: int = 1
    max_stations_checked: int = 8
    request_deadline_seconds: float = 90.0
    result_cache_seconds: float = 300.0
    max_cache_entries: int = 128

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[1]
        _load_simple_dotenv(project_root / ".env")

        api_key = os.getenv("OPENAQ_API_KEY", "").strip()
        if not api_key or api_key.upper().startswith("YOUR_"):
            raise ConfigurationError(
                "OPENAQ_API_KEY is missing. Add it to the project's .env file."
            )

        max_age = _env_float("MAX_DATA_AGE_HOURS", 24.0, minimum=1.0, maximum=168.0)
        deadline = _env_float(
            "LIVE_AIR_QUALITY_DEADLINE_SECONDS",
            90.0,
            minimum=10.0,
            maximum=110.0,
        )
        cache_seconds = _env_float(
            "LIVE_AIR_QUALITY_CACHE_SECONDS",
            300.0,
            minimum=0.0,
            maximum=3600.0,
        )
        user_agent = os.getenv(
            "NOMINATIM_USER_AGENT", ""
        ).strip()
        if not user_agent:
            raise ConfigurationError(
                "NOMINATIM_USER_AGENT must identify the application; including "
                "a contact email address or repository URL is recommended."
            )

        return cls(
            openaq_api_key=api_key,
            openaq_base_url=os.getenv(
                "OPENAQ_BASE_URL", "https://api.openaq.org/v3"
            ).rstrip("/"),
            nominatim_base_url=os.getenv(
                "NOMINATIM_BASE_URL", "https://nominatim.openstreetmap.org"
            ).rstrip("/"),
            nominatim_user_agent=user_agent,
            max_data_age_hours=max_age,
            request_deadline_seconds=deadline,
            result_cache_seconds=cache_seconds,
        )


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be numeric.") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(
            f"{name} must be between {minimum:g} and {maximum:g}."
        )
    return value


@dataclass(frozen=True)
class ResolvedLocation:
    query: str
    display_name: str
    latitude: float
    longitude: float
    country_code: str


@dataclass(frozen=True)
class Measurement:
    pollutant: str
    value: float
    unit: str
    measured_at: datetime
    sensor_id: int
    location_id: int
    station_name: str
    provider_name: str | None
    source_attributions: tuple[
        tuple[str | None, str | None, str | None],
        ...,
    ]
    distance_meters: float
    station_latitude: float | None
    station_longitude: float | None


class AirQualityService:
    """Geocode a place and retrieve allowlisted, recent OpenAQ measurements."""

    _geocoder_lock = threading.Lock()
    _last_geocoder_request_monotonic = 0.0

    def __init__(
        self,
        settings: Settings,
        *,
        transport: JsonTransport | None = None,
        now: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.transport = transport or UrllibJsonTransport(sleeper=sleeper)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleeper
        self._monotonic = monotonic
        self._cache_lock = threading.RLock()
        self._parameter_lock = threading.Lock()
        self._geocode_cache: OrderedDict[str, ResolvedLocation] = OrderedDict()
        self._result_cache: OrderedDict[
            str, tuple[float, dict[str, Any]]
        ] = OrderedDict()
        self._parameter_ids: dict[str, int] | None = None

    def get_current_air_quality(self, location: str) -> dict[str, Any]:
        """Return recent PM2.5 and NO2 measurements for an allowlisted place."""
        query = _validate_location_query(location)
        cache_key = query.casefold()
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached

        request_started = self._monotonic()
        deadline = request_started + self.settings.request_deadline_seconds
        now = self._utc_now()
        resolved = self._geocode(query, deadline=deadline)
        parameter_ids = self._get_target_parameter_ids(deadline=deadline)
        locations = self._find_candidate_locations(
            resolved,
            parameter_ids,
            deadline=deadline,
        )
        measurements = self._select_nearest_recent_measurements(
            resolved,
            locations,
            now=now,
            deadline=deadline,
        )

        pollutants: dict[str, dict[str, Any] | None] = {}
        warnings: list[str] = []
        for pollutant in TARGET_POLLUTANTS:
            measurement = measurements.get(pollutant)
            if measurement is None:
                pollutants[pollutant] = None
                warnings.append(
                    f"No {POLLUTANT_LABELS[pollutant]} measurement newer than "
                    f"{self.settings.max_data_age_hours:g} hours was found within "
                    f"{self.settings.radius_meters / 1000:g} km."
                )
            else:
                pollutants[pollutant] = _measurement_to_dict(measurement, now)

        available_count = sum(value is not None for value in pollutants.values())
        status = "ok" if available_count == 2 else "partial" if available_count else "no_data"

        result = {
            "status": status,
            "requested_location": query,
            "resolved_location": resolved.display_name,
            "country": {
                "code": resolved.country_code,
                "name": ALLOWED_COUNTRIES[resolved.country_code],
            },
            "coordinates": {
                "latitude": resolved.latitude,
                "longitude": resolved.longitude,
            },
            "pollutants": pollutants,
            "retrieved_at_utc": _isoformat_utc(now),
            "freshness_limit_hours": self.settings.max_data_age_hours,
            "search_radius_km": self.settings.radius_meters / 1000,
            "selection_method": (
                "nearest fresh fixed station per pollutant among bounded "
                "OpenAQ candidates"
            ),
            "cache": {
                "hit": False,
                "ttl_seconds": self.settings.result_cache_seconds,
            },
            "allowed_countries": list(ALLOWED_COUNTRIES),
            "warnings": warnings,
            "attribution": [
                "Air-quality data: OpenAQ and the provider identified per measurement.",
                "Geocoding: © OpenStreetMap contributors, ODbL 1.0.",
            ],
        }
        self._store_cached_result(cache_key, result)
        return result

    def _utc_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _ensure_before_deadline(self, deadline: float) -> None:
        if self._monotonic() >= deadline:
            raise UpstreamServiceError(
                "The live air-quality lookup exceeded its overall deadline."
            )

    def _get_cached_result(self, key: str) -> dict[str, Any] | None:
        if self.settings.result_cache_seconds <= 0:
            return None
        with self._cache_lock:
            cached = self._result_cache.get(key)
            if cached is None:
                return None
            stored_at, result = cached
            age = self._monotonic() - stored_at
            if age < 0 or age > self.settings.result_cache_seconds:
                self._result_cache.pop(key, None)
                return None
            self._result_cache.move_to_end(key)
            copied = copy.deepcopy(result)
            now = self._utc_now()
            for measurement in copied.get("pollutants", {}).values():
                if not isinstance(measurement, dict):
                    continue
                try:
                    measured_at = _parse_datetime(measurement["measured_at_utc"])
                except (KeyError, TypeError, ValueError):
                    continue
                measurement["age_hours"] = round(
                    max(0.0, (now - measured_at).total_seconds() / 3600),
                    2,
                )
            copied["cache"] = {
                "hit": True,
                "age_seconds": round(age, 2),
                "ttl_seconds": self.settings.result_cache_seconds,
            }
            return copied

    def _store_cached_result(self, key: str, result: dict[str, Any]) -> None:
        if self.settings.result_cache_seconds <= 0:
            return
        with self._cache_lock:
            self._result_cache[key] = (self._monotonic(), copy.deepcopy(result))
            self._result_cache.move_to_end(key)
            while len(self._result_cache) > self.settings.max_cache_entries:
                self._result_cache.popitem(last=False)

    def _geocode(self, query: str, *, deadline: float) -> ResolvedLocation:
        cache_key = query.casefold()
        with self._cache_lock:
            cached = self._geocode_cache.get(cache_key)
            if cached is not None:
                self._geocode_cache.move_to_end(cache_key)
                return cached

        # Public Nominatim requires at most one request per second per app.
        with self._geocoder_lock:
            self._ensure_before_deadline(deadline)
            elapsed = (
                self._monotonic()
                - AirQualityService._last_geocoder_request_monotonic
            )
            if elapsed < 1.0 and AirQualityService._last_geocoder_request_monotonic:
                self._sleep(1.0 - elapsed)
            self._ensure_before_deadline(deadline)

            # Record the attempt before transport execution. Nominatim requests
            # are never retried inside the transport, preserving the 1 req/s cap.
            AirQualityService._last_geocoder_request_monotonic = self._monotonic()
            payload = self.transport.get_json(
                f"{self.settings.nominatim_base_url}/search",
                params={
                    "q": query,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "featureType": "settlement",
                    "countrycodes": "fr,it,de",
                    "accept-language": "fr,en",
                    "limit": 5,
                },
                headers={"User-Agent": self.settings.nominatim_user_agent},
                service="Nominatim",
            )

        if not isinstance(payload, list):
            raise UpstreamServiceError("Nominatim returned an unexpected response.")

        candidates: list[ResolvedLocation] = []
        candidate_descriptions: list[dict[str, Any]] = []
        seen_places: set[tuple[str, str, str]] = set()

        for item in payload:
            if not isinstance(item, dict):
                continue
            address = _mapping(item.get("address"))
            country_code = str(address.get("country_code", "")).upper()
            if country_code not in ALLOWED_COUNTRIES:
                continue
            try:
                resolved = ResolvedLocation(
                    query=query,
                    display_name=str(item["display_name"]),
                    latitude=float(item["lat"]),
                    longitude=float(item["lon"]),
                    country_code=country_code,
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not _valid_coordinates(resolved.latitude, resolved.longitude):
                continue

            locality = _candidate_locality_name(item, address)
            region = _candidate_region_name(address)
            deduplication_key = (
                _normalize_place_text(locality),
                _normalize_place_text(region),
                country_code,
            )
            if deduplication_key in seen_places:
                continue
            seen_places.add(deduplication_key)
            candidates.append(resolved)
            candidate_descriptions.append(
                {
                    "display_name": resolved.display_name,
                    "locality": locality,
                    "region": region or None,
                    "country": {
                        "code": country_code,
                        "name": ALLOWED_COUNTRIES[country_code],
                    },
                    "suggested_query": ", ".join(
                        part
                        for part in (locality, region, ALLOWED_COUNTRIES[country_code])
                        if part
                    ),
                }
            )

        if len(candidates) > 1:
            raise LocationAmbiguousError(query, candidate_descriptions)

        if candidates:
            with self._cache_lock:
                self._geocode_cache[cache_key] = candidates[0]
                self._geocode_cache.move_to_end(cache_key)
                while len(self._geocode_cache) > self.settings.max_cache_entries:
                    self._geocode_cache.popitem(last=False)
            return candidates[0]

        raise LocationRejectedError(
            "Location not found in France, Italy, or Germany. No air-quality "
            "data was requested."
        )

    def _get_target_parameter_ids(self, *, deadline: float) -> dict[str, int]:
        if self._parameter_ids is not None:
            return self._parameter_ids

        with self._parameter_lock:
            if self._parameter_ids is not None:
                return self._parameter_ids

            payload = self._openaq_get(
                "/parameters",
                params={"parameter_type": "pollutant", "limit": 100, "page": 1},
                deadline=deadline,
            )
            results = _results_list(payload, "OpenAQ parameters")
            found: dict[str, int] = {}
            for item in results:
                if not isinstance(item, dict):
                    continue
                normalized = _normalize_pollutant_name(str(item.get("name", "")))
                unit = _canonical_mass_unit(item.get("units"))
                if normalized in TARGET_POLLUTANTS and unit is not None:
                    parameter_id = _safe_int(item.get("id"))
                    if parameter_id is not None:
                        # Be deterministic if the API exposes duplicate metadata.
                        found.setdefault(normalized, parameter_id)

            missing = set(TARGET_POLLUTANTS) - set(found)
            if missing:
                raise UpstreamServiceError(
                    "OpenAQ did not expose the required PM2.5 and NO2 "
                    f"mass-concentration parameters; missing: {sorted(missing)}."
                )
            self._parameter_ids = found
            return found

    def _find_candidate_locations(
        self,
        resolved: ResolvedLocation,
        parameter_ids: Mapping[str, int],
        *,
        deadline: float,
    ) -> list[dict[str, Any]]:
        locations_by_id: dict[int, dict[str, Any]] = {}

        # Search each pollutant independently so separate stations are supported
        # regardless of how the API interprets a multi-value parameter filter.
        for pollutant in TARGET_POLLUTANTS:
            parameter_id = parameter_ids[pollutant]
            for page in range(1, self.settings.max_location_pages_per_pollutant + 1):
                payload = self._openaq_get(
                    "/locations",
                    params={
                        "coordinates": f"{resolved.latitude:.4f},{resolved.longitude:.4f}",
                        "radius": self.settings.radius_meters,
                        "parameters_id": parameter_id,
                        "iso": resolved.country_code,
                        "mobile": "false",
                        "limit": OPENAQ_LOCATION_PAGE_SIZE,
                        "page": page,
                    },
                    deadline=deadline,
                )
                results = _results_list(payload, "OpenAQ locations")
                for location in results:
                    if not isinstance(location, dict):
                        continue
                    country = _mapping(location.get("country"))
                    country_code = str(country.get("code", "")).upper()
                    # L4 action gate: never consume or return a cross-country result.
                    if (
                        country_code != resolved.country_code
                        or country_code not in ALLOWED_COUNTRIES
                    ):
                        continue
                    try:
                        location_id = int(location["id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    locations_by_id[location_id] = location

                meta = _mapping(payload.get("meta")) if isinstance(payload, dict) else {}
                found = _safe_int(meta.get("found"))
                if (
                    len(results) < OPENAQ_LOCATION_PAGE_SIZE
                    or (
                        found is not None
                        and page * OPENAQ_LOCATION_PAGE_SIZE >= found
                    )
                ):
                    break

        locations = [
            location
            for location in locations_by_id.values()
            if _usable_distance(
                _location_distance_meters(location, resolved),
                self.settings.radius_meters,
            )
        ]
        locations.sort(key=lambda item: _location_distance_meters(item, resolved))
        return locations

    def _select_nearest_recent_measurements(
        self,
        resolved: ResolvedLocation,
        locations: list[dict[str, Any]],
        *,
        now: datetime,
        deadline: float,
    ) -> dict[str, Measurement]:
        selected: dict[str, Measurement] = {}
        cutoff = now - timedelta(hours=self.settings.max_data_age_hours)
        latest_acceptable = now + MAX_FUTURE_CLOCK_SKEW

        candidates = _balanced_station_candidates(
            locations,
            self.settings.max_stations_checked,
        )
        for location in candidates:
            sensor_metadata = _target_sensor_metadata(location)
            if not sensor_metadata:
                continue
            remaining = set(TARGET_POLLUTANTS) - set(selected)
            if not any(
                metadata["pollutant"] in remaining
                for metadata in sensor_metadata.values()
            ):
                continue

            location_id = _safe_int(location.get("id"))
            if location_id is None:
                continue
            payload = self._openaq_get(
                f"/locations/{location_id}/latest",
                params={
                    "limit": 100,
                    "page": 1,
                    "datetime_min": _isoformat_utc(cutoff),
                },
                deadline=deadline,
            )
            latest_results = _results_list(payload, "OpenAQ latest measurements")
            distance = _location_distance_meters(location, resolved)

            per_station: dict[str, Measurement] = {}
            for item in latest_results:
                measurement = _parse_measurement(
                    item=item,
                    location=location,
                    sensor_metadata=sensor_metadata,
                    distance_meters=distance,
                )
                if (
                    measurement is None
                    or measurement.measured_at < cutoff
                    or measurement.measured_at > latest_acceptable
                ):
                    continue
                current = per_station.get(measurement.pollutant)
                if current is None or measurement.measured_at > current.measured_at:
                    per_station[measurement.pollutant] = measurement

            for pollutant, measurement in per_station.items():
                # Locations are sorted by distance, so the first fresh result is
                # the nearest fresh measurement for that pollutant.
                selected.setdefault(pollutant, measurement)

            if all(pollutant in selected for pollutant in TARGET_POLLUTANTS):
                break

        return selected

    def _openaq_get(
        self,
        path: str,
        *,
        params: Mapping[str, Any],
        deadline: float,
    ) -> Any:
        self._ensure_before_deadline(deadline)
        return self.transport.get_json(
            f"{self.settings.openaq_base_url}{path}",
            params=params,
            headers={"X-API-Key": self.settings.openaq_api_key},
            service="OpenAQ",
        )


def _validate_location_query(location: str) -> str:
    if not isinstance(location, str):
        raise LocationRejectedError("Location must be a text value.")
    if any(ord(character) < 32 or ord(character) == 127 for character in location):
        raise LocationRejectedError("Location contains invalid control characters.")
    value = " ".join(location.strip().split())
    if not value:
        raise LocationRejectedError("Location cannot be empty.")
    if len(value) > MAX_LOCATION_LENGTH:
        raise LocationRejectedError(
            f"Location is too long (maximum {MAX_LOCATION_LENGTH} characters)."
        )
    return value


def _normalize_pollutant_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _normalize_place_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _canonical_mass_unit(value: Any) -> str | None:
    normalized = (
        str(value or "")
        .strip()
        .casefold()
        .replace("μ", "µ")
        .replace("³", "3")
        .replace("^3", "3")
        .replace(" ", "")
    )
    if normalized in {"µg/m3", "ug/m3", "mcg/m3"}:
        return CANONICAL_MASS_UNIT
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _valid_coordinates(latitude: float, longitude: float) -> bool:
    return (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    )


def _usable_distance(distance_meters: float, radius_meters: int) -> bool:
    return (
        math.isfinite(distance_meters)
        and 0 <= distance_meters <= radius_meters
    )


def _candidate_locality_name(
    item: Mapping[str, Any], address: Mapping[str, Any]
) -> str:
    for field in (
        "city",
        "town",
        "village",
        "municipality",
        "hamlet",
        "county",
    ):
        value = address.get(field)
        if value:
            return str(value)
    if item.get("name"):
        return str(item["name"])
    return str(item.get("display_name", "Unknown place")).split(",", 1)[0].strip()


def _candidate_region_name(address: Mapping[str, Any]) -> str:
    for field in ("state", "region", "state_district", "county"):
        value = address.get(field)
        if value:
            return str(value)
    return ""


def _results_list(payload: Any, label: str) -> list[Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise UpstreamServiceError(f"{label} response has an invalid structure.")
    return payload["results"]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _target_sensor_metadata(location: Mapping[str, Any]) -> dict[int, dict[str, str]]:
    metadata: dict[int, dict[str, str]] = {}
    sensors = location.get("sensors")
    if not isinstance(sensors, list):
        return metadata
    for sensor in sensors:
        if not isinstance(sensor, dict):
            continue
        parameter = _mapping(sensor.get("parameter"))
        pollutant = _normalize_pollutant_name(str(parameter.get("name", "")))
        if pollutant not in TARGET_POLLUTANTS:
            continue
        unit = _canonical_mass_unit(parameter.get("units"))
        if unit is None:
            continue
        sensor_id = _safe_int(sensor.get("id"))
        if sensor_id is None:
            continue
        metadata[sensor_id] = {
            "pollutant": pollutant,
            "unit": unit,
        }
    return metadata


def _balanced_station_candidates(
    locations: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Interleave nearest candidates so one pollutant cannot consume the cap."""
    if limit <= 0:
        return []

    metadata_by_location = [
        (location, _target_sensor_metadata(location))
        for location in locations
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    cursor = {pollutant: 0 for pollutant in TARGET_POLLUTANTS}

    while len(selected) < limit:
        added_this_round = False
        for pollutant in TARGET_POLLUTANTS:
            entries = metadata_by_location
            while cursor[pollutant] < len(entries):
                index = cursor[pollutant]
                cursor[pollutant] += 1
                location, sensor_metadata = entries[index]
                if pollutant not in {
                    metadata["pollutant"]
                    for metadata in sensor_metadata.values()
                }:
                    continue
                location_id = _safe_int(location.get("id"))
                if location_id is None or location_id in selected_ids:
                    continue
                selected.append(location)
                selected_ids.add(location_id)
                added_this_round = True
                break
            if len(selected) >= limit:
                break
        if not added_this_round:
            break

    return selected


def _parse_measurement(
    *,
    item: Any,
    location: Mapping[str, Any],
    sensor_metadata: Mapping[int, Mapping[str, str]],
    distance_meters: float,
) -> Measurement | None:
    if not isinstance(item, dict):
        return None
    sensor_id = _safe_int(item.get("sensorsId"))
    location_id = _safe_int(item.get("locationsId"))
    if sensor_id is None or location_id is None or sensor_id not in sensor_metadata:
        return None
    if location_id != _safe_int(location.get("id")):
        return None

    try:
        datetime_payload = _mapping(item.get("datetime"))
        measured_at = _parse_datetime(datetime_payload["utc"])
        value = float(item["value"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None

    coordinates = _mapping(location.get("coordinates"))
    if not coordinates:
        coordinates = _mapping(item.get("coordinates"))
    station_latitude = _safe_float(coordinates.get("latitude"))
    station_longitude = _safe_float(coordinates.get("longitude"))
    if (
        station_latitude is not None
        and station_longitude is not None
        and not _valid_coordinates(station_latitude, station_longitude)
    ):
        station_latitude = None
        station_longitude = None
    provider = _mapping(location.get("provider"))
    sensor = sensor_metadata[sensor_id]
    source_attributions: list[
        tuple[str | None, str | None, str | None]
    ] = []
    licenses = location.get("licenses")
    if isinstance(licenses, list):
        for license_item in licenses:
            license_payload = _mapping(license_item)
            attribution = _mapping(license_payload.get("attribution"))
            license_name = (
                str(license_payload["name"])
                if license_payload.get("name")
                else None
            )
            attribution_name = (
                str(attribution["name"]) if attribution.get("name") else None
            )
            attribution_url = (
                str(attribution["url"]) if attribution.get("url") else None
            )
            if license_name or attribution_name or attribution_url:
                source_attributions.append(
                    (license_name, attribution_name, attribution_url)
                )

    return Measurement(
        pollutant=sensor["pollutant"],
        value=value,
        unit=sensor["unit"],
        measured_at=measured_at,
        sensor_id=sensor_id,
        location_id=location_id,
        station_name=str(location.get("name") or f"OpenAQ location {location_id}"),
        provider_name=str(provider.get("name")) if provider.get("name") else None,
        source_attributions=tuple(source_attributions),
        distance_meters=distance_meters,
        station_latitude=station_latitude,
        station_longitude=station_longitude,
    )


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Datetime must be a string.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _location_distance_meters(
    location: Mapping[str, Any], resolved: ResolvedLocation
) -> float:
    api_distance = _safe_float(location.get("distance"))
    if api_distance is not None and api_distance >= 0:
        return api_distance

    coordinates = _mapping(location.get("coordinates"))
    latitude = _safe_float(coordinates.get("latitude"))
    longitude = _safe_float(coordinates.get("longitude"))
    if (
        latitude is None
        or longitude is None
        or not _valid_coordinates(latitude, longitude)
    ):
        return float("inf")
    return _haversine_meters(
        resolved.latitude, resolved.longitude, latitude, longitude
    )


def _haversine_meters(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    earth_radius_meters = 6_371_008.8
    lat_1 = math.radians(latitude_1)
    lat_2 = math.radians(latitude_2)
    delta_lat = lat_2 - lat_1
    delta_lon = math.radians(longitude_2 - longitude_1)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * earth_radius_meters * math.asin(math.sqrt(min(max(a, 0.0), 1.0)))


def _measurement_to_dict(measurement: Measurement, now: datetime) -> dict[str, Any]:
    age_hours = max(
        0.0,
        (now.astimezone(timezone.utc) - measurement.measured_at).total_seconds()
        / 3600,
    )
    return {
        "pollutant": POLLUTANT_LABELS[measurement.pollutant],
        "value": measurement.value,
        "unit": measurement.unit,
        "measured_at_utc": _isoformat_utc(measurement.measured_at),
        "age_hours": round(age_hours, 2),
        "station": {
            "name": measurement.station_name,
            "location_id": measurement.location_id,
            "sensor_id": measurement.sensor_id,
            "provider": measurement.provider_name,
            "source_attribution": [
                {
                    "license": license_name,
                    "name": attribution_name,
                    "url": attribution_url,
                }
                for (
                    license_name,
                    attribution_name,
                    attribution_url,
                ) in measurement.source_attributions
            ],
            "distance_km": round(measurement.distance_meters / 1000, 2),
            "coordinates": {
                "latitude": measurement.station_latitude,
                "longitude": measurement.station_longitude,
            },
        },
    }


@lru_cache(maxsize=1)
def _default_service() -> AirQualityService:
    return AirQualityService(Settings.from_env())


def get_air_quality(location: str) -> dict[str, Any]:
    """Safe function for agents: never emits raw upstream responses or secrets."""
    try:
        return _default_service().get_current_air_quality(location)
    except LocationAmbiguousError as exc:
        options = " ; ".join(
            candidate["suggested_query"] for candidate in exc.candidates
        )
        return {
            "status": "ambiguous_location",
            "requested_location": exc.query,
            "message": str(exc),
            "clarification_question": (
                f"Plusieurs villes correspondent à « {exc.query} ». "
                f"De laquelle parlez-vous : {options} ?"
            ),
            "next_action": "ask_user_to_choose_one_candidate_then_retry",
            "candidates": exc.candidates,
            "allowed_countries": list(ALLOWED_COUNTRIES),
            "pollutants": {"pm25": None, "no2": None},
        }
    except LocationRejectedError as exc:
        return {
            "status": "rejected",
            "requested_location": location if isinstance(location, str) else None,
            "error": str(exc),
            "next_action": "ask_user_to_check_spelling_or_add_country_and_region",
            "allowed_countries": list(ALLOWED_COUNTRIES),
            "pollutants": {"pm25": None, "no2": None},
        }
    except ConfigurationError as exc:
        return {
            "status": "configuration_error",
            "error": str(exc),
            "allowed_countries": list(ALLOWED_COUNTRIES),
            "pollutants": {"pm25": None, "no2": None},
        }
    except UpstreamServiceError as exc:
        return {
            "status": "upstream_error",
            "error": str(exc),
            "allowed_countries": list(ALLOWED_COUNTRIES),
            "pollutants": {"pm25": None, "no2": None},
        }
    except Exception:
        # The MCP boundary must remain structured even if an unexpected
        # implementation bug or malformed nested payload reaches this point.
        logger.exception("Unexpected live air-quality tool failure")
        return {
            "status": "upstream_error",
            "error": "The live air-quality lookup failed unexpectedly.",
            "allowed_countries": list(ALLOWED_COUNTRIES),
            "pollutants": {"pm25": None, "no2": None},
        }


__all__ = [
    "ALLOWED_COUNTRIES",
    "AirQualityService",
    "ConfigurationError",
    "LocationAmbiguousError",
    "LocationRejectedError",
    "Settings",
    "UpstreamServiceError",
    "get_air_quality",
]
