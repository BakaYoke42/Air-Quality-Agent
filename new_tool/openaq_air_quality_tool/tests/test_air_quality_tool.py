from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from air_quality_tool import (  # noqa: E402
    AirQualityService,
    ConfigurationError,
    LocationAmbiguousError,
    LocationRejectedError,
    Settings,
    get_air_quality,
)


NOW = datetime(2026, 7, 22, 10, 0, tzinfo=timezone.utc)


def station(
    *,
    location_id: int,
    name: str,
    country: str,
    distance: float,
    pollutant: str,
    parameter_id: int,
    sensor_id: int,
    unit: str = "µg/m³",
) -> dict[str, Any]:
    return {
        "id": location_id,
        "name": name,
        "country": {"code": country, "name": country},
        "provider": {"id": 1, "name": "Test provider"},
        "licenses": [
            {
                "name": "CC BY 4.0",
                "attribution": {
                    "name": "Test data owner",
                    "url": "https://example.test/data",
                },
            }
        ],
        "coordinates": {"latitude": 43.48, "longitude": -1.56},
        "distance": distance,
        "sensors": [
            {
                "id": sensor_id,
                "parameter": {
                    "id": parameter_id,
                    "name": pollutant,
                    "units": unit,
                    "displayName": pollutant.upper(),
                },
            }
        ],
    }


def latest(
    *,
    location_id: int,
    sensor_id: int,
    value: float,
    measured_at: str = "2026-07-22T09:30:00Z",
) -> dict[str, Any]:
    return {
        "locationsId": location_id,
        "sensorsId": sensor_id,
        "value": value,
        "datetime": {"utc": measured_at, "local": measured_at},
        "coordinates": {"latitude": 43.48, "longitude": -1.56},
    }


class FakeTransport:
    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, dict[str, Any], dict[str, str], str]] = []

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        service: str,
    ) -> Any:
        params_copy = dict(params)
        headers_copy = dict(headers)
        self.calls.append((url, params_copy, headers_copy, service))
        return self.handler(url, params_copy, headers_copy, service)


def parameters_response() -> dict[str, Any]:
    return {
        "meta": {"found": 2},
        "results": [
            {"id": 2, "name": "pm25", "units": "µg/m³"},
            {"id": 6, "name": "no2", "units": "µg/m³"},
        ],
    }


def geocode_response(country: str = "fr") -> list[dict[str, Any]]:
    return [
        {
            "display_name": "Biarritz, Bayonne, France",
            "lat": "43.4832",
            "lon": "-1.5586",
            "address": {"country": "France", "country_code": country},
        }
    ]


def ambiguous_geocode_response() -> list[dict[str, Any]]:
    return [
        {
            "display_name": "Neustadt, Hesse, Germany",
            "name": "Neustadt",
            "lat": "50.8500",
            "lon": "9.1167",
            "address": {
                "town": "Neustadt",
                "state": "Hesse",
                "country": "Germany",
                "country_code": "de",
            },
        },
        {
            "display_name": "Neustadt, Rhineland-Palatinate, Germany",
            "name": "Neustadt",
            "lat": "49.3500",
            "lon": "8.1500",
            "address": {
                "town": "Neustadt",
                "state": "Rhineland-Palatinate",
                "country": "Germany",
                "country_code": "de",
            },
        },
    ]


class AirQualityServiceTests(unittest.TestCase):
    def make_service(self, handler) -> tuple[AirQualityService, FakeTransport]:
        transport = FakeTransport(handler)
        settings = Settings(
            openaq_api_key="test-key",
            max_data_age_hours=24,
            max_location_pages_per_pollutant=1,
            max_stations_checked=10,
            result_cache_seconds=0,
        )
        service = AirQualityService(
            settings,
            transport=transport,
            now=lambda: NOW,
            sleeper=lambda _: None,
        )
        return service, transport

    def test_uses_two_independent_stations(self):
        pm_station = station(
            location_id=10,
            name="Biarritz PM",
            country="FR",
            distance=1_200,
            pollutant="pm25",
            parameter_id=2,
            sensor_id=101,
        )
        no2_station = station(
            location_id=20,
            name="Anglet NO2",
            country="FR",
            distance=3_400,
            pollutant="no2",
            parameter_id=6,
            sensor_id=202,
        )

        def handler(url, params, _headers, _service):
            if url.endswith("/search"):
                return geocode_response()
            if url.endswith("/parameters"):
                return parameters_response()
            if url.endswith("/locations"):
                item = pm_station if params["parameters_id"] == 2 else no2_station
                return {"meta": {"found": 1}, "results": [item]}
            if url.endswith("/locations/10/latest"):
                return {"meta": {"found": 1}, "results": [latest(location_id=10, sensor_id=101, value=8.4)]}
            if url.endswith("/locations/20/latest"):
                return {"meta": {"found": 1}, "results": [latest(location_id=20, sensor_id=202, value=17.1)]}
            raise AssertionError(f"Unexpected URL: {url}")

        service, _ = self.make_service(handler)
        result = service.get_current_air_quality("Biarritz")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["country"]["code"], "FR")
        self.assertEqual(result["pollutants"]["pm25"]["value"], 8.4)
        self.assertEqual(result["pollutants"]["no2"]["value"], 17.1)
        self.assertEqual(
            result["pollutants"]["pm25"]["station"]["location_id"], 10
        )
        self.assertEqual(
            result["pollutants"]["no2"]["station"]["location_id"], 20
        )
        self.assertEqual(
            result["pollutants"]["pm25"]["station"]["source_attribution"][0][
                "name"
            ],
            "Test data owner",
        )

    def test_geocoder_is_hard_limited_to_three_countries(self):
        def handler(url, params, _headers, _service):
            if url.endswith("/search"):
                self.assertEqual(params["countrycodes"], "fr,it,de")
                return []
            raise AssertionError("OpenAQ must not be called for rejected locations")

        service, transport = self.make_service(handler)

        with self.assertRaises(LocationRejectedError):
            service.get_current_air_quality("Madrid")

        self.assertEqual(len(transport.calls), 1)
        self.assertTrue(transport.calls[0][0].endswith("/search"))

    def test_ambiguous_city_requires_user_clarification(self):
        def handler(url, params, _headers, _service):
            if url.endswith("/search"):
                self.assertEqual(params["countrycodes"], "fr,it,de")
                return ambiguous_geocode_response()
            raise AssertionError("OpenAQ must not be called before clarification")

        service, transport = self.make_service(handler)

        with self.assertRaises(LocationAmbiguousError) as caught:
            service.get_current_air_quality("Neustadt")
        self.assertEqual(len(caught.exception.candidates), 2)
        self.assertEqual(len(transport.calls), 1)

        with patch("air_quality_tool._default_service", return_value=service):
            result = get_air_quality("Neustadt")

        self.assertEqual(result["status"], "ambiguous_location")
        self.assertEqual(
            result["next_action"], "ask_user_to_choose_one_candidate_then_retry"
        )
        self.assertEqual(len(result["candidates"]), 2)
        self.assertIn("De laquelle parlez-vous", result["clarification_question"])

    def test_cross_country_openaq_result_is_discarded(self):
        leaked_station = station(
            location_id=99,
            name="Spanish station",
            country="ES",
            distance=500,
            pollutant="pm25",
            parameter_id=2,
            sensor_id=999,
        )

        def handler(url, _params, _headers, _service):
            if url.endswith("/search"):
                return geocode_response()
            if url.endswith("/parameters"):
                return parameters_response()
            if url.endswith("/locations"):
                return {"meta": {"found": 1}, "results": [leaked_station]}
            raise AssertionError("A rejected station must never be queried")

        service, _ = self.make_service(handler)
        result = service.get_current_air_quality("Biarritz")

        self.assertEqual(result["status"], "no_data")
        self.assertIsNone(result["pollutants"]["pm25"])
        self.assertIsNone(result["pollutants"]["no2"])

    def test_stale_measurement_is_not_returned_as_current(self):
        pm_station = station(
            location_id=10,
            name="Old PM station",
            country="FR",
            distance=1_200,
            pollutant="pm25",
            parameter_id=2,
            sensor_id=101,
        )

        def handler(url, params, _headers, _service):
            if url.endswith("/search"):
                return geocode_response()
            if url.endswith("/parameters"):
                return parameters_response()
            if url.endswith("/locations"):
                results = [pm_station] if params["parameters_id"] == 2 else []
                return {"meta": {"found": len(results)}, "results": results}
            if url.endswith("/locations/10/latest"):
                return {
                    "meta": {"found": 1},
                    "results": [
                        latest(
                            location_id=10,
                            sensor_id=101,
                            value=7.2,
                            measured_at="2026-07-20T08:00:00Z",
                        )
                    ],
                }
            raise AssertionError(f"Unexpected URL: {url}")

        service, _ = self.make_service(handler)
        result = service.get_current_air_quality("Biarritz")

        self.assertEqual(result["status"], "no_data")
        self.assertIsNone(result["pollutants"]["pm25"])

    def test_openaq_requests_retain_country_and_pollutant_filters(self):
        def handler(url, params, _headers, _service):
            if url.endswith("/search"):
                return geocode_response(country="de")
            if url.endswith("/parameters"):
                return parameters_response()
            if url.endswith("/locations"):
                self.assertEqual(params["iso"], "DE")
                self.assertIn(params["parameters_id"], {2, 6})
                self.assertLessEqual(params["radius"], 25_000)
                self.assertEqual(params["limit"], 1_000)
                return {"meta": {"found": 0}, "results": []}
            raise AssertionError(f"Unexpected URL: {url}")

        service, _ = self.make_service(handler)
        result = service.get_current_air_quality("Berlin")
        self.assertEqual(result["country"]["code"], "DE")

    def test_future_measurement_is_rejected(self):
        pm_station = station(
            location_id=10,
            name="Future PM station",
            country="FR",
            distance=1_200,
            pollutant="pm25",
            parameter_id=2,
            sensor_id=101,
        )

        def handler(url, params, _headers, _service):
            if url.endswith("/search"):
                return geocode_response()
            if url.endswith("/parameters"):
                return parameters_response()
            if url.endswith("/locations"):
                results = [pm_station] if params["parameters_id"] == 2 else []
                return {"meta": {"found": len(results)}, "results": results}
            if url.endswith("/locations/10/latest"):
                return {
                    "meta": {"found": 1},
                    "results": [
                        latest(
                            location_id=10,
                            sensor_id=101,
                            value=7.2,
                            measured_at="2026-07-22T11:00:01Z",
                        )
                    ],
                }
            raise AssertionError(f"Unexpected URL: {url}")

        service, _ = self.make_service(handler)
        result = service.get_current_air_quality("Biarritz")

        self.assertEqual(result["status"], "no_data")
        self.assertIsNone(result["pollutants"]["pm25"])

    def test_non_mass_unit_and_negative_value_are_rejected(self):
        ppb_station = station(
            location_id=10,
            name="NO2 ppb station",
            country="FR",
            distance=900,
            pollutant="no2",
            parameter_id=6,
            sensor_id=101,
            unit="ppb",
        )
        pm_station = station(
            location_id=20,
            name="Negative PM station",
            country="FR",
            distance=1_100,
            pollutant="pm25",
            parameter_id=2,
            sensor_id=202,
        )

        def handler(url, params, _headers, _service):
            if url.endswith("/search"):
                return geocode_response()
            if url.endswith("/parameters"):
                return {
                    "meta": {"found": 3},
                    "results": [
                        {"id": 90, "name": "no2", "units": "ppb"},
                        {"id": 2, "name": "pm25", "units": "ug/m3"},
                        {"id": 6, "name": "no2", "units": "µg/m³"},
                    ],
                }
            if url.endswith("/locations"):
                item = pm_station if params["parameters_id"] == 2 else ppb_station
                return {"meta": {"found": 1}, "results": [item]}
            if url.endswith("/locations/20/latest"):
                return {
                    "meta": {"found": 1},
                    "results": [latest(location_id=20, sensor_id=202, value=-1.0)],
                }
            raise AssertionError(
                "A station with an unsupported unit must not be queried"
            )

        service, transport = self.make_service(handler)
        result = service.get_current_air_quality("Biarritz")

        self.assertEqual(result["status"], "no_data")
        location_calls = [
            call for call in transport.calls if call[0].endswith("/locations")
        ]
        self.assertEqual(
            {call[1]["parameters_id"] for call in location_calls},
            {2, 6},
        )
        self.assertFalse(
            any(call[0].endswith("/locations/10/latest") for call in transport.calls)
        )

    def test_missing_or_out_of_radius_distance_is_discarded(self):
        missing_distance = station(
            location_id=10,
            name="Unknown-distance station",
            country="FR",
            distance=float("nan"),
            pollutant="pm25",
            parameter_id=2,
            sensor_id=101,
        )
        missing_distance["coordinates"] = {}
        far_station = station(
            location_id=20,
            name="Too-far station",
            country="FR",
            distance=30_000,
            pollutant="no2",
            parameter_id=6,
            sensor_id=202,
        )

        def handler(url, params, _headers, _service):
            if url.endswith("/search"):
                return geocode_response()
            if url.endswith("/parameters"):
                return parameters_response()
            if url.endswith("/locations"):
                item = (
                    missing_distance
                    if params["parameters_id"] == 2
                    else far_station
                )
                return {"meta": {"found": 1}, "results": [item]}
            raise AssertionError("Unusable-distance stations must not be queried")

        service, transport = self.make_service(handler)
        result = service.get_current_air_quality("Biarritz")

        self.assertEqual(result["status"], "no_data")
        self.assertFalse(
            any(call[0].endswith("/latest") for call in transport.calls)
        )

    def test_rejects_control_characters_before_normalizing(self):
        service, transport = self.make_service(
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("No network call was expected")
            )
        )

        with self.assertRaises(LocationRejectedError):
            service.get_current_air_quality("Berlin\nGermany")

        self.assertEqual(transport.calls, [])

    def test_repeated_lookup_uses_short_lived_result_cache(self):
        def handler(url, _params, _headers, _service):
            if url.endswith("/search"):
                return geocode_response()
            if url.endswith("/parameters"):
                return parameters_response()
            if url.endswith("/locations"):
                return {"meta": {"found": 0}, "results": []}
            raise AssertionError(f"Unexpected URL: {url}")

        transport = FakeTransport(handler)
        service = AirQualityService(
            Settings(
                openaq_api_key="test-key",
                max_location_pages_per_pollutant=1,
                max_stations_checked=8,
                result_cache_seconds=300,
            ),
            transport=transport,
            now=lambda: NOW,
            sleeper=lambda _: None,
        )

        first = service.get_current_air_quality("Biarritz")
        call_count = len(transport.calls)
        second = service.get_current_air_quality("Biarritz")

        self.assertFalse(first["cache"]["hit"])
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(len(transport.calls), call_count)

    def test_cold_lookup_has_a_bounded_upstream_request_count(self):
        pm_stations = [
            station(
                location_id=index,
                name=f"PM station {index}",
                country="FR",
                distance=500 + index,
                pollutant="pm25",
                parameter_id=2,
                sensor_id=1_000 + index,
            )
            for index in range(1, 21)
        ]
        no2_stations = [
            station(
                location_id=100 + index,
                name=f"NO2 station {index}",
                country="FR",
                distance=700 + index,
                pollutant="no2",
                parameter_id=6,
                sensor_id=2_000 + index,
            )
            for index in range(1, 21)
        ]

        def handler(url, params, _headers, _service):
            if url.endswith("/search"):
                return geocode_response()
            if url.endswith("/parameters"):
                return parameters_response()
            if url.endswith("/locations"):
                results = (
                    pm_stations
                    if params["parameters_id"] == 2
                    else no2_stations
                )
                return {"meta": {"found": len(results)}, "results": results}
            if url.endswith("/latest"):
                return {"meta": {"found": 0}, "results": []}
            raise AssertionError(f"Unexpected URL: {url}")

        transport = FakeTransport(handler)
        service = AirQualityService(
            Settings(
                openaq_api_key="test-key",
                max_location_pages_per_pollutant=1,
                max_stations_checked=8,
                result_cache_seconds=0,
            ),
            transport=transport,
            now=lambda: NOW,
            sleeper=lambda _: None,
        )

        result = service.get_current_air_quality("Biarritz")

        self.assertEqual(result["status"], "no_data")
        self.assertLessEqual(len(transport.calls), 12)
        latest_calls = [
            call for call in transport.calls if call[0].endswith("/latest")
        ]
        self.assertEqual(len(latest_calls), 8)

    def test_missing_key_is_a_controlled_configuration_error(self):
        with (
            patch("air_quality_tool._load_simple_dotenv"),
            patch.dict(
                "os.environ",
                {
                    "NOMINATIM_USER_AGENT": (
                        "air-quality-test/1.0 (contact: test@example.com)"
                    )
                },
                clear=True,
            ),
        ):
            with self.assertRaises(ConfigurationError) as caught:
                Settings.from_env()

        self.assertIn("OPENAQ_API_KEY is missing", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
