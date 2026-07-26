"""Deterministic queries over the processed 2024 EEA measurement dataset.

The RAG corpus explains standards and methodology.  This module answers numeric
questions from structured data without asking an LLM to calculate statistics.
It reads exactly one annual dataset: Parquet is preferred, with CSV as a
portable fallback.  The two formats are never concatenated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


class MeasurementDataError(RuntimeError):
    """Raised when the processed measurement files are absent or inconsistent."""


class InvalidMeasurementQuery(ValueError):
    """Raised when a tool argument is outside the supported project scope."""


COUNTRY_ALIASES = {
    "de": "DE",
    "germany": "DE",
    "german": "DE",
    "allemagne": "DE",
    "deutschland": "DE",
    "fr": "FR",
    "france": "FR",
    "french": "FR",
    "it": "IT",
    "italy": "IT",
    "italia": "IT",
    "italie": "IT",
}

POLLUTANT_ALIASES = {
    "8": "NO2",
    "no2": "NO2",
    "nitrogen dioxide": "NO2",
    "nitrogen_dioxide": "NO2",
    "6001": "PM2.5",
    "pm2.5": "PM2.5",
    "pm25": "PM2.5",
    "pm2_5": "PM2.5",
    "fine particulate matter": "PM2.5",
}

BENCHMARKS = {
    "who_2021": {
        "aliases": {"who", "who2021", "who_2021", "who 2021"},
        "threshold_column": "who_2021_threshold",
        "flag_column": "above_who_2021",
    },
    "eu_2030": {
        "aliases": {"eu2030", "eu_2030", "eu 2030"},
        "threshold_column": "eu_2030_threshold",
        "flag_column": "above_eu_2030",
    },
    "eu_current": {
        "aliases": {"current", "eu_current", "eu current", "current eu"},
        "threshold_column": "eu_current_threshold",
        "flag_column": "above_eu_current",
    },
}

REQUIRED_ANNUAL_COLUMNS = {
    "country_code",
    "country_name",
    "sampling_point",
    "pollutant_code",
    "pollutant_name",
    "unit",
    "year",
    "raw_rows",
    "valid_hours",
    "annual_mean",
    "coverage_pct",
    "source_zip",
    "source_member",
    "who_2021_threshold",
    "eu_2030_threshold",
    "eu_current_threshold",
    "above_who_2021",
    "above_eu_2030",
    "above_eu_current",
    "distance_to_who_2021",
    "distance_to_eu_2030",
}

NUMERIC_COLUMNS = {
    "pollutant_code",
    "year",
    "raw_rows",
    "valid_hours",
    "annual_mean",
    "coverage_pct",
    "who_2021_threshold",
    "eu_2030_threshold",
    "eu_current_threshold",
    "distance_to_who_2021",
    "distance_to_eu_2030",
}

BOOLEAN_COLUMNS = {
    "above_who_2021",
    "above_eu_2030",
    "above_eu_current",
}


def project_root() -> Path:
    """Return the repository root when this file is located in ``src/``."""

    return Path(__file__).resolve().parents[1]


def _round(value: Any, digits: int = 2) -> float:
    return round(float(value), digits)


def _normalise_boolean(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    normalised = series.astype(str).str.strip().str.lower().map(mapping)
    if normalised.isna().any():
        bad = sorted(series[normalised.isna()].astype(str).unique().tolist())[:5]
        raise MeasurementDataError(
            f"Column '{column}' contains invalid boolean values: {bad}"
        )
    return normalised.astype(bool)


class MeasurementStore:
    """Load and query the processed annual EEA measurements.

    The store is designed to be created once when the MCP server starts and
    reused for every tool call.
    """

    def __init__(
        self,
        *,
        root: Path | None = None,
        annual_path: Path | None = None,
        excluded_path: Path | None = None,
        country_summary_path: Path | None = None,
    ) -> None:
        self.root = (root or project_root()).resolve()
        self.processed_dir = self.root / "data" / "measurements" / "processed"

        env_annual = os.getenv("AIR_QUALITY_ANNUAL_DATA")
        env_excluded = os.getenv("AIR_QUALITY_EXCLUDED_DATA")
        env_summary = os.getenv("AIR_QUALITY_COUNTRY_SUMMARY_DATA")

        self._explicit_annual = (
            Path(annual_path or env_annual).expanduser().resolve()
            if annual_path or env_annual
            else None
        )
        self.excluded_path = (
            Path(excluded_path or env_excluded).expanduser().resolve()
            if excluded_path or env_excluded
            else self.processed_dir / "eea_excluded_low_coverage_2024.csv"
        )
        self.country_summary_path = (
            Path(country_summary_path or env_summary).expanduser().resolve()
            if country_summary_path or env_summary
            else self.processed_dir / "eea_country_summary_2024.csv"
        )

        self.annual, self.annual_path = self._load_annual()
        self.excluded = self._load_excluded()
        self._validate_annual()
        self.country_summary_reconciled = self._reconcile_country_summary()

    def _annual_candidates(self) -> list[Path]:
        if self._explicit_annual is not None:
            return [self._explicit_annual]
        return [
            self.processed_dir / "eea_sampling_point_annual_2024.parquet",
            self.processed_dir / "eea_sampling_point_annual_2024.csv",
        ]

    def _load_annual(self) -> tuple[pd.DataFrame, Path]:
        errors: list[str] = []
        candidates = self._annual_candidates()
        for path in candidates:
            if not path.is_file():
                continue
            try:
                if path.suffix.lower() == ".parquet":
                    frame = pd.read_parquet(path)
                elif path.suffix.lower() == ".csv":
                    frame = pd.read_csv(path)
                else:
                    raise MeasurementDataError(
                        "Annual measurement data must be a .parquet or .csv file"
                    )
                return frame, path
            except (ImportError, ModuleNotFoundError) as exc:
                # If the Parquet engine is unavailable, continue to the CSV copy.
                errors.append(f"{path.name}: {exc}")
                continue
            except Exception as exc:
                raise MeasurementDataError(
                    f"Could not read annual measurement data '{path}': {exc}"
                ) from exc

        looked_for = ", ".join(str(path) for path in candidates)
        detail = f" Read errors: {'; '.join(errors)}" if errors else ""
        raise MeasurementDataError(
            f"Annual measurement dataset not found. Looked for: {looked_for}.{detail}"
        )

    def _load_excluded(self) -> pd.DataFrame:
        if not self.excluded_path.is_file():
            return pd.DataFrame()
        try:
            frame = pd.read_csv(self.excluded_path)
        except Exception as exc:
            raise MeasurementDataError(
                f"Could not read excluded-series data '{self.excluded_path}': {exc}"
            ) from exc
        if "country_code" in frame:
            frame["country_code"] = frame["country_code"].astype(str).str.upper()
        if "pollutant_name" in frame:
            frame["pollutant_name"] = frame["pollutant_name"].astype(str).str.upper()
            frame.loc[frame["pollutant_name"].eq("PM2.5"), "pollutant_name"] = "PM2.5"
        return frame

    def _validate_annual(self) -> None:
        missing = sorted(REQUIRED_ANNUAL_COLUMNS - set(self.annual.columns))
        if missing:
            raise MeasurementDataError(
                f"Annual measurement dataset is missing columns: {missing}"
            )
        if self.annual.empty:
            raise MeasurementDataError("Annual measurement dataset is empty")

        for column in NUMERIC_COLUMNS:
            try:
                self.annual[column] = pd.to_numeric(self.annual[column], errors="raise")
            except Exception as exc:
                raise MeasurementDataError(
                    f"Column '{column}' must contain numeric values"
                ) from exc
        for column in BOOLEAN_COLUMNS:
            self.annual[column] = _normalise_boolean(self.annual[column], column)

        self.annual["country_code"] = (
            self.annual["country_code"].astype(str).str.strip().str.upper()
        )
        pollutant = self.annual["pollutant_name"].astype(str).str.strip().str.upper()
        self.annual["pollutant_name"] = pollutant.replace({"PM2.5": "PM2.5"})

        if self.annual[list(REQUIRED_ANNUAL_COLUMNS)].isna().any().any():
            raise MeasurementDataError("Required annual measurement fields contain nulls")

        keys = ["country_code", "sampling_point", "pollutant_name", "year"]
        duplicate_count = int(self.annual.duplicated(keys).sum())
        if duplicate_count:
            raise MeasurementDataError(
                f"Annual measurement dataset contains {duplicate_count} duplicate series"
            )

        unsupported_countries = sorted(set(self.annual["country_code"]) - {"DE", "FR", "IT"})
        unsupported_pollutants = sorted(
            set(self.annual["pollutant_name"]) - {"NO2", "PM2.5"}
        )
        if unsupported_countries or unsupported_pollutants:
            raise MeasurementDataError(
                "Dataset is outside the configured scope: "
                f"countries={unsupported_countries}, pollutants={unsupported_pollutants}"
            )

    def _reconcile_country_summary(self) -> bool:
        """Check the optional six-row summary against calculations from annual data."""

        if not self.country_summary_path.is_file():
            return False
        try:
            expected = pd.read_csv(self.country_summary_path)
        except Exception as exc:
            raise MeasurementDataError(
                f"Could not read country summary '{self.country_summary_path}': {exc}"
            ) from exc

        required = {
            "country_code",
            "pollutant_name",
            "sampling_points",
            "minimum",
            "q25",
            "median",
            "q75",
            "maximum",
            "pct_above_who_2021",
            "pct_above_eu_2030",
            "pct_above_eu_current",
        }
        if required - set(expected.columns):
            raise MeasurementDataError("Country summary has an unexpected schema")

        for row in expected.to_dict(orient="records"):
            subset = self._filtered(
                str(row["country_code"]), str(row["pollutant_name"]), 2024
            )
            actual = self._aggregate(subset)
            comparisons = {
                "sampling_points": actual["sampling_points"],
                "minimum": actual["annual_mean_ug_m3"]["minimum"],
                "q25": actual["annual_mean_ug_m3"]["q25"],
                "median": actual["annual_mean_ug_m3"]["median"],
                "q75": actual["annual_mean_ug_m3"]["q75"],
                "maximum": actual["annual_mean_ug_m3"]["maximum"],
                "pct_above_who_2021": actual["benchmarks"]["who_2021"]["pct_above"],
                "pct_above_eu_2030": actual["benchmarks"]["eu_2030"]["pct_above"],
                "pct_above_eu_current": actual["benchmarks"]["eu_current"]["pct_above"],
            }
            for field, actual_value in comparisons.items():
                expected_value = float(row[field])
                tolerance = 0.0 if field == "sampling_points" else 0.02
                if abs(float(actual_value) - expected_value) > tolerance:
                    raise MeasurementDataError(
                        "Country summary reconciliation failed for "
                        f"{row['country_code']}/{row['pollutant_name']}/{field}: "
                        f"computed={actual_value}, file={expected_value}"
                    )
        return True

    @staticmethod
    def normalise_country(country: str) -> str:
        value = str(country).strip().lower()
        code = COUNTRY_ALIASES.get(value)
        if code is None:
            raise InvalidMeasurementQuery(
                "Unsupported country. Use FR/France, DE/Germany, or IT/Italy."
            )
        return code

    @staticmethod
    def normalise_pollutant(pollutant: str) -> str:
        value = str(pollutant).strip().lower()
        name = POLLUTANT_ALIASES.get(value)
        if name is None:
            raise InvalidMeasurementQuery("Unsupported pollutant. Use PM2.5 or NO2.")
        return name

    @staticmethod
    def normalise_benchmark(benchmark: str) -> str:
        value = str(benchmark).strip().lower()
        for name, config in BENCHMARKS.items():
            if value == name or value in config["aliases"]:
                return name
        raise InvalidMeasurementQuery(
            "Unsupported benchmark. Use who_2021, eu_2030, or eu_current."
        )

    def _validate_year(self, year: int) -> int:
        try:
            value = int(year)
        except (TypeError, ValueError) as exc:
            raise InvalidMeasurementQuery("Year must be an integer.") from exc
        available = sorted(int(v) for v in self.annual["year"].unique())
        if value not in available:
            raise InvalidMeasurementQuery(
                f"Year {value} is unavailable. Available years: {available}."
            )
        return value

    def _filtered(self, country: str, pollutant: str, year: int) -> pd.DataFrame:
        code = self.normalise_country(country)
        name = self.normalise_pollutant(pollutant)
        selected_year = self._validate_year(year)
        frame = self.annual[
            self.annual["country_code"].eq(code)
            & self.annual["pollutant_name"].eq(name)
            & self.annual["year"].eq(selected_year)
        ].copy()
        if frame.empty:
            raise InvalidMeasurementQuery(
                f"No retained measurements for {code}/{name}/{selected_year}."
            )
        return frame

    @staticmethod
    def _constant_value(frame: pd.DataFrame, column: str) -> float:
        values = frame[column].dropna().unique()
        if len(values) != 1:
            raise MeasurementDataError(
                f"Expected one value for '{column}', found {values.tolist()}"
            )
        return _round(values[0], 2)

    def _aggregate(self, frame: pd.DataFrame) -> dict[str, Any]:
        annual = frame["annual_mean"]
        benchmarks: dict[str, Any] = {}
        for name, config in BENCHMARKS.items():
            flag = config["flag_column"]
            count = int(frame[flag].sum())
            benchmarks[name] = {
                "threshold_ug_m3": self._constant_value(
                    frame, config["threshold_column"]
                ),
                "sampling_points_above": count,
                "pct_above": _round(100 * count / len(frame), 2),
            }
        return {
            "sampling_points": int(len(frame)),
            "unit": str(frame["unit"].iloc[0]),
            "annual_mean_ug_m3": {
                "minimum": _round(annual.min(), 2),
                "q25": _round(annual.quantile(0.25), 2),
                "median": _round(annual.median(), 2),
                "mean": _round(annual.mean(), 2),
                "q75": _round(annual.quantile(0.75), 2),
                "maximum": _round(annual.max(), 2),
            },
            "coverage_pct": {
                "minimum": _round(frame["coverage_pct"].min(), 2),
                "median": _round(frame["coverage_pct"].median(), 2),
                "maximum": _round(frame["coverage_pct"].max(), 2),
            },
            "benchmarks": benchmarks,
        }

    def _excluded_summary(self, country_code: str, pollutant: str, year: int) -> dict[str, Any]:
        if self.excluded.empty:
            return {"available": False, "sampling_points_excluded": None}
        required = {"country_code", "pollutant_name", "year", "coverage_pct"}
        if required - set(self.excluded.columns):
            return {"available": False, "sampling_points_excluded": None}
        frame = self.excluded[
            self.excluded["country_code"].eq(country_code)
            & self.excluded["pollutant_name"].eq(pollutant)
            & pd.to_numeric(self.excluded["year"], errors="coerce").eq(year)
        ]
        result: dict[str, Any] = {
            "available": True,
            "sampling_points_excluded": int(len(frame)),
        }
        if not frame.empty:
            coverage = pd.to_numeric(frame["coverage_pct"], errors="coerce")
            result["excluded_coverage_pct"] = {
                "minimum": _round(coverage.min(), 2),
                "median": _round(coverage.median(), 2),
                "maximum": _round(coverage.max(), 2),
            }
        return result

    def get_country_air_quality(
        self, country: str, pollutant: str, year: int = 2024
    ) -> dict[str, Any]:
        """Return deterministic annual station statistics for one country/pollutant."""

        frame = self._filtered(country, pollutant, year)
        code = str(frame["country_code"].iloc[0])
        name = str(frame["pollutant_name"].iloc[0])
        selected_year = int(frame["year"].iloc[0])
        result = {
            "country_code": code,
            "country_name": str(frame["country_name"].iloc[0]),
            "pollutant": name,
            "year": selected_year,
            **self._aggregate(frame),
            "data_quality": {
                "retention_rule": "coverage_pct >= 75",
                **self._excluded_summary(code, name, selected_year),
            },
            "provenance": {
                "annual_dataset": self.annual_path.name,
                "annual_format": self.annual_path.suffix.lstrip(".").lower(),
                "raw_archives": sorted(frame["source_zip"].astype(str).unique().tolist()),
                "country_summary_reconciled": self.country_summary_reconciled,
            },
            "interpretation_limits": [
                "Statistics describe retained sampling points, not population exposure.",
                "Country summaries are unweighted across sampling points.",
            ],
        }
        return result

    @staticmethod
    def _parse_countries(countries: str | Iterable[str] | None) -> list[str]:
        if countries is None:
            return ["FR", "DE", "IT"]
        if isinstance(countries, str):
            values = [part.strip() for part in countries.split(",") if part.strip()]
        else:
            values = [str(part).strip() for part in countries if str(part).strip()]
        if not values:
            raise InvalidMeasurementQuery("At least one country is required.")
        return values

    def compare_countries(
        self,
        pollutant: str,
        countries: str | Iterable[str] | None = "FR,DE,IT",
        year: int = 2024,
        benchmark: str = "who_2021",
        rank_by: str = "median",
    ) -> dict[str, Any]:
        """Compare annual station statistics across the selected countries."""

        benchmark_name = self.normalise_benchmark(benchmark)
        ranking_field = str(rank_by).strip().lower()
        if ranking_field not in {"median", "pct_above"}:
            raise InvalidMeasurementQuery("rank_by must be median or pct_above.")

        normalised_codes: list[str] = []
        for value in self._parse_countries(countries):
            code = self.normalise_country(value)
            if code not in normalised_codes:
                normalised_codes.append(code)

        rows: list[dict[str, Any]] = []
        for code in normalised_codes:
            summary = self.get_country_air_quality(code, pollutant, year)
            benchmark_result = summary["benchmarks"][benchmark_name]
            rows.append(
                {
                    "country_code": summary["country_code"],
                    "country_name": summary["country_name"],
                    "sampling_points": summary["sampling_points"],
                    "excluded_low_coverage": summary["data_quality"][
                        "sampling_points_excluded"
                    ],
                    "median_ug_m3": summary["annual_mean_ug_m3"]["median"],
                    "mean_ug_m3": summary["annual_mean_ug_m3"]["mean"],
                    "minimum_ug_m3": summary["annual_mean_ug_m3"]["minimum"],
                    "maximum_ug_m3": summary["annual_mean_ug_m3"]["maximum"],
                    "benchmark": benchmark_name,
                    "benchmark_threshold_ug_m3": benchmark_result[
                        "threshold_ug_m3"
                    ],
                    "sampling_points_above": benchmark_result[
                        "sampling_points_above"
                    ],
                    "pct_above": benchmark_result["pct_above"],
                }
            )

        key = "median_ug_m3" if ranking_field == "median" else "pct_above"
        rows.sort(key=lambda row: (-float(row[key]), row["country_code"]))
        for index, row in enumerate(rows, start=1):
            row["rank"] = index

        pollutant_name = self.normalise_pollutant(pollutant)
        return {
            "pollutant": pollutant_name,
            "year": self._validate_year(year),
            "benchmark": benchmark_name,
            "rank_by": ranking_field,
            "countries": rows,
            "highest": rows[0]["country_code"],
            "interpretation_limits": [
                "Ranking is based on retained sampling points and is not population-weighted.",
                "Sampling-point counts differ between countries.",
            ],
            "provenance": {"annual_dataset": self.annual_path.name},
        }

    def find_station_extremes(
        self,
        country: str,
        pollutant: str,
        year: int = 2024,
        direction: str = "highest",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Return the highest or lowest annual sampling-point means."""

        direction_name = str(direction).strip().lower()
        if direction_name not in {"highest", "lowest"}:
            raise InvalidMeasurementQuery("direction must be highest or lowest.")
        try:
            selected_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise InvalidMeasurementQuery("limit must be an integer.") from exc
        if selected_limit < 1 or selected_limit > 20:
            raise InvalidMeasurementQuery("limit must be between 1 and 20.")

        frame = self._filtered(country, pollutant, year)
        ascending = direction_name == "lowest"
        selected = frame.sort_values(
            ["annual_mean", "sampling_point"], ascending=[ascending, True]
        ).head(selected_limit)

        records: list[dict[str, Any]] = []
        for rank, row in enumerate(selected.to_dict(orient="records"), start=1):
            records.append(
                {
                    "rank": rank,
                    "sampling_point": str(row["sampling_point"]),
                    "annual_mean_ug_m3": _round(row["annual_mean"], 3),
                    "coverage_pct": _round(row["coverage_pct"], 2),
                    "valid_hours": int(row["valid_hours"]),
                    "above_who_2021": bool(row["above_who_2021"]),
                    "above_eu_2030": bool(row["above_eu_2030"]),
                    "above_eu_current": bool(row["above_eu_current"]),
                    "distance_to_who_2021_ug_m3": _round(
                        row["distance_to_who_2021"], 3
                    ),
                    "distance_to_eu_2030_ug_m3": _round(
                        row["distance_to_eu_2030"], 3
                    ),
                    "source_member": str(row["source_member"]),
                }
            )

        return {
            "country_code": str(frame["country_code"].iloc[0]),
            "country_name": str(frame["country_name"].iloc[0]),
            "pollutant": str(frame["pollutant_name"].iloc[0]),
            "year": int(frame["year"].iloc[0]),
            "direction": direction_name,
            "results": records,
            "warning": (
                "Sampling-point identifiers are not city names; location metadata is not "
                "included in the current project dataset."
            ),
            "provenance": {"annual_dataset": self.annual_path.name},
        }


__all__ = [
    "InvalidMeasurementQuery",
    "MeasurementDataError",
    "MeasurementStore",
]
