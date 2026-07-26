from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


YEAR = 2024
EXPECTED_HOURS = 8_784  # 2024 is a leap year.
MIN_COVERAGE_PCT = 75.0

COUNTRIES = {
    "FR": "France",
    "DE": "Germany",
    "IT": "Italy",
}

POLLUTANTS = {
    6001: "PM2.5",
    8: "NO2",
}

THRESHOLDS = {
    6001: {"who_2021": 5.0, "eu_2030": 10.0, "eu_current": 25.0},
    8: {"who_2021": 10.0, "eu_2030": 20.0, "eu_current": 40.0},
}

# Eionet observation validity: 1, 2, and 3 are valid observations.
VALIDITY_CODES = {1, 2, 3}
VERIFIED_CODE = 1

REQUIRED_COLUMNS = {
    "Samplingpoint",
    "Pollutant",
    "Start",
    "End",
    "Value",
    "Unit",
    "AggType",
    "Validity",
    "Verification",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_member(zip_file: zipfile.ZipFile, member: str) -> pd.DataFrame:
    data = zip_file.read(member)
    table = pq.read_table(io.BytesIO(data))
    return table.to_pandas()


def normalize_and_summarize(
    frame: pd.DataFrame,
    source_zip: str,
    source_member: str,
) -> list[dict]:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{source_zip}/{source_member} is missing columns: {sorted(missing)}"
        )

    raw_rows = len(frame)
    df = pd.DataFrame(
        {
            "sampling_point": frame["Samplingpoint"].astype("string").str.strip(),
            "pollutant_code": pd.to_numeric(frame["Pollutant"], errors="coerce"),
            "start_time": pd.to_datetime(frame["Start"], errors="coerce"),
            "end_time": pd.to_datetime(frame["End"], errors="coerce"),
            "value": pd.to_numeric(frame["Value"], errors="coerce"),
            "unit": frame["Unit"].astype("string").str.strip(),
            "agg_type": frame["AggType"].astype("string").str.lower().str.strip(),
            "validity": pd.to_numeric(frame["Validity"], errors="coerce"),
            "verification": pd.to_numeric(frame["Verification"], errors="coerce"),
        }
    )

    if "ResultTime" in frame.columns:
        df["result_time"] = pd.to_datetime(frame["ResultTime"], errors="coerce")
    else:
        df["result_time"] = pd.NaT

    df["country_code"] = (
        df["sampling_point"].str.extract(r"^([A-Za-z]{2})", expand=False).str.upper()
    )

    df = df[
        df["country_code"].isin(COUNTRIES)
        & df["pollutant_code"].isin(POLLUTANTS)
        & (df["start_time"].dt.year == YEAR)
        & (df["agg_type"] == "hour")
        & df["validity"].isin(VALIDITY_CODES)
        & (df["verification"] == VERIFIED_CODE)
        & df["value"].notna()
    ].copy()

    if df.empty:
        return []

    # When the same timestamp was reported more than once, retain the latest report.
    df = df.sort_values("result_time", na_position="first")
    df = df.drop_duplicates(
        subset=["sampling_point", "pollutant_code", "start_time"],
        keep="last",
    )

    summaries: list[dict] = []
    group_columns = ["country_code", "sampling_point", "pollutant_code", "unit"]
    for (country, sampling_point, pollutant, unit), group in df.groupby(
        group_columns, dropna=False
    ):
        valid_hours = int(group["start_time"].nunique())
        summaries.append(
            {
                "country_code": str(country),
                "sampling_point": str(sampling_point),
                "pollutant_code": int(pollutant),
                "unit": str(unit),
                "year": YEAR,
                "raw_rows": raw_rows,
                "valid_hours": valid_hours,
                "annual_mean": float(group["value"].mean()),
                "first_observation": group["start_time"].min(),
                "last_observation_end": group["end_time"].max(),
                "source_zip": source_zip,
                "source_member": source_member,
            }
        )

    return summaries


def prepare() -> None:
    root = project_root()
    measurement_dir = root / "data" / "measurements"
    output_dir = measurement_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted(measurement_dir.glob("raw_*_2024.zip"))
    if not zip_paths:
        raise FileNotFoundError(
            f"No raw_*_2024.zip files found in {measurement_dir}"
        )

    print(f"ZIP files found: {len(zip_paths)}")
    records: list[dict] = []
    parquet_files_read = 0

    for zip_path in zip_paths:
        print(f"Reading {zip_path.name} ...")
        try:
            with zipfile.ZipFile(zip_path) as archive:
                listed_members = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(".parquet")
                ]
                members = sorted(set(listed_members))
                if not members:
                    raise ValueError(f"No Parquet files found in {zip_path.name}")

                repeated_members = len(listed_members) - len(members)
                if repeated_members:
                    print(
                        f"  Ignoring {repeated_members} repeated archive member name(s)"
                    )

                for member in members:
                    frame = load_member(archive, member)
                    records.extend(
                        normalize_and_summarize(frame, zip_path.name, member)
                    )
                    parquet_files_read += 1
                    if parquet_files_read % 100 == 0:
                        print(f"  {parquet_files_read} Parquet files read")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid ZIP file: {zip_path}") from exc

    if not records:
        raise RuntimeError("No valid 2024 observations were found.")

    annual = pd.DataFrame(records).drop_duplicates()
    key = ["country_code", "sampling_point", "pollutant_code", "year"]

    # Multiple source files for the same series need timestamp-level merging. Stop
    # instead of silently double-counting them.
    duplicate_path = output_dir / "duplicate_series_to_review.csv"
    duplicates = annual[annual.duplicated(key, keep=False)].sort_values(key)
    if not duplicates.empty:
        duplicates.to_csv(duplicate_path, index=False)
        raise RuntimeError(
            "Some sampling-point series occur in multiple Parquet files. "
            f"Review {duplicate_path}; no annual output was written."
        )
    if duplicate_path.exists():
        duplicate_path.unlink()

    annual["coverage_pct"] = annual["valid_hours"] * 100.0 / EXPECTED_HOURS
    annual["country_name"] = annual["country_code"].map(COUNTRIES)
    annual["pollutant_name"] = annual["pollutant_code"].map(POLLUTANTS)

    found_pairs = set(
        annual[["country_code", "pollutant_code"]].itertuples(index=False, name=None)
    )
    expected_pairs = {
        (country, pollutant)
        for country in COUNTRIES
        for pollutant in POLLUTANTS
    }
    missing_pairs = expected_pairs.difference(found_pairs)
    if missing_pairs:
        formatted = ", ".join(
            f"{country}/{POLLUTANTS[pollutant]}"
            for country, pollutant in sorted(missing_pairs)
        )
        raise RuntimeError(f"Missing downloaded data for: {formatted}")

    units = sorted(annual["unit"].dropna().unique().tolist())
    if units != ["ug.m-3"]:
        raise RuntimeError(f"Unexpected units found: {units}")

    excluded = annual[annual["coverage_pct"] < MIN_COVERAGE_PCT].copy()
    kept = annual[annual["coverage_pct"] >= MIN_COVERAGE_PCT].copy()

    kept["who_2021_threshold"] = kept["pollutant_code"].map(
        lambda code: THRESHOLDS[code]["who_2021"]
    )
    kept["eu_2030_threshold"] = kept["pollutant_code"].map(
        lambda code: THRESHOLDS[code]["eu_2030"]
    )
    kept["eu_current_threshold"] = kept["pollutant_code"].map(
        lambda code: THRESHOLDS[code]["eu_current"]
    )
    kept["above_who_2021"] = kept["annual_mean"] > kept["who_2021_threshold"]
    kept["above_eu_2030"] = kept["annual_mean"] > kept["eu_2030_threshold"]
    kept["above_eu_current"] = (
        kept["annual_mean"] > kept["eu_current_threshold"]
    )
    kept["distance_to_who_2021"] = (
        kept["annual_mean"] - kept["who_2021_threshold"]
    )
    kept["distance_to_eu_2030"] = (
        kept["annual_mean"] - kept["eu_2030_threshold"]
    )

    kept["annual_mean"] = kept["annual_mean"].round(3)
    kept["coverage_pct"] = kept["coverage_pct"].round(2)
    kept["distance_to_who_2021"] = kept["distance_to_who_2021"].round(3)
    kept["distance_to_eu_2030"] = kept["distance_to_eu_2030"].round(3)
    kept = kept.sort_values(["country_code", "pollutant_code", "sampling_point"])

    summary = (
        kept.groupby(["country_code", "country_name", "pollutant_name"], observed=True)
        .agg(
            sampling_points=("sampling_point", "nunique"),
            minimum=("annual_mean", "min"),
            q25=("annual_mean", lambda values: values.quantile(0.25)),
            median=("annual_mean", "median"),
            q75=("annual_mean", lambda values: values.quantile(0.75)),
            maximum=("annual_mean", "max"),
            pct_above_who_2021=("above_who_2021", lambda values: values.mean() * 100),
            pct_above_eu_2030=("above_eu_2030", lambda values: values.mean() * 100),
            pct_above_eu_current=(
                "above_eu_current",
                lambda values: values.mean() * 100,
            ),
        )
        .reset_index()
    )

    numeric_summary_columns = [
        "minimum",
        "q25",
        "median",
        "q75",
        "maximum",
        "pct_above_who_2021",
        "pct_above_eu_2030",
        "pct_above_eu_current",
    ]
    summary[numeric_summary_columns] = summary[numeric_summary_columns].round(2)

    annual_parquet = output_dir / "eea_sampling_point_annual_2024.parquet"
    annual_csv = output_dir / "eea_sampling_point_annual_2024.csv"
    summary_csv = output_dir / "eea_country_summary_2024.csv"
    excluded_csv = output_dir / "eea_excluded_low_coverage_2024.csv"

    kept.to_parquet(annual_parquet, index=False)
    kept.to_csv(annual_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    excluded.to_csv(excluded_csv, index=False)

    print("\nPreparation complete")
    print(f"Parquet source files read: {parquet_files_read}")
    print(f"Sampling-point series before coverage filter: {len(annual)}")
    print(f"Sampling-point series retained: {len(kept)}")
    print(f"Sampling-point series excluded (<75%): {len(excluded)}")
    print(f"Annual dataset: {annual_parquet}")
    print(f"Country summary: {summary_csv}")
    print("\nCountry/pollutant summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        prepare()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc