"""Load local, Thai-government, or cached/downloaded USGS observations."""

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def load_csv(source: dict[str, Any], root: Path) -> pd.DataFrame:
    path = _resolve(source["path"], root)
    if not path.exists():
        raise FileNotFoundError(f"Time-series CSV not found: {path}")
    return pd.read_csv(
        path,
        sep=source.get("separator", ","),
        encoding=source.get("encoding", "utf-8"),
        decimal=source.get("decimal", "."),
        thousands=source.get("thousands"),
    )


def _read_thai_csv(source: dict[str, Any], root: Path) -> pd.DataFrame:
    """Read a Thai CSV using explicit or common Thai encodings."""
    path = _resolve(source["path"], root)
    if not path.exists():
        raise FileNotFoundError(f"Thai time-series CSV not found: {path}")
    encodings = source.get("encodings", ["utf-8-sig", "utf-8", "cp874", "tis-620"])
    failures = []
    for encoding in encodings:
        try:
            skiprows = source.get("skiprows", 0)
            header_starts_with = source.get("header_starts_with")
            if header_starts_with:
                with path.open("r", encoding=encoding) as handle:
                    for line_number, line in enumerate(handle):
                        if line.lstrip("\ufeff").startswith(header_starts_with):
                            skiprows = line_number
                            break
                    else:
                        raise ValueError(
                            f"Thai CSV header beginning with {header_starts_with!r} was not found"
                        )
            return pd.read_csv(
                path,
                sep=source.get("separator", ","),
                encoding=encoding,
                decimal=source.get("decimal", "."),
                thousands=source.get("thousands"),
                skiprows=skiprows,
                na_values=source.get("missing_values"),
            )
        except UnicodeDecodeError as exc:
            failures.append(f"{encoding}: {exc}")
    raise UnicodeError("Could not decode Thai CSV with configured encodings: " + "; ".join(failures))


def _convert_buddhist_year(value: Any) -> Any:
    """Convert a four-digit Thai Buddhist year in a date string to Gregorian."""
    import re

    if pd.isna(value):
        return value
    text = str(value)
    match = re.search(r"(?<!\d)(2[4-6]\d{2})(?!\d)", text)
    if not match:
        return text
    gregorian = int(match.group(1)) - 543
    return text[: match.start()] + str(gregorian) + text[match.end() :]


def load_thai_csv(source: dict[str, Any], root: Path) -> pd.DataFrame:
    """Normalize a local Thai agency CSV into the standard forecasting schema."""
    frame = _read_thai_csv(source, root)
    for column, required_value in source.get("filters", {}).items():
        if column not in frame:
            raise ValueError(f"Thai CSV filter column is missing: {column}")
        frame = frame[frame[column].astype(str).str.strip() == str(required_value).strip()]
    if frame.empty:
        raise ValueError("Thai CSV contains no rows after applying station/province filters")

    timestamp_column = source["timestamp_column"]
    timestamp_from = source.get("timestamp_from")
    timestamp_from_year_day = source.get("timestamp_from_year_day")
    if timestamp_from_year_day:
        year_column = timestamp_from_year_day["year"]
        day_column = timestamp_from_year_day["day_of_year"]
        missing = [column for column in (year_column, day_column) if column not in frame]
        if missing:
            raise ValueError(
                f"Thai CSV year/day timestamp columns are missing: {', '.join(missing)}"
            )
        years = pd.to_numeric(frame[year_column], errors="coerce").astype("Int64").astype(str)
        days = (
            pd.to_numeric(frame[day_column], errors="coerce")
            .astype("Int64")
            .astype(str)
            .str.zfill(3)
        )
        frame[timestamp_column] = pd.to_datetime(
            years + days,
            format="%Y%j",
            errors="coerce",
        )
    if timestamp_from:
        missing = [column for column in timestamp_from if column not in frame]
        if missing:
            raise ValueError(f"Thai CSV timestamp source columns are missing: {', '.join(missing)}")
        frame[timestamp_column] = frame[timestamp_from].astype(str).agg(" ".join, axis=1)

    frame = frame.rename(columns=source.get("column_map", {}))
    if source.get("buddhist_year", False):
        if timestamp_column not in frame:
            raise ValueError(f"Thai CSV timestamp column is missing: {timestamp_column}")
        frame[timestamp_column] = frame[timestamp_column].map(_convert_buddhist_year)
    return frame


def _download_usgs(source: dict[str, Any]) -> pd.DataFrame:
    params = {
        "f": "json",
        "monitoring_location_id": source.get(
            "monitoring_location_id", f"USGS-{source['site_no']}"
        ),
        "parameter_code": source.get("parameter_code", "72019"),
        "statistic_id": source.get("statistic_id", "00003"),
        "datetime": f"{source['start_date']}/{source['end_date']}",        
        "properties": "value,time,approval_status,qualifier",
        "skipGeometry": "true",
        "limit": 50000,
    }
    endpoint = source.get(
        "endpoint",
        "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items",
    )
    request = urllib.request.Request(
        endpoint + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": "CrewAI-rebuild forecasting study", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    observations = [feature.get("properties", {}) for feature in payload.get("features", [])]
    rows = [
        {
            source["timestamp_column"]: item["time"],
            source["target_column"]: item["value"],
            "approval_status": item.get("approval_status"),
            "qualifier": item.get("qualifier"),
        }
        for item in observations
        if item.get("value") not in {None, "", "-999999"}
    ]
    if not rows:
        raise ValueError("USGS OGC daily API returned no usable observations")
    return pd.DataFrame(rows)


def load_usgs(source: dict[str, Any], root: Path, refresh: bool = False) -> pd.DataFrame:
    cache_path = _resolve(source["cache_path"], root)
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)
    frame = _download_usgs(source)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def load_source(config: dict[str, Any], root: Path, refresh: bool = False) -> pd.DataFrame:
    source = config["data"]
    kind = source["source"].lower()
    if kind in {"csv", "jena"}:
        return load_csv(source, root)
    if kind == "thai_csv":
        return load_thai_csv(source, root)
    if kind == "usgs":
        return load_usgs(source, root, refresh=refresh)
    raise ValueError(f"Unsupported forecasting data source: {kind}")
