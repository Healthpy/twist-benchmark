#!/usr/bin/env python
"""Build the real covariate profiles of the semi-synthetic energy worlds (plan §Week 9).

Source: Open Power System Data (CC-BY 4.0), Netherlands, hourly, 2016-01-01 to 2019-12-31 UTC.

* ``load``  — ``NL_load_actual_entsoe_transparency`` (ENTSO-E Transparency, MW), divided by its
  mean over the period (dimensionless, ≈ 1).
* ``temp``  — ``NL_temperature`` (°C, population-weighted MERRA-2 reanalysis).
* ``ghi``   — ``NL_radiation_direct_horizontal + NL_radiation_diffuse_horizontal`` (W/m² → kW/m²),
  global horizontal irradiance. PV is derived from irradiance because the OPSD NL solar
  generation series is incomplete (it collapses in 2019).

Usage::

    python scripts/fetch_energy_profiles.py                    # download (≈360 MB) and build
    python scripts/fetch_energy_profiles.py --ts ts.csv --weather weather.csv   # local copies

Writes ``src/twist/data/energy/opsd_nl_2016_2019.npz`` and a ``.json`` provenance file with the
SHA-256 of the written arrays (checked by ``twist.worlds.semi_synthetic.load_profiles``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

TS_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)
WEATHER_URL = "https://data.open-power-system-data.org/weather_data/2020-09-16/weather_data.csv"
OUT = Path(__file__).resolve().parents[1] / "src/twist/data/energy/opsd_nl_2016_2019.npz"
START, END = "2016-01-01", "2019-12-31"


def _get(url: str, path: Path | None, tmp: Path) -> Path:
    if path is not None:
        return path
    dest = tmp / url.rsplit("/", 1)[-1]
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def build(ts_csv: Path, weather_csv: Path, out: Path = OUT) -> Path:
    ts = pd.read_csv(
        ts_csv,
        usecols=["utc_timestamp", "NL_load_actual_entsoe_transparency"],
        parse_dates=["utc_timestamp"],
    ).set_index("utc_timestamp")
    w = pd.read_csv(
        weather_csv,
        usecols=[
            "utc_timestamp",
            "NL_temperature",
            "NL_radiation_direct_horizontal",
            "NL_radiation_diffuse_horizontal",
        ],
        parse_dates=["utc_timestamp"],
    ).set_index("utc_timestamp")
    df = ts.join(w, how="inner").loc[START:END]
    if df.isna().any().any() or len(df) != len(pd.date_range(START, f"{END} 23:00", freq="h")):
        raise ValueError("expected a complete hourly NL record for 2016–2019")
    load = df["NL_load_actual_entsoe_transparency"].to_numpy(float)
    arrays = {
        "load": (load / load.mean()).astype(np.float32),
        "temp": df["NL_temperature"].to_numpy(np.float32),
        "ghi": (
            (df["NL_radiation_direct_horizontal"] + df["NL_radiation_diffuse_horizontal"]) / 1000.0
        ).to_numpy(np.float32),
        "hour_utc": df.index.hour.to_numpy(np.int8),
        "month": df.index.month.to_numpy(np.int8),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    digest = hashlib.sha256(b"".join(arrays[k].tobytes() for k in sorted(arrays))).hexdigest()
    meta = {
        "name": "opsd_nl_2016_2019",
        "hours": len(df),
        "start_utc": str(df.index[0]),
        "end_utc": str(df.index[-1]),
        "load_mean_mw": float(load.mean()),
        "sources": {
            "time_series": TS_URL,
            "weather_data": WEATHER_URL,
            "columns": list(df.columns),
        },
        "license": "CC-BY 4.0 (Open Power System Data; ENTSO-E Transparency; MERRA-2 via OPSD)",
        "citation": "Open Power System Data. 2020. Data Package Time series / Weather Data. "
        "Versions 2020-10-06 / 2020-09-16. https://doi.org/10.25832/time_series/2020-10-06, "
        "https://doi.org/10.25832/weather_data/2020-09-16",
        "arrays_sha256": digest,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB), sha256 {digest[:12]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ts", type=Path, help="local time_series_60min_singleindex.csv")
    ap.add_argument("--weather", type=Path, help="local weather_data.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        build(_get(TS_URL, a.ts, Path(d)), _get(WEATHER_URL, a.weather, Path(d)), a.out)


if __name__ == "__main__":
    main()
