"""ERCOT price loading, normalisation, and the DuckDB store.

Canonical schema for both `rtm` (15-min) and `dam` (hourly) tables:
    location              settlement point, e.g. HB_WEST
    interval_start_utc    timestamp, UTC (unambiguous across DST)
    interval_start_local  naive Central wall-clock time
    delivery_date         local (Central) calendar date of the interval
    spp                   settlement point price, $/MWh
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from bess.config import Config

MARKETS = {
    "rtm": {"per_hour": 4, "gridstatus_method": "get_rtm_spp"},
    "dam": {"per_hour": 1, "gridstatus_method": "get_dam_spp"},
}


def standardize(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Map a gridstatus SPP frame (Interval Start, Location, SPP) to the canonical schema."""
    start = pd.to_datetime(df["Interval Start"])
    if start.dt.tz is None:
        raise ValueError("Interval Start must be timezone-aware")
    local = start.dt.tz_convert(tz)
    out = pd.DataFrame(
        {
            "location": df["Location"].astype(str),
            "interval_start_utc": start.dt.tz_convert("UTC"),
            "interval_start_local": local.dt.tz_localize(None),
            "delivery_date": local.dt.date,
            "spp": df["SPP"].astype(float),
        }
    )
    return out.sort_values(["location", "interval_start_utc"]).reset_index(drop=True)


def raw_path(cfg: Config, market: str, year: int) -> Path:
    return cfg.path("raw_dir") / f"{market}_{year}.parquet"


def fetch_year(market: str, year: int, cfg: Config) -> pd.DataFrame:
    """Download one year from ERCOT through gridstatus, keep configured hubs.

    Uses the yearly historical archive (NP6-785-ER / NP4-180-ER). Needs network
    access to ercot.com. The in-progress year may not have an archive yet; the
    caller handles that (see scripts/load_data.py).
    """
    from gridstatus import Ercot

    method = getattr(Ercot(), MARKETS[market]["gridstatus_method"])
    df = method(year)
    df = df[df["Location"].isin(cfg.data["hubs"])]
    return standardize(df, cfg.data["timezone"])


def local_day_hours(d: date, tz: str) -> int:
    """Hours in the local calendar day: 24 normally, 23 / 25 on DST changes."""
    start = pd.Timestamp(d).tz_localize(tz)
    end = (pd.Timestamp(d) + pd.DateOffset(days=1)).tz_localize(tz)
    return int(round((end - start).total_seconds() / 3600))


def calendar(start: date, end: date, tz: str) -> pd.DataFrame:
    """Expected interval counts per local day, DST-aware."""
    days = pd.date_range(start, end, freq="D")
    hours = [local_day_hours(d.date(), tz) for d in days]
    return pd.DataFrame(
        {
            "delivery_date": [d.date() for d in days],
            "hours": hours,
            "expected_rtm": [h * MARKETS["rtm"]["per_hour"] for h in hours],
            "expected_dam": [h * MARKETS["dam"]["per_hour"] for h in hours],
        }
    )


def build_db(cfg: Config) -> duckdb.DuckDBPyConnection:
    """(Re)build the DuckDB store from the Parquet files in raw_dir."""
    raw_dir = cfg.path("raw_dir")
    db_path = cfg.path("db_path")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    for market in MARKETS:
        files = sorted(raw_dir.glob(f"{market}_*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {market}_*.parquet in {raw_dir}")
        con.execute(
            f"CREATE OR REPLACE TABLE {market} AS "
            f"SELECT * FROM read_parquet({[str(f) for f in files]}) "
            "WHERE delivery_date >= ? ORDER BY location, interval_start_utc",
            [cfg.data["start_date"]],
        )
    bounds = con.execute(
        "SELECT min(delivery_date), max(delivery_date) FROM rtm"
    ).fetchone()
    cal = calendar(bounds[0], bounds[1], cfg.data["timezone"])
    con.register("cal_df", cal)
    con.execute("CREATE OR REPLACE TABLE calendar AS SELECT * FROM cal_df")
    return con
