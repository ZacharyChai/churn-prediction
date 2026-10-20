"""Download ERCOT RTM (15-min) and DAM (hourly) hub prices to Parquet.

    python scripts/load_data.py              # every year from config start_date to now
    python scripts/load_data.py --years 2025 # one year

Needs network access to ercot.com (via gridstatus). The current year may have no
yearly archive yet (NP6-785-ER / NP4-180-ER publish after year end); in that case
the script falls back to gridstatus' recent-data reader, month by month. That
fallback path has not been exercised against live data: check its output with
scripts/check_data.py before trusting it.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bess.config import load_config  # noqa: E402
from bess.data import MARKETS, fetch_year, raw_path, standardize  # noqa: E402


def fetch_recent(market: str, year: int, cfg) -> pd.DataFrame:
    from gridstatus import Ercot, Markets

    gs_market = (
        Markets.REAL_TIME_15_MIN if market == "rtm" else Markets.DAY_AHEAD_HOURLY
    )
    ercot = Ercot()
    today = pd.Timestamp(date.today())
    months = pd.period_range(f"{year}-01", min(today, pd.Timestamp(f"{year}-12-31")), freq="M")
    parts = []
    for m in months:
        start = m.to_timestamp()
        end = min(start + pd.offsets.MonthBegin(1), today)
        df = ercot.get_spp(
            start, end, market=gs_market, location_type="Trading Hub"
        )
        parts.append(df[df["Location"].isin(cfg.data["hubs"])])
    return standardize(pd.concat(parts), cfg.data["timezone"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    start_year = cfg.data["start_date"].year
    years = args.years or list(range(start_year, date.today().year + 1))
    cfg.path("raw_dir").mkdir(parents=True, exist_ok=True)
    for market in MARKETS:
        for year in years:
            try:
                df = fetch_year(market, year, cfg)
                source = "yearly archive"
            except Exception as exc:  # archive missing for in-progress year, or network error
                if year != date.today().year:
                    raise
                print(f"{market} {year}: archive unavailable ({exc!r}); trying recent-data reader")
                df = fetch_recent(market, year, cfg)
                source = "recent-data reader"
            df.to_parquet(raw_path(cfg, market, year), index=False)
            print(f"{market} {year}: {len(df):,} rows from {source}")


if __name__ == "__main__":
    main()
