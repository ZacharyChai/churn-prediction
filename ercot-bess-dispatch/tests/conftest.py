"""Synthetic ERCOT-shaped data for tests. Never used for reported results."""
import sys
from dataclasses import replace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bess.config import load_config  # noqa: E402
from bess.data import calendar, standardize  # noqa: E402

TZ = "America/Chicago"


def local_day_index(day: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    """All interval starts of one local day, DST-correct (92 / 96 / 100 at 15-min)."""
    start = day.tz_localize(TZ)
    end = (day + pd.DateOffset(days=1)).tz_localize(TZ)
    return pd.date_range(start, end, freq=freq, inclusive="left")


def synthetic_prices(days, hubs=("HB_WEST",), seed=0, negative_days=()):
    rtm, dam = [], []
    for hub in hubs:
        for day in days:
            day = pd.Timestamp(day)
            rng = np.random.default_rng([seed, day.toordinal(), hash(hub) % 1000])
            for freq, store, noise in (("15min", rtm, 6.0), ("60min", dam, 3.0)):
                idx = local_day_index(day, freq)
                hour = idx.tz_convert(TZ).hour + idx.tz_convert(TZ).minute / 60
                price = 30 + 25 * np.cos(2 * np.pi * (hour - 19) / 24)
                price = price + rng.normal(0, noise, len(idx))
                if day in {pd.Timestamp(d) for d in negative_days}:
                    price = price - 80
                store.append(
                    pd.DataFrame({"Interval Start": idx, "Location": hub, "SPP": price})
                )
    return (
        standardize(pd.concat(rtm), TZ),
        standardize(pd.concat(dam), TZ),
    )


def make_db(rtm, dam, tz=TZ):
    con = duckdb.connect()
    con.register("rtm_df", rtm)
    con.register("dam_df", dam)
    con.execute("CREATE TABLE rtm AS SELECT * FROM rtm_df")
    con.execute("CREATE TABLE dam AS SELECT * FROM dam_df")
    lo, hi = rtm["delivery_date"].min(), rtm["delivery_date"].max()
    cal = calendar(lo, hi, tz)
    con.register("cal_df", cal)
    con.execute("CREATE TABLE calendar AS SELECT * FROM cal_df")
    return con


@pytest.fixture(scope="session")
def cfg():
    base = load_config()
    return replace(
        base,
        scenarios=[base.scenarios[0]],
        data={**base.data, "hubs": ["HB_WEST"]},
        solver={**base.solver, "workers": 1},
    )


@pytest.fixture(scope="session")
def asset(cfg):
    return cfg.asset
