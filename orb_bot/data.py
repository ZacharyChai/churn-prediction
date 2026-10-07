"""Loading 1-minute bars from CSV and generating synthetic bars for tests/demos."""

from __future__ import annotations

import math
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import ContractSpec, ES
from .models import Bar

_TS_CANDIDATES = ("timestamp", "datetime", "date_time", "time", "date", "ts")
_COL_CANDIDATES = {
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "last"),
    "volume": ("volume", "vol", "v"),
}


def _pick(columns: list[str], candidates: tuple[str, ...], required: bool = True) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    if required:
        raise ValueError(f"CSV is missing a column for {candidates[0]!r}; columns are {columns}")
    return None


def load_bars_csv(
    path: str | Path,
    source_tz: str = "America/New_York",
    tz: str = "America/New_York",
) -> list[Bar]:
    """Read a 1-minute OHLCV CSV.

    Timestamps are the bar OPEN time. Naive timestamps are localised to
    ``source_tz``; tz-aware or UTC-suffixed timestamps are converted. Rows
    are sorted and de-duplicated on timestamp.
    """
    df = pd.read_csv(path)
    columns = list(df.columns)

    ts_col = _pick(columns, _TS_CANDIDATES, required=False)
    if ts_col is None:
        # Separate date and time columns
        d_col = _pick(columns, ("date",))
        t_col = _pick(columns, ("time",))
        raw = df[d_col].astype(str) + " " + df[t_col].astype(str)
    else:
        raw = df[ts_col]
    return frame_to_bars(df.assign(_ts=raw), ts_col="_ts", source_tz=source_tz, tz=tz)


def frame_to_bars(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    source_tz: str = "America/New_York",
    tz: str = "America/New_York",
) -> list[Bar]:
    columns = list(df.columns)
    o = _pick(columns, _COL_CANDIDATES["open"])
    h = _pick(columns, _COL_CANDIDATES["high"])
    l = _pick(columns, _COL_CANDIDATES["low"])
    c = _pick(columns, _COL_CANDIDATES["close"])
    v = _pick(columns, _COL_CANDIDATES["volume"], required=False)

    ts = pd.to_datetime(df[ts_col], utc=False)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(ZoneInfo(source_tz), ambiguous="infer", nonexistent="shift_forward")
    ts = ts.dt.tz_convert(ZoneInfo(tz))

    out = pd.DataFrame(
        {
            "ts": ts,
            "open": df[o].astype(float),
            "high": df[h].astype(float),
            "low": df[l].astype(float),
            "close": df[c].astype(float),
            "volume": df[v].astype(float) if v is not None else 0.0,
        }
    )
    out = out.dropna(subset=["ts", "open", "high", "low", "close"])
    out = out.sort_values("ts").drop_duplicates(subset="ts", keep="last")

    bars = []
    for row in out.itertuples(index=False):
        bars.append(Bar(ts=row.ts.to_pydatetime(), open=row.open, high=row.high, low=row.low, close=row.close, volume=row.volume))
    return bars


def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [b.ts for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    )


def generate_synthetic_bars(
    days: int = 60,
    seed: int = 7,
    start: date = date(2025, 1, 6),
    start_price: float = 5800.0,
    minute_vol: float = 0.00045,
    spec: ContractSpec = ES,
    tz: str = "America/New_York",
    session_start: time = time(9, 30),
    session_end: time = time(16, 0),
) -> list[Bar]:
    """Random-walk regular-session bars for ``days`` weekdays.

    Each day gets a random drift so that roughly a third of sessions trend
    and produce clean breakouts. This is for testing the plumbing; it has
    no information about real ES behaviour.
    """
    rng = random.Random(seed)
    zone = ZoneInfo(tz)
    bars: list[Bar] = []
    price = start_price
    day = start
    produced = 0
    while produced < days:
        if day.weekday() >= 5:
            day += timedelta(days=1)
            continue
        regime = rng.random()
        drift = rng.choice([-1, 1]) * minute_vol * (0.35 if regime < 0.35 else 0.0)
        vol = minute_vol * rng.uniform(0.6, 1.6)
        price *= math.exp(rng.gauss(0, 0.004))  # overnight gap
        current = datetime.combine(day, session_start, tzinfo=zone)
        end = datetime.combine(day, session_end, tzinfo=zone)
        while current < end:
            o = price
            path = [o]
            for _ in range(4):
                path.append(path[-1] * math.exp(drift + rng.gauss(0, vol / 2)))
            c = path[-1]
            hi = max(path) * math.exp(abs(rng.gauss(0, vol / 3)))
            lo = min(path) * math.exp(-abs(rng.gauss(0, vol / 3)))
            o_, h_, l_, c_ = (spec.round_to_tick(x) for x in (o, hi, lo, c))
            h_ = max(h_, o_, c_)
            l_ = min(l_, o_, c_)
            bars.append(Bar(ts=current, open=o_, high=h_, low=l_, close=c_, volume=float(rng.randint(500, 5000))))
            price = c
            current += timedelta(minutes=1)
        produced += 1
        day += timedelta(days=1)
    return bars
