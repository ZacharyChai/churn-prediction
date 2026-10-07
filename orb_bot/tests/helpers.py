"""Builders for hand-crafted bar sequences."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from orb_bot.models import Bar

NY = ZoneInfo("America/New_York")


def ts(day: date, hh: int, mm: int) -> datetime:
    return datetime.combine(day, time(hh, mm), tzinfo=NY)


def bar(day: date, hh: int, mm: int, o: float, h: float, l: float, c: float) -> Bar:
    return Bar(ts=ts(day, hh, mm), open=o, high=h, low=l, close=c, volume=100.0)


def flat_range(day: date, high: float = 5010.0, low: float = 5000.0, start: time = time(9, 30), minutes: int = 30) -> list[Bar]:
    """30 one-minute bars that together span exactly [low, high]."""
    bars = []
    current = datetime.combine(day, start, tzinfo=NY)
    mid = (high + low) / 2
    wiggle = (high - low) / 4
    for i in range(minutes):
        if i == 0:
            bars.append(Bar(ts=current, open=mid, high=high, low=low, close=mid))
        else:
            bars.append(Bar(ts=current, open=mid, high=mid + wiggle, low=mid - wiggle, close=mid))
        current += timedelta(minutes=1)
    return bars


def flat_bars(day: date, start: tuple[int, int], end: tuple[int, int], price: float) -> list[Bar]:
    """Flat bars from start (inclusive) to end (exclusive), one per minute."""
    out = []
    current = datetime.combine(day, time(*start), tzinfo=NY)
    stop = datetime.combine(day, time(*end), tzinfo=NY)
    while current < stop:
        out.append(Bar(ts=current, open=price, high=price, low=price, close=price))
        current += timedelta(minutes=1)
    return out
