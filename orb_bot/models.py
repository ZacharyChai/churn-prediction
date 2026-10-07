"""Plain data types shared by the strategy, brokers and backtester."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Union


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1

    @property
    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar. ``ts`` is the tz-aware OPEN time of the bar."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise ValueError("Bar.ts must be timezone-aware")
        if not (self.low <= min(self.open, self.close) and self.high >= max(self.open, self.close)):
            raise ValueError(f"inconsistent OHLC in bar at {self.ts}: {self}")


@dataclass(frozen=True)
class EntryOrder:
    """A request to enter a position with an attached bracket.

    ``kind == "stop"``   : rest a stop order at ``trigger``; it fills when price trades through.
    ``kind == "market"`` : enter at the next available price (``trigger`` is the reference close).
    """

    side: Side
    kind: str
    trigger: float
    stop: float
    target: float
    qty: int

    @property
    def risk_points(self) -> float:
        return abs(self.trigger - self.stop)


@dataclass(frozen=True)
class CancelEntries:
    """Cancel every resting entry order."""


@dataclass(frozen=True)
class Flatten:
    """Close the open position at market."""

    reason: str


Action = Union[EntryOrder, CancelEntries, Flatten]


@dataclass
class Position:
    side: Side
    qty: int
    entry_price: float
    entry_ts: datetime
    stop: float
    target: float
    order: EntryOrder


@dataclass(frozen=True)
class Trade:
    side: Side
    qty: int
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    exit_reason: str  # "stop" | "target" | "eod" | "session_end"
    stop: float
    target: float
    pnl_points: float  # per contract, after direction, before costs
    pnl_usd: float  # total, after commissions
    r_multiple: float  # pnl_points / initial risk in points

    @property
    def is_win(self) -> bool:
        return self.pnl_usd > 0


@dataclass(frozen=True)
class BrokerState:
    """What the strategy is allowed to know about the account."""

    position: Position | None
    pending_entries: tuple[EntryOrder, ...]
    trades: tuple[Trade, ...]


@dataclass
class DayRecord:
    """Diagnostics for one session."""

    date: object  # datetime.date
    range_high: float | None = None
    range_low: float | None = None
    range_bars: int = 0
    eligible: bool = False
    skip_reason: str | None = None

    @property
    def range_size(self) -> float | None:
        if self.range_high is None or self.range_low is None:
            return None
        return self.range_high - self.range_low
