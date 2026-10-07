"""Strategy configuration and futures contract specifications."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import time
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContractSpec:
    """Static facts about a futures contract needed for sizing and P&L."""

    symbol: str
    point_value: float  # USD per 1.00 index point
    tick_size: float  # minimum price increment in points
    exchange: str = "CME"
    currency: str = "USD"

    @property
    def tick_value(self) -> float:
        return self.point_value * self.tick_size

    def round_to_tick(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)


ES = ContractSpec(symbol="ES", point_value=50.0, tick_size=0.25)
MES = ContractSpec(symbol="MES", point_value=5.0, tick_size=0.25)

CONTRACTS: dict[str, ContractSpec] = {"ES": ES, "MES": MES}

_TIME_FIELDS = ("session_open", "last_entry", "flatten_at")


def _parse_time(value: Any) -> time:
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        parts = value.strip().split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"time must be HH:MM or HH:MM:SS, got {value!r}")
        return time(*(int(p) for p in parts))
    raise TypeError(f"cannot interpret {value!r} as a time of day")


@dataclass
class StrategyConfig:
    """All tunable parameters of the ORB strategy.

    Times are wall-clock times in ``tz`` (default New York). Bar timestamps
    are the bar's OPEN time and the strategy acts when a bar CLOSES. With
    1-minute bars and ``range_minutes=30`` the range is built from the 09:30
    bar through the 09:59 bar; entry stop orders are placed as the 09:59 bar
    closes (10:00:00) and can fill during the 10:00 bar. Resting entries are
    cancelled as the bar ending at ``last_entry`` closes, and an open position
    is flattened as the bar ending at ``flatten_at`` closes.
    """

    # Instrument
    contract: str = "ES"
    tz: str = "America/New_York"

    # Session timing (wall clock in tz)
    bar_minutes: int = 1  # bar size of the data feed
    session_open: time = field(default_factory=lambda: time(9, 30))
    range_minutes: int = 30
    min_range_fraction: float = 0.8  # skip the day if fewer than this share of the window's bars arrived
    last_entry: time = field(default_factory=lambda: time(11, 30))  # resting entries are cancelled at this time
    flatten_at: time = field(default_factory=lambda: time(15, 55))  # an open position is closed at market at this time

    # Entry
    entry_mode: str = "stop"  # "stop": resting stop orders beyond the range; "close": enter on a bar close beyond it
    breakout_ticks: int = 1  # buffer beyond the range high/low before an entry triggers

    # Exit
    stop_mode: str = "range"  # "range": opposite side of range; "mid": range midpoint; "fixed": fixed_stop_points
    fixed_stop_points: float = 5.0
    target_mode: str = "range"  # "range": multiple of the range size; "r": multiple of the initial risk
    target_mult: float = 1.0
    conservative_fills: bool = True  # if stop and target are both touched in one bar, assume the stop filled

    # Filters
    min_range_points: float = 2.0
    max_range_points: float = 40.0
    max_trades_per_day: int = 1  # each direction can be traded at most once per day regardless
    max_daily_loss_usd: float | None = None  # stop trading for the day once realised losses reach this

    # Risk and costs
    risk_per_trade_usd: float = 2000.0  # a range-wide stop on ES is often $1,000-1,750 per contract
    max_contracts: int = 5
    commission_per_side: float = 2.25  # USD per contract per fill
    slippage_ticks: int = 1  # applied to market and stop fills, not to limit (target) fills

    def __post_init__(self) -> None:
        for name in _TIME_FIELDS:
            setattr(self, name, _parse_time(getattr(self, name)))
        self.validate()

    # ------------------------------------------------------------------ helpers
    @property
    def spec(self) -> ContractSpec:
        return CONTRACTS[self.contract]

    def validate(self) -> None:
        if self.contract not in CONTRACTS:
            raise ValueError(f"unknown contract {self.contract!r}; choose from {sorted(CONTRACTS)}")
        if self.entry_mode not in ("stop", "close"):
            raise ValueError("entry_mode must be 'stop' or 'close'")
        if self.stop_mode not in ("range", "mid", "fixed"):
            raise ValueError("stop_mode must be 'range', 'mid' or 'fixed'")
        if self.target_mode not in ("range", "r"):
            raise ValueError("target_mode must be 'range' or 'r'")
        if self.bar_minutes <= 0 or self.range_minutes <= 0:
            raise ValueError("bar_minutes and range_minutes must be positive")
        if self.range_minutes % self.bar_minutes:
            raise ValueError("range_minutes must be a multiple of bar_minutes")
        if not 0 < self.min_range_fraction <= 1:
            raise ValueError("min_range_fraction must be in (0, 1]")
        if self.target_mult <= 0:
            raise ValueError("target_mult must be positive")
        if self.risk_per_trade_usd <= 0 or self.max_contracts < 1:
            raise ValueError("risk_per_trade_usd must be positive and max_contracts at least 1")
        if self.min_range_points < 0 or self.max_range_points <= self.min_range_points:
            raise ValueError("need 0 <= min_range_points < max_range_points")
        if self.breakout_ticks < 0 or self.slippage_ticks < 0:
            raise ValueError("breakout_ticks and slippage_ticks cannot be negative")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day must be at least 1")
        if self.max_daily_loss_usd is not None and self.max_daily_loss_usd <= 0:
            raise ValueError("max_daily_loss_usd must be positive when set")
        if not (self.session_open < self.last_entry <= self.flatten_at):
            raise ValueError("need session_open < last_entry <= flatten_at")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for name in _TIME_FIELDS:
            d[name] = getattr(self, name).strftime("%H:%M")
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "StrategyConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")
