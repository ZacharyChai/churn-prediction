"""Backtest engine: feeds bars through the strategy and the paper broker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from zoneinfo import ZoneInfo

from .broker import PaperBroker
from .config import StrategyConfig
from .models import Bar, DayRecord, Trade
from .strategy import ORBStrategy


@dataclass
class BacktestResult:
    cfg: StrategyConfig
    trades: list[Trade]
    days: list[DayRecord]
    skipped_entries: list

    @property
    def net_pnl(self) -> float:
        return sum(t.pnl_usd for t in self.trades)


def run_backtest(bars: Iterable[Bar], cfg: StrategyConfig) -> BacktestResult:
    strategy = ORBStrategy(cfg)
    broker = PaperBroker(cfg, cfg.spec)
    tz = ZoneInfo(cfg.tz)

    current_date = None
    prev_ts = None
    for bar in bars:
        if prev_ts is not None and bar.ts <= prev_ts:
            raise ValueError(f"bars must be strictly increasing in time; got {bar.ts} after {prev_ts}")
        prev_ts = bar.ts

        session_date = bar.ts.astimezone(tz).date()
        if current_date is not None and session_date != current_date:
            broker.end_of_session()
        current_date = session_date

        broker.process_bar(bar)
        broker.apply(strategy.on_bar(bar, broker.state()))

    broker.end_of_session()
    return BacktestResult(
        cfg=cfg,
        trades=list(broker.trades),
        days=list(strategy.day_log),
        skipped_entries=list(strategy.skipped_entries),
    )
