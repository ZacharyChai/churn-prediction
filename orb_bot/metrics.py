"""Performance statistics for a list of trades."""

from __future__ import annotations

import math
from collections import Counter
from zoneinfo import ZoneInfo

import pandas as pd

from .models import DayRecord, Trade


def trades_to_frame(trades: list[Trade], tz: str = "America/New_York") -> pd.DataFrame:
    zone = ZoneInfo(tz)
    rows = []
    for t in trades:
        rows.append(
            {
                "entry_ts": t.entry_ts.astimezone(zone),
                "exit_ts": t.exit_ts.astimezone(zone),
                "side": t.side.value,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "stop": t.stop,
                "target": t.target,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "pnl_points": round(t.pnl_points, 4),
                "r_multiple": round(t.r_multiple, 4),
                "pnl_usd": round(t.pnl_usd, 2),
            }
        )
    columns = [
        "entry_ts", "exit_ts", "side", "qty", "entry_price", "stop", "target",
        "exit_price", "exit_reason", "pnl_points", "r_multiple", "pnl_usd",
    ]
    return pd.DataFrame(rows, columns=columns)


def max_drawdown(pnls: list[float]) -> float:
    peak = equity = 0.0
    worst = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def longest_streak(flags: list[bool]) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def summarize(trades: list[Trade], days: list[DayRecord] | None = None, tz: str = "America/New_York") -> dict:
    """Return a flat dict of headline statistics."""
    zone = ZoneInfo(tz)
    n = len(trades)
    pnls = [t.pnl_usd for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)

    # Daily P&L across every session the strategy saw (zero on days without a trade).
    daily: dict = {}
    if days:
        for d in days:
            daily[d.date] = 0.0
    for t in trades:
        day = t.entry_ts.astimezone(zone).date()
        daily[day] = daily.get(day, 0.0) + t.pnl_usd
    daily_series = pd.Series(daily, dtype="float64").sort_index()
    sharpe = float("nan")
    if len(daily_series) > 1 and daily_series.std(ddof=1) > 0:
        sharpe = daily_series.mean() / daily_series.std(ddof=1) * math.sqrt(252)

    stats = {
        "sessions": len(days) if days is not None else len(daily_series),
        "eligible_sessions": sum(1 for d in days if d.eligible) if days is not None else None,
        "trades": n,
        "win_rate": (len(wins) / n) if n else float("nan"),
        "net_pnl_usd": sum(pnls),
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else float("nan"),
        "avg_trade_usd": (sum(pnls) / n) if n else float("nan"),
        "avg_win_usd": (gross_profit / len(wins)) if wins else float("nan"),
        "avg_loss_usd": (-gross_loss / len(losses)) if losses else float("nan"),
        "expectancy_r": (sum(t.r_multiple for t in trades) / n) if n else float("nan"),
        "max_drawdown_usd": max_drawdown(pnls),
        "max_consecutive_losses": longest_streak([p <= 0 for p in pnls]),
        "sharpe_daily_annualized": sharpe,
        "best_trade_usd": max(pnls) if pnls else float("nan"),
        "worst_trade_usd": min(pnls) if pnls else float("nan"),
        "exit_reasons": dict(Counter(t.exit_reason for t in trades)),
        "long_trades": sum(1 for t in trades if t.side.value == "long"),
        "short_trades": sum(1 for t in trades if t.side.value == "short"),
    }
    return stats


def format_summary(stats: dict) -> str:
    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "n/a"
            if math.isinf(v):
                return "inf"
            return f"{v:,.2f}"
        return str(v)

    width = max(len(k) for k in stats)
    return "\n".join(f"{k.ljust(width)}  {fmt(v)}" for k, v in stats.items())
