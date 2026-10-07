"""The ORB strategy as a pure state machine.

``ORBStrategy.on_bar`` receives one completed bar plus a read-only view of the
account and returns a list of actions (entry orders, cancels, flatten). It
never touches a broker directly, so the same object drives the backtester and
the live runner.

Idempotency matters for live trading: if there is no position and no resting
entry orders, the strategy re-issues its entry orders on every bar while the
day is still tradeable. That lets the bot be restarted mid-session without
losing the setup.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .config import StrategyConfig
from .models import Action, Bar, BrokerState, CancelEntries, DayRecord, EntryOrder, Flatten, Side, Trade


class ORBStrategy:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.spec = cfg.spec
        self.tz = ZoneInfo(cfg.tz)
        self.bar_delta = timedelta(minutes=cfg.bar_minutes)
        self.expected_range_bars = cfg.range_minutes // cfg.bar_minutes
        self.min_range_bars = max(1, math.ceil(cfg.min_range_fraction * self.expected_range_bars))
        self.day_log: list[DayRecord] = []
        self.skipped_entries: list[tuple[date, Side, str]] = []
        self._reset_day(None)

    # ------------------------------------------------------------- lifecycle
    def _reset_day(self, session_date: date | None) -> None:
        self.session_date = session_date
        self.range_high: float | None = None
        self.range_low: float | None = None
        self.range_bars = 0
        self.range_final = False
        self.eligible = False
        self.day_done = False
        self.blocked_sides: set[Side] = set()  # sides skipped today (e.g. unaffordable); never retried
        self.day: DayRecord | None = DayRecord(date=session_date) if session_date else None
        if self.day is not None:
            self.day_log.append(self.day)
        if session_date is not None:
            cfg = self.cfg
            self.open_dt = datetime.combine(session_date, cfg.session_open, tzinfo=self.tz)
            self.range_end_dt = self.open_dt + timedelta(minutes=cfg.range_minutes)
            self.last_entry_dt = datetime.combine(session_date, cfg.last_entry, tzinfo=self.tz)
            self.flatten_dt = datetime.combine(session_date, cfg.flatten_at, tzinfo=self.tz)

    @property
    def range_size(self) -> float | None:
        if self.range_high is None or self.range_low is None:
            return None
        return self.range_high - self.range_low

    # ------------------------------------------------------------- main hook
    def on_bar(self, bar: Bar, state: BrokerState) -> list[Action]:
        local = bar.ts.astimezone(self.tz)
        if local.date() != self.session_date:
            self._reset_day(local.date())
        bar_end = local + self.bar_delta
        cfg = self.cfg

        if local < self.open_dt:
            return []  # overnight / pre-open bar

        if bar_end <= self.range_end_dt:
            self._update_range(bar)
            if bar_end < self.range_end_dt:
                return []
            # This bar completes the range: fall through and arm entries now.

        if not self.range_final:
            self._finalize_range()

        if self.day_done:
            return []

        actions: list[Action] = []

        if bar_end >= self.flatten_dt:
            self.day_done = True
            if state.pending_entries:
                actions.append(CancelEntries())
            if state.position is not None:
                actions.append(Flatten(reason="eod"))
            return actions

        if state.position is not None:
            return []  # bracket orders manage the open trade

        if not self.eligible:
            return []

        trades_today, sides_taken, realised = self._today_activity(state)

        loss_limit_hit = cfg.max_daily_loss_usd is not None and realised <= -cfg.max_daily_loss_usd
        if bar_end >= self.last_entry_dt or trades_today >= cfg.max_trades_per_day or loss_limit_hit:
            if state.pending_entries:
                actions.append(CancelEntries())
            return actions

        sides = [s for s in (Side.LONG, Side.SHORT) if s not in sides_taken and s not in self.blocked_sides]
        if not sides:
            if state.pending_entries:
                actions.append(CancelEntries())
            return actions

        if cfg.entry_mode == "stop":
            if not state.pending_entries:
                for side in sides:
                    order = self._build_entry(side, kind="stop")
                    if order is not None:
                        actions.append(order)
            return actions

        # entry_mode == "close": act on a bar that closed beyond the range
        buf = cfg.breakout_ticks * self.spec.tick_size
        if Side.LONG in sides and bar.close > self.range_high + buf:
            order = self._build_entry(Side.LONG, kind="market", ref_price=bar.close)
        elif Side.SHORT in sides and bar.close < self.range_low - buf:
            order = self._build_entry(Side.SHORT, kind="market", ref_price=bar.close)
        else:
            order = None
        if order is not None:
            actions.append(order)
        return actions

    # ------------------------------------------------------------- internals
    def _update_range(self, bar: Bar) -> None:
        self.range_high = bar.high if self.range_high is None else max(self.range_high, bar.high)
        self.range_low = bar.low if self.range_low is None else min(self.range_low, bar.low)
        self.range_bars += 1

    def _finalize_range(self) -> None:
        self.range_final = True
        cfg = self.cfg
        day = self.day
        assert day is not None
        day.range_high, day.range_low, day.range_bars = self.range_high, self.range_low, self.range_bars

        if self.range_bars < self.min_range_bars or self.range_size is None:
            day.skip_reason = f"incomplete range ({self.range_bars} bars)"
        elif self.range_size < cfg.min_range_points:
            day.skip_reason = f"range too small ({self.range_size:.2f} pts)"
        elif self.range_size > cfg.max_range_points:
            day.skip_reason = f"range too large ({self.range_size:.2f} pts)"
        else:
            self.eligible = True
        day.eligible = self.eligible
        if not self.eligible:
            self.day_done = True

    def _today_activity(self, state: BrokerState) -> tuple[int, set[Side], float]:
        """Today's trade count, sides traded and realised P&L.

        Derived from the broker's trade list so that a trade which opened and
        closed inside a single bar is still counted.
        """
        count = 0
        sides: set[Side] = set()
        realised = 0.0
        for trade in reversed(state.trades):
            if trade.entry_ts.astimezone(self.tz).date() != self.session_date:
                break
            count += 1
            sides.add(trade.side)
            realised += trade.pnl_usd
        if state.position is not None:
            count += 1
            sides.add(state.position.side)
        return count, sides, realised

    def _build_entry(self, side: Side, kind: str, ref_price: float | None = None) -> EntryOrder | None:
        cfg, spec = self.cfg, self.spec
        assert self.range_high is not None and self.range_low is not None
        buf = cfg.breakout_ticks * spec.tick_size
        rng = self.range_size or 0.0

        if kind == "stop":
            trigger = self.range_high + buf if side is Side.LONG else self.range_low - buf
        else:
            assert ref_price is not None
            trigger = ref_price
        trigger = spec.round_to_tick(trigger)

        if cfg.stop_mode == "range":
            stop = self.range_low if side is Side.LONG else self.range_high
        elif cfg.stop_mode == "mid":
            stop = (self.range_high + self.range_low) / 2.0
        else:
            stop = trigger - side.sign * cfg.fixed_stop_points
        stop = spec.round_to_tick(stop)

        risk = (trigger - stop) * side.sign
        if risk <= 0:
            self._skip(side, "non-positive risk")
            return None

        move = rng * cfg.target_mult if cfg.target_mode == "range" else risk * cfg.target_mult
        target = spec.round_to_tick(trigger + side.sign * move)
        if (target - trigger) * side.sign <= 0:
            self._skip(side, "non-positive target distance")
            return None

        qty = math.floor(cfg.risk_per_trade_usd / (risk * spec.point_value))
        qty = min(qty, cfg.max_contracts)
        if qty < 1:
            self._skip(side, f"risk {risk:.2f} pts exceeds budget for 1 contract")
            return None

        return EntryOrder(side=side, kind=kind, trigger=trigger, stop=stop, target=target, qty=qty)

    def _skip(self, side: Side, reason: str) -> None:
        assert self.session_date is not None
        self.blocked_sides.add(side)
        self.skipped_entries.append((self.session_date, side, reason))

