"""Bar-driven fill simulator used by the backtester.

Fill conventions
----------------
* Market orders (entries in "close" mode, flattens) fill at the NEXT bar's
  open, plus slippage against you. The strategy only sees completed bars, so
  this mirrors what a live bot can actually do.
* Resting stop entries fill inside the bar that trades through the trigger,
  at max(trigger, open) for longs / min(trigger, open) for shorts, plus
  slippage. If both the long and short triggers are hit in one bar, the one
  closer to the bar's open is assumed to have filled first.
* Stop losses fill at the stop price plus slippage, or at the open if the
  bar gaps through. Targets are limit orders: they fill at the target with
  no slippage, or at the open if the bar gaps through favourably.
* If the stop and target are both touched in the same bar, the stop is
  assumed to have filled first when ``conservative_fills`` is true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .config import ContractSpec, StrategyConfig
from .models import Action, Bar, BrokerState, CancelEntries, EntryOrder, Flatten, Position, Side, Trade


@dataclass
class PaperBroker:
    cfg: StrategyConfig
    spec: ContractSpec
    position: Position | None = None
    pending: list[EntryOrder] = field(default_factory=list)
    market_queue: list[EntryOrder] = field(default_factory=list)
    flatten_reason: str | None = None
    trades: list[Trade] = field(default_factory=list)
    last_bar: Bar | None = None

    def __post_init__(self) -> None:
        self.slip = self.cfg.slippage_ticks * self.spec.tick_size

    # ------------------------------------------------------------ interface
    def state(self) -> BrokerState:
        return BrokerState(
            position=self.position,
            pending_entries=tuple(self.pending),
            trades=tuple(self.trades),
        )

    def apply(self, actions: list[Action]) -> None:
        for action in actions:
            if isinstance(action, CancelEntries):
                self.pending.clear()
                self.market_queue.clear()
            elif isinstance(action, Flatten):
                self.pending.clear()
                self.market_queue.clear()
                if self.position is not None:
                    self.flatten_reason = action.reason
            elif isinstance(action, EntryOrder):
                if action.kind == "stop":
                    self.pending.append(action)
                else:
                    self.market_queue.append(action)
            else:  # pragma: no cover - defensive
                raise TypeError(f"unknown action {action!r}")

    def process_bar(self, bar: Bar) -> None:
        """Simulate everything that could fill during ``bar``."""
        self.last_bar = bar
        entered_this_bar = False

        # 1. Flatten requested on the previous bar: exit at this open.
        if self.position is not None and self.flatten_reason is not None:
            px = bar.open - self.position.side.sign * self.slip
            self._close(px, bar.ts, self.flatten_reason)
            self.flatten_reason = None

        # 2. Market entries queued on the previous bar: fill at this open.
        if self.position is None and self.market_queue:
            order = self.market_queue[0]
            self.market_queue.clear()
            self.pending.clear()
            self._open(order, bar.open + order.side.sign * self.slip, bar.ts)
            entered_this_bar = True

        # 3. Manage a position carried into this bar.
        if self.position is not None and not entered_this_bar:
            self._check_exits(bar, check_gap=True)

        # 4. Resting stop entries.
        if self.position is None and self.pending:
            triggered = [o for o in self.pending if self._triggered(o, bar)]
            if triggered:
                order = min(triggered, key=lambda o: abs(o.trigger - bar.open))
                if order.side is Side.LONG:
                    px = max(order.trigger, bar.open) + self.slip
                else:
                    px = min(order.trigger, bar.open) - self.slip
                self.pending.clear()  # one-cancels-other
                self._open(order, px, bar.ts)
                entered_this_bar = True

        # 5. A trade opened inside this bar can also be stopped/targeted inside it.
        if self.position is not None and entered_this_bar:
            self._check_exits(bar, check_gap=False)

    def end_of_session(self) -> None:
        """Called when the data for a session ends. Closes anything still open at the last close."""
        if self.position is not None and self.last_bar is not None:
            self._close(self.last_bar.close, self.last_bar.ts, "session_end")
        self.pending.clear()
        self.market_queue.clear()
        self.flatten_reason = None

    # ------------------------------------------------------------ internals
    @staticmethod
    def _triggered(order: EntryOrder, bar: Bar) -> bool:
        if order.side is Side.LONG:
            return bar.high >= order.trigger
        return bar.low <= order.trigger

    def _open(self, order: EntryOrder, price: float, ts: datetime) -> None:
        self.position = Position(
            side=order.side,
            qty=order.qty,
            entry_price=price,
            entry_ts=ts,
            stop=order.stop,
            target=order.target,
            order=order,
        )

    def _check_exits(self, bar: Bar, check_gap: bool) -> None:
        p = self.position
        assert p is not None
        sign = p.side.sign

        if check_gap:
            if (bar.open - p.stop) * sign <= 0:
                self._close(bar.open - sign * self.slip, bar.ts, "stop")
                return
            if (bar.open - p.target) * sign >= 0:
                self._close(bar.open, bar.ts, "target")
                return

        if p.side is Side.LONG:
            stop_hit, target_hit = bar.low <= p.stop, bar.high >= p.target
        else:
            stop_hit, target_hit = bar.high >= p.stop, bar.low <= p.target

        if stop_hit and target_hit:
            if self.cfg.conservative_fills:
                self._close(p.stop - sign * self.slip, bar.ts, "stop")
            else:
                self._close(p.target, bar.ts, "target")
        elif stop_hit:
            self._close(p.stop - sign * self.slip, bar.ts, "stop")
        elif target_hit:
            self._close(p.target, bar.ts, "target")

    def _close(self, price: float, ts: datetime, reason: str) -> None:
        p = self.position
        assert p is not None
        pnl_points = (price - p.entry_price) * p.side.sign
        commission = self.cfg.commission_per_side * 2 * p.qty
        pnl_usd = pnl_points * self.spec.point_value * p.qty - commission
        risk = abs(p.entry_price - p.stop)
        self.trades.append(
            Trade(
                side=p.side,
                qty=p.qty,
                entry_ts=p.entry_ts,
                entry_price=p.entry_price,
                exit_ts=ts,
                exit_price=price,
                exit_reason=reason,
                stop=p.stop,
                target=p.target,
                pnl_points=pnl_points,
                pnl_usd=pnl_usd,
                r_multiple=pnl_points / risk if risk > 0 else 0.0,
            )
        )
        self.position = None
