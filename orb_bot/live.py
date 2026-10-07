"""Live / paper trading through Interactive Brokers using ib_insync.

The strategy object is identical to the one used in backtests. This module
only (1) turns IB's streaming 1-minute bars into ``Bar`` objects and
(2) turns the strategy's actions into IB bracket orders.

Order mapping
-------------
EntryOrder(kind="stop")   -> parent STP order at the trigger, with a child STP (stop loss)
                             and a child LMT (target) attached. The long and short
                             parents share an OCA group so one fill cancels the other.
EntryOrder(kind="market") -> parent MKT order with the same children.
CancelEntries             -> cancel every open parent entry order on the contract.
Flatten                   -> cancel all open orders on the contract, then a MKT order
                             for the opposite side of the current position.

This module has NOT been exercised against a live gateway in this repository.
Run it with --dry-run first, then against a paper account (port 7497 / 4002),
and watch the TWS order log for the first few sessions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from .config import ContractSpec, StrategyConfig
from .models import Action, Bar, BrokerState, CancelEntries, EntryOrder, Flatten, Position, Side, Trade
from .strategy import ORBStrategy

log = logging.getLogger("orb_bot.live")


def _require_ib_insync():
    try:
        import ib_insync  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit("live trading requires ib_insync: pip install ib_insync") from exc
    return ib_insync


@dataclass
class IBBroker:
    """Thin adapter that satisfies what ORBStrategy expects from a broker."""

    ib: object
    contract: object
    cfg: StrategyConfig
    spec: ContractSpec
    dry_run: bool = True
    position: Position | None = None
    trades: list[Trade] = field(default_factory=list)
    entry_trades: dict = field(default_factory=dict)  # Side -> ib_insync.Trade (parent)
    _oca_counter: int = 0

    # ------------------------------------------------------------ state
    def state(self) -> BrokerState:
        pending = tuple(
            t.order._orb_entry
            for t in self.entry_trades.values()
            if t is not None and not t.isDone() and hasattr(t.order, "_orb_entry")
        )
        return BrokerState(position=self.position, pending_entries=pending, trades=tuple(self.trades))

    def reconcile(self) -> None:
        """Compare the tracked position with IB's view and trust IB."""
        if self.dry_run:
            return
        ib_qty = 0
        for p in self.ib.positions():
            if p.contract.conId == self.contract.conId:
                ib_qty += int(p.position)
        tracked = 0 if self.position is None else self.position.side.sign * self.position.qty
        if ib_qty != tracked:
            log.warning("position mismatch: IB=%s tracked=%s; adopting IB's view", ib_qty, tracked)
            if ib_qty == 0:
                self.position = None
            elif self.position is None:
                side = Side.LONG if ib_qty > 0 else Side.SHORT
                now = datetime.now(timezone.utc)
                self.position = Position(side=side, qty=abs(ib_qty), entry_price=float("nan"), entry_ts=now,
                                         stop=float("nan"), target=float("nan"),
                                         order=EntryOrder(side, "market", float("nan"), float("nan"), float("nan"), abs(ib_qty)))
            else:
                self.position.qty = abs(ib_qty)

    # ------------------------------------------------------------ actions
    def apply(self, actions: list[Action]) -> None:
        for action in actions:
            if isinstance(action, CancelEntries):
                self._cancel_entries()
            elif isinstance(action, Flatten):
                self._flatten(action.reason)
            elif isinstance(action, EntryOrder):
                self._place_entry(action)

    def _cancel_entries(self) -> None:
        for side, t in list(self.entry_trades.items()):
            if t is not None and not t.isDone():
                log.info("cancel %s entry order", side.value)
                if not self.dry_run:
                    self.ib.cancelOrder(t.order)
        self.entry_trades.clear()

    def _flatten(self, reason: str) -> None:
        from ib_insync import MarketOrder

        self._cancel_entries()
        if not self.dry_run:
            for t in self.ib.openTrades():
                if t.contract.conId == self.contract.conId:
                    self.ib.cancelOrder(t.order)
        if self.position is None:
            return
        action = "SELL" if self.position.side is Side.LONG else "BUY"
        log.info("FLATTEN (%s): %s %d %s at market", reason, action, self.position.qty, self.spec.symbol)
        if self.dry_run:
            self._record_exit(float("nan"), datetime.now(timezone.utc), reason)
            return
        order = MarketOrder(action, self.position.qty, tif="DAY", outsideRth=True)
        trade = self.ib.placeOrder(self.contract, order)
        trade.fillEvent += lambda tr, fill, reason=reason: self._on_exit_fill(tr, fill, reason)

    def _place_entry(self, entry: EntryOrder) -> None:
        from ib_insync import LimitOrder, MarketOrder, StopOrder

        action = "BUY" if entry.side is Side.LONG else "SELL"
        reverse = "SELL" if action == "BUY" else "BUY"
        log.info(
            "ENTRY %s %s %d @ %s  stop %s  target %s",
            entry.kind.upper(), action, entry.qty, entry.trigger, entry.stop, entry.target,
        )
        if self.dry_run:
            self.entry_trades[entry.side] = _DryTrade(entry)
            return

        if not self.entry_trades:
            self._oca_counter += 1
        oca = f"orb-{datetime.now(timezone.utc):%Y%m%d}-{self._oca_counter}"

        if entry.kind == "stop":
            parent = StopOrder(action, entry.qty, entry.trigger, tif="DAY", outsideRth=True)
        else:
            parent = MarketOrder(action, entry.qty, tif="DAY", outsideRth=True)
        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False
        parent.ocaGroup = oca
        parent.ocaType = 1
        parent._orb_entry = entry

        stop_loss = StopOrder(reverse, entry.qty, entry.stop, tif="GTC", outsideRth=True)
        stop_loss.orderId = self.ib.client.getReqId()
        stop_loss.parentId = parent.orderId
        stop_loss.transmit = False

        target = LimitOrder(reverse, entry.qty, entry.target, tif="GTC", outsideRth=True)
        target.orderId = self.ib.client.getReqId()
        target.parentId = parent.orderId
        target.transmit = True  # transmitting the last child sends the whole bracket

        parent_trade = self.ib.placeOrder(self.contract, parent)
        stop_trade = self.ib.placeOrder(self.contract, stop_loss)
        target_trade = self.ib.placeOrder(self.contract, target)

        parent_trade.fillEvent += lambda tr, fill, e=entry: self._on_entry_fill(tr, fill, e)
        stop_trade.fillEvent += lambda tr, fill: self._on_exit_fill(tr, fill, "stop")
        target_trade.fillEvent += lambda tr, fill: self._on_exit_fill(tr, fill, "target")
        self.entry_trades[entry.side] = parent_trade

    # ------------------------------------------------------------ fills
    def _on_entry_fill(self, trade, fill, entry: EntryOrder) -> None:
        if self.position is None:
            self.position = Position(
                side=entry.side, qty=0, entry_price=0.0, entry_ts=fill.time, stop=entry.stop, target=entry.target, order=entry
            )
        p = self.position
        filled = int(fill.execution.shares)
        total = p.qty + filled
        p.entry_price = (p.entry_price * p.qty + fill.execution.price * filled) / total if total else fill.execution.price
        p.qty = total
        log.info("entry filled: %s %d @ %.2f", p.side.value, filled, fill.execution.price)
        # Our OCA group cancels the other parent; drop both from the pending view.
        self.entry_trades.clear()

    def _on_exit_fill(self, trade, fill, reason: str) -> None:
        if self.position is None:
            return
        log.info("exit filled (%s): %d @ %.2f", reason, int(fill.execution.shares), fill.execution.price)
        self.position.qty -= int(fill.execution.shares)
        if self.position.qty <= 0:
            self._record_exit(fill.execution.price, fill.time, reason)

    def _record_exit(self, price: float, ts: datetime, reason: str) -> None:
        p = self.position
        assert p is not None
        qty = p.order.qty
        pnl_points = (price - p.entry_price) * p.side.sign
        pnl_usd = pnl_points * self.spec.point_value * qty - self.cfg.commission_per_side * 2 * qty
        risk = abs(p.entry_price - p.stop)
        self.trades.append(
            Trade(
                side=p.side, qty=qty, entry_ts=p.entry_ts, entry_price=p.entry_price, exit_ts=ts, exit_price=price,
                exit_reason=reason, stop=p.stop, target=p.target, pnl_points=pnl_points, pnl_usd=pnl_usd,
                r_multiple=pnl_points / risk if risk > 0 else 0.0,
            )
        )
        log.info("round trip closed: %s pnl=%.2f USD (%s)", p.side.value, pnl_usd, reason)
        self.position = None

    # ------------------------------------------------------------ startup
    def rebuild_today_from_executions(self, session_date: date, tz: ZoneInfo) -> None:
        """Reconstruct today's completed round trips from IB executions after a restart."""
        if self.dry_run:
            return
        fills = [f for f in self.ib.fills() if f.contract.conId == self.contract.conId
                 and f.time.astimezone(tz).date() == session_date]
        fills.sort(key=lambda f: f.time)
        net = 0
        entry_price = entry_ts = side = None
        for f in fills:
            qty = int(f.execution.shares) * (1 if f.execution.side == "BOT" else -1)
            prev = net
            net += qty
            if prev == 0 and net != 0:
                side = Side.LONG if net > 0 else Side.SHORT
                entry_price, entry_ts = f.execution.price, f.time
            elif prev != 0 and net == 0 and side is not None:
                pnl_points = (f.execution.price - entry_price) * side.sign
                n = abs(prev)
                self.trades.append(Trade(
                    side=side, qty=n, entry_ts=entry_ts, entry_price=entry_price, exit_ts=f.time,
                    exit_price=f.execution.price, exit_reason="restored", stop=float("nan"), target=float("nan"),
                    pnl_points=pnl_points, pnl_usd=pnl_points * self.spec.point_value * n - self.cfg.commission_per_side * 2 * n,
                    r_multiple=0.0,
                ))
        if self.trades:
            log.warning("restored %d completed trade(s) from today's executions", len(self.trades))
        if net != 0:
            log.warning("IB shows an open position of %d from before this restart", net)
        self.reconcile()


class _DryTrade:
    """Stand-in for an ib_insync Trade so dry runs track pending entries."""

    def __init__(self, entry: EntryOrder):
        class _O:  # noqa: D401 - tiny attribute bag
            pass

        self.order = _O()
        self.order._orb_entry = entry

    def isDone(self) -> bool:
        return False


def _bar_from_ib(b, tz: ZoneInfo) -> Bar:
    ts = b.date
    if isinstance(ts, date) and not isinstance(ts, datetime):
        ts = datetime.combine(ts, datetime.min.time())
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return Bar(ts=ts.astimezone(tz), open=b.open, high=b.high, low=b.low, close=b.close, volume=float(b.volume or 0))


def run_live(cfg: StrategyConfig, host: str, port: int, client_id: int, expiry: str, dry_run: bool) -> None:
    ib_insync = _require_ib_insync()
    from ib_insync import IB, Future

    tz = ZoneInfo(cfg.tz)
    spec = cfg.spec
    ib = IB()
    log.info("connecting to %s:%d (clientId %d) %s", host, port, client_id, "[DRY RUN]" if dry_run else "")
    ib.connect(host, port, clientId=client_id, readonly=dry_run)

    contract = Future(spec.symbol, lastTradeDateOrContractMonth=expiry, exchange=spec.exchange, currency=spec.currency)
    ib.qualifyContracts(contract)
    log.info("trading %s", contract.localSymbol)

    strategy = ORBStrategy(cfg)
    broker = IBBroker(ib=ib, contract=contract, cfg=cfg, spec=spec, dry_run=dry_run)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
        keepUpToDate=True,
    )

    # Warm up on today's completed bars so the range is known if we start late.
    today = datetime.now(tz).date()
    broker.rebuild_today_from_executions(today, tz)
    last_ts = None
    for b in list(bars)[:-1]:
        bar = _bar_from_ib(b, tz)
        strategy.on_bar(bar, broker.state())  # replay: discard actions
        last_ts = bar.ts
    log.info("warm-up done through %s; range high=%s low=%s eligible=%s",
             last_ts, strategy.range_high, strategy.range_low, strategy.eligible)

    def on_update(bar_list, has_new_bar: bool) -> None:
        nonlocal last_ts
        if not has_new_bar or len(bar_list) < 2:
            return
        completed = [_bar_from_ib(b, tz) for b in list(bar_list)[:-1]]
        for bar in completed:
            if last_ts is not None and bar.ts <= last_ts:
                continue
            last_ts = bar.ts
            broker.reconcile()
            actions = strategy.on_bar(bar, broker.state())
            if actions:
                log.info("%s close=%.2f -> %s", bar.ts.strftime("%H:%M"), bar.close, [type(a).__name__ for a in actions])
            broker.apply(actions)

    bars.updateEvent += on_update
    log.info("streaming 1-minute bars; Ctrl-C to stop")
    try:
        ib.run()
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        ib.disconnect()
