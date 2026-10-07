"""Command-line interface: ``python -m orb_bot <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .backtest import run_backtest
from .config import StrategyConfig
from .data import bars_to_frame, generate_synthetic_bars, load_bars_csv
from .metrics import format_summary, summarize, trades_to_frame


def _load_config(path: str | None, overrides: list[str]) -> StrategyConfig:
    data = StrategyConfig.load(path).to_dict() if path else StrategyConfig().to_dict()
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        if key not in data:
            raise SystemExit(f"unknown config key {key!r}")
        current = data[key]
        if value.lower() in ("none", "null", ""):
            data[key] = None
        elif isinstance(current, bool):
            data[key] = value.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            data[key] = int(value)
        elif isinstance(current, float):
            data[key] = float(value)
        elif current is None:
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
        else:
            data[key] = value
    return StrategyConfig.from_dict(data)


def cmd_init_config(args: argparse.Namespace) -> int:
    StrategyConfig().save(args.out)
    print(f"wrote default config to {args.out}")
    return 0


def cmd_synthetic(args: argparse.Namespace) -> int:
    bars = generate_synthetic_bars(days=args.days, seed=args.seed)
    frame = bars_to_frame(bars)
    frame["timestamp"] = frame["timestamp"].map(lambda d: d.strftime("%Y-%m-%d %H:%M:%S"))
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(bars)} synthetic bars over {args.days} sessions to {args.out} (timestamps in America/New_York)")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config, args.set)
    if args.csv:
        bars = load_bars_csv(args.csv, source_tz=args.source_tz, tz=cfg.tz)
        source = args.csv
    else:
        bars = generate_synthetic_bars(days=args.synthetic_days, seed=args.seed)
        source = f"synthetic ({args.synthetic_days} sessions, seed {args.seed})"
    if not bars:
        raise SystemExit("no bars loaded")

    result = run_backtest(bars, cfg)
    stats = summarize(result.trades, result.days, tz=cfg.tz)

    print(f"data      : {source}")
    print(f"bars      : {len(bars)}  ({bars[0].ts:%Y-%m-%d} to {bars[-1].ts:%Y-%m-%d})")
    print(f"contract  : {cfg.contract}  range={cfg.range_minutes}m  entry={cfg.entry_mode}  "
          f"stop={cfg.stop_mode}  target={cfg.target_mult}x{cfg.target_mode}  risk=${cfg.risk_per_trade_usd:,.0f}")
    print()
    print(format_summary(stats))

    if result.skipped_entries and args.verbose:
        print("\nskipped entries:")
        for day, side, reason in result.skipped_entries:
            print(f"  {day} {side.value:5s} {reason}")

    if args.out:
        trades_to_frame(result.trades, tz=cfg.tz).to_csv(args.out, index=False)
        print(f"\nwrote {len(result.trades)} trades to {args.out}")
    if args.days_out:
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "date": d.date,
                    "range_high": d.range_high,
                    "range_low": d.range_low,
                    "range_size": d.range_size,
                    "range_bars": d.range_bars,
                    "eligible": d.eligible,
                    "skip_reason": d.skip_reason,
                }
                for d in result.days
            ]
        ).to_csv(args.days_out, index=False)
        print(f"wrote {len(result.days)} session records to {args.days_out}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .live import run_live

    cfg = _load_config(args.config, args.set)
    if not args.dry_run and not args.i_understand_this_sends_real_orders:
        raise SystemExit(
            "Refusing to send orders. Re-run with --dry-run to log only, or add "
            "--i-understand-this-sends-real-orders once you have tested against a paper account."
        )
    run_live(
        cfg,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        expiry=args.expiry,
        dry_run=args.dry_run,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orb_bot", description="Opening Range Breakout bot for ES/MES futures")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init-config", help="write a default JSON config")
    s.add_argument("out", nargs="?", default="orb_config.json")
    s.set_defaults(func=cmd_init_config)

    s = sub.add_parser("synthetic", help="write a synthetic 1-minute CSV for plumbing tests")
    s.add_argument("--days", type=int, default=120)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="es_synthetic_1min.csv")
    s.set_defaults(func=cmd_synthetic)

    s = sub.add_parser("backtest", help="run the strategy over a 1-minute CSV")
    s.add_argument("--csv", help="1-minute OHLCV CSV (timestamp = bar open time)")
    s.add_argument("--source-tz", default="America/New_York", help="timezone of naive timestamps in the CSV")
    s.add_argument("--config", help="JSON config written by init-config")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    s.add_argument("--synthetic-days", type=int, default=120, help="used when --csv is omitted")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", help="write the trade list to this CSV")
    s.add_argument("--days-out", help="write per-session range diagnostics to this CSV")
    s.add_argument("-v", "--verbose", action="store_true", help="also list skipped entries")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("live", help="trade through Interactive Brokers TWS / Gateway (needs ib_insync)")
    s.add_argument("--config")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7497, help="7497 = TWS paper, 7496 = TWS live, 4002/4001 = Gateway paper/live")
    s.add_argument("--client-id", type=int, default=17)
    s.add_argument("--expiry", required=True, help="contract month, e.g. 202612 for the December 2026 ES")
    s.add_argument("--dry-run", action="store_true", help="log the orders the bot would send without sending them")
    s.add_argument("--i-understand-this-sends-real-orders", action="store_true")
    s.set_defaults(func=cmd_live)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
