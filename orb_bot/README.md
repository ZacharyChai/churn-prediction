# ORB bot for ES / MES futures

An Opening Range Breakout day-trading bot. It builds the high and low of the
first 30 minutes after the New York cash open (09:30 to 10:00 ET), rests a buy
stop above the range and a sell stop below it, manages the trade with a
bracket (stop loss and profit target), and is flat before the cash close.

The strategy is a pure state machine. The same object drives the backtester
and the live Interactive Brokers runner, so what you backtest is what trades.

```
orb_bot/
  config.py     StrategyConfig (all parameters) and ES / MES contract specs
  models.py     Bar, EntryOrder, Position, Trade, BrokerState dataclasses
  strategy.py   ORBStrategy.on_bar(bar, state) -> actions
  broker.py     PaperBroker: bar-driven fill simulator for backtests
  backtest.py   run_backtest(bars, cfg)
  metrics.py    win rate, profit factor, drawdown, Sharpe, trade export
  data.py       CSV loader (timezone aware) and synthetic bar generator
  live.py       Interactive Brokers runner (ib_insync), dry-run by default
  cli.py        python -m orb_bot {init-config, synthetic, backtest, live}
  tests/        39 pytest cases covering fills, timing, sizing, filters, DST
```

## Strategy rules (defaults)

| Step | Rule | Config key |
|---|---|---|
| Range | High/low of the 1-minute bars from 09:30 through 09:59 ET | `session_open`, `range_minutes`, `bar_minutes` |
| Filters | Skip the day if the range is under 2 or over 40 points, or if fewer than 80% of the window's bars arrived | `min_range_points`, `max_range_points`, `min_range_fraction` |
| Entry | Buy stop at range high + 1 tick, sell stop at range low - 1 tick, placed as the 09:59 bar closes. One cancels the other | `entry_mode="stop"`, `breakout_ticks` |
| Stop loss | Opposite side of the range | `stop_mode` (`range`, `mid`, `fixed`) |
| Target | 1.0 x range size from the entry | `target_mode` (`range`, `r`), `target_mult` |
| Cutoff | Resting entries cancelled at 11:30 ET | `last_entry` |
| Flatten | Open position closed at market at 15:55 ET | `flatten_at` |
| Frequency | One trade per day; each direction at most once per day | `max_trades_per_day` |
| Kill switch | Optional: stop for the day once realised losses reach a limit | `max_daily_loss_usd` |
| Sizing | floor(risk budget / (stop distance x point value)), capped | `risk_per_trade_usd`, `max_contracts` |
| Costs | $2.25 per side per contract, 1 tick slippage on market and stop fills | `commission_per_side`, `slippage_ticks` |

ES is $50 per point, MES is $5 per point, both tick in 0.25. A range-wide
stop on ES is commonly $1,000 to $1,750 per contract, so the default risk
budget is $2,000. For MES, set `contract="MES"`, `risk_per_trade_usd=200`
and raise `max_contracts`.

`entry_mode="close"` is the alternative entry: wait for a 1-minute bar to
close beyond the range, then enter at the next bar's open. It filters wicks
at the cost of a worse fill.

### Timing convention

Bar timestamps are the bar's open time. The strategy acts when a bar closes.
The 09:59 bar closes at 10:00:00, so the entry stops are live during the 10:00
bar. The 11:29 bar closing cancels the entries; the 15:54 bar closing flattens
the position at the 15:55 open. All times are New York wall clock, so daylight
saving is handled, and Globex overnight bars are ignored.

## Backtest fill model

* Resting stop entries fill inside the bar that trades through the trigger,
  at max(trigger, open) for longs, plus slippage. If both triggers are hit in
  one bar, the one nearer the bar's open is taken.
* Stop losses fill at the stop plus slippage, or at the open if the bar gaps
  through. Targets are limit orders and fill without slippage.
* If the stop and target are both touched in the same bar, the stop is
  assumed to have filled first (`conservative_fills=true`).
* A trade that opens and gets stopped inside one bar still counts toward the
  daily trade limit.

Backtests have no knowledge of a bar's internal path, so expect live results
to be somewhat worse than the simulation on the same data.

## Quick start

```bash
pip install -r orb_bot/requirements.txt

# Default parameters as an editable JSON file
python -m orb_bot init-config orb_config.json

# Plumbing check on random-walk data (no edge expected; it should roughly lose the costs)
python -m orb_bot backtest --synthetic-days 120

# Real data: a 1-minute OHLCV CSV with columns timestamp,open,high,low,close,volume
python -m orb_bot backtest --csv es_1min.csv --config orb_config.json \
    --out trades.csv --days-out sessions.csv

# Override any parameter from the command line
python -m orb_bot backtest --csv es_1min.csv --set target_mult=1.5 --set stop_mode=mid \
    --set max_trades_per_day=2 --set max_daily_loss_usd=1500
```

Naive CSV timestamps are assumed to be New York time; pass `--source-tz UTC`
(or any IANA zone) if your vendor exports otherwise. Timestamps with an
offset are converted automatically. Separate `date` and `time` columns work
too. Use a back-adjusted continuous contract so quarterly rolls do not appear
as gaps; the bot does not handle rolls itself.

Run the tests with `python -m pytest orb_bot`.

## Live trading with Interactive Brokers

```bash
pip install ib_insync
# TWS or IB Gateway must be running with API access enabled.
# 7497 = TWS paper, 7496 = TWS live, 4002 / 4001 = Gateway paper / live.
python -m orb_bot live --expiry 202612 --port 7497 --dry-run
python -m orb_bot live --expiry 202612 --port 7497 --i-understand-this-sends-real-orders
```

The runner streams 1-minute bars (`reqHistoricalData` with `keepUpToDate`),
replays today's completed bars on startup so the range is known even if you
start late, and reconstructs today's completed round trips from IB executions
so a restart does not re-trade a direction that was already taken.

Order mapping: each entry is a parent stop (or market) order with a child
stop loss and a child limit target attached. The long and short parents share
an OCA group, so one fill cancels the other. Flatten cancels everything on the
contract and sends a market order for the opposite side of the position.

The live module has not been exercised against a gateway in this repository.
Run `--dry-run` first, then a paper account, and watch the TWS order log for
the first few sessions before considering real money. Futures carry leverage
and the opening half hour is the most volatile part of the session; size
accordingly.
