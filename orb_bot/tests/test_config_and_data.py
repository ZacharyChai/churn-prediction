import json
from datetime import time

import pandas as pd
import pytest

from orb_bot import StrategyConfig, run_backtest
from orb_bot.config import ES, MES
from orb_bot.data import bars_to_frame, frame_to_bars, generate_synthetic_bars, load_bars_csv
from orb_bot.metrics import max_drawdown, summarize, trades_to_frame


def test_config_round_trip(tmp_path):
    cfg = StrategyConfig(last_entry="11:00", flatten_at=time(15, 50), target_mult=1.5)
    path = tmp_path / "cfg.json"
    cfg.save(path)
    loaded = StrategyConfig.load(path)
    assert loaded == cfg
    assert json.loads(path.read_text())["last_entry"] == "11:00"


@pytest.mark.parametrize(
    "kw",
    [
        dict(contract="NQ"),
        dict(entry_mode="limit"),
        dict(min_range_fraction=1.5),
        dict(bar_minutes=7),
        dict(last_entry="09:00"),
        dict(max_daily_loss_usd=-5),
        dict(min_range_points=50, max_range_points=40),
    ],
)
def test_config_validation(kw):
    with pytest.raises(ValueError):
        StrategyConfig(**kw)


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        StrategyConfig.from_dict({"nope": 1})


def test_contract_specs():
    assert ES.tick_value == 12.5
    assert MES.tick_value == 1.25
    assert ES.round_to_tick(5010.13) == 5010.25
    assert ES.round_to_tick(5010.12) == 5010.0


def test_csv_round_trip_with_naive_new_york_timestamps(tmp_path):
    bars = generate_synthetic_bars(days=2, seed=1)
    frame = bars_to_frame(bars)
    frame["timestamp"] = frame["timestamp"].map(lambda d: d.strftime("%Y-%m-%d %H:%M:%S"))
    path = tmp_path / "bars.csv"
    frame.to_csv(path, index=False)

    loaded = load_bars_csv(path)
    assert len(loaded) == len(bars)
    assert loaded[0].ts == bars[0].ts
    assert loaded[-1].close == bars[-1].close


def test_csv_with_utc_timestamps_is_converted(tmp_path):
    df = pd.DataFrame(
        {
            "Date": ["2025-03-03 14:30:00+00:00", "2025-03-03 14:31:00+00:00"],
            "Open": [1, 2], "High": [2, 3], "Low": [0, 1], "Close": [1.5, 2.5], "Vol": [10, 20],
        }
    )
    path = tmp_path / "utc.csv"
    df.to_csv(path, index=False)
    bars = load_bars_csv(path)
    assert bars[0].ts.hour == 9 and bars[0].ts.minute == 30
    assert bars[0].ts.tzinfo.key == "America/New_York"
    assert bars[1].volume == 20


def test_duplicate_and_unsorted_rows_are_cleaned():
    df = pd.DataFrame(
        {
            "timestamp": ["2025-03-03 09:31", "2025-03-03 09:30", "2025-03-03 09:31"],
            "open": [1, 1, 5], "high": [2, 2, 6], "low": [0, 0, 4], "close": [1, 1, 5],
        }
    )
    bars = frame_to_bars(df)
    assert [b.ts.minute for b in bars] == [30, 31]
    assert bars[1].close == 5  # last duplicate wins


def test_synthetic_backtest_runs_end_to_end():
    bars = generate_synthetic_bars(days=20, seed=3)
    assert len(bars) == 20 * 390
    result = run_backtest(bars, StrategyConfig())
    assert len(result.days) == 20
    stats = summarize(result.trades, result.days)
    assert stats["sessions"] == 20
    assert stats["trades"] == len(result.trades)
    frame = trades_to_frame(result.trades)
    assert list(frame.columns)[:3] == ["entry_ts", "exit_ts", "side"]
    if result.trades:
        assert frame["pnl_usd"].sum() == pytest.approx(result.net_pnl, abs=0.05)


def test_max_drawdown():
    assert max_drawdown([100, -50, -100, 200, -300]) == -300
    assert max_drawdown([]) == 0.0
    assert max_drawdown([10, 20]) == 0.0


def test_summary_with_no_trades_does_not_crash():
    stats = summarize([], [])
    assert stats["trades"] == 0
