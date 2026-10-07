from datetime import date, time

import pytest

from orb_bot import StrategyConfig, run_backtest
from orb_bot.models import Side

from .helpers import bar, flat_bars, flat_range

DAY = date(2025, 3, 3)  # a Monday


def cfg(**kw) -> StrategyConfig:
    base = dict(risk_per_trade_usd=1000.0, slippage_ticks=0, commission_per_side=0.0)
    base.update(kw)
    return StrategyConfig(**base)


def test_range_is_built_from_first_30_bars_and_entry_waits_for_breakout():
    bars = flat_range(DAY, 5010, 5000)
    bars += flat_bars(DAY, (10, 0), (10, 30), 5005.0)  # inside the range: nothing should happen
    bars.append(bar(DAY, 10, 30, 5005, 5012, 5004, 5011))  # trades through 5010.25
    bars += flat_bars(DAY, (10, 31), (11, 0), 5011.0)
    bars.append(bar(DAY, 11, 0, 5011, 5021, 5010, 5020))  # 1x range target at 5020.25
    bars += flat_bars(DAY, (11, 1), (16, 0), 5019.0)

    result = run_backtest(bars, cfg())
    day = result.days[0]
    assert (day.range_high, day.range_low, day.range_bars) == (5010.0, 5000.0, 30)
    assert day.eligible

    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.side is Side.LONG
    assert t.entry_price == pytest.approx(5010.25)
    assert t.entry_ts.time() == time(10, 30)
    assert t.stop == pytest.approx(5000.0)
    assert t.target == pytest.approx(5020.25)
    assert t.exit_reason == "target"
    assert t.exit_price == pytest.approx(5020.25)
    assert t.qty == 1  # $1000 / (10.25 pts * $50) = 1.95 -> 1
    assert t.pnl_usd == pytest.approx(10.0 * 50)


def test_short_breakout_stopped_out_at_range_high():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5003, 4998, 4999))  # through 4999.75
    bars.append(bar(DAY, 10, 1, 4999, 5011, 4999, 5010))  # back above the range high -> stop
    bars += flat_bars(DAY, (10, 2), (16, 0), 5010.0)

    result = run_backtest(bars, cfg(slippage_ticks=1))
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.side is Side.SHORT
    assert t.entry_price == pytest.approx(4999.75 - 0.25)
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(5010.0 + 0.25)
    assert t.pnl_usd < 0
    assert t.r_multiple == pytest.approx(-1.0, abs=0.05)


def test_one_trade_per_day_and_opposite_side_not_rearmed():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))  # long entry
    bars.append(bar(DAY, 10, 1, 5011, 5011, 4999, 4999))  # stopped
    bars.append(bar(DAY, 10, 2, 4999, 4999, 4990, 4991))  # would be a short breakout
    bars += flat_bars(DAY, (10, 3), (16, 0), 4991.0)

    result = run_backtest(bars, cfg(max_trades_per_day=1))
    assert [t.side for t in result.trades] == [Side.LONG]

    result = run_backtest(bars, cfg(max_trades_per_day=2))
    assert [t.side for t in result.trades] == [Side.LONG, Side.SHORT]


def test_same_side_never_traded_twice_even_with_two_trades_allowed():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))  # long
    bars.append(bar(DAY, 10, 1, 5011, 5011, 4999, 5005))  # stopped, closes inside the range
    bars.append(bar(DAY, 10, 2, 5005, 5013, 5005, 5012))  # second long break must be ignored
    bars += flat_bars(DAY, (10, 3), (16, 0), 5012.0)

    result = run_backtest(bars, cfg(max_trades_per_day=2))
    assert len(result.trades) == 1


def test_entry_and_stop_inside_one_bar_still_counts_as_a_trade():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 4999, 5000))  # breaks out and reverses through the stop
    bars.append(bar(DAY, 10, 1, 5000, 5013, 5000, 5012))  # new long break must NOT be taken
    bars += flat_bars(DAY, (10, 2), (16, 0), 5012.0)

    result = run_backtest(bars, cfg())
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "stop"


def test_eod_flatten_signals_on_the_bar_before_flatten_at_and_fills_at_that_open():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))
    bars += flat_bars(DAY, (10, 1), (15, 55), 5015.0)  # the 15:54 bar closes at 15:55 -> flatten
    bars.append(bar(DAY, 15, 55, 5016, 5016, 5016, 5016))  # ... fills at this open
    bars += flat_bars(DAY, (15, 56), (16, 0), 5017.0)

    result = run_backtest(bars, cfg())
    t = result.trades[0]
    assert t.exit_reason == "eod"
    assert t.exit_ts.time() == time(15, 55)
    assert t.exit_price == pytest.approx(5016.0)


def test_position_still_open_when_data_ends_is_closed_at_last_close():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))
    bars.append(bar(DAY, 10, 1, 5011, 5013, 5011, 5012.5))
    result = run_backtest(bars, cfg())
    assert result.trades[0].exit_reason == "session_end"
    assert result.trades[0].exit_price == pytest.approx(5012.5)


def test_resting_entries_are_cancelled_at_last_entry_time():
    bars = flat_range(DAY, 5010, 5000)
    bars += flat_bars(DAY, (10, 0), (11, 30), 5005.0)  # the 11:29 bar closes at 11:30 -> cancel
    bars.append(bar(DAY, 11, 30, 5005, 5020, 5005, 5019))  # breakout after the cutoff
    bars += flat_bars(DAY, (11, 31), (16, 0), 5019.0)
    assert run_backtest(bars, cfg(last_entry=time(11, 30))).trades == []
    assert len(run_backtest(bars, cfg(last_entry=time(12, 0))).trades) == 1


def test_entry_orders_are_live_during_the_first_bar_after_the_range():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))  # breaks out on the very first bar
    bars += flat_bars(DAY, (10, 1), (16, 0), 5011.0)
    t = run_backtest(bars, cfg()).trades[0]
    assert t.entry_ts.time() == time(10, 0)
    assert t.entry_price == pytest.approx(5010.25)


def test_five_minute_bars():
    from datetime import timedelta
    from .helpers import NY
    from datetime import datetime
    bars = []
    current = datetime.combine(DAY, time(9, 30), tzinfo=NY)
    for i in range(6):  # 09:30 .. 09:55 -> range 5000-5010
        bars.append(bar(DAY, current.hour, current.minute, 5005, 5010 if i == 0 else 5007, 5000 if i == 0 else 5003, 5005))
        current += timedelta(minutes=5)
    bars.append(bar(DAY, 10, 0, 5005, 5012, 5004, 5011))
    bars.append(bar(DAY, 10, 5, 5011, 5021, 5010, 5020))
    result = run_backtest(bars, cfg(bar_minutes=5))
    assert result.days[0].range_bars == 6
    assert result.days[0].eligible
    assert result.trades[0].entry_ts.time() == time(10, 0)
    assert result.trades[0].exit_reason == "target"


@pytest.mark.parametrize("high,low,reason", [(5001.0, 5000.0, "too small"), (5100.0, 5000.0, "too large")])
def test_range_filters_skip_the_day(high, low, reason):
    bars = flat_range(DAY, high, low)
    bars.append(bar(DAY, 10, 0, high, high + 5, high, high + 4))
    bars += flat_bars(DAY, (10, 1), (16, 0), high + 4)
    result = run_backtest(bars, cfg())
    assert result.trades == []
    assert not result.days[0].eligible
    assert reason in result.days[0].skip_reason


def test_incomplete_range_skips_the_day():
    bars = flat_range(DAY, 5010, 5000, minutes=10)
    bars += flat_bars(DAY, (9, 40), (9, 50), 5005.0)  # 20 bars total, below 80% of 30
    bars.append(bar(DAY, 10, 0, 5005, 5020, 5005, 5019))
    bars += flat_bars(DAY, (10, 1), (16, 0), 5019.0)
    result = run_backtest(bars, cfg())
    assert result.trades == []
    assert "incomplete" in result.days[0].skip_reason


def test_close_mode_enters_at_next_open_after_a_close_beyond_the_range():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5009.0))  # wick above but closes inside: no entry
    bars.append(bar(DAY, 10, 1, 5009, 5012, 5008, 5011.0))  # closes above 5010.25
    bars.append(bar(DAY, 10, 2, 5011.5, 5012, 5011, 5011.5))  # fill at this open
    bars += flat_bars(DAY, (10, 3), (16, 0), 5011.5)
    result = run_backtest(bars, cfg(entry_mode="close"))
    t = result.trades[0]
    assert t.entry_ts.time() == time(10, 2)
    assert t.entry_price == pytest.approx(5011.5)
    assert t.stop == pytest.approx(5000.0)
    assert t.target == pytest.approx(5011.0 + 10.0)  # 1x range from the reference close


def test_stop_modes_and_r_target():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))
    bars += flat_bars(DAY, (10, 1), (16, 0), 5011.0)

    t = run_backtest(bars, cfg(stop_mode="mid")).trades[0]
    assert t.stop == pytest.approx(5005.0)

    t = run_backtest(bars, cfg(stop_mode="fixed", fixed_stop_points=3.0, target_mode="r", target_mult=2.0)).trades[0]
    assert t.stop == pytest.approx(5010.25 - 3.0)
    assert t.target == pytest.approx(5010.25 + 6.0)


def test_position_sizing_respects_risk_budget_and_cap():
    bars = flat_range(DAY, 5010, 5000)  # 10.25 pts risk on a long = $512.50 per ES contract
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))
    bars += flat_bars(DAY, (10, 1), (16, 0), 5011.0)

    assert run_backtest(bars, cfg(risk_per_trade_usd=400.0)).trades == []  # cannot afford one contract
    assert run_backtest(bars, cfg(risk_per_trade_usd=1100.0)).trades[0].qty == 2
    assert run_backtest(bars, cfg(risk_per_trade_usd=5000.0, max_contracts=3)).trades[0].qty == 3
    assert run_backtest(bars, cfg(contract="MES", risk_per_trade_usd=400.0, max_contracts=20)).trades[0].qty == 7


def test_conservative_fill_when_stop_and_target_touch_in_one_bar():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))
    bars.append(bar(DAY, 10, 1, 5011, 5025, 4995, 5010))
    bars += flat_bars(DAY, (10, 2), (16, 0), 5010.0)
    assert run_backtest(bars, cfg(conservative_fills=True)).trades[0].exit_reason == "stop"
    assert run_backtest(bars, cfg(conservative_fills=False)).trades[0].exit_reason == "target"


def test_gap_through_stop_fills_at_open():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))
    bars.append(bar(DAY, 10, 1, 4990, 4992, 4988, 4991))  # opens far below the stop
    bars += flat_bars(DAY, (10, 2), (16, 0), 4991.0)
    t = run_backtest(bars, cfg()).trades[0]
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(4990.0)


def test_daily_loss_limit_blocks_further_trades():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))  # long, risk 10.25 pts = $512.50
    bars.append(bar(DAY, 10, 1, 5011, 5011, 4999, 4999))  # stopped: about -$512
    bars.append(bar(DAY, 10, 2, 4999, 4999, 4990, 4991))  # short break
    bars += flat_bars(DAY, (10, 3), (16, 0), 4991.0)
    assert len(run_backtest(bars, cfg(max_trades_per_day=2)).trades) == 2
    assert len(run_backtest(bars, cfg(max_trades_per_day=2, max_daily_loss_usd=500.0)).trades) == 1


def test_multi_day_state_resets_and_overnight_bars_are_ignored():
    day2 = date(2025, 3, 4)
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bar(DAY, 10, 0, 5002, 5012, 5002, 5011))
    bars += flat_bars(DAY, (10, 1), (16, 0), 5011.0)
    bars += flat_bars(DAY, (18, 0), (23, 59), 5050.0)  # Globex bars must not pollute the next range
    bars += flat_bars(day2, (0, 0), (9, 30), 5050.0)
    bars += flat_range(day2, 5060, 5050)
    bars.append(bar(day2, 10, 0, 5052, 5062, 5052, 5061))
    bars += flat_bars(day2, (10, 1), (16, 0), 5061.0)

    result = run_backtest(bars, cfg())
    assert len(result.days) == 2
    assert (result.days[1].range_high, result.days[1].range_low) == (5060.0, 5050.0)
    assert len(result.trades) == 2
    assert result.trades[0].exit_reason == "eod"


def test_dst_transition_day_uses_new_york_wall_clock():
    from datetime import timezone

    day = date(2025, 3, 10)  # first trading day after US clocks spring forward
    bars = flat_range(day, 5010, 5000)
    # the 09:30 ET bar is 13:30 UTC on this day (EDT), not 14:30
    assert bars[0].ts.astimezone(timezone.utc).hour == 13
    bars.append(bar(day, 10, 0, 5002, 5012, 5002, 5011))
    bars += flat_bars(day, (10, 1), (16, 0), 5011.0)
    result = run_backtest(bars, cfg())
    assert result.days[0].range_bars == 30
    assert len(result.trades) == 1


def test_bars_out_of_order_are_rejected():
    bars = flat_range(DAY, 5010, 5000)
    bars.append(bars[0])
    with pytest.raises(ValueError):
        run_backtest(bars, cfg())


def test_unaffordable_side_is_skipped_once_per_day_not_every_bar():
    bars = flat_range(DAY, 5010, 5000)
    bars += flat_bars(DAY, (10, 0), (16, 0), 5005.0)
    result = run_backtest(bars, cfg(risk_per_trade_usd=400.0))
    assert result.trades == []
    assert len(result.skipped_entries) == 2  # one long, one short
