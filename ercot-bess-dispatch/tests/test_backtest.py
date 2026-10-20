from datetime import date

import numpy as np
import pandas as pd
import pytest

from bess.backtest import build_tasks, run_all
from bess.metrics import run_metrics
from conftest import make_db, synthetic_prices

DAYS = pd.date_range("2025-03-06", "2025-03-12")  # spring-forward on the 9th


@pytest.fixture(scope="module")
def results(cfg):
    rtm, dam = synthetic_prices(DAYS, seed=1)
    con = make_db(rtm, dam)
    return con, run_all(cfg, con)


def test_one_task_per_day_and_s3_forecast_alignment(results):
    con, _ = results
    tasks, skipped = build_tasks(con, "HB_WEST")
    assert skipped.empty and len(tasks) == len(DAYS)
    by_day = {pd.Timestamp(t.delivery_date).date(): t for t in tasks}
    assert by_day[date(2025, 3, 6)].rt_forecast is None  # no prior day in the window
    spring = by_day[date(2025, 3, 9)]
    assert len(spring.rt_prices) == 92 and len(spring.rt_forecast) == 92
    after = by_day[date(2025, 3, 10)]
    assert len(after.rt_forecast) == 96
    # yesterday (the 9th) has no 02:00-02:45; those slots take the 01:45 price
    assert after.rt_forecast[8] == pytest.approx(spring.rt_prices[7])
    assert after.rt_forecast[11] == pytest.approx(spring.rt_prices[7])
    # forecast is literally yesterday's price at the same clock time
    assert after.rt_forecast[0] == pytest.approx(spring.rt_prices[0])


def test_incomplete_day_is_skipped_and_breaks_next_days_forecast(cfg):
    rtm, dam = synthetic_prices(DAYS, seed=1)
    hole = rtm[rtm["delivery_date"] == date(2025, 3, 8)].index[3]
    con = make_db(rtm.drop(index=hole), dam)
    tasks, skipped = build_tasks(con, "HB_WEST")
    assert skipped["delivery_date"].dt.date.tolist() == [date(2025, 3, 8)]
    by_day = {pd.Timestamp(t.delivery_date).date(): t for t in tasks}
    assert by_day[date(2025, 3, 9)].rt_forecast is None
    assert by_day[date(2025, 3, 10)].rt_forecast is not None


def test_perfect_foresight_dominates_naive_rt_every_day(results):
    daily, _, _ = results[1]
    wide = daily.pivot(index="delivery_date", columns="strategy", values="revenue_usd")
    both = wide.dropna(subset=["S1", "S3"])
    assert len(both) == len(DAYS) - 1
    assert (both["S1"] >= both["S3"] - 1e-6).all()
    assert (wide["S1"] > 0).all()


def test_interval_counts_per_strategy(results):
    daily, _, _ = results[1]
    s1 = daily[daily.strategy == "S1"].set_index("delivery_date")["intervals"]
    assert s1[pd.Timestamp("2025-03-09")] == 92
    s2 = daily[daily.strategy == "S2"].set_index("delivery_date")["intervals"]
    assert s2[pd.Timestamp("2025-03-09")] == 23


def test_formulation_reported_and_lp_used_on_ordinary_prices(results):
    _, form, _ = results[1]
    assert form["formulation"].tolist() == ["LP"]
    assert form["lp_simultaneity_share"].iloc[0] <= 0.01


def test_exceeding_simultaneity_threshold_switches_to_milp(cfg):
    from dataclasses import replace

    forced = replace(cfg, solver={**cfg.solver, "simultaneity_threshold": -1.0})
    rtm, dam = synthetic_prices(DAYS, seed=2, negative_days=DAYS[:5])
    con = make_db(rtm, dam)
    daily, form, _ = run_all(forced, con)
    assert form["formulation"].iloc[0] == "MILP"
    assert daily["simultaneous_intervals"].sum() == 0
    # S1 >= S3 must still hold under the MILP
    wide = daily.pivot(index="delivery_date", columns="strategy", values="revenue_usd").dropna()
    assert (wide["S1"] >= wide["S3"] - 1e-6).all()


def test_metrics_sql_runs_and_is_consistent(cfg, results):
    daily, form, _ = results[1]
    m = run_metrics(cfg, daily, form)
    assert set(m) == {
        "monthly_revenue", "capture_rate", "rtcb_vs_placebo",
        "top_days", "bound_checks", "simultaneity",
    }
    cap = m["capture_rate"]
    assert (cap["capture_s3"] <= 1 + 1e-9).all()
    assert m["bound_checks"]["s3_above_s1_bug"].sum() == 0
    assert len(m["top_days"]) <= 10
    # $/kW-month: revenue over 100 MW = 100,000 kW, scaled to the full calendar month
    mr = m["monthly_revenue"].query("strategy == 'S1'").iloc[0]
    assert mr["usd_per_kw_raw"] == pytest.approx(mr["revenue_usd"] / 100_000)
    assert mr["usd_per_kw_month"] == pytest.approx(
        mr["usd_per_kw_raw"] * mr["days_in_month"] / mr["days"]
    )
    # the window ends before RTC+B, so no real windows are covered
    assert not m["rtcb_vs_placebo"]["windows_covered"].any()
