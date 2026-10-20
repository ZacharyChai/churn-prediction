"""Check the pre/post and placebo window logic in sql/metrics against hand-computed values.

The daily table here is a test fixture with round numbers, not model output.
"""
import pandas as pd
import pytest

from bess.metrics import run_metrics


@pytest.fixture(scope="module")
def metrics(cfg):
    days = pd.date_range("2023-01-01", "2026-09-30")
    rows = []
    for d in days:
        s2 = 70.0 if d >= pd.Timestamp("2025-12-01") else 50.0
        for strat, rev in (("S1", 100.0), ("S2", s2), ("S3", 40.0)):
            rows.append(("HB_WEST", "base", d, strat, rev, 0, 96))
    daily = pd.DataFrame(
        rows,
        columns=["hub", "scenario", "delivery_date", "strategy", "revenue_usd",
                 "simultaneous_intervals", "intervals"],
    )
    form = pd.DataFrame(
        [{"hub": "HB_WEST", "scenario": "base", "formulation": "LP", "lp_simultaneity_share": 0.0}]
    )
    return run_metrics(cfg, daily, form)


def test_actual_window_is_dec_to_latest_complete_month_vs_year_earlier(metrics):
    r = metrics["rtcb_vs_placebo"].set_index("label").loc["rtcb"]
    assert r["post_start"] == pd.Timestamp("2025-12-01")
    assert r["post_end"] == pd.Timestamp("2026-09-30")
    assert r["pre_start"] == pd.Timestamp("2024-12-01")
    assert r["pre_end"] == pd.Timestamp("2025-09-30")
    assert r["n_months"] == 10
    assert r["capture_s2_post"] == pytest.approx(0.70)
    assert r["capture_s2_pre"] == pytest.approx(0.50)
    assert r["change_s2_pp"] == pytest.approx(20.0)
    assert r["change_s3_pp"] == pytest.approx(0.0)
    assert bool(r["windows_covered"])


def test_placebo_with_no_change_shows_zero_and_uncovered_placebo_is_flagged(metrics):
    t = metrics["rtcb_vs_placebo"].set_index("label")
    p24 = t.loc["placebo_2024-12-05"]
    assert p24["post_start"] == pd.Timestamp("2024-12-01")
    assert p24["post_end"] == pd.Timestamp("2025-09-30")
    assert p24["change_s2_pp"] == pytest.approx(0.0)
    assert bool(p24["windows_covered"])
    p23 = t.loc["placebo_2023-12-05"]  # needs Dec 2022, before the data starts
    assert not bool(p23["windows_covered"])
    assert pd.isna(p23["change_s2_pp"])


def test_capture_rate_and_monthly_revenue_values(metrics):
    cap = metrics["capture_rate"].set_index("month")
    assert cap.loc[pd.Timestamp("2025-01-01"), "capture_s2"] == pytest.approx(0.5)
    assert cap.loc[pd.Timestamp("2026-01-01"), "capture_s2"] == pytest.approx(0.7)
    mr = metrics["monthly_revenue"].query("strategy == 'S1' and month == @pd.Timestamp('2025-02-01')").iloc[0]
    assert mr["revenue_usd"] == pytest.approx(28 * 100.0)
    assert mr["usd_per_kw_month"] == pytest.approx(28 * 100.0 / 100_000)
