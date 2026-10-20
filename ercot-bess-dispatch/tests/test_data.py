from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from bess.data import calendar, local_day_hours, standardize
from conftest import make_db, synthetic_prices

SQL = Path(__file__).resolve().parent.parent / "sql" / "checks"
DAYS = pd.date_range("2025-03-07", "2025-03-11")  # spans spring-forward on the 9th


def _run(con, name):
    return con.execute((SQL / f"{name}.sql").read_text()).df()


@pytest.mark.parametrize(
    "d,hours", [(date(2025, 3, 8), 24), (date(2025, 3, 9), 23), (date(2025, 11, 2), 25)]
)
def test_dst_aware_day_length(d, hours):
    assert local_day_hours(d, "America/Chicago") == hours


def test_calendar_interval_counts_and_year_total():
    cal = calendar(date(2025, 1, 1), date(2025, 12, 31), "America/Chicago")
    row = cal.set_index("delivery_date")
    assert row.loc[date(2025, 3, 9), "expected_rtm"] == 92
    assert row.loc[date(2025, 11, 2), "expected_rtm"] == 100
    assert row.loc[date(2025, 6, 1), "expected_rtm"] == 96
    # DST adds and removes an hour, so a full non-leap year is 35,040 intervals.
    assert cal["expected_rtm"].sum() == 35_040
    assert cal["expected_dam"].sum() == 8_760


def test_standardize_requires_timezone_and_handles_dst():
    rtm, dam = synthetic_prices(DAYS)
    counts = rtm.groupby("delivery_date").size()
    assert counts[date(2025, 3, 9)] == 92
    assert counts[date(2025, 3, 8)] == 96
    assert dam.groupby("delivery_date").size()[date(2025, 3, 9)] == 23
    assert rtm["interval_start_utc"].is_unique
    naive = pd.DataFrame(
        {"Interval Start": pd.to_datetime(["2025-01-01"]), "Location": "HB_WEST", "SPP": [1.0]}
    )
    with pytest.raises(ValueError):
        standardize(naive, "America/Chicago")


def test_fall_back_day_has_100_distinct_utc_intervals():
    rtm, _ = synthetic_prices(pd.date_range("2025-11-01", "2025-11-03"))
    day = rtm[rtm["delivery_date"] == date(2025, 11, 2)]
    assert len(day) == 100
    assert day["interval_start_utc"].is_unique
    assert day["interval_start_local"].duplicated().sum() == 4  # repeated 01:00-02:00 hour


def test_checks_pass_on_clean_data():
    rtm, dam = synthetic_prices(DAYS)
    con = make_db(rtm, dam)
    assert _run(con, "duplicates").empty
    assert _run(con, "interval_counts").empty


def test_checks_catch_duplicate_timestamp():
    rtm, dam = synthetic_prices(DAYS)
    con = make_db(pd.concat([rtm, rtm.iloc[[5]]]), dam)
    assert len(_run(con, "duplicates")) == 1
    assert not _run(con, "interval_counts").empty  # the extra row also breaks that day's count


def test_checks_catch_missing_interval_and_missing_day():
    rtm, dam = synthetic_prices(DAYS)
    con = make_db(rtm.drop(index=10), dam)
    bad = _run(con, "interval_counts")
    assert len(bad) == 1
    assert bad["actual_rtm"].iloc[0] == bad["expected_rtm"].iloc[0] - 1
    gap = dam[dam["delivery_date"] != date(2025, 3, 8)]
    bad = _run(make_db(rtm, gap), "interval_counts")
    assert bad["delivery_date"].tolist() == [pd.Timestamp("2025-03-08")]
    assert bad["actual_dam"].iloc[0] == 0
