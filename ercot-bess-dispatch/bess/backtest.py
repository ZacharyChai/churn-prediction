"""Daily backtest of the three strategies (S1 / S2 / S3) per hub and scenario.

All three use the same LP. Only the price the schedule is optimised on and the
price it is settled at differ:

    S1 perfect foresight  optimise on today's RT 15-min,   settle at today's RT
    S2 day-ahead schedule optimise on today's DAM hourly,  settle at today's DAM
    S3 naive RT           optimise on yesterday's RT 15-min (matched by local
                          clock time), settle at today's RT
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from bess.config import Asset, Config
from bess.lp import (
    FEASIBILITY_TOL,
    dispatch_violations,
    settle,
    simultaneous_intervals,
    solve_day,
)

RT_DT = 0.25
DAM_DT = 1.0
STRATEGIES = ("S1", "S2", "S3")
ALL_CLOCKS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]


@dataclass(frozen=True)
class DayTask:
    delivery_date: object
    rt_prices: np.ndarray
    dam_prices: np.ndarray
    rt_forecast: np.ndarray | None  # yesterday's RT prices at today's clock times


def _clock_key(local_ts: pd.Series) -> pd.Series:
    return local_ts.dt.strftime("%H:%M")


def build_tasks(
    con: duckdb.DuckDBPyConnection, hub: str
) -> tuple[list[DayTask], pd.DataFrame]:
    """One DayTask per complete day. Returns (tasks, skipped-days frame)."""
    rt = con.execute(
        "SELECT r.delivery_date, r.interval_start_local, r.spp, "
        "c.expected_rtm FROM rtm r JOIN calendar c USING (delivery_date) "
        "WHERE r.location = ? ORDER BY r.interval_start_utc",
        [hub],
    ).df()
    dam = con.execute(
        "SELECT d.delivery_date, d.spp, c.expected_dam "
        "FROM dam d JOIN calendar c USING (delivery_date) "
        "WHERE d.location = ? ORDER BY d.interval_start_utc",
        [hub],
    ).df()
    cal = con.execute(
        "SELECT delivery_date FROM calendar ORDER BY delivery_date"
    ).df()["delivery_date"]

    rt["clock"] = _clock_key(rt["interval_start_local"])
    rt_days = {k: g for k, g in rt.groupby("delivery_date", sort=True)}
    dam_days = {k: g for k, g in dam.groupby("delivery_date", sort=True)}

    complete: dict = {}
    skipped = []
    for day in cal:
        r, d = rt_days.get(day), dam_days.get(day)
        if r is None or d is None:
            skipped.append((hub, day, "missing RT or DAM"))
        elif len(r) != r["expected_rtm"].iloc[0] or len(d) != d["expected_dam"].iloc[0]:
            skipped.append((hub, day, "interval count mismatch"))
        elif not (np.isfinite(r["spp"]).all() and np.isfinite(d["spp"]).all()):
            skipped.append((hub, day, "non-finite price"))
        else:
            complete[day] = (r, d)

    tasks = []
    for day, (r, d) in complete.items():
        prev = complete.get(day - pd.DateOffset(days=1))
        forecast = None
        if prev is not None:
            # Match by local clock time. Repeated clock times (fall-back hour) average within the
            # day; clock times missing yesterday (spring-forward gap) take the previous slot's price.
            by_clock = prev[0].groupby("clock")["spp"].mean()
            by_clock = by_clock.reindex(ALL_CLOCKS).ffill().bfill()
            forecast = by_clock.reindex(r["clock"]).to_numpy()
        tasks.append(
            DayTask(day, r["spp"].to_numpy(), d["spp"].to_numpy(), forecast)
        )
    skipped_df = pd.DataFrame(skipped, columns=["hub", "delivery_date", "reason"])
    return tasks, skipped_df


def _run_strategy(
    strategy: str, task: DayTask, asset: Asset, milp: bool, tol: float
) -> dict | None:
    if strategy == "S1":
        opt_prices, opt_dt, settle_prices, settle_dt = (
            task.rt_prices, RT_DT, task.rt_prices, RT_DT,
        )
    elif strategy == "S2":
        opt_prices, opt_dt, settle_prices, settle_dt = (
            task.dam_prices, DAM_DT, task.dam_prices, DAM_DT,
        )
    else:
        if task.rt_forecast is None:
            return None
        opt_prices, opt_dt, settle_prices, settle_dt = (
            task.rt_forecast, RT_DT, task.rt_prices, RT_DT,
        )

    disp = solve_day(opt_prices, opt_dt, asset, milp=milp)
    worst = max(dispatch_violations(disp, opt_dt, asset).values())
    if worst > FEASIBILITY_TOL:
        raise AssertionError(
            f"{strategy} {task.delivery_date}: constraint breach {worst:.2e}"
        )
    return {
        "delivery_date": task.delivery_date,
        "strategy": strategy,
        "revenue_usd": settle(
            disp.charge_mw, disp.discharge_mw, settle_prices, settle_dt, asset
        ),
        "charged_mwh": float(disp.charge_mw.sum() * opt_dt),
        "discharged_mwh": float(disp.discharge_mw.sum() * opt_dt),
        "intervals": len(opt_prices),
        "simultaneous_intervals": simultaneous_intervals(
            disp.charge_mw, disp.discharge_mw, tol
        ),
    }


def run_day(args: tuple) -> list[dict]:
    task, asset, milp, tol = args
    rows = (_run_strategy(s, task, asset, milp, tol) for s in STRATEGIES)
    return [r for r in rows if r is not None]


def run_hub_scenario(
    tasks: list[DayTask], asset: Asset, milp: bool, cfg: Config
) -> pd.DataFrame:
    tol = cfg.solver["simultaneity_tolerance_mw"]
    jobs = [(t, asset, milp, tol) for t in tasks]
    workers = cfg.solver["workers"]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(run_day, jobs, chunksize=8))
    else:
        chunks = [run_day(j) for j in jobs]
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def simultaneity_share(daily: pd.DataFrame) -> float:
    """Largest share of intervals (across strategies) with charge and discharge both on."""
    by = daily.groupby("strategy")[["simultaneous_intervals", "intervals"]].sum()
    return float((by["simultaneous_intervals"] / by["intervals"]).max())


def run_all(cfg: Config, con: duckdb.DuckDBPyConnection):
    """Backtest every hub x scenario. Returns (daily, formulation, skipped)."""
    threshold = cfg.solver["simultaneity_threshold"]
    daily_parts, form_rows, skipped_parts = [], [], []
    for hub in cfg.data["hubs"]:
        tasks, skipped = build_tasks(con, hub)
        skipped_parts.append(skipped)
        for sc in cfg.scenarios:
            df = run_hub_scenario(tasks, sc.asset, milp=False, cfg=cfg)
            share_lp = simultaneity_share(df)
            formulation = "LP"
            if share_lp > threshold:
                df = run_hub_scenario(tasks, sc.asset, milp=True, cfg=cfg)
                formulation = "MILP"
            df.insert(0, "scenario", sc.name)
            df.insert(0, "hub", hub)
            daily_parts.append(df)
            form_rows.append(
                {
                    "hub": hub,
                    "scenario": sc.name,
                    "formulation": formulation,
                    "lp_simultaneity_share": share_lp,
                }
            )
    return (
        pd.concat(daily_parts, ignore_index=True),
        pd.DataFrame(form_rows),
        pd.concat(skipped_parts, ignore_index=True),
    )
