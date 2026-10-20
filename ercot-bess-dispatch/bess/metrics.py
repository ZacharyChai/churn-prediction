"""Run the SQL in sql/metrics against backtest output. Every reported number comes from these queries."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from bess.config import ROOT, Config

SQL_DIR = ROOT / "sql" / "metrics"


def run_metrics(
    cfg: Config, daily: pd.DataFrame, formulation: pd.DataFrame, sql_dir: Path = SQL_DIR
) -> dict[str, pd.DataFrame]:
    con = duckdb.connect()
    con.register("daily", daily)
    con.register("formulation", formulation)
    params = pd.DataFrame(
        {"primary_hub": [cfg.data["primary_hub"]], "power_mw": [cfg.asset.power_mw]}
    )
    con.register("params", params)
    breaks = pd.DataFrame(
        [("rtcb", "actual", cfg.study["rtcb_go_live"])]
        + [(f"placebo_{d.isoformat()}", "placebo", d) for d in cfg.study["placebo_breaks"]],
        columns=["label", "kind", "break_date"],
    )
    breaks["break_date"] = pd.to_datetime(breaks["break_date"])
    con.register("breaks", breaks)

    con.execute((sql_dir / "00_views.sql").read_text())
    out = {}
    for f in sorted(sql_dir.glob("[0-9][0-9]_*.sql")):
        if f.name.startswith("00_"):
            continue
        out[f.stem[3:]] = con.execute(f.read_text()).df()
    return out
