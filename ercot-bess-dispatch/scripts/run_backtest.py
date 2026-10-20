"""Run the full backtest and write results/ and powerbi/ outputs.

    python scripts/check_data.py && python scripts/run_backtest.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bess.backtest import run_all  # noqa: E402
from bess.config import load_config  # noqa: E402
from bess.data import build_db  # noqa: E402
from bess.metrics import run_metrics  # noqa: E402


def main() -> None:
    cfg = load_config()
    con = build_db(cfg)
    daily, formulation, skipped = run_all(cfg, con)

    out = cfg.path("results_dir")
    out.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out / "daily.parquet", index=False)
    formulation.to_parquet(out / "formulation.parquet", index=False)
    skipped.to_parquet(out / "skipped_days.parquet", index=False)

    metrics = run_metrics(cfg, daily, formulation)
    pbi = cfg.root / "powerbi"
    pbi.mkdir(exist_ok=True)
    for name, df in metrics.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        df.to_csv(pbi / f"{name}.csv", index=False)
        print(f"{name}: {len(df):,} rows")
    print(f"skipped days: {len(skipped):,}")
    print(formulation.to_string(index=False))


if __name__ == "__main__":
    main()
