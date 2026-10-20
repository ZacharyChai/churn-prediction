"""Run the data checks in sql/checks against the loaded Parquet. Exit 1 on any failure.

Run this before the backtest. A full year must load cleanly first.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bess.config import load_config  # noqa: E402
from bess.data import build_db  # noqa: E402


def main() -> int:
    cfg = load_config()
    con = build_db(cfg)
    print(con.execute("SELECT * FROM (" + (ROOT / "sql/checks/year_totals.sql").read_text().rstrip().rstrip(";") + ")").df().to_string(index=False))
    failed = False
    for name in ("duplicates", "interval_counts"):
        bad = con.execute((ROOT / f"sql/checks/{name}.sql").read_text()).df()
        status = "OK" if bad.empty else f"FAIL ({len(bad)} rows)"
        print(f"\n{name}: {status}")
        if not bad.empty:
            failed = True
            print(bad.head(20).to_string(index=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
