# Findings

**Status: not run.** The backtest has not been executed on ERCOT data yet. This file gets
filled in from `results/*.parquet` only, after `scripts/run_backtest.py` completes. No number
below may be typed in by hand.

## Headline capture rates (HB_WEST, base scenario)
Pending. Source: `results/capture_rate.parquet`.

## Pre vs post RTC+B (go-live 5 Dec 2025)
Pending. Source: `results/rtcb_vs_placebo.parquet`, row `rtcb`.

## Placebo comparison (5 Dec 2024, 5 Dec 2023)
Pending. Source: same table, rows `placebo_*`. If a placebo shows a change of similar size to the
actual window, the post-RTC+B change is not attributed to RTC+B. A null is reported as a null.

## Robustness
Pending: HB_HOUSTON, 1.5 and 2 cycles per day, $10/MWh degradation cost.

## Limits
See README. The study is descriptive; battery buildout, weather and gas prices all move spreads.
