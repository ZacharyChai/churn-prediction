# ERCOT battery dispatch backtest

Energy-arbitrage backtest for a 100 MW / 200 MWh battery at ERCOT hubs. A linear program
(PuLP, CBC) schedules the battery each day. Three strategies are compared against a
perfect-foresight upper bound, with a before/after look at RTC+B (live 5 Dec 2025).

**Status:** code and tests are complete; the backtest has not been run on ERCOT data yet, so
there are no results in this repo. See `FINDINGS.md`.

## Question (fixed before any results were seen)
How much of the theoretical energy-arbitrage revenue at an ERCOT hub can a simple,
implementable strategy capture, and did that change after RTC+B went live on 5 Dec 2025?

- **Primary metric:** capture rate = realized revenue of the implementable strategy / perfect-foresight revenue, by month. Pooled over a window as sum of revenue over sum of perfect-foresight revenue.
- **Comparison:** Dec 2025 through the latest complete month, against the same calendar months one year earlier.
- **Falsification test:** repeat with placebo breaks at 5 Dec 2024 and 5 Dec 2023 (same window length, same one-year-earlier comparison). The 2023 placebo needs Dec 2022 prices, which the Jan 2023 start does not have, so it is reported as not covered.
- **Descriptive only.** Battery buildout (ERCOT passed about 15 GW of installed BESS in Q1 2026), weather and gas prices all move spreads. A null is written up as a null.

## Strategies
All three use the same LP. Only the price input and the settlement price differ.

| ID | Optimizes on | Settles at | Role |
|---|---|---|---|
| S1 Perfect foresight | Actual RT 15-min prices | RT | Upper bound |
| S2 Day-ahead schedule | DAM hourly prices | DAM | Implementable (primary) |
| S3 Naive RT | Yesterday's RT prices at the same local clock times | Today's RT | Implementable, forecast baseline |

## Asset assumptions
All in `config.toml`.

| Parameter | Value |
|---|---|
| Power / energy | 100 MW / 200 MWh |
| Round-trip efficiency | 85%, split evenly (sqrt(0.85) each way) |
| State of charge | 5% to 95%; each day starts and ends at 50% |
| Throughput cap | 1 cycle/day (200 MWh discharged); sensitivities at 1.5 and 2 |
| Degradation cost | $0/MWh; sensitivity at $10/MWh of discharge |
| Market impact | None (price-taker) |
| Locations | HB_WEST (primary), HB_HOUSTON (robustness) |
| Window | Jan 2023 to latest complete month |

## Model
Per day, decision variables are charge, discharge (MW) and state of charge (MWh) per interval.
Maximize price x (discharge - charge) x dt, minus degradation cost x discharged MWh, subject to the
state-of-charge balance, SoC and power bounds, charge + discharge <= P, the daily throughput cap,
and start = end SoC. This is an LP relaxation: it does not force a choice between charging and
discharging. If more than 1% of intervals charge and discharge at once, that hub/scenario is
re-solved as a MILP with a binary per interval. `results/formulation.parquet` records which
formulation produced each result.

Days are the local Central calendar day, so DST days have 92 or 100 15-minute intervals and 23 or
25 hourly intervals. Days with missing or extra intervals are skipped and listed in
`results/skipped_days.parquet`. Capture rates use only days where all three strategies exist.

## Reproduce
```
pip install -r requirements.txt
python -m pytest -q                 # no network or data needed
python scripts/load_data.py         # needs access to ercot.com via gridstatus
python scripts/check_data.py        # must pass before the backtest
python scripts/run_backtest.py      # writes results/ and powerbi/*.csv
```
Every metric comes from a query in `sql/metrics/`. The pipeline needs no accounts: raw data goes
to Parquet in `data/raw/`, DuckDB queries it. Set `BESS_CONFIG` to use a different config file.

## Power BI
`powerbi/*.csv` are the inputs for the one-page report (monthly $/kW-month by strategy, capture
rate with the 5 Dec 2025 marker, pre/post vs placebo bars, top 10 days). The `.pbix` and
screenshots are not in the repo yet.

## Limits and known deviations
- **Energy arbitrage only.** No ancillary services, no bid/offer curves, no trading, no price forecasting beyond S3, no RTC+B mechanics.
- **S2 is not strictly bounded by S1.** S2 settles at DAM prices and S1 at RT prices, so a day where DAM spreads beat RT spreads can give S2 > S1. `sql/metrics/05_bound_checks.sql` counts these days. The hard bug check is S1 >= S3, which must hold every day.
- **Feasibility tolerance is 1e-4 MWh, not 1e-6.** CBC writes solution values to 8 significant digits, so energy-balance residuals reach about 1e-5 MWh from rounding alone.
- **LP simultaneity can be a tie, not a gain.** At sustained negative prices the LP can pick an optimum that charges and discharges in the same interval; the MILP reaches the same revenue by alternating intervals. The 1% rule is applied mechanically anyway.
- **S3 on odd days.** Repeated fall-back clock times in yesterday's prices are averaged; clock times missing yesterday (spring-forward gap) take the previous slot's price.
- **No uncertainty intervals.** Window capture rates are point estimates.
- **PuLP is pinned to 2.9.0** because 4.x does not bundle CBC.
