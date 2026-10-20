"""Single-day battery dispatch as an LP (or MILP) in PuLP, solved with CBC.

Variables per interval t: charge c_t and discharge d_t (MW), end-of-interval
state of charge s_t (MWh). Energy-only arbitrage, price-taker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pulp

from bess.config import Asset


# CBC writes solution values to 8 significant digits, so a 100 MW / 200 MWh asset shows
# energy-balance residuals up to ~1e-5 MWh from output rounding alone. Anything above this
# tolerance is a real constraint breach.
FEASIBILITY_TOL = 1e-4


@dataclass(frozen=True)
class Dispatch:
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    objective: float


def solve_day(
    prices: np.ndarray, dt: float, asset: Asset, milp: bool = False
) -> Dispatch:
    """Optimal dispatch for one day on the given price vector.

    prices: $/MWh per interval; dt: interval length in hours.
    milp: add a binary per interval so charge and discharge cannot overlap.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or len(prices) == 0:
        raise ValueError("prices must be a non-empty 1-D array")
    if not np.isfinite(prices).all():
        raise ValueError("prices contain NaN or inf")

    n = len(prices)
    eta = asset.eta
    P = asset.power_mw
    prob = pulp.LpProblem("bess_day", pulp.LpMaximize)

    c = [pulp.LpVariable(f"c{t}", 0, P) for t in range(n)]
    d = [pulp.LpVariable(f"d{t}", 0, P) for t in range(n)]
    s = [pulp.LpVariable(f"s{t}", asset.soc_min, asset.soc_max) for t in range(n)]

    deg = asset.degradation_usd_per_mwh
    prob += pulp.lpSum(
        float(prices[t]) * (d[t] - c[t]) * dt - deg * d[t] * dt for t in range(n)
    )

    for t in range(n):
        prev = asset.soc_start if t == 0 else s[t - 1]
        prob += s[t] == prev + eta * c[t] * dt - d[t] * dt / eta
        prob += c[t] + d[t] <= P
    prob += s[n - 1] == asset.soc_start
    prob += pulp.lpSum(d[t] * dt for t in range(n)) <= asset.daily_discharge_cap_mwh

    if milp:
        u = [pulp.LpVariable(f"u{t}", cat=pulp.LpBinary) for t in range(n)]
        for t in range(n):
            prob += c[t] <= P * u[t]
            prob += d[t] <= P * (1 - u[t])

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"CBC returned {pulp.LpStatus[status]}")

    return Dispatch(
        charge_mw=np.array([v.value() or 0.0 for v in c]),
        discharge_mw=np.array([v.value() or 0.0 for v in d]),
        soc_mwh=np.array([v.value() or 0.0 for v in s]),
        objective=pulp.value(prob.objective),
    )


def settle(
    charge_mw: np.ndarray,
    discharge_mw: np.ndarray,
    prices: np.ndarray,
    dt: float,
    asset: Asset,
) -> float:
    """Net revenue ($) of a dispatch schedule settled at `prices`."""
    gross = float(np.sum(prices * (discharge_mw - charge_mw) * dt))
    return gross - asset.degradation_usd_per_mwh * float(np.sum(discharge_mw) * dt)


def simultaneous_intervals(
    charge_mw: np.ndarray, discharge_mw: np.ndarray, tol: float
) -> int:
    """Intervals where charge and discharge are both above tol."""
    return int(np.sum((charge_mw > tol) & (discharge_mw > tol)))


def dispatch_violations(
    disp: Dispatch, dt: float, asset: Asset
) -> dict[str, float]:
    """Largest breach of each constraint; every value should be <= FEASIBILITY_TOL."""
    c, d, s = disp.charge_mw, disp.discharge_mw, disp.soc_mwh
    prev = np.concatenate([[asset.soc_start], s[:-1]])
    balance = s - prev - (asset.eta * c * dt - d * dt / asset.eta)
    return {
        "energy_balance": float(np.max(np.abs(balance))),
        "terminal_soc": abs(float(s[-1]) - asset.soc_start),
        "soc_low": max(0.0, float(asset.soc_min - s.min())),
        "soc_high": max(0.0, float(s.max() - asset.soc_max)),
        "power": max(0.0, float((c + d).max() - asset.power_mw)),
        "negative_power": max(0.0, float(-min(c.min(), d.min()))),
        "throughput": max(0.0, float(d.sum() * dt - asset.daily_discharge_cap_mwh)),
    }
