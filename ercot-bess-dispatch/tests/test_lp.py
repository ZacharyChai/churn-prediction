import numpy as np
import pytest

from bess.lp import FEASIBILITY_TOL, Dispatch, dispatch_violations, settle, simultaneous_intervals, solve_day


def test_two_price_day_matches_hand_calculation(asset):
    # 12 cheap hours then 12 expensive hours. SoC starts and ends at 100 MWh and peaks at 190,
    # so at most 90 MWh is stored: 90/eta bought at the low price, 90*eta sold at the high price.
    low, high = 20.0, 100.0
    prices = np.array([low] * 12 + [high] * 12)
    disp = solve_day(prices, 1.0, asset)
    eta = asset.eta
    stored = asset.soc_max - asset.soc_start
    expected = high * stored * eta - low * stored / eta
    revenue = settle(disp.charge_mw, disp.discharge_mw, prices, 1.0, asset)
    assert revenue == pytest.approx(expected, abs=1e-4)
    assert disp.objective == pytest.approx(expected, abs=1e-4)
    assert disp.charge_mw[:12].sum() > 0 and disp.charge_mw[12:].sum() == pytest.approx(0, abs=1e-9)
    assert disp.discharge_mw[12:].sum() > 0 and disp.discharge_mw[:12].sum() == pytest.approx(0, abs=1e-9)


def test_throughput_cap_binds_and_matches_hand_calculation(asset):
    from dataclasses import replace

    capped = replace(asset, cycles_per_day=0.25)  # 50 MWh discharged
    prices = np.array([20.0] * 12 + [100.0] * 12)
    disp = solve_day(prices, 1.0, capped)
    assert disp.discharge_mw.sum() == pytest.approx(50.0, abs=1e-6)
    expected = 100.0 * 50.0 - 20.0 * 50.0 / capped.round_trip_efficiency
    assert settle(disp.charge_mw, disp.discharge_mw, prices, 1.0, capped) == pytest.approx(expected, abs=1e-4)


def test_flat_prices_earn_nothing(asset):
    disp = solve_day(np.full(96, 40.0), 0.25, asset)
    assert settle(disp.charge_mw, disp.discharge_mw, np.full(96, 40.0), 0.25, asset) == pytest.approx(0, abs=1e-6)


def test_degradation_cost_reduces_revenue_and_matches_settle(asset):
    from dataclasses import replace

    deg = replace(asset, degradation_usd_per_mwh=10.0)
    prices = np.array([20.0] * 12 + [100.0] * 12)
    base = solve_day(prices, 1.0, asset)
    withdeg = solve_day(prices, 1.0, deg)
    assert withdeg.objective < base.objective
    assert withdeg.objective == pytest.approx(
        settle(withdeg.charge_mw, withdeg.discharge_mw, prices, 1.0, deg), abs=1e-4
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("dt,n", [(0.25, 96), (0.25, 92), (0.25, 100), (1.0, 24)])
def test_constraints_hold_on_random_days(asset, seed, dt, n):
    rng = np.random.default_rng(seed)
    prices = 30 + 40 * rng.standard_normal(n)  # includes negative prices
    disp = solve_day(prices, dt, asset)
    v = dispatch_violations(disp, dt, asset)
    assert max(v.values()) <= FEASIBILITY_TOL, v


def test_milp_removes_simultaneous_charge_discharge(asset):
    # At sustained negative prices the LP can pick an optimum that charges and discharges in the
    # same interval; the MILP forbids it and reaches the same revenue by alternating intervals.
    prices = np.full(96, -50.0)
    lp = solve_day(prices, 0.25, asset)
    milp = solve_day(prices, 0.25, asset, milp=True)
    assert simultaneous_intervals(lp.charge_mw, lp.discharge_mw, 1e-6) > 0
    assert simultaneous_intervals(milp.charge_mw, milp.discharge_mw, 1e-6) == 0
    assert milp.objective == pytest.approx(lp.objective, abs=1e-3)


def test_rejects_bad_prices(asset):
    with pytest.raises(ValueError):
        solve_day(np.array([1.0, np.nan]), 1.0, asset)
    with pytest.raises(ValueError):
        solve_day(np.array([]), 1.0, asset)


def test_violation_check_detects_tampered_dispatch(asset):
    prices = np.array([20.0] * 12 + [100.0] * 12)
    disp = solve_day(prices, 1.0, asset)
    assert max(dispatch_violations(disp, 1.0, asset).values()) <= FEASIBILITY_TOL
    bad_balance = Dispatch(disp.charge_mw, disp.discharge_mw, disp.soc_mwh + 0.01, 0.0)
    assert dispatch_violations(bad_balance, 1.0, asset)["energy_balance"] > FEASIBILITY_TOL
    over_cap = Dispatch(disp.charge_mw, disp.discharge_mw * 3, disp.soc_mwh, 0.0)
    assert dispatch_violations(over_cap, 1.0, asset)["throughput"] > FEASIBILITY_TOL
