"""Battery dispatch strategies for the INTRADAY exercise (Part B). YOU implement.

Every strategy has the same shape:

    strategy(prices, battery, dt_h) -> net_export_mw   # np.ndarray, len == len(prices)

net_export_mw[t] > 0 sells (discharge), < 0 buys (charge). Whatever you return
MUST be feasible: |net_export| <= power_mw and the implied SoC must stay in
[0, energy_mwh] at every step given the efficiency in `battery`. Keeping a
single `is_feasible(...)` helper and asserting it in your tests will save you.

Implement at least three. Suggested set (mix a benchmark, a rule, a signal):
  1. perfect_foresight   -- LP / greedy with full knowledge of the path (upper bound)
  2. threshold           -- charge below the p-th percentile, discharge above (100-p)-th
  3. da_spread           -- trade the intraday price vs the day-ahead reference
"""
#%%
from typing import Optional

import numpy as np
import pandas as pd 
from data import Battery, INTRADAY_BATTERY
#%%

intraday_df = pd.read_csv('../data/intraday.csv', parse_dates=['timestamp'])
idc_prices = intraday_df['id_price_eur_mwh'].to_numpy()
da_prices = intraday_df['da_price_eur_mwh'].to_numpy()
battery = INTRADAY_BATTERY
dt_h = 0.25                       # 15-min intraday steps
#%%

def perfect_foresight(prices, battery: Battery, dt_h: float) -> np.ndarray:
    """Upper bound: knowing the whole price path, maximise arbitrage revenue
    subject to power, energy and efficiency limits. LP (pulp / scipy.linprog)
    or a clever greedy both work."""

    import cvxpy as cp

    
    fh = len(idc_prices)

    p_ch = cp.Variable(fh, nonneg=True)
    p_dis = cp.Variable(fh, nonneg=True)
    energy = cp.Variable(fh + 1, nonneg=True) 

    constraints = [
        energy[0] == battery.soc0_mwh,
        energy[1:] == energy[:-1]
                    + battery.eta_charge * p_ch * dt_h
                    - p_dis / battery.eta_discharge * dt_h,
        p_ch <= battery.power_mw,
        p_dis <= battery.power_mw,
        energy <= battery.energy_mwh,       
        
    ]

    revenue = cp.sum(cp.multiply(idc_prices, p_dis - p_ch)) * dt_h
    problem = cp.Problem(cp.Maximize(revenue), constraints)
    problem.solve(solver=cp.CLARABEL)

    if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        print("Optimal value:", problem.value)
        print("Optimal p_ch:", p_ch.value)
        print("Optimal p_dis:", p_dis.value)
        print("Optimal energy:", energy.value)

    return {'optimal_p_ch': p_ch.value, 'optimal_p_dis': p_dis.value, 'optimal_energy': energy.value, 'revenue': problem.value}


opt_results = perfect_foresight(idc_prices, battery, dt_h)
print(opt_results)
#%%

def threshold(prices, battery: Battery, dt_h: float, low_pct: float = 25.0,
              high_pct: float = 75.0) -> np.ndarray:
    """Charge at full power when price <= low percentile, discharge at full
    power when price >= high percentile, else idle -- clipped to stay feasible."""
    
    low_pct_prices = np.percentile(prices, low_pct)
    high_pct_prices = np.percentile(prices, high_pct)
    batt_power = np.zeros(len(prices))
    energy = np.zeros(len(prices) + 1)

    for t in range(len(prices)):

        if prices[t] <= low_pct_prices:
            batt_power[t] = -battery.power_mw  # Charge at full power
        elif prices[t] >= high_pct_prices:
            batt_power[t] = battery.power_mw  # Discharge at full power
        else:
            batt_power[t] = 0.0  # Idle

        energy[t + 1] = energy[t] + (batt_power[t] * dt_h * (battery.eta_charge if batt_power[t] < 0 else 1 / battery.eta_discharge))

    return energy, batt_power

threshold_results = threshold(idc_prices, battery, dt_h)

#%%
    


def da_spread(prices, da_reference, battery: Battery, dt_h: float,
              band_eur: float = 10.0) -> np.ndarray:
    """Buy when intraday trades a band below the day-ahead reference, sell when
    it trades a band above -- a simple convergence trade."""
    raise NotImplementedError
