"""Day-ahead co-optimisation of a wind + solar + battery hybrid (Part A).
YOU implement.

One grid connection shared by all three assets, capped at GRID_LIMIT_MW. The
battery may charge from the renewables and/or the grid. Curtailment of wind/
solar is allowed (and sometimes optimal when prices go negative).

Formulate and solve the LP that maximises day-ahead revenue over the horizon,
then return the resulting schedule so pnl.py can score it. A good decomposition
of decision variables per step t:
    p_wind[t], p_solar[t]        used renewable power (<= availability, >= 0)
    charge[t], discharge[t]      battery (>= 0, <= power_mw)
    grid_export[t]               net power over the meter (can be < 0 if importing)
with  grid_export[t] = p_wind[t] + p_solar[t] + discharge[t] - charge[t]
      |grid_export[t]| <= GRID_LIMIT_MW
      SoC dynamics + bounds as in data.Battery.
"""
#%%
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from .data import Battery


#%%

da_prices_df = pd.read_csv("../data/day_ahead.csv")

#%%

da_prices = da_prices_df["da_price_eur_mwh"].to_numpy()
wind_prod_avail = da_prices_df["wind_avail_mw"].to_numpy()
solar_prod_avail = da_prices_df["solar_avail_mw"].to_numpy()

eta_ch = 0.95
eta_dis = 0.90
dt = 1.0

fh = len(da_prices)

batt_power_mw = 20.0
batt_capacity_mwh = 100.0
grid_connection_capacity_mw = 120.0
initial_energy_mwh = 0.0

p_charge = cp.Variable(fh, nonneg=True)
p_discharge = cp.Variable(fh, nonneg=True)
p_wind = cp.Variable(fh, nonneg=True)
p_solar = cp.Variable(fh, nonneg=True)
grid_export = cp.Variable(fh)

energy = cp.Variable(fh + 1)

is_charging = cp.Variable(fh, boolean=True)

constraints = [
    energy[0] == initial_energy_mwh,
    energy >= 0,
    energy <= batt_capacity_mwh,

    p_wind <= wind_prod_avail,
    p_solar <= solar_prod_avail,

    p_charge <= batt_power_mw,
    p_discharge <= batt_power_mw,

    grid_export <= grid_connection_capacity_mw,
    grid_export >= -grid_connection_capacity_mw,

    # Prevent simultaneous charging and discharging.
    p_charge <= batt_power_mw * is_charging,
    p_discharge <= batt_power_mw * (1 - is_charging),
]

for t in range(fh):
    constraints += [
        energy[t + 1] == energy[t]
        + eta_ch * p_charge[t] * dt
        - p_discharge[t] * dt / eta_dis,

        grid_export[t]
        == p_wind[t]
        + p_solar[t]
        + p_discharge[t]
        - p_charge[t],
    ]

# constraints += [energy[-1] == energy[0]]

revenue = cp.sum(cp.multiply(da_prices, grid_export)) * dt
problem = cp.Problem(cp.Maximize(revenue), constraints)

problem.solve(solver=cp.HIGHS)

if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
    print("Optimal value:", problem.value)
    print("Charge:", p_charge.value)
    print("Discharge:", p_discharge.value)
    print("Grid export:", grid_export.value)
    print("Energy:", energy.value)
else:
    print("Solver status:", problem.status)



# %%
import matplotlib.pyplot as plt
import pandas as pd

time = pd.to_datetime(da_prices_df["timestamp"])  # change column name if needed
soc_time = list(time) + [time.iloc[-1] + pd.Timedelta(hours=dt)]

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax[0].step(time, da_prices, where="post")
ax[0].set_ylabel("€/MWh")
ax[0].grid()

ax[1].bar(time, p_charge.value, width=0.02,  label="Charge")
ax[1].bar(time, -p_discharge.value, width=0.02,  label="Discharge")
ax[1].set_ylabel("MW")
ax[1].legend()
ax[1].grid()

soc_ax = ax[1].twinx()
soc_ax.step(soc_time, 100 * energy.value / batt_capacity_mwh, "k")
soc_ax.set_ylabel("SoC (%)")
soc_ax.set_ylim(0, 105)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# %%
