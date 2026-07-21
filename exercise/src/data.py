"""Data loading + shared constants. This part is done for you -- the modelling
lives in optimize.py / strategies.py / pnl.py.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class Battery:
    """A generic storage asset. Powers in MW, energy in MWh, prices in EUR/MWh.

    Efficiency convention (round-trip = eta_charge * eta_discharge):
        soc[t] = soc[t-1] + eta_charge * charge_mw[t] * dt
                          - discharge_mw[t] * dt / eta_discharge
    Cashflow only ever depends on what crosses the grid meter (see pnl.py),
    NOT on the efficiency -- efficiency only shrinks the feasible schedule.
    """
    power_mw: float
    energy_mwh: float
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc0_mwh: float = 0.0            # state of charge at t=0


# Reference assets used by the two exercises (feel free to tweak).
INTRADAY_BATTERY = Battery(power_mw=10.0, energy_mwh=20.0)      # 2h battery
DAYAHEAD_BATTERY = Battery(power_mw=20.0, energy_mwh=40.0)      # 2h battery
GRID_LIMIT_MW = 120.0                                          # hybrid export cap


def load_day_ahead() -> pd.DataFrame:
    """Hourly week. Columns: timestamp, da_price_eur_mwh, wind_avail_mw, solar_avail_mw."""
    df = pd.read_csv(DATA_DIR / "day_ahead.csv", parse_dates=["timestamp"])
    return df


def load_intraday() -> pd.DataFrame:
    """15-min single day. Columns: timestamp, da_price_eur_mwh, id_price_eur_mwh."""
    df = pd.read_csv(DATA_DIR / "intraday.csv", parse_dates=["timestamp"])
    return df
