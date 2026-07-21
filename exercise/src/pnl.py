"""PnL + performance metrics. YOU implement these.

The single source of truth for money is `pnl_eur`. Every strategy, no matter
how it decides to dispatch, is scored by the same accounting so comparisons
are fair.
"""
from typing import Sequence

import numpy as np


def pnl_eur(prices: Sequence[float], net_export_mw: Sequence[float], dt_h: float) -> float:
    """Cash earned over the horizon, in EUR.

    Sign convention: net_export_mw[t] > 0 means selling to the market
    (discharge / generation), < 0 means buying (charging). We get paid the
    price for what we sell and pay the price for what we buy, so

        pnl = sum_t  price[t] * net_export_mw[t] * dt_h

    dt_h is the length of one step in hours (1.0 hourly, 0.25 for 15-min).

    Worked example (see tests/test_pnl.py):
        prices      = [10, 50, 20, 40]
        net_export  = [-1,  0, -1,  1]   # charge, idle, charge, discharge
        dt_h        = 0.25
        pnl = 0.25 * (10*-1 + 50*0 + 20*-1 + 40*1) = 2.5
    """
    raise NotImplementedError


def equivalent_full_cycles(discharge_mwh: Sequence[float], energy_mwh: float) -> float:
    """Throughput expressed in whole battery capacities discharged.

        cycles = sum(discharge_mwh) / energy_mwh

    A useful proxy for battery wear when comparing strategies.
    """
    raise NotImplementedError


def summarise(name: str, prices, net_export_mw, dt_h, energy_mwh=None) -> dict:
    """Return a small dict of headline metrics for one strategy.

    Suggested keys: name, pnl_eur, mwh_bought, mwh_sold, cycles (if energy_mwh
    given). Used to build the comparison table in Part B.
    """
    raise NotImplementedError
