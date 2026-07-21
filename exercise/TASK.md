# Trading Analyst — battery & hybrid optimisation exercise

Two modelling problems plus a shared PnL/back-testing harness. Aim to have
Part B (the three-strategy comparison) fully working and tested; Part A is the
"can you formulate an LP" piece. Budget roughly: 30 min data exploration,
90 min Part A, 90 min Part B, 30 min write-up.

## Data

`data/day_ahead.csv` — one week, **hourly** (168 rows)
| column | unit | meaning |
|---|---|---|
| `timestamp` | — | hour start |
| `da_price_eur_mwh` | EUR/MWh | day-ahead clearing price (occasionally negative) |
| `wind_avail_mw` | MW | wind power *available* this hour (100 MW farm) |
| `solar_avail_mw` | MW | solar power *available* this hour (50 MW farm) |

`data/intraday.csv` — one day, **15-minute** (96 rows)
| column | unit | meaning |
|---|---|---|
| `timestamp` | — | quarter-hour start |
| `da_price_eur_mwh` | EUR/MWh | day-ahead price for the hour (reference) |
| `id_price_eur_mwh` | EUR/MWh | intraday price, more volatile, with spikes |

Assets (`src/data.py`): intraday battery 10 MW / 20 MWh; hybrid battery
20 MW / 40 MWh; shared grid connection 120 MW; round-trip efficiency
`eta_charge * eta_discharge` (0.95 × 0.95 ≈ 0.90). All batteries start empty.

**Conventions used everywhere**
- `net_export_mw > 0` = selling (discharge / generation), `< 0` = buying (charging).
- `dt_h` = step length in hours: **1.0** day-ahead, **0.25** intraday.
- SoC update: `soc[t] = soc[t-1] + eta_charge*charge*dt - discharge*dt/eta_discharge`,
  bounded to `[0, energy_mwh]`.
- Cash depends only on what crosses the meter: `pnl = Σ price[t]·net_export[t]·dt_h`.

---

## Part A — Day-ahead co-optimisation (wind + solar + battery)

One grid connection shared by wind, solar and the battery, capped at 120 MW.
The battery may charge from the renewables or from the grid, and wind/solar may
be curtailed. Prices are known (day-ahead), so this is a deterministic optimisation.

**Do:**
1. Formulate the LP that maximises day-ahead revenue over the 168 hours.
   Decision variables per hour: used wind, used solar, battery charge, battery
   discharge, net grid export. Constraints: availability, battery power/energy/
   efficiency, `|export| ≤ 120 MW`. Objective: `Σ price·export·dt`. Implement in
   `src/optimize.py::cooptimize`.
2. Compare its PnL against a **no-battery baseline** (sell all available
   renewables, curtailing only to respect the 120 MW cap). Report the uplift the
   battery adds, plus battery cycles and any curtailed energy.

**Talking points to be ready for:** why the LP is linear (no if/else, efficiency
handled with separate charge/discharge vars); when negative prices make
curtailment optimal; whether you'd ever charge from the grid.

---

## Part B — Intraday battery, three strategies + PnL (main deliverable)

The 10 MW / 20 MWh battery trades the 15-minute intraday price. Implement **at
least three** dispatch strategies in `src/strategies.py`, score them all with the
**same** `src/pnl.py::pnl_eur`, and produce a comparison table.

Suggested set (a benchmark, a rule, a signal):
1. `perfect_foresight` — LP / greedy with full knowledge of the path. Upper bound.
2. `threshold` — charge below the p-th price percentile, discharge above the (100−p)-th.
3. `da_spread` — buy when intraday trades a band below the day-ahead reference, sell above.

**Do:**
1. Implement the three strategies. Each returns a feasible `net_export_mw` array
   (respect power and SoC limits — write one `is_feasible()` helper and reuse it).
2. Implement `pnl_eur`, `equivalent_full_cycles`, `summarise` in `src/pnl.py`.
3. Build a table: strategy → PnL (EUR), MWh bought/sold, cycles, and % of the
   perfect-foresight benchmark captured.
4. Make `pytest -q` green (see `tests/test_pnl.py`), then add one test asserting
   each strategy's output is feasible.

**Stretch (only if time):** a rolling-horizon strategy that re-optimises each
step on a naive forecast (e.g. price persists) and acts only on the first step —
this is the honest "no look-ahead" version, and the gap to `perfect_foresight`
is the value of a better forecast. Add a bid/ask spread and see which strategies
survive it.

---

## Deliverables
- Working `optimize.py`, `strategies.py`, `pnl.py`.
- A short script or notebook printing Part A uplift and the Part B table.
- Green tests + your own feasibility test.
- 5 lines of commentary: which strategy wins, why, and what you'd add with more time.
