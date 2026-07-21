"""Generate synthetic-but-realistic market data for the two exercises.

Run:  python scripts/generate_data.py
Writes: data/day_ahead.csv  and  data/intraday.csv

The numbers are made up but shaped to look like a Central-European power
market: a daily price profile with a morning ramp, a midday solar dip
(occasionally negative), and a sharp evening peak. Wind is a smooth random
walk; solar is a deterministic bell curve scaled by daily cloudiness.
Nothing here is fitted to real data -- it only needs to be *plausible* so the
optimisation and strategy code has something interesting to chew on.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# --- horizons -------------------------------------------------------------
DA_START = "2026-07-13"          # Monday
DA_DAYS = 7                       # one week, hourly
ID_DATE = "2026-07-15"           # the intraday exercise zooms into this day

# --- asset ratings (also documented in TASK.md) ---------------------------
WIND_MW = 100.0                   # wind farm nameplate
SOLAR_MW = 50.0                   # solar farm nameplate


# 24h base price shape (EUR/MWh), typical summer weekday
BASE_SHAPE = np.array([
    58, 54, 51, 50, 51, 56,       # 00-05 night
    68, 88, 98, 84, 66, 50,       # 06-11 morning ramp then solar pushes it down
    42, 38, 41, 55, 74, 96,       # 12-17 midday dip -> late afternoon climb
    138, 150, 118, 92, 76, 64,    # 18-23 evening peak, then wind-down
], dtype=float)

SOLAR_SHAPE = np.clip(              # normalised 0..1 bell, zero at night
    np.sin((np.arange(24) - 6) / 12 * np.pi), 0, None
)


def day_ahead():
    idx = pd.date_range(DA_START, periods=DA_DAYS * 24, freq="H")
    hour = idx.hour.to_numpy()
    day = (idx - idx[0]).days.to_numpy()

    # price: base shape + per-day level shift + AR(1)-ish noise
    level = RNG.normal(0, 8, size=DA_DAYS)[day]
    noise = RNG.normal(0, 6, size=len(idx))
    price = BASE_SHAPE[hour] + level + noise

    # occasionally drive the solar hours negative (renewable glut)
    glut = (RNG.random(DA_DAYS) < 0.35)[day] & (SOLAR_SHAPE[hour] > 0.7)
    price = np.where(glut, price - 55, price)
    price = price.round(2)

    # wind: smooth random walk in capacity-factor space, clipped to [.03,.92]
    steps = RNG.normal(0, 0.08, size=len(idx))
    cf = np.clip(0.4 + np.cumsum(steps) % 1.0, 0.03, 0.92)
    cf = 0.03 + (cf - cf.min()) / (cf.max() - cf.min()) * (0.92 - 0.03)
    wind = (cf * WIND_MW).round(2)

    # solar: bell curve * daily cloudiness in [.55,1.0]
    cloud = RNG.uniform(0.55, 1.0, size=DA_DAYS)[day]
    solar = (SOLAR_SHAPE[hour] * cloud * SOLAR_MW).round(2)

    return pd.DataFrame({
        "timestamp": idx.strftime("%Y-%m-%d %H:%M"),
        "da_price_eur_mwh": price,
        "wind_avail_mw": wind,
        "solar_avail_mw": solar,
    })


def intraday(da_df):
    # 15-min grid for the chosen day
    idx = pd.date_range(ID_DATE, periods=96, freq="15min")

    # broadcast that day's hourly DA price down to quarter-hours
    day_da = da_df.set_index(pd.to_datetime(da_df["timestamp"]))
    day_da = day_da.loc[ID_DATE, "da_price_eur_mwh"].to_numpy()
    da_q = np.repeat(day_da, 4)

    # intraday = DA + mean-reverting deviation (higher vol) + rare spikes
    dev = np.zeros(96)
    for t in range(1, 96):
        dev[t] = 0.75 * dev[t - 1] + RNG.normal(0, 9)
    spike_at = RNG.choice(96, size=5, replace=False)
    dev[spike_at] += RNG.normal(0, 1, size=5).round() * 0 + RNG.choice(
        [-120, -80, 90, 130, 160], size=5
    )

    id_price = (da_q + dev).round(2)

    return pd.DataFrame({
        "timestamp": idx.strftime("%Y-%m-%d %H:%M"),
        "da_price_eur_mwh": da_q.round(2),
        "id_price_eur_mwh": id_price,
    })


def main():
    da = day_ahead()
    da.to_csv("data/day_ahead.csv", index=False)
    idf = intraday(da)
    idf.to_csv("data/intraday.csv", index=False)

    print("wrote data/day_ahead.csv", da.shape,
          "price range", da.da_price_eur_mwh.min(), "->", da.da_price_eur_mwh.max())
    print("wrote data/intraday.csv", idf.shape,
          "id range", idf.id_price_eur_mwh.min(), "->", idf.id_price_eur_mwh.max())


if __name__ == "__main__":
    main()
