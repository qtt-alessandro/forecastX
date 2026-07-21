# Interview prep — battery / hybrid optimisation

Practice kit for a trading-analyst technical. The **task** is in [TASK.md](TASK.md);
the data is generated; the modelling code is left for you to write.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py     # (re)creates data/*.csv, seeded & reproducible
pytest -q                            # currently RED — that's the starting point
```

## Layout
```
data/            day_ahead.csv (hourly week), intraday.csv (15-min day)  [generated]
scripts/         generate_data.py — how the data is built (good to be able to explain)
src/data.py      loaders + asset constants                               [done]
src/optimize.py  Part A: wind+solar+battery co-optimisation LP           [TODO]
src/strategies.py Part B: 3 intraday battery strategies                  [TODO]
src/pnl.py       shared PnL + metrics                                    [TODO]
tests/test_pnl.py example tests = the PnL spec                           [make green]
```

## How to work through it
1. Skim `data/*.csv` and `scripts/generate_data.py` so you can explain the market shape.
2. Do `pnl_eur` first — everything is scored through it — and turn the tests green.
3. Part B strategies, then the comparison table.
4. Part A LP last (biggest single chunk).

Everything in `src/*.py` that raises `NotImplementedError` is yours to fill in.
