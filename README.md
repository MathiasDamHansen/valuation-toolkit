# Reverse-DCF + Analyst Consensus + Multiples Toolkit

A standardized, repeatable **equity valuation pipeline for any listed company** —
mega-cap or micro-cap, heavily covered or not covered at all. It combines three
methodologies usually done separately (DCF, analyst consensus, trading comps)
inside one Monte Carlo framework, and regenerating the entire input dataset for
a new company is **one prompt to an AI research agent**, not a manual chore.

Ships pre-loaded with **NVIDIA (NVDA)** as a fully-covered worked example and
**TINYCO** as a no-coverage small-cap example, so you can see graceful
degradation in action.

## The big picture

```
AGENT_DATA_COLLECTION_PROMPT.md  →  data/*.csv  →  run_full_analysis.py  →  charts + CSVs
   (hand to any AI agent               (standard         build_dashboard.py →  one-page HTML/PDF
    with web search)                    5-file schema)    app.py             →  live Streamlit app
```

You never touch the Python for a new company — just regenerate the CSVs and re-run.

## What's inside (11 valuation lenses + risk views)

**Base engine**
- Linked **3-statement model** (IS/BS/CF) with day-based working capital.
- **DCF** with a terminal value that can be pure Gordon Growth, a peer **exit
  EV/EBITDA multiple**, or any blend of the two.
- **Reverse-DCF**: solve the terminal growth / WACC / Year-1 growth the current
  price implies.
- **Dynamic share count** — SBC dilution and buybacks compound over the horizon
  (per-share value uses the terminal share count, not a flat base-year count).
- Optional **sum-of-the-parts** revenue build from `segments.csv`.

**Consensus layer (auto-skips if no coverage)**
- Consensus-anchored DCF with a **multi-year fade**: anchors Year-1 *and* Year-2
  to the Street when FY+1 estimates exist, then fades — which tames the
  well-known "one year of hyper-growth compounded for a decade" overshoot.
- Consensus-weighted Monte Carlo and a bank-price-target comparison.

**Multiples layer (auto-skips if no clean peers)**
- Peer median P/E, EV/EBITDA, EV/Revenue → implied price; subject vs peer
  premium/discount; exit-multiple-blended Monte Carlo. Falls back to the model's
  own forward forecast when there's no analyst NTM estimate.

**Risk & presentation add-ons**
- **Tornado** chart ranking each driver's price impact.
- Deterministic **WACC × terminal-growth sensitivity grid**.
- **Bear / Base / Bull** scenario table.
- The company's **own historical multiple range** (if `valuation_history.csv`).
- **Backtest**: feed a prior year in, check the engine reproduces actuals.
- **Fair-value-today → 12-month-target** reconciliation (rolls FV forward one
  year at the cost of capital so it's comparable to a Street target).
- **One-page HTML/PDF dashboard** and an interactive **Streamlit app**.

## Quick start

```bash
pip install -r requirements.txt

python run_full_analysis.py                 # full run on the bundled NVDA data
python build_dashboard.py                   # one-page HTML (+PDF) dashboard
streamlit run app.py                        # interactive live dashboard
```

Everything reads `data/*.csv` and writes to the folder in `settings.OUTPUT_DIR`
(override with the `VALUATION_OUTPUT_DIR` env var).

## Files

| File | Role |
|---|---|
| `AGENT_DATA_COLLECTION_PROMPT.md` | The one prompt to (re)generate the CSVs for any ticker. **Start here for a new company.** |
| `data/` | The 5 required CSVs (+ optional `segments.csv`, `valuation_history.csv`). |
| `data_smallcap/` | A no-coverage micro-cap example (only 2 CSVs) to demo graceful degradation. |
| `settings.py` | Output dir + neutral (company-agnostic) chart palette. |
| `data_loader.py` | Parses CSVs → objects; sets `has_consensus` / `has_comps` flags. |
| `model.py` | 3-statement model, DCF, reverse-solve, share dynamics, segments. |
| `monte_carlo.py` | Generic Monte Carlo engine (auto-centers ranges on the company). |
| `analyst_data.py` | `AnalystConsensus` / `BankTarget` dataclasses. |
| `consensus_analysis.py` | Consensus-anchored & multi-year-fade DCF + weighted MC. |
| `multiples_valuation.py` | Peer comps, implied price, exit-multiple-blended MC. |
| `enhancements.py` | Tornado, 2D grid, scenarios, historical multiples, backtest, FV reconciliation. |
| `run_full_analysis.py` | **The one script to run.** Orchestrates everything, robustly. |
| `build_dashboard.py` | Assembles the self-contained one-page HTML/PDF dashboard. |
| `app.py` | Streamlit interactive dashboard (sliders + CSV upload). |
| `valuation_toolkit.ipynb` | Jupyter/Colab notebook version. |

## Running on a different company

1. Copy the prompt from `AGENT_DATA_COLLECTION_PROMPT.md`, swap in a ticker,
   hand it to any AI agent with web search.
2. Save the returned CSVs into a folder (e.g. `data_msft/`).
3. `python run_full_analysis.py --data-dir data_msft` (or point the Streamlit
   app / notebook at it). No code changes needed.

**Small / uncovered names:** you only strictly need `company_inputs.csv` and
`financial_history.csv`. Without `analyst_consensus.csv` the consensus layers are
skipped; without a usable `multiples_comps.csv` the comps layers are skipped. The
DCF, reverse-solve, Monte Carlo, tornado, sensitivity grid, scenarios and
backtest still run.

## Deployment (GitHub / Streamlit / Colab / Lovable)

See **`DEPLOYMENT.md`** for copy-paste steps. Short version:

- **Live web dashboard (recommended, easiest):** push to GitHub → deploy `app.py`
  free on Streamlit Community Cloud → you get a public URL. This is the
  Python-native equivalent of a "Lovable site" but can actually run the model.
- **Colab/Jupyter:** open `valuation_toolkit.ipynb`.
- **Lovable:** possible, but Lovable builds a React front-end and can't execute
  this Python — you'd host the model as a small API and have Lovable call it.
  `DEPLOYMENT.md` explains that pattern if you want it.

## A key finding worth flagging

With the **single-year** consensus anchor, running the Street's ~54% Year-1
growth through a DCF and fading over 10 years implies a very high price (~$430
for NVDA). With the **multi-year** anchor — pinning Year-2 to the Street's own
~20% estimate too — the same DCF drops to ~$175. Neither is "right"; the gap is
the point: DCF outputs are dominated by how fast you assume growth decelerates,
which is exactly why multiples-based bank targets (which bake in deceleration
implicitly) and a naive DCF can diverge so far.

## Caveats

The bundled `data/` CSVs are approximate, illustrative figures as of ~August
2026 and a snapshot, not a live feed. **Regenerate via the agent prompt before
relying on any number.** This is an educational tool, not investment advice.
