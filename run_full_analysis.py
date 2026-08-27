"""
run_full_analysis.py
====================
Single entry point that runs the entire pipeline end to end from the CSVs in
`data/`. Company-agnostic and robust: layers that need data you don't have
(analyst consensus, clean peer comps) are skipped automatically, so a tiny
uncovered small-cap runs just as cleanly as a mega-cap.

Pipeline
--------
  1. Load inputs (data_loader.load_all) + graceful has_consensus / has_comps flags
  2. Deterministic base-case DCF (Gordon Growth terminal value, dynamic shares)
  3. Reverse solve (implied terminal growth & WACC at the current price)
  4. Consensus-anchored DCF (multi-year fade if FY+1 available)    [if coverage]
  5. Multiples / comps valuation                                    [if peers]
  6. Monte Carlo: own / consensus-weighted / exit-multiple-blend
  7. Enhancements: tornado, 2D sensitivity grid, bear/base/bull scenarios,
     historical own-multiple range, backtest, fair-value-vs-12m-target
  8. Consolidated summary table + master chart, all saved to OUTPUT_DIR

Run:
    python run_full_analysis.py
    python run_full_analysis.py --data-dir data_msft --n-trials 20000
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import ThreeStatementModel
import monte_carlo as mc
import consensus_analysis as ca
import multiples_valuation as mv
import enhancements as enh
from data_loader import load_all
from settings import OUTPUT_DIR, BRAND, BRAND_ALT, POS, NEG, INK, ensure_output_dir, output_path


def run(data_dir="data", n_trials=20_000, make_charts=True):
    ensure_output_dir()
    data = load_all(data_dir)
    base, drv, history = data["base"], data["drivers"], data["history"]
    meta = data["meta"]
    consensus, has_consensus = data["consensus"], data["has_consensus"]
    comps, has_comps = data["comps"], data["has_comps"]
    ticker = meta["ticker"]

    print("\n" + "#" * 72)
    print(f"# FULL VALUATION: {meta['company_name']} ({ticker})  as of {meta['as_of_date']}")
    print(f"# coverage: consensus={'YES' if has_consensus else 'no'} | "
          f"comps={'YES' if has_comps else 'no'} | "
          f"segments={'YES' if data['segments'] else 'no'}")
    print("#" * 72)

    print("\nHistorical financials:")
    print(history[["fiscal_year", "revenue", "revenue_growth", "gross_margin",
                   "ebit_margin", "net_margin"]].round(3).to_string(index=False))

    # --- 1. Base-case DCF ---
    base_model = ThreeStatementModel(base, drv)
    base_model.run()
    base_val = base_model.dcf_value()

    # --- 2. Reverse solve ---
    reverse = {}
    try:
        reverse["implied_terminal_growth"] = base_model.solve_for_terminal_growth(base.current_price, wacc=drv.wacc)
    except ValueError:
        reverse["implied_terminal_growth"] = None
    try:
        reverse["implied_wacc"] = base_model.solve_for_wacc(base.current_price, terminal_growth=drv.terminal_growth)
    except ValueError:
        reverse["implied_wacc"] = None
    try:
        reverse["implied_year1_growth"] = base_model.solve_for_year1_growth(base.current_price)
    except ValueError:
        reverse["implied_year1_growth"] = None

    # --- 3. Consensus-anchored DCF (optional) ---
    cons_val = None
    cons_drv = None
    if has_consensus:
        cons_model, cons_val, cons_drv = ca.run_consensus_case(base, drv, consensus, multiyear=True)

    # --- 4. Multiples valuation (optional) ---
    implied_multiples = None
    svp = None
    if has_comps:
        implied_multiples = mv.implied_prices_from_multiples(base, consensus, comps)
        svp = mv.subject_vs_peers(comps)

    # --- 5. Monte Carlo variants ---
    print(f"\nMonte Carlo A (own assumptions, Gordon Growth)... [{n_trials:,}]")
    trials_a = mc.run_monte_carlo(base, drv, n_trials=n_trials, seed=42)

    trials_b = None
    if has_consensus:
        print(f"Monte Carlo B (consensus-weighted growth)...        [{n_trials:,}]")
        trials_b = ca.run_consensus_weighted_monte_carlo(base, drv, consensus, n_trials=n_trials, seed=7)

    trials_c = None
    if has_comps:
        print(f"Monte Carlo C (exit-multiple blend)...              [{n_trials:,}]")
        trials_c = mv.run_exit_multiple_blend(base, drv, comps, blend_weight=0.5, n_trials=n_trials, seed=11)

    # --- 6. Enhancements ---
    print("\nEnhancements: tornado, 2D grid, scenarios, backtest, historical multiples...")
    tornado = enh.tornado_sensitivity(base, drv)
    grid = enh.sensitivity_grid_2d(base, drv)
    scenarios = enh.scenario_table(base, drv)
    backtest = enh.backtest_calibration(base, drv, history)
    subj_row = comps[comps["is_subject"]].iloc[0] if has_comps else None
    hist_mult = enh.historical_own_multiples(meta, history, data["valuation_history"], subj_row)
    cost_of_equity = drv.wacc  # WACC used as the roll-forward rate (cost of capital)
    street_target = consensus.avg_target if has_consensus else None
    fv_recon = enh.reconcile_fv_to_target(base_val["price_per_share"], cost_of_equity, street_target)

    # --- Consolidated summary ---
    rows = [dict(method="Current market price", price=base.current_price)]
    if has_consensus:
        rows += [dict(method="Street avg analyst target", price=consensus.avg_target),
                 dict(method="Street median analyst target", price=consensus.median_target)]
    rows += [dict(method="Base-case DCF (Gordon Growth)", price=base_val["price_per_share"])]
    if cons_val is not None:
        rows += [dict(method="Consensus-anchored DCF", price=cons_val["price_per_share"])]
    if implied_multiples is not None:
        rows += [dict(method=f"Multiples: {r['method']}", price=r["implied_price"])
                 for _, r in implied_multiples.iterrows()]
    rows += [dict(method="Monte Carlo median — own assumptions", price=trials_a["implied_price"].median())]
    if trials_b is not None:
        rows += [dict(method="Monte Carlo median — consensus-weighted", price=trials_b["implied_price"].median())]
    if trials_c is not None:
        rows += [dict(method="Monte Carlo median — exit-multiple blend", price=trials_c["implied_price"].median())]
    for s in ["Bear", "Base", "Bull"]:
        p = float(scenarios.loc[scenarios["scenario"] == s, "implied_price"].iloc[0])
        rows += [dict(method=f"Scenario: {s}", price=p)]

    summary = pd.DataFrame(rows).sort_values("price", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 72)
    print("CONSOLIDATED VALUATION SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False, formatters={"price": lambda x: f"${x:,.2f}"}))

    pct_below = (trials_a["implied_price"] < base.current_price).mean() * 100
    print(f"\nCurrent price sits at the {pct_below:.0f}th percentile of the own-assumptions MC distribution.")
    if reverse["implied_terminal_growth"] is not None:
        print(f"Reverse-DCF: at WACC {drv.wacc:.1%}, market price implies terminal growth "
              f"{reverse['implied_terminal_growth']:.2%}.")
    if reverse["implied_wacc"] is not None:
        print(f"Reverse-DCF: at terminal growth {drv.terminal_growth:.1%}, market price implies WACC "
              f"{reverse['implied_wacc']:.2%}.")
    print(f"Fair value today ${fv_recon['fair_value_today']:,.0f} -> rolled 12m at {cost_of_equity:.1%} = "
          f"${fv_recon['fair_value_12m']:,.0f}" +
          (f" vs Street target ${street_target:,.0f} (gap {fv_recon['gap_pct']:+.1%})" if street_target else ""))
    if backtest.get("available"):
        print(f"Backtest {backtest['prior_year']}->{backtest['target_year']}: revenue error "
              f"{backtest['revenue_error_pct']:+.1%}, EBIT error {backtest['ebit_error_pct']:+.1%}.")

    charts = {}
    if make_charts:
        charts = _make_all_charts(ticker, meta, base, drv, history, consensus, has_consensus,
                                  comps, has_comps, base_val, cons_val, implied_multiples, svp,
                                  trials_a, trials_b, trials_c, tornado, grid, scenarios,
                                  hist_mult, subj_row, summary)

    # Save summary CSV
    summary.to_csv(output_path(ticker, "valuation_summary", "csv"), index=False)
    print(f"\nSaved: {output_path(ticker, 'valuation_summary', 'csv')}")

    return dict(
        meta=meta, base=base, drv=drv, history=history,
        consensus=consensus, has_consensus=has_consensus,
        comps=comps, has_comps=has_comps, segments=data["segments"],
        base_val=base_val, cons_val=cons_val, cons_drv=cons_drv,
        reverse=reverse, implied_multiples=implied_multiples, svp=svp,
        trials_a=trials_a, trials_b=trials_b, trials_c=trials_c,
        tornado=tornado, grid=grid, scenarios=scenarios, backtest=backtest,
        hist_mult=hist_mult, fv_recon=fv_recon,
        summary=summary, charts=charts, forecast=base_model.results,
        pct_below=pct_below,
    )


def _make_all_charts(ticker, meta, base, drv, history, consensus, has_consensus,
                     comps, has_comps, base_val, cons_val, implied_multiples, svp,
                     trials_a, trials_b, trials_c, tornado, grid, scenarios,
                     hist_mult, subj_row, summary):
    charts = {}

    # Master chart
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = []
    for m in summary["method"]:
        if "market price" in m:
            colors.append(INK)
        elif "Street" in m:
            colors.append("#555555")
        elif "Monte Carlo" in m:
            colors.append("#9467bd")
        elif "Multiples" in m:
            colors.append(BRAND)
        elif "Scenario" in m:
            colors.append(BRAND_ALT)
        else:
            colors.append(NEG)
    y = np.arange(len(summary))
    ax.barh(y, summary["price"], color=colors)
    ax.set_yticks(y); ax.set_yticklabels(summary["method"], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(base.current_price, color=INK, lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("Price ($/share)")
    ax.set_title(f"{ticker}: All Valuation Methods Compared")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    charts["master"] = output_path(ticker, "master_valuation_summary")
    plt.savefig(charts["master"], dpi=150); plt.close()

    # Monte Carlo panel (own assumptions)
    charts["monte_carlo"] = mc.make_plots(trials_a, base.current_price, ticker=ticker)

    # Tornado
    charts["tornado"] = enh.plot_tornado(tornado, base, ticker=ticker)

    # 2D grid
    charts["grid"] = enh.plot_sensitivity_grid(grid, base, ticker=ticker)

    # Scenarios
    charts["scenarios"] = enh.plot_scenarios(scenarios, base, ticker=ticker)

    # Historical own multiples (optional)
    hm = enh.plot_historical_own_multiples(hist_mult, meta, subj_row, ticker=ticker)
    if hm:
        charts["hist_mult"] = hm

    # Consensus layer charts
    if has_consensus and cons_val is not None:
        try:
            base.ticker = ticker  # for title
            charts["bank_targets"] = ca.plot_bank_targets_vs_model(
                consensus, base, base_val["price_per_share"], cons_val["price_per_share"],
                trials_a["implied_price"].median(),
                (trials_b["implied_price"].median() if trials_b is not None else trials_a["implied_price"].median()),
                ticker=ticker)
        except Exception as ex:
            print(f"[bank targets chart skipped: {ex}]")
        if trials_b is not None:
            charts["consensus_hist"] = ca.plot_consensus_weighted_histogram(
                trials_a, trials_b, base, consensus, ticker=ticker)

    # Multiples layer charts
    if has_comps and implied_multiples is not None:
        street_avg = consensus.avg_target if has_consensus else base.current_price
        charts["multiples"] = mv.plot_multiples_comparison(
            comps, implied_multiples, base, base_val["price_per_share"], street_avg, ticker=ticker)
        if trials_c is not None:
            charts["exit_blend"] = mv.plot_exit_multiple_blend_histogram(trials_a, trials_c, base, ticker=ticker)

    for k, v in charts.items():
        if v:
            print(f"Saved chart [{k}]: {v}")
    return charts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-trials", type=int, default=20_000)
    args = parser.parse_args()
    run(data_dir=args.data_dir, n_trials=args.n_trials)
