"""
consensus_analysis.py
======================
Layers Wall Street analyst consensus into the reverse-DCF tool two ways:

1. CONSENSUS-ANCHORED DCF: instead of guessing Year-1 revenue growth
   ourselves, anchor it to the Street's consensus revenue estimate for
   NVIDIA's current/next fiscal year, run it through the same 3-statement
   model, and see what price that implies.

2. CONSENSUS-WEIGHTED MONTE CARLO: re-center the Monte Carlo's Year-1 growth
   distribution on the consensus estimate (instead of our own subjective
   guess), so "what does the Street expect" becomes an input to the
   simulation, not just a side comparison.

It also produces a chart and table comparing individual bank price targets
against: today's market price, our base-case DCF, our consensus-anchored
DCF, and the Monte Carlo median.

NOTE ON METHOD: analyst price targets are usually NOT built from a DCF —
most banks anchor to a forward P/E or EV/Sales multiple applied to their own
estimates. So "our consensus-anchored DCF price" and "the Street's average
target" are two different methodologies that happen to both use consensus-ish
inputs. Presenting them side by side (rather than treating them as the same
number) is intentional — the gap between them is itself informative about
whether DCF and multiple-based approaches agree.

Run:  python consensus_analysis.py
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import BaseYearActuals, ForecastDrivers, ThreeStatementModel
from analyst_data import AnalystConsensus
import monte_carlo as mc

from settings import OUTPUT_DIR, BRAND, BRAND_ALT, POS, NEG, INK, output_path


# ---------------------------------------------------------------------------
# 1. Consensus-anchored deterministic DCF
# ---------------------------------------------------------------------------

def consensus_implied_year1_growth(base: BaseYearActuals, consensus: AnalystConsensus) -> float:
    """Approximate Year-1 growth implied by the Street's forward revenue estimate.

    Caveat: consensus 'FY2027' revenue covers a slightly different set of
    quarters than our TTM base, since NVIDIA's fiscal year doesn't align
    cleanly with 'trailing twelve months as of today'. Treated here as a
    reasonable one-year-forward growth proxy, not an exact match.
    """
    return consensus.fy_next_revenue_estimate / base.revenue - 1


def consensus_implied_year2_growth(consensus: AnalystConsensus):
    """Year-2 growth implied by FY+1 over FY revenue estimates, if both exist."""
    r1 = getattr(consensus, "fy_next_revenue_estimate", None)
    r2 = getattr(consensus, "fy_next2_revenue_estimate", None)
    if r1 and r2:
        return r2 / r1 - 1
    return None


def consensus_multiyear_fade_drivers(base_drivers: ForecastDrivers, base: BaseYearActuals,
                                     consensus: AnalystConsensus) -> ForecastDrivers:
    """Anchor BOTH Year-1 and Year-2 growth to the Street (when FY+1 is
    available), then fade to terminal. Implemented as a 2-segment build so the
    first two years hit the consensus path exactly, after which a single blended
    segment fades from the Year-2 growth down to terminal growth.

    Falls back to the single-year anchor if FY+1 revenue isn't available."""
    import numpy as np
    g1 = consensus_implied_year1_growth(base, consensus)
    g2 = consensus_implied_year2_growth(consensus)
    drv = copy.deepcopy(base_drivers)
    drv.segments = None
    if g2 is None:
        drv.year1_growth = g1
        drv.custom_growth_path = None
        return drv
    # Explicit path: Year-1 = g1 (Street), Year-2 = g2 (Street), then linear
    # fade from g2 down to terminal growth across the remaining years.
    n = drv.n_years
    path = [g1, g2]
    if n > 2:
        path += list(np.linspace(g2, drv.terminal_growth, n - 1))[1:]
    drv.custom_growth_path = path[:n]
    drv.year1_growth = g1
    return drv


def consensus_anchored_drivers(base_drivers: ForecastDrivers, base: BaseYearActuals,
                                consensus: AnalystConsensus) -> ForecastDrivers:
    """Same margin/capex/WACC assumptions as the base case — only Year-1 growth
    is replaced with the Street's consensus growth rate. Margins remain our own
    assumption (see module docstring / README for why we don't force-fit them
    to consensus EPS, which is usually non-GAAP and share-count sensitive)."""
    drv = copy.deepcopy(base_drivers)
    drv.year1_growth = consensus_implied_year1_growth(base, consensus)
    return drv


def run_consensus_case(base: BaseYearActuals, base_drivers: ForecastDrivers,
                        consensus: AnalystConsensus, multiyear: bool = True):
    """Consensus-anchored DCF. If `multiyear` and FY+1 revenue is available,
    anchors BOTH Year-1 and Year-2 to the Street and then fades; otherwise
    anchors Year-1 only (original behavior)."""
    if multiyear:
        drv = consensus_multiyear_fade_drivers(base_drivers, base, consensus)
    else:
        drv = consensus_anchored_drivers(base_drivers, base, consensus)
    model = ThreeStatementModel(base, drv)
    model.run()
    val = model.dcf_value()
    return model, val, drv


# ---------------------------------------------------------------------------
# 2. Consensus-weighted Monte Carlo (re-centered Year-1 growth distribution)
# ---------------------------------------------------------------------------

def consensus_weighted_dist(base: BaseYearActuals, base_drivers: ForecastDrivers,
                            consensus: AnalystConsensus, spread=0.15) -> dict:
    """Monte Carlo distribution auto-centered on the company's own drivers, but
    with Year-1 growth re-centered on the Street consensus growth rate
    (+/- `spread`), reflecting the analyst range rather than a subjective guess."""
    g = consensus_implied_year1_growth(base, consensus)
    dist = mc.default_dist(base_drivers)
    dist["year1_growth"] = dict(kind="triangular",
                                low=max(-0.10, g - spread),
                                mode=g,
                                high=g + spread)
    return dist


def run_consensus_weighted_monte_carlo(base, base_drivers, consensus,
                                        n_trials=20_000, seed=7, spread=0.15):
    dist = consensus_weighted_dist(base, base_drivers, consensus, spread=spread)
    return mc.run_monte_carlo(base, base_drivers, n_trials=n_trials, seed=seed, dist=dist)


# ---------------------------------------------------------------------------
# 3. Bank price-target table + chart
# ---------------------------------------------------------------------------

def bank_target_table(consensus: AnalystConsensus) -> pd.DataFrame:
    df = pd.DataFrame([t.__dict__ for t in consensus.bank_targets])
    df = df.sort_values("target", ascending=False).reset_index(drop=True)
    return df


def plot_bank_targets_vs_model(consensus: AnalystConsensus, base: BaseYearActuals,
                                dcf_base_price: float, dcf_consensus_price: float,
                                mc_median: float, mc_consensus_median: float,
                                outdir=OUTPUT_DIR, ticker="subject"):
    df = bank_target_table(consensus)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y_pos = np.arange(len(df))
    colors = ["#1f77b4" if "2026-08" in d or "2026-07" in d else "#a6c8e8" for d in df["date"]]
    ax.barh(y_pos, df["target"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{b}  ({a})" for b, a in zip(df["bank"], df["analyst"])], fontsize=9)
    ax.invert_yaxis()

    lines = [
        (base.current_price, "black", "-", f"Current price (${base.current_price:,.0f})"),
        (consensus.avg_target, "#333333", ":", f"Street avg target (${consensus.avg_target:,.0f}, "
                                                 f"{consensus.num_analysts} analysts)"),
        (dcf_base_price, "#d62728", "--", f"Our base-case DCF (${dcf_base_price:,.0f})"),
        (dcf_consensus_price, "#ff7f0e", "--", f"Our consensus-anchored DCF (${dcf_consensus_price:,.0f})"),
        (mc_median, "#9467bd", "-.", f"Monte Carlo median, own assumptions (${mc_median:,.0f})"),
        (mc_consensus_median, "#2ca02c", "-.", f"Monte Carlo median, consensus-weighted (${mc_consensus_median:,.0f})"),
    ]
    for x, color, style, label in lines:
        ax.axvline(x, color=color, linestyle=style, linewidth=1.6, label=label)

    ax.set_xlabel("Price target / implied value ($/share)")
    ax.set_title(f"{getattr(base,'ticker','')} Bank Price Targets vs. Model-Implied Values (as of {consensus.as_of})".strip())
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = output_path(ticker, "analyst_targets_vs_model")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_consensus_weighted_histogram(trials_own: pd.DataFrame, trials_consensus: pd.DataFrame,
                                       base: BaseYearActuals, consensus: AnalystConsensus,
                                       outdir=OUTPUT_DIR, ticker="subject"):
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(0, max(trials_own["implied_price"].quantile(0.995),
                               trials_consensus["implied_price"].quantile(0.995)), 90)
    ax.hist(trials_own["implied_price"], bins=bins, alpha=0.55, color=BRAND,
            label="Monte Carlo — own growth assumptions")
    ax.hist(trials_consensus["implied_price"], bins=bins, alpha=0.55, color="#1f77b4",
            label="Monte Carlo — consensus-weighted growth")
    ax.axvline(base.current_price, color="black", linewidth=2, linestyle="--",
               label=f"Current price (${base.current_price:,.0f})")
    ax.axvline(consensus.avg_target, color="#d62728", linewidth=2, linestyle=":",
               label=f"Street avg target (${consensus.avg_target:,.0f})")
    ax.set_title("Simulated Implied Price: Own Assumptions vs. Consensus-Weighted Growth")
    ax.set_xlabel("Implied price ($/share)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = output_path(ticker, "consensus_vs_own_montecarlo")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(data_dir="data"):
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    from data_loader import load_all
    data = load_all(data_dir)
    base, base_drivers, consensus = data["base"], data["drivers"], data["consensus"]
    ticker = data["meta"]["ticker"]
    if consensus is None:
        print("No analyst consensus available for this company — skipping consensus layer.")
        return

    # --- Base case (own assumptions) ---
    base_model = ThreeStatementModel(base, base_drivers)
    base_model.run()
    base_val = base_model.dcf_value()

    # --- Consensus-anchored case ---
    cons_model, cons_val, cons_drv = run_consensus_case(base, base_drivers, consensus)
    implied_g = consensus_implied_year1_growth(base, consensus)

    print("\n" + "=" * 70)
    print("ANALYST CONSENSUS INPUTS")
    print("=" * 70)
    print(f"As of:                           {consensus.as_of}")
    print(f"Analysts covering {ticker}:          {consensus.num_analysts}")
    print(f"Consensus rating:                {consensus.rating}")
    print(f"Average 12-month price target:   ${consensus.avg_target:,.2f}")
    print(f"Median 12-month price target:    ${consensus.median_target:,.2f}")
    print(f"Range:                           ${consensus.low_target:,.0f} - ${consensus.high_target:,.0f}")
    print(f"{consensus.fy_next_label} revenue estimate: ${consensus.fy_next_revenue_estimate:,.0f}mm")
    print(f"{consensus.fy_next_label} EPS estimate (non-GAAP): ${consensus.fy_next_eps_estimate:,.2f}")
    print(f"Implied Year-1 growth vs. our TTM base: {implied_g:.1%} "
          f"(vs. {base_drivers.year1_growth:.1%} in our own base case)")

    print("\n" + "=" * 70)
    print("VALUATION COMPARISON — DETERMINISTIC CASES")
    print("=" * 70)
    print(f"{'Case':40s}{'Year-1 growth':>16s}{'Implied price':>16s}")
    print(f"{'Our base case (subjective)':40s}{base_drivers.year1_growth:>15.1%} "
          f"${base_val['price_per_share']:>14,.2f}")
    print(f"{'Consensus-anchored (Street growth)':40s}{cons_drv.year1_growth:>15.1%} "
          f"${cons_val['price_per_share']:>14,.2f}")
    print(f"{'Current market price':40s}{'':>16s}${base.current_price:>14,.2f}")
    print(f"{'Street avg analyst price target':40s}{'':>16s}${consensus.avg_target:>14,.2f}")
    print("(Note: the Street's price targets are typically built off forward P/E or "
          "EV/Sales multiples, not a DCF — so this is a cross-methodology comparison, "
          "not an apples-to-apples check.)")

    # --- Monte Carlo: own assumptions vs. consensus-weighted ---
    print("\nRunning Monte Carlo (own assumptions)...")
    trials_own = mc.run_monte_carlo(base, base_drivers, n_trials=20_000, seed=42)
    mc.summarize(trials_own, base.current_price,
                 label="MONTE CARLO — OWN GROWTH ASSUMPTIONS")

    print("\nRunning Monte Carlo (consensus-weighted growth)...")
    trials_cons = run_consensus_weighted_monte_carlo(base, base_drivers, consensus,
                                                       n_trials=20_000, seed=7)
    mc.summarize(trials_cons, base.current_price,
                 label="MONTE CARLO — CONSENSUS-WEIGHTED GROWTH")
    mc.summarize(trials_cons, consensus.avg_target,
                 label="MONTE CARLO (consensus-weighted) vs. STREET AVG TARGET",
                 price_label="Street avg analyst target")

    # --- Bank targets table ---
    df_targets = bank_target_table(consensus)
    print("\n" + "=" * 70)
    print("INDIVIDUAL BANK / ANALYST PRICE TARGETS")
    print("=" * 70)
    print(df_targets.to_string(index=False))

    # --- Charts ---
    chart1 = plot_bank_targets_vs_model(
        consensus, base,
        dcf_base_price=base_val["price_per_share"],
        dcf_consensus_price=cons_val["price_per_share"],
        mc_median=trials_own["implied_price"].median(),
        mc_consensus_median=trials_cons["implied_price"].median(), ticker=ticker,
    )
    print(f"\nSaved chart: {chart1}")

    chart2 = plot_consensus_weighted_histogram(trials_own, trials_cons, base, consensus, ticker=ticker)
    print(f"Saved chart: {chart2}")

    # --- Save data ---
    df_targets.to_csv(output_path(ticker, "bank_price_targets", "csv"), index=False)
    trials_cons.to_csv(output_path(ticker, "consensus_weighted_trials", "csv"), index=False)
    print(f"Saved: {output_path(ticker, 'bank_price_targets', 'csv')}")
    print(f"Saved: {output_path(ticker, 'consensus_weighted_trials', 'csv')}")


if __name__ == "__main__":
    main()
