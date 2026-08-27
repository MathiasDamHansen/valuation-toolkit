"""
multiples_valuation.py
========================
Comps-based ("relative") valuation to sit alongside the DCF.

Three things happen here:

1. PEER COMP STATS: pull median/mean forward P/E, EV/EBITDA, and EV/Revenue
   across the peer set (excluding the subject company), and show where the
   subject trades today relative to that peer group (premium/discount).

2. IMPLIED PRICE VIA MULTIPLES: apply the peer median multiple to the
   subject's own forward metrics (NTM EBITDA, NTM revenue, NTM EPS — sourced
   from the analyst consensus estimate, or the model's own Year-1 forecast as
   a fallback) to get an implied share price under each method.

3. EXIT-MULTIPLE DCF BLEND: builds a Monte Carlo distribution for the
   terminal value's EV/EBITDA exit multiple directly from the peer comp
   range, and blends it with the Gordon Growth terminal value inside the DCF
   (see `exit_multiple_weight` / `exit_ev_ebitda_multiple` on ForecastDrivers
   in model.py). This is what "factors multiples into the DCF" rather than
   just placing them side by side.

Run:  python multiples_valuation.py
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import BaseYearActuals, ForecastDrivers, ThreeStatementModel
from analyst_data import AnalystConsensus
import monte_carlo as mc

from settings import OUTPUT_DIR, BRAND, NEG, INK, output_path


# ---------------------------------------------------------------------------
# 1. Peer comp stats
# ---------------------------------------------------------------------------

def peer_stats(comps: pd.DataFrame) -> pd.DataFrame:
    """Median/mean/min/max for each multiple column, across peers only (excludes subject)."""
    peers = comps[~comps["is_subject"]]
    cols = ["pe_ttm", "pe_fwd", "ev_ebitda_ttm", "ev_ebitda_fwd", "ev_revenue_ttm", "ev_revenue_fwd"]
    stats = peers[cols].agg(["median", "mean", "min", "max"]).T
    return stats


def subject_vs_peers(comps: pd.DataFrame) -> pd.DataFrame:
    """Subject's own multiple vs. peer median, with premium/discount %."""
    subject = comps[comps["is_subject"]].iloc[0]
    stats = peer_stats(comps)
    rows = []
    for col in stats.index:
        subj_val = subject[col]
        peer_med = stats.loc[col, "median"]
        if pd.isna(subj_val) or pd.isna(peer_med) or peer_med == 0:
            continue
        premium = subj_val / peer_med - 1
        rows.append(dict(multiple=col, subject_value=subj_val, peer_median=peer_med,
                          premium_discount=premium))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Implied price via multiples
# ---------------------------------------------------------------------------

def implied_prices_from_multiples(base: BaseYearActuals, consensus: AnalystConsensus,
                                   comps: pd.DataFrame) -> pd.DataFrame:
    """Apply peer median multiples to the subject's own forward metrics.

    NTM EBITDA / revenue are approximated from the consensus next-FY revenue
    estimate combined with the subject's own EBITDA margin (held flat) —
    a simplification flagged in the README. NTM EPS comes directly from the
    consensus EPS estimate.
    """
    stats = peer_stats(comps)
    ttm_ebitda_margin = base.gross_margin - base.opex_pct_revenue  # EBITDA margin from base year

    # Prefer Street NTM estimates; if there's no coverage, fall back to the
    # model's own Year-1 forecast so the multiples layer still runs for
    # uncovered / small-cap names.
    if consensus is not None and consensus.fy_next_revenue_estimate:
        ntm_revenue = consensus.fy_next_revenue_estimate
        ntm_eps = consensus.fy_next_eps_estimate
    else:
        from model import ThreeStatementModel, ForecastDrivers
        drv = ForecastDrivers(wacc=0.10, terminal_gross_margin=base.gross_margin,
                              opex_pct_revenue=base.opex_pct_revenue,
                              da_pct_revenue=base.da_pct_revenue, tax_rate=base.tax_rate)
        y1 = ThreeStatementModel(base, drv).run().iloc[0]
        ntm_revenue = float(y1["revenue"])
        ntm_eps = float(y1["net_income"] / base.shares_diluted)
    ntm_ebitda = ntm_revenue * ttm_ebitda_margin

    rows = []

    pe = stats.loc["pe_fwd", "median"]
    if pd.notna(pe) and ntm_eps is not None and pd.notna(ntm_eps):
        implied_price_pe = pe * ntm_eps
        rows.append(dict(method="P/E (fwd, peer median)", multiple=pe,
                          metric_applied="NTM EPS", metric_value=ntm_eps,
                          implied_price=implied_price_pe))

    ev_ebitda = stats.loc["ev_ebitda_fwd", "median"]
    if pd.notna(ev_ebitda):
        implied_equity_1 = ev_ebitda * ntm_ebitda - base.net_debt
        rows.append(dict(method="EV/EBITDA (fwd, peer median)", multiple=ev_ebitda,
                          metric_applied="NTM EBITDA ($mm)", metric_value=ntm_ebitda,
                          implied_price=implied_equity_1 / base.shares_diluted))

    ev_rev = stats.loc["ev_revenue_fwd", "median"]
    if pd.notna(ev_rev):
        implied_equity_2 = ev_rev * ntm_revenue - base.net_debt
        rows.append(dict(method="EV/Revenue (fwd, peer median)", multiple=ev_rev,
                          metric_applied="NTM Revenue ($mm)", metric_value=ntm_revenue,
                          implied_price=implied_equity_2 / base.shares_diluted))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Exit-multiple-blended DCF + Monte Carlo
# ---------------------------------------------------------------------------

def exit_multiple_dist_from_peers(comps: pd.DataFrame, pad=0.15) -> dict:
    """Build a triangular distribution for the exit EV/EBITDA multiple directly
    from the peer set's forward EV/EBITDA range (mode = peer median, bounds
    widened slightly beyond peer min/max to avoid an artificially tight range)."""
    peers = comps[~comps["is_subject"]]["ev_ebitda_fwd"].dropna()
    lo, mid, hi = peers.min(), peers.median(), peers.max()
    span = hi - lo
    return dict(kind="triangular", low=max(1.0, lo - pad * span), mode=mid, high=hi + pad * span)


def run_exit_multiple_blend(base, base_drivers, comps, blend_weight=0.5, n_trials=20_000, seed=11):
    """Monte Carlo where the terminal value is a blend of Gordon Growth and an
    exit EV/EBITDA multiple drawn from the peer comp distribution — this is
    the mechanism that puts multiples INSIDE the DCF simulation rather than
    just next to it."""
    dist = mc.default_dist(base_drivers)
    dist["exit_ev_ebitda_multiple"] = exit_multiple_dist_from_peers(comps)

    drv = copy.deepcopy(base_drivers)
    drv.exit_multiple_weight = blend_weight  # fixed weight; the multiple itself is randomized

    trials = mc.run_monte_carlo(base, drv, n_trials=n_trials, seed=seed, dist=dist)
    return trials


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_multiples_comparison(comps: pd.DataFrame, implied: pd.DataFrame, base: BaseYearActuals,
                               dcf_base_price: float, street_avg: float, outdir=OUTPUT_DIR, ticker="subject"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: peer forward EV/EBITDA multiples, subject highlighted
    ax = axes[0]
    comps_sorted = comps.sort_values("ev_ebitda_fwd")
    colors = [NEG if s else BRAND for s in comps_sorted["is_subject"]]
    ax.barh(comps_sorted["ticker"], comps_sorted["ev_ebitda_fwd"], color=colors)
    ax.set_xlabel("EV/EBITDA (forward)")
    ax.set_title("Peer Set: Forward EV/EBITDA Multiples")
    ax.grid(axis="x", alpha=0.3)

    # Right: implied prices from each multiples method vs DCF vs street vs current
    ax = axes[1]
    methods = list(implied["method"]) + ["Our base-case DCF", "Street avg target", "Current price"]
    values = list(implied["implied_price"]) + [dcf_base_price, street_avg, base.current_price]
    colors2 = ["#1f77b4"] * len(implied) + ["#d62728", "#333333", "black"]
    y_pos = np.arange(len(methods))
    ax.barh(y_pos, values, color=colors2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Implied price ($/share)")
    ax.set_title("Implied Price: Multiples vs. DCF vs. Street")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = output_path(ticker, "multiples_valuation")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_exit_multiple_blend_histogram(trials_gordon: pd.DataFrame, trials_blend: pd.DataFrame,
                                        base: BaseYearActuals, outdir=OUTPUT_DIR, ticker="subject"):
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(0, max(trials_gordon["implied_price"].quantile(0.995),
                               trials_blend["implied_price"].quantile(0.995)), 90)
    ax.hist(trials_gordon["implied_price"], bins=bins, alpha=0.55, color=BRAND,
            label="Pure Gordon Growth terminal value")
    ax.hist(trials_blend["implied_price"], bins=bins, alpha=0.55, color="#ff7f0e",
            label="50/50 blend: Gordon Growth + peer exit multiple")
    ax.axvline(base.current_price, color=INK, linewidth=2, linestyle="--",
               label=f"Current price (${base.current_price:,.0f})")
    ax.set_title("Terminal Value Method: Gordon Growth vs. Multiples-Blended")
    ax.set_xlabel("Implied price ($/share)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = output_path(ticker, "exit_multiple_blend")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from data_loader import load_all
    data = load_all("data")
    base, base_drivers, consensus, comps = data["base"], data["drivers"], data["consensus"], data["comps"]
    ticker = data["meta"]["ticker"]
    if not data["has_comps"]:
        print("No usable peer comps for this company — skipping multiples layer.")
        return

    print("\n" + "=" * 70)
    print("PEER COMP MULTIPLES — SUMMARY")
    print("=" * 70)
    print(comps[["entity", "ticker", "is_subject", "pe_fwd", "ev_ebitda_fwd", "ev_revenue_fwd"]]
          .to_string(index=False))

    print("\nPeer stats (excludes subject):")
    print(peer_stats(comps).round(2))

    print("\nSubject vs. peer median:")
    svp = subject_vs_peers(comps)
    svp["premium_discount"] = svp["premium_discount"].apply(lambda x: f"{x:+.1%}")
    print(svp.to_string(index=False))

    print("\n" + "=" * 70)
    print("IMPLIED PRICE VIA MULTIPLES (peer median applied to subject's own NTM metrics)")
    print("=" * 70)
    implied = implied_prices_from_multiples(base, consensus, comps)
    print(implied.to_string(index=False))

    # Base-case DCF for comparison
    base_model = ThreeStatementModel(base, base_drivers)
    base_model.run()
    base_val = base_model.dcf_value()
    print(f"\nOur base-case DCF (Gordon Growth terminal value): ${base_val['price_per_share']:,.2f}")
    print(f"Current market price: ${base.current_price:,.2f}")
    print(f"Street avg analyst target: ${consensus.avg_target:,.2f}")

    street_avg = consensus.avg_target if consensus is not None else base.current_price
    chart1 = plot_multiples_comparison(comps, implied, base, base_val["price_per_share"], street_avg, ticker=ticker)
    print(f"\nSaved chart: {chart1}")

    # --- Exit-multiple-blended DCF Monte Carlo ---
    print("\nRunning Monte Carlo with pure Gordon Growth terminal value...")
    trials_gordon = mc.run_monte_carlo(base, base_drivers, n_trials=20_000, seed=42)
    mc.summarize(trials_gordon, base.current_price, label="MONTE CARLO — PURE GORDON GROWTH TV")

    print("\nRunning Monte Carlo with 50/50 Gordon Growth / peer-exit-multiple blend...")
    trials_blend = run_exit_multiple_blend(base, base_drivers, comps, blend_weight=0.5)
    mc.summarize(trials_blend, base.current_price, label="MONTE CARLO — 50/50 GORDON GROWTH / EXIT MULTIPLE BLEND")

    chart2 = plot_exit_multiple_blend_histogram(trials_gordon, trials_blend, base, ticker=ticker)
    print(f"Saved chart: {chart2}")

    implied.to_csv(output_path(ticker, "multiples_implied_prices", "csv"), index=False)
    svp.to_csv(output_path(ticker, "subject_vs_peer_multiples", "csv"), index=False)
    trials_blend.to_csv(output_path(ticker, "exit_multiple_blend_trials", "csv"), index=False)
    print("Saved multiples CSVs to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
