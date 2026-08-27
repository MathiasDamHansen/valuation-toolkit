"""
monte_carlo.py
===============
Monte Carlo reverse-DCF for NVIDIA.

Instead of solving for a single "market-implied" assumption (holding
everything else fixed), this randomizes several key drivers simultaneously,
runs the full 3-statement model + DCF thousands of times, and produces a
DISTRIBUTION of implied share prices. Comparing that distribution to NVIDIA's
actual market price answers: "how aggressive is the market's embedded
assumption set, and what combinations of growth/margin/WACC are consistent
with today's price?"

Run:  python monte_carlo.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import BaseYearActuals, ForecastDrivers, ThreeStatementModel
from settings import OUTPUT_DIR, BRAND, INK, output_path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
N_TRIALS = 20_000
RANDOM_SEED = 42

# Distributions for randomized drivers. Triangular(min, mode, max) is used
# throughout because it's the easiest to reason about and doesn't require
# assuming a particular statistical shape — just "plausible range + best guess".
DIST = dict(
    year1_growth=dict(kind="triangular", low=0.15, mode=0.35, high=0.55),
    terminal_growth=dict(kind="triangular", low=0.02, mode=0.04, high=0.065),
    terminal_gross_margin=dict(kind="triangular", low=0.60, mode=0.68, high=0.74),
    wacc=dict(kind="triangular", low=0.085, mode=0.11, high=0.145),
    terminal_capex_pct_revenue=dict(kind="triangular", low=0.03, mode=0.045, high=0.07),
)

# Correlation knobs (simple, illustrative): higher growth tends to come with
# somewhat higher capex intensity and a *lower* discount rate demanded by the
# market during "risk-on" periods. We apply a mild rank-based nudge rather
# than a full copula, to keep this transparent and easy to edit.
CORRELATE_GROWTH_WITH_CAPEX = 0.4   # 0 = independent, 1 = fully rank-correlated
CORRELATE_GROWTH_WITH_WACC = -0.25  # negative: high growth trials skew to lower WACC


def default_dist(drv):
    """Build a Monte Carlo distribution dict CENTERED on this company's own base
    drivers, rather than a fixed NVDA-flavored range. Widths are proportional so
    it scales sensibly from a slow-growing micro-cap to a hyper-grower."""
    g = drv.year1_growth
    tg = drv.terminal_growth
    gm = drv.terminal_gross_margin
    w = drv.wacc
    capx = drv.terminal_capex_pct_revenue
    return dict(
        year1_growth=dict(kind="triangular",
                          low=max(-0.10, g - max(0.10, 0.4 * abs(g))),
                          mode=g,
                          high=g + max(0.12, 0.5 * abs(g))),
        terminal_growth=dict(kind="triangular",
                             low=max(0.0, tg - 0.015), mode=tg, high=min(w - 0.01, tg + 0.02)),
        terminal_gross_margin=dict(kind="triangular",
                                   low=max(0.05, gm - 0.06), mode=gm, high=min(0.95, gm + 0.05)),
        wacc=dict(kind="triangular", low=max(0.04, w - 0.025), mode=w, high=w + 0.03),
        terminal_capex_pct_revenue=dict(kind="triangular",
                                        low=max(0.005, capx - 0.02), mode=capx, high=capx + 0.025),
    )


def sample_triangular(rng, low, mode, high, size):
    return rng.triangular(low, mode, high, size)


# Fields where we apply the mild rank-correlation nudges below (only used if
# both fields happen to be present in the distribution dict being sampled).
def build_trial_dataframe(rng, n, dist=None) -> pd.DataFrame:
    """Generic sampler: draws every field in `dist` as an independent triangular,
    then applies a couple of illustrative rank-correlation nudges if the
    relevant fields are present. Works with any subset/superset of fields
    (e.g. adding exit_ev_ebitda_multiple for a multiples-blended run) without
    code changes here."""
    dist = DIST if dist is None else dist
    cols = {}
    for key, spec in dist.items():
        if spec.get("kind", "triangular") != "triangular":
            raise ValueError(f"Unsupported distribution kind for '{key}': {spec.get('kind')}")
        cols[key] = sample_triangular(rng, spec["low"], spec["mode"], spec["high"], size=n)

    def nudge_rank_correlate(base, target, corr):
        """Blend `target`'s rank order toward `base`'s rank order by `corr` (simple, transparent)."""
        order_base = np.argsort(np.argsort(base))
        order_target_sorted = np.sort(target)
        matched = order_target_sorted[order_base]
        return (1 - abs(corr)) * target + abs(corr) * (matched if corr >= 0 else matched[::-1])

    if "year1_growth" in cols and "terminal_capex_pct_revenue" in cols:
        cols["terminal_capex_pct_revenue"] = nudge_rank_correlate(
            cols["year1_growth"], cols["terminal_capex_pct_revenue"], CORRELATE_GROWTH_WITH_CAPEX)
    if "year1_growth" in cols and "wacc" in cols:
        cols["wacc"] = nudge_rank_correlate(cols["year1_growth"], cols["wacc"], CORRELATE_GROWTH_WITH_WACC)
    if "terminal_growth" in cols and "wacc" in cols:
        # keep terminal growth strictly below wacc (required for finite terminal value)
        cols["terminal_growth"] = np.minimum(cols["terminal_growth"], cols["wacc"] - 0.01)

    return pd.DataFrame(cols)


# ForecastDrivers fields that a Monte Carlo trial is allowed to override.
# Anything sampled in `dist` with a matching name here gets passed straight
# through to ForecastDrivers(); everything else on ForecastDrivers keeps its
# value from `base_drivers`.
_OVERRIDABLE_DRIVER_FIELDS = {
    "year1_growth", "terminal_growth", "terminal_gross_margin",
    "opex_pct_revenue", "da_pct_revenue", "capex_pct_revenue",
    "terminal_capex_pct_revenue", "tax_rate", "wacc",
    "exit_multiple_weight", "exit_ev_ebitda_multiple",
    "sbc_dilution_pct", "buyback_pct",
}


def run_monte_carlo(base: BaseYearActuals, base_drivers: ForecastDrivers,
                     n_trials=N_TRIALS, seed=RANDOM_SEED, dist=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if dist is None:
        dist = default_dist(base_drivers)   # auto-center on this company's drivers
    trials = build_trial_dataframe(rng, n_trials, dist=dist)

    prices = np.empty(n_trials)
    ev_list = np.empty(n_trials)

    base_kwargs = dict(
        n_years=base_drivers.n_years,
        year1_growth=base_drivers.year1_growth,
        terminal_growth=base_drivers.terminal_growth,
        terminal_gross_margin=base_drivers.terminal_gross_margin,
        opex_pct_revenue=base_drivers.opex_pct_revenue,
        da_pct_revenue=base_drivers.da_pct_revenue,
        capex_pct_revenue=base_drivers.capex_pct_revenue,
        terminal_capex_pct_revenue=base_drivers.terminal_capex_pct_revenue,
        tax_rate=base_drivers.tax_rate,
        ar_days=base_drivers.ar_days,
        inventory_days=base_drivers.inventory_days,
        ap_days=base_drivers.ap_days,
        other_nwc_pct_revenue=base_drivers.other_nwc_pct_revenue,
        wacc=base_drivers.wacc,
        dividend_payout_pct_ni=base_drivers.dividend_payout_pct_ni,
        exit_multiple_weight=base_drivers.exit_multiple_weight,
        exit_ev_ebitda_multiple=base_drivers.exit_ev_ebitda_multiple,
        sbc_dilution_pct=base_drivers.sbc_dilution_pct,
        buyback_pct=base_drivers.buyback_pct,
        segments=base_drivers.segments,
    )
    overridable_cols = [c for c in trials.columns if c in _OVERRIDABLE_DRIVER_FIELDS]

    for i, row in trials.iterrows():
        kwargs = dict(base_kwargs)
        for col in overridable_cols:
            kwargs[col] = row[col]
        drv = ForecastDrivers(**kwargs)

        m = ThreeStatementModel(base, drv)
        m.run()
        val = m.dcf_value()  # uses drv's own wacc/terminal_growth/exit-multiple fields by default
        prices[i] = val["price_per_share"]
        ev_list[i] = val["enterprise_value"]

    trials["implied_price"] = prices
    trials["implied_ev"] = ev_list
    return trials


def summarize(trials: pd.DataFrame, market_price: float, label="MONTE CARLO REVERSE-DCF — SUMMARY",
              price_label="Current market price"):
    pct_below = (trials["implied_price"] < market_price).mean() * 100
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(f"Trials run:                     {len(trials):,}")
    print(f"{price_label + ':':<33}${market_price:,.2f}")
    print(f"Simulated implied price — mean: ${trials['implied_price'].mean():,.2f}")
    print(f"Simulated implied price — median:${trials['implied_price'].median():,.2f}")
    print(f"Simulated implied price — std:  ${trials['implied_price'].std():,.2f}")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p:>2}th percentile:            ${np.percentile(trials['implied_price'], p):,.2f}")
    print(f"\nMarket price sits at the {pct_below:.1f}th percentile of the simulated distribution.")
    print("(i.e., that share of random draws produced a LOWER fair value than today's price.)")

    # "Market-consistent" subset: trials landing within +/-3% of actual price
    band = 0.03
    consistent = trials[
        (trials["implied_price"] >= market_price * (1 - band)) &
        (trials["implied_price"] <= market_price * (1 + band))
    ]
    print(f"\nTrials landing within +/-{band*100:.0f}% of market price: {len(consistent):,} "
          f"({len(consistent)/len(trials)*100:.1f}% of all trials)")
    if len(consistent) > 20:
        print("Characteristics of those 'market-consistent' trials (5th/50th/95th pct):")
        for col in ["year1_growth", "terminal_growth", "terminal_gross_margin", "wacc",
                    "terminal_capex_pct_revenue"]:
            lo, mid, hi = np.percentile(consistent[col], [5, 50, 95])
            print(f"  {col:28s}: {lo:6.2%}  |  {mid:6.2%}  |  {hi:6.2%}")
    print("=" * 70 + "\n")
    return consistent


def make_plots(trials: pd.DataFrame, market_price: float, outdir=OUTPUT_DIR, ticker="subject"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 1. Histogram of implied price vs market price
    ax = axes[0, 0]
    ax.hist(trials["implied_price"], bins=80, color=BRAND, alpha=0.85, edgecolor="none")
    ax.axvline(market_price, color=INK, linewidth=2, linestyle="--",
               label=f"Market price (${market_price:,.0f})")
    ax.set_title("Distribution of Simulated Implied Share Price")
    ax.set_xlabel("Implied price ($/share)")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, np.percentile(trials["implied_price"], 99.5))
    ax.legend()

    # 2-6. Scatter of each driver vs implied price
    drivers = ["year1_growth", "terminal_growth", "terminal_gross_margin",
               "wacc", "terminal_capex_pct_revenue"]
    titles = ["Year-1 Revenue Growth", "Terminal Growth Rate", "Terminal Gross Margin",
              "WACC", "Terminal Capex % of Revenue"]
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for drv, title, pos in zip(drivers, titles, positions):
        ax = axes[pos]
        sc = ax.scatter(trials[drv], trials["implied_price"], s=3, alpha=0.15, color=BRAND)
        ax.axhline(market_price, color=INK, linewidth=1.5, linestyle="--")
        ax.set_title(f"Implied Price vs. {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Implied price ($)")
        ax.set_ylim(0, np.percentile(trials["implied_price"], 99))

    plt.tight_layout()
    path = output_path(ticker, "monte_carlo_results")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved chart: {path}")
    return path


def main(data_dir="data"):
    from data_loader import load_all
    data = load_all(data_dir)
    base, base_drivers = data["base"], data["drivers"]
    ticker = data["meta"]["ticker"]

    # --- 1. Deterministic base case ---
    m = ThreeStatementModel(base, base_drivers)
    df = m.run()
    base_val = m.dcf_value()
    print("\nBASE-CASE 3-STATEMENT MODEL (first 5 forecast years):")
    print(df[["revenue", "growth", "ebit_margin", "ufcf"]].head().round(3))
    print(f"\nBase-case DCF: EV=${base_val['enterprise_value']:,.0f}mm | "
          f"Equity=${base_val['equity_value']:,.0f}mm | "
          f"Implied price=${base_val['price_per_share']:,.2f} "
          f"(actual market price=${base.current_price:,.2f})")

    # --- 2. Single-point reverse solve: what terminal growth reconciles to market price? ---
    try:
        implied_tg = m.solve_for_terminal_growth(base.current_price, wacc=base_drivers.wacc)
        print(f"\nHolding WACC at {base_drivers.wacc:.1%}, the terminal growth rate that "
              f"reconciles the model to the current price of ${base.current_price:,.2f} is "
              f"{implied_tg:.2%}.")
    except ValueError as e:
        print(f"\n[Reverse solve for terminal growth failed: {e}]")

    try:
        implied_wacc = m.solve_for_wacc(base.current_price, terminal_growth=base_drivers.terminal_growth)
        print(f"Holding terminal growth at {base_drivers.terminal_growth:.1%}, the WACC that "
              f"reconciles the model to the current price is {implied_wacc:.2%}.")
    except ValueError as e:
        print(f"[Reverse solve for WACC failed: {e}]")

    # --- 3. Monte Carlo ---
    print(f"\nRunning Monte Carlo with {N_TRIALS:,} trials...")
    trials = run_monte_carlo(base, base_drivers)
    consistent = summarize(trials, base.current_price)

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    make_plots(trials, base.current_price, ticker=ticker)

    trials.to_csv(output_path(ticker, "monte_carlo_trials", "csv"), index=False)
    print(f"Saved raw trial data: {output_path(ticker, 'monte_carlo_trials', 'csv')}")

    if len(consistent) > 0:
        consistent.to_csv(output_path(ticker, "market_consistent_trials", "csv"), index=False)
        print(f"Saved market-consistent subset: {output_path(ticker, 'market_consistent_trials', 'csv')}")


if __name__ == "__main__":
    main()
