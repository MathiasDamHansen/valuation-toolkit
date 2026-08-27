"""
market_consistent.py
====================
"What must hold true for today's price to be fair?"

Given a Monte Carlo trial set (each row = one random combination of growth,
WACC, margin, terminal growth, capex — and, when peers exist, an exit
EV/EBITDA multiple — plus the resulting DCF `implied_price`), this isolates the
subset of trials whose DCF value ≈ the current share price. The spread of each
driver within that subset is the answer to "what assumption set does the market
appear to be pricing in?"

It also computes the trading multiples the current price implies (fwd
EV/EBITDA, P/E, EV/Revenue) and compares them to the peer set, so multiples are
factored in two complementary ways:
  1. as a randomized dimension inside the reverse-DCF cloud (exit multiple), and
  2. as a direct implied-multiple read-out vs peers.

Pure functions, no Streamlit/Matplotlib dependency here, so they're easy to test.
"""

import numpy as np
import pandas as pd


# Drivers we report on, with display label and formatting kind.
DRIVER_META = [
    ("year1_growth",               "Year-1 revenue growth", "pct"),
    ("terminal_growth",            "Terminal growth",       "pct"),
    ("terminal_gross_margin",      "Terminal gross margin", "pct"),
    ("wacc",                       "WACC",                  "pct"),
    ("terminal_capex_pct_revenue", "Terminal capex % rev",  "pct"),
    ("exit_ev_ebitda_multiple",    "Exit EV/EBITDA",        "mult"),
]


def fmt_value(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.1%}" if kind == "pct" else f"{v:.1f}x"


def market_consistent_subset(trials: pd.DataFrame, current_price: float,
                             band: float = 0.025, target_min: int = 150,
                             max_band: float = 0.15):
    """Return (subset_df, band_used, status).

    status:
      'ok'     -> found a usable subset around the current price
      'sparse' -> price is inside the cloud but very few trials landed near it
      'above'  -> current price exceeds EVERY simulated fair value
                  (even the most bullish scenario doesn't justify it)
      'below'  -> current price is under every simulated fair value
                  (even the most bearish scenario is above it)

    The band auto-widens from `band` up to `max_band` until at least
    `target_min` trials fall within it, so the read-out stays stable.
    """
    ip = trials["implied_price"]
    lo_all, hi_all = ip.min(), ip.max()
    if current_price > hi_all:
        return trials.iloc[0:0], band, "above"
    if current_price < lo_all:
        return trials.iloc[0:0], band, "below"

    b = band
    while b <= max_band + 1e-9:
        lo, hi = current_price * (1 - b), current_price * (1 + b)
        sub = trials[(ip >= lo) & (ip <= hi)]
        if len(sub) >= target_min:
            return sub, b, "ok"
        b += 0.005

    lo, hi = current_price * (1 - max_band), current_price * (1 + max_band)
    sub = trials[(ip >= lo) & (ip <= hi)]
    return sub, max_band, ("ok" if len(sub) >= 20 else "sparse")


def consistent_ranges(subset: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Percentile ranges (10th/50th/90th) of each driver within the
    market-consistent subset, alongside the full-simulation 10th/90th for
    context. Returns a display-ready frame plus raw numeric columns for charts.
    """
    rows = []
    for col, label, kind in DRIVER_META:
        if col not in subset.columns or subset[col].isna().all():
            continue
        c10, c50, c90 = np.percentile(subset[col], [10, 50, 90])
        if col in full.columns:
            f10, f90 = np.percentile(full[col], [10, 90])
        else:
            f10 = f90 = np.nan
        rows.append(dict(
            driver=label, kind=kind,
            need_low=fmt_value(c10, kind), need_mid=fmt_value(c50, kind), need_high=fmt_value(c90, kind),
            full_low=fmt_value(f10, kind), full_high=fmt_value(f90, kind),
            _c10=c10, _c50=c50, _c90=c90, _f10=f10, _f90=f90,
        ))
    return pd.DataFrame(rows)


def implied_multiples_at_price(base, forecast, comps: pd.DataFrame = None) -> dict:
    """Trading multiples embedded in TODAY's price, using the model's own
    Year-1 (NTM) forecast for the denominators. Compares to peer medians when
    a peer set is available."""
    shares = base.shares_diluted
    ev_now = base.current_price * shares + base.net_debt
    ntm_rev = float(forecast["revenue"].iloc[0])
    ntm_ebitda = float(forecast["ebitda"].iloc[0])
    ntm_ni = float(forecast["net_income"].iloc[0])
    ntm_eps = ntm_ni / shares if shares else np.nan

    implied = {
        "Fwd EV/EBITDA": (ev_now / ntm_ebitda) if ntm_ebitda else np.nan,
        "Fwd P/E": (base.current_price / ntm_eps) if ntm_eps else np.nan,
        "Fwd EV/Revenue": (ev_now / ntm_rev) if ntm_rev else np.nan,
    }
    peer = {}
    if comps is not None and len(comps):
        try:
            import multiples_valuation as mv
            st = mv.peer_stats(comps)
            peer = {
                "Fwd EV/EBITDA": st.loc["ev_ebitda_fwd", "median"],
                "Fwd P/E": st.loc["pe_fwd", "median"],
                "Fwd EV/Revenue": st.loc["ev_revenue_fwd", "median"],
            }
        except Exception:
            peer = {}
    return dict(implied=implied, peer=peer)


def narrative(subset: pd.DataFrame, base_drivers, status: str, band: float) -> str:
    """A one-line plain-English summary of what the market is pricing in."""
    if status == "above":
        return ("Today's price sits **above every** simulated fair value — even the most "
                "optimistic combination of growth, margin and multiple in the range tested "
                "doesn't reach it. The market is either pricing in something beyond these "
                "ranges, or the stock looks expensive on these assumptions.")
    if status == "below":
        return ("Today's price sits **below every** simulated fair value — even the most "
                "conservative scenario values it higher. On these assumptions the stock "
                "looks cheap.")
    if len(subset) == 0:
        return "Not enough trials landed near the current price to characterise it — widen the band."
    g = np.median(subset["year1_growth"]) if "year1_growth" in subset else None
    w = np.median(subset["wacc"]) if "wacc" in subset else None
    parts = []
    if g is not None:
        parts.append(f"~**{g:.0%}** Year-1 revenue growth")
    if w is not None:
        parts.append(f"a **{w:.1%}** WACC")
    if "exit_ev_ebitda_multiple" in subset.columns and not subset["exit_ev_ebitda_multiple"].isna().all():
        em = np.median(subset["exit_ev_ebitda_multiple"])
        parts.append(f"an exit multiple near **{em:.0f}x** EV/EBITDA")
    joined = ", ".join(parts) if parts else "a specific mix of assumptions"
    return (f"To justify today's price (±{band*100:.0f}% band, {len(subset):,} matching "
            f"scenarios), the market needs roughly {joined} — that combination, held together, "
            f"reproduces the current share price.")
