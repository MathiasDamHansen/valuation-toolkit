"""
implied_assumptions.py
======================
"What needs to hold true for today's price to be correct?"

This is a REVERSE Monte Carlo / market-implied assumptions view. Instead of
asking "what is the stock worth", it asks "which combinations of growth, WACC,
margin (and exit multiple) produce a DCF value equal to today's market price?"

Method
------
1. Run (or reuse) a Monte Carlo that randomizes the key value drivers around the
   company's own base assumptions.
2. Keep only the trials whose implied DCF price lands within a tight band of the
   current share price (the "market-consistent" set). The band auto-widens until
   there are enough matches; if even a wide band barely matches, that itself is
   the finding (the price is hard to justify inside plausible ranges).
3. Report, for each driver, the range (P5 / P25 / median / P75 / P95) across the
   market-consistent set — i.e. the assumptions the market appears to embed.

Factoring in multiples
----------------------
When peer comps are available, a second pass blends the DCF terminal value with
an exit EV/EBITDA multiple drawn from the peer set and randomizes THAT multiple
too. The market-consistent subset then also yields an *implied exit EV/EBITDA
range*, which we compare to where peers actually trade — connecting the DCF story
to the multiples story ("for today's price to be right on a DCF basis, you'd also
have to believe an exit multiple of ~X, vs peers at ~Y").

All functions except `render()` are pure (no Streamlit) so they're easy to test.
"""

import copy
import numpy as np
import pandas as pd

import monte_carlo as mc
import multiples_valuation as mv


# Human-readable labels + formatting for each randomized driver.
DRIVER_LABELS = {
    "year1_growth": ("Year-1 revenue growth", "pct"),
    "terminal_growth": ("Terminal growth", "pct"),
    "terminal_gross_margin": ("Terminal gross margin", "pct"),
    "wacc": ("WACC (discount rate)", "pct"),
    "terminal_capex_pct_revenue": ("Terminal capex % of revenue", "pct"),
    "exit_ev_ebitda_multiple": ("Exit EV/EBITDA multiple", "mult"),
}


def _fmt(value, kind):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "mult":
        return f"{value:.1f}x"
    return f"{value:,.2f}"


# ---------------------------------------------------------------------------
# 1. Find the market-consistent subset
# ---------------------------------------------------------------------------

def market_consistent_subset(trials: pd.DataFrame, current_price: float,
                             target_count: int = 80,
                             bands=(0.02, 0.03, 0.05, 0.075, 0.10)) -> tuple:
    """Return (subset, band_used, match_rate). Widen the band until at least
    `target_count` trials fall within +/- band of the current price."""
    n = len(trials)
    best = None
    for band in bands:
        mask = (trials["implied_price"] >= current_price * (1 - band)) & \
               (trials["implied_price"] <= current_price * (1 + band))
        sub = trials[mask]
        best = (sub, band, len(sub) / n)
        if len(sub) >= target_count:
            return best
    return best  # widest band tried, even if sparse


# ---------------------------------------------------------------------------
# 2. Per-driver implied ranges
# ---------------------------------------------------------------------------

_RANGE_COLS = ["field", "driver", "kind", "p5", "p25", "median", "p75", "p95"]


def driver_ranges(subset: pd.DataFrame, fields=None) -> pd.DataFrame:
    """P5 / P25 / median / P75 / P95 for each randomized driver in the subset.
    Always returns a DataFrame with the standard columns (empty if no matches)."""
    if fields is None:
        fields = [c for c in DRIVER_LABELS if c in subset.columns]
    rows = []
    for f in fields:
        if f not in subset.columns or subset[f].dropna().empty:
            continue
        label, kind = DRIVER_LABELS[f]
        p5, p25, p50, p75, p95 = np.percentile(subset[f], [5, 25, 50, 75, 95])
        rows.append(dict(field=f, driver=label, kind=kind,
                         p5=p5, p25=p25, median=p50, p75=p75, p95=p95))
    return pd.DataFrame(rows, columns=_RANGE_COLS)


def ranges_display(ranges: pd.DataFrame) -> pd.DataFrame:
    """Pretty, ready-to-show version of driver_ranges (formatted strings)."""
    out = []
    for _, r in ranges.iterrows():
        k = r["kind"]
        out.append({
            "Driver": r["driver"],
            "Bearish (P5)": _fmt(r["p5"], k),
            "Low (P25)": _fmt(r["p25"], k),
            "Central (median)": _fmt(r["median"], k),
            "High (P75)": _fmt(r["p75"], k),
            "Bullish (P95)": _fmt(r["p95"], k),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 3. Multiples-aware pass (exit EV/EBITDA drawn from peers, then randomized)
# ---------------------------------------------------------------------------

def run_multiples_blended_trials(base, drv, comps, blend_weight=0.5,
                                 n_trials=6000, seed=202) -> pd.DataFrame:
    """Monte Carlo where the terminal value blends Gordon Growth with a peer
    exit EV/EBITDA multiple, and the multiple itself is randomized from the peer
    range. Adds an `exit_ev_ebitda_multiple` column to the trials so we can read
    off the market-implied exit multiple."""
    return mv.run_exit_multiple_blend(base, drv, comps, blend_weight=blend_weight,
                                      n_trials=n_trials, seed=seed)


def peer_fwd_ev_ebitda_median(comps) -> float:
    peers = comps[~comps["is_subject"]]["ev_ebitda_fwd"].dropna()
    return float(peers.median()) if len(peers) else np.nan


# ---------------------------------------------------------------------------
# 4. Narrative
# ---------------------------------------------------------------------------

def build_narrative(ranges: pd.DataFrame, current_price: float, band_used: float,
                    match_rate: float, subset_n: int,
                    implied_exit=None, peer_exit_median=None,
                    price_positioning: dict = None) -> str:
    # No plausible scenario reaches today's price -> that's the finding.
    if subset_n == 0 or ranges.empty:
        msg = (f"⚠️ **Today's price of ${current_price:,.2f} cannot be reconciled to a DCF within a "
               f"plausible range of assumptions.** None of the simulated scenarios (even at a ±{band_used:.0%} "
               f"band) reach it.")
        if price_positioning:
            med = price_positioning.get("median_sim_price")
            p99 = price_positioning.get("p99_sim_price")
            pct_below = price_positioning.get("pct_sims_below_price")
            if med is not None:
                msg += (f" The median simulated fair value is **${med:,.2f}** and even the 99th-percentile "
                        f"scenario only reaches **${p99:,.2f}** — so the market is pricing in materially more "
                        f"than the model's plausible drivers support ({pct_below:.0f}% of scenarios fall below "
                        f"today's price). In short: the price already embeds a very optimistic, out-of-range story.")
        return msg

    def band_txt(field):
        row = ranges[ranges["field"] == field]
        if row.empty:
            return None
        r = row.iloc[0]; k = r["kind"]
        return f"{_fmt(r['p25'], k)}–{_fmt(r['p75'], k)} (mid {_fmt(r['median'], k)})"

    parts = []
    for f in ["year1_growth", "terminal_growth", "terminal_gross_margin", "wacc"]:
        t = band_txt(f)
        if t:
            label = DRIVER_LABELS[f][0].split(" (")[0].lower()
            parts.append(f"**{label}** of {t}")

    lead = (f"For today's price of **${current_price:,.2f}** to be fair on a DCF basis, "
            f"the market appears to be embedding roughly: " + "; ".join(parts) + ".")

    if implied_exit is not None and not np.isnan(implied_exit):
        cmp_txt = ""
        if peer_exit_median is not None and not np.isnan(peer_exit_median):
            rel = "in line with" if abs(implied_exit / peer_exit_median - 1) < 0.1 else (
                "above" if implied_exit > peer_exit_median else "below")
            cmp_txt = f" — {rel} peers at ~{peer_exit_median:.1f}x"
        lead += (f" On a multiples basis, the same price implies an **exit EV/EBITDA of "
                 f"~{implied_exit:.1f}x**{cmp_txt}.")

    if match_rate < 0.03:
        lead += (f" ⚠️ Only {match_rate:.1%} of plausible scenarios ({subset_n:,} trials) land near "
                 f"today's price even at a ±{band_used:.0%} band — the current price sits toward the edge "
                 f"of the plausible range, i.e. it needs a fairly specific set of assumptions to hold.")
    else:
        lead += (f" This is based on {subset_n:,} market-consistent scenarios (within ±{band_used:.0%} of "
                 f"the price, {match_rate:.0%} of all simulations).")
    return lead


# ---------------------------------------------------------------------------
# 5. Full compute bundle (pure — testable without Streamlit)
# ---------------------------------------------------------------------------

def compute(base, drv, trials_pure: pd.DataFrame, has_comps: bool, comps=None,
            n_trials_multiples: int = 6000, target_count: int = 80):
    """Returns a dict with everything needed to render the section."""
    current = base.current_price

    sub, band, rate = market_consistent_subset(trials_pure, current, target_count=target_count)
    ranges = driver_ranges(sub)

    price_positioning = dict(
        median_sim_price=float(trials_pure["implied_price"].median()),
        p99_sim_price=float(np.percentile(trials_pure["implied_price"], 99)),
        pct_sims_below_price=float((trials_pure["implied_price"] < current).mean() * 100),
    )

    implied_exit = None
    peer_exit_med = None
    sub_mult = None
    trials_mult = None
    if has_comps and comps is not None:
        try:
            trials_mult = run_multiples_blended_trials(base, drv, comps, n_trials=n_trials_multiples)
            sub_mult, _, _ = market_consistent_subset(trials_mult, current, target_count=target_count)
            if sub_mult is not None and not sub_mult.empty and "exit_ev_ebitda_multiple" in sub_mult.columns:
                implied_exit = float(np.median(sub_mult["exit_ev_ebitda_multiple"]))
                em_ranges = driver_ranges(sub_mult, fields=["exit_ev_ebitda_multiple"])
                if not em_ranges.empty and not ranges.empty:
                    ranges = pd.concat([ranges, em_ranges], ignore_index=True)
            peer_exit_med = peer_fwd_ev_ebitda_median(comps)
        except Exception:
            pass

    narrative = build_narrative(ranges, current, band, rate, len(sub),
                                implied_exit=implied_exit, peer_exit_median=peer_exit_med,
                                price_positioning=price_positioning)

    return dict(subset=sub, band_used=band, match_rate=rate, ranges=ranges,
                narrative=narrative, implied_exit=implied_exit, peer_exit_median=peer_exit_med,
                subset_multiples=sub_mult, trials_multiples=trials_mult,
                price_positioning=price_positioning)


# ---------------------------------------------------------------------------
# 6. Streamlit rendering
# ---------------------------------------------------------------------------

def render(st, base, drv, trials_pure, has_comps=False, comps=None,
           n_trials_multiples=6000):
    """Render the whole 'what needs to hold true' section at the top of the app."""
    import matplotlib.pyplot as plt
    from settings import BRAND, POS, NEG, INK, BRAND_ALT

    st.markdown("## 🔍 What needs to hold true for today's price?")
    st.caption("A reverse Monte Carlo: of thousands of randomized growth / WACC / margin (and, where "
               "peers exist, exit-multiple) scenarios, these are the ones whose DCF lands on the current "
               "share price — i.e. the assumptions the market appears to be embedding.")

    res = compute(base, drv, trials_pure, has_comps=has_comps, comps=comps,
                  n_trials_multiples=n_trials_multiples)
    ranges = res["ranges"]

    st.markdown(res["narrative"])

    # Headline "market is embedding" metrics (median of the consistent set)
    def med(field):
        row = ranges[ranges["field"] == field]
        return None if row.empty else row.iloc[0]["median"]

    cols = st.columns(5 if res["implied_exit"] is not None else 4)
    cols[0].metric("Implied Year-1 growth", _fmt(med("year1_growth"), "pct"))
    cols[1].metric("Implied terminal growth", _fmt(med("terminal_growth"), "pct"))
    cols[2].metric("Implied terminal margin", _fmt(med("terminal_gross_margin"), "pct"))
    cols[3].metric("Implied WACC", _fmt(med("wacc"), "pct"))
    if res["implied_exit"] is not None:
        delta = None
        if res["peer_exit_median"] is not None and not np.isnan(res["peer_exit_median"]):
            delta = f"vs peers {res['peer_exit_median']:.1f}x"
        cols[4].metric("Implied exit EV/EBITDA", f"{res['implied_exit']:.1f}x", delta)

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("**Assumption ranges that justify today's price**")
        st.dataframe(ranges_display(ranges), use_container_width=True, hide_index=True)
        st.caption(f"Ranges are percentiles across the market-consistent scenarios "
                   f"(±{res['band_used']:.0%} band, {len(res['subset']):,} of {len(trials_pure):,} trials).")

    with right:
        sub = res["subset"]
        if {"wacc", "year1_growth"}.issubset(sub.columns) and len(sub) >= 5:
            st.markdown("**The trade-off frontier**")
            fig, ax = plt.subplots(figsize=(6, 4.2))
            color_field = "terminal_gross_margin" if "terminal_gross_margin" in sub.columns else None
            sc = ax.scatter(sub["wacc"], sub["year1_growth"],
                            c=(sub[color_field] if color_field else BRAND),
                            cmap="viridis" if color_field else None, s=14, alpha=0.7)
            ax.set_xlabel("WACC"); ax.set_ylabel("Year-1 revenue growth")
            ax.set_title("Combinations that reprice to today's price")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            if color_field:
                cb = fig.colorbar(sc, ax=ax); cb.set_label("Terminal gross margin")
                cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            st.caption("Each dot is a scenario that reproduces today's price. Higher growth can justify the "
                       "price only alongside a higher discount rate (and vice-versa) — that's the market's implied trade-off.")
        else:
            st.info("Not enough market-consistent scenarios to draw the trade-off frontier — "
                    "today's price sits near the edge of the plausible range (see the note above).")

    with st.expander("How to read this / method notes"):
        st.markdown(
            "- **What it answers:** rather than a single fair value, it shows the *set* of assumptions "
            "consistent with the market price — a direct read on what you'd need to believe.\n"
            "- **Band:** a scenario 'matches' if its DCF is within the shown ± band of the price; the band "
            "auto-widens only if too few scenarios match.\n"
            "- **Multiples link:** when peers are available, a second pass lets the terminal value be set by a "
            "peer exit EV/EBITDA multiple (randomized across the peer range), so the price is also expressed as "
            "an *implied exit multiple* you can sanity-check against where peers actually trade.\n"
            "- **Edge cases:** a very low match rate means the price is hard to justify within plausible ranges "
            "— informative in itself.")

    return res
