"""
app.py — Streamlit interactive dashboard
========================================
A live, interactive front-end for the reverse-DCF + consensus + multiples
toolkit. Works for ANY company: pick a bundled dataset, or upload your own five
CSVs (generated via AGENT_DATA_COLLECTION_PROMPT.md). Consensus/comps layers
auto-appear only when that data is present, so it works for uncovered small-caps
too.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy free: push the repo to GitHub, then on share.streamlit.io point a new app
at app.py. That gives you a public live dashboard URL — the Python-native
equivalent of a "Lovable website", but able to actually run this model.
"""

import copy
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_loader import load_all
from model import ThreeStatementModel
import monte_carlo as mc
import consensus_analysis as ca
import multiples_valuation as mv
import enhancements as enh
from settings import BRAND, BRAND_ALT, POS, NEG, INK

st.set_page_config(page_title="Reverse-DCF Valuation", layout="wide")

BUNDLED = {"NVDA (full coverage)": "data", "TINYCO (no coverage small-cap)": "data_smallcap"}
REQUIRED = ["company_inputs.csv", "financial_history.csv"]
OPTIONAL = ["analyst_consensus.csv", "analyst_bank_targets.csv", "multiples_comps.csv",
            "segments.csv", "valuation_history.csv"]


@st.cache_data(show_spinner=False)
def _load(data_dir):
    return load_all(data_dir)


def _dir_from_upload(files):
    tmp = tempfile.mkdtemp(prefix="valuation_")
    for f in files:
        with open(os.path.join(tmp, f.name), "wb") as out:
            out.write(f.getbuffer())
    return tmp


# --------------------------------------------------------------------------
# Sidebar: data source + assumption overrides
# --------------------------------------------------------------------------
st.sidebar.title("Valuation toolkit")
source = st.sidebar.radio("Data source", ["Bundled example", "Upload my 5 CSVs"])

if source == "Bundled example":
    choice = st.sidebar.selectbox("Company", list(BUNDLED.keys()))
    data_dir = BUNDLED[choice]
else:
    st.sidebar.caption("Upload the CSVs from AGENT_DATA_COLLECTION_PROMPT.md. "
                       "Required: company_inputs.csv + financial_history.csv. "
                       "Consensus/comps/segments are optional.")
    ups = st.sidebar.file_uploader("CSV files", type="csv", accept_multiple_files=True)
    data_dir = None
    if ups:
        names = [u.name for u in ups]
        if all(r in names for r in REQUIRED):
            data_dir = _dir_from_upload(ups)
        else:
            st.sidebar.error(f"Need at least: {', '.join(REQUIRED)}")

if not data_dir:
    st.title("Reverse-DCF + Consensus + Multiples")
    st.info("Pick a bundled example or upload your CSVs in the sidebar to begin.")
    st.stop()

data = _load(data_dir)
base, drv0, meta = data["base"], data["drivers"], data["meta"]
has_consensus, has_comps = data["has_consensus"], data["has_comps"]

# --------------------------------------------------------------------------
# 5-year historical averages (from financial_history.csv) to guide the sliders
# --------------------------------------------------------------------------
def _hist_avg(col, n):
    """Mean of the last up to `n` non-null values of a history column, or None."""
    try:
        hist = data.get("history")
        if hist is None or col not in hist.columns:
            return None
        s = hist[col].dropna().tail(n)
        return float(s.mean()) if len(s) else None
    except Exception:
        return None


def _pct(v):
    return f"{v:.1%}" if v is not None else "n/a"


# 2-year (recent trend) and 5-year (longer-run) historical averages
avg_rev_growth_2y = _hist_avg("revenue_growth", 2)
avg_rev_growth_5y = _hist_avg("revenue_growth", 5)
avg_gross_margin_2y = _hist_avg("gross_margin", 2)
avg_gross_margin_5y = _hist_avg("gross_margin", 5)

st.sidebar.markdown("### Assumptions")
n_trials = st.sidebar.select_slider("Monte Carlo trials", [1000, 2000, 5000, 10000, 20000], value=5000)

y1 = st.sidebar.slider(
    "Year-1 revenue growth", -0.20, 1.50, float(round(drv0.year1_growth, 3)), 0.01,
    help=(f"Historical guide — revenue growth: "
          f"2-yr avg {_pct(avg_rev_growth_2y)}, 5-yr avg {_pct(avg_rev_growth_5y)}"))
st.sidebar.caption(f"↳ Revenue growth — 2-yr avg: **{_pct(avg_rev_growth_2y)}** · "
                   f"5-yr avg: **{_pct(avg_rev_growth_5y)}**")

tg = st.sidebar.slider(
    "Terminal growth", 0.00, 0.08, float(round(drv0.terminal_growth, 3)), 0.005,
    help=(f"Reference — revenue growth: 2-yr avg {_pct(avg_rev_growth_2y)}, "
          f"5-yr avg {_pct(avg_rev_growth_5y)}. Terminal growth is normally set well "
          f"below these, toward a long-run / GDP-like rate."))
st.sidebar.caption(f"↳ Revenue growth (ref) — 2-yr avg: **{_pct(avg_rev_growth_2y)}** · "
                   f"5-yr avg: **{_pct(avg_rev_growth_5y)}**")

gm = st.sidebar.slider(
    "Terminal gross margin", 0.05, 0.95, float(round(drv0.terminal_gross_margin, 3)), 0.01,
    help=(f"Historical guide — gross margin: "
          f"2-yr avg {_pct(avg_gross_margin_2y)}, 5-yr avg {_pct(avg_gross_margin_5y)}"))
st.sidebar.caption(f"↳ Gross margin — 2-yr avg: **{_pct(avg_gross_margin_2y)}** · "
                   f"5-yr avg: **{_pct(avg_gross_margin_5y)}**")

wacc = st.sidebar.slider("WACC", 0.04, 0.20, float(round(drv0.wacc, 3)), 0.005)
exit_w = st.sidebar.slider("Exit-multiple weight in TV", 0.0, 1.0, float(drv0.exit_multiple_weight), 0.05)
sbc = st.sidebar.slider("SBC dilution % / yr", 0.0, 0.05, float(drv0.sbc_dilution_pct), 0.005)
bb = st.sidebar.slider("Buyback % / yr", 0.0, 0.05, float(drv0.buyback_pct), 0.005)

drv = copy.deepcopy(drv0)
drv.year1_growth, drv.terminal_growth, drv.terminal_gross_margin = y1, tg, gm
drv.wacc, drv.exit_multiple_weight = wacc, exit_w
drv.sbc_dilution_pct, drv.buyback_pct = sbc, bb
if drv.custom_growth_path is not None:  # user override wins over auto path
    drv.custom_growth_path = None

# --------------------------------------------------------------------------
# Core computations
# --------------------------------------------------------------------------
m = ThreeStatementModel(base, drv)
forecast = m.run()
val = m.dcf_value()
dcf_price = val["price_per_share"]

with st.spinner(f"Running {n_trials:,} Monte Carlo trials..."):
    trials_a = mc.run_monte_carlo(base, drv, n_trials=n_trials, seed=42)
pct_below = (trials_a["implied_price"] < base.current_price).mean() * 100

# reverse solves
try:
    imp_tg = m.solve_for_terminal_growth(base.current_price, wacc=drv.wacc)
except ValueError:
    imp_tg = None
try:
    imp_wacc = m.solve_for_wacc(base.current_price, terminal_growth=drv.terminal_growth)
except ValueError:
    imp_wacc = None

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.title(f"{meta['company_name']} ({meta['ticker']})")
st.caption(f"As of {meta['as_of_date']} · coverage — consensus: "
           f"{'yes' if has_consensus else 'no'}, peers: {'yes' if has_comps else 'no'}, "
           f"segments: {'yes' if data['segments'] else 'no'}")

k = st.columns(5)
k[0].metric("Current price", f"${base.current_price:,.2f}")
k[1].metric("Base-case DCF", f"${dcf_price:,.2f}", f"{dcf_price/base.current_price-1:+.0%}")
k[2].metric("MC percentile of price", f"{pct_below:.0f}th")
k[3].metric("Implied terminal growth", f"{imp_tg:.2%}" if imp_tg is not None else "n/a")
k[4].metric("Implied WACC", f"{imp_wacc:.2%}" if imp_wacc is not None else "n/a")

# --------------------------------------------------------------------------
# NEW: "What must hold true for today's price to be fair?"
# --------------------------------------------------------------------------
import market_consistent as mcx

st.markdown("## 🎯 What must hold true for today's price to be fair?")
st.caption("The reverse-DCF run below keeps only the simulated scenarios whose DCF value "
           "lands on today's share price. The spread of each assumption in that surviving "
           "set is, in effect, what the market appears to be pricing in. When a peer set "
           "exists, the exit EV/EBITDA multiple is randomised too, so multiples are baked in.")

mc_band = st.slider("Match tolerance around current price (±%)", 1.0, 10.0, 2.5, 0.5,
                    help="A scenario 'reproduces' the price if its DCF value is within this "
                         "band of the current price. Widen it if too few scenarios match.") / 100.0

# A dedicated reverse-DCF cloud for this question. When peers exist we use the
# exit-multiple-blended variant (adds the exit EV/EBITDA multiple as a dimension);
# otherwise the standard own-assumptions cloud. Capped at 8k trials for speed.
_mc_trials = min(max(n_trials, 4000), 8000)
with st.spinner(f"Solving the reverse-DCF cloud ({_mc_trials:,} scenarios)..."):
    if has_comps:
        trials_consistent = mv.run_exit_multiple_blend(base, drv, data["comps"],
                                                       blend_weight=0.5, n_trials=_mc_trials, seed=11)
    else:
        trials_consistent = mc.run_monte_carlo(base, drv, n_trials=_mc_trials, seed=42)

sub, band_used, status = mcx.market_consistent_subset(trials_consistent, base.current_price, band=mc_band)

st.info(mcx.narrative(sub, drv, status, band_used))

mc_left, mc_right = st.columns([1.35, 1])

with mc_left:
    st.markdown("**What each assumption must be (10th–90th percentile of matching scenarios)**")
    if status in ("above", "below"):
        st.warning("No scenarios matched — today's price is outside the entire simulated range, "
                   "so there's no assumption set within these ranges that reproduces it. "
                   "Try widening the driver ranges, or read this as the market pricing in "
                   "something beyond the tested bounds.")
    else:
        ranges = mcx.consistent_ranges(sub, trials_consistent)
        show = ranges[["driver", "need_low", "need_mid", "need_high", "full_low", "full_high"]].rename(
            columns={"driver": "Assumption", "need_low": "Need: low (10th)",
                     "need_mid": "Need: mid (50th)", "need_high": "Need: high (90th)",
                     "full_low": "Full range low", "full_high": "Full range high"})
        st.dataframe(show, use_container_width=True, hide_index=True)
        # Range-band chart: market-consistent band (bold) vs full simulated band (faint)
        rr = ranges[ranges["kind"] == "pct"]
        if len(rr):
            figc, axc = plt.subplots(figsize=(6.4, 0.6 * len(rr) + 1.2))
            yv = np.arange(len(rr))
            for i, (_, r) in enumerate(rr.iterrows()):
                axc.plot([r["_f10"], r["_f90"]], [i, i], color="#c9ced6", lw=7, solid_capstyle="round")
                axc.plot([r["_c10"], r["_c90"]], [i, i], color=BRAND, lw=7, solid_capstyle="round")
                axc.plot(r["_c50"], i, "o", color=INK, ms=5)
            axc.set_yticks(yv); axc.set_yticklabels(rr["driver"], fontsize=8)
            axc.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            axc.set_title("Blue = must-hold range · grey = full tested range", fontsize=9)
            axc.grid(axis="x", alpha=0.3)
            st.pyplot(figc)

with mc_right:
    st.markdown("**Multiples the current price implies**")
    im = mcx.implied_multiples_at_price(base, forecast, data["comps"] if has_comps else None)
    mult_rows = []
    for kkey in ["Fwd EV/EBITDA", "Fwd P/E", "Fwd EV/Revenue"]:
        iv = im["implied"].get(kkey)
        pv = im["peer"].get(kkey)
        mult_rows.append({
            "Multiple": kkey,
            "At today's price": (f"{iv:.1f}x" if iv == iv else "—"),
            "Peer median": (f"{pv:.1f}x" if (pv is not None and pv == pv) else "—"),
        })
    st.dataframe(pd.DataFrame(mult_rows), use_container_width=True, hide_index=True)
    if has_comps:
        st.caption("If 'at today's price' sits below the peer median, the market is paying "
                   "less than peers for the same forward metric (a relative discount), and "
                   "vice-versa.")
    else:
        st.caption("No peer set loaded, so only the implied side is shown. Add a "
                   "multiples_comps.csv to compare against peers.")

st.divider()

# --------------------------------------------------------------------------
# Consolidated summary
# --------------------------------------------------------------------------
rows = [("Current market price", base.current_price),
        ("Base-case DCF", dcf_price),
        ("Monte Carlo median (own)", trials_a["implied_price"].median())]
if has_consensus:
    cons = data["consensus"]
    _, cons_val, _ = ca.run_consensus_case(base, drv, cons, multiyear=True)
    rows += [("Consensus-anchored DCF", cons_val["price_per_share"]),
             ("Street avg target", cons.avg_target),
             ("Street median target", cons.median_target)]
if has_comps:
    imp = mv.implied_prices_from_multiples(base, data["consensus"], data["comps"])
    rows += [(f"Multiples: {r['method']}", r["implied_price"]) for _, r in imp.iterrows()]
scen = enh.scenario_table(base, drv)
for s in ["Bear", "Base", "Bull"]:
    rows.append((f"Scenario: {s}", float(scen.loc[scen.scenario == s, "implied_price"].iloc[0])))
summary = pd.DataFrame(rows, columns=["method", "price"]).sort_values("price", ascending=False)

left, right = st.columns([1, 1])
with left:
    st.subheader("Consolidated valuation")
    st.dataframe(summary.style.format({"price": "${:,.2f}"}), use_container_width=True, height=430)
with right:
    st.subheader("Monte Carlo distribution")
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.hist(trials_a["implied_price"], bins=70, color=BRAND, alpha=0.85)
    ax.axvline(base.current_price, color=INK, ls="--", lw=2, label=f"Price ${base.current_price:,.0f}")
    ax.axvline(dcf_price, color=NEG, ls="-", lw=2, label=f"Base DCF ${dcf_price:,.0f}")
    ax.set_xlim(0, np.percentile(trials_a["implied_price"], 99))
    ax.set_xlabel("Implied price ($)"); ax.legend(fontsize=8)
    st.pyplot(fig)

# --------------------------------------------------------------------------
# Tornado + 2D grid + scenarios
# --------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Tornado — driver sensitivity")
    torn = enh.tornado_sensitivity(base, drv)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bp = torn.attrs["base_price"]
    for i, r in torn.iterrows():
        ax2.barh([i], [r["price_high"] - bp], left=[bp], color=POS)
        ax2.barh([i], [r["price_low"] - bp], left=[bp], color=NEG)
    ax2.axvline(bp, color=INK, lw=1.5)
    ax2.set_yticks(range(len(torn))); ax2.set_yticklabels(torn["driver"], fontsize=8)
    ax2.set_xlabel("Implied price ($)")
    st.pyplot(fig2)
with c2:
    st.subheader("Scenarios (bear / base / bull)")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    cmap = {"Bear": NEG, "Base": BRAND, "Bull": POS}
    ax3.bar(scen["scenario"], scen["implied_price"], color=[cmap[s] for s in scen["scenario"]])
    ax3.axhline(base.current_price, color=INK, ls="--")
    for i, r in scen.iterrows():
        ax3.text(i, r["implied_price"], f"${r['implied_price']:,.0f}", ha="center", va="bottom", fontsize=8)
    ax3.set_ylabel("Implied price ($)")
    st.pyplot(fig3)

st.subheader("WACC × terminal-growth sensitivity")
grid = enh.sensitivity_grid_2d(base, drv)
st.dataframe(grid.style.format("${:,.0f}", na_rep="—"), use_container_width=True)

with st.expander("Forecast detail (first years)"):
    st.dataframe(forecast[["revenue", "growth", "ebitda", "ebit_margin", "ufcf", "shares", "eps"]]
                 .round(2), use_container_width=True)

st.caption("Educational tool, not investment advice. Regenerate inputs via "
           "AGENT_DATA_COLLECTION_PROMPT.md before relying on any figure.")
