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

st.sidebar.markdown("### Assumptions")
n_trials = st.sidebar.select_slider("Monte Carlo trials", [1000, 2000, 5000, 10000, 20000], value=5000)
y1 = st.sidebar.slider("Year-1 revenue growth", -0.20, 1.50, float(round(drv0.year1_growth, 3)), 0.01)
tg = st.sidebar.slider("Terminal growth", 0.00, 0.08, float(round(drv0.terminal_growth, 3)), 0.005)
gm = st.sidebar.slider("Terminal gross margin", 0.05, 0.95, float(round(drv0.terminal_gross_margin, 3)), 0.01)
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

# ==========================================================================
# NEW SECTION — "What combinations of levers reproduce today's price?"
# Reverse Monte Carlo: keep only the simulated scenarios whose DCF value lands
# on the current market price, then show what growth / WACC / margin / capex
# (and, when peers exist, exit EV/EBITDA multiple) those scenarios require.
# Everything is driven straight from the Monte Carlo trials, and the resulting
# combinations are downloadable as CSV.
# ==========================================================================
st.divider()
st.subheader("🎯 What must be true for the DCF to equal today's price?")
st.caption("Of thousands of randomized lever combinations, these are the ones whose DCF value "
           "lands on the current share price — i.e. the assumption sets the market appears to be "
           "embedding. When peers exist, the exit EV/EBITDA multiple is randomized too, so "
           "multiples are factored in.")

band_pct = st.slider("Match tolerance around current price (±%)", 1.0, 10.0, 2.5, 0.5,
                     help="A scenario 'reproduces' the price if its DCF is within this band of "
                          "today's price. Widen it if too few scenarios match.") / 100.0

# Use a multiples-aware cloud when peers exist (adds exit EV/EBITDA as a lever);
# otherwise reuse the standard own-assumptions Monte Carlo already computed.
if has_comps:
    with st.spinner("Solving lever combinations (multiples-aware)..."):
        trials_solve = mv.run_exit_multiple_blend(
            base, drv, data["comps"], blend_weight=0.5,
            n_trials=max(4000, n_trials), seed=11)
else:
    trials_solve = trials_a

cur = base.current_price
# find the price-consistent subset, auto-widening the band until enough match
_b = band_pct
mask = trials_solve["implied_price"].between(cur * (1 - _b), cur * (1 + _b))
consistent = trials_solve[mask].copy()
while len(consistent) < 60 and _b < 0.15:
    _b += 0.01
    mask = trials_solve["implied_price"].between(cur * (1 - _b), cur * (1 + _b))
    consistent = trials_solve[mask].copy()

# levers to report (only those present as columns get shown)
_LEVERS = [
    ("year1_growth", "Year-1 revenue growth", "pct"),
    ("terminal_growth", "Terminal growth", "pct"),
    ("terminal_gross_margin", "Terminal gross margin", "pct"),
    ("wacc", "WACC (discount rate)", "pct"),
    ("terminal_capex_pct_revenue", "Terminal capex % of revenue", "pct"),
    ("exit_ev_ebitda_multiple", "Exit EV/EBITDA multiple", "mult"),
]


def _fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.1%}" if kind == "pct" else f"{v:.1f}x"


if len(consistent) == 0:
    # price sits outside the entire simulated cloud — informative in itself
    med_sim = float(trials_solve["implied_price"].median())
    p99_sim = float(np.percentile(trials_solve["implied_price"], 99))
    st.warning(
        f"**No lever combination reproduces ${cur:,.2f} within ±{_b:.0%}.** Even the most "
        f"favorable simulated scenario only reaches ${p99_sim:,.2f} (median ${med_sim:,.2f}). "
        f"The market is pricing in more than these lever ranges support — i.e. today's price "
        f"already embeds an out-of-range, very optimistic story.")
else:
    # ---- headline: median lever values across the price-consistent set ----
    def _med(col):
        return float(np.median(consistent[col])) if col in consistent.columns else None

    present = [(c, lbl, kind) for c, lbl, kind in _LEVERS if c in consistent.columns]
    mcols = st.columns(len(present))
    for (c, lbl, kind), col_ui in zip(present, mcols):
        short = lbl.split(" (")[0]
        col_ui.metric(f"Implied {short.lower()}", _fmt(_med(c), kind))

    # ---- range table (P5 / P25 / median / P75 / P95) ----
    range_rows = []
    for c, lbl, kind in present:
        p5, p25, p50, p75, p95 = np.percentile(consistent[c], [5, 25, 50, 75, 95])
        range_rows.append({
            "Lever": lbl,
            "Bearish (P5)": _fmt(p5, kind),
            "Low (P25)": _fmt(p25, kind),
            "Central (median)": _fmt(p50, kind),
            "High (P75)": _fmt(p75, kind),
            "Bullish (P95)": _fmt(p95, kind),
        })
    ranges_df = pd.DataFrame(range_rows)

    left_c, right_c = st.columns([1.05, 1])
    with left_c:
        st.markdown(f"**Lever ranges that reproduce ${cur:,.2f}**  \n"
                    f"<span style='color:#888'>{len(consistent):,} matching scenarios "
                    f"(±{_b:.0%} band, {len(consistent)/len(trials_solve):.0%} of the "
                    f"{len(trials_solve):,} simulated)</span>", unsafe_allow_html=True)
        st.dataframe(ranges_df, use_container_width=True, hide_index=True)

    with right_c:
        if {"wacc", "year1_growth"}.issubset(consistent.columns):
            st.markdown("**The trade-off frontier**")
            fig0, ax0 = plt.subplots(figsize=(6, 4.2))
            color_col = "terminal_gross_margin" if "terminal_gross_margin" in consistent.columns else None
            sc = ax0.scatter(consistent["wacc"], consistent["year1_growth"],
                             c=(consistent[color_col] if color_col else BRAND),
                             cmap="viridis" if color_col else None, s=16, alpha=0.75)
            ax0.set_xlabel("WACC"); ax0.set_ylabel("Year-1 revenue growth")
            ax0.set_title(f"Combinations that reprice to ${cur:,.0f}")
            ax0.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            if color_col:
                cb = fig0.colorbar(sc, ax=ax0); cb.set_label("Terminal gross margin")
                cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax0.grid(alpha=0.3)
            st.pyplot(fig0)
            st.caption("Each dot is one scenario whose DCF equals today's price. Higher growth only "
                       "reproduces the price alongside a higher discount rate — the market's implied trade-off.")

    # ---- sample of the actual combinations + CSV export ----
    show_cols = [c for c, _, _ in present] + ["implied_price"]
    preview = consistent[show_cols].copy().sort_values("implied_price").reset_index(drop=True)
    with st.expander(f"See the {len(consistent):,} price-consistent lever combinations (raw)"):
        st.dataframe(preview.round(4), use_container_width=True, height=320)

    st.download_button(
        "⬇️ Download price-consistent combinations (CSV)",
        data=preview.to_csv(index=False).encode("utf-8"),
        file_name=f"{meta['ticker']}_price_consistent_combinations.csv",
        mime="text/csv")

st.download_button(
    "⬇️ Download full Monte Carlo simulation (CSV)",
    data=trials_solve.to_csv(index=False).encode("utf-8"),
    file_name=f"{meta['ticker']}_monte_carlo_trials.csv",
    mime="text/csv")

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
