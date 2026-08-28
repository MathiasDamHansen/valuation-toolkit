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
"""

import copy
import os
import re
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
# Number formatting helpers
# --------------------------------------------------------------------------
def _num(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def f_money(v, dp=0):
    v = _num(v)
    return "—" if v is None else f"${v:,.{dp}f}"


def f_mm(v):
    """Money in $mm with thousands separators."""
    v = _num(v)
    return "—" if v is None else f"{v:,.0f}"


def f_pct(v, dp=1):
    v = _num(v)
    return "—" if v is None else f"{v:.{dp}%}"


def f_eps(v):
    v = _num(v)
    return "—" if v is None else f"{v:,.2f}"


def f_mult(v):
    v = _num(v)
    return "—" if v is None else f"{v:.1f}x"


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
history = data.get("history")
valuation_history = data.get("valuation_history")

# --------------------------------------------------------------------------
# 5-year historical averages (from financial_history.csv) to guide the sliders
# --------------------------------------------------------------------------
def _avg5(col):
    if history is None or col not in getattr(history, "columns", []):
        return None
    s = history[col].dropna().tail(5)
    return float(s.mean()) if len(s) else None


avg_rev_growth_5y = _avg5("revenue_growth")
avg_gross_margin_5y = _avg5("gross_margin")


def _avg_caption(value, what):
    if value is None:
        return f"📊 5-yr avg {what}: n/a (insufficient history)"
    return f"📊 5-yr avg {what}: {value:.1%}"


st.sidebar.markdown("### Assumptions")
n_trials = st.sidebar.select_slider("Monte Carlo trials", [1000, 2000, 5000, 10000, 20000], value=5000)

y1 = st.sidebar.slider("Year-1 revenue growth", -0.20, 1.50, float(round(drv0.year1_growth, 3)), 0.01)
st.sidebar.caption(_avg_caption(avg_rev_growth_5y, "revenue growth"))

tg = st.sidebar.slider("Terminal growth", 0.00, 0.08, float(round(drv0.terminal_growth, 3)), 0.005)
st.sidebar.caption(_avg_caption(avg_rev_growth_5y, "revenue growth (long-run reference)"))

gm = st.sidebar.slider("Terminal gross margin", 0.05, 0.95, float(round(drv0.terminal_gross_margin, 3)), 0.01)
st.sidebar.caption(_avg_caption(avg_gross_margin_5y, "gross margin"))

wacc = st.sidebar.slider("WACC", 0.04, 0.20, float(round(drv0.wacc, 3)), 0.005)
exit_w = st.sidebar.slider("Exit-multiple weight in TV", 0.0, 1.0, float(drv0.exit_multiple_weight), 0.05)
sbc = st.sidebar.slider("SBC dilution % / yr", 0.0, 0.05, float(drv0.sbc_dilution_pct), 0.005)
bb = st.sidebar.slider("Buyback % / yr", 0.0, 0.05, float(drv0.buyback_pct), 0.005)

drv = copy.deepcopy(drv0)
drv.year1_growth, drv.terminal_growth, drv.terminal_gross_margin = y1, tg, gm
drv.wacc, drv.exit_multiple_weight = wacc, exit_w
drv.sbc_dilution_pct, drv.buyback_pct = sbc, bb
if drv.custom_growth_path is not None:
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

try:
    imp_tg = m.solve_for_terminal_growth(base.current_price, wacc=drv.wacc)
except ValueError:
    imp_tg = None
try:
    imp_wacc = m.solve_for_wacc(base.current_price, terminal_growth=drv.terminal_growth)
except ValueError:
    imp_wacc = None

# --------------------------------------------------------------------------
# Extra valuations for the KPI row
# --------------------------------------------------------------------------
# Analyst valuation = Street MEDIAN 12-month price target (if coverage exists)
analyst_val = None
if has_consensus and data.get("consensus") is not None:
    try:
        analyst_val = float(data["consensus"].median_target)
    except Exception:
        analyst_val = None

# Multiples-based valuation = peer forward P/E × the company's NTM EPS.
multiples_val = None
if has_comps:
    try:
        _imp = mv.implied_prices_from_multiples(base, data.get("consensus"), data["comps"])
        _pe = _imp[_imp["method"].astype(str).str.startswith("P/E")]
        if len(_pe):
            multiples_val = float(_pe["implied_price"].iloc[0])
    except Exception:
        multiples_val = None

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.title(f"{meta['company_name']} ({meta['ticker']})")
st.caption(f"As of {meta['as_of_date']}")


def _delta(v):
    if v is None or base.current_price in (None, 0):
        return None
    return f"{v / base.current_price - 1:+.0%}"


k = st.columns(7)
k[0].metric("Current price", f"${base.current_price:,.2f}")
k[1].metric("Base-case DCF", f"${dcf_price:,.2f}", _delta(dcf_price))
k[2].metric("Multiples valuation (P/E)",
            f"${multiples_val:,.2f}" if multiples_val is not None else "n/a", _delta(multiples_val))
k[3].metric("Analyst valuation (median)",
            f"${analyst_val:,.2f}" if analyst_val is not None else "n/a", _delta(analyst_val))
k[4].metric("MC percentile of price", f"{pct_below:.0f}th")
k[5].metric("Implied terminal growth", f"{imp_tg:.2%}" if imp_tg is not None else "n/a")
k[6].metric("Implied WACC", f"{imp_wacc:.2%}" if imp_wacc is not None else "n/a")

# --------------------------------------------------------------------------
# "What must hold true for today's price to be fair?"
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
        rr = ranges[ranges["kind"] == "pct"]
        if len(rr):
            figc, axc = plt.subplots(figsize=(6.4, 0.6 * len(rr) + 1.2))
            for i, (_, r) in enumerate(rr.iterrows()):
                axc.plot([r["_f10"], r["_f90"]], [i, i], color="#c9ced6", lw=7, solid_capstyle="round")
                axc.plot([r["_c10"], r["_c90"]], [i, i], color=BRAND, lw=7, solid_capstyle="round")
                axc.plot(r["_c50"], i, "o", color=INK, ms=5)
            axc.set_yticks(np.arange(len(rr))); axc.set_yticklabels(rr["driver"], fontsize=8)
            axc.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            axc.set_title("Blue = must-hold range · grey = full tested range", fontsize=9)
            axc.grid(axis="x", alpha=0.3)
            st.pyplot(figc)

with mc_right:
    st.markdown("**Multiples the current price implies**")
    im = mcx.implied_multiples_at_price(base, forecast, data["comps"] if has_comps else None)
    mult_rows = []
    for kkey in ["Fwd EV/EBITDA", "Fwd P/E", "Fwd EV/Revenue"]:
        iv = im["implied"].get(kkey); pv = im["peer"].get(kkey)
        mult_rows.append({"Multiple": kkey,
                          "At today's price": (f"{iv:.1f}x" if iv == iv else "—"),
                          "Peer median": (f"{pv:.1f}x" if (pv is not None and pv == pv) else "—")})
    st.dataframe(pd.DataFrame(mult_rows), use_container_width=True, hide_index=True)
    if has_comps:
        st.caption("If 'at today's price' sits below the peer median, the market is paying "
                   "less than peers for the same forward metric (a relative discount), and vice-versa.")
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
# Tornado + scenarios + 2D grid
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

# ==========================================================================
# BOTTOM SECTION — 5 years actuals + 5 years forecast, with cash flow,
# margins, ROIC, ROE and FCF yield.
# ==========================================================================
st.divider()
st.header("📅 Financials: 5-year actuals + 5-year forecast")

_TAX = drv.tax_rate
_MKTCAP = (base.current_price or 0) * (base.shares_diluted or 0)


def _roic(ebit, debt, equity, cash, tax):
    ebit, debt, equity, cash = _num(ebit), _num(debt), _num(equity), _num(cash)
    if None in (ebit, debt, equity, cash):
        return None
    den = debt + equity - cash
    return None if den == 0 else ebit * (1 - tax) / den


def _roe(ni, equity):
    ni, equity = _num(ni), _num(equity)
    return None if None in (ni, equity) or equity == 0 else ni / equity


def _fcf_yield(fcf):
    fcf = _num(fcf)
    return None if fcf is None or _MKTCAP == 0 else fcf / _MKTCAP


def _forecast_year_labels(last_label, n):
    yr = None
    if last_label:
        mo = re.search(r"(\d{4})", str(last_label))
        if mo:
            yr = int(mo.group(1))
    return [f"FY{yr + i}" if yr else f"Year +{i}" for i in range(1, n + 1)]


def build_table(hist, fcast, base, drv, tax, n_actual=5, n_forecast=5):
    recs = []
    last_equity, last_label = None, None

    # ---- Actuals (last n_actual years) ----
    if hist is not None and len(hist):
        h_tail = hist.tail(n_actual)
        for _, h in h_tail.iterrows():
            rev = _num(h.get("revenue")); ebit = _num(h.get("ebit")); ebitda = _num(h.get("ebitda"))
            ni = _num(h.get("net_income")); capex = _num(h.get("capex"))
            equity = _num(h.get("total_equity"))
            da = (ebitda - ebit) if (ebitda is not None and ebit is not None) else None
            fcf = (ni + da - capex) if (ni is not None and da is not None and capex is not None) else None
            recs.append(dict(
                Period=str(h.get("fiscal_year", "")), Type="Actual",
                Revenue=rev, RevGrowth=_num(h.get("revenue_growth")),
                GrossMargin=_num(h.get("gross_margin")),
                EBITDA=ebitda, EBITDAMargin=(ebitda / rev if ebitda is not None and rev else None),
                EBIT=ebit, EBITMargin=(ebit / rev if ebit is not None and rev else None),
                NetProfit=ni, NetMargin=(ni / rev if ni is not None and rev else None),
                FCF=fcf, FCFYield=_fcf_yield(fcf),
                ROIC=_roic(ebit, h.get("total_debt"), equity, h.get("cash_and_investments"), tax),
                ROE=_roe(ni, equity)))
            if equity is not None:
                last_equity = equity
            last_label = h.get("fiscal_year")

    # ---- Forecast (first n_forecast model years) ----
    labels = _forecast_year_labels(last_label, n_forecast)
    prev_shares, equity = base.shares_diluted, last_equity
    f_head = fcast.head(n_forecast)
    for (yr, f), lab in zip(f_head.iterrows(), labels):
        rev = _num(f["revenue"]); ebit = _num(f["ebit"]); ebitda = _num(f["ebitda"])
        ni = _num(f["net_income"]); capex = _num(f["capex"])
        da = _num(f["da"]) if "da" in f.index else ((ebitda - ebit) if (ebitda is not None and ebit is not None) else None)
        fcf = (ni + da - capex) if (ni is not None and da is not None and capex is not None) else None
        # roll equity forward via retained earnings (net of dividends & buybacks)
        if equity is not None and ni is not None:
            dividends = ni * drv.dividend_payout_pct_ni
            buyback_cash = (prev_shares or 0) * drv.buyback_pct * (base.current_price or 0)
            equity = equity + ni - dividends - buyback_cash
        recs.append(dict(
            Period=lab, Type="Forecast",
            Revenue=rev, RevGrowth=_num(f["growth"]),
            GrossMargin=_num(f["gross_margin"]),
            EBITDA=ebitda, EBITDAMargin=(ebitda / rev if ebitda is not None and rev else None),
            EBIT=ebit, EBITMargin=_num(f["ebit_margin"]),
            NetProfit=ni, NetMargin=(ni / rev if ni is not None and rev else None),
            FCF=fcf, FCFYield=_fcf_yield(fcf),
            ROIC=(_roic(ebit, f["debt"], equity, f["cash"], tax) if equity is not None else None),
            ROE=(_roe(ni, equity) if equity is not None else None)))
        prev_shares = _num(f["shares"])
    return pd.DataFrame(recs)


tbl = build_table(history, forecast, base, drv, _TAX, n_actual=5, n_forecast=5)

st.caption(f"Actuals from financial_history.csv · forecast from the model. "
           f"FCF ≈ net income + D&A − capex. FCF yield = FCF ÷ current market cap "
           f"(${_MKTCAP:,.0f}mm). ROIC = NOPAT / (debt + equity − cash), NOPAT = EBIT × (1 − {_TAX:.0%}). "
           f"ROE = net income ÷ book equity. Forecast equity is rolled forward from the last "
           f"actual, so forecast ROIC/ROE are estimates. Blanks = data not available.")

disp = pd.DataFrame({
    "Period": tbl["Period"],
    "Type": tbl["Type"],
    "Revenue ($mm)": tbl["Revenue"].map(f_mm),
    "Rev growth": tbl["RevGrowth"].map(f_pct),
    "Gross margin": tbl["GrossMargin"].map(f_pct),
    "EBITDA ($mm)": tbl["EBITDA"].map(f_mm),
    "EBITDA margin": tbl["EBITDAMargin"].map(f_pct),
    "EBIT ($mm)": tbl["EBIT"].map(f_mm),
    "EBIT margin": tbl["EBITMargin"].map(f_pct),
    "Net profit ($mm)": tbl["NetProfit"].map(f_mm),
    "Net margin": tbl["NetMargin"].map(f_pct),
    "FCF ($mm)": tbl["FCF"].map(f_mm),
    "FCF yield": tbl["FCFYield"].map(f_pct),
    "ROIC": tbl["ROIC"].map(f_pct),
    "ROE": tbl["ROE"].map(f_pct),
})
st.dataframe(disp, use_container_width=True, hide_index=True, height=430)
st.download_button("⬇️ Download this table (CSV)",
                   data=tbl.to_csv(index=False).encode("utf-8"),
                   file_name=f"{meta['ticker']}_actuals_forecast.csv", mime="text/csv")

# Returns-through-time chart: ROIC vs ROE, WACC reference line
_ret = tbl.dropna(subset=["ROIC"])
if len(_ret) >= 2:
    figr, axr = plt.subplots(figsize=(10, 3.4))
    x = np.arange(len(_ret))
    axr.bar(x - 0.2, _ret["ROIC"], width=0.4, label="ROIC",
            color=[POS if t == "Actual" else BRAND for t in _ret["Type"]])
    if _ret["ROE"].notna().any():
        axr.bar(x + 0.2, _ret["ROE"], width=0.4, label="ROE", color=BRAND_ALT, alpha=0.85)
    axr.axhline(drv.wacc, color=NEG, ls="--", lw=1.5, label=f"WACC {drv.wacc:.1%}")
    axr.set_xticks(x); axr.set_xticklabels(_ret["Period"], rotation=45, fontsize=8)
    axr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axr.set_title("Returns over time — ROIC & ROE vs WACC")
    axr.legend(fontsize=8)
    st.pyplot(figr)

# --------------------------------------------------------------------------
# Historical valuation multiples (actuals)
# --------------------------------------------------------------------------
st.subheader("Historical valuation multiples (actuals)")
if valuation_history is None or len(valuation_history) == 0:
    st.info("No valuation_history.csv found for this company, so historical actual multiples "
            "can't be shown. Add one with columns: date, price, pe, ev_ebitda, ev_revenue.")
else:
    vh = valuation_history.copy()
    hrows = []
    for _, r in vh.iterrows():
        hrows.append({
            "Period": str(r.get("date", "")),
            "Price": f_money(r.get("price"), 2),
            "P/E": f_mult(r.get("pe")),
            "EV/EBITDA": f_mult(r.get("ev_ebitda")),
            "EV/Revenue": f_mult(r.get("ev_revenue")),
        })
    # Current (fwd) row implied by today's price vs the model's NTM estimates
    try:
        f1 = forecast.iloc[0]
        ev_now = base.current_price * base.shares_diluted + base.net_debt
        ntm_ebitda = _num(f1["ebitda"]); ntm_rev = _num(f1["revenue"])
        ntm_eps = (_num(f1["net_income"]) / base.shares_diluted) if base.shares_diluted else None
        hrows.append({
            "Period": "Current (fwd)",
            "Price": f_money(base.current_price, 2),
            "P/E": (f"{base.current_price / ntm_eps:.1f}x" if ntm_eps else "—"),
            "EV/EBITDA": (f"{ev_now / ntm_ebitda:.1f}x" if ntm_ebitda else "—"),
            "EV/Revenue": (f"{ev_now / ntm_rev:.1f}x" if ntm_rev else "—"),
        })
    except Exception:
        pass
    st.dataframe(pd.DataFrame(hrows), use_container_width=True, hide_index=True)
    plot_cols = [c for c in ["pe", "ev_ebitda", "ev_revenue"] if c in vh.columns and vh[c].notna().any()]
    if plot_cols and "date" in vh.columns:
        figm, axm = plt.subplots(figsize=(9, 3.4))
        lbl = {"pe": "P/E", "ev_ebitda": "EV/EBITDA", "ev_revenue": "EV/Revenue"}
        pal = {"pe": BRAND, "ev_ebitda": BRAND_ALT, "ev_revenue": POS}
        for c in plot_cols:
            axm.plot(vh["date"].astype(str), vh[c], marker="o", label=lbl[c], color=pal[c])
        axm.set_ylabel("Multiple (x)"); axm.legend(fontsize=8); axm.grid(alpha=0.3)
        axm.set_title("Historical trading multiples")
        st.pyplot(figm)

st.caption("Educational tool, not investment advice. Regenerate inputs via "
           "AGENT_DATA_COLLECTION_PROMPT.md before relying on any figure.")
