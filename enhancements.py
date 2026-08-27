"""
enhancements.py
===============
The analytical add-ons requested on top of the base DCF/consensus/multiples
engine. Every function is company-agnostic and degrades gracefully when the
relevant optional data is missing (small / uncovered names).

Contents
--------
1. historical_own_multiples()      -> the company's OWN valuation range over time
2. tornado_sensitivity()           -> ranked driver impact (banker tornado chart)
3. sensitivity_grid_2d()           -> deterministic WACC x terminal-growth price grid
4. backtest_calibration()          -> feed a prior year in, compare to actuals
5. scenario_table()                -> discrete bear / base / bull cases
6. reconcile_fv_to_target()        -> today's fair value vs a 12-month target

Each has a paired plot_* helper that writes a PNG and returns its path.
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import ThreeStatementModel, ForecastDrivers, BaseYearActuals
from settings import BRAND, BRAND_ALT, POS, NEG, INK, output_path, OUTPUT_DIR


# ===========================================================================
# 1. Historical OWN multiple range
# ===========================================================================

def historical_own_multiples(meta: dict, history: pd.DataFrame,
                             valuation_history: pd.DataFrame = None,
                             current_comps_row: pd.Series = None) -> pd.DataFrame:
    """Return a tidy frame of the company's own historical P/E and EV/EBITDA.

    Preferred source: an explicit valuation_history.csv (date, price, pe, ev_ebitda).
    Fallback: derive a rough P/E series from financial_history.csv using
    eps_diluted and a supplied historical price if present; EV/EBITDA is left
    blank when we lack a historical EV series (we don't fabricate it).
    """
    if valuation_history is not None and len(valuation_history):
        vh = valuation_history.copy()
        for c in ["pe", "ev_ebitda"]:
            if c not in vh.columns:
                vh[c] = np.nan
        return vh[["date", "pe", "ev_ebitda"]]

    # Fallback: P/E from history if a price column exists
    rows = []
    if "price" in history.columns:
        for _, r in history.iterrows():
            eps = r.get("eps_diluted")
            price = r.get("price")
            pe = (price / eps) if (pd.notna(eps) and eps not in (0, None) and pd.notna(price)) else np.nan
            rows.append(dict(date=r.get("fiscal_year"), pe=pe, ev_ebitda=np.nan))
    return pd.DataFrame(rows)


def plot_historical_own_multiples(hist_mult: pd.DataFrame, meta: dict,
                                  current_comps_row: pd.Series = None,
                                  ticker="subject", outdir=OUTPUT_DIR):
    if hist_mult is None or hist_mult.empty or hist_mult[["pe", "ev_ebitda"]].isna().all().all():
        return None
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(hist_mult)))
    labels = [str(d) for d in hist_mult["date"]]
    plotted = False
    if hist_mult["pe"].notna().any():
        ax.plot(x, hist_mult["pe"], marker="o", color=BRAND, label="P/E (historical)")
        plotted = True
    if hist_mult["ev_ebitda"].notna().any():
        ax.plot(x, hist_mult["ev_ebitda"], marker="s", color=BRAND_ALT, label="EV/EBITDA (historical)")
        plotted = True
    # current multiples as reference lines
    if current_comps_row is not None:
        if pd.notna(current_comps_row.get("pe_ttm")):
            ax.axhline(current_comps_row["pe_ttm"], color=BRAND, ls="--", alpha=0.6,
                       label=f"P/E now ({current_comps_row['pe_ttm']:.1f}x)")
        if pd.notna(current_comps_row.get("ev_ebitda_ttm")):
            ax.axhline(current_comps_row["ev_ebitda_ttm"], color=BRAND_ALT, ls="--", alpha=0.6,
                       label=f"EV/EBITDA now ({current_comps_row['ev_ebitda_ttm']:.1f}x)")
    if not plotted:
        plt.close(); return None
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Multiple (x)")
    ax.set_title(f"{meta.get('ticker','')}: Own Valuation Multiple vs. History")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    path = output_path(ticker, "historical_own_multiples")
    plt.savefig(path, dpi=150); plt.close()
    return path


# ===========================================================================
# 2. Tornado sensitivity
# ===========================================================================

TORNADO_DRIVERS = [
    ("year1_growth", "Year-1 growth", 0.6),      # +/- 60% relative
    ("terminal_growth", "Terminal growth", None),  # absolute +/-1.0pp handled below
    ("terminal_gross_margin", "Terminal gross margin", None),  # +/-3pp
    ("wacc", "WACC", None),                        # +/-1.5pp
    ("terminal_capex_pct_revenue", "Terminal capex %", 0.4),
    ("opex_pct_revenue", "Opex % of revenue", 0.25),
]


def tornado_sensitivity(base: BaseYearActuals, drv: ForecastDrivers) -> pd.DataFrame:
    """Vary each driver low/high (holding others at base) and record the implied
    price at each end. Returns a frame sorted by absolute swing (widest first)."""
    base_price = ThreeStatementModel(base, drv).dcf_value()["price_per_share"]

    def price_with(**over):
        d = copy.deepcopy(drv)
        for k, v in over.items():
            setattr(d, k, v)
        try:
            return ThreeStatementModel(base, d).dcf_value()["price_per_share"]
        except ValueError:
            return np.nan

    rows = []
    for field, label, rel in TORNADO_DRIVERS:
        cur = getattr(drv, field, None)
        if cur is None:
            continue
        if field == "terminal_growth":
            lo_v, hi_v = max(0.0, cur - 0.01), min(drv.wacc - 0.005, cur + 0.01)
        elif field == "terminal_gross_margin":
            lo_v, hi_v = cur - 0.03, cur + 0.03
        elif field == "wacc":
            lo_v, hi_v = cur - 0.015, cur + 0.015
        else:
            lo_v, hi_v = cur * (1 - rel), cur * (1 + rel)

        # for WACC and capex, HIGH input -> LOWER price, so we still label by input value
        p_lo = price_with(**{field: lo_v})
        p_hi = price_with(**{field: hi_v})
        low_price, high_price = min(p_lo, p_hi), max(p_lo, p_hi)
        rows.append(dict(driver=label, field=field,
                         input_low=lo_v, input_high=hi_v,
                         price_low=low_price, price_high=high_price,
                         swing=high_price - low_price))
    df = pd.DataFrame(rows).sort_values("swing", ascending=True).reset_index(drop=True)
    df.attrs["base_price"] = base_price
    return df


def plot_tornado(tornado: pd.DataFrame, base: BaseYearActuals, ticker="subject", outdir=OUTPUT_DIR):
    if tornado is None or tornado.empty:
        return None
    base_price = tornado.attrs.get("base_price", base.current_price)
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(tornado) + 2))
    y = np.arange(len(tornado))
    for i, r in tornado.iterrows():
        ax.barh([i], [r["price_high"] - base_price], left=[base_price], color=POS, alpha=0.85)
        ax.barh([i], [r["price_low"] - base_price], left=[base_price], color=NEG, alpha=0.85)
    ax.axvline(base_price, color=INK, lw=1.5)
    ax.text(base_price, len(tornado) - 0.3, f" base ${base_price:,.0f}", color=INK, fontsize=8, va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(tornado["driver"], fontsize=9)
    ax.set_xlabel("Implied price ($/share)")
    ax.set_title("Tornado: Driver Sensitivity of DCF Price (widest at top)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = output_path(ticker, "tornado_sensitivity")
    plt.savefig(path, dpi=150); plt.close()
    return path


# ===========================================================================
# 3. Deterministic 2D sensitivity grid (WACC x terminal growth)
# ===========================================================================

def sensitivity_grid_2d(base: BaseYearActuals, drv: ForecastDrivers,
                        wacc_range=None, tg_range=None) -> pd.DataFrame:
    if wacc_range is None:
        wacc_range = np.round(np.linspace(drv.wacc - 0.02, drv.wacc + 0.02, 5), 4)
    if tg_range is None:
        tg_range = np.round(np.linspace(drv.terminal_growth - 0.01, drv.terminal_growth + 0.015, 6), 4)
    m = ThreeStatementModel(base, drv)
    m.run()
    grid = pd.DataFrame(index=[f"{w:.1%}" for w in wacc_range],
                        columns=[f"{g:.1%}" for g in tg_range], dtype=float)
    for w in wacc_range:
        for g in tg_range:
            if w <= g:
                grid.loc[f"{w:.1%}", f"{g:.1%}"] = np.nan
            else:
                grid.loc[f"{w:.1%}", f"{g:.1%}"] = m.dcf_value(wacc=w, terminal_growth=g)["price_per_share"]
    grid.index.name = "WACC \\ term. growth"
    return grid


def plot_sensitivity_grid(grid: pd.DataFrame, base: BaseYearActuals, ticker="subject", outdir=OUTPUT_DIR):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = grid.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(grid.columns))); ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index))); ax.set_yticklabels(grid.index)
    ax.set_xlabel("Terminal growth"); ax.set_ylabel("WACC")
    ax.set_title("DCF Price Sensitivity: WACC x Terminal Growth")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"${v:,.0f}", ha="center", va="center", fontsize=8, color=INK)
    fig.colorbar(im, ax=ax, label="Implied price ($)")
    plt.tight_layout()
    path = output_path(ticker, "sensitivity_grid")
    plt.savefig(path, dpi=150); plt.close()
    return path


# ===========================================================================
# 4. Backtest / calibration
# ===========================================================================

def backtest_calibration(base: BaseYearActuals, drv: ForecastDrivers,
                         history: pd.DataFrame) -> dict:
    """Sanity check on mechanics: take the SECOND-TO-LAST historical year as the
    starting point, forecast one year forward using that year's realized growth,
    and compare the model's Year-1 revenue/EBIT to the ACTUAL last historical
    year. This checks the engine reproduces reality when fed real inputs (it is
    not a market-price backtest)."""
    h = history.dropna(subset=["revenue"]).reset_index(drop=True)
    if len(h) < 2:
        return dict(available=False, reason="need >=2 historical years")

    prev, actual = h.iloc[-2], h.iloc[-1]
    realized_growth = actual["revenue"] / prev["revenue"] - 1

    # Build a base-year object from the PRIOR year
    prior_base = copy.deepcopy(base)
    prior_base.revenue = float(prev["revenue"])
    prior_base.gross_margin = float(prev["gross_margin"])
    prior_base.shares_diluted = float(prev.get("shares_diluted", base.shares_diluted))

    d = copy.deepcopy(drv)
    d.n_years = 1
    d.year1_growth = realized_growth
    d.terminal_growth = realized_growth  # single-year, no fade
    d.segments = None
    m = ThreeStatementModel(prior_base, d)
    res = m.run()

    model_rev = float(res["revenue"].iloc[0])
    model_ebit = float(res["ebit"].iloc[0])
    actual_rev = float(actual["revenue"])
    actual_ebit = float(actual["ebit"])

    return dict(
        available=True,
        prior_year=str(prev.get("fiscal_year", "prior")),
        target_year=str(actual.get("fiscal_year", "actual")),
        realized_growth=realized_growth,
        model_revenue=model_rev, actual_revenue=actual_rev,
        revenue_error_pct=model_rev / actual_rev - 1,
        model_ebit=model_ebit, actual_ebit=actual_ebit,
        ebit_error_pct=(model_ebit / actual_ebit - 1) if actual_ebit else np.nan,
    )


# ===========================================================================
# 5. Discrete scenario table (bear / base / bull)
# ===========================================================================

def scenario_table(base: BaseYearActuals, drv: ForecastDrivers) -> pd.DataFrame:
    """Three named, self-consistent assumption sets. Deltas are relative to the
    base drivers so this scales to any company."""
    def make(name, dg, dtg, dgm, dwacc):
        d = copy.deepcopy(drv)
        d.year1_growth = max(-0.5, drv.year1_growth + dg)
        d.terminal_growth = min(drv.wacc + dwacc - 0.005, max(0.0, drv.terminal_growth + dtg))
        d.terminal_gross_margin = min(0.95, max(0.05, drv.terminal_gross_margin + dgm))
        d.wacc = max(0.03, drv.wacc + dwacc)
        d.segments = None if name != "Base" else drv.segments
        val = ThreeStatementModel(base, d).dcf_value()
        return dict(scenario=name, year1_growth=d.year1_growth, terminal_growth=d.terminal_growth,
                    terminal_gross_margin=d.terminal_gross_margin, wacc=d.wacc,
                    implied_price=val["price_per_share"])

    rows = [
        make("Bear", dg=-0.15, dtg=-0.01, dgm=-0.05, dwacc=+0.015),
        make("Base", dg=0.0, dtg=0.0, dgm=0.0, dwacc=0.0),
        make("Bull", dg=+0.15, dtg=+0.01, dgm=+0.04, dwacc=-0.015),
    ]
    df = pd.DataFrame(rows)
    df["vs_current"] = df["implied_price"] / base.current_price - 1
    return df


def plot_scenarios(scen: pd.DataFrame, base: BaseYearActuals, ticker="subject", outdir=OUTPUT_DIR):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Bear": NEG, "Base": BRAND, "Bull": POS}
    ax.bar(scen["scenario"], scen["implied_price"],
           color=[colors.get(s, BRAND) for s in scen["scenario"]])
    ax.axhline(base.current_price, color=INK, ls="--", label=f"Current ${base.current_price:,.0f}")
    for i, r in scen.iterrows():
        ax.text(i, r["implied_price"], f"${r['implied_price']:,.0f}\n({r['vs_current']:+.0%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Implied price ($/share)")
    ax.set_title("Bear / Base / Bull DCF Scenarios")
    ax.legend()
    plt.tight_layout()
    path = output_path(ticker, "scenarios")
    plt.savefig(path, dpi=150); plt.close()
    return path


# ===========================================================================
# 6. Fair value today vs 12-month target reconciliation
# ===========================================================================

def reconcile_fv_to_target(fair_value_today: float, cost_of_equity: float,
                           street_target: float = None) -> dict:
    """A DCF gives a value TODAY; a Street 'price target' is a value ~12 months
    out. Rolling fair value forward one year at the cost of equity puts them on
    the same footing so the comparison is apples-to-apples."""
    fv_12m = fair_value_today * (1 + cost_of_equity)
    out = dict(fair_value_today=fair_value_today, cost_of_equity=cost_of_equity,
               fair_value_12m=fv_12m)
    if street_target is not None:
        out["street_target"] = street_target
        out["gap_pct"] = fv_12m / street_target - 1
    return out
