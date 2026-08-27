"""
data_loader.py
==============
Reads the standardized CSV inputs (produced by hand or by an AI research agent
following AGENT_DATA_COLLECTION_PROMPT.md) and builds the Python objects the
rest of the toolkit uses.

Company-agnostic and DEGRADES GRACEFULLY for thinly-covered / small-cap names:
  - required : company_inputs.csv   (long: field,value,unit,notes)
  - required : financial_history.csv (>=1 fiscal year row)
  - optional : analyst_consensus.csv (long: field,value)   -> has_consensus flag
  - optional : analyst_bank_targets.csv                     -> may be empty
  - optional : multiples_comps.csv   (subject + peers)      -> has_comps flag
  - optional : segments.csv          (segment revenue build)-> sum-of-parts
  - optional : valuation_history.csv (historical own multiples)

If a file is missing/empty the corresponding layer is simply skipped downstream,
so a company with NO analyst coverage and NO clean peer set still runs the DCF,
reverse-solve, Monte Carlo, sensitivity, scenarios and backtest.
"""

import os
import pandas as pd

from model import BaseYearActuals, ForecastDrivers, Segment
from analyst_data import AnalystConsensus, BankTarget

DEFAULTS = dict(
    ar_days=60.0,
    inventory_days=95.0,
    ap_days=70.0,
    interest_rate_on_debt=0.045,
    interest_yield_on_cash=0.035,
)


def _read_long_csv(path: str) -> dict:
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        val = row["value"]
        out[str(row["field"]).strip()] = None if pd.isna(val) else val
    return out


def _to_float(x, default=None):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return default


def _exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# 1. Base year actuals + starting WACC
# ---------------------------------------------------------------------------

def load_base_year_actuals(data_dir: str) -> BaseYearActuals:
    raw = _read_long_csv(os.path.join(data_dir, "company_inputs.csv"))

    rev = _to_float(raw.get("ttm_revenue"))
    gm = _to_float(raw.get("ttm_gross_margin"))
    ebitda = _to_float(raw.get("ttm_ebitda"))
    ebit = _to_float(raw.get("ttm_ebit"))

    # opex% = gross profit - EBITDA, over revenue (fall back to explicit if given)
    opex_pct = _to_float(raw.get("ttm_opex_pct_revenue"))
    if opex_pct is None and None not in (gm, ebitda, rev) and rev:
        opex_pct = gm - ebitda / rev
    if opex_pct is None:
        opex_pct = 0.20

    da_pct = _to_float(raw.get("ttm_da_pct_revenue"))
    if da_pct is None and None not in (ebitda, ebit, rev) and rev:
        da_pct = (ebitda - ebit) / rev
    if da_pct is None:
        da_pct = 0.03

    capex = _to_float(raw.get("ttm_capex"), 0.0)

    return BaseYearActuals(
        revenue=rev,
        gross_margin=gm,
        opex_pct_revenue=opex_pct,
        da_pct_revenue=da_pct,
        tax_rate=_to_float(raw.get("tax_rate_effective"), 0.21),
        capex_pct_revenue=(capex / rev) if rev else 0.05,
        ar_days=_to_float(raw.get("ar_days"), DEFAULTS["ar_days"]),
        inventory_days=_to_float(raw.get("inventory_days"), DEFAULTS["inventory_days"]),
        ap_days=_to_float(raw.get("ap_days"), DEFAULTS["ap_days"]),
        other_nwc_pct_revenue=_to_float(raw.get("other_nwc_pct_revenue"), 0.03),
        cash_and_investments=_to_float(raw.get("cash_and_investments"), 0.0),
        total_debt=_to_float(raw.get("total_debt"), 0.0),
        interest_rate_on_debt=_to_float(raw.get("interest_rate_on_debt"), DEFAULTS["interest_rate_on_debt"]),
        interest_yield_on_cash=_to_float(raw.get("interest_yield_on_cash"), DEFAULTS["interest_yield_on_cash"]),
        ppe_net=_to_float(raw.get("ppe_net"), 0.0),
        shares_diluted=_to_float(raw.get("shares_diluted")),
        current_price=_to_float(raw.get("current_price")),
    )


def load_company_meta(data_dir: str) -> dict:
    raw = _read_long_csv(os.path.join(data_dir, "company_inputs.csv"))
    return dict(
        ticker=str(raw.get("ticker") or "SUBJECT"),
        company_name=str(raw.get("company_name") or raw.get("ticker") or "Subject Company"),
        as_of_date=str(raw.get("as_of_date") or ""),
        ttm_eps_diluted=_to_float(raw.get("ttm_eps_diluted")),
        ttm_ebitda=_to_float(raw.get("ttm_ebitda")),
        ttm_revenue=_to_float(raw.get("ttm_revenue")),
        ttm_net_income=_to_float(raw.get("ttm_net_income")),
        beta=_to_float(raw.get("beta")),
        risk_free_rate=_to_float(raw.get("risk_free_rate")),
        equity_risk_premium=_to_float(raw.get("equity_risk_premium")),
    )


def load_starting_wacc(data_dir: str) -> float:
    raw = _read_long_csv(os.path.join(data_dir, "company_inputs.csv"))
    wacc = _to_float(raw.get("wacc_estimate"))
    if wacc is not None:
        return wacc
    beta = _to_float(raw.get("beta"), 1.1)
    rf = _to_float(raw.get("risk_free_rate"), 0.04)
    erp = _to_float(raw.get("equity_risk_premium"), 0.05)
    return rf + beta * erp


def load_share_dynamics(data_dir: str) -> dict:
    raw = _read_long_csv(os.path.join(data_dir, "company_inputs.csv"))
    return dict(
        sbc_dilution_pct=_to_float(raw.get("sbc_dilution_pct"), 0.0),
        buyback_pct=_to_float(raw.get("buyback_pct"), 0.0),
        dividend_payout_pct_ni=_to_float(raw.get("dividend_payout_pct_ni"), 0.0),
    )


# ---------------------------------------------------------------------------
# 2. Financial history
# ---------------------------------------------------------------------------

def load_financial_history(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "financial_history.csv")
    df = pd.read_csv(path)
    df["revenue_growth"] = df["revenue"].pct_change()
    df["gross_profit"] = df["revenue"] * df["gross_margin"]
    df["ebit_margin"] = df["ebit"] / df["revenue"]
    df["net_margin"] = df["net_income"] / df["revenue"]
    return df


# ---------------------------------------------------------------------------
# 3. Analyst consensus + bank targets (OPTIONAL)
# ---------------------------------------------------------------------------

def load_full_consensus(data_dir: str):
    """Returns (AnalystConsensus | None, has_consensus: bool)."""
    path = os.path.join(data_dir, "analyst_consensus.csv")
    if not _exists(path):
        return None, False
    raw = _read_long_csv(path)
    avg = _to_float(raw.get("avg_price_target"))
    if avg is None:  # no usable coverage
        return None, False

    consensus = AnalystConsensus(
        as_of=str(raw.get("as_of_date") or ""),
        num_analysts=int(_to_float(raw.get("num_analysts"), 0) or 0),
        avg_target=avg,
        median_target=_to_float(raw.get("median_price_target"), avg),
        low_target=_to_float(raw.get("low_price_target"), avg),
        high_target=_to_float(raw.get("high_price_target"), avg),
        rating=str(raw.get("consensus_rating") or "n/a"),
        fy_next_label=str(raw.get("next_fy_label") or "next FY"),
        fy_next_revenue_estimate=_to_float(raw.get("next_fy_revenue_estimate")),
        fy_next_eps_estimate=_to_float(raw.get("next_fy_eps_estimate")),
        fy_next2_label=str(raw.get("next_fy2_label") or "next FY+1"),
        fy_next2_revenue_estimate=_to_float(raw.get("next_fy2_revenue_estimate")),
        fy_next2_eps_estimate=_to_float(raw.get("next_fy2_eps_estimate")),
        bank_targets=[],
    )
    # bank targets (optional)
    bt_path = os.path.join(data_dir, "analyst_bank_targets.csv")
    if _exists(bt_path):
        df = pd.read_csv(bt_path)
        for _, row in df.iterrows():
            pt = _to_float(row.get("price_target"))
            if pt is None:
                continue
            consensus.bank_targets.append(BankTarget(
                bank=str(row.get("bank", "")), analyst=str(row.get("analyst", "")),
                target=pt, rating=str(row.get("rating", "")), date=str(row.get("date", "")),
            ))
    return consensus, True


# ---------------------------------------------------------------------------
# 4. Trading comps / multiples (OPTIONAL)
# ---------------------------------------------------------------------------

def load_multiples_comps(data_dir: str):
    """Returns (DataFrame | None, has_comps: bool). has_comps requires a subject
    row plus at least one peer with a usable forward multiple."""
    path = os.path.join(data_dir, "multiples_comps.csv")
    if not _exists(path):
        return None, False
    df = pd.read_csv(path)
    df["is_subject"] = df["is_subject"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    peers = df[~df["is_subject"]]
    usable = peers[["pe_fwd", "ev_ebitda_fwd", "ev_revenue_fwd"]].notna().any(axis=1).sum()
    has = bool(df["is_subject"].any()) and usable >= 1
    return df, has


# ---------------------------------------------------------------------------
# 5. Segments (OPTIONAL) — enables sum-of-the-parts revenue build
# ---------------------------------------------------------------------------

def load_segments(data_dir: str):
    path = os.path.join(data_dir, "segments.csv")
    if not _exists(path):
        return None
    df = pd.read_csv(path)
    segs = []
    for _, r in df.iterrows():
        base_rev = _to_float(r.get("base_revenue"))
        if base_rev is None:
            continue
        segs.append(Segment(
            name=str(r.get("segment", "seg")),
            base_revenue=base_rev,
            year1_growth=_to_float(r.get("year1_growth"), 0.05),
            terminal_growth=_to_float(r.get("terminal_growth"), 0.03),
            gross_margin=_to_float(r.get("gross_margin")),
        ))
    return segs or None


# ---------------------------------------------------------------------------
# 6. Valuation history (OPTIONAL) — the company's OWN historical multiples
# ---------------------------------------------------------------------------

def load_valuation_history(data_dir: str):
    path = os.path.join(data_dir, "valuation_history.csv")
    if not _exists(path):
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Convenience: load everything at once
# ---------------------------------------------------------------------------

def load_all(data_dir: str = "data"):
    base = load_base_year_actuals(data_dir)
    meta = load_company_meta(data_dir)
    wacc = load_starting_wacc(data_dir)
    sd = load_share_dynamics(data_dir)
    segments = load_segments(data_dir)

    drv = ForecastDrivers(wacc=wacc)
    # seed driver defaults from the base year so a bare CSV still forecasts sensibly
    drv.terminal_gross_margin = base.gross_margin
    drv.opex_pct_revenue = base.opex_pct_revenue
    drv.da_pct_revenue = base.da_pct_revenue
    drv.capex_pct_revenue = base.capex_pct_revenue
    drv.terminal_capex_pct_revenue = max(0.02, base.capex_pct_revenue * 0.85)
    drv.tax_rate = base.tax_rate
    drv.ar_days, drv.inventory_days, drv.ap_days = base.ar_days, base.inventory_days, base.ap_days
    drv.other_nwc_pct_revenue = base.other_nwc_pct_revenue
    drv.sbc_dilution_pct = sd["sbc_dilution_pct"]
    drv.buyback_pct = sd["buyback_pct"]
    drv.dividend_payout_pct_ni = sd["dividend_payout_pct_ni"]
    drv.segments = segments

    # Generic, conservative Year-1 growth seed for the OWN base case: average of
    # the last up to 3 years of revenue growth, then clipped to [2%, 35%]. This
    # keeps hyper-growers from producing an absurd base case (the reverse-DCF's
    # job is to reveal what's priced in, so the base case stays deliberately
    # sober) while faithfully tracking normal/mature companies' real trend.
    history = load_financial_history(data_dir)
    recent = history["revenue_growth"].dropna().tail(3)
    if len(recent):
        seed_g = float(recent.mean())
        drv.year1_growth = float(min(max(seed_g, 0.02), 0.35))
    else:
        drv.year1_growth = 0.08
    drv.terminal_growth = 0.03

    consensus, has_consensus = load_full_consensus(data_dir)
    comps, has_comps = load_multiples_comps(data_dir)
    val_hist = load_valuation_history(data_dir)

    return dict(
        base=base, meta=meta, drivers=drv, history=history,
        consensus=consensus, has_consensus=has_consensus,
        comps=comps, has_comps=has_comps,
        segments=segments, valuation_history=val_hist,
    )
