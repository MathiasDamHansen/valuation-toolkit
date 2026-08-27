"""
model.py
========
A linked 3-statement financial model (Income Statement, Balance Sheet,
Cash Flow Statement) that feeds a DCF / reverse-DCF valuation for ANY listed
company — mega-cap or micro-cap, covered or not.

DESIGN NOTES
------------
- Base year = trailing twelve months (TTM) as of the most recent quarter.
- Forecast horizon is explicit (default 10 years) + a terminal value.
- Revenue can be built two ways:
    (a) a single blended line whose growth fades linearly from a Year-1 rate
        down to the terminal rate (the standard mega-cap reverse-DCF pattern), or
    (b) a SUM-OF-SEGMENTS build, if a segments table is supplied — each segment
        fades from its own Year-1 growth to its own terminal growth, so a
        Data-Center-vs-Gaming style mix is captured instead of one blended rate.
- Working capital is modeled via day-based drivers (AR/inventory/AP days).
- SHARE COUNT is now dynamic: stock-based comp dilutes and buybacks shrink the
  count each year, compounding over the horizon (previously held flat).
- Unlevered Free Cash Flow (UFCF) is computed separately from levered net
  income, so the DCF is discounted at WACC (capital-structure neutral).

All $ figures are in millions unless noted. Shares are in millions.

NOTE: The dataclass defaults below are deliberately GENERIC placeholders so an
empty object still runs. In normal use every field is populated from the input
CSVs via data_loader.py, so these defaults never actually drive a real result.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. BASE-YEAR ACTUALS
# ---------------------------------------------------------------------------

@dataclass
class BaseYearActuals:
    revenue: float = 1_000.0             # TTM revenue, $mm
    gross_margin: float = 0.50           # TTM gross margin
    opex_pct_revenue: float = 0.20       # cash R&D + SG&A as % of revenue
    da_pct_revenue: float = 0.03         # D&A as % of revenue
    tax_rate: float = 0.21               # effective tax rate

    capex_pct_revenue: float = 0.05      # capex as % of revenue
    ar_days: float = 60.0
    inventory_days: float = 95.0
    ap_days: float = 70.0
    other_nwc_pct_revenue: float = 0.03

    cash_and_investments: float = 100.0
    total_debt: float = 100.0
    interest_rate_on_debt: float = 0.05
    interest_yield_on_cash: float = 0.03

    ppe_net: float = 200.0
    shares_diluted: float = 100.0

    current_price: float = 10.0
    market_cap: float = field(init=False)

    def __post_init__(self):
        self.market_cap = self.current_price * self.shares_diluted

    @property
    def net_debt(self) -> float:
        return self.total_debt - self.cash_and_investments  # negative = net cash

    @property
    def ttm_ebitda(self) -> float:
        gross_profit = self.revenue * self.gross_margin
        opex = self.revenue * self.opex_pct_revenue
        return gross_profit - opex

    @property
    def ttm_ebit(self) -> float:
        return self.ttm_ebitda - self.revenue * self.da_pct_revenue


# ---------------------------------------------------------------------------
# 2. FORECAST DRIVER ASSUMPTIONS
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One revenue segment for a sum-of-the-parts build."""
    name: str
    base_revenue: float          # $mm, TTM
    year1_growth: float          # Year-1 growth
    terminal_growth: float       # long-run growth for this segment
    gross_margin: Optional[float] = None  # optional segment-specific GM (else company GM used)


@dataclass
class ForecastDrivers:
    n_years: int = 10

    # Blended revenue growth fades linearly from year1_growth to terminal_growth
    year1_growth: float = 0.10
    terminal_growth: float = 0.03

    # Margins fade linearly from current gross_margin to a normalized long-run margin
    terminal_gross_margin: float = 0.50

    opex_pct_revenue: float = 0.20
    da_pct_revenue: float = 0.03
    capex_pct_revenue: float = 0.05
    terminal_capex_pct_revenue: float = 0.045

    tax_rate: float = 0.21

    ar_days: float = 60.0
    inventory_days: float = 95.0
    ap_days: float = 70.0
    other_nwc_pct_revenue: float = 0.03

    wacc: float = 0.10
    dividend_payout_pct_ni: float = 0.0

    # --- Share-count dynamics (NEW) ---
    # Net annual change in diluted shares = SBC dilution minus buybacks.
    # e.g. sbc_dilution_pct=0.010 and buyback_pct=0.015 -> shares shrink ~0.5%/yr.
    sbc_dilution_pct: float = 0.0        # gross new shares issued per year (as % of shares)
    buyback_pct: float = 0.0             # shares repurchased per year (as % of shares)

    # --- Terminal value method: blend of Gordon Growth and Exit Multiple ---
    exit_multiple_weight: float = 0.0        # 0 = pure Gordon Growth, 1 = pure exit multiple
    exit_ev_ebitda_multiple: float = 15.0    # applied to final forecast year EBITDA

    # --- Optional sum-of-segments revenue build ---
    segments: Optional[List[Segment]] = None

    # --- Optional explicit per-year revenue growth path (length n_years). ---
    # When set, it overrides the blended fade AND segments for revenue growth
    # (used e.g. to anchor Year-1 and Year-2 exactly to Street consensus, then
    # fade). Margins/other drivers are unaffected.
    custom_growth_path: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# 3. THE LINKED 3-STATEMENT MODEL
# ---------------------------------------------------------------------------

class ThreeStatementModel:
    def __init__(self, base: BaseYearActuals, drv: ForecastDrivers):
        self.base = base
        self.drv = drv
        self.results = None  # populated by run()

    def _fade(self, start, end, n):
        return np.linspace(start, end, n)

    def _segment_revenue_path(self):
        """Return an (n_years,) array of TOTAL revenue built from segments, plus a
        blended-gross-margin path implied by segment mix. If segments carry
        gross margins, the company GM path is overridden by the mix-weighted GM."""
        d, b = self.drv, self.base
        n = d.n_years
        seg_rev = np.zeros(n)
        seg_gp = np.zeros(n)
        any_gm = any(s.gross_margin is not None for s in d.segments)
        for s in d.segments:
            g_path = self._fade(s.year1_growth, s.terminal_growth, n)
            rev_prev = s.base_revenue
            gm = s.gross_margin if s.gross_margin is not None else b.gross_margin
            for i in range(n):
                rev_prev = rev_prev * (1 + g_path[i])
                seg_rev[i] += rev_prev
                seg_gp[i] += rev_prev * gm
        gm_path = (seg_gp / seg_rev) if any_gm else None
        return seg_rev, gm_path

    def run(self) -> pd.DataFrame:
        b, d = self.base, self.drv
        n = d.n_years

        # Revenue path priority: custom growth path > segments > blended fade
        if d.custom_growth_path is not None:
            growth_path = np.array(d.custom_growth_path, dtype=float)
            if len(growth_path) != n:
                raise ValueError("custom_growth_path length must equal n_years")
            revenue_path = None
            gm_path = self._fade(b.gross_margin, d.terminal_gross_margin, n)
        elif d.segments:
            seg_rev, seg_gm_path = self._segment_revenue_path()
            revenue_path = seg_rev
            growth_path = np.empty(n)
            prev = b.revenue
            for i in range(n):
                growth_path[i] = revenue_path[i] / prev - 1
                prev = revenue_path[i]
            gm_path = seg_gm_path if seg_gm_path is not None else self._fade(b.gross_margin, d.terminal_gross_margin, n)
        else:
            growth_path = self._fade(d.year1_growth, d.terminal_growth, n)
            revenue_path = None
            gm_path = self._fade(b.gross_margin, d.terminal_gross_margin, n)

        capex_path = self._fade(d.capex_pct_revenue, d.terminal_capex_pct_revenue, n)

        rows = []
        revenue_prev = b.revenue
        ppe_prev = b.ppe_net
        cash_prev = b.cash_and_investments
        debt = b.total_debt
        shares_prev = b.shares_diluted
        ar_prev = b.revenue * b.ar_days / 365
        inv_prev = b.revenue * (1 - b.gross_margin) * b.inventory_days / 365
        ap_prev = b.revenue * (1 - b.gross_margin) * b.ap_days / 365
        other_nwc_prev = b.revenue * b.other_nwc_pct_revenue

        for yr in range(1, n + 1):
            gm = gm_path[yr - 1]
            capex_pct = capex_path[yr - 1]

            if revenue_path is not None:
                revenue = revenue_path[yr - 1]
                g = growth_path[yr - 1]
            else:
                g = growth_path[yr - 1]
                revenue = revenue_prev * (1 + g)

            cogs = revenue * (1 - gm)
            gross_profit = revenue - cogs
            opex = revenue * d.opex_pct_revenue
            ebitda = gross_profit - opex
            da = revenue * d.da_pct_revenue
            ebit = ebitda - da

            # working capital (day-based)
            ar = revenue * d.ar_days / 365
            inv = cogs * d.inventory_days / 365
            ap = cogs * d.ap_days / 365
            other_nwc = revenue * d.other_nwc_pct_revenue

            nwc = ar + inv + other_nwc - ap
            nwc_prev = ar_prev + inv_prev + other_nwc_prev - ap_prev
            delta_nwc = nwc - nwc_prev

            capex = revenue * capex_pct

            # --- Unlevered free cash flow (for DCF) ---
            ufcf = ebit * (1 - d.tax_rate) + da - capex - delta_nwc

            # --- Levered path (for the 3-statement / net income) ---
            interest_expense = debt * b.interest_rate_on_debt
            interest_income = cash_prev * b.interest_yield_on_cash
            pretax_income = ebit - interest_expense + interest_income
            tax = pretax_income * d.tax_rate
            net_income = pretax_income - tax

            dividends = net_income * d.dividend_payout_pct_ni

            # --- Share count dynamics ---
            net_share_change = shares_prev * (d.sbc_dilution_pct - d.buyback_pct)
            shares = shares_prev + net_share_change
            # cash impact of buybacks net of SBC issuance proceeds (approx: net buyback at price)
            # kept out of the UFCF (equity financing), but flows through CFF for the BS to balance
            buyback_cash = shares_prev * d.buyback_pct * b.current_price
            sbc_proceeds = shares_prev * d.sbc_dilution_pct * b.current_price * 0.0  # SBC is non-cash; proceeds ~0
            net_buyback_cash = buyback_cash - sbc_proceeds

            cfo = net_income + da - delta_nwc
            cfi = -capex
            cff = -dividends - net_buyback_cash
            change_in_cash = cfo + cfi + cff
            cash = cash_prev + change_in_cash

            ppe = ppe_prev + capex - da

            rows.append(dict(
                year=yr, revenue=revenue, growth=g, gross_margin=gm,
                cogs=cogs, gross_profit=gross_profit, opex=opex, ebitda=ebitda,
                da=da, ebit=ebit, ebit_margin=ebit / revenue,
                interest_expense=interest_expense, interest_income=interest_income,
                pretax_income=pretax_income, tax=tax, net_income=net_income,
                eps=net_income / shares,
                ar=ar, inventory=inv, ap=ap, other_nwc=other_nwc, nwc=nwc,
                delta_nwc=delta_nwc, capex=capex, ppe=ppe,
                cfo=cfo, cfi=cfi, cff=cff, change_in_cash=change_in_cash, cash=cash,
                debt=debt, shares=shares, ufcf=ufcf,
            ))

            revenue_prev, ppe_prev, cash_prev, shares_prev = revenue, ppe, cash, shares
            ar_prev, inv_prev, ap_prev, other_nwc_prev = ar, inv, ap, other_nwc

        self.results = pd.DataFrame(rows).set_index("year")
        return self.results

    # -----------------------------------------------------------------
    # DCF VALUATION
    # -----------------------------------------------------------------
    def dcf_value(self, wacc: float = None, terminal_growth: float = None,
                  exit_multiple_weight: float = None, exit_ev_ebitda_multiple: float = None) -> dict:
        """Discount the UFCF path + terminal value to today; bridge to price/share.

        Terminal value blends Gordon Growth and an exit EV/EBITDA multiple per
        `exit_multiple_weight` (0 = pure Gordon Growth, 1 = pure exit multiple).

        Per-share value uses the FINAL-YEAR diluted share count so multi-year
        dilution/buybacks are reflected (previously used a flat base-year count).
        """
        if self.results is None:
            self.run()
        wacc = self.drv.wacc if wacc is None else wacc
        tg = self.drv.terminal_growth if terminal_growth is None else terminal_growth
        w = self.drv.exit_multiple_weight if exit_multiple_weight is None else exit_multiple_weight
        exit_mult = self.drv.exit_ev_ebitda_multiple if exit_ev_ebitda_multiple is None else exit_ev_ebitda_multiple

        if wacc <= tg:
            raise ValueError("WACC must exceed terminal growth rate for a finite terminal value.")

        ufcf = self.results["ufcf"].values
        n = len(ufcf)
        discount_factors = np.array([(1 + wacc) ** -t for t in range(1, n + 1)])
        pv_ufcf = ufcf * discount_factors

        gordon_tv = ufcf[-1] * (1 + tg) / (wacc - tg)

        if w > 0:
            final_ebitda = self.results["ebitda"].iloc[-1]
            exit_tv = exit_mult * final_ebitda
            terminal_value = (1 - w) * gordon_tv + w * exit_tv
        else:
            terminal_value = gordon_tv

        pv_terminal = terminal_value * discount_factors[-1]

        enterprise_value = pv_ufcf.sum() + pv_terminal
        equity_value = enterprise_value - self.base.net_debt

        # Use average of base and terminal share count for a mid-horizon view,
        # but report both. Per-share uses terminal shares (dilution fully reflected).
        terminal_shares = self.results["shares"].iloc[-1]
        price_per_share = equity_value / terminal_shares

        return dict(
            wacc=wacc, terminal_growth=tg, exit_multiple_weight=w, exit_ev_ebitda_multiple=exit_mult,
            pv_ufcf=pv_ufcf.sum(), pv_terminal=pv_terminal, gordon_terminal_value=gordon_tv,
            enterprise_value=enterprise_value, equity_value=equity_value,
            price_per_share=price_per_share,
            terminal_shares=terminal_shares, base_shares=self.base.shares_diluted,
            terminal_value_pct_of_ev=pv_terminal / enterprise_value,
        )

    # -----------------------------------------------------------------
    # REVERSE DCF
    # -----------------------------------------------------------------
    def solve_for_terminal_growth(self, target_price: float, wacc: float = None,
                                  lo=-0.05, hi=None, tol=1e-6, max_iter=100) -> float:
        wacc = self.drv.wacc if wacc is None else wacc
        hi = wacc - 1e-4 if hi is None else hi

        def f(tg):
            return self.dcf_value(wacc=wacc, terminal_growth=tg)["price_per_share"] - target_price

        return self._bisect(f, lo, hi, tol, max_iter)

    def solve_for_wacc(self, target_price: float, terminal_growth: float = None,
                       lo=None, hi=0.30, tol=1e-6, max_iter=100) -> float:
        tg = self.drv.terminal_growth if terminal_growth is None else terminal_growth
        lo = tg + 1e-4 if lo is None else lo

        def f(w):
            return self.dcf_value(wacc=w, terminal_growth=tg)["price_per_share"] - target_price

        return self._bisect(f, lo, hi, tol, max_iter)

    def solve_for_year1_growth(self, target_price: float, lo=-0.20, hi=2.0, tol=1e-6, max_iter=100) -> float:
        """Reverse-solve the Year-1 revenue growth that reconciles to target_price
        (only meaningful for the single blended-line build, not segments)."""
        import copy

        def f(g):
            drv = copy.deepcopy(self.drv)
            drv.year1_growth = g
            drv.segments = None
            m = ThreeStatementModel(self.base, drv)
            m.run()
            return m.dcf_value()["price_per_share"] - target_price

        return self._bisect(f, lo, hi, tol, max_iter)

    @staticmethod
    def _bisect(f, lo, hi, tol, max_iter):
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            raise ValueError(
                f"No sign change in bracket [{lo:.4f}, {hi:.4f}] "
                f"(f(lo)={flo:.4f}, f(hi)={fhi:.4f}). Widen the search range."
            )
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            fmid = f(mid)
            if abs(fmid) < tol:
                return mid
            if flo * fmid < 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return (lo + hi) / 2
