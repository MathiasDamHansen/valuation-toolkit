"""
analyst_data.py
================
Wall Street analyst consensus data for NVIDIA, as of mid-to-late August 2026.

Sources: aggregated from S&P Global Market Intelligence / TipRanks-sourced
consensus feeds (stockanalysis.com, chartmill.com), and individual bank notes
reported via financial media (TipRanks analyst-action feed, Simply Wall St).

IMPORTANT: Analyst price targets change every few days, especially around
earnings (NVDA's next print is Aug 26, 2026). Treat these as a snapshot, not
a live feed. Re-pull before using this for real decisions — a free source
like stockanalysis.com/stocks/nvda/forecast or your broker's research tab
will have the latest.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BankTarget:
    bank: str
    analyst: str
    target: float          # $/share, 12-month price target
    rating: str
    date: str               # ISO date of the note


@dataclass
class AnalystConsensus:
    as_of: str = "mid-August 2026"
    num_analysts: int = 62
    avg_target: float = 304.73
    median_target: float = 300.0
    low_target: float = 180.0
    high_target: float = 500.0
    rating: str = "Strong Buy"

    # Street consensus estimate for NVIDIA's current/next fiscal year
    # (FY2027, year ending on or around Jan 2027). Non-GAAP basis, as is
    # standard for consensus EPS feeds.
    fy_next_label: str = "FY2027 (NVIDIA fiscal year ending ~Jan 2027)"
    fy_next_revenue_estimate: float = 391_300.0   # $mm
    fy_next_eps_estimate: float = 9.34            # $/share, non-GAAP

    # Second forward year (optional) — enables a multi-year consensus fade
    # (anchor Year-1 AND Year-2 growth to the Street instead of just Year-1).
    fy_next2_label: str = "FY2028 (year ending ~Jan 2028)"
    fy_next2_revenue_estimate: float = None       # $mm, optional
    fy_next2_eps_estimate: float = None           # $/share, optional

    # Individual bank / analyst price targets (most recent available note
    # for each firm as of the as_of date above). A couple of the entries
    # are from earlier in 2026 where a firm hadn't refreshed since; flagged
    # via the date field so you can see freshness at a glance.
    bank_targets: List[BankTarget] = field(default_factory=lambda: [
        BankTarget("Wells Fargo", "Aaron Rakers", 315.0, "Overweight", "2026-08-11"),
        BankTarget("Bernstein", "Stacy Rasgon", 315.0, "Outperform", "2026-08-03"),
        BankTarget("Tigress Financial", "Ivan Feinseth", 360.0, "Strong Buy", "2026-03-05"),
        BankTarget("Evercore ISI", "Mark Lipacis", 352.0, "Outperform", "2026-03-02"),
        BankTarget("Morgan Stanley", "Joseph Moore", 288.0, "Overweight", "2026-08-14"),
        BankTarget("Goldman Sachs", "James Schneider", 285.0, "Buy", "2026-08-12"),
        BankTarget("Bank of America", "Vivek Arya", 275.0, "Buy", "2026-01-28"),
        BankTarget("Susquehanna", "Christopher Rolland", 275.0, "Positive", "2026-08-12"),
        BankTarget("JPMorgan", "Harlan Sur", 265.0, "Overweight", "2026-03-02"),
    ])
