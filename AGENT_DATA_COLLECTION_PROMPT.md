# Data Collection Prompt — Reverse-DCF / Multiples Toolkit

Copy everything in the box below into a fresh conversation with an AI agent
that has **live web search** (e.g. Claude with web search enabled, or any
agent with an equivalent tool). Replace `{TICKER}` with the company you want
(e.g. `NVDA`, `MSFT`, `ASML`). The agent will do the research and hand back
five CSV files in the *exact* schema this toolkit expects — drop them into
`data/` and run `python run_full_analysis.py`.

You can re-run this prompt any time (e.g. weekly, or right after earnings) to
refresh the dataset for the same company, or swap in a different ticker.

---

## PROMPT TO COPY

```
You are a financial research agent. Research the company with ticker
{TICKER} and produce exactly five CSV files, formatted precisely as
specified below. Use web search to find current, sourced data — do not
estimate or fabricate figures. If a figure genuinely cannot be found after
a reasonable search, leave it blank rather than guessing, and note the gap
in a final summary.

Work through these steps in order:

STEP 1 — Identify the company and current market data
- Full company name, ticker, exchange
- Current share price, diluted shares outstanding, market cap
- Today's date (use as the as_of_date for market-based figures)

STEP 2 — Trailing-twelve-month (TTM) financials
Find the most recent quarterly/annual filings (10-K, 10-Q, or local
equivalent) and compute or extract TTM:
- Revenue, gross margin, EBITDA, EBIT, net income, diluted EPS
- Capital expenditures
- Total debt, cash & marketable securities, net PP&E
- Effective/guided tax rate
- If available: AR days, inventory days, AP days (else note as "not found,
  use default")
- Beta, and a reasonable current risk-free rate (10yr govt bond yield) and
  equity risk premium (~4-5.5% is typical) — or a directly-stated WACC/cost
  of capital estimate from a source if you find one

STEP 3 — Annual financial history (last 5 fiscal years)
For each of the last 5 fiscal years, find: period end date, revenue, gross
margin, EBIT, net income, diluted EPS, capex, total debt, cash &
investments, diluted shares outstanding.

STEP 4 — Analyst consensus
- Number of analysts covering the stock
- Average, median, low, and high 12-month price targets
- Consensus rating (e.g. Buy/Hold/Sell or Strong Buy, etc.)
- Consensus revenue and EPS estimates for the current/next fiscal year
  (state which fiscal year label this is, e.g. "FY2027")
- If available, next quarter's consensus revenue and EPS estimate

STEP 5 — Individual bank/analyst price targets
Find as many recent (ideally within the last 1-2 months) individual price
targets as you can from major banks/research shops (aim for at least 6-10):
bank name, analyst name, price target, rating, and the date of the note.
Prioritize freshness — note the date for each so staleness is visible.

STEP 6 — Peer comps / trading multiples
Identify 4-6 direct public peers/competitors. For the subject company AND
each peer, find:
- Current price, shares outstanding, market cap
- Net debt (total debt minus cash) and enterprise value
- Trailing P/E, forward P/E
- Trailing EV/EBITDA, forward (NTM) EV/EBITDA
- Trailing EV/Revenue, forward (NTM) EV/Revenue
- Forward PEG ratio, if available
- Date of the data and any methodology notes (e.g. GAAP vs non-GAAP basis)

STEP 7 — Output exactly these five CSV files (use these exact filenames,
column names, and column order — do not add, remove, or rename columns; use
a blank value rather than "N/A" or "-" for missing data; use plain numbers
with no currency symbols, commas, or % signs; percentages as decimals e.g.
0.11 not 11%; do not include commas inside any text field, since these are
plain CSVs, not quoted):

=== FILE 1: company_inputs.csv ===
Long format, exactly these rows (field,value,unit,notes):
ticker,,text,
company_name,,text,
as_of_date,,date,
current_price,,USD/share,
shares_diluted,,millions,
market_cap,,USD mm,leave blank to auto-derive
ttm_revenue,,USD mm,
ttm_gross_margin,,decimal,
ttm_ebitda,,USD mm,
ttm_ebit,,USD mm,
ttm_net_income,,USD mm,
ttm_eps_diluted,,USD/share,
ttm_capex,,USD mm,
total_debt,,USD mm,
cash_and_investments,,USD mm,
ppe_net,,USD mm,
tax_rate_effective,,decimal,
ar_days,,days,leave blank if not found
inventory_days,,days,leave blank if not found
ap_days,,days,leave blank if not found
interest_rate_on_debt,,decimal,leave blank if not found
interest_yield_on_cash,,decimal,leave blank if not found
beta,,number,
risk_free_rate,,decimal,
equity_risk_premium,,decimal,
wacc_estimate,,decimal,leave blank to auto-derive from CAPM

=== FILE 2: financial_history.csv ===
Wide format, one row per fiscal year (5 rows), exact header:
fiscal_year,period_end_date,revenue,gross_margin,ebit,net_income,eps_diluted,capex,total_debt,cash_and_investments,shares_diluted,source_note

=== FILE 3: analyst_consensus.csv ===
Long format, exactly these rows (field,value):
ticker,
as_of_date,
num_analysts,
avg_price_target,
median_price_target,
low_price_target,
high_price_target,
consensus_rating,
next_fy_label,
next_fy_revenue_estimate,
next_fy_eps_estimate,
next_fy2_label,
next_fy2_revenue_estimate,
next_fy2_eps_estimate,
next_quarter_revenue_estimate,
next_quarter_eps_estimate

=== FILE 4: analyst_bank_targets.csv ===
Wide format, one row per bank/analyst, exact header:
bank,analyst,price_target,rating,date

=== FILE 5: multiples_comps.csv ===
Wide format, one row per company (subject company FIRST, then peers), exact
header:
entity,ticker,is_subject,price,shares_diluted,market_cap,net_debt,enterprise_value,pe_ttm,pe_fwd,ev_ebitda_ttm,ev_ebitda_fwd,ev_revenue_ttm,ev_revenue_fwd,peg_fwd,as_of_date,notes
(use TRUE/FALSE for is_subject; use semicolons instead of commas inside the
notes field)

STEP 7b — OPTIONAL extra files (produce these when the data exists; the
toolkit uses them automatically and simply skips them if absent). Same
formatting rules as above.

=== OPTIONAL FILE A: segments.csv ===
One row per reporting segment (enables a sum-of-the-parts revenue build
instead of one blended growth rate). Exact header:
segment,base_revenue,year1_growth,terminal_growth,gross_margin
(base_revenue in USD mm = segment TTM revenue; growths as decimals; leave
gross_margin blank to inherit the company-level margin.)

=== OPTIONAL FILE B: valuation_history.csv ===
The company's OWN historical trading multiples (so the model can show whether
today's multiple is rich/cheap vs its own history, not just vs peers). One row
per date (e.g. fiscal year-ends over the last 3-5 years). Exact header:
date,price,pe,ev_ebitda
(leave pe or ev_ebitda blank if not available for a date.)

Also, in company_inputs.csv you MAY add these optional rows to model share-count
dynamics and shareholder returns (all as decimals; the toolkit defaults them to
0 if absent):
sbc_dilution_pct,,decimal,gross new shares issued per year as % of shares (SBC)
buyback_pct,,decimal,shares repurchased per year as % of shares
dividend_payout_pct_ni,,decimal,dividend payout as % of net income

STEP 8 — Final summary
After producing the five CSVs, give a short bullet list of:
- Any fields you could not find and left blank
- Any figures you're less confident in (e.g. derived/estimated rather than
  directly sourced) and why
- The overall as-of date range of the data collected
```

---

## What to do with the output

1. Save the five CSVs into this project's `data/` folder, overwriting the
   existing example files (or into a new folder, e.g. `data_msft/`, if you
   want to keep multiple companies side by side).
2. Run:
   ```bash
   python run_full_analysis.py --data-dir data
   ```
   (or `--data-dir data_msft` etc.)
3. Everything downstream — the 3-statement model, DCF, reverse-solve,
   consensus-anchored valuation, comps valuation, and all three Monte Carlo
   variants — runs automatically off those five files. No code changes
   needed for a new company.

## Why this schema (not something looser)

The loader (`data_loader.py`) parses these CSVs with strict column names and
order because that's what makes the "one prompt in, working analysis out"
pattern reliable — a research agent that free-forms its output format would
require you to hand-edit the CSVs or the parser every time. Keeping the
schema fixed means the same prompt works unmodified for any ticker.
