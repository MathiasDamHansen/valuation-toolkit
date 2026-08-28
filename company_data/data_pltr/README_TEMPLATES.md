# Data templates — with the NEW columns for the bottom-of-dashboard build

These templates add the fields needed to power the requested bottom section
(5 years actuals + 5 years forecast with EBITDA, margins, cash flow, ROIC, ROE,
FCF yield) and the historical multiples table (EV/Sales, EV/EBITDA, EV/EBIT, P/E).

## Two folders

- **`data_nvidia_example/`** — a fully filled NVDA example so you can see the exact format.
- **`data_BLANK_template/`** — empty versions to copy for your own companies.

## How to use in your repo

1. In your GitHub repo, go to the `company_data/` folder.
2. Create a subfolder named `data_<company>` (e.g. `data_msft`).
3. Upload the CSVs from `data_BLANK_template/` into it and fill them in
   (only `company_inputs.csv` + `financial_history.csv` are strictly required).
4. Commit. It appears automatically in the Streamlit dropdown.

## WHAT'S NEW vs. the old templates (this is the important part)

### `company_inputs.csv`
- **`shareholders_equity`** — base-year book equity. Enables the model to roll
  equity forward each year (begin equity + net income − dividends − buybacks),
  which is what makes **forecast ROE** possible. Without it, ROE shows n/a.

### `financial_history.csv` — several NEW columns so the ACTUALS side can show
the same metrics as the forecast (previously impossible):
- **`ebitda`** and **`da`** (D&A) — needed for actual EBITDA + EBITDA margin.
  (Provide at least one; if you give `da`, EBITDA = EBIT + D&A.)
- **`cfo`** — operating cash flow → the cash-flow row and actual FCF (= cfo − capex).
- **`shareholders_equity`** — enables **ROE** and **ROIC** on the actuals.
- **`price`** (year-end share price) — enables **FCF yield** on the actuals.
- (already present and still required: revenue, gross_margin, ebit, net_income,
  eps_diluted, capex, total_debt, cash_and_investments, ppe_net, shares_diluted)

### `valuation_history.csv` — NEW multiple columns for the historical table:
- **`ev_sales`** (EV/Sales) — was missing before; you asked for it.
- **`ev_ebit`** (EV/EBIT) — the enterprise-level "EV/Earnings" view.
- (already present: `price`, `pe`, `ev_ebitda`)

## Notes / conventions

- Percentages as decimals (0.74 = 74%), no % signs, no thousands commas.
- `capex` is a positive number (the model subtracts it).
- Leave any cell blank if you genuinely can't source it — the dashboard shows
  "n/a" for that metric rather than inventing a value. The more of the NEW
  columns you fill, the more of the actuals block lights up.
- "EV/Earnings" is shown as **P/E** (equity) and **EV/EBIT** (enterprise), since
  a literal "EV/Earnings" isn't a standard multiple.
- ROIC used: operating **NOPAT / (net PP&E + net working capital)**.
  ROE used: **net income / average shareholders' equity**.
  FCF yield used: **FCF / market cap** (that year's price × shares for actuals;
  current price for the forecast).
