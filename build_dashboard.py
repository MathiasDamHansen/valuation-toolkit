"""
build_dashboard.py
==================
Assembles a single self-contained one-page HTML dashboard from the outputs of
run_full_analysis.py. All charts are embedded as base64 so the file is fully
portable (email it, host it on GitHub Pages, drop it in SharePoint — no external
image files needed).

Run:
    python build_dashboard.py                       # uses data/
    python build_dashboard.py --data-dir data_msft  # any company

Optionally also writes a PDF if a converter is available (weasyprint), else it
just tells you the HTML path.
"""

import argparse
import base64
import os
import datetime as dt

import run_full_analysis as rfa
from settings import OUTPUT_DIR, output_path, ensure_output_dir


def _img_b64(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _card(title, caption, img_b64, full=False):
    if not img_b64:
        return ""
    cls = "card full-width" if full else "card"
    return f"""
      <div class="{cls}">
        <h2>{title}</h2>
        <p class="caption">{caption}</p>
        <img src="{img_b64}" alt="{title}"/>
      </div>"""


def _table_html(df, fmt=None, index=False):
    fmt = fmt or {}
    df2 = df.copy()
    for c, fn in fmt.items():
        if c in df2.columns:
            df2[c] = df2[c].apply(fn)
    return df2.to_html(classes="data-table", index=index, border=0, escape=False)


def build(data_dir="data", n_trials=20_000):
    ensure_output_dir()
    res = rfa.run(data_dir=data_dir, n_trials=n_trials)
    meta, base = res["meta"], res["base"]
    ticker = meta["ticker"]
    charts = res["charts"]

    # KPI strip
    cur = base.current_price
    dcf = res["base_val"]["price_per_share"]
    upside = dcf / cur - 1
    kpis = [
        ("Current price", f"${cur:,.2f}"),
        ("Base-case DCF", f"${dcf:,.2f}"),
        ("DCF vs price", f"{upside:+.0%}"),
        ("MC percentile of price", f"{res['pct_below']:.0f}th"),
    ]
    if res["has_consensus"]:
        kpis.append(("Street avg target", f"${res['consensus'].avg_target:,.2f}"))
    if res["reverse"]["implied_terminal_growth"] is not None:
        kpis.append(("Mkt-implied term. growth", f"{res['reverse']['implied_terminal_growth']:.2%}"))
    if res["reverse"]["implied_wacc"] is not None:
        kpis.append(("Mkt-implied WACC", f"{res['reverse']['implied_wacc']:.2%}"))

    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-val">{v}</div><div class="kpi-lbl">{k}</div></div>'
        for k, v in kpis
    )

    # Summary + scenarios tables
    summary_tbl = _table_html(res["summary"], fmt={"price": lambda x: f"${x:,.2f}"})
    scen = res["scenarios"][["scenario", "year1_growth", "terminal_growth",
                             "terminal_gross_margin", "wacc", "implied_price", "vs_current"]].copy()
    scen_tbl = _table_html(scen, fmt={
        "year1_growth": lambda x: f"{x:.1%}", "terminal_growth": lambda x: f"{x:.1%}",
        "terminal_gross_margin": lambda x: f"{x:.1%}", "wacc": lambda x: f"{x:.1%}",
        "implied_price": lambda x: f"${x:,.2f}", "vs_current": lambda x: f"{x:+.0%}"})

    # cards
    cards = []
    cards.append(_card("Master Valuation Summary",
                       "Every method — DCF, consensus, multiples, scenarios, Monte Carlo, Street — on one chart.",
                       _img_b64(charts.get("master")), full=True))
    cards.append(_card("Monte Carlo (own assumptions)",
                       "Distribution of implied price and each driver's relationship to it.",
                       _img_b64(charts.get("monte_carlo")), full=True))
    cards.append(_card("Tornado — driver sensitivity",
                       "Which assumptions move the DCF price most (widest bar = biggest swing).",
                       _img_b64(charts.get("tornado"))))
    cards.append(_card("2D sensitivity grid",
                       "Deterministic WACC x terminal-growth price grid.",
                       _img_b64(charts.get("grid"))))
    cards.append(_card("Bear / Base / Bull scenarios",
                       "Three named, self-consistent assumption sets.",
                       _img_b64(charts.get("scenarios"))))
    if "hist_mult" in charts:
        cards.append(_card("Own valuation vs history",
                           "Where the company trades now vs its own multiple history.",
                           _img_b64(charts.get("hist_mult"))))
    if "bank_targets" in charts:
        cards.append(_card("Bank targets vs model",
                           "Individual analyst targets against model-implied values.",
                           _img_b64(charts.get("bank_targets")), full=True))
    if "consensus_hist" in charts:
        cards.append(_card("Consensus-weighted Monte Carlo",
                           "Own vs Street-anchored growth distribution.",
                           _img_b64(charts.get("consensus_hist"))))
    if "multiples" in charts:
        cards.append(_card("Peer multiples valuation",
                           "Peer forward EV/EBITDA and implied price by method.",
                           _img_b64(charts.get("multiples")), full=True))
    if "exit_blend" in charts:
        cards.append(_card("Exit-multiple-blended terminal value",
                           "Gordon Growth vs peer-exit-multiple terminal value.",
                           _img_b64(charts.get("exit_blend"))))

    coverage = (f"consensus: {'yes' if res['has_consensus'] else 'no coverage'} · "
                f"peers: {'yes' if res['has_comps'] else 'none'} · "
                f"segments: {'yes' if res['segments'] else 'no'}")

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker} Valuation Dashboard</title>
<style>
  :root {{ --brand:#2563eb; --ink:#111827; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:#f4f5f7; margin:0; padding:24px; color:#1a1a1a; }}
  h1 {{ font-size:1.6rem; margin-bottom:2px; }}
  .subtitle {{ color:#666; margin-bottom:18px; font-size:.9rem; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:22px; }}
  .kpi {{ background:white; border-radius:10px; padding:12px 18px; box-shadow:0 1px 4px rgba(0,0,0,.08); min-width:130px; }}
  .kpi-val {{ font-size:1.3rem; font-weight:700; color:var(--brand); }}
  .kpi-lbl {{ color:#777; font-size:.78rem; margin-top:2px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:20px; }}
  .card {{ background:white; border-radius:10px; padding:16px 20px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  .card h2 {{ font-size:1.05rem; margin:0 0 4px; }}
  .caption {{ color:#777; font-size:.85rem; margin:0 0 12px; }}
  img {{ width:100%; height:auto; border-radius:6px; }}
  table.data-table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
  table.data-table th, table.data-table td {{ text-align:left; padding:6px 8px; border-bottom:1px solid #eee; }}
  table.data-table th {{ background:#fafafa; }}
  .full-width {{ grid-column:1 / -1; }}
  .footer {{ color:#999; font-size:.78rem; margin-top:26px; }}
</style></head><body>
  <h1>{meta['company_name']} ({ticker}) — Valuation Dashboard</h1>
  <p class="subtitle">As of {meta['as_of_date']} · {coverage} · generated {dt.date.today().isoformat()} by the reverse-DCF toolkit</p>
  <div class="kpis">{kpi_html}</div>
  <div class="grid">
    <div class="card full-width"><h2>Consolidated valuation summary</h2>
      <p class="caption">All methods, sorted high to low.</p>{summary_tbl}</div>
    <div class="card full-width"><h2>Scenario detail (bear / base / bull)</h2>
      <p class="caption">Assumption sets and implied prices.</p>{scen_tbl}</div>
    {''.join(cards)}
  </div>
  <p class="footer">Educational tool, not investment advice. Outputs are only as good as the input CSVs —
  regenerate them via AGENT_DATA_COLLECTION_PROMPT.md and re-run. Multiples/consensus layers auto-skip
  when coverage is unavailable.</p>
</body></html>"""

    out_path = output_path(ticker, "dashboard", "html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nSaved dashboard: {out_path}")

    # Optional PDF
    try:
        from weasyprint import HTML  # noqa
        pdf_path = output_path(ticker, "dashboard", "pdf")
        HTML(string=html).write_pdf(pdf_path)
        print(f"Saved PDF: {pdf_path}")
    except Exception:
        print("(PDF export skipped — install weasyprint for PDF, or print the HTML to PDF from your browser.)")

    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n-trials", type=int, default=20_000)
    args = p.parse_args()
    build(data_dir=args.data_dir, n_trials=args.n_trials)
