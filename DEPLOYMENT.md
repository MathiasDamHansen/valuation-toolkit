# Deployment guide — GitHub, Streamlit, Colab (and a note on Lovable)

You asked to run this via GitHub / Lovable / Jupyter with a live dashboard.
Here's the honest recommendation and exact steps for each.

---

## TL;DR — the easiest path that actually works

**GitHub (code home) → Streamlit Community Cloud (free live dashboard).**

Streamlit is the Python-native equivalent of a "Lovable website": one file
(`app.py`), a public URL, sliders, CSV upload — and it can actually execute the
Monte Carlo / DCF engine. Lovable builds a React/Tailwind front-end and **cannot
run Python**, so using Lovable would mean rebuilding the model as a web API and
having Lovable call it (pattern described at the bottom if you still want it).

---

## 1. Put the repo on GitHub (one time)

```bash
cd valuation_toolkit
git init
git add .
git commit -m "Reverse-DCF + consensus + multiples toolkit"
# create an empty repo on github.com first, then:
git remote add origin https://github.com/<you>/valuation-toolkit.git
git branch -M main
git push -u origin main
```

That's your code home. Anyone can `git clone` it and run `python run_full_analysis.py`.

---

## 2. Live interactive dashboard — Streamlit Community Cloud (free)

1. Push the repo (step 1). Make sure `app.py` and `requirements.txt` are at the root.
2. Go to **share.streamlit.io** → *New app* → pick your repo/branch → main file
   `app.py` → **Deploy**.
3. You get a public URL like `https://<you>-valuation-toolkit.streamlit.app`.
   It has company selector, assumption sliders, and a CSV uploader so you can
   value any ticker by dropping in the 5 CSVs from the agent prompt.

Local test first:
```bash
pip install -r requirements.txt
streamlit run app.py
```

> Tip: `weasyprint` (PDF export) can be awkward on some cloud builders. It's
> already marked optional in `requirements.txt`; the app itself doesn't need it,
> only `build_dashboard.py`'s PDF step does, and that degrades gracefully.

---

## 3. Static one-page dashboard — GitHub Pages (no server needed)

`build_dashboard.py` writes a **self-contained** `*_dashboard.html` (charts
embedded as base64 — no external files). To publish it:

```bash
python build_dashboard.py                 # produces <ticker>_dashboard.html
mkdir -p docs && cp /path/to/<ticker>_dashboard.html docs/index.html
git add docs && git commit -m "publish dashboard" && git push
```
Then in the repo: **Settings → Pages → Source: `main` / `docs`**. Your dashboard
is live at `https://<you>.github.io/valuation-toolkit/`. Re-run the script and
push to refresh. Great for a fixed snapshot; use Streamlit if you want
interactivity.

---

## 4. Jupyter / Google Colab

- **Local Jupyter:** `pip install -r requirements.txt jupyter` → `jupyter lab`
  → open `valuation_toolkit.ipynb` → Run All.
- **Colab:** upload the folder (or `!git clone <your repo>` in the first cell),
  then run top to bottom. The notebook sets `VALUATION_OUTPUT_DIR` to a local
  `outputs/` folder and displays every chart inline.

---

## 5. If you specifically want Lovable

Lovable is great for a polished marketing/UX shell but can't run this model.
The clean pattern is **Lovable front-end → your Python back-end**:

1. Wrap the engine in a tiny API. Create `api.py`:
   ```python
   from fastapi import FastAPI
   from data_loader import load_all
   from model import ThreeStatementModel
   app = FastAPI()

   @app.get("/value")
   def value(data_dir: str = "data"):
       d = load_all(data_dir)
       m = ThreeStatementModel(d["base"], d["drivers"]); m.run()
       v = m.dcf_value()
       return {"ticker": d["meta"]["ticker"],
               "price": d["base"].current_price,
               "dcf": v["price_per_share"]}
   ```
   Run with `uvicorn api:app` and deploy it on Render/Railway/Fly.io (free tiers).
2. In Lovable, build the UI and have it `fetch()` your API's JSON, then render
   the numbers/charts.

This is strictly more work than Streamlit for the same result, so only go here
if you need Lovable's specific look-and-feel.

---

## Recommended for you

1. **GitHub** for the code.
2. **Streamlit Community Cloud** for the live, interactive, company-agnostic
   dashboard (public URL, CSV upload, sliders).
3. **GitHub Pages** if you also want a static shareable snapshot.
4. **Colab notebook** for ad-hoc deep dives.

Skip Lovable unless you need its front-end styling specifically.
