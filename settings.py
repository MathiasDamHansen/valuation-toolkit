"""
settings.py
===========
Central, company-agnostic configuration. Nothing here is NVIDIA-specific — it
just controls where outputs go and how charts are branded, so the same code
works for any ticker (mega-cap or micro-cap, heavily covered or not).
"""

import os

# Where charts/CSVs are written. Overridable via env var so the same code runs
# locally, in CI, in a notebook, or inside the Streamlit app.
OUTPUT_DIR = os.environ.get("VALUATION_OUTPUT_DIR", "/mnt/user-data/outputs")

# Neutral brand palette (was hard-coded NVIDIA green before). These are generic
# and readable for ANY company.
BRAND = "#2563eb"      # primary (blue)
BRAND_ALT = "#f59e0b"  # secondary (amber)
POS = "#16a34a"        # positive / peers
NEG = "#dc2626"        # negative / subject highlight
INK = "#111827"        # near-black for reference lines


def output_path(ticker: str, name: str, ext: str = "png") -> str:
    """Ticker-prefixed output path, e.g. output_path('NVDA','master','png')."""
    tk = (ticker or "subject").lower().replace(" ", "_")
    return os.path.join(OUTPUT_DIR, f"{tk}_{name}.{ext}")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
