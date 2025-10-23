#!/usr/bin/env python3
"""
scripts/eda_refactored.py

Refactored, production-grade EDA for time series datasets.
Generates decomposition, ACF/PACF, ADF/KPSS tests, and Markdown reports.
"""

import argparse
import logging

# import sys
# import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("eda")


# ---------- Helpers ----------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def infer_freq(idx):
    try:
        return pd.infer_freq(idx)
    except Exception as e:
        print(f"Error running ADF test: {e}")
        return None


# ---------- Core functions ----------
def run_adf(ts):
    res = adfuller(ts.dropna(), autolag="AIC")
    return {"stat": res[0], "p": res[1], "crit": res[4]}


def run_kpss(ts, reg="c"):
    res = kpss(ts.dropna(), regression=reg, nlags="auto")
    return {"stat": res[0], "p": res[1], "crit": res[3]}


def save_decomposition(ts, period, model, path):
    res = seasonal_decompose(ts, model=model, period=period, extrapolate_trend="freq")
    fig = res.plot()
    fig.set_size_inches(10, 8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return res


def save_acf_pacf(ts, path, lags=48):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(ts.dropna(), ax=ax[0])
    plot_pacf(ts.dropna(), ax=ax[1])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(path, meta, adf, kpss_l, kpss_t):
    with open(path, "w") as f:
        f.write(f"# Data Profile — {meta['name']}\n\n")
        f.write(f"**Obs:** {meta['n']}, Start: {meta['start']}, End: {meta['end']}  \n")
        f.write(f"**Freq:** {meta['freq']}\n\n")
        f.write("## ADF Test\n")
        f.write(f"- Statistic: {adf['stat']:.4f}, p={adf['p']:.4f}\n")
        f.write("Stationary \n\n" if adf["p"] < 0.05 else "Non-stationary \n\n")
        f.write("## KPSS (level)\n")
        f.write(f"- Statistic: {kpss_l['stat']:.4f}, p={kpss_l['p']:.4f}\n")
        f.write("Non-stationary \n\n" if kpss_l["p"] < 0.05 else "Stationary \n\n")
        f.write("## KPSS (trend)\n")
        f.write(f"- Statistic: {kpss_t['stat']:.4f}, p={kpss_t['p']:.4f}\n\n")
        f.write("## Recommendations\n")
        f.write("- Difference series if ADF fails & KPSS rejects.\n")


# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--date-col", default="Month")
    p.add_argument("--target", default="Passengers")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--model", default="additive")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df = df.sort_values(args.date_col)
    df.set_index(args.date_col, inplace=True)
    ts = df[args.target].astype(float)
    freq = infer_freq(ts.index)
    # period = 12 if (freq and "M" in freq.upper()) else None
    out = Path(args.output_dir)
    ensure_dir(out)

    # dec = save_decomposition(ts, period, args.model, out / "seasonal_decompose.png")
    save_acf_pacf(ts, out / "acf_pacf.png")

    adf = run_adf(ts)
    kpss_l = run_kpss(ts, "c")
    kpss_t = run_kpss(ts, "ct")
    pd.DataFrame([adf | kpss_l | kpss_t]).to_csv(
        out / "adf_kpss_results.csv", index=False
    )

    meta = {
        "name": args.target,
        "n": len(ts.dropna()),
        "start": ts.index.min(),
        "end": ts.index.max(),
        "freq": freq,
    }
    write_report(out / "data_profile.md", meta, adf, kpss_l, kpss_t)
    log.info(f"EDA complete. Results saved in {out}/")


if __name__ == "__main__":
    main()
