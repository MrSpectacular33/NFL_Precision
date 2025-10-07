#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_week.py

Make weekly predictions for margin and total using pre-trained OLS/Ridge-style
coefficients saved as JSON. Supports two presets for coefficient files:
- "mkt"   -> reports/cv/margin_ols.json and total_ols.json
- "nomkt" -> reports/cv/margin_ols_nomkt.json and total_ols_nomkt.json

Reads design matrix rows from:
  data/processed/team_games_features_lag.parquet

Outputs:
  reports/picks/predictions_s{season}_w{week}_{coefs}_{rows}.parquet
  reports/picks/predictions_s{season}_w{week}_{coefs}_{rows}.csv

Columns in output:
  game_id, team, opponent, market_spread, pred_margin_ols,
  market_total, pred_total_ols_home, season, week
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------- Paths ----------
DATA_PARQUET = Path("data/processed/team_games_features_lag.parquet")
OUT_DIR = Path("reports/picks")
CV_DIR = Path("reports/cv")


# ---------- Utilities for loading coefficients ----------
def _normalize_coef_dict(d: Dict) -> Tuple[float, List[str], List[float]]:
    """
    Accepts multiple JSON formats and returns (intercept, features, coeffs):

    Supported formats:
    1) {"features":[...], "coefficients":[...], ...maybe metrics...}
       (optionally includes "intercept" as a feature like "const"/"(Intercept)")
    2) {"coefficients": {"feat1": val1, "feat2": val2, ..., "intercept": b0}, ...}
    3) {"feat1": val1, "feat2": val2, "intercept": b0}  (flat mapping)

    Any of "const", "Const", "(Intercept)", "Intercept" found in features will be
    promoted to the intercept and removed from features/coeffs.
    """
    intercept = 0.0
    features: List[str] = []
    coeffs: List[float] = []

    # Case 1: features + coefficients lists
    if "features" in d and "coefficients" in d and isinstance(d["coefficients"], (list, tuple)):
        feats = [str(x) for x in d["features"]]
        coef_list = [float(x) for x in d["coefficients"]]

        if len(feats) != len(coef_list):
            raise ValueError("Coef JSON mismatch: len(features) != len(coefficients)")

        # Promote any 'const'/intercept-like entry
        for name_variant in ("const", "Const", "(Intercept)", "Intercept"):
            if name_variant in feats:
                idx = feats.index(name_variant)
                intercept += float(coef_list[idx])
                del feats[idx], coef_list[idx]
                break

        # If they explicitly provided an intercept key too, add it
        if "intercept" in d and isinstance(d["intercept"], (int, float)):
            intercept += float(d["intercept"])

        features = feats
        coeffs = coef_list
        return intercept, features, coeffs

    # Case 2: coefficients is a dict mapping
    if "coefficients" in d and isinstance(d["coefficients"], dict):
        mapping = d["coefficients"]
        for k, v in mapping.items():
            if k.lower() in {"const", "intercept", "(intercept)"}:
                intercept += float(v)
            else:
                features.append(str(k))
                coeffs.append(float(v))
        # Optional explicit intercept at top-level
        if "intercept" in d and isinstance(d["intercept"], (int, float)):
            intercept += float(d["intercept"])
        return intercept, features, coeffs

    # Case 3: flat dict mapping
    # Heuristic: if it looks like metrics (rmse, r2, etc.), it's not flat mapping.
    metric_like = {"rmse", "r2", "adj_r2", "aic", "bic", "n"}
    if any(k in d for k in ("features", "coefficients")):
        # We tried list/dict formats above; if we’re here, format is unsupported.
        raise KeyError(f"Unrecognized coef file format for keys: {list(d.keys())}")

    if isinstance(d, dict) and not metric_like.issuperset(set(d.keys())):
        # Treat non-metric keys as coefficients
        for k, v in d.items():
            if k.lower() in {"const", "intercept", "(intercept)"}:
                intercept += float(v)
            else:
                # Avoid swallowing obvious metrics if included
                if k not in metric_like:
                    features.append(str(k))
                    coeffs.append(float(v))
        return intercept, features, coeffs

    raise KeyError(f"Unrecognized coef file format for keys: {list(d.keys())}")


def _read_coef(path: Path) -> Tuple[float, List[str], List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return _normalize_coef_dict(d)


def load_coefs(preset: str) -> Tuple[float, List[str], List[float], float, List[str], List[float], str, str]:
    """
    Load margin and total coefficient sets given a preset ("mkt" or "nomkt").
    Returns:
        (m_intercept, m_feats, m_coefs, t_intercept, t_feats, t_coefs, mname, tname)
    """
    if preset.lower() == "mkt":
        mfile = CV_DIR / "margin_ols.json"
        tfile = CV_DIR / "total_ols.json"
    elif preset.lower() == "nomkt":
        mfile = CV_DIR / "margin_ols_nomkt.json"
        tfile = CV_DIR / "total_ols_nomkt.json"
    else:
        raise ValueError("--coefs must be one of: mkt, nomkt")

    if not mfile.exists():
        raise FileNotFoundError(f"Missing coefficients file: {mfile}")
    if not tfile.exists():
        raise FileNotFoundError(f"Missing coefficients file: {tfile}")

    mi, mf, mc = _read_coef(mfile)
    ti, tf, tc = _read_coef(tfile)

    print(f"[predict_week] using margin coefs from: {mfile.name}")
    print(f"[predict_week] using total  coefs from: {tfile.name}")

    return mi, mf, mc, ti, tf, tc, mfile.name, tfile.name


# ---------- Prediction ----------
def _predict_linear(
    df: pd.DataFrame,
    intercept: float,
    features: List[str],
    coeffs: List[float],
    label: str,
) -> pd.Series:
    """
    Compute linear predictions with robust handling for missing features.
    Prints coverage diagnostics.
    """
    assert len(features) == len(coeffs), "features / coeffs length mismatch"

    # Determine which requested features exist
    existing = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        print(f"[predict_week:warn] missing {len(missing)} features (treated as 0): {missing}")

    coverage = 100.0 * (len(existing) / max(1, len(features)))
    tag = "margin" if "margin" in label.lower() else "total "
    print(f"[predict_week:{tag:7}] feature coverage {len(existing)}/{len(features)} = {coverage:.1f}%")

    # Build aligned matrix (fill missing with 0)
    X = pd.DataFrame(index=df.index)
    if existing:
        X = df[existing].copy()
    for mf in missing:
        X[mf] = 0.0

    # Ensure numeric dtypes
    X = X.astype(float)

    # Align coeff vector in same columns order
    coef_map = dict(zip(features, coeffs))
    coef_vec = np.array([coef_map[c] for c in X.columns], dtype=float)

    # y = b0 + X @ beta
    y = float(intercept) + X.values @ coef_vec
    return pd.Series(y, index=df.index, name=label)


# ---------- Main ----------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True, help="Target season, e.g. 2025")
    parser.add_argument("--week", type=int, required=True, help="Target week number")
    parser.add_argument("--coefs", type=str, choices=["mkt", "nomkt"], default="mkt", help="Which coefficient preset to use")
    parser.add_argument("--rows", type=str, choices=["home", "both"], default="home", help="Which rows to output (home-only or both teams)")
    args = parser.parse_args()

    # Load data
    if not DATA_PARQUET.exists():
        raise FileNotFoundError(f"Missing input design matrix: {DATA_PARQUET}")

    df = pd.read_parquet(DATA_PARQUET)

    # Filter season/week
    if "season" not in df.columns or "week" not in df.columns:
        raise KeyError("Input parquet must include 'season' and 'week' columns.")
    df = df[(df["season"] == args.season) & (df["week"] == args.week)].copy()

    if df.empty:
        raise ValueError(f"No rows found for season={args.season}, week={args.week} in {DATA_PARQUET}")

    # Keep only desired rows
    if args.rows == "home":
        if "is_home" in df.columns:
            df = df[df["is_home"] == 1].copy()
        else:
            # Fallback: if no is_home, try to dedupe by game_id with a stable order.
            df = df.sort_values(["game_id", "team"]).drop_duplicates("game_id", keep="first").copy()

    # Sanity columns for output (do not fail if absent; fill with NaN)
    for col in ["game_id", "team", "opponent", "market_spread", "market_total"]:
        if col not in df.columns:
            df[col] = np.nan

    # Load coefficients
    mi, mf, mc, ti, tf, tc, mname, tname = load_coefs(args.coefs)

    # Predictions
    y_margin = _predict_linear(df, mi, mf, mc, label="pred_margin_ols")
    y_total  = _predict_linear(df, ti, tf, tc, label="pred_total_ols_home")

    out = pd.DataFrame(
        {
            "game_id": df["game_id"].astype(str),
            "team": df["team"].astype(str),
            "opponent": df["opponent"].astype(str),
            "market_spread": pd.to_numeric(df["market_spread"], errors="coerce"),
            "pred_margin_ols": y_margin,
            "market_total": pd.to_numeric(df["market_total"], errors="coerce"),
            "pred_total_ols_home": y_total,
            "season": df["season"].astype(int),
            "week": df["week"].astype(int),
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.coefs}_{args.rows}"
    out_parquet = OUT_DIR / f"predictions_s{args.season}_w{args.week}_{suffix}.parquet"
    out_csv     = OUT_DIR / f"predictions_s{args.season}_w{args.week}_{suffix}.csv"

    out.to_parquet(out_parquet, index=False)
    out.to_csv(out_csv, index=False)

    print(
        f"[predict_week] wrote {len(out)} games -> "
        f"{out_parquet.as_posix()} and {out_csv.as_posix()}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Bubble up a readable error for PowerShell/CI logs
        print(f"ERROR: {e}", file=sys.stderr)
        raise
