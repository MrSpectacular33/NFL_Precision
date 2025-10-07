# scripts/eval/edge_sheet.py
# -*- coding: utf-8 -*-
"""
Create an "edges" sheet from a predictions CSV.

Inputs (CSV):
  game_id, team, opponent, market_spread, pred_margin_ols,
  market_total, pred_total_ols_home, season, week

Outputs (CSV):
  game_id, team, opponent, season, week,
  market_spread, pred_margin_ols, edge_margin_pts, p_cover, kelly_spread, edge_spread_ev,
  market_total,  pred_total_ols_home, edge_total_pts,  p_over,  kelly_total,  edge_total_ev

Notes
- Probabilities are computed by Gaussian assumption using args.sigma_spread / args.sigma_total.
- Kelly fraction uses American odds (default -110) via (b*p - (1-p))/b, where b is profit per $1 staked.
- EV per $1 staked uses same odds and is EV = p*b - (1-p).
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build edges sheet from predictions.")
    p.add_argument(
        "--file",
        required=True,
        help="Path to predictions CSV (e.g., reports/picks/predictions_s2025_w4_mkt_home.csv)",
    )
    p.add_argument(
        "--outdir",
        default="reports/edges",
        help="Directory to write edges CSV (default: reports/edges)",
    )
    p.add_argument(
        "--sigma-spread",
        type=float,
        default=13.0,
        help="Std dev (points) for spread probability (default: 13.0)",
    )
    p.add_argument(
        "--sigma-total",
        type=float,
        default=13.0,
        help="Std dev (points) for total probability (default: 13.0)",
    )
    p.add_argument(
        "--price-spread",
        type=int,
        default=-110,
        help="American odds for spread bets (default: -110)",
    )
    p.add_argument(
        "--price-total",
        type=int,
        default=-110,
        help="American odds for totals bets (default: -110)",
    )
    return p.parse_args()


def _cdf_from_edge(edge_pts: float, sigma: float) -> float:
    """Normal CDF style (two-sided -> convert to cover/over probability)."""
    # P = 0.5 + 0.5 * erf(edge / (sigma * sqrt(2)))
    return 0.5 + 0.5 * math.erf(edge_pts / (sigma * math.sqrt(2.0)))


def _b_from_american(price: int) -> float:
    """Profit per $1 staked from American odds."""
    if price < 0:
        return 100.0 / abs(price)
    else:
        return price / 100.0


def kelly_fraction(p: float, price: int) -> float:
    """
    Kelly fraction for fraction of bankroll to stake (per $1 edge math).
    Uses b as profit per $1 staked: f* = (b*p - (1-p)) / b.
    Negative values are floored at 0 (no-bet).
    """
    b = _b_from_american(price)
    f = (b * p - (1.0 - p)) / b
    return max(0.0, float(f))


def ev_from_prob(p: float, price: int) -> float:
    """
    Expected value per $1 staked given win prob p and American odds.
    EV = p*b - (1 - p) ; for -110, b = 100/110 -> EV = 1.90909*p - 1
    """
    b = _b_from_american(price)
    return float(p) * b - (1.0 - float(p))


def _infer_outfile_name(pred_path: str) -> str:
    """
    Turn .../predictions_s2025_w4_mkt_home.csv -> edges_s2025_w4_mkt_home.csv
    Fallback to edges_<basename>.csv if pattern not found.
    """
    base = os.path.basename(pred_path)
    m = re.match(r"predictions_(.+)\.csv$", base)
    if m:
        return f"edges_{m.group(1)}.csv"
    # fallback
    stem = os.path.splitext(base)[0]
    return f"edges_{stem}.csv"


def _require_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in predictions file: {missing}")


def main() -> None:
    args = _parse_args()

    df = pd.read_csv(args.file)
    _require_cols(
        df,
        (
            "game_id",
            "team",
            "opponent",
            "market_spread",
            "pred_margin_ols",
            "market_total",
            "pred_total_ols_home",
            "season",
            "week",
        ),
    )

    # Ensure numeric
    for col in [
        "market_spread",
        "pred_margin_ols",
        "market_total",
        "pred_total_ols_home",
        "season",
        "week",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Edges in points
    df["edge_margin_pts"] = df["pred_margin_ols"] - df["market_spread"]
    df["edge_total_pts"] = df["pred_total_ols_home"] - df["market_total"]

    # Probabilities (Gaussian assumption)
    df["p_cover"] = df["edge_margin_pts"].apply(
        lambda x: _cdf_from_edge(float(x), args.sigma_spread)
    )
    df["p_over"] = df["edge_total_pts"].apply(
        lambda x: _cdf_from_edge(float(x), args.sigma_total)
    )

    # Kelly suggestions (fraction of bankroll)
    df["kelly_spread"] = df["p_cover"].apply(lambda p: kelly_fraction(p, args.price_spread))
    df["kelly_total"] = df["p_over"].apply(lambda p: kelly_fraction(p, args.price_total))

    # Expected Value per $1 staked (spread/total)
    df["edge_spread_ev"] = df["p_cover"].apply(lambda p: ev_from_prob(p, args.price_spread))
    df["edge_total_ev"] = df["p_over"].apply(lambda p: ev_from_prob(p, args.price_total))

    # Column order for readability
    out_cols = [
        "game_id",
        "team",
        "opponent",
        "season",
        "week",
        "market_spread",
        "pred_margin_ols",
        "edge_margin_pts",
        "p_cover",
        "kelly_spread",
        "edge_spread_ev",
        "market_total",
        "pred_total_ols_home",
        "edge_total_pts",
        "p_over",
        "kelly_total",
        "edge_total_ev",
    ]
    # Keep any extras at the end (if present)
    extra_cols = [c for c in df.columns if c not in out_cols]
    out_df = df[out_cols + extra_cols] if extra_cols else df[out_cols]

    # Write
    os.makedirs(args.outdir, exist_ok=True)
    out_name = _infer_outfile_name(args.file)
    out_path = os.path.join(args.outdir, out_name)
    out_df.to_csv(out_path, index=False)

    print(f"[edge_sheet] wrote -> {out_path}")


if __name__ == "__main__":
    main()
