#!/usr/bin/env python3
"""
edge_buckets.py — summarize model edges into reliability buckets.

Inputs (via --edges-dir):
  - CSVs produced by edge_sheet.py such as:
      edges_all_s2019.csv
      edges_all_s2020.csv
      edges_s2025_w4_mkt_home.csv
      edges_s2025_w4_nomkt_both.csv
  Each row should have (when present):
      game_id, team, edge_margin_pts, edge_total_pts,
      p_cover, p_over, kelly_spread, kelly_total,
      edge_spread_ev, edge_total_ev
  (If *_ev columns are missing, EV gating is skipped safely.)

Outputs (to --outdir):
  - spread_buckets.csv
  - total_buckets.csv
  - (optional) outcomes_all.csv when --export-outcomes provided

Notes
- Buckets are based on absolute model edge (pts): [0-0.5), [0.5-1), ... [5+).
- ROI columns are *expected* ROI using model probabilities and the price (American).
- Empirical results are not required. (plot_buckets.py can still chart expected curves.)
"""

import argparse
import glob
import math
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd


# ------------------------
# Helpers
# ------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def american_to_payout(price: float) -> float:
    """Return net payout per unit stake (e.g. -110 -> 0.9091)."""
    if price < 0:
        return 100.0 / abs(price)
    return price / 100.0


def to_bucket_labels() -> List[str]:
    # Human-friendly ranges
    labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3",
              "3-3.5", "3.5-4", "4-5", "5+"]
    return labels


def bucketize_edges(abs_edge: pd.Series) -> pd.Series:
    # Bin edges: [0,0.5), [0.5,1), ... [4,5), [5, inf)
    bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, np.inf]
    labels = to_bucket_labels()
    return pd.cut(abs_edge, bins=bins, labels=labels, right=False, include_lowest=True)


def parse_season_week_from_game_id(game_id: str) -> (Optional[int], Optional[int]):
    # Expect formats like "2019_10_ARI_TB" or "2024_01_KC_BAL"
    try:
        # Split at underscores; first two tokens should be season and week
        parts = str(game_id).split("_")
        season = int(parts[0]) if len(parts) > 0 else None
        week = int(parts[1]) if len(parts) > 1 else None
        return season, week
    except Exception:
        return None, None


def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_all_edges(edges_dir: str) -> pd.DataFrame:
    patterns = [
        os.path.join(edges_dir, "edges_all_s*.csv"),
        os.path.join(edges_dir, "edges_s*_w*_*.csv"),
        os.path.join(edges_dir, "edges_*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No edge CSVs found in {edges_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] failed to read {f}: {e}")
    if not dfs:
        raise RuntimeError("No edge rows loaded (all reads failed).")
    out = pd.concat(dfs, ignore_index=True)

    # normalize expected cols (create if missing)
    for col in [
        "edge_margin_pts", "edge_total_pts",
        "p_cover", "p_over",
        "kelly_spread", "kelly_total",
        "edge_spread_ev", "edge_total_ev",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    # coerce types
    for col in ["edge_margin_pts", "edge_total_pts", "p_cover", "p_over",
                "kelly_spread", "kelly_total", "edge_spread_ev", "edge_total_ev"]:
        out[col] = safe_to_numeric(out[col])

    # Always keep game_id/team if present
    if "game_id" not in out.columns:
        out["game_id"] = None
    if "team" not in out.columns:
        out["team"] = None

    return out


def expected_roi_from_prob(p: pd.Series, price: float) -> pd.Series:
    """Expected ROI per 1u stake using model probability p and American odds."""
    payout = american_to_payout(price)
    # ROI = p * payout - (1 - p)
    return p * payout - (1.0 - p)


def summarize_side(df: pd.DataFrame, side: str, price: float) -> pd.DataFrame:
    """
    Summarize spread or total into buckets using model probabilities only.
    side = "spread" uses columns: edge_margin_pts, p_cover
    side = "total"  uses columns: edge_total_pts,  p_over
    """
    if side == "spread":
        edge_col = "edge_margin_pts"
        p_col = "p_cover"
    else:
        edge_col = "edge_total_pts"
        p_col = "p_over"

    if df.empty:
        # Return empty table with expected columns
        return pd.DataFrame(columns=[
            "bucket", "n_bets", "avg_edge", "model_p",
            "emp_p", "push_rate", "roi_flat_1u"
        ])

    work = df[[edge_col, p_col]].copy()
    work["abs_edge"] = work[edge_col].abs()
    work["bucket"] = bucketize_edges(work["abs_edge"])
    work["model_p"] = work[p_col]
    work["roi_flat_1u"] = expected_roi_from_prob(work["model_p"], price)

    grp = work.groupby("bucket", dropna=False)

    out = pd.DataFrame({
        "n_bets": grp.size(),
        "avg_edge": grp["abs_edge"].mean(),
        "model_p": grp["model_p"].mean(),
        # Placeholders (no empirical results in this script)
        "emp_p": np.nan,
        "push_rate": np.nan,
        "roi_flat_1u": grp["roi_flat_1u"].mean(),
    }).reset_index()

    # ensure sorting by bucket order
    labels = to_bucket_labels()
    cat = pd.Categorical(out["bucket"], categories=labels, ordered=True)
    out = out.assign(bucket=cat).sort_values("bucket").reset_index(drop=True)
    return out


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-dir", default="reports/backtest", help="Directory containing edges CSVs.")
    ap.add_argument("--games-parquet", default="data/processed/team_games_features.parquet",
                    help="(Optional/unused for expected ROI) Team games parquet; not required.")
    ap.add_argument("--outdir", default="reports/backtest", help="Output directory.")
    ap.add_argument("--price", type=float, default=-110, help="American odds used for ROI calc (default -110).")
    ap.add_argument("--min-edge", type=float, default=0.0, help="Minimum absolute edge (pts) to include.")
    ap.add_argument("--use-ev", action="store_true",
                    help="If set, keep only rows with positive edge_*_ev when available.")
    ap.add_argument("--export-outcomes", default=None,
                    help="If set, write per-bet rows to this CSV (expected-only fields).")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # Load & normalize
    edges = read_all_edges(args.edges_dir)

    # Optional EV gating
    if args.use_ev:
        # Keep if spread EV>0 or total EV>0 when those columns exist; otherwise don't drop merely for NaN.
        spread_ok = (edges["edge_spread_ev"].notna() & (edges["edge_spread_ev"] > 0))
        total_ok = (edges["edge_total_ev"].notna() & (edges["edge_total_ev"] > 0))
        # If both EVs are NaN, keep the row (we don't know). If at least one exists, require >0 for at least one.
        both_nan = edges["edge_spread_ev"].isna() & edges["edge_total_ev"].isna()
        ev_mask = both_nan | spread_ok | total_ok
        edges = edges[ev_mask].copy()

    # Apply min-edge filter independently for spread vs total when summarizing.
    # We'll create filtered views for each side.
    spread_mask = edges["edge_margin_pts"].abs() >= float(args.min_edge)
    total_mask = edges["edge_total_pts"].abs() >= float(args.min_edge)

    spread_view = edges.loc[spread_mask, :].copy()
    total_view = edges.loc[total_mask, :].copy()

    # Summaries (expected ROI only)
    spread_tbl = summarize_side(spread_view, side="spread", price=args.price)
    total_tbl = summarize_side(total_view, side="total", price=args.price)

    spread_path = os.path.join(args.outdir, "spread_buckets.csv")
    total_path = os.path.join(args.outdir, "total_buckets.csv")
    spread_tbl.to_csv(spread_path, index=False)
    total_tbl.to_csv(total_path, index=False)

    print(f"[OK] wrote -> {spread_path} (rows considered: {len(spread_view)})")
    print(f"[OK] wrote -> {total_path} (rows considered: {len(total_view)})")

    # Optional outcomes export (expected-only fields, no realized results)
    if args.export_outcomes:
        out = edges.copy()

        # parse season/week from game_id if possible
        seasons = []
        weeks = []
        for gid in out.get("game_id", []):
            s, w = parse_season_week_from_game_id(gid)
            seasons.append(s)
            weeks.append(w)
        out["season"] = seasons
        out["week"] = weeks

        # keep a tidy subset likely useful for downstream checks
        keep_cols = [
            "season", "week", "game_id", "team", "opponent", "is_home",
            "__source_file",
            "market_spread", "market_total",
            "edge_margin_pts", "edge_total_pts",
            "p_cover", "p_over",
            "kelly_spread", "kelly_total",
            "edge_spread_ev", "edge_total_ev",
        ]
        for c in keep_cols:
            if c not in out.columns:
                out[c] = np.nan

        out = out[keep_cols]
        out_path = os.path.join(args.outdir, os.path.basename(args.export_outcomes))
        out.to_csv(out_path, index=False)
        print(f"[OK] wrote -> {out_path}")


if __name__ == "__main__":
    main()
