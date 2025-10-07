#!/usr/bin/env python3
"""
bankroll_sim.py

Reads an edge sheet CSV (from edge_sheet.py) and produces a bet-sizing plan
and an expected (not realized) bankroll growth estimate using fractional Kelly.

Inputs (from edge sheet):
- p_cover, kelly_spread, edge_margin_pts (spread bets)
- p_over,  kelly_total,  edge_total_pts  (total bets)

By default assumes -110 (1.909 decimal) odds for both spread and total.
"""

import argparse
import os
import sys
import math
import json
import pandas as pd

def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    a = float(american)
    if a > 0:
        return 1.0 + (a / 100.0)
    else:
        return 1.0 + (100.0 / abs(a))

def expected_log_growth(f: float, p: float, dec_odds: float) -> float:
    """
    Expected log growth for a single bet that risks fraction f of bankroll:
    G = p*ln(1 + f*(dec_odds-1)) + (1-p)*ln(1 - f)
    Valid only for 0 < f < 1.
    """
    if f <= 0 or f >= 1:
        return float('-inf')
    edge = dec_odds - 1.0
    win_term = 1.0 + f * edge
    lose_term = 1.0 - f
    if win_term <= 0 or lose_term <= 0:
        return float('-inf')
    return p * math.log(win_term) + (1.0 - p) * math.log(lose_term)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-file", required=True, help="CSV produced by edge_sheet.py")
    ap.add_argument("--starting-bankroll", type=float, default=1000.0)
    ap.add_argument("--kelly-fraction", type=float, default=0.5, help="Fraction of Kelly to bet (e.g., 0.5 = half Kelly)")
    ap.add_argument("--min-kelly", type=float, default=0.01, help="Minimum Kelly fraction (edge_sheet field) to include")
    ap.add_argument("--max-bets", type=int, default=9999, help="Cap number of bets to include (after filtering)")
    ap.add_argument("--odds-spread", type=float, default=-110.0, help="American odds for spread bets")
    ap.add_argument("--odds-total", type=float, default=-110.0, help="American odds for total bets")
    ap.add_argument("--outdir", default="reports/bankroll")
    args = ap.parse_args()

    df = pd.read_csv(args.edges_file)
    base = os.path.splitext(os.path.basename(args.edges_file))[0]

    # Build bet candidates: one row for spread, one for total (when available)
    bets = []

    dec_spread = american_to_decimal(args.odds_spread)
    dec_total  = american_to_decimal(args.odds_total)

    def push_bet(kind, row, p_col, kelly_col, edge_pts_col, dec_odds):
        p = row.get(p_col)
        k = row.get(kelly_col)
        edge_pts = row.get(edge_pts_col)
        try:
            p = float(p)
            k = float(k)
        except Exception:
            return
        if not (0.0 <= p <= 1.0):
            return
        if k is None or k <= 0:
            return
        if k < args.min_kelly:
            return
        # fractional Kelly: stake fraction of BR
        f = args.kelly_fraction * k
        gl = expected_log_growth(f, p, dec_odds)
        bets.append({
            "game_id": row.get("game_id"),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "market": kind,  # 'spread' or 'total'
            "edge_pts": edge_pts,
            "p": p,
            "kelly": k,
            "fractional_kelly_used": args.kelly_fraction,
            "stake_frac": f,
            "decimal_odds": dec_odds,
            "exp_log_growth": gl
        })

    # Spread (cover) and Total (over)
    for _, r in df.iterrows():
        # spread side
        if "p_cover" in r and "kelly_spread" in r and "edge_margin_pts" in r:
            push_bet("spread", r, "p_cover", "kelly_spread", "edge_margin_pts", dec_spread)
        # total side (use p_over)
        if "p_over" in r and "kelly_total" in r and "edge_total_pts" in r:
            push_bet("total", r, "p_over", "kelly_total", "edge_total_pts", dec_total)

    if not bets:
        print("[bankroll_sim] No qualifying bets (check min-kelly or edge sheet fields).")
        # still write empty outputs
        os.makedirs(args.outdir, exist_ok=True)
        pd.DataFrame().to_csv(os.path.join(args.outdir, f"bankroll_plan_{base}.csv"), index=False)
        with open(os.path.join(args.outdir, f"bankroll_summary_{base}.json"), "w", encoding="utf-8") as f:
            json.dump({"note": "no bets"}, f, indent=2)
        sys.exit(0)

    # Rank by expected log growth (desc), then by stake fraction (desc)
    bets_sorted = sorted(bets, key=lambda b: (b["exp_log_growth"], b["stake_frac"]), reverse=True)
    bets_selected = bets_sorted[: args.max_bets]

    # Expected bankroll curve with sequential betting (using expected log growth additivity):
    # Expected log bankroll after N bets = log(B0) + sum(G_i)  => E[BN] = B0 * exp(sum G_i)
    sum_g = sum(b["exp_log_growth"] for b in bets_selected if math.isfinite(b["exp_log_growth"]))
    B0 = float(args.starting_bankroll)
    expected_final_bankroll = B0 * math.exp(sum_g)
    expected_roi = (expected_final_bankroll / B0) - 1.0

    # Compute dollar stakes using current bankroll * stake_frac.
    # For plan output we keep them as B0 * stake_frac (single-pass planning).
    for b in bets_selected:
        b["stake_dollars"] = round(B0 * b["stake_frac"], 2)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    plan_path = os.path.join(outdir, f"bankroll_plan_{base}.csv")
    summary_path = os.path.join(outdir, f"bankroll_summary_{base}.json")

    pd.DataFrame(bets_selected).to_csv(plan_path, index=False)

    summary = {
        "starting_bankroll": B0,
        "num_bets": len(bets_selected),
        "kelly_fraction": args.kelly_fraction,
        "min_kelly_filter": args.min_kelly,
        "odds": {
            "spread_decimal": dec_spread,
            "total_decimal": dec_total
        },
        "expected_final_bankroll": round(expected_final_bankroll, 2),
        "expected_roi": expected_roi,
        "sum_expected_log_growth": sum_g,
        "plan_csv": plan_path
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[bankroll_sim] wrote plan -> {plan_path}")
    print(f"[bankroll_sim] wrote summary -> {summary_path}")
    print(f"[bankroll_sim] Expected final bankroll: {expected_final_bankroll:.2f} (ROI {expected_roi:.2%})")

if __name__ == "__main__":
    main()
