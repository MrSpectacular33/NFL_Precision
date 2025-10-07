#!/usr/bin/env python3
"""
make_bet_slip.py

Combine one or more bankroll plan CSVs (from bankroll_sim.py) and render a
Markdown "bet slip" with the top recommended wagers.

Each plan CSV is expected to contain:
- game_id, team, opponent, market ('spread' or 'total'), edge_pts, p, kelly,
  fractional_kelly_used, stake_frac, decimal_odds, exp_log_growth, stake_dollars

Example:
  python scripts/eval/make_bet_slip.py \
    --plan-files reports/bankroll/bankroll_plan_edges_s2025_w4_mkt_home.csv \
                 reports/bankroll/bankroll_plan_edges_s2025_w4_nomkt_both.csv \
    --top 25 \
    --min-stake 5 \
    --title "Week 4 Bet Slip" \
    --outdir reports/bankroll
"""

import argparse
import os
import sys
import math
from datetime import datetime
import pandas as pd


def fmt_money(x):
    try:
        return f"${float(x):.2f}"
    except Exception:
        return ""

def fmt_pct(x):
    try:
        return f"{100.0*float(x):.1f}%"
    except Exception:
        return ""

def fmt_num(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""

def infer_pick(row):
    """
    For 'spread': pick is the 'team' listed (cover).
    For 'total' : if edge_pts > 0 => 'Over', if < 0 => 'Under', else 'Total (no edge)'.
    """
    mkt = str(row.get("market", "")).lower()
    if mkt == "spread":
        return row.get("team", "")
    if mkt == "total":
        try:
            e = float(row.get("edge_pts", 0.0))
        except Exception:
            e = 0.0
        if e > 0:
            return "Over"
        elif e < 0:
            return "Under"
        else:
            return "Total"
    return ""

def safe_str(x):
    return "" if x is None else str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-files", nargs="+", required=True, help="One or more bankroll_plan_*.csv files")
    ap.add_argument("--top", type=int, default=25, help="Max number of bets to include (after filters), across all sources")
    ap.add_argument("--min-stake", type=float, default=0.0, help="Minimum stake dollars to include")
    ap.add_argument("--title", default=None, help="Title for the bet slip (optional)")
    ap.add_argument("--outdir", default="reports/bankroll", help="Directory for the markdown output")
    args = ap.parse_args()

    plans = []
    for path in args.plan_files:
        if not os.path.exists(path):
            print(f"[make_bet_slip:warn] missing plan file: {path}")
            continue
        df = pd.read_csv(path)
        base = os.path.splitext(os.path.basename(path))[0]
        df["source"] = base
        plans.append(df)

    if not plans:
        print("[make_bet_slip] No plan files found. Nothing to do.")
        sys.exit(0)

    df_all = pd.concat(plans, ignore_index=True)

    # Basic sanity for required columns
    required_cols = [
        "game_id","team","opponent","market","edge_pts","p","kelly",
        "fractional_kelly_used","stake_frac","decimal_odds","exp_log_growth",
        "stake_dollars","source"
    ]
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        print(f"[make_bet_slip:error] Plan file(s) missing columns: {missing}")
        sys.exit(2)

    # Filter by min stake and drop non-sensical rows
    try:
        df_all["stake_dollars"] = pd.to_numeric(df_all["stake_dollars"], errors="coerce")
    except Exception:
        pass
    df_all = df_all[(df_all["stake_dollars"].fillna(0) >= args.min_stake)]

    # Add helper columns
    df_all["pick"] = df_all.apply(infer_pick, axis=1)
    df_all["matchup"] = df_all.apply(lambda r: f'{safe_str(r.get("team"))} vs {safe_str(r.get("opponent"))}', axis=1)

    # Rank — primary sort by stake_dollars desc, tie-breaker exp_log_growth desc
    df_all = df_all.sort_values(by=["stake_dollars","exp_log_growth"], ascending=[False, False])

    # Cap to top N
    df_top = df_all.head(args.top).copy()

    # Split by market
    spreads = df_top[df_top["market"].str.lower() == "spread"].copy()
    totals  = df_top[df_top["market"].str.lower() == "total"].copy()

    # Build Markdown
    os.makedirs(args.outdir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Pick a filename that includes all sources (compact)
    unique_sources = sorted(df_top["source"].unique().tolist())
    fname_tag = "_".join(s.replace("bankroll_plan_", "") for s in unique_sources)
    if len(fname_tag) > 80:
        fname_tag = "multi"
    md_path = os.path.join(args.outdir, f"bet_slip_{fname_tag}.md")

    title = args.title if args.title else "Bet Slip"
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated: {today}_")
    lines.append("")
    total_wagers = len(df_top)
    total_staked = df_top["stake_dollars"].fillna(0).sum()
    lines.append(f"**Wagers:** {total_wagers} &nbsp;&nbsp; **Total Staked:** {fmt_money(total_staked)}")
    lines.append("")
    lines.append("**Sources:** " + ", ".join(unique_sources))
    lines.append("")

    def table_block(df, subtitle):
        if df.empty:
            return [f"### {subtitle}", "", "_(none)_", ""]
        block = [f"### {subtitle}", ""]
        block.append("| Game | Pick | Kelly | Stake | Prob | Odds | Edge | Exp log growth | Src |")
        block.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in df.iterrows():
            game = f"{safe_str(r.get('game_id'))}<br>{safe_str(r.get('matchup'))}"
            pick = safe_str(r.get("pick"))
            kelly = fmt_pct(r.get("kelly"))
            stake = fmt_money(r.get("stake_dollars"))
            prob = fmt_pct(r.get("p"))
            odds = fmt_num(r.get("decimal_odds"), 3)
            edge = fmt_num(r.get("edge_pts"), 2)
            elg  = fmt_num(r.get("exp_log_growth"), 4)
            src  = safe_str(r.get("source"))
            block.append(f"| {game} | {pick} | {kelly} | {stake} | {prob} | {odds} | {edge} | {elg} | {src} |")
        block.append("")
        return block

    lines += table_block(spreads, "Spread bets")
    lines += table_block(totals,  "Totals bets")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[make_bet_slip] wrote -> {md_path}")
    print(f"[make_bet_slip] wagers={total_wagers} total_staked={fmt_money(total_staked)}")

if __name__ == "__main__":
    main()
