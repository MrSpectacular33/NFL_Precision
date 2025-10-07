import argparse, pandas as pd, numpy as np
from pathlib import Path

def choose_week(placed, season=None, week=None, min_games=10, regular_only=False):
    if regular_only:
        placed = placed[placed["week"] <= 18]
    if season is not None and week is not None:
        key = {"season": int(season), "week": int(week)}
        d = placed[(placed["season"]==key["season"]) & (placed["week"]==key["week"])]
        return key if len(d) else None
    cand = (placed.groupby(["season","week"])["game_id"]
                  .nunique()
                  .reset_index(name="ngames")
                  .sort_values(["season","week"]))
    cand = cand[cand["ngames"]>=min_games]
    if len(cand)==0:
        last = placed.sort_values(["season","week"]).tail(1)[["season","week"]].iloc[0]
        return {"season": int(last["season"]), "week": int(last["week"])}
    row = cand.tail(1).iloc[0]
    return {"season": int(row["season"]), "week": int(row["week"])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placed", default="reports/backtest/placed_bets.csv")
    ap.add_argument("--out", default="reports/backtest/weekly_picks.csv")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--min-games", type=int, default=10)
    ap.add_argument("--regular-season-only", action="store_true")
    ap.add_argument("--unit-divisor", type=float, default=100.0)
    args = ap.parse_args()

    p = pd.read_csv(args.placed)
    for col in ("season","week"): p[col] = p[col].astype(int)

    key = choose_week(p, args.season, args.week, args.min_games, args.regular_season_only)
    if key is None: raise SystemExit("No week found matching criteria.")
    w = p[(p["season"]==key["season"]) & (p["week"]==key["week"])].copy()
    if len(w)==0: raise SystemExit(f"No placed bets for {key}")

    w["bet_type"] = w["bet_type"].astype(str).str.lower()
    w = (w.sort_values(["bet_type","game_id","ev"], ascending=[True, True, False])
           .drop_duplicates(subset=["bet_type","game_id"], keep="first"))

    w["units"] = (w["stake"] / float(args.unit_divisor)).round(2)
    cols = ["season","week","bet_type","game_id","pick","bet_line","price","p_cal","ev","units"]
    w[cols].to_csv(args.out, index=False)
    print(f"[OK] weekly_picks.csv for {key}  (games: {w['game_id'].nunique()}, picks: {len(w)})")

if __name__ == "__main__":
    main()
