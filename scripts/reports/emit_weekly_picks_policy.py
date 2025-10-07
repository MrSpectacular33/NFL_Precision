import os, json, argparse, pandas as pd

def choose_week(df, season=None, week=None, min_games=8, regular_only=False):
    d = df.copy()
    if regular_only:
        d = d[d["week"].astype(int) <= 18]
    if season is not None and week is not None:
        key = {"season": int(season), "week": int(week)}
        hit = d[(d["season"].astype(int)==key["season"]) & (d["week"].astype(int)==key["week"])]
        return key if len(hit) else None
    # latest (season,week) with at least min_games
    cand = (d.groupby(["season","week"])["game_id"]
              .nunique()
              .reset_index(name="ngames")
              .sort_values(["season","week"]))
    cand = cand[cand["ngames"]>=min_games]
    if len(cand)==0:
        # fall back to absolute latest in the file
        last = d.sort_values(["season","week"]).tail(1)[["season","week"]].iloc[0]
        return {"season": int(last["season"]), "week": int(last["week"])}
    row = cand.tail(1).iloc[0]
    return {"season": int(row["season"]), "week": int(row["week"])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placed", default="reports/backtest/placed_bets.csv")
    ap.add_argument("--policy", default="reports/policy/policy_by_season.json")
    ap.add_argument("--out",     default="reports/backtest/weekly_picks.csv")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week",   type=int)
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--regular-season-only", action="store_true")
    ap.add_argument("--unit-divisor", type=float, default=100.0, help="stake/unit divisor")
    args = ap.parse_args()

    if not os.path.exists(args.placed):
        raise SystemExit(f"Missing placed bets: {args.placed}. Run bankroll_sim first.")
    if not os.path.exists(args.policy):
        raise SystemExit(f"Missing policy file: {args.policy}. Run your policy selector first.")

    p = pd.read_csv(args.placed)
    need_cols = {"season","week","game_id","bet_type","pick","bet_line","price","p_cal","ev","stake"}
    missing = need_cols - set(p.columns)
    if missing:
        raise SystemExit(f"Missing columns in placed_bets.csv: {missing}")

    # Load policy (not strictly required to filter here since bankroll already applied gates,
    # but we read it so the script fails fast if season isn't present).
    with open(args.policy, "r") as f:
        policy = json.load(f)

    # Decide target season/week
    key = choose_week(p, args.season, args.week, args.min_games, args.regular_season_only)
    if key is None:
        raise SystemExit("No week found matching criteria.")
    if str(key["season"]) not in policy:
        # Not fatal—just warn; the bets were already filtered by bankroll gates.
        print(f"[warn] No policy entry for season {key['season']}; using placed_bets as-is.")

    # Filter and dedupe one pick per game per market (highest EV)
    w = p[(p["season"].astype(int)==key["season"]) & (p["week"].astype(int)==key["week"])].copy()
    if w.empty:
        raise SystemExit(f"No placed bets for {key}")
    w["bet_type"] = w["bet_type"].astype(str).str.lower()
    w = (w.sort_values(["bet_type","game_id","ev"], ascending=[True, True, False])
           .drop_duplicates(subset=["bet_type","game_id"], keep="first"))

    # Units from stake
    w["units"] = (w["stake"] / float(args.unit_divisor)).round(2)

    outcols = ["season","week","bet_type","game_id","pick","bet_line","price","p_cal","ev","units"]
    w[outcols].to_csv(args.out, index=False)
    print(f"[OK] weekly_picks.csv for {key}  (games: {w['game_id'].nunique()}, picks: {len(w)})")

if __name__ == "__main__":
    main()
