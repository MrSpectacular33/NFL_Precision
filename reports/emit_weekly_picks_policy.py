# scripts/reports/emit_weekly_picks_policy.py
import os, argparse, json
import pandas as pd
from pathlib import Path

def choose_week(placed: pd.DataFrame, season: int|None, week: int|None, min_games: int = 10, regular_only: bool = True):
    p = placed.copy()
    if regular_only:
        p = p[p["week"] <= 18]
    if season is not None and week is not None:
        key = {"season": int(season), "week": int(week)}
        if len(p[(p["season"]==key["season"]) & (p["week"]==key["week"])]) > 0:
            return key
        return None
    # latest (season, week) with at least min_games
    cand = (p.groupby(["season","week"])["game_id"]
              .nunique()
              .reset_index(name="ngames")
              .sort_values(["season","week"]))
    cand = cand[cand["ngames"] >= min_games]
    if len(cand)==0:
        last = p.sort_values(["season","week"]).tail(1)[["season","week"]].iloc[0]
        return {"season": int(last["season"]), "week": int(last["week"])}
    row = cand.tail(1).iloc[0]
    return {"season": int(row["season"]), "week": int(row["week"])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placed", default="reports/backtest/placed_bets.csv")
    ap.add_argument("--policy-json", default="reports/policy/policy_by_season.json")
    ap.add_argument("--out", default="reports/backtest/weekly_picks.csv")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--regular-season-only", action="store_true")
    ap.add_argument("--min-games", type=int, default=10)
    ap.add_argument("--unit-divisor", type=float, default=100.0, help="stake / divisor = units")
    ap.add_argument("--price-col", type=str, default="price")
    ap.add_argument("--prob-col", type=str, default="p_cal")
    args = ap.parse_args()

    # load placed bets
    if not os.path.exists(args.placed):
        raise SystemExit(f"Missing placed bets: {args.placed}")
    p = pd.read_csv(args.placed)

    # types
    for col in ("season","week"):
        if col in p.columns:
            p[col] = p[col].astype(int)
        else:
            raise SystemExit(f"Missing column {col} in {args.placed}")

    # read policy
    if not os.path.exists(args.policy_json):
        raise SystemExit(f"Missing policy json: {args.policy_json}")
    with open(args.policy_json, "r") as f:
        policy = json.load(f)

    # choose week
    key = choose_week(p, args.season, args.week, args.min_games, args.regular_season_only)
    if key is None:
        raise SystemExit("No week found matching criteria.")
    w = p[(p["season"]==key["season"]) & (p["week"]==key["week"])].copy()
    if len(w)==0:
        raise SystemExit(f"No placed bets for {key}")

    # season gates from policy (fallbacks if missing)
    skey = str(key["season"])
    if skey not in policy:
        raise SystemExit(f"No policy entry for season {skey} in {args.policy_json}")
    min_ev_spread  = float(policy[skey].get("min_ev_spread", 0.015))
    min_edge_spread= float(policy[skey].get("min_edge_spread", 1.0))

    # probability & price columns
    prob_col = args.prob_col if args.prob_col in w.columns else "p_cal"
    if prob_col not in w.columns:
        raise SystemExit(f"Missing prob column {prob_col} in weekly frame.")
    price_col = args.price_col if args.price_col in w.columns else None

    # compute EV from American price (default -110 if absent)
    def american_to_b(px):
        px = float(px)
        return (px/100.0) if px>0 else (100.0/abs(px))
    if price_col is None:
        w["price"] = -110.0
    else:
        w["price"] = w[price_col]
    b = w["price"].apply(american_to_b)
    w["ev"] = w[prob_col]*b - (1 - w[prob_col])

    # filters: spreads only, policy gates
    w = w[w["bet_type"].astype(str).str.lower().eq("spread")].copy()
    w = w[(w["ev"] >= min_ev_spread) & (w["edge_pts"].abs() >= min_edge_spread)]

    if w.empty:
        # still write an empty file with header
        cols = ["season","week","bet_type","game_id","pick","bet_line","price",prob_col,"ev","units"]
        pd.DataFrame(columns=cols).to_csv(args.out, index=False)
        print(f"[OK] weekly_picks.csv for {key}  (games: 0, picks: 0)  [no selections after policy gates]")
        return

    # one pick per game: keep highest EV
    w = (w.sort_values(["game_id","ev"], ascending=[True, False])
           .drop_duplicates(subset=["game_id"], keep="first"))

    # units
    w["units"] = (w["stake"] / float(args.unit_divisor)).round(2) if "stake" in w.columns else 1.00

    # output (keep prob column name as 'p_cal' in file for downstream tools)
    if prob_col != "p_cal":
        w = w.rename(columns={prob_col: "p_cal"})

    cols = ["season","week","bet_type","game_id","pick","bet_line","price","p_cal","ev","units"]
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    w[cols].to_csv(args.out, index=False)

    print(f"[OK] weekly_picks.csv for {key}  (games: {w['game_id'].nunique()}, picks: {len(w)})")

if __name__ == "__main__":
    main()
