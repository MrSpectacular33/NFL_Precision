import os, argparse, json
import numpy as np, pandas as pd

def american_to_b(price):
    price=float(price); return (price/100.0) if price>0 else (100.0/abs(price))

def make_ev(df, prob_col="p_cal", price_col="price", default_price=-110.0):
    out = df.copy()
    if price_col not in out.columns: out[price_col] = default_price
    b = out[price_col].abs().apply(american_to_b)
    out["ev"] = out[prob_col]*b - (1 - out[prob_col])
    return out

def clv_for_spreads(df):
    cands = ["closing_spread_pick","closing_spread_team","closing_spread"]
    col=None
    for c in cands:
        if c in df.columns: col=c; break
    if col is None or "bet_line" not in df.columns: return float("nan")
    return float((df[col] - df["bet_line"]).mean()) if len(df) else float("nan")

def apply_policy(df_spread, min_ev, min_edge):
    m = (df_spread["ev"] >= min_ev) & (df_spread["edge_pts"].abs() >= min_edge)
    return df_spread[m].copy()

def season_roi(df):
    if df.empty: return 0.0
    y = df["won"].astype(int).mean()
    b = df["price"].abs().apply(american_to_b).mean()
    return float(y*b - (1-y))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="reports/backtest/outcomes_long_with_pcal.csv")
    ap.add_argument("--outdir", default="reports/policy")
    ap.add_argument("--prob-col", default="p_cal")
    ap.add_argument("--price-col", default="price")
    ap.add_argument("--season-col", default="season")
    ap.add_argument("--min-ev-grid", default="0.010,0.012,0.015,0.018,0.020")
    ap.add_argument("--min-edge-grid", default="0.8,1.0,1.2,1.4")
    ap.add_argument("--regular-season-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.infile)
    need = {args.prob_col, args.season_col, "bet_type", "won", "edge_pts"}
    miss = need - set(df.columns)
    if miss: raise SystemExit(f"Missing columns in {args.infile}: {miss}")

    if args.regular_season_only and "week" in df.columns:
        df = df[df["week"] <= 18].copy()

    df["bet_type"] = df["bet_type"].astype(str).str.lower()
    sp = df[df["bet_type"]=="spread"].copy()
    if "price" not in sp.columns: sp["price"] = -110.0
    sp = make_ev(sp, prob_col=args.prob_col, price_col=args.price_col)

    # make sure we have plain Python ints, not numpy integers
    seasons = sorted(int(x) for x in sp[args.season_col].dropna().unique())

    grid_ev   = [float(x) for x in args.min_ev_grid.split(",")]
    grid_edge = [float(x) for x in args.min_edge_grid.split(",")]

    policy: dict[int, dict] = {}
    live_rows = []

    for i, tgt in enumerate(seasons):
        season_key = int(tgt)

        if i==0:
            policy[season_key] = {"min_ev_spread": grid_ev[0], "min_edge_spread": grid_edge[0], "note":"bootstrap"}
            continue

        train = sp[sp[args.season_col] < season_key].copy()
        best = None
        for ev in grid_ev:
            for edge in grid_edge:
                pick = apply_policy(train, ev, edge)
                mean_clv = clv_for_spreads(pick)
                roi = season_roi(pick)
                if not np.isnan(mean_clv):
                    score = (1, round(mean_clv,6), round(roi,6))
                else:
                    score = (0, round(roi,6))
                cand = (score, ev, edge, len(pick))
                if best is None or cand > best: best = cand
        _, ev_star, edge_star, n_star = best
        policy[season_key] = {
            "min_ev_spread": ev_star,
            "min_edge_spread": edge_star,
            "train_picks": int(n_star)
        }

        live = apply_policy(sp[sp[args.season_col]==season_key], ev_star, edge_star)
        live_rows.append({
            "season": season_key,
            "min_ev_spread": ev_star,
            "min_edge_spread": edge_star,
            "n_picks": int(len(live)),
            "clv_mean_pts": clv_for_spreads(live),
            "roi_proxy": season_roi(live)
        })

    # Ensure JSON keys are proper ints
    policy_dump = {int(k): v for k, v in policy.items()}
    with open(os.path.join(args.outdir,"policy_by_season.json"), "w", encoding="utf-8") as f:
        json.dump(policy_dump, f, indent=2)

    out_csv = os.path.join(args.outdir,"policy_live_summary.csv")
    pd.DataFrame(live_rows).to_csv(out_csv, index=False)
    print(f"[OK] wrote {os.path.join(args.outdir,'policy_by_season.json')}")
    print(f"[OK] wrote {out_csv}")

if __name__=="__main__":
    main()
