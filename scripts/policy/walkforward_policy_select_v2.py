import os, json, argparse
import numpy as np
import pandas as pd

def ev_from_prob(p, price=-110.0):
    """EV per $1 risked at American odds."""
    b = (price/100.0) if price > 0 else (100.0/abs(price))
    return p*b - (1-p)

def choose_grid():
    # conservative defaults; widen if you like
    ev_grid   = [0.012, 0.015, 0.018, 0.020, 0.022]
    edge_grid = [1.0, 1.2, 1.4, 1.6]
    return ev_grid, edge_grid

def score_subset(df):
    """Return proxy score: mean EV and simple ROI-proxy."""
    if len(df) == 0:
        return -1e9, -1e9
    return float(df["ev"].mean()), float(df["ev"].sum()/max(len(df),1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True,
                    help="Outcomes/placements-like CSV with season/week/edge_pts/prob column")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prob-col", default="p_cal", help="Probability column to evaluate (e.g., p_blend, p_cal_meta)")
    ap.add_argument("--market", default="spread", choices=["spread","total"])
    ap.add_argument("--price", type=float, default=-110.0)
    ap.add_argument("--regular-season-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.infile)

    need = {"season","week","bet_type","game_id","edge_pts", args.prob_col}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Missing columns in {args.infile}: {miss}")

    d = df.copy()
    d["season"] = d["season"].astype(int)
    d["week"]   = d["week"].astype(int)
    if args.regular_season_only:
        d = d[d["week"] <= 18]
    d = d[d["bet_type"].astype(str).str.lower()==args.market]

    # EV from chosen probability
    d["ev"] = ev_from_prob(d[args.prob_col].astype(float), price=args.price)
    d["abs_edge"] = d["edge_pts"].abs()

    ev_grid, edge_grid = choose_grid()

    policies = {}
    live_rows = []

    for s, ds in d.groupby("season", sort=True):
        best = {"min_ev_spread": None, "min_edge_spread": None, "train_picks": 0, "score": -1e9}
        for ev_thr in ev_grid:
            for edge_thr in edge_grid:
                cand = ds[(ds["ev"]>=ev_thr) & (ds["abs_edge"]>=edge_thr)]
                mean_ev, roi_proxy = score_subset(cand)
                score = roi_proxy  # prioritize ROI-proxy; you can blend criteria later
                if score > best["score"]:
                    best.update({
                        "min_ev_spread": float(ev_thr),
                        "min_edge_spread": float(edge_thr),
                        "train_picks": int(len(cand)),
                        "score": float(score)
                    })
        policies[str(int(s))] = {
            "min_ev_spread": best["min_ev_spread"],
            "min_edge_spread": best["min_edge_spread"],
            "train_picks": best["train_picks"],
            "prob_col": args.prob_col
        }
        live_rows.append({
            "season": int(s),
            "min_ev_spread": best["min_ev_spread"],
            "min_edge_spread": best["min_edge_spread"],
            "n_picks": best["train_picks"],
            "clv_mean_pts": "",    # placeholder; filled by clv_audit if you have closes
            "roi_proxy": best["score"]
        })

    with open(os.path.join(args.outdir, "policy_by_season.json"), "w") as f:
        json.dump(policies, f, indent=2)
    pd.DataFrame(live_rows).sort_values("season").to_csv(
        os.path.join(args.outdir, "policy_live_summary.csv"), index=False
    )
    print(f"[OK] wrote {os.path.join(args.outdir,'policy_by_season.json')}")
    print(f"[OK] wrote {os.path.join(args.outdir,'policy_live_summary.csv')}")
if __name__ == "__main__":
    main()
