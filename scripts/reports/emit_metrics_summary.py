# scripts/reports/emit_metrics_summary.py
import os, argparse, json
import numpy as np, pandas as pd

def brier(y, p): return float(np.mean((p - y) ** 2))

def ece(y, p, bins=10):
    # equal-frequency bins (robust to skew); duplicates="drop" handles ties
    df = pd.DataFrame({"y":y, "p":p}).dropna()
    if df.empty: return float("nan")
    df["bin"] = pd.qcut(df["p"], q=min(bins, max(2, df["p"].nunique())), duplicates="drop")
    g = df.groupby("bin", observed=True).agg(p_bar=("p","mean"), y_bar=("y","mean"), n=("y","size"))
    w = g["n"] / g["n"].sum()
    return float(np.sum(np.abs(g["p_bar"] - g["y_bar"]) * w))

def ks_stat(y, p):
    # KS = sup_x |F1(x) - F0(x)|, via sorted thresholds
    df = pd.DataFrame({"y":y, "p":p}).sort_values("p")
    pos = (df["y"]==1).to_numpy()
    n1, n0 = pos.sum(), (~pos).sum()
    if n1==0 or n0==0: return float("nan")
    cum1 = np.cumsum(pos) / n1
    cum0 = np.cumsum(~pos) / n0
    return float(np.max(np.abs(cum1 - cum0)))

def american_to_b(price):
    price=float(price)
    return (price/100.0) if price>0 else (100.0/abs(price))

def try_clv(placed_bets_csv, market_csv=None):
    """
    CLV report if we can line up closing lines.
    Returns (dict, msg). If columns are missing, returns {}, reason.
    """
    if not os.path.exists(placed_bets_csv):
        return {}, f"missing {placed_bets_csv}"
    pb = pd.read_csv(placed_bets_csv)
    if "bet_type" not in pb.columns or "bet_line" not in pb.columns:
        return {}, "placed_bets.csv lacks bet_type/bet_line"
    sp = pb[pb["bet_type"].astype(str).str.lower()=="spread"].copy()
    if sp.empty:
        return {}, "no spread bets to compute CLV"
    # Try to find a closing line to compare against
    closing_col_candidates = ["closing_spread_pick","closing_spread_team","closing_spread"]
    closing_col = None
    src = None
    if market_csv and os.path.exists(market_csv):
        mkt = pd.read_csv(market_csv)
        merge_keys = [c for c in ["game_id","season","week","bet_type","pick"] if c in sp.columns and c in mkt.columns]
        if merge_keys:
            sp = sp.merge(mkt, on=merge_keys, how="left", suffixes=("","_mkt"))
            for c in closing_col_candidates:
                if c in sp.columns:
                    closing_col = c; src = f"merged:{os.path.basename(market_csv)}"; break
    if closing_col is None:
        # fall back to column already present in placed_bets.csv
        for c in closing_col_candidates:
            if c in sp.columns:
                closing_col = c; src = "placed_bets.csv"; break
    if closing_col is None:
        return {}, "no closing spread column found"

    sp = sp.copy()
    sp["clv_pts"] = sp[closing_col] - sp["bet_line"]
    # Positive CLV means we beat the close (got a better number for our side)
    clv_mean = float(sp["clv_pts"].mean())
    clv_median = float(sp["clv_pts"].median())
    pct_positive = float((sp["clv_pts"]>0).mean()) if len(sp) else float("nan")
    by_season = sp.groupby("season", dropna=False)["clv_pts"].agg(["mean","median","count"]).reset_index().to_dict(orient="records")
    return {
        "source": src,
        "n_bets": int(len(sp)),
        "clv_mean_pts": round(clv_mean,4),
        "clv_median_pts": round(clv_median,4),
        "pct_positive": round(pct_positive,4),
        "by_season": by_season
    }, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="reports/backtest/outcomes_long_with_pcal.csv")
    ap.add_argument("--outdir", default="reports/metrics")
    ap.add_argument("--placed-bets", default="reports/backtest/placed_bets.csv")
    ap.add_argument("--market-csv", default="", help="optional CSV with closing lines to join for CLV")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--prob-col", default="p_cal")
    ap.add_argument("--label-col", default="won")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.infile)
    need = {args.prob_col, args.label_col, "bet_type"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Missing columns in {args.infile}: {miss}")

    out_rows = []
    for mkt in ("spread","total"):
        d = df[df["bet_type"].astype(str).str.lower()==mkt].copy()
        if d.empty: 
            continue
        p = d[args.prob_col].clip(0.0,1.0).to_numpy()
        y = d[args.label_col].astype(int).to_numpy()
        row = {
            "scope": mkt,
            "n": int(len(d)),
            "brier": round(brier(y,p),6),
            "ece": round(ece(y,p, bins=args.bins),6),
            "ks": round(ks_stat(y,p),6),
            "corr_p_won": round(float(pd.DataFrame({"p":p,"y":y}).corr().iloc[0,1]),6) if len(d)>1 else float("nan")
        }
        out_rows.append(row)

    # Overall row
    p = df[args.prob_col].clip(0.0,1.0).to_numpy()
    y = df[args.label_col].astype(int).to_numpy()
    overall = {
        "scope": "overall",
        "n": int(len(df)),
        "brier": round(brier(y,p),6),
        "ece": round(ece(y,p, bins=args.bins),6),
        "ks": round(ks_stat(y,p),6),
        "corr_p_won": round(float(pd.DataFrame({"p":p,"y":y}).corr().iloc[0,1]),6) if len(df)>1 else float("nan")
    }
    out_rows.append(overall)

    metrics_csv = os.path.join(args.outdir, "metrics_summary.csv")
    pd.DataFrame(out_rows).to_csv(metrics_csv, index=False)

    clv_json, msg = try_clv(args.placed_bets, args.market_csv if args.market_csv else None)
    with open(os.path.join(args.outdir,"clv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(clv_json if clv_json else {"note": msg}, f, indent=2)

    print(f"[OK] wrote {metrics_csv}")
    print(f"[OK] wrote {os.path.join(args.outdir,'clv_summary.json')}")

if __name__=="__main__":
    main()
