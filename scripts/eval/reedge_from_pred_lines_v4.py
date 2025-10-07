import argparse, os, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

def norm_cdf_arr(z):
    try: return 0.5*(1.0 + np.erf(z/np.sqrt(2.0)))
    except Exception:
        import math
        z = np.asarray(z, float)
        return np.array([0.5*(1.0 + math.erf(float(v)/math.sqrt(2.0))) for v in z])

def fit_iso(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    m = ~(np.isnan(p)|np.isnan(y)); p, y = p[m], y[m]
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(p, y); return iso

def pick(cols, names):
    cl = {c.lower(): c for c in cols}
    for n in names:
        if n in cl: return cl[n]
    return None

def oof_isotonic(df_long, bet_type_col="bet_type", p_col="p_raw", y_col="won"):
    out = []
    seasons = sorted(df_long["season"].astype(int).unique())
    for mkt in ("spread","total"):
        dm = df_long[df_long[bet_type_col]==mkt].copy()
        # LOSO: leave-one-season-out
        dm["p_cal"] = np.nan
        for s in seasons:
            tr = dm[dm["season"]!=s]
            te = dm[dm["season"]==s]
            # If training set too small, fit global
            if len(tr)>=100:
                iso = fit_iso(tr[p_col].values, tr[y_col].values)
            else:
                iso = fit_iso(dm[p_col].values, dm[y_col].values)
            dm.loc[dm["season"]==s, "p_cal"] = np.clip(iso.transform(te[p_col].values), 0.001, 0.999)
        out.append(dm)
    return pd.concat(out, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes-wide", default="reports/backtest/outcomes_all.csv")
    ap.add_argument("--pred-parquet", required=True)
    ap.add_argument("--schedules-parquet", default="data/raw/schedules_2019_2024.parquet")
    ap.add_argument("--sigma-spread", type=float, default=14.0)
    ap.add_argument("--sigma-total",  type=float, default=11.0)
    ap.add_argument("--out-long", default="reports/backtest/outcomes_long_with_pcal.csv")
    ap.add_argument("--rels-out", default="reports/backtest/reliability_oof.csv")
    args = ap.parse_args()

    ow   = pd.read_csv(args.outcomes_wide)
    pred = pd.read_parquet(args.pred_parquet)

    m = ow.merge(pred[["game_id","season","week","pred_margin_line","pred_total_line"]],
                 on=["game_id","season","week"], how="left")

    sc = pd.read_parquet(args.schedules_parquet)
    low = {c.lower(): c for c in sc.columns}
    ht = pick(sc.columns, ["home_team","home","home_abbr"])
    at = pick(sc.columns, ["away_team","away","away_abbr"])
    hs = pick(sc.columns, ["home_score","home_points","home_pts","home_final"])
    as_ = pick(sc.columns, ["away_score","away_points","away_pts","away_final"])
    gsel = sc[[low.get("game_id","game_id"), low.get("season","season"), low.get("week","week"), ht, at, hs, as_]].copy()
    gsel.columns = ["game_id","season","week","home_team","away_team","home_score","away_score"]
    gsel["actual_margin_home_minus_away"] = gsel["home_score"].astype(float) - gsel["away_score"].astype(float)
    gsel["actual_total"] = gsel["home_score"].astype(float) + gsel["away_score"].astype(float)
    m = m.merge(gsel, on=["game_id","season","week"], how="left")

    m["team_is_home"] = (m["team"].astype(str).str.upper()==m["home_team"].astype(str).str.upper())
    m["team_market_spread"] = np.where(m["team_is_home"], m["market_spread"].astype(float), -m["market_spread"].astype(float))
    m["team_margin"] = np.where(m["team_is_home"], m["actual_margin_home_minus_away"], -m["actual_margin_home_minus_away"])

    m["edge_margin_pts_new"] = (np.where(m["team_is_home"], m["pred_margin_line"], -m["pred_margin_line"]) - m["team_market_spread"]).astype(float)
    m["edge_total_pts_new"]  = (m["pred_total_line"] - m["market_total"]).astype(float)

    m["p_cover_new"] = norm_cdf_arr(m["edge_margin_pts_new"]/args.sigma_spread)
    m["p_over_new"]  = norm_cdf_arr(m["edge_total_pts_new"]/args.sigma_total)

    rows=[]
    for _, r in m.iterrows():
        season, week, gid = int(r["season"]), int(r["week"]), r["game_id"]
        sgn = 1.0 if r["edge_margin_pts_new"]>=0 else -1.0
        pick_name = r["team"] if sgn>0 else r["opponent"]
        won_spread = 1 if (sgn*(r["team_margin"] - r["team_market_spread"]) > 0) else 0
        rows.append({"season":season,"week":week,"game_id":gid,"bet_type":"spread","bet_line":float(r["market_spread"]),
                     "pick":str(pick_name),"price":-110,"edge_pts":float(r["edge_margin_pts_new"]),
                     "p_raw":float(r["p_cover_new"]),"won":int(won_spread)})
        over = r["edge_total_pts_new"]>=0
        won_total = 1 if ((r["actual_total"] - float(r["market_total"]) > 0) == bool(over)) else 0
        rows.append({"season":season,"week":week,"game_id":gid,"bet_type":"total","bet_line":float(r["market_total"]),
                     "pick":"OVER" if over else "UNDER","price":-110,"edge_pts":float(r["edge_total_pts_new"]),
                     "p_raw":float(r["p_over_new"]),"won":int(won_total)})

    long = pd.DataFrame(rows)
    long["p_raw"] = long["p_raw"].astype(float).clip(0.001,0.999)

    # OOF isotonic
    long_cal = oof_isotonic(long)

    os.makedirs(os.path.dirname(args.out_long), exist_ok=True)
    long_cal.to_csv(args.out_long, index=False)
    print(f"[OK] wrote {args.out_long} (rows: {len(long_cal)})")

    # Reliability table (deciles) for quick QA
    out=[]
    for mkt in ("spread","total"):
        d=long_cal[long_cal["bet_type"]==mkt].copy()
        d["bin"]=pd.qcut(d["p_cal"],10,duplicates="drop")
        g=d.groupby("bin").agg(p=("p_cal","mean"), hit=("won","mean"), n=("won","size")).reset_index()
        g.insert(0,"bet_type",mkt); out.append(g)
    rel=pd.concat(out, ignore_index=True)
    rel.to_csv(args.rels_out, index=False)
    print(f"[OK] wrote {args.rels_out}")
if __name__ == "__main__":
    main()
