from __future__ import annotations
import argparse, os, joblib, numpy as np, pandas as pd

def pick(cols, names):
    cl = {c.lower(): c for c in cols}
    for n in names:
        if n in cl: return cl[n]
    return None

def standardize_keys(df):
    df = df.copy()
    m = {c.lower(): c for c in df.columns}
    for want, opts in {
        "game_id": ["game_id"],
        "season":  ["season"],
        "week":    ["week","gameday_week","game_week"],
    }.items():
        col = None
        for o in opts:
            if o in m: col = m[o]; break
        if col and col != want:
            df = df.rename(columns={col: want})
    return df

def coalesce(df, a, b, out):
    if a in df.columns and b in df.columns:
        df[out] = df[a].where(~df[a].isna(), df[b])
        df.drop(columns=[a,b], inplace=True, errors="ignore")
    elif a in df.columns:
        df[out] = df[a]; df.drop(columns=[a], inplace=True, errors="ignore")
    elif b in df.columns:
        df[out] = df[b]; df.drop(columns=[b], inplace=True, errors="ignore")
    return df

def early_shrink(err, week, strength=0.5):
    w = max(1, int(week))
    if w >= 5: return float(err)
    return float(err) * (1.0 - strength*(5-w)/4.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True)   # v4 parquet
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outcomes-wide", default="reports/backtest/outcomes_all.csv")
    ap.add_argument("--shrink-strength", type=float, default=0.5)
    args = ap.parse_args()

    feats = pd.read_parquet(args.features_parquet)
    feats = standardize_keys(feats)

    # Prefer closers from outcomes_all.csv; else from features_v4 (market_* or closing_*)
    ow = None
    if os.path.exists(args.outcomes_wide):
        tmp = pd.read_csv(args.outcomes_wide)
        tmp = standardize_keys(tmp)
        need = ["game_id","season","week","market_spread","market_total"]
        if all(c in tmp.columns for c in need):
            ow = tmp[need].drop_duplicates(["game_id","season","week"]).rename(
                columns={"market_spread":"closing_spread","market_total":"closing_total"})
    if ow is None:
        sp = pick(feats.columns, ["closing_spread","market_spread"])
        to = pick(feats.columns, ["closing_total","market_total"])
        if sp and to:
            ow = feats[["game_id","season","week",sp,to]].drop_duplicates(["game_id","season","week"]).rename(
                columns={sp:"closing_spread", to:"closing_total"})
        else:
            raise SystemExit("No closing lines found in outcomes_all.csv or features parquet.")

    df = feats.merge(ow, on=["game_id","season","week"], how="inner")
    # If duplicates exist (e.g., feat already had closing_*), coalesce them
    for a,b,o in [("closing_spread_x","closing_spread_y","closing_spread"),
                  ("closing_total_x","closing_total_y","closing_total")]:
        if a in df.columns or b in df.columns:
            df = coalesce(df, a, b, o)

    if "closing_spread" not in df.columns or "closing_total" not in df.columns:
        raise SystemExit("Missing closing_spread/closing_total after merge in predictor.")

    me_b = joblib.load(os.path.join(args.models_dir,"margin_error_gbm.pkl"))
    te_b = joblib.load(os.path.join(args.models_dir,"total_error_gbm.pkl"))

    # Align features lists exactly
    Xm = df.reindex(columns=me_b["features"], fill_value=np.nan).astype(float).values
    Xt = df.reindex(columns=te_b["features"], fill_value=np.nan).astype(float).values
    me = me_b["gbm"].predict(Xm)
    te = te_b["gbm"].predict(Xt)

    weeks = df["week"].fillna(5).astype(int).values
    me_s = np.array([early_shrink(x, w, args.shrink_strength) for x,w in zip(me, weeks)])
    te_s = np.array([early_shrink(x, w, args.shrink_strength) for x,w in zip(te, weeks)])

    df["pred_margin_line"] = df["closing_spread"] + me_s
    df["pred_total_line"]  = df["closing_total"]  + te_s
    df["pred_margin_error"] = me_s
    df["pred_total_error"]  = te_s

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df[["game_id","season","week","pred_margin_line","pred_total_line","pred_margin_error","pred_total_error"]].to_parquet(args.out, index=False)
    print(f"[OK] wrote {args.out} (rows: {len(df)})")
if __name__ == "__main__":
    main()
