from __future__ import annotations
import argparse, os, joblib, numpy as np, pandas as pd

def early_shrink(err, week, strength=0.5):
    w = max(1, int(week))
    if w >= 5: return float(err)
    return float(err) * (1.0 - strength*(5-w)/4.0)

def coalesce(df, a, b, out):
    ha, hb = a in df.columns, b in df.columns
    if ha and hb:
        df[out] = pd.to_numeric(df[a], errors="coerce").combine_first(
                  pd.to_numeric(df[b], errors="coerce"))
    elif ha:
        df[out] = pd.to_numeric(df[a], errors="coerce")
    elif hb:
        df[out] = pd.to_numeric(df[b], errors="coerce")
    else:
        raise SystemExit(f"Missing both '{a}' and '{b}' after merge.")
    return df

def predict_bundle(bundle, X):
    preds = []
    for _, mdl in bundle["models"]:
        preds.append(mdl.predict(X))
    return np.mean(preds, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outcomes-wide", default="reports/backtest/outcomes_all.csv")
    ap.add_argument("--shrink-strength", type=float, default=0.5)
    args = ap.parse_args()

    feats = pd.read_parquet(args.features_parquet)
    ow = pd.read_csv(args.outcomes_wide)[["game_id","season","week","market_spread","market_total"]].drop_duplicates(["game_id","season","week"])
    ow = ow.rename(columns={"market_spread":"closing_spread","market_total":"closing_total"})
    df = feats.merge(ow, on=["game_id","season","week"], how="inner")

    if "closing_spread" not in df.columns:
        df = coalesce(df, "closing_spread_x", "closing_spread_y", "closing_spread")
    if "closing_total"  not in df.columns:
        df = coalesce(df, "closing_total_x",  "closing_total_y",  "closing_total")

    m_b = joblib.load(os.path.join(args.models_dir,"margin_err_ens.pkl"))
    t_b = joblib.load(os.path.join(args.models_dir,"total_err_ens.pkl"))

    Xm = df.reindex(columns=m_b["features"], fill_value=np.nan).astype(float).values
    Xt = df.reindex(columns=t_b["features"], fill_value=np.nan).astype(float).values
    me = predict_bundle(m_b, Xm)
    te = predict_bundle(t_b, Xt)

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
