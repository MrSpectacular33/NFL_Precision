from __future__ import annotations
import argparse, os, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib

def wf_splits(df, min_train_seasons=3):
    seasons = sorted(df["season"].dropna().astype(int).unique())
    for i in range(min_train_seasons, len(seasons)):
        tr_seasons, va_season = seasons[:i], seasons[i]
        tr = df[df["season"].isin(tr_seasons)]
        va = df[df["season"].eq(va_season)]
        yield tr, va, va_season

def build_gbm():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("gbm", GradientBoostingRegressor(
            loss="huber", learning_rate=0.05, n_estimators=800,
            max_depth=3, subsample=0.8, random_state=42))
    ])

def build_hgbr():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler(with_mean=False)),  # tree-based; scaling harmless
        ("hgb", HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.08,
            max_depth=6, max_bins=255, random_state=42))
    ])

def build_ridge(alpha=1.0):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("rg",  Ridge(alpha=alpha))
    ])

def choose_features(df):
    bad = {"game_id","season","week","team","opponent","actual_margin","actual_total",
           "closing_spread","closing_total"}
    num = []
    for c in df.columns:
        if c in bad: continue
        if not pd.api.types.is_numeric_dtype(df[c]): continue
        if not df[c].notna().any(): continue
        num.append(c)
    return num[:60]

def pick(sc_df, names):
    m = {c.lower(): c for c in sc_df.columns}
    for n in names:
        if n in m: return m[n]
    return None

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True)  # v5 parquet
    ap.add_argument("--outcomes-wide", default="reports/backtest/outcomes_all.csv")
    ap.add_argument("--schedules-parquet", default="data/raw/schedules_2019_2024.parquet")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--clip-margin", type=float, default=21)
    ap.add_argument("--clip-total",  type=float, default=28)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    feats = pd.read_parquet(args.features_parquet)

    # Closing lines
    ow = pd.read_csv(args.outcomes_wide)[["game_id","season","week","market_spread","market_total"]].drop_duplicates(["game_id","season","week"])
    ow = ow.rename(columns={"market_spread":"closing_spread","market_total":"closing_total"})

    # Scores (flexible columns)
    sc = pd.read_parquet(args.schedules_parquet)
    gid  = pick(sc, ["game_id"]) or "game_id"
    seas = pick(sc, ["season"])  or "season"
    wk   = pick(sc, ["week","gameday_week","game_week"]) or "week"
    hs   = pick(sc, ["home_score","home_points","home_pts","home_final"])
    as_  = pick(sc, ["away_score","away_points","away_pts","away_final"])
    if hs is None or as_ is None:
        raise SystemExit("Schedules parquet missing home/away scores.")
    g = sc[[gid,seas,wk,hs,as_]].copy()
    g.columns = ["game_id","season","week","home_score","away_score"]
    g["actual_margin"] = g["home_score"].astype(float) - g["away_score"].astype(float)
    g["actual_total"]  = g["home_score"].astype(float) + g["away_score"].astype(float)

    # Merge
    df = feats.merge(ow, on=["game_id","season","week"], how="inner") \
              .merge(g,  on=["game_id","season","week"], how="inner")

    # Coalesce any _x/_y closers
    if "closing_spread" not in df.columns:
        df = coalesce(df, "closing_spread_x", "closing_spread_y", "closing_spread")
    if "closing_total"  not in df.columns:
        df = coalesce(df, "closing_total_x",  "closing_total_y",  "closing_total")

    # Targets
    df["margin_error"] = (df["actual_margin"] - df["closing_spread"]).clip(-args.clip_margin, args.clip_margin)
    df["total_error"]  = (df["actual_total"]  - df["closing_total"]).clip(-args.clip_total,  args.clip_total)

    feats_list = choose_features(df)

    # Ensemble per target
    bundles = {}
    for target in ["margin_error","total_error"]:
        models = [("gbm", build_gbm()), ("hgb", build_hgbr()), ("ridge", build_ridge(1.0))]
        rmses = []
        for tr, va, _ in wf_splits(df, min_train_seasons=3):
            preds_va = []
            for _, mdl in models:
                mdl.fit(tr[feats_list], tr[target])
                preds_va.append(mdl.predict(va[feats_list]))
            pred = np.mean(preds_va, axis=0)
            rmse = float(np.sqrt(np.mean((pred - va[target].values)**2)))
            rmses.append(rmse)
        rmse_cv = float(np.mean(rmses))

        # Fit on all rows
        fitted = []
        for name, mdl in models:
            fitted.append((name, mdl.fit(df[feats_list], df[target])))

        bundles[target] = {"models": fitted, "features": feats_list, "rmse_cv": rmse_cv}

    joblib.dump(bundles["margin_error"], os.path.join(args.outdir,"margin_err_ens.pkl"))
    joblib.dump(bundles["total_error"],  os.path.join(args.outdir,"total_err_ens.pkl"))
    with open(os.path.join(args.outdir,"phase6_models_summary.json"), "w") as f:
        json.dump({"features": feats_list,
                   "rmse_margin_cv": bundles["margin_error"]["rmse_cv"],
                   "rmse_total_cv":  bundles["total_error"]["rmse_cv"],
                   "n_rows": int(len(df))}, f, indent=2)
    print("[OK] trained Phase-6 Ensemble")

if __name__ == "__main__":
    main()
