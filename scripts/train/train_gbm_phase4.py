from __future__ import annotations
import argparse, os, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib

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

def choose_features(df):
    bad = {
        "game_id", "season", "week", "team", "opponent",
        "actual_margin", "actual_total",
        "closing_spread", "closing_total"
    }
    num = []
    for c in df.columns:
        if c in bad:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if not df[c].notna().any():  # drop all-NaN columns
            continue
        num.append(c)
    return num[:60]

def wf_splits(df, min_train_seasons=3):
    seasons = sorted(df["season"].dropna().astype(int).unique())
    for i in range(min_train_seasons, len(seasons)):
        tr_seasons, va_season = seasons[:i], seasons[i]
        yield df[df["season"].isin(tr_seasons)], df[df["season"].eq(va_season)], va_season

def build_gbm():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler(with_mean=True, with_std=True)),
        ("gbm", GradientBoostingRegressor(
            loss="huber", learning_rate=0.05, n_estimators=800,
            max_depth=3, subsample=0.8, random_state=42))
    ])

def build_ridge(alpha=1.0):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("rg",  Ridge(alpha=alpha))
    ])

def coalesce(df, a, b, out):
    import numpy as np
    if a in df.columns and b in df.columns:
        df[out] = df[a].where(~df[a].isna(), df[b])
        df.drop(columns=[a,b], inplace=True, errors="ignore")
    elif a in df.columns:
        df[out] = df[a]; df.drop(columns=[a], inplace=True, errors="ignore")
    elif b in df.columns:
        df[out] = df[b]; df.drop(columns=[b], inplace=True, errors="ignore")
    else:
        df[out] = np.nan
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True)
    ap.add_argument("--outcomes-wide", default="reports/backtest/outcomes_all.csv")
    ap.add_argument("--schedules-parquet", default="data/raw/schedules_2019_2024.parquet")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--clip-margin", type=float, default=21)
    ap.add_argument("--clip-total",  type=float, default=28)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Features (v4)
    feats = pd.read_parquet(args.features_parquet)
    feats = standardize_keys(feats)

    # Closers (prefer outcomes_all.csv; otherwise from feats)
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

    # Scores from schedules
    sc = pd.read_parquet(args.schedules_parquet)
    sc = standardize_keys(sc)
    hs = pick(sc.columns, ["home_score","home_points","home_pts","home_final"])
    as_ = pick(sc.columns, ["away_score","away_points","away_pts","away_final"])
    if hs is None or as_ is None:
        raise SystemExit("Schedules parquet missing score columns.")
    g = sc[["game_id","season","week",hs,as_]].copy()
    g.columns = ["game_id","season","week","home_score","away_score"]
    g["actual_margin"] = g["home_score"].astype(float) - g["away_score"].astype(float)
    g["actual_total"]  = g["home_score"].astype(float) + g["away_score"].astype(float)

    # Merge & coalesce
    df = feats.merge(ow, on=["game_id","season","week"], how="inner") \
              .merge(g,  on=["game_id","season","week"], how="inner")
    # If duplicates exist, coalesce them
    for pair in [("closing_spread_x","closing_spread_y","closing_spread"),
                 ("closing_total_x","closing_total_y","closing_total")]:
        a,b,o = pair
        if a in df.columns or b in df.columns:
            df = coalesce(df, a, b, o)

    if "closing_spread" not in df.columns or "closing_total" not in df.columns:
        raise SystemExit("Missing closing_spread/closing_total after merge.")

    # Targets
    df["margin_error"] = (df["actual_margin"] - df["closing_spread"]).clip(-args.clip_margin, args.clip_margin)
    df["total_error"]  = (df["actual_total"]  - df["closing_total"]).clip(-args.clip_total,  args.clip_total)

    feats_list = choose_features(df)

    bundles = {}
    for target in ["margin_error","total_error"]:
        pipe = build_gbm()
        rmses = []
        for tr, va, _ in wf_splits(df, min_train_seasons=3):
            pipe.fit(tr[feats_list], tr[target])
            rmse = float(np.sqrt(np.mean((pipe.predict(va[feats_list]) - va[target].values)**2)))
            rmses.append(rmse)
        rmse_cv = float(np.mean(rmses))
        ridge = build_ridge(alpha=1.0).fit(df[feats_list], df[target])  # fallback
        bundles[target] = {"gbm": pipe, "ridge": ridge, "features": feats_list, "rmse_cv": rmse_cv}

    joblib.dump(bundles["margin_error"], os.path.join(args.outdir,"margin_error_gbm.pkl"))
    joblib.dump(bundles["total_error"],  os.path.join(args.outdir,"total_error_gbm.pkl"))
    with open(os.path.join(args.outdir,"models_phase4_summary.json"), "w") as f:
        json.dump({"features": feats_list,
                   "rmse_margin_cv": bundles["margin_error"]["rmse_cv"],
                   "rmse_total_cv":  bundles["total_error"]["rmse_cv"],
                   "n_rows": int(len(df))}, f, indent=2)
    print("[OK] trained Phase-4 GBMs")
if __name__ == "__main__":
    main()
