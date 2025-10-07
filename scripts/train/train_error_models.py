from __future__ import annotations
import argparse, os, json, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
import joblib

# ---------- utilities ----------
def _pick_ci(cols, names):
    cl = {c.lower(): c for c in cols}
    for n in names:
        if n in cl: return cl[n]
    return None

def load_scores_from_schedules(path):
    sc = pd.read_parquet(path)
    gid   = _pick_ci(sc.columns, ["game_id"])
    season= _pick_ci(sc.columns, ["season"])
    week  = _pick_ci(sc.columns, ["week","gameday_week","game_week"])
    hs    = _pick_ci(sc.columns, ["home_score","home_points","home_pts","home_final"])
    aas   = _pick_ci(sc.columns, ["away_score","away_points","away_pts","away_final"])
    req = [gid, season, week, hs, aas]
    if any(x is None for x in req):
        raise RuntimeError("Schedules parquet is missing game_id/season/week/home_score/away_score")
    g = sc[[gid,season,week,hs,aas]].copy()
    g.columns = ["game_id","season","week","home_score","away_score"]
    g["actual_margin"] = g["home_score"].astype(float) - g["away_score"].astype(float)
    g["actual_total"]  = g["home_score"].astype(float) + g["away_score"].astype(float)
    return g[["game_id","season","week","actual_margin","actual_total"]]

def load_closing_from_outcomes(path):
    ow = pd.read_csv(path)
    cols_needed = ["game_id","season","week","market_spread","market_total"]
    for c in cols_needed:
        if c not in ow.columns:
            raise RuntimeError(f"outcomes_all.csv missing required column: {c}")
    base = (ow[cols_needed]
            .drop_duplicates(["game_id","season","week"])
            .rename(columns={"market_spread":"closing_spread","market_total":"closing_total"}))
    return base

def choose_features(df):
    # Preferred stable signals (use intersection)
    wish = [
        "off_epa","def_epa","success_rate_off","success_rate_def",
        "pace_seconds","pass_rate_over_expected","rest_days_diff",
        "qb_downgrade","ol_continuity","wind_mph","is_dome"
    ]
    have = [c for c in wish if c in df.columns]
    # fallback to numeric features if too few
    if len(have) < 5:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        bad_prefix = ("actual_", "closing_", "market_", "p_", "edge_", "outcome", "won")
        num_cols = [c for c in num_cols if not c.lower().startswith(bad_prefix)]
        # drop identifiers
        num_cols = [c for c in num_cols if c not in ("season","week")]
        have = num_cols[:40]
    return have

def rolling_splits(df, min_train_seasons=3):
    seasons = sorted(df["season"].unique())
    for i in range(min_train_seasons, len(seasons)):
        tr = df[df["season"].isin(seasons[:i])]
        va = df[df["season"].eq(seasons[i])]
        yield tr, va, seasons[i]

def build_pipe(kind="huber", alpha=1.0, epsilon=1.35):
    est = HuberRegressor(alpha=alpha, epsilon=epsilon, fit_intercept=True) if kind=="huber" else Ridge(alpha=alpha, fit_intercept=True)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler(with_mean=True, with_std=True)),
        ("est",     est),
    ])

def train_one(df, ycol, features, clip=None, kind="huber"):
    # drop rows with NaN target
    df = df[~df[ycol].isna()].copy()
    if clip is not None:
        df[ycol] = df[ycol].clip(-clip, clip)

    # restrict to selected features, drop columns with too many NaNs
    X = df[features].copy()
    na_rate = X.isna().mean()
    keep = na_rate[na_rate <= 0.30].index.tolist()  # drop features with >30% NaN
    X = X[keep]

    # numeric safety
    X = X.replace([np.inf, -np.inf], np.nan)

    best, best_rmse = None, 1e9
    for alpha in (0.1, 0.5, 1.0, 2.0):
        pipe = build_pipe(kind=kind, alpha=alpha)
        rmses = []
        for tr, va, _ in rolling_splits(df):
            Xtr, ytr = tr[keep], tr[ycol]
            Xva, yva = va[keep], va[ycol]
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xva)
            rmses.append(float(np.sqrt(np.mean((pred - yva.values)**2))))
        rmse = float(np.mean(rmses))
        if rmse < best_rmse:
            best_rmse, best = rmse, pipe
    return best, best_rmse, keep

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-parquet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--schedules-parquet", default="data/raw/schedules_2019_2024.parquet")
    ap.add_argument("--outcomes-wide", dest="outcomes_wide", default="reports/backtest/outcomes_all.csv")
    ap.add_argument("--model", choices=["huber","ridge"], default="huber")
    ap.add_argument("--clip-margin", type=float, default=21.0)
    ap.add_argument("--clip-total",  type=float, default=28.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    feats = pd.read_parquet(args.games_parquet)
    if not all(k in feats.columns for k in ("game_id","season","week")):
        raise RuntimeError("team_games_features.parquet must include game_id, season, week")

    scores  = load_scores_from_schedules(args.schedules_parquet)
    closing = load_closing_from_outcomes(args.outcomes_wide)

    df = feats.merge(scores, on=["game_id","season","week"], how="inner") \
              .merge(closing, on=["game_id","season","week"], how="inner")

    # Targets
    df["margin_error"] = df["actual_margin"] - df["closing_spread"]
    df["total_error"]  = df["actual_total"]  - df["closing_total"]

    # Choose feature set, then train
    candidate_feats = choose_features(df)

    m_model, m_rmse, m_feats = train_one(df, "margin_error", candidate_feats, clip=args.clip_margin, kind=args.model)
    t_model, t_rmse, t_feats = train_one(df, "total_error",  candidate_feats, clip=args.clip_total,  kind=args.model)

    joblib.dump({"pipe": m_model, "features": m_feats, "target": "margin_error", "rmse_cv": m_rmse},
                os.path.join(args.outdir, "margin_error_model.pkl"))
    joblib.dump({"pipe": t_model, "features": t_feats, "target": "total_error", "rmse_cv": t_rmse},
                os.path.join(args.outdir, "total_error_model.pkl"))

    with open(os.path.join(args.outdir,"error_models_summary.json"), "w") as f:
        json.dump({
            "rmse_margin_error_cv": m_rmse,
            "rmse_total_error_cv":  t_rmse,
            "n_rows": int(len(df)),
            "features_margin": m_feats,
            "features_total":  t_feats
        }, f, indent=2)

if __name__ == "__main__":
    main()
