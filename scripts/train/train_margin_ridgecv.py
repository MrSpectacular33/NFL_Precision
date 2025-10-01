import json, pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

def ridge_cv_groups(X, y, groups, alphas):
    gk = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    best_alpha, best_rmse = None, None
    for a in alphas:
        rmses = []
        for tr, te in gk.split(X, y, groups):
            pipe = Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=a, random_state=42))])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict(X.iloc[te])
            rmses.append(np.sqrt(mean_squared_error(y.iloc[te], p)))
        rmse = float(np.mean(rmses))
        if best_rmse is None or rmse < best_rmse:
            best_rmse, best_alpha = rmse, a
    return best_alpha, best_rmse

def main():
    df = pd.read_parquet("data/processed/team_games_features_lag.parquet")
    XCOLS = [
        "pf_avg3","pa_avg3","marg_avg3","tot_avg3",
        "pf_avg8","pa_avg8","marg_avg8","tot_avg8",
        "delta_pf_avg3","delta_pa_avg3","delta_marg_avg3","delta_tot_avg3",
        "delta_pf_avg8","delta_pa_avg8","delta_marg_avg8","delta_tot_avg8",
        "is_home","is_divisional","rest_days","short_week",
        "market_spread",
    ]

    X = df[XCOLS].copy()
    for c in ("is_home","is_divisional","short_week"):
        if c in X: X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    if "rest_days" in X:
        X["rest_days"] = pd.to_numeric(X["rest_days"], errors="coerce").fillna(7.0)
    if "market_spread" in X:
        ms = pd.to_numeric(df["market_spread"], errors="coerce")
        season_mean = ms.groupby(df["season"]).transform("mean")
        X["market_spread"] = pd.to_numeric(X["market_spread"], errors="coerce").fillna(season_mean)

    for c in X.columns:
        if c not in ("is_home","is_divisional","short_week","rest_days","market_spread"):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    y = pd.to_numeric(df["margin"], errors="coerce")
    X = X.replace([np.inf,-np.inf], np.nan); y = y.replace([np.inf,-np.inf], np.nan)

    m = X.notna().all(axis=1) & y.notna()
    X, y = X[m].astype(float), y[m].astype(float)
    groups = df.loc[m, "season"].astype(int)

    alphas = np.logspace(-4, 4, 41)
    a, rmse_cv = ridge_cv_groups(X, y, groups, alphas)

    pipe = Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=a, random_state=42))])
    pipe.fit(X, y)
    rmse_in = float(np.sqrt(mean_squared_error(y, pipe.predict(X))))

    pathlib.Path("reports/cv").mkdir(parents=True, exist_ok=True)
    with open("reports/cv/margin_ridgecv.json","w",encoding="utf-8") as f:
        json.dump({"target":"margin","n":int(len(y)),"xcols":XCOLS,"alpha":float(a),"rmse_cv":float(rmse_cv),"rmse_in":rmse_in}, f, indent=2)
    print(f"[train_margin_ridgecv] alpha={a:.4g} RMSE_CV={rmse_cv:.3f} n={len(y)} -> reports/cv/margin_ridgecv.json")

if __name__ == "__main__":
    main()
