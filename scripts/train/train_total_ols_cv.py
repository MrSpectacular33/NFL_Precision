import json, pathlib, sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error

def debug(msg):
    print(f"[total_ols_cv] {msg}", flush=True)

def build_X(df, xcols):
    X = df.reindex(columns=xcols, fill_value=np.nan).copy()
    for c in ("is_home","is_divisional","short_week"):
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    if "rest_days" in X:
        X["rest_days"] = pd.to_numeric(X["rest_days"], errors="coerce").fillna(7.0)
    if "market_total" in X:
        mt = pd.to_numeric(df.get("market_total", pd.Series(index=df.index)), errors="coerce")
        season = pd.to_numeric(df.get("season", pd.Series(index=df.index)), errors="coerce")
        season_mean = mt.groupby(season).transform("mean")
        X["market_total"] = pd.to_numeric(X["market_total"], errors="coerce").fillna(season_mean)
    for c in X.columns:
        if c not in ("is_home","is_divisional","short_week","rest_days","market_total"):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def main():
    df = pd.read_parquet("data/processed/team_games_features_lag.parquet")
    XCOLS = [
        "pf_avg3","pa_avg3","tot_avg3",
        "pf_avg8","pa_avg8","tot_avg8",
        "delta_pf_avg3","delta_pa_avg3","delta_tot_avg3",
        "delta_pf_avg8","delta_pa_avg8","delta_tot_avg8",
        "is_home","is_divisional","rest_days","short_week",
        "market_total",
    ]
    debug("building design matrix")
    X = build_X(df, XCOLS)
    y = pd.to_numeric(df.get("total_points"), errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    m = X.notna().all(axis=1) & y.notna()
    n_before = len(X)
    X, y = X[m].astype(float), y[m].astype(float)
    debug(f"mask kept {len(X)}/{n_before} rows")

    if len(X) == 0:
        raise SystemExit("[FATAL] No rows left after NA filtering. Check feature columns and fills.")

    groups_raw = pd.to_numeric(df.loc[m, "season"], errors="coerce")
    n_groups = groups_raw.nunique(dropna=True)
    debug(f"unique groups (seasons): {n_groups}")

    if n_groups >= 2:
        n_splits = min(5, int(n_groups))
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups_raw.astype(int))
        debug(f"using GroupKFold with {n_splits} splits")
    else:
        n_splits = min(5, max(2, len(X)//100))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y)
        debug(f"using KFold fallback with {n_splits} splits")

    rmses, split_ct = [], 0
    for tr, te in splitter:
        split_ct += 1
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict(X.iloc[te])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], p)))

    rmse_cv = float(np.mean(rmses))
    debug(f"completed {split_ct} splits; RMSEs={list(map(lambda v: round(float(v),3), rmses))}")

    out = {"target":"total_points","n":int(len(y)),"xcols":XCOLS,"rmse_cv":rmse_cv}
    pathlib.Path("reports/cv").mkdir(parents=True, exist_ok=True)
    outp = "reports/cv/total_ols_cv.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[train_total_ols_cv] RMSE_CV={rmse_cv:.3f} n={len(y)} -> {outp}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[total_ols_cv][ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise
