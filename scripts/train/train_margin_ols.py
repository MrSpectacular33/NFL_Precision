import json, pathlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

def _as_num_series(obj, index):
    if isinstance(obj, pd.Series):
        s = obj
    else:
        s = pd.Series(obj, index=index, dtype="float64")
    return pd.to_numeric(s, errors="coerce").astype("float64")

def _reconstruct_target_from_base(df_lag: pd.DataFrame, target: str) -> pd.Series:
    # Try to reconstruct from base file by (game_id, team)
    base_path = "data/processed/team_games_features.parquet"
    try:
        base = pd.read_parquet(base_path)
    except Exception:
        return pd.Series(index=df_lag.index, dtype="float64")  # all-NaN

    base = base.copy()
    for c in ("points_for","points_against"):
        base[c] = pd.to_numeric(base.get(c), errors="coerce")

    base["margin"] = base["points_for"] - base["points_against"]
    base["total_points"] = base["points_for"] + base["points_against"]

    join_keys = ["game_id","team"]
    for c in join_keys:
        if c not in df_lag.columns or c not in base.columns:
            return pd.Series(index=df_lag.index, dtype="float64")

    merged = df_lag[join_keys].merge(
        base[join_keys + ["margin","total_points"]],
        on=join_keys, how="left"
    ).set_index(df_lag.index)

    return pd.to_numeric(merged.get(target), errors="coerce")

def ols(df: pd.DataFrame, target: str, xcols: list[str], out_path: str):
    X = df.reindex(columns=xcols, fill_value=np.nan).copy()

    # Flags/simple fills
    for c in ("is_home","is_divisional","short_week"):
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    if "rest_days" in X:
        X["rest_days"] = pd.to_numeric(X["rest_days"], errors="coerce").fillna(7.0)

    # Market spread
    if "market_spread" in X:
        ms = _as_num_series(df.get("market_spread", pd.Series(index=df.index, dtype="float64")), df.index)
        season = pd.to_numeric(df.get("season", pd.Series(index=df.index, dtype="float64")), errors="coerce")
        season_mean = ms.groupby(season).transform("mean")
        X["market_spread"] = pd.to_numeric(X["market_spread"], errors="coerce").fillna(season_mean)
        if X["market_spread"].isna().all():
            X = X.drop(columns=["market_spread"])

    # Cast remaining numeric
    for c in list(X.columns):
        if c not in ("is_home","is_divisional","short_week","rest_days"):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Drop all-NaN columns
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        print(f"[train_margin_ols] dropping all-NaN cols: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)

    # Robust NA fill (medians)
    na_before = X.isna().sum().sum()
    if na_before:
        col_medians = X.median(numeric_only=True)
        X = X.fillna(col_medians)
    print(f"[train_margin_ols] NA cells before fill: {na_before}, after fill: {X.isna().sum().sum()}")

    # Target y (with reconstruction fallback)
    y = _as_num_series(df.get(target), df.index)
    if y.notna().sum() == 0:
        print("[train_margin_ols] target missing in lag file; reconstructing from base…")
        y = _reconstruct_target_from_base(df, target)

    y = y.mask(~np.isfinite(y), np.nan)

    # Only require y to be present
    m = y.notna()
    X, y = X.loc[m].astype("float64"), y.loc[m].astype("float64")

    if len(X) == 0:
        raise SystemExit("[FATAL] No rows left after NA filtering. Check feature columns and fills (target).")

    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X).fit()
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))

    out = {
        "rmse": rmse,
        "n": int(len(y)),
        "features": list(X.columns.drop("const")),
        "coefficients": {k: float(v) for k, v in model.params.items()},
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return rmse, int(len(y))

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/team_games_features_lag.parquet")
    XCOLS = [
        "pf_avg3","pa_avg3","marg_avg3","tot_avg3",
        "pf_avg8","pa_avg8","marg_avg8","tot_avg8",
        "delta_pf_avg3","delta_pa_avg3","delta_marg_avg3","delta_tot_avg3",
        "delta_pf_avg8","delta_pa_avg8","delta_marg_avg8","delta_tot_avg8",
        "is_home","is_divisional","rest_days","short_week",
        "market_spread",
    ]
    rmse, n = ols(df, "margin", XCOLS, "reports/cv/margin_ols.json")
    print(f"[train_margin_ols] RMSE={rmse:.3f} n={n} -> reports/cv/margin_ols.json")
