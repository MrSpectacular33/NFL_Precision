import json, pathlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = pathlib.Path("reports/cv/margin_ols_nomkt.json")
SRC = pathlib.Path("data/processed/team_games_features_lag.parquet")

XCOLS = [
    "pf_avg3","pa_avg3","marg_avg3","tot_avg3",
    "pf_avg8","pa_avg8","marg_avg8","tot_avg8",
    "delta_pf_avg3","delta_pa_avg3","delta_marg_avg3","delta_tot_avg3",
    "delta_pf_avg8","delta_pa_avg8","delta_marg_avg8","delta_tot_avg8",
    "is_home","is_divisional","rest_days","short_week",
    # EXCLUDE market_spread
]

def main():
    df = pd.read_parquet(SRC)

    # target
    y = pd.to_numeric(df.get("margin", pd.Series(index=df.index, dtype="float64")), errors="coerce")

    # fills for flags (identical to other trainers)
    for c in ("is_home","is_divisional","short_week"):
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "rest_days" in df: df["rest_days"] = pd.to_numeric(df["rest_days"], errors="coerce").fillna(7)

    # design matrix
    X = df.reindex(columns=XCOLS).apply(pd.to_numeric, errors="coerce")

    # drop any rows with NA in X or y
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].astype("float64")
    y = y.loc[mask].astype("float64")

    # force purely numeric and add intercept
    X = sm.add_constant(X, has_constant="add")
    assert (X.dtypes != "object").sum() == len(X.columns), f"Object dtypes linger in X: {X.dtypes[X.dtypes=='object']}"

    model = sm.OLS(y.values.astype("float64"), X.values.astype("float64")).fit()

    rmse = float(np.sqrt(np.mean((model.predict(X.values) - y.values)**2)))
    coefs = dict(zip(["const"] + XCOLS, map(float, model.params)))
    out = {
        "rmse": rmse,
        "n": int(len(y)),
        "features": XCOLS,
        "coefficients": coefs,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[train_margin_ols_nomkt] RMSE={rmse:.3f} n={out['n']} -> {OUT}")

if __name__ == "__main__":
    main()
