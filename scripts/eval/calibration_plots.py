import json, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CALIB_DIR = pathlib.Path("reports/calibration")
CALIB_DIR.mkdir(parents=True, exist_ok=True)

def load_coef(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def add_const(X):
    if "const" not in X:
        X = X.copy()
        X["const"] = 1.0
    cols = ["const"] + [c for c in X.columns if c != "const"]
    return X[cols]

def predict_from_coef(df, coef_dict):
    beta = pd.Series({k: float(v) for k, v in coef_dict.items()})
    X = df.reindex(columns=[c for c in beta.index if c != "const"], fill_value=0.0).copy()
    X = add_const(X)
    for c in beta.index:
        if c not in X.columns:
            X[c] = 0.0
    X = X[beta.index]
    return (X * beta).sum(axis=1)

def reliability_curve(df, y_col, yhat_col, n_bins, title, out_png):
    d = df[[y_col, yhat_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if d.empty:
        print(f"[calibration] No rows for {title}; skipping.")
        return
    d["bin"] = pd.qcut(d[yhat_col], q=n_bins, duplicates="drop")
    grp = d.groupby("bin", observed=True).agg(
        yhat=(yhat_col, "mean"), yact=(y_col, "mean"), n=(y_col, "size")
    ).reset_index(drop=True)

    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()
    ax.plot(grp["yhat"], grp["yact"], marker="o")
    lim_min = float(min(grp["yhat"].min(), grp["yact"].min()))
    lim_max = float(max(grp["yhat"].max(), grp["yact"].max()))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
    ax.set_xlabel("Predicted (bin mean)")
    ax.set_ylabel("Actual (bin mean)")
    ax.set_title(title)
    for _, r in grp.iterrows():
        ax.annotate(str(int(r["n"])), (r["yhat"], r["yact"]), xytext=(3,3), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[calibration] wrote {out_png} (bins={len(grp)})")

def main():
    lag = pd.read_parquet("data/processed/team_games_features_lag.parquet")
    m_cfg = load_coef("reports/cv/margin_ols.json")
    t_cfg = load_coef("reports/cv/total_ols.json")

    # Keep trainer’s feature sets
    margin_cols = [
        "pf_avg3","pa_avg3","marg_avg3","tot_avg3",
        "pf_avg8","pa_avg8","marg_avg8","tot_avg8",
        "delta_pf_avg3","delta_pa_avg3","delta_marg_avg3","delta_tot_avg3",
        "delta_pf_avg8","delta_pa_avg8","delta_marg_avg8","delta_tot_avg8",
        "is_home","is_divisional","rest_days","short_week","market_spread",
    ]
    total_cols = [
        "pf_avg3","pa_avg3","tot_avg3",
        "pf_avg8","pa_avg8","tot_avg8",
        "delta_pf_avg3","delta_pa_avg3","delta_tot_avg3",
        "delta_pf_avg8","delta_pa_avg8","delta_tot_avg8",
        "is_home","is_divisional","rest_days","short_week","market_total",
    ]

    for c in ("is_home","is_divisional","short_week"):
        if c in lag: lag[c] = pd.to_numeric(lag[c], errors="coerce").fillna(0.0)
    if "rest_days" in lag: lag["rest_days"] = pd.to_numeric(lag["rest_days"], errors="coerce").fillna(7.0)

    lag["yhat_margin_ols"] = predict_from_coef(lag[margin_cols], m_cfg["coefficients"])
    lag["yhat_total_ols"]  = predict_from_coef(lag[total_cols],  t_cfg["coefficients"])

    if "margin" not in lag.columns: lag["margin"] = pd.NA
    if "total_points" not in lag.columns: lag["total_points"] = pd.NA

    reliability_curve(lag, "margin",       "yhat_margin_ols", 10, "Margin OLS calibration", CALIB_DIR / "calib_margin_ols.png")
    reliability_curve(lag, "total_points", "yhat_total_ols",  10, "Total  OLS calibration", CALIB_DIR / "calib_total_ols.png")

if __name__ == "__main__":
    main()
