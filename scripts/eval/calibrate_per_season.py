import argparse, os
import numpy as np
import pandas as pd
from math import erf, sqrt
from sklearn.isotonic import IsotonicRegression

def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    # vectorize math.erf (NumPy doesn't always expose np.erf)
    v_erf = np.vectorize(erf)
    return 0.5*(1.0 + v_erf(x/np.sqrt(2.0)))

def fit_iso(p_raw, y, min_n=300, min_unique=10):
    """Fit isotonic p_raw -> calibrated probability; fallback to identity if too sparse."""
    p_raw = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p_raw) & np.isfinite(y)
    p_raw, y = p_raw[mask], y[mask]
    if len(p_raw) < min_n or np.unique(p_raw).size < min_unique:
        return None  # identity fallback
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(p_raw, y)
    return iso

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="reports/backtest/outcomes_long_with_pcal.csv")
    ap.add_argument("--out-long", default="reports/backtest/outcomes_long_with_pcal_season.csv")
    ap.add_argument("--reliability-out", default="reports/backtest/reliability_seasonal.csv")
    ap.add_argument("--base-prob-col", default="p_cal", help="probability column to recalibrate (default: p_cal)")
    ap.add_argument("--bins", type=int, default=8, help="quantile bins for reliability table")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    if args.base_prob_col not in df.columns:
        raise SystemExit(f"Missing column {args.base_prob_col} in {args.infile}")
    for col in ("season","bet_type","won"):
        if col not in df.columns:
            raise SystemExit(f"Missing column {col} in {args.infile}")

    df["p_in"] = df[args.base_prob_col].astype(float).clip(0.01, 0.99)

    # Per-(season, bet_type) isotonic
    p_out = np.empty(len(df), dtype=float); p_out[:] = np.nan
    groups = df.groupby(["season","bet_type"], sort=True)
    for (season, mkt), g in groups:
        iso = fit_iso(g["p_in"].values, g["won"].values, min_n=300, min_unique=10)
        if iso is None:
            # identity fallback
            p = g["p_in"].values
        else:
            p = iso.predict(g["p_in"].values)
        p_out[g.index] = np.clip(p, 0.01, 0.99)

    df["p_cal_season"] = p_out

    # Reliability per season x market (quantile bins on p_cal_season)
    rel_rows = []
    for (season, mkt), g in df.groupby(["season","bet_type"], sort=True):
        g = g.dropna(subset=["p_cal_season","won"])
        if len(g) == 0:
            continue
        try:
            g["bin"] = pd.qcut(g["p_cal_season"], q=min(args.bins, g["p_cal_season"].nunique()), duplicates="drop")
        except Exception:
            continue
        agg = (g.groupby("bin")
                .agg(p=("p_cal_season","mean"), hit=("won","mean"), n=("won","size"))
                .reset_index())
        for _, r in agg.iterrows():
            rel_rows.append({
                "season": int(season),
                "bet_type": str(mkt),
                "bin": str(r["bin"]),
                "p": float(r["p"]),
                "hit": float(r["hit"]),
                "n": int(r["n"]),
            })

    os.makedirs(os.path.dirname(args.out_long), exist_ok=True)
    df.to_csv(args.out_long, index=False)
    pd.DataFrame(rel_rows).to_csv(args.reliability_out, index=False)

    # Quick sanity print
    corr_all = float(df[["p_cal_season","won"]].corr().iloc[0,1])
    print(f"[OK] wrote {args.out_long} (rows: {len(df)})")
    print(f"[OK] wrote {args.reliability_out}")
    print(f"[INFO] corr(p_cal_season, won) = {corr_all:.4f}")

if __name__ == "__main__":
    main()
