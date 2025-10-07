from __future__ import annotations
import argparse, os, joblib, numpy as np, pandas as pd

def early_season_shrinkage(pred_error: float, week: int, strength: float = 0.5) -> float:
    w = max(1, int(week))
    if w >= 5: return float(pred_error)
    factor = 1.0 - strength * (5 - w) / 4.0
    return float(pred_error) * max(0.0, min(1.0, factor))

def load_closing_from_outcomes(path):
    ow = pd.read_csv(path)
    cols = ["game_id","season","week","market_spread","market_total"]
    for c in cols:
        if c not in ow.columns:
            raise RuntimeError(f"outcomes_all.csv missing {c}")
    base = ow[cols].drop_duplicates(["game_id","season","week"])
    return base.rename(columns={"market_spread":"closing_spread","market_total":"closing_total"})

def apply_model(bundle_path: str, df: pd.DataFrame):
    """
    Use the model's own saved feature list (exact order). Add missing cols as NaN so
    SimpleImputer in the pipeline can handle them. Return np.ndarray of predictions.
    """
    b = joblib.load(bundle_path)
    pipe = b["pipe"]
    feats = b["features"]
    # Reindex to exact features (adds missing as NaN) and keep column order
    X = df.reindex(columns=feats, fill_value=np.nan).astype(float)
    return pipe.predict(X.values)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-parquet", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outcomes-wide", dest="outcomes_wide", default="reports/backtest/outcomes_all.csv")
    ap.add_argument("--shrink-strength", type=float, default=0.5)
    args = ap.parse_args()

    feats = pd.read_parquet(args.games_parquet)
    for k in ("game_id","season","week"):
        if k not in feats.columns: raise RuntimeError("features parquet missing keys")

    closing = load_closing_from_outcomes(args.outcomes_wide)
    df = feats.merge(closing, on=["game_id","season","week"], how="inner")

    # Predict errors using model feature lists
    me = apply_model(os.path.join(args.models_dir,"margin_error_model.pkl"), df)
    te = apply_model(os.path.join(args.models_dir,"total_error_model.pkl"),  df)

    weeks = df["week"].fillna(5).astype(int).values
    me_shrunk = np.array([early_season_shrinkage(x, w, args.shrink_strength) for x, w in zip(me, weeks)])
    te_shrunk = np.array([early_season_shrinkage(x, w, args.shrink_strength) for x, w in zip(te, weeks)])

    df["pred_margin_line"] = df["closing_spread"] + me_shrunk
    df["pred_total_line"]  = df["closing_total"]  + te_shrunk
    df["pred_margin_error"] = me_shrunk
    df["pred_total_error"]  = te_shrunk

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df[["game_id","season","week","pred_margin_line","pred_total_line","pred_margin_error","pred_total_error"]].to_parquet(args.out, index=False)

if __name__ == "__main__":
    main()
