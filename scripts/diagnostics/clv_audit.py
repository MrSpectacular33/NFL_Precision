import os, argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", default="reports/diagnostics/clv_summary.csv")
    ap.add_argument("--market", default="spread", choices=["spread","total"])
    ap.add_argument("--regular-season-only", action="store_true")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.infile)

    need = {"season","week","bet_type","game_id","bet_line"}
    miss = need - set(df.columns)
    if miss: raise SystemExit(f"Missing required columns: {miss}")

    d = df.copy()
    d["season"] = d["season"].astype(int)
    d["week"]   = d["week"].astype(int)
    d = d[d["bet_type"].astype(str).str.lower()==args.market]
    if args.regular_season_only:
        d = d[d["week"]<=18]

    if "close_line" not in d.columns:
        pd.DataFrame([{"note":"no close_line column present; nothing to compute"}]).to_csv(args.out, index=False)
        print(f"[warn] No close_line column; wrote note to {args.out}")
        return

    d = d[~d["close_line"].isna()].copy()
    if d.empty:
        pd.DataFrame([{"note":"close_line present but all NaN"}]).to_csv(args.out, index=False)
        print(f"[warn] close_line empty; wrote note to {args.out}")
        return

    d["clv_signed"] = d["close_line"].astype(float) - d["bet_line"].astype(float)
    d["clv_abs"]    = d["clv_signed"].abs()
    d["beat_close"] = (np.sign(d["bet_line"]) == np.sign(d["close_line"])) & (d["clv_abs"]>0) | (d["bet_line"]==d["close_line"])
    d["beat_close_0p5"] = d["clv_abs"] >= 0.5
    d["beat_close_1p0"] = d["clv_abs"] >= 1.0

    by_season = d.groupby("season").agg(
        n=("clv_signed","size"),
        clv_mean_signed=("clv_signed","mean"),
        clv_median_signed=("clv_signed","median"),
        clv_mean_abs=("clv_abs","mean"),
        beat_close_rate=("beat_close","mean"),
        beat_close_rate_0p5=("beat_close_0p5","mean"),
        beat_close_rate_1p0=("beat_close_1p0","mean"),
    ).reset_index()

    overall = pd.DataFrame([{
        "season":"ALL",
        "n": int(len(d)),
        "clv_mean_signed": float(d["clv_signed"].mean()),
        "clv_median_signed": float(d["clv_signed"].median()),
        "clv_mean_abs": float(d["clv_abs"].mean()),
        "beat_close_rate": float(d["beat_close"].mean()),
        "beat_close_rate_0p5": float(d["beat_close_0p5"].mean()),
        "beat_close_rate_1p0": float(d["beat_close_1p0"].mean()),
    }])

    out = pd.concat([by_season, overall], ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()
