# scripts/deploy/auto_calibrate.py  (LF)
# Learns a monotone piecewise-linear calibration from recent resolved bets.
import pandas as pd, numpy as np, json, os, datetime as dt, sys

WAREHOUSE = r"data/warehouse/bet_outcomes_all.csv"
PREDS     = r"reports/risk/preds_final_for_kelly.csv"
OUT_MAP   = r"configs/calibration/calibration_map.json"
OUT_TABLE = r"reports/calibration/calibration_latest.csv"
META      = r"reports/qa/live_drift.json"

os.makedirs(os.path.dirname(OUT_MAP), exist_ok=True)
os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)

def _coerce_result(x:str)->int:
    x=str(x).strip().lower()
    if x in ("win","won","1","true","t"): return 1
    if x in ("lose","loss","0","false","f"): return 0
    return -1

def main():
    wh = pd.read_csv(WAREHOUSE)
    preds = pd.read_csv(PREDS)

    # Need bet_id, p_blend in preds; join with outcomes
    if "bet_id" not in preds.columns: sys.exit("preds missing bet_id")
    if "p_blend" not in preds.columns: sys.exit("preds missing p_blend")

    wh["y"] = wh["result"].map(_coerce_result)
    wh = wh[wh["y"].isin([0,1])].copy()

    # Join on bet_id; de-dup by keeping max stake row if any dupes
    j = preds[["bet_id","p_blend"]].merge(wh[["bet_id","y"]], on="bet_id", how="inner").dropna()

    # If too small, bail gracefully
    if len(j) < 200:
        calib = {"bins": [0.0,1.0], "map": [0.0,1.0], "as_of": dt.datetime.utcnow().isoformat(), "samples": int(len(j))}
        json.dump(calib, open(OUT_MAP,"w"))
        pd.DataFrame({"p_hat":[0.0,1.0],"p_obs":[0.0,1.0]}).to_csv(OUT_TABLE, index=False)
        meta = {"as_of": dt.datetime.utcnow().isoformat(), "samples": int(len(j)), "note": "too few samples; identity map"}
        json.dump(meta, open(META,"w"))
        print(json.dumps({"samples": len(j), "used_bins": 1, "wrote": [OUT_MAP, OUT_TABLE, META]}))
        return

    # Quantile bins for monotone calibration
    nbins = min(12, max(6, int(np.sqrt(len(j)//25)+6)))  # adaptive 6–12
    j = j.sort_values("p_blend").reset_index(drop=True)
    j["bin"] = pd.qcut(j["p_blend"], q=nbins, duplicates="drop")
    g = j.groupby("bin", observed=True)
    p_hat = g["p_blend"].mean().to_numpy()
    p_obs = g["y"].mean().to_numpy()

    # Ensure strict monotonicity by cumulative max/min smoothing
    # enforce increasing both in x and y
    order = np.argsort(p_hat)
    p_hat = np.clip(p_hat[order], 1e-6, 1-1e-6)
    p_obs = np.clip(p_obs[order], 1e-6, 1-1e-6)
    # monotone adjust (PAV-lite)
    for i in range(1,len(p_obs)):
        if p_obs[i] < p_obs[i-1]:
            p_obs[i] = p_obs[i-1]

    # Add anchors at (0,0) and (1,1)
    bins = np.concatenate([[0.0], p_hat, [1.0]])
    maps = np.concatenate([[0.0], p_obs, [1.0]])

    # Persist
    calib = {"bins": bins.tolist(), "map": maps.tolist(), "as_of": dt.datetime.utcnow().isoformat(), "samples": int(len(j))}
    json.dump(calib, open(OUT_MAP,"w"))
    pd.DataFrame({"p_hat":bins, "p_obs":maps}).to_csv(OUT_TABLE, index=False)

    # Drift summary
    mae = float(np.mean(np.abs(p_hat - p_obs[1:-1])) if len(p_hat)>2 else 0.0)
    meta = {"as_of": dt.datetime.utcnow().isoformat(), "samples": int(len(j)), "nbins": int(nbins), "calib_mae_bins": mae}
    json.dump(meta, open(META,"w"))

    print(json.dumps({"samples": len(j), "used_bins": int(nbins), "wrote": [OUT_MAP, OUT_TABLE, META]}))

if __name__ == "__main__":
    main()