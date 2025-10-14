# scripts/deploy/apply_calibration.py  (LF)
import pandas as pd, numpy as np, json, sys, os

IN_PREDS = r"reports/risk/preds_final_for_kelly.csv"
CAL_MAP  = r"configs/calibration/calibration_map.json"
OUT      = r"reports/calibration/preds_with_pcal.csv"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

def interp_map(x, bins, vals):
    x = np.asarray(x, dtype=float)
    return np.interp(x, bins, vals)

def main():
    preds = pd.read_csv(IN_PREDS)
    if "p_blend" not in preds.columns:
        sys.exit("preds missing p_blend")
    # keep one row per bet_id (prefer higher edge when available)
    if "edge" in preds.columns:
        preds = preds.sort_values(["bet_id","edge"], ascending=[True, False]).drop_duplicates("bet_id", keep="first")
    else:
        preds = preds.drop_duplicates("bet_id", keep="first")

    calib = json.load(open(CAL_MAP,"r"))
    bins = np.array(calib["bins"], dtype=float)
    vals = np.array(calib["map"], dtype=float)
    preds["p_cal"] = interp_map(preds["p_blend"].clip(1e-6, 1-1e-6), bins, vals).clip(1e-6, 1-1e-6)
    preds.to_csv(OUT, index=False)
    print(json.dumps({"rows": int(len(preds)), "wrote": OUT}))
if __name__ == "__main__":
    main()