import pandas as pd, numpy as np, json, os, sys
from math import log

# Inputs
PRED_CSV = r"reports/risk/preds_final_for_kelly.csv"    # probs per bet_id (p_blend)
OUTCOME_CSV = r"data/warehouse/bet_outcomes_all.csv"    # resolved outcomes joined
OUT = "reports/qa"

def safe_log(x, eps=1e-12): return log(min(max(x, eps), 1-eps))

def main():
    if not os.path.exists(PRED_CSV):
        sys.exit(f"missing {PRED_CSV}")
    preds = pd.read_csv(PRED_CSV)

    if "bet_id" not in preds.columns: sys.exit("preds missing bet_id")
    # pick probability field
    pcol = "p_blend" if "p_blend" in preds.columns else None
    if not pcol: sys.exit("p_blend not found in preds")

    # Read cumulative outcomes (if not present, create empty shell)
    if os.path.exists(OUTCOME_CSV):
        out = pd.read_csv(OUTCOME_CSV)
    else:
        out = pd.DataFrame(columns=["bet_id","result"])

    # Build binary target: win=1, lose=0; ignore pushes/pending
    res = out[out["result"].isin(["win","lose"])].copy()
    res["y"] = (res["result"]=="win").astype(int)

    # Merge on bet_id
    m = preds.merge(res[["bet_id","y"]], on="bet_id", how="left")

    # Keep rows with y defined for calibration metrics
    k = m.dropna(subset=["y"]).copy()
    if k.empty:
        # still write empty summary so pipelines don’t fail
        os.makedirs(OUT, exist_ok=True)
        open(os.path.join(OUT,"validation_summary.json"),"w").write(json.dumps({
            "resolved": 0
        }, indent=2))
        print(json.dumps({"resolved":0}))
        return

    p = k[pcol].clip(1e-6, 1-1e-6)
    y = k["y"].astype(int)

    # Metrics
    brier = float(np.mean((p - y)**2))
    logloss = float(np.mean([- (yi*safe_log(pi) + (1-yi)*safe_log(1-pi)) for pi,yi in zip(p,y)]))
    # calibration table
    bins = np.linspace(0,1,11)
    k["bin"] = pd.cut(p, bins, include_lowest=True)
    cal = k.groupby("bin").agg(
        n=("y","size"),
        mean_pred=(pcol,"mean"),
        frac_win=("y","mean")
    ).reset_index()

    # drift: compare latest 200 vs prior 200 by Brier
    recent = k.tail(200)
    prior  = k.iloc[max(0, len(k)-400): len(k)-200]
    drift = {}
    if len(recent)>=50 and len(prior)>=50:
        drift["brier_recent"] = float(np.mean((recent[pcol].clip(1e-6,1-1e-6) - recent["y"])**2))
        drift["brier_prior"]  = float(np.mean((prior[pcol].clip(1e-6,1-1e-6) - prior["y"])**2))
        drift["brier_diff"]   = float(drift["brier_recent"] - drift["brier_prior"])

    # Save artifacts
    os.makedirs(OUT, exist_ok=True)
    cal_path = os.path.join(OUT,"calibration_table.csv")
    k_path   = os.path.join(OUT,"validation_joined_sample.csv")
    k[["bet_id",pcol,"y"]].tail(500).to_csv(k_path, index=False)
    cal.to_csv(cal_path, index=False)

    summary = {
        "resolved": int(len(k)),
        "brier": brier,
        "logloss": logloss,
        **drift
    }
    open(os.path.join(OUT,"validation_summary.json"),"w").write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
