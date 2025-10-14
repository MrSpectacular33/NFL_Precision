# scripts/deploy/shadow_portfolio.py  (LF)
import pandas as pd, numpy as np, json, os

STAKES = r"reports/risk/stakes_optimized_latest.csv"
PCAL   = r"reports/calibration/preds_with_pcal.csv"
OUT_SH = r"reports/portfolio/shadow_portfolio.csv"
OUT_CMP= r"reports/portfolio/shadow_vs_live.csv"
OUT_SUM= r"reports/portfolio/shadow_summary.json"

os.makedirs(os.path.dirname(OUT_SH), exist_ok=True)

def american_to_decimal(x):
    x=float(x); return 1+(100/abs(x)) if x<0 else 1+(x/100)

def main():
    s = pd.read_csv(STAKES)
    # Guard: compute live_total BEFORE any merge
    live_total = float(pd.to_numeric(s["stake"], errors="coerce").fillna(0).sum())

    # Use a single p_cal per bet_id to avoid multiplicative joins
    p = pd.read_csv(PCAL)[["bet_id","p_cal"]].drop_duplicates(subset=["bet_id"], keep="first")

    need = {"bet_id","stake","price","edge","p_blend"}
    missing = need - set(s.columns)
    if missing:
        raise SystemExit(f"stakes csv missing columns: {missing}")

    m = s.merge(p, on="bet_id", how="left")

    dec = m["price"].map(american_to_decimal)
    edge_cal = m["p_cal"]*(dec-1.0) - (1.0 - m["p_cal"])

    ratio = (edge_cal.clip(lower=0) / pd.to_numeric(m["edge"], errors="coerce").replace(0,np.nan)).fillna(0.0)
    stake_cal = pd.to_numeric(m["stake"], errors="coerce").fillna(0.0) * ratio

    sh_total = float(stake_cal.sum())
    if sh_total > 0:
        stake_cal = stake_cal * (live_total / sh_total)

    out = m.copy()
    out["stake_shadow"] = np.round(stake_cal, 0)
    out["edge_cal"] = edge_cal
    out.to_csv(OUT_SH, index=False)

    cmp = out[["bet_id","game_id","market","price","stake","stake_shadow","p_blend","p_cal","edge","edge_cal"]].copy()
    cmp["delta_stake"] = cmp["stake_shadow"] - cmp["stake"]
    cmp.to_csv(OUT_CMP, index=False)

    positions_all = int(len(cmp))
    positions_live = int((pd.to_numeric(cmp["stake"], errors="coerce").fillna(0) > 0).sum())
    summary = {
        "positions_all": positions_all,
        "positions_live": positions_live,
        "live_total": float(live_total),
        "shadow_total": float(float(out["stake_shadow"].sum())),
        "stake_shift_abs": float(cmp["delta_stake"].abs().sum()),
        "pct_shift": float(cmp["delta_stake"].abs().sum()/live_total) if live_total>0 else 0.0
    }
    json.dump(summary, open(OUT_SUM,"w"))
    print(json.dumps({"wrote":[OUT_SH, OUT_CMP, OUT_SUM], **summary}))

if __name__ == "__main__":
    main()