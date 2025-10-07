import os, numpy as np, pandas as pd

INP  = "data/processed/_tmp_v5_roll.parquet"
OUT  = "data/processed/_tmp_v5_sched.parquet"

def main():
    if not os.path.exists(INP): raise SystemExit(f"Missing {INP}")
    df = pd.read_parquet(INP).copy()
    # assume these cols exist from v4 builder
    for k in ("season","week","team","is_home"):
        if k not in df.columns: df[k]=np.nan
    df["season"]=df["season"].astype(int, errors="ignore")
    df["week"]=df["week"].astype(int, errors="ignore")
    df["is_home"] = (df["is_home"].astype(int, errors="ignore")>0).astype(int)

    # prior game flags (same team)
    df = df.sort_values(["season","team","week"]).copy()
    prev_home = df.groupby(["season","team"])["is_home"].shift(1)
    df["b2b_road"] = ((prev_home==0) & (df["is_home"]==0)).astype(int)

    # rest-derived flags (v4 had rest_days_diff; if missing, approximate by week gaps)
    if "rest_days_diff" in df.columns:
        rest = df["rest_days_diff"].astype(float)
        df["bye_flag"]     = (rest >= 10).astype(int)
        df["short_week"]   = (rest <= 5).astype(int)
    else:
        # naive: week gap 2 => bye, 0 => short
        wk_gap = df["week"] - df.groupby(["season","team"])["week"].shift(1)
        df["bye_flag"]   = (wk_gap>=2).astype(int)
        df["short_week"] = (wk_gap==0).astype(int)

    df.to_parquet(OUT, index=False)
    print(f"[OK] wrote {OUT} (rows: {len(df)})")
if __name__=="__main__": main()
