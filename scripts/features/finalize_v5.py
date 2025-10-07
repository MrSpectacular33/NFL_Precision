import os, numpy as np, pandas as pd

INP  = "data/processed/_tmp_v5_weather.parquet"
OUT  = "data/processed/team_games_features_v5.parquet"

def main():
    if not os.path.exists(INP): raise SystemExit(f"Missing {INP}")
    df = pd.read_parquet(INP).copy()

    if "qb_downgrade" not in df.columns:
        df["qb_downgrade"] = 0.0
    if "ol_continuity" not in df.columns:
        df["ol_continuity"] = np.nan

    # optional: qb stability if qb_name present
    if "qb_name" in df.columns:
        tmp = (df.sort_values(["season","team","week"])
                 .groupby(["season","team"])["qb_name"]
                 .apply(lambda s: s.shift(1).ne(s.shift(2)).cumsum()))  # crude changes count
        df["qb_changes_to_date"] = tmp.values
    else:
        df["qb_changes_to_date"] = np.nan

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[OK] wrote {OUT} (rows: {len(df)})")
if __name__=="__main__": main()
