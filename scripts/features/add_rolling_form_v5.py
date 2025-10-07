import os, numpy as np, pandas as pd

INP = "data/processed/team_games_features_v4.parquet"
OUT = "data/processed/_tmp_v5_roll.parquet"

def roll_cols(df, cols, wins=(3,5)):
    df = df.sort_values(["season","team","week"]).copy()
    for c in cols:
        s = df[c].astype(float)
        s = s.groupby([df["season"], df["team"]]).shift(1)  # PAST ONLY
        for w in wins:
            df[f"{c}_roll{w}"] = (
                s.groupby([df["season"], df["team"]]).rolling(w, min_periods=1).mean().reset_index(level=[0,1], drop=True)
            )
        # to-date (season) mean up to prev week
        df[f"{c}_to_date"] = s.groupby([df["season"], df["team"]]).expanding(min_periods=1).mean().reset_index(level=[0,1], drop=True)
    return df

def main():
    if not os.path.exists(INP): raise SystemExit(f"Missing {INP}")
    df = pd.read_parquet(INP).copy()
    num = []
    for c in ["off_epa","def_epa","success_rate_off","success_rate_def","pace_seconds","proe","epa_diff","sr_diff"]:
        if c in df.columns: num.append(c)
    if not num: raise SystemExit("No numeric base cols found for rolling form.")

    out = roll_cols(df, num, wins=(3,5))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[OK] wrote {OUT} (rows: {len(out)})")
if __name__=="__main__": main()
