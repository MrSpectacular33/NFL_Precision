import os, numpy as np, pandas as pd

INP  = "data/processed/_tmp_v5_sched.parquet"
OUT  = "data/processed/_tmp_v5_weather.parquet"

def main():
    if not os.path.exists(INP): raise SystemExit(f"Missing {INP}")
    df = pd.read_parquet(INP).copy()
    wind = df.get("wind_mph", pd.Series(0, index=df.index)).astype(float).fillna(0)
    dome = df.get("is_dome", pd.Series(0, index=df.index)).astype(int)
    # effective wind only outdoor
    df["wind_outdoor"] = wind * (1 - dome)

    temp = None
    for cand in ["temp_f","temperature_f","game_temp_f","temp"]:
        if cand in df.columns:
            temp = df[cand].astype(float)
            break
    if temp is None:
        df["temp_f_eff"] = np.nan
        df["temp_bin"]   = "unk"
    else:
        df["temp_f_eff"] = temp
        bins = [-100, 25, 40, 55, 70, 85, 200]
        labels = ["frigid","cold","cool","mild","warm","hot"]
        df["temp_bin"] = pd.cut(df["temp_f_eff"], bins=bins, labels=labels)

    # a heuristic “weather headwind” score for totals (bounded)
    w = df["wind_outdoor"].clip(0,30)/30.0
    t = ( (df["temp_f_eff"]-60).abs()/40.0 ).clip(0,1) if "temp_f_eff" in df else 0
    df["weather_headwind"] = (0.7*w + 0.3*t).astype(float)

    df.to_parquet(OUT, index=False)
    print(f"[OK] wrote {OUT} (rows: {len(df)})")
if __name__=="__main__": main()
