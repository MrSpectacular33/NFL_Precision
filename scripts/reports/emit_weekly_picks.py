import pandas as pd, numpy as np
p = pd.read_csv("reports/backtest/placed_bets.csv")
# latest week by (season, week)
p["season"]=p["season"].astype(int); p["week"]=p["week"].astype(int)
key = p.sort_values(["season","week"]).tail(1)[["season","week"]].iloc[0].to_dict()
w = p[(p.season==key["season"]) & (p.week==key["week"])].copy()
w["units"] = (w["stake"] / 100).round(2)  # $100 = 1u (adjust if you prefer)
cols = ["season","week","bet_type","pick","bet_line","price","p_cal","ev","units"]
w[cols].to_csv("reports/backtest/weekly_picks.csv", index=False)
print(f"[OK] weekly_picks.csv for {key}")
