import pandas as pd, numpy as np
df = pd.read_csv("reports/backtest/outcomes_long_with_pcal.csv")

def ev_from(p, price=-110):
    b = (price/100.0) if price>0 else (100.0/abs(price))
    return p*b - (1-p)

bet = df.copy()
bet["ev"] = ev_from(bet["p_cal"], -110)
bet = bet[(bet["ev"]>=0.015) & (bet["edge_pts"].abs()>=1.0)]
for mkt in ("spread","total"):
    d = bet[bet.bet_type.eq(mkt)]
    if len(d):
        print(mkt, "rows:", len(d), "corr:", float(d[["p_cal","won"]].corr().iloc[0,1]))
