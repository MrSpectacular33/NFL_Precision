import pandas as pd
df=pd.read_csv("reports/backtest/outcomes_long_with_pcal.csv")
print("corr(p_cal,won)=", float(df[["p_cal","won"]].corr().iloc[0,1]))
