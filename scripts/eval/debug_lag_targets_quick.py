import pandas as pd
df = pd.read_parquet("data/processed/team_games_features_lag.parquet")
print("has margin:", "margin" in df.columns, " non-null:", df["margin"].notna().sum())
print("has total_points:", "total_points" in df.columns, " non-null:", df["total_points"].notna().sum())
