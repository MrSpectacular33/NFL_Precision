import pandas as pd
df = pd.read_parquet("data/processed/team_games_features_lag.parquet")
cols = list(df.columns)
print("[lag] cols:", len(cols))
print("has margin:", "margin" in df.columns, " non-null:", df["margin"].notna().sum() if "margin" in df.columns else 0)
print("has total_points:", "total_points" in df.columns, " non-null:", df["total_points"].notna().sum() if "total_points" in df.columns else 0)
print(df.head(3)[["game_id","team","opponent"] + [c for c in ["margin","total_points"] if c in df.columns]])
