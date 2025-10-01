import pathlib
import pandas as pd
import numpy as np

OUT = pathlib.Path("data/processed/team_games_features_lag.parquet")

def main():
    src = pathlib.Path("data/processed/team_games_features.parquet")
    if not src.exists():
        raise SystemExit(f"missing {src}; run build_team_games.py first")
    df = pd.read_parquet(src)

    # Normalize ids/order
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"]   = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    for c in ["game_id","team","opponent"]:
        df[c] = df[c].astype("string")
    df = df.sort_values(["team","season","week","game_id"]).reset_index(drop=True)

    # Targets
    df["points_for"]     = pd.to_numeric(df["points_for"], errors="coerce")
    df["points_against"] = pd.to_numeric(df["points_against"], errors="coerce")
    df["margin"]         = df["points_for"] - df["points_against"]
    df["total_points"]   = df["points_for"] + df["points_against"]

    # Pre-game rolling means (shift(1) => no leakage)
    base_cols = {
        "pf":  "points_for",
        "pa":  "points_against",
        "marg":"margin",
        "tot": "total_points",
    }
    def add_rolls(g):
        g = g.copy()
        for short, col in base_cols.items():
            g[f"{short}_avg3"] = g[col].shift(1).rolling(window=3, min_periods=1).mean()
            g[f"{short}_avg8"] = g[col].shift(1).rolling(window=8, min_periods=1).mean()
        return g
    df = df.groupby("team", group_keys=False).apply(add_rolls)

    LAG_COLS = ["pf_avg3","pa_avg3","marg_avg3","tot_avg3",
                "pf_avg8","pa_avg8","marg_avg8","tot_avg8"]

    # Start output with ids + season/week + TARGETS + lags
    out = df[["game_id","team","opponent","season","week","margin","total_points"] + LAG_COLS].copy()

    # Opponent lags
    opp = out[["game_id","team"] + LAG_COLS].rename(columns={"team":"__opp_team", **{c:f"opp_{c}" for c in LAG_COLS}})
    out = out.merge(opp, how="left", left_on=["game_id","opponent"], right_on=["game_id","__opp_team"])
    out.drop(columns="__opp_team", inplace=True)

    # Deltas
    for c in LAG_COLS:
        out[f"delta_{c}"] = out[c] - out[f"opp_{c}"]

    # Carry flags/market anchors
    carry_cols = ["is_home","is_divisional","rest_days","short_week","market_total","market_spread"]
    carry_cols = [c for c in carry_cols if c in df.columns]
    out = out.merge(df[["game_id","team"] + carry_cols].drop_duplicates(), on=["game_id","team"], how="left")

    # Cast + tidy
    num_cols = LAG_COLS + [f"delta_{c}" for c in LAG_COLS] + ["rest_days","market_total","market_spread","margin","total_points"]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["is_home","is_divisional","short_week"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("Int8")

    out = out.sort_values(["season","week","game_id","team"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[build_team_lag_features] wrote {len(out):,} rows -> {OUT}")

if __name__ == "__main__":
    main()
