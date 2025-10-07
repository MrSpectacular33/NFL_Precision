import os, numpy as np, pandas as pd

INP  = "data/processed/team_games_features.parquet"
OUT  = "data/processed/team_games_features_v4.parquet"

def safe_col(df, names, default=np.nan):
    for n in names:
        if n in df.columns: return df[n]
        if n.lower() in map(str.lower, df.columns):
            return df[[c for c in df.columns if c.lower()==n.lower()][0]]
    return pd.Series(default, index=df.index)

def mk_bool(x): 
    if x.dtype==bool: return x.astype(int)
    return x.astype(str).str.lower().isin(["1","true","t","yes","y"]).astype(int)

def main():
    if not os.path.exists(INP):
        raise SystemExit(f"Missing {INP}")
    df = pd.read_parquet(INP).copy()

    # Keys
    for k in ("game_id","season","week","team","opponent"):
        if k not in df.columns: df[k] = np.nan

    # Baseline signals (safe fetch)
    off_epa   = safe_col(df, ["off_epa","offensive_epa","team_epa"])
    def_epa   = safe_col(df, ["def_epa","defensive_epa","opp_epa"])
    suc_off   = safe_col(df, ["success_rate_off","off_success","off_sr"])
    suc_def   = safe_col(df, ["success_rate_def","def_success","def_sr"])
    pace_s    = safe_col(df, ["pace_seconds","seconds_per_play","pace"])
    proe      = safe_col(df, ["pass_rate_over_expected","PROE","proe"])
    rest_diff = safe_col(df, ["rest_days_diff","rest_diff","days_rest_diff"], 0.0)
    wind_mph  = safe_col(df, ["wind_mph","wind_speed"], 0.0)
    is_dome   = mk_bool(safe_col(df, ["is_dome","dome","indoor"], 0))

    # Home/away normalization (if present)
    is_home   = mk_bool(safe_col(df, ["is_home","home_flag"], 0))
    # Shift signals slightly for home/away; helps model interaction cheaply
    home_adj = (is_home*0.5 - (1-is_home)*0.5)

    # Market anchors (if outcome CSV was merged upstream, these may exist)
    close_sp  = safe_col(df, ["closing_spread","market_spread"])
    close_tot = safe_col(df, ["closing_total","market_total"])

    # Simple continuity proxies (safe fallbacks)
    qb_dg     = safe_col(df, ["qb_downgrade","qb_change","qb_out"], 0.0)
    ol_cont   = safe_col(df, ["ol_continuity","ol_same_starters","ol_cont"], np.nan).fillna(0)

    # Derived features
    out = pd.DataFrame({
        "game_id": df["game_id"], "season": df["season"], "week": df["week"],
        "team": df["team"], "opponent": df["opponent"],
        "off_epa": off_epa, "def_epa": def_epa,
        "success_rate_off": suc_off, "success_rate_def": suc_def,
        "pace_seconds": pace_s, "proe": proe,
        "rest_days_diff": rest_diff,
        "wind_mph": wind_mph, "is_dome": is_dome,
        "is_home": is_home,
        "home_bias": home_adj,
        "qb_downgrade": qb_dg,
        "ol_continuity": ol_cont,
        "closing_spread": close_sp, "closing_total": close_tot,
    })

    # Interaction taps (cheap nonlinearity)
    out["epa_diff"] = out["off_epa"] - out["def_epa"]
    out["sr_diff"]  = out["success_rate_off"] - out["success_rate_def"]
    out["pace_x_proe"] = (out["pace_seconds"].astype(float).fillna(out["pace_seconds"].median())) * (out["proe"].astype(float).fillna(0))
    out["rest_home_inter"] = out["rest_days_diff"].astype(float).fillna(0) * out["home_bias"]
    out["wind_outdoor"] = out["wind_mph"].astype(float).fillna(0) * (1 - out["is_dome"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[OK] wrote {OUT} (rows: {len(out)})")

if __name__ == "__main__":
    main()
