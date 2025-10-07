import os, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

# try to import a reliable CDF
try:
    from scipy.stats import norm
    def norm_cdf(x):
        return norm.cdf(x)
except Exception:
    import math
    def norm_cdf(x):
        x = np.asarray(x, dtype=float)
        # use math.erf elementwise
        return np.array([0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in x])

OUT_WIDE   = "reports/backtest/outcomes_all.csv"
SCHEDULES  = "data/raw/schedules_2019_2024.parquet"
OUT_LONG   = "reports/backtest/outcomes_long_with_pcal.csv"
OUT_WIDE_O = "reports/backtest/outcomes_all_with_pcal.csv"

def fit_iso_safe(p_raw, y):
    p_raw = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(p_raw) | np.isnan(y))
    p, t = p_raw[mask], y[mask]
    if len(p) < 50:
        raise RuntimeError(f"Not enough non-NaN samples to calibrate (got {len(p)}).")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, t)
    return iso

def pick_col(cands, cols_lower):
    for c in cands:
        if c in cols_lower: return cols_lower[c]
    return None

# ---- load inputs ----
ow = pd.read_csv(OUT_WIDE)
if not os.path.exists(SCHEDULES):
    raise RuntimeError(f"Missing schedules parquet at {SCHEDULES}. Run the backtest to generate it.")
sc = pd.read_parquet(SCHEDULES)
map_lower = {c.lower(): c for c in sc.columns}

gid   = pick_col(["game_id"], map_lower) or "game_id"
season= pick_col(["season"], map_lower) or "season"
week  = pick_col(["week","gameday_week","game_week"], map_lower) or "week"
ht    = pick_col(["home_team","home","home_abbr"], map_lower)
at    = pick_col(["away_team","away","away_abbr"], map_lower)
hs    = pick_col(["home_score","home_points","home_pts","home_final"], map_lower)
as_   = pick_col(["away_score","away_points","away_pts","away_final"], map_lower)
needed = [gid,season,week,ht,at,hs,as_]
if any(x is None for x in needed):
    raise RuntimeError("Schedules parquet missing required score/team columns.")

gsel = sc[[gid,season,week,ht,at,hs,as_]].copy()
gsel.columns = ["game_id","season","week","home_team","away_team","home_score","away_score"]
gsel["actual_margin_home_minus_away"] = gsel["home_score"].astype(float) - gsel["away_score"].astype(float)
gsel["actual_total"] = gsel["home_score"].astype(float) + gsel["away_score"].astype(float)

req_cols = ["season","week","game_id","team","opponent","market_spread","market_total",
            "edge_margin_pts","edge_total_pts","p_cover","p_over"]
miss = [c for c in req_cols if c not in ow.columns]
if miss: raise RuntimeError(f"Missing columns in outcomes_all.csv: {miss}")

m = ow.merge(gsel, on=["game_id","season","week"], how="left")
if m["actual_total"].isna().any():
    raise RuntimeError("Some games lack scores after merge; check keys.")

m["team_is_home"] = (m["team"].astype(str).str.upper() == m["home_team"].astype(str).str.upper())
m["team_line"]    = np.where(m["team_is_home"], m["market_spread"].astype(float), -m["market_spread"].astype(float))
m["team_margin"]  = np.where(m["team_is_home"], m["actual_margin_home_minus_away"], -m["actual_margin_home_minus_away"])

rows = []
for _, r in m.iterrows():
    season, week, gid = int(r["season"]), int(r["week"]), r["game_id"]
    # SPREAD
    spread_edge = float(r["edge_margin_pts"])
    pick_team   = (spread_edge >= 0)
    sgn = 1.0 if pick_team else -1.0
    won_spread = 1 if (sgn*(r["team_margin"] - r["team_line"]) > 0) else 0
    rows.append({
        "season": season, "week": week, "game_id": gid,
        "bet_type": "spread",
        "bet_line": float(r["market_spread"]),
        "pick": str(r["team"] if pick_team else r["opponent"]),
        "price": -110,
        "edge_pts": spread_edge,
        "p_raw": float(r["p_cover"]) if pd.notna(r["p_cover"]) else np.nan,
        "won": won_spread
    })
    # TOTAL
    total_edge = float(r["edge_total_pts"])
    over = (total_edge >= 0)
    won_total = 1 if ((r["actual_total"] - float(r["market_total"])) > 0) == over else 0
    rows.append({
        "season": season, "week": week, "game_id": gid,
        "bet_type": "total",
        "bet_line": float(r["market_total"]),
        "pick": "OVER" if over else "UNDER",
        "price": -110,
        "edge_pts": total_edge,
        "p_raw": float(r["p_over"]) if pd.notna(r["p_over"]) else np.nan,
        "won": won_total
    })

long = pd.DataFrame(rows)
# Fallback fill for missing p_raw
for mkt, sigma in (("spread",13.0),("total",10.0)):
    miss = long["bet_type"].eq(mkt) & long["p_raw"].isna()
    if miss.any():
        z = long.loc[miss,"edge_pts"].astype(float).to_numpy() / sigma
        long.loc[miss,"p_raw"] = np.clip(norm_cdf(z),0.01,0.99)

long["p_raw"] = long["p_raw"].astype(float).clip(0.001,0.999)
long = long.dropna(subset=["p_raw","won"])

# Isotonic calibration
out_frames=[]
for mkt in ("spread","total"):
    dm=long[long["bet_type"]==mkt].copy()
    iso=fit_iso_safe(dm["p_raw"].values,dm["won"].values)
    dm["p_cal"]=np.clip(iso.transform(dm["p_raw"].values),0.001,0.999)
    out_frames.append(dm)
long_cal=pd.concat(out_frames,ignore_index=True)

os.makedirs(os.path.dirname(OUT_LONG),exist_ok=True)
long_cal.to_csv(OUT_LONG,index=False)
print(f"[OK] wrote {OUT_LONG} (rows: {len(long_cal)})")

# Wide output
wide=ow.copy()
try:
    sp=long_cal[long_cal["bet_type"]=="spread"]["p_cal"].reset_index(drop=True)
    to=long_cal[long_cal["bet_type"]=="total"]["p_cal"].reset_index(drop=True)
    wide["p_cal_spread"]=sp.values
    wide["p_cal_total"]=to.values
except Exception:
    sp_key=long_cal[long_cal["bet_type"]=="spread"][["game_id","season","week","p_cal"]].rename(columns={"p_cal":"p_cal_spread"})
    to_key=long_cal[long_cal["bet_type"]=="total"][["game_id","season","week","p_cal"]].rename(columns={"p_cal":"p_cal_total"})
    wide=wide.merge(sp_key,on=["game_id","season","week"],how="left")
    wide=wide.merge(to_key,on=["game_id","season","week"],how="left")
wide.to_csv(OUT_WIDE_O,index=False)
print(f"[OK] wrote {OUT_WIDE_O} with p_cal_spread / p_cal_total")
