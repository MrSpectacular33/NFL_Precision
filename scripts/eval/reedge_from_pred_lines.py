import os, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

OUT_WIDE = "reports/backtest/outcomes_all.csv"
PRED_PARQ = "data/processed/predicted_lines_error.parquet"
SCHED = "data/raw/schedules_2019_2024.parquet"
OUT_LONG = "reports/backtest/outcomes_long_with_pcal.csv"

# --- helpers ---
def pick_col(cands, mp): 
    for c in cands:
        if c in mp: return mp[c]
    return None

def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    # pure numpy implementation
    return 0.5*(1.0 + (2/np.sqrt(np.pi)) * np.vectorize(lambda v: np.math.erf(v/np.sqrt(2.0)))(x) * 0 + 
                np.erf(x/np.sqrt(2.0)) if hasattr(np, "erf") else
                np.array([0.5*(1.0 + np.math.erf(float(v)/np.sqrt(2.0))) for v in x]))  # fallback handled below

try:
    _ = np.erf  # some builds lack this
    def norm_cdf(x): 
        x = np.asarray(x, dtype=float)
        return 0.5*(1.0 + np.erf(x/np.sqrt(2.0)))
except Exception:
    import math
    def norm_cdf(x):
        x = np.asarray(x, dtype=float)
        return np.array([0.5*(1.0 + math.erf(float(v)/math.sqrt(2.0))) for v in x])

def fit_iso(p, y):
    mask = ~(np.isnan(p) | np.isnan(y))
    p, y = p[mask].astype(float), y[mask].astype(float)
    if len(p) < 50:
        raise RuntimeError("Not enough samples for isotonic calibration.")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return iso

# --- load ---
ow = pd.read_csv(OUT_WIDE)
pred = pd.read_parquet(PRED_PARQ)

# sanity: need keys
for k in ("game_id","season","week","pred_margin_line","pred_total_line"):
    if k not in pred.columns: 
        raise RuntimeError(f"{PRED_PARQ} missing {k}")

# join predicted lines onto wide outcomes
m = ow.merge(pred[["game_id","season","week","pred_margin_line","pred_total_line"]], 
             on=["game_id","season","week"], how="left")
if m["pred_margin_line"].isna().all():
    raise RuntimeError("Join failed: no predicted lines matched outcomes_all.csv")

# pull actual scores from schedules for win/lose calculation
sc = pd.read_parquet(SCHED)
mp = {c.lower(): c for c in sc.columns}
gid   = pick_col(["game_id"], mp) or "game_id"
season= pick_col(["season"], mp) or "season"
week  = pick_col(["week","gameday_week","game_week"], mp) or "week"
ht    = pick_col(["home_team","home","home_abbr"], mp)
at    = pick_col(["away_team","away","away_abbr"], mp)
hs    = pick_col(["home_score","home_points","home_pts","home_final"], mp)
as_   = pick_col(["away_score","away_points","away_pts","away_final"], mp)
if any(x is None for x in (ht,at,hs,as_)):
    raise RuntimeError("Schedules parquet missing team/score columns.")

gsel = sc[[gid,season,week,ht,at,hs,as_]].copy()
gsel.columns = ["game_id","season","week","home_team","away_team","home_score","away_score"]
gsel["actual_margin_home_minus_away"] = gsel["home_score"].astype(float) - gsel["away_score"].astype(float)
gsel["actual_total"] = gsel["home_score"].astype(float) + gsel["away_score"].astype(float)

m = m.merge(gsel, on=["game_id","season","week"], how="left")

# compute team home/away alignment
m["team_is_home"] = (m["team"].astype(str).str.upper() == m["home_team"].astype(str).str.upper())
# team-line in points from market (spread is home minus away; flip for away team)
m["team_market_spread"] = np.where(m["team_is_home"], m["market_spread"].astype(float), -m["market_spread"].astype(float))
m["team_margin"] = np.where(m["team_is_home"], m["actual_margin_home_minus_away"], -m["actual_margin_home_minus_away"])

# --- recompute edges from predicted model lines ---
# spread: model advantage (for 'team' side) = (pred_margin_line - team_market_spread)
# total: model advantage for OVER = (pred_total_line - market_total)
m["edge_margin_pts_new"] = (np.where(m["team_is_home"], m["pred_margin_line"], -m["pred_margin_line"]) - m["team_market_spread"]).astype(float)
m["edge_total_pts_new"]  = (m["pred_total_line"] - m["market_total"]).astype(float)

# raw probabilities from edge via Normal CDF (conservative sigmas)
sigma_spread, sigma_total = 13.0, 10.0
m["p_cover_new"] = norm_cdf(m["edge_margin_pts_new"]/sigma_spread)
m["p_over_new"]  = norm_cdf(m["edge_total_pts_new"]/sigma_total)

# --- build long-form rows with new p_raw and realized outcomes ---
rows=[]
for _, r in m.iterrows():
    season, week, gid = int(r["season"]), int(r["week"]), r["game_id"]

    # Spread pick: follow edge sign
    sgn = 1.0 if r["edge_margin_pts_new"]>=0 else -1.0
    pick_name = r["team"] if sgn>0 else r["opponent"]
    won_spread = 1 if (sgn*(r["team_margin"] - r["team_market_spread"]) > 0) else 0
    rows.append({
        "season": season, "week": week, "game_id": gid,
        "bet_type": "spread",
        "bet_line": float(r["market_spread"]),
        "pick": str(pick_name),
        "price": -110,
        "edge_pts": float(r["edge_margin_pts_new"]),
        "p_raw": float(r["p_cover_new"]),
        "won": int(won_spread)
    })

    # Total pick: OVER if edge >= 0 else UNDER
    over = r["edge_total_pts_new"]>=0
    won_total = 1 if ((r["actual_total"] - float(r["market_total"]) > 0) == bool(over)) else 0
    rows.append({
        "season": season, "week": week, "game_id": gid,
        "bet_type": "total",
        "bet_line": float(r["market_total"]),
        "pick": "OVER" if over else "UNDER",
        "price": -110,
        "edge_pts": float(r["edge_total_pts_new"]),
        "p_raw": float(r["p_over_new"]),
        "won": int(won_total)
    })

long = pd.DataFrame(rows)

# clip/fix any stray NaNs
long["p_raw"] = long["p_raw"].astype(float).clip(0.001, 0.999)
long = long.dropna(subset=["p_raw","won"])

# isotonic per market
out = []
for mkt in ("spread","total"):
    dm = long[long["bet_type"]==mkt].copy()
    iso = fit_iso(dm["p_raw"].values, dm["won"].values)
    dm["p_cal"] = np.clip(iso.transform(dm["p_raw"].values), 0.001, 0.999)
    out.append(dm)

out_long = pd.concat(out, ignore_index=True)
os.makedirs(os.path.dirname(OUT_LONG), exist_ok=True)
out_long.to_csv(OUT_LONG, index=False)
print(f"[OK] re-edged + calibrated → {OUT_LONG} (rows: {len(out_long)})")
