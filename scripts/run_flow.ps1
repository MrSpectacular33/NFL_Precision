param(
  [double]$bankroll = 10000,
  [double]$targetStake = 500
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

# --- Paths ---
$picksSlim  = "reports\risk\placed_bets_keys_slim.csv"
$predsSrc   = "reports\cv\preds_with_p_cal.csv"   # switch to preds_with_p_blend.csv if desired
$predsFinal = "reports\risk\preds_final_for_kelly.csv"
$mc         = "reports\risk\mc_from_placed_bets_summary_h20.csv"
$kellyCfg   = "configs\risk\adaptive_kelly.yml"
$stakesRaw  = "reports\risk\stakes_week_latest.csv"
$optCfg     = "configs\portfolio\optimizer.yml"
$stakesOpt  = "reports\risk\stakes_optimized_latest.csv"
$optDiff    = "reports\risk\stakes_opt_diff_latest.csv"
$execCsv    = "reports\exec\execution_sheet_latest.csv"

# Ensure output dir exists
New-Item -ItemType Directory -Force "reports\exec" | Out-Null

Write-Host "`n[Preflight]"
& $py scripts\qa\preflight_config.py
if ($LASTEXITCODE -ne 0) { throw "Preflight failed - fix adaptive_kelly.yml first." }

# Clean old outputs to avoid stale reads
Remove-Item -ErrorAction SilentlyContinue $predsFinal, $stakesRaw, $stakesOpt, $optDiff, $execCsv

# -------------------------------------------------------
# A) Build preds_final_for_kelly.csv from preds + keys
# -------------------------------------------------------
Write-Host "`n[A] Build preds_final_for_kelly.csv"
$code = @'
import yaml, pandas as pd, json, sys
def american_to_decimal(x):
    x=float(x); return 1+(100/abs(x)) if x<0 else 1+(x/100)

cfg = yaml.safe_load(open(r"{CFG}","r",encoding="utf-8")) or {}
kelly_in = str(cfg.get("kelly_input","p_blend")).strip()

p    = pd.read_csv(r"{PREDSSRC}")
keys = pd.read_csv(r"{PICKSSLIM}")

if "bet_id" not in p.columns:
    if {"game_id","market"}.issubset(p.columns):
        p["bet_id"] = p["game_id"].astype(str)+"_"+p["market"].astype(str)
    else:
        sys.exit("preds_src must contain bet_id or (game_id, market)")

m = keys.merge(p, on="bet_id", how="inner")
if kelly_in not in m.columns:
    sys.exit(f"kelly_input '{kelly_in}' not in predictions")

m["price"] = pd.to_numeric(m.get("price",-110), errors="coerce").fillna(-110.0)
prob = pd.to_numeric(m[kelly_in], errors="coerce")
dec  = m["price"].map(american_to_decimal)

m["p_blend"] = prob
m["edge"]    = prob*(dec-1.0) - (1.0 - prob)

out = m[["bet_id","game_id","market","price","p_blend","edge"]].copy()
out.to_csv(r"{PREDSFINAL}", index=False)
print(json.dumps({"rows": int(len(out)), "pos_edge": int((out.edge>0).sum()), "kelly_input": kelly_in, "wrote": r"{PREDSFINAL}"}))
'@
$code = $code.Replace("{CFG}", $kellyCfg).Replace("{PREDSSRC}", $predsSrc).Replace("{PICKSSLIM}", $picksSlim).Replace("{PREDSFINAL}", $predsFinal)
$code | & $py -
if ($LASTEXITCODE -ne 0) { throw "[A] preds build failed." }

Write-Host "`n[preds_final_for_kelly head]"
Get-Content $predsFinal -TotalCount 5

# -------------------------------------------------------
# B) Kelly allocation
# -------------------------------------------------------
Write-Host "`n[B] adaptive_kelly.py"
& $py scripts\risk\adaptive_kelly.py `
  --picks $picksSlim `
  --preds $predsFinal `
  --mc $mc `
  --out $stakesRaw `
  --cfg $kellyCfg `
  --bankroll $bankroll
if ($LASTEXITCODE -ne 0) { throw "[B] adaptive_kelly failed." }

Write-Host "[Kelly head]"
Get-Content $stakesRaw -TotalCount 10

# -------------------------------------------------------
# C) Optimize positions
# -------------------------------------------------------
Write-Host "`n[C] optimize_positions.py"
& $py scripts\portfolio\optimize_positions.py `
  --in $stakesRaw `
  --cfg $optCfg `
  --bankroll $bankroll `
  --out $stakesOpt `
  --diff $optDiff
if ($LASTEXITCODE -ne 0) { throw "[C] optimize_positions failed." }

Write-Host "[Optimized head]"
Get-Content $stakesOpt -TotalCount 10

# -------------------------------------------------------
# D) Scale to target stake (per-game caps observed)
# -------------------------------------------------------
Write-Host "`n[D] scale to target"
$scale = @'
import yaml, pandas as pd, numpy as np, json
opt_csv=r"{OPTCSV}"; cfg_yml=r"{CFGYML}"; target={TARGET}
opt=pd.read_csv(opt_csv)
cur=float(opt["stake"].sum()) if "stake" in opt.columns else 0.0

with open(cfg_yml,"r",encoding="utf-8") as f:
    cfg=yaml.safe_load(f) or {}
cap=float(cfg.get("per_game_cap",250))
inc=float(cfg.get("stake_increment",1))
min_s=float(cfg.get("min_stake",1))

opt["headroom"] = np.maximum(
    0.0,
    pd.to_numeric(opt.get("stake_raw", opt["stake"]), errors="coerce").fillna(0)
    - pd.to_numeric(opt["stake"], errors="coerce").fillna(0)
)

need = max(0.0, target - cur)
if need > 0.5:
    pos = opt["stake"] > 0
    w = opt.loc[pos, "stake"].astype(float)
    add = pd.Series(0.0, index=opt.index)
    add.loc[pos] = need * (w / w.sum())
    add = np.minimum(add, opt["headroom"])
    if "game_id" in opt.columns:
        for _, idx in opt.groupby("game_id").indices.items():
            room = max(0.0, cap - float(opt.loc[idx, "stake"].sum()))
            tot  = float(add.loc[idx].sum())
            if tot > room + 1e-9:
                add.loc[idx] *= (room / tot) if tot > 0 else 0.0
    opt["stake"] = np.round((opt["stake"] + add) / inc) * inc
    opt.loc[opt["stake"] < min_s, "stake"] = 0.0

opt.to_csv(opt_csv, index=False)
print(json.dumps({"scaled": True, "to": float(opt["stake"].sum())}))
'@
$scale = $scale.Replace("{OPTCSV}", $stakesOpt).Replace("{CFGYML}", $optCfg).Replace("{TARGET}", [string]$targetStake)
$scale | & $py -
if ($LASTEXITCODE -ne 0) { throw "[D] scale failed." }

Write-Host "[Scaled head]"
Get-Content $stakesOpt -TotalCount 10

# -------------------------------------------------------
# E) Execution sheet
# -------------------------------------------------------
Write-Host "`n[E] execution sheet"
$exec = @'
import pandas as pd
inp=r"{OPTCSV}"; out=r"{EXECCSV}"
use=["bet_id","market","game_id","price","stake"]
df=pd.read_csv(inp)[use]
df["book"]="DK"; df["notes"]=""
df.to_csv(out, index=False)
print(f"Execution sheet -> {out}")
'@
$exec = $exec.Replace("{OPTCSV}", $stakesOpt).Replace("{EXECCSV}", $execCsv)
$exec | & $py -
if ($LASTEXITCODE -ne 0) { throw "[E] execution sheet failed." }

# -------------------------------------------------------
# Summary + simple constraints check
# -------------------------------------------------------
Write-Host "`n[Final Portfolio Totals]"
"Raw Kelly total:  " + (Import-Csv $stakesRaw  | Measure-Object stake -Sum).Sum
"Optimized total:  " + (Import-Csv $stakesOpt  | Measure-Object stake -Sum).Sum
"Positions count:  " + (Import-Csv $stakesOpt  | Where-Object {[double]$_.stake -gt 0}).Count

# Per-game cap check (optional hard-stop)
$cap = 250
$viol = @()
Import-Csv $stakesOpt | Group-Object game_id | ForEach-Object {
  $sum = ($_.Group | Measure-Object stake -Sum).Sum
  if($sum -gt $cap){ $viol += ("{0} over per_game_cap: {1}" -f $_.Name,$sum) }
}
if($viol.Count){ Write-Warning ("Per-game cap violations:`n" + ($viol -join "`n")) }

