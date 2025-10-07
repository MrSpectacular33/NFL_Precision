Param(
  [int]$Season = (Get-Date).Year,
  [int]$Week = 1,
  [switch]$FromScratch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Paths & helpers ---
$Root = Resolve-Path (Join-Path $PSScriptRoot '..')
$PyPath = Join-Path $Root '.\.venv\Scripts\python.exe'
if (-not (Test-Path $PyPath)) {
  Write-Warning "Virtualenv python not found at $PyPath. Falling back to 'python' on PATH."
  $PyPath = 'python'
}

function Run-Step {
  param(
    [Parameter(Mandatory)] [string]$Title,
    [Parameter(Mandatory)] [string]$RelScript,
    [string[]]$Args = @()
  )
  $scriptPath = Join-Path $Root $RelScript
  if (-not (Test-Path $scriptPath)) {
    throw "Script not found: $scriptPath"
  }

  $cmdPreview = @($PyPath, $scriptPath) + $Args
  Write-Host ""
  Write-Host "[RUN] $Title"
  Write-Host "[CMD] $($cmdPreview -join ' ')"
  & $PyPath $scriptPath @Args

  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Title"
  } else {
    Write-Host "[OK] $Title"
  }
}

# --- Clean (optional) ---
if ($FromScratch) {
  Write-Host "[CLEAN] wiping data and reports (safe paths only)"
  $safeDirs = @(
    (Join-Path $Root 'data\raw'),
    (Join-Path $Root 'data\processed'),
    (Join-Path $Root 'reports')
  )
  foreach ($d in $safeDirs) {
    if (Test-Path $d) {
      Get-ChildItem -Path $d -Recurse -Force -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
}

# --- Ensure output dirs exist ---
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'reports') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'reports\picks') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'reports\edges') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'reports\calibration') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'reports\diagnostics') | Out-Null

# --- Ingest ---
Run-Step -Title "Pull schedules"      -RelScript 'scripts\ingest\pull_schedules.py' -Args @('--season-start','2019','--season-end',"$Season")
Run-Step -Title "Pull games (weekly)" -RelScript 'scripts\ingest\pull_games.py'     -Args @('--season-start','2019','--season-end',"$Season")

# --- Features ---
Run-Step -Title "Build team games"            -RelScript 'scripts\features\build_team_games.py'       -Args @('--season-start','2019','--season-end',"$Season")
Run-Step -Title "Build lag features (L3/L8)"  -RelScript 'scripts\features\build_team_lag_features.py'

# --- Train (anchored) ---
Run-Step -Title "Train margin OLS"            -RelScript 'scripts\train\train_margin_ols.py'
Run-Step -Title "Train total OLS"             -RelScript 'scripts\train\train_total_ols.py'
Run-Step -Title "Train margin OLS (CV)"       -RelScript 'scripts\train\train_margin_ols_cv.py'
Run-Step -Title "Train total  OLS (CV)"       -RelScript 'scripts\train\train_total_ols_cv.py'
Run-Step -Title "Train margin RidgeCV"        -RelScript 'scripts\train\train_margin_ridgecv.py'
Run-Step -Title "Train total  RidgeCV"        -RelScript 'scripts\train\train_total_ridgecv.py'

# --- Train (no market) ---
Run-Step -Title "Train margin OLS (no market)" -RelScript 'scripts\train\train_margin_ols_nomkt.py'
Run-Step -Title "Train total  OLS (no market)" -RelScript 'scripts\train\train_total_ols_nomkt.py'

# --- QA gates ---
Run-Step -Title "QA gate: margin" -RelScript 'scripts\eval\qa_ridge_gate.py' -Args @(
  '--baseline','reports/cv/margin_ols_cv.json',
  '--candidate','reports/cv/margin_ridgecv.json',
  '--tolerance','1'
)
Run-Step -Title "QA gate: total" -RelScript 'scripts\eval\qa_ridge_gate.py' -Args @(
  '--baseline','reports/cv/total_ols_cv.json',
  '--candidate','reports/cv/total_ridgecv.json',
  '--tolerance','1'
)

# --- Predict (two flavors) ---
# These produce: reports/picks/predictions_s{Season}_w{Week}_mkt_home.csv and _nomkt_both.csv
Run-Step -Title "Predict week (mkt, home)"   -RelScript 'scripts\eval\predict_week.py' -Args @('--season',"$Season",'--week',"$Week",'--coefs','mkt','--rows','home')
Run-Step -Title "Predict week (nomkt, both)" -RelScript 'scripts\eval\predict_week.py' -Args @('--season',"$Season",'--week',"$Week",'--coefs','nomkt','--rows','both')

# --- Print picks (both files) ---
$picksMkt   = "reports/picks/predictions_s${Season}_w${Week}_mkt_home.csv"
$picksNomkt = "reports/picks/predictions_s${Season}_w${Week}_nomkt_both.csv"
Run-Step -Title "Print picks (mkt, home)"   -RelScript 'scripts\eval\print_picks.py' -Args @('--file', $picksMkt)
Run-Step -Title "Print picks (nomkt, both)" -RelScript 'scripts\eval\print_picks.py' -Args @('--file', $picksNomkt)

# --- Edge sheets (automated for both files) ---
Run-Step -Title "Edge sheet (mkt, home)"   -RelScript 'scripts\eval\edge_sheet.py' -Args @('--file', $picksMkt,   '--outdir','reports/edges')
Run-Step -Title "Edge sheet (nomkt, both)" -RelScript 'scripts\eval\edge_sheet.py' -Args @('--file', $picksNomkt, '--outdir','reports/edges')

# --- Diagnostics ---
Run-Step -Title "Calibration plots" -RelScript 'scripts\eval\calibration_plots.py'
Run-Step -Title "OLS feature audit" -RelScript 'scripts\eval\feature_audit.py'

Write-Host ""
Write-Host "[DONE] NFL_Precision end-to-end OK"

# ensure this new dir exists near the other New-Item lines
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'reports\bankroll') | Out-Null

# ... (unchanged steps above)

# --- Edge sheets (automated for both files) ---
Run-Step -Title "Edge sheet (mkt, home)"   -RelScript 'scripts\eval\edge_sheet.py' -Args @('--file', $picksMkt,   '--outdir','reports/edges')
Run-Step -Title "Edge sheet (nomkt, both)" -RelScript 'scripts\eval\edge_sheet.py' -Args @('--file', $picksNomkt, '--outdir','reports/edges')

# --- Bankroll simulation (for both edge sheets) ---
$edgesMkt   = "reports/edges/edges_s${Season}_w${Week}_mkt_home.csv"
$edgesNomkt = "reports/edges/edges_s${Season}_w${Week}_nomkt_both.csv"

Run-Step -Title "Bankroll sim (mkt, home)" -RelScript 'scripts\eval\bankroll_sim.py' -Args @(
  '--edges-file', $edgesMkt,
  '--starting-bankroll','1000',
  '--kelly-fraction','0.5',
  '--min-kelly','0.01',
  '--max-bets','999',
  '--odds-spread','-110',
  '--odds-total','-110',
  '--outdir','reports/bankroll'
)

Run-Step -Title "Bankroll sim (nomkt, both)" -RelScript 'scripts\eval\bankroll_sim.py' -Args @(
  '--edges-file', $edgesNomkt,
  '--starting-bankroll','1000',
  '--kelly-fraction','0.5',
  '--min-kelly','0.01',
  '--max-bets','999',
  '--odds-spread','-110',
  '--odds-total','-110',
  '--outdir','reports/bankroll'
)

# --- Diagnostics ---
Run-Step -Title "Calibration plots" -RelScript 'scripts\eval\calibration_plots.py'
Run-Step -Title "OLS feature audit" -RelScript 'scripts\eval\feature_audit.py'

# --- Bet slip (Markdown) from bankroll plans ---
$planMkt   = "reports/bankroll/bankroll_plan_edges_s${Season}_w${Week}_mkt_home.csv"
$planNomkt = "reports/bankroll/bankroll_plan_edges_s${Season}_w${Week}_nomkt_both.csv"

Run-Step -Title "Bet slip (Markdown)" -RelScript 'scripts\eval\make_bet_slip.py' -Args @(
  '--plan-files', $planMkt, $planNomkt,
  '--top', '25',
  '--min-stake', '1',
  '--title', "Season $Season, Week $Week — Bet Slip",
  '--outdir', 'reports/bankroll'
)
