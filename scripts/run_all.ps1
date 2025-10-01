param(
  [int]$Season = 2025,
  [int]$Week   = 4,
  [switch]$FromScratch
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$py = Join-Path $RepoRoot ".\.venv\Scripts\python.exe"

function Step($title, $argv) {
  $scriptPath = Join-Path $RepoRoot $argv[0]
  $argsOnly   = $argv[1..($argv.Count-1)]
  Write-Host ("[RUN] {0}" -f $title)
  if ($argsOnly.Count -gt 0) {
    Write-Host ("[CMD] {0} {1} {2}" -f $py, $scriptPath, (($argsOnly) -join ' '))
    & $py $scriptPath @argsOnly
  } else {
    Write-Host ("[CMD] {0} {1}" -f $py, $scriptPath)
    & $py $scriptPath
  }
  if ($LASTEXITCODE -ne 0) { throw "Step failed: $title" }
  Write-Host ("[OK] {0}" -f $title)
}

if ($FromScratch) {
  Write-Host "[CLEAN] wiping data and reports (safe paths only)"
  Remove-Item -Recurse -Force (Join-Path $RepoRoot "data\raw\*")       -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force (Join-Path $RepoRoot "data\interim\*")   -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force (Join-Path $RepoRoot "data\processed\*") -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force (Join-Path $RepoRoot "reports\*")        -ErrorAction SilentlyContinue

  New-Item -Type Directory -Force -Path `
    (Join-Path $RepoRoot "data\raw"), `
    (Join-Path $RepoRoot "data\interim"), `
    (Join-Path $RepoRoot "data\processed"), `
    (Join-Path $RepoRoot "reports\cv"), `
    (Join-Path $RepoRoot "reports\holdouts"), `
    (Join-Path $RepoRoot "reports\picks"), `
    (Join-Path $RepoRoot "reports\bankroll"), `
    (Join-Path $RepoRoot "reports\calibration"), `
    (Join-Path $RepoRoot "reports\diagnostics") | Out-Null
}

# 1) Ingest
Step "Pull schedules"         @("scripts\ingest\pull_schedules.py", "--season-start", 2019, "--season-end", $Season)
Step "Pull games (weekly)"    @("scripts\ingest\pull_games.py", "--season-start", 2019, "--season-end", $Season)
Step "Pull odds (cached stub)"@("scripts\ingest\pull_odds_cached.py", "--season", $Season, "--week", $Week)

# 2) Features
Step "Build team games"       @("scripts\features\build_team_games.py", "--season-start", 2019, "--season-end", $Season)
Step "Build lag features (L3/L8)" @("scripts\features\build_team_lag_features.py")

# 3) Baselines (in-sample OLS – optional)
Step "Train margin OLS"       @("scripts\train\train_margin_ols.py")
Step "Train total OLS"        @("scripts\train\train_total_ols.py")

# 4) Cross-validated OLS (CV baselines for fair QA)
Step "Train margin OLS (CV)"  @("scripts\train\train_margin_ols_cv.py")
Step "Train total  OLS (CV)"  @("scripts\train\train_total_ols_cv.py")

# 5) Regularized models
Step "Train margin RidgeCV"   @("scripts\train\train_margin_ridgecv.py")
Step "Train total  RidgeCV"   @("scripts\train\train_total_ridgecv.py")

# 6) QA: compare CV vs CV at strict tolerance
Step "QA gate: margin" @("scripts\eval\qa_ridge_gate.py", "--baseline", "reports/cv/margin_ols_cv.json", "--candidate", "reports/cv/margin_ridgecv.json", "--tolerance", 1.00)
Step "QA gate: total"  @("scripts\eval\qa_ridge_gate.py", "--baseline", "reports/cv/total_ols_cv.json",  "--candidate", "reports/cv/total_ridgecv.json",  "--tolerance", 1.00)

# smoke
Step "Hello eval"             @("scripts\eval\hello_eval.py")

Write-Host "[DONE] NFL_Precision end-to-end OK"
