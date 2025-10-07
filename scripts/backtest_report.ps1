param(
    [int]$SeasonStart = 2019,
    [int]$SeasonEnd   = 2024,
    [int]$Price       = -110,
    [double]$MinEdge  = 0.5,
    [switch]$UseEV
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Split-Path $ScriptDir

$BacktestAll     = Join-Path $ScriptDir 'backtest_all.ps1'
$EdgeBucketsPy   = Join-Path (Join-Path $ScriptDir 'eval') 'edge_buckets.py'
$PlotBucketsPy   = Join-Path (Join-Path $ScriptDir 'eval') 'plot_buckets.py'
$GamesParquet    = Join-Path (Join-Path $RepoRoot 'data\processed') 'team_games_features.parquet'
$EdgesDir        = Join-Path (Join-Path $RepoRoot 'reports') 'backtest'
$OutcomesCSV     = Join-Path $EdgesDir 'outcomes_all.csv'
$SpreadCSV       = Join-Path $EdgesDir 'spread_buckets.csv'
$TotalCSV        = Join-Path $EdgesDir 'total_buckets.csv'
$VenvPython      = Join-Path (Join-Path $RepoRoot '.venv\Scripts') 'python.exe'

function Assert-Path([string]$Path) {
    if (-not (Test-Path $Path)) { throw "Missing required path: $Path" }
}

Write-Host "[report] Starting Phase-2 rebuild…" -ForegroundColor Cyan

Assert-Path $BacktestAll
Assert-Path $EdgeBucketsPy
Assert-Path $PlotBucketsPy
Assert-Path $GamesParquet
Assert-Path $VenvPython

Write-Host "[report] 1/3 backtest_all.ps1 ($SeasonStart-$SeasonEnd)…"
powershell -ExecutionPolicy Bypass -File $BacktestAll `
  -SeasonStart $SeasonStart -SeasonEnd $SeasonEnd `
  -Price $Price -MinEdge $MinEdge $(if ($UseEV) { "-UseEV" })

Write-Host "[report] 2/3 edge_buckets.py…"
& $VenvPython $EdgeBucketsPy `
  --edges-dir $EdgesDir `
  --games-parquet $GamesParquet `
  --outdir $EdgesDir `
  --price $Price `
  --min-edge $MinEdge `
  $(if ($UseEV) { "--use-ev" }) `
  --export-outcomes $OutcomesCSV

Assert-Path $SpreadCSV
Assert-Path $TotalCSV

Write-Host "[report] 3/3 plot_buckets.py…"
& $VenvPython $PlotBucketsPy `
  --spread-file $SpreadCSV `
  --total-file  $TotalCSV `
  --outdir      $EdgesDir

Write-Host "[report] Done. Artifacts in $EdgesDir." -ForegroundColor Green
