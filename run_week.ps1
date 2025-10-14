param(
  [double]$bankroll = 10000,
  [double]$targetStake = 500
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

Write-Host "`n=== WEEKLY RUN START ===" -ForegroundColor Cyan

# 1) Risk flow (preds -> Kelly -> optimize -> scale -> exec sheet -> settlement/warehouse)
.\scripts\run_flow.ps1 -bankroll $bankroll -targetStake $targetStake

# 2) Portfolio diversification/caps
.\scripts\portfolio\run_portfolio.ps1

# 3) Package artifacts + summary
Write-Host "`n[Package] Zipping artifacts + writing summary..." -ForegroundColor Cyan
$pack = & $py "scripts/deploy/zip_and_summary.py"
Write-Output $pack

Write-Host "`n=== WEEKLY RUN DONE ===" -ForegroundColor Cyan
