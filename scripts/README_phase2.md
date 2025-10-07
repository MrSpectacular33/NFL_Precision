# NFL_Precision — Phase 2 Checkpoint

This checkpoint freezes our working backtest pipeline so you can always rebuild the historical ROI/diagnostics in one shot.

## What this bundle includes
- `scripts/backtest_report.ps1` – one-command regenerates:
  - Weekly edges for all seasons
  - Bucket tables (spread & total)
  - ROI-vs-threshold curves
  - Summary CSVs & plots
- This README (what to run, where outputs go)

## Prereqs
- Repo root: `F:\NFL_Precision`
- Virtual env: `.venv` (already used in prior runs)
- PowerShell (ExecutionPolicy Bypass allowed)

## One-liner run (default 2019–2024)
```powershell
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\backtest_report.ps1 `
  -SeasonStart 2019 -SeasonEnd 2024 `
  -Price -110 -MinEdge 0.5 -UseEV
