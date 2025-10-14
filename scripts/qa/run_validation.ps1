param(
  [string]$Preds = ".\reports\risk\preds_final_for_kelly.csv",
  [string]$Outcomes = ".\data\warehouse\bet_outcomes_all.csv"
)
Write-Host "`n[QA] Computing validation metrics..."
.\.venv\Scripts\python.exe .\scripts\qa\validation_metrics.py
Write-Host "[QA] Wrote reports\qa\validation_summary.json and calibration_table.csv"
