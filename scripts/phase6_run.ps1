param(
  [double]$SigmaSpread=14, [double]$SigmaTotal=11,
  [double]$MinEvSpread=0.015, [double]$MinEdgeSpread=1.0,
  [double]$MinEvTotal =0.020,  [double]$MinEdgeTotal=1.2,
  [double]$KellyFrac=0.12, [double]$PerBetCap=0.008,
  [double]$WeeklyCap=0.05, [double]$PauseDD=0.30
)
$py = ".\.venv\Scripts\python.exe"

# Predict Phase-6
& $py .\scripts\eval\predict_error_ensemble_phase6.py `
  --features-parquet data\processed\team_games_features_v5.parquet `
  --models-dir models\phase6 `
  --out data\processed\predicted_lines_v6.parquet `
  --shrink-strength 0.5

# Re-edge + OOF isotonic
& $py .\scripts\eval\reedge_from_pred_lines_v4.py `
  --pred-parquet data\processed\predicted_lines_v6.parquet `
  --sigma-spread $SigmaSpread --sigma-total $SigmaTotal

# Bankroll
& $py .\scripts\bankroll\bankroll_sim.py `
  --in reports\backtest\outcomes_long_with_pcal.csv `
  --outdir reports\backtest `
  --min-ev-spread $MinEvSpread --min-edge-pts-spread $MinEdgeSpread `
  --min-ev-total  $MinEvTotal  --min-edge-pts-total  $MinEdgeTotal `
  --bankroll0 10000 --kelly-frac $KellyFrac `
  --per-bet-cap $PerBetCap --weekly-risk-cap $WeeklyCap --pause-drawdown $PauseDD
