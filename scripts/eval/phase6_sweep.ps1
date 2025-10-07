param(
  [string]$Pred="data\processed\predicted_lines_v6.parquet",
  [string]$OutDir="reports\backtest"
)

$py = ".\.venv\Scripts\python.exe"
$sweepDir = Join-Path $OutDir "sweeps"
New-Item -Type Directory -Force $sweepDir | Out-Null
$results = @()

# combos: feel free to add/remove values
$SIGMA_SPREAD = 13, 14, 15
$SIGMA_TOTAL  = 10, 11, 12
$MIN_EV_SPREAD = 0.012, 0.015, 0.018, 0.020
$MIN_EDGE_SPREAD = 1.0, 1.2
$KELLY = 0.10, 0.12
$PERCAP = 0.006, 0.008
$WEEKCAP = 0.04, 0.05

foreach ($ss in $SIGMA_SPREAD) {
  foreach ($st in $SIGMA_TOTAL) {
    & $py .\scripts\eval\reedge_from_pred_lines_v4.py `
        --pred-parquet $Pred `
        --sigma-spread $ss --sigma-total $st | Out-Null

    foreach ($ev in $MIN_EV_SPREAD) {
      foreach ($ed in $MIN_EDGE_SPREAD) {
        foreach ($k in $KELLY) {
          foreach ($pc in $PERCAP) {
            foreach ($wc in $WEEKCAP) {

              & $py .\scripts\bankroll\bankroll_sim.py `
                --in $OutDir\outcomes_long_with_pcal.csv `
                --outdir $OutDir `
                --min-ev-spread $ev --min-edge-pts-spread $ed `
                --bankroll0 10000 --kelly-frac $k `
                --per-bet-cap $pc --weekly-risk-cap $wc --pause-drawdown 0.30 | Out-Null

              $sum = Import-Csv "$OutDir\bankroll_summary.csv" | Select-Object -First 1
              $row = [PSCustomObject]@{
                sigma_spread      = $ss
                sigma_total       = $st
                min_ev_spread     = $ev
                min_edge_spread   = $ed
                kelly_frac        = $k
                per_bet_cap       = $pc
                weekly_risk_cap   = $wc
                final_bankroll    = [double]$sum.final_bankroll
                roi               = [double]$sum.roi
                max_drawdown      = [double]$sum.max_drawdown
                n_placed_bets     = [int]$sum.n_placed_bets
              }
              $results += $row

              $tag = "ss${ss}_st${st}_ev${ev}_ed${ed}_k${k}_pc${pc}_wc${wc}"
              Copy-Item "$OutDir\bankroll_summary.csv" "$sweepDir\summary_$tag.csv" -Force
              Copy-Item "$OutDir\bankroll_path.csv"    "$sweepDir\path_$tag.csv"     -Force
            }
          }
        }
      }
    }
  }
}

$results `
| Sort-Object roi -Descending `
| Tee-Object -File "$sweepDir\_sweep_results.csv" `
| Format-Table -AutoSize
Write-Host "`n[OK] Wrote sweep summaries to $sweepDir"
