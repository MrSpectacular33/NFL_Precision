param(
  [int]$SeasonStart = 2019,
  [int]$SeasonEnd   = 2024,
  [switch]$FromScratch
)

# ---------- Hardening ----------
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-Python {
  param([string]$RepoRoot)
  $venv = Join-Path $RepoRoot ".\.venv\Scripts\python.exe"
  if (Test-Path $venv) { return $venv }
  return "python"
}

function Invoke-Step {
  param(
    [string]$Title,
    [string[]]$Cmd,
    [string]$WorkingDir
  )
  if (-not $Cmd -or $Cmd.Count -lt 1) { throw "Invoke-Step called with empty command for step: $Title" }
  Write-Host "[RUN] $Title" -ForegroundColor Cyan
  Write-Host "[CMD] $($Cmd -join ' ')" -ForegroundColor DarkGray
  Push-Location $WorkingDir
  try {
    if ($Cmd.Count -gt 1) {
      & $Cmd[0] $Cmd[1..($Cmd.Count-1)]
    } else {
      & $Cmd[0]
    }
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Title" }
  } finally {
    Pop-Location
  }
  Write-Host "[OK] $Title" -ForegroundColor Green
}

function Safe-Clean {
  param([string]$PathToWipe)
  if (-not (Test-Path $PathToWipe)) { return }
  $full = (Resolve-Path $PathToWipe).Path
  if ($full -like "*\reports\backtest*") {
    Write-Host "[CLEAN] wiping $full" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $full
  } else {
    Write-Host "[SKIP CLEAN] refused to wipe $full (not in reports\backtest)" -ForegroundColor DarkYellow
  }
}

function Try-InvokePythonAbs {
  param(
    [string]$Py,
    [string]$RepoRoot,
    [string]$ScriptRelPath,
    [string[]]$ScriptArgs
  )
  $abs = Join-Path $RepoRoot $ScriptRelPath
  try {
    Push-Location $RepoRoot
    & $Py $abs $ScriptArgs
    Pop-Location
    return $true
  } catch {
    try { Pop-Location } catch {}
    return $false
  }
}

function Join-CsvFiles {
  param(
    [System.IO.FileInfo[]]$Files,
    [string]$OutPath
  )
  if (-not $Files -or $Files.Count -eq 0) { return $false }

  $all = @()
  foreach ($f in $Files) {
    try {
      $rows = Import-Csv -Path $f.FullName
      if ($rows) { $all += $rows }
    } catch {
      Write-Host ("[WARN] failed to read {0} :: {1}" -f $f.FullName, $_.Exception.Message) -ForegroundColor DarkYellow
    }
  }
  if ($all.Count -gt 0) {
    $outDir = Split-Path -Parent $OutPath
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    $all | Export-Csv -Path $OutPath -NoTypeInformation
    return $true
  }
  return $false
}

# ---------- Paths ----------
$ScriptRoot   = $PSScriptRoot             # ...\NFL_Precision\scripts
$RepoRoot     = Split-Path $ScriptRoot -Parent
$PythonExe    = Resolve-Python -RepoRoot $RepoRoot

$BacktestRoot = Join-Path $RepoRoot "reports\backtest"
$ByWeekDir    = Join-Path $BacktestRoot "by_week"

# ---------- Clean ----------
if ($FromScratch) {
  Safe-Clean -PathToWipe $BacktestRoot
}
if (-not (Test-Path $ByWeekDir)) {
  New-Item -ItemType Directory -Path $ByWeekDir | Out-Null
}

# ---------- 1) Pull schedules once (ABSOLUTE script paths + fixed working dir) ----------
Invoke-Step -Title "Pull schedules" -Cmd @(
  $PythonExe,
  (Join-Path $RepoRoot "scripts\ingest\pull_schedules.py"),
  "--season-start", $SeasonStart, "--season-end", $SeasonEnd
) -WorkingDir $RepoRoot

# ---------- 2) Loop seasons/weeks ----------
for ($season = $SeasonStart; $season -le $SeasonEnd; $season++) {

  Write-Host ("`n=== SEASON {0} ===" -f $season) -ForegroundColor Magenta

  for ($week = 1; $week -le 21; $week++) {
    # 2a) Predict (mkt, home)
    $ok = Try-InvokePythonAbs -Py $PythonExe -RepoRoot $RepoRoot -ScriptRelPath "scripts\eval\predict_week.py" -ScriptArgs @(
      "--season", "$season", "--week", "$week", "--coefs", "mkt", "--rows", "home"
    )
    if (-not $ok) {
      Write-Host ("[SKIP] predict (mkt,home) s{0} w{1}" -f $season, $week) -ForegroundColor DarkYellow
      continue
    }

    # 2b) Predict (nomkt, both)
    $ok2 = Try-InvokePythonAbs -Py $PythonExe -RepoRoot $RepoRoot -ScriptRelPath "scripts\eval\predict_week.py" -ScriptArgs @(
      "--season", "$season", "--week", "$week", "--coefs", "nomkt", "--rows", "both"
    )
    if (-not $ok2) {
      Write-Host ("[SKIP] predict (nomkt,both) s{0} w{1}" -f $season, $week) -ForegroundColor DarkYellow
    }

    # Expected prediction filenames
    $predMkt  = Join-Path $RepoRoot ("reports\picks\predictions_s{0}_w{1}_mkt_home.csv"   -f $season, $week)
    $predNoMk = Join-Path $RepoRoot ("reports\picks\predictions_s{0}_w{1}_nomkt_both.csv" -f $season, $week)

    # 2c) Edges (mkt, home)
    if (Test-Path $predMkt) {
      $ok3 = Try-InvokePythonAbs -Py $PythonExe -RepoRoot $RepoRoot -ScriptRelPath "scripts\eval\edge_sheet.py" -ScriptArgs @(
        "--file", $predMkt, "--outdir", $ByWeekDir
      )
      if (-not $ok3) {
        Write-Host ("[SKIP] edges (mkt,home) s{0} w{1}" -f $season, $week) -ForegroundColor DarkYellow
      }
    } else {
      Write-Host ("[SKIP] edges (mkt,home) missing preds s{0} w{1}" -f $season, $week) -ForegroundColor DarkYellow
    }

    # 2d) Edges (nomkt, both)
    if (Test-Path $predNoMk) {
      $ok4 = Try-InvokePythonAbs -Py $PythonExe -RepoRoot $RepoRoot -ScriptRelPath "scripts\eval\edge_sheet.py" -ScriptArgs @(
        "--file", $predNoMk, "--outdir", $ByWeekDir
      )
      if (-not $ok4) {
        Write-Host ("[SKIP] edges (nomkt,both) s{0} w{1}" -f $season, $week) -ForegroundColor DarkYellow
      }
    } else {
      Write-Host ("[SKIP] edges (nomkt,both) missing preds s{0} w{1}" -f $season, $week) -ForegroundColor DarkYellow
    }
  }

  # 2e) Concatenate weekly edges for the season
  $pattern     = "edges_s{0}_w*.csv" -f $season
  $seasonFiles = Get-ChildItem -Path $ByWeekDir -Filter $pattern -ErrorAction SilentlyContinue
  $outSeason   = Join-Path $BacktestRoot ("edges_all_s{0}.csv" -f $season)

  if (Join-CsvFiles -Files $seasonFiles -OutPath $outSeason) {
    Write-Host ("[OK] wrote {0}" -f $outSeason) -ForegroundColor Green
  } else {
    Write-Host ("[WARN] no weekly edges to join for season {0}" -f $season) -ForegroundColor DarkYellow
  }
}

# ---------- 3) Build summary across seasons ----------
Write-Host "`nRUN Build backtest summary" -ForegroundColor Cyan

$summaryRows = @()
for ($season = $SeasonStart; $season -le $SeasonEnd; $season++) {
  $seasonPath = Join-Path $BacktestRoot ("edges_all_s{0}.csv" -f $season)
  if (-not (Test-Path $seasonPath)) {
    Write-Host ("WARN Could not summarize {0} :: file missing" -f (Split-Path -Leaf $seasonPath)) -ForegroundColor DarkYellow
    continue
  }

  try {
    $rows = Import-Csv -Path $seasonPath

    $spreadEvCol = 'edge_spread_ev'
    $totalEvCol  = 'edge_total_ev'

    $nSpread = ($rows | Where-Object { $_.$spreadEvCol -ne $null -and $_.$spreadEvCol -ne "" }).Count
    $nTotal  = ($rows | Where-Object { $_.$totalEvCol  -ne $null -and $_.$totalEvCol  -ne "" }).Count

    $evSpread = 0.0
    $evTotal  = 0.0
    foreach ($r in $rows) {
      $v1 = 0.0; [void][double]::TryParse( ("" + $r.$spreadEvCol), [ref]$v1 ); $evSpread += $v1
      $v2 = 0.0; [void][double]::TryParse( ("" + $r.$totalEvCol ), [ref]$v2 ); $evTotal  += $v2
    }
    $evAll = $evSpread + $evTotal

    $kellySpreadSum = 0.0
    $kellyTotalSum  = 0.0
    if ($rows.Count -gt 0 -and $rows[0].PSObject.Properties.Name -contains 'kelly_spread') {
      foreach ($r in $rows) { $t=0.0; [void][double]::TryParse( ("" + $r.kelly_spread), [ref]$t ); $kellySpreadSum += $t }
    }
    if ($rows.Count -gt 0 -and $rows[0].PSObject.Properties.Name -contains 'kelly_total') {
      foreach ($r in $rows) { $t=0.0; [void][double]::TryParse( ("" + $r.kelly_total),  [ref]$t ); $kellyTotalSum  += $t }
    }

    $summaryRows += [pscustomobject]@{
      season            = $season
      n_spread_bets     = $nSpread
      n_total_bets      = $nTotal
      ev_spread         = '{0:N2}' -f $evSpread
      ev_total          = '{0:N2}' -f $evTotal
      ev_all            = '{0:N2}' -f $evAll
      kelly_spread_sum  = '{0:N3}' -f $kellySpreadSum
      kelly_total_sum   = '{0:N3}' -f $kellyTotalSum
    }
  } catch {
    Write-Host ("WARN Could not summarize {0} :: {1}" -f (Split-Path -Leaf $seasonPath), $_.Exception.Message) -ForegroundColor DarkYellow
  }
}

$summaryPath = Join-Path $BacktestRoot ("summary_seasons_{0}_{1}.csv" -f $SeasonStart, $SeasonEnd)
if ($summaryRows.Count -gt 0) {
  $summaryRows | Sort-Object season | Export-Csv -Path $summaryPath -NoTypeInformation
  Write-Host ("SUMMARY wrote -> {0}" -f $summaryPath) -ForegroundColor Green
} else {
  Write-Host "SUMMARY: no rows produced." -ForegroundColor DarkYellow
}

Write-Host "`n[DONE] Backtest completed." -ForegroundColor Green
