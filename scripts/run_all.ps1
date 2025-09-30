param(
  [int] = 2025,
  [int]   = 4,
  [switch]
)

# Fail fast
\Continue = "Stop"

# Python path
\ = ".\.venv\Scripts\python.exe"

function Step(\, \) {
  Write-Host ("[RUN] {0}" -f \)
  & \ @args
  if (\0 -ne 0) { throw "Step failed: \" }
  Write-Host ("[OK] {0}" -f \)
}

# Optional clean
if (\) {
  Write-Host "[CLEAN] wiping data and reports (safe paths only)"
  Remove-Item -Recurse -Force .\data\raw\*       -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force .\data\interim\*   -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force .\data\processed\* -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force .\reports\*        -ErrorAction SilentlyContinue

  New-Item -Type Directory -Force -Path 
    .\data\raw, 
    .\data\interim, 
    .\data\processed, 
    .\reports\cv, 
    .\reports\holdouts, 
    .\reports\picks, 
    .\reports\bankroll, 
    .\reports\calibration, 
    .\reports\diagnostics | Out-Null
}

# 1) Ingest (stub)
Step "Hello ingest"   @(".\scripts\ingest\hello_ingest.py", "--season", \, "--week", \)

# 2) Features (stub)
Step "Hello features" @(".\scripts\features\hello_features.py", "--season", \)

# 3) Train (stub)
Step "Hello train"    @(".\scripts\train\hello_train.py")

# 4) Eval (stub)
Step "Hello eval"     @(".\scripts\eval\hello_eval.py")

Write-Host "[DONE] NFL_Precision Phase 0 skeleton OK"
