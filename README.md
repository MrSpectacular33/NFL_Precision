# NFL_Precision

Reproducible NFL modeling pipeline (margin & total) with:
- PowerShell orchestration
- DuckDB for joins/feature assembly
- Python for modeling (OLS baseline -> regularized CV -> calibration)
- Minimal external calls, secrets in .env

## Quickstart
1) python -m venv .venv
2) .\.venv\Scripts\Activate.ps1
3) pip install -r requirements.txt
4) powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1 -Season 2025 -Week 4 -FromScratch
