import json, os, zipfile, datetime as dt, pathlib as p

DIST = p.Path("dist"); DIST.mkdir(parents=True, exist_ok=True)

stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
zip_path = DIST / f"run_{stamp}.zip"
summary_path = DIST / f"summary_{stamp}.txt"

candidates = [
    "reports/risk/preds_final_for_kelly.csv",
    "reports/risk/stakes_week_latest.csv",
    "reports/risk/stakes_optimized_latest.csv",
    "reports/exec/execution_sheet_latest.csv",
    "reports/risk/portfolio_week_latest.csv",
    "reports/portfolio/exposure_report.csv",
    "reports/portfolio/portfolio_summary.json",
    "reports/qa/live_metrics.json",
    "reports/qa/validation_summary.json",
    "reports/qa/calibration_table.csv",
    "reports/qa/reliability_curve.png",
    "reports/qa/pred_prob_hist.png",
]

def try_read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

portfolio = try_read_json("reports/portfolio/portfolio_summary.json", {})
metrics   = try_read_json("reports/qa/live_metrics.json", {})
valid     = try_read_json("reports/qa/validation_summary.json", {})

# Build ASCII-safe summary
lines = []
lines.append(f"Run Summary - {stamp}")
lines.append("-" * 60)
lines.append(f"Positions: {portfolio.get('positions','?')}")
lines.append(f"Total Stake: {portfolio.get('total_stake','?')}")
by_mkt = portfolio.get("by_market", {})
by_book = portfolio.get("by_book", {})
if by_mkt: lines.append("By Market: " + ", ".join(f"{k}={v}" for k,v in by_mkt.items()))
if by_book: lines.append("By Book: " + ", ".join(f"{k}={v}" for k,v in by_book.items()))
lines.append("")
if metrics:
    lines.append("[Live Outcomes]")
    lines.append(f"Resolved: {metrics.get('resolved','?')}  Wins: {metrics.get('wins','?')}  Losses: {metrics.get('losses','?')}")
    lines.append(f"ROI: {metrics.get('roi','?'):.4f}  Avg CLV: {metrics.get('avg_clv','?'):.4f}  % Beats Close: {metrics.get('pct_beats_close','?'):.4f}")
lines.append("")
if valid:
    lines.append("[Validation]")
    lines.append(f"Brier: {valid.get('brier','?')}  LogLoss: {valid.get('logloss','?')}")

text = ("\r\n").join(lines) + "\r\n"
with open(summary_path, "w", encoding="utf-8", newline="\r\n") as f:
    f.write(text)

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for fpath in candidates:
        if os.path.exists(fpath):
            z.write(fpath)

print(json.dumps({"zip": str(zip_path), "summary": str(summary_path)}))
