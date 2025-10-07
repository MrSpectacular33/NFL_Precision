# scripts/eval/plot_buckets.py
import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def _require_cols(df, needed, label):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}. "
                         f"Has: {list(df.columns)}")

def _bucket_lower(b):
    """Extract a numeric lower bound from bucket labels like '0-0.5', '3.5-4', '5+'."""
    s = str(b).strip()
    if s.endswith("+"):
        s = s[:-1]
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return float("inf")
    return float(m.group(1))

def _sort_buckets(df):
    out = df.copy()
    out["__key"] = out["bucket"].map(_bucket_lower)
    return out.sort_values("__key").drop(columns="__key")

def plot_calibration(df, title, outfile):
    plt.figure(figsize=(8, 6))
    plt.plot(df["model_p"], df["emp_p"], marker="o", label="Empirical")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Model predicted probability")
    plt.ylabel("Empirical probability")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote -> {outfile}")

def plot_roi_bars(df, title, outfile):
    df_sorted = _sort_buckets(df)
    plt.figure(figsize=(10, 6))
    x = range(len(df_sorted))
    y = df_sorted["roi_flat_1u"]
    plt.bar(x, y)
    plt.xticks(ticks=x, labels=df_sorted["bucket"], rotation=45, ha="right")
    plt.ylabel("ROI (flat 1u)")
    plt.xlabel("Edge bucket")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    # annotate with n_bets
    for xi, yi, nb in zip(x, y, df_sorted["n_bets"]):
        plt.text(xi, yi if yi >= 0 else 0, f"n={int(nb)}", ha="center",
                 va="bottom" if yi >= 0 else "top", fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote -> {outfile}")

def cumulative_roi_from_buckets(df, side_label):
    """
    Build a cumulative ROI vs min-edge curve using bucket summaries.
    Method: sort buckets by lower edge bound descending; at each threshold,
    compute weighted mean ROI across all buckets whose lower bound >= threshold.
    """
    work = df.copy()
    work["edge_min"] = work["bucket"].map(_bucket_lower)
    # Sort by edge_min descending for threshold sweep
    work = work.sort_values("edge_min", ascending=False)

    rows = []
    cum_bets = 0
    cum_pnl_flat = 0.0
    for _, r in work.iterrows():
        n = int(r["n_bets"])
        roi = float(r["roi_flat_1u"])  # per-bet average ROI for the bucket
        pnl_bucket = roi * n           # total P&L (in "units") for that bucket
        cum_bets += n
        cum_pnl_flat += pnl_bucket
        if cum_bets > 0:
            cum_roi = cum_pnl_flat / cum_bets
        else:
            cum_roi = 0.0
        rows.append({
            "side": side_label,
            "threshold_edge": r["edge_min"],
            "n_bets_cum": cum_bets,
            "roi_flat_1u_cum": cum_roi
        })

    cum_df = pd.DataFrame(rows)
    # Collapse same thresholds (rare) by taking the last (most cumulative)
    cum_df = cum_df.groupby(["side", "threshold_edge"], as_index=False).last()
    # Sort by threshold descending for a clean right-to-left curve
    cum_df = cum_df.sort_values("threshold_edge", ascending=False)
    return cum_df

def plot_cumulative(cum_df, title, outfile):
    plt.figure(figsize=(10, 6))
    for side, g in cum_df.groupby("side"):
        g_sorted = g.sort_values("threshold_edge", ascending=False)
        plt.plot(g_sorted["threshold_edge"], g_sorted["roi_flat_1u_cum"], marker="o", label=side)
    plt.xlabel("Minimum edge threshold (bucket lower bound)")
    plt.ylabel("Cumulative ROI (flat 1u)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote -> {outfile}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spread-file", required=True, help="CSV from edge_buckets.py for spreads")
    ap.add_argument("--total-file", required=True, help="CSV from edge_buckets.py for totals")
    ap.add_argument("--outdir", required=True, help="Where to save plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    spread = pd.read_csv(args.spread_file)
    total  = pd.read_csv(args.total_file)

    needed = ["bucket", "n_bets", "model_p", "emp_p", "roi_flat_1u"]
    _require_cols(spread, needed, "spread-file")
    _require_cols(total,  needed, "total-file")

    # 1) Calibration plots
    plot_calibration(
        spread,
        "Spread Bucket Calibration",
        os.path.join(args.outdir, "spread_buckets.png"),
    )
    plot_calibration(
        total,
        "Total Bucket Calibration",
        os.path.join(args.outdir, "total_buckets.png"),
    )

    # 2) ROI-by-bucket bar charts
    plot_roi_bars(
        spread,
        "Spread ROI by Edge Bucket (flat 1u)",
        os.path.join(args.outdir, "spread_bucket_roi.png"),
    )
    plot_roi_bars(
        total,
        "Total ROI by Edge Bucket (flat 1u)",
        os.path.join(args.outdir, "total_bucket_roi.png"),
    )

    # 3) Cumulative ROI vs edge threshold (using bucket summaries)
    spread_cum = cumulative_roi_from_buckets(spread, side_label="Spread")
    total_cum  = cumulative_roi_from_buckets(total,  side_label="Total")
    cum_all = pd.concat([spread_cum, total_cum], ignore_index=True)

    # Save CSVs
    spread_cum_path = os.path.join(args.outdir, "spread_cum_roi.csv")
    total_cum_path  = os.path.join(args.outdir, "total_cum_roi.csv")
    cum_all_path    = os.path.join(args.outdir, "cum_roi_all.csv")
    spread_cum.to_csv(spread_cum_path, index=False)
    total_cum.to_csv(total_cum_path,  index=False)
    cum_all.to_csv(cum_all_path,      index=False)
    print(f"[OK] wrote -> {spread_cum_path}")
    print(f"[OK] wrote -> {total_cum_path}")
    print(f"[OK] wrote -> {cum_all_path}")

    # Plot combined cumulative chart
    plot_cumulative(
        cum_all,
        "Cumulative ROI vs Edge Threshold (flat 1u)",
        os.path.join(args.outdir, "cum_roi_vs_threshold.png"),
    )

if __name__ == "__main__":
    main()
