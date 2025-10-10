import os, argparse, numpy as np, pandas as pd

def american_to_decimal(price):
    price = float(price)
    if price > 0:
        return 1.0 + price/100.0
    else:
        return 1.0 + 100.0/abs(price)

def simulate_path(win_probs, stakes, dec_odds, rng):
    """
    Simulate one full sequence (same order) to allow max drawdown.
    Returns: total PnL, max_drawdown (positive number).
    """
    # Bernoulli results
    wins = rng.random(len(win_probs)) < win_probs
    # PnL per bet
    pnl = np.where(wins, stakes*(dec_odds-1.0), -stakes)
    # cumulative equity and drawdown
    equity = pnl.cumsum()
    peak = np.maximum.accumulate(np.insert(equity, 0, 0.0))[1:]
    dd = peak - equity
    max_dd = float(dd.max() if dd.size else 0.0)
    return float(pnl.sum()), max_dd

def main():
    ap = argparse.ArgumentParser(description="Monte Carlo bankroll from placed_bets.csv")
    ap.add_argument("--in", dest="infile", default="reports/backtest/placed_bets.csv")
    ap.add_argument("--out", default="reports/risk/mc_from_placed_bets_summary.csv")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prob-col", default=None, help="Override prob col (default: p_blend if present else p_cal)")
    args = ap.parse_args()

    if not os.path.exists(args.infile):
        raise SystemExit(f"Missing input: {args.infile}")

    df = pd.read_csv(args.infile)
    need = {"stake","price"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Missing required columns in placed_bets: {miss}")

    # choose probability column
    prob_col = args.prob_col
    if prob_col is None:
        if "p_blend" in df.columns: prob_col = "p_blend"
        elif "p_cal" in df.columns: prob_col = "p_cal"
        else: raise SystemExit("No p_blend or p_cal column found. Provide --prob-col.")
    if prob_col not in df.columns:
        raise SystemExit(f"Column {prob_col} not in placed_bets.")

    d = df.copy()
    d = d[~d[prob_col].isna()].copy()
    if d.empty:
        raise SystemExit("No rows with probabilities to simulate.")

    # Vectorize odds to decimal
    dec = d["price"].astype(float).apply(american_to_decimal).values.astype(float)
    p   = d[prob_col].astype(float).clip(1e-5, 1-1e-5).values
    st  = d["stake"].astype(float).values

    rng = np.random.default_rng(args.seed)

    totals = np.empty(args.sims, dtype=float)
    maxdds = np.empty(args.sims, dtype=float)

    # Simulate respecting current bet order (one path ~ a season path),
    # and also a random shuffle each time to capture sequencing risk.
    idx = np.arange(len(p))
    for i in range(args.sims):
        # shuffle once per path to vary sequencing risk
        rng.shuffle(idx)
        pnl, mdd = simulate_path(p[idx], st[idx], dec[idx], rng)
        totals[i] = pnl
        maxdds[i] = mdd

    def pct(a, q): return float(np.percentile(a, q))

    out = pd.DataFrame([{
        "n_bets": int(len(d)),
        "sims": int(args.sims),
        "mean_pnl": float(totals.mean()),
        "median_pnl": float(np.median(totals)),
        "pnl_p05": pct(totals, 5),
        "pnl_p01": pct(totals, 1),
        "pnl_p95": pct(totals, 95),
        "prob_loss": float((totals < 0).mean()),
        "mean_max_drawdown": float(maxdds.mean()),
        "median_max_drawdown": float(np.median(maxdds)),
        "maxdd_p95": pct(maxdds, 95),
        "maxdd_p99": pct(maxdds, 99),
    }])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
