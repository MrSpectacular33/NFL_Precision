import argparse, os, pandas as pd, numpy as np

def american_to_b(price):
    price = float(price)
    return (price/100.0) if price>0 else (100.0/abs(price))  # net decimal b

def simulate(df, bankroll0=10000, kelly_frac=0.25,
             per_bet_cap=0.02, weekly_risk_cap=0.10, pause_drawdown=0.30,
             # global thresholds (fallbacks)
             min_ev=0.0, min_edge_pts=0.0,
             # per-market thresholds
             min_ev_spread=None, min_ev_total=None,
             min_edge_pts_spread=None, min_edge_pts_total=None,
             placed_out=None):
    df = df.copy()

    # Require calibrated probabilities
    if "p_cal" not in df.columns:
        raise ValueError("Expected p_cal in input. Use outcomes_long_with_pcal.csv.")
    if "price" not in df.columns:
        df["price"] = -110
    if "edge_pts" not in df.columns:
        df["edge_pts"] = 0.0

    # Per-market gates (default to global if not provided)
    def ev_gate(r):
        if str(r["bet_type"]).lower()=="spread" and min_ev_spread is not None: return float(min_ev_spread)
        if str(r["bet_type"]).lower()=="total"  and min_ev_total  is not None: return float(min_ev_total)
        return float(min_ev)
    def edge_gate(r):
        if str(r["bet_type"]).lower()=="spread" and min_edge_pts_spread is not None: return float(min_edge_pts_spread)
        if str(r["bet_type"]).lower()=="total"  and min_edge_pts_total  is not None: return float(min_edge_pts_total)
        return float(min_edge_pts)

    # EV per row
    b = df["price"].abs().apply(american_to_b)
    df["ev"] = df["p_cal"]*b - (1 - df["p_cal"])

    # Selection mask with per-market gates
    take_mask = []
    for _, r in df.iterrows():
        if r["ev"] >= ev_gate(r) and abs(r["edge_pts"]) >= edge_gate(r):
            take_mask.append(True)
        else:
            take_mask.append(False)
    df = df.loc[take_mask].copy().sort_values(["season","week"]).reset_index(drop=True)

    bankroll = bankroll0
    peak = bankroll0
    dd_pause = False
    path = []
    placed = []   # placed-bets log
    placed_total = 0
    stake_total = 0.0

    for (season, week), wk in df.groupby(["season","week"], sort=True):
        if dd_pause:
            path.append({"season":season, "week":week, "bankroll":bankroll, "bets":0})
            if bankroll >= (1 - pause_drawdown) * peak:
                dd_pause = False
            continue

        wk_risk = 0.0
        bets_this_week = 0

        for _, r in wk.iterrows():
            p = float(r["p_cal"])
            b = american_to_b(r["price"])
            q = 1 - p
            k = (b*p - q) / b  # Kelly
            f = max(0.0, min(per_bet_cap, kelly_frac * k))
            remaining = max(0.0, weekly_risk_cap - wk_risk)
            f = min(f, remaining)
            if f <= 0.0:
                continue
            stake = bankroll * f
            won = bool(r.get("won", 0))
            pnl = stake * (b if won else -1)

            # record bet
            placed.append({
                "season": int(r.get("season", 0)),
                "week":   int(r.get("week", 0)),
                "game_id": r.get("game_id", ""),
                "bet_type": r.get("bet_type", ""),
                "pick":    r.get("pick",""),
                "bet_line": float(r.get("bet_line", np.nan)),
                "price":   float(r.get("price", -110)),
                "p_cal":   float(p),
                "ev":      float(r["ev"]),
                "edge_pts": float(r.get("edge_pts", 0.0)),
                "stake":   float(stake),
                "won":     int(won),
                "pnl":     float(pnl)
            })

            bankroll += pnl
            bets_this_week += 1
            placed_total += 1
            stake_total += stake
            wk_risk += f
            peak = max(peak, bankroll)

        if bankroll < (1 - pause_drawdown) * peak:
            dd_pause = True

        path.append({"season":season, "week":week, "bankroll":bankroll, "bets":bets_this_week})

    path = pd.DataFrame(path)
    if len(path)==0:
        roi = 0.0
        max_dd = 0.0
    else:
        roi = (bankroll - bankroll0)/bankroll0
        max_dd = 1 - (path["bankroll"]/path["bankroll"].cummax()).min()

    summary = pd.DataFrame([{
        "final_bankroll": round(bankroll,2),
        "roi": round(roi,4),
        "max_drawdown": round(max_dd,4),
        "n_weeks": int(len(path)),
        "n_rows_input": int(len(df)),
        "n_placed_bets": int(placed_total),
        "avg_stake": round(stake_total/max(1,placed_total),2),
        "kelly_frac": kelly_frac,
        "per_bet_cap": per_bet_cap,
        "weekly_risk_cap": weekly_risk_cap,
        "pause_drawdown": pause_drawdown
    }])

    if placed_out:
        os.makedirs(os.path.dirname(placed_out), exist_ok=True)
        pd.DataFrame(placed).to_csv(placed_out, index=False)

    return path, summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bankroll0", type=float, default=10000)
    ap.add_argument("--kelly-frac", type=float, default=0.25)
    ap.add_argument("--per-bet-cap", type=float, default=0.02)
    ap.add_argument("--weekly-risk-cap", type=float, default=0.10)
    ap.add_argument("--pause-drawdown", type=float, default=0.30)
    ap.add_argument("--min-ev", type=float, default=0.0)
    ap.add_argument("--min-edge-pts", type=float, default=0.0)
    ap.add_argument("--min-ev-spread", type=float)
    ap.add_argument("--min-ev-total", type=float)
    ap.add_argument("--min-edge-pts-spread", type=float)
    ap.add_argument("--min-edge-pts-total", type=float)
    ap.add_argument("--placed-out", type=str, default="reports/backtest/placed_bets.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    os.makedirs(args.outdir, exist_ok=True)
    path, summ = simulate(
        df,
        bankroll0=args.bankroll0,
        kelly_frac=args.kelly_frac,
        per_bet_cap=args.per_bet_cap,
        weekly_risk_cap=args.weekly_risk_cap,
        pause_drawdown=args.pause_drawdown,
        min_ev=args.min_ev, min_edge_pts=args.min_edge_pts,
        min_ev_spread=args.min_ev_spread, min_ev_total=args.min_ev_total,
        min_edge_pts_spread=args.min_edge_pts_spread, min_edge_pts_total=args.min_edge_pts_total,
        placed_out=args.placed_out
    )
    path.to_csv(os.path.join(args.outdir,"bankroll_path.csv"), index=False)
    summ.to_csv(os.path.join(args.outdir,"bankroll_summary.csv"), index=False)
    print(f"[OK] wrote bankroll_path.csv and bankroll_summary.csv to {args.outdir}")
    print(f"[OK] wrote placed bets -> {args.placed_out}")
