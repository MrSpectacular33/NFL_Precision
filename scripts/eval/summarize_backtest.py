# scripts/eval/summarize_backtest.py
import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def kelly_fraction(p: np.ndarray, dec_odds: float) -> np.ndarray:
    """
    Kelly f* for +EV single-outcome bet at decimal odds.
    f* = (b*p - (1-p)) / b where b = dec_odds - 1
    Negative results -> 0.
    """
    b = dec_odds - 1.0
    f = (b * p - (1.0 - p)) / b
    return np.clip(f, 0.0, 1.0)


def implied_ev_per_dollar(p: np.ndarray, american: float) -> np.ndarray:
    """
    Expected value per $1 staked at given odds assuming
    win prob = p, lose prob = (1-p).
    """
    dec = american_to_decimal(american)
    b = dec - 1.0
    # Win returns: +b per $1 (profit only). Lose: -1.
    return p * b - (1.0 - p) * 1.0


def load_schedules_parquet(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # Normalize column names a bit
    df.columns = [c.lower() for c in df.columns]
    # Expect at least: game_id, season, week, home_team, away_team, home_score, away_score
    need = {"game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"[schedules] missing columns: {sorted(missing)} in {parquet_path}")
    return df


def compute_outcomes(edges: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Join realized outcomes:
      - Spread (cover/push/lose) for listed 'team' using market_spread (home-perspective)
      - Total (over/push/under) using market_total
    Assumptions:
      - edges has columns: ['game_id','team','market_spread','p_cover','market_total','p_over', ...]
      - schedules has: game_id, home_team, away_team, home_score, away_score
      - market_spread is home-perspective (negative if home favored).
      - For an away 'team' row, effective line for that team is -market_spread.
    """
    e = edges.copy()
    # lower-case to merge safely
    for c in ("team",):
        if c in e.columns:
            e[c] = e[c].astype(str)

    sched = schedules.copy()

    merged = e.merge(
        sched[["game_id", "home_team", "away_team", "home_score", "away_score", "season", "week"]],
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    # sanity
    if merged["home_team"].isna().any():
        missing = merged[merged["home_team"].isna()]["game_id"].unique().tolist()
        raise ValueError(f"[compute_outcomes] schedule join failed for game_ids: {missing[:5]}...")

    # figure out if the listed row 'team' is home or away
    merged["is_home_row"] = (merged["team"] == merged["home_team"])

    # realized team margin
    merged["team_score"] = np.where(merged["is_home_row"], merged["home_score"], merged["away_score"])
    merged["opp_score"] = np.where(merged["is_home_row"], merged["away_score"], merged["home_score"])
    merged["final_margin_team"] = merged["team_score"] - merged["opp_score"]
    merged["final_total"] = merged["home_score"] + merged["away_score"]

    # line from that team's perspective
    # home perspective spread provided (home minus away). For away team, invert.
    merged["team_line"] = np.where(merged["is_home_row"], merged["market_spread"], -merged["market_spread"])

    # spread outcomes for the row's side
    merged["spread_result"] = merged["final_margin_team"] - merged["team_line"]
    merged["won_spread"] = (merged["spread_result"] > 0).astype(int)
    merged["push_spread"] = (merged["spread_result"] == 0).astype(int)
    merged["lost_spread"] = (merged["spread_result"] < 0).astype(int)

    # total outcomes (OVER relative to market_total)
    merged["total_result"] = merged["final_total"] - merged["market_total"]
    merged["won_over"] = (merged["total_result"] > 0).astype(int)
    merged["push_total"] = (merged["total_result"] == 0).astype(int)
    merged["lost_over"] = (merged["total_result"] < 0).astype(int)

    return merged


def simulate_bankroll(rows: pd.DataFrame,
                      prob_col: str,
                      american_odds: float,
                      kelly_mult: float = 0.5,
                      cap_pct: float = 0.02,
                      floor_pct: float = 0.0,
                      initial_bankroll: float = 10000.0,
                      outcome_win_col: str = "won_spread",
                      outcome_push_col: str = "push_spread") -> dict:
    """
    Simulate realized bankroll path for a set of homogeneous bets (all spreads OR all totals).
    Bets are processed in input order.
    """
    dec = american_to_decimal(american_odds)
    b = dec - 1.0

    bk = initial_bankroll
    equity_curve = [bk]
    stakes = []
    wins = 0
    pushes = 0
    losses = 0
    pnl = 0.0

    for _, r in rows.iterrows():
        p = float(r[prob_col])
        f_star = kelly_fraction(np.array([p]), dec_odds=dec)[0]
        f = min(kelly_mult * f_star, cap_pct)   # half-Kelly with cap
        if f < floor_pct:
            continue
        stake = bk * f
        stakes.append(stake)

        # realized outcome
        win = int(r[outcome_win_col]) == 1
        push = int(r[outcome_push_col]) == 1
        if push:
            # stake returned
            equity_curve.append(bk)
            pushes += 1
            continue

        if win:
            profit = stake * b
            pnl += profit
            bk += profit
            wins += 1
        else:
            pnl -= stake
            bk -= stake
            losses += 1

        equity_curve.append(bk)

    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "bets": wins + losses + pushes,
        "pnl": pnl,
        "final_bankroll": bk,
        "max_drawdown": float(np.max(np.maximum.accumulate(equity_curve) - np.array(equity_curve))) if equity_curve else 0.0,
        "avg_stake_pct": float(np.mean(stakes) / initial_bankroll) if stakes else 0.0,
    }


def bucket_calibration(rows: pd.DataFrame,
                       prob_col: str,
                       outcome_win_col: str,
                       n_bins: int = 10) -> pd.DataFrame:
    """
    Reliability table: bins of predicted prob vs realized win rate.
    """
    df = rows.copy()
    df["bin"] = pd.cut(df[prob_col], bins=np.linspace(0.5, 0.7, n_bins + 1), include_lowest=True, right=False)
    agg = df.groupby("bin").agg(
        n=("game_id", "count"),
        p_mean=(prob_col, "mean"),
        hit_rate=(outcome_win_col, "mean"),
    ).reset_index()
    agg["calibration_gap"] = agg["hit_rate"] - agg["p_mean"]
    return agg


def plot_reliability(calib: pd.DataFrame, title: str, out_png: Path):
    plt.figure()
    x = calib["p_mean"]
    y = calib["hit_rate"]
    plt.scatter(x, y)
    # 45-degree line
    minxy = min(x.min(), y.min()) if len(x) and len(y) else 0
    maxxy = max(x.max(), y.max()) if len(x) and len(y) else 1
    plt.plot([minxy, maxxy], [minxy, maxxy])
    plt.xlabel("Predicted probability")
    plt.ylabel("Realized hit rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges-dir", default="reports/backtest", help="Folder containing edges_all_s{SEASON}.csv")
    ap.add_argument("--schedules-parquet", default="data/raw/schedules_2019_2025.parquet")
    ap.add_argument("--season-start", type=int, required=True)
    ap.add_argument("--season-end", type=int, required=True)
    ap.add_argument("--american-odds", type=float, default=-110)
    ap.add_argument("--flat-stake", type=float, default=100.0)
    ap.add_argument("--kelly-mult", type=float, default=0.5)
    ap.add_argument("--kelly-cap", type=float, default=0.02)
    ap.add_argument("--kelly-floor", type=float, default=0.0)
    ap.add_argument("--p-min", type=float, default=0.525, help="min p_cover to take a spread bet")
    ap.add_argument("--p-over-min", type=float, default=0.525, help="min p_over to take a totals bet")
    ap.add_argument("--outdir", default="reports/backtest")
    ap.add_argument("--n-bins", type=int, default=10, help="bins for reliability tables")
    args = ap.parse_args()

    edges_dir = Path(args.edges_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    schedules = load_schedules_parquet(Path(args.schedules_parquet))

    rows = []
    for season in range(args.season_start, args.season_end + 1):
        f = edges_dir / f"edges_all_s{season}.csv"
        if not f.exists():
            print(f"[WARN] No edges file for {season}: {f}")
            continue
        e = pd.read_csv(f)
        e["season"] = season
        rows.append(e)

    if not rows:
        print("[ERROR] No edges to summarize.")
        return

    edges_all = pd.concat(rows, ignore_index=True)
    # ensure required columns
    need_cols = ["game_id", "team", "market_spread", "market_total", "p_cover", "p_over"]
    missing = [c for c in need_cols if c not in edges_all.columns]
    if missing:
        raise ValueError(f"[edges] missing columns: {missing}")

    # attach outcomes
    full = compute_outcomes(edges_all, schedules)

    # theoretical EV per $ staked
    full["ev_per_dollar_spread"] = implied_ev_per_dollar(full["p_cover"].values, args.american_odds)
    full["ev_per_dollar_total"] = implied_ev_per_dollar(full["p_over"].values, args.american_odds)

    # decide which bets to place
    full["take_spread"] = (full["p_cover"] >= args.p_min) & (full["ev_per_dollar_spread"] > 0)
    full["take_total"] = (full["p_over"] >= args.p_over_min) & (full["ev_per_dollar_total"] > 0)

    # Flat $ stakes, realized PnL
    dec = american_to_decimal(args.american_odds)
    b = dec - 1.0

    spread_bets = full[full["take_spread"]].copy()
    total_bets = full[full["take_total"]].copy()

    def realized_pnl_flat(df: pd.DataFrame, win_col: str, push_col: str) -> float:
        if df.empty:
            return 0.0
        wins = df[win_col].sum()
        pushes = df[push_col].sum()
        losses = len(df) - wins - pushes
        return args.flat_stake * (wins * b - losses * 1.0)

    pnl_flat_spread = realized_pnl_flat(spread_bets, "won_spread", "push_spread")
    pnl_flat_total = realized_pnl_flat(total_bets, "won_over", "push_total")
    pnl_flat_all = pnl_flat_spread + pnl_flat_total

    stake_flat_all = args.flat_stake * (len(spread_bets) - spread_bets["push_spread"].sum() +
                                        len(total_bets) - total_bets["push_total"].sum())

    roi_flat = (pnl_flat_all / stake_flat_all) if stake_flat_all > 0 else 0.0

    # Kelly simulations (realized)
    kelly_spread = simulate_bankroll(
        spread_bets, prob_col="p_cover", american_odds=args.american_odds,
        kelly_mult=args.kelly_mult, cap_pct=args.kelly_cap, floor_pct=args.kelly_floor,
        initial_bankroll=10000.0, outcome_win_col="won_spread", outcome_push_col="push_spread",
    )
    kelly_total = simulate_bankroll(
        total_bets, prob_col="p_over", american_odds=args.american_odds,
        kelly_mult=args.kelly_mult, cap_pct=args.kelly_cap, floor_pct=args.kelly_floor,
        initial_bankroll=10000.0, outcome_win_col="won_over", outcome_push_col="push_total",
    )
    pnl_kelly = kelly_spread["pnl"] + kelly_total["pnl"]
    roi_kelly = pnl_kelly / 10000.0

    # Theoretical EV sums (per-season we’ll group next)
    full["ev_spread_$"] = full["take_spread"] * args.flat_stake * full["ev_per_dollar_spread"]
    full["ev_total_$"] = full["take_total"] * args.flat_stake * full["ev_per_dollar_total"]

    # per-season summary
    def summarize_season(df: pd.DataFrame, season: int) -> dict:
        df_s = df[df["season"] == season]
        sp = df_s[df_s["take_spread"]]
        to = df_s[df_s["take_total"]]

        pnl_flat_s = realized_pnl_flat(sp, "won_spread", "push_spread")
        pnl_flat_t = realized_pnl_flat(to, "won_over", "push_total")
        staked_s = args.flat_stake * (len(sp) - sp["push_spread"].sum())
        staked_t = args.flat_stake * (len(to) - to["push_total"].sum())
        staked_all = staked_s + staked_t
        roi_flat_szn = (pnl_flat_s + pnl_flat_t) / staked_all if staked_all > 0 else 0.0

        # kelly per season
        k_sp = simulate_bankroll(
            sp, prob_col="p_cover", american_odds=args.american_odds,
            kelly_mult=args.kelly_mult, cap_pct=args.kelly_cap, floor_pct=args.kelly_floor,
            initial_bankroll=10000.0, outcome_win_col="won_spread", outcome_push_col="push_spread",
        )
        k_to = simulate_bankroll(
            to, prob_col="p_over", american_odds=args.american_odds,
            kelly_mult=args.kelly_mult, cap_pct=args.kelly_cap, floor_pct=args.kelly_floor,
            initial_bankroll=10000.0, outcome_win_col="won_over", outcome_push_col="push_total",
        )
        roi_kelly_szn = (k_sp["pnl"] + k_to["pnl"]) / 10000.0

        return {
            "season": season,
            "n_spread_bets": int(len(sp)),
            "n_total_bets": int(len(to)),
            "ev_spread": float(sp["ev_spread_$"].sum()),
            "ev_total": float(to["ev_total_$"].sum()),
            "ev_all": float(sp["ev_spread_$"].sum() + to["ev_total_$"].sum()),
            "roi_flat": float(roi_flat_szn),
            "roi_kelly": float(roi_kelly_szn),
            "kelly_spread_max_dd": float(k_sp["max_drawdown"]),
            "kelly_total_max_dd": float(k_to["max_drawdown"]),
            "avg_kelly_stake_pct_spread": float(k_sp["avg_stake_pct"]),
            "avg_kelly_stake_pct_total": float(k_to["avg_stake_pct"]),
        }

    seasons = sorted(full["season"].unique())
    rows = [summarize_season(full, s) for s in seasons]
    summary = pd.DataFrame(rows)
    out_summary = outdir / f"summary_seasons_{seasons[0]}_{seasons[-1]}.csv"
    summary.to_csv(out_summary, index=False)
    print(f"[SUMMARY] wrote -> {out_summary}")

    # Reliability (bucket) tables + plots
    # spreads
    sp_calib = bucket_calibration(full[full["take_spread"]], "p_cover", "won_spread", n_bins=args.n_bins)
    out_spread_csv = outdir / f"calibration_spreads_{seasons[0]}_{seasons[-1]}.csv"
    sp_calib.to_csv(out_spread_csv, index=False)
    plot_reliability(sp_calib, "Spread reliability", outdir / f"calibration_spreads_{seasons[0]}_{seasons[-1]}.png")

    # totals
    to_calib = bucket_calibration(full[full["take_total"]], "p_over", "won_over", n_bins=args.n_bins)
    out_totals_csv = outdir / f"calibration_totals_{seasons[0]}_{seasons[-1]}.csv"
    to_calib.to_csv(out_totals_csv, index=False)
    plot_reliability(to_calib, "Total reliability", outdir / f"calibration_totals_{seasons[0]}_{seasons[-1]}.png")

    print("[DONE] summarize_backtest OK")


if __name__ == "__main__":
    main()
