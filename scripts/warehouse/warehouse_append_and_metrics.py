import pandas as pd, os, json, datetime as dt, sys

def main(exec_csv, outcomes_csv, out_csv):
    # Join execution sheet to outcomes
    bets = pd.read_csv(exec_csv)
    oc   = pd.read_csv(outcomes_csv)

    # Normalize keys
    for c in ("bet_id","result"):
        if c not in oc.columns: sys.exit(f"outcomes missing {c}")
    if "bet_id" not in bets.columns: sys.exit("exec missing bet_id")

    m = bets.merge(oc, on="bet_id", how="left")
    m["price"]  = pd.to_numeric(m["price"], errors="coerce").fillna(-110)
    m["stake"]  = pd.to_numeric(m["stake"], errors="coerce").fillna(0)
    m["result"] = m["result"].fillna("pending")
    m["closing_price"] = pd.to_numeric(m.get("closing_price"), errors="coerce")

    # P&L using American odds
    def dec(x): x=float(x); return 1+(100/abs(x)) if x<0 else 1+(x/100)
    m["our_dec"]   = m["price"].map(dec)
    m["close_dec"] = m["closing_price"].map(dec) if "closing_price" in m else None
    m["pnl"] = 0.0
    m.loc[m["result"].eq("win"),  "pnl"] = m["stake"] * (m["our_dec"] - 1.0)
    m.loc[m["result"].eq("lose"), "pnl"] = -m["stake"]
    # push => 0

    # CLV
    if "close_dec" in m:
        m["clv"] = (m["our_dec"] - m["close_dec"]) / m["close_dec"]

    # Write joined output
    cols = ["bet_id","game_id","market","price","stake","result","closing_price","pnl","our_dec","close_dec","clv"]
    cols = [c for c in cols if c in m.columns]
    m[cols].to_csv(out_csv, index=False)

    # Warehouse + metrics
    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_date = dt.date.today()
    arch = f"data/warehouse/bet_outcomes_{run_ts}.csv"
    cum  = "data/warehouse/bet_outcomes_all.csv"
    os.makedirs("data/warehouse", exist_ok=True)
    os.makedirs("reports/qa", exist_ok=True)

    df = pd.read_csv(out_csv)
    df["run_date"] = str(run_date)
    df.to_csv(arch, index=False)
    if os.path.exists(cum):
        df.to_csv(cum, mode="a", header=False, index=False)
    else:
        df.to_csv(cum, index=False)

    k = df[df["result"].isin(["win","lose","push"])].copy()
    pnl_total   = float(k["pnl"].sum()) if len(k) else 0.0
    stake_total = float(k["stake"].sum()) if len(k) else 0.0
    roi         = (pnl_total / stake_total) if stake_total else 0.0
    wins        = int((k["pnl"] > 0).sum())
    losses      = int((k["pnl"] < 0).sum())
    avg_clv     = float(df["clv"].dropna().mean()) if "clv" in df.columns else float("nan")
    pct_beats   = float((df["clv"] > 0).mean())     if "clv" in df.columns else float("nan")

    metrics = {
      "as_of": dt.datetime.now().isoformat(timespec="seconds"),
      "resolved": int(len(k)),
      "wins": wins, "losses": losses,
      "pnl_total": pnl_total, "stake_total": stake_total, "roi": roi,
      "avg_clv": avg_clv, "pct_beats_close": pct_beats
    }
    open("reports/qa/live_metrics.json","w").write(json.dumps(metrics, indent=2))

    print(json.dumps({
      "archived": arch,
      "cumulative": cum,
      "metrics_file": "reports/qa/live_metrics.json",
      **metrics
    }, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", dest="exec", required=True)
    ap.add_argument("--outcomes", dest="outcomes", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()
    main(a.exec, a.outcomes, a.out)
