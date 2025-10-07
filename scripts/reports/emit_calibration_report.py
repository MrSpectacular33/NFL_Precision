import os, argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def qbin(s, q=8):
    return pd.qcut(s, q, duplicates="drop")

def rel_table(dframe, label="overall"):
    g = dframe.groupby("bin", observed=True).agg(
        p=("p_cal","mean"), hit=("won","mean"), n=("won","size")
    ).reset_index()
    g.insert(0, "scope", label)
    return g

def plot_reliability(df, title, out_png):
    if df.empty: return
    p = df["p"].values; h = df["hit"].values
    fig = plt.figure(figsize=(6,6)); ax = plt.gca()
    ax.plot([0,1],[0,1], linestyle="--")
    ax.scatter(p, h, s=np.clip(df["n"].values/10, 10, 100))
    for _, r in df.iterrows():
        ax.annotate(f"{int(r['n'])}", (r["p"], r["hit"]), xytext=(3,3), textcoords="offset points")
    ax.set_xlabel("Predicted win prob"); ax.set_ylabel("Observed win rate")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="reports/backtest/outcomes_long_with_pcal.csv")
    ap.add_argument("--outdir", default="reports/metrics")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--min-ev-spread", type=float, default=0.015)
    ap.add_argument("--min-edge-pts-spread", type=float, default=1.0)
    ap.add_argument("--price", type=float, default=-110.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.infile)
    need = {"p_cal","won","bet_type"}
    miss = need - set(df.columns)
    if miss: raise SystemExit(f"Missing columns in {args.infile}: {miss}")

    outs_overall, outs_season = [], []
    for mkt in ("spread","total"):
        d = df[df["bet_type"]==mkt].copy()
        if d.empty: continue
        d["bin"] = qbin(d["p_cal"], q=args.bins)
        over = rel_table(d, label=mkt); outs_overall.append(over)
        if "season" in d.columns:
            for s, ds in d.groupby("season", sort=True):
                ds = ds.copy()
                ds["bin"] = qbin(ds["p_cal"], q=min(args.bins, max(2, ds["p_cal"].nunique())))
                tab = rel_table(ds, label=f"{int(s)}_{mkt}"); tab.insert(1,"season",int(s))
                outs_season.append(tab)
        plot_reliability(over[["p","hit","n"]],
            f"Reliability – {mkt} (overall)", os.path.join(args.outdir, f"reliability_{mkt}.png"))

    if outs_overall:
        pd.concat(outs_overall, ignore_index=True).to_csv(
            os.path.join(args.outdir,"reliability_overall.csv"), index=False)
    if outs_season:
        pd.concat(outs_season, ignore_index=True).to_csv(
            os.path.join(args.outdir,"reliability_seasonal.csv"), index=False)

    sp = df[df["bet_type"]=="spread"].copy()
    if not sp.empty:
        b = (args.price/100.0) if args.price>0 else (100.0/abs(args.price))
        sp["ev"] = sp["p_cal"]*b - (1-sp["p_cal"])
        mask = (sp["ev"]>=args.min_ev_spread) & (sp["edge_pts"].abs()>=args.min_edge_pts_spread)
        spb = sp[mask].copy()
        if len(spb):
            spb["bin"] = qbin(spb["p_cal"], q=min(args.bins, max(2, spb["p_cal"].nunique())))
            bettab = rel_table(spb, label="spread_bettable")
            bettab.to_csv(os.path.join(args.outdir,"reliability_spread_bettable.csv"), index=False)
            plot_reliability(bettab[["p","hit","n"]],
                "Reliability – spread (bettable subset)", os.path.join(args.outdir, "reliability_spread_bettable.png"))

    print(f"[OK] wrote reliability to {args.outdir}")

if __name__ == "__main__":
    main()
