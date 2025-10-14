import pandas as pd, numpy as np, os, json, matplotlib.pyplot as plt

CAL = r"reports/qa/calibration_table.csv"
JOIN = r"reports/qa/validation_joined_sample.csv"
OUTDIR = r"reports/qa"

def plot_reliability(cal_path, out_png):
    cal = pd.read_csv(cal_path)
    if cal.empty or cal["n"].sum() == 0:
        return
    x = cal["mean_pred"].values
    y = cal["frac_win"].values
    n = cal["n"].values

    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.scatter(x, y, s=np.clip(n*2, 10, 200))
    plt.title("Reliability (Calibration)"); plt.xlabel("Predicted win prob"); plt.ylabel("Empirical win rate")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_pit(join_path, out_png):
    if not os.path.exists(join_path): return
    df = pd.read_csv(join_path)
    if {"p_blend","y"}.issubset(df.columns) and len(df)>=20:
        # PIT-style: show histogram of predicted probs for positives vs negatives
        pos = df.loc[df["y"]==1,"p_blend"].clip(0,1)
        neg = df.loc[df["y"]==0,"p_blend"].clip(0,1)
        plt.figure(figsize=(6,4))
        plt.hist(pos, bins=20, alpha=0.6, label="Wins")
        plt.hist(neg, bins=20, alpha=0.6, label="Losses")
        plt.title("Predicted Prob Distribution (Wins vs Losses)")
        plt.xlabel("Predicted win prob"); plt.ylabel("Count"); plt.legend()
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

os.makedirs(OUTDIR, exist_ok=True)
plot_reliability(CAL, os.path.join(OUTDIR,"reliability_curve.png"))
plot_pit(JOIN, os.path.join(OUTDIR,"pred_prob_hist.png"))
print(json.dumps({"wrote":["reliability_curve.png","pred_prob_hist.png"]}))
