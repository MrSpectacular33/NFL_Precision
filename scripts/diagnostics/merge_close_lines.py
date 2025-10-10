import os, argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-outcomes", required=True, help="outcomes_long_*.csv with bet_line")
    ap.add_argument("--in-close", required=True,
                    help="CSV with columns: game_id,bet_type,close_line (and optionally close_spread/close_total)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    outc = pd.read_csv(args.in_outcomes)
    close = pd.read_csv(args.in_close)

    need_o = {"game_id","bet_type","bet_line"}
    miss_o = need_o - set(outc.columns)
    if miss_o: raise SystemExit(f"Missing in outcomes: {miss_o}")

    need_c = {"game_id","bet_type"}
    miss_c = need_c - set(close.columns)
    if miss_c: raise SystemExit(f"Missing in close file: {miss_c}")

    # normalize
    outc["bet_type"] = outc["bet_type"].astype(str).str.lower()
    close["bet_type"] = close["bet_type"].astype(str).str.lower()

    # prefer explicit close_spread/close_total; else use close_line
    c = close.copy()
    if "close_line" in c.columns:
        c["close_line"] = c["close_line"].astype(float)
    if "close_spread" in c.columns:
        c.loc[c["bet_type"]=="spread", "close_line"] = c["close_spread"]
    if "close_total" in c.columns:
        c.loc[c["bet_type"]=="total", "close_line"] = c["close_total"]

    merged = outc.merge(c[["game_id","bet_type","close_line"]],
                        on=["game_id","bet_type"], how="left")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"[OK] wrote merged outcomes with close_line -> {args.out}")

if __name__ == "__main__":
    main()
