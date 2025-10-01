import argparse, pathlib, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()

    out_dir = pathlib.Path("data/raw"); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "season":[args.season],
        "week":[args.week],
        "team":["PLACEHOLDER"],
        "opponent":["PLACEHOLDER"],
        "spread_close":[0.0],
        "total_close":[44.5],
        "open_spread":[0.0],
        "open_total":[44.5]
    })
    path = out_dir / f"market_stub_s{args.season}_w{args.week}.parquet"
    df.to_parquet(path, index=False)
    print(f"[pull_odds_cached] wrote {len(df)} rows -> {path}")

if __name__ == "__main__":
    main()
